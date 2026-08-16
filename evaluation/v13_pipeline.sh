#!/usr/bin/env bash
# v13 fine-tune: add FRESH dense real photos (ImageNet shards 00000-00002,
# 00048-00051 — never used before) to the real class, from v12 checkpoint.
# Goal: cut the truly-unseen hard-subtype real FP below v12's 65.7% while
# holding AI recall. Uses the EXACT training transform for every eval.
set -uo pipefail
cd ~/aidet
source .venv/bin/activate
# Production-lane guard: train ONLY on the RTX 5090 (PCI bus 0000:01:00.0).
# Pin by PCI bus ID so it can never resolve to the PRO 6000 production lane.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0000:01:00.0
exec >> v13_pipeline.log 2>&1
echo "=== v13 pipeline start $(date) ==="
echo "=== train_v13 (from run_v12/best.pt) ==="
python train_v4.py --root data_v13/train --ckpt run_v12/best.pt --out run_v13 --epochs 8 --batch 80 2>&1 | grep -E "^ep[0-9]|BEST" | tail -10
echo "=== export v13 ==="
python export_onnx.py --ckpt run_v13/best.pt --out model/v13_detector.onnx 2>&1 | grep -E "fp16 size" | tail -1
echo "=== KEY: FRESH unseen real val (never trained) FP @ EXACT transform ==="
python - <<PY 2>&1 | grep -vE "RuntimeWarning|Warning" | tail -8
import onnxruntime as ort, numpy as np, glob
from PIL import Image
import torchvision.transforms as T
tf=T.Compose([T.Resize(288),T.CenterCrop(256),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
def fp(model, pattern, thr=0.5):
    sess=ort.InferenceSession(model); fs=glob.glob(pattern); tot=fp=0
    for f in fs:
        try:
            x=tf(Image.open(f).convert("RGB")).unsqueeze(0).numpy()
            p=np.exp(sess.run(None,{"input":x})[0][0]); p=p/p.sum(); tot+=1
            if p[1]>=thr: fp+=1
        except: pass
    return f"{fp/tot:.4f} ({fp}/{tot})"
print("  v13 unseen-real FP@0.5:", fp("model/v13_detector_fp16.onnx","data_v13/test/real_unseen/*.jpg"))
print("  v13 unseen-real FP@0.65:", fp("model/v13_detector_fp16.onnx","data_v13/test/real_unseen/*.jpg",0.65))
print("  v12 reference FP@0.5:", fp("model/v12_detector_fp16.onnx","data_v13/test/real_unseen/*.jpg"))
PY
echo "=== eval: per-generator AI-recall + held-out real spec ==="
for d in deepfake frontier dalle3 midjourney ideogram; do
  echo "----- $d -----"
  python eval_aid.py --real-dir data_v9/test/real --fake-dir "data_v9/test/fake/$d" --onnx model/v13_detector_fp16.onnx 2>&1 | grep -iE "balanced acc" | tail -1
done
echo "=== v13 pipeline DONE $(date) ==="
