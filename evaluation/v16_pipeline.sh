#!/usr/bin/env bash
# v16 fine-tune: add 7,500 FRESH, oracle-deduped ImageNet real photos (shards
# 00026-00028, never used by v13/v14/v15 or the oracle) to the real class, from
# the v14 checkpoint. Goal: test whether the v15 regression was a REAL
# diminishing-returns plateau or an artifact of v15's BROKEN build (stale
# data_v15 tree -> FileExistsError -> trained on a mixed/incomplete set).
# v16 fixes the build (clean OUT dir + hard-fail on stale trees) so this is the
# first clean retest of the fresh-real lever since v14.
# Uses the EXACT training transform for every eval. Production-lane guard pins
# training to the RTX 5090 (PCI 0000:01:00.0) — never the PRO 6000.
set -uo pipefail
cd ~/aidet
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0000:01:00.0
# hard-reject if the PRO 6000 production lane is somehow the only CUDA device
if ! nvidia-smi -L | grep -q "5090"; then
  echo "ABORT: no RTX 5090 visible to training"; exit 1
fi
exec >> v16_pipeline.log 2>&1
echo "=== v16 pipeline start $(date) ==="
echo "=== train_v16 (from run_v14/best.pt) ==="
python train_v4.py --root data_v16/train --ckpt run_v14/best.pt --out run_v16 --epochs 8 --batch 80 2>&1 | grep -E "^ep[0-9]|BEST" | tail -10
echo "=== export v16 ==="
python export_onnx.py --ckpt run_v16/best.pt --out model/v16_detector.onnx 2>&1 | grep -E "fp16 size|exported|fp32 size" | tail -2
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
print("  v16 unseen-real FP@0.5:", fp("model/v16_detector_fp16.onnx","data_v16/test/real_unseen/*.jpg"))
print("  v16 unseen-real FP@0.65:", fp("model/v16_detector_fp16.onnx","data_v16/test/real_unseen/*.jpg",0.65))
print("  v14 reference FP@0.5:", fp("model/v14_detector_fp16.onnx","data_v16/test/real_unseen/*.jpg"))
PY
echo "=== eval: per-generator AI-recall + held-out real spec ==="
for d in deepfake frontier dalle3 midjourney ideogram; do
  echo "----- $d -----"
  python eval_aid.py --real-dir data_v9/test/real --fake-dir "data_v9/test/fake/$d" --onnx model/v16_detector_fp16.onnx 2>&1 | grep -iE "balanced acc" | tail -1
done
echo "=== v16 pipeline DONE $(date) ==="
