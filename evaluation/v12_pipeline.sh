#!/usr/bin/env bash
# v12 fine-tune: MASSLY broadened real class (17.5K diverse real photos) —
# fix the severe unseen-real false-positive rate (v11 = 79% on unseen val).
set -uo pipefail
cd ~/aidet
source .venv/bin/activate
exec >> v12_pipeline.log 2>&1
echo "=== v12 pipeline start $(date) ==="
echo "=== train_v12 (from run_v11/best.pt) ==="
python train_v4.py --root data_v12/train --ckpt run_v11/best.pt --out run_v12 --epochs 8 --batch 80 2>&1 | grep -E "^ep[0-9]|BEST" | tail -10
echo "=== export v12 ==="
python export_onnx.py --ckpt run_v12/best.pt --out model/v12_detector.onnx 2>&1 | grep -E "fp16 size" | tail -1
echo "=== KEY: FRESH unseen real val (never trained) FP — must drop from 79% ==="
python - <<PY 2>&1 | grep -vE "RuntimeWarning|Warning" | tail -10
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
print("  v12 FRESH unseen real val (thr 0.5): FP=", fp("model/v12_detector_fp16.onnx","data_unseen/real_eval/*.jpg"))
print("  v12 FRESH unseen real val (thr 0.65): FP=", fp("model/v12_detector_fp16.onnx","data_unseen/real_eval/*.jpg",0.65))
print("  v11 reference (thr 0.5): FP=", fp("model/v11_detector_fp16.onnx","data_unseen/real_eval/*.jpg"))
PY
echo "=== eval: per-generator AI-recall regression ==="
for d in deepfake aura dalle3 frontier ideogram imagine leonardo midjourney mobius synthfaces; do
  [ -d "data_v12/test/fake/$d" ] || continue
  echo "----- $d -----"
  python eval_aid.py --real-dir data_v12/test/real --fake-dir "data_v12/test/fake/$d" --onnx model/v12_detector_fp16.onnx 2>&1 | grep -E "balanced acc" | tail -1
done
echo "=== v12 pipeline DONE $(date) ==="
