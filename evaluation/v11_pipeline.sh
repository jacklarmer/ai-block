#!/usr/bin/env bash
# v11 fine-tune: REAL "looks-like-AI" photography subtype class — fix the 87%
# unseen-real false-positive rate.
set -uo pipefail
cd ~/aidet
source .venv/bin/activate
exec >> v11_pipeline.log 2>&1
echo "=== v11 pipeline start $(date) ==="
echo "=== train_v11 (from run_v10/best.pt) ==="
python train_v4.py --root data_v11/train --ckpt run_v10/best.pt --out run_v11 --epochs 6 --batch 80 2>&1 | grep -E "^ep[0-9]|BEST" | tail -8
echo "=== export v11 ==="
python export_onnx.py --ckpt run_v11/best.pt --out model/v11_detector.onnx 2>&1 | grep -E "fp16 size" | tail -1
echo "=== KEY: FRESH unseen real val (never trained) FP — must drop from 87% ==="
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
print("  v11 FRESH unseen real val (thr 0.5): FP=", fp("model/v11_detector_fp16.onnx","data_unseen/real_eval/*.jpg"))
print("  v11 FRESH unseen real val (thr 0.65): FP=", fp("model/v11_detector_fp16.onnx","data_unseen/real_eval/*.jpg",0.65))
print("  v10 reference (thr 0.5): FP=", fp("model/v10_detector_fp16.onnx","data_unseen/real_eval/*.jpg"))
PY
echo "=== eval: per-generator AI-recall regression ==="
for d in deepfake aura dalle3 frontier ideogram imagine leonardo midjourney mobius synthfaces; do
  [ -d "data_v11/test/fake/$d" ] || continue
  echo "----- $d -----"
  python eval_aid.py --real-dir data_v11/test/real --fake-dir "data_v11/test/fake/$d" --onnx model/v11_detector_fp16.onnx 2>&1 | grep -E "balanced acc" | tail -1
done
echo "=== v11 pipeline DONE $(date) ==="
