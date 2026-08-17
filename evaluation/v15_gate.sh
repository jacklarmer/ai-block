#!/usr/bin/env bash
# v15 official gate measurement — EXACT training transform on the shipped ONNX.
cd ~/aidet
source .venv/bin/activate
python - <<'PY'
import onnxruntime as ort, numpy as np, glob
from PIL import Image
import torchvision.transforms as T
tf=T.Compose([T.Resize(288),T.CenterCrop(256),T.ToTensor(),
              T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
def fp(model, pattern, thr):
    sess=ort.InferenceSession(model); fs=glob.glob(pattern); tot=fp=0
    for f in fs:
        try:
            x=tf(Image.open(f).convert("RGB")).unsqueeze(0).numpy()
            p=np.exp(sess.run(None,{"input":x})[0][0]); p=p/p.sum(); tot+=1
            if p[1]>=thr: fp+=1
        except Exception: pass
    return f"{fp/max(tot,1):.4f} ({fp}/{tot})"
print("v15 FINAL unseen-real FP@0.5:", fp("model/v15_detector_fp16.onnx","data_unseen/real_eval/*.jpg",0.5))
print("v15 FINAL unseen-real FP@0.65:", fp("model/v15_detector_fp16.onnx","data_unseen/real_eval/*.jpg",0.65))
print("v14 ref unseen-real FP@0.5:", fp("model/v14_detector_fp16.onnx","data_unseen/real_eval/*.jpg",0.5))
PY
echo "=== per-generator recall (v15 FINAL) ==="
for d in deepfake frontier dalle3 midjourney ideogram; do
  echo "--- $d ---"
  python eval_aid.py --real-dir data_v9/test/real --fake-dir "data_v9/test/fake/$d" --onnx model/v15_detector_fp16.onnx 2>&1 | grep -E "balanced acc" | tail -1
done
echo "V15_GATE_DONE $(date)"
