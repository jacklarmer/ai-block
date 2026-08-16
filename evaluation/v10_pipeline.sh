#!/usr/bin/env bash
# v10 fine-tune: LOW-RESOLUTION real-photo class — fixes the resolution
# shortcut (v9 flagged downscaled real photos as AI).
set -uo pipefail
cd ~/aidet
source .venv/bin/activate
exec >> v10_pipeline.log 2>&1
echo "=== v10 pipeline start $(date) ==="
echo "=== train_v10 (from run_v9/best.pt) ==="
python train_v4.py --root data_v10/train --ckpt run_v9/best.pt --out run_v10 --epochs 6 --batch 80 2>&1 | grep -E "^ep[0-9]|BEST" | tail -8
echo "=== export v10 ==="
python export_onnx.py --ckpt run_v10/best.pt --out model/v10_detector.onnx 2>&1 | grep -E "exported|fp16" | tail -2
echo "=== consistency ==="
python check_consistent.py --pt run_v10/best.pt --onnx model/v10_detector_fp16.onnx --image-dir /tmp/v10_consist 2>&1 | tail -1 || true
echo "=== KEY: unseen real photos (ImageNet) FP — v10 must drop v9's ~40% ==="
python - <<PY 2>&1 | grep -vE "RuntimeWarning|Warning" | tail -8
import onnxruntime as ort, numpy as np, glob
from PIL import Image
import torchvision.transforms as T
tf=T.Compose([T.Resize(288),T.CenterCrop(256),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
def fp_rate(model, pattern, thr=0.5):
    sess=ort.InferenceSession(model)
    fs=glob.glob(pattern); tot=fp=0
    for f in fs:
        try:
            x=tf(Image.open(f).convert("RGB")).unsqueeze(0).numpy()
            p=np.exp(sess.run(None,{"input":x})[0][0]); p=p/p.sum(); tot+=1
            if p[1]>=thr: fp+=1
        except: pass
    return tot, fp, fp/tot if tot else 0
for name,pat in [("unseen ImageNet real","data_unseen/real/*.jpg"),
                 ("held-out COCO real","data_v9/test/real/*")]:
    t,f,r=fp_rate("model/v10_detector_fp16.onnx", pat)
    print(f"  v10 {name}: FP@0.5={r:.4f} ({f}/{t})")
for name,pat in [("unseen ImageNet real","data_unseen/real/*.jpg")]:
    t,f,r=fp_rate("model/v9_detector_fp16.onnx", pat)
    print(f"  (v9 {name}: FP@0.5={r:.4f} for reference)")
PY
echo "=== eval: per-generator regression ==="
for d in deepfake aura dalle3 frontier ideogram imagine leonardo midjourney mobius synthfaces; do
  [ -d "data_v10/test/fake/$d" ] || continue
  echo "----- $d -----"
  python eval_aid.py --real-dir data_v10/test/real --fake-dir "data_v10/test/fake/$d" --onnx model/v10_detector_fp16.onnx 2>&1 | grep -E "balanced acc" | tail -1
done
echo "=== v10 pipeline DONE $(date) ==="
