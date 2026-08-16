#!/usr/bin/env bash
# v9 fine-tune: DEEPFAKE / synthetic-face class (real-world AI face fraud).
set -euo pipefail
cd ~/aidet && source .venv/bin/activate
exec >> v9_pipeline.log 2>&1
echo "=== v9 pipeline start $(date) ==="
echo "=== train_v9 (from run_v8/best.pt) ==="
python train_v4.py --root data_v9/train --ckpt run_v8/best.pt --out run_v9 --epochs 8 --batch 80 2>&1 | grep -E "^ep[0-9]|BEST" | tail -10
echo "=== export v9 ==="
python export_onnx.py --ckpt run_v9/best.pt --out model/v9_detector.onnx
echo "=== consistency ==="
python check_consistent.py --pt run_v9/best.pt --onnx model/v9_detector_fp16.onnx --image-dir /tmp/v9_consist_imgs 2>&1 | tail -1 || true
echo "=== eval: deepfake held-out + all-generator regression ==="
for d in deepfake aura dalle3 frontier ideogram imagine leonardo midjourney mobius synthfaces; do
  [ -d "data_v9/test/fake/$d" ] || continue
  echo "----- $d -----"
  python eval_aid.py --real-dir data_v9/test/real --fake-dir "data_v9/test/fake/$d" \
    --onnx model/v9_detector_fp16.onnx 2>&1 | grep -E "balanced acc" | tail -1
done
echo "=== v9 pipeline DONE $(date) ==="
