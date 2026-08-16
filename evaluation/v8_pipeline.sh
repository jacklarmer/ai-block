#!/usr/bin/env bash
# v8 fine-tune: FRONTIER AI class (CogView4/Gemini-flash/FLUX/Janus/RealVis).
set -euo pipefail
cd ~/aidet && source .venv/bin/activate
exec >> v8_pipeline.log 2>&1
echo "=== v8 pipeline start $(date) ==="

echo "=== train_v8 (from run_v7/best.pt) ==="
python train_v4.py --root data_v8/train --ckpt run_v7/best.pt \
  --out run_v8 --epochs 8 --batch 80 2>&1 | grep -E "^ep[0-9]|BEST" | tail -10

echo "=== export v8 ==="
python export_onnx.py --ckpt run_v8/best.pt --out model/v8_detector.onnx 2>&1 | tail -3

echo "=== consistency ==="
python check_consistent.py --pt run_v8/best.pt --onnx model/v8_detector_fp16.onnx \
  --image-dir data_v8/test/real 2>&1 | tail -2

echo "=== eval: per-generator (regression) + frontier held-out ==="
for g in $(ls data_v8/test/fake); do
  echo "----- $g -----"
  python eval_aid.py --real-dir data_v8/test/real --fake-dir "data_v8/test/fake/$g" \
    --onnx model/v8_detector_fp16.onnx 2>&1 | grep -E "balanced acc" | tail -1
done

echo "=== v8 pipeline DONE $(date) ==="
