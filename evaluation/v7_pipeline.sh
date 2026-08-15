#!/usr/bin/env bash
# v7 fine-tune + eval: real-ART class (fix Wikipedia art/illustration false-pos).
set -euo pipefail
cd ~/aidet && source .venv/bin/activate
exec >> v7_pipeline.log 2>&1
echo "=== v7 pipeline start $(date) ==="

echo "=== build_v7 ==="
python build_v7.py /home/jack/aidet/data_v7_add/real_art 200 3 \
  /home/jack/aidet/data_v6_add/real_coco 200 2>&1 | tail -4

echo "=== train_v7 (from run_v6/best.pt) ==="
python train_v4.py \
  --root data_v7/train \
  --ckpt run_v6/best.pt \
  --out run_v7 \
  --epochs 8 \
  --batch 80 2>&1 | grep -E "^ep[0-9]|BEST" | tail -10

echo "=== export v7 ==="
cp run_v7/best.pt model/v7_detector.pt
python export_onnx.py --ckpt run_v7/best.pt --out model/v7_detector.onnx 2>&1 | tail -4

echo "=== consistency ==="
python check_consistent.py --pt run_v7/best.pt --onnx model/v7_detector_fp16.onnx \
  --image-dir data_v7/test/real 2>&1 | tail -2

echo "=== eval: per-generator held-out (regression) + real-art/photo FP ==="
for g in $(ls data_v7/test/fake); do
  echo "----- $g -----"
  python eval_aid.py --real-dir data_v7/test/real --fake-dir "data_v7/test/fake/$g" \
    --onnx model/v7_detector_fp16.onnx 2>&1 | grep -E "balanced acc" | tail -1
done

echo "=== v7 pipeline DONE $(date) ==="
