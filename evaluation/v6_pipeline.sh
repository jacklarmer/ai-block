#!/usr/bin/env bash
# v6 fine-tune + eval: real-photography class addition.
set -euo pipefail
cd ~/aidet && source .venv/bin/activate
exec >> v6_pipeline.log 2>&1
echo "=== v6 pipeline start $(date) ==="
echo "=== train_v6 ==="
python train_v4.py --root data_v6/train --ckpt run_v5/best.pt --out run_v6 --epochs 8
echo "=== export v6 ==="
python export_onnx.py --ckpt run_v6/best.pt --out model/v6_detector.onnx 2>&1 | grep -E "exported|fp16|size"
echo "=== consistency ==="
python check_consistent.py --pt run_v6/best.pt --onnx model/v6_detector_fp16.onnx --image-dir data_v6/test/real 2>&1 | tail -1
echo "=== eval: COCO held-out real (WOULD-X false-positives) vs AI sets ==="
for g in /home/jack/aidet/data_v6/test/fake/*; do
  [ -d "$g" ] || continue
  d=$(basename "$g")
  echo "----- $d -----"
  python eval_aid.py --onnx /home/jack/aidet/model/v6_detector_fp16.onnx --real-dir /home/jack/aidet/data_v6/test/real --fake-dir "$g" --max-per-class 1000 2>&1 | grep -E "balanced acc @65|recall" | head -2
done
echo "=== v6 pipeline DONE $(date) ==="
