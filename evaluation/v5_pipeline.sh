#!/usr/bin/env bash
# Auto-drive the FULL v5 pipeline on jack-gpu2: wait for gather_v5 to finish,
# then build data, fine-tune v5, export ONNX, and run per-generator held-out eval.
# Logs everything to v5_pipeline.log. Safe to re-run (idempotent-ish via dir checks).
set -u
cd ~/aidet && source .venv/bin/activate

exec >> /home/jack/aidet/v5_pipeline.log 2>&1
echo "=== v5 pipeline start $(date) ==="

# 1. Wait for the gatherer to finish (it runs up to a few hours)
for i in $(seq 1 240); do
  if ! pgrep -f "python gather_v5.py" >/dev/null 2>&1; then
    echo "[wait] gatherer finished (iteration $i)"; break
  fi
  sleep 60
done
if pgrep -f "python gather_v5.py" >/dev/null 2>&1; then
  echo "[wait] TIMEOUT waiting for gatherer (still running) — continuing anyway"; 
fi

# 2. Build the v5 data layout (held-out per-gen slices)
echo "=== build_v5 ==="
python build_v5.py /home/jack/aidet/data_v5_add 200 4

# 3. Fine-tune v5 from the v4 checkpoint
echo "=== train_v5 ==="
python train_v4.py --root /home/jack/aidet/data_v5/train --ckpt /home/jack/aidet/run_v4/best.pt --out /home/jack/aidet/run_v5 --epochs 8

# 4. Export the shipped fp16 ONNX
echo "=== export ==="
python export_onnx.py --ckpt /home/jack/aidet/run_v5/best.pt --out /home/jack/aidet/model/v5_detector.onnx

# 5. Per-generator held-out recall (all test slices under data_v5/test/fake)
echo "=== eval per-gen (v5 fp16 ONNX) ==="
real=/home/jack/aidet/data_v5/test/real
for g in /home/jack/aidet/data_v5/test/fake/*; do
  [ -d "$g" ] || continue
  d=$(basename "$g")
  echo "----- $d -----"
  python eval_aid.py --onnx /home/jack/aidet/model/v5_detector_fp16.onnx --real-dir $real --fake-dir "$g" --max-per-class 1000 2>&1 \
     | grep -E "balanced acc @65" | head -1
done

echo "=== v5 pipeline DONE $(date) ==="
echo "PIPELINE_COMPLETE" 
