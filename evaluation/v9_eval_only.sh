#!/usr/bin/env bash
# v9 eval-only re-run: v9 model is trained+exported; just fix the eval (which
# previously crashed because --onnx was omitted) + flat consistency check.
set -uo pipefail
cd ~/aidet
source .venv/bin/activate
exec >> v9_pipeline.log 2>&1
echo "=== v9 eval-only re-run $(date) ==="
# flat consistency image dir (check_consistent needs flat files, not subdirs)
mkdir -p /tmp/v9_consist_imgs
find data_v9/test/real -type l | head -120 | xargs -I{} cp -L {} /tmp/v9_consist_imgs/
find data_v9/test/fake/deepfake -type l | head -80 | xargs -I{} cp -L {} /tmp/v9_consist_imgs/
echo "consist imgs: $(ls /tmp/v9_consist_imgs | wc -l)"
echo "=== consistency ==="
python check_consistent.py --pt run_v9/best.pt --onnx model/v9_detector_fp16.onnx --image-dir /tmp/v9_consist_imgs 2>&1 | tail -1 || true
echo "=== eval: deepfake held-out + all-generator regression ==="
for d in deepfake aura dalle3 frontier ideogram imagine leonardo midjourney mobius synthfaces; do
  [ -d "data_v9/test/fake/$d" ] || continue
  echo "----- $d -----"
  python eval_aid.py --real-dir data_v9/test/real --fake-dir "data_v9/test/fake/$d" \
    --onnx model/v9_detector_fp16.onnx 2>&1 | grep -E "balanced acc" | tail -1
done
echo "=== v9 eval-only DONE $(date) ==="
