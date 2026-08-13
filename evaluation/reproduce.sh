#!/usr/bin/env bash
# Reproduce LocalLens end-to-end from source.
#
# Stages:
#   1. assemble training data (real + multi-generator AI) -> data_all2/
#   2. stream a DALL-E 3 slice + build the v4 layout -> data_v4_add/, data_v4/
#   3. fine-tune the v4 classifier (v3 ckpt + DALL-E 3) -> run_v4/best.pt
#   4. export to ONNX (fp32/fp16) -> model/*.onnx
#   5. verify export consistency (PyTorch vs ONNX)
#   6. evaluate the shipped fp16 artifact on the mixed held-out set
#
# Requires: an NVIDIA GPU node with a Python venv (torch, timm, onnxruntime,
# onnx, huggingface_hub, datasets, Pillow, opencv-python-headless, scipy, sklearn).
set -euo pipefail

VENV="${VENV:-$HOME/aidet/.venv}"
DATA="${DATA:-$HOME/aidet}"
PY="$VENV/bin/python"

echo "== 1. assemble base data =="
"$PY" assemble.py

echo "== 2. gather DALL-E 3 + build v4 layout =="
"$PY" gather_dalle.py "$DATA/data_v4_add/dalle3" 15000
"$PY" build_v4.py "$DATA/data_v4_add/dalle3" 615 200

echo "== 3. train v3 (if needed), then fine-tune v4 =="
"$PY" build_v3.py 2>/dev/null || true
"$PY" train_aid2.py --root data_all2/train --out run_v3 --epochs 12 --batch 64 --lr 2e-4
"$PY" train_v4.py --root data_v4/train --ckpt run_v3/best.pt --out run_v4 --epochs 8

echo "== 4. export ONNX (shipped fp16 = detector.onnx) =="
"$PY" export_onnx.py --ckpt run_v4/best.pt --out model/detector.onnx

echo "== 5. consistency (PyTorch vs ONNX fp16) =="
"$PY" check_consistent.py --pt run_v4/best.pt --onnx model/detector_fp16.onnx --image-dir data_v4/test/real

echo "== 6. mixed held-out eval (shipped fp16) =="
"$PY" eval_mixed.py run_v4/best.pt --real data_v4/test/real --fake data_v4/test/fake/mobius --fake data_v4/test/fake/dalle --maxreal 2000 --maxfake 2000

echo "DONE"

