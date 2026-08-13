#!/usr/bin/env bash
# Reproduce LocalLens end-to-end from source.
#
# Stages:
#   1. assemble training data (real + multi-generator AI) -> data_all2/
#   2. train the v3 classifier -> run_v3/best.pt
#   3. export to ONNX (fp32/fp16) -> model/*.onnx
#   4. verify export consistency (PyTorch vs ONNX)
#   5. evaluate the shipped fp16 artifact on the held-out generalization set
#
# Requires: an NVIDIA GPU node with a Python venv (torch, timm, onnxruntime,
# onnx, huggingface_hub, datasets, Pillow, opencv-python-headless, scipy, sklearn).
set -euo pipefail

VENV="${VENV:-$HOME/aidet/.venv}"
DATA="${DATA:-$HOME/aidet}"
PY="$VENV/bin/python"

echo "== 1. assemble data =="
"$PY" assemble.py

echo "== 2. train v3 =="
"$PY" train_aid2.py --root data_all2/train --out run_v3 --epochs 12 --batch 64 --lr 2e-4

echo "== 3. export ONNX =="
"$PY" export_onnx.py --ckpt run_v3/best.pt --out model/v3_detector.onnx

echo "== 4. consistency (PyTorch vs ONNX fp32/fp16) =="
"$PY" check_consistent.py

echo "== 5. held-out generalization eval (shipped fp16) =="
"$PY" eval_aid.py --onnx model/v3_detector_fp16.onnx \
      --real-dir data_v3/test/real --fake-dir data_v3/test/fake --max-per-class 4000

echo "DONE"
