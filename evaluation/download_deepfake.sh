#!/usr/bin/env bash
# Download + extract the bitmind DeepfakeDataset AI.zip (synthetic-face / AI
# face-swap images) — the source for the v9 deepfake class.
set -euo pipefail
cd "$(dirname "$0")"
URL="https://huggingface.co/datasets/bitmind/DeepfakeDataset/resolve/main/AI.zip"
DEST="${1:-data_v9_dl}"
mkdir -p "$DEST"
echo "Downloading AI.zip (~1.3GB) from bitmind/DeepfakeDataset ..."
curl -sL -o "$DEST/AI.zip" "$URL"
echo "Extracting to $DEST/deepfake_ai ..."
mkdir -p "$DEST/deepfake_ai"
unzip -q -o "$DEST/AI.zip" -d "$DEST/deepfake_ai"
n=$(find "$DEST/deepfake_ai" -type f | wc -l)
echo "Extracted $n images -> $DEST/deepfake_ai"
