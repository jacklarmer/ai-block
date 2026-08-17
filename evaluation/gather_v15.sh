#!/usr/bin/env bash
# v15: gather FRESH ImageNet real shards NOT used by v13 (00000-00002,00048-00051)
# or v14 (00020-00022). These are 00023-00025 — never touched. Oracle (data_unseen)
# is content-hash-deduped out in build_v15.py so it stays clean.
# NOTE: This candidacy was measured and DISCARDED (see README) — the fresh-real
# diversity lever plateaued, so v15 did NOT ship. Kept here for reproducibility.
set -uo pipefail
cd ~/aidet
source .venv/bin/activate
OUT=data_v15_more
mkdir -p "$OUT"
declare -a SHARDS=(
"data/train-00023-of-00052-8112dbb3c625d13a.parquet"
"data/train-00024-of-00052-c4d3713f0afbe4aa.parquet"
"data/train-00025-of-00052-6f44606ad3c37c83.parquet"
)
for s in "${SHARDS[@]}"; do
  echo "=== gather $s ==="
  python gather_imagenet_direct.py "$OUT" "$s" 2500 || echo "GATHER FAILED for $s"
done
echo "RAW TOTAL: $(find "$OUT" -type f | wc -l)"
echo "GATHER V15 DONE $(date)"
