#!/usr/bin/env bash
# v16: gather FRESH ImageNet real shards 00026-00028 (never used by v14/v15 or
# the oracle), then the build content-hash-dedups them against the truly-unseen
# oracle so the oracle stays clean.
set -uo pipefail
cd ~/aidet
source .venv/bin/activate
OUT=data_v16_more
rm -rf "$OUT"; mkdir -p "$OUT"
declare -a SHARDS=(
"data/train-00026-of-00052-f14be58421a67130.parquet"
"data/train-00027-of-00052-9fc8a75241a3bff4.parquet"
"data/train-00028-of-00052-177f34038e9c9a41.parquet"
)
for s in "${SHARDS[@]}"; do
  echo "=== gather $s ==="
  python gather_imagenet_direct.py "$OUT" "$s" 2500 || echo "GATHER FAILED for $s"
done
echo "RAW TOTAL: $(find "$OUT" -type f | wc -l)"
echo "GATHER V16 DONE $(date)"
