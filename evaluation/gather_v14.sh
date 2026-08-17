#!/usr/bin/env bash
# v14: gather FRESH ImageNet real shards NOT used by v13 (00020, 00021, 00022).
# Each shard yields 2,500 JPEGs into data_v14_more/. The build step then
# content-hash-dedups against the truly-unseen eval oracle so the oracle stays
# clean (never-trained). Shards 00020-00022 were verified reachable from the HF
# dataset evanarlian/imagenet_1k_resized_256 via direct parquet GET.
set -uo pipefail
cd ~/aidet
source .venv/bin/activate
OUT=data_v14_more
mkdir -p "$OUT"
declare -a SHARDS=(
"data/train-00020-of-00052-3d2f3be76c1ba810.parquet"
"data/train-00021-of-00052-3660c4ef5916a594.parquet"
"data/train-00022-of-00052-5009c46203164a5b.parquet"
)
for s in "${SHARDS[@]}"; do
  echo "=== gather $s ==="
  python gather_imagenet_direct.py "$OUT" "$s" 2500 || echo "GATHER FAILED for $s"
done
echo "RAW TOTAL: $(find "$OUT" -type f | wc -l)"
echo "GATHER V14 DONE $(date)"
