#!/usr/bin/env bash
# v13: gather FRESH ImageNet real shards NOT used by v10/v11/v12 training
# (00000-00002, 00048-00051) for the broadened real class. Each shard yields
# 2500 diverse real photographs (macros / clinical / abstract / low-light /
# heavy-JPEG subtypes). Run on jack-gpu2 (~/aidet) — network to HF.
set -uo pipefail
cd ~/aidet
source .venv/bin/activate
OUT=data_v13_more
mkdir -p "$OUT"
declare -a SHARDS=(
  data/train-00000-of-00052-ab3669701d34fafd.parquet
  data/train-00001-of-00052-886eb11e764e42fe.parquet
  data/train-00002-of-00052-571cd07ccaf0aba0.parquet
  data/train-00048-of-00052-b19b8bd9a8957bd9.parquet
  data/train-00049-of-00052-2a355f06f10edd93.parquet
  data/train-00050-of-00052-8633d68c8a494520.parquet
  data/train-00051-of-00052-e00f2fc9aebbf12b.parquet
)
for s in "${SHARDS[@]}"; do
  echo "=== gather $s ==="
  python gather_imagenet_direct.py "$OUT" "$s" 2500
done
echo "TOTAL: $(find "$OUT" -type f | wc -l) in $OUT"
echo "GATHER V13 DONE $(date)"
