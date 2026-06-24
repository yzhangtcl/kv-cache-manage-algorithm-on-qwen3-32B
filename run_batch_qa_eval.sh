#!/usr/bin/env bash
set -euo pipefail

SPEED_CASES="${SPEED_CASES:-10}"
OOM_CASES="${OOM_CASES:-100}"
SPEED_REPEATS="${SPEED_REPEATS:-220}"
OOM_REPEATS="${OOM_REPEATS:-360}"

OUTPUT_CSV=/root/autodl-tmp/kvcache_outputs/reliability_speed.csv
rm -f "$OUTPUT_CSV"

python3 make_reliability_datasets.py \
  --speed-cases "$SPEED_CASES" \
  --speed-repeats "$SPEED_REPEATS" \
  --oom-cases "$OOM_CASES" \
  --oom-repeats "$OOM_REPEATS"

python3 batch_qa_eval.py \
  --model Qwen/Qwen3-32B-AWQ \
  --dataset datasets/reliability_speed.jsonl \
  --dtype auto \
  --mode both \
  --max-gpu-memory 20GiB \
  --max-cpu-memory "" \
  --offload-folder /root/autodl-tmp/offload \
  --prefill-chunk-tokens 4096 \
  --max-cache-tokens 4096 \
  --recent-window 2048 \
  --hot-cache-tokens 1536 \
  --hot-raw-tokens -1 \
  --merge-similarity 0.90 \
  --attention-decay 0.995 \
  --importance-update 0 \
  --output-csv "$OUTPUT_CSV" \
  --artifacts-dir /root/autodl-tmp/kvcache_outputs/reliability_speed

python3 summarize_eval.py "$OUTPUT_CSV"
