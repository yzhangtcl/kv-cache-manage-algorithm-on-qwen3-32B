#!/usr/bin/env bash
set -euo pipefail

SPEED_CASES="${SPEED_CASES:-10}"
OOM_CASES="${OOM_CASES:-100}"
SPEED_REPEATS="${SPEED_REPEATS:-220}"
OOM_REPEATS="${OOM_REPEATS:-360}"
MAX_GPU_MEMORY="${MAX_GPU_MEMORY:-18GiB}"
MAX_CPU_MEMORY="${MAX_CPU_MEMORY:-110GiB}"
PREFILL_CHUNK_TOKENS="${PREFILL_CHUNK_TOKENS:-512}"

OUTPUT_CSV=/root/autodl-tmp/kvcache_outputs/oom_stress.csv
rm -f "$OUTPUT_CSV"

python3 make_reliability_datasets.py \
  --speed-cases "$SPEED_CASES" \
  --speed-repeats "$SPEED_REPEATS" \
  --oom-cases "$OOM_CASES" \
  --oom-repeats "$OOM_REPEATS"

python3 batch_qa_eval.py \
  --model Qwen/Qwen3-32B-AWQ \
  --dataset datasets/oom_stress.jsonl \
  --dtype auto \
  --mode both \
  --max-gpu-memory "$MAX_GPU_MEMORY" \
  --max-cpu-memory "$MAX_CPU_MEMORY" \
  --offload-folder /root/autodl-tmp/offload \
  --prefill-chunk-tokens "$PREFILL_CHUNK_TOKENS" \
  --max-cache-tokens 4096 \
  --recent-window 2048 \
  --hot-cache-tokens 1536 \
  --hot-raw-tokens -1 \
  --merge-similarity 0.90 \
  --attention-decay 0.995 \
  --importance-update 0 \
  --output-csv "$OUTPUT_CSV" \
  --artifacts-dir /root/autodl-tmp/kvcache_outputs/oom_stress

python3 summarize_eval.py "$OUTPUT_CSV"
