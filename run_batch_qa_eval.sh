#!/usr/bin/env bash
set -euo pipefail

SPEED_CASES="${SPEED_CASES:-100}"
OOM_CASES="${OOM_CASES:-100}"
SPEED_REPEATS="${SPEED_REPEATS:-220}"
OOM_REPEATS="${OOM_REPEATS:-360}"
MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/models/Qwen3-8B-AWQ}"
MAX_GPU_MEMORY="${MAX_GPU_MEMORY:-22GiB}"
MAX_CPU_MEMORY="${MAX_CPU_MEMORY:-}"
PREFILL_CHUNK_TOKENS="${PREFILL_CHUNK_TOKENS:-1024}"
COMPRESS_EVERY="${COMPRESS_EVERY:-4}"
MAX_CACHE_TOKENS="${MAX_CACHE_TOKENS:-4096}"
RECENT_WINDOW="${RECENT_WINDOW:-2048}"
HOT_CACHE_TOKENS="${HOT_CACHE_TOKENS:-1536}"
HOT_RAW_TOKENS="${HOT_RAW_TOKENS:--1}"
export WORD_DEDUP_ENABLED="${WORD_DEDUP_ENABLED:-0}"
export WORD_DEDUP_PATTERN_WORDS="${WORD_DEDUP_PATTERN_WORDS:-6}"
export WORD_DEDUP_MIN_REPEATS="${WORD_DEDUP_MIN_REPEATS:-4}"
export WORD_DEDUP_KEEP_PER_PATTERN="${WORD_DEDUP_KEEP_PER_PATTERN:-4}"
export WORD_DEDUP_BOOST="${WORD_DEDUP_BOOST:-2.0}"

OUTPUT_CSV=/root/autodl-tmp/kvcache_outputs/reliability_speed.csv
rm -f "$OUTPUT_CSV"

python3 make_reliability_datasets.py \
  --speed-cases "$SPEED_CASES" \
  --speed-repeats "$SPEED_REPEATS" \
  --oom-cases "$OOM_CASES" \
  --oom-repeats "$OOM_REPEATS"

python3 batch_qa_eval.py \
  --model "$MODEL_NAME" \
  --dataset datasets/reliability_speed.jsonl \
  --dtype auto \
  --mode both \
  --max-gpu-memory "$MAX_GPU_MEMORY" \
  --max-cpu-memory "$MAX_CPU_MEMORY" \
  --offload-folder /root/autodl-tmp/offload \
  --prefill-chunk-tokens "$PREFILL_CHUNK_TOKENS" \
  --max-cache-tokens "$MAX_CACHE_TOKENS" \
  --recent-window "$RECENT_WINDOW" \
  --hot-cache-tokens "$HOT_CACHE_TOKENS" \
  --hot-raw-tokens "$HOT_RAW_TOKENS" \
  --merge-similarity 0.90 \
  --attention-decay 0.995 \
  --importance-update 0 \
  --compress-every "$COMPRESS_EVERY" \
  --output-csv "$OUTPUT_CSV" \
  --artifacts-dir /root/autodl-tmp/kvcache_outputs/reliability_speed

python3 summarize_eval.py "$OUTPUT_CSV"
