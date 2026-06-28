#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/models/Qwen3-8B-AWQ}"
DATASET="${DATASET:-data/longmemeval_s_cleaned.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/kvcache_outputs/longmemeval_s}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$OUTPUT_DIR/artifacts}"
MODE="${MODE:-all}"
LIMIT="${LIMIT:-0}"
MAX_GPU_MEMORY="${MAX_GPU_MEMORY:-22GiB}"
MAX_CPU_MEMORY="${MAX_CPU_MEMORY:-}"
PREFILL_CHUNK_TOKENS="${PREFILL_CHUNK_TOKENS:-1024}"
COMPRESS_EVERY="${COMPRESS_EVERY:-4}"
MAX_CACHE_TOKENS="${MAX_CACHE_TOKENS:-4096}"
RECENT_WINDOW="${RECENT_WINDOW:-2048}"
HOT_CACHE_TOKENS="${HOT_CACHE_TOKENS:-1536}"
SLIDING_CACHE_TOKENS="${SLIDING_CACHE_TOKENS:-4096}"
MAX_RETRIEVAL_TOKENS="${MAX_RETRIEVAL_TOKENS:-129000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-800}"
ROPE_FACTOR="${ROPE_FACTOR:-4.0}"

mkdir -p "$OUTPUT_DIR" "$ARTIFACTS_DIR"

python3 longmemeval_eval.py \
  --model "$MODEL_NAME" \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --artifacts-dir "$ARTIFACTS_DIR" \
  --dtype auto \
  --device auto \
  --max-gpu-memory "$MAX_GPU_MEMORY" \
  --max-cpu-memory "$MAX_CPU_MEMORY" \
  --offload-folder /root/autodl-tmp/offload \
  --mode "$MODE" \
  --limit "$LIMIT" \
  --history-format json \
  --reading-method con \
  --topk-context 1000 \
  --max-retrieval-tokens "$MAX_RETRIEVAL_TOKENS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --prefill-chunk-tokens "$PREFILL_CHUNK_TOKENS" \
  --max-cache-tokens "$MAX_CACHE_TOKENS" \
  --recent-window "$RECENT_WINDOW" \
  --hot-cache-tokens "$HOT_CACHE_TOKENS" \
  --sliding-cache-tokens "$SLIDING_CACHE_TOKENS" \
  --compress-every "$COMPRESS_EVERY" \
  --rope-factor "$ROPE_FACTOR" \
  --resume \
  --continue-on-error

echo
echo "Outputs:"
echo "- $OUTPUT_DIR/runs.csv"
echo "- $OUTPUT_DIR/full.jsonl"
echo "- $OUTPUT_DIR/kvmanage.jsonl"
echo "- $OUTPUT_DIR/sliding_window.jsonl"
