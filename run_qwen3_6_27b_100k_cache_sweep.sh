#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/models/Qwen3.6-27B-AWQ}"
DATASET="${DATASET:-data/longmemeval_s_cleaned.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/kvcache_outputs/qwen3_6_27b_100k_cache_sweep}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$OUTPUT_DIR/artifacts}"
LIMIT="${LIMIT:-0}"
MAX_GPU_MEMORY="${MAX_GPU_MEMORY:-22GiB}"
MAX_CPU_MEMORY="${MAX_CPU_MEMORY:-}"
DTYPE="${DTYPE:-float16}"
PREFILL_CHUNK_TOKENS="${PREFILL_CHUNK_TOKENS:-512}"
COMPRESS_EVERY="${COMPRESS_EVERY:-4}"
KV_CACHE_TOKENS_LIST="${KV_CACHE_TOKENS_LIST:-20000 40000}"
RECENT_WINDOW_OVERRIDE="${RECENT_WINDOW:-}"
HOT_CACHE_TOKENS_OVERRIDE="${HOT_CACHE_TOKENS:-}"
HOT_RAW_TOKENS="${HOT_RAW_TOKENS:--1}"
MAX_RETRIEVAL_TOKENS="${MAX_RETRIEVAL_TOKENS:-100000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-800}"
IMPORTANCE_UPDATE="${IMPORTANCE_UPDATE:-0.02}"
MERGE_SIMILARITY="${MERGE_SIMILARITY:-0.90}"
ATTENTION_DECAY="${ATTENTION_DECAY:-0.995}"
ROPE_FACTOR="${ROPE_FACTOR:-4.0}"
ROPE_THETA="${ROPE_THETA:-1000000.0}"

mkdir -p "$OUTPUT_DIR" "$ARTIFACTS_DIR"

for cache_tokens in $KV_CACHE_TOKENS_LIST; do
  if (( cache_tokens <= 0 )); then
    echo "KV cache token budget must be positive, got: $cache_tokens" >&2
    exit 1
  fi

  if [[ -n "$RECENT_WINDOW_OVERRIDE" ]]; then
    recent_window="$RECENT_WINDOW_OVERRIDE"
  else
    recent_window=$((cache_tokens / 2))
  fi

  if [[ -n "$HOT_CACHE_TOKENS_OVERRIDE" ]]; then
    hot_cache_tokens="$HOT_CACHE_TOKENS_OVERRIDE"
  else
    hot_cache_tokens=$((cache_tokens * 3 / 8))
  fi

  if (( cache_tokens % 1000 == 0 )); then
    cache_label="$((cache_tokens / 1000))k"
  else
    cache_label="$cache_tokens"
  fi

  mode_label="kvmanage_${cache_label}"

  echo
  echo "Running Qwen3.6-27B-AWQ LongMemEval-S: input=${MAX_RETRIEVAL_TOKENS} tokens, cache=${cache_tokens}, recent=${recent_window}, hot=${hot_cache_tokens}"

  python3 longmemeval_eval.py \
    --model "$MODEL_NAME" \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --dtype "$DTYPE" \
    --device auto \
    --max-gpu-memory "$MAX_GPU_MEMORY" \
    --max-cpu-memory "$MAX_CPU_MEMORY" \
    --offload-folder /root/autodl-tmp/offload \
    --mode kvmanage \
    --mode-label "$mode_label" \
    --limit "$LIMIT" \
    --history-format json \
    --reading-method con \
    --topk-context 1000 \
    --max-retrieval-tokens "$MAX_RETRIEVAL_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --prefill-chunk-tokens "$PREFILL_CHUNK_TOKENS" \
    --max-cache-tokens "$cache_tokens" \
    --recent-window "$recent_window" \
    --hot-cache-tokens "$hot_cache_tokens" \
    --hot-raw-tokens "$HOT_RAW_TOKENS" \
    --merge-similarity "$MERGE_SIMILARITY" \
    --attention-decay "$ATTENTION_DECAY" \
    --importance-update "$IMPORTANCE_UPDATE" \
    --compress-every "$COMPRESS_EVERY" \
    --rope-factor "$ROPE_FACTOR" \
    --rope-theta "$ROPE_THETA" \
    --resume \
    --continue-on-error
done

echo
echo "Outputs:"
echo "- $OUTPUT_DIR/runs.csv"
for cache_tokens in $KV_CACHE_TOKENS_LIST; do
  if (( cache_tokens % 1000 == 0 )); then
    cache_label="$((cache_tokens / 1000))k"
  else
    cache_label="$cache_tokens"
  fi
  echo "- $OUTPUT_DIR/kvmanage_${cache_label}.jsonl"
done
