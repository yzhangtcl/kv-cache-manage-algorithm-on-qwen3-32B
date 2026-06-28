#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/models/Qwen3-8B-AWQ}"
DATASET="${DATASET:-data/longmemeval_s_cleaned.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/kvcache_outputs/qwen3_8b_awq_longmemeval_s_compare}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$OUTPUT_DIR/artifacts}"
LIMIT="${LIMIT:-0}"
MAX_GPU_MEMORY="${MAX_GPU_MEMORY:-22GiB}"
MAX_CPU_MEMORY="${MAX_CPU_MEMORY:-}"
DTYPE="${DTYPE:-auto}"
PREFILL_CHUNK_TOKENS="${PREFILL_CHUNK_TOKENS:-512}"
COMPRESS_EVERY="${COMPRESS_EVERY:-4}"
KV_CACHE_TOKENS_LIST="${KV_CACHE_TOKENS_LIST:-20000}"
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
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-${DEEPSEEK_JUDGE_MODEL:-deepseek-chat}}"
JUDGE_LIMIT="${JUDGE_LIMIT:-0}"
JUDGE_SLEEP_SEC="${JUDGE_SLEEP_SEC:-0}"
LOG_EVERY="${LOG_EVERY:-0}"
RUN_KVMANAGE="${RUN_KVMANAGE:-1}"
RUN_SLIDING="${RUN_SLIDING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WORD_DEDUP_PATTERN_WORDS="${WORD_DEDUP_PATTERN_WORDS:-4}"
export WORD_DEDUP_MIN_REPEATS="${WORD_DEDUP_MIN_REPEATS:-2}"
export WORD_DEDUP_KEEP_PER_PATTERN="${WORD_DEDUP_KEEP_PER_PATTERN:-1}"
export WORD_DEDUP_BOOST="${WORD_DEDUP_BOOST:-2.0}"

if [[ "$JUDGE_LIMIT" != "0" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is not set. Run: export DEEPSEEK_API_KEY=\"your_deepseek_api_key\"" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$ARTIFACTS_DIR"

hypotheses=()
mode_labels=()

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

  kv_label="kvmanage_${cache_label}"
  sliding_label="sliding_window_${cache_label}"

  if [[ "$RUN_KVMANAGE" != "0" ]]; then
    echo
    echo "Running Qwen3-8B-AWQ KVManage: input=${MAX_RETRIEVAL_TOKENS}, cache=${cache_tokens}, recent=${recent_window}, hot=${hot_cache_tokens}"
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
      --mode-label "$kv_label" \
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
      --log-every "$LOG_EVERY" \
      --rope-factor "$ROPE_FACTOR" \
      --rope-theta "$ROPE_THETA" \
      --resume \
      --continue-on-error
    hypotheses+=("$OUTPUT_DIR/${kv_label}.jsonl")
    mode_labels+=("$kv_label")
  fi

  if [[ "$RUN_SLIDING" != "0" ]]; then
    echo
    echo "Running Qwen3-8B-AWQ sliding window: input=${MAX_RETRIEVAL_TOKENS}, cache=${cache_tokens}"
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
      --mode sliding \
      --mode-label "$sliding_label" \
      --limit "$LIMIT" \
      --history-format json \
      --reading-method con \
      --topk-context 1000 \
      --max-retrieval-tokens "$MAX_RETRIEVAL_TOKENS" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --prefill-chunk-tokens "$PREFILL_CHUNK_TOKENS" \
      --sliding-cache-tokens "$cache_tokens" \
      --merge-similarity "$MERGE_SIMILARITY" \
      --attention-decay "$ATTENTION_DECAY" \
      --compress-every "$COMPRESS_EVERY" \
      --log-every "$LOG_EVERY" \
      --rope-factor "$ROPE_FACTOR" \
      --rope-theta "$ROPE_THETA" \
      --resume \
      --continue-on-error
    hypotheses+=("$OUTPUT_DIR/${sliding_label}.jsonl")
    mode_labels+=("$sliding_label")
  fi
done

if [[ "$JUDGE_LIMIT" != "0" ]]; then
  echo
  echo "Judging KVManage and sliding window outputs with DeepSeek"
  python3 longmemeval_deepseek_judge.py \
    --reference "$DATASET" \
    --hypothesis "${hypotheses[@]}" \
    --mode-labels "${mode_labels[@]}" \
    --output-csv "$OUTPUT_DIR/deepseek_judge.csv" \
    --summary-csv "$OUTPUT_DIR/deepseek_judge_summary.csv" \
    --model "$DEEPSEEK_MODEL" \
    --limit "$JUDGE_LIMIT" \
    --sleep-sec "$JUDGE_SLEEP_SEC" \
    --resume
else
  echo
  echo "Skipping DeepSeek judge because JUDGE_LIMIT=0."
fi

echo
echo "Wrote:"
echo "- $OUTPUT_DIR/runs.csv"
for hypothesis in "${hypotheses[@]}"; do
  echo "- $hypothesis"
done
echo "- $OUTPUT_DIR/deepseek_judge.csv"
echo "- $OUTPUT_DIR/deepseek_judge_summary.csv"
if [[ -f "$OUTPUT_DIR/deepseek_judge_summary.csv" ]]; then
  echo
  cat "$OUTPUT_DIR/deepseek_judge_summary.csv"
fi
