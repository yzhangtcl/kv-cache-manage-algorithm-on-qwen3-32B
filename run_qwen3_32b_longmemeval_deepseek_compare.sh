#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/models/Qwen3-32B-AWQ}"
DATASET="${DATASET:-data/longmemeval_s_cleaned.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/kvcache_outputs/qwen3_32b_longmemeval_s_compare}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$OUTPUT_DIR/artifacts}"
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
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-${DEEPSEEK_JUDGE_MODEL:-deepseek-chat}}"
JUDGE_LIMIT="${JUDGE_LIMIT:-0}"
JUDGE_SLEEP_SEC="${JUDGE_SLEEP_SEC:-0}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is not set. Run: export DEEPSEEK_API_KEY=\"your_deepseek_api_key\"" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$ARTIFACTS_DIR"

echo "[1/3] Running Qwen3-32B-AWQ KVManage on LongMemEval-S"
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
  --mode kvmanage \
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
  --compress-every "$COMPRESS_EVERY" \
  --rope-factor "$ROPE_FACTOR" \
  --resume \
  --continue-on-error

echo "[2/3] Running Qwen3-32B-AWQ sliding window on LongMemEval-S"
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
  --mode sliding \
  --limit "$LIMIT" \
  --history-format json \
  --reading-method con \
  --topk-context 1000 \
  --max-retrieval-tokens "$MAX_RETRIEVAL_TOKENS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --prefill-chunk-tokens "$PREFILL_CHUNK_TOKENS" \
  --sliding-cache-tokens "$SLIDING_CACHE_TOKENS" \
  --compress-every "$COMPRESS_EVERY" \
  --rope-factor "$ROPE_FACTOR" \
  --resume \
  --continue-on-error

echo "[3/3] Judging KVManage and sliding window with DeepSeek"
python3 longmemeval_deepseek_judge.py \
  --reference "$DATASET" \
  --hypothesis "$OUTPUT_DIR/kvmanage.jsonl" "$OUTPUT_DIR/sliding_window.jsonl" \
  --mode-labels kvmanage sliding_window \
  --output-csv "$OUTPUT_DIR/deepseek_judge.csv" \
  --summary-csv "$OUTPUT_DIR/deepseek_judge_summary.csv" \
  --model "$DEEPSEEK_MODEL" \
  --limit "$JUDGE_LIMIT" \
  --sleep-sec "$JUDGE_SLEEP_SEC" \
  --resume

echo
echo "Wrote:"
echo "- $OUTPUT_DIR/kvmanage.jsonl"
echo "- $OUTPUT_DIR/sliding_window.jsonl"
echo "- $OUTPUT_DIR/runs.csv"
echo "- $OUTPUT_DIR/deepseek_judge.csv"
echo "- $OUTPUT_DIR/deepseek_judge_summary.csv"
echo
cat "$OUTPUT_DIR/deepseek_judge_summary.csv"
