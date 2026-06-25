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
DATASET="${DATASET:-datasets/reliability_speed.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/kvcache_outputs/accuracy_compare}"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"
MAX_PROMPT_CHARS="${MAX_PROMPT_CHARS:-0}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is not set. Run: export DEEPSEEK_API_KEY=\"your_deepseek_api_key\"" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

KVMANAGE_CSV="$OUTPUT_DIR/kvmanage.csv"
SLIDING_CSV="$OUTPUT_DIR/sliding_window.csv"
JUDGE_CSV="$OUTPUT_DIR/deepseek_judge.csv"
JUDGE_SUMMARY="$OUTPUT_DIR/deepseek_judge_summary.csv"

python3 make_reliability_datasets.py \
  --speed-cases "$SPEED_CASES" \
  --speed-repeats "$SPEED_REPEATS" \
  --oom-cases "$OOM_CASES" \
  --oom-repeats "$OOM_REPEATS"

rm -f "$KVMANAGE_CSV" "$SLIDING_CSV" "$JUDGE_CSV" "$JUDGE_SUMMARY"
rm -rf "$OUTPUT_DIR/kvmanage_artifacts" "$OUTPUT_DIR/sliding_window_artifacts"

echo "[1/3] Running KVManage: max=4096 recent=2048 hot=1536"
python3 batch_qa_eval.py \
  --model "$MODEL_NAME" \
  --dataset "$DATASET" \
  --dtype auto \
  --mode kvmanage \
  --mode-label kvmanage \
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
  --compress-every "$COMPRESS_EVERY" \
  --output-csv "$KVMANAGE_CSV" \
  --artifacts-dir "$OUTPUT_DIR/kvmanage_artifacts"

echo "[2/3] Running sliding window baseline: max=4096 recent=4096 hot=0"
python3 batch_qa_eval.py \
  --model "$MODEL_NAME" \
  --dataset "$DATASET" \
  --dtype auto \
  --mode kvmanage \
  --mode-label sliding_window \
  --max-gpu-memory "$MAX_GPU_MEMORY" \
  --max-cpu-memory "$MAX_CPU_MEMORY" \
  --offload-folder /root/autodl-tmp/offload \
  --prefill-chunk-tokens "$PREFILL_CHUNK_TOKENS" \
  --max-cache-tokens 4096 \
  --recent-window 4096 \
  --hot-cache-tokens 0 \
  --hot-raw-tokens 0 \
  --merge-similarity 0.90 \
  --attention-decay 0.995 \
  --importance-update 0 \
  --compress-every "$COMPRESS_EVERY" \
  --output-csv "$SLIDING_CSV" \
  --artifacts-dir "$OUTPUT_DIR/sliding_window_artifacts"

echo "[3/3] Judging answers with DeepSeek"
python3 deepseek_judge_eval.py \
  --dataset "$DATASET" \
  --eval-csv "$KVMANAGE_CSV" "$SLIDING_CSV" \
  --artifacts-dir "$OUTPUT_DIR/kvmanage_artifacts" "$OUTPUT_DIR/sliding_window_artifacts" \
  --output-csv "$JUDGE_CSV" \
  --summary-csv "$JUDGE_SUMMARY" \
  --model "$DEEPSEEK_MODEL" \
  --include-prompt \
  --max-prompt-chars "$MAX_PROMPT_CHARS" \
  --resume

echo
echo "Wrote:"
echo "- $KVMANAGE_CSV"
echo "- $SLIDING_CSV"
echo "- $JUDGE_CSV"
echo "- $JUDGE_SUMMARY"
echo
cat "$JUDGE_SUMMARY"
