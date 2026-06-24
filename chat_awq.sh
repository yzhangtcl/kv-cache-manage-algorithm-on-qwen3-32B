#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-32B-AWQ}"

python3 chat_qwen_awq.py \
  --model "$MODEL_PATH" \
  --dtype auto \
  --max-gpu-memory 18GiB \
  --max-cpu-memory 110GiB \
  --use-kvcache \
  --prefill-chunk-tokens 4096 \
  --max-cache-tokens 3072 \
  --recent-window 1024 \
  --hot-cache-tokens 1024 \
  --hot-raw-tokens -1 \
  --merge-similarity 0.90 \
  --attention-decay 0.995 \
  --importance-update 0.02 \
  --history-file /root/autodl-tmp/kvcache_outputs/chat_history.json
