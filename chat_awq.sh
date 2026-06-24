#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-32B-AWQ}"

python3 chat_qwen_awq.py \
  --model "$MODEL_PATH" \
  --dtype auto \
  --max-gpu-memory 16GiB \
  --max-cpu-memory 110GiB \
  --awq-version gemm \
  --fresh-start \
  --max-input-tokens 0 \
  --max-new-tokens 4096 \
  --use-kvcache \
  --prefill-chunk-tokens 256 \
  --max-cache-tokens 2048 \
  --recent-window 1024 \
  --hot-cache-tokens 768 \
  --hot-raw-tokens -1 \
  --merge-similarity 0.90 \
  --attention-decay 0.995 \
  --importance-update 0.02 \
  --history-file /root/autodl-tmp/kvcache_outputs/chat_history.json
