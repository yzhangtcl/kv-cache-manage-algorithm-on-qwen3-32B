#!/usr/bin/env bash
set -euo pipefail

python3 batch_qa_eval.py \
  --model Qwen/Qwen3-32B-AWQ \
  --dataset datasets/all_qa.jsonl \
  --dtype auto \
  --max-gpu-memory 20GiB \
  --max-cpu-memory "" \
  --offload-folder /root/autodl-tmp/offload \
  --prefill-chunk-tokens 4096 \
  --max-cache-tokens 3072 \
  --recent-window 1024 \
  --hot-cache-tokens 1024 \
  --hot-raw-tokens -1 \
  --merge-similarity 0.90 \
  --attention-decay 0.995 \
  --importance-update 0 \
  --output-csv /root/autodl-tmp/kvcache_outputs/awq_32b_qa.csv \
  --artifacts-dir /root/autodl-tmp/kvcache_outputs/awq_32b_qa

