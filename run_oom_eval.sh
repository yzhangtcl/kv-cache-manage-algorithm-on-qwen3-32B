#!/usr/bin/env bash
set -euo pipefail

python3 make_reliability_datasets.py

python3 batch_qa_eval.py \
  --model Qwen/Qwen3-32B-AWQ \
  --dataset datasets/oom_stress.jsonl \
  --dtype auto \
  --mode both \
  --max-gpu-memory 20GiB \
  --max-cpu-memory "" \
  --offload-folder /root/autodl-tmp/offload \
  --prefill-chunk-tokens 2048 \
  --max-cache-tokens 3072 \
  --recent-window 1024 \
  --hot-cache-tokens 1024 \
  --hot-raw-tokens -1 \
  --merge-similarity 0.90 \
  --attention-decay 0.995 \
  --importance-update 0 \
  --output-csv /root/autodl-tmp/kvcache_outputs/oom_stress.csv \
  --artifacts-dir /root/autodl-tmp/kvcache_outputs/oom_stress

python3 summarize_eval.py /root/autodl-tmp/kvcache_outputs/oom_stress.csv
