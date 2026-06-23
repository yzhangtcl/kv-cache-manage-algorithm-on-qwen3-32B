#!/usr/bin/env bash
set -euo pipefail

python3 scripts/make_long_prompt.py --repeats 900 --output long_prompt.txt

python3 kv_cache_sim.py \
  --seq-len 2048 \
  --dim 128 \
  --attention-labels 64 \
  --gpu-capacity 512 \
  --cpu-capacity 512 \
  --merge-similarity 0.6 \
  --merge-candidates 32 \
  --replace-candidates 32 \
  --probe-every 32 \
  --probe-topk 16

python3 run_server_experiment.py \
  --model Qwen/Qwen2.5-32B-Instruct \
  --prompt-file long_prompt.txt \
  --dtype float16 \
  --max-gpu-memory 22GiB \
  --max-cpu-memory 96GiB \
  --offload-folder offload \
  --max-new-tokens 96 \
  --prefill-chunk-tokens 512 \
  --max-cache-tokens 2048 \
  --recent-window 1536 \
  --hot-cache-tokens -1 \
  --hot-raw-tokens -1 \
  --merge-similarity 0.82 \
  --attention-decay 0.98 \
  --importance-update 0.05 \
  --log-every 2048 \
  --output-file outputs/qwen32b_budgeted_kv.txt

python3 evaluate_quality.py \
  --model Qwen/Qwen2.5-32B-Instruct \
  --dtype float16 \
  --max-gpu-memory 22GiB \
  --max-cpu-memory 96GiB \
  --offload-folder offload \
  --max-new-tokens 64 \
  --prefill-chunk-tokens 512 \
  --max-cache-tokens 2048 \
  --recent-window 1536 \
  --hot-cache-tokens -1 \
  --hot-raw-tokens -1 \
  --merge-similarity 0.82 \
  --attention-decay 0.98 \
  --importance-update 0.05 \
  --output-csv outputs/qwen32b_quality.csv
