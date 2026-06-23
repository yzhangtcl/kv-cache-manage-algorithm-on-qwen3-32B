# KV Cache Offload Simulator

This repository contains a local NumPy simulator for KV cache storage policies during long-context autoregressive decoding. It compares a full-KV quality baseline with several bounded GPU/CPU cache policies, then estimates speedup against a full-KV system backed by GPU, CPU memory, and disk tiers.

The simulator does not require a real GPU. It is intended for policy exploration, not hardware benchmarking.

## Real LLM Server Experiment

This repo also includes an experimental Transformers runner that applies a bounded KV-cache policy to a real causal language model:

- `llm_kvcache.py`: chunked prefill plus hot/cold budgeted KV compression helpers.
- `run_server_experiment.py`: run one long-context prompt with budgeted KV cache.
- `evaluate_quality.py`: compare exact generation and budgeted-KV generation on small quality cases.
- `batch_needle_eval.py`: append-only batch needle-in-haystack accuracy evaluation.
- `batch_qa_eval.py`: append-only QA/reasoning/long-document evaluation over JSONL datasets.
- `scripts/build_eval_datasets.py`: generates local JSONL evaluation suites under `datasets/`.
- `scripts/make_long_prompt.py`: create a deterministic long-context prompt.
- `SERVER_RUNBOOK.md`: step-by-step Chinese instructions for running on a 4090 server.

Recommended server-side starting point:

```bash
python3 scripts/make_long_prompt.py --repeats 900 --output long_prompt.txt
python3 run_server_experiment.py \
  --model Qwen/Qwen2.5-32B-Instruct \
  --prompt-file long_prompt.txt \
  --dtype float16 \
  --max-new-tokens 96 \
  --prefill-chunk-tokens 512 \
  --max-cache-tokens 2048 \
  --recent-window 1536 \
  --hot-cache-tokens -1 \
  --hot-raw-tokens -1 \
  --merge-similarity 0.82
```

The real runner keeps recent KV entries exactly, splits old KV into hot and cold groups, keeps a small number of hot anchors exactly, merges redundant hot old KV into representatives, and merges cold old KV into representatives. `--hot-cache-tokens -1` assigns half of the non-recent cache budget to hot old KV; `--hot-raw-tokens -1` keeps one quarter of that hot budget as exact anchors.

Important: the real runner is an experimental approximation. RoPE-based KV tensors cannot be freely merged or reordered without quality risk, so use `evaluate_quality.py` to measure task-level degradation against an exact baseline whenever that baseline fits.

Batch accuracy run:

```bash
python3 batch_needle_eval.py \
  --model /root/autodl-tmp/models/Qwen3-32B \
  --cases 12 \
  --repeats 200 \
  --max-new-tokens 16 \
  --max-cache-tokens 1024 \
  --recent-window 768 \
  --hot-cache-tokens -1 \
  --hot-raw-tokens -1 \
  --output-csv /root/autodl-tmp/kvcache_outputs/batch_needle_eval.csv \
  --resume
```

General QA/reasoning run:

```bash
python3 scripts/build_eval_datasets.py --output-dir datasets
python3 batch_qa_eval.py \
  --model /root/autodl-tmp/models/Qwen3-32B \
  --dataset datasets/all_qa.jsonl \
  --limit 3 \
  --max-cache-tokens 2048 \
  --recent-window 1024 \
  --hot-cache-tokens 768 \
  --hot-raw-tokens -1 \
  --output-csv /root/autodl-tmp/kvcache_outputs/batch_qa_eval.csv \
  --artifacts-dir /root/autodl-tmp/kvcache_outputs/qa_artifacts \
  --resume
```

## Policies

The script compares five cache policies:

- `recent`: GPU keeps only the newest KV entries. Older entries are dropped.
- `hot`: GPU evicts low-score entries to CPU. CPU is periodically probed and likely useful entries are moved back to GPU.
- `hot_cluster`: same hot policy, but cold CPU entries can be merged with similar CPU representatives.
- `hybrid_cluster`: protects a recent GPU window and uses the remaining GPU slots for hot long-range entries. CPU overflow is clustered.
- `adaptive_hybrid_cluster`: starts from `hybrid_cluster` and adjusts the protected recent window based on whether CPU-returned entries receive attention later.

## Workload

Current code has one workload:

```bash
--workload attention
```

The old `synthetic` and `text` workloads have been removed. The current `AttentionTransitionWorkload` creates a controlled KV trace that mimics common decoder attention patterns:

- attention sinks near the beginning of the sequence
- strong local recency attention
- a smoothly moving long-range focus
- occasional long-range recall
- clustered key vectors generated from a smaller set of key prototypes

The key prototype mechanism matters for small experiments. If every 128-dimensional key were independently random, cosine similarities would be close to zero and CPU merge would rarely trigger. Prototype-centered keys make `--merge-similarity` meaningful at small sequence lengths.

## Baselines

There are two baseline concepts:

- Quality baseline: full-KV attention output. Cache policies are compared against this output for `base_acc`, `acc_drop`, `top1`, `cos`, and `mse`.
- Cost baseline: full-KV attention over a tiered GPU -> CPU -> disk storage model.

The old impossible infinite-GPU cost baseline has been removed. Speedup is always computed against the tiered full-KV cost:

```text
newest gpu_capacity KV entries      -> GPU
next cpu_capacity KV entries        -> CPU
older KV entries                    -> disk
```

By default, one CPU KV entry costs `10x` one GPU KV entry, and one disk KV entry costs `10x` one CPU KV entry.

## Run

Basic run:

```bash
python3 kv_cache_sim.py
```

Small bounded-memory run that makes merge behavior visible:

```bash
python3 kv_cache_sim.py \
  --seq-len 256 \
  --dim 64 \
  --gpu-capacity 64 \
  --cpu-capacity 64 \
  --merge-similarity 0.6
```

Larger recommended experiment:

```bash
python3 kv_cache_sim.py \
  --seq-len 2048 \
  --dim 128 \
  --attention-labels 64 \
  --attention-local-window 128 \
  --attention-sink-tokens 4 \
  --attention-sink-weight 0.25 \
  --attention-local-weight 0.50 \
  --attention-focus-weight 0.20 \
  --attention-recall-weight 0.05 \
  --attention-focus-drift 0.18 \
  --attention-transition-noise 0.04 \
  --attention-key-prototypes 128 \
  --attention-key-prototype-noise 0.08 \
  --attention-key-stay-prob 0.92 \
  --gpu-capacity 512 \
  --cpu-capacity 512 \
  --merge-similarity 0.6 \
  --merge-candidates 32 \
  --replace-candidates 32 \
  --probe-every 32 \
  --probe-topk 16 \
  --attention-scale 12
```

Example output shape:

```text
workload=attention, sequence=2048, dim=128, gpu_capacity=512, cpu_capacity=512, protected_recent=448
adaptive_min_recent=384, adaptive_max_recent=486, adaptive_step=16, adaptive_interval=64
attention_labels=64, attention_local_window=128, attention_sink_tokens=4, attend_before_insert=True
attention_weights=sink:0.25, local:0.5, focus:0.2, recall:0.05, focus_drift=0.18, transition_noise=0.04
attention_key_prototypes=128, attention_key_prototype_noise=0.08, attention_key_stay_prob=0.92
cpu/gpu speed ratio=10x, disk/cpu speed ratio=10x, transfer_cost=2, attention_scale=12
speedup_baseline=tiered_full_kv, tiered_full_kv_baseline=59845888.0
merge_similarity=0.6, merge_candidates=32, replace_candidates=32, merge_cost_ratio=0.1
policy                     base_acc   accuracy   acc_drop       top1       cos        mse        cost   base_cost   speedup      gpu      cpu     prot   cpu_comp   moved  merged   mops   repl
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
recent                      100.00%     91.85%      8.15%     91.85%    0.9631    0.00016    917248.0  59845888.0    65.25x    448.1      0.0      0.0      0.00x       0       0      0      0
hot                         100.00%     95.85%      4.15%     95.85%    0.9788    0.00011   1299232.0  59845888.0    46.06x    448.1    576.4      0.0      1.00x    3072       0      0      0
hot_cluster                 100.00%     96.19%      3.81%     96.19%    0.9894    0.00006   1118557.1  59845888.0    53.50x    448.1    286.2      0.0      1.10x    3072     351    351    673
hybrid_cluster              100.00%     96.48%      3.52%     96.48%    0.9864    0.00007   1096219.6  59845888.0    54.59x    448.1    252.4    448.0      1.26x    3072     515    515    513
adaptive_hybrid_cluster     100.00%     95.51%      4.49%     95.51%    0.9835    0.00008   1078687.5  59845888.0    55.48x    448.1    225.6    486.0      1.56x    3056     595    595    433
```

Exact values depend on random seed and parameters.

## Important Parameters

Workload shape:

```bash
--seq-len 2048
--dim 128
--attention-labels 64
--seed 7
```

Teacher attention pattern:

```bash
--attention-local-window 128
--attention-sink-tokens 4
--attention-sink-weight 0.25
--attention-local-weight 0.50
--attention-focus-weight 0.20
--attention-recall-weight 0.05
--attention-focus-drift 0.18
--attention-transition-noise 0.04
```

Key clustering:

```bash
--attention-key-prototypes 128
--attention-key-prototype-noise 0.08
--attention-key-stay-prob 0.92
```

Memory sizes and costs:

```bash
--gpu-capacity 512
--cpu-capacity 512
--cpu-gpu-speed-ratio 10
--disk-cpu-speed-ratio 10
--transfer-cost 2
```

CPU probing:

```bash
--probe-every 32
--probe-topk 16
```

Merge and replacement:

```bash
--merge-similarity 0.6
--merge-candidates 32
--replace-candidates 32
--merge-cost-ratio 0.1
```

Hybrid/adaptive recent window:

```bash
--protected-recent 0
--adaptive-interval 64
--adaptive-step 0
--adaptive-min-recent 0
--adaptive-max-recent 0
--adaptive-hot-high 0.55
--adaptive-hot-low 0.15
```

`0` means the code uses defaults derived from `gpu_capacity`:

- `protected_recent = 87.5% of gpu_capacity`
- `adaptive_min_recent = 75% of gpu_capacity`
- `adaptive_max_recent = 95% of gpu_capacity`
- `adaptive_step = gpu_capacity / 32`

## Metrics

- `base_acc`: accuracy of the full-KV quality baseline.
- `accuracy`: accuracy of the cache policy.
- `acc_drop`: absolute accuracy loss versus the full-KV quality baseline.
- `top1`: percentage of steps whose predicted label matches the full-KV baseline prediction.
- `cos`: mean cosine similarity between policy output and full-KV output.
- `mse`: mean squared error between policy output and full-KV output.
- `cost`: estimated runtime cost of the policy.
- `base_cost`: tiered full-KV cost used as the speedup denominator.
- `speedup`: `base_cost / cost`.
- `gpu`: average number of physical GPU KV entries.
- `cpu`: average number of physical CPU KV entries.
- `prot`: final protected recent window size.
- `cpu_comp`: CPU logical KV entries divided by CPU physical entries. `1.00x` means no compression.
- `moved`: total number of KV moves.
- `merged`: number of unique original KV entries that entered a merge at least once.
- `mops`: number of merge operations.
- `repl`: number of CPU replacement operations when CPU memory is full.

`base_acc` is usually close to 100% because targets are defined from the full-KV baseline itself. This is intentional: the experiment measures how closely each cache policy preserves full-KV behavior.

## Algorithm Summary

Each decode step inserts or attends to one KV item depending on `attend_before_insert`. The current attention workload uses `attend_before_insert=True`, so each step first attends over historical KV and then inserts the current KV.

`hot` keeps a bounded GPU cache. When GPU exceeds capacity, it evicts the item with the lowest score:

```text
score = importance * attention_decay ** age
```

Most steps attend only over GPU-resident KV. Every `probe_every` steps, CPU entries are scored against the current query, the top `probe_topk` entries are moved back to GPU, and GPU capacity is enforced again.

`hybrid_cluster` protects a recent window on GPU. Items inside the protected window are not preferred for hot eviction. The remaining GPU slots are used for older hot entries.

`adaptive_hybrid_cluster` adjusts the protected recent window every `adaptive_interval` steps. If CPU-probed entries later receive a large share of attention, the recent window shrinks to leave more room for long-range hot KV. If CPU-probed entries do not help much, the recent window grows.

`hot_cluster`, `hybrid_cluster`, and `adaptive_hybrid_cluster` compress CPU memory. When a GPU item is offloaded to CPU:

1. The policy checks up to `merge_candidates` CPU representatives.
2. If the best cosine similarity is at least `merge_similarity`, the item is merged by count-weighted averaging.
3. If no merge target exists and CPU is not full, the item is appended.
4. If CPU is full, the policy checks up to `replace_candidates` representatives and replaces the lowest-score candidate.

Merged representatives store `count`. Attention scoring adds `log(count)`:

```text
attention_score = attention_scale * (key @ query) + log(count)
```

This approximates the softmax mass of multiple similar KV rows represented by one physical CPU/GPU item.

## Additional Documentation

- `kv_cache_sim_中文注释.md`: Chinese overview of the current simulator.
- `HierarchicalKVCache_中文说明.md`: Chinese function-by-function explanation of `HierarchicalKVCache`.

## Limitations

This is a policy simulator, not a production inference engine.

Current limitations:

- No real GPU kernels, PCIe/NVLink transfers, async DMA, or batching.
- Cost model is hand-written, not measured from hardware.
- Workload simulates attention transition patterns but does not replay real Transformer tensors.
- No multi-layer or multi-head KV behavior.
- CPU probing scans CPU periodically instead of using ANN or an index.
- CPU merge uses bounded round-robin candidates, not global nearest neighbors.
- Disk is included only in the tiered full-KV cost baseline; optimized policies do not implement an explicit disk cache.
