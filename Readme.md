# KV Cache Management on Qwen3-32B-AWQ

本项目用于在 Qwen3-32B-AWQ 上实验 KV cache 管理算法。核心思路是在长上下文推理时限制 `past_key_values` 的缓存规模：保留最近 token，选择部分重要的 hot token，并对冗余或 cold token 做合并表示，从而降低显存压力。

## 文件说明

- `kvcache.py`：KV cache 管理算法核心实现，包含 cache 压缩、hot/cold token 选择、分块 prefill 生成函数，以及 `model.generate()` 包装式压缩接口。
- `chat_qwen_awq.py`：交互式聊天主程序，支持普通生成、KV cache 管理、CPU offload、AWQ backend 配置和流式输出。
- `chat_awq.sh`：CPU offload 聊天启动脚本，适合显存不足但可以接受较慢速度的场景。
- `chat_awq_chunked.sh`：无 CPU offload 的分块 KV 聊天启动脚本，显存行为更接近 QA 脚本，会分块 prefill 并边跑边压缩 cache。
- `chat_history_chunked.json`: chunked 模式运行测试
- `batch_qa_eval.py`：批量 QA/长上下文评测脚本，用于在数据集上测试 KV cache 管理算法的效果。
- `make_reliability_datasets.py`：生成重复事实长上下文数据集，避免只考察单次出现的唯一答案。
- `summarize_eval.py`：汇总 full KV 与 kvmanage 的耗时、峰值显存、正确率和 OOM 情况。
- `run_batch_qa_eval.sh`：批量 QA 评测启动脚本。
- `run_oom_eval.sh`：full KV OOM、kvmanage 继续运行的压力评测脚本。
- `deepseek_judge_eval.py`：调用 DeepSeek API 对已有评测 CSV 中的答案做 LLM 裁判判分。
- `DEEPSEEK_JUDGE.md`：DeepSeek API key 获取、配置、判分命令和输出说明。
- `datasets/all_qa.jsonl`：QA 评测数据集。
- `requirements.txt`：Python 依赖列表。
- `logqa.txt`：QA 运行日志或实验记录。

## 常用命令

无 offload、分块 KV 聊天：

```bash
./chat_awq_chunked.sh
```

CPU offload 聊天：

```bash
./chat_awq.sh
```

批量 QA 评测：

```bash
./run_batch_qa_eval.sh
```

OOM 压力评测：

```bash
./run_oom_eval.sh
```

修改测试组数：

```bash
SPEED_CASES=10 ./run_batch_qa_eval.sh
OOM_CASES=100 ./run_oom_eval.sh
```

大上下文且允许模型权重 CPU offload：

```bash
MAX_GPU_MEMORY=18GiB MAX_CPU_MEMORY=110GiB PREFILL_CHUNK_TOKENS=512 ./run_batch_qa_eval.sh
```

`MAX_CPU_MEMORY` 只允许模型权重通过 `device_map=auto` 放到 CPU/offload folder；Transformers 不会自动把运行时 `past_key_values` KV cache 或 attention 临时张量溢出到 CPU。大输入如果仍 OOM，优先降低 `PREFILL_CHUNK_TOKENS`，例如 256 或 128。

减少 kvmanage 压缩开销：

```bash
COMPRESS_EVERY=4 ./run_batch_qa_eval.sh
COMPRESS_EVERY=8 ./run_batch_qa_eval.sh
```

`COMPRESS_EVERY` 越大，Python 压缩次数越少、速度可能更好，但 cache 会在两次压缩之间短暂超过预算，显存压力也更高。

如果 32B-AWQ 权重本身占用过高，速度对照建议换小一些的模型，让 full KV baseline 也能完成：

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-8B-AWQ \
  --local-dir /root/autodl-tmp/models/Qwen3-8B-AWQ \
  --local-dir-use-symlinks False \
  --resume-download

./run_batch_qa_eval.sh
```

OOM 展示可以继续用默认 32B：

```bash
MODEL_NAME=Qwen/Qwen3-32B-AWQ ./run_oom_eval.sh
```

## 实验建议

先运行 `python3 make_reliability_datasets.py` 生成：

- `datasets/reliability_speed.jsonl`：多条约 8k/10k/12k 级别长上下文，用于比较 full KV 与 4k kvmanage 的速度和显存。
- `datasets/oom_stress.jsonl`：多条更长的重复事实上下文，用于制造 full KV OOM、kvmanage 能完成的场景。

`batch_qa_eval.py --mode both` 会对同一个 case 先跑 `full`，再跑 `kvmanage`，输出 CSV 中的关键字段：

- `mode`：`full` 或 `kvmanage`。
- `status`：`ok`、`oom` 或 `error`。
- `elapsed_sec`、`peak_memory_gb`：耗时和峰值显存。
- `avg_kept_cache_tokens`、`dropped_tokens_total`、`merged_tokens_total`：kvmanage 的压缩统计。

`summarize_eval.py` 会按 `mode` 分开统计 accuracy、平均耗时、平均峰值显存、OOM 数和 error 数，避免把 full KV 的 OOM 与 kvmanage 的正确率混在一起。

`batch_qa_eval.py` 默认会在 CSV 末尾追加两行 summary：`__summary_full__` 和 `__summary_kvmanage__`。summary 行中 `ok` 是该方法 accuracy，`elapsed_sec`、`peak_memory_gb`、`prompt_tokens` 等数值列是该方法在 `status=ok` 样本上的均值。

汇总结果：

```bash
python3 summarize_eval.py /root/autodl-tmp/kvcache_outputs/reliability_speed.csv
python3 summarize_eval.py /root/autodl-tmp/kvcache_outputs/oom_stress.csv
```

使用 DeepSeek V4 Flash 作为答案裁判：

```bash
export DEEPSEEK_API_KEY="你的DeepSeek API Key"
python3 deepseek_judge_eval.py \
  --dataset datasets/reliability_speed.jsonl \
  --eval-csv speed4.csv \
  --output-csv outputs/deepseek_judge_speed4.csv \
  --summary-csv outputs/deepseek_judge_speed4_summary.csv \
  --resume
```

完整说明见 `DEEPSEEK_JUDGE.md`。

一键对比你的 KVManage 和滑动窗口 baseline 的准确率：

```bash
export DEEPSEEK_API_KEY="你的DeepSeek API Key"
./run_kvmanage_vs_sliding_accuracy.sh
```

该脚本会分别运行：

- KVManage：`--max-cache-tokens 4096 --recent-window 2048 --hot-cache-tokens 1536`
- Sliding window：`--max-cache-tokens 4096 --recent-window 4096 --hot-cache-tokens 0`

然后用 DeepSeek V4 Flash 带原 prompt 做裁判，汇总结果在 `/root/autodl-tmp/kvcache_outputs/accuracy_compare/deepseek_judge_summary.csv`。
