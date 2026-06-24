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

## 实验建议

先运行 `python3 make_reliability_datasets.py` 生成：

- `datasets/reliability_speed.jsonl`：中等长上下文，用于比较 full KV 与 kvmanage 的速度和显存。
- `datasets/oom_stress.jsonl`：更长的重复事实上下文，用于制造 full KV OOM、kvmanage 能完成的场景。

`batch_qa_eval.py --mode both` 会对同一个 case 先跑 `full`，再跑 `kvmanage`，输出 CSV 中的关键字段：

- `mode`：`full` 或 `kvmanage`。
- `status`：`ok`、`oom` 或 `error`。
- `elapsed_sec`、`peak_memory_gb`：耗时和峰值显存。
- `avg_kept_cache_tokens`、`dropped_tokens_total`、`merged_tokens_total`：kvmanage 的压缩统计。

汇总结果：

```bash
python3 summarize_eval.py /root/autodl-tmp/kvcache_outputs/reliability_speed.csv
python3 summarize_eval.py /root/autodl-tmp/kvcache_outputs/oom_stress.csv
```

