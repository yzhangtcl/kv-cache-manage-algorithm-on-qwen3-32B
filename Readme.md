# KV Cache Management on Qwen3 AWQ

本项目用于在 Qwen3 AWQ 模型上实验长上下文推理时的 KV cache 管理策略。核心目标是在接近或超过显存预算的场景中，限制 `past_key_values` 的规模，同时尽量保留回答质量。

当前实现提供两类能力：

- 交互式聊天：在 Qwen3 AWQ 模型上进行 CLI 对话，支持普通生成、`model.generate()` 包装式 KV 压缩、分块 prefill + KV 压缩、CPU offload。
- 批量评测：生成长上下文 QA 数据集，对比 full KV 与 KVManage 在正确率、耗时、峰值显存和 OOM 情况上的差异，并可调用 DeepSeek API 做语义裁判。

## 核心思路

KVManage 在缓存超过预算后执行近似压缩：

- 保留最近 token，保证局部上下文连续性。
- 从旧 token 中选择部分 hot token，保留或聚类压缩重要内容。
- 将 cold token 合并为代表性 key/value 状态，降低缓存长度。
- 支持按间隔压缩，减少 Python 侧压缩开销。

注意：当前策略是实验性实现。对 RoPE 模型来说，压缩后无法完全恢复原始绝对位置关系，因此应把它作为研究和工程验证工具，而不是生产推理内核。

## 目录结构

| 文件 | 说明 |
| --- | --- |
| `kvcache.py` | KV cache 压缩、hot/cold token 选择、分块 prefill、full KV 与 KVManage 生成函数 |
| `chat_qwen_awq.py` | 交互式聊天入口，支持 AWQ、offload、流式输出和 KVManage 参数 |
| `chat_awq_chunked.sh` | 无 CPU offload 的分块 KV 聊天脚本 |
| `chat_awq.sh` | 带 CPU offload 的聊天脚本 |
| `batch_qa_eval.py` | 批量 QA/长上下文评测入口 |
| `make_reliability_datasets.py` | 生成重复事实长上下文评测集 |
| `summarize_eval.py` | 汇总 full KV 与 KVManage 的结果 |
| `deepseek_judge_eval.py` | 使用 DeepSeek API 对生成答案做语义裁判 |
| `run_batch_qa_eval.sh` | 速度/显存对比评测脚本，默认使用 8B AWQ |
| `run_oom_eval.sh` | OOM 压力评测脚本，默认使用 Qwen3-32B-AWQ |
| `run_kvmanage_vs_sliding_accuracy.sh` | KVManage 与 sliding window baseline 的准确率对比脚本 |
| `run_qwen3_6_27b_longmemeval_deepseek_compare.sh` | Qwen3.6-27B-AWQ，100k 输入，20k/40k KVManage 与 sliding window DeepSeek 对比脚本 |
| `plot_oom_chart.py` | 根据 OOM 评测 CSV 绘制图表 |
| `datasets/` | 评测数据集 |
| `outputs/` | 示例结果和图表 |
| `requirements.txt` | Python 依赖 |
| `DEPLOYMENT.md` | 环境部署、模型下载、运行和排错说明 |

## 快速开始

完整部署流程见 [DEPLOYMENT.md](DEPLOYMENT.md)。如果环境和模型已经准备好，可以直接使用下面的命令。

### 交互式聊天

分块 KVManage 聊天：

```bash
./chat_awq_chunked.sh
```

带 CPU offload 的聊天：

```bash
./chat_awq.sh
```

脚本默认读取：

```bash
/root/autodl-tmp/models/Qwen3-32B-AWQ
```

如果模型在其他位置：

```bash
MODEL_PATH=/path/to/Qwen3-32B-AWQ ./chat_awq_chunked.sh
```

聊天命令支持：

- `/exit`：退出。
- `/reset`：清空当前对话。
- `/history`：查看历史摘要。
- `/save`：保存历史到 `--history-file`。

### 批量评测

生成长上下文数据集并运行速度/显存对比：

```bash
./run_batch_qa_eval.sh
```

运行 OOM 压力评测：

```bash
./run_oom_eval.sh
```

调整样本数量：

```bash
SPEED_CASES=10 ./run_batch_qa_eval.sh
OOM_CASES=100 ./run_oom_eval.sh
```

汇总 CSV：

```bash
python3 summarize_eval.py /root/autodl-tmp/kvcache_outputs/reliability_speed.csv
python3 summarize_eval.py /root/autodl-tmp/kvcache_outputs/oom_stress.csv
```

### LongMemEval-S 本地复现

老师提到的 LongMemEval-S 建议单独跑，因为它的输入是真实多会话记忆数据，单题上下文可到 100k token 以上。先下载官方 cleaned 数据到本项目目录：

```bash
mkdir -p data
wget -O data/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

先用 1 条样本做 smoke test：

```bash
MODEL_NAME=/root/autodl-tmp/models/Qwen3-8B-AWQ \
DATASET=data/longmemeval_s_cleaned.json \
MODE=all \
LIMIT=1 \
./run_longmemeval_s.sh
```

全量跑 KVManage、sliding window 和 full context：

```bash
MODEL_NAME=/root/autodl-tmp/models/Qwen3-8B-AWQ \
DATASET=data/longmemeval_s_cleaned.json \
MODE=all \
LIMIT=0 \
OUTPUT_DIR=/root/autodl-tmp/kvcache_outputs/longmemeval_s \
./run_longmemeval_s.sh
```

`MODE=all` 会依次写出：

```bash
/root/autodl-tmp/kvcache_outputs/longmemeval_s/full.jsonl
/root/autodl-tmp/kvcache_outputs/longmemeval_s/kvmanage.jsonl
/root/autodl-tmp/kvcache_outputs/longmemeval_s/sliding_window.jsonl
/root/autodl-tmp/kvcache_outputs/longmemeval_s/runs.csv
```

三个 `jsonl` 文件使用官方评测需要的 `question_id` 和 `hypothesis` 字段；`runs.csv` 只提供本地快速排查用的 token 数、耗时、显存、压缩次数和简单 answer substring 命中，不等价于官方 LLM judge 分数。

如果使用 Qwen3 系列跑 100k+ 上下文，脚本默认启用 YaRN：

```bash
ROPE_FACTOR=4.0
MAX_RETRIEVAL_TOKENS=129000
```

单卡显存不足时，先把 full context 单独去掉，只对比 KVManage 和 sliding：

```bash
MODE=kvmanage ./run_longmemeval_s.sh
MODE=sliding ./run_longmemeval_s.sh
```

如果要在 Qwen3.6-27B-AWQ 上对比 KVManage 和 sliding window，并用 DeepSeek 判正确性：

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"

MODEL_NAME=/root/autodl-tmp/models/Qwen3.6-27B-AWQ \
DATASET=data/longmemeval_s_cleaned.json \
LIMIT=0 \
./run_qwen3_6_27b_longmemeval_deepseek_compare.sh
```

先小样本试跑：

```bash
LIMIT=1 JUDGE_LIMIT=4 ./run_qwen3_6_27b_longmemeval_deepseek_compare.sh
```

脚本默认使用：

```bash
MAX_RETRIEVAL_TOKENS=100000
KV_CACHE_TOKENS_LIST="20000 40000"
PREFILL_CHUNK_TOKENS=512
```

输出：

```bash
/root/autodl-tmp/kvcache_outputs/qwen3_6_27b_longmemeval_s_compare/kvmanage_20k.jsonl
/root/autodl-tmp/kvcache_outputs/qwen3_6_27b_longmemeval_s_compare/sliding_window_20k.jsonl
/root/autodl-tmp/kvcache_outputs/qwen3_6_27b_longmemeval_s_compare/kvmanage_40k.jsonl
/root/autodl-tmp/kvcache_outputs/qwen3_6_27b_longmemeval_s_compare/sliding_window_40k.jsonl
/root/autodl-tmp/kvcache_outputs/qwen3_6_27b_longmemeval_s_compare/deepseek_judge.csv
/root/autodl-tmp/kvcache_outputs/qwen3_6_27b_longmemeval_s_compare/deepseek_judge_summary.csv
```

再视显存调整：

```bash
PREFILL_CHUNK_TOKENS=512
MAX_CACHE_TOKENS=4096
RECENT_WINDOW=2048
HOT_CACHE_TOKENS=1536
SLIDING_CACHE_TOKENS=4096
```

绘制 OOM 图表需要额外安装 `matplotlib`：

```bash
pip install matplotlib
python3 plot_oom_chart.py \
  --csv /root/autodl-tmp/kvcache_outputs/oom_stress.csv \
  --output outputs/oom_stress_chart.png \
  --aggregate-csv outputs/oom_stress_aggregated.csv
```

## 常用参数

### 模型和显存

- `MODEL_PATH`：聊天脚本使用的本地模型路径。
- `MODEL_NAME`：评测脚本使用的模型路径或 Hugging Face 模型名。
- `MAX_GPU_MEMORY`：传给 Transformers `max_memory[0]` 的 GPU 预算。
- `MAX_CPU_MEMORY`：允许 `device_map=auto` 将模型权重放到 CPU/offload folder。
- `PREFILL_CHUNK_TOKENS`：分块 prefill 的 chunk 大小。

`MAX_CPU_MEMORY` 只影响模型权重放置。Transformers 不会自动把运行时 `past_key_values` 或 attention 临时张量溢出到 CPU。大输入仍然 OOM 时，优先降低 `PREFILL_CHUNK_TOKENS`，例如 `512`、`256` 或 `128`。

### KVManage

- `--max-cache-tokens`：压缩后目标 KV cache token 数。
- `--recent-window`：始终保留的最近 token 数。
- `--hot-cache-tokens`：旧 token 中的 hot token/cluster 预算。
- `--hot-raw-tokens`：hot token 中原样保留的数量，`-1` 表示自动。
- `--merge-similarity`：选择代表 token 时的相似度阈值。
- `--attention-decay`：hot token 重要性的时间衰减。
- `--importance-update`：根据新 query 更新重要性的强度。
- `--compress-every`：每隔多少次 forward 压缩一次。

例如减少压缩频率：

```bash
COMPRESS_EVERY=4 ./run_batch_qa_eval.sh
COMPRESS_EVERY=8 ./run_batch_qa_eval.sh
```

`COMPRESS_EVERY` 越大，压缩调用越少，速度可能更好；但两次压缩之间 cache 会短暂超过预算，显存压力也会更高。

## DeepSeek 语义裁判

先配置 API key：

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
```

对已有评测结果判分：

```bash
python3 deepseek_judge_eval.py \
  --dataset datasets/reliability_speed.jsonl \
  --eval-csv /root/autodl-tmp/kvcache_outputs/reliability_speed.csv \
  --artifacts-dir /root/autodl-tmp/kvcache_outputs/reliability_speed \
  --output-csv outputs/deepseek_judge.csv \
  --summary-csv outputs/deepseek_judge_summary.csv \
  --include-prompt \
  --resume
```

一键对比 KVManage 和 sliding window baseline：

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
./run_kvmanage_vs_sliding_accuracy.sh
```

该脚本会分别运行：

- KVManage：`--max-cache-tokens 4096 --recent-window 2048 --hot-cache-tokens 1536`
- Sliding window：`--max-cache-tokens 4096 --recent-window 4096 --hot-cache-tokens 0`

结果默认写入：

```bash
/root/autodl-tmp/kvcache_outputs/accuracy_compare/
```

## 输出说明

`batch_qa_eval.py` 的 CSV 关键字段：

- `mode`：`full`、`kvmanage` 或自定义 `--mode-label`。
- `status`：`ok`、`oom` 或 `error`。
- `ok`：基于规则判分的正确性。
- `prompt_tokens`、`generated_tokens`：输入和输出 token 数。
- `elapsed_sec`：单 case 耗时。
- `peak_memory_gb`：峰值显存。
- `compress_calls`：KVManage 压缩次数。
- `avg_kept_cache_tokens`：平均保留 cache token 数。
- `dropped_tokens_total`、`merged_tokens_total`：压缩统计。

默认会在 CSV 末尾追加 `__summary_<mode>__` 汇总行。`summarize_eval.py` 会忽略这些 summary 行并重新按 mode 统计。

## 推荐实验路径

1. 先用 8B AWQ 跑 `run_batch_qa_eval.sh`，确认 full KV 与 KVManage 都能完成，观察速度、显存和正确率。
2. 再用 32B AWQ 跑 `run_oom_eval.sh`，制造 full KV OOM、KVManage 继续完成的场景。
3. 对关键结果使用 `deepseek_judge_eval.py` 复核语义正确性。
4. 如果要和简单 baseline 对比，运行 `run_kvmanage_vs_sliding_accuracy.sh`。

## 已知限制

- KV 合并是近似方法，不能保证对所有长上下文任务都保持准确。
- 当前实现以实验可读性为主，压缩逻辑在 Python/PyTorch 层，速度不是生产级 kernel。
- CPU offload 只缓解模型权重显存压力，不能完全解决 attention 临时张量或 KV cache 峰值。
- full KV baseline 在 32B + 长上下文场景下可能直接 OOM，这是 OOM 压力评测预期行为。
