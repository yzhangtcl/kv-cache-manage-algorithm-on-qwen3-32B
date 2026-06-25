# 部署文档

本文档说明如何从一台干净的 GPU 机器部署并运行本项目。示例路径默认使用 AutoDL 风格目录：

```bash
/root/autodl-tmp/
```

如果你的机器路径不同，把命令中的模型、输出和 offload 目录替换为实际路径即可。

## 1. 环境要求

建议环境：

- Linux
- Python 3.10 或 3.11
- NVIDIA GPU 和可用 CUDA 版 PyTorch
- 32B AWQ：建议 24GB 及以上显存；如果显存较小，需要 CPU offload 或更小模型
- 8B AWQ：更适合速度、正确率和 baseline 对比实验

本项目依赖 Transformers 加载 AWQ 模型。不同机器的 CUDA、PyTorch 和 AWQ backend 组合可能有差异；如果 AWQ backend 报错，先尝试不显式指定 `--awq-version` 和 `--awq-backend`，再根据本机环境调整。

## 2. 创建 Python 环境

```bash
cd /path/to/kv-cache-manage-algorithm-on-qwen3-32B

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
pip install -r requirements.txt
```

安装 CUDA 版 PyTorch 时，以你的机器 CUDA 版本为准。例如 CUDA 12.1：

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch
```

如果需要绘图：

```bash
pip install matplotlib
```

检查 GPU：

```bash
python3 - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

## 3. 准备目录

```bash
mkdir -p /root/autodl-tmp/models
mkdir -p /root/autodl-tmp/offload
mkdir -p /root/autodl-tmp/kvcache_outputs
```

如果没有 `/root/autodl-tmp` 权限，可以使用自己的目录，例如：

```bash
mkdir -p "$HOME/models" "$HOME/offload" "$HOME/kvcache_outputs"
```

随后在命令中替换 `MODEL_PATH`、`MODEL_NAME`、`--offload-folder`、`--output-csv` 和 `--artifacts-dir`。

## 4. 下载模型

安装 Hugging Face CLI：

```bash
pip install -U "huggingface_hub[cli]"
```

下载 Qwen3-32B-AWQ：

```bash
huggingface-cli download Qwen/Qwen3-32B-AWQ \
  --local-dir /root/autodl-tmp/models/Qwen3-32B-AWQ \
  --local-dir-use-symlinks False \
  --resume-download
```

如果主要做速度/准确率对比，建议先下载 8B AWQ：

```bash
huggingface-cli download Qwen/Qwen3-8B-AWQ \
  --local-dir /root/autodl-tmp/models/Qwen3-8B-AWQ \
  --local-dir-use-symlinks False \
  --resume-download
```

国内或受限网络环境可提前在其他机器下载模型，再同步到上述目录。

## 5. 运行交互式聊天

推荐先运行分块 KVManage 聊天：

```bash
MODEL_PATH=/root/autodl-tmp/models/Qwen3-32B-AWQ ./chat_awq_chunked.sh
```

如果显存不足，使用 CPU offload 版本：

```bash
MODEL_PATH=/root/autodl-tmp/models/Qwen3-32B-AWQ ./chat_awq.sh
```

手动运行示例：

```bash
python3 chat_qwen_awq.py \
  --model /root/autodl-tmp/models/Qwen3-32B-AWQ \
  --dtype auto \
  --max-gpu-memory 18GiB \
  --max-cpu-memory "" \
  --fresh-start \
  --max-input-tokens 0 \
  --max-new-tokens 1024 \
  --use-kvcache \
  --generation-backend chunked-kv \
  --prefill-chunk-tokens 256 \
  --max-cache-tokens 4096 \
  --recent-window 2048 \
  --hot-cache-tokens 1536 \
  --history-file /root/autodl-tmp/kvcache_outputs/chat_history_chunked.json
```

如果报 OOM，优先调整：

```bash
--prefill-chunk-tokens 128
--max-cache-tokens 2048
--recent-window 1024
--hot-cache-tokens 768
```

如果模型权重本身放不下，再设置：

```bash
--max-cpu-memory 110GiB
--offload-folder /root/autodl-tmp/offload
```

## 6. 运行批量评测

### 6.1 生成评测数据

```bash
python3 make_reliability_datasets.py \
  --speed-cases 10 \
  --speed-repeats 220 \
  --oom-cases 100 \
  --oom-repeats 360
```

会生成：

- `datasets/reliability_speed.jsonl`
- `datasets/oom_stress.jsonl`

### 6.2 速度/显存对比

默认脚本使用本地 8B AWQ：

```bash
MODEL_NAME=/root/autodl-tmp/models/Qwen3-8B-AWQ ./run_batch_qa_eval.sh
```

输出：

```bash
/root/autodl-tmp/kvcache_outputs/reliability_speed.csv
/root/autodl-tmp/kvcache_outputs/reliability_speed/
```

手动运行示例：

```bash
python3 batch_qa_eval.py \
  --model /root/autodl-tmp/models/Qwen3-8B-AWQ \
  --dataset datasets/reliability_speed.jsonl \
  --dtype auto \
  --mode both \
  --max-gpu-memory 22GiB \
  --max-cpu-memory "" \
  --offload-folder /root/autodl-tmp/offload \
  --prefill-chunk-tokens 1024 \
  --max-cache-tokens 4096 \
  --recent-window 2048 \
  --hot-cache-tokens 1536 \
  --compress-every 4 \
  --output-csv /root/autodl-tmp/kvcache_outputs/reliability_speed.csv \
  --artifacts-dir /root/autodl-tmp/kvcache_outputs/reliability_speed
```

### 6.3 OOM 压力评测

默认脚本使用 `Qwen/Qwen3-32B-AWQ`，可改成本地路径：

```bash
MODEL_NAME=/root/autodl-tmp/models/Qwen3-32B-AWQ ./run_oom_eval.sh
```

输出：

```bash
/root/autodl-tmp/kvcache_outputs/oom_stress.csv
/root/autodl-tmp/kvcache_outputs/oom_stress/
```

汇总：

```bash
python3 summarize_eval.py /root/autodl-tmp/kvcache_outputs/oom_stress.csv
```

绘图：

```bash
python3 plot_oom_chart.py \
  --csv /root/autodl-tmp/kvcache_outputs/oom_stress.csv \
  --output outputs/oom_stress_chart.png \
  --aggregate-csv outputs/oom_stress_aggregated.csv
```

## 7. DeepSeek 语义裁判

安装依赖已包含在 `requirements.txt` 中。配置 API key：

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
```

对已有 CSV 判分：

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

只检查请求内容、不调用 API：

```bash
python3 deepseek_judge_eval.py \
  --dataset datasets/reliability_speed.jsonl \
  --eval-csv /root/autodl-tmp/kvcache_outputs/reliability_speed.csv \
  --output-csv outputs/deepseek_judge_dry_run.csv \
  --summary-csv outputs/deepseek_judge_dry_run_summary.csv \
  --dry-run \
  --limit 3
```

一键对比 KVManage 与 sliding window：

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
MODEL_NAME=/root/autodl-tmp/models/Qwen3-8B-AWQ ./run_kvmanage_vs_sliding_accuracy.sh
```

## 8. 常见参数模板

### 24GB GPU，8B 对比实验

```bash
MODEL_NAME=/root/autodl-tmp/models/Qwen3-8B-AWQ \
MAX_GPU_MEMORY=22GiB \
MAX_CPU_MEMORY="" \
PREFILL_CHUNK_TOKENS=1024 \
COMPRESS_EVERY=4 \
./run_batch_qa_eval.sh
```

### 24GB GPU，32B OOM 展示

```bash
MODEL_NAME=/root/autodl-tmp/models/Qwen3-32B-AWQ \
MAX_GPU_MEMORY=18GiB \
MAX_CPU_MEMORY=110GiB \
PREFILL_CHUNK_TOKENS=512 \
COMPRESS_EVERY=4 \
./run_oom_eval.sh
```

### 更保守的显存配置

```bash
MODEL_NAME=/root/autodl-tmp/models/Qwen3-32B-AWQ \
MAX_GPU_MEMORY=16GiB \
MAX_CPU_MEMORY=120GiB \
PREFILL_CHUNK_TOKENS=256 \
COMPRESS_EVERY=2 \
./run_oom_eval.sh
```

## 9. 常见问题

### CUDA OOM

优先按顺序降低：

1. `PREFILL_CHUNK_TOKENS`
2. `--max-cache-tokens`
3. `--recent-window`
4. `--hot-cache-tokens`
5. `--max-new-tokens`

如果是模型加载阶段 OOM，增加 CPU offload：

```bash
MAX_CPU_MEMORY=110GiB
```

如果是长 prompt prefill 阶段 OOM，降低：

```bash
PREFILL_CHUNK_TOKENS=256
```

### full KV baseline 总是 OOM

这是 32B + 长上下文压力评测的预期现象。要比较速度和正确率，请先用 8B AWQ 或减少 `SPEED_REPEATS`。

### KVManage 速度不一定更快

当前压缩逻辑运行在 Python/PyTorch 层，压缩本身有开销。KVManage 的主要收益是把 cache 长度限制在预算内，降低峰值显存并避免 OOM；速度收益取决于 prompt 长度、压缩频率和 GPU 状态。

### AWQ backend 报错

可以先去掉聊天命令中的：

```bash
--awq-version gemm
--awq-backend torch_awq
```

让 Transformers 使用模型配置里的默认量化设置。如果仍失败，确认当前 `transformers`、`torch` 与模型量化格式兼容。

### DeepSeek preflight failed

检查：

- `DEEPSEEK_API_KEY` 是否设置。
- 机器是否能访问 `https://api.deepseek.com`。
- `DEEPSEEK_JUDGE_MODEL` 或 `--model` 是否为可用模型名。

## 10. 部署验收清单

完成部署后，建议依次确认：

1. `python3 -c "import torch; print(torch.cuda.is_available())"` 输出 `True`。
2. 模型目录存在 `config.json`、tokenizer 文件和权重文件。
3. `MODEL_PATH=/root/autodl-tmp/models/Qwen3-32B-AWQ ./chat_awq_chunked.sh` 可以进入对话。
4. `SPEED_CASES=1 ./run_batch_qa_eval.sh` 可以生成 CSV。
5. `python3 summarize_eval.py /root/autodl-tmp/kvcache_outputs/reliability_speed.csv` 可以输出 per-mode 统计。
