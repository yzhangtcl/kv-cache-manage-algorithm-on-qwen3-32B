# 4090 服务器运行说明

目标：在单张 RTX 4090 上尝试运行一个常规长上下文很容易 OOM 的模型，并用本目录里的 KV cache 预算/压缩策略降低显存占用，再用小型质量集检查输出有没有明显退化。

## 重要边界

`kv_cache_sim.py` 是 NumPy 模拟器，用来验证策略趋势。真实 LLM 的 KV cache 是每层、每头、带位置编码语义的张量，不能把模拟器里的 CPU/GPU/offload 逻辑原样当成生产内核。

新增的真实推理脚本采用近似策略：

- prompt 分块 prefill，避免一次性构造完整长上下文 KV。
- 保留最近 `recent_window` 个 KV token。
- 对更老 KV 先分 hot/cold：少量 hot old KV 原样保留，重复 hot old KV 去重合并，cold old KV 按 key 相似度合并成均值代表。
- 每次 cache 超过 `max_cache_tokens` 后压缩。

这能用于服务器实验和论文/报告里的原型结果，但不是 vLLM/TensorRT-LLM 级别的高性能内核。

## 推荐模型

默认模型：

```bash
Qwen/Qwen2.5-32B-Instruct
```

选择原因：32B 参数在 fp16 约需要 64GB 权重显存，单张 24GB 4090 不能常规完整加载；即使通过 `device_map=auto` 把一部分权重放到 CPU，长上下文 KV cache 也很容易 OOM。用本实验的 KV cache 预算可以尝试让长 prompt 的生成跑起来。

如果服务器 CPU 内存不够，先用下面的小模型做冒烟测试：

```bash
Qwen/Qwen2.5-7B-Instruct
```

## 1. 上传代码

在本机进入仓库目录：

```bash
cd /home/yzhang/kvcachemodel
tar czf kvcachemodel.tar.gz .
scp kvcachemodel.tar.gz USER@SERVER:/path/to/workdir/
```

在服务器上：

```bash
cd /path/to/workdir
mkdir -p kvcachemodel
tar xzf kvcachemodel.tar.gz -C kvcachemodel
cd kvcachemodel
```

## 2. 创建 Python 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-server.txt
```

如果服务器 CUDA/PyTorch 版本不匹配，到 PyTorch 官网选择对应 CUDA 版本安装，例如 CUDA 12.1：

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -r requirements-server.txt
```

检查 GPU：

```bash
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
PY
```

## 3. 登录 Hugging Face

如果模型需要授权或下载速度慢，先设置 token：

```bash
pip install huggingface_hub
huggingface-cli login
```

也可以设置缓存目录：

```bash
export HF_HOME=/path/to/big_disk/hf_home
```

## 4. 先跑模拟器

```bash
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
```

你应该看到 `recent`、`hot`、`hot_cluster`、`hybrid_cluster`、`adaptive_hybrid_cluster` 的质量和估算 speedup 表格。

## 5. 生成长 prompt

```bash
python3 scripts/make_long_prompt.py --repeats 900 --output long_prompt.txt
```

如果想更容易 OOM，把 `--repeats` 调大；如果 32B 模型太慢，先调到 `200`。

## 6. 冒烟测试

先用 7B 验证脚本逻辑：

```bash
python3 run_server_experiment.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --prompt-file long_prompt.txt \
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
  --log-every 2048 \
  --output-file outputs/qwen7b_budgeted_kv.txt
```

查看输出：

```bash
cat outputs/qwen7b_budgeted_kv.txt
```

重点看：

- `peak_memory_gb`
- `prompt_tokens`
- `avg_kept_cache_tokens`
- `output`

## 7. 跑 32B 实验

```bash
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
```

如果显存还是不够，按顺序尝试：

```bash
--max-cache-tokens 1536 --recent-window 1024
```

再不够：

```bash
--max-cache-tokens 1024 --recent-window 768 --prefill-chunk-tokens 256
```

如果 CPU 内存不够，32B 权重 offload 本身也可能失败，这时不是 KV cache 的问题，需要更多 CPU 内存、更小模型，或 4bit 权重量化加载。

## 8. 质量测试

短上下文 exact baseline 能跑时：

```bash
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
```

如果 exact baseline OOM，只跑压缩版本：

```bash
python3 evaluate_quality.py \
  --model Qwen/Qwen2.5-32B-Instruct \
  --dtype float16 \
  --max-gpu-memory 22GiB \
  --max-cpu-memory 96GiB \
  --offload-folder offload \
  --skip-exact \
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
```

看结果：

```bash
column -s, -t < outputs/qwen32b_quality.csv | less -S
```

关注这些列：

- `budgeted_ok`: 是否包含标准答案。
- `budgeted_f1`: 输出与标准答案的 token F1。
- `exact_ok` / `exact_f1`: exact baseline 成功时用于对照。
- `budgeted_peak_gb`: 压缩 KV 的峰值显存。
- `exact_peak_gb`: exact baseline 的峰值显存。

批量 long-context needle 测试：

```bash
python3 batch_needle_eval.py \
  --model /root/autodl-tmp/models/Qwen3-32B \
  --dtype float16 \
  --max-gpu-memory 18GiB \
  --max-cpu-memory 110GiB \
  --offload-folder /root/autodl-tmp/offload \
  --cases 12 \
  --repeats 200 \
  --max-new-tokens 16 \
  --prefill-chunk-tokens 256 \
  --max-cache-tokens 1024 \
  --recent-window 768 \
  --hot-cache-tokens -1 \
  --hot-raw-tokens -1 \
  --merge-similarity 0.82 \
  --attention-decay 0.98 \
  --importance-update 0.05 \
  --output-csv /root/autodl-tmp/kvcache_outputs/batch_needle_eval.csv \
  --resume
```

这个脚本每完成一条 case 就追加写入 CSV，中途停止后加 `--resume` 会跳过已完成 case。先用 `--cases 3 --repeats 80` 做冒烟测试，再扩大到 `--cases 12 --repeats 200`。

通用 QA / 推理 / 长文综合测试：

```bash
python3 scripts/build_eval_datasets.py --output-dir datasets
```

会生成：

- `datasets/short_qa.jsonl`: 短问答，主要检查基础问答能力。
- `datasets/reasoning.jsonl`: 简单多步推理。
- `datasets/long_qa.jsonl`: 长文档综合问答。
- `datasets/recall_qa.jsonl`: 长文档事实召回。
- `datasets/all_qa.jsonl`: 上面所有 case 合并。

先跑 3 条冒烟测试：

```bash
python3 batch_qa_eval.py \
  --model /root/autodl-tmp/models/Qwen3-32B \
  --dataset datasets/all_qa.jsonl \
  --dtype float16 \
  --max-gpu-memory 18GiB \
  --max-cpu-memory 110GiB \
  --offload-folder /root/autodl-tmp/offload \
  --limit 3 \
  --prefill-chunk-tokens 256 \
  --max-cache-tokens 2048 \
  --recent-window 1024 \
  --hot-cache-tokens 768 \
  --hot-raw-tokens -1 \
  --merge-similarity 0.90 \
  --attention-decay 0.995 \
  --importance-update 0.02 \
  --output-csv /root/autodl-tmp/kvcache_outputs/batch_qa_eval.csv \
  --artifacts-dir /root/autodl-tmp/kvcache_outputs/qa_artifacts \
  --resume
```

`batch_qa_eval.py` 默认会使用 tokenizer 的 chat template 包装 prompt，这对 Qwen3 这类聊天模型很重要。如果之前裸 prompt 跑出一串 `!!!!!!!!`，那不是 KV 压缩问题，而是输入格式不对。修正后建议换一个新 CSV 路径，或删除旧 CSV 再跑，避免 `--resume` 跳过旧失败 case。

再分别跑完整小套件：

```bash
for ds in short_qa reasoning long_qa recall_qa; do
  python3 batch_qa_eval.py \
    --model /root/autodl-tmp/models/Qwen3-32B \
    --dataset datasets/${ds}.jsonl \
    --dtype float16 \
    --max-gpu-memory 18GiB \
    --max-cpu-memory 110GiB \
    --offload-folder /root/autodl-tmp/offload \
    --prefill-chunk-tokens 256 \
    --max-cache-tokens 2048 \
    --recent-window 1024 \
    --hot-cache-tokens 768 \
    --hot-raw-tokens -1 \
    --merge-similarity 0.90 \
    --attention-decay 0.995 \
    --importance-update 0.02 \
    --output-csv /root/autodl-tmp/kvcache_outputs/${ds}_qa_eval.csv \
    --artifacts-dir /root/autodl-tmp/kvcache_outputs/${ds}_artifacts \
    --resume
done
```

查看正确率：

```bash
python3 - <<'PY'
import csv, glob
for path in glob.glob("/root/autodl-tmp/kvcache_outputs/*_qa_eval.csv"):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    ok = sum(r["ok"] == "True" for r in rows)
    print(path, f"{ok}/{len(rows)}", f"{ok / max(1, len(rows)):.2%}")
PY
```

每条 case 的完整 prompt 和 output 会保存在 `--artifacts-dir`，适合人工抽查。

## 9. 一键脚本

环境配好后可以直接跑：

```bash
bash scripts/run_4090_qwen32b.sh
```

建议第一次不要一键跑，先按上面的 4 到 8 步逐个验证，这样能更快定位是模型下载、显存、CPU 内存还是脚本参数问题。

## 10. 实验报告建议

至少记录这些信息：

- GPU 型号和显存：`nvidia-smi`
- PyTorch/Transformers 版本。
- 模型名和 dtype。
- prompt tokens、max new tokens。
- `max_cache_tokens`、`recent_window`、`merge_similarity`。
- exact baseline 是否 OOM。
- 优化版本峰值显存、速度、输出。
- 质量 CSV 的 `budgeted_ok`、`budgeted_f1`，以及 exact baseline 可用时的差距。
