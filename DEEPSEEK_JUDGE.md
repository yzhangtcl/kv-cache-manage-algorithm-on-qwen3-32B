# DeepSeek API 作为答案裁判

本项目原来的 `batch_qa_eval.py` 使用 `expected_regex`、`expected_exact`、`expected_keywords` 自动判分。这个方法快，但对长答案比较粗。老师说的 DeepSeek V4 Flash API 可以作为 LLM 裁判：你的 KV cache 管理算法和滑动窗口 baseline 先各自生成答案，然后把答案交给 DeepSeek 判断是否正确，最后比较两个方法的准确率。

## 1. 获取 DeepSeek API Key

1. 打开 DeepSeek 开放平台：https://platform.deepseek.com/
2. 注册或登录账号。
3. 进入 API keys 页面：https://platform.deepseek.com/api_keys
4. 点击创建 API key。
5. 复制 key。这个 key 只显示一次，建议保存到本机环境变量或服务器密钥管理里，不要写进代码或提交到 git。
6. 如果账号没有余额，进入充值/计费页面充值。API 文档和价格说明见：https://api-docs.deepseek.com/

DeepSeek API 使用 OpenAI 兼容调用方式：

- `base_url`: `https://api.deepseek.com`
- API key 环境变量：本项目默认读 `DEEPSEEK_API_KEY`
- judge 模型：老师指定 `deepseek-v4-flash`，脚本默认也使用这个名字。如果控制台或文档里的模型名有变化，用 `--model` 覆盖即可。

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

或者只安装 API SDK：

```bash
pip install openai
```

## 3. 配置 API Key

Linux/macOS：

```bash
export DEEPSEEK_API_KEY="你的DeepSeek API Key"
```

如果你的模型名要换，可以同时设置：

```bash
export DEEPSEEK_JUDGE_MODEL="deepseek-v4-flash"
```

如果临时只想对单条命令生效：

```bash
DEEPSEEK_API_KEY="你的DeepSeek API Key" python3 deepseek_judge_eval.py --help
```

## 4. 先 dry-run 检查裁判 prompt

dry-run 不会调用 API，也不会花钱，只会把准备发给 DeepSeek 的裁判输入写到 CSV 里，适合先检查格式。

```bash
python3 deepseek_judge_eval.py \
  --dataset datasets/reliability_speed.jsonl \
  --eval-csv speed4.csv \
  --output-csv outputs/deepseek_judge_dry_run.csv \
  --summary-csv outputs/deepseek_judge_dry_run_summary.csv \
  --limit 3 \
  --dry-run
```

## 5. 对已有 speed 结果进行 DeepSeek 判分

```bash
python3 deepseek_judge_eval.py \
  --dataset datasets/reliability_speed.jsonl \
  --eval-csv speed4.csv \
  --output-csv outputs/deepseek_judge_speed4.csv \
  --summary-csv outputs/deepseek_judge_speed4_summary.csv \
  --resume
```

如果你同时想评多个 CSV：

```bash
python3 deepseek_judge_eval.py \
  --dataset datasets/reliability_speed.jsonl \
  --eval-csv speed.csv speed2.csv speed3.csv speed4.csv \
  --output-csv outputs/deepseek_judge_speed_all.csv \
  --summary-csv outputs/deepseek_judge_speed_all_summary.csv \
  --resume
```

## 6. 对 OOM 压测结果判分

OOM CSV 里 full 模式通常是 `status=oom`，脚本只会评 `status=ok` 的答案，所以会主要评 kvmanage 成功生成的答案。

```bash
python3 deepseek_judge_eval.py \
  --dataset datasets/oom_stress.jsonl \
  --eval-csv oom.csv \
  --output-csv outputs/deepseek_judge_oom.csv \
  --summary-csv outputs/deepseek_judge_oom_summary.csv \
  --resume
```

## 7. 如果有完整输出 artifacts

`batch_qa_eval.py` 的 CSV 里 `output` 字段只保存前 700 个字符。如果你运行评测时用了 `--artifacts-dir`，脚本可以读取完整答案：

```bash
python3 deepseek_judge_eval.py \
  --dataset datasets/reliability_speed.jsonl \
  --eval-csv /root/autodl-tmp/kvcache_outputs/reliability_speed.csv \
  --artifacts-dir /root/autodl-tmp/kvcache_outputs/reliability_speed \
  --output-csv outputs/deepseek_judge_reliability_speed.csv \
  --summary-csv outputs/deepseek_judge_reliability_speed_summary.csv \
  --resume
```

没有 `--artifacts-dir` 也可以跑，脚本会使用 CSV 里的 `output` 字段。

## 8. 输出文件怎么看

`outputs/deepseek_judge_*.csv` 是逐条判分结果，关键字段：

- `id`: case id。
- `mode`: 被评测的方法，例如 `full`、`kvmanage`，以后滑动窗口 baseline 可以写成 `sliding_window`。
- `original_ok`: 原来的关键词/正则判分结果。
- `judge_correct`: DeepSeek 裁判判断是否正确。
- `judge_score`: 0 到 1 的分数。
- `judge_reason`: 简短判分理由。
- `elapsed_sec`、`peak_memory_gb`: 原实验的耗时和显存，方便同表追溯。

`outputs/deepseek_judge_*_summary.csv` 是按 `source_csv + mode` 聚合的准确率：

- `judged`: 成功判分的数量。
- `correct`: DeepSeek 判正确的数量。
- `accuracy`: DeepSeek 判分准确率。
- `avg_score`: 平均分。
- `judge_errors`: API 调用或解析失败数量。

最终汇报时主要比较：

```text
kvmanage accuracy vs sliding_window accuracy
kvmanage peak_memory_gb vs sliding_window peak_memory_gb
kvmanage elapsed_sec vs sliding_window elapsed_sec
```

准确率用 DeepSeek judge，速度和显存用原评测 CSV。

## 9. 推荐实验流程

1. 准备同一批测试数据，例如 `datasets/reliability_speed.jsonl`。
2. 运行你的 KV cache 管理算法，得到 CSV。
3. 运行滑动窗口 baseline，得到同格式 CSV，`mode` 最好标为 `sliding_window`。
4. 用 `deepseek_judge_eval.py` 分别或一起判分。
5. 用 summary CSV 比较准确率。
6. 再结合 speed 图、OOM 图说明显存和速度收益。

注意：不要在裁判 prompt 里告诉 DeepSeek “这个答案来自我的算法” 或 “这个答案来自 baseline”。脚本只提供 `mode` 到输出 CSV，不会把方法名写进裁判 prompt，避免裁判偏向某一种方法。

## 10. 一键对比 KVManage 和滑动窗口

你的方法参数：

```text
--max-cache-tokens 4096
--recent-window 2048
--hot-cache-tokens 1536
```

滑动窗口 baseline 参数：

```text
--max-cache-tokens 4096
--recent-window 4096
--hot-cache-tokens 0
--hot-raw-tokens 0
```

这等价于同样最多保留 4096 个 KV token，但 baseline 只保留最近 4096 个 token，不保留 hot token，也不做 hot/cold 选择。

已经提供一键脚本：

```bash
export DEEPSEEK_API_KEY="你的DeepSeek API Key"

./run_kvmanage_vs_sliding_accuracy.sh
```

默认输出目录：

```text
/root/autodl-tmp/kvcache_outputs/accuracy_compare
```

关键输出：

- `kvmanage.csv`：你的 KVManage 生成结果。
- `sliding_window.csv`：滑动窗口 baseline 生成结果。
- `deepseek_judge.csv`：DeepSeek 对每条答案的逐条判分。
- `deepseek_judge_summary.csv`：按方法聚合后的准确率。

看准确率：

```bash
cat /root/autodl-tmp/kvcache_outputs/accuracy_compare/deepseek_judge_summary.csv
```

如果你只想先跑少量样本确认流程：

```bash
SPEED_CASES=5 ./run_kvmanage_vs_sliding_accuracy.sh
```

如果完整 prompt 太长导致 API 上下文超限，可以截断 prompt：

```bash
MAX_PROMPT_CHARS=16000 ./run_kvmanage_vs_sliding_accuracy.sh
```

默认 `MAX_PROMPT_CHARS=0` 表示尽量发送完整 prompt 给 DeepSeek 裁判。
