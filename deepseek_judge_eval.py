#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """你是严格、稳定、保守的答案评测器。你的任务是根据题目要求、参考答案或判分标准，判断待评测答案是否正确。

判分规则：
1. 只判断语义是否正确，不要求措辞完全一致。
2. 如果答案覆盖核心事实，且没有与参考标准矛盾的明显错误，判 correct=true。
3. 如果答案缺少核心事实、答非所问、出现明显事实错误、或只说了泛泛内容，判 correct=false。
4. 对关键词标准，不要求逐字匹配，但必须覆盖相同语义。
5. 不要因为格式、大小写、中英文标点差异扣分。
6. 只能输出 JSON，不要输出 Markdown 或额外解释。

JSON 格式：
{"correct": true, "score": 1.0, "reason": "简短中文理由"}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge eval CSV answers with DeepSeek API.")
    parser.add_argument("--dataset", type=Path, required=True, help="Original JSONL dataset used for generation.")
    parser.add_argument(
        "--eval-csv",
        type=Path,
        nargs="+",
        required=True,
        help="One or more batch_qa_eval.py output CSV files.",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/deepseek_judge.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/deepseek_judge_summary.csv"))
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="Optional artifacts directory created by batch_qa_eval.py; full outputs are read from here when present.",
    )
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_JUDGE_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--limit", type=int, default=0, help="Judge at most this many unjudged rows.")
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in --output-csv.")
    parser.add_argument("--sleep-sec", type=float, default=0.0, help="Optional delay between API calls.")
    parser.add_argument("--max-answer-chars", type=int, default=5000)
    parser.add_argument("--max-prompt-chars", type=int, default=3000)
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        help="Include a shortened prompt excerpt in the judge request. Expected fields are always included.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the API; write judge prompts as failed rows for inspection.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cases[str(item["id"])] = item
    return cases


def clean_row(raw: dict[str, str]) -> dict[str, str]:
    return {str(key).strip(): value.strip() for key, value in raw.items() if key is not None}


def load_eval_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, skipinitialspace=True)
            for raw in reader:
                row = clean_row(raw)
                if row.get("category") == "summary" or row.get("id", "").startswith("__summary_"):
                    continue
                row["source_csv"] = str(path)
                rows.append(row)
    return rows


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def artifact_output(artifacts_dir: Path | None, case_id: str, mode: str) -> str | None:
    if artifacts_dir is None:
        return None
    path = artifacts_dir / f"{safe_name(case_id)}.{safe_name(mode)}.output.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def shorten(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 3
    tail = max_chars - head
    return text[:head] + "\n...[中间内容已截断]...\n" + text[-tail:]


def expected_spec(case: dict[str, Any]) -> str:
    parts = []
    if case.get("expected_exact"):
        parts.append(f"expected_exact: {case['expected_exact']}")
    if case.get("expected_regex"):
        parts.append(f"expected_regex: {case['expected_regex']}")
    keywords = case.get("expected_keywords") or []
    if keywords:
        parts.append("expected_keywords: " + ", ".join(str(item) for item in keywords))
        parts.append(f"min_keyword_hits: {case.get('min_keyword_hits', len(keywords))}")
    if not parts:
        parts.append("无显式参考答案；请根据题目本身判断。")
    return "\n".join(parts)


def judge_payload(case: dict[str, Any], answer: str, include_prompt: bool, max_prompt_chars: int, max_answer_chars: int) -> str:
    question_part = ""
    if include_prompt:
        question_part = f"\n题目/上下文摘录：\n{shorten(str(case.get('prompt', '')), max_prompt_chars)}\n"

    return f"""请评测下面这个模型答案是否正确。

case_id: {case.get('id', '')}
category: {case.get('category', '')}
{question_part}
参考答案或判分标准：
{expected_spec(case)}

待评测答案：
{shorten(answer, max_answer_chars)}

请只输出 JSON。"""


def parse_judge_json(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    return {
        "correct": bool(data.get("correct", False)),
        "score": float(data.get("score", 1.0 if data.get("correct") else 0.0)),
        "reason": str(data.get("reason", "")),
    }


def make_row_key(row: dict[str, str]) -> str:
    raw = "\t".join([row.get("source_csv", ""), row.get("id", ""), row.get("mode", ""), row.get("output", "")])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            key = row.get("row_key")
            if key:
                keys.add(key)
    return keys


def append_result(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()


def call_deepseek(client: Any, model: str, user_prompt: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content
    return parse_judge_json(content or "{}")


def summary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("source_csv", ""), row.get("mode", ""))].append(row)

    summaries = []
    for (source_csv, mode), items in sorted(grouped.items()):
        judged = [item for item in items if item.get("judge_status") == "ok"]
        correct = sum(1 for item in judged if item.get("judge_correct") == "True")
        scores = []
        for item in judged:
            try:
                scores.append(float(item.get("judge_score", "")))
            except ValueError:
                pass
        summaries.append(
            {
                "source_csv": source_csv,
                "mode": mode,
                "rows": str(len(items)),
                "judged": str(len(judged)),
                "correct": str(correct),
                "accuracy": f"{correct / len(judged):.4f}" if judged else "0.0000",
                "avg_score": f"{sum(scores) / len(scores):.4f}" if scores else "0.0000",
                "judge_errors": str(sum(1 for item in items if item.get("judge_status") != "ok")),
            }
        )
    return summaries


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    summaries = summary_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = ["source_csv", "mode", "rows", "judged", "correct", "accuracy", "avg_score", "judge_errors"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def load_output_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    args = parse_args()
    cases = load_jsonl(args.dataset)
    rows = load_eval_rows(args.eval_csv)
    done = existing_keys(args.output_csv) if args.resume else set()

    client = None
    if not args.dry_run:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit("missing dependency: run `pip install openai` first") from exc
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(f"missing API key: export {args.api_key_env}=your_deepseek_api_key")
        client = OpenAI(api_key=api_key, base_url=args.base_url)

    attempted = 0
    status_counts: Counter[str] = Counter()

    for row in rows:
        if row.get("status") != "ok":
            continue
        case_id = row.get("id", "")
        mode = row.get("mode", "")
        case = cases.get(case_id)
        if case is None:
            print(f"[skip] dataset case not found: {case_id}", flush=True)
            continue

        full_output = artifact_output(args.artifacts_dir, case_id, mode) or row.get("output", "")
        row_for_key = dict(row)
        row_for_key["output"] = full_output
        row_key = make_row_key(row_for_key)
        if row_key in done:
            continue
        if args.limit > 0 and attempted >= args.limit:
            break

        prompt = judge_payload(
            case=case,
            answer=full_output,
            include_prompt=args.include_prompt,
            max_prompt_chars=args.max_prompt_chars,
            max_answer_chars=args.max_answer_chars,
        )

        judge_status = "ok"
        judge_correct = False
        judge_score = 0.0
        judge_reason = ""
        error = ""
        try:
            if args.dry_run:
                judge_status = "dry_run"
                judge_reason = shorten(prompt, 1000)
            else:
                assert client is not None
                judged = call_deepseek(client, args.model, prompt)
                judge_correct = bool(judged["correct"])
                judge_score = float(judged["score"])
                judge_reason = str(judged["reason"])
        except Exception as exc:
            judge_status = "error"
            error = f"{type(exc).__name__}: {exc}"

        out = {
            "row_key": row_key,
            "source_csv": row.get("source_csv", ""),
            "dataset": str(args.dataset),
            "id": case_id,
            "mode": mode,
            "category": row.get("category", ""),
            "generation_status": row.get("status", ""),
            "original_ok": row.get("ok", ""),
            "original_score_type": row.get("score_type", ""),
            "judge_status": judge_status,
            "judge_model": args.model,
            "judge_correct": str(judge_correct),
            "judge_score": f"{judge_score:.4f}",
            "judge_reason": judge_reason.replace("\n", " ")[:700],
            "prompt_tokens": row.get("prompt_tokens", ""),
            "generated_tokens": row.get("generated_tokens", ""),
            "elapsed_sec": row.get("elapsed_sec", ""),
            "peak_memory_gb": row.get("peak_memory_gb", ""),
            "error": error.replace("\n", " ")[:700],
            "answer": full_output.replace("\n", " ")[:1200],
        }
        append_result(args.output_csv, out)
        attempted += 1
        status_counts[judge_status] += 1
        print(
            f"[judge {attempted}] {case_id} mode={mode} status={judge_status} "
            f"correct={judge_correct} score={judge_score:.2f}",
            flush=True,
        )
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    output_rows = load_output_rows(args.output_csv)
    write_summary(args.summary_csv, output_rows)
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.summary_csv}")
    if status_counts:
        print("judge_status:", dict(status_counts))


if __name__ == "__main__":
    main()
