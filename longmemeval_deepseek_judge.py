#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a strict and conservative evaluator for LongMemEval answers.
Judge whether the model response satisfies the reference answer or rubric.
Return JSON only: {"correct": true, "score": 1.0, "reason": "short reason"}."""


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_references(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["question_id"]): row for row in load_records(path)}


def shorten(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 3
    tail = max_chars - head
    return text[:head] + "\n...[truncated]...\n" + text[-tail:]


def task_instruction(question_id: str, question_type: str) -> str:
    if "_abs" in question_id:
        return (
            "This is an unanswerable question. Mark correct only if the response "
            "clearly says the answer cannot be determined from the provided memory, "
            "or explains that the needed information is missing."
        )
    if question_type == "temporal-reasoning":
        return (
            "Mark correct if the response contains the correct answer or equivalent reasoning. "
            "For day/week/month count questions, tolerate off-by-one counting errors."
        )
    if question_type == "knowledge-update":
        return (
            "Mark correct if the response gives the updated answer. Do not penalize it "
            "for also mentioning older information, as long as the updated answer is clear."
        )
    if question_type == "single-session-preference":
        return (
            "The reference is a rubric for a personalized response. Mark correct if the "
            "response recalls and uses the user's personal information appropriately; "
            "it does not need to cover every rubric point."
        )
    return (
        "Mark correct if the response contains the correct answer. Equivalent wording "
        "or sufficient intermediate reasoning is acceptable. Mark incorrect if it only "
        "contains a subset of the required information."
    )


def judge_prompt(reference: dict[str, Any], hypothesis: str, max_answer_chars: int) -> str:
    question_id = str(reference.get("question_id", ""))
    question_type = str(reference.get("question_type", ""))
    answer_label = "Explanation/Rubric" if "_abs" in question_id else "Correct Answer"
    return f"""Evaluate the model response for a LongMemEval question.

Question ID: {question_id}
Question Type: {question_type}
Instruction: {task_instruction(question_id, question_type)}

Question:
{reference.get("question", "")}

{answer_label}:
{reference.get("answer", "")}

Model Response:
{shorten(hypothesis, max_answer_chars)}

Return JSON only with fields correct, score, reason."""


def parse_judge_json(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            lowered = text.lower()
            correct = "yes" in lowered and "no" not in lowered
            data = {"correct": correct, "score": 1.0 if correct else 0.0, "reason": text}
    correct = bool(data.get("correct", False))
    return {
        "correct": correct,
        "score": float(data.get("score", 1.0 if correct else 0.0)),
        "reason": str(data.get("reason", "")),
    }


def call_deepseek(client: Any, model: str, prompt: str, retries: int, sleep_sec: float) -> dict[str, Any]:
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return parse_judge_json(response.choices[0].message.content or "{}")
        except Exception as exc:  # noqa: BLE001 - preserve API error text in CSV.
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(max(sleep_sec, 1.0) * (attempt + 1))
    raise last_error  # type: ignore[misc]


def row_key(hyp_file: Path, question_id: str, hypothesis: str) -> str:
    raw = "\t".join([str(hyp_file), question_id, hypothesis])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        return {row["row_key"] for row in csv.DictReader(fh) if row.get("row_key")}


def append_csv(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("mode", ""), row.get("question_type", ""))].append(row)

    fieldnames = ["mode", "question_type", "rows", "judged", "correct", "accuracy", "avg_score", "errors"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for (mode, qtype), items in sorted(grouped.items()):
            judged = [item for item in items if item.get("judge_status") == "ok"]
            correct = sum(1 for item in judged if item.get("judge_correct") == "True")
            scores = []
            for item in judged:
                try:
                    scores.append(float(item.get("judge_score", "")))
                except ValueError:
                    pass
            writer.writerow(
                {
                    "mode": mode,
                    "question_type": qtype,
                    "rows": str(len(items)),
                    "judged": str(len(judged)),
                    "correct": str(correct),
                    "accuracy": f"{correct / len(judged):.4f}" if judged else "0.0000",
                    "avg_score": f"{sum(scores) / len(scores):.4f}" if scores else "0.0000",
                    "errors": str(sum(1 for item in items if item.get("judge_status") != "ok")),
                }
            )

        all_judged = [item for item in rows if item.get("judge_status") == "ok"]
        all_correct = sum(1 for item in all_judged if item.get("judge_correct") == "True")
        all_scores = [float(item["judge_score"]) for item in all_judged if item.get("judge_score")]
        writer.writerow(
            {
                "mode": "__all__",
                "question_type": "__all__",
                "rows": str(len(rows)),
                "judged": str(len(all_judged)),
                "correct": str(all_correct),
                "accuracy": f"{all_correct / len(all_judged):.4f}" if all_judged else "0.0000",
                "avg_score": f"{sum(all_scores) / len(all_scores):.4f}" if all_scores else "0.0000",
                "errors": str(sum(1 for item in rows if item.get("judge_status") != "ok")),
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge LongMemEval hypotheses with DeepSeek API.")
    parser.add_argument("--reference", type=Path, required=True, help="LongMemEval reference JSON/JSONL.")
    parser.add_argument("--hypothesis", type=Path, nargs="+", required=True, help="Generated hypothesis jsonl files.")
    parser.add_argument("--mode-labels", nargs="+", help="Optional labels matching --hypothesis.")
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/longmemeval_deepseek_judge.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/longmemeval_deepseek_summary.csv"))
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_JUDGE_MODEL", "deepseek-chat"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-answer-chars", type=int, default=6000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode_labels and len(args.mode_labels) != len(args.hypothesis):
        raise SystemExit("--mode-labels must have the same length as --hypothesis")

    references = load_references(args.reference)
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
    for file_index, hyp_file in enumerate(args.hypothesis):
        mode = args.mode_labels[file_index] if args.mode_labels else hyp_file.stem
        for hyp_row in load_records(hyp_file):
            question_id = str(hyp_row.get("question_id", ""))
            hypothesis = str(hyp_row.get("hypothesis", ""))
            reference = references.get(question_id)
            if reference is None:
                print(f"[skip] reference not found: {question_id}", flush=True)
                continue

            key = row_key(hyp_file, question_id, hypothesis)
            if key in done:
                continue
            if args.limit > 0 and attempted >= args.limit:
                break

            prompt = judge_prompt(reference, hypothesis, args.max_answer_chars)
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
                    judged = call_deepseek(client, args.model, prompt, args.retries, args.sleep_sec)
                    judge_correct = bool(judged["correct"])
                    judge_score = float(judged["score"])
                    judge_reason = str(judged["reason"])
            except Exception as exc:  # noqa: BLE001 - write judge failure and continue.
                judge_status = "error"
                error = f"{type(exc).__name__}: {exc}"
                if args.fail_fast:
                    raise SystemExit(f"judge failed at {question_id} mode={mode}: {error}") from exc

            out = {
                "row_key": key,
                "mode": mode,
                "hypothesis_file": str(hyp_file),
                "question_id": question_id,
                "question_type": str(reference.get("question_type", "")),
                "judge_status": judge_status,
                "judge_model": args.model,
                "judge_correct": str(judge_correct),
                "judge_score": f"{judge_score:.4f}",
                "judge_reason": judge_reason.replace("\n", " ")[:1000],
                "error": error.replace("\n", " ")[:1000],
                "question": str(reference.get("question", "")).replace("\n", " ")[:800],
                "answer": str(reference.get("answer", "")).replace("\n", " ")[:800],
                "hypothesis": hypothesis.replace("\n", " ")[:1200],
            }
            append_csv(args.output_csv, out)
            attempted += 1
            print(
                f"[judge {attempted}] {question_id} mode={mode} "
                f"status={judge_status} correct={judge_correct} score={judge_score:.2f}",
                flush=True,
            )
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

        if args.limit > 0 and attempted >= args.limit:
            break

    rows = load_csv(args.output_csv)
    write_summary(args.summary_csv, rows)
    print(f"wrote {args.output_csv}", flush=True)
    print(f"wrote {args.summary_csv}", flush=True)


if __name__ == "__main__":
    main()
