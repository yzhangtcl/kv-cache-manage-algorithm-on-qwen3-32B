#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")


def short_qa() -> list[dict]:
    return [
        {
            "id": "short_qa_earth_seasons",
            "category": "short_qa",
            "prompt": "Question: Why does Earth have seasons? Answer in two concise sentences.",
            "expected_keywords": ["tilt", "axis", "sunlight"],
            "min_keyword_hits": 2,
            "max_new_tokens": 64,
            "stop_after_sentences": 2,
        },
        {
            "id": "short_qa_photosynthesis",
            "category": "short_qa",
            "prompt": "Question: What is photosynthesis? Answer in two concise sentences.",
            "expected_keywords": ["light", "carbon dioxide", "glucose", "oxygen"],
            "min_keyword_hits": 2,
            "max_new_tokens": 64,
            "stop_after_sentences": 2,
        },
        {
            "id": "short_qa_http_404",
            "category": "short_qa",
            "prompt": "Question: What does HTTP status code 404 mean? Answer in one sentence.",
            "expected_keywords": ["not found", "resource", "server"],
            "min_keyword_hits": 1,
            "max_new_tokens": 48,
            "stop_after_sentences": 1,
        },
        {
            "id": "short_qa_overfitting",
            "category": "short_qa",
            "prompt": "Question: In machine learning, what is overfitting? Answer in two concise sentences.",
            "expected_keywords": ["training", "generalize", "unseen", "data"],
            "min_keyword_hits": 2,
            "max_new_tokens": 64,
            "stop_after_sentences": 2,
        },
    ]


def reasoning() -> list[dict]:
    return [
        {
            "id": "reasoning_boxes",
            "category": "reasoning",
            "prompt": (
                "Solve the problem and give the final answer clearly. "
                "Maya has 3 boxes with 4 blue pens in each box and 2 loose blue pens. "
                "How many blue pens does Maya have?"
            ),
            "expected_regex": r"\b14\b",
            "max_new_tokens": 64,
            "stop_after_sentences": 2,
        },
        {
            "id": "reasoning_train",
            "category": "reasoning",
            "prompt": (
                "Solve the problem and give the final answer clearly. "
                "A train travels 45 miles per hour for 2 hours, then 30 miles per hour for 1 hour. "
                "What total distance does it travel?"
            ),
            "expected_regex": r"\b120\b",
            "max_new_tokens": 64,
            "stop_after_sentences": 2,
        },
        {
            "id": "reasoning_apples",
            "category": "reasoning",
            "prompt": (
                "Solve the problem and give the final answer clearly. "
                "There are 18 apples. Sam eats 5, then buys 9 more. How many apples are there?"
            ),
            "expected_regex": r"\b22\b",
            "max_new_tokens": 64,
            "stop_after_sentences": 2,
        },
        {
            "id": "reasoning_schedule",
            "category": "reasoning",
            "prompt": (
                "Solve the problem and give the final answer clearly. "
                "A meeting starts at 9:15 and lasts 95 minutes. What time does it end?"
            ),
            "expected_regex": r"\b10:50\b|\bten fifty\b",
            "max_new_tokens": 64,
            "stop_after_sentences": 2,
        },
    ]


def long_doc() -> str:
    sections = [
        (
            "Design Overview",
            "The cache policy protects a recent window because nearby tokens dominate many decoder attention patterns. "
            "Older tokens are split into hot and cold groups so that important long-range information can survive compression.",
        ),
        (
            "Hot Cache",
            "Hot entries are selected using an importance score with age decay. "
            "A small number of hot anchors are retained exactly, while redundant hot entries are merged into representatives.",
        ),
        (
            "Cold Cache",
            "Cold entries are lower-priority old tokens. They are clustered by key similarity and represented by averaged key/value tensors.",
        ),
        (
            "Risks",
            "Merging key/value tensors is approximate for RoPE-based models. It can harm recall, trigger repetition, or distort long-range reasoning.",
        ),
        (
            "Evaluation",
            "The system should be evaluated with short question answering, multi-step reasoning, long-document synthesis, and targeted recall tests.",
        ),
    ]
    noise = []
    for i in range(160):
        title, body = sections[i % len(sections)]
        noise.append(f"Section {i:03d} - {title}: {body}")
    return "\n".join(noise)


def long_qa() -> list[dict]:
    doc = long_doc()
    return [
        {
            "id": "long_qa_tradeoffs",
            "category": "long_qa",
            "prompt": (
                f"Document:\n{doc}\n\n"
                "Question: According to the document, what are two main risks of merging KV cache entries? "
                "Answer in two sentences."
            ),
            "expected_keywords": ["RoPE", "recall", "repetition", "reasoning"],
            "min_keyword_hits": 2,
            "max_new_tokens": 96,
            "stop_after_sentences": 2,
        },
        {
            "id": "long_qa_eval_types",
            "category": "long_qa",
            "prompt": (
                f"Document:\n{doc}\n\n"
                "Question: What kinds of evaluations does the document recommend? Answer concisely."
            ),
            "expected_keywords": ["short question answering", "reasoning", "long-document", "recall"],
            "min_keyword_hits": 2,
            "max_new_tokens": 96,
            "stop_after_sentences": 2,
        },
        {
            "id": "long_qa_hot_cache",
            "category": "long_qa",
            "prompt": (
                f"Document:\n{doc}\n\n"
                "Question: How does the document describe the hot cache? Answer in two sentences."
            ),
            "expected_keywords": ["importance", "age decay", "anchors", "merged"],
            "min_keyword_hits": 2,
            "max_new_tokens": 96,
            "stop_after_sentences": 2,
        },
    ]


def recall_qa() -> list[dict]:
    rows = []
    projects = [
        ("Atlas", "blue lane", "113"),
        ("Borealis", "amber key", "227"),
        ("Cygnus", "vault seven", "389"),
        ("Dione", "silver mark", "431"),
    ]
    for idx, (name, phrase, code) in enumerate(projects):
        lines = [
            "Operations archive. Use the archive to answer the final question.",
            "",
        ]
        for i in range(220):
            if i == 30 + idx * 40:
                lines.append(f"Critical fact: project {name} uses {phrase} and code {code}.")
            else:
                other = projects[(idx + i + 1) % len(projects)]
                lines.append(f"Archive line {i:03d}: project {other[0]} has routine marker {other[2]}.")
        lines.append("")
        lines.append(f"Question: Which code belongs to project {name}? Answer with the code and a short justification.")
        rows.append(
            {
                "id": f"recall_{name.lower()}",
                "category": "recall_qa",
                "prompt": "\n".join(lines),
                "expected_regex": rf"\b{code}\b",
                "max_new_tokens": 48,
                "stop_after_sentences": 1,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local JSONL evaluation datasets.")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suites = {
        "short_qa.jsonl": short_qa(),
        "reasoning.jsonl": reasoning(),
        "long_qa.jsonl": long_qa(),
        "recall_qa.jsonl": recall_qa(),
    }
    all_rows = []
    for filename, rows in suites.items():
        write_jsonl(args.output_dir / filename, rows)
        all_rows.extend(rows)
    write_jsonl(args.output_dir / "all_qa.jsonl", all_rows)


if __name__ == "__main__":
    main()
