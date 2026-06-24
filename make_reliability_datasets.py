#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def repeated_policy_prompt(repeats: int, label: str = "standard") -> str:
    blocks = []
    for idx in range(repeats):
        blocks.append(
            "\n".join(
                [
                    f"Block {idx:04d} [{label}]: Budgeted KV cache keeps a recent window for local context.",
                    "The same block says hot tokens are retained when they carry repeated importance.",
                    "Cold tokens may be merged into representative key/value states.",
                    "The reliability risk is that compression is approximate, so evaluation must check quality.",
                ]
            )
        )
    return (
        "Read the repeated engineering notes and answer the final question.\n\n"
        + "\n".join(blocks)
        + "\n\nQuestion: Summarize the cache policy and the main reliability risk in three bullets."
    )


def repeated_ops_prompt(repeats: int, label: str = "standard") -> str:
    projects = [
        ("Atlas", "113", "daily backups"),
        ("Borealis", "227", "index audits"),
        ("Cygnus", "389", "latency checks"),
        ("Dione", "431", "capacity reviews"),
    ]
    lines = []
    for idx in range(repeats):
        name, code, routine = projects[idx % len(projects)]
        lines.append(
            f"Archive line {idx:04d} [{label}]: project {name} uses code {code} and routine {routine}. "
            f"The repeated project record says {name} keeps the same code {code} across reviews. "
            f"Operators should answer from the repeated pattern, not from a single rare line."
        )
    return (
        "Operations archive. Each project fact is repeated many times; answer from the repeated pattern.\n\n"
        + "\n".join(lines)
        + "\n\nQuestion: List the code and routine for Atlas, Cygnus, and Dione."
    )


def repeated_oom_prompt(repeats: int, label: str) -> str:
    blocks = []
    for idx in range(repeats):
        blocks.append(
            "\n".join(
                [
                    f"Stress record {idx:04d} [{label}]: The approved deployment verdict is reliable-with-monitoring.",
                    "Evidence repeated here: short QA remains stable, long summaries retain policy facts, and memory drops.",
                    "The accepted caveat is that exact rare-token recall should not be the only benchmark.",
                    "The action item is to compare full KV against kvmanage under the same prompt.",
                ]
            )
        )
    return (
        "Large repeated evaluation file. The final question asks for conclusions that appear throughout the file.\n\n"
        + "\n".join(blocks)
        + "\n\nQuestion: What is the deployment verdict, what evidence supports it, and what caveat should be reported?"
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create repeated-fact reliability datasets.")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--speed-cases", type=int, default=10)
    parser.add_argument(
        "--speed-repeats",
        type=int,
        default=220,
        help="Base repeat count for 8k-ish speed cases; later cases get longer contexts.",
    )
    parser.add_argument("--oom-repeats", type=int, default=360)
    parser.add_argument("--oom-cases", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    speed_rows = []
    for case_idx in range(args.speed_cases):
        level = case_idx % 3
        family_idx = case_idx // 3
        suffix = ["16k", "20k", "24k"][level]
        repeats = args.speed_repeats + level * 70 + family_idx * 8
        if case_idx % 2 == 0:
            speed_rows.append(
                {
                    "id": f"speed_policy_summary_{case_idx + 1:03d}_{suffix}",
                    "category": "speed_long",
                    "prompt": repeated_policy_prompt(repeats, label=f"{suffix}_{case_idx + 1:03d}"),
                    "expected_keywords": ["recent", "hot", "cold", "merged", "risk"],
                    "min_keyword_hits": 4,
                    "max_new_tokens": 160,
                    "stop_after_sentences": 0,
                }
            )
        else:
            speed_rows.append(
                {
                    "id": f"speed_ops_summary_{case_idx + 1:03d}_{suffix}",
                    "category": "speed_long",
                    "prompt": repeated_ops_prompt(repeats, label=f"{suffix}_{case_idx + 1:03d}"),
                    "expected_keywords": ["Atlas", "113", "Cygnus", "389", "Dione", "431"],
                    "min_keyword_hits": 5,
                    "max_new_tokens": 160,
                    "stop_after_sentences": 0,
                }
            )

    oom_rows = []
    for case_idx in range(args.oom_cases):
        label = f"oom{case_idx + 1}"
        oom_rows.append(
            {
                "id": f"oom_repeated_verdict_{case_idx + 1:02d}",
                "category": "oom_stress",
                "prompt": repeated_oom_prompt(args.oom_repeats + case_idx * 10, label=label),
                "expected_keywords": [
                    "reliable-with-monitoring",
                    "short QA",
                    "long summaries",
                    "memory",
                    "rare-token recall",
                ],
                "min_keyword_hits": 4,
                "max_new_tokens": 192,
                "stop_after_sentences": 0,
            }
        )
    write_jsonl(args.output_dir / "reliability_speed.jsonl", speed_rows)
    write_jsonl(args.output_dir / "oom_stress.jsonl", oom_rows)
    print(f"wrote {args.output_dir / 'reliability_speed.jsonl'}")
    print(f"wrote {args.output_dir / 'oom_stress.jsonl'}")


if __name__ == "__main__":
    main()
