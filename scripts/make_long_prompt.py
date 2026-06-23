#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


FACTS = [
    ("Atlas", "runs the blue cache lane", "113"),
    ("Borealis", "owns the amber routing key", "227"),
    ("Cygnus", "stores snapshots in vault seven", "389"),
    ("Dione", "uses the silver eviction mark", "431"),
    ("Erebus", "keeps backup shards near node nine", "557"),
    ("Fornax", "rotates the green attention beacon", "619"),
    ("Gaia", "pins the first three memory anchors", "743"),
    ("Helios", "moves cold blocks after probe thirty two", "811"),
]


def build_prompt(repeats: int) -> str:
    lines = [
        "You are given an operations log. Answer only from facts in the log.",
        "The log intentionally repeats and paraphrases facts to create a long context.",
        "",
    ]
    for round_id in range(repeats):
        for name, action, code in FACTS:
            lines.append(
                f"Record {round_id:04d}-{name}: project {name} {action}; "
                f"its verification code is {code}."
            )
        lines.append(
            "Noise: cache scheduling should preserve recent facts, old facts, "
            "and repeated long-range facts without inventing new codes."
        )
    lines.extend(
        [
            "",
            "Question: What is the verification code for project Helios?",
            "Answer with only the code and a one-sentence justification.",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic long prompt.")
    parser.add_argument("--repeats", type=int, default=900)
    parser.add_argument("--output", type=Path, default=Path("long_prompt.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.write_text(build_prompt(args.repeats), encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
