#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_kvcache import generate_with_budgeted_kv


PROJECTS = [
    ("Atlas", "113"),
    ("Borealis", "227"),
    ("Cygnus", "389"),
    ("Dione", "431"),
    ("Erebus", "557"),
    ("Fornax", "619"),
    ("Gaia", "743"),
    ("Helios", "811"),
    ("Icarus", "929"),
    ("Juno", "1049"),
    ("Kepler", "1153"),
    ("Lumen", "1217"),
]

NOISE_ACTIONS = [
    "rotates cold cache segments",
    "pins recent attention blocks",
    "checks the routing ledger",
    "moves old summaries into archive",
    "refreshes the probe schedule",
    "audits local token windows",
]


@dataclass
class EvalCase:
    case_id: str
    prompt: str
    expected: str
    needle_project: str
    needle_depth: float


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def load_model(
    model_name: str,
    dtype: str,
    device: str,
    max_gpu_memory: str,
    max_cpu_memory: str,
    offload_folder: Path,
):
    torch_dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    max_memory = None
    if max_gpu_memory or max_cpu_memory:
        max_memory = {}
        if torch.cuda.is_available() and max_gpu_memory:
            max_memory[0] = max_gpu_memory
        if max_cpu_memory:
            max_memory["cpu"] = max_cpu_memory
    offload_folder.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        device_map=device,
        max_memory=max_memory,
        offload_folder=str(offload_folder),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


def build_case(
    case_index: int,
    repeats: int,
    needle_depth: float,
    rng: random.Random,
) -> EvalCase:
    project, code = PROJECTS[case_index % len(PROJECTS)]
    insert_at = int(max(0, min(repeats - 1, round((repeats - 1) * needle_depth))))
    lines = [
        "You are given a long operations log. Answer only from facts in the log.",
        "Many records are distracting and should not override the requested project code.",
        "",
    ]

    for round_id in range(repeats):
        if round_id == insert_at:
            lines.append(
                f"Critical record {round_id:04d}: project {project} has verification code {code}."
            )
        other_project, other_code = PROJECTS[(case_index + round_id + 3) % len(PROJECTS)]
        action = rng.choice(NOISE_ACTIONS)
        lines.append(
            f"Record {round_id:04d}: project {other_project} {action}; "
            f"temporary audit number {other_code} is not the answer unless asked."
        )
        if round_id % 7 == 0:
            lines.append(
                "Noise: preserve old, recent, and hot cache facts without inventing codes."
            )

    lines.extend(
        [
            "",
            f"Question: What is the verification code for project {project}?",
            "Answer with only the code.",
        ]
    )
    return EvalCase(
        case_id=f"needle_{case_index:03d}_depth_{needle_depth:.2f}",
        prompt="\n".join(lines),
        expected=code,
        needle_project=project,
        needle_depth=needle_depth,
    )


def existing_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        return {row["case_id"] for row in csv.DictReader(fh)}


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch long-context needle eval for budgeted KV.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-gpu-memory", default="18GiB")
    parser.add_argument("--max-cpu-memory", default="110GiB")
    parser.add_argument("--offload-folder", type=Path, default=Path("/root/autodl-tmp/offload"))
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--depths", default="0.05,0.25,0.50,0.75,0.95")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=256)
    parser.add_argument("--max-cache-tokens", type=int, default=1024)
    parser.add_argument("--recent-window", type=int, default=768)
    parser.add_argument("--hot-cache-tokens", type=int, default=-1)
    parser.add_argument("--hot-raw-tokens", type=int, default=-1)
    parser.add_argument("--merge-similarity", type=float, default=0.82)
    parser.add_argument("--attention-decay", type=float, default=0.98)
    parser.add_argument("--importance-update", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument(
        "--stop-after-regex",
        default="",
        help="Optional task-specific stop regex. Empty means disabled.",
    )
    parser.add_argument(
        "--stop-after-sentences",
        type=int,
        default=0,
        help="Stop after this many complete sentences. 0 means disabled.",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/batch_needle_eval.csv"))
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    depths = [float(item) for item in args.depths.split(",") if item.strip()]
    completed = existing_case_ids(args.output_csv) if args.resume else set()

    model, tokenizer = load_model(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
        max_gpu_memory=args.max_gpu_memory,
        max_cpu_memory=args.max_cpu_memory,
        offload_folder=args.offload_folder,
    )

    total = args.cases
    correct = 0
    finished = 0
    started = time.perf_counter()
    for idx in range(total):
        depth = depths[idx % len(depths)]
        case = build_case(idx, args.repeats, depth, rng)
        if case.case_id in completed:
            print(f"[skip] {case.case_id}", flush=True)
            continue

        result = generate_with_budgeted_kv(
            model=model,
            tokenizer=tokenizer,
            prompt=case.prompt,
            max_new_tokens=args.max_new_tokens,
            prefill_chunk_tokens=args.prefill_chunk_tokens,
            max_cache_tokens=args.max_cache_tokens,
            recent_window=args.recent_window,
            hot_cache_tokens=args.hot_cache_tokens,
            hot_raw_tokens=args.hot_raw_tokens,
            merge_similarity=args.merge_similarity,
            attention_decay=args.attention_decay,
            importance_update=args.importance_update,
            log_every=args.log_every,
            stop_after_regex=args.stop_after_regex,
            stop_after_sentences=args.stop_after_sentences,
            temperature=0.0,
            top_p=1.0,
            greedy=True,
            use_chat_template=args.use_chat_template,
            chat_template_enable_thinking=args.enable_thinking if args.use_chat_template else None,
        )

        ok = normalize(case.expected) in normalize(result.text)
        correct += int(ok)
        finished += 1
        elapsed_total = time.perf_counter() - started
        append_row(
            args.output_csv,
            {
                "case_id": case.case_id,
                "ok": str(ok),
                "expected": case.expected,
                "needle_project": case.needle_project,
                "needle_depth": f"{case.needle_depth:.2f}",
                "prompt_tokens": str(result.prompt_tokens),
                "generated_tokens": str(result.generated_tokens),
                "elapsed_sec": f"{result.elapsed_sec:.2f}",
                "peak_memory_gb": f"{result.peak_memory_gb:.2f}",
                "compress_calls": str(result.compression.compress_calls),
                "avg_kept_cache_tokens": f"{result.compression.avg_kept_tokens:.1f}",
                "avg_hot_cache_tokens": f"{result.compression.avg_hot_tokens:.1f}",
                "avg_hot_raw_tokens": f"{result.compression.avg_hot_raw_tokens:.1f}",
                "avg_hot_cluster_tokens": f"{result.compression.avg_hot_cluster_tokens:.1f}",
                "avg_cold_cluster_tokens": f"{result.compression.avg_cold_tokens:.1f}",
                "dropped_tokens_total": str(result.compression.dropped_tokens),
                "merged_tokens_total": str(result.compression.merged_tokens),
                "output": result.text.replace("\n", " ")[:500],
            },
        )
        print(
            f"[case {idx + 1}/{total}] {case.case_id} ok={ok} "
            f"running_acc={correct / max(1, finished):.2%} "
            f"case_time={result.elapsed_sec:.1f}s total_time={elapsed_total:.1f}s",
            flush=True,
        )

    print(f"wrote {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
