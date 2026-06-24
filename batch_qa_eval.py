#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvcache import generate_with_budgeted_kv


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9:]+", " ", text)
    return " ".join(text.split())


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def existing_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        return {row["id"] for row in csv.DictReader(fh)}


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()


def save_artifacts(path: Path | None, case_id: str, prompt: str, output: str) -> None:
    if path is None:
        return
    path.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", case_id)
    (path / f"{safe_id}.prompt.txt").write_text(prompt, encoding="utf-8")
    (path / f"{safe_id}.output.txt").write_text(output, encoding="utf-8")


def score_output(case: dict, output: str) -> tuple[bool, str, int, int]:
    output_norm = normalize(output)
    if case.get("expected_regex"):
        pattern = str(case["expected_regex"])
        return bool(re.search(pattern, output, flags=re.IGNORECASE)), "regex", 0, 0

    if case.get("expected_exact"):
        expected = normalize(str(case["expected_exact"]))
        return expected in output_norm, "exact", int(expected in output_norm), 1

    keywords = [str(item) for item in case.get("expected_keywords", [])]
    if keywords:
        hits = sum(1 for keyword in keywords if normalize(keyword) in output_norm)
        required = int(case.get("min_keyword_hits", len(keywords)))
        return hits >= required, "keywords", hits, required

    raise ValueError(f"case {case.get('id')} has no expected_regex, expected_exact, or expected_keywords")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch QA/reasoning/long-context eval for budgeted KV.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-gpu-memory", default="18GiB")
    parser.add_argument("--max-cpu-memory", default="110GiB")
    parser.add_argument("--offload-folder", type=Path, default=Path("/root/autodl-tmp/offload"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--default-max-new-tokens", type=int, default=1024)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=256)
    parser.add_argument("--max-cache-tokens", type=int, default=2048)
    parser.add_argument("--recent-window", type=int, default=1024)
    parser.add_argument("--hot-cache-tokens", type=int, default=768)
    parser.add_argument("--hot-raw-tokens", type=int, default=-1)
    parser.add_argument("--merge-similarity", type=float, default=0.90)
    parser.add_argument("--attention-decay", type=float, default=0.995)
    parser.add_argument("--importance-update", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/batch_qa_eval.csv"))
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable tokenizer chat template wrapping.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen-style thinking mode in chat template when supported.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_jsonl(args.dataset)
    if args.limit > 0:
        cases = cases[: args.limit]
    completed = existing_case_ids(args.output_csv) if args.resume else set()

    model, tokenizer = load_model(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
        max_gpu_memory=args.max_gpu_memory,
        max_cpu_memory=args.max_cpu_memory,
        offload_folder=args.offload_folder,
    )

    correct = 0
    finished = 0
    started = time.perf_counter()
    for idx, case in enumerate(cases):
        case_id = str(case["id"])
        if case_id in completed:
            print(f"[skip] {case_id}", flush=True)
            continue

        result = generate_with_budgeted_kv(
            model=model,
            tokenizer=tokenizer,
            prompt=str(case["prompt"]),
            max_new_tokens=int(case.get("max_new_tokens", args.default_max_new_tokens)),
            prefill_chunk_tokens=args.prefill_chunk_tokens,
            max_cache_tokens=args.max_cache_tokens,
            recent_window=args.recent_window,
            hot_cache_tokens=args.hot_cache_tokens,
            hot_raw_tokens=args.hot_raw_tokens,
            merge_similarity=args.merge_similarity,
            attention_decay=args.attention_decay,
            importance_update=args.importance_update,
            log_every=args.log_every,
            stop_after_regex=str(case.get("stop_after_regex", "")),
            stop_after_sentences=int(case.get("stop_after_sentences", 0)),
            temperature=0.0,
            top_p=1.0,
            greedy=True,
            use_chat_template=not args.no_chat_template,
            chat_template_enable_thinking=args.enable_thinking,
        )

        ok, score_type, hits, required = score_output(case, result.text)
        correct += int(ok)
        finished += 1
        save_artifacts(args.artifacts_dir, case_id, str(case["prompt"]), result.text)
        append_row(
            args.output_csv,
            {
                "id": case_id,
                "category": str(case.get("category", "")),
                "ok": str(ok),
                "score_type": score_type,
                "keyword_hits": str(hits),
                "keyword_required": str(required),
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
                "output": result.text.replace("\n", " ")[:700],
            },
        )
        total_elapsed = time.perf_counter() - started
        print(
            f"[case {idx + 1}/{len(cases)}] {case_id} ok={ok} "
            f"running_acc={correct / max(1, finished):.2%} "
            f"case_time={result.elapsed_sec:.1f}s total_time={total_elapsed:.1f}s",
            flush=True,
        )

    print(f"wrote {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
