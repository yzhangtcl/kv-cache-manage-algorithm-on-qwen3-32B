#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_kvcache import generate_with_budgeted_kv


@dataclass
class Case:
    case_id: str
    prompt: str
    expected: str


DEFAULT_CASES = [
    Case(
        "needle_helios",
        "Facts:\n"
        + "\n".join(
            f"Line {i}: project Helios verification code is 811. Project Atlas code is 113."
            for i in range(180)
        )
        + "\nQuestion: What is the verification code for project Helios? Answer only the code.",
        "811",
    ),
    Case(
        "needle_borealis",
        "Operations notes:\n"
        + "\n".join(
            f"Block {i}: Borealis owns amber routing key 227; Dione uses mark 431."
            for i in range(160)
        )
        + "\nQuestion: Which routing key belongs to Borealis?",
        "227",
    ),
    Case(
        "local_reasoning",
        "A cache has 2048 total slots. 1536 slots are protected recent slots. "
        "How many slots remain for older hot entries? Answer only the number.",
        "512",
    ),
    Case(
        "simple_summary",
        "Summarize in one sentence: The experiment keeps a recent KV window, "
        "compresses older KV entries, and checks whether outputs remain close "
        "to an exact baseline.",
        "keeps recent kv and compresses older kv",
    ),
]


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, expected: str) -> float:
    pred_tokens = normalize(prediction).split()
    exp_tokens = normalize(expected).split()
    if not pred_tokens or not exp_tokens:
        return 0.0
    common = 0
    used = [False] * len(pred_tokens)
    for exp in exp_tokens:
        for i, pred in enumerate(pred_tokens):
            if not used[i] and pred == exp:
                used[i] = True
                common += 1
                break
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(exp_tokens)
    return 2 * precision * recall / (precision + recall)


def load_cases(path: Path | None) -> list[Case]:
    if path is None:
        return DEFAULT_CASES
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Case(str(item["id"]), str(item["prompt"]), str(item["expected"])) for item in payload]


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


def exact_generate(model, tokenizer, prompt: str, max_new_tokens: int) -> tuple[str, float, float]:
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start
    peak_gb = 0.0
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    new_tokens = output[:, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens[0], skip_special_tokens=True), elapsed, peak_gb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare exact generation and budgeted KV generation.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--cases-json", type=Path)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-gpu-memory", default="22GiB")
    parser.add_argument("--max-cpu-memory", default="96GiB")
    parser.add_argument("--offload-folder", type=Path, default=Path("offload"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=512)
    parser.add_argument("--max-cache-tokens", type=int, default=2048)
    parser.add_argument("--recent-window", type=int, default=1536)
    parser.add_argument(
        "--hot-cache-tokens",
        type=int,
        default=-1,
        help="Old-window total hot KV budget. -1 means half of old budget.",
    )
    parser.add_argument(
        "--hot-raw-tokens",
        type=int,
        default=-1,
        help="Hot KV tokens kept exactly. -1 means one quarter of hot budget.",
    )
    parser.add_argument("--merge-similarity", type=float, default=0.82)
    parser.add_argument("--attention-decay", type=float, default=0.98)
    parser.add_argument("--importance-update", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--stop-after-regex", default="")
    parser.add_argument("--stop-after-sentences", type=int, default=0)
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--skip-exact",
        action="store_true",
        help="Use this when exact baseline OOMs on the selected model/context.",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/quality_eval.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.cases_json)
    model, tokenizer = load_model(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
        max_gpu_memory=args.max_gpu_memory,
        max_cpu_memory=args.max_cpu_memory,
        offload_folder=args.offload_folder,
    )

    rows = []
    for case in cases:
        exact_text = ""
        exact_error = ""
        exact_elapsed = 0.0
        exact_peak = 0.0
        if not args.skip_exact:
            try:
                exact_text, exact_elapsed, exact_peak = exact_generate(
                    model, tokenizer, case.prompt, args.max_new_tokens
                )
            except torch.cuda.OutOfMemoryError as exc:
                exact_error = f"OOM: {exc}"
                torch.cuda.empty_cache()

        budgeted = generate_with_budgeted_kv(
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

        expected_norm = normalize(case.expected)
        budget_norm = normalize(budgeted.text)
        exact_norm = normalize(exact_text)
        rows.append(
            {
                "case_id": case.case_id,
                "expected": case.expected,
                "exact_ok": "" if exact_error else str(expected_norm in exact_norm),
                "budgeted_ok": str(expected_norm in budget_norm),
                "exact_f1": "" if exact_error else f"{token_f1(exact_text, case.expected):.4f}",
                "budgeted_f1": f"{token_f1(budgeted.text, case.expected):.4f}",
                "exact_elapsed_sec": f"{exact_elapsed:.2f}",
                "budgeted_elapsed_sec": f"{budgeted.elapsed_sec:.2f}",
                "exact_peak_gb": f"{exact_peak:.2f}",
                "budgeted_peak_gb": f"{budgeted.peak_memory_gb:.2f}",
                "prompt_tokens": str(budgeted.prompt_tokens),
                "budgeted_generated_tokens": str(budgeted.generated_tokens),
                "budgeted_avg_kept_cache_tokens": f"{budgeted.compression.avg_kept_tokens:.1f}",
                "budgeted_avg_hot_cache_tokens": f"{budgeted.compression.avg_hot_tokens:.1f}",
                "budgeted_avg_hot_raw_tokens": f"{budgeted.compression.avg_hot_raw_tokens:.1f}",
                "budgeted_avg_hot_cluster_tokens": f"{budgeted.compression.avg_hot_cluster_tokens:.1f}",
                "budgeted_avg_cold_cluster_tokens": f"{budgeted.compression.avg_cold_tokens:.1f}",
                "exact_error": exact_error[:300],
                "exact_output": exact_text.replace("\n", " ")[:500],
                "budgeted_output": budgeted.text.replace("\n", " ")[:500],
            }
        )
        print(f"{case.case_id}: budgeted_ok={rows[-1]['budgeted_ok']} exact_ok={rows[-1]['exact_ok']}")

    with args.output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
