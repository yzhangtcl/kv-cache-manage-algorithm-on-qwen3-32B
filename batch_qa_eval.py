#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvcache import GenerationResult, generate_full_kv, generate_with_budgeted_kv


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


def existing_runs(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        completed = set()
        for row in csv.DictReader(fh):
            if row.get("category") == "summary" or row.get("id", "").startswith("__summary_"):
                continue
            mode = row.get("mode") or "kvmanage"
            completed.add((row["id"], mode))
        return completed


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()


def append_summary_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    summaries = build_summary_rows(rows, fieldnames)
    if not summaries:
        return
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        for row in summaries:
            writer.writerow(row)
        fh.flush()


def to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_summary_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> list[dict[str, str]]:
    summaries = []
    modes = sorted({row.get("mode", "") for row in rows if row.get("mode")})
    for mode in modes:
        mode_rows = [row for row in rows if row.get("mode") == mode]
        ok_rows = [row for row in mode_rows if row.get("status") == "ok"]
        correct = sum(1 for row in ok_rows if row.get("ok") == "True")
        elapsed = [value for row in ok_rows if (value := to_float(row.get("elapsed_sec", ""))) is not None]
        memory = [
            value for row in ok_rows if (value := to_float(row.get("peak_memory_gb", ""))) is not None
        ]
        generated = [
            value for row in ok_rows if (value := to_float(row.get("generated_tokens", ""))) is not None
        ]
        prompt_tokens = [
            value for row in ok_rows if (value := to_float(row.get("prompt_tokens", ""))) is not None
        ]
        kept = [
            value for row in ok_rows if (value := to_float(row.get("avg_kept_cache_tokens", ""))) is not None
        ]
        dropped = [
            value for row in ok_rows if (value := to_float(row.get("dropped_tokens_total", ""))) is not None
        ]
        merged = [
            value for row in ok_rows if (value := to_float(row.get("merged_tokens_total", ""))) is not None
        ]
        oom_count = sum(1 for row in mode_rows if row.get("status") == "oom")
        error_count = sum(1 for row in mode_rows if row.get("status") == "error")
        summary = {field: "" for field in fieldnames}
        summary.update(
            {
                "id": f"__summary_{mode}__",
                "mode": mode,
                "category": "summary",
                "status": "summary",
                "ok": f"{correct / len(ok_rows):.4f}" if ok_rows else "0.0000",
                "score_type": "mean",
                "keyword_hits": str(correct),
                "keyword_required": str(len(ok_rows)),
                "prompt_tokens": f"{avg(prompt_tokens):.1f}",
                "generated_tokens": f"{avg(generated):.1f}",
                "elapsed_sec": f"{avg(elapsed):.2f}",
                "peak_memory_gb": f"{avg(memory):.2f}",
                "avg_kept_cache_tokens": f"{avg(kept):.1f}",
                "dropped_tokens_total": f"{avg(dropped):.1f}",
                "merged_tokens_total": f"{avg(merged):.1f}",
                "error": f"rows={len(mode_rows)} scored={len(ok_rows)} oom={oom_count} error={error_count}",
                "output": "summary row: ok is accuracy; numeric columns are means over status=ok rows",
            }
        )
        summaries.append(summary)
    return summaries


def save_artifacts(path: Path | None, case_id: str, mode: str, prompt: str, output: str) -> None:
    if path is None:
        return
    path.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", case_id)
    safe_mode = re.sub(r"[^a-zA-Z0-9_.-]+", "_", mode)
    (path / f"{safe_id}.prompt.txt").write_text(prompt, encoding="utf-8")
    (path / f"{safe_id}.{safe_mode}.output.txt").write_text(output, encoding="utf-8")


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


def is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text or "cublas_status_alloc_failed" in text


def clear_cuda_state() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def run_modes(mode: str) -> list[str]:
    if mode == "both":
        return ["full", "kvmanage"]
    return [mode]


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
    device_map = device
    if max_cpu_memory and device in {"cuda", "cuda:0", "gpu", "0"}:
        device_map = "auto"
    offload_folder.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        device_map=device_map,
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
    parser.add_argument(
        "--mode",
        choices=["kvmanage", "full", "both"],
        default="kvmanage",
        help="Run budgeted KV, full KV baseline, or both.",
    )
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
    parser.add_argument("--compress-every", type=int, default=1)
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
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record non-OOM exceptions as failed rows instead of aborting.",
    )
    parser.add_argument(
        "--no-csv-summary",
        action="store_true",
        help="Do not append per-mode summary rows to the end of the output CSV.",
    )
    return parser.parse_args()


def generate_case(args: argparse.Namespace, model, tokenizer, case: dict, mode: str) -> GenerationResult:
    common = {
        "model": model,
        "tokenizer": tokenizer,
        "prompt": str(case["prompt"]),
        "max_new_tokens": int(case.get("max_new_tokens", args.default_max_new_tokens)),
        "prefill_chunk_tokens": args.prefill_chunk_tokens,
        "stop_after_regex": str(case.get("stop_after_regex", "")),
        "stop_after_sentences": int(case.get("stop_after_sentences", 0)),
        "temperature": 0.0,
        "top_p": 1.0,
        "greedy": True,
        "use_chat_template": not args.no_chat_template,
        "chat_template_enable_thinking": args.enable_thinking,
    }
    if mode == "full":
        return generate_full_kv(**common)
    if mode != "kvmanage":
        raise ValueError(f"unsupported mode: {mode}")
    return generate_with_budgeted_kv(
        **common,
        max_cache_tokens=args.max_cache_tokens,
        recent_window=args.recent_window,
        hot_cache_tokens=args.hot_cache_tokens,
        hot_raw_tokens=args.hot_raw_tokens,
        merge_similarity=args.merge_similarity,
        attention_decay=args.attention_decay,
        importance_update=args.importance_update,
        compress_every=args.compress_every,
        log_every=args.log_every,
    )


def result_row(
    case: dict,
    mode: str,
    status: str,
    ok: bool,
    score_type: str,
    hits: int,
    required: int,
    result: GenerationResult | None,
    error: str = "",
) -> dict[str, str]:
    compression = result.compression if result is not None else None
    return {
        "id": str(case["id"]),
        "mode": mode,
        "category": str(case.get("category", "")),
        "status": status,
        "ok": str(ok),
        "score_type": score_type,
        "keyword_hits": str(hits),
        "keyword_required": str(required),
        "prompt_tokens": str(result.prompt_tokens if result is not None else ""),
        "generated_tokens": str(result.generated_tokens if result is not None else ""),
        "elapsed_sec": f"{result.elapsed_sec:.2f}" if result is not None else "",
        "peak_memory_gb": f"{result.peak_memory_gb:.2f}" if result is not None else "",
        "compress_calls": str(compression.compress_calls if compression is not None else ""),
        "avg_kept_cache_tokens": (
            f"{compression.avg_kept_tokens:.1f}" if compression is not None else ""
        ),
        "avg_hot_cache_tokens": (
            f"{compression.avg_hot_tokens:.1f}" if compression is not None else ""
        ),
        "avg_hot_raw_tokens": (
            f"{compression.avg_hot_raw_tokens:.1f}" if compression is not None else ""
        ),
        "avg_hot_cluster_tokens": (
            f"{compression.avg_hot_cluster_tokens:.1f}" if compression is not None else ""
        ),
        "avg_cold_cluster_tokens": (
            f"{compression.avg_cold_tokens:.1f}" if compression is not None else ""
        ),
        "dropped_tokens_total": str(compression.dropped_tokens if compression is not None else ""),
        "merged_tokens_total": str(compression.merged_tokens if compression is not None else ""),
        "error": error.replace("\n", " ")[:700],
        "output": result.text.replace("\n", " ")[:700] if result is not None else "",
    }


def main() -> None:
    args = parse_args()
    cases = load_jsonl(args.dataset)
    if args.limit > 0:
        cases = cases[: args.limit]
    completed = existing_runs(args.output_csv) if args.resume else set()
    modes = run_modes(args.mode)

    model, tokenizer = load_model(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
        max_gpu_memory=args.max_gpu_memory,
        max_cpu_memory=args.max_cpu_memory,
        offload_folder=args.offload_folder,
    )

    mode_correct = {mode: 0 for mode in modes}
    mode_scored = {mode: 0 for mode in modes}
    written_rows: list[dict[str, str]] = []
    started = time.perf_counter()
    for idx, case in enumerate(cases):
        case_id = str(case["id"])
        for mode in modes:
            if (case_id, mode) in completed:
                print(f"[skip] {case_id} mode={mode}", flush=True)
                continue

            result = None
            try:
                result = generate_case(args, model, tokenizer, case, mode)
                ok, score_type, hits, required = score_output(case, result.text)
                status = "ok"
                error = ""
                save_artifacts(args.artifacts_dir, case_id, mode, str(case["prompt"]), result.text)
            except RuntimeError as exc:
                if not is_oom_error(exc) and not args.continue_on_error:
                    raise
                status = "oom" if is_oom_error(exc) else "error"
                ok, score_type, hits, required = False, "error", 0, 0
                error = f"{type(exc).__name__}: {exc}"
                clear_cuda_state()
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                status = "error"
                ok, score_type, hits, required = False, "error", 0, 0
                error = f"{type(exc).__name__}: {exc}"
                clear_cuda_state()

            if status == "ok":
                mode_correct[mode] += int(ok)
                mode_scored[mode] += 1
            row = result_row(
                case=case,
                mode=mode,
                status=status,
                ok=ok,
                score_type=score_type,
                hits=hits,
                required=required,
                result=result,
                error=error,
            )
            append_row(args.output_csv, row)
            written_rows.append(row)
            total_elapsed = time.perf_counter() - started
            case_time = result.elapsed_sec if result is not None else 0.0
            print(
                f"[case {idx + 1}/{len(cases)}] {case_id} mode={mode} "
                f"status={status} ok={ok} "
                f"{mode}_acc={mode_correct[mode] / max(1, mode_scored[mode]):.2%} "
                f"case_time={case_time:.1f}s total_time={total_elapsed:.1f}s",
                flush=True,
            )
            clear_cuda_state()

    if not args.no_csv_summary:
        append_summary_rows(args.output_csv, written_rows)
    print(f"wrote {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
