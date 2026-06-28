#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import time
from copy import deepcopy
from pathlib import Path


def load_longmemeval(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def simple_answer_hit(answer: str, output: str) -> bool:
    answer_norm = normalize(answer)
    output_norm = normalize(output)
    return bool(answer_norm) and answer_norm in output_norm


def clean_turn(turn: dict) -> dict:
    return {
        "role": str(turn.get("role", "")),
        "content": str(turn.get("content", "")),
    }


def session_to_json(session: list[dict], useronly: bool) -> str:
    turns = [clean_turn(turn) for turn in session if not useronly or turn.get("role") == "user"]
    return json.dumps(turns, ensure_ascii=False)


def session_to_nl(session: list[dict], useronly: bool) -> str:
    lines = []
    for turn in session:
        if useronly and turn.get("role") != "user":
            continue
        role = str(turn.get("role", "")).strip()
        content = str(turn.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def render_history(entry: dict, history_format: str, useronly: bool, topk_context: int) -> str:
    dates = entry.get("haystack_dates", [])
    sessions = entry.get("haystack_sessions", [])
    pairs = list(zip(dates, sessions))
    if topk_context > 0:
        pairs = pairs[-topk_context:]
    pairs.sort(key=lambda item: item[0])

    chunks = []
    for idx, (session_date, session) in enumerate(pairs, start=1):
        clean_session = deepcopy(session)
        if history_format == "json":
            content = "\n" + session_to_json(clean_session, useronly=useronly)
        elif history_format == "nl":
            content = session_to_nl(clean_session, useronly=useronly)
        else:
            raise ValueError(f"unsupported history_format: {history_format}")
        chunks.append(
            f"\n### Session {idx}:\n"
            f"Session Date: {session_date}\n"
            f"Session Content:\n{content}\n"
        )
    return "".join(chunks)


def truncate_history(
    tokenizer,
    history: str,
    max_retrieval_tokens: int,
) -> str:
    if max_retrieval_tokens <= 0:
        return history
    encoded = tokenizer(
        history,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=False,
    )
    input_ids = encoded["input_ids"][0]
    if int(input_ids.shape[0]) <= max_retrieval_tokens:
        return history
    truncated = input_ids[:max_retrieval_tokens]
    return tokenizer.decode(truncated, skip_special_tokens=True)


def build_prompt(
    entry: dict,
    tokenizer,
    history_format: str,
    useronly: bool,
    topk_context: int,
    reading_method: str,
    max_retrieval_tokens: int,
) -> str:
    history = render_history(
        entry=entry,
        history_format=history_format,
        useronly=useronly,
        topk_context=topk_context,
    )
    history = truncate_history(tokenizer, history, max_retrieval_tokens)

    if reading_method == "direct":
        template = (
            "I will give you several history chats between you and a user.\n"
            "Please answer the question based on the relevant chat history.\n\n\n"
            "History Chats:\n\n{}\n\n"
            "Current Date: {}\n"
            "Question: {}\n"
            "Answer:"
        )
    elif reading_method == "con":
        template = (
            "I will give you several history chats between you and a user.\n"
            "Please answer the question based on the relevant chat history. "
            "Answer the question step by step: first extract all the relevant information, "
            "and then reason over the information to get the answer.\n\n\n"
            "History Chats:\n\n{}\n\n"
            "Current Date: {}\n"
            "Question: {}\n"
            "Answer (step by step):"
        )
    else:
        raise ValueError(f"unsupported reading_method: {reading_method}")

    return template.format(
        history,
        str(entry.get("question_date", "")),
        str(entry.get("question", "")),
    )


def output_mode_name(args: argparse.Namespace, mode: str) -> str:
    if mode == "sliding":
        return "sliding_window"
    return args.mode_label or mode


def modes_to_run(mode: str) -> list[str]:
    if mode == "all":
        return ["full", "kvmanage", "sliding"]
    if mode == "both":
        return ["full", "kvmanage"]
    return [mode]


def existing_jsonl_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                ids.add(str(row["question_id"]))
    return ids


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        print(json.dumps(row, ensure_ascii=False), file=fh, flush=True)


def append_csv(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()


def save_artifacts(path: Path | None, question_id: str, mode: str, prompt: str, output: str) -> None:
    if path is None:
        return
    path.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", question_id)
    safe_mode = re.sub(r"[^a-zA-Z0-9_.-]+", "_", mode)
    (path / f"{safe_id}.{safe_mode}.prompt.txt").write_text(prompt, encoding="utf-8")
    (path / f"{safe_id}.{safe_mode}.output.txt").write_text(output, encoding="utf-8")


def clear_cuda_state() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text or "cublas_status_alloc_failed" in text


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if args.rope_factor > 0:
        original = args.rope_original_max_position_embeddings
        config.rope_scaling = {
            "rope_type": "yarn",
            "factor": float(args.rope_factor),
            "original_max_position_embeddings": int(original),
        }
        config.max_position_embeddings = max(
            int(getattr(config, "max_position_embeddings", 0) or 0),
            int(original * args.rope_factor),
        )

    max_memory = None
    if args.max_gpu_memory or args.max_cpu_memory:
        max_memory = {}
        if torch.cuda.is_available() and args.max_gpu_memory:
            max_memory[0] = args.max_gpu_memory
        if args.max_cpu_memory:
            max_memory["cpu"] = args.max_cpu_memory

    device_map = args.device
    if args.max_cpu_memory and args.device in {"cuda", "cuda:0", "gpu", "0"}:
        device_map = "auto"
    args.offload_folder.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        dtype=torch_dtype,
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=str(args.offload_folder),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    return model, tokenizer


def run_generation(
    args: argparse.Namespace,
    model,
    tokenizer,
    prompt: str,
    mode: str,
):
    from kvcache import generate_full_kv, generate_with_budgeted_kv

    common = {
        "model": model,
        "tokenizer": tokenizer,
        "prompt": prompt,
        "max_new_tokens": args.max_new_tokens,
        "prefill_chunk_tokens": args.prefill_chunk_tokens,
        "stop_after_regex": "",
        "stop_after_sentences": 0,
        "temperature": 0.0,
        "top_p": 1.0,
        "greedy": True,
        "use_chat_template": not args.no_chat_template,
        "chat_template_enable_thinking": args.enable_thinking,
        "repetition_penalty": args.repetition_penalty,
    }
    if mode == "full":
        return generate_full_kv(**common)
    if mode == "sliding":
        return generate_with_budgeted_kv(
            **common,
            max_cache_tokens=args.sliding_cache_tokens,
            recent_window=args.sliding_cache_tokens,
            hot_cache_tokens=0,
            hot_raw_tokens=0,
            merge_similarity=args.merge_similarity,
            attention_decay=args.attention_decay,
            importance_update=0.0,
            compress_every=args.compress_every,
            log_every=args.log_every,
        )
    if mode == "kvmanage":
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
    raise ValueError(f"unsupported mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Transformers baselines on LongMemEval.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/longmemeval_s"))
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-gpu-memory", default="22GiB")
    parser.add_argument("--max-cpu-memory", default="")
    parser.add_argument("--offload-folder", type=Path, default=Path("/root/autodl-tmp/offload"))
    parser.add_argument(
        "--mode",
        choices=["full", "kvmanage", "sliding", "both", "all"],
        default="kvmanage",
    )
    parser.add_argument("--mode-label", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--history-format", choices=["json", "nl"], default="json")
    parser.add_argument("--useronly", action="store_true")
    parser.add_argument(
        "--reading-method",
        choices=["direct", "con"],
        default="con",
        help="LongMemEval's con mode is the step-by-step reading prompt.",
    )
    parser.add_argument(
        "--topk-context",
        type=int,
        default=1000,
        help="Large default keeps all LongMemEval-S sessions.",
    )
    parser.add_argument(
        "--max-retrieval-tokens",
        type=int,
        default=129000,
        help="Token budget for just the rendered history before question/template text. Use 0 to disable.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=800)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=1024)
    parser.add_argument("--max-cache-tokens", type=int, default=4096)
    parser.add_argument("--recent-window", type=int, default=2048)
    parser.add_argument("--hot-cache-tokens", type=int, default=1536)
    parser.add_argument("--hot-raw-tokens", type=int, default=-1)
    parser.add_argument("--sliding-cache-tokens", type=int, default=4096)
    parser.add_argument("--merge-similarity", type=float, default=0.90)
    parser.add_argument("--attention-decay", type=float, default=0.995)
    parser.add_argument("--importance-update", type=float, default=0.0)
    parser.add_argument("--compress-every", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--rope-factor",
        type=float,
        default=4.0,
        help="Set 0 to disable YaRN. Qwen3 uses factor 4 for about 131k tokens.",
    )
    parser.add_argument("--rope-original-max-position-embeddings", type=int, default=32768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_longmemeval(args.dataset)
    if args.limit > 0:
        entries = entries[: args.limit]
    modes = modes_to_run(args.mode)

    model, tokenizer = load_model(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "runs.csv"
    started = time.perf_counter()

    completed_by_mode = {}
    for mode in modes:
        out_mode = output_mode_name(args, mode)
        completed_by_mode[out_mode] = (
            existing_jsonl_ids(args.output_dir / f"{out_mode}.jsonl") if args.resume else set()
        )

    for index, entry in enumerate(entries, start=1):
        question_id = str(entry["question_id"])
        prompt = build_prompt(
            entry=entry,
            tokenizer=tokenizer,
            history_format=args.history_format,
            useronly=args.useronly,
            topk_context=args.topk_context,
            reading_method=args.reading_method,
            max_retrieval_tokens=args.max_retrieval_tokens,
        )
        for mode in modes:
            out_mode = output_mode_name(args, mode)
            if question_id in completed_by_mode[out_mode]:
                print(f"[skip] {question_id} mode={out_mode}", flush=True)
                continue

            result = None
            status = "ok"
            error = ""
            try:
                result = run_generation(args, model, tokenizer, prompt, mode)
                hypothesis = result.text.strip()
                save_artifacts(args.artifacts_dir, question_id, out_mode, prompt, hypothesis)
                append_jsonl(
                    args.output_dir / f"{out_mode}.jsonl",
                    {"question_id": question_id, "hypothesis": hypothesis},
                )
            except RuntimeError as exc:
                if not is_oom_error(exc) and not args.continue_on_error:
                    raise
                status = "oom" if is_oom_error(exc) else "error"
                error = f"{type(exc).__name__}: {exc}"
                hypothesis = ""
                clear_cuda_state()
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                hypothesis = ""
                clear_cuda_state()

            compression = result.compression if result is not None else None
            row = {
                "question_id": question_id,
                "mode": out_mode,
                "question_type": str(entry.get("question_type", "")),
                "status": status,
                "simple_answer_hit": str(
                    simple_answer_hit(str(entry.get("answer", "")), hypothesis)
                    if status == "ok"
                    else False
                ),
                "prompt_tokens": str(result.prompt_tokens if result is not None else ""),
                "generated_tokens": str(result.generated_tokens if result is not None else ""),
                "elapsed_sec": f"{result.elapsed_sec:.2f}" if result is not None else "",
                "peak_memory_gb": f"{result.peak_memory_gb:.2f}" if result is not None else "",
                "compress_calls": str(compression.compress_calls if compression is not None else ""),
                "avg_kept_cache_tokens": (
                    f"{compression.avg_kept_tokens:.1f}" if compression is not None else ""
                ),
                "dropped_tokens_total": (
                    str(compression.dropped_tokens) if compression is not None else ""
                ),
                "merged_tokens_total": (
                    str(compression.merged_tokens) if compression is not None else ""
                ),
                "answer": str(entry.get("answer", "")).replace("\n", " ")[:500],
                "error": error.replace("\n", " ")[:700],
                "output": hypothesis.replace("\n", " ")[:700],
            }
            append_csv(csv_path, row)
            total_elapsed = time.perf_counter() - started
            case_time = result.elapsed_sec if result is not None else 0.0
            print(
                f"[case {index}/{len(entries)}] {question_id} mode={out_mode} "
                f"status={status} prompt_tokens={row['prompt_tokens']} "
                f"case_time={case_time:.1f}s total_time={total_elapsed:.1f}s",
                flush=True,
            )
            clear_cuda_state()

    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
