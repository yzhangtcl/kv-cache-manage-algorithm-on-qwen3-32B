#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_kvcache import generate_with_budgeted_kv


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
    parser = argparse.ArgumentParser(
        description="Run long-context generation with budgeted KV cache compression."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--prompt-file", type=Path, default=Path("long_prompt.txt"))
    parser.add_argument("--prompt", default="")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument(
        "--device",
        default="auto",
        help="Transformers device_map, usually 'auto' on a single 4090 server.",
    )
    parser.add_argument("--max-gpu-memory", default="22GiB")
    parser.add_argument("--max-cpu-memory", default="96GiB")
    parser.add_argument("--offload-folder", type=Path, default=Path("offload"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
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
    parser.add_argument(
        "--log-every",
        type=int,
        default=2048,
        help="Print progress every N prompt tokens. 0 disables progress logging.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
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
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Wrap prompt with tokenizer chat template before generation.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen-style thinking mode in chat template when supported.",
    )
    parser.add_argument("--output-file", type=Path, default=Path("outputs/server_generation.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = args.prompt or args.prompt_file.read_text(encoding="utf-8")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
        max_gpu_memory=args.max_gpu_memory,
        max_cpu_memory=args.max_cpu_memory,
        offload_folder=args.offload_folder,
    )
    result = generate_with_budgeted_kv(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
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
        temperature=args.temperature,
        top_p=args.top_p,
        greedy=args.temperature <= 0,
        use_chat_template=args.use_chat_template,
        chat_template_enable_thinking=args.enable_thinking if args.use_chat_template else None,
    )

    report = [
        f"model: {args.model}",
        f"prompt_tokens: {result.prompt_tokens}",
        f"generated_tokens: {result.generated_tokens}",
        f"elapsed_sec: {result.elapsed_sec:.2f}",
        f"tokens_per_sec: {result.generated_tokens / max(result.elapsed_sec, 1e-6):.2f}",
        f"peak_memory_gb: {result.peak_memory_gb:.2f}",
        f"compress_calls: {result.compression.compress_calls}",
        f"avg_kept_cache_tokens: {result.compression.avg_kept_tokens:.1f}",
        f"avg_hot_cache_tokens: {result.compression.avg_hot_tokens:.1f}",
        f"avg_hot_raw_tokens: {result.compression.avg_hot_raw_tokens:.1f}",
        f"avg_hot_cluster_tokens: {result.compression.avg_hot_cluster_tokens:.1f}",
        f"avg_cold_cluster_tokens: {result.compression.avg_cold_tokens:.1f}",
        f"dropped_tokens_total: {result.compression.dropped_tokens}",
        f"merged_tokens_total: {result.compression.merged_tokens}",
        "",
        "output:",
        result.text.strip(),
        "",
    ]
    text = "\n".join(report)
    args.output_file.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
