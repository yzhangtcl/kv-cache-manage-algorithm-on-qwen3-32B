#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from kvcache import budgeted_kv_cache


def load_model(
    model_name: str,
    dtype: str,
    device: str,
    max_gpu_memory: str,
    max_cpu_memory: str,
    offload_folder: Path,
    awq_version: str,
):
    torch_dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    device_map: str | dict[str, int] = device
    if device in {"cuda", "cuda:0", "gpu", "0"} or (
        device == "auto" and torch.cuda.is_available() and not max_cpu_memory
    ):
        device_map = {"": 0}
    max_memory = None
    if max_gpu_memory or max_cpu_memory:
        max_memory = {}
        if torch.cuda.is_available() and max_gpu_memory:
            max_memory[0] = max_gpu_memory
        if max_cpu_memory:
            max_memory["cpu"] = max_cpu_memory
    offload_folder.mkdir(parents=True, exist_ok=True)
    model_config = None
    if awq_version:
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        raw_quantization_config = getattr(model_config, "quantization_config", None)
        if isinstance(raw_quantization_config, dict):
            raw_quantization_config = dict(raw_quantization_config)
            raw_quantization_config["version"] = awq_version
            model_config.quantization_config = raw_quantization_config
        elif raw_quantization_config is not None and hasattr(raw_quantization_config, "version"):
            raw_quantization_config.version = awq_version
        else:
            raise RuntimeError("model config does not contain an AWQ quantization_config")
        print(f"using AWQ kernel version: {awq_version}", flush=True)
    model_kwargs = {
        "dtype": torch_dtype,
        "device_map": device_map,
        "max_memory": max_memory,
        "offload_folder": str(offload_folder),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if model_config is not None:
        model_kwargs["config"] = model_config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, tokenizer


def render_messages(tokenizer, messages: list[dict[str, str]], enable_thinking: bool) -> torch.Tensor:
    kwargs = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_tensors": "pt",
    }
    try:
        rendered = tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(rendered, torch.Tensor):
        return rendered
    if isinstance(rendered, dict) and "input_ids" in rendered:
        return rendered["input_ids"]
    if hasattr(rendered, "input_ids"):
        return rendered.input_ids
    raise TypeError(f"unexpected chat template return type: {type(rendered)!r}")


def trim_history(
    tokenizer,
    messages: list[dict[str, str]],
    max_input_tokens: int,
    enable_thinking: bool,
) -> list[dict[str, str]]:
    if max_input_tokens <= 0:
        return messages
    kept = list(messages)
    while len(kept) > 1:
        tokens = render_messages(tokenizer, kept, enable_thinking)
        if int(tokens.shape[-1]) <= max_input_tokens:
            return kept
        if kept[0]["role"] == "system":
            if len(kept) > 2:
                del kept[1:3]
            else:
                del kept[1:]
        else:
            del kept[:2]
    return kept


def generate_reply(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    enable_thinking: bool,
    stream: bool,
    use_kvcache: bool,
    prefill_chunk_tokens: int,
    max_cache_tokens: int,
    recent_window: int,
    hot_cache_tokens: int,
    hot_raw_tokens: int,
    merge_similarity: float,
    attention_decay: float,
    importance_update: float,
    kv_log_every: int,
) -> str:
    input_ids = render_messages(tokenizer, messages, enable_thinking).to(model.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    if use_kvcache:
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        generation_kwargs = {
            key: value for key, value in generation_kwargs.items() if value is not None
        }
        with budgeted_kv_cache(
            model=model,
            max_cache_tokens=max_cache_tokens,
            recent_window=recent_window,
            hot_cache_tokens=hot_cache_tokens,
            hot_raw_tokens=hot_raw_tokens,
            merge_similarity=merge_similarity,
            attention_decay=attention_decay,
            importance_update=importance_update,
            log_every=kv_log_every,
        ):
            if not stream:
                with torch.inference_mode():
                    output = model.generate(**generation_kwargs)
                new_tokens = output[0, input_ids.shape[-1] :]
                return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs["streamer"] = streamer
            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            chunks = []
            for chunk in streamer:
                print(chunk, end="", flush=True)
                chunks.append(chunk)
            thread.join()
            print()
            return "".join(chunks).strip()

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}

    if not stream:
        with torch.inference_mode():
            output = model.generate(**generation_kwargs)
        new_tokens = output[0, input_ids.shape[-1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs["streamer"] = streamer
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    chunks = []
    for chunk in streamer:
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    thread.join()
    print()
    return "".join(chunks).strip()


def save_history(path: Path, messages: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")


def load_history(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive CLI chat for Qwen AWQ models.")
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-32B-AWQ")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-gpu-memory", default="22GiB")
    parser.add_argument("--max-cpu-memory", default="")
    parser.add_argument("--offload-folder", type=Path, default=Path("/root/autodl-tmp/offload"))
    parser.add_argument("--awq-version", choices=["", "gemm", "gemv", "exllama"], default="")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--history-file", type=Path)
    parser.add_argument("--fresh-start", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=24000)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--use-kvcache", action="store_true")
    parser.add_argument("--prefill-chunk-tokens", type=int, default=256)
    parser.add_argument("--max-cache-tokens", type=int, default=2048)
    parser.add_argument("--recent-window", type=int, default=1024)
    parser.add_argument("--hot-cache-tokens", type=int, default=768)
    parser.add_argument("--hot-raw-tokens", type=int, default=-1)
    parser.add_argument("--merge-similarity", type=float, default=0.90)
    parser.add_argument("--attention-decay", type=float, default=0.995)
    parser.add_argument("--importance-update", type=float, default=0.02)
    parser.add_argument("--kv-log-every", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"loading model: {args.model}", flush=True)
    model, tokenizer = load_model(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
        max_gpu_memory=args.max_gpu_memory,
        max_cpu_memory=args.max_cpu_memory,
        offload_folder=args.offload_folder,
        awq_version=args.awq_version,
    )

    if args.history_file and args.history_file.exists() and not args.fresh_start:
        messages = load_history(args.history_file)
    else:
        messages = [{"role": "system", "content": args.system}]

    print("ready. commands: /exit, /reset, /history, /save", flush=True)
    while True:
        try:
            user_text = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text == "/exit":
            break
        if user_text == "/reset":
            messages = [{"role": "system", "content": args.system}]
            print("history reset.")
            continue
        if user_text == "/history":
            for idx, message in enumerate(messages):
                print(f"[{idx}] {message['role']}: {message['content'][:300]}")
            continue
        if user_text == "/save":
            if not args.history_file:
                print("no --history-file was provided.")
            else:
                save_history(args.history_file, messages)
                print(f"saved {args.history_file}")
            continue

        messages.append({"role": "user", "content": user_text})
        messages = trim_history(tokenizer, messages, args.max_input_tokens, args.enable_thinking)
        print("AI> ", end="", flush=True)
        reply = generate_reply(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            enable_thinking=args.enable_thinking,
            stream=not args.no_stream,
            use_kvcache=args.use_kvcache,
            prefill_chunk_tokens=args.prefill_chunk_tokens,
            max_cache_tokens=args.max_cache_tokens,
            recent_window=args.recent_window,
            hot_cache_tokens=args.hot_cache_tokens,
            hot_raw_tokens=args.hot_raw_tokens,
            merge_similarity=args.merge_similarity,
            attention_decay=args.attention_decay,
            importance_update=args.importance_update,
            kv_log_every=args.kv_log_every,
        )
        if args.no_stream:
            print(reply)
        messages.append({"role": "assistant", "content": reply})
        if args.history_file:
            save_history(args.history_file, messages)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"runtime error: {exc}", file=sys.stderr)
        raise
