#!/usr/bin/env python3

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterable

import torch


PastKeyValues = tuple[tuple[torch.Tensor, torch.Tensor], ...]


@dataclass
class CompressionStats:
    compress_calls: int = 0
    input_tokens_seen: int = 0
    kept_tokens_sum: int = 0
    dropped_tokens: int = 0
    merged_tokens: int = 0
    hot_tokens_sum: int = 0
    hot_raw_tokens_sum: int = 0
    hot_cluster_tokens_sum: int = 0
    cold_tokens_sum: int = 0

    @property
    def avg_kept_tokens(self) -> float:
        if self.compress_calls == 0:
            return 0.0
        return self.kept_tokens_sum / self.compress_calls

    @property
    def avg_hot_tokens(self) -> float:
        if self.compress_calls == 0:
            return 0.0
        return self.hot_tokens_sum / self.compress_calls

    @property
    def avg_hot_raw_tokens(self) -> float:
        if self.compress_calls == 0:
            return 0.0
        return self.hot_raw_tokens_sum / self.compress_calls

    @property
    def avg_hot_cluster_tokens(self) -> float:
        if self.compress_calls == 0:
            return 0.0
        return self.hot_cluster_tokens_sum / self.compress_calls

    @property
    def avg_cold_tokens(self) -> float:
        if self.compress_calls == 0:
            return 0.0
        return self.cold_tokens_sum / self.compress_calls


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    generated_tokens: int
    elapsed_sec: float
    peak_memory_gb: float
    compression: CompressionStats


@dataclass
class HotKVState:
    importance: torch.Tensor | None = None
    token_positions: torch.Tensor | None = None

    def align(self, length: int, device: torch.device) -> None:
        if self.importance is None or self.token_positions is None:
            self.importance = torch.zeros(length, device=device, dtype=torch.float32)
            self.token_positions = torch.arange(length, device=device, dtype=torch.long)
            return
        self.importance = self.importance.to(device=device, dtype=torch.float32)
        self.token_positions = self.token_positions.to(device=device, dtype=torch.long)
        current = int(self.importance.shape[0])
        if current == length:
            return
        if current > length:
            self.importance = self.importance[-length:].contiguous()
            self.token_positions = self.token_positions[-length:].contiguous()
            return
        add = length - current
        last = int(self.token_positions[-1].item()) if current else -1
        self.importance = torch.cat(
            [self.importance, torch.zeros(add, device=device, dtype=torch.float32)]
        )
        self.token_positions = torch.cat(
            [
                self.token_positions,
                torch.arange(last + 1, last + 1 + add, device=device, dtype=torch.long),
            ]
        )

    def update_from_query(
        self,
        past_key_values,
        query: torch.Tensor,
        update_strength: float,
    ) -> None:
        entries = _cache_entries(past_key_values)
        if not entries or update_strength <= 0:
            return
        key = entries[0][0]
        length = int(key.shape[-2])
        self.align(length, key.device)
        summary = _layer_key_summary(key)
        query_vec = torch.nn.functional.normalize(query.detach().float().reshape(-1), dim=0)
        if int(query_vec.shape[0]) != int(summary.shape[-1]):
            return
        scores = torch.relu(summary @ query_vec)
        self.importance = (1.0 - update_strength) * self.importance + update_strength * scores

    def subset(self, indices: torch.Tensor) -> None:
        if self.importance is None or self.token_positions is None:
            return
        indices = indices.to(self.importance.device)
        self.importance = self.importance.index_select(0, indices).contiguous()
        self.token_positions = self.token_positions.index_select(0, indices).contiguous()


def _cache_entries(past_key_values) -> list[tuple[torch.Tensor, torch.Tensor, object | None]]:
    if past_key_values is None:
        return []
    if isinstance(past_key_values, tuple):
        return [(key, value, None) for key, value in past_key_values]
    if hasattr(past_key_values, "layers"):
        entries = []
        for layer in past_key_values.layers:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor):
                entries.append((key, value, layer))
        return entries
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return [
            (key, value, None)
            for key, value in zip(past_key_values.key_cache, past_key_values.value_cache)
        ]
    try:
        entries = []
        for item in past_key_values:
            key, value = item[0], item[1]
            if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor):
                entries.append((key, value, None))
        return entries
    except TypeError:
        return []


def _cache_seq_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    entries = _cache_entries(past_key_values)
    if not entries:
        return 0
    return int(entries[0][0].shape[-2])


def _replace_cache_entries(original, compressed_layers: PastKeyValues):
    if isinstance(original, tuple):
        return compressed_layers
    entries = _cache_entries(original)
    if entries and len(entries) == len(compressed_layers):
        for (_old_key, _old_value, layer), (new_key, new_value) in zip(entries, compressed_layers):
            if layer is None:
                continue
            layer.keys = new_key
            layer.values = new_value
            layer.is_initialized = True
            if hasattr(layer, "cumulative_length"):
                current = getattr(layer, "cumulative_length")
                if isinstance(current, int):
                    layer.cumulative_length = int(new_key.shape[-2])
                elif isinstance(current, torch.Tensor):
                    current.fill_(int(new_key.shape[-2]))
        return original
    return compressed_layers


def _position_ids(start: int, length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(start, start + length, device=device, dtype=torch.long).unsqueeze(0)


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    greedy: bool,
    repetition_penalty: float = 1.0,
    penalty_token_ids: Iterable[int] | None = None,
) -> torch.Tensor:
    logits = logits[:, -1, :]
    if repetition_penalty != 1.0 and penalty_token_ids:
        token_ids = torch.tensor(list(penalty_token_ids), device=logits.device, dtype=torch.long)
        token_logits = logits.index_select(dim=1, index=token_ids)
        penalized = torch.where(
            token_logits < 0,
            token_logits * repetition_penalty,
            token_logits / repetition_penalty,
        )
        logits.scatter_(dim=1, index=token_ids.unsqueeze(0).expand_as(penalized), src=penalized)
    if greedy or temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        logits = torch.full_like(logits, -float("inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _layer_key_summary(key: torch.Tensor) -> torch.Tensor:
    # [batch, heads, seq, dim] -> [seq, dim], averaged across batch and heads.
    return torch.nn.functional.normalize(key[0].float().mean(dim=0), dim=-1)


def _merge_by_assignment(
    tensor: torch.Tensor,
    selected: torch.Tensor,
    assignment: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    old = tensor[:, :, : int(assignment.shape[0])]
    out = torch.zeros(
        (*old.shape[:2], groups, old.shape[-1]),
        device=old.device,
        dtype=old.dtype,
    )
    out.index_add_(2, assignment.to(old.device), old)
    counts = torch.bincount(assignment.to(old.device), minlength=groups).to(
        device=old.device,
        dtype=old.dtype,
    )

    empty = counts == 0
    counts = counts.clamp_min(1).view(1, 1, groups, 1)
    out = out / counts
    if bool(empty.any()):
        empty_idx = torch.nonzero(empty, as_tuple=False).flatten()
        fallback = tensor.index_select(2, selected.to(tensor.device).index_select(0, empty_idx))
        out[:, :, empty_idx] = fallback
    return out


def compress_past_key_values(
    past_key_values,
    recent_window: int,
    max_cache_tokens: int,
    hot_cache_tokens: int,
    hot_raw_tokens: int,
    merge_similarity: float,
    attention_decay: float,
    hot_state: HotKVState | None = None,
    current_position: int | None = None,
    stats: CompressionStats | None = None,
) -> PastKeyValues:
    """Compress a Transformers past_key_values cache.

    The policy mirrors the simulator at a practical level: recent KV tokens are
    protected, a small set of hot old tokens is retained exactly, redundant hot
    old tokens are merged, and cold old tokens are represented by centroids.
    It is approximate for RoPE-based models because absolute positions are not
    recoverable after merging. Use it for experiments, not as a production kernel.
    """
    seq_len = _cache_seq_len(past_key_values)
    if seq_len <= max_cache_tokens:
        return past_key_values
    if max_cache_tokens <= 0:
        raise ValueError("max_cache_tokens must be positive")
    if hot_cache_tokens < -1:
        raise ValueError("hot_cache_tokens must be -1 or non-negative")
    if hot_raw_tokens < -1:
        raise ValueError("hot_raw_tokens must be -1 or non-negative")
    if not 0.0 <= attention_decay <= 1.0:
        raise ValueError("attention_decay must be between 0 and 1")

    entries = _cache_entries(past_key_values)
    if not entries:
        return past_key_values
    if hot_state is not None:
        hot_state.align(seq_len, entries[0][0].device)

    recent = min(recent_window, max_cache_tokens, seq_len)
    old_count = seq_len - recent
    old_budget = max(0, max_cache_tokens - recent)
    auto_hot_budget = old_budget // 2
    requested_hot_budget = auto_hot_budget if hot_cache_tokens < 0 else hot_cache_tokens
    hot_budget = min(requested_hot_budget, old_budget, old_count)
    cold_budget = max(0, old_budget - hot_budget)
    requested_hot_raw = max(1, hot_budget // 4) if hot_raw_tokens < 0 else hot_raw_tokens
    hot_raw_budget = min(requested_hot_raw, hot_budget, old_count)
    hot_cluster_budget = max(0, hot_budget - hot_raw_budget)

    if old_budget == 0:
        compressed = tuple(
            (k[:, :, -recent:].contiguous(), v[:, :, -recent:].contiguous())
            for k, v, _layer in entries
        )
        if stats is not None:
            stats.compress_calls += 1
            stats.input_tokens_seen += seq_len
            stats.kept_tokens_sum += recent
            stats.dropped_tokens += old_count
        if hot_state is not None:
            keep_indices = torch.arange(seq_len - recent, seq_len, device=entries[0][0].device)
            hot_state.subset(keep_indices)
        return _replace_cache_entries(past_key_values, compressed)

    first_key = entries[0][0]
    old_summary = _layer_key_summary(first_key[:, :, :old_count])
    hot_pool_indices = _select_hot_indices(
        old_count=old_count,
        hot_budget=hot_budget,
        hot_state=hot_state,
        current_position=current_position,
        attention_decay=attention_decay,
        device=first_key.device,
    )
    hot_raw_indices = hot_pool_indices[:hot_raw_budget]
    hot_cluster_source_indices = hot_pool_indices[hot_raw_budget:]
    hot_cluster_selected = _select_hot_cluster_representatives(
        old_summary=old_summary,
        hot_source_indices=hot_cluster_source_indices,
        budget=hot_cluster_budget,
        merge_similarity=merge_similarity,
    )
    if int(hot_cluster_selected.numel()) > 0:
        hot_selected_summary = old_summary.index_select(0, hot_cluster_selected)
        hot_cluster_assignment = torch.argmax(
            old_summary.index_select(0, hot_cluster_source_indices) @ hot_selected_summary.T,
            dim=1,
        ).to(torch.long)
    else:
        hot_cluster_assignment = torch.empty(0, device=first_key.device, dtype=torch.long)

    cold_mask = torch.ones(old_count, device=first_key.device, dtype=torch.bool)
    if int(hot_pool_indices.numel()) > 0:
        cold_mask[hot_pool_indices] = False
    cold_indices = torch.nonzero(cold_mask, as_tuple=False).flatten()
    cold_selected_relative = _select_old_representatives(
        old_summary.index_select(0, cold_indices),
        cold_budget,
        merge_similarity,
    )
    cold_selected = cold_indices.index_select(0, cold_selected_relative)
    if int(cold_selected.numel()) > 0:
        selected_summary = old_summary.index_select(0, cold_selected)
        assignment = torch.argmax(
            old_summary.index_select(0, cold_indices) @ selected_summary.T,
            dim=1,
        ).to(torch.long)
    else:
        assignment = torch.empty(0, device=first_key.device, dtype=torch.long)

    compressed_layers = []
    hot_merged_tokens = max(
        0,
        int(hot_cluster_source_indices.numel()) - int(hot_cluster_selected.numel()),
    )
    merged_tokens = max(0, int(cold_indices.numel()) - int(cold_selected.numel()))
    for key, value, _layer in entries:
        hot_raw_key = (
            key.index_select(2, hot_raw_indices.to(key.device))
            if hot_raw_indices.numel()
            else key[:, :, :0]
        )
        hot_raw_value = (
            value.index_select(2, hot_raw_indices.to(value.device))
            if hot_raw_indices.numel()
            else value[:, :, :0]
        )
        if int(hot_cluster_selected.numel()) > 0:
            hot_cluster_key = _merge_subset_by_assignment(
                key,
                source_indices=hot_cluster_source_indices,
                selected_indices=hot_cluster_selected,
                assignment=hot_cluster_assignment,
            )
            hot_cluster_value = _merge_subset_by_assignment(
                value,
                source_indices=hot_cluster_source_indices,
                selected_indices=hot_cluster_selected,
                assignment=hot_cluster_assignment,
            )
        else:
            hot_cluster_key = key[:, :, :0]
            hot_cluster_value = value[:, :, :0]
        hot_key = torch.cat([hot_raw_key, hot_cluster_key], dim=2)
        hot_value = (
            torch.cat([hot_raw_value, hot_cluster_value], dim=2)
        )
        if int(cold_selected.numel()) > 0:
            cold_key = _merge_subset_by_assignment(
                key,
                source_indices=cold_indices,
                selected_indices=cold_selected,
                assignment=assignment,
            )
            cold_value = _merge_subset_by_assignment(
                value,
                source_indices=cold_indices,
                selected_indices=cold_selected,
                assignment=assignment,
            )
        else:
            cold_key = key[:, :, :0]
            cold_value = value[:, :, :0]
        recent_key = key[:, :, -recent:]
        recent_value = value[:, :, -recent:]
        compressed_layers.append(
            (
                torch.cat([hot_key, cold_key, recent_key], dim=2).contiguous(),
                torch.cat([hot_value, cold_value, recent_value], dim=2).contiguous(),
            )
        )

    if hot_state is not None:
        recent_indices = torch.arange(old_count, seq_len, device=first_key.device)
        keep_indices = torch.cat(
            [hot_raw_indices, hot_cluster_selected, cold_selected, recent_indices],
            dim=0,
        )
        hot_state.subset(keep_indices)

    kept = _cache_seq_len(tuple(compressed_layers))
    if stats is not None:
        stats.compress_calls += 1
        stats.input_tokens_seen += seq_len
        stats.kept_tokens_sum += kept
        stats.dropped_tokens += max(0, seq_len - kept)
        stats.merged_tokens += merged_tokens + hot_merged_tokens
        stats.hot_tokens_sum += int(hot_raw_indices.numel()) + int(hot_cluster_selected.numel())
        stats.hot_raw_tokens_sum += int(hot_raw_indices.numel())
        stats.hot_cluster_tokens_sum += int(hot_cluster_selected.numel())
        stats.cold_tokens_sum += int(cold_selected.numel())
    return _replace_cache_entries(past_key_values, tuple(compressed_layers))


def _merge_subset_by_assignment(
    tensor: torch.Tensor,
    source_indices: torch.Tensor,
    selected_indices: torch.Tensor,
    assignment: torch.Tensor,
) -> torch.Tensor:
    groups = int(selected_indices.shape[0])
    if groups == 0:
        return tensor[:, :, :0]
    source = tensor.index_select(2, source_indices.to(tensor.device))
    out = torch.zeros(
        (*source.shape[:2], groups, source.shape[-1]),
        device=source.device,
        dtype=source.dtype,
    )
    out.index_add_(2, assignment.to(source.device), source)
    counts = torch.bincount(assignment.to(source.device), minlength=groups).to(
        device=source.device,
        dtype=source.dtype,
    )
    empty = counts == 0
    out = out / counts.clamp_min(1).view(1, 1, groups, 1)
    if bool(empty.any()):
        empty_idx = torch.nonzero(empty, as_tuple=False).flatten()
        fallback = tensor.index_select(2, selected_indices.to(tensor.device).index_select(0, empty_idx))
        out[:, :, empty_idx] = fallback
    return out


def _select_hot_indices(
    old_count: int,
    hot_budget: int,
    hot_state: HotKVState | None,
    current_position: int | None,
    attention_decay: float,
    device: torch.device,
) -> torch.Tensor:
    if old_count <= 0 or hot_budget <= 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if hot_state is None or hot_state.importance is None or hot_state.token_positions is None:
        return torch.arange(max(0, old_count - hot_budget), old_count, device=device, dtype=torch.long)
    importance = hot_state.importance[:old_count].to(device=device, dtype=torch.float32)
    positions = hot_state.token_positions[:old_count].to(device=device)
    if current_position is None:
        current_position = int(positions[-1].item()) + 1 if int(positions.numel()) else old_count
    age = torch.clamp(torch.tensor(current_position, device=device) - positions, min=0).float()
    scores = importance * torch.pow(torch.tensor(attention_decay, device=device), age)
    take = min(hot_budget, old_count)
    if take <= 0:
        return torch.empty(0, device=device, dtype=torch.long)
    return torch.topk(scores, k=take, largest=True).indices


def _select_hot_cluster_representatives(
    old_summary: torch.Tensor,
    hot_source_indices: torch.Tensor,
    budget: int,
    merge_similarity: float,
) -> torch.Tensor:
    if budget <= 0 or int(hot_source_indices.numel()) == 0:
        return torch.empty(0, device=old_summary.device, dtype=torch.long)
    if int(hot_source_indices.numel()) <= budget:
        return hot_source_indices

    source_summary = old_summary.index_select(0, hot_source_indices)
    selected_relative = _select_old_representatives(
        source_summary,
        budget=budget,
        merge_similarity=merge_similarity,
    )
    return hot_source_indices.index_select(0, selected_relative)


def _select_old_representatives(
    old_summary: torch.Tensor,
    budget: int,
    merge_similarity: float,
) -> torch.Tensor:
    old_count = int(old_summary.shape[0])
    if budget <= 0:
        return torch.empty(0, device=old_summary.device, dtype=torch.long)
    if old_count <= budget:
        return torch.arange(old_count, device=old_summary.device, dtype=torch.long)

    stride = max(1, old_count // budget)
    candidates = torch.arange(0, old_count, stride, device=old_summary.device, dtype=torch.long)
    candidates = candidates[-budget:]
    if merge_similarity >= 1.0 or candidates.numel() >= budget:
        return candidates[:budget]

    # Add a cheap diversity pass around the strided representatives. This keeps
    # dissimilar old keys when a long prompt has repeated blocks.
    selected = []
    selected_matrix = None
    probe_step = max(1, old_count // (budget * 4))
    for idx in range(0, old_count, probe_step):
        vec = old_summary[idx : idx + 1]
        if selected_matrix is None:
            selected.append(idx)
            selected_matrix = vec
        else:
            sim = torch.max(selected_matrix @ vec.T).item()
            if sim < merge_similarity:
                selected.append(idx)
                selected_matrix = torch.cat([selected_matrix, vec], dim=0)
        if len(selected) >= budget:
            break

    if len(selected) < budget:
        used = set(selected)
        for idx in candidates.tolist():
            if idx not in used:
                selected.append(idx)
            if len(selected) >= budget:
                break

    return torch.tensor(selected[:budget], device=old_summary.device, dtype=torch.long)


def generate_with_budgeted_kv(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    prefill_chunk_tokens: int,
    max_cache_tokens: int,
    recent_window: int,
    hot_cache_tokens: int,
    hot_raw_tokens: int,
    merge_similarity: float,
    attention_decay: float,
    importance_update: float,
    log_every: int,
    stop_after_regex: str,
    stop_after_sentences: int,
    temperature: float,
    top_p: float,
    greedy: bool,
    use_chat_template: bool = False,
    chat_template_enable_thinking: bool | None = None,
    repetition_penalty: float = 1.0,
    stream_callback=None,
) -> GenerationResult:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        chat_kwargs = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        if chat_template_enable_thinking is not None:
            chat_kwargs["enable_thinking"] = chat_template_enable_thinking
        try:
            rendered_prompt = tokenizer.apply_chat_template(messages, **chat_kwargs)
        except TypeError:
            chat_kwargs.pop("enable_thinking", None)
            rendered_prompt = tokenizer.apply_chat_template(messages, **chat_kwargs)
        encoded = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=False)
    else:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

    return generate_with_budgeted_kv_from_input_ids(
        model=model,
        tokenizer=tokenizer,
        input_ids=encoded["input_ids"],
        max_new_tokens=max_new_tokens,
        prefill_chunk_tokens=prefill_chunk_tokens,
        max_cache_tokens=max_cache_tokens,
        recent_window=recent_window,
        hot_cache_tokens=hot_cache_tokens,
        hot_raw_tokens=hot_raw_tokens,
        merge_similarity=merge_similarity,
        attention_decay=attention_decay,
        importance_update=importance_update,
        log_every=log_every,
        stop_after_regex=stop_after_regex,
        stop_after_sentences=stop_after_sentences,
        temperature=temperature,
        top_p=top_p,
        greedy=greedy,
        repetition_penalty=repetition_penalty,
        stream_callback=stream_callback,
    )


def generate_with_budgeted_kv_from_input_ids(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    prefill_chunk_tokens: int,
    max_cache_tokens: int,
    recent_window: int,
    hot_cache_tokens: int,
    hot_raw_tokens: int,
    merge_similarity: float,
    attention_decay: float,
    importance_update: float,
    log_every: int,
    stop_after_regex: str,
    stop_after_sentences: int,
    temperature: float,
    top_p: float,
    greedy: bool,
    repetition_penalty: float = 1.0,
    stream_callback=None,
) -> GenerationResult:
    device = model.device
    input_ids = input_ids.to(device)
    if input_ids.dim() != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("input_ids must have shape [1, sequence_length]")
    if prefill_chunk_tokens <= 0:
        raise ValueError("prefill_chunk_tokens must be positive")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    stats = CompressionStats()
    hot_state = HotKVState()
    start_time = time.perf_counter()

    past_key_values = None
    logits = None
    prompt_len = int(input_ids.shape[1])
    model_limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(model_limit, int) and model_limit > 0 and prompt_len > model_limit:
        raise ValueError(
            f"prompt has {prompt_len} tokens, but model max_position_embeddings is "
            f"{model_limit}. Regenerate a shorter prompt, for example: "
            "python3 scripts/make_long_prompt.py --repeats 300 --output long_prompt.txt"
        )
    cursor = 0
    while cursor < prompt_len:
        chunk = input_ids[:, cursor : cursor + prefill_chunk_tokens]
        position_ids = _position_ids(cursor, int(chunk.shape[1]), device)
        with torch.inference_mode():
            out = model(
                input_ids=chunk,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = out.past_key_values
        logits = out.logits
        hot_state.update_from_query(
            past_key_values,
            query=_last_query_from_cache(past_key_values),
            update_strength=importance_update,
        )
        past_key_values = compress_past_key_values(
            past_key_values=past_key_values,
            recent_window=recent_window,
            max_cache_tokens=max_cache_tokens,
            hot_cache_tokens=hot_cache_tokens,
            hot_raw_tokens=hot_raw_tokens,
            merge_similarity=merge_similarity,
            attention_decay=attention_decay,
            hot_state=hot_state,
            current_position=cursor + int(chunk.shape[1]),
            stats=stats,
        )
        cursor += int(chunk.shape[1])
        if log_every > 0 and (cursor >= prompt_len or cursor % log_every < int(chunk.shape[1])):
            _log_progress(
                phase="prefill",
                done=cursor,
                total=prompt_len,
                cache_tokens=_cache_seq_len(past_key_values),
                stats=stats,
                start_time=start_time,
            )

    generated: list[int] = []
    penalty_token_ids = set(int(token_id) for token_id in input_ids[0].tolist())
    next_token = _sample_next_token(
        logits,
        temperature,
        top_p,
        greedy,
        repetition_penalty=repetition_penalty,
        penalty_token_ids=penalty_token_ids,
    )
    eos_ids = _eos_token_ids(tokenizer)
    absolute_position = prompt_len
    stop_pattern = re.compile(stop_after_regex) if stop_after_regex else None
    streamed_text = ""

    for _ in range(max_new_tokens):
        token_id = int(next_token.item())
        generated.append(token_id)
        penalty_token_ids.add(token_id)
        if stream_callback is not None and token_id not in eos_ids:
            text = tokenizer.decode(generated, skip_special_tokens=True)
            if text.startswith(streamed_text):
                delta = text[len(streamed_text) :]
            else:
                delta = text
            if delta:
                stream_callback(delta)
            streamed_text = text
        if token_id in eos_ids:
            break
        if stop_pattern is not None:
            partial_text = tokenizer.decode(generated, skip_special_tokens=True)
            if stop_pattern.search(partial_text):
                break
        if stop_after_sentences > 0:
            partial_text = tokenizer.decode(generated, skip_special_tokens=True)
            if _sentence_count(partial_text) >= stop_after_sentences:
                break

        with torch.inference_mode():
            out = model(
                input_ids=next_token,
                position_ids=_position_ids(absolute_position, 1, device),
                past_key_values=past_key_values,
                use_cache=True,
            )
        absolute_position += 1
        past_key_values = out.past_key_values
        hot_state.update_from_query(
            past_key_values,
            query=_last_query_from_cache(past_key_values),
            update_strength=importance_update,
        )
        past_key_values = compress_past_key_values(
            past_key_values=past_key_values,
            recent_window=recent_window,
            max_cache_tokens=max_cache_tokens,
            hot_cache_tokens=hot_cache_tokens,
            hot_raw_tokens=hot_raw_tokens,
            merge_similarity=merge_similarity,
            attention_decay=attention_decay,
            hot_state=hot_state,
            current_position=absolute_position,
            stats=stats,
        )
        next_token = _sample_next_token(
            out.logits,
            temperature,
            top_p,
            greedy,
            repetition_penalty=repetition_penalty,
            penalty_token_ids=penalty_token_ids,
        )
        if log_every > 0 and len(generated) % max(1, min(log_every, 32)) == 0:
            _log_progress(
                phase="decode",
                done=len(generated),
                total=max_new_tokens,
                cache_tokens=_cache_seq_len(past_key_values),
                stats=stats,
                start_time=start_time,
            )

    elapsed = time.perf_counter() - start_time
    peak_gb = 0.0
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    return GenerationResult(
        text=tokenizer.decode(generated, skip_special_tokens=True),
        prompt_tokens=prompt_len,
        generated_tokens=len(generated),
        elapsed_sec=elapsed,
        peak_memory_gb=peak_gb,
        compression=stats,
    )


def _eos_token_ids(tokenizer) -> set[int]:
    values: Iterable[int | list[int] | tuple[int, ...] | None] = [
        getattr(tokenizer, "eos_token_id", None)
    ]
    ids = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, int):
            ids.add(value)
        else:
            ids.update(int(item) for item in value)
    return ids


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?。！？](?:\s|$)", text))


def _last_query_from_cache(past_key_values) -> torch.Tensor:
    entries = _cache_entries(past_key_values)
    if not entries:
        return torch.empty(0)
    return entries[0][0][0, :, -1, :].float().mean(dim=0)


def _log_progress(
    phase: str,
    done: int,
    total: int,
    cache_tokens: int,
    stats: CompressionStats,
    start_time: float,
) -> None:
    elapsed = time.perf_counter() - start_time
    rate = done / max(elapsed, 1e-6)
    remaining = max(0, total - done)
    eta = remaining / max(rate, 1e-6)
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    print(
        f"[{phase}] {done}/{total} tokens, "
        f"cache={cache_tokens}, compress={stats.compress_calls}, "
        f"avg_hot={stats.avg_hot_tokens:.1f}, "
        f"avg_hot_raw={stats.avg_hot_raw_tokens:.1f}, "
        f"avg_hot_cluster={stats.avg_hot_cluster_tokens:.1f}, "
        f"avg_cold={stats.avg_cold_tokens:.1f}, "
        f"peak={peak_gb:.2f}GB, elapsed={elapsed:.1f}s, eta={eta:.1f}s",
        flush=True,
    )
