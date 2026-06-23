#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np


Array = np.ndarray


def softmax(scores: Array) -> Array:
    if scores.size == 0:
        return scores
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def l2_normalize(x: Array, axis: int = -1, eps: float = 1e-8) -> Array:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


@dataclass
class SequenceBatch:
    queries: Array
    keys: Array
    values: Array
    topic_labels: Array
    num_topics: int


@dataclass
class StepResult:
    output: Array
    pred: int
    target: int
    gpu_items: int
    cpu_items: int
    compute_cost: float
    transfer_cost: float
    moved_items: int
    merged_items: int = 0
    merge_ops: int = 0
    replaced_items: int = 0
    cpu_logical_items: int = 0


@dataclass
class RunMetrics:
    policy: str
    baseline_accuracy: float
    accuracy: float
    accuracy_drop: float
    top1_match: float
    mean_cosine: float
    mse: float
    estimated_cost: float
    baseline_cost: float
    speedup: float
    avg_gpu_items: float
    avg_cpu_items: float
    final_protected_recent: float
    cpu_compression: float
    moved_items: int
    merged_items: int
    merge_ops: int
    replaced_items: int


@dataclass
class CacheMoveStats:
    moved_items: int = 0
    merged_items: int = 0
    merge_ops: int = 0
    replaced_items: int = 0

    def add(self, other: "CacheMoveStats") -> None:
        self.moved_items += other.moved_items
        self.merged_items += other.merged_items
        self.merge_ops += other.merge_ops
        self.replaced_items += other.replaced_items


class AttentionTransitionWorkload:
    """Generates a KV trace from causal attention-transition patterns.

    The trace is driven by a teacher attention distribution over previous KV
    entries. That distribution mixes attention sinks, local recency, a smoothly
    moving focus, and occasional long-range recall, which mirrors common
    decoder attention behavior better than independent synthetic topics.
    """

    def __init__(
        self,
        seq_len: int,
        dim: int,
        labels: int,
        local_window: int,
        sink_tokens: int,
        sink_weight: float,
        local_weight: float,
        focus_weight: float,
        recall_weight: float,
        focus_drift: float,
        transition_noise: float,
        key_prototypes: int,
        key_prototype_noise: float,
        key_stay_prob: float,
        seed: int,
    ) -> None:
        self.seq_len = seq_len
        self.dim = dim
        self.labels = labels
        self.local_window = local_window
        self.sink_tokens = sink_tokens
        self.sink_weight = sink_weight
        self.local_weight = local_weight
        self.focus_weight = focus_weight
        self.recall_weight = recall_weight
        self.focus_drift = focus_drift
        self.transition_noise = transition_noise
        self.key_prototypes = key_prototypes
        self.key_prototype_noise = key_prototype_noise
        self.key_stay_prob = key_stay_prob
        self.rng = np.random.default_rng(seed)

    def generate(self) -> SequenceBatch:
        if self.seq_len <= 1:
            raise ValueError("seq_len must be greater than 1")
        if self.labels <= 1:
            raise ValueError("attention labels must be greater than 1")
        if self.dim < self.labels:
            raise ValueError("dim must be >= attention labels so labels fit in values")
        if self.local_window <= 0:
            raise ValueError("attention local window must be positive")
        if self.sink_tokens < 0:
            raise ValueError("attention sink tokens must be non-negative")
        if self.transition_noise < 0:
            raise ValueError("attention transition noise must be non-negative")
        if self.key_prototypes <= 0:
            raise ValueError("attention key prototypes must be positive")
        if self.key_prototype_noise < 0:
            raise ValueError("attention key prototype noise must be non-negative")
        if not 0.0 <= self.key_stay_prob <= 1.0:
            raise ValueError("attention key stay probability must be between 0 and 1")

        keys, key_prototype_ids = self._sample_keys()
        values = np.zeros((self.seq_len, self.dim), dtype=np.float64)
        labels = self._sample_labels(key_prototype_ids)
        for i, label in enumerate(labels):
            values[i, label] = 1.0
            values[i] += self.rng.normal(scale=0.015, size=self.dim)

        queries = np.empty((self.seq_len, self.dim), dtype=np.float64)
        targets = np.empty(self.seq_len, dtype=np.int64)
        focus = 0

        for step in range(self.seq_len):
            if step == 0:
                queries[step] = keys[step] + self.rng.normal(scale=0.05, size=self.dim)
                targets[step] = 0
                continue

            if step > 1 and self.rng.random() < self.focus_drift:
                focus = self._move_focus(focus, step)

            weights = self._teacher_attention(step, focus)
            attended_key = weights @ keys[:step]
            queries[step] = attended_key + self.rng.normal(
                scale=self.transition_noise, size=self.dim
            )
            full_weights = softmax(12.0 * (keys[:step] @ l2_normalize(queries[step])))
            attended_value = full_weights @ values[:step]
            targets[step] = int(np.argmax(attended_value[: self.labels]))

        return SequenceBatch(
            queries=l2_normalize(queries),
            keys=keys,
            values=values,
            topic_labels=targets,
            num_topics=self.labels,
        )

    def _sample_keys(self) -> tuple[Array, Array]:
        prototype_count = min(self.key_prototypes, self.seq_len)
        centers = l2_normalize(self.rng.normal(size=(prototype_count, self.dim)))
        prototype_ids = np.empty(self.seq_len, dtype=np.int64)
        keys = np.empty((self.seq_len, self.dim), dtype=np.float64)
        current = int(self.rng.integers(prototype_count))

        for step in range(self.seq_len):
            if step > 0 and self.rng.random() > self.key_stay_prob:
                current = int(self.rng.integers(prototype_count))
            prototype_ids[step] = current
            keys[step] = centers[current] + self.rng.normal(
                scale=self.key_prototype_noise, size=self.dim
            )

        return l2_normalize(keys), prototype_ids

    def _sample_labels(self, key_prototype_ids: Array) -> Array:
        labels = np.empty(self.seq_len, dtype=np.int64)
        prototype_labels = self.rng.integers(self.labels, size=self.key_prototypes)
        label = int(prototype_labels[int(key_prototype_ids[0])])
        for step in range(self.seq_len):
            if step == 0 or self.rng.random() < 0.85:
                label = int(prototype_labels[int(key_prototype_ids[step])])
            elif self.rng.random() < 0.18:
                label = int(self.rng.integers(self.labels))
            labels[step] = label
        return labels

    def _move_focus(self, focus: int, step: int) -> int:
        if self.rng.random() < 0.75:
            delta = int(self.rng.choice([-2, -1, 1, 2]))
            return int(np.clip(focus + delta, 0, step - 1))
        return int(self.rng.integers(step))

    def _teacher_attention(self, step: int, focus: int) -> Array:
        scores = np.full(step, 1e-6, dtype=np.float64)

        sink_count = min(self.sink_tokens, step)
        if sink_count > 0 and self.sink_weight > 0:
            sink_idx = np.arange(sink_count)
            scores[sink_idx] += self.sink_weight * np.exp(-sink_idx / max(1.0, sink_count))

        if self.local_weight > 0:
            start = max(0, step - self.local_window)
            local_idx = np.arange(start, step)
            distance = step - local_idx
            scores[local_idx] += self.local_weight * np.exp(
                -distance / max(1.0, self.local_window / 2.0)
            )

        if self.focus_weight > 0:
            idx = np.arange(step)
            scores += self.focus_weight * np.exp(-np.abs(idx - focus) / 6.0)

        if self.recall_weight > 0 and step > self.local_window:
            recall_center = int(self.rng.integers(0, max(1, step - self.local_window)))
            idx = np.arange(step)
            scores += self.recall_weight * np.exp(-np.abs(idx - recall_center) / 8.0)

        return scores / np.sum(scores)


class HierarchicalKVCache:
    """GPU hot cache with CPU offload and optional query-time CPU probing."""

    def __init__(
        self,
        dim: int,
        num_topics: int,
        gpu_capacity: int,
        cpu_gpu_speed_ratio: float,
        transfer_cost: float,
        policy: str,
        probe_every: int,
        probe_topk: int,
        attention_decay: float,
        attention_scale: float,
        cpu_capacity: int,
        merge_similarity: float,
        merge_candidates: int,
        merge_cost_ratio: float,
        replace_candidates: int,
        protected_recent: int,
        adaptive_interval: int,
        adaptive_step: int,
        adaptive_min_recent: int,
        adaptive_max_recent: int,
        adaptive_hot_high: float,
        adaptive_hot_low: float,
        attend_before_insert: bool = False,
    ) -> None:
        if gpu_capacity <= 0:
            raise ValueError("gpu_capacity must be positive")
        if policy not in {
            "recent",
            "hot",
            "hot_cluster",
            "hybrid_cluster",
            "adaptive_hybrid_cluster",
        }:
            raise ValueError(
                "policy must be 'recent', 'hot', 'hot_cluster', 'hybrid_cluster', "
                "or 'adaptive_hybrid_cluster'"
            )
        if not -1.0 <= merge_similarity <= 1.0:
            raise ValueError("merge_similarity must be between -1 and 1")
        if cpu_capacity < 0:
            raise ValueError("cpu_capacity must be non-negative")
        if merge_candidates < 0:
            raise ValueError("merge_candidates must be non-negative")
        if merge_cost_ratio < 0:
            raise ValueError("merge_cost_ratio must be non-negative")
        if replace_candidates < 0:
            raise ValueError("replace_candidates must be non-negative")
        if protected_recent < 0:
            raise ValueError("protected_recent must be non-negative")
        if adaptive_interval < 0:
            raise ValueError("adaptive_interval must be non-negative")
        if adaptive_step < 0:
            raise ValueError("adaptive_step must be non-negative")

        self.dim = dim
        self.num_topics = num_topics
        self.gpu_capacity = gpu_capacity
        self.cpu_gpu_speed_ratio = cpu_gpu_speed_ratio
        self.transfer_cost_per_item = transfer_cost
        self.policy = policy
        self.probe_every = probe_every
        self.probe_topk = probe_topk
        self.attention_decay = attention_decay
        self.attention_scale = attention_scale
        self.cpu_capacity = cpu_capacity
        self.merge_similarity = merge_similarity
        self.merge_candidates = merge_candidates
        self.merge_cost_ratio = merge_cost_ratio
        self.replace_candidates = replace_candidates
        self.protected_recent = min(protected_recent, gpu_capacity)
        self.adaptive_interval = adaptive_interval
        self.adaptive_step = adaptive_step
        self.adaptive_min_recent = min(max(0, adaptive_min_recent), gpu_capacity)
        self.adaptive_max_recent = min(max(0, adaptive_max_recent), gpu_capacity)
        if self.adaptive_min_recent > self.adaptive_max_recent:
            self.adaptive_min_recent, self.adaptive_max_recent = (
                self.adaptive_max_recent,
                self.adaptive_min_recent,
            )
        self.adaptive_hot_high = adaptive_hot_high
        self.adaptive_hot_low = adaptive_hot_low
        self.recent_attention_ema = 0.0
        self.hot_attention_ema = 0.0
        self.probed_attention_ema = 0.0
        self.attention_ema_decay = 0.90
        self.attend_before_insert = attend_before_insert

        self.gpu: list[dict[str, Array | float | int]] = []
        self.cpu: list[dict[str, Array | float | int]] = []
        self.clock = 0
        self.merge_cursor = 0
        self.replace_cursor = 0
        self.next_item_id = 0
        self.merged_ids: set[int] = set()

    def step(self, query: Array, key: Array, value: Array, target: int) -> StepResult:
        self.clock += 1
        self._last_cpu_maintenance_items = 0
        item_id = self.next_item_id
        self.next_item_id += 1
        stats = self._probe_cpu(query)

        if self.attend_before_insert:
            result = self._attend(query, target, stats)
            maintenance_before_insert = self._last_cpu_maintenance_items
            self.gpu.append(
                {
                    "key": key,
                    "value": value,
                    "last_seen": self.clock,
                    "step": self.clock,
                    "importance": 1.0,
                    "count": 1,
                    "ids": {item_id},
                }
            )
            stats.add(self._enforce_capacity())
            result.gpu_items = len(self.gpu)
            result.cpu_items = len(self.cpu)
            result.cpu_logical_items = self._logical_count(self.cpu)
            result.moved_items = stats.moved_items
            result.merged_items = stats.merged_items
            result.merge_ops = stats.merge_ops
            result.replaced_items = stats.replaced_items
            result.transfer_cost = stats.moved_items * self.transfer_cost_per_item
            insert_maintenance = self._last_cpu_maintenance_items - maintenance_before_insert
            result.compute_cost += insert_maintenance * self.merge_cost_ratio
            return result

        self.gpu.append(
            {
                "key": key,
                "value": value,
                "last_seen": self.clock,
                "step": self.clock,
                "importance": 1.0,
                "count": 1,
                "ids": {item_id},
            }
        )
        stats.add(self._enforce_capacity())

        return self._attend(query, target, stats)

    def _attend(self, query: Array, target: int, stats: CacheMoveStats) -> StepResult:
        if not self.gpu:
            return StepResult(
                output=np.zeros(self.dim, dtype=np.float64),
                pred=0,
                target=target,
                gpu_items=0,
                cpu_items=len(self.cpu),
                compute_cost=self._last_probe_items * self.cpu_gpu_speed_ratio,
                transfer_cost=stats.moved_items * self.transfer_cost_per_item,
                moved_items=stats.moved_items,
                merged_items=stats.merged_items,
                merge_ops=stats.merge_ops,
                replaced_items=stats.replaced_items,
                cpu_logical_items=self._logical_count(self.cpu),
            )

        keys = np.vstack([item["key"] for item in self.gpu])
        values = np.vstack([item["value"] for item in self.gpu])
        counts = np.array([int(item.get("count", 1)) for item in self.gpu], dtype=np.float64)
        scores = self._attention_scores(keys, query, counts)
        weights = softmax(scores)
        output = weights @ values
        pred = int(np.argmax(output[: self.num_topics]))

        if self.policy in {
            "hot",
            "hot_cluster",
            "hybrid_cluster",
            "adaptive_hybrid_cluster",
        }:
            self._observe_gpu_partitions(weights)
            self._update_importance(weights)
            self._maybe_adapt_protected_recent()

        gpu_cost = float(len(self.gpu))
        cpu_probe_cost = self._last_probe_items * self.cpu_gpu_speed_ratio
        merge_cost = self._last_cpu_maintenance_items * self.merge_cost_ratio
        transfer_cost = stats.moved_items * self.transfer_cost_per_item
        return StepResult(
            output=output,
            pred=pred,
            target=target,
            gpu_items=len(self.gpu),
            cpu_items=len(self.cpu),
            compute_cost=gpu_cost + cpu_probe_cost + merge_cost,
            transfer_cost=transfer_cost,
            moved_items=stats.moved_items,
            merged_items=stats.merged_items,
            merge_ops=stats.merge_ops,
            replaced_items=stats.replaced_items,
            cpu_logical_items=self._logical_count(self.cpu),
        )

    def _probe_cpu(self, query: Array) -> CacheMoveStats:
        self._last_probe_items = 0
        if self.policy not in {
            "hot",
            "hot_cluster",
            "hybrid_cluster",
            "adaptive_hybrid_cluster",
        }:
            return CacheMoveStats()
        if self.probe_every <= 0 or self.probe_topk <= 0:
            return CacheMoveStats()
        if self.clock % self.probe_every != 0 or not self.cpu:
            return CacheMoveStats()

        self._last_probe_items = len(self.cpu)
        cpu_keys = np.vstack([item["key"] for item in self.cpu])
        cpu_counts = np.array([int(item.get("count", 1)) for item in self.cpu], dtype=np.float64)
        scores = self._attention_scores(cpu_keys, query, cpu_counts)
        take = min(self.probe_topk, len(self.cpu))
        selected = np.argpartition(scores, -take)[-take:]
        selected_set = set(int(i) for i in selected)

        moved_items = []
        kept_cpu = []
        for idx, item in enumerate(self.cpu):
            if idx in selected_set:
                item["last_seen"] = self.clock
                item["probe_promoted_at"] = self.clock
                item["importance"] = float(item["importance"]) + 0.5
                moved_items.append(item)
            else:
                kept_cpu.append(item)

        self.cpu = kept_cpu
        self.gpu.extend(moved_items)
        stats = self._enforce_capacity()
        stats.moved_items += len(moved_items)
        return stats

    def _enforce_capacity(self) -> CacheMoveStats:
        stats = CacheMoveStats()
        while len(self.gpu) > self.gpu_capacity:
            evict_idx = self._choose_evict_idx()
            evicted = self.gpu.pop(evict_idx)
            if self.policy == "recent":
                continue
            stats.moved_items += 1
            if self.policy in {
                "hot_cluster",
                "hybrid_cluster",
                "adaptive_hybrid_cluster",
            }:
                stats.add(self._offload_clustered(evicted))
            else:
                self.cpu.append(evicted)
        return stats

    def _choose_evict_idx(self) -> int:
        if self.policy == "recent":
            return int(np.argmin([int(item["last_seen"]) for item in self.gpu]))

        candidate_indices = self._evictable_indices()
        scores = [self._cache_score(self.gpu[idx]) for idx in candidate_indices]
        return candidate_indices[int(np.argmin(scores))]

    def _evictable_indices(self) -> list[int]:
        if (
            self.policy not in {"hybrid_cluster", "adaptive_hybrid_cluster"}
            or self.protected_recent <= 0
        ):
            return list(range(len(self.gpu)))

        protected_cutoff = self.clock - self.protected_recent + 1
        candidates = [
            idx
            for idx, item in enumerate(self.gpu)
            if int(item.get("step", item["last_seen"])) < protected_cutoff
        ]
        return candidates or list(range(len(self.gpu)))

    def _is_recent_protected(self, item: dict[str, Array | float | int]) -> bool:
        if self.protected_recent <= 0:
            return False
        protected_cutoff = self.clock - self.protected_recent + 1
        return int(item.get("step", item["last_seen"])) >= protected_cutoff

    def _observe_gpu_partitions(self, weights: Array) -> None:
        if self.policy != "adaptive_hybrid_cluster" or not self.gpu:
            return
        recent_weight = 0.0
        hot_weight = 0.0
        probed_weight = 0.0
        for item, attn in zip(self.gpu, weights):
            if self._is_recent_protected(item):
                recent_weight += float(attn)
            else:
                hot_weight += float(attn)
            if int(item.get("probe_promoted_at", -1)) > 0:
                probed_weight += float(attn)
        decay = self.attention_ema_decay
        self.recent_attention_ema = decay * self.recent_attention_ema + (1.0 - decay) * recent_weight
        self.hot_attention_ema = decay * self.hot_attention_ema + (1.0 - decay) * hot_weight
        self.probed_attention_ema = decay * self.probed_attention_ema + (
            1.0 - decay
        ) * probed_weight

    def _maybe_adapt_protected_recent(self) -> None:
        if self.policy != "adaptive_hybrid_cluster":
            return
        if self.adaptive_interval <= 0 or self.adaptive_step <= 0:
            return
        if self.clock % self.adaptive_interval != 0:
            return

        total = self.recent_attention_ema + self.hot_attention_ema
        if total <= 1e-8:
            return
        probed_share = self.probed_attention_ema / total
        if probed_share > self.adaptive_hot_high:
            self.protected_recent = max(
                self.adaptive_min_recent,
                self.protected_recent - self.adaptive_step,
            )
        elif probed_share < self.adaptive_hot_low:
            self.protected_recent = min(
                self.adaptive_max_recent,
                self.protected_recent + self.adaptive_step,
            )

    def _update_importance(self, weights: Array) -> None:
        for item, attn in zip(self.gpu, weights):
            item["importance"] = 0.95 * float(item["importance"]) + float(attn)
            if attn > 1.0 / max(1, len(self.gpu)):
                item["last_seen"] = self.clock

    def _offload_clustered(self, evicted: dict[str, Array | float | int]) -> CacheMoveStats:
        """Offload a cold item using high-similarity merge or bounded replacement."""
        stats = CacheMoveStats()
        if self.cpu_capacity == 0:
            return stats

        merge_idx, similarity = self._find_merge_target(evicted)
        if merge_idx is not None:
            incoming_ids = self._item_ids(evicted)
            newly_merged_ids = incoming_ids - self.merged_ids
            self.merged_ids.update(newly_merged_ids)
            stats.merged_items = len(newly_merged_ids)
            stats.merge_ops = 1
            self._merge_items(self.cpu[merge_idx], evicted, similarity)
            return stats

        if len(self.cpu) < self.cpu_capacity:
            self.cpu.append(evicted)
            return stats

        replace_idx = self._choose_replace_idx()
        if replace_idx is None:
            return stats

        self.cpu[replace_idx] = evicted
        stats.replaced_items = 1
        return stats

    def _find_merge_target(
        self, evicted: dict[str, Array | float | int]
    ) -> tuple[int | None, float]:
        if not self.cpu or self.merge_candidates == 0:
            return None, 0.0

        candidate_indices = self._round_robin_indices(self.merge_cursor, self.merge_candidates)
        self.merge_cursor = (self.merge_cursor + len(candidate_indices)) % max(1, len(self.cpu))
        if not candidate_indices:
            return None, 0.0

        self._last_cpu_maintenance_items += len(candidate_indices)
        evicted_key = evicted["key"]
        scores = np.array([float(self.cpu[idx]["key"] @ evicted_key) for idx in candidate_indices])
        best_local = int(np.argmax(scores))
        best_score = float(scores[best_local])
        if best_score < self.merge_similarity:
            return None, best_score
        return candidate_indices[best_local], best_score

    def _choose_replace_idx(self) -> int | None:
        if not self.cpu or self.replace_candidates == 0:
            return None

        candidate_indices = self._round_robin_indices(self.replace_cursor, self.replace_candidates)
        self.replace_cursor = (self.replace_cursor + len(candidate_indices)) % max(1, len(self.cpu))
        if not candidate_indices:
            return None

        self._last_cpu_maintenance_items += len(candidate_indices)
        scores = [self._cache_score(self.cpu[idx]) for idx in candidate_indices]
        return candidate_indices[int(np.argmin(scores))]

    def _round_robin_indices(self, start: int, limit: int) -> list[int]:
        take = min(limit, len(self.cpu))
        if take <= 0:
            return []
        return [(start + offset) % len(self.cpu) for offset in range(take)]

    def _cache_score(self, item: dict[str, Array | float | int]) -> float:
        age = self.clock - int(item["last_seen"])
        return float(item["importance"]) * (self.attention_decay**age)

    def _merge_items(
        self,
        representative: dict[str, Array | float | int],
        incoming: dict[str, Array | float | int],
        similarity: float,
    ) -> None:
        rep_count = int(representative.get("count", 1))
        in_count = int(incoming.get("count", 1))
        total = rep_count + in_count

        representative["key"] = l2_normalize(
            (representative["key"] * rep_count + incoming["key"] * in_count) / total
        )
        representative["value"] = (
            representative["value"] * rep_count + incoming["value"] * in_count
        ) / total
        representative["importance"] = (
            float(representative["importance"]) * rep_count
            + float(incoming["importance"]) * in_count
        ) / total
        representative["last_seen"] = max(int(representative["last_seen"]), int(incoming["last_seen"]))
        representative["count"] = total
        representative["ids"] = self._item_ids(representative) | self._item_ids(incoming)
        representative["merge_similarity_sum"] = float(
            representative.get("merge_similarity_sum", 0.0)
        ) + similarity * in_count

    def _logical_count(self, items: list[dict[str, Array | float | int]]) -> int:
        return int(sum(int(item.get("count", 1)) for item in items))

    def _item_ids(self, item: dict[str, Array | float | int]) -> set[int]:
        ids = item.get("ids")
        if isinstance(ids, set):
            return set(ids)
        return set()

    def _attention_scores(self, keys: Array, query: Array, counts: Array) -> Array:
        return self.attention_scale * (keys @ query) + np.log(counts)


def run_full_kv_baseline(
    batch: SequenceBatch,
    attention_scale: float,
    attend_before_insert: bool,
) -> list[StepResult]:
    keys: list[Array] = []
    values: list[Array] = []
    results: list[StepResult] = []

    for query, key, value, target in zip(
        batch.queries, batch.keys, batch.values, batch.topic_labels
    ):
        if attend_before_insert:
            result = attend_full_kv(
                query=query,
                keys=keys,
                values=values,
                target=int(target),
                dim=batch.keys.shape[1],
                num_topics=batch.num_topics,
                attention_scale=attention_scale,
            )
            keys.append(key)
            values.append(value)
            result.gpu_items = len(keys)
        else:
            keys.append(key)
            values.append(value)
            result = attend_full_kv(
                query=query,
                keys=keys,
                values=values,
                target=int(target),
                dim=batch.keys.shape[1],
                num_topics=batch.num_topics,
                attention_scale=attention_scale,
            )
        results.append(result)

    return results


def attend_full_kv(
    query: Array,
    keys: list[Array],
    values: list[Array],
    target: int,
    dim: int,
    num_topics: int,
    attention_scale: float,
) -> StepResult:
    if not keys:
        return StepResult(
            output=np.zeros(dim, dtype=np.float64),
            pred=0,
            target=target,
            gpu_items=0,
            cpu_items=0,
            compute_cost=0.0,
            transfer_cost=0.0,
            moved_items=0,
        )

    key_matrix = np.vstack(keys)
    value_matrix = np.vstack(values)
    weights = softmax(attention_scale * (key_matrix @ query))
    output = weights @ value_matrix
    pred = int(np.argmax(output[:num_topics]))
    return StepResult(
        output=output,
        pred=pred,
        target=target,
        gpu_items=len(keys),
        cpu_items=0,
        compute_cost=float(len(keys)),
        transfer_cost=0.0,
        moved_items=0,
    )


def tiered_full_kv_cost(
    steps: int,
    gpu_capacity: int,
    cpu_capacity: int,
    cpu_gpu_speed_ratio: float,
    disk_cpu_speed_ratio: float,
    attend_before_insert: bool,
) -> float:
    """Cost of exact full-KV attention on ordinary GPU -> CPU -> disk tiers."""
    total = 0.0
    disk_gpu_speed_ratio = cpu_gpu_speed_ratio * disk_cpu_speed_ratio
    for step in range(steps):
        n_items = step if attend_before_insert else step + 1
        gpu_items = min(n_items, gpu_capacity)
        remaining = max(0, n_items - gpu_items)
        cpu_items = min(remaining, cpu_capacity)
        disk_items = max(0, remaining - cpu_items)
        total += (
            gpu_items
            + cpu_items * cpu_gpu_speed_ratio
            + disk_items * disk_gpu_speed_ratio
        )
    return float(total)


def run_hierarchical(
    batch: SequenceBatch,
    baseline: list[StepResult],
    gpu_capacity: int,
    speed_ratio: float,
    transfer_cost: float,
    policy: str,
    probe_every: int,
    probe_topk: int,
    attention_decay: float,
    attention_scale: float,
    cpu_capacity: int,
    merge_similarity: float,
    merge_candidates: int,
    merge_cost_ratio: float,
    replace_candidates: int,
    baseline_cost: float,
    protected_recent: int,
    adaptive_interval: int,
    adaptive_step: int,
    adaptive_min_recent: int,
    adaptive_max_recent: int,
    adaptive_hot_high: float,
    adaptive_hot_low: float,
    attend_before_insert: bool,
) -> RunMetrics:
    cache = HierarchicalKVCache(
        dim=batch.keys.shape[1],
        num_topics=batch.num_topics,
        gpu_capacity=gpu_capacity,
        cpu_gpu_speed_ratio=speed_ratio,
        transfer_cost=transfer_cost,
        policy=policy,
        probe_every=probe_every,
        probe_topk=probe_topk,
        attention_decay=attention_decay,
        attention_scale=attention_scale,
        cpu_capacity=cpu_capacity,
        merge_similarity=merge_similarity,
        merge_candidates=merge_candidates,
        merge_cost_ratio=merge_cost_ratio,
        replace_candidates=replace_candidates,
        protected_recent=protected_recent,
        adaptive_interval=adaptive_interval,
        adaptive_step=adaptive_step,
        adaptive_min_recent=adaptive_min_recent,
        adaptive_max_recent=adaptive_max_recent,
        adaptive_hot_high=adaptive_hot_high,
        adaptive_hot_low=adaptive_hot_low,
        attend_before_insert=attend_before_insert,
    )
    results = [
        cache.step(q, k, v, int(target))
        for q, k, v, target in zip(batch.queries, batch.keys, batch.values, batch.topic_labels)
    ]

    baseline_outputs = np.vstack([result.output for result in baseline])
    outputs = np.vstack([result.output for result in results])
    baseline_preds = np.array([result.pred for result in baseline])
    preds = np.array([result.pred for result in results])
    targets = np.array([result.target for result in results])

    cosine = np.sum(baseline_outputs * outputs, axis=1) / (
        np.linalg.norm(baseline_outputs, axis=1) * np.linalg.norm(outputs, axis=1) + 1e-8
    )
    total_cost = sum(result.compute_cost + result.transfer_cost for result in results)
    cpu_logical_items = np.array([result.cpu_logical_items for result in results], dtype=np.float64)
    cpu_physical_items = np.array([result.cpu_items for result in results], dtype=np.float64)
    total_cpu_logical = float(np.sum(cpu_logical_items))
    total_cpu_physical = float(np.sum(cpu_physical_items))
    return RunMetrics(
        policy=policy,
        baseline_accuracy=float(np.mean(baseline_preds == targets)),
        accuracy=float(np.mean(preds == targets)),
        accuracy_drop=float(np.mean(baseline_preds == targets) - np.mean(preds == targets)),
        top1_match=float(np.mean(baseline_preds == preds)),
        mean_cosine=float(np.mean(cosine)),
        mse=float(np.mean((baseline_outputs - outputs) ** 2)),
        estimated_cost=float(total_cost),
        baseline_cost=float(baseline_cost),
        speedup=float(baseline_cost / max(total_cost, 1e-8)),
        avg_gpu_items=float(np.mean([result.gpu_items for result in results])),
        avg_cpu_items=float(np.mean([result.cpu_items for result in results])),
        final_protected_recent=float(cache.protected_recent),
        cpu_compression=float(total_cpu_logical / max(total_cpu_physical, 1.0)),
        moved_items=int(sum(result.moved_items for result in results)),
        merged_items=int(sum(result.merged_items for result in results)),
        merge_ops=int(sum(result.merge_ops for result in results)),
        replaced_items=int(sum(result.replaced_items for result in results)),
    )


def print_metrics(metrics: Iterable[RunMetrics]) -> None:
    rows = []
    for item in metrics:
        rows.append(
            {
                "policy": item.policy,
                "base_acc": f"{item.baseline_accuracy:.2%}",
                "accuracy": f"{item.accuracy:.2%}",
                "acc_drop": f"{item.accuracy_drop:.2%}",
                "top1": f"{item.top1_match:.2%}",
                "cos": f"{item.mean_cosine:.4f}",
                "mse": f"{item.mse:.5f}",
                "cost": f"{item.estimated_cost:.1f}",
                "base_cost": f"{item.baseline_cost:.1f}",
                "speedup": f"{item.speedup:.2f}x",
                "gpu": f"{item.avg_gpu_items:.1f}",
                "cpu": f"{item.avg_cpu_items:.1f}",
                "prot": f"{item.final_protected_recent:.1f}",
                "cpu_comp": f"{item.cpu_compression:.2f}x",
                "moved": f"{item.moved_items:d}",
                "merged": f"{item.merged_items:d}",
                "mops": f"{item.merge_ops:d}",
                "repl": f"{item.replaced_items:d}",
            }
        )

    columns = [
        ("policy", "policy", 24, "<"),
        ("base_acc", "base_acc", 9, ">"),
        ("accuracy", "accuracy", 9, ">"),
        ("acc_drop", "acc_drop", 9, ">"),
        ("top1", "top1", 9, ">"),
        ("cos", "cos", 8, ">"),
        ("mse", "mse", 9, ">"),
        ("cost", "cost", 10, ">"),
        ("base_cost", "base_cost", 10, ">"),
        ("speedup", "speedup", 8, ">"),
        ("gpu", "gpu", 7, ">"),
        ("cpu", "cpu", 7, ">"),
        ("prot", "prot", 7, ">"),
        ("cpu_comp", "cpu_comp", 9, ">"),
        ("moved", "moved", 6, ">"),
        ("merged", "merged", 6, ">"),
        ("mops", "mops", 5, ">"),
        ("repl", "repl", 5, ">"),
    ]

    def format_cell(value: str, width: int, align: str) -> str:
        return f"{value:{align}{width}}"

    header = "  ".join(format_cell(label, width, align) for _key, label, width, align in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            "  ".join(
                format_cell(row[key], width, align)
                for key, _label, width, align in columns
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full-KV baseline behavior with CPU-offloaded hierarchical KV cache."
    )
    parser.add_argument("--workload", choices=["attention"], default="attention")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument(
        "--attention-labels",
        type=int,
        default=32,
        help="Number of output labels encoded in value vectors.",
    )
    parser.add_argument(
        "--attention-local-window",
        type=int,
        default=64,
        help="Recent-token window used by the teacher attention trace.",
    )
    parser.add_argument(
        "--attention-sink-tokens",
        type=int,
        default=4,
        help="Initial tokens that act as persistent attention sinks.",
    )
    parser.add_argument("--attention-sink-weight", type=float, default=0.25)
    parser.add_argument("--attention-local-weight", type=float, default=0.50)
    parser.add_argument("--attention-focus-weight", type=float, default=0.20)
    parser.add_argument("--attention-recall-weight", type=float, default=0.05)
    parser.add_argument(
        "--attention-focus-drift",
        type=float,
        default=0.18,
        help="Probability that the long-range focus moves at each decoding step.",
    )
    parser.add_argument(
        "--attention-transition-noise",
        type=float,
        default=0.04,
        help="Noise added when converting teacher attention into query vectors.",
    )
    parser.add_argument(
        "--attention-key-prototypes",
        type=int,
        default=128,
        help="Number of key prototype vectors used to create clustered KV keys.",
    )
    parser.add_argument(
        "--attention-key-prototype-noise",
        type=float,
        default=0.08,
        help="Noise added around each key prototype before normalization.",
    )
    parser.add_argument(
        "--attention-key-stay-prob",
        type=float,
        default=0.92,
        help="Probability that consecutive tokens keep using the same key prototype.",
    )
    parser.add_argument("--gpu-capacity", type=int, default=128)
    parser.add_argument("--cpu-gpu-speed-ratio", type=float, default=10.0)
    parser.add_argument(
        "--disk-cpu-speed-ratio",
        type=float,
        default=10.0,
        help="Estimated disk KV processing cost relative to CPU KV processing.",
    )
    parser.add_argument("--transfer-cost", type=float, default=2.0)
    parser.add_argument("--probe-every", type=int, default=32)
    parser.add_argument("--probe-topk", type=int, default=8)
    parser.add_argument("--attention-decay", type=float, default=0.98)
    parser.add_argument("--attention-scale", type=float, default=12.0)
    parser.add_argument(
        "--protected-recent",
        type=int,
        default=0,
        help=(
            "Recent GPU entries protected from hot eviction in hybrid_cluster. "
            "0 means 87.5%% of gpu_capacity."
        ),
    )
    parser.add_argument(
        "--adaptive-interval",
        type=int,
        default=64,
        help="Steps between protected-recent adjustments in adaptive_hybrid_cluster.",
    )
    parser.add_argument(
        "--adaptive-step",
        type=int,
        default=0,
        help="Protected-recent adjustment size. 0 means 1/32 of gpu_capacity.",
    )
    parser.add_argument(
        "--adaptive-min-recent",
        type=int,
        default=0,
        help="Minimum protected recent window. 0 means 75%% of gpu_capacity.",
    )
    parser.add_argument(
        "--adaptive-max-recent",
        type=int,
        default=0,
        help="Maximum protected recent window. 0 means 95%% of gpu_capacity.",
    )
    parser.add_argument(
        "--adaptive-hot-high",
        type=float,
        default=0.55,
        help="Shrink recent window if probed CPU-return attention share rises above this.",
    )
    parser.add_argument(
        "--adaptive-hot-low",
        type=float,
        default=0.15,
        help="Grow recent window if probed CPU-return attention share falls below this.",
    )
    parser.add_argument(
        "--cpu-capacity",
        type=int,
        default=0,
        help="Max physical CPU KV entries for hot_cluster. 0 means 4 * gpu_capacity.",
    )
    parser.add_argument(
        "--merge-similarity",
        type=float,
        default=0.90,
        help="Cosine threshold for merging cold CPU KV entries in hot_cluster.",
    )
    parser.add_argument(
        "--merge-candidates",
        type=int,
        default=16,
        help="Max CPU representatives tested per offloaded item in hot_cluster.",
    )
    parser.add_argument(
        "--replace-candidates",
        type=int,
        default=16,
        help="Max CPU representatives tested when hot_cluster must replace a full CPU cache.",
    )
    parser.add_argument(
        "--merge-cost-ratio",
        type=float,
        default=0.10,
        help="Estimated cost of one merge-candidate similarity check in GPU-item units.",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def build_workload(args: argparse.Namespace) -> tuple[SequenceBatch, bool]:
    workload = AttentionTransitionWorkload(
        seq_len=args.seq_len,
        dim=args.dim,
        labels=args.attention_labels,
        local_window=args.attention_local_window,
        sink_tokens=args.attention_sink_tokens,
        sink_weight=args.attention_sink_weight,
        local_weight=args.attention_local_weight,
        focus_weight=args.attention_focus_weight,
        recall_weight=args.attention_recall_weight,
        focus_drift=args.attention_focus_drift,
        transition_noise=args.attention_transition_noise,
        key_prototypes=args.attention_key_prototypes,
        key_prototype_noise=args.attention_key_prototype_noise,
        key_stay_prob=args.attention_key_stay_prob,
        seed=args.seed,
    )
    return workload.generate(), True


def main() -> None:
    args = parse_args()
    cpu_capacity = args.cpu_capacity or args.gpu_capacity * 4
    protected_recent = args.protected_recent or max(1, int(args.gpu_capacity * 0.875))
    adaptive_step = args.adaptive_step or max(1, int(args.gpu_capacity / 32))
    adaptive_min_recent = args.adaptive_min_recent or max(1, int(args.gpu_capacity * 0.75))
    adaptive_max_recent = args.adaptive_max_recent or max(
        adaptive_min_recent, int(args.gpu_capacity * 0.95)
    )
    batch, attend_before_insert = build_workload(args)
    baseline = run_full_kv_baseline(
        batch,
        attention_scale=args.attention_scale,
        attend_before_insert=attend_before_insert,
    )
    tiered_baseline_cost = tiered_full_kv_cost(
        steps=batch.keys.shape[0],
        gpu_capacity=args.gpu_capacity,
        cpu_capacity=cpu_capacity,
        cpu_gpu_speed_ratio=args.cpu_gpu_speed_ratio,
        disk_cpu_speed_ratio=args.disk_cpu_speed_ratio,
        attend_before_insert=attend_before_insert,
    )
    baseline_cost = tiered_baseline_cost

    metrics = [
        run_hierarchical(
            batch=batch,
            baseline=baseline,
            gpu_capacity=args.gpu_capacity,
            speed_ratio=args.cpu_gpu_speed_ratio,
            transfer_cost=args.transfer_cost,
            policy="recent",
            probe_every=0,
            probe_topk=0,
            attention_decay=args.attention_decay,
            attention_scale=args.attention_scale,
            cpu_capacity=cpu_capacity,
            merge_similarity=args.merge_similarity,
            merge_candidates=args.merge_candidates,
            merge_cost_ratio=args.merge_cost_ratio,
            replace_candidates=args.replace_candidates,
            baseline_cost=baseline_cost,
            protected_recent=0,
            adaptive_interval=args.adaptive_interval,
            adaptive_step=adaptive_step,
            adaptive_min_recent=adaptive_min_recent,
            adaptive_max_recent=adaptive_max_recent,
            adaptive_hot_high=args.adaptive_hot_high,
            adaptive_hot_low=args.adaptive_hot_low,
            attend_before_insert=attend_before_insert,
        ),
        run_hierarchical(
            batch=batch,
            baseline=baseline,
            gpu_capacity=args.gpu_capacity,
            speed_ratio=args.cpu_gpu_speed_ratio,
            transfer_cost=args.transfer_cost,
            policy="hot",
            probe_every=args.probe_every,
            probe_topk=args.probe_topk,
            attention_decay=args.attention_decay,
            attention_scale=args.attention_scale,
            cpu_capacity=cpu_capacity,
            merge_similarity=args.merge_similarity,
            merge_candidates=args.merge_candidates,
            merge_cost_ratio=args.merge_cost_ratio,
            replace_candidates=args.replace_candidates,
            baseline_cost=baseline_cost,
            protected_recent=0,
            adaptive_interval=args.adaptive_interval,
            adaptive_step=adaptive_step,
            adaptive_min_recent=adaptive_min_recent,
            adaptive_max_recent=adaptive_max_recent,
            adaptive_hot_high=args.adaptive_hot_high,
            adaptive_hot_low=args.adaptive_hot_low,
            attend_before_insert=attend_before_insert,
        ),
        run_hierarchical(
            batch=batch,
            baseline=baseline,
            gpu_capacity=args.gpu_capacity,
            speed_ratio=args.cpu_gpu_speed_ratio,
            transfer_cost=args.transfer_cost,
            policy="hot_cluster",
            probe_every=args.probe_every,
            probe_topk=args.probe_topk,
            attention_decay=args.attention_decay,
            attention_scale=args.attention_scale,
            cpu_capacity=cpu_capacity,
            merge_similarity=args.merge_similarity,
            merge_candidates=args.merge_candidates,
            merge_cost_ratio=args.merge_cost_ratio,
            replace_candidates=args.replace_candidates,
            baseline_cost=baseline_cost,
            protected_recent=0,
            adaptive_interval=args.adaptive_interval,
            adaptive_step=adaptive_step,
            adaptive_min_recent=adaptive_min_recent,
            adaptive_max_recent=adaptive_max_recent,
            adaptive_hot_high=args.adaptive_hot_high,
            adaptive_hot_low=args.adaptive_hot_low,
            attend_before_insert=attend_before_insert,
        ),
        run_hierarchical(
            batch=batch,
            baseline=baseline,
            gpu_capacity=args.gpu_capacity,
            speed_ratio=args.cpu_gpu_speed_ratio,
            transfer_cost=args.transfer_cost,
            policy="hybrid_cluster",
            probe_every=args.probe_every,
            probe_topk=args.probe_topk,
            attention_decay=args.attention_decay,
            attention_scale=args.attention_scale,
            cpu_capacity=cpu_capacity,
            merge_similarity=args.merge_similarity,
            merge_candidates=args.merge_candidates,
            merge_cost_ratio=args.merge_cost_ratio,
            replace_candidates=args.replace_candidates,
            baseline_cost=baseline_cost,
            protected_recent=protected_recent,
            adaptive_interval=args.adaptive_interval,
            adaptive_step=adaptive_step,
            adaptive_min_recent=adaptive_min_recent,
            adaptive_max_recent=adaptive_max_recent,
            adaptive_hot_high=args.adaptive_hot_high,
            adaptive_hot_low=args.adaptive_hot_low,
            attend_before_insert=attend_before_insert,
        ),
        run_hierarchical(
            batch=batch,
            baseline=baseline,
            gpu_capacity=args.gpu_capacity,
            speed_ratio=args.cpu_gpu_speed_ratio,
            transfer_cost=args.transfer_cost,
            policy="adaptive_hybrid_cluster",
            probe_every=args.probe_every,
            probe_topk=args.probe_topk,
            attention_decay=args.attention_decay,
            attention_scale=args.attention_scale,
            cpu_capacity=cpu_capacity,
            merge_similarity=args.merge_similarity,
            merge_candidates=args.merge_candidates,
            merge_cost_ratio=args.merge_cost_ratio,
            replace_candidates=args.replace_candidates,
            baseline_cost=baseline_cost,
            protected_recent=protected_recent,
            adaptive_interval=args.adaptive_interval,
            adaptive_step=adaptive_step,
            adaptive_min_recent=adaptive_min_recent,
            adaptive_max_recent=adaptive_max_recent,
            adaptive_hot_high=args.adaptive_hot_high,
            adaptive_hot_low=args.adaptive_hot_low,
            attend_before_insert=attend_before_insert,
        ),
    ]

    print(
        f"workload={args.workload}, sequence={batch.keys.shape[0]}, dim={args.dim}, "
        f"gpu_capacity={args.gpu_capacity}, cpu_capacity={cpu_capacity}, "
        f"protected_recent={protected_recent}"
    )
    print(
        f"adaptive_min_recent={adaptive_min_recent}, "
        f"adaptive_max_recent={adaptive_max_recent}, adaptive_step={adaptive_step}, "
        f"adaptive_interval={args.adaptive_interval}"
    )
    print(
        f"attention_labels={args.attention_labels}, "
        f"attention_local_window={args.attention_local_window}, "
        f"attention_sink_tokens={args.attention_sink_tokens}, "
        f"attend_before_insert={attend_before_insert}"
    )
    print(
        f"attention_weights=sink:{args.attention_sink_weight:g}, "
        f"local:{args.attention_local_weight:g}, "
        f"focus:{args.attention_focus_weight:g}, "
        f"recall:{args.attention_recall_weight:g}, "
        f"focus_drift={args.attention_focus_drift:g}, "
        f"transition_noise={args.attention_transition_noise:g}"
    )
    print(
        f"attention_key_prototypes={args.attention_key_prototypes}, "
        f"attention_key_prototype_noise={args.attention_key_prototype_noise:g}, "
        f"attention_key_stay_prob={args.attention_key_stay_prob:g}"
    )
    print(
        f"cpu/gpu speed ratio={args.cpu_gpu_speed_ratio:g}x, "
        f"disk/cpu speed ratio={args.disk_cpu_speed_ratio:g}x, "
        f"transfer_cost={args.transfer_cost:g}, attention_scale={args.attention_scale:g}"
    )
    print(
        f"speedup_baseline=tiered_full_kv, "
        f"tiered_full_kv_baseline={tiered_baseline_cost:.1f}"
    )
    print(
        f"merge_similarity={args.merge_similarity:g}, "
        f"merge_candidates={args.merge_candidates}, "
        f"replace_candidates={args.replace_candidates}, "
        f"merge_cost_ratio={args.merge_cost_ratio:g}"
    )
    print_metrics(metrics)


if __name__ == "__main__":
    main()
