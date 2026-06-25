#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot OOM stress outcomes and kvmanage resource use.")
    parser.add_argument("--csv", type=Path, default=Path("oom.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/oom_stress_chart.png"))
    parser.add_argument("--aggregate-csv", type=Path, default=Path("outputs/oom_stress_aggregated.csv"))
    return parser.parse_args()


def clean_row(raw: dict[str, str]) -> dict[str, str]:
    return {str(key).strip(): value.strip() for key, value in raw.items() if key is not None}


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_gpu_capacity_gb(error: str) -> float | None:
    match = re.search(r"total capacity of\s+([0-9.]+)\s+GiB", error)
    return float(match.group(1)) if match else None


def load_cases(path: Path) -> tuple[list[dict[str, object]], Counter[tuple[str, str]], float | None]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, skipinitialspace=True)
        for raw in reader:
            row = clean_row(raw)
            if not row.get("id") or row.get("id", "").startswith("__summary_"):
                continue
            rows.append(row)

    by_case: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    status_counts: Counter[tuple[str, str]] = Counter()
    capacities = []

    for row in rows:
        mode = row.get("mode", "")
        status = row.get("status", "")
        if mode:
            by_case[row["id"]][mode] = row
            status_counts[(mode, status)] += 1
        capacity = parse_gpu_capacity_gb(row.get("error", ""))
        if capacity is not None:
            capacities.append(capacity)

    cases = []
    for case_id in sorted(by_case):
        full = by_case[case_id].get("full", {})
        kv = by_case[case_id].get("kvmanage", {})
        prompt_tokens = as_float(kv.get("prompt_tokens")) or as_float(full.get("prompt_tokens"))
        if prompt_tokens is None:
            continue
        cases.append(
            {
                "id": case_id,
                "prompt_tokens": prompt_tokens,
                "full_status": full.get("status", ""),
                "kvmanage_status": kv.get("status", ""),
                "kv_elapsed_sec": as_float(kv.get("elapsed_sec")),
                "kv_peak_memory_gb": as_float(kv.get("peak_memory_gb")),
                "kv_generated_tokens": as_float(kv.get("generated_tokens")),
                "kv_compress_calls": as_float(kv.get("compress_calls")),
                "kv_kept_cache_tokens": as_float(kv.get("avg_kept_cache_tokens")),
                "kv_dropped_tokens_total": as_float(kv.get("dropped_tokens_total")),
                "kv_merged_tokens_total": as_float(kv.get("merged_tokens_total")),
                "gpu_capacity_gb": parse_gpu_capacity_gb(full.get("error", "")),
            }
        )

    cases.sort(key=lambda item: float(item["prompt_tokens"]))
    gpu_capacity = sum(capacities) / len(capacities) if capacities else None
    return cases, status_counts, gpu_capacity


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_aggregate(path: Path, cases: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "prompt_tokens",
        "full_status",
        "kvmanage_status",
        "kv_elapsed_sec",
        "kv_peak_memory_gb",
        "kv_generated_tokens",
        "kv_compress_calls",
        "kv_kept_cache_tokens",
        "kv_dropped_tokens_total",
        "kv_merged_tokens_total",
        "gpu_capacity_gb",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow({key: fmt(case.get(key)) for key in fieldnames})


def plot(path: Path, cases: list[dict[str, object]], status_counts: Counter[tuple[str, str]], gpu_capacity: float | None) -> None:
    if not cases:
        raise SystemExit("no kvmanage cases with prompt_tokens found")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 10))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.2, 3.2, 2.4])
    fig.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.08, hspace=0.48)

    ax_status = fig.add_subplot(grid[0])
    modes = ["full", "kvmanage"]
    ok_counts = [status_counts.get((mode, "ok"), 0) for mode in modes]
    oom_counts = [status_counts.get((mode, "oom"), 0) for mode in modes]
    error_counts = [
        sum(count for (mode, status), count in status_counts.items() if mode == item and status not in {"ok", "oom"})
        for item in modes
    ]
    xs = range(len(modes))
    ax_status.bar(xs, ok_counts, color="#2f855a", label="ok")
    ax_status.bar(xs, oom_counts, bottom=ok_counts, color="#c53030", label="oom")
    bottoms = [ok + oom for ok, oom in zip(ok_counts, oom_counts)]
    ax_status.bar(xs, error_counts, bottom=bottoms, color="#805ad5", label="other error")
    for idx, mode in enumerate(modes):
        total = ok_counts[idx] + oom_counts[idx] + error_counts[idx]
        ax_status.text(idx, total + 0.25, str(total), ha="center", va="bottom", fontsize=9)
    ax_status.set_xticks(list(xs), modes)
    ax_status.set_ylabel("Rows")
    ax_status.set_title("OOM Stress Outcomes")
    ax_status.set_ylim(0, max(ok_counts + oom_counts + error_counts + [1]) * 1.25)
    ax_status.grid(True, axis="y", alpha=0.25)
    ax_status.legend(ncol=3, fontsize=9)

    prompt_tokens = [float(item["prompt_tokens"]) for item in cases]
    kv_memory = [float(item["kv_peak_memory_gb"]) for item in cases if item["kv_peak_memory_gb"] is not None]
    memory_x = [float(item["prompt_tokens"]) for item in cases if item["kv_peak_memory_gb"] is not None]

    ax_resource = fig.add_subplot(grid[1])
    mem_line = ax_resource.plot(
        memory_x,
        kv_memory,
        color="#1f77b4",
        marker="o",
        linewidth=2.2,
        markersize=4,
        label="kvmanage peak memory",
    )
    if gpu_capacity is not None:
        ax_resource.axhline(gpu_capacity, color="#c53030", linestyle="--", linewidth=1.4, label="GPU capacity from full OOM")
        ax_resource.fill_between(
            [min(prompt_tokens), max(prompt_tokens)],
            gpu_capacity,
            gpu_capacity * 1.02,
            color="#c53030",
            alpha=0.08,
        )
    ax_resource.set_ylabel("Peak memory (GiB)")
    ax_resource.set_title("KVManage Peak Memory Across Full-KV OOM Prompts")
    ax_resource.grid(True, alpha=0.3)
    lines = mem_line + ax_resource.lines[1:]
    labels = [line.get_label() for line in lines]
    ax_resource.legend(lines, labels, loc="upper left", fontsize=9)

    ax_cache = fig.add_subplot(grid[2], sharex=ax_resource)
    ax_calls = ax_cache.twinx()
    dropped = [float(item["kv_dropped_tokens_total"]) for item in cases if item["kv_dropped_tokens_total"] is not None]
    dropped_x = [float(item["prompt_tokens"]) for item in cases if item["kv_dropped_tokens_total"] is not None]
    kept = [float(item["kv_kept_cache_tokens"]) for item in cases if item["kv_kept_cache_tokens"] is not None]
    kept_x = [float(item["prompt_tokens"]) for item in cases if item["kv_kept_cache_tokens"] is not None]
    calls = [float(item["kv_compress_calls"]) for item in cases if item["kv_compress_calls"] is not None]
    calls_x = [float(item["prompt_tokens"]) for item in cases if item["kv_compress_calls"] is not None]

    dropped_line = ax_cache.plot(
        dropped_x,
        dropped,
        color="#6b46c1",
        marker="o",
        linewidth=2.1,
        markersize=3.8,
        label="dropped/merged tokens",
    )
    kept_line = ax_cache.plot(
        kept_x,
        kept,
        color="#2f855a",
        linestyle="--",
        linewidth=1.8,
        label="kept cache tokens",
    )
    calls_line = ax_calls.plot(
        calls_x,
        calls,
        color="#4a5568",
        marker="^",
        linewidth=1.7,
        markersize=3.6,
        label="compress calls",
    )
    ax_cache.set_xlabel("Prompt tokens")
    ax_cache.set_ylabel("Tokens")
    ax_calls.set_ylabel("Compress calls")
    ax_cache.set_title("KVManage Compression Work")
    ax_cache.grid(True, alpha=0.3)
    lines = dropped_line + kept_line + calls_line
    labels = [line.get_label() for line in lines]
    ax_cache.legend(lines, labels, loc="upper left", fontsize=9)

    full_oom = sum(1 for item in cases if item["full_status"] == "oom")
    kv_ok = sum(1 for item in cases if item["kvmanage_status"] == "ok")
    avg_mem = sum(kv_memory) / len(kv_memory) if kv_memory else 0.0
    fig.text(
        0.01,
        0.01,
        f"Cases: {len(cases)}. Full OOM: {full_oom}. KVManage OK: {kv_ok}. "
        f"KVManage avg peak memory: {avg_mem:.2f} GiB.",
        fontsize=8.5,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cases, status_counts, gpu_capacity = load_cases(args.csv)
    write_aggregate(args.aggregate_csv, cases)
    plot(args.output, cases, status_counts, gpu_capacity)

    kv_memory = [float(item["kv_peak_memory_gb"]) for item in cases if item["kv_peak_memory_gb"] is not None]
    print(f"wrote {args.output}")
    print(f"wrote {args.aggregate_csv}")
    print(f"cases={len(cases)} full_oom={sum(1 for item in cases if item['full_status'] == 'oom')} kvmanage_ok={sum(1 for item in cases if item['kvmanage_status'] == 'ok')}")
    if kv_memory:
        print(f"kvmanage_peak_memory_gb avg={sum(kv_memory) / len(kv_memory):.2f} min={min(kv_memory):.2f} max={max(kv_memory):.2f}")


if __name__ == "__main__":
    main()
