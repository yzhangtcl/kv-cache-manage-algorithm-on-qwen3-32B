#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def data_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("category") != "summary" and not row.get("id", "").startswith("__summary_")
    ]


def mode_stats(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    stats: dict[str, Counter] = defaultdict(Counter)
    sums: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        mode = row.get("mode") or "kvmanage"
        status = row.get("status") or "ok"
        stats[mode]["rows"] += 1
        stats[mode][status] += 1
        if status == "ok":
            stats[mode]["scored"] += 1
            if row.get("ok") == "True":
                stats[mode]["correct"] += 1
            elapsed = as_float(row.get("elapsed_sec", ""))
            peak_memory = as_float(row.get("peak_memory_gb", ""))
            if elapsed is not None:
                sums[mode]["elapsed_sec"] += elapsed
                sums[mode]["elapsed_count"] += 1
            if peak_memory is not None:
                sums[mode]["peak_memory_gb"] += peak_memory
                sums[mode]["memory_count"] += 1
    for mode, item in stats.items():
        elapsed_count = sums[mode].get("elapsed_count", 0)
        memory_count = sums[mode].get("memory_count", 0)
        if elapsed_count:
            item["avg_elapsed_sec_x100"] = round(
                100 * sums[mode]["elapsed_sec"] / elapsed_count
            )
        if memory_count:
            item["avg_peak_memory_gb_x100"] = round(
                100 * sums[mode]["peak_memory_gb"] / memory_count
            )
    return {mode: dict(counter) for mode, counter in stats.items()}


def print_mode_stats(rows: list[dict[str, str]]) -> None:
    stats = mode_stats(rows)
    if not stats:
        return
    print()
    print("Per-mode results:")
    print("| mode | rows | scored | correct | accuracy | avg_sec | avg_gb | oom | error |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for mode in sorted(stats):
        item = stats[mode]
        scored = item.get("scored", 0)
        correct = item.get("correct", 0)
        accuracy = correct / scored if scored else 0.0
        avg_sec = item.get("avg_elapsed_sec_x100", 0) / 100
        avg_gb = item.get("avg_peak_memory_gb_x100", 0) / 100
        print(
            f"| {mode} | {item.get('rows', 0)} | {scored} | {correct} | "
            f"{accuracy:.2%} | {avg_sec:.2f} | {avg_gb:.2f} | "
            f"{item.get('oom', 0)} | {item.get('error', 0)} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize full KV vs kvmanage eval CSV.")
    parser.add_argument("csv_path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows = load_rows(args.csv_path)
    rows = data_rows(all_rows)
    by_case: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_case[row["id"]][row.get("mode") or "kvmanage"] = row

    comparable = []
    full_oom = []
    kv_ok_when_full_oom = []
    for case_id, modes in by_case.items():
        full = modes.get("full")
        kv = modes.get("kvmanage")
        if full and full.get("status") == "oom":
            full_oom.append(case_id)
            if kv and kv.get("status") == "ok":
                kv_ok_when_full_oom.append(case_id)
        if not full or not kv:
            continue
        if full.get("status") != "ok" or kv.get("status") != "ok":
            continue
        full_time = as_float(full.get("elapsed_sec", ""))
        kv_time = as_float(kv.get("elapsed_sec", ""))
        full_mem = as_float(full.get("peak_memory_gb", ""))
        kv_mem = as_float(kv.get("peak_memory_gb", ""))
        if full_time and kv_time and full_mem is not None and kv_mem is not None:
            comparable.append((case_id, full_time, kv_time, full_mem, kv_mem, full, kv))

    print(f"# Summary for {args.csv_path}")
    print()
    print(f"- rows: {len(rows)}")
    print(f"- cases with full OOM: {len(full_oom)}")
    print(f"- cases where full OOM but kvmanage succeeded: {len(kv_ok_when_full_oom)}")
    print(f"- comparable successful full/kvmanage pairs: {len(comparable)}")
    print_mode_stats(rows)

    if comparable:
        speedups = [full_time / kv_time for _, full_time, kv_time, _, _, _, _ in comparable]
        mem_ratios = [full_mem / max(kv_mem, 1e-9) for _, _, _, full_mem, kv_mem, _, _ in comparable]
        print(f"- average speedup full/kvmanage: {sum(speedups) / len(speedups):.2f}x")
        print(f"- average peak-memory ratio full/kvmanage: {sum(mem_ratios) / len(mem_ratios):.2f}x")
        print()
        print("| case | full_sec | kv_sec | speedup | full_gb | kv_gb | mem_ratio |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for case_id, full_time, kv_time, full_mem, kv_mem, _full, _kv in comparable:
            print(
                f"| {case_id} | {full_time:.2f} | {kv_time:.2f} | "
                f"{full_time / kv_time:.2f}x | {full_mem:.2f} | {kv_mem:.2f} | "
                f"{full_mem / max(kv_mem, 1e-9):.2f}x |"
            )

    if full_oom:
        print()
        print("Full KV OOM cases:")
        for case_id in full_oom:
            suffix = " (kvmanage ok)" if case_id in kv_ok_when_full_oom else ""
            print(f"- {case_id}{suffix}")


if __name__ == "__main__":
    main()
