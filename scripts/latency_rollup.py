#!/usr/bin/env python3
"""Read-only latency summary: python scripts/latency_rollup.py --days 7.

Uses only the standard library; does not import the application or open stores.
Days are calendar days in the reporting timezone (local time by default).
Wall time is the explicit ingress-to-final-response ``wall_elapsed_s`` field.
Older debug ``total_wall`` values omit answer checks, and nested/concurrent
phases cannot be summed, so historical rows without wall time show n/a.
P90 uses linear interpolation between adjacent sorted observations.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median


def _seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if value >= 0 and math.isfinite(value) else None
    except (OverflowError, ValueError):
        return None


def _summary(values):
    if not values:
        return "median=n/a p90=n/a n=0"
    values = sorted(values)
    position = (len(values) - 1) * 0.9
    lower, upper = math.floor(position), math.ceil(position)
    p90 = values[lower] + (values[upper] - values[lower]) * (position - lower)
    return f"median={median(values):.3f} p90={p90:.3f} n={len(values)}"


def main(argv=None, *, now=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--path", type=Path,
                        default=Path(__file__).resolve().parents[1] / "logs/turn_records.jsonl")
    args = parser.parse_args(argv)
    if args.days < 1:
        parser.error("--days must be positive")
    now = now or datetime.now().astimezone()
    if now.tzinfo is None:
        now = now.astimezone()
    first_day = now.date() - timedelta(days=args.days - 1)
    groups = {}
    ignored = 0
    try:
        with args.path.open("r", encoding="utf-8", errors="replace") as source:
            for line in source:
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or row.get("test_env"):
                        ignored += 1
                        continue
                    timestamp = datetime.fromisoformat(row["ts"])
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=now.tzinfo)
                    timestamp = timestamp.astimezone(now.tzinfo)
                    mode = row.get("mode", "unknown")
                    if not isinstance(mode, str):
                        raise ValueError("malformed mode")
                    if timestamp.date() < first_day or timestamp > now:
                        ignored += 1
                        continue
                except (ValueError, TypeError, KeyError, OverflowError):
                    ignored += 1
                    continue
                tasks = row.get("task_timings")
                tasks = tasks if isinstance(tasks, dict) else {}
                has_images = row.get("has_images") is True or any(
                    "image" in str(key).lower() or "vision" in str(key).lower()
                    for key in tasks
                )
                key = (timestamp.date().isoformat(), mode, "image" if has_images else "text")
                group = groups.setdefault(key, {
                    "n": 0, "wall": [], "prepare": [], "pre_prepare": [],
                    "grounding": [], "tasks": defaultdict(list),
                })
                group["n"] += 1
                for field, bucket in (
                    ("wall_elapsed_s", "wall"), ("prepare_elapsed_s", "prepare"),
                    ("pre_prepare_elapsed_s", "pre_prepare"),
                    ("grounding_verifier_elapsed_s", "grounding"),
                ):
                    value = _seconds(row.get(field))
                    if value is not None:
                        group[bucket].append(value)
                for task, elapsed in tasks.items():
                    value = _seconds(elapsed)
                    if value is not None:
                        group["tasks"][str(task)].append(value)
    except OSError as error:
        print(f"No telemetry available: {error}")
        return 0

    print(f"Latency {first_day} through {now.date()} (seconds; ignored={ignored})")
    for (day, mode, kind), group in sorted(groups.items()):
        print(f"{day} {mode} {kind} n={group['n']} "
              f"wall_s {_summary(group['wall'])} prepare_s {_summary(group['prepare'])} "
              f"pre_prepare_s {_summary(group['pre_prepare'])} grounding_s {_summary(group['grounding'])}")
        top_tasks = sorted(
            ((name, median(values)) for name, values in group["tasks"].items()),
            key=lambda item: (-item[1], item[0]),
        )[:6]
        print("  tasks_s: " + (", ".join(f"{name}={value:.3f}" for name, value in top_tasks) or "n/a"))
    if not groups:
        print("No production turns in the requested window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
