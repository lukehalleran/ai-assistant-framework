#!/usr/bin/env python3
"""Privacy-preserving, bounded rollup for one explicit turn telemetry JSONL file."""
from __future__ import annotations

import argparse
import json
import math
import stat
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, BinaryIO
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SCHEMA_VERSION = 1
DEFAULT_MAX_RECORDS = 100_000
DEFAULT_MAX_LINE_BYTES = 1_048_576
MAX_SAMPLE_ROWS = 10

# Only these values may appear in output. Unknown values share one fixed bucket.
KNOWN_MODES = frozenset({
    "agentic-search", "best-of-duel", "doc-generation", "enhanced", "failed",
    "insight-assembly", "self-note", "uncertainty-fallback",
})
KNOWN_MODELS = frozenset({
    "kimi-3", "deepseek-v4", "claude-fable-5", "claude-opus-4.8", "glm-5.2",
    "deepseek-v4.1-flash", "claude-fable-5.1", "astra",
})
KNOWN_PHASES = frozenset({
    "agentic_loop", "context_pipeline", "llm_streaming", "prepare_prompt",
    "prompt_build", "total_wall",
})
KNOWN_TASKS = frozenset({
    "daemon_self_notes", "dreams", "git_commits", "google_calendar", "graph_context",
    "memories", "personal_notes", "proactive_insights", "procedural_skills", "proposed_features",
    "recent", "reference_docs", "reflections", "relevant_emails", "semantic", "summaries",
    "unresolved_threads", "upcoming_schedule", "user_profile", "user_uploads",
    "visual_memories", "web_search", "wiki",
})

# These producer fields are deliberately exact-name expectations; aliases are not guessed.
COVERAGE_FIELDS = {
    "correlation_id": "turn_id",
    "build_version": "build_sha",
    "finish_or_outcome_reason": "outcome_reason_code",
    "time_to_first_token_s": "ttft_s",
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "cache_tokens": "cache_tokens",
    "cost_usd": "cost_usd",
    "provider_retries": "provider_retries",
}
INTEGER_COVERAGE_FIELDS = frozenset({"input_tokens", "output_tokens", "cache_tokens", "provider_retries"})


def _safe_input_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if any(part.casefold() == "data" for part in candidate.parts):
        raise ValueError("input paths containing a data component are forbidden")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("input path cannot be resolved") from exc
    # Check both user spelling and fully resolved target before opening anything.
    if any(part.casefold() == "data" for part in candidate.parts + resolved.parts):
        raise ValueError("input paths containing a data component are forbidden")
    try:
        mode = resolved.stat().st_mode
    except OSError as exc:
        raise ValueError("input is not readable") from exc
    if not stat.S_ISREG(mode):
        raise ValueError("input must be one regular file")
    return resolved


def _parse_timestamp(value: Any, zone: ZoneInfo) -> tuple[datetime | None, bool]:
    if not isinstance(value, str) or not value or len(value) > 128:
        return None, False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, OverflowError):
        return None, False
    naive = parsed.tzinfo is None or parsed.utcoffset() is None
    if naive:
        parsed = parsed.replace(tzinfo=zone)
    try:
        return parsed.astimezone(zone), naive
    except (ValueError, OverflowError):
        return None, naive


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(result) or result < 0:
        return None
    return result


def _coverage_number_valid(public_name: str, value: Any) -> bool:
    if public_name in INTEGER_COVERAGE_FIELDS:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    return _number(value) is not None


def _label(value: Any, allowed: frozenset[str]) -> str:
    if isinstance(value, str) and value in allowed:
        return value
    return "other"


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = percentile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    # Inputs are nonnegative, so the delta form cannot overflow by subtraction.
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "median": _percentile(values, 0.5),
        "p90": _percentile(values, 0.9),
        "max": max(values) if values else None,
    }


def _discard_line(stream: BinaryIO) -> None:
    while True:
        chunk = stream.readline(65536)
        if not chunk or chunk.endswith(b"\n"):
            return


def rollup(path: str, selected_date: date, zone_name: str, *,
           max_records: int = DEFAULT_MAX_RECORDS,
           max_line_bytes: int = DEFAULT_MAX_LINE_BYTES) -> dict[str, Any]:
    if not (1 <= max_records <= DEFAULT_MAX_RECORDS):
        raise ValueError(f"max_records must be between 1 and {DEFAULT_MAX_RECORDS}")
    if not (128 <= max_line_bytes <= DEFAULT_MAX_LINE_BYTES):
        raise ValueError(f"max_line_bytes must be between 128 and {DEFAULT_MAX_LINE_BYTES}")
    try:
        zone = ZoneInfo(zone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError("timezone must be a valid IANA timezone") from exc
    safe_path = _safe_input_path(path)
    start = datetime.combine(selected_date, time.min, tzinfo=zone)
    end = datetime.combine(selected_date + timedelta(days=1), time.min, tzinfo=zone)

    counts = Counter()
    wall_values: list[float] = []
    phases: dict[str, list[float]] = defaultdict(list)
    tasks: dict[str, list[float]] = defaultdict(list)
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    coverage_present = Counter()
    coverage_records = Counter()
    coverage_values = Counter()
    samples: list[dict[str, Any]] = []
    records_seen = 0
    naive_timestamps = 0
    unknown_timing_entries = 0

    # Open the resolved path, never the user-supplied symlink spelling.
    with safe_path.open("rb") as stream:
        while records_seen < max_records:
            raw = stream.readline(max_line_bytes + 1)
            if not raw:
                break
            records_seen += 1
            if len(raw) > max_line_bytes and not raw.endswith(b"\n"):
                _discard_line(stream)
                counts["oversized"] += 1
                continue
            if len(raw) > max_line_bytes:
                counts["oversized"] += 1
                continue
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
                counts["malformed"] += 1
                continue
            if not isinstance(record, dict):
                counts["malformed"] += 1
                continue
            if record.get("test_env") is True or record.get("model") == "test-model":
                counts["test"] += 1
                continue
            timestamp, was_naive = _parse_timestamp(record.get("ts"), zone)
            if timestamp is None:
                counts["invalid_time"] += 1
                continue
            if was_naive:
                naive_timestamps += 1
            if timestamp < start or timestamp >= end:
                counts["out_of_window"] += 1
                continue

            counts["selected"] += 1
            for public_name, field_name in COVERAGE_FIELDS.items():
                value = record.get(field_name)
                if public_name in {"correlation_id", "build_version", "finish_or_outcome_reason"}:
                    if isinstance(value, str) and value.strip() and len(value) <= 128:
                        coverage_present[public_name] += 1
                        coverage_values[public_name] += 1
                elif _coverage_number_valid(public_name, value):
                    coverage_present[public_name] += 1
                    coverage_values[public_name] += 1

            wall = _number(record.get("wall_elapsed_s"))
            if wall is not None:
                wall_values.append(wall)
                coverage_records["wall_elapsed_s"] += 1
                coverage_values["wall_elapsed_s"] += 1
            mode = _label(record.get("mode"), KNOWN_MODES)
            model = _label(record.get("model"), KNOWN_MODELS)
            group = groups.setdefault((mode, model), {"rows": 0, "wall": []})
            group["rows"] += 1
            if wall is not None:
                group["wall"].append(wall)
            row_number = records_seen
            if wall is not None:
                sample = {"row_number": row_number, "wall_elapsed_s": wall,
                          "mode": mode, "model": model}
                if len(samples) < MAX_SAMPLE_ROWS:
                    samples.append(sample)
                else:
                    slowest_sample = min(samples, key=lambda item: (item["wall_elapsed_s"], -item["row_number"]))
                    if (wall, -row_number) > (slowest_sample["wall_elapsed_s"], -slowest_sample["row_number"]):
                        samples.remove(slowest_sample)
                        samples.append(sample)

            for field_name, allowed, target in (
                ("phase_timings", KNOWN_PHASES, phases),
                ("task_timings", KNOWN_TASKS, tasks),
            ):
                section = record.get(field_name)
                if not isinstance(section, dict):
                    continue
                valid_in_record = False
                for key, value in section.items():
                    duration = _number(value)
                    if duration is None:
                        continue
                    valid_in_record = True
                    label = _label(key, allowed)
                    if label == "other":
                        unknown_timing_entries += 1
                        continue
                    coverage_values[field_name] += 1
                    target[label].append(duration)
                if valid_in_record:
                    coverage_records[field_name] += 1

    samples.sort(key=lambda item: (-item["wall_elapsed_s"], item["row_number"]))
    capped = records_seen >= max_records
    coverage = {
        name: {"present": coverage_present[name], "missing_or_invalid": max(0, counts["selected"] - coverage_present[name]),
               "valid_values": coverage_values[name]}
        for name in COVERAGE_FIELDS
    }
    for name in ("wall_elapsed_s", "phase_timings", "task_timings"):
        coverage[name] = {"present": coverage_records[name],
                          "missing_or_invalid": max(0, counts["selected"] - coverage_records[name]),
                          "valid_values": coverage_values[name]}
    group_output = []
    for (mode, model), values in sorted(groups.items()):
        group_output.append({"mode": mode, "model": model, "rows": groups[(mode, model)]["rows"],
                             "wall_elapsed_s": _summary(groups[(mode, model)]["wall"])})
    return {
        "schema_version": SCHEMA_VERSION,
        "selection": {"local_date": selected_date.isoformat(), "timezone": zone_name},
        "counts": {
            "records_examined": records_seen,
            "selected": counts["selected"],
            "malformed": counts["malformed"],
            "oversized": counts["oversized"],
            "test_excluded": counts["test"],
            "invalid_time": counts["invalid_time"],
            "out_of_window": counts["out_of_window"],
            "naive_timestamps": naive_timestamps,
        },
        "truncation": {"record_cap": max_records, "record_cap_reached": capped,
                       "notice": "Input stopped at the hard record cap." if capped else None},
        "coverage": coverage,
        "unknown_timing_entries_dropped": unknown_timing_entries,
        "wall_elapsed_s": _summary(wall_values),
        "phases_elapsed_s": {key: _summary(values) for key, values in sorted(phases.items())},
        "tasks_elapsed_s": {key: _summary(values) for key, values in sorted(tasks.items())},
        "groups": group_output,
        "top_slow_rows": samples[:MAX_SAMPLE_ROWS],
        "privacy": {"free_text_emitted": False, "unknown_labels_bucketed": True,
                    "nested_timings_summed": False},
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", help="one explicit telemetry JSONL file")
    parser.add_argument("--date", required=True, help="selected local date (YYYY-MM-DD)")
    parser.add_argument("--timezone", required=True, help="IANA timezone name")
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument("--max-line-bytes", type=int, default=DEFAULT_MAX_LINE_BYTES)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        selected_date = date.fromisoformat(args.date)
        result = rollup(args.input_jsonl, selected_date, args.timezone,
                        max_records=args.max_records, max_line_bytes=args.max_line_bytes)
    except OSError:
        print("diagnostic_rollup: input could not be read", file=sys.stderr)
        return 2
    except ValueError:
        print("diagnostic_rollup: invalid input, date, timezone, or bounds", file=sys.stderr)
        return 2
    json.dump(result, sys.stdout, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
