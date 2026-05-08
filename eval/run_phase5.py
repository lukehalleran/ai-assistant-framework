#!/usr/bin/env python3
"""
Phase 5 runner: judge baseline vs variant responses from a Phase 4 run.

Usage:
    # Dry run — show pair count
    python -m eval.run_phase5 --gen-run eval/runs/20260506_100331 --dry-run

    # Run with default judge (gpt-4o-mini)
    python -m eval.run_phase5 --gen-run eval/runs/20260506_100331

    # Use a stronger judge model
    python -m eval.run_phase5 --gen-run eval/runs/20260506_100331 --judge-model gpt-4o

    # Resume
    python -m eval.run_phase5 --gen-run eval/runs/20260506_100331 --run-id 20260506_120000

    # Report only (no generation, just analyze existing verdicts)
    python -m eval.run_phase5 --gen-run eval/runs/20260506_100331 --report-only
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from eval.judge import (
    BatchJudge,
    PairwiseJudge,
    aggregate_by_section,
    format_section_report,
    load_verdicts,
)
from eval.no_store_generation import EvalGenerator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 5 pairwise judging on Phase 4 generation results."
    )
    parser.add_argument(
        "--gen-run", required=True,
        help="Path to Phase 4 generation run directory",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for judgments (default: <gen-run>/judgments/)",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Run ID for resume",
    )
    parser.add_argument(
        "--judge-model", default="gpt-4o-mini",
        help="Model to use as judge (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--rpm", type=int, default=30,
        help="Requests per minute rate limit (default: 30)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Plan pairs and show summary without judging",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Skip judging, just analyze existing verdicts and print report",
    )
    args = parser.parse_args()

    gen_run_dir = args.gen_run
    if not Path(gen_run_dir).exists():
        print(f"ERROR: Generation run directory not found: {gen_run_dir}")
        sys.exit(1)

    output_dir = args.output or str(Path(gen_run_dir) / "judgments")

    # Report only mode
    if args.report_only:
        _print_report(output_dir)
        return

    # Set up generator (for judge LLM calls)
    if args.dry_run:
        generator = EvalGenerator(model_manager=None)
    else:
        try:
            from models.model_manager import ModelManager
            model_manager = ModelManager()
            generator = EvalGenerator(model_manager=model_manager)
        except Exception as e:
            print(f"ERROR: Could not initialize ModelManager: {e}")
            sys.exit(1)

    judge = PairwiseJudge(
        generator=generator,
        judge_model=args.judge_model,
    )
    batch = BatchJudge(judge=judge, requests_per_minute=args.rpm)

    # Plan
    pairs = batch.plan_judgments(gen_run_dir)
    print(f"Planned {len(pairs)} judgment pairs")

    strategy_counts: dict[str, int] = {}
    for p in pairs:
        strategy_counts[p["strategy"]] = strategy_counts.get(p["strategy"], 0) + 1
    for strategy, count in sorted(strategy_counts.items()):
        print(f"  {strategy}: {count}")

    if args.dry_run:
        print("\n[DRY RUN] No judging performed.")
        return

    # Run
    print(f"\nJudging with {args.judge_model} at {args.rpm} rpm...")
    manifest = asyncio.run(
        batch.run(gen_run_dir, output_dir=output_dir, run_id=args.run_id)
    )

    print(f"\n{'='*50}")
    print(f"Judging complete: {manifest.run_id}")
    print(f"  Completed: {manifest.completed_pairs}")
    print(f"  Failed:    {manifest.failed_pairs}")
    print(f"  Skipped:   {manifest.skipped_pairs}")
    print(f"  Baseline wins: {manifest.baseline_wins}")
    print(f"  Variant wins:  {manifest.variant_wins}")
    print(f"  Ties:          {manifest.ties}")

    if manifest.errors:
        print(f"\n  Errors ({len(manifest.errors)}):")
        for err in manifest.errors[:5]:
            print(f"    {err.get('variant_id', '?')}: {err.get('error', '?')[:80]}")

    # Print report
    _print_report(output_dir)


def _print_report(output_dir: str) -> None:
    """Find the latest judge run and print the report."""
    judge_base = Path(output_dir)
    if not judge_base.exists():
        print("No judgments found.")
        return

    # Find latest run dir
    run_dirs = sorted(
        [d for d in judge_base.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    if not run_dirs:
        print("No judgment runs found.")
        return

    latest = run_dirs[0]
    verdicts = load_verdicts(str(latest))
    if not verdicts:
        print(f"No verdicts found in {latest}")
        return

    print(f"\nLoaded {len(verdicts)} verdicts from {latest.name}")

    section_stats = aggregate_by_section(verdicts)
    report = format_section_report(section_stats)
    print(f"\n{report}")

    # Save report
    report_path = latest / "section_report.txt"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
