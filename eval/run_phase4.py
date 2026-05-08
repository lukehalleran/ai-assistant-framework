#!/usr/bin/env python3
"""
Phase 4 runner: generate LLM responses for all (snapshot, variant) pairs.

Usage:
    # Dry run — show what would be generated
    python -m eval.run_phase4 --dry-run

    # Run with default settings (gpt-4o-mini, temp 0.3)
    python -m eval.run_phase4

    # Run with specific model
    python -m eval.run_phase4 --model sonnet-4.5

    # Include reorder variants
    python -m eval.run_phase4 --include-reorder

    # Resume a previous run
    python -m eval.run_phase4 --run-id 20260505_131500

    # Custom rate limit
    python -m eval.run_phase4 --rpm 10
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from eval.harness import (
    BASELINE_VARIANT_ID,
    GenerationHarness,
    HarnessConfig,
    load_run_manifest,
)
from eval.no_store_generation import EvalGenerator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 4 generation harness against captured snapshots."
    )
    parser.add_argument(
        "--snapshot-dir", default="eval/snapshots",
        help="Directory containing snapshot JSON files",
    )
    parser.add_argument(
        "--output", default="eval/runs",
        help="Base directory for run output",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Run ID (for resume). Auto-generated if not provided.",
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="Model name for generation",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Sampling temperature (default: 0.3)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Max response tokens (default: 2048)",
    )
    parser.add_argument(
        "--rpm", type=int, default=30,
        help="Requests per minute rate limit (default: 30)",
    )
    parser.add_argument(
        "--include-reorder", action="store_true",
        help="Include reorder-to-high-attention variants",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Plan pairs and show summary without generating",
    )
    args = parser.parse_args()

    config = HarnessConfig(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        requests_per_minute=args.rpm,
        include_reorder=args.include_reorder,
    )

    # For dry run, we don't need a real model manager
    if args.dry_run:
        generator = EvalGenerator(model_manager=None)
    else:
        # Import ModelManager only when actually generating
        try:
            from models.model_manager import ModelManager
            model_manager = ModelManager()
            generator = EvalGenerator(model_manager=model_manager)
        except Exception as e:
            print(f"ERROR: Could not initialize ModelManager: {e}")
            print("Make sure your .env file has the required API keys.")
            sys.exit(1)

    harness = GenerationHarness(generator=generator, config=config)

    # Load snapshots
    snapshots = harness.load_snapshots(args.snapshot_dir)
    if not snapshots:
        print(f"No snapshots found in {args.snapshot_dir}")
        print("Capture some first with: DAEMON_EVAL_CAPTURE=1 python main.py")
        sys.exit(1)

    print(f"Loaded {len(snapshots)} snapshots")

    # Plan pairs
    pairs = harness.plan_pairs(snapshots)
    baselines = [p for p in pairs if p.variant_id == BASELINE_VARIANT_ID]
    variants = [p for p in pairs if p.variant_id != BASELINE_VARIANT_ID]

    print(f"Planned {len(pairs)} total pairs:")
    print(f"  {len(baselines)} baselines")
    print(f"  {len(variants)} variants")

    # Strategy breakdown
    strategy_counts: dict[str, int] = {}
    for p in pairs:
        strategy_counts[p.strategy] = strategy_counts.get(p.strategy, 0) + 1
    for strategy, count in sorted(strategy_counts.items()):
        print(f"    {strategy}: {count}")

    # Cost estimate
    total_tokens = sum(p.token_count_estimate for p in pairs)
    print(f"\nEstimated input tokens: ~{total_tokens:,}")
    print(f"Model: {config.model}")
    print(f"Temperature: {config.temperature}")
    print(f"Rate limit: {config.requests_per_minute} rpm")

    if args.dry_run:
        print("\n[DRY RUN] No generation performed.")

        # Show first few pairs
        print("\nFirst 10 pairs:")
        for p in pairs[:10]:
            print(f"  {p.snapshot_id[:8]} | {p.strategy:<14} | {p.description[:50]}")
        if len(pairs) > 10:
            print(f"  ... and {len(pairs) - 10} more")
        return

    # Check for resume
    if args.run_id:
        run_dir = Path(args.output) / args.run_id
        if run_dir.exists():
            old_manifest = load_run_manifest(str(run_dir))
            if old_manifest:
                completed = sum(
                    1 for p in old_manifest.pairs if p.get("status") == "completed"
                )
                print(f"\nResuming run {args.run_id}: {completed} already completed")

    # Run
    print(f"\nStarting generation...")
    manifest = asyncio.run(
        harness.run(snapshots, output_dir=args.output, run_id=args.run_id)
    )

    print(f"\n{'='*50}")
    print(f"Run complete: {manifest.run_id}")
    print(f"  Completed: {manifest.completed_pairs}")
    print(f"  Failed:    {manifest.failed_pairs}")
    print(f"  Skipped:   {manifest.skipped_pairs}")
    print(f"  Total time: {manifest.total_generation_time_ms / 1000:.1f}s")
    print(f"  Prompt tokens:   ~{manifest.total_prompt_tokens:,}")
    print(f"  Response tokens: ~{manifest.total_response_tokens:,}")

    if manifest.errors:
        print(f"\n  Errors ({len(manifest.errors)}):")
        for err in manifest.errors[:5]:
            print(f"    {err.get('stage', '?')}: {err.get('error', '?')[:80]}")

    print(f"\nResults saved to: {args.output}/{manifest.run_id}/")


if __name__ == "__main__":
    main()
