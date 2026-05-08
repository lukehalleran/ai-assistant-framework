#!/usr/bin/env python3
"""
Phase 6 runner: run objective checks on Phase 4 generation results.

Usage:
    python -m eval.run_phase6 --gen-run eval/runs/20260506_100331
    python -m eval.run_phase6 --gen-run eval/runs/20260506_100331 --output eval/phase6_output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.checks import (
    aggregate_checks_by_section,
    format_checks_report,
    run_checks_on_generation_run,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 6 objective checks on Phase 4 generation results."
    )
    parser.add_argument(
        "--gen-run", required=True,
        help="Path to Phase 4 generation run directory",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: <gen-run>/checks/)",
    )
    args = parser.parse_args()

    gen_run_dir = args.gen_run
    if not Path(gen_run_dir).exists():
        print(f"ERROR: Generation run directory not found: {gen_run_dir}")
        return

    output_dir = Path(args.output or f"{gen_run_dir}/checks")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run checks
    print(f"Running checks on {gen_run_dir}...")
    results = run_checks_on_generation_run(gen_run_dir)
    print(f"Checked {len(results)} responses")

    # Count pass/fail per check
    from collections import Counter
    check_totals: dict[str, Counter] = {}
    for r in results:
        for c in r.checks:
            if c.check_name not in check_totals:
                check_totals[c.check_name] = Counter()
            check_totals[c.check_name]["pass" if c.passed else "fail"] += 1

    print("\n--- Overall pass/fail rates ---")
    for check_name, counts in sorted(check_totals.items()):
        total = counts["pass"] + counts["fail"]
        pass_rate = counts["pass"] / total * 100 if total else 0
        print(f"  {check_name:<24s}  pass={counts['pass']:>5d}  fail={counts['fail']:>5d}  ({pass_rate:.0f}%)")

    # Aggregate by section (LOO)
    section_stats = aggregate_checks_by_section(results)
    report = format_checks_report(section_stats)

    # Save
    results_path = output_dir / "check_results.json"
    results_path.write_text(json.dumps(
        [r.to_dict() for r in results], indent=2
    ))

    stats_path = output_dir / "section_stats.json"
    stats_path.write_text(json.dumps(section_stats, indent=2))

    report_path = output_dir / "checks_report.txt"
    report_path.write_text(report)

    print(f"\n{report}")
    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    main()
