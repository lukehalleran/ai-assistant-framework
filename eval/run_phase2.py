#!/usr/bin/env python3
"""
Phase 2 runner: given a directory of snapshots, generate all variants
and produce a utilization report.

Usage:
    python -m eval.run_phase2 --snapshot-dir eval/snapshots --output eval/phase2_output
    python -m eval.run_phase2 --include-reorder  # also generate reorder variants
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.snapshots import load_snapshot
from eval.variants import VariantGenerator, DEFAULT_BUNDLES
from eval.corpus import CorpusManager
from eval.utilization import UtilizationAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ablation variants and utilization report from snapshots."
    )
    parser.add_argument(
        "--snapshot-dir", default="eval/snapshots",
        help="Directory containing snapshot JSON files",
    )
    parser.add_argument(
        "--output", default="eval/phase2_output",
        help="Directory for output files",
    )
    parser.add_argument(
        "--corpus", default="eval/corpus.json",
        help="Path to corpus JSON file",
    )
    parser.add_argument(
        "--include-reorder", action="store_true",
        help="Include reorder-to-high-attention variants",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load snapshots
    snapshot_dir = Path(args.snapshot_dir)
    snapshots = {}
    if snapshot_dir.exists():
        for path in sorted(snapshot_dir.glob("*.json")):
            try:
                snapshot = load_snapshot(path)
                snapshots[path.stem] = snapshot
            except Exception as e:
                print(f"  WARN: skipping {path.name}: {e}")

    print(f"Loaded {len(snapshots)} snapshots")

    if not snapshots:
        print(
            "No snapshots found. Capture some first with DAEMON_EVAL_CAPTURE=1"
        )
        return

    # Load corpus
    corpus = CorpusManager(Path(args.corpus))
    print(
        f"Corpus: {len(corpus.queries)} queries, "
        f"coverage: {corpus.get_intent_coverage()}"
    )
    intent_gaps = corpus.get_intent_gaps()
    tone_gaps = corpus.get_tone_gaps()
    if intent_gaps:
        print(f"  Intent gaps (< 3 queries): {intent_gaps}")
    if tone_gaps:
        print(f"  Tone gaps (< 2 queries): {tone_gaps}")

    # Generate variants for each snapshot
    generator = VariantGenerator()
    all_variants = {}
    total_variant_count = 0

    for snap_id, snapshot in snapshots.items():
        variants = generator.generate_all(
            snapshot,
            DEFAULT_BUNDLES,
            include_reorder=args.include_reorder,
        )
        total_variant_count += len(variants)
        all_variants[snap_id] = {
            "snapshot_id": snap_id,
            "query_text": snapshot.query_text,
            "variant_count": len(variants),
            "variants": [v.to_dict() for v in variants],
        }

    # Save variant manifest
    manifest_path = output_dir / "variant_manifest.json"
    manifest_path.write_text(json.dumps(all_variants, indent=2))
    print(f"\nVariant manifest: {manifest_path}")
    print(
        f"Total variants across {len(snapshots)} snapshots: "
        f"{total_variant_count}"
    )

    # Run utilization analysis
    analyzer = UtilizationAnalyzer()
    report = analyzer.analyze(snapshots, corpus.queries)

    # Save reports
    report_json_path = output_dir / "utilization_report.json"
    report_json_path.write_text(json.dumps(report.to_dict(), indent=2))

    text_report = analyzer.format_report(report)
    report_txt_path = output_dir / "utilization_report.txt"
    report_txt_path.write_text(text_report)

    print(f"\n{text_report}")
    print(f"\nReports saved to {output_dir}/")


if __name__ == "__main__":
    main()
