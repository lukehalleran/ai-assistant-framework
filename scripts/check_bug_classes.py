#!/usr/bin/env python3
"""Bug-class catalog inspection and the repo-static scan ratchet.

``inspect`` (B1) checks ``docs/BUG_CLASSES.md`` for internal consistency.

``scan`` runs the DM scanners in ``bug_class_guards/scanners`` against the
content-anchored baseline in ``config/bug_class_baseline.json`` and ratchets
in both directions: a NEW finding fails (exit 1) and a STALE baseline entry
fails too, so fixed debt must be removed from the baseline rather than left
behind.  A gated scanner that processed ZERO files also fails — empty input
is a failure, never a pass (BC-64).  A scanner that raises exits 2: "could
not check" must never read like "nothing wrong" (BC-47).

Baseline entries are written ONLY by an explicit ``--write-baseline`` run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from bug_class_guards import baseline as baseline_mod
from bug_class_guards.catalog import CatalogParseError, catalog_diagnostics, parse_catalog
from bug_class_guards.scanners import SCANNERS, ScanResult
from bug_class_guards.scanners.common import ScannerError

# Report-only scanners can be large; print a bounded sample so the pre-push
# output stays readable. --json always carries every finding.
REPORT_SAMPLE = 10

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2


def inspect_catalog(path: Path) -> int:
    try:
        catalog = parse_catalog(path)
    except CatalogParseError as exc:
        print(f"catalog error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    findings = catalog_diagnostics(catalog)
    print(f"entries: {len(catalog.entries)}")
    print(f"detection methods: {len(catalog.detection_methods)}")
    print(f"closure methods: {len(catalog.closure_methods)}")
    print(f"singleton bullets: {len(catalog.singleton_bullets)}")
    if findings:
        print("diagnostics:")
        for finding in findings:
            print(f"- {finding}")
        return EXIT_FINDINGS
    print("diagnostics: none")
    return EXIT_OK


def _selected(scanner_ids: list[str] | None) -> list[str]:
    if not scanner_ids:
        return list(SCANNERS)
    unknown = [name for name in scanner_ids if name not in SCANNERS]
    if unknown:
        raise SystemExit(f"unknown scanner(s): {', '.join(unknown)}")
    return [name for name in SCANNERS if name in set(scanner_ids)]


def _fmt_row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip()


def _print_table(rows: list[list[str]], stream) -> None:
    header = ["scanner", "mode", "files", "found", "base", "new", "stale"]
    widths = [
        max(len(row[i]) for row in [header, *rows]) for i in range(len(header))
    ]
    print(_fmt_row(header, widths), file=stream)
    print(_fmt_row(["-" * w for w in widths], widths), file=stream)
    for row in rows:
        print(_fmt_row(row, widths), file=stream)


def _describe(key: tuple[str, str, str, str]) -> str:
    scanner, path, symbol, text = key
    scope = f" [{symbol}]" if symbol else ""
    return f"  {scanner}: {path}{scope} {text!r}"


def scan_repo(
    root: Path,
    baseline_path: Path,
    write_baseline: bool,
    scanner_ids: list[str] | None,
    as_json: bool,
) -> int:
    selected = _selected(scanner_ids)
    # With --json, stdout is PURE JSON for tooling and the human report goes to
    # stderr; without it, everything goes to stdout.
    report = sys.stderr if as_json else sys.stdout
    results: dict[str, ScanResult] = {}
    for name in selected:
        try:
            results[name] = SCANNERS[name].scan(root)
        except ScannerError as exc:
            print(f"scanner error: {name}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            print(f"scanner error: {name}: {exc!r}", file=sys.stderr)
            traceback.print_exc(limit=6, file=sys.stderr)
            return EXIT_ERROR

    gated = [name for name in selected if SCANNERS[name].mode == "gate"]
    found_all = baseline_mod.fingerprints(
        finding for name in gated for finding in results[name].findings
    )

    if write_baseline:
        baseline_mod.save(baseline_path, found_all)
        print(
            f"baseline written: {baseline_path} ({sum(found_all.values())} entries)",
            file=report,
        )

    try:
        stored = baseline_mod.load(baseline_path)
    except baseline_mod.BaselineError as exc:
        print(f"baseline error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    stored = baseline_mod.subset(stored, gated)

    comparison = baseline_mod.compare(found_all, stored)

    rows: list[list[str]] = []
    per_scanner_new = {name: 0 for name in selected}
    per_scanner_stale = {name: 0 for name in selected}
    for key in comparison.new:
        per_scanner_new[key[0]] = per_scanner_new.get(key[0], 0) + 1
    for key in comparison.stale:
        per_scanner_stale[key[0]] = per_scanner_stale.get(key[0], 0) + 1

    empty_gated = []
    for name in selected:
        scanner = SCANNERS[name]
        result = results[name]
        gate = scanner.mode == "gate"
        if gate and result.files_processed == 0:
            empty_gated.append(name)
        based = (
            sum(v for k, v in stored.items() if k[0] == name) if gate else None
        )
        rows.append(
            [
                name,
                scanner.mode,
                str(result.files_processed),
                str(len(result.findings)),
                str(based) if gate else "-",
                str(per_scanner_new.get(name, 0)) if gate else "-",
                str(per_scanner_stale.get(name, 0)) if gate else "-",
            ]
        )
    _print_table(rows, report)

    for name in selected:
        if SCANNERS[name].mode != "report":
            continue
        result = results[name]
        if not result.findings:
            continue
        print(
            f"\nreport-only {name} ({len(result.findings)} — never fails the gate):",
            file=report,
        )
        for finding in result.findings[:REPORT_SAMPLE]:
            print(_describe(finding.fingerprint()), file=report)
        hidden = len(result.findings) - REPORT_SAMPLE
        if hidden > 0:
            print(
                f"  … {hidden} more (--scanner {name} --json for all)", file=report
            )

    exit_code = EXIT_OK
    if comparison.new:
        print(
            f"\nNEW findings ({len(comparison.new)}) — fix, or record deliberately:",
            file=report,
        )
        for key in comparison.new:
            print(_describe(key), file=report)
        exit_code = EXIT_FINDINGS
    if comparison.stale:
        print(
            f"\nSTALE baseline entries ({len(comparison.stale)}) — the debt is gone or "
            "moved; delete the entry (never leave it behind):",
            file=report,
        )
        for key in comparison.stale:
            print(_describe(key), file=report)
        exit_code = EXIT_FINDINGS
    if empty_gated:
        print(
            "\nEMPTY INPUT — these gated scanners processed zero files, which is a "
            f"failure, not a pass: {', '.join(empty_gated)}",
            file=report,
        )
        exit_code = EXIT_FINDINGS

    if as_json:
        payload = {
            "root": ".",
            "scanners": {
                name: {
                    "mode": SCANNERS[name].mode,
                    "class_ids": list(SCANNERS[name].class_ids),
                    "files_processed": results[name].files_processed,
                    "findings": [
                        {
                            "path": f.path,
                            "symbol": f.symbol,
                            "line": f.line,
                            "text": f.text,
                        }
                        for f in results[name].findings
                    ],
                }
                for name in selected
            },
            "new": [list(key) for key in comparison.new],
            "stale": [list(key) for key in comparison.stale],
            "baselined": comparison.baselined,
            "empty_gated": empty_gated,
            "exit": exit_code,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False))

    if exit_code == EXIT_OK:
        print("\nbug-class scan: OK", file=report)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_bug_classes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="inspect a Markdown catalog")
    inspect.add_argument("--catalog", type=Path, required=True)

    scan = subparsers.add_parser("scan", help="run the repo-static bug-class scanners")
    scan.add_argument("--root", type=Path, default=Path("."))
    scan.add_argument("--baseline", type=Path, default=None)
    scan.add_argument("--write-baseline", action="store_true")
    scan.add_argument("--scanner", action="append", dest="scanners", default=None)
    scan.add_argument("--json", action="store_true", dest="as_json")

    args = parser.parse_args(argv)
    if args.command == "inspect":
        return inspect_catalog(args.catalog)
    if args.command == "scan":
        root = args.root.resolve()
        baseline_path = args.baseline or (root / baseline_mod.DEFAULT_BASELINE)
        return scan_repo(
            root, baseline_path, args.write_baseline, args.scanners, args.as_json
        )
    parser.error(f"unsupported command: {args.command}")
    return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
