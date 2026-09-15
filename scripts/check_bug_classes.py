#!/usr/bin/env python3
"""Bug-class catalog inspection, the repo-static scan ratchet, and its receipts.

``inspect`` checks ``docs/BUG_CLASSES.md`` for internal consistency.

``scan --root .`` evaluates the enforcement contract in
``config/bug_class_policy.json``, in this order, and reports every stage:

1. policy validity, then full parity between the policy and the scanner
   registry (a deleted, renamed, re-moded, re-classed, narrowed or unreviewed
   scanner stops here);
2. the shallow top-level root inventory (full scans only);
3. a syntax preflight over every declared Python input, before any
   scanner-specific matching;
4. every scanner, with a receipt per input leg and per root;
5. the strict baseline (a missing file is never "no debt") and the reviewed
   dispositions ledger;
6. the two-way ratchet for the gate scanners, then disposition reconciliation.

Exit 2: the contract could not be evaluated (usage, policy, registry, syntax,
unreadable input, scanner error, invalid baseline or ledger).  Exit 1: it was
evaluated and failed (NEW or STALE occurrences, unreviewed/orphaned/re-review
dispositions, a missing or empty required input, an unclassified Python root).
Exit 0: every required gate contract passed; report-only evidence is labelled
and never affects the exit.

``--scanner`` is an explicit PARTIAL scan: it still validates the policy,
registry, selected inputs and ledger syntax, compares only the selection, and
never claims a full scan.  ``--write-baseline`` is a developer bootstrap only
(full scope, never in hook or CI), and its rows still need reviewed
dispositions.  ``--json`` prints the report on stdout; ``--receipt PATH``
writes the same report to a file for ``verify-receipts``.

This is a scoped structural-detector lane; the report lists the catalog
classes no scanner covers.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

from bug_class_guards import baseline as baseline_mod
from bug_class_guards import dispositions as dispositions_mod
from bug_class_guards import receipts as receipts_mod
from bug_class_guards.catalog import CatalogParseError, catalog_diagnostics, parse_catalog
from bug_class_guards.policy import (
    POLICY_PATH,
    PolicyError,
    evaluate_inventory,
    load_policy,
    observed_contract,
    registry_mismatches,
)
from bug_class_guards.scanners import SCANNERS
from bug_class_guards.scanners.common import ScannerError, relpath, resolve_leg

# Report-only scanners can be large; print a bounded sample so the pre-push
# output stays readable. --json always carries every finding.
REPORT_SAMPLE = 10
REPORT_SCHEMA = 1

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2

FATAL_CODES = frozenset(
    {
        "usage",
        "policy_invalid",
        "registry_mismatch",
        "syntax_error",
        "read_error",
        "scanner_error",
        "baseline_invalid",
        "dispositions_invalid",
    }
)


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


# ---------------------------------------------------------------- the scan


def _display(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _class_key(class_id: str) -> tuple[int, str]:
    digits = class_id.split("-", 1)[-1]
    return (int(digits) if digits.isdigit() else 10**6, class_id)


def _anchor_row(anchor, excerpt: str) -> dict:
    row = dict(zip(baseline_mod.ANCHOR_FIELDS, anchor))
    row["excerpt"] = excerpt
    return row


class _Scan:
    def __init__(self, root: Path, scanner_ids: list[str] | None):
        self.root = root
        self.partial = bool(scanner_ids)
        self.requested = list(scanner_ids or [])
        self.results = {}
        self.report = {
            "report_schema": REPORT_SCHEMA,
            "command": "scan",
            "root": ".",
            "source_sha": os.environ.get("GITHUB_SHA") or None,
            "scope": "partial" if self.partial else "full",
            "selected_scanners": [],
            "claim": None,
            "policy": None,
            "registry": {"expected": [], "observed": [], "mismatches": []},
            "inventory": {"evaluated": False},
            "syntax": {"files_checked": 0, "errors": []},
            "scanners": {},
            "coverage": {
                "catalog_classes": None,
                "gate_classes": [],
                "report_only_classes": [],
                "unavailable_report_classes": [],
                "uncovered_classes": [],
                "uncovered_count": None,
            },
            "baseline": {"path": None, "occurrences": None, "by_scanner": {}},
            "dispositions": {
                "path": None,
                "active": None,
                "confirmed_fixed": None,
                "false_positive": None,
                "legacy_occurrences": None,
            },
            "comparison": {"compared_scanners": [], "not_compared": [], "matched": 0, "new": [], "stale": []},
            "failures": [],
            "exit": EXIT_OK,
        }
        self._failure_keys: set[tuple[str, str]] = set()

    def fail(self, code: str, detail: str) -> None:
        if (code, detail) in self._failure_keys:
            return
        self._failure_keys.add((code, detail))
        self.report["failures"].append({"code": code, "detail": detail})

    @property
    def fatal(self) -> bool:
        return any(failure["code"] in FATAL_CODES for failure in self.report["failures"])

    def exit_code(self) -> int:
        if self.fatal:
            return EXIT_ERROR
        return EXIT_FINDINGS if self.report["failures"] else EXIT_OK

    # -- stages ---------------------------------------------------------

    def run(self, policy_path, baseline_path, dispositions_path, write_baseline) -> None:
        unknown = [sid for sid in self.requested if sid not in SCANNERS]
        if unknown:
            self.fail("usage", f"unknown scanner(s): {', '.join(unknown)}")
            return
        if write_baseline and self.partial:
            self.fail("usage", "--write-baseline needs the full scope; a partial write would drop other gates' rows")
            return

        path = policy_path or (self.root / POLICY_PATH)
        try:
            policy = load_policy(path)
        except PolicyError as exc:
            self.fail("policy_invalid", str(exc))
            return
        self.policy = policy
        self.report["claim"] = policy.claim
        self.report["policy"] = {"path": _display(self.root, path), "version": policy.version, "sha256": policy.sha256}
        self.report["registry"] = {
            "expected": [contract.to_json() for contract in policy.scanners],
            "observed": [observed_contract(scanner).to_json() for scanner in SCANNERS.values()],
            "mismatches": registry_mismatches(policy, SCANNERS),
        }
        for mismatch in self.report["registry"]["mismatches"]:
            self.fail("registry_mismatch", mismatch)
        if self.fatal:
            return

        selected = [sid for sid in policy.scanner_ids if not self.partial or sid in self.requested]
        self.selected = selected
        self.report["selected_scanners"] = selected
        for contract in policy.scanners:
            scanner = SCANNERS[contract.id]
            chosen = contract.id in selected
            self.report["scanners"][contract.id] = {
                "mode": contract.mode,
                "class_ids": list(contract.class_ids),
                "contract_version": contract.contract_version,
                "selected": chosen,
                "status": "pending" if chosen else "not_selected",
                "files_processed": 0,
                "finding_count": 0,
                "unresolved": 0,
                "findings": [],
                "legs": [
                    {
                        "id": leg.id,
                        "required": leg.required,
                        "status": "not_selected",
                        "available": False,
                        "files_processed": 0,
                        "unresolved": 0,
                        "roots": [],
                    }
                    for leg in scanner.legs
                ],
            }

        if not self.partial:
            inventory = evaluate_inventory(self.root, policy)
            self.report["inventory"] = inventory.to_json()
            if inventory.unclassified_python:
                self.fail(
                    "inventory_unclassified",
                    "top-level Python root(s) not classified by the policy: "
                    + ", ".join(inventory.unclassified_python),
                )

        self._syntax_preflight()
        if self.fatal:
            return
        self._run_scanners()
        if self.fatal:
            return
        self._coverage()
        self._ratchet(baseline_path, dispositions_path, write_baseline)

    def _syntax_preflight(self) -> None:
        files: dict[str, Path] = {}
        for sid in self.selected:
            for leg in SCANNERS[sid].legs:
                if not leg.is_python:
                    continue
                for path in resolve_leg(self.root, leg).files:
                    files.setdefault(relpath(self.root, path), path)
        errors = []
        for rel in sorted(files):
            try:
                source = files[rel].read_bytes().decode("utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                self.fail("read_error", f"{rel}: {exc}")
                continue
            try:
                ast.parse(source, filename=rel)
            except (SyntaxError, ValueError) as exc:
                errors.append({
                    "path": rel,
                    "line": getattr(exc, "lineno", None),
                    "message": getattr(exc, "msg", None) or str(exc),
                })
        self.report["syntax"] = {"files_checked": len(files), "errors": errors}
        for error in errors:
            self.fail("syntax_error", f"{error['path']}:{error['line']}: {error['message']}")

    def _run_scanners(self) -> None:
        for sid in self.selected:
            scanner = SCANNERS[sid]
            try:
                result = scanner.scan(self.root)
            except ScannerError as exc:
                self.fail("scanner_error", f"{sid}: {exc}")
                return
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                traceback.print_exc(limit=6, file=sys.stderr)
                self.fail("scanner_error", f"{sid}: {exc!r}")
                return
            self.results[sid] = result
            entry = self.report["scanners"][sid]
            entry["legs"] = [leg.to_json() for leg in result.legs]
            entry["files_processed"] = result.files_processed
            entry["findings"] = [finding.to_json() for finding in result.findings]
            entry["finding_count"] = len(result.findings)
            entry["unresolved"] = result.unresolved
            incomplete = False
            for leg in result.legs:
                if not leg.required or leg.available:
                    continue
                incomplete = True
                code = "leg_missing" if leg.status == "missing" else "leg_empty"
                for root in leg.roots:
                    if root.status != "available":
                        self.fail(code, f"required leg {leg.id} root {root.path} is {root.status}")
            if incomplete:
                entry["status"] = "incomplete"
            elif not any(leg.available for leg in result.legs):
                entry["status"] = "unavailable"
            else:
                entry["status"] = "ok"

    def _coverage(self) -> None:
        policy = self.policy
        gate = {c for contract in policy.scanners if contract.mode == "gate" for c in contract.class_ids}
        report_only = {
            c for contract in policy.scanners if contract.mode == "report" for c in contract.class_ids
        } - gate
        usable = set(gate)
        unavailable = set()
        for contract in policy.scanners:
            status = self.report["scanners"][contract.id]["status"]
            if contract.mode == "report" and status == "ok":
                usable.update(contract.class_ids)
            if contract.mode == "report" and status == "unavailable":
                unavailable.update(contract.class_ids)
        coverage = self.report["coverage"]
        coverage["gate_classes"] = sorted(gate, key=_class_key)
        coverage["report_only_classes"] = sorted(report_only, key=_class_key)
        coverage["unavailable_report_classes"] = sorted(unavailable - usable, key=_class_key)
        catalog_path = self.root / policy.catalog_path
        if catalog_path.is_file():
            try:
                catalog = parse_catalog(catalog_path)
            except CatalogParseError:
                return
            ids = set(catalog.entries) | set(catalog.index)
            uncovered = sorted(ids - gate - report_only, key=_class_key)
            coverage["catalog_classes"] = len(ids)
            coverage["uncovered_classes"] = uncovered
            coverage["uncovered_count"] = len(uncovered)

    def _ratchet(self, baseline_path, dispositions_path, write_baseline) -> None:
        policy = self.policy
        gate_kinds = {c.id: c.kinds for c in policy.scanners if c.mode == "gate"}
        known = policy.scanner_ids
        gate_ids = [sid for sid in self.selected if sid in gate_kinds]

        path = baseline_path or (self.root / policy.baseline_path)
        self.report["baseline"]["path"] = _display(self.root, path)
        if write_baseline:
            incomplete = [sid for sid in gate_ids if self.report["scanners"][sid]["status"] != "ok"]
            if incomplete:
                self.fail("usage", f"--write-baseline refused: gate input incomplete for {', '.join(incomplete)}")
                return
            findings = [f for sid in gate_ids for f in self.results[sid].findings]
            baseline_mod.save(path, baseline_mod.from_findings(findings))
            print(f"baseline written: {path} (bootstrap only; dispositions still need review)", file=sys.stderr)
        try:
            stored = baseline_mod.load(path, gate_kinds, known)
        except baseline_mod.BaselineError as exc:
            self.fail("baseline_invalid", str(exc))
            return
        self.report["baseline"].update(occurrences=stored.occurrences, by_scanner=stored.by_scanner())

        ledger_path = dispositions_path or (self.root / policy.dispositions_path)
        self.report["dispositions"]["path"] = _display(self.root, ledger_path)
        try:
            ledger = dispositions_mod.load(ledger_path, gate_kinds, known)
        except dispositions_mod.DispositionsError as exc:
            self.fail("dispositions_invalid", str(exc))
            return
        self.report["dispositions"].update(
            active=ledger.count("accepted_debt"),
            confirmed_fixed=ledger.count("confirmed_fixed"),
            false_positive=ledger.count("false_positive"),
            legacy_occurrences=ledger.legacy_records,
        )

        compared = [sid for sid in gate_ids if self.report["scanners"][sid]["status"] == "ok"]
        comparison = self.report["comparison"]
        comparison["compared_scanners"] = compared
        comparison["not_compared"] = [sid for sid in gate_ids if sid not in compared]
        found: Counter = Counter()
        excerpts = dict(stored.excerpts)
        for sid in compared:
            for finding in self.results[sid].findings:
                found[finding.fingerprint()] += 1
                excerpts.setdefault(finding.fingerprint(), finding.excerpt)
        current_excerpts = {
            finding.fingerprint(): finding.excerpt
            for sid in compared
            for finding in self.results[sid].findings
        }
        result = baseline_mod.compare(found, baseline_mod.subset(stored.counter, compared))
        comparison["matched"] = result.matched
        comparison["new"] = [_anchor_row(a, current_excerpts.get(a, "")) for a in result.new]
        comparison["stale"] = [_anchor_row(a, stored.excerpts.get(a, "")) for a in result.stale]
        if result.new:
            self.fail("new_findings", f"{len(result.new)} NEW gate occurrence(s) — fix, or review and record deliberately")
        if result.stale:
            self.fail("stale_baseline", f"{len(result.stale)} STALE baseline occurrence(s) — the debt is gone or changed; remove the row")
        for problem in dispositions_mod.reconcile(
            ledger, stored.counter, self.root, compared, full_scope=not self.partial
        ):
            self.fail(problem.code, problem.detail)


# ------------------------------------------------------------- the report


def _fmt_row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip()


def _print_human(report: dict, stream) -> None:
    policy = report.get("policy") or {}
    header = f"bug-class scan — scope: {report['scope']}"
    if policy:
        header += f" — policy {policy['version']} (sha256 {policy['sha256'][:12]}…)"
    print(header, file=stream)

    baseline_counts = report["baseline"].get("by_scanner") or {}
    new = Counter(row["scanner"] for row in report["comparison"]["new"])
    stale = Counter(row["scanner"] for row in report["comparison"]["stale"])
    compared = set(report["comparison"]["compared_scanners"])
    rows = []
    for sid, entry in report["scanners"].items():
        status = entry["status"]
        legs = entry["legs"]
        ran = status in {"ok", "incomplete"}
        gate = entry["mode"] == "gate"
        rows.append([
            sid,
            entry["mode"],
            status,
            f"{sum(1 for leg in legs if leg['available'])}/{len(legs)}",
            str(entry["files_processed"]) if ran else "-",
            str(entry["finding_count"]) if ran else "-",
            str(entry["unresolved"]) if ran else "-",
            str(baseline_counts.get(sid, 0)) if gate and sid in compared else "-",
            str(new.get(sid, 0)) if gate and sid in compared else "-",
            str(stale.get(sid, 0)) if gate and sid in compared else "-",
        ])
    if rows:
        head = ["scanner", "mode", "status", "legs", "files", "found", "unres", "base", "new", "stale"]
        widths = [max(len(r[i]) for r in [head, *rows]) for i in range(len(head))]
        print(_fmt_row(head, widths), file=stream)
        print(_fmt_row(["-" * w for w in widths], widths), file=stream)
        for row in rows:
            print(_fmt_row(row, widths), file=stream)

    for sid, entry in report["scanners"].items():
        for leg in entry["legs"]:
            if leg["available"] or leg["status"] == "not_selected":
                continue
            label = "required" if leg["required"] else "optional report evidence"
            roots = ", ".join(f"{r['path']} {r['status']}" for r in leg["roots"] if r["status"] != "available")
            note = "" if leg["required"] else f" — {', '.join(entry['class_ids'])} NOT scanned by this leg"
            print(f"  {sid} leg {leg['id']} ({label}): {leg['status']} [{roots}]{note}", file=stream)

    coverage = report["coverage"]
    if coverage["catalog_classes"] is not None:
        covered = coverage["catalog_classes"] - coverage["uncovered_count"]
        print(
            f"\ncoverage (scoped structural lane, not behavioral coverage): {covered} of "
            f"{coverage['catalog_classes']} catalog classes have a scanner "
            f"(gate {len(coverage['gate_classes'])}, report-only {len(coverage['report_only_classes'])}); "
            f"{coverage['uncovered_count']} uncovered: {', '.join(coverage['uncovered_classes']) or 'none'}",
            file=stream,
        )
    if coverage["unavailable_report_classes"]:
        print(f"unavailable report evidence: {', '.join(coverage['unavailable_report_classes'])}", file=stream)

    inventory = report["inventory"]
    if inventory.get("evaluated"):
        if inventory["unscanned_python_present"]:
            print(f"python roots outside every scanner (classified): {', '.join(inventory['unscanned_python_present'])}", file=stream)
        if inventory["not_inspected"]:
            print(f"top-level entries not inspected: {', '.join(inventory['not_inspected'])}", file=stream)
        if inventory["unclassified_non_python"]:
            print(f"unclassified non-Python roots: {', '.join(inventory['unclassified_non_python'])}", file=stream)

    for sid, entry in report["scanners"].items():
        if entry["mode"] != "report" or not entry["findings"]:
            continue
        print(f"\nreport-only {sid} ({entry['finding_count']}, {entry['unresolved']} unresolved — never fails the gate):", file=stream)
        for finding in entry["findings"][:REPORT_SAMPLE]:
            print(f"  {finding['path']} [{finding['symbol']}] {finding['kind']} {finding['excerpt']!r}", file=stream)
        hidden = entry["finding_count"] - REPORT_SAMPLE
        if hidden > 0:
            print(f"  … {hidden} more (--json for all)", file=stream)

    for title, key in (("NEW gate occurrences", "new"), ("STALE baseline occurrences", "stale")):
        rows = report["comparison"][key]
        if rows:
            print(f"\n{title} ({len(rows)}):", file=stream)
            for row in rows:
                print(f"  {row['scanner']}: {row['path']} [{row['symbol']}] {row['kind']} {row['digest'][:12]} {row['excerpt']!r}", file=stream)

    if report["failures"]:
        print("", file=stream)
        for failure in report["failures"]:
            print(f"FAILED [{failure['code']}] {failure['detail']}", file=stream)
        return
    if report["scope"] == "full":
        print("\nbug-class scan: OK", file=stream)
    else:
        print(
            f"\nbug-class partial scan: OK for {', '.join(report['selected_scanners'])} only — no full-scan claim",
            file=stream,
        )


def scan_repo(
    root: Path,
    *,
    policy_path: Path | None = None,
    baseline_path: Path | None = None,
    dispositions_path: Path | None = None,
    write_baseline: bool = False,
    scanner_ids: list[str] | None = None,
    as_json: bool = False,
    receipt_path: Path | None = None,
) -> int:
    scan = _Scan(root, scanner_ids)
    scan.run(policy_path, baseline_path, dispositions_path, write_baseline)
    code = scan.exit_code()
    scan.report["exit"] = code
    _print_human(scan.report, sys.stderr if as_json else sys.stdout)
    rendered = json.dumps(scan.report, indent=2, ensure_ascii=False)
    if as_json:
        print(rendered)
    if receipt_path is not None:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(rendered + "\n", encoding="utf-8")
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_bug_classes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="inspect a Markdown catalog")
    inspect.add_argument("--catalog", type=Path, required=True)

    scan = subparsers.add_parser("scan", help="evaluate the bug-class enforcement contract")
    scan.add_argument("--root", type=Path, default=Path("."))
    scan.add_argument("--policy", type=Path, default=None)
    scan.add_argument("--baseline", type=Path, default=None)
    scan.add_argument("--dispositions", type=Path, default=None)
    scan.add_argument("--write-baseline", action="store_true")
    scan.add_argument("--scanner", action="append", dest="scanners", default=None)
    scan.add_argument("--json", action="store_true", dest="as_json")
    scan.add_argument("--receipt", type=Path, default=None)

    verify = subparsers.add_parser("verify-receipts", help="check CI/hook evidence against the checkout")
    verify.add_argument("--root", type=Path, default=Path("."))
    verify.add_argument("--scan", type=Path, default=None)
    verify.add_argument("--harness-junit", type=Path, default=None)
    verify.add_argument("--guards-junit", type=Path, default=None)
    verify.add_argument("--expect-sha", default=None)

    migrate = subparsers.add_parser("migrate-baseline", help="propose the reviewed schema-2 migration (writes only --out)")
    migrate.add_argument("--root", type=Path, default=Path("."))
    migrate.add_argument("--legacy", type=Path, required=True)
    migrate.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "inspect":
        return inspect_catalog(args.catalog)
    if args.command == "scan":
        return scan_repo(
            args.root.resolve(),
            policy_path=args.policy,
            baseline_path=args.baseline,
            dispositions_path=args.dispositions,
            write_baseline=args.write_baseline,
            scanner_ids=args.scanners,
            as_json=args.as_json,
            receipt_path=args.receipt,
        )
    if args.command == "verify-receipts":
        if not (args.scan or args.harness_junit or args.guards_junit):
            print("verify-receipts: give at least one of --scan, --harness-junit, --guards-junit", file=sys.stderr)
            return EXIT_ERROR
        problems = receipts_mod.verify(
            args.root.resolve(),
            scan=args.scan,
            harness_junit=args.harness_junit,
            guards_junit=args.guards_junit,
            expect_sha=args.expect_sha,
        )
        for problem in problems:
            print(f"receipt rejected: {problem}", file=sys.stderr)
        if problems:
            return EXIT_FINDINGS
        print("bug-class receipts: OK", file=sys.stderr)
        return EXIT_OK
    if args.command == "migrate-baseline":
        from bug_class_guards import migration

        return migration.run(args.root.resolve(), args.legacy, args.out)
    parser.error(f"unsupported command: {args.command}")
    return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
