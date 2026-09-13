"""Verify the evidence a green bug-class check must carry.

GitHub lets successful, skipped, or neutral checks satisfy a required status,
and a job's colour says nothing about what it actually examined.  The final
gate therefore re-reads the CHECKED-OUT policy, baseline and dispositions and
accepts a run only when its receipts agree with them:

* the scan receipt (``scan --receipt``) is a full-scope, exit-0, failure-free
  report for the expected commit, produced under this exact policy file, with
  the pinned scanner IDs/modes/classes/versions, an available receipt for
  every required leg and root, and baseline/disposition counts equal to the
  committed files;
* the isolated harness JUnit report has exactly the policy's expected test
  case count, all from ``tests/bug_class_guards``, none failed, errored or
  skipped (an empty or shrunken collection is a failure);
* the repo-wide guard JUnit report contains at least one passing case from
  each of the five guard modules.

This is a consistency check of machine-readable evidence, not a signed
receipt framework: whoever can change the guard code can change this code,
and both remain ordinary reviewed changes.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from . import baseline as baseline_mod
from . import dispositions as dispositions_mod
from .policy import POLICY_PATH, Policy, PolicyError, load_policy


def verify(
    root: Path,
    *,
    scan: Path | None = None,
    harness_junit: Path | None = None,
    guards_junit: Path | None = None,
    expect_sha: str | None = None,
) -> list[str]:
    try:
        policy = load_policy(root / POLICY_PATH)
    except PolicyError as exc:
        return [f"policy: {exc}"]
    problems: list[str] = []
    if scan is not None:
        problems += _verify_scan(root, policy, scan, expect_sha)
    elif expect_sha:
        problems.append("commit: --expect-sha needs a --scan receipt to compare against")
    if harness_junit is not None:
        problems += _verify_harness(policy, harness_junit)
    if guards_junit is not None:
        problems += _verify_guards(policy, guards_junit)
    return problems


# ------------------------------------------------------------- scan receipt


def _verify_scan(root: Path, policy: Policy, path: Path, expect_sha: str | None) -> list[str]:
    if not path.is_file():
        return [f"scan receipt missing: {path}"]
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"scan receipt unreadable: {path}: {exc}"]
    if not isinstance(report, dict) or report.get("report_schema") != 1 or report.get("command") != "scan":
        return [f"scan receipt {path} is not a schema-1 scan report"]
    problems: list[str] = []
    if expect_sha is not None and report.get("source_sha") != expect_sha:
        problems.append(f"commit: receipt is for {report.get('source_sha')!r}, expected {expect_sha}")
    if report.get("exit") != 0:
        problems.append(f"scan exit {report.get('exit')!r}, expected 0")
    if report.get("scope") != "full":
        problems.append(f"scan scope {report.get('scope')!r}, expected full")
    failures = report.get("failures")
    if failures != []:
        codes = sorted({f.get("code") for f in failures or [] if isinstance(f, dict)})
        problems.append(f"scan recorded failure(s): {', '.join(map(str, codes)) or failures!r}")
    receipt_policy = report.get("policy") or {}
    if receipt_policy.get("sha256") != policy.sha256:
        problems.append(
            f"policy: receipt produced under sha256 {receipt_policy.get('sha256')!r}, "
            f"checkout policy is {policy.sha256}"
        )

    observed = {
        entry.get("id"): entry
        for entry in (report.get("registry") or {}).get("observed") or []
        if isinstance(entry, dict)
    }
    expected_ids = set(policy.scanner_ids)
    for extra in sorted(set(observed) - expected_ids, key=str):
        problems.append(f"registry: receipt lists unreviewed scanner {extra}")
    scanners = report.get("scanners") or {}
    for contract in policy.scanners:
        entry = observed.get(contract.id)
        if entry is None:
            problems.append(f"registry: scanner {contract.id} missing from the receipt")
            continue
        for field_name, want in contract.to_json().items():
            if entry.get(field_name) != want:
                problems.append(
                    f"{contract.id}: receipt {field_name} {entry.get(field_name)!r}, policy {want!r}"
                )
        problems += _verify_legs(policy, contract, scanners.get(contract.id))

    gate_kinds = {c.id: c.kinds for c in policy.scanners if c.mode == "gate"}
    try:
        stored = baseline_mod.load(root / policy.baseline_path, gate_kinds, policy.scanner_ids)
        occurrences = stored.occurrences
    except baseline_mod.BaselineError as exc:
        problems.append(f"baseline: {exc}")
        occurrences = None
    receipt_baseline = (report.get("baseline") or {}).get("occurrences")
    if occurrences is not None and receipt_baseline != occurrences:
        problems.append(f"baseline: receipt counts {receipt_baseline!r} occurrences, checkout has {occurrences}")
    try:
        ledger = dispositions_mod.load(root / policy.dispositions_path, gate_kinds, policy.scanner_ids)
        active = ledger.count("accepted_debt")
    except dispositions_mod.DispositionsError as exc:
        problems.append(f"dispositions: {exc}")
        active = None
    receipt_active = (report.get("dispositions") or {}).get("active")
    if active is not None:
        if receipt_active != active:
            problems.append(f"dispositions: receipt counts {receipt_active!r} active records, checkout has {active}")
        if occurrences is not None and active != occurrences:
            problems.append(f"dispositions: {active} accepted_debt records for {occurrences} baseline occurrences")
    return problems


def _verify_legs(policy: Policy, contract, entry) -> list[str]:
    if not isinstance(entry, dict):
        return [f"{contract.id}: scanner receipt missing"]
    problems: list[str] = []
    status = entry.get("status")
    if status != "ok" and not (contract.mode == "report" and status == "unavailable"):
        problems.append(f"{contract.id}: status {status!r}")
    legs = {leg.get("id"): leg for leg in entry.get("legs") or [] if isinstance(leg, dict)}
    for leg_id in contract.legs:
        declared = policy.legs[leg_id]
        receipt = legs.get(leg_id)
        if receipt is None:
            problems.append(f"{contract.id} leg {leg_id}: receipt missing")
            continue
        if receipt.get("required") is not declared.required or type(receipt.get("available")) is not bool:
            problems.append(f"{contract.id} leg {leg_id}: required/available flags malformed")
            continue
        roots = [r.get("path") for r in receipt.get("roots") or [] if isinstance(r, dict)]
        if roots != list(declared.roots):
            problems.append(f"{contract.id} leg {leg_id}: roots {roots!r}, policy {list(declared.roots)!r}")
        if declared.required:
            files = receipt.get("files_processed")
            root_ok = all(r.get("status") == "available" for r in receipt.get("roots") or [])
            if not (receipt["available"] and receipt.get("status") == "available" and root_ok
                    and type(files) is int and files > 0):
                problems.append(
                    f"{contract.id} leg {leg_id}: required input not fully available "
                    f"(status {receipt.get('status')!r}, files {files!r})"
                )
        elif receipt.get("status") not in {"available", "unavailable"}:
            problems.append(f"{contract.id} leg {leg_id}: optional leg status {receipt.get('status')!r}")
    return problems


# ------------------------------------------------------------------ JUnit


def _cases(path: Path, label: str):
    if not path.is_file():
        return None, [f"{label} junit missing: {path}"]
    try:
        tree = ET.parse(path)
    except (ET.ParseError, OSError) as exc:
        return None, [f"{label} junit unreadable: {path}: {exc}"]
    cases = []
    for case in tree.getroot().iter("testcase"):
        tags = {child.tag for child in case}
        cases.append((case.get("classname") or "", case.get("name") or "", tags))
    return cases, []


def _outcome_problems(cases, label: str) -> list[str]:
    problems = []
    for tag, word in (("failure", "failed"), ("error", "errored"), ("skipped", "skipped")):
        bad = [f"{classname}::{name}" for classname, name, tags in cases if tag in tags]
        if bad:
            problems.append(f"{label}: {len(bad)} case(s) {word}: {', '.join(bad[:5])}")
    return problems


def _verify_harness(policy: Policy, path: Path) -> list[str]:
    cases, problems = _cases(path, "harness")
    if cases is None:
        return problems
    expected = policy.expected_test_cases
    if len(cases) != expected:
        problems.append(
            f"harness: {len(cases)} test cases, expected {expected} "
            "(policy harness.expected_test_cases; update both together)"
        )
    prefix = policy.harness_path.replace("/", ".")
    outside = [c for c, _, _ in cases if not (c == prefix or c.startswith(prefix + "."))]
    if outside:
        problems.append(f"harness: {len(outside)} case(s) outside {policy.harness_path}")
    return problems + _outcome_problems(cases, "harness")


def _verify_guards(policy: Policy, path: Path) -> list[str]:
    cases, problems = _cases(path, "guards")
    if cases is None:
        return problems
    for guard in policy.repo_wide_guards:
        module = guard[: -len(".py")].replace("/", ".")
        own = [case for case in cases if case[0] == module or case[0].startswith(module + ".")]
        if not own:
            problems.append(f"guards: no test case from {module}")
            continue
        problems += _outcome_problems(own, f"guards {module}")
    return problems
