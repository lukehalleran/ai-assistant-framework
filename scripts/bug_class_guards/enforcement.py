"""Literal-command checks for the enforcement wiring (hook and workflow).

The scan contract means nothing if the places that enforce it run less than
all of it.  These validators read ``hooks/pre-push`` and
``.github/workflows/tests.yml`` as TEXT and require the exact reviewed
commands: the full policy scan with a receipt (no ``--scanner``,
``--write-baseline`` or ledger/policy override), the isolated harness with
its JUnit receipt, receipt verification, and the five repo-wide guards.  They
reject failure masking (``|| true``, ``set +e``, ``continue-on-error``), path
filters, and — for the workflow — a final ``bug-class-gate`` status that could
be skipped, that does not require every other job's result to be
``success``, or that does not verify the receipts for the tested commit.

Deliberately not a shell or YAML engine: an exact reviewed command string is
the contract, so an unrecognized rewrite is a problem to review, not a
spelling to accept.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_TOOL = "scripts/check_bug_classes.py"
FORBIDDEN_FLAGS = ("--scanner", "--write-baseline", "--baseline", "--dispositions", "--policy")
MASKS = ("|| true", "|| :", "set +e")
GATE = "bug-class-gate"

HOOK_SCAN = 'python scripts/check_bug_classes.py scan --root . --receipt "$receipts/bug-class-scan.json" || fail "bug-class scan"'
HOOK_HARNESS = (
    'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider '
    '--confcutdir=tests/bug_class_guards --junitxml="$receipts/bug-class-harness.xml" '
    'tests/bug_class_guards || fail "bug-class guard tests"'
)
HOOK_VERIFY = (
    'python scripts/check_bug_classes.py verify-receipts --root . '
    '--scan "$receipts/bug-class-scan.json" --harness-junit "$receipts/bug-class-harness.xml" '
    '|| fail "bug-class receipts"'
)
HOOK_GUARD_VERIFY = (
    'python scripts/check_bug_classes.py verify-receipts --root . '
    '--guards-junit "$receipts/repo-wide-guards.xml" || fail "repo-wide guard receipts"'
)

CI_SCAN = "python scripts/check_bug_classes.py scan --root . --receipt receipts/bug-class-scan.json"
CI_HARNESS = (
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider "
    "--confcutdir=tests/bug_class_guards --junitxml=receipts/bug-class-harness.xml "
    "tests/bug_class_guards"
)
CI_VERIFY = (
    "python scripts/check_bug_classes.py verify-receipts --root . "
    "--scan receipts/bug-class-scan.json --harness-junit receipts/bug-class-harness.xml "
    '--guards-junit receipts/repo-wide-guards.xml --expect-sha "$GITHUB_SHA"'
)


def repo_wide_guards() -> list[str]:
    policy = json.loads((REPO_ROOT / "config" / "bug_class_policy.json").read_text(encoding="utf-8"))
    return list(policy["repo_wide_guards"])


def logical_lines(text: str) -> list[str]:
    """Stripped, comment-free lines with backslash continuations joined."""
    lines: list[str] = []
    pending = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not pending and stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            pending += stripped[:-1].strip() + " "
            continue
        lines.append((pending + stripped).strip())
        pending = ""
    if pending:
        lines.append(pending.strip())
    return [line for line in lines if line]


def _shared_problems(lines: list[str], where: str) -> list[str]:
    problems = []
    for line in lines:
        for mask in MASKS:
            if mask in line:
                problems.append(f"{where}: failure masked by {mask!r}: {line}")
        if SCAN_TOOL in line:
            for flag in FORBIDDEN_FLAGS:
                if re.search(rf"(^|\s){re.escape(flag)}(\s|=|$)", line):
                    problems.append(f"{where}: enforcement command uses {flag}: {line}")
    return problems


def _exactly_once(lines: list[str], command: str, where: str, label: str) -> list[str]:
    count = lines.count(command)
    if count != 1:
        return [f"{where}: {label} must appear exactly once as {command!r} (found {count})"]
    return []


def hook_problems(text: str, guards: list[str] | None = None) -> list[str]:
    guards = repo_wide_guards() if guards is None else guards
    lines = logical_lines(text)
    problems = _shared_problems(lines, "hook")
    if "set -euo pipefail" not in lines:
        problems.append("hook: set -euo pipefail is required")
    problems += _exactly_once(lines, HOOK_SCAN, "hook", "the full policy scan")
    scans = [line for line in lines if f"{SCAN_TOOL} scan" in line]
    if len(scans) > 1:
        problems.append(f"hook: {len(scans)} scan invocations; only the full scan is allowed")
    problems += _exactly_once(lines, HOOK_HARNESS, "hook", "the isolated harness")
    problems += _exactly_once(lines, HOOK_VERIFY, "hook", "scan/harness receipt verification")
    problems += _exactly_once(lines, HOOK_GUARD_VERIFY, "hook", "repo-wide guard receipt verification")

    match = re.search(r"^guards=\((?P<body>.*?)^\)", text, re.MULTILINE | re.DOTALL)
    listed = match.group("body").split() if match else []
    if listed != guards:
        problems.append(f"hook: guards array {listed!r} must be exactly {guards!r}")
    if '[ -f "$g" ] || fail' not in text or '[ -f "$g" ] &&' in text:
        problems.append("hook: a missing repo-wide guard file must fail the push, never be skipped")
    guard_run = [
        line for line in lines
        if "python -m pytest" in line and '--junitxml="$receipts/repo-wide-guards.xml"' in line
    ]
    if len(guard_run) != 1 or '"${selection[@]}"' not in guard_run[0] or not guard_run[0].endswith('|| fail "tests"'):
        problems.append("hook: the guard selection must run once with its JUnit receipt and fail the push")
    return problems


def _jobs(text: str) -> dict[str, dict]:
    jobs: dict[str, dict] = {}
    in_jobs = False
    current = None
    for raw in text.splitlines():
        if raw.strip().startswith("#"):
            continue
        if re.match(r"^jobs:\s*$", raw):
            in_jobs = True
            continue
        if in_jobs and re.match(r"^\S", raw):
            in_jobs = False
        if not in_jobs:
            continue
        header = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", raw)
        if header:
            current = jobs.setdefault(header.group(1), {"lines": [], "name": None, "needs": None, "if": None})
            continue
        if current is None:
            continue
        current["lines"].append(raw)
        attr = re.match(r"^    (name|needs|if):\s*(.*?)\s*$", raw)
        if attr:
            key, value = attr.groups()
            if key == "needs":
                inline = re.match(r"^\[(.*)\]$", value)
                current["needs"] = [item.strip() for item in inline.group(1).split(",") if item.strip()] if inline else []
            else:
                current[key] = value
        elif current["needs"] == [] and re.match(r"^      - ([A-Za-z0-9_-]+)\s*$", raw):
            current["needs"].append(raw.strip()[2:].strip())
    return jobs


def workflow_problems(text: str, guards: list[str] | None = None) -> list[str]:
    guards = repo_wide_guards() if guards is None else guards
    raw_code = [line for line in text.splitlines() if not line.strip().startswith("#")]
    lines = logical_lines(text)
    problems = _shared_problems(lines, "workflow")
    for line in raw_code:
        if re.match(r"^\s*paths(-ignore)?\s*:", line):
            problems.append(f"workflow: path filter {line.strip()!r} — the lane must run on every change")
        if "continue-on-error" in line:
            problems.append(f"workflow: {line.strip()!r} masks a failing step")
    if "  push:" not in raw_code or "    branches: [master, refactor/prompt-modular]" not in raw_code:
        problems.append("workflow: push to master and refactor/prompt-modular must trigger the run")
    if "  pull_request:" not in raw_code or "    branches: [master]" not in raw_code:
        problems.append("workflow: pull requests into master must trigger the run")

    problems += _exactly_once(lines, CI_SCAN, "workflow", "the full policy scan")
    scans = [line for line in lines if f"{SCAN_TOOL} scan" in line]
    if len(scans) > 1:
        problems.append(f"workflow: {len(scans)} scan invocations; only the full scan is allowed")
    problems += _exactly_once(lines, CI_HARNESS, "workflow", "the isolated harness")
    guard_command = (
        "python -m pytest -q -p no:cacheprovider --junitxml=receipts/repo-wide-guards.xml "
        + " ".join(guards)
    )
    problems += _exactly_once(lines, guard_command, "workflow", "the five repo-wide guards")
    problems += _exactly_once(lines, CI_VERIFY, "workflow", "receipt verification for the tested commit")

    jobs = _jobs(text)
    gates = [job_id for job_id, job in jobs.items() if job_id == GATE or job["name"] == GATE]
    if gates != [GATE] or jobs.get(GATE, {}).get("name") != GATE:
        problems.append(f"workflow: exactly one job {GATE!r} named {GATE!r} must publish the final status (found {gates})")
        return problems
    gate = jobs[GATE]
    others = sorted(job_id for job_id in jobs if job_id != GATE)
    if sorted(gate["needs"] or []) != others:
        problems.append(f"workflow: {GATE} must need every other job {others} (needs {gate['needs']})")
    if gate["if"] != "${{ always() }}":
        problems.append(f"workflow: {GATE} must run with if: ${{{{ always() }}}} so a failed or skipped dependency cannot skip it into green")
    always = sum(line.count("always()") for line in raw_code)
    if always != 1:
        problems.append(f"workflow: always() appears {always} times; only {GATE}'s job condition may use it")
    gate_text = "\n".join(gate["lines"])
    for job_id in others:
        if f"${{{{ needs.{job_id}.result }}}}" not in gate_text:
            problems.append(f"workflow: {GATE} never checks needs.{job_id}.result")
    if '!= "success"' not in gate_text and '= "success"' not in gate_text:
        problems.append(f"workflow: {GATE} must require each dependency result to equal success")
    if CI_VERIFY not in logical_lines(gate_text):
        problems.append(f"workflow: {GATE} must run receipt verification itself")
    return problems
