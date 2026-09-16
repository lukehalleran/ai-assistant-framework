"""
Repo-wide guard for the pyright type-checking lane
(docs/PLAN_20260916_typecheck_lane.md §2). ``config/typecheck_baseline.json``
is a per-(file, rule) diagnostic ceiling plus a suppression-comment ceiling;
``scripts/check_typecheck.py --update-baseline`` may only LOWER a row, never
raise one silently. This test never runs pyright and never reads git state
(``tests/unit/test_no_git_state_in_tests.py`` bans that) — it only reads the
checked-in baseline file and rescans the included tree's suppression
comments with the SAME regexes the gate script uses (imported from the
script itself, so there is one definition, not two that can drift apart).

Mode: CEILING RATCHET, mirroring ``tests/unit/test_import_hygiene_guard.py``.
``MAX_DIAGNOSTICS``/``MAX_IGNORES`` are the totals at the time the baseline
was last lowered; the first test in each pair fails when the tree exceeds
them, the second fails when the ceiling is left slack after a fix batch.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / "config" / "typecheck_baseline.json"
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_typecheck.py"

# Lowered by every typecheck fix batch; raised by nobody. See module docstring.
MAX_DIAGNOSTICS = 740  # 2026-09-16 initial baseline (docs/PLAN_20260916_typecheck_lane.md §2.2)
MAX_IGNORES = 9  # 0 valid + 9 legacy (F3: pre-existing bare `# type: ignore` sites)
CEILING_SLACK = 5

RULE_NAME_RE = re.compile(r"^report[A-Z]\w+$|^<none>$")
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _load_script():
    """Import scripts/check_typecheck.py by path — one definition of the
    suppression regexes and the included-file scan, never a copy that can
    drift from the gate script."""
    spec = importlib.util.spec_from_file_location("_typecheck_guard_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_baseline() -> dict:
    assert BASELINE_PATH.is_file(), f"missing {BASELINE_PATH.name} — run scripts/check_typecheck.py --update-baseline"
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _measured_suppressions():
    script = _load_script()
    config = script.pyright_config(REPO_ROOT)
    files = script.included_python_files(REPO_ROOT, config)
    return script.scan_suppressions(REPO_ROOT, files)


def test_baseline_schema_is_well_formed():
    baseline = _load_baseline()
    assert baseline["schema"] == 1
    assert baseline["tool"] == "pyright"
    assert VERSION_RE.match(baseline["version"]), baseline["version"]
    assert baseline["mode"] == "standard"
    for file, rules in baseline["files"].items():
        for rule, count in rules.items():
            assert RULE_NAME_RE.match(rule), f"{file}: rule name {rule!r} does not look like a pyright rule"
            assert isinstance(count, int) and count > 0, f"{file}::{rule} count must be a positive int, got {count!r}"


def test_diagnostic_total_does_not_grow():
    baseline = _load_baseline()
    total = sum(sum(rules.values()) for rules in baseline["files"].values())
    assert total <= MAX_DIAGNOSTICS, (
        f"baseline now carries {total} error diagnostics, ceiling is {MAX_DIAGNOSTICS}. "
        "A raised ceiling means scripts/check_typecheck.py --update-baseline --allow-raise "
        "was run without owner review — lower MAX_DIAGNOSTICS to match a reviewed raise, or "
        "fix the new diagnostics instead."
    )


def test_diagnostic_ceiling_is_lowered_after_each_batch():
    baseline = _load_baseline()
    total = sum(sum(rules.values()) for rules in baseline["files"].values())
    slack = MAX_DIAGNOSTICS - total
    assert slack <= CEILING_SLACK, (
        f"MAX_DIAGNOSTICS ({MAX_DIAGNOSTICS}) is {slack} above the baseline total ({total}); "
        f"lower it to {total} in this batch so the ratchet only ever tightens."
    )


def test_suppression_total_does_not_grow():
    baseline = _load_baseline()
    valid, legacy = _measured_suppressions()
    measured_total = sum(valid.values()) + sum(legacy.values())
    assert measured_total <= MAX_IGNORES, (
        f"{measured_total} suppression comments in the included tree, ceiling is {MAX_IGNORES}. "
        "A new bare `# type: ignore` (or `# pyright: ignore` without a rule+why:) fails this "
        "even though pyright itself was never run here."
    )
    for file, count in legacy.items():
        ceiling = baseline.get("legacy_ignores", {}).get(file, 0)
        assert count <= ceiling, (
            f"{file}: {count} legacy (non-conforming) suppressions, baseline ceiling is {ceiling}. "
            "Use the conforming form: `# pyright: ignore[reportXxx]  # why: <reason>`."
        )
    for file, count in valid.items():
        ceiling = baseline.get("ignores", {}).get(file, 0)
        assert count <= ceiling, f"{file}: {count} suppressions, baseline ceiling is {ceiling}."


def test_suppression_ceiling_is_lowered_after_each_batch():
    valid, legacy = _measured_suppressions()
    measured_total = sum(valid.values()) + sum(legacy.values())
    slack = MAX_IGNORES - measured_total
    assert slack <= CEILING_SLACK, (
        f"MAX_IGNORES ({MAX_IGNORES}) is {slack} above the measured count ({measured_total}); "
        f"lower it to {measured_total} in this batch so the ratchet only ever tightens."
    )
