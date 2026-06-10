# tests/agent_branch/test_eval_gate.py
"""Tests for the evaluation gates.

- Layer 1a (static): pure kill-decision logic — runs everywhere.
- Layer 1b (sandboxed): tests execute in a disposable container, never the host
  — marked ``slow``, skipped without podman.
"""

import textwrap

import pytest

from agent_branch.eval_static import (
    GateFlag,
    RunStats,
    matches_forbidden,
    parse_diff,
    run_static_gate,
)
from agent_branch.eval_sandboxed import evaluate_in_sandbox
from agent_branch.manifest import BranchManifest, ManifestTamperError


def _m(**over):
    base = dict(branch_id="g", objective="o", allowed_paths=["sandbox/**"],
                max_diff_lines=50, required_tests=[])
    base.update(over)
    return BranchManifest.issue(**base)


def _diff(path, added=("+x = 1",), removed=()):
    head = f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n"
    return head + "\n".join([*removed, *added]) + "\n"


# -- Layer 1a: static kill decisions ----------------------------------------

def test_empty_diff_killed():
    r = run_static_gate("", _m())
    assert r.killed and GateFlag.EMPTY_DIFF in r.flags


def test_whitespace_only_killed():
    d = ("diff --git a/sandbox/c.py b/sandbox/c.py\n--- a/sandbox/c.py\n+++ b/sandbox/c.py\n"
         "@@ -1 +1 @@\n-    return a+b\n+    return a + b\n")
    r = run_static_gate(d, _m())
    assert r.killed and GateFlag.WHITESPACE_ONLY in r.flags


def test_real_in_scope_change_passes():
    d = ("diff --git a/sandbox/c.py b/sandbox/c.py\n--- a/sandbox/c.py\n+++ b/sandbox/c.py\n"
         '@@ -1 +1,2 @@\n+"""doc"""\n def add(): ...\n')
    r = run_static_gate(d, _m())
    assert not r.killed
    assert r.touched_paths == ["sandbox/c.py"]


def test_touching_tests_is_high_risk_kill():
    r = run_static_gate(_diff("tests/test_x.py"), _m())
    assert r.killed and r.high_risk and GateFlag.SAFETY_TOUCHED in r.flags


def test_conftest_payload_caught_at_any_depth():
    """Red-team #8: a conftest payload anywhere is forbidden (basename match)."""
    r = run_static_gate(_diff("sandbox/conftest.py"), _m())
    assert r.killed and r.high_risk and GateFlag.SAFETY_TOUCHED in r.flags
    assert matches_forbidden("deep/nested/conftest.py", ["conftest.py"])


def test_out_of_scope_path_killed():
    r = run_static_gate(_diff("core/orchestrator.py"), _m())
    assert r.killed and GateFlag.OUT_OF_SCOPE in r.flags


def test_config_default_tamper_killed_high_risk():
    """D2: 'fake success via config default' is a distinct high-risk kill."""
    r = run_static_gate(_diff("config/defaults.py"), _m())
    assert r.killed and r.high_risk and GateFlag.CONFIG_TAMPER in r.flags


def test_benchmark_threshold_tamper_killed():
    r = run_static_gate(_diff("tests/benchmarks/relaxed.yaml"), _m())
    assert r.killed and GateFlag.CONFIG_TAMPER in r.flags


def test_plain_test_edit_is_safety_not_config_tamper():
    # a weakened test is SAFETY_TOUCHED; only config/threshold paths are CONFIG_TAMPER
    r = run_static_gate(_diff("tests/test_x.py"), _m())
    assert GateFlag.SAFETY_TOUCHED in r.flags and GateFlag.CONFIG_TAMPER not in r.flags


def test_diff_too_large_killed():
    big = _diff("sandbox/c.py", added=tuple(f"+line{i}" for i in range(80)))
    r = run_static_gate(big, _m(max_diff_lines=10))
    assert r.killed and GateFlag.DIFF_TOO_LARGE in r.flags


def test_budget_exceeded_killed():
    d = _diff("sandbox/c.py")
    r = run_static_gate(d, _m(token_budget=100),
                        run_stats=RunStats(tokens_spent=999))
    assert r.killed and GateFlag.BUDGET_EXCEEDED in r.flags


def test_wallclock_budget_exceeded_killed():
    d = _diff("sandbox/c.py")
    r = run_static_gate(d, _m(wallclock_seconds=10),
                        run_stats=RunStats(wallclock_elapsed_s=99))
    assert r.killed and GateFlag.BUDGET_EXCEEDED in r.flags


def test_secrets_access_marker_in_audit_log_killed():
    d = _diff("sandbox/c.py")
    r = run_static_gate(d, _m(), audit_log="worker tried to read .env for OPENROUTER_API_KEY")
    assert r.killed and GateFlag.SECRETS_ACCESS in r.flags and r.high_risk


def test_static_gate_rejects_tampered_manifest():
    m = _m()
    bad = m.model_copy(update={"objective": "changed-without-rehash"})
    with pytest.raises(ManifestTamperError):
        run_static_gate(_diff("sandbox/c.py"), bad)


def test_parse_diff_collects_both_rename_sides():
    d = ("diff --git a/sandbox/a.py b/sandbox/b.py\nrename from sandbox/a.py\n"
         "rename to sandbox/b.py\n")
    touched, *_ = parse_diff(d)
    assert "sandbox/a.py" in touched and "sandbox/b.py" in touched


# -- Layer 1b: sandboxed dynamic evaluation (podman) ------------------------

@pytest.mark.slow
def test_sandboxed_eval_passes_for_valid_diff(podman, tiny_repo):
    # a docstring prepend keeps add() correct → trusted check passes
    diff = ("diff --git a/sandbox/calc.py b/sandbox/calc.py\n"
            "--- a/sandbox/calc.py\n+++ b/sandbox/calc.py\n"
            '@@ -1,2 +1,3 @@\n+"""doc"""\n def add(a, b):\n     return a + b\n')
    res = evaluate_in_sandbox(tiny_repo, diff, ["check_calc.py"],
                              test_cmd=["python", "check_calc.py"])
    assert res.applied and res.passed and not res.killed


@pytest.mark.slow
def test_sandboxed_eval_kills_on_failing_test(podman, tiny_repo):
    # break add() → the trusted check fails inside the sandbox → killed
    diff = ("diff --git a/sandbox/calc.py b/sandbox/calc.py\n"
            "--- a/sandbox/calc.py\n+++ b/sandbox/calc.py\n"
            "@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a + b\n+    return a - b\n")
    res = evaluate_in_sandbox(tiny_repo, diff, ["check_calc.py"],
                              test_cmd=["python", "check_calc.py"])
    assert res.applied and not res.passed and res.killed


@pytest.mark.slow
def test_sandboxed_eval_kills_on_unapplyable_diff(podman, tiny_repo):
    bogus = ("diff --git a/sandbox/calc.py b/sandbox/calc.py\n"
             "--- a/sandbox/calc.py\n+++ b/sandbox/calc.py\n"
             "@@ -50,1 +50,1 @@\n-nonexistent context\n+whatever\n")
    res = evaluate_in_sandbox(tiny_repo, bogus, ["check_calc.py"],
                              test_cmd=["python", "check_calc.py"])
    assert not res.applied and res.killed
