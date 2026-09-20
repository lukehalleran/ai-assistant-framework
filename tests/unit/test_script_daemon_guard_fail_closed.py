"""
W2-L2b (2026-09-19): store-writing scripts' ``_daemon_running()`` fails OPEN
when ``utils.daemon_guard`` cannot be imported.

Verified shape (``scripts/purge_profile_facts.py`` was the model): a
try/except around ``from utils.daemon_guard import daemon_running`` either
fell through to a cmdline ``pgrep -af main.py`` heuristic whose own comment
records a known hole (a relative-path launch has no repo name in its
cmdline, 2026-08-21), or -- the simpler variant most of these scripts used --
went straight from ``except Exception:`` to ``return False``. Either way,
when the real guard module is unavailable, ``--apply`` proceeds as though the
Daemon were NOT running: the exact mechanism behind the 2026-08-05 profile
clobber (CLAUDE.md: "prime directive: never wrong > always active").

Fixed: every free (non-accepted-debt) ``scripts/*.py`` defining
``_daemon_running`` now fails CLOSED on any exception from the guard import
or call -- it returns ``True`` (treat the Daemon as running, refuse
``--apply``) and prints a ``[daemon-guard]`` line to stderr instead of
guessing. ``scripts/report_claim_contamination.py`` is deliberately excluded:
it is read-only (no ``--apply`` exists anywhere in it) and its own docstring
documents that a live Daemon changing nothing there makes the check a
WARNING, never a refusal -- a different, intentional design, not this bug.

Safety: this test NEVER imports a whole script module and NEVER calls any
script's ``main()``. Several of these scripts have module-level side effects
(``sys.path.insert``, ``os.environ[...] = ...``, a module logger) that have
nothing to do with the guard. Instead each script's source is parsed with
``ast`` and ONLY the top-level ``_daemon_running`` function definition is
extracted and exec'd, in an isolated namespace that provides just the ``sys``
name its body references (for ``file=sys.stderr``). No script file is ever
imported or executed as a whole; no store, network call, or subprocess is
ever touched.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Accepted-debt scripts.py entries from ../debt_files.txt as of 2026-09-19 --
# none of these define _daemon_running today, kept here only so
# test_all_free_daemon_guard_scripts_covered stays honest if that changes.
DEBT_SCRIPTS = {
    "latency_rollup.py",
    "migrate_proposals_supervision.py",
    "restore_backup.py",
    "test_reflection_self_rewrite.py",
}

# Deliberately NOT fixed: read-only report, no --apply anywhere in it, and
# its own docstring documents the WARNING-only (never refusal) design.
INTENTIONALLY_DIFFERENT = {"report_claim_contamination.py"}

# The free scripts this batch edited to fail CLOSED.
FIXED_SCRIPTS = [
    "add_profile_fact.py",
    "backfill_stance.py",
    "budget_experiment.py",
    "cleanup_stale_illness.py",
    "dedup_reference_docs.py",
    "graph_relation_normalize.py",
    "purge_adaptive_exemplars.py",
    "purge_calibration_facts.py",
    "purge_daemon_self_notes.py",
    "purge_error_memories.py",
    "purge_junk_facts.py",
    "purge_profile_facts.py",
    "quarantine_facts.py",
    "quarantine_graph_edges.py",
    "reclassify_proposals.py",
    "strip_special_token_artifacts.py",
]


def _extract_daemon_running(script_name: str):
    """Parse ``script_name``'s source and exec ONLY its top-level
    ``_daemon_running`` function definition -- never the rest of the module.
    Returns the resulting function object."""
    source = (SCRIPTS_DIR / script_name).read_text()
    tree = ast.parse(source, filename=script_name)
    node = next(
        (n for n in tree.body
         if isinstance(n, ast.FunctionDef) and n.name == "_daemon_running"),
        None,
    )
    assert node is not None, f"{script_name}: no top-level _daemon_running() found"
    func_src = ast.get_source_segment(source, node)
    assert func_src, f"{script_name}: could not recover _daemon_running source"
    namespace: dict = {"sys": sys}
    exec(compile(func_src, f"<{script_name}:_daemon_running>", "exec"), namespace)
    return namespace["_daemon_running"]


class _StubDaemonGuardModule(ModuleType):
    """A minimal stand-in for utils.daemon_guard exposing only daemon_running."""

    def __init__(self, result: bool):
        super().__init__("utils.daemon_guard")
        self.daemon_running = lambda: result


@pytest.mark.parametrize("script_name", FIXED_SCRIPTS)
def test_fails_closed_when_daemon_guard_unimportable(script_name, monkeypatch, capsys):
    fn = _extract_daemon_running(script_name)
    # Forces ImportError/ModuleNotFoundError on `from utils.daemon_guard import ...`
    # without needing the real module to be absent from the environment.
    monkeypatch.setitem(sys.modules, "utils.daemon_guard", None)

    result = fn()

    assert result is True, (
        f"{script_name}: _daemon_running() must fail CLOSED (return True) "
        f"when utils.daemon_guard cannot be imported"
    )
    err = capsys.readouterr().err
    assert "[daemon-guard]" in err, (
        f"{script_name}: must print a [daemon-guard] warning to stderr on fallback"
    )


@pytest.mark.parametrize("script_name", FIXED_SCRIPTS)
def test_defers_to_real_guard_when_available(script_name, monkeypatch, capsys):
    fn = _extract_daemon_running(script_name)

    monkeypatch.setitem(sys.modules, "utils.daemon_guard", _StubDaemonGuardModule(False))
    assert fn() is False, f"{script_name}: must return the guard's own False verdict"
    assert "[daemon-guard]" not in capsys.readouterr().err

    monkeypatch.setitem(sys.modules, "utils.daemon_guard", _StubDaemonGuardModule(True))
    assert fn() is True, f"{script_name}: must return the guard's own True verdict"
    assert "[daemon-guard]" not in capsys.readouterr().err


def test_all_free_daemon_guard_scripts_are_covered():
    """Anti-drift guard on FIXED_SCRIPTS itself: every scripts/*.py that
    defines _daemon_running and is not accepted debt must be in this test's
    parametrization, except the one script that is intentionally different
    (read-only, documented WARNING-only design)."""
    all_scripts = {
        p.name for p in SCRIPTS_DIR.glob("*.py")
        if "def _daemon_running" in p.read_text()
    }
    expected = (all_scripts - DEBT_SCRIPTS) - INTENTIONALLY_DIFFERENT
    assert expected == set(FIXED_SCRIPTS), (
        f"missing={expected - set(FIXED_SCRIPTS)} "
        f"extra={set(FIXED_SCRIPTS) - expected}"
    )
