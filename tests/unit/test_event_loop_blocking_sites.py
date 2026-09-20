"""Tests for event-loop blocking-call fixes (2026-09-19, BC-69 neighbour).

get_codebase_changes() previously called subprocess.run() directly on the
event loop for git rev-parse/log/diff/status (core/prompt/gatherer_knowledge.py).
That could stall every concurrent prompt-gather task for up to ~30s if git
hung (git status on a big tree normally still costs 100s of ms). The fix
wraps each call in asyncio.to_thread(subprocess.run, ...).

T1 is behavioural (drives the deployed get_codebase_changes exactly the way
tests/unit/test_session_diff.py's happy-path tests do) and proves both that
the four git subprocess calls now run OFF the main thread and that the
returned dict is unchanged.

T2 is a source-reading AST guard scoped to THIS ONE function only. A
source-reading test is acceptable here ONLY because T1 is its behavioural
sibling (BC-63) — together they prove the call is both wrapped AND still
functionally correct; a bare source assertion alone would be vacuous.
"""

import ast
import threading
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest


def _make_gatherer():
    """Create a minimal ContextGatherer for testing (mirrors test_session_diff.py)."""
    from core.prompt.context_gatherer import ContextGatherer

    mc = MagicMock()
    mc.corpus_manager = MagicMock()
    mm = MagicMock()
    tm = MagicMock()
    gs = MagicMock()
    time_mgr = MagicMock()

    return ContextGatherer(
        memory_coordinator=mc,
        model_manager=mm,
        token_manager=tm,
        gate_system=gs,
        time_manager=time_mgr,
    )


def _make_subprocess_result(stdout="", returncode=0):
    result = MagicMock()
    result.stdout = stdout
    result.returncode = returncode
    return result


class TestGetCodebaseChangesOffMainThread:
    """T1: subprocess.run for git rev-parse/log/diff/status runs off the main thread."""

    @pytest.mark.asyncio
    async def test_all_four_git_calls_run_off_main_thread(self):
        gatherer = _make_gatherer()
        thread_flags = []

        def mock_run(cmd, **kwargs):
            thread_flags.append(threading.current_thread() is threading.main_thread())
            if "rev-parse" in cmd:
                return _make_subprocess_result(stdout="/fake/repo\n")
            if "log" in cmd:
                return _make_subprocess_result(
                    stdout="abc1234 feat: Add login\ndef5678 fix: Bug fix\n"
                )
            if cmd[1] == "diff":
                return _make_subprocess_result(
                    stdout="core/orchestrator.py\nmemory/store.py\n"
                )
            if "status" in cmd:
                return _make_subprocess_result(
                    stdout="?? memory/new_module.py\n?? tests/test_new.py\n"
                )
            return _make_subprocess_result()

        with patch("config.app_config.SESSION_DIFF_ENABLED", True), \
             patch("config.app_config.SESSION_DIFF_MAX_COMMITTED", 20), \
             patch("config.app_config.SESSION_DIFF_MAX_UNCOMMITTED", 20), \
             patch("config.app_config.SESSION_DIFF_EXTENSIONS", [".py"]), \
             patch("subprocess.run", side_effect=mock_run):
            result = await gatherer.get_codebase_changes(
                datetime.now() - timedelta(hours=2)
            )

        # All four git subprocess calls (rev-parse, log, diff, status) fired.
        assert len(thread_flags) == 4
        assert not any(thread_flags), (
            "one or more git subprocess.run calls ran ON the event-loop thread"
        )

        # Same result shape/content as before the fix (test_session_diff.py
        # TestGetCodebaseChanges happy-path expectations).
        assert len(result["committed"]) == 2
        assert "abc1234 feat: Add login" in result["committed"]
        assert len(result["uncommitted_modified"]) == 2
        assert "core/orchestrator.py" in result["uncommitted_modified"]
        assert len(result["uncommitted_new"]) == 2
        assert "memory/new_module.py" in result["uncommitted_new"]


class TestGetCodebaseChangesASTGuard:
    """T2: no direct (non-to_thread-wrapped) subprocess.run Call inside the function.

    Source-reading test, deliberately paired with T1 above (BC-63): a
    call like `asyncio.to_thread(subprocess.run, ...)` never appears as an
    `ast.Call` whose func is `subprocess.run` (it appears as a plain
    ast.Attribute reference passed as an argument), so this guard only
    flags a real regression back to a bare `subprocess.run(...)` call.
    """

    def test_no_direct_subprocess_run_call(self):
        path = "core/prompt/gatherer_knowledge.py"
        with open(path) as f:
            tree = ast.parse(f.read(), filename=path)

        target = None
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "get_codebase_changes":
                target = node
                break
        assert target is not None, "get_codebase_changes not found"

        def _is_subprocess_run_call(call: ast.Call) -> bool:
            func = call.func
            return (
                isinstance(func, ast.Attribute)
                and func.attr == "run"
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
            )

        offending = [
            node.lineno
            for node in ast.walk(target)
            if isinstance(node, ast.Call) and _is_subprocess_run_call(node)
        ]

        assert offending == [], (
            f"direct subprocess.run(...) call(s) inside get_codebase_changes "
            f"at line(s) {offending} — must be wrapped in asyncio.to_thread(...)"
        )
