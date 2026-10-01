"""
2026-09-27 lane B batch B8 (class: BC-69, BC-71).

FOLLOWUPS.md #180: "`print(` outside documented CLI paths — 260 app sites
(gui/launch.py 69 banners, eval/run_phase* 69, utils/bootstrap.py 24,
utils/startup.py 21). Closure: ruff T201 scoped to app dirs minus the two
documented exceptions (S)."

Mechanism (BC-69 silent ops failures / BC-71 doc-tool self-description drift):
a print() in app code bypasses logging levels/handlers, so an error printed
instead of logged is invisible to anything that watches logs (and vanishes
entirely in a frozen build with no attached console). `pyproject.toml` now
selects ruff's T201 rule, scoped away from dev tooling (scripts/tests/docs/
hooks/agent_branch) and away from documented CLI/banner paths (main.py,
gui/launch.py, utils/bootstrap.py, utils/startup.py, the eval harness), with
the remaining pre-existing app-code print sites grandfathered per-file so the
rule only fails on NEW prints.

Per CLAUDE.md's validation doctrine ("validation must call the deployed
function"), these tests do not parse pyproject.toml and reason about what it
SHOULD do — they invoke the real `ruff` binary against the actual shipped
config (read from disk at test time) and check its actual exit code /
findings, the same way `hooks/pre-push` and `.github/workflows/tests.yml`
invoke it (`python -m ruff check .`).

Test 2 is the one that proves the fix: run against the CURRENT repo's
pyproject.toml, a brand-new print() in an in-scope, non-ignored app file (a
synthetic ``core/probe_module.py``) IS caught, while the same print() in a
documented-exception path (``gui/launch.py``) and a grandfathered-debt path
(``utils/preflight.py``) is NOT. Run against `laneB_0927_base`'s (pre-fix)
pyproject.toml, the same probe file is NOT caught at all (T201 was never
selected) — verified manually against `~/daemon_exec/laneB_0927_base` before
writing this test: `ruff check .` on an isolated tmp project using that
file's pyproject.toml reports "All checks passed!" for the identical
core/probe_module.py print(), where the post-fix config reports
"T201 `print` found ... core/probe_module.py:2:5". This test's assertion
(a T201 finding for the unexempted probe file) therefore fails against the
base clone's config and passes against this batch's config.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
# Same convention as every other invocation in this batch's clone (see
# BRIEF_LANEB.md): drop PYTHONPATH so a login shell's usercustomize.py can't
# shadow-import a different clone's modules into the ruff subprocess.
_SUBPROCESS_ENV = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def _ruff_importable() -> bool:
    """ruff ships as a Python package with a native extension — import-check
    with the SAME interpreter the test itself is running under (sys.executable),
    not a `ruff` on PATH, which can resolve to a different interpreter/version
    depending on shell/pyenv state."""
    probe = subprocess.run(
        [sys.executable, "-c", "import ruff"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(
    not _ruff_importable(), reason="ruff not importable under this interpreter"
)


def _run_ruff(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-s", "-m", "ruff", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=120,
        env=_SUBPROCESS_ENV,
    )


class TestT201ConfigShape:
    """Cheap sanity checks on the shipped config (not a substitute for the
    subprocess tests below, which validate the deployed behavior)."""

    def test_t201_selected(self):
        config = tomllib.loads(PYPROJECT.read_text())
        select = config["tool"]["ruff"]["lint"]["select"]
        assert "T201" in select

    def test_documented_exceptions_present(self):
        config = tomllib.loads(PYPROJECT.read_text())
        ignores = config["tool"]["ruff"]["lint"]["per-file-ignores"]
        for documented in (
            "main.py",
            "gui/launch.py",
            "utils/bootstrap.py",
            "utils/startup.py",
        ):
            assert documented in ignores, documented
            assert ignores[documented] == ["T201"]

    def test_dev_tooling_dirs_exempted(self):
        config = tomllib.loads(PYPROJECT.read_text())
        ignores = config["tool"]["ruff"]["lint"]["per-file-ignores"]
        for dev_glob in ("scripts/**", "tests/**", "docs/**", "hooks/**", "agent_branch/**"):
            assert dev_glob in ignores, dev_glob


class TestT201RepoCheckStaysGreen:
    def test_repo_wide_ruff_check_is_green(self):
        """The repo's normal invocation (hooks/pre-push, .github/workflows/tests.yml:
        `python -m ruff check .`) must stay green with T201 enabled — the whole point
        of grandfathering existing sites via per-file-ignores."""
        result = _run_ruff(["check", "."], cwd=REPO_ROOT)
        assert result.returncode == 0, result.stdout + result.stderr


class TestT201CatchesNewPrints:
    """Builds an isolated tmp project (pytest's tmp_path — auto-cleaned) that
    copies the REAL shipped pyproject.toml so ruff resolves per-file-ignores
    the same way it would in the real tree, then drops synthetic files at
    paths that are (a) unexempted app code, (b) a documented CLI/banner
    exception, and (c) grandfathered debt — proving the rule's scope, not just
    its presence in the TOML."""

    @pytest.fixture(autouse=True)
    def _isolated_project(self, tmp_path):
        for d in ("core", "gui", "utils"):
            (tmp_path / d).mkdir()
        shutil.copyfile(PYPROJECT, tmp_path / "pyproject.toml")
        (tmp_path / "core" / "probe_module.py").write_text(
            'def f():\n    print("new debug line")\n'
        )
        (tmp_path / "gui" / "launch.py").write_text(
            'def g():\n    print("startup banner")\n'
        )
        (tmp_path / "utils" / "preflight.py").write_text(
            'def h():\n    print("grandfathered pre-existing debt")\n'
        )
        self.tmp_path = tmp_path

    def test_new_print_in_unexempted_app_file_is_flagged(self):
        result = _run_ruff(["check", "."], cwd=self.tmp_path)
        assert result.returncode != 0
        assert "core/probe_module.py" in result.stdout

    def test_documented_banner_path_is_exempt(self):
        result = _run_ruff(["check", "."], cwd=self.tmp_path)
        assert "gui/launch.py" not in result.stdout

    def test_grandfathered_debt_path_is_exempt(self):
        result = _run_ruff(["check", "."], cwd=self.tmp_path)
        assert "utils/preflight.py" not in result.stdout
