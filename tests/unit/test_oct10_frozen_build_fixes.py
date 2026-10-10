"""P3 frozen-build findings F1/F3/F5/F6 (2026-10-10).

class: BC-71, BC-83, BC-16, BC-70

F1 is pinned as a SOURCE-STRUCTURE test: daemon.spec is a PyInstaller spec
(needs the PyInstaller runtime to execute), so the test parses it with ``ast``
and asserts ``rfc3987_syntax`` sits in the ``collect_data_files`` package loop.
"""
import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _spec_loop_packages() -> list[str]:
    tree = ast.parse((REPO / "daemon.spec").read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.iter, ast.List):
            continue
        # the loop must feed collect_data_files(<loop var>)
        calls = [
            c for c in ast.walk(node)
            if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "collect_data_files"
        ]
        if calls:
            found += [e.value for e in node.iter.elts if isinstance(e, ast.Constant)]
    return found


class TestSpecBundlesRfc3987Data:
    def test_rfc3987_syntax_in_collect_data_files_loop(self):
        pkgs = _spec_loop_packages()
        assert "rfc3987_syntax" in pkgs, pkgs

    def test_loop_still_has_original_packages(self):
        pkgs = _spec_loop_packages()
        assert {"tomlkit", "httpcore", "uvicorn"} <= set(pkgs)


def _run_stage_script(cwd: Path, tmp: Path, *extra: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    script = REPO / "scripts" / "stage_frozen_models.py"
    # -s: no user site (keeps usercustomize.py out); -I is NOT used so the
    # script directory stays on sys.path exactly as in a real invocation.
    return subprocess.run(
        [sys.executable, "-s", str(script), "--out", str(tmp / "stage_out"), "--cache",
         str(tmp / "empty_hf_cache"), *extra],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=120,
    )


class TestStageScriptGuardImport:
    def test_guard_importable_from_repo_root_and_scripts_dir(self, tmp_path):
        # The script's _daemon_running() fails CLOSED with this message when the
        # utils.daemon_guard import fails (scripts/ on sys.path, not the root).
        code = (
            "import runpy, sys\n"
            f"ns = runpy.run_path({str(REPO / 'scripts' / 'stage_frozen_models.py')!r})\n"
            "import utils.daemon_guard as g\n"
            f"assert g.__file__.startswith({str(REPO)!r}), g.__file__\n"
            "print('GUARD_OK')\n"
        )
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for cwd in (REPO, REPO / "scripts"):
            r = subprocess.run([sys.executable, "-s", "-c", code], cwd=str(cwd), env=env,
                               capture_output=True, text=True, timeout=120)
            assert "GUARD_OK" in r.stdout, (cwd, r.stdout, r.stderr)

    def test_daemon_running_does_not_fail_closed_on_import(self, tmp_path):
        # Run the real script path (as BUILD_GUIDE does) in a subprocess and ask
        # _daemon_running() directly: an import failure prints the fail-closed note.
        code = (
            "import sys, runpy\n"
            f"ns = runpy.run_path({str(REPO / 'scripts' / 'stage_frozen_models.py')!r})\n"
            "ns['_daemon_running']()\n"
        )
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        r = subprocess.run([sys.executable, "-s", "-c", code], cwd=str(REPO / "scripts"),
                           env=env, capture_output=True, text=True, timeout=120)
        assert "utils.daemon_guard unavailable" not in r.stderr, r.stderr

    def test_apply_from_scripts_dir_is_not_refused_by_a_failed_guard_import(self, tmp_path):
        # --apply is the path that consults the guard. Empty cache -> staging
        # itself fails afterwards, which is fine: only the refusal matters here.
        (tmp_path / "empty_hf_cache").mkdir()
        for cwd in (REPO, REPO / "scripts"):
            r = _run_stage_script(cwd, tmp_path, "--apply")
            assert "utils.daemon_guard unavailable" not in r.stderr, (cwd, r.stderr)
            assert "Refusing --apply" not in r.stderr, (cwd, r.stderr)


class TestCoreInitDeadImport:
    def test_no_prompt_builder_v2_reference(self):
        src = (REPO / "core" / "__init__.py").read_text(encoding="utf-8")
        assert "prompt_builder_v2" not in src

    def test_core_package_imports(self):
        importlib.import_module("core")
        assert not (REPO / "core" / "prompt_builder_v2.py").exists()


class TestPreflightLlmKeyNames:
    def _check(self, monkeypatch, openrouter=None, openai=None):
        from utils.preflight import PreflightResult, _check_llm_key
        for name, val in (("OPENROUTER_API_KEY", openrouter), ("OPENAI_API_KEY", openai)):
            if val is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, val)
        result = PreflightResult()
        _check_llm_key(result)
        return result

    def test_openrouter_only_does_not_warn(self, monkeypatch):
        key = "sk-" + "or-" + "v1-" + "abc123" + "def456"
        assert not self._check(monkeypatch, openrouter=key).warnings

    def test_openai_only_does_not_warn(self, monkeypatch):
        key = "sk-" + "or-" + "v1-" + "abc123" + "def456"
        assert not self._check(monkeypatch, openai=key).warnings

    def test_neither_warns_naming_openrouter_first(self, monkeypatch):
        w = self._check(monkeypatch).warnings
        assert len(w) == 1
        assert "OPENROUTER_API_KEY" in w[0]
        assert w[0].index("OPENROUTER_API_KEY") < w[0].index("OPENAI_API_KEY")
        assert "wizard" in w[0]

    def test_placeholder_openrouter_warns(self, monkeypatch):
        w = self._check(monkeypatch, openrouter="your_key_here").warnings
        assert len(w) == 1 and "placeholder" in w[0]
