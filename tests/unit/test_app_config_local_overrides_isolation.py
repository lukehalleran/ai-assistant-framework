"""Tests never load the owner's gitignored config.local.yaml.

Incident 2026-10-22: a README-only `git push` from the live checkout was blocked
by hooks/pre-push -- four unit tests failed because config.app_config deep-merges
config/config.local.yaml at import (daemon.mode dev, decision_model tone_arbiter
shadow, a vault path). class: BC-58 (test-state isolation), BC-83.

The switch is DAEMON_IGNORE_LOCAL_CONFIG=1, set by tests/conftest.py before any
project import and read by config.app_config.load_local_overrides.
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

from config import app_config

REPO = Path(__file__).resolve().parents[2]
HOSTILE = (
    "daemon:\n  mode: dev\n"
    "decision_model:\n  roles:\n    tone_arbiter: shadow\n"
)


def _plant(monkeypatch, tmp_path):
    (tmp_path / "config.local.yaml").write_text(HOSTILE)
    monkeypatch.chdir(tmp_path)
    assert Path("config.local.yaml") in app_config._candidate_config_paths("config.local.yaml")


def test_switch_set_ignores_a_planted_local_file(monkeypatch, tmp_path):
    _plant(monkeypatch, tmp_path)
    assert app_config.load_local_overrides() == {}


def test_fresh_import_does_not_pick_up_planted_values(tmp_path):
    (tmp_path / "config.local.yaml").write_text(HOSTILE)
    import os
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(REPO)
    assert env.get("DAEMON_IGNORE_LOCAL_CONFIG") == "1"
    code = ("import config.app_config as c;"
            "print(c.config.get('daemon',{}).get('mode'),"
            "c.config.get('decision_model',{}).get('roles',{}).get('tone_arbiter'))")
    r = subprocess.run([sys.executable, "-s", "-c", code], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    mode, arbiter = r.stdout.strip().splitlines()[-1].split()
    assert arbiter != "shadow"


def test_switch_removed_loads_the_same_planted_file(monkeypatch, tmp_path):
    _plant(monkeypatch, tmp_path)
    monkeypatch.delenv("DAEMON_IGNORE_LOCAL_CONFIG")
    loaded = app_config.load_local_overrides()
    assert loaded["daemon"]["mode"] == "dev"
    assert loaded["decision_model"]["roles"]["tone_arbiter"] == "shadow"


def test_switch_other_value_does_not_suppress(monkeypatch, tmp_path):
    _plant(monkeypatch, tmp_path)
    monkeypatch.setenv("DAEMON_IGNORE_LOCAL_CONFIG", "0")
    assert app_config.load_local_overrides()["daemon"]["mode"] == "dev"


def test_conftest_sets_switch_before_any_project_import():
    tree = ast.parse((REPO / "tests" / "conftest.py").read_text())
    set_line = None
    first_project_import = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "DAEMON_IGNORE_LOCAL_CONFIG":
            set_line = node.lineno if set_line is None else min(set_line, node.lineno)
    # module-level (not nested) imports and statements only
    for node in tree.body:
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module]
        for m in mods:
            top = m.split(".")[0]
            if top not in sys.stdlib_module_names and first_project_import is None:
                first_project_import = node.lineno
    assert set_line is not None, "conftest must set DAEMON_IGNORE_LOCAL_CONFIG"
    assert first_project_import is None or set_line < first_project_import
    # also: nothing project-level is imported earlier inside try blocks at module level
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node.lineno < set_line:
            name = node.names[0].name if isinstance(node, ast.Import) else (node.module or "")
            assert name.split(".")[0] in sys.stdlib_module_names, (name, node.lineno)
