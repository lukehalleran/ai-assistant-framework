"""
utils/daemon_guard.py — live-Daemon detection for store-writing scripts.

Regression (2026-08-21): the per-script guards grepped the pgrep command line
for "Daemon_v1", which a relative-path launch defeats (`python main.py` from
inside the repo has no repo name in its cmdline). The guard silently passed
and `purge_adaptive_exemplars.py --apply` ran against a live store. The
shared guard resolves /proc/<pid>/cwd against the repo root instead — launch
style can't hide the working directory.
"""

import subprocess
import sys
import time
from pathlib import Path

from utils.daemon_guard import _looks_like_daemon, daemon_running


class TestLooksLikeDaemon:
    def test_python_running_main_py(self):
        assert _looks_like_daemon(["/usr/bin/python3", "main.py"]) is True

    def test_python_absolute_path_relative_launch(self):
        # THE regression shape: pyenv python, bare relative "main.py" —
        # no repo name anywhere in argv.
        assert _looks_like_daemon(
            ["/home/lukeh/.pyenv/versions/3.11.8/bin/python", "main.py"]
        ) is True

    def test_frozen_executable(self):
        assert _looks_like_daemon(["./main.py"]) is True

    def test_editor_on_main_py_excluded(self):
        assert _looks_like_daemon(["vim", "main.py"]) is False

    def test_python_other_script_excluded(self):
        assert _looks_like_daemon(
            ["/usr/bin/python3", "scripts/purge_error_memories.py"]
        ) is False

    def test_unrelated_process_with_main_py_substring(self):
        # e.g. ibus-typing-booster's engine/main.py — different cwd anyway,
        # but the argv shape alone must not be enough without python+main.py
        assert _looks_like_daemon(
            ["/usr/bin/python3", "/usr/share/ibus-typing-booster/engine/main.py"]
        ) is True  # argv-shape matches; the CWD check is what excludes it


class TestDaemonRunning:
    def test_no_daemon_in_empty_root(self, tmp_path):
        # nothing runs with this tmp dir as cwd
        assert daemon_running(repo_root=tmp_path) is False

    def test_detects_relative_path_launch(self, tmp_path):
        # Reproduce the exact miss: `python main.py` launched with the repo
        # as cwd and NO repo name in argv.
        fake_main = tmp_path / "main.py"
        fake_main.write_text("import time\ntime.sleep(30)\n")
        proc = subprocess.Popen(
            [sys.executable, "main.py"], cwd=str(tmp_path),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(0.3)  # let it start
            assert daemon_running(repo_root=tmp_path) is True
        finally:
            proc.kill()
            proc.wait()

    def test_other_cwd_not_detected(self, tmp_path):
        # a main.py running elsewhere must not trip a guard scoped to tmp_path
        other = tmp_path / "other"
        other.mkdir()
        watched = tmp_path / "watched"
        watched.mkdir()
        fake_main = other / "main.py"
        fake_main.write_text("import time\ntime.sleep(30)\n")
        proc = subprocess.Popen(
            [sys.executable, "main.py"], cwd=str(other),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(0.3)
            assert daemon_running(repo_root=watched) is False
        finally:
            proc.kill()
            proc.wait()


class TestScriptsDelegate:
    def test_all_store_writing_scripts_use_shared_guard(self):
        scripts = [
            "scripts/add_profile_fact.py",
            "scripts/budget_experiment.py",
            "scripts/dedup_reference_docs.py",
            "scripts/purge_junk_facts.py",
            "scripts/purge_error_memories.py",
            "scripts/purge_adaptive_exemplars.py",
            "scripts/purge_profile_facts.py",
            "scripts/repair_thinking_leaks.py",
        ]
        for s in scripts:
            src = Path(s).read_text()
            assert "daemon_guard" in src, f"{s} does not delegate to utils.daemon_guard"
