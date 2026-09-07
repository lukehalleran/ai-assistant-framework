"""SIGHUP → clean shutdown (2026-09-07): a dropped SSH session must trigger the
same shutdown as Ctrl+C, never kill the daemon mid-shutdown.

The handler is exercised in a real child interpreter: the child installs it,
reports ready, and the test sends SIGHUP. Without the handler the default
action terminates the child (exit -1 / 129, marker never written)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

CHILD = r"""
import os, signal, sys, time
sys.path.insert(0, {repo!r})
from utils.process_signals import install_hangup_handler
marker = {marker!r}
ready = {ready!r}
installed = install_hangup_handler()
open(ready, "w").write("installed" if installed else "skipped")
try:
    while True:
        time.sleep(0.05)
except KeyboardInterrupt:
    # stdout was detached by the handler: this print must not raise
    print("after hangup")
    open(marker, "w").write("clean shutdown ran")
    sys.exit(0)
"""


def _spawn(tmp_path: Path):
    marker = tmp_path / "marker.txt"
    ready = tmp_path / "ready.txt"
    script = CHILD.format(repo=str(REPO_ROOT), marker=str(marker), ready=str(ready))
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={**os.environ, "DAEMON_TEST_MODE": "1"},
    )
    for _ in range(100):
        if ready.exists():
            break
        time.sleep(0.05)
    else:
        proc.kill()
        pytest.fail("child never signalled readiness")
    return proc, marker, ready


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="no SIGHUP on this platform")
class TestHangupBecomesCleanShutdown:
    def test_sighup_runs_the_keyboardinterrupt_path(self, tmp_path):
        proc, marker, ready = _spawn(tmp_path)
        assert ready.read_text() == "installed"
        os.kill(proc.pid, signal.SIGHUP)
        rc = proc.wait(timeout=10)
        assert rc == 0, f"child exit {rc}: SIGHUP did not become a clean shutdown"
        assert marker.read_text() == "clean shutdown ran"

    def test_without_handler_sighup_kills(self, tmp_path):
        """Control: the default SIGHUP action terminates the interpreter."""
        script = "import time\nopen(%r,'w').write('x')\nwhile True: time.sleep(0.05)" % str(tmp_path / "r")
        proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            if (tmp_path / "r").exists():
                break
            time.sleep(0.05)
        os.kill(proc.pid, signal.SIGHUP)
        rc = proc.wait(timeout=10)
        assert rc != 0

    def test_second_hangup_is_ignored(self):
        from utils import process_signals as ps
        ps._HANGUP_SEEN = True
        try:
            saved = signal.getsignal(signal.SIGHUP)
            assert ps.install_hangup_handler() is True
            handler = signal.getsignal(signal.SIGHUP)
            handler(signal.SIGHUP, None)  # must NOT forward SIGINT to the test process
        finally:
            signal.signal(signal.SIGHUP, saved)
            ps._HANGUP_SEEN = False
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__


# ---------------------------------------------------------------------------
# main.py wiring (source-level, like test_main_module_alias.py — importing
# main pulls the whole orchestrator stack into a unit test).
# ---------------------------------------------------------------------------

class TestMainWiring:
    SRC = (REPO_ROOT / "main.py").read_text()

    def test_hangup_handler_installed_on_both_launch_paths(self):
        assert self.SRC.count("install_hangup_handler(logger=logger)") == 2

    def test_second_shutdown_entrant_waits_for_inflight_run(self):
        body = self.SRC.split("async def run_shutdown_tasks_async", 1)[1].split("\ndef ", 1)[0]
        assert "_shutdown_done.wait" in body, "lifespan entrant must wait for an in-flight idle shutdown"
        assert body.index("_shutdown_done.wait") < body.index("_shutdown_requested = True")
        assert body.rstrip().endswith("_shutdown_done.set()")
        sync_body = self.SRC.split("def _run_shutdown_tasks(orchestrator)", 1)[1].split("\nasync def ", 1)[0]
        assert "finally:\n        _shutdown_done.set()" in sync_body
