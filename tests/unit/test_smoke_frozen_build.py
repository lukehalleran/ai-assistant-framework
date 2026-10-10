"""scripts/smoke_frozen_build.py: the pure parts, with NO binary launched.

The script is loaded by path (scripts/ is not a package). Nothing here starts a
process, binds 8000/7860, or touches ~/.daemon.
"""
from __future__ import annotations

import importlib.util
import json
import os
import socket
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "smoke_frozen_build.py"


@pytest.fixture(scope="module")
def smoke():
    spec = importlib.util.spec_from_file_location("smoke_frozen_build_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -- strict write filter -------------------------------------------------------

def test_allowed_writes_pass(smoke):
    paths = ["", "home", "home/.daemon", "home/.daemon/backups/20261010/x.json",
             "home/.daemon/logs/daemon_debug.log", "home/.cache/chroma/telemetry_user_id",
             "home/.nv/ComputeCache", "run.log", ".stamp"]
    assert smoke.strict_write_violations(paths) == []


# `data/...` literals are built with os.path.join so no test names a production
# store path as a string constant (DM-17); these are strings the filter sees.
STRAY_DATA = os.path.join("data", "x.json")


@pytest.mark.parametrize("stray", [
    STRAY_DATA, "logs/curation_audit.jsonl", "conversation_logs/c.txt", "daemon_debug.log",
    os.path.join("home", "data", "x.json"), "home/.daemonx/y", "home/.config/z", "other/run.log",
    os.path.join("home", ".daemon", "..", "data", "x.json"),
])
def test_stray_writes_are_violations(smoke, stray):
    assert smoke.strict_write_violations(["home/.daemon/ok.json", stray]) == [stray]


def test_find_newer_matches_find_newer_semantics(smoke, tmp_path):
    (tmp_path / "home" / ".daemon").mkdir(parents=True)
    stamp = tmp_path / ".stamp"
    stamp.touch()
    old = time.time() - 100
    os.utime(stamp, (old, old))  # make the stamp clearly older than what follows
    (tmp_path / "home" / ".daemon" / "a.json").write_text("{}")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.json").write_text("{}")
    newer = smoke.find_newer(tmp_path, stamp)
    assert "home/.daemon/a.json" in newer and STRAY_DATA in newer
    assert smoke.strict_write_violations(newer) == ["data", STRAY_DATA]


# -- ports ----------------------------------------------------------------------

def test_pick_free_ports_never_forbidden_and_distinct(smoke):
    for _ in range(25):
        ports = smoke.pick_free_ports(2)
        assert len(set(ports)) == 2
        assert not (set(ports) & {8000, 7860})


def test_pick_free_ports_skips_forbidden_ports(smoke, monkeypatch):
    seq = iter([8000, 7860, 45001, 45002])

    class FakeSock:
        def __init__(self, *a, **k): ...
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def bind(self, addr): ...
        def getsockname(self): return ("127.0.0.1", next(seq))
        def settimeout(self, t): ...
        def connect_ex(self, addr): return 1  # nothing listening

    monkeypatch.setattr(smoke.socket, "socket", FakeSock)
    assert smoke.pick_free_ports(2) == [45001, 45002]


def test_assert_ports_safe_refuses_forbidden_and_occupied(smoke):
    with pytest.raises(SystemExit, match="forbidden"):
        smoke.assert_ports_safe([8000])
    with pytest.raises(SystemExit, match="forbidden"):
        smoke.assert_ports_safe([7860])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        busy = srv.getsockname()[1]
        assert smoke.port_in_use(busy)
        with pytest.raises(SystemExit, match="already in use"):
            smoke.assert_ports_safe([busy])
    free = smoke.pick_free_ports(1)[0]
    smoke.assert_ports_safe([free])  # no raise


# -- environment ----------------------------------------------------------------

def test_build_env_is_from_scratch(smoke, tmp_path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    monkeypatch.setenv("OPENAI_API_KEY", "real-looking")
    env = smoke.build_env(tmp_path / "home", 41000, 41001)
    assert env["HOME"] == str(tmp_path / "home")
    assert "DISPLAY" not in env and "PYTHONPATH" not in env and "OPENAI_API_KEY" not in env
    assert env["DAEMON_API_PORT"] == "41000"
    assert env["GRADIO_PORT"] == env["GRADIO_SERVER_PORT"] == "41001"
    assert env["DAEMON_API_HOST"] == "127.0.0.1"
    assert set(env) <= {"PATH", "HOME", "LANG", "DAEMON_API_HOST", "DAEMON_API_PORT",
                        "GRADIO_SERVER_NAME", "GRADIO_SERVER_PORT", "GRADIO_PORT"}


# -- synthetic pre-seed -----------------------------------------------------------

def test_seed_writes_only_synthetic_values(smoke, tmp_path):
    made = smoke.seed_returning_user(tmp_path / "home")
    assert {p.name for p in made} == {"user_profile.json", ".env"}
    assert all(str(p).startswith(str(tmp_path)) for p in made)
    profile = json.loads((tmp_path / "home/.daemon/user_profile.json").read_text())
    assert profile["identity"]["name"] == "Smoke Tester"
    env_lines = (tmp_path / "home/.daemon/.env").read_text().splitlines()
    assert "OPENROUTER_API_KEY=sk-or-v1-fake-smoke" in env_lines
    assert all(line.startswith(("OPENROUTER_API_KEY=", "DAEMON_MODE=")) for line in env_lines)
    # nothing but those two files
    assert sorted(p.name for p in (tmp_path / "home/.daemon").iterdir()) == [".env", "user_profile.json"]


# -- report / exit code ------------------------------------------------------------

def test_exit_code_and_table(smoke):
    ok = [("a", smoke.PASS, "fine"), ("b", smoke.SKIP, "")]
    bad = ok + [("c", smoke.FAIL, "stray: data/x.json")]
    assert smoke.exit_code(ok) == 0 and smoke.exit_code(bad) == 1
    table = smoke.render_table(bad)
    assert "OVERALL: FAIL" in table and "stray: data/x.json" in table
    assert "OVERALL: PASS" in smoke.render_table(ok)


def test_write_report(smoke, tmp_path):
    path = smoke.write_report(tmp_path, [("x", smoke.FAIL, "boom")], ["- note"])
    text = path.read_text()
    assert path.name == "report.md" and "boom" in text and "- note" in text


# -- small parsers ----------------------------------------------------------------

def test_shutdown_phases(smoke):
    log = "[Shutdown] 1/6 Draining\n[Shutdown] 5/6 Curation\n[Shutdown] 6/6 Backup written"
    assert smoke.shutdown_phases(log) == (6, 6)
    assert smoke.shutdown_phases("[Shutdown] 2/6 x") == (2, 6)
    assert smoke.shutdown_phases("nothing") == (0, 0)


def test_extract_launch_token(smoke):
    html = '<head><meta name="daemon-launch-token" content="abc_DEF-123"></head>'
    assert smoke.extract_launch_token(html) == "abc_DEF-123"
    assert smoke.extract_launch_token("<head></head>") is None


def test_snapshot_dir_detects_change_without_reading_contents(smoke, tmp_path):
    d = tmp_path / "guard"
    d.mkdir()
    (d / "f").write_text("one")
    a = smoke.snapshot_dir(d)
    assert a == smoke.snapshot_dir(d)
    (d / "g").write_text("two")
    assert a != smoke.snapshot_dir(d)
    assert smoke.snapshot_dir(tmp_path / "absent")["exists"] is False


def test_freeze_command_is_capped_and_niced(smoke, tmp_path):
    cmd = smoke.freeze_command(tmp_path)
    assert cmd[:3] == ["nice", "-n", "10"]
    assert "PyInstaller" in cmd and "daemon.spec" in cmd
    assert ("env" in cmd) and "-u" in cmd and "PYTHONPATH" in cmd
