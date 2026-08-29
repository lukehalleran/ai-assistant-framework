"""DAEMON_TEST_MODE isolation — test traffic must not pollute prod records.

2026-08-28 retrospective: benchmark/test rows sat un-flagged in prod
turn_records.jsonl (95+ rows, placeholder model values) skewing fire-rate
metrics, and a pytest import of main/gui.launch ROTATED the live daemon's
debug log and wrote test output into daemon_debug.log. The sentinel is set
by tests/conftest.py; these tests drive the two deployed consumers.
"""
import json
import logging
import os

import pytest


def test_conftest_sets_sentinel():
    # Running under pytest, the root conftest must have set the sentinel.
    assert os.getenv("DAEMON_TEST_MODE") == "1"


def test_record_turn_stamps_test_env(tmp_path, monkeypatch):
    import config.app_config as ac
    import utils.turn_telemetry as tt
    p = tmp_path / "records.jsonl"
    monkeypatch.setattr(ac, "TURN_TELEMETRY_PATH", str(p))
    monkeypatch.setattr(ac, "TURN_TELEMETRY_ENABLED", True)
    assert tt.record_turn({"query": "hello from a test"})
    row = json.loads(p.read_text().strip())
    assert row.get("test_env") is True


def test_record_turn_no_stamp_without_sentinel(tmp_path, monkeypatch):
    import config.app_config as ac
    import utils.turn_telemetry as tt
    monkeypatch.delenv("DAEMON_TEST_MODE", raising=False)
    p = tmp_path / "records.jsonl"
    monkeypatch.setattr(ac, "TURN_TELEMETRY_PATH", str(p))
    monkeypatch.setattr(ac, "TURN_TELEMETRY_ENABLED", True)
    assert tt.record_turn({"query": "prod-shaped row"})
    row = json.loads(p.read_text().strip())
    assert "test_env" not in row


def test_configure_logging_redirects_file_sink(tmp_path, monkeypatch):
    """Under the sentinel, the file handler must NEVER point at the prod
    daemon_debug.log path (which configure_logging would also rotate)."""
    from utils.logging_utils import configure_logging
    monkeypatch.chdir(tmp_path)  # rotation/creation happens in cwd — sandbox it
    (tmp_path / "logs").mkdir()
    root = logging.getLogger()
    saved = root.handlers[:]
    try:
        configure_logging(file_path="daemon_debug.log")
        fhs = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert fhs, "file handler expected"
        assert all("test_debug" in h.baseFilename for h in fhs)
        assert not (tmp_path / "daemon_debug.log").exists()
    finally:
        for h in root.handlers[:]:
            if isinstance(h, logging.FileHandler):
                h.close()
        root.handlers.clear()
        root.handlers.extend(saved)
