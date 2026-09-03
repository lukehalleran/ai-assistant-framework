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
from pathlib import Path

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


def _flush_file_handlers(root):
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()


def test_configure_logging_suppresses_raw_transport_debug(tmp_path, monkeypatch):
    """A DEBUG file sink must not persist an SDK request body by default."""
    from utils.logging_utils import configure_logging

    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    monkeypatch.setenv("DAEMON_MODE", "dev")
    monkeypatch.delenv("DAEMON_ALLOW_SENSITIVE_HTTP_LOGS", raising=False)
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_levels = {
        name: logging.getLogger(name).level
        for name in ("openai", "httpx", "httpcore", "urllib3")
    }
    try:
        configure_logging(file_path="ignored-under-test.log")
        logging.getLogger("daemon.test").debug("ordinary-debug-marker")
        logging.getLogger("openai._base_client").debug(
            "Request options: {'messages': 'private-prompt-marker'}"
        )
        _flush_file_handlers(root)
        content = Path("logs/test_debug.log").read_text()
        assert "ordinary-debug-marker" in content
        assert "private-prompt-marker" not in content
    finally:
        for handler in root.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
        root.handlers.clear()
        root.handlers.extend(saved_handlers)
        for name, level in saved_levels.items():
            logging.getLogger(name).setLevel(level)


def test_sensitive_transport_logging_requires_both_dev_flags(monkeypatch):
    from utils.logging_utils import _sensitive_http_logging_enabled

    monkeypatch.setenv("DAEMON_ALLOW_SENSITIVE_HTTP_LOGS", "1")
    monkeypatch.setenv("DAEMON_MODE", "user")
    assert _sensitive_http_logging_enabled() is False

    monkeypatch.setenv("DAEMON_MODE", "dev")
    assert _sensitive_http_logging_enabled() is True
