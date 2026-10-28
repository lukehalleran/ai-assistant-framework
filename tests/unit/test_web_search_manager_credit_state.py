"""WebSearchRateLimiter credit-state persistence fails CLOSED, and save failures are visible.

Incident: 2026-10-27 wave-3b swallow audit. ``_load_state`` swallowed a corrupt or
wrong-shape credit file and restarted usage from ZERO, so a restart could overspend
the daily Tavily credit budget; ``_save_state`` failures were DEBUG-only.
Classes: BC-69 (silent swallow), BC-70 (diagnostic invisible), BC-58.
Drives the deployed WebSearchRateLimiter against tmp_path state files.
"""
import glob
import json
import logging
import os
from datetime import datetime, timedelta

import pytest

from knowledge.web_search_manager import WebSearchRateLimiter

TODAY = datetime.now().strftime("%Y-%m-%d")


def _limiter(path, limit=10):
    return WebSearchRateLimiter(daily_limit=limit, state_file=str(path))


def _quarantined(path):
    return glob.glob(str(path) + ".corrupt-*")


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_missing_file_starts_at_zero_and_allows(tmp_path):
    lim = _limiter(tmp_path / "credits.json")
    assert lim.can_search(1.0) is True
    assert lim._credits_today == 0.0


def test_valid_state_loads(tmp_path):
    p = tmp_path / "credits.json"
    p.write_text(json.dumps({"credits_today": 4, "date": TODAY}))
    lim = _limiter(p)
    assert lim._credits_today == 4.0
    assert lim.can_search(1.0) is True
    assert lim.can_search(7.0) is False


@pytest.mark.parametrize("payload", [
    b"{not json at all",
    json.dumps([1, 2, 3]).encode(),
    json.dumps({"credits_today": "lots", "date": TODAY}).encode(),
    json.dumps({"date": TODAY}).encode(),
])
def test_unreadable_or_wrong_shape_fails_closed_with_quarantine_and_warning(tmp_path, caplog, payload):
    p = tmp_path / "credits.json"
    p.write_bytes(payload)
    with caplog.at_level(logging.DEBUG):
        lim = _limiter(p)
    assert lim.can_search(1.0) is False
    assert lim._credits_today == float(lim.daily_limit)
    assert _quarantined(p), "no quarantine copy"
    msgs = _warnings(caplog)
    assert any("paused until the date changes" in m for m in msgs), msgs


def test_fail_closed_state_clears_on_next_day(tmp_path):
    p = tmp_path / "credits.json"
    p.write_bytes(b"{garbage")
    lim = _limiter(p)
    assert lim.can_search(1.0) is False
    lim._current_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    assert lim.can_search(1.0) is True
    assert lim._credits_today == 0.0


def test_save_failure_warns_and_keeps_memory_state(tmp_path, caplog, monkeypatch):
    lim = _limiter(tmp_path / "credits.json")

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("knowledge.web_search_manager.atomic_write_json", boom)
    with caplog.at_level(logging.DEBUG):
        lim.record_usage(2.0)
    assert lim._credits_today == 2.0
    msgs = _warnings(caplog)
    assert any("Failed to persist credit state" in m and "OSError" in m for m in msgs), msgs


def test_save_roundtrip_persists_and_reloads(tmp_path):
    p = tmp_path / "credits.json"
    lim = _limiter(p)
    lim.record_usage(3.0)
    again = _limiter(p)
    assert again._credits_today == 3.0
    assert not _quarantined(p)


def test_fail_closed_state_is_persisted_so_a_restart_does_not_quarantine_again(tmp_path):
    """Referee addition: after a quarantined fail-closed load the bad file is replaced by the
    fail-closed state, so the next process reads 'spent for today' without a second quarantine copy."""
    p = tmp_path / "credits.json"
    p.write_bytes(b"{not json at all")
    first = _limiter(p)
    assert first.can_search(1.0) is False
    assert len(_quarantined(p)) == 1
    assert json.loads(p.read_text()) == {"credits_today": 10.0, "date": TODAY}
    second = _limiter(p)
    assert second.can_search(1.0) is False
    assert len(_quarantined(p)) == 1, "a restart re-quarantined the already-replaced file"


def test_failed_quarantine_leaves_the_original_in_place(tmp_path, monkeypatch):
    """When the quarantine copy cannot be made, the only copy of the bad file is never overwritten."""
    import utils.safe_json as sj
    monkeypatch.setattr(sj, "quarantine_corrupt_file", lambda path: None)
    p = tmp_path / "credits.json"
    p.write_bytes(b"{not json at all")
    lim = _limiter(p)
    assert lim.can_search(1.0) is False
    assert p.read_bytes() == b"{not json at all"


def test_default_limiter_in_tests_never_uses_the_cwd_credits_file(tmp_path):
    """Referee addition: the conftest sandbox points WEB_SEARCH_CREDITS_PATH into the test's tmp dir,
    so a default-constructed limiter can never read or rewrite a checkout's data/web_search_credits.json."""
    lim = WebSearchRateLimiter(daily_limit=10)
    assert os.path.abspath(lim.state_file).startswith(str(tmp_path))
    assert lim.can_search(1.0) is True
