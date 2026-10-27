"""Failures that lose owner-visible work in utils/ log a greppable WARNING.

Incident date: 2026-10-27 (oct27b wave 3a, lane SWB).
Bug classes: BC-69 (silent ops failure), BC-70 (log severity misdescribes control flow).

Each handler below used to swallow its failure at DEBUG. They now keep the
non-raising contract but log `[Module] <what failed>: <ExcType>` at WARNING.
"""
import logging

import pytest

from utils import narrative_staleness, turn_telemetry
from utils.institution_resolver import InstitutionResolver
from utils.timezone_resolver import TimezoneResolver
from utils.user_identity import UserIdentityResolver


def _bad_profile(tmp_path):
    p = tmp_path / "profile.json"
    p.write_text("{not json", encoding="utf-8")
    return str(p)


@pytest.mark.parametrize(
    "factory,call,tag",
    [
        (InstitutionResolver, "get_institution", "[Institution] profile read failed"),
        (TimezoneResolver, "get_timezone", "[Timezone] profile read failed"),
        (UserIdentityResolver, "get_display_name", "[UserIdentity] profile read failed"),
    ],
)
def test_corrupt_profile_read_warns(tmp_path, caplog, factory, call, tag):
    resolver = factory(profile_path=_bad_profile(tmp_path))
    with caplog.at_level(logging.WARNING):
        getattr(resolver, call)()  # must not raise
    recs = [r for r in caplog.records if tag in r.getMessage()]
    assert recs and all(r.levelno == logging.WARNING for r in recs)
    assert "JSONDecodeError" in recs[0].getMessage()
    assert "not json" not in recs[0].getMessage()


def test_mark_stale_failure_warns(tmp_path, monkeypatch, caplog):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    monkeypatch.setenv("NARRATIVE_STALE_FLAG_PATH", str(blocker / "sub" / "flag.json"))
    with caplog.at_level(logging.WARNING):
        assert narrative_staleness.mark_stale("a correction") is False
    recs = [r for r in caplog.records if "[NarrativeStaleness] mark_stale failed" in r.getMessage()]
    assert recs and recs[0].levelno == logging.WARNING


def test_telemetry_write_failure_warns(tmp_path, monkeypatch, caplog):
    import config.app_config as cfg

    monkeypatch.setattr(cfg, "TURN_TELEMETRY_ENABLED", True)
    monkeypatch.setattr(cfg, "TURN_TELEMETRY_PATH", str(tmp_path))  # a directory: open() fails
    with caplog.at_level(logging.WARNING):
        assert turn_telemetry.record_turn({"query": "q"}) is False
    recs = [r for r in caplog.records if "[TurnTelemetry] write failed" in r.getMessage()]
    assert recs and recs[0].levelno == logging.WARNING
    assert "IsADirectoryError" in recs[0].getMessage()
