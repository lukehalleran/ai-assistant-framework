"""A move the user states mid-session reaches the location resolver in that session.

Incident date: 2026-10-27 (M1 V4: "I moved to <city>" only reached `get_user_location()` after
the shutdown profile write, i.e. the next process; searches kept localizing to the old place).
Class: BC-46 (session state invisible to a consumer), BC-04 (localization correctness).

Drives THE deployed LocationResolver.get_location with a temp profile + the in-process corpus entries
(handed over by a provider callable, as main.py registers `corpus_manager.visible_entries`; no file read). Statements are judged by the deployed per-turn
fact extractor. Cities are invented-use only (Portland/Salem/Reno are public places, not the owner's).
Every statement is asserted clean and wrapped/indented.
"""
import json
import textwrap
import time
from datetime import datetime, timedelta

import pytest

import utils.location_resolver as lr
from utils.location_resolver import LocationResolver


def _profile(tmp_path, place="Springfield, IL"):
    p = tmp_path / "profile.json"
    p.write_text(json.dumps({"categories": {"identity": [{
        "relation": "lives_in", "value": place,
        "timestamp": "2026-03-22T23:57:07", "is_current": True,
    }]}}))
    return str(p)


def _turn(query, minutes_ago=1, **extra):
    ts = (datetime.now() - timedelta(minutes=minutes_ago)).isoformat()  # clock-safe: compared only against session_start (also now-relative); no calendar-day boundary
    return {"query": query, "response": "ok", "timestamp": ts, **extra}


@pytest.fixture(autouse=True)
def _chain(monkeypatch):
    monkeypatch.setattr(lr, "LOCATION_ENABLED", True)
    monkeypatch.setattr(lr, "LOCATION_IP_LOOKUP_ENABLED", False)
    monkeypatch.setattr(lr, "LOCATION_OVERRIDE", "")


def _resolver(tmp_path, turns, **kw):
    """`turns` is the live in-process list; the provider hands it out by reference."""
    return LocationResolver(
        profile_path=_profile(tmp_path), turn_provider=lambda: turns,
        session_start=time.time() - 3600, **kw,  # clock-safe: compared only against session_start (also now-relative); no calendar-day boundary
    )


def _wrap(text):
    return "\n".join("    " + ln for ln in textwrap.wrap(text, 14))


@pytest.mark.parametrize("wrap", [False, True])
def test_stated_move_beats_the_profile_in_the_same_session(tmp_path, wrap):
    q = "I just moved to Portland, Oregon."
    r = _resolver(tmp_path, [_turn(_wrap(q) if wrap else q)])
    assert r.get_location() == "Portland, Oregon"


def test_same_instance_before_and_after_the_statement(tmp_path):
    turns = [_turn("what a day")]
    r = _resolver(tmp_path, turns)
    assert r.get_location() == "Springfield, IL"          # nothing stated yet -> profile, unchanged
    turns.append(_turn("Update: I live in Salem, Oregon now", 0.5))
    assert r.get_location() == "Salem, Oregon"            # same instance sees the new statement


def test_newest_statement_wins(tmp_path):
    r = _resolver(tmp_path, [_turn("I moved to Reno, Nevada", 20), _turn("I moved to Salem, Oregon", 2)])
    assert r.get_location() == "Salem, Oregon"


def test_config_override_still_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(lr, "LOCATION_OVERRIDE", "Geneva, IL")
    r = _resolver(tmp_path, [_turn("I just moved to Portland, Oregon.")])
    assert r.get_location() == "Geneva, IL"


@pytest.mark.parametrize("statement", [
    "I'm moving to Denver, Colorado next month.",          # a plan, not a residence
    "my sister moved to Austin, Texas",                    # someone else's move
    "I used to live in Reno, Nevada",                      # past residence
    "If I moved to Miami, Florida I'd be warmer",          # conditional
    "I live in a bad mood",                                # not a place
])
@pytest.mark.parametrize("wrap", [False, True])
def test_non_residence_statements_do_not_move_the_user(tmp_path, statement, wrap):
    r = _resolver(tmp_path, [_turn(_wrap(statement) if wrap else statement)])
    assert r.get_location() == "Springfield, IL"


def test_previous_session_turns_are_left_to_the_profile(tmp_path):
    old = _turn("I moved to Reno, Nevada", minutes_ago=60 * 24)
    r = _resolver(tmp_path, [old])
    assert r.get_location() == "Springfield, IL"


def test_probe_and_quarantined_turns_are_ignored(tmp_path):
    turns = [
        _turn("[test] I just moved to Portland, Oregon. [/test]", 3),
        _turn("I just moved to Salem, Oregon.", 2, curation_quarantined=True),
    ]
    assert _resolver(tmp_path, turns).get_location() == "Springfield, IL"


def test_attachment_text_is_not_a_statement(tmp_path):
    # `query` is the merged blob; the user's own typed text is `user_text`
    turn = _turn("please summarize this. Resume: I live in Reno, Nevada", user_text="please summarize this.")
    assert _resolver(tmp_path, [turn]).get_location() == "Springfield, IL"


def test_failing_provider_falls_back_to_profile(tmp_path):
    def boom():
        raise RuntimeError("provider down")
    r = LocationResolver(profile_path=_profile(tmp_path), turn_provider=boom, session_start=0)
    assert r.get_location() == "Springfield, IL"


def test_no_provider_means_no_session_evidence(tmp_path):
    r = LocationResolver(profile_path=_profile(tmp_path))
    assert r.turn_provider is None
    assert r.get_location() == "Springfield, IL"


def test_session_move_is_picked_up_without_opening_any_file(tmp_path, monkeypatch):
    import builtins
    r = _resolver(tmp_path, [_turn("I just moved to Portland, Oregon.")])
    r.get_location()   # warm: profile path may be read once on a miss
    opened = []
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (opened.append(str(f)), real_open(f, *a, **k))[1])
    turns = [_turn("I just moved to Salem, Oregon.", 0.2)]
    r2 = LocationResolver(profile_path=_profile(tmp_path), turn_provider=lambda: turns,
                          session_start=time.time() - 3600)  # clock-safe: compared only against session_start (also now-relative); no calendar-day boundary
    assert r2.get_location() == "Salem, Oregon"
    assert not [f for f in opened if "corpus" in f], opened


def test_watermark_second_call_scans_only_new_entries(tmp_path, monkeypatch):
    calls = []
    real = lr._stated_place_from_text
    monkeypatch.setattr(lr, "_stated_place_from_text", lambda t: (calls.append(t), real(t))[1])
    turns = [_turn("what a day", 6), _turn("walked the dog", 5)]
    r = _resolver(tmp_path, turns)
    r.get_location()
    assert len(calls) == 2
    r.get_location()
    assert len(calls) == 2, "unchanged corpus must cost no extraction"
    turns.append(_turn("I live in Salem, Oregon now", 0.1))
    assert r.get_location() == "Salem, Oregon"
    assert len(calls) == 3, "only the new turn is extracted"
    r.get_location()
    assert len(calls) == 3


def test_composition_root_registers_the_provider():
    # the module-level hook main.py calls; a resolver without its own provider uses it
    entries = [_turn("I moved to Reno, Nevada", 1)]
    lr.set_session_turn_provider(lambda: entries)
    try:
        r = LocationResolver(profile_path="/nonexistent/profile.json", session_start=time.time() - 3600)  # clock-safe: compared only against session_start (also now-relative); no calendar-day boundary
        assert r.get_location() == "Reno, Nevada"
    finally:
        lr.set_session_turn_provider(None)


def test_datetime_timestamps_as_held_in_process(tmp_path):
    # CorpusManager holds datetime objects in memory (ISO strings only on disk)
    turns = [{"query": "I live in Salem, Oregon now", "response": "ok", "timestamp": datetime.now()}]
    assert _resolver(tmp_path, turns).get_location() == "Salem, Oregon"
