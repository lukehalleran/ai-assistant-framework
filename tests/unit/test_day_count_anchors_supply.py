"""[YOUR DAY COUNTS]: store round-trip, deployed builder supply, formatter, dated notes render.

Incident 2026-09-11..20 (class: BC-103, BC-100, BC-75, BC-52): a note's "Day 23" taken as today's
count. The user's own dated counts are stored by the post-response hook, then handed to the reply
prompt already projected to today. Invented activity throughout.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pytest

import memory.day_count_anchors as dca
from config import app_config
from core.prompt.formatter import PromptFormatter, _note_date_label
from tests.unit.test_independent_prompt_audit import full_builder, retrieval_limits
from utils import safe_json

TODAY = date.today()


def ago(n):
    return TODAY - timedelta(days=n)


def note(title, day, content="body", basis="content", **meta):
    return {"content": content, "relevance_score": 0.9,
            "metadata": {"title": title, "note_date": day.isoformat() if day else "", "date_basis": basis, **meta}}


# --- store ---------------------------------------------------------------------------------------

def test_round_trip_and_dedupe():
    assert dca.record_from_user_text("I'm on day 16 of the cold-shower habit", ago(60)) == 1
    assert dca.record_from_user_text("I'm on day 16 of the cold-shower habit", ago(60)) == 0
    raw = json.load(open(dca.store_path()))
    assert raw["schema_version"] == 1 and raw["claims"][0]["count"] == 16
    dca.reset_for_tests(dca.store_path())                       # fresh process: strict reload
    lines = dca.supply_lines("how are the cold showers")
    assert len(lines) == 1 and f"day {16 + 60} today ({TODAY.isoformat()})" in lines[0]


def test_corrupt_store_is_quarantined_and_never_overwritten():
    open(dca.store_path(), "w").write("{ broken")
    with pytest.raises(safe_json.CorruptStoreError):
        dca.record_from_user_text("day 16 of the cold-shower habit", TODAY)
    assert open(dca.store_path()).read() == "{ broken"          # untouched
    assert dca.supply_lines("cold showers") == []                # prompt path degrades, no raise
    import glob
    assert glob.glob(dca.store_path() + ".corrupt-*")


def test_newer_schema_is_refused():
    json.dump({"schema_version": 99, "claims": []}, open(dca.store_path(), "w"))
    with pytest.raises(safe_json.StoreVersionError):
        dca.record_from_user_text("day 16 of the cold-shower habit", TODAY)


def test_paste_and_disabled_record_nothing(monkeypatch):
    assert dca.record_from_user_text("day 16 of the cold-shower habit " + "x " * 800, TODAY) == 0
    monkeypatch.setattr(app_config, "DAY_COUNTS_ENABLED", False)
    assert dca.record_from_user_text("day 16 of the cold-shower habit", TODAY) == 0
    assert dca.supply_lines("cold showers") == []


# --- notes: user-authored, content-dated only ------------------------------------------------------------

def test_note_claims_filter_authorship_and_date_basis():
    ok = note("7 28 26 Daily Note (Day 23)", ago(5))
    assert [c.count for c in dca.note_claims([ok])] == [23]
    assert dca.note_claims([note("Note (Day 23)", ago(5), basis="index-time")]) == []
    assert dca.note_claims([note("Note (Day 23)", ago(5), author="daemon")]) == []
    assert dca.note_claims([note("Note (Day 23)", ago(5), source_type="daemon_daily_summary")]) == []
    assert dca.note_claims([note("Note (Day 23)", None)]) == []
    # body: attributed shapes only (a bare "day 3" in prose is not a count)
    assert dca.note_claims([note("Daily", ago(5), "slept badly on day 3. 9 days into cold showers")])[0].count == 9


@pytest.mark.parametrize("wrapped", [False, True])
def test_incident_replay_message_then_dated_note_then_turn_sixty_days_later(wrapped):
    msg = "I'm on day 16 of the cold-shower habit"
    dca.record_from_user_text(("\n   " + msg + "\n   ") if wrapped else msg, ago(66))
    n = note("7 28 26 Daily Note (Day 23)", ago(60))             # D+6, unattributed -> binds by start date
    lines = dca.supply_lines("is the cold-shower habit still worth it?", [n])
    assert len(lines) == 1 and f"day {23 + 60} today" in lines[0]
    assert dca.supply_lines("what should I cook tonight", [n]) == []       # relevance gate: no block


def test_daemon_authored_count_never_becomes_an_anchor():
    # The hook passes the USER's text only; assistant wording is not an input to record_from_user_text,
    # and a Daemon-authored summary note is refused as a source.
    assert dca.note_claims([note("Summary", ago(3), "You are 23 days into the cold-shower habit",
                                 author="daemon")]) == []
    assert dca.supply_lines("cold showers", [note("S", ago(3), "23 days into cold showers", source_type="daemon_daily_summary")]) == []


# --- deployed builder + formatter ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_builder_supplies_projected_count_when_activity_is_in_play(monkeypatch):
    dca.record_from_user_text("I'm on day 16 of the cold-shower habit", ago(60))
    builder = full_builder(monkeypatch, [{"query": "hi", "response": "hello"}], budget=10000)
    ctx = await builder.build_prompt("how long have I been doing the cold-shower habit?",
                                     retrieval_overrides=retrieval_limits())
    assert len(ctx["day_counts"]) == 1 and "day 76" in ctx["day_counts"][0]
    off_topic = await builder.build_prompt("what is a good pasta recipe", retrieval_overrides=retrieval_limits())
    assert off_topic["day_counts"] == []
    monkeypatch.setattr(app_config, "DAY_COUNTS_ENABLED", False)           # same instance, live flag
    off = await builder.build_prompt("how long have I been doing the cold-shower habit?",
                                     retrieval_overrides=retrieval_limits())
    assert off["day_counts"] == []


@pytest.mark.asyncio
async def test_builder_never_supplies_on_a_local_repo_audit(monkeypatch):
    dca.record_from_user_text("day 16 of the repo audit habit", ago(60))
    builder = full_builder(monkeypatch, [{"query": "hi", "response": "hello"}], budget=10000)
    ctx = await builder.build_prompt("Do a read-only audit of the repo at /home/lukeh/project and run pytest",
                                     retrieval_overrides=retrieval_limits())
    assert ctx["day_counts"] == []


def test_formatter_renders_the_section_only_with_lines():
    fmt = PromptFormatter(token_manager=None, time_manager=None)
    out = fmt._assemble_prompt({"day_counts": ["- cold-shower habit: day 76 today"]}, "q", "")
    assert "[YOUR DAY COUNTS" in out and "day 76 today" in out and "never reuse it" in out
    assert "[YOUR DAY COUNTS" not in fmt._assemble_prompt({"day_counts": []}, "q", "")


def test_notes_header_carries_the_content_date_and_age():
    fmt = PromptFormatter(token_manager=None, time_manager=None)
    n = note("7 28 26 Daily Note (Day 23)", ago(91), "Felt ok today.")
    out = fmt._assemble_prompt({"personal_notes": [n]}, "q", "")
    assert f"[note dated {ago(91).isoformat()} (" in out and " ago)]" in out
    idx = note("7 28 26 Daily Note (Day 23)", ago(91), "Felt ok.", basis="index-time")
    assert "[note dated" not in fmt._assemble_prompt({"personal_notes": [idx]}, "q", "")
    assert _note_date_label({"note_date": "garbage", "date_basis": "content"}) == ""
    assert _note_date_label(None) == ""
