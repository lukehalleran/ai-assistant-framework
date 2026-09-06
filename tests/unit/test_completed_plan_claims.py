"""Tests for utils/completed_plan_claims.py.

Covers the six acceptance scenarios from the Codex audit handoff (completed
social plan removed with a dated CAUTION line; unrelated statement left
alone; digit-conflict guard; earlier-than-plan-date guard; idempotency) plus
the narrative-generator wiring order (streak check runs before the
completed-plan check).
"""

from datetime import date

import pytest

from utils.completed_plan_claims import (
    completed_by_user,
    plan_sentences,
    remove_completed_plan_claims,
)


def _stmt(text, day):
    return {"user_text": text, "timestamp": day}


# --- (i) completed plan sentence removed with dated CAUTION -------------

def test_completed_plan_removed_with_dated_caution():
    narrative = (
        "Sent weekend hangout invites; hanging out with a friend Saturday is pending. "
        "Work has been steady this week."
    )
    as_of = date(2026, 9, 4)  # Friday
    statements = [
        _stmt(
            "went out to movie with my friend and his dad and had drinks",
            date(2026, 9, 5),  # Saturday
        ),
    ]
    revised, removed = remove_completed_plan_claims(narrative, statements, as_of=as_of)

    body = revised.split("[CAUTION")[0]
    assert "pending" not in body
    assert "hanging out with a friend Saturday" not in body
    assert "Work has been steady this week." in body
    assert len(removed) == 1
    assert "pending" in removed[0]
    assert "[CAUTION:" in revised
    assert "2026-09-05" in revised
    assert "went out to movie" in revised


# --- (ii) unrelated past-tense statement -> untouched ---------------------

def test_unrelated_past_tense_statement_leaves_narrative_untouched():
    narrative = "He is going to finish the garage cleanup this weekend."
    as_of = date(2026, 9, 4)
    statements = [
        _stmt("I finished watching a documentary about volcanoes last night.", date(2026, 9, 5)),
    ]
    revised, removed = remove_completed_plan_claims(narrative, statements, as_of=as_of)
    assert revised == narrative
    assert removed == []


# --- (iii) digit-conflict guard -------------------------------------------

def test_digit_conflict_blocks_resolution():
    narrative = "He plans to finish HW6 by Friday."
    as_of = date(2026, 9, 4)
    statements = [
        _stmt("finished hw7 last night, turned it in early", date(2026, 9, 5)),
    ]
    revised, removed = remove_completed_plan_claims(narrative, statements, as_of=as_of)
    assert revised == narrative
    assert removed == []


# --- (iv) statement earlier than the plan's own date -> untouched --------

def test_statement_earlier_than_plan_date_untouched():
    narrative = "Hanging out with a friend Saturday is pending."
    as_of = date(2026, 9, 5)  # narrative generated Saturday
    statements = [
        # Reports something superficially similar but dated BEFORE as_of.
        _stmt("went out with my friend last weekend and had a great time", date(2026, 9, 1)),
    ]
    revised, removed = remove_completed_plan_claims(narrative, statements, as_of=as_of)
    assert revised == narrative
    assert removed == []


# --- (v) idempotent --------------------------------------------------------

def test_remove_completed_plan_claims_is_idempotent():
    narrative = (
        "Sent weekend hangout invites; hanging out with a friend Saturday is pending. "
        "Work has been steady this week."
    )
    as_of = date(2026, 9, 4)
    statements = [
        _stmt("went out to movie with my friend and his dad and had drinks", date(2026, 9, 5)),
    ]
    once, removed_once = remove_completed_plan_claims(narrative, statements, as_of=as_of)
    twice, removed_twice = remove_completed_plan_claims(once, statements, as_of=as_of)
    assert once == twice
    assert removed_once
    assert removed_twice == []


# --- plan_sentences ---------------------------------------------------

def test_plan_sentences_extracts_forward_looking_cues():
    narrative = (
        "Hanging out with a friend Saturday is pending. "
        "The user completed their taxes last week. "
        "He hasn't scheduled the dentist appointment yet."
    )
    sentences = plan_sentences(narrative)
    assert any("pending" in s for s in sentences)
    assert any("hasn't" in s.lower() for s in sentences)
    assert not any("completed their taxes" in s for s in sentences)


def test_plan_sentences_empty_narrative():
    assert plan_sentences("") == []
    assert plan_sentences(None) == []


# --- completed_by_user direct unit coverage -------------------------------

def test_completed_by_user_returns_none_without_statements():
    assert completed_by_user("He is going to call the dentist.", []) is None


def test_completed_by_user_requires_completion_cue():
    plan = "He is going to call the dentist about his appointment."
    as_of = date(2026, 9, 4)
    statements = [_stmt("thinking about calling the dentist appointment tomorrow", date(2026, 9, 5))]
    # No completion cue present -> should not resolve.
    assert completed_by_user(plan, statements, as_of=as_of) is None


def test_completed_by_user_requires_two_token_overlap():
    plan = "He is going to call the dentist about his appointment."
    as_of = date(2026, 9, 4)
    # Completion cue present but overlaps on only "the" (stopword, filtered).
    statements = [_stmt("finished the report", date(2026, 9, 5))]
    assert completed_by_user(plan, statements, as_of=as_of) is None


# --- wiring: streak check runs before the completed-plan check -----------

def test_narrative_generator_calls_completed_plan_check_after_streak_check():
    import inspect

    import memory.memory_consolidator as consolidator_mod

    source = inspect.getsource(consolidator_mod)
    streak_idx = source.index("remove_stale_streak_claims(narrative, streak_claims, _today)")
    plan_idx = source.index("remove_completed_plan_claims(\n")
    assert streak_idx < plan_idx, (
        "remove_completed_plan_claims must be wired in AFTER remove_stale_streak_claims "
        "in generate_narrative_context's post-check chain"
    )
    assert "from utils.completed_plan_claims import remove_completed_plan_claims" in source


def test_narrative_generator_wiring_call_order_via_monkeypatch(monkeypatch):
    """Belt-and-suspenders: actually invoke generate_narrative_context with
    stubbed dependencies and assert the streak check is called before the
    completed-plan check."""
    import memory.memory_consolidator as consolidator_mod

    call_order = []

    def fake_remove_stale_streak_claims(narrative, claims, as_of):
        call_order.append("streak")
        return narrative, []

    def fake_remove_completed_plan_claims(narrative, statements, as_of):
        call_order.append("plan")
        return narrative, []

    def fake_remove_conflicting_claims(narrative, facts):
        return narrative, []

    def fake_streak_ledger(statements, as_of=None):
        return []

    def fake_streak_ledger_block(claims, as_of):
        return ""

    def fake_authoritative_facts_block(facts):
        return ""

    monkeypatch.setattr(consolidator_mod, "remove_stale_streak_claims", fake_remove_stale_streak_claims)
    monkeypatch.setattr(consolidator_mod, "remove_completed_plan_claims", fake_remove_completed_plan_claims)
    monkeypatch.setattr(consolidator_mod, "remove_conflicting_claims", fake_remove_conflicting_claims)
    monkeypatch.setattr(consolidator_mod, "streak_ledger", fake_streak_ledger)
    monkeypatch.setattr(consolidator_mod, "streak_ledger_block", fake_streak_ledger_block)
    monkeypatch.setattr(consolidator_mod, "authoritative_facts_block", fake_authoritative_facts_block)

    # Build a minimal instance without running the real __init__ dependency
    # wiring (which needs a live chroma store / model manager).
    cls = None
    for name, obj in vars(consolidator_mod).items():
        if hasattr(obj, "generate_narrative_context") and isinstance(obj, type):
            cls = obj
            break
    if cls is None:
        pytest.skip("No class with generate_narrative_context found in memory_consolidator")

    instance = cls.__new__(cls)
    instance._read_obsidian_monthly_summaries = lambda limit: []
    instance._read_obsidian_weekly_summaries = lambda limit: []
    instance._read_obsidian_daily_notes = lambda limit: []
    instance._current_status_facts = lambda: []

    class _FakeModelManager:
        async def generate_once(self, *a, **k):
            return "He is going to call the dentist. Work has been steady."

    instance.model_manager = _FakeModelManager()
    instance.NARRATIVE_SYNTHESIS_PROMPT = "{today}{monthly_summaries}{weekly_summaries}{daily_notes}{corpus_summaries}{authoritative_status_facts}{streak_ledger}"

    import asyncio
    result = asyncio.run(instance.generate_narrative_context(
        recent_weeklies=[{"content": "x"}], user_statements=[],
    ))
    assert result
    assert call_order == ["streak", "plan"]


def test_completion_reported_yesterday_removes_plan_under_horizon_floor():
    """Fable referee (2026-09-06): the narrative's user_statements are the
    last 60 corpus entries (count-windowed), so a "today" floor skipped a
    plan the user reported done YESTERDAY. The caller now passes its own
    two-week horizon as the floor."""
    from datetime import date, timedelta
    from utils.completed_plan_claims import remove_completed_plan_claims
    today = date(2026, 9, 6)
    narrative = "Social reconnection: hanging out with a friend on Saturday is pending."
    statements = [{"user_text": "went out to a movie with my friend and his dad and had drinks",
                   "timestamp": "2026-09-05T23:10:00"}]
    revised, removed = remove_completed_plan_claims(
        narrative, statements, as_of=today - timedelta(days=14))
    assert removed and "pending" not in revised.split("[CAUTION")[0]
    assert "2026-09-05" in revised


def test_later_replan_cancels_completion():
    """Newest statement wins: after reporting a plan done, the user made the
    same plan again — the narrative's pending sentence is then correct."""
    from datetime import date, timedelta
    from utils.completed_plan_claims import remove_completed_plan_claims
    today = date(2026, 9, 6)
    narrative = "Plans to hang out with a friend on Saturday are pending."
    statements = [
        {"user_text": "hung out with my friend on Saturday, was great", "timestamp": "2026-08-30T20:00:00"},
        {"user_text": "planning to hang out with my friend again on Saturday", "timestamp": "2026-09-04T18:00:00"},
    ]
    revised, removed = remove_completed_plan_claims(
        narrative, statements, as_of=today - timedelta(days=14))
    assert removed == [] and revised == narrative
