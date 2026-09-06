"""Regression tests for the 2026-09-06 phase-bounds sentinel fix.

Root cause (verified against the deployed code): the planner prompt showed
the literal word "null" as an example end-date value, and a quoted JSON
string "null" is truthy where a real JSON null is not. A phase like
``{"label": "post-rest period", "start": "2026-09-06", "end": "null"}``
therefore looked like an EXPLICIT bound to ``_resolve_phases``, which then
tried to parse the literal string "null" as a date, failed, and discarded
the entire comparison (``phases=None`` -> ``scan=None`` -> zero computed
aggregates for the whole spec) even though only one phase's end date was
missing.

These tests call the deployed ``validate_and_freeze`` and ``_resolve_phases``
directly. No expected values are copied from the incident transcript beyond
the generic sentinel-word shape described in the fix.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from core.insight.coordinator import LongitudinalDeliberationCoordinator, _resolve_phases
from core.insight.deliberation import _PLANNER_PROMPT, validate_and_freeze
from memory.pattern_engine import EvidencePhase, LongitudinalEvidenceSpec


def _proposal(phases, *, analysis_kind="event_phased", channels=None, anchor=None):
    return {
        "analysis_kind": analysis_kind,
        "claims": [
            {"claim_id": "A", "proposition": "Engagement differs between phases",
             "dependencies": [], "authority": "assessment"},
        ],
        "outcome_terms": ["engagement"],
        "phases": phases,
        "anchor": anchor,
        "requested_channels": channels or ["pattern", "corpus"],
    }


def _phase_spec(phases) -> LongitudinalEvidenceSpec:
    return LongitudinalEvidenceSpec(claims=[], outcome_terms=["engagement"], phases=phases)


# --- Prompt wording (fix a) -------------------------------------------------

def test_planner_prompt_no_longer_invites_the_literal_word_null():
    assert '"ISO date or null"' not in _PLANNER_PROMPT
    assert "YYYY-MM-DD" in _PLANNER_PROMPT


# --- Freeze-time sentinel normalization (fix a) -----------------------------

@pytest.mark.parametrize("sentinel", ["null", "NULL", "None", "present", "TBD", "ongoing", ""])
def test_freeze_normalizes_end_sentinels_to_none(sentinel):
    proposal = _proposal([
        {"label": "before", "start": "2026-01-01", "end": "2026-01-31"},
        {"label": "after", "start": "2026-02-01", "end": sentinel},
    ])
    result = validate_and_freeze(proposal)
    assert result.status == "ready", result.limitations
    after = next(p for p in result.spec.phases if p.label == "after")
    assert after.end is None


def test_freeze_normalizes_anchor_date_sentinel_to_none():
    proposal = _proposal(
        [{"label": "only", "start": "2026-01-01", "end": "2026-01-31"}],
        analysis_kind="time_series",
        anchor={"label": "kickoff", "date": "null", "search_terms": ["kickoff"]},
    )
    result = validate_and_freeze(proposal)
    assert result.status == "ready", result.limitations
    assert result.spec.anchor.date is None


def test_freeze_preserves_a_real_explicit_date():
    proposal = _proposal([
        {"label": "before", "start": "2026-01-01", "end": "2026-01-31"},
        {"label": "after", "start": "2026-02-01", "end": "2026-02-28"},
    ])
    result = validate_and_freeze(proposal)
    assert result.status == "ready", result.limitations
    after = next(p for p in result.spec.phases if p.label == "after")
    assert after.end == "2026-02-28"


# --- _resolve_phases bound handling (fix b) ---------------------------------

def test_open_ended_phase_is_coerced_to_now_with_a_note():
    now = datetime(2026, 9, 6)
    spec = _phase_spec([
        EvidencePhase(label="before", start="2026-08-01", end="2026-08-15"),
        EvidencePhase(label="after", start="2026-08-16", end=None),
    ])
    phases, _candidates, notes = _resolve_phases(spec, events=[], now=now)
    assert phases is not None
    after = next(p for p in phases if p.label == "after")
    assert after.end == "2026-09-06"
    assert any("after" in note and "open end coerced to 2026-09-06" in note for note in notes)


def test_invalid_start_drops_only_that_phase():
    now = datetime(2026, 9, 6)
    spec = _phase_spec([
        EvidencePhase(label="before", start="not-a-date", end="2026-08-15"),
        EvidencePhase(label="after", start="2026-08-16", end="2026-08-31"),
        EvidencePhase(label="later", start="2026-09-01", end="2026-09-05"),
    ])
    phases, _candidates, notes = _resolve_phases(spec, events=[], now=now)
    assert phases is not None
    assert [p.label for p in phases] == ["after", "later"]
    assert any("before" in note and "dropped: invalid bounds" in note for note in notes)


def test_fewer_than_two_valid_phases_yields_insufficient():
    now = datetime(2026, 9, 6)
    spec = _phase_spec([
        EvidencePhase(label="before", start="garbage", end="2026-08-15"),
        EvidencePhase(label="after", start="2026-08-16", end="2026-08-31"),
    ])
    phases, _candidates, notes = _resolve_phases(spec, events=[], now=now)
    assert phases is None
    assert any("fewer than 2 valid phases" in note for note in notes)


def test_valid_explicit_pair_is_unchanged():
    spec = _phase_spec([
        EvidencePhase(label="before", start="2026-01-01", end="2026-01-31"),
        EvidencePhase(label="after", start="2026-02-01", end="2026-02-28"),
    ])
    phases, _candidates, notes = _resolve_phases(spec, events=[], now=datetime(2026, 9, 6))
    assert phases is not None
    assert [(p.start, p.end) for p in phases] == [
        ("2026-01-01", "2026-01-31"), ("2026-02-01", "2026-02-28"),
    ]
    assert notes == ["planner supplied explicit phase bounds"]


def test_unresolved_anchor_case_still_returns_none_untouched():
    """Sanity pin: the anchor-relative branch (no explicit starts at all) is
    a DIFFERENT code path from the open-end coercion above and must still
    fail closed when the anchor cannot be located — this is not the
    open-ended case the fix addresses."""
    spec = LongitudinalEvidenceSpec(
        claims=[], outcome_terms=["engagement"],
        phases=[
            EvidencePhase(label="before", metadata={"start_offset_days": -30, "end_offset_days": -1}),
            EvidencePhase(label="after", metadata={"start_offset_days": 0, "end_offset_days": 30}),
        ],
        anchor=None,
    )
    phases, _candidates, notes = _resolve_phases(spec, events=[], now=datetime(2026, 9, 6))
    assert phases is None
    assert "no temporal anchor was specified" in notes


# --- End-to-end: freeze -> resolve, and manifest surfacing (fix c) ---------

def test_end_to_end_null_sentinel_resolves_instead_of_killing_the_whole_spec():
    now = datetime(2026, 9, 6)
    proposal = _proposal([
        {"label": "before", "start": "2026-08-01", "end": "2026-08-31"},
        {"label": "after", "start": "2026-09-01", "end": "null"},
    ])
    frozen = validate_and_freeze(proposal)
    assert frozen.status == "ready", frozen.limitations
    phases, _candidates, notes = _resolve_phases(frozen.spec, events=[], now=now)
    assert phases is not None
    after = next(p for p in phases if p.label == "after")
    assert after.end == now.date().isoformat()
    assert any("open end coerced" in note for note in notes)


class _Corpus:
    corpus: list = []


@pytest.mark.asyncio
async def test_coordinator_run_surfaces_the_coercion_note_and_completes_a_scan():
    proposal = _proposal([
        {"label": "before", "start": "2026-08-01", "end": "2026-08-31"},
        {"label": "after", "start": "2026-09-01", "end": "null"},
    ])
    coordinator = LongitudinalDeliberationCoordinator(
        corpus_manager=_Corpus(), now=datetime(2026, 9, 6),
    )
    result = await coordinator.run("compare my engagement before and after", proposal)
    assert result.scan is not None
    assert any("open end coerced" in note for note in result.manifest["limitations"])
