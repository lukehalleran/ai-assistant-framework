"""Structural phase fallback for event_phased specs with no planner phases
(2026-09-06 live: the planner returned zero phases; the whole pattern channel
died with "requires at least 2 phase(s)"). All cases drive the deployed
validate_and_freeze / structural_phases."""

from datetime import date, datetime, timedelta

from core.insight.deliberation import structural_phases, validate_and_freeze

NOW = datetime(2026, 9, 6, 15, 0)


def _proposal(phases=None, anchor=None, kind="event_phased"):
    p = {
        "analysis_kind": kind,
        "claims": [{"claim_id": "A", "proposition": "X differed", "authority": "assessment",
                    "required_channels": ["pattern", "corpus"], "evidence_standard": "dated"}],
        "outcome_terms": ["energy"], "series_terms": {}, "phases": phases or [],
        "requested_channels": ["pattern", "corpus"], "research_queries": {},
    }
    if anchor:
        p["anchor"] = anchor
    return p


def test_zero_phases_event_phased_becomes_ready_with_structural_split():
    r = validate_and_freeze(_proposal(), query="does my history support rest days", now=NOW)
    assert r.status == "ready" and r.spec is not None
    assert len(r.spec.phases) == 2
    assert any("derived structurally" in n for n in r.limitations)
    labels = [p.label for p in r.spec.phases]
    assert labels == ["earlier period", "recent period"]
    assert r.spec.phases[1].end == NOW.date().isoformat()


def test_without_query_prior_insufficient_behavior_is_unchanged():
    r = validate_and_freeze(_proposal(), now=NOW)
    assert r.status == "insufficient"
    assert any("at least 2 phase" in n for n in r.limitations)


def test_planner_phases_are_never_overridden():
    phases = [{"label": "a", "start": "2026-08-01", "end": "2026-08-15"},
              {"label": "b", "start": "2026-08-16", "end": "2026-09-05"}]
    r = validate_and_freeze(_proposal(phases), query="anything", now=NOW)
    assert r.status == "ready" and [p.label for p in r.spec.phases] == ["a", "b"]
    assert not any("derived structurally" in n for n in r.limitations)


def test_named_window_drives_the_split():
    ph = structural_phases("over the last 8 weeks", now=NOW)
    start = date.fromisoformat(ph[0]["start"])
    assert (NOW.date() - start).days == 56
    recent = date.fromisoformat(ph[1]["start"])
    assert (NOW.date() - recent).days == max(3, 56 // 4)


def test_explicit_iso_window_splits_in_half():
    ph = structural_phases("between 2026-08-01 and 2026-08-31", now=NOW)
    assert ph[0]["start"] == "2026-08-01" and ph[1]["end"] == "2026-08-31"
    assert date.fromisoformat(ph[1]["start"]) == date.fromisoformat(ph[0]["end"]) + timedelta(days=1)


def test_anchor_date_gives_before_after():
    ph = structural_phases("since I stopped", now=NOW, anchor={"label": "stopping", "date": "2026-07-15"})
    assert ph[0]["label"] == "before stopping" and ph[0]["end"] == "2026-07-14"
    assert ph[1]["label"] == "after stopping" and ph[1]["start"] == "2026-07-15"
    assert ph[1]["end"] == NOW.date().isoformat()


def test_anchor_sentinel_date_falls_back_to_window():
    r = validate_and_freeze(_proposal(anchor={"label": "x", "date": "null"}),
                            query="my rest days lately", now=NOW)
    assert r.status == "ready" and [p.label for p in r.spec.phases] == ["earlier period", "recent period"]


def test_time_series_kind_is_untouched():
    r = validate_and_freeze(_proposal(kind="time_series"), query="how often", now=NOW)
    # time_series needs 1 phase; zero → still insufficient, no structural split.
    assert r.status == "insufficient"
