from core.agentic.protocols import XMLMarkerHandler, NativeToolsHandler
from core.agentic.types import SearchDecision
from memory.pattern_engine import (
    EvidencePhase, LongitudinalEvidenceSpec, run_longitudinal_scan,
)


def test_longitudinal_phases_dedup_and_assistant_exclusion():
    spec = LongitudinalEvidenceSpec(
        propositions=["irritability changes after cessation"],
        outcome_terms=["irritable"],
        directional_indicators={
            "increase": ["irritable again"],
            "decrease": ["less irritable"],
        },
        phases=[EvidencePhase(label="stable-on", start="2026-01-01", end="2026-01-31"),
                EvidencePhase(label="later-off", start="2026-02-01", end="2026-02-28")],
    )
    result = run_longitudinal_scan(spec, [
        {"timestamp": "2026-01-10T10:00:00", "text": "less irritable", "source_id": "a"},
        {"timestamp": "2026-01-10T10:00:00", "text": "less irritable", "source_id": "a"},
        {"timestamp": "2026-02-10T10:00:00", "text": "irritable again", "source_id": "b"},
        {"timestamp": "2026-02-11T10:00:00", "text": "irritable", "speaker": "assistant"},
        {"timestamp": "2026-02-12T10:00:00", "text": "fine today"},
    ])
    assert [c.event_count for c in result.comparisons] == [1, 1]
    assert result.exclusion_reasons["duplicate"] == 1
    assert result.exclusion_reasons["assistant_derived"] == 1
    assert "absence of mention" in result.render_manifest()
    assert result.comparisons[0].decreased_count == 1
    assert result.comparisons[1].increased_count == 1


def test_calendar_bounds_are_inclusive_but_adjacent_phases_do_not_overlap():
    spec = LongitudinalEvidenceSpec(
        outcome_terms=["pain"],
        phases=[
            EvidencePhase(label="january", start="2026-01-01", end="2026-01-31"),
            EvidencePhase(label="february", start="2026-02-01", end="2026-02-28"),
        ],
    )
    result = run_longitudinal_scan(spec, [
        {"timestamp": "2026-01-01T00:00:00", "text": "pain", "source_id": "j1"},
        {"timestamp": "2026-01-31T23:59:59", "text": "pain", "source_id": "j2"},
        {"timestamp": "2026-02-01T00:00:00", "text": "pain", "source_id": "f1"},
    ])
    assert [row.event_count for row in result.comparisons] == [2, 1]


def test_denominator_and_coverage_use_all_admissible_observations_not_hits():
    spec = LongitudinalEvidenceSpec(
        outcome_terms=["sleep"],
        phases=[EvidencePhase(label="week", start="2026-01-01", end="2026-01-07")],
    )
    result = run_longitudinal_scan(spec, [
        {"timestamp": "2026-01-01T10:00:00", "text": "sleep was broken", "source_id": "1"},
        {"timestamp": "2026-01-02T10:00:00", "text": "worked all day", "source_id": "2"},
        {"timestamp": "2026-01-02T12:00:00", "text": "ate lunch", "source_id": "3"},
    ])
    row = result.comparisons[0]
    assert row.event_count == 1
    assert row.observation_denominator == 3
    assert row.observed_days == 2
    assert row.coverage == 2 / 7


def test_lineage_dedup_cross_store_and_word_boundaries_prevent_false_hits():
    spec = LongitudinalEvidenceSpec(
        outcome_terms=["pain"],
        admissible_source_classes=["user", "users-own-note"],
        phases=[EvidencePhase(label="all", start="2026-01-01", end="2026-01-31")],
    )
    result = run_longitudinal_scan(spec, [
        {"timestamp": "2026-01-10", "text": "pain today", "source_class": "user", "source_id": "corpus-1", "provenance": {"canonical_source_id": "turn-1"}},
        {"timestamp": "2026-01-10", "text": "pain today", "source_class": "users-own-note", "source_id": "note-1", "provenance": {"derived_from": "turn-1"}},
        {"timestamp": "2026-01-11", "text": "finished a painting", "source_class": "user", "source_id": "corpus-2"},
    ])
    assert result.comparisons[0].event_count == 1
    assert result.exclusion_reasons["duplicate"] == 1


def test_direction_is_frozen_and_domain_neutral_not_good_or_bad():
    spec = LongitudinalEvidenceSpec(
        outcome_terms=["exercise"],
        directional_indicators={
            "increase": ["exercised more"],
            "decrease": ["exercised less"],
        },
        phases=[EvidencePhase(label="month", start="2026-01-01", end="2026-01-31")],
    )
    result = run_longitudinal_scan(spec, [
        {"timestamp": "2026-01-01", "text": "I exercised more", "source_id": "1"},
        {"timestamp": "2026-01-02", "text": "I exercised less", "source_id": "2"},
        {"timestamp": "2026-01-03", "text": "exercise happened", "source_id": "3"},
    ])
    row = result.comparisons[0]
    assert (row.increased_count, row.decreased_count, row.unclear_direction_count) == (1, 1, 1)


def test_frozen_confounders_are_retained_separately_from_outcomes_by_phase():
    spec = LongitudinalEvidenceSpec(
        outcome_terms=["focus"],
        confounder_indicators={
            "sleep disruption": ["slept two hours", "awake all night"],
            "work schedule": ["late shift"],
        },
        phases=[EvidencePhase(label="after", start="2026-01-01", end="2026-01-31")],
    )
    result = run_longitudinal_scan(spec, [
        {"timestamp": "2026-01-01", "text": "focus was low", "source_id": "o1"},
        {"timestamp": "2026-01-02", "text": "I slept two hours after a late shift", "source_id": "c1"},
        {"timestamp": "2026-01-03", "text": "unrelated observation", "source_id": "x1"},
    ])
    row = result.comparisons[0]
    assert row.event_count == 1
    assert row.covariate_counts == {"sleep disruption": 1, "work schedule": 1}
    assert [event.source_id for event in row.covariate_events] == ["c1"]


def test_named_series_count_and_joint_events_are_phase_scoped():
    spec = LongitudinalEvidenceSpec(
        analysis_kind="co_occurrence",
        series_terms={"study": ["studied"], "exercise": ["worked out"]},
        phases=[EvidencePhase(label="week", start="2026-01-01", end="2026-01-07")],
    )
    result = run_longitudinal_scan(spec, [
        {"timestamp": "2026-01-01", "text": "studied", "source_id": "a"},
        {"timestamp": "2026-01-02", "text": "worked out", "source_id": "b"},
        {"timestamp": "2026-01-03", "text": "studied and worked out", "source_id": "c"},
    ])
    row = result.comparisons[0]
    assert row.series_counts == {"study": 2, "exercise": 2}
    assert row.joint_event_count == 1


def test_xml_and_native_pattern_scan_parse():
    spec = '{"outcome_terms":["irritable"],"phases":[]}'
    decision = XMLMarkerHandler().parse_response(f'<pattern_scan spec=\'{spec}\'></pattern_scan>')[0]
    assert decision.wants_pattern_scan and decision.pattern_spec["outcome_terms"] == ["irritable"]
    native = NativeToolsHandler()._parse_single_tool_call({
        "function": {"name": "pattern_scan", "arguments": '{"spec": {"phases": []}}'}
    })
    assert native.wants_pattern_scan
