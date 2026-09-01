from datetime import datetime
import json

import pytest

from core.insight.coordinator import (
    LongitudinalDeliberationCoordinator,
    assess_claim_chain,
    _event_from_row,
    normalize_chroma_rows,
)
from core.insight.deliberation import compact_recovery_plan, plan_deliberation, validate_and_freeze
from memory.pattern_engine import (
    DeliberationClaimSpec,
    EvidencePhase,
    LongitudinalEvent,
    LongitudinalEvidenceSpec,
    run_longitudinal_scan,
)
from knowledge.pubmed_search import PubMedRows


def test_generated_daily_note_is_assistant_summary_not_user_observation():
    event = _event_from_row({
        "id": "daily-1", "timestamp": "2026-07-14",
        "title": "7 14 26 Daily Note",
        "text": "Luke seemed agitated today.",
        "metadata": {"author": "daemon", "source_type": "daemon_daily_summary",
                     "derived_from": "conversation_corpus"},
    }, "notes")
    assert event is not None
    assert event.source_class == "assistant-summary"
    assert event.speaker == "assistant"
    assert event.evidence_class == "assistant_summary"


def test_corpus_excludes_prior_insight_analysis_artifacts():
    class Corpus:
        corpus = [
            {"query": "Analyze my sleep pattern", "timestamp": "2026-01-01", "response_mode": "insight-assembly"},
            {"query": "I slept badly last night", "timestamp": "2026-01-02", "response_mode": "chat"},
        ]
    from core.insight.coordinator import _corpus_events
    events = _corpus_events(Corpus())
    assert [event.text for event in events] == ["I slept badly last night"]


def test_note_content_date_survives_normalization_and_index_date_is_marked():
    rows = normalize_chroma_rows([{
        "id": "note-1", "content": "study session", "metadata": {
            "timestamp": "2026-08-31T12:00:00", "note_date": "2026-06-04",
        },
    }], channel="notes")
    assert rows[0]["content_date"] == "2026-06-04"
    event = _event_from_row(rows[0], "notes")
    assert event.timestamp.startswith("2026-06-04")
    assert event.provenance["date_basis"] == "content"

    legacy = _event_from_row({
        "id": "legacy", "timestamp": "2026-08-31T12:00:00",
        "text": "retrospective study summary", "date_basis": "index-time",
        "metadata": {"source_type": "daemon_daily_summary"},
    }, "notes")
    assert legacy is not None
    assert legacy.provenance["date_basis"] == "index-time"


def _spec(
    *, outcome="sleep quality", phases=None, channels=None,
    research_queries=None, anchor=None, analysis_kind="period_comparison",
):
    return {
        "analysis_kind": analysis_kind,
        "claims": [
            {"claim_id": "A", "proposition": f"The record differs in {outcome} between phases", "dependencies": [], "authority": "assessment"},
            {"claim_id": "B", "proposition": "An outside source describes a relevant mechanism", "dependencies": [], "authority": "assessment"},
            {"claim_id": "C", "proposition": "A professional should prescribe a particular treatment", "dependencies": ["A", "B"], "authority": "outside_authority"},
        ],
        "outcome_terms": [outcome],
        "behavioral_indicators": ["slept through the night"],
        "directional_indicators": {
            "increase": [f"more {outcome}"],
            "decrease": [f"less {outcome}"],
        },
        "phases": phases or [
            {"label": "before", "start": "2026-01-01", "end": "2026-01-31"},
            {"label": "after", "start": "2026-02-01", "end": "2026-02-28"},
        ],
        "anchor": anchor,
        "admissible_source_classes": ["user", "users-own-note"],
        "rival_explanations": ["schedule changed"],
        "supporting_facets": [f"reports of {outcome}"],
        "refuting_facets": [f"counterexamples for {outcome}"],
        "requested_channels": channels or ["pattern", "corpus"],
        "research_queries": research_queries or {},
        "assumptions": ["phase windows are descriptive"],
    }


class Corpus:
    corpus = [
        {"timestamp": "2026-01-15T10:00:00", "query": "I had less sleep quality last night", "id": "u-before"},
        {"timestamp": "2026-01-16T10:00:00", "query": "ordinary observation without the outcome", "id": "u-denominator"},
        {"timestamp": "2026-02-15T10:00:00", "query": "I had more sleep quality and slept through the night", "id": "u-after"},
    ]


@pytest.mark.asyncio
async def test_compact_recovery_plan_is_generic_and_validates_source_queries():
    class Planner:
        async def generate_once(self, *_args, **_kwargs):
            return json.dumps({
                "analysis_kind": "period_comparison",
                "outcome_terms": ["study hours", "exercise frequency"],
                "series_terms": {"study": ["studied"], "exercise": ["worked out"]},
                "behavioral_indicators": ["hours studied", "workout"],
                "phases": [
                    {"label": "before", "start": "2026-06-01", "end": "2026-06-30"},
                    {"label": "after", "start": "2026-07-01", "end": "2026-07-31"},
                ],
                "requested_channels": ["pattern", "corpus", "notes", "web"],
                "research_queries": {"web": "study hours exercise frequency research"},
                "relation": "co-occurrence",
            })

    result = await compact_recovery_plan(
        "Analyze whether my study and exercise activity co-occur", model_manager=Planner(),
        available_channels=["pattern", "corpus", "notes", "web"],
    )
    assert result.status == "ready"
    assert result.planner_provenance == "compact-recovery"
    assert result.spec.outcome_terms == ["study hours", "exercise frequency"]
    assert result.spec.series_terms == {"study": ["studied"], "exercise": ["worked out"]}
    assert result.spec.phases[0].start == "2026-06-01"


@pytest.mark.asyncio
async def test_compact_recovery_plan_fails_closed_without_outcomes_or_phases():
    class Planner:
        async def generate_once(self, *_args, **_kwargs):
            return json.dumps({"analysis_kind": "time_series", "outcome_terms": []})

    result = await compact_recovery_plan("Analyze my life", model_manager=Planner())
    assert result.status == "insufficient"
    assert result.spec is None


@pytest.mark.asyncio
async def test_compact_cooccurrence_recovers_weekly_phases_from_explicit_window():
    class Planner:
        async def generate_once(self, *_args, **_kwargs):
            return json.dumps({
                "analysis_kind": "co_occurrence",
                "series_terms": {"social": ["visited friend"], "household": ["did laundry"]},
                "phases": [],
                "requested_channels": ["pattern", "corpus", "notes"],
            })

    result = await compact_recovery_plan(
        "Examine whether social contact and household maintenance co-occur from 2025-01-01 through 2025-03-31",
        model_manager=Planner(), available_channels=["pattern", "corpus", "notes"],
    )
    assert result.status == "ready"
    assert len(result.spec.phases) == 13
    assert result.spec.phases[0].label == "week-2025-01-01"


@pytest.mark.asyncio
async def test_generic_coordinator_merges_requested_channels_and_exact_phases():
    proposed = _spec(
        channels=["pattern", "corpus", "notes", "pubmed", "web", "wiki"],
        research_queries={
            "pubmed": "sleep continuity intervention trial",
            "web": "sleep continuity mechanism review",
            "wiki": "sleep continuity",
        },
    )
    coordinator = LongitudinalDeliberationCoordinator(
        corpus_manager=Corpus(),
        adapters={
            "notes": lambda q: [{"timestamp": "2026-02-16", "text": "slept through the night", "id": "note-1"}],
            "pubmed": lambda q: [{"pmid": "1", "title": "Trial", "abstract": "Direct abstract", "url": "https://pubmed.example/1"}],
            "web": lambda q: [{"id": "w1", "title": "Review", "snippet": "Web evidence", "url": "https://example.test/review"}],
            "wiki": lambda q: [{"id": "k1", "title": "Sleep", "text": "Reference evidence"}],
        },
        now=datetime(2026, 3, 1),
    )
    result = await coordinator.run("compare my sleep before and after", proposed)

    assert result.scan is not None
    assert [row.phase.label for row in result.scan.comparisons] == ["before", "after"]
    assert [row.event_count for row in result.scan.comparisons] == [1, 2]
    assert result.scan.comparisons[0].observation_denominator == 2
    statuses = {row.channel: row.status for row in result.channels}
    assert set(statuses.items()) >= set({
        "corpus": "succeeded", "notes": "succeeded", "pubmed": "succeeded",
        "web": "succeeded", "wiki": "succeeded", "pattern": "succeeded",
    }.items())
    assert {row["source_class"] for row in result.external_evidence} == {"pubmed", "web", "wiki"}
    assert result.manifest["phase_summary"][0]["decreased"] == 1
    assert result.manifest["phase_summary"][1]["increased"] == 1
    repro = result.manifest["reproducibility"]
    assert len(repro["evidence_fingerprint"]) == 64
    assert len(repro["spec_fingerprint"]) == 64
    assert len(repro["external_fingerprint"]) == 64
    assert len(repro["counted_output_fingerprint"]) == 64
    assert repro["status"] == "captured"
    assert repro["event_input_count"] == len(result.internal_events)
    assert result.claim_chain[-1].status == "outside_authority"


@pytest.mark.asyncio
async def test_unresolved_anchor_does_not_discard_successful_external_research():
    proposed = _spec(
        outcome="focus", analysis_kind="event_phased",
        phases=[
            {"label": "before", "metadata": {"start_offset_days": -30, "end_offset_days": -1}},
            {"label": "after", "metadata": {"start_offset_days": 0, "end_offset_days": 30}},
        ],
        anchor={"label": "the move", "search_terms": ["moved apartments"], "uncertainty_days": 3},
        channels=["pattern", "corpus", "pubmed"],
        research_queries={"pubmed": "attention environmental change"},
    )
    result = await LongitudinalDeliberationCoordinator(
        corpus_manager=Corpus(),
        adapters={"pubmed": lambda q: [{"pmid": "9", "abstract": "Research"}]},
    ).run("has my focus changed since I moved; use PubMed", proposed)

    assert result.scan is None
    assert result.external_evidence[0]["source_id"] == "pmid:9"
    statuses = {row.channel: row.status for row in result.channels}
    assert statuses["pubmed"] == "succeeded"
    assert statuses["pattern"] == "insufficient"
    assert "no dated user-authored event" in " ".join(result.manifest["limitations"])


@pytest.mark.asyncio
async def test_adapter_failure_is_channel_specific_and_other_channels_survive():
    def broken(_query):
        raise RuntimeError("source offline")

    proposed = _spec(
        outcome="work output", channels=["pattern", "corpus", "web", "wiki"],
        research_queries={"web": "work output shift schedules", "wiki": "shift work"},
    )
    result = await LongitudinalDeliberationCoordinator(
        corpus_manager=Corpus(),
        adapters={"web": broken, "wiki": lambda q: [{"id": "wiki-1", "text": "Shift work"}]},
    ).run("compare my work output and use web and wiki", proposed)
    statuses = {row.channel: row for row in result.channels}
    assert statuses["web"].status == "failed"
    assert "source offline" in statuses["web"].reason
    assert statuses["wiki"].status == "succeeded"


@pytest.mark.asyncio
async def test_planner_failure_is_fail_closed_not_keyword_guessing():
    result = await LongitudinalDeliberationCoordinator(
        corpus_manager=Corpus(), model_manager=None,
    ).run("what tends to happen when I skip breakfast?")
    assert result.freeze.status == "insufficient"
    assert result.scan is None
    assert result.internal_events == []
    assert "not guessed" in " ".join(result.freeze.limitations)


@pytest.mark.asyncio
async def test_pubmed_recovery_uses_ladder_and_preserves_no_relevant_status():
    seen = []

    async def pubmed(query):
        seen.append(query)
        return PubMedRows(status="no_relevant_results")

    result = await LongitudinalDeliberationCoordinator(
        corpus_manager=Corpus(), adapters={"pubmed": pubmed},
    ).run(
        "Therapist says cariprazine may help irritability in autism; use PubMed",
    )
    assert result.freeze.status == "insufficient"
    assert len(seen) >= 2
    assert all("therapist" not in query.lower() for query in seen)
    pubmed_status = next(row for row in result.channels if row.channel == "pubmed")
    assert pubmed_status.status == "no_relevant_results"


def test_validator_rejects_bad_claim_graph_and_missing_source_query():
    proposed = _spec(channels=["pattern", "corpus", "pubmed"], research_queries={})
    proposed["claims"][0]["dependencies"] = ["B"]
    proposed["claims"][1]["dependencies"] = ["A"]
    result = validate_and_freeze(proposed)
    assert result.status == "insufficient"
    joined = " ".join(result.limitations)
    assert "cycle" in joined
    assert "pubmed" in joined and "source-specific query" in joined


@pytest.mark.asyncio
async def test_planner_supports_non_medication_time_series_shape():
    proposal = _spec(
        outcome="exercise frequency", analysis_kind="time_series",
        phases=[{"label": "observation", "start": "2026-01-01", "end": "2026-03-01"}],
    )

    class Planner:
        async def generate_once(self, *_args, **_kwargs):
            return json.dumps(proposal)

    frozen = await plan_deliberation(
        "Across my history, is exercise frequency trending over time?",
        model_manager=Planner(),
    )
    assert frozen.status == "ready"
    assert frozen.spec.analysis_kind == "time_series"
    assert frozen.spec.outcome_terms == ["exercise frequency"]


@pytest.mark.asyncio
async def test_claim_assessor_requires_real_citations_and_honors_dependencies():
    spec = LongitudinalEvidenceSpec(
        claims=[
            DeliberationClaimSpec(claim_id="A", proposition="A historical change occurred"),
            DeliberationClaimSpec(claim_id="B", proposition="A mechanism explains it", dependencies=["A"]),
        ],
        outcome_terms=["focus"],
        phases=[EvidencePhase(label="before", start="2026-01-01", end="2026-01-31")],
    )
    scan = run_longitudinal_scan(spec, [
        LongitudinalEvent(timestamp="2026-01-10", text="focus", source_id="u1"),
    ])

    class Assessor:
        async def generate_once(self, *_args, **_kwargs):
            return json.dumps({"claims": [
                {"claim_id": "A", "status": "supported", "confidence": 0.9, "support_source_ids": ["invented"], "refute_source_ids": []},
                {"claim_id": "B", "status": "supported", "confidence": 0.9, "support_source_ids": ["pmid:1"], "refute_source_ids": []},
            ]})

    claims = await assess_claim_chain(
        spec, scan,
        [{"pmid": "1", "source_id": "pmid:1", "abstract": "mechanism", "source_class": "pubmed"}],
        [], [], model_manager=Assessor(),
    )
    assert claims[0].status == "insufficient"
    assert claims[1].status == "insufficient"


@pytest.mark.asyncio
async def test_claim_required_channel_cannot_be_replaced_by_another_source():
    spec = LongitudinalEvidenceSpec(
        claims=[DeliberationClaimSpec(
            claim_id="A", proposition="Research establishes the mechanism",
            required_channels=["pubmed"],
        )],
        outcome_terms=["focus"],
        phases=[EvidencePhase(label="all", start="2026-01-01", end="2026-01-31")],
    )

    class Assessor:
        async def generate_once(self, *_args, **_kwargs):
            return json.dumps({"claims": [{
                "claim_id": "A", "status": "supported", "confidence": 0.8,
                "support_source_ids": ["web:1"], "refute_source_ids": [],
            }]})

    claims = await assess_claim_chain(
        spec, None,
        [{"source_id": "web:1", "text": "a web claim", "source_class": "web"}],
        [], [], model_manager=Assessor(),
    )
    assert claims[0].status == "insufficient"
    assert "pubmed" in claims[0].rationale


@pytest.mark.asyncio
async def test_assessor_call_disables_reasoning_and_has_recovery_headroom():
    """2026-08-31 live run: the assessor timed out at 35s (kimi-3 reasoning
    channel) and the verdict collapsed to the 0.0-confidence default chain.
    The strict-JSON assessor call must suppress native reasoning, and the
    default timeout must leave headroom because the assessor — unlike the
    planner — has no deterministic fallback."""
    captured = {}

    class Assessor:
        async def generate_once(self, *_args, **kwargs):
            captured.update(kwargs)
            return json.dumps({"claims": [{
                "claim_id": "A", "status": "insufficient", "confidence": 0.1,
                "support_source_ids": [], "refute_source_ids": [],
            }]})

    spec = LongitudinalEvidenceSpec(
        claims=[DeliberationClaimSpec(claim_id="A", proposition="A change occurred")],
        outcome_terms=["sleep"],
        phases=[EvidencePhase(label="off", start="2026-07-13", end="2026-09-11")],
    )
    await assess_claim_chain(spec, None, [], [], [], model_manager=Assessor())
    assert captured.get("disable_reasoning") is True
    assert LongitudinalDeliberationCoordinator().assessor_timeout_s >= 60
