"""Behavioural siblings for the insight-route source-text tests.

Runs the DEPLOYED ``handlers._run_insight_mode`` with the stage functions replaced at
their source modules (the same seam test_insight_mode_handler uses) and asserts what
the route does: the pattern stage runs for a ``pattern_temporal`` intent and its
output reaches the synthesizer, storage is tracked before turn telemetry, and the
prompt-only suffix is handed to storage.

Incident class: BC-63 (source-text test proves spelling only). Lane SRCH, 2026-10-27.
Source tests kept: test_insight_pattern_facet::test_insight_runner_threads_patterns,
test_background_storage_receipts (insight site), test_handlers_navigation_not_stored_as_user_text.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gui.handlers as handlers
from core.insight.types import Assessment, ClaimAssessment, EvidenceItem, FacetPlan, FacetQuery

pytestmark = pytest.mark.asyncio

ITEMS = [EvidenceItem(doc_id="d1", text="slept badly on the sample night", date="2026-08-18",
                      collection="conversations", speaker="user")]
SUFFIX = "\n\n[ATTACHMENT NOTE] sample note"


def _ctx(intent_dict):
    orchestrator = MagicMock()
    orchestrator.model_manager.get_active_model_name = MagicMock(return_value="test-model")
    ctx = handlers.SubmitContext(
        user_text=intent_dict["raw_query"], files=None, history=[], use_raw_gpt=False,
        orchestrator=orchestrator, personality=None, fast_mode=False,
        conversation_logger=None, file_names=[], merged_input="q" + SUFFIX,
        files_result=None, prompt_only_suffix=SUFFIX,
    )
    ctx.raw_context = {}
    ctx.gate_decision = MagicMock()
    ctx.gate_decision.insight_intent = intent_dict
    return ctx


@pytest.fixture
def route(monkeypatch):
    import core.insight.assessor as assessor_mod
    import core.insight.facets as facets_mod
    import core.insight.sweep as sweep_mod
    import core.insight.synthesizer as synth_mod

    rec = SimpleNamespace(events=[], storage=[], synth={}, stage_calls=[])
    plan = FacetPlan(facets=[FacetQuery(name="f", query_text="q")], claims=["c"])
    monkeypatch.setattr(facets_mod, "decompose", AsyncMock(return_value=plan))
    monkeypatch.setattr(sweep_mod, "run_sweep", AsyncMock(return_value=list(ITEMS)))
    monkeypatch.setattr(assessor_mod, "assess", AsyncMock(return_value=Assessment(
        overall="agree", claims=[ClaimAssessment(claim="c", verdict="agree", support=["E1"])])))

    async def synthesize_stream(intent, evidence, assessment, **kw):
        rec.synth.update(kw)
        yield "The record shows you slept badly (2026-08-18)."

    monkeypatch.setattr(synth_mod, "synthesize_stream", synthesize_stream)

    def _dispatch(*args, **kwargs):
        rec.storage.append((args, kwargs))
        rec.events.append("dispatch_storage")

    monkeypatch.setattr(handlers, "_dispatch_storage", _dispatch)
    monkeypatch.setattr(handlers, "_track_storage_task",
                        lambda ctx, value: rec.events.append("track_storage"))
    monkeypatch.setattr(handlers, "_write_turn_telemetry",
                        lambda *a, **k: rec.events.append("write_telemetry"))
    return rec


async def _run(ctx):
    return [c async for c in handlers._run_insight_mode(ctx)]


async def test_insight_site_tracks_storage_before_turn_telemetry_and_passes_the_suffix(route):
    ctx = _ctx({"kind": "theme_sweep", "theme": "sleep", "wants_document": False, "raw_query": "q"})
    chunks = await _run(ctx)
    assert ctx.handled is True and ctx.storage_dispatched is True
    assert route.events == ["dispatch_storage", "track_storage", "write_telemetry"]
    args, kwargs = route.storage[0]
    assert kwargs["prompt_only_suffix"] == SUFFIX
    assert args[1].endswith(SUFFIX)
    assert chunks[-1]["debug"]["mode"] == "insight-assembly"


async def test_pattern_temporal_runs_the_pattern_stage_and_threads_it_to_the_synthesizer(
        route, monkeypatch):
    spec = SimpleNamespace(
        analysis_kind="time_series", outcome_terms=["sleep"], behavioral_indicators=[],
        supporting_facets=["slept badly"], refuting_facets=["slept well"], claims=[], phases=[],
    )
    freeze = SimpleNamespace(status="ready", spec=spec)
    deliberation = SimpleNamespace(freeze=freeze, scan=None, external_evidence=[],
                                   manifest="manifest-sentinel")

    class FakeCoordinator:
        def __init__(self, **kwargs):
            pass

        async def run(self, request):
            return deliberation

    patterns = [SimpleNamespace(name="weekly-counts-sentinel")]

    def fake_stage(intent, **kwargs):
        route.stage_calls.append(kwargs)
        return patterns, []

    monkeypatch.setattr(handlers.coordinator, "LongitudinalDeliberationCoordinator", FakeCoordinator)
    monkeypatch.setattr(handlers.temporal, "run_pattern_stage", fake_stage)

    ctx = _ctx({"kind": "pattern_temporal", "theme": "sleep", "wants_document": False,
                "raw_query": "how has my sleep changed"})
    await _run(ctx)

    assert len(route.stage_calls) == 1
    assert route.stage_calls[0]["spec"] is spec              # the FROZEN contract drives the stage
    assert route.synth["patterns"] is patterns               # its output reaches the synthesizer
    assert route.synth["deliberation_manifest"] == "manifest-sentinel"
    assert ctx.handled is True


async def test_unfrozen_plan_never_turns_prose_into_a_computed_pattern(route, monkeypatch):
    deliberation = SimpleNamespace(freeze=SimpleNamespace(status="failed", spec=None),
                                   scan=None, external_evidence=[], manifest=None)

    class FakeCoordinator:
        def __init__(self, **kwargs):
            pass

        async def run(self, request):
            return deliberation

    def fake_stage(intent, **kwargs):  # pragma: no cover - must not run
        route.stage_calls.append(kwargs)
        return [SimpleNamespace(name="invented")], []

    monkeypatch.setattr(handlers.coordinator, "LongitudinalDeliberationCoordinator", FakeCoordinator)
    monkeypatch.setattr(handlers.temporal, "run_pattern_stage", fake_stage)
    ctx = _ctx({"kind": "pattern_temporal", "theme": "sleep", "wants_document": False,
                "raw_query": "how has my sleep changed"})
    await _run(ctx)
    assert route.stage_calls == []
    assert route.synth["patterns"] == []
