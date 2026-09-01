"""
Insight-mode handler (_run_insight_mode) — routing, provenance, doc gating,
fall-through contract (2026-08-23).

Stage functions (decompose/run_sweep/assess/synthesize_stream) are patched at
their source modules (the handler lazy-imports them per call), so these tests
exercise handler logic only: turn ownership, storage dispatch + provenance
mode, honest doc withholding, exception fall-through, tone threading.
"""

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

import gui.handlers as handlers
from core.insight.types import (
    Assessment,
    ClaimAssessment,
    EvidenceItem,
    FacetPlan,
    FacetQuery,
)


def _ctx(intent_dict, tone_level=None):
    orchestrator = MagicMock()
    orchestrator.model_manager.get_active_model_name = MagicMock(return_value="kimi-3")
    orchestrator.memory_system.chroma_store = MagicMock()
    ctx = handlers.SubmitContext(
        user_text=intent_dict.get("raw_query", "q"),
        files=None, history=[], use_raw_gpt=False,
        orchestrator=orchestrator, personality=None, fast_mode=False,
        conversation_logger=None, file_names=[], merged_input="q",
        files_result=None,
    )
    ctx.raw_context = {"tone_level": tone_level} if tone_level else {}
    ctx.gate_decision = MagicMock()
    ctx.gate_decision.insight_intent = intent_dict
    return ctx


_SWEEP_ITEMS = [
    EvidenceItem(doc_id="d1", text="she wasn't abusive", date="2026-08-18",
                 collection="conversations", speaker="user"),
    EvidenceItem(doc_id="d2", text="casey | is | evil", date="2026-08-18",
                 collection="facts"),
]


@pytest.fixture
def stages(monkeypatch):
    """Patch the four stage functions; returns the mocks for assertions."""
    import core.insight.assessor as assessor_mod
    import core.insight.facets as facets_mod
    import core.insight.sweep as sweep_mod
    import core.insight.synthesizer as synth_mod

    plan = FacetPlan(facets=[FacetQuery(name="f", query_text="q")], claims=["c"])
    decompose = AsyncMock(return_value=plan)
    run_sweep = AsyncMock(return_value=list(_SWEEP_ITEMS))
    assess = AsyncMock(return_value=Assessment(
        overall="agree",
        claims=[ClaimAssessment(claim="c", verdict="agree", support=["E1"])],
    ))
    synth_calls = {}

    async def synthesize_stream(intent, evidence, assessment, **kw):
        synth_calls.update(kw)
        synth_calls["assessment"] = assessment
        yield "The record shows: you said she wasn't abusive (2026-08-18)."

    monkeypatch.setattr(facets_mod, "decompose", decompose)
    monkeypatch.setattr(sweep_mod, "run_sweep", run_sweep)
    monkeypatch.setattr(assessor_mod, "assess", assess)
    monkeypatch.setattr(synth_mod, "synthesize_stream", synthesize_stream)
    return {"decompose": decompose, "run_sweep": run_sweep,
            "assess": assess, "synth_calls": synth_calls}


def _collect(ctx):
    async def _run():
        chunks = []
        async for c in handlers._run_insight_mode(ctx):
            chunks.append(c)
        return chunks
    return asyncio.run(_run())


class TestInsightModeHandler:
    def test_theme_sweep_owns_turn_with_provenance(self, stages, monkeypatch):
        dispatched = {}
        monkeypatch.setattr(
            handlers, "_dispatch_storage",
            lambda *a, **k: dispatched.update({"prov": a[9], "mode": a[10]}),
        )
        ctx = _ctx({"kind": "theme_sweep", "theme": "sleep",
                    "wants_document": False, "raw_query": "gather it"})
        chunks = _collect(ctx)
        assert ctx.handled is True
        assert ctx.storage_dispatched is True
        final = chunks[-1]
        assert "record shows" in final["content"]
        assert "debug" in final
        assert dispatched["mode"] == "insight-assembly"
        assert dispatched["prov"]["response_mode"] == "insight-assembly"
        assert dispatched["prov"]["insight_kind"] == "theme_sweep"
        assert dispatched["prov"]["evidence_count"] == 2
        stages["assess"].assert_not_called()  # sweeps never run the assessor

    def test_assessment_kind_runs_assessor(self, stages, monkeypatch):
        monkeypatch.setattr(handlers, "_dispatch_storage", lambda *a, **k: None)
        ctx = _ctx({"kind": "insight_assessment", "theme": "my problem is x",
                    "wants_document": False, "raw_query": "am I right"})
        _collect(ctx)
        stages["assess"].assert_awaited_once()
        assert stages["synth_calls"]["assessment"].overall == "agree"

    def test_sweep_exception_fails_honestly(self, stages, monkeypatch):
        import core.insight.sweep as sweep_mod
        monkeypatch.setattr(
            sweep_mod, "run_sweep", AsyncMock(side_effect=RuntimeError("boom"))
        )
        ctx = _ctx({"kind": "theme_sweep", "theme": "sleep",
                    "wants_document": False, "raw_query": "gather it"})
        chunks = _collect(ctx)  # must not raise
        # 2026-08-31 contract change: insight failures are terminal and
        # honest — never silently replaced by an unrelated agentic/enhanced
        # answer (evidence-sensitive workflow).
        assert ctx.handled is True
        assert "couldn't complete the insight synthesis" in chunks[-1]["content"]

    def test_elevated_tone_threads_to_synthesizer(self, stages, monkeypatch):
        monkeypatch.setattr(handlers, "_dispatch_storage", lambda *a, **k: None)
        ctx = _ctx({"kind": "theme_sweep", "theme": "my pattern with casey",
                    "wants_document": False, "raw_query": "gather it"},
                   tone_level="CONCERN")
        _collect(ctx)
        assert stages["synth_calls"]["tone_elevated"] is True

    def test_empty_synthesis_fails_honestly(self, stages, monkeypatch):
        import core.insight.synthesizer as synth_mod

        async def empty_stream(*a, **kw):
            if False:
                yield ""
        monkeypatch.setattr(synth_mod, "synthesize_stream", empty_stream)
        ctx = _ctx({"kind": "theme_sweep", "theme": "sleep",
                    "wants_document": False, "raw_query": "gather it"})
        chunks = _collect(ctx)
        # Same 2026-08-31 contract: empty synthesis (after the reasoning-off
        # retry) surfaces a bounded failure instead of falling through.
        assert ctx.handled is True
        assert "couldn't complete the insight synthesis" in chunks[-1]["content"]


class TestDocumentGating:
    def _dg_spy(self, monkeypatch):
        import knowledge.document_generator as dg_mod
        saved = {}
        fake_doc = MagicMock(title="My Pattern", path="/tmp/x/doc.md")

        class FakeDG:
            def __init__(self, **kw):
                pass

            def save_prewritten(self, markdown, **kw):
                saved["markdown"] = markdown
                saved.update(kw)
                return fake_doc

        monkeypatch.setattr(dg_mod, "DocumentGenerator", FakeDG)
        monkeypatch.setattr(handlers, "_dispatch_storage", lambda *a, **k: None)
        return saved

    def test_wants_document_saves(self, stages, monkeypatch):
        saved = self._dg_spy(monkeypatch)
        ctx = _ctx({"kind": "theme_sweep", "theme": "my pattern with casey",
                    "wants_document": True, "raw_query": "write a summary"})
        chunks = _collect(ctx)
        assert saved["topic"] == "my pattern with casey"
        assert sorted(saved["source_types"]) == ["conversations", "facts"]
        assert "Saved as" in chunks[-1]["content"]

    def test_disagree_withholds_document(self, stages, monkeypatch):
        import core.insight.assessor as assessor_mod
        saved = self._dg_spy(monkeypatch)
        monkeypatch.setattr(assessor_mod, "assess", AsyncMock(
            return_value=Assessment(
                overall="disagree",
                claims=[ClaimAssessment(claim="c", verdict="disagree", refute=["E2"])],
            )
        ))
        ctx = _ctx({"kind": "insight_assessment", "theme": "t",
                    "wants_document": True, "raw_query": "check and save it"})
        chunks = _collect(ctx)
        assert "markdown" not in saved  # never written
        assert "held off" in chunks[-1]["content"]

    def test_agreed_assessment_saves_doc_on_agreement(self, stages, monkeypatch):
        # goal 2's contract: agreement itself warrants the provenance document
        saved = self._dg_spy(monkeypatch)
        ctx = _ctx({"kind": "insight_assessment", "theme": "t",
                    "wants_document": False, "raw_query": "am I right"})
        _collect(ctx)
        assert "markdown" in saved

    def test_doc_save_failure_is_honest_not_fatal(self, stages, monkeypatch):
        import knowledge.document_generator as dg_mod

        class BoomDG:
            def __init__(self, **kw):
                pass

            def save_prewritten(self, markdown, **kw):
                raise OSError("disk full")

        monkeypatch.setattr(dg_mod, "DocumentGenerator", BoomDG)
        monkeypatch.setattr(handlers, "_dispatch_storage", lambda *a, **k: None)
        ctx = _ctx({"kind": "theme_sweep", "theme": "t",
                    "wants_document": True, "raw_query": "write a summary"})
        chunks = _collect(ctx)
        assert ctx.handled is True
        assert "couldn't save" in chunks[-1]["content"]


class TestDispatcherWiring:
    def test_insight_branch_precedes_doc_gen(self):
        src = inspect.getsource(handlers.handle_submit_streaming_enhanced) \
            if hasattr(handlers, "handle_submit_streaming_enhanced") \
            else inspect.getsource(handlers)
        i_insight = src.index("_run_insight_mode(ctx)")
        i_docgen = src.index("_run_doc_generation(ctx)")
        # the dispatcher CALL comes after the function DEFINITIONS; compare
        # the last occurrences (the dispatch sites)
        assert src.rindex("_run_insight_mode(ctx)") < src.rindex("_run_doc_generation(ctx)")
        assert i_insight and i_docgen

    def test_offer_note_wired(self):
        src = inspect.getsource(handlers)
        assert "[INSIGHT OFFER]" in src
        assert "maybe_arm_insight_offer" in src
        # the offer must never push — check the DISPATCH-SITE occurrence (the
        # module docstring also mentions the marker; rindex targets the code)
        assert "never" in src[src.rindex("[INSIGHT OFFER]"):src.rindex("[INSIGHT OFFER]") + 600].lower()


class TestInterleavePhaseEvents:
    """2026-08-31 sleep/functioning run: phase evidence appended in phase
    order (25 stable-on + 29 taper) filled the 50-item pattern-evidence cap
    before ANY of the 62 post-cessation events — the synthesis prompt had no
    quotable off-phase statement while the manifest reported 25/29/62."""

    @staticmethod
    def _comparison(label, n_events, n_proxy=0):
        from types import SimpleNamespace
        return SimpleNamespace(
            events=[SimpleNamespace(phase=label, idx=i) for i in range(n_events)],
            proxy_events=[
                SimpleNamespace(phase=f"{label}-proxy", idx=i) for i in range(n_proxy)
            ],
        )

    def test_live_run_shape_keeps_off_phase_under_cap(self):
        comparisons = [
            self._comparison("stable-on", 25, 14),
            self._comparison("taper", 29, 4),
            self._comparison("off", 62, 9),
        ]
        interleaved = handlers._interleave_phase_events(comparisons)
        assert len(interleaved) == 25 + 14 + 29 + 4 + 62 + 9
        capped_phases = {e.phase for e in interleaved[:50]}
        # Every phase survives a 50-item cap — including the final one.
        assert {"stable-on", "taper", "off"} <= capped_phases
        # Fair sampling: roughly a third of the cap is off-phase.
        assert sum(1 for e in interleaved[:50] if e.phase == "off") >= 15

    def test_outcome_events_precede_proxies_within_a_phase(self):
        comparisons = [self._comparison("only", 2, 2)]
        interleaved = handlers._interleave_phase_events(comparisons)
        assert [e.phase for e in interleaved] == [
            "only", "only", "only-proxy", "only-proxy"]

    def test_empty_and_uneven_phases(self):
        comparisons = [self._comparison("a", 0, 0), self._comparison("b", 3, 0)]
        interleaved = handlers._interleave_phase_events(comparisons)
        assert [e.idx for e in interleaved] == [0, 1, 2]
        assert handlers._interleave_phase_events([]) == []
