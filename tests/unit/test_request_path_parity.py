"""
Regression tests for the 2026-09-04 dead-wiring-class fix (FIX A of the
class-closing batch): DaemonOrchestrator.process_user_query() (RAW mode via
gui/handlers._run_raw, and `python main.py cli`) and the GUI dispatcher
(gui/handlers._write_turn_telemetry, driving the enhanced/agentic/duel/
doc-gen/self-note/insight-assembly/... modes) used to run separate,
hand-maintained per-turn hook sequences that silently drifted apart. That
was the root cause of three real production bugs: the 2026-08-21
EscalationTracker/SafetyCanary gap, the 2026-08-23 correction-pipeline gap,
and (closed here) the fact that RAW-mode turns never wrote turn telemetry
or taught the adaptive "search_worthy" exemplar store.

Both paths now iterate the SAME module-level registry,
core.orchestrator.POST_RESPONSE_HOOKS / run_post_response_hooks(). This
file pins:

  (3a) every hook in the registry actually fires on a stubbed ENHANCED
       handle_submit turn (the GUI dispatcher's own driving path);
  (3b) a source-level guard: escalation_tracker.update( and the public
       orchestrator.run_post_response_detectors( each have exactly ONE
       call site in the scanned source tree, and it lives in
       core/orchestrator.py (the module hosting the registry) — no other
       module may call either directly;
  (4)  a MODE parity check: every response mode handle_submit can end a
       turn in yields a FINAL chunk carrying a debug dict with 'mode' set
       (api/chat_service.py's SSE `complete` event and the Debug/
       Provenance SPA views key off this — a mode missing it silently
       vanishes from both).

Before this batch: doc-generation and self-note already carried debug
records as of an earlier same-day fix (see gui/handlers.py "2026-09-04"
comments near _run_doc_generation / _run_self_note) — this file's mode
sweep re-verifies that fix rather than re-doing it. All 8 modes checked
here (raw, best-of-duel, doc-generation, self-note, insight-assembly,
agentic-search, enhanced, uncertainty-fallback) pass as of this commit;
none required a NEW handlers.py fix from this batch.
"""

import asyncio
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import core.orchestrator as orchestrator_module
import gui.handlers as handlers
from core.insight.types import Assessment, ClaimAssessment, EvidenceItem, FacetPlan, FacetQuery
from tests.unit.test_handle_submit import (
    _debug_record,
    _gate_decision,
    _make_orchestrator,
    _no_trigger_decision,
    _run_submit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ===========================================================================
# Step 3(a) — every POST_RESPONSE_HOOKS entry fires on a stubbed ENHANCED turn
# ===========================================================================
class TestRegistryHooksFireOnEnhancedTurn:
    @pytest.mark.asyncio
    async def test_all_post_response_hooks_invoked(self, monkeypatch):
        calls = []
        spied = []
        for name, fn in orchestrator_module.POST_RESPONSE_HOOKS:
            def _spy(ctx, _name=name, _fn=fn):
                calls.append(_name)
                return _fn(ctx)
            spied.append((name, _spy))
        assert len(spied) >= 4, "expected turn_telemetry/search_worthy_teach/" \
            "escalation_record_response/post_response_detectors at minimum"
        monkeypatch.setattr(orchestrator_module, "POST_RESPONSE_HOOKS", spied)

        orch = _make_orchestrator(streaming_chunks=["Hello", " world"])
        await _run_submit("How are you?", orch)

        expected = {name for name, _ in spied}
        fired = set(calls)
        assert fired == expected, f"hooks never invoked: {expected - fired}"

    @pytest.mark.asyncio
    async def test_hooks_fire_exactly_once_per_turn(self, monkeypatch):
        """No hook double-fires within a single completed enhanced turn —
        the exact failure mode a naive 'both paths call everything' merge
        would have introduced (e.g. run_post_response_detectors running
        twice would double-apply a truth-event/staleness cascade)."""
        counts = {}
        spied = []
        for name, fn in orchestrator_module.POST_RESPONSE_HOOKS:
            def _spy(ctx, _name=name, _fn=fn):
                counts[_name] = counts.get(_name, 0) + 1
                return _fn(ctx)
            spied.append((name, _spy))
        monkeypatch.setattr(orchestrator_module, "POST_RESPONSE_HOOKS", spied)

        orch = _make_orchestrator(streaming_chunks=["Hello", " world"])
        await _run_submit("How are you?", orch)

        assert all(c == 1 for c in counts.values()), counts

    @pytest.mark.asyncio
    async def test_process_user_query_raw_mode_also_runs_registry(self, monkeypatch):
        """The other production caller of process_user_query (RAW mode,
        reached via gui/handlers._run_raw) must run the SAME registry —
        this is the concrete gap FIX A closes: RAW turns previously wrote
        no turn telemetry and never taught adaptive exemplars."""
        from core.orchestrator import DaemonOrchestrator, PostResponseHookContext

        calls = []
        spied = []
        for name, fn in orchestrator_module.POST_RESPONSE_HOOKS:
            def _spy(ctx, _name=name, _fn=fn):
                calls.append(_name)
            spied.append((name, _spy))
        monkeypatch.setattr(orchestrator_module, "POST_RESPONSE_HOOKS", spied)

        orch = DaemonOrchestrator.__new__(DaemonOrchestrator)
        orch.logger = None
        orch.conversation_logger = None
        orch.current_topic = "general"
        orch.current_tone_level = None
        orch.enable_citations = False
        orch._web_source_map = {}
        orch._current_memory_id_map = {}
        orch.model_manager = MagicMock()
        orch.model_manager.get_active_model_name = MagicMock(return_value="mock-model")
        orch.memory_system = MagicMock()
        orch.memory_system.session_id = "sess-1"
        orch.memory_system.get_memories = AsyncMock(return_value=[])
        orch.memory_system.store_interaction = AsyncMock()
        orch.escalation_tracker = None
        orch.safety_canary = None
        orch._agentic_controller = None
        orch._agentic_config = {}
        orch.user_profile = None
        orch.correction_detector = None
        orch.best_of_handler = MagicMock()
        orch.best_of_handler.should_use_best_of = MagicMock(return_value=False)
        orch.response_generator = MagicMock()

        async def _gen(*a, **k):
            for c in ("Hi", " there"):
                yield c
        orch.response_generator.generate_streaming_response = _gen

        from core.context_pipeline import ContextResult, ToneLevel
        ctx = ContextResult(
            processed_query="hi there", original_query="hi there",
            tone_level=ToneLevel.CONVERSATIONAL, tone_instructions="",
            emotional_context=None,
        )
        orch.build_context = AsyncMock(return_value=ctx)
        orch.build_full_prompt = AsyncMock(return_value=("PROMPT", "SYSTEM", {}))

        text, debug = await orch.process_user_query("hi there", use_raw_mode=True)

        assert "Hi" in text and "there" in text
        expected = {name for name, _ in spied}
        assert set(calls) == expected, f"RAW mode skipped hooks: {expected - set(calls)}"


# ===========================================================================
# Step 3(b) — source-level single-call-site guard
# ===========================================================================
class TestSingleCallSiteGuard:
    """escalation_tracker.update( and the public
    orchestrator.run_post_response_detectors( must each have exactly ONE
    call site in the scanned source tree, living in core/orchestrator.py
    (the module that hosts POST_RESPONSE_HOOKS / run_post_response_hooks
    and the _update_safety_trackers / _run_post_response_detectors methods
    the registry wraps). Every other caller (gui/handlers.py included) must
    reach these exclusively through run_post_response_hooks()."""

    _SCAN_DIRS = ("core", "gui", "memory", "utils", "api", "knowledge", "config")
    _SCAN_FILES = ("main.py",)
    _REGISTRY_FILE = "core/orchestrator.py"

    @classmethod
    def _iter_py_files(cls):
        for d in cls._SCAN_DIRS:
            base = REPO_ROOT / d
            if base.exists():
                yield from base.rglob("*.py")
        for f in cls._SCAN_FILES:
            p = REPO_ROOT / f
            if p.exists():
                yield p

    @classmethod
    def _code_hits(cls, pattern):
        rx = re.compile(pattern)
        hits = []
        for path in cls._iter_py_files():
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if line.strip().startswith("#"):
                    continue
                if rx.search(line):
                    hits.append((str(path.relative_to(REPO_ROOT)), lineno, line.strip()))
        return hits

    def test_escalation_update_single_call_site_in_registry(self):
        hits = self._code_hits(r"escalation_tracker\.update\(")
        assert len(hits) == 1, hits
        assert hits[0][0] == self._REGISTRY_FILE, hits

    def test_run_post_response_detectors_single_call_site_in_registry(self):
        # Leading-dot anchor excludes the flow-shaped wrapper
        # _run_post_response_detectors( (registered as the
        # "post_response_detectors" hook's own implementation) — only a
        # call to the PUBLIC method counts.
        hits = self._code_hits(r"\.run_post_response_detectors\(")
        assert len(hits) == 1, hits
        assert hits[0][0] == self._REGISTRY_FILE, hits

    def test_handlers_no_longer_calls_either_directly(self):
        """The GUI dispatcher must reach both exclusively via
        run_post_response_hooks() — this is the concrete drift FIX A
        eliminates (gui/handlers.py used to hold its own copy of the
        run_post_response_detectors call)."""
        src = Path(handlers.__file__).read_text(encoding="utf-8")
        assert "escalation_tracker.update(" not in src
        assert ".run_post_response_detectors(" not in src


# ===========================================================================
# Step 4 — every response mode's final yield carries debug['mode']
# ===========================================================================
async def _run_insight_assembly():
    """Drive gui.handlers._run_insight_mode directly (no ready-made fake
    exists in test_handle_submit.py for this mode; pattern mirrors
    tests/unit/test_insight_mode_handler.py's `stages`/`_ctx` fixtures)."""
    import core.insight.assessor as assessor_mod
    import core.insight.facets as facets_mod
    import core.insight.sweep as sweep_mod
    import core.insight.synthesizer as synth_mod

    plan = FacetPlan(facets=[FacetQuery(name="f", query_text="q")], claims=["c"])
    items = [EvidenceItem(doc_id="d1", text="evidence text", date="2026-08-18",
                           collection="conversations", speaker="user")]

    async def synthesize_stream(intent, evidence, assessment, **kw):
        yield "The record shows: evidence text (2026-08-18)."

    with patch.object(facets_mod, "decompose", AsyncMock(return_value=plan)), \
         patch.object(sweep_mod, "run_sweep", AsyncMock(return_value=items)), \
         patch.object(assessor_mod, "assess", AsyncMock(return_value=Assessment(
             overall="agree",
             claims=[ClaimAssessment(claim="c", verdict="agree", support=["E1"])],
         ))), \
         patch.object(synth_mod, "synthesize_stream", synthesize_stream), \
         patch.object(handlers, "_dispatch_storage", lambda *a, **k: None):

        orchestrator = MagicMock()
        orchestrator.model_manager.get_active_model_name = MagicMock(return_value="kimi-3")
        orchestrator.memory_system.chroma_store = MagicMock()
        ctx = handlers.SubmitContext(
            user_text="gather it", files=None, history=[], use_raw_gpt=False,
            orchestrator=orchestrator, personality=None, fast_mode=False,
            conversation_logger=None, file_names=[], merged_input="gather it",
            files_result=None,
        )
        ctx.raw_context = {}
        ctx.gate_decision = MagicMock()
        ctx.gate_decision.insight_intent = {
            "kind": "theme_sweep", "theme": "sleep",
            "wants_document": False, "raw_query": "gather it",
        }
        chunks = []
        async for c in handlers._run_insight_mode(ctx):
            chunks.append(c)
        return chunks


class TestModeParityDebugRecord:
    """Every mode handle_submit can end a turn in must leave a FINAL yield
    carrying a debug dict with 'mode' set (api/chat_service.py's
    `is_final = "debug" in chunk` check, and the SPA Debug/Provenance
    views, key off this). A mode failing this is a real bug — see the
    module docstring for which modes were already fixed same-day by a
    different batch before this file was written."""

    def _assert_final_debug_mode(self, results, expected_mode):
        debug = _debug_record(results)
        assert debug is not None, f"no debug record yielded for mode={expected_mode!r}"
        assert debug.get("mode") == expected_mode, debug.get("mode")
        # The debug-bearing chunk must be the FINAL substantive yield.
        finals = [r for r in reversed(results)
                  if not r.get("is_progress") and not r.get("is_thinking")]
        assert finals, "no non-progress yield at all"
        assert finals[0].get("debug") is not None, \
            "debug record exists but is not on the final yielded chunk"

    @pytest.mark.asyncio
    async def test_raw_mode(self):
        orch = _make_orchestrator(raw_result=("Raw answer.", {}))
        results = await _run_submit("test", orch, use_raw_gpt=True)
        self._assert_final_debug_mode(results, "raw")

    @pytest.mark.asyncio
    async def test_best_of_duel_mode(self):
        orch = _make_orchestrator(duel_enabled=True, duel_gens=["a", "b"], duel_sels=["j"])
        results = await _run_submit("test", orch)
        self._assert_final_debug_mode(results, "best-of-duel")

    @pytest.mark.asyncio
    async def test_agentic_search_mode(self):
        orch = _make_orchestrator(
            agentic_enabled=True, agentic_items=["The 10th Fibonacci number is 55."],
        )
        results = await _run_submit("calculate fibonacci 10", orch)
        self._assert_final_debug_mode(results, "agentic-search")

    @pytest.mark.asyncio
    async def test_enhanced_mode(self):
        orch = _make_orchestrator(streaming_chunks=["Hello", " world"])
        results = await _run_submit("How are you?", orch)
        self._assert_final_debug_mode(results, "enhanced")

    @pytest.mark.asyncio
    async def test_uncertainty_fallback_mode(self):
        query = "Can you describe the color of my cat's fur exactly?"
        orch = _make_orchestrator(
            agentic_enabled=True,
            streaming_chunks=["I don't have any information about that."],
        )
        extra = [
            patch("config.app_config.UNCERTAINTY_FALLBACK_ENABLED", True),
            patch("config.app_config.UNCERTAINTY_SEMANTIC_THRESHOLD", 0.7),
            patch("config.app_config.UNCERTAINTY_MAX_LENGTH", 1200),
            patch(
                "utils.web_search_trigger.analyze_for_web_search_llm",
                new_callable=AsyncMock, return_value=_no_trigger_decision(),
            ),
            patch(
                "gui.handlers._silent_agentic_retry", new_callable=AsyncMock,
                return_value=("Here is the information you wanted about your cat.", ""),
            ),
        ]
        results = await _run_submit(query, orch, extra_patches=extra)
        self._assert_final_debug_mode(results, "uncertainty-fallback")

    @pytest.mark.asyncio
    async def test_doc_generation_mode(self):
        orch = _make_orchestrator(agentic_enabled=True)
        dg = MagicMock()
        doc_result = MagicMock()
        doc_result.title, doc_result.path, doc_result.doc_type = "Q", "documents/q.md", "report"
        doc_result.sources, doc_result.sections_count, doc_result.word_count = ["a"], 2, 100
        dg.generate = AsyncMock(return_value=doc_result)
        extra = [
            patch(
                "core.agentic.gate.evaluate_agentic_gate", new_callable=AsyncMock,
                return_value=_gate_decision(
                    doc_gen_intent={"topic": "Q", "doc_type": "report", "focus": None},
                ),
            ),
            patch("knowledge.document_generator.DocumentGenerator", return_value=dg),
        ]
        results = await _run_submit("write a report on Q", orch, extra_patches=extra)
        self._assert_final_debug_mode(results, "doc-generation")

    @pytest.mark.asyncio
    async def test_self_note_mode(self):
        orch = _make_orchestrator(agentic_enabled=True)
        note_result = MagicMock()
        note_result.title, note_result.path = "Status", "daemon_notes/status.md"
        note_result.category, note_result.confidence, note_result.id = "project", "medium", "n1"
        dnm = MagicMock()
        dnm._generate_summary = AsyncMock(return_value="A summary.")
        dnm.create_note = AsyncMock(return_value=note_result)
        extra = [
            patch(
                "core.agentic.gate.evaluate_agentic_gate", new_callable=AsyncMock,
                return_value=_gate_decision(
                    self_note_intent={"topic": "Status", "category": "project"},
                ),
            ),
            patch("knowledge.daemon_notes_manager.DaemonNotesManager", return_value=dnm),
        ]
        results = await _run_submit("make a note about status", orch, extra_patches=extra)
        self._assert_final_debug_mode(results, "self-note")

    @pytest.mark.asyncio
    async def test_insight_assembly_mode(self):
        results = await _run_insight_assembly()
        self._assert_final_debug_mode(results, "insight-assembly")
