"""Small guards from the 2026-08-28 log retrospective.

1. Crisis+URL veto pierce: "I don't want to die I am fucking scared and hurt
   https://en.wikip…" (08-18 14:28, MEDIUM, harm_score 12) ran an agentic
   loop because the pasted link made the vent veto_exempt. A bare URL inside
   an ACUTE first-person vent is shared context, not a fetch request — the
   acute tone arm may pierce THAT exemption only.
2. Insight empty-intent guard: `INSIGHT MODE: {}` crashed InsightIntent
   validation (kind/theme Field required) before falling through.
3. FSGuard daemon-state exemption: the web-search credits tracker's atomic
   save under data/ was blocked on every agentic turn since 07-16.
4. Shutdown LLM failure marker: 08-24 shutdown hit OpenRouter 402s visible
   only at DEBUG — session memory processing failed invisibly.
"""
import asyncio
from types import SimpleNamespace

import pytest

from core.agentic.gate import AgenticDecision, apply_intent_veto, evaluate_agentic_gate


LIVE_CRISIS_URL_QUERY = (
    "I don't want to die I am fucking scared and hurt "
    "https://en.wikipedia.org/wiki/Neuroleptic_induced_deficit_syndrome"
)


class TestCrisisUrlVetoPierce:
    @pytest.mark.asyncio
    async def test_live_turn_acute_vent_with_bare_url_vetoed(self):
        d = await evaluate_agentic_gate(LIVE_CRISIS_URL_QUERY, None, None, None,
                                        intent_info=None)
        assert d.should_trigger and d.veto_exempt and d.veto_exempt_url_only
        out = apply_intent_veto(
            d, {"intent_type": "general", "confidence": 0.0},
            tone_level="CrisisLevel.MEDIUM", query=LIVE_CRISIS_URL_QUERY,
        )
        assert out.should_trigger is False

    @pytest.mark.asyncio
    async def test_high_tone_also_pierces(self):
        d = await evaluate_agentic_gate(LIVE_CRISIS_URL_QUERY, None, None, None,
                                        intent_info=None)
        out = apply_intent_veto(
            d, {"intent_type": "general", "confidence": 0.0},
            tone_level="CrisisLevel.HIGH", query=LIVE_CRISIS_URL_QUERY,
        )
        assert out.should_trigger is False

    @pytest.mark.asyncio
    async def test_explicit_url_request_stays_exempt_at_acute(self):
        q = "go to https://registrar.gatech.edu and check the drop deadline"
        d = await evaluate_agentic_gate(q, None, None, None, intent_info=None)
        assert d.veto_exempt and not d.veto_exempt_url_only
        out = apply_intent_veto(
            d, {"intent_type": "general", "confidence": 0.0},
            tone_level="CrisisLevel.MEDIUM", query=q,
        )
        assert out.should_trigger is True

    @pytest.mark.asyncio
    async def test_bare_url_conversational_tone_still_triggers(self):
        q = "thoughts on this? https://example.com/article"
        d = await evaluate_agentic_gate(q, None, None, None, intent_info=None)
        out = apply_intent_veto(
            d, {"intent_type": "general", "confidence": 0.0},
            tone_level="CrisisLevel.CONVERSATIONAL", query=q,
        )
        assert out.should_trigger is True

    @pytest.mark.asyncio
    async def test_concern_tone_does_not_pierce(self):
        # Pierce is ACUTE-only (MEDIUM/HIGH): the sticky CONCERN floor held
        # across whole sessions historically — a bare link at CONCERN keeps
        # its exemption (under-fire by design).
        d = await evaluate_agentic_gate(LIVE_CRISIS_URL_QUERY, None, None, None,
                                        intent_info=None)
        out = apply_intent_veto(
            d, {"intent_type": "general", "confidence": 0.0},
            tone_level="CrisisLevel.CONCERN", query=LIVE_CRISIS_URL_QUERY,
        )
        assert out.should_trigger is True

    def test_info_seeking_url_vent_not_pierced(self):
        # Info-seeking shape means it's not vent-shaped → no pierce.
        d = AgenticDecision(should_trigger=True, modes=["web_search"],
                            veto_exempt=True, veto_exempt_url_only=True)
        q = "can you look up https://example.com/paper for me?"
        out = apply_intent_veto(
            d, {"intent_type": "general", "confidence": 0.0},
            tone_level="CrisisLevel.MEDIUM", query=q,
        )
        assert out.should_trigger is True


class TestInsightEmptyIntentGuard:
    @pytest.mark.asyncio
    async def test_empty_intent_falls_through_without_crash(self):
        import gui.handlers as handlers
        ctx = SimpleNamespace(
            orchestrator=SimpleNamespace(),
            gate_decision=SimpleNamespace(insight_intent={}),
            raw_context={},
            handled=False,
            telemetry={},
            user_text="whatever",
        )
        chunks = []
        async for c in handlers._run_insight_mode(ctx):
            chunks.append(c)
        assert ctx.handled is False
        assert chunks == []  # silent fall-through to the normal flow

    @pytest.mark.asyncio
    async def test_missing_theme_falls_through(self):
        import gui.handlers as handlers
        ctx = SimpleNamespace(
            orchestrator=SimpleNamespace(),
            gate_decision=SimpleNamespace(insight_intent={"kind": "insight_assessment"}),
            raw_context={}, handled=False, telemetry={}, user_text="x",
        )
        async for _c in handlers._run_insight_mode(ctx):
            pass
        assert ctx.handled is False


class TestDaemonStateFsExemption:
    def test_credits_file_exempt(self):
        from utils.python_fs_guard import _DAEMON_STATE_EXEMPT_PREFIXES
        assert any("web_search_credits" in p for p in _DAEMON_STATE_EXEMPT_PREFIXES)

    def test_exemption_covers_tmp_sibling(self):
        from utils.python_fs_guard import _DAEMON_STATE_EXEMPT_PREFIXES
        rel = "data/web_search_credits.json.tmp"
        assert rel.startswith(_DAEMON_STATE_EXEMPT_PREFIXES)

    def test_other_data_paths_not_exempt(self):
        from utils.python_fs_guard import _DAEMON_STATE_EXEMPT_PREFIXES
        for rel in ("data/knowledge_graph.json", "data/corpus_v4.json",
                    "data/entity_aliases.json"):
            assert not rel.startswith(_DAEMON_STATE_EXEMPT_PREFIXES)

    def test_check_and_maybe_block_allows_credits_in_agent_mode(self):
        import utils.python_fs_guard as g
        from pathlib import Path
        # Simulate guard active + agent mode, then check the deployed gate fn.
        saved_active = g._active
        saved_root = g._repo_root
        token = g._agent_mode.set(True)
        g._active = True
        g._repo_root = Path(__file__).resolve().parents[2]
        try:
            # exempt: must not raise
            g._check_and_maybe_block(
                "os.replace", str(g._repo_root / "data/web_search_credits.json.tmp"))
            # non-exempt protected sibling: must raise
            with pytest.raises(PermissionError):
                g._check_and_maybe_block(
                    "os.remove", str(g._repo_root / "data/knowledge_graph.json"))
        finally:
            g._active = saved_active
            g._repo_root = saved_root
            g._agent_mode.reset(token)


class TestShutdownLlmFailureMarker:
    def test_credit_failure_marked_loudly(self, caplog):
        from memory.shutdown_processor import _log_shutdown_llm_failure
        import logging
        with caplog.at_level(logging.ERROR):
            _log_shutdown_llm_failure(
                "extraction",
                RuntimeError("Client error '402 Payment Required' for url ..."))
        joined = " ".join(r.message for r in caplog.records)
        assert "[SHUTDOWN-LLM-FAILURE]" in joined
        assert "LOST" in joined

    def test_generic_failure_still_marked(self, caplog):
        from memory.shutdown_processor import _log_shutdown_llm_failure
        import logging
        with caplog.at_level(logging.ERROR):
            _log_shutdown_llm_failure("generation", ValueError("boom"))
        assert any("[SHUTDOWN-LLM-FAILURE]" in r.message for r in caplog.records)
