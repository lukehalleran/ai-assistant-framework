"""
Conversation-sourced document generation (2026-08-23).

The live incident: "Please summerize insight with direct evidence so I can
text that to my therapist" ran RESEARCH mode — handlers passed only the
~70-char request as source_material, which failed DOCUMENT_PROVIDED_MIN_CHARS
(400), so the generator web-searched the topic instead of writing up the
conversation the user wanted summarized.

Fix has three layers, all driven here as THE deployed functions:
  1. LLM trigger declares document_source ("research" | "conversation")
     end-to-end: prompt schema → LLMSearchTriggerResponse.parse →
     WebSearchDecision → gate Tier-4 doc_gen_intent["source"].
  2. Deterministic backstop in handlers (_DOC_CONVERSATION_SOURCE_RE +
     _resolve_doc_source) — Tier-3 regex intents carry no source and the
     LLM may omit the field; the backstop MUST match the live message.
  3. Conversation mode builds a transcript from ctx.history as
     source_material (clears the 400-char bar → conversation content is
     PRIMARY, web/wiki suppressed by the generator's existing logic).
"""

import inspect
from types import SimpleNamespace

import pytest

from gui.handlers import (
    _DOC_CONVERSATION_SOURCE_RE,
    _DOC_TRANSCRIPT_MAX_CHARS,
    _build_conversation_source_material,
    _resolve_doc_source,
    _run_doc_generation,
)
from utils.web_search_trigger import LLMSearchTriggerResponse

LIVE_MESSAGE = (
    "Please summerize insight with direct evidence so I can text that to my therapist"
)


class TestDocumentSourceParsing:
    def _parse(self, extra: str) -> LLMSearchTriggerResponse:
        resp = LLMSearchTriggerResponse.parse(
            '{"should_search": false, "needs_document_generation": true, '
            '"document_topic": "therapy insights", "document_type": "summary"'
            + extra + "}"
        )
        assert resp is not None
        return resp

    def test_conversation_source_parsed(self):
        assert self._parse(', "document_source": "conversation"').document_source == "conversation"

    def test_research_source_parsed(self):
        assert self._parse(', "document_source": "research"').document_source == "research"

    def test_missing_source_defaults_empty(self):
        assert self._parse("").document_source == ""

    def test_invalid_source_normalized_to_empty(self):
        assert self._parse(', "document_source": "banana"').document_source == ""

    def test_case_and_whitespace_normalized(self):
        assert self._parse(', "document_source": " Conversation "').document_source == "conversation"

    def test_prompt_documents_the_field(self):
        """The LLM can only set the field if the prompt asks for it."""
        import utils.web_search_trigger as wst
        src = inspect.getsource(wst)
        assert '"document_source"' in src
        assert src.count("document_source") >= 5  # both dataclasses, parse, prompt, blend

    def test_blend_site_threads_source_through(self):
        """WebSearchDecision built from the LLM response carries the field."""
        from utils.web_search_trigger import WebSearchDecision, WebSearchDepth
        d = WebSearchDecision(
            should_search=False, depth=WebSearchDepth.QUICK, confidence=0.5,
            reason="", matched_keywords=[], matched_patterns=[],
            document_source="conversation",
        )
        assert d.document_source == "conversation"


class TestGateCarriesSource:
    def test_tier4_doc_gen_intent_includes_source(self):
        """Source-level: the Tier-4 dict must thread document_source →
        doc_gen_intent["source"] (behavioral test would need a full mocked
        gate run; the dict literal is the single construction site)."""
        from core.agentic import gate
        src = inspect.getsource(gate.evaluate_agentic_gate)
        assert '"source": getattr(trigger_decision, \'document_source\'' in src


class TestConversationBackstopRegex:
    def test_live_therapist_message_matches(self):
        assert _DOC_CONVERSATION_SOURCE_RE.search(LIVE_MESSAGE)

    @pytest.mark.parametrize("text", [
        "Can you summarize what we just discussed?",
        "summarize our conversation into a doc",
        "Summarise this discussion for me",
        "write this up so I can send it to my doctor",
        "please write it up so I can share it with my mom",
        "summarize these insights and save them",
    ])
    def test_conversation_shapes_match(self, text):
        assert _DOC_CONVERSATION_SOURCE_RE.search(text)

    @pytest.mark.parametrize("text", [
        "write a report about climate change",
        "save a summary on quantum computing",
        "summarize quantum computing for me",
        "prepare a report on economic trends",
        "make me a research document about transformers",
        "generate a report and save it",
    ])
    def test_research_shapes_do_not_match(self, text):
        assert not _DOC_CONVERSATION_SOURCE_RE.search(text)


class TestResolveDocSource:
    def test_llm_conversation_declaration_wins(self):
        assert _resolve_doc_source({"source": "conversation"},
                                   "write a report on X") == "conversation"

    def test_backstop_fires_when_source_absent(self):
        # Tier-3 regex intents have no "source" key at all.
        assert _resolve_doc_source({"topic": "insights", "doc_type": "summary"},
                                   LIVE_MESSAGE) == "conversation"

    def test_backstop_overrides_llm_research_on_deterministic_evidence(self):
        # Deterministic signal beats a wrong LLM label (house doctrine:
        # deterministic backstops win — strip_unjustified_location etc.)
        assert _resolve_doc_source({"source": "research"},
                                   LIVE_MESSAGE) == "conversation"

    def test_plain_research_request_stays_research(self):
        assert _resolve_doc_source({"source": None},
                                   "write a report about climate change") == "research"

    def test_none_intent_and_empty_text(self):
        assert _resolve_doc_source(None, "") == "research"
        assert _resolve_doc_source(None, None) == "research"


class TestTranscriptBuilder:
    def _history(self):
        return [
            {"role": "user", "content": "I noticed I always crash after seeing my dad"},
            {"role": "assistant", "content": "That pattern shows up in three prior sessions too."},
            {"role": "user", "content": LIVE_MESSAGE},
        ]

    def test_roles_rendered_and_request_appended(self):
        out = _build_conversation_source_material(self._history(), LIVE_MESSAGE)
        assert "User: I noticed I always crash" in out
        assert "Daemon: That pattern shows up" in out
        assert out.rstrip().endswith(LIVE_MESSAGE)
        assert "[USER REQUEST]" in out

    def test_progress_chunks_and_empty_skipped(self):
        history = [
            {"role": "assistant", "content": "📝 Researching: something..."},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": None},
            "not-a-dict",
            {"role": "user", "content": "real message"},
        ]
        out = _build_conversation_source_material(history, "req")
        assert "📝" not in out
        assert "not-a-dict" not in out
        assert "User: real message" in out

    def test_char_cap_keeps_newest(self):
        history = (
            [{"role": "user", "content": "OLDEST " + "x" * 500}]
            + [{"role": "user", "content": f"msg {i} " + "y" * 400} for i in range(25)]
        )
        out = _build_conversation_source_material(history, "req")
        # Transcript portion capped; the newest message survives, oldest trimmed.
        assert len(out) <= _DOC_TRANSCRIPT_MAX_CHARS + 200  # + request/header slack
        assert "msg 24" in out
        assert "OLDEST" not in out

    def test_empty_history_still_carries_request(self):
        out = _build_conversation_source_material(None, LIVE_MESSAGE)
        assert LIVE_MESSAGE in out


class TestHandlersWiring:
    def test_run_doc_generation_uses_resolver_and_builder(self):
        src = inspect.getsource(_run_doc_generation)
        assert "_resolve_doc_source" in src
        assert "_build_conversation_source_material" in src
        assert "source_material=_source_material" in src

    @pytest.mark.asyncio
    async def test_conversation_mode_passes_transcript_as_material(self, monkeypatch):
        """End-to-end through THE deployed _run_doc_generation with a fake
        DocumentGenerator: conversation-sourced request → source_material is
        the transcript (clears the 400-char bar), progress says summarizing."""
        import gui.handlers as handlers
        import knowledge.document_generator as dg_mod

        captured = {}

        class FakeGenerator:
            def __init__(self, **kwargs):
                pass

            async def generate(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    title="Therapy Insights", path="/tmp/doc.md",
                    doc_type="summary", sources=[], sections_count=2,
                    word_count=300,
                )

        monkeypatch.setattr(dg_mod, "DocumentGenerator", FakeGenerator)
        monkeypatch.setattr(handlers, "_write_turn_telemetry",
                            lambda *a, **k: None)
        monkeypatch.setattr(handlers, "_get_session_id", lambda *a: "s1")

        ctx = SimpleNamespace(
            orchestrator=SimpleNamespace(
                prompt_builder=None, memory_system=None,
                model_manager=SimpleNamespace(
                    get_active_model_name=lambda: "test-model"),
            ),
            doc_gen_intent={"topic": "therapy insights", "doc_type": "summary",
                            "focus": None, "source": "conversation"},
            user_text=LIVE_MESSAGE,
            history=[
                {"role": "user", "content": "I crash after seeing my dad"},
                {"role": "assistant", "content": "Three sessions show that pattern."},
            ],
            handled=False,
        )

        chunks = [c async for c in handlers._run_doc_generation(ctx)]

        assert ctx.handled is True
        material = captured["source_material"]
        assert "User: I crash after seeing my dad" in material
        assert "Daemon: Three sessions show that pattern." in material
        assert "[USER REQUEST]" in material
        progress = [c for c in chunks if c.get("is_progress")]
        assert progress and "Summarizing our conversation" in progress[0]["content"]

    @pytest.mark.asyncio
    async def test_research_mode_unchanged(self, monkeypatch):
        import gui.handlers as handlers
        import knowledge.document_generator as dg_mod

        captured = {}

        class FakeGenerator:
            def __init__(self, **kwargs):
                pass

            async def generate(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    title="Climate", path="/tmp/doc.md", doc_type="report",
                    sources=[], sections_count=3, word_count=500,
                )

        monkeypatch.setattr(dg_mod, "DocumentGenerator", FakeGenerator)
        monkeypatch.setattr(handlers, "_write_turn_telemetry",
                            lambda *a, **k: None)
        monkeypatch.setattr(handlers, "_get_session_id", lambda *a: "s1")

        user_text = "write a report about climate change"
        ctx = SimpleNamespace(
            orchestrator=SimpleNamespace(
                prompt_builder=None, memory_system=None,
                model_manager=SimpleNamespace(
                    get_active_model_name=lambda: "test-model"),
            ),
            doc_gen_intent={"topic": "climate change", "doc_type": "report",
                            "focus": None},
            user_text=user_text,
            history=[{"role": "user", "content": "unrelated chatter"}],
            handled=False,
        )

        chunks = [c async for c in handlers._run_doc_generation(ctx)]

        assert ctx.handled is True
        assert captured["source_material"] == user_text  # not the transcript
        progress = [c for c in chunks if c.get("is_progress")]
        assert progress and "Researching: climate change" in progress[0]["content"]
