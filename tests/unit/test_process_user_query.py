"""
Characterization tests for DaemonOrchestrator.process_user_query and build_full_prompt.

These pin the OBSERVABLE behavior of the orchestrator's two god-methods so the
behavior-preserving decomposition (see plan: orchestrator god-method refactor) can be
verified at every step.

Strategy:
- Flow tests build a DaemonOrchestrator via a bypassed __init__ (the MockOrchestrator
  pattern from test_orchestrator_helpers.py) and mock build_context / build_full_prompt,
  so we exercise process_user_query's *control flow* (routing, storage, debug keys,
  exception handling) rather than prompt building.
- System-prompt tests run the REAL build_full_prompt with a mocked prompt_builder, pinning
  the assembled system prompt's sections (the part Stage 1 extracts into _build_system_prompt).
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.orchestrator import DaemonOrchestrator
from core.context_pipeline import ContextResult, ToneLevel


# =============================================================================
# Test scaffolding
# =============================================================================
class _StreamRecorder:
    """Callable that records call kwargs and returns an async generator of chunks."""
    def __init__(self, *chunks):
        self._chunks = chunks
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        chunks = self._chunks

        async def _gen():
            for c in chunks:
                yield c
        return _gen()

    @property
    def last_kwargs(self):
        return self.calls[-1][1] if self.calls else {}


def _agentic_stream(*chunks):
    """Return an object whose run_agentic_search yields the given chunks (async gen)."""
    def run_agentic_search(*args, **kwargs):
        async def _gen():
            for c in chunks:
                yield c
        return _gen()
    ctrl = MagicMock()
    ctrl.run_agentic_search = run_agentic_search
    return ctrl


def _make_context(**overrides):
    """Build a ContextResult with sane defaults for flow tests."""
    kwargs = dict(
        processed_query="processed",
        original_query="original query",
        tone_level=ToneLevel.CONVERSATIONAL,
        tone_instructions="",
        emotional_context=None,
        topics=["python"],
        primary_topic="python",
        is_heavy_topic=False,
        intent=None,
    )
    kwargs.update(overrides)
    return ContextResult(**kwargs)


def _make_flow_orch(**overrides):
    """
    Build an orchestrator for process_user_query flow tests.

    build_context / build_full_prompt are mocked so we test the flow, not prompt building.
    """
    # agentic_controller is a read-only property — control it via its backing field.
    agentic_controller = overrides.pop("agentic_controller", None)
    ctx = overrides.pop("context", _make_context())

    orch = DaemonOrchestrator.__new__(DaemonOrchestrator)
    orch.logger = None
    orch.conversation_logger = None
    orch.current_topic = "general"
    orch.current_tone_level = None
    orch.enable_citations = False
    orch._web_source_map = {}
    orch._current_memory_id_map = {}

    # Collaborators
    orch.model_manager = MagicMock()
    orch.model_manager.get_active_model_name = MagicMock(return_value="mock-model")

    orch.memory_system = MagicMock()
    orch.memory_system.session_id = "sess-1"
    orch.memory_system.get_memories = AsyncMock(return_value=[])
    orch.memory_system.store_interaction = AsyncMock()

    orch.escalation_tracker = None
    orch._agentic_controller = agentic_controller
    orch._agentic_config = {}
    orch.user_profile = None
    orch.correction_detector = None

    orch.best_of_handler = MagicMock()
    orch.best_of_handler.should_use_best_of = MagicMock(return_value=False)

    orch.response_generator = MagicMock()
    orch.response_generator.generate_streaming_response = _StreamRecorder("Hello", "world")

    # Mock the prompt-building methods (tested separately)
    orch.build_context = AsyncMock(return_value=ctx)
    orch.build_full_prompt = AsyncMock(return_value=("PROMPT", "SYSTEM", {}))

    for k, v in overrides.items():
        setattr(orch, k, v)
    return orch


async def _run(orch, text, **kwargs):
    return await orch.process_user_query(text, **kwargs)


# =============================================================================
# Command early-exit path
# =============================================================================
class TestCommandPath:
    @pytest.mark.asyncio
    async def test_topic_switch_command(self):
        orch = _make_flow_orch()
        text, debug = await _run(orch, "/topic astronomy")
        assert "astronomy" in text
        assert debug["command"] == "topic_switch"
        assert orch.current_topic == "astronomy"
        # No generation / storage on a command
        orch.memory_system.store_interaction.assert_not_called()
        orch.build_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_clear_topic_command(self):
        orch = _make_flow_orch()
        text, debug = await _run(orch, "/clear_topic")
        assert debug["command"] == "topic_clear"
        assert orch.current_topic == "general"


# =============================================================================
# Deictic clarification early-exit path
# =============================================================================
class TestDeicticPath:
    @pytest.mark.asyncio
    async def test_deictic_clarification_returns_early(self, monkeypatch):
        monkeypatch.setattr("core.orchestrator.is_deictic", lambda _t: True)
        orch = _make_flow_orch()
        orch.memory_system.get_memories = AsyncMock(
            return_value=[{"metadata": {"needs_clarification": True}}]
        )
        text, debug = await _run(orch, "what about it")
        assert "not sure what you're referring to" in text
        assert debug["prompt_length"] == 0
        assert "duration" in debug
        # clarification stored with its own tag; build never runs
        orch.memory_system.store_interaction.assert_awaited_once()
        orch.build_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_deictic_without_clarification_falls_through(self, monkeypatch):
        monkeypatch.setattr("core.orchestrator.is_deictic", lambda _t: True)
        orch = _make_flow_orch()
        orch.memory_system.get_memories = AsyncMock(return_value=[{"metadata": {}}])
        text, debug = await _run(orch, "what about it")
        # falls through to normal generation
        assert text == "Hello world"
        assert debug["mode"] == "enhanced"


# =============================================================================
# Raw mode
# =============================================================================
class TestRawMode:
    @pytest.mark.asyncio
    async def test_raw_mode_no_storage(self):
        orch = _make_flow_orch()
        text, debug = await _run(orch, "hi there", use_raw_mode=True)
        assert text == "Hello world"
        assert debug["mode"] == "raw"
        orch.memory_system.store_interaction.assert_not_called()


# =============================================================================
# Standard enhanced path
# =============================================================================
class TestEnhancedStandard:
    @pytest.mark.asyncio
    async def test_normal_flow_stores_once_with_conversation_tag(self):
        orch = _make_flow_orch()
        text, debug = await _run(orch, "tell me about python")
        assert text == "Hello world"
        assert debug["mode"] == "enhanced"
        orch.memory_system.store_interaction.assert_awaited_once()
        _, kwargs = orch.memory_system.store_interaction.await_args
        assert kwargs["tags"] == ["conversation"]
        assert kwargs["query"] == "tell me about python"

    @pytest.mark.asyncio
    async def test_debug_has_phase_timings_and_citation_flag(self):
        orch = _make_flow_orch()
        _, debug = await _run(orch, "tell me about python")
        assert set(debug["phase_timings"].keys()) == {
            "context_pipeline", "prompt_build", "llm_generation", "memory_store"
        }
        assert "citations_enabled" in debug
        assert debug["citations"] == []
        assert "task_timings" in debug
        assert "gather_elapsed" in debug

    @pytest.mark.asyncio
    async def test_escalation_tracker_update_and_record(self):
        esc = MagicMock()
        esc.get_debug_info = MagicMock(return_value={"strategy": "x"})
        esc.get_token_budget_override = MagicMock(return_value=None)
        orch = _make_flow_orch(escalation_tracker=esc)
        await _run(orch, "tell me about python")
        esc.update.assert_called_once()
        esc.record_response.assert_called_once()


# =============================================================================
# Token-limit logic (characterized via the max_tokens kwarg to generation)
# =============================================================================
class TestTokenLimits:
    async def _max_tokens_for(self, **ctx_over):
        from config.app_config import DEFAULT_MAX_TOKENS  # noqa: F401
        orch = _make_flow_orch(context=_make_context(**ctx_over))
        await _run(orch, "tell me about python")
        return orch.response_generator.generate_streaming_response.last_kwargs["max_tokens"]

    @pytest.mark.asyncio
    async def test_conversational(self):
        assert await self._max_tokens_for(tone_level=ToneLevel.CONVERSATIONAL) == 600

    @pytest.mark.asyncio
    async def test_concern(self):
        assert await self._max_tokens_for(tone_level=ToneLevel.CONCERN) == 1000

    @pytest.mark.asyncio
    async def test_elevated(self):
        assert await self._max_tokens_for(tone_level=ToneLevel.ELEVATED) == 1500

    @pytest.mark.asyncio
    async def test_crisis(self):
        assert await self._max_tokens_for(tone_level=ToneLevel.CRISIS) == 2000

    @pytest.mark.asyncio
    async def test_heavy_topic_overrides_tone(self):
        from config.app_config import HEAVY_TOPIC_MAX_TOKENS
        got = await self._max_tokens_for(
            tone_level=ToneLevel.CONVERSATIONAL, is_heavy_topic=True
        )
        assert got == HEAVY_TOPIC_MAX_TOKENS

    @pytest.mark.asyncio
    async def test_escalation_override_wins(self):
        esc = MagicMock()
        esc.get_debug_info = MagicMock(return_value={})
        esc.get_token_budget_override = MagicMock(return_value=42)
        esc.current_strategy = MagicMock()
        esc.current_strategy.value = "QUIET"
        orch = _make_flow_orch(escalation_tracker=esc)
        await _run(orch, "tell me about python")
        assert orch.response_generator.generate_streaming_response.last_kwargs["max_tokens"] == 42


# =============================================================================
# Document-generation bypass
# =============================================================================
class TestDocGenPath:
    @pytest.mark.asyncio
    async def test_doc_gen_success(self, monkeypatch):
        monkeypatch.setattr(
            "knowledge.document_generator.detect_document_intent",
            lambda _q: {"topic": "black holes", "doc_type": "report", "focus": None},
        )
        orch = _make_flow_orch(agentic_controller=_agentic_stream("Doc ", "body"))
        text, debug = await _run(orch, "write a report on black holes", use_agentic_search=True)
        assert text == "Doc body"
        assert debug["document_generation_used"] is True
        assert debug["doc_intent"]["topic"] == "black holes"
        orch.memory_system.store_interaction.assert_awaited_once()
        _, kwargs = orch.memory_system.store_interaction.await_args
        assert kwargs["tags"] == ["document_generation"]

    @pytest.mark.asyncio
    async def test_doc_gen_failure_falls_through(self, monkeypatch):
        monkeypatch.setattr(
            "knowledge.document_generator.detect_document_intent",
            lambda _q: {"topic": "x", "doc_type": "report"},
        )
        ctrl = MagicMock()
        ctrl.run_agentic_search = MagicMock(side_effect=RuntimeError("boom"))
        orch = _make_flow_orch(agentic_controller=ctrl)
        text, debug = await _run(orch, "write a report on x", use_agentic_search=True)
        # falls through to standard generation
        assert text == "Hello world"
        assert "doc_gen_error" in debug


# =============================================================================
# Agentic-search bypass
# =============================================================================
class TestAgenticPath:
    @pytest.mark.asyncio
    async def test_agentic_success(self, monkeypatch):
        monkeypatch.setattr(
            "knowledge.document_generator.detect_document_intent", lambda _q: None
        )
        decision = MagicMock()
        decision.should_search = True
        decision.search_terms = ["weather today"]
        monkeypatch.setattr(
            "utils.web_search_trigger.analyze_for_web_search_llm",
            AsyncMock(return_value=decision),
        )
        orch = _make_flow_orch(agentic_controller=_agentic_stream("Agentic ", "answer"))
        text, debug = await _run(orch, "search the web for weather", use_agentic_search=True)
        assert text == "Agentic answer"
        assert debug["agentic_search_used"] is True
        assert "phase_timings" in debug
        orch.memory_system.store_interaction.assert_awaited_once()
        _, kwargs = orch.memory_system.store_interaction.await_args
        assert kwargs["tags"] == ["agentic_search"]

    @pytest.mark.asyncio
    async def test_agentic_no_search_falls_through(self, monkeypatch):
        monkeypatch.setattr(
            "knowledge.document_generator.detect_document_intent", lambda _q: None
        )
        decision = MagicMock()
        decision.should_search = False
        decision.search_terms = []
        monkeypatch.setattr(
            "utils.web_search_trigger.analyze_for_web_search_llm",
            AsyncMock(return_value=decision),
        )
        orch = _make_flow_orch(agentic_controller=_agentic_stream("nope"))
        text, debug = await _run(orch, "just chatting", use_agentic_search=True)
        assert text == "Hello world"
        assert "agentic_search_used" not in debug


# =============================================================================
# Best-of / duel path
# =============================================================================
class TestBestOfPath:
    @pytest.mark.asyncio
    async def test_best_of_used(self):
        orch = _make_flow_orch()
        orch.best_of_handler.should_use_best_of = MagicMock(return_value=True)
        result = MagicMock()
        result.response = "best answer"
        result.mode = "best_of"
        result.used_best_of = True
        result.metadata = {}
        orch.best_of_handler.generate = AsyncMock(return_value=result)
        text, debug = await _run(orch, "tell me about python")
        assert text == "best answer"
        assert debug["best_of_mode"] == "best_of"
        assert debug["best_of_used"] is True


# =============================================================================
# Exception handling
# =============================================================================
class TestExceptionPath:
    @pytest.mark.asyncio
    async def test_build_context_error_reraises_and_logs(self):
        conv_logger = MagicMock()
        orch = _make_flow_orch(conversation_logger=conv_logger)
        orch.build_context = AsyncMock(side_effect=RuntimeError("kaboom"))
        with pytest.raises(RuntimeError, match="kaboom"):
            await _run(orch, "tell me about python")
        conv_logger.log_system_event.assert_called_once()


# =============================================================================
# build_full_prompt — system-prompt assembly (the Stage 1 extraction target)
# =============================================================================
def _make_bfp_orch(**overrides):
    """Orchestrator running the REAL build_full_prompt with a mocked prompt_builder."""
    orch = DaemonOrchestrator.__new__(DaemonOrchestrator)
    orch.logger = None
    orch.enable_citations = False
    orch.user_profile = None
    orch.escalation_tracker = None
    orch.response_planner = None
    orch.time_manager = None

    orch.model_manager = MagicMock()
    orch.model_manager.get_active_model_name = MagicMock(return_value="mock-model")
    orch.model_manager.supports_reasoning = MagicMock(return_value=False)

    orch.prompt_builder = MagicMock()
    orch.prompt_builder.build_prompt_from_context = AsyncMock(return_value={})
    orch.prompt_builder._assemble_prompt = MagicMock(return_value="ASSEMBLED_PROMPT")

    for k, v in overrides.items():
        setattr(orch, k, v)
    return orch


class TestBuildFullPrompt:
    @pytest.mark.asyncio
    async def test_returns_three_tuple_with_raw_context(self):
        orch = _make_bfp_orch()
        result = await orch.build_full_prompt(_make_context(), return_raw_context=True)
        assert len(result) == 3
        prompt, system_prompt, prompt_ctx = result
        assert prompt == "ASSEMBLED_PROMPT"
        assert isinstance(prompt_ctx, dict)

    @pytest.mark.asyncio
    async def test_returns_two_tuple_without_raw_context(self):
        orch = _make_bfp_orch()
        result = await orch.build_full_prompt(_make_context(), return_raw_context=False)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_sets_current_instance_attrs(self):
        orch = _make_bfp_orch()
        await orch.build_full_prompt(_make_context(), return_raw_context=True)
        assert orch._current_memory_id_map == {}
        assert orch._web_source_map == {}
        assert orch._current_response_plan is None

    @pytest.mark.asyncio
    async def test_topic_hint_always_present(self):
        orch = _make_bfp_orch()
        _, system_prompt, _ = await orch.build_full_prompt(
            _make_context(primary_topic="astrophysics"), return_raw_context=True
        )
        assert "Query topic: astrophysics" in system_prompt

    @pytest.mark.asyncio
    async def test_citation_block_present_when_enabled(self):
        orch = _make_bfp_orch(enable_citations=True)
        _, system_prompt, _ = await orch.build_full_prompt(
            _make_context(), return_raw_context=True
        )
        assert "MANDATORY MEMORY CITATION PROTOCOL" in system_prompt

    @pytest.mark.asyncio
    async def test_citation_block_absent_when_disabled(self):
        orch = _make_bfp_orch(enable_citations=False)
        _, system_prompt, _ = await orch.build_full_prompt(
            _make_context(), return_raw_context=True
        )
        assert "MANDATORY MEMORY CITATION PROTOCOL" not in system_prompt

    @pytest.mark.asyncio
    async def test_thread_context_present_when_has_thread(self):
        ctx = _make_context(
            thread_context={"thread_id": "t1", "thread_depth": 3, "thread_topic": "cars"}
        )
        orch = _make_bfp_orch()
        _, system_prompt, _ = await orch.build_full_prompt(ctx, return_raw_context=True)
        assert "[THREAD CONTEXT]" in system_prompt
        assert "cars" in system_prompt

    @pytest.mark.asyncio
    async def test_thinking_instruction_present_without_native_reasoning(self):
        orch = _make_bfp_orch()
        orch.model_manager.supports_reasoning = MagicMock(return_value=False)
        _, system_prompt, _ = await orch.build_full_prompt(
            _make_context(), return_raw_context=True
        )
        assert "<thinking>" in system_prompt

    @pytest.mark.asyncio
    async def test_thinking_instruction_absent_with_native_reasoning(self):
        orch = _make_bfp_orch()
        orch.model_manager.supports_reasoning = MagicMock(return_value=True)
        _, system_prompt, _ = await orch.build_full_prompt(
            _make_context(), return_raw_context=True
        )
        assert "<thinking>" not in system_prompt

    @pytest.mark.asyncio
    async def test_raw_mode_skips_thread_and_thinking(self):
        ctx = _make_context(thread_context={"thread_id": "t1", "thread_depth": 3})
        orch = _make_bfp_orch()
        _, system_prompt, _ = await orch.build_full_prompt(
            ctx, use_raw_mode=True, return_raw_context=True
        )
        assert "[THREAD CONTEXT]" not in system_prompt
        assert "<thinking>" not in system_prompt

    @pytest.mark.asyncio
    async def test_identity_placeholders_substituted(self):
        profile = MagicMock()
        profile.identity.name = "Sam"
        profile.identity.pronouns = "they/them"
        orch = _make_bfp_orch(user_profile=profile)
        orch.prompt_builder._assemble_prompt = MagicMock(
            side_effect=lambda context, user_input, system_prompt: system_prompt
        )
        # Use a personality that contains a placeholder by checking no raw placeholder leaks.
        _, system_prompt, _ = await orch.build_full_prompt(
            _make_context(), return_raw_context=True
        )
        assert "{USER_NAME}" not in system_prompt
        assert "{PRONOUN_SUBJ}" not in system_prompt
