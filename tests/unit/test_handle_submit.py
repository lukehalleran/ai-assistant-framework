"""Golden-path tests for gui/handlers.py handle_submit().

Covers all 6 mode paths: raw, duel, agentic, enhanced streaming,
uncertainty fallback, and review gate. These tests exist to catch
regressions before and after refactoring.
"""

import asyncio
import pytest
import sys
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _collect(async_gen):
    """Drain an async generator into a list."""
    results = []
    async for item in async_gen:
        results.append(item)
    return results


def _final_content(results):
    """Return the content of the last non-progress yield."""
    for r in reversed(results):
        if not r.get("is_progress") and not r.get("is_thinking"):
            return r.get("content", "")
    return ""


def _debug_record(results):
    """Return the first yield that carries a debug dict."""
    for r in results:
        if "debug" in r and r["debug"] is not None:
            return r["debug"]
    return None


def _async_gen_factory(items):
    """Create an async-generator-returning callable from a list of items."""
    async def _gen(*args, **kwargs):
        for item in items:
            yield item
    return _gen


# ---------------------------------------------------------------------------
# Orchestrator mock factory
# ---------------------------------------------------------------------------

def _make_orchestrator(
    *,
    agentic_enabled=False,
    duel_enabled=False,
    duel_gens=None,
    duel_sels=None,
    streaming_chunks=None,
    agentic_items=None,
    duel_result=None,
    raw_result=None,
    prepare_result=None,
    entity_resolver=None,
):
    """Build a fully-wired orchestrator mock.

    Only override the bits each test cares about.
    """
    orch = MagicMock()

    # --- config ---
    features = {
        "best_of_duel_mode": duel_enabled,
        "best_of_generator_models": duel_gens or [],
        "best_of_selector_models": duel_sels or [],
    }
    orch.config = {
        "agentic_search": {"enabled": agentic_enabled},
        "features": features,
    }

    # --- prepare_prompt ---
    if prepare_result is None:
        prepare_result = ("Test prompt", "Test system prompt", {})
    orch.prepare_prompt = AsyncMock(return_value=prepare_result)

    # --- process_user_query (raw mode) ---
    if raw_result is None:
        raw_result = ("Raw response", {})
    orch.process_user_query = AsyncMock(return_value=raw_result)

    # --- response_generator ---
    rg = MagicMock()
    if streaming_chunks is None:
        streaming_chunks = ["Hello", " world"]
    rg.generate_streaming_response = _async_gen_factory(streaming_chunks)
    if duel_result is None:
        duel_result = {
            "answer": "Duel answer",
            "thinking_a": "",
            "thinking_b": "",
            "model_a": "m1",
            "model_b": "m2",
            "winner": "A",
            "scores": {"A": 8.0, "B": 6.0},
        }
    rg.generate_duel_and_judge = AsyncMock(return_value=duel_result)
    orch.response_generator = rg

    # --- agentic controller ---
    ac = MagicMock()
    if agentic_items is None:
        agentic_items = ["Agentic answer"]
    ac.run_agentic_search = _async_gen_factory(agentic_items)
    ac._last_session = None
    ac._current_web_source_map = None
    orch.agentic_controller = ac
    orch._agentic_controller = ac

    # --- model manager ---
    mm = MagicMock()
    mm.get_active_model_name = MagicMock(return_value="test-model")
    mm.embed_model = None
    orch.model_manager = mm

    # --- memory system ---
    ms = MagicMock()
    ms.session_id = "test-session"
    ms.entity_resolver = entity_resolver
    ms.corpus_manager = MagicMock()
    ms.corpus_manager.get_recent_memories = MagicMock(return_value=[])
    ms.store_interaction = AsyncMock(return_value="mem-123")
    ms.chroma_store = MagicMock()
    orch.memory_system = ms
    orch.memory_coordinator = MagicMock()

    # --- prompt builder ---
    pb = MagicMock()
    pb.context_gatherer = MagicMock()
    pb.context_gatherer._fast_mode = False
    orch.prompt_builder = pb

    # --- misc attributes handlers.py reads ---
    orch.tokenizer_manager = None
    orch.enable_citations = False
    orch.current_topic = "general"
    orch._current_memory_id_map = {}
    orch._last_phase_timings = {}
    orch._last_task_timings = {}
    orch._last_gather_elapsed = 0.0
    orch._current_response_plan = None
    orch.response_planner = None
    orch._web_source_map = None

    return orch


# ---------------------------------------------------------------------------
# Fake ProcessedFilesResult (avoid importing real one and its heavy deps)
# ---------------------------------------------------------------------------

@dataclass
class _FakeProcessedFilesResult:
    text_content: str = ""
    processed_files: list = field(default_factory=list)
    images: list = field(default_factory=list)
    documents: list = field(default_factory=list)


def _make_file_processor_mock(user_text="test"):
    mock = MagicMock()
    mock.process_files_structured = AsyncMock(
        return_value=_FakeProcessedFilesResult(text_content=user_text)
    )
    return mock


# ---------------------------------------------------------------------------
# Common patches applied to every test
# ---------------------------------------------------------------------------

def _base_patches():
    """Return a list of patch context managers common to every test."""
    return [
        patch("gui.handlers.file_processor", _make_file_processor_mock()),
        patch("gui.handlers.get_conversation_logger", return_value=MagicMock()),
    ]


async def _run_submit(user_text, orchestrator, *, use_raw_gpt=False, fast_mode=False, extra_patches=None):
    """Run handle_submit with base patches and collect all yields."""
    from gui.handlers import handle_submit

    patches = _base_patches()
    if extra_patches:
        patches.extend(extra_patches)

    # Stack all patches
    for p in patches:
        p.start()

    # Patch file processor to return the actual user_text
    fp = _make_file_processor_mock(user_text)
    patch("gui.handlers.file_processor", fp).start()

    try:
        gen = handle_submit(
            user_text=user_text,
            files=None,
            history=[],
            use_raw_gpt=use_raw_gpt,
            orchestrator=orchestrator,
            fast_mode=fast_mode,
        )
        results = await _collect(gen)
    finally:
        patch.stopall()

    return results


# ===========================================================================
# TESTS
# ===========================================================================


class TestEnhancedStreaming:
    """Test 1: Standard enhanced streaming (no agentic, no duel)."""

    @pytest.mark.asyncio
    async def test_enhanced_streaming(self):
        orch = _make_orchestrator(streaming_chunks=["Hello", " world"])
        results = await _run_submit("How are you?", orch)

        debug = _debug_record(results)
        assert debug is not None, "Expected debug record in yields"
        assert debug["mode"] == "enhanced"

        content = _final_content(results)
        assert "Hello" in content
        assert "world" in content

    @pytest.mark.asyncio
    async def test_prepare_prompt_called(self):
        orch = _make_orchestrator()
        await _run_submit("How are you?", orch)
        orch.prepare_prompt.assert_awaited_once()


class TestRawMode:
    """Test 2: Raw mode bypass."""

    @pytest.mark.asyncio
    async def test_raw_mode(self):
        orch = _make_orchestrator(raw_result=("Raw response text", {}))
        results = await _run_submit("test", orch, use_raw_gpt=True)

        debug = _debug_record(results)
        assert debug is not None
        assert debug["mode"] == "raw"

        content = _final_content(results)
        assert content == "Raw response text"

    @pytest.mark.asyncio
    async def test_raw_mode_skips_prepare_prompt(self):
        orch = _make_orchestrator()
        await _run_submit("test", orch, use_raw_gpt=True)
        orch.prepare_prompt.assert_not_awaited()


class TestDuelMode:
    """Test 3: Duel mode routing."""

    @pytest.mark.asyncio
    async def test_duel_mode(self):
        orch = _make_orchestrator(
            duel_enabled=True,
            duel_gens=["model-a", "model-b"],
            duel_sels=["judge-model"],
        )
        results = await _run_submit("What is the meaning of life?", orch)

        debug = _debug_record(results)
        assert debug is not None
        assert debug["mode"] == "best-of-duel"

        content = _final_content(results)
        assert content == "Duel answer"

        orch.response_generator.generate_duel_and_judge.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_duel_mode_yields_thinking(self):
        orch = _make_orchestrator(
            duel_enabled=True,
            duel_gens=["model-a", "model-b"],
            duel_sels=["judge-model"],
        )
        results = await _run_submit("test", orch)

        thinking_yields = [r for r in results if "thinking" in r and r["thinking"]]
        assert len(thinking_yields) >= 1
        td = thinking_yields[0]["thinking"]
        assert td["winner"] == "A"


class TestAgenticComputation:
    """Test 4: Agentic trigger via computation keyword."""

    @pytest.mark.asyncio
    async def test_agentic_computation(self):
        # Response must have >= 4 words per line to survive bare-tool-call filter
        orch = _make_orchestrator(
            agentic_enabled=True,
            agentic_items=["The 10th Fibonacci number is 55."],
        )
        results = await _run_submit("calculate fibonacci 10", orch)

        debug = _debug_record(results)
        assert debug is not None
        assert debug["mode"] == "agentic-search"

        content = _final_content(results)
        assert "55" in content


class TestAgenticMemory:
    """Test 5: Agentic trigger via memory keyword."""

    @pytest.mark.asyncio
    async def test_agentic_memory(self):
        orch = _make_orchestrator(
            agentic_enabled=True,
            agentic_items=["Your brother's name is Auggie, as you mentioned previously."],
        )
        results = await _run_submit("do you remember my brother's name?", orch)

        debug = _debug_record(results)
        assert debug is not None
        assert debug["mode"] == "agentic-search"


class TestAgenticEntityRecall:
    """Test 6: Agentic trigger via entity match + recall signal."""

    @pytest.mark.asyncio
    async def test_entity_recall(self):
        orch = _make_orchestrator(
            agentic_enabled=True,
            agentic_items=["Flapjack is your cat, a senior feline you care for deeply."],
        )
        # Mock entity resolver so Tier 2 fires
        resolver = MagicMock()
        orch.memory_system.entity_resolver = resolver

        with patch("memory.graph_utils.extract_graph_entities", return_value={"flapjack"}):
            results = await _run_submit("what do you know about Flapjack?", orch)

        debug = _debug_record(results)
        assert debug is not None
        assert debug["mode"] == "agentic-search"


class TestCasualSkipsAgentic:
    """Test 7: Casual/short message skips agentic even when enabled."""

    @pytest.mark.asyncio
    async def test_casual_skips(self):
        orch = _make_orchestrator(
            agentic_enabled=True,
            streaming_chunks=["You're welcome!"],
        )
        results = await _run_submit("thanks", orch)

        debug = _debug_record(results)
        assert debug is not None
        assert debug["mode"] == "enhanced"


def _no_trigger_decision():
    """Return a WebSearchDecision-like object that triggers nothing."""
    return MagicMock(
        should_search=False,
        search_terms=[],
        needs_memory_search=False,
        needs_knowledge_search=False,
        needs_document_generation=False,
    )


class TestUncertaintyFallback:
    """Tests 8-9: Uncertainty detection and retry.

    These tests need agentic_enabled=True (uncertainty gate checks it) but
    must NOT trigger the agentic gate itself. Use a query that avoids all
    keyword lists and patch the LLM fallback to return no-trigger.
    """

    # Query: >= 5 words, has '?', but matches no agentic keyword list
    _QUERY = "Can you describe the color of my cat's fur exactly?"

    @pytest.mark.asyncio
    async def test_uncertainty_accepts_retry(self):
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
                new_callable=AsyncMock,
                return_value=_no_trigger_decision(),
            ),
            patch(
                "gui.handlers._silent_agentic_retry",
                new_callable=AsyncMock,
                return_value=("Here is the information you wanted about your cat.", ""),
            ),
        ]
        results = await _run_submit(self._QUERY, orch, extra_patches=extra)

        content = _final_content(results)
        assert "information you wanted" in content

    @pytest.mark.asyncio
    async def test_uncertainty_rejects_similar(self):
        uncertain_text = "I don't have any information about that topic."
        orch = _make_orchestrator(
            agentic_enabled=True,
            streaming_chunks=[uncertain_text],
        )

        extra = [
            patch("config.app_config.UNCERTAINTY_FALLBACK_ENABLED", True),
            patch("config.app_config.UNCERTAINTY_SEMANTIC_THRESHOLD", 0.7),
            patch("config.app_config.UNCERTAINTY_MAX_LENGTH", 1200),
            patch(
                "utils.web_search_trigger.analyze_for_web_search_llm",
                new_callable=AsyncMock,
                return_value=_no_trigger_decision(),
            ),
            # Retry returns None (too similar / rejected by overlap check)
            patch(
                "gui.handlers._silent_agentic_retry",
                new_callable=AsyncMock,
                return_value=(None, None),
            ),
        ]
        results = await _run_submit(self._QUERY, orch, extra_patches=extra)

        content = _final_content(results)
        # Original uncertain response should be kept
        assert "don't have" in content or "information" in content


class TestReviewGate:
    """Test 10: Post-answer review gate triggers retry.

    Needs agentic_enabled=True but must NOT trigger agentic gate.
    Response must be >= 120 chars for review gate to fire.
    """

    # Query that avoids all keyword lists but is long enough to not be casual-skipped
    _QUERY = "Can you describe the color of my cat's fur and general appearance in some detail?"

    @pytest.mark.asyncio
    async def test_review_gate_retry(self):
        # Response must be >= 120 chars for review gate
        initial_response = "The cat has some fur. " * 8  # ~176 chars
        orch = _make_orchestrator(
            agentic_enabled=True,
            streaming_chunks=[initial_response],
        )

        # Set up response planner on orchestrator
        mock_planner = MagicMock()
        review_result = MagicMock()
        review_result.passes = False
        review_result.confidence = 0.95
        review_result.issues = ["missing detail"]
        review_result.suggestion = "Add more context"
        mock_planner.review_answer = AsyncMock(return_value=review_result)
        orch.response_planner = mock_planner

        mock_plan = MagicMock()
        orch._current_response_plan = mock_plan

        extra = [
            patch("config.app_config.RESPONSE_REVIEW_ENABLED", True),
            patch("config.app_config.RESPONSE_REVIEW_CONFIDENCE_THRESHOLD", 0.9),
            # Uncertainty must be disabled to reach the review gate
            patch("config.app_config.UNCERTAINTY_FALLBACK_ENABLED", False),
            # Prevent agentic trigger via LLM fallback
            patch(
                "utils.web_search_trigger.analyze_for_web_search_llm",
                new_callable=AsyncMock,
                return_value=_no_trigger_decision(),
            ),
            patch(
                "gui.handlers._silent_agentic_retry",
                new_callable=AsyncMock,
                return_value=("Here is a much better and complete answer with full context and detail.", ""),
            ),
        ]
        results = await _run_submit(self._QUERY, orch, extra_patches=extra)

        content = _final_content(results)
        assert "better and complete" in content


class TestEmptyResponse:
    """Test 11: Empty response from model yields error message."""

    @pytest.mark.asyncio
    async def test_empty_response(self):
        orch = _make_orchestrator(streaming_chunks=[])
        results = await _run_submit("test query", orch)

        content = _final_content(results)
        assert "empty response" in content.lower() or "⚠️" in content


class TestStorageOnce:
    """Test 12: Storage dispatched exactly once for a successful response."""

    @pytest.mark.asyncio
    async def test_storage_dispatched_once(self):
        orch = _make_orchestrator(streaming_chunks=["A real response."])

        with patch("gui.handlers._dispatch_storage") as mock_dispatch:
            results = await _run_submit("How are you?", orch)

        assert mock_dispatch.call_count == 1
        call_kwargs = mock_dispatch.call_args
        # Verify mode is correct
        assert call_kwargs[1].get("mode", call_kwargs[0][-1] if call_kwargs[0] else None) is not None
