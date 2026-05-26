"""
Unit tests for LLM-based memory compression in builder.py.

Tests the _llm_compress_oversized() async pre-pass that replaces
middle-out character slicing for heavily oversized items (>= 3x
over their token limit) with LLM summaries.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model_manager():
    """Mock model manager with generate_once."""
    manager = Mock()
    manager.get_active_model_name = Mock(return_value="gpt-4o")
    manager.generate_once = AsyncMock(return_value="Compressed summary of the content.")
    return manager


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer that estimates ~1 token per 4 chars."""
    tokenizer = Mock()
    def _count(text, model_name):
        return max(1, len(text) // 4)
    tokenizer.count_tokens = Mock(side_effect=_count)
    return tokenizer


@pytest.fixture
def builder(mock_model_manager, mock_tokenizer):
    """Create a UnifiedPromptBuilder with mocked dependencies."""
    from core.prompt import UnifiedPromptBuilder
    coordinator = Mock()
    coordinator.corpus_manager = Mock()
    coordinator.corpus_manager.get_recent_memories = Mock(return_value=[])
    coordinator.get_summaries = Mock(return_value=[])
    return UnifiedPromptBuilder(
        memory_coordinator=coordinator,
        model_manager=mock_model_manager,
        tokenizer_manager=mock_tokenizer,
        token_budget=40000,
    )


def _make_item(content: str, **extra) -> dict:
    """Build a context item dict with the given content."""
    d = {"content": content}
    d.update(extra)
    return d


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLLMCompression:
    """Tests for _llm_compress_oversized."""

    @pytest.mark.asyncio
    async def test_disabled_returns_context_unchanged(self, builder):
        """When LLM_COMPRESSION_ENABLED is False, context passes through."""
        ctx = {"memories": [_make_item("x" * 8000)]}
        with patch("core.prompt.builder.LLM_COMPRESSION_ENABLED", False):
            result = await builder._llm_compress_oversized(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_no_model_manager_returns_unchanged(self, mock_tokenizer):
        """When model_manager is None, context passes through."""
        from core.prompt import UnifiedPromptBuilder
        coordinator = Mock()
        coordinator.corpus_manager = Mock()
        coordinator.corpus_manager.get_recent_memories = Mock(return_value=[])
        coordinator.get_summaries = Mock(return_value=[])
        b = UnifiedPromptBuilder(
            memory_coordinator=coordinator,
            model_manager=None,
            tokenizer_manager=mock_tokenizer,
            token_budget=40000,
        )
        ctx = {"memories": [_make_item("x" * 8000)]}
        result = await b._llm_compress_oversized(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_no_oversized_items_skips_compression(self, builder):
        """Items below threshold are not sent to LLM."""
        # 512 * 3 = 1536 tokens threshold for memories.
        # Item is ~250 tokens (1000 chars / 4), well below 1536.
        ctx = {"memories": [_make_item("x" * 1000)]}
        result = await builder._llm_compress_oversized(ctx)
        builder.model_manager.generate_once.assert_not_called()
        assert result["memories"][0]["content"] == "x" * 1000

    @pytest.mark.asyncio
    async def test_oversized_item_gets_compressed(self, builder):
        """Items >= 3x over limit get LLM compression."""
        # 512 max * 3 threshold = 1536 tokens needed.
        # 8000 chars / 4 = 2000 tokens > 1536 → should compress.
        big_content = "Important fact about the user. " * 250  # ~7500 chars
        ctx = {"memories": [_make_item(big_content)]}

        builder.model_manager.generate_once = AsyncMock(
            return_value="User has an important fact."
        )

        result = await builder._llm_compress_oversized(ctx)
        builder.model_manager.generate_once.assert_called_once()
        assert result["memories"][0]["content"] == "User has an important fact."

    @pytest.mark.asyncio
    async def test_preserves_item_metadata(self, builder):
        """Compression preserves non-text metadata fields."""
        big_content = "x" * 8000
        ctx = {"memories": [_make_item(big_content, metadata={"source": "test"}, id="abc123")]}

        builder.model_manager.generate_once = AsyncMock(
            return_value="Compressed summary preserving key facts about the user."
        )

        result = await builder._llm_compress_oversized(ctx)
        item = result["memories"][0]
        assert item["content"] == "Compressed summary preserving key facts about the user."
        assert item["metadata"] == {"source": "test"}
        assert item["id"] == "abc123"

    @pytest.mark.asyncio
    async def test_timeout_falls_back_gracefully(self, builder):
        """On timeout, item stays unchanged (middle-out handles it later)."""
        big_content = "y" * 8000

        async def _slow(*args, **kwargs):
            await asyncio.sleep(10)
            return "should not reach"

        builder.model_manager.generate_once = _slow
        ctx = {"memories": [_make_item(big_content)]}

        with patch("core.prompt.builder.LLM_COMPRESSION_TIMEOUT", 0.01):
            result = await builder._llm_compress_oversized(ctx)

        # Content unchanged — timeout caused fallback
        assert result["memories"][0]["content"] == big_content

    @pytest.mark.asyncio
    async def test_llm_error_falls_back_gracefully(self, builder):
        """On LLM error, item stays unchanged."""
        big_content = "z" * 8000

        builder.model_manager.generate_once = AsyncMock(side_effect=RuntimeError("API down"))
        ctx = {"memories": [_make_item(big_content)]}

        result = await builder._llm_compress_oversized(ctx)
        assert result["memories"][0]["content"] == big_content

    @pytest.mark.asyncio
    async def test_empty_llm_response_falls_back(self, builder):
        """If LLM returns empty/very short string, item stays unchanged."""
        big_content = "w" * 8000
        builder.model_manager.generate_once = AsyncMock(return_value="ok")
        ctx = {"memories": [_make_item(big_content)]}

        result = await builder._llm_compress_oversized(ctx)
        # "ok" is only 2 chars — below the 20-char guard
        assert result["memories"][0]["content"] == big_content

    @pytest.mark.asyncio
    async def test_max_batch_cap(self, builder):
        """Only max_batch items are compressed even if more qualify."""
        items = [_make_item("a" * 8000) for _ in range(12)]
        ctx = {"memories": items}

        call_count = 0
        async def _track(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return "compressed text that is long enough."

        builder.model_manager.generate_once = _track

        with patch("core.prompt.builder.LLM_COMPRESSION_MAX_BATCH", 3):
            await builder._llm_compress_oversized(ctx)

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_multiple_sections(self, builder):
        """Compression works across different context sections."""
        ctx = {
            "memories": [_make_item("m" * 8000)],
            "summaries": [_make_item("s" * 12000)],
            "reflections": [_make_item("r" * 100)],  # too small
        }

        call_count = 0
        async def _compress(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return "compressed content for this section."

        builder.model_manager.generate_once = _compress

        result = await builder._llm_compress_oversized(ctx)

        # memories (8000/4=2000 > 1536) and summaries (12000/4=3000 > 2400) should compress
        # reflections (100/4=25 < 2400) should not
        assert call_count == 2
        assert result["memories"][0]["content"] == "compressed content for this section."
        assert result["summaries"][0]["content"] == "compressed content for this section."
        assert result["reflections"][0]["content"] == "r" * 100

    @pytest.mark.asyncio
    async def test_skips_protected_sections(self, builder):
        """stm_summary, user_profile, narrative_state are never compressed."""
        ctx = {
            "stm_summary": {"intent": "test", "summary": "x" * 8000},
            "user_profile": "x" * 8000,
            "narrative_state": "x" * 8000,
        }

        result = await builder._llm_compress_oversized(ctx)
        builder.model_manager.generate_once.assert_not_called()

    @pytest.mark.asyncio
    async def test_string_item_in_list(self, builder):
        """Handles list items that are plain strings (not dicts)."""
        big_str = "a" * 8000
        ctx = {"memories": [big_str]}

        builder.model_manager.generate_once = AsyncMock(
            return_value="compressed string result here."
        )

        result = await builder._llm_compress_oversized(ctx)
        assert result["memories"][0] == "compressed string result here."

    @pytest.mark.asyncio
    async def test_prioritizes_largest_ratio(self, builder):
        """Items with highest ratio are compressed first when batch is limited."""
        # Both items must stay below skip_threshold (ratio_threshold * 2 = 6x max_tokens = 3072 tokens)
        # Item A: ~1600 tokens, ratio 1600/512 ≈ 3.1
        # Item B: ~2500 tokens, ratio 2500/512 ≈ 4.9 → higher priority
        item_a = _make_item("a" * 6400)   # ~1600 tokens
        item_b = _make_item("b" * 10000)  # ~2500 tokens
        ctx = {"memories": [item_a, item_b]}

        compressed_items = []
        async def _track_order(*args, **kwargs):
            # Extract which item was sent by checking the prompt content
            prompt = args[0] if args else kwargs.get("prompt", "")
            if "bbbb" in prompt:
                compressed_items.append("B")
            elif "aaaa" in prompt:
                compressed_items.append("A")
            return "compressed result that is long enough."

        builder.model_manager.generate_once = _track_order

        with patch("core.prompt.builder.LLM_COMPRESSION_MAX_BATCH", 1):
            await builder._llm_compress_oversized(ctx)

        # Only 1 item compressed (max_batch=1), should be B (higher ratio)
        assert len(compressed_items) == 1
        assert compressed_items[0] == "B"

    @pytest.mark.asyncio
    async def test_partial_batch_failure(self, builder):
        """If some items fail and others succeed, successful ones still apply."""
        item_a = _make_item("a" * 8000)
        item_b = _make_item("b" * 8000)
        ctx = {"memories": [item_a, item_b]}

        call_num = 0
        async def _partial_fail(*args, **kwargs):
            nonlocal call_num
            call_num += 1
            if call_num == 1:
                raise RuntimeError("API error")
            return "second item compressed successfully."

        builder.model_manager.generate_once = _partial_fail

        result = await builder._llm_compress_oversized(ctx)

        # One should have original content, one should be compressed
        contents = [item["content"] for item in result["memories"]]
        assert "second item compressed successfully." in contents
        # The other kept its original
        originals = [c for c in contents if c != "second item compressed successfully."]
        assert len(originals) == 1
        assert len(originals[0]) == 8000

    @pytest.mark.asyncio
    async def test_uses_configured_model(self, builder):
        """Compression uses LLM_COMPRESSION_MODEL, not the active model."""
        big_content = "x" * 8000
        ctx = {"memories": [_make_item(big_content)]}

        builder.model_manager.generate_once = AsyncMock(
            return_value="compressed output that is long enough."
        )

        with patch("core.prompt.builder.LLM_COMPRESSION_MODEL", "test-model-123"):
            await builder._llm_compress_oversized(ctx)

        call_kwargs = builder.model_manager.generate_once.call_args
        assert call_kwargs.kwargs.get("model_name") == "test-model-123"

    @pytest.mark.asyncio
    async def test_dict_with_text_key(self, builder):
        """Items with 'text' key instead of 'content' are handled."""
        # SEMANTIC_ITEM_MAX_TOKENS=800, threshold=800*3=2400 tokens
        # Need >2400 tokens → >9600 chars (at ~4 chars/token)
        big = "x" * 12000  # ~3000 tokens > 2400
        ctx = {"summaries": [{"text": big, "metadata": {"id": "1"}}]}

        builder.model_manager.generate_once = AsyncMock(
            return_value="compressed text key result here."
        )

        result = await builder._llm_compress_oversized(ctx)
        assert result["summaries"][0]["text"] == "compressed text key result here."
        assert result["summaries"][0]["metadata"] == {"id": "1"}
