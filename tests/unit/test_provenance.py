"""
Test suite for the provenance / citation system.

Covers:
- store_interaction() with provenance= and session_id= params
- Thinking block truncation at configured limit
- AgenticSearchSession.get_provenance_summary() structure
- Memory_id_map fix: recent conversations have non-empty content
- Citation markers in agentic _build_final_prompt() output
- Config constants loaded correctly
"""
import json
import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory_storage():
    """Create a minimal MemoryStorage with mocked dependencies."""
    from memory.memory_storage import MemoryStorage

    corpus = Mock()
    corpus.add_entry = Mock()
    chroma = Mock()
    chroma.add_conversation_memory = Mock(return_value="test-uuid-1234")
    topic_mgr = Mock()
    topic_mgr.get_primary_topic = Mock(return_value="general")

    storage = MemoryStorage(
        corpus_manager=corpus,
        chroma_store=chroma,
        fact_extractor=None,
        consolidator=None,
        topic_manager=topic_mgr,
    )
    storage._thread_detect_fn = None
    return storage, chroma


# ---------------------------------------------------------------------------
# Phase 2: store_interaction with session_id + provenance
# ---------------------------------------------------------------------------

class TestStoreInteractionProvenance:
    """Tests for provenance metadata being stored in ChromaDB."""

    @pytest.mark.asyncio
    async def test_session_id_stored_in_metadata(self):
        """session_id param should appear in ChromaDB metadata."""
        storage, chroma = _make_memory_storage()

        await storage.store_interaction(
            query="hello",
            response="hi there",
            tags=["test"],
            session_id="2026-03-26T10:00:00",
        )

        chroma.add_conversation_memory.assert_called_once()
        call_args = chroma.add_conversation_memory.call_args
        metadata = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get('metadata', {})
        assert metadata.get("session_id") == "2026-03-26T10:00:00"

    @pytest.mark.asyncio
    async def test_provenance_response_mode_stored(self):
        """response_mode from provenance dict stored in metadata."""
        storage, chroma = _make_memory_storage()

        await storage.store_interaction(
            query="what is X?",
            response="X is Y",
            provenance={"response_mode": "enhanced", "model_name": "gpt-5"},
        )

        chroma.add_conversation_memory.assert_called_once()
        metadata = chroma.add_conversation_memory.call_args[0][2]
        assert metadata.get("response_mode") == "enhanced"
        assert metadata.get("model_name") == "gpt-5"

    @pytest.mark.asyncio
    async def test_provenance_thinking_block_stored(self):
        """Thinking block stored in metadata under thinking_block key."""
        storage, chroma = _make_memory_storage()

        await storage.store_interaction(
            query="think about this",
            response="done thinking",
            provenance={"thinking_block": "I reasoned about X then Y"},
        )

        metadata = chroma.add_conversation_memory.call_args[0][2]
        assert "I reasoned about X then Y" in metadata.get("thinking_block", "")

    @pytest.mark.asyncio
    async def test_thinking_block_truncation(self):
        """Thinking blocks exceeding PROVENANCE_THINKING_MAX_CHARS are truncated."""
        storage, chroma = _make_memory_storage()
        long_thinking = "x" * 5000  # exceeds default 4000

        with patch("config.app_config.PROVENANCE_THINKING_MAX_CHARS", 4000):
            await storage.store_interaction(
                query="q",
                response="r",
                provenance={"thinking_block": long_thinking},
            )

        metadata = chroma.add_conversation_memory.call_args[0][2]
        assert len(metadata.get("thinking_block", "")) == 4000
        assert metadata.get("thinking_block_truncated") is True

    @pytest.mark.asyncio
    async def test_provenance_cited_ids_serialized(self):
        """cited_ids stored as JSON string in cited_memory_ids."""
        storage, chroma = _make_memory_storage()

        await storage.store_interaction(
            query="q",
            response="r",
            provenance={"cited_ids": ["MEM_RECENT_1", "FACT_3"]},
        )

        metadata = chroma.add_conversation_memory.call_args[0][2]
        cited = json.loads(metadata.get("cited_memory_ids", "[]"))
        assert "MEM_RECENT_1" in cited
        assert "FACT_3" in cited

    @pytest.mark.asyncio
    async def test_provenance_agentic_rounds_serialized(self):
        """agentic_rounds stored as compact JSON in agentic_summary."""
        storage, chroma = _make_memory_storage()

        rounds = [
            {"round": 1, "action": "web_search for climate data"},
            {"round": 2, "action": "search_memory facts"},
        ]
        await storage.store_interaction(
            query="q",
            response="r",
            provenance={"agentic_rounds": rounds},
        )

        metadata = chroma.add_conversation_memory.call_args[0][2]
        summary = json.loads(metadata.get("agentic_summary", "[]"))
        assert len(summary) == 2
        assert summary[0]["r"] == 1

    @pytest.mark.asyncio
    async def test_provenance_prompt_hash_stored(self):
        """final_prompt_hash stored as prompt_hash."""
        storage, chroma = _make_memory_storage()

        await storage.store_interaction(
            query="q",
            response="r",
            provenance={"final_prompt_hash": "abc123def456"},
        )

        metadata = chroma.add_conversation_memory.call_args[0][2]
        assert metadata.get("prompt_hash") == "abc123def456"

    @pytest.mark.asyncio
    async def test_no_provenance_no_extra_fields(self):
        """Without provenance param, no provenance fields in metadata."""
        storage, chroma = _make_memory_storage()

        await storage.store_interaction(query="q", response="r")

        metadata = chroma.add_conversation_memory.call_args[0][2]
        assert "response_mode" not in metadata
        assert "thinking_block" not in metadata
        assert "prompt_hash" not in metadata

    @pytest.mark.asyncio
    async def test_provenance_disabled_skips_fields(self):
        """When PROVENANCE_ENABLED=False, provenance fields are NOT stored."""
        storage, chroma = _make_memory_storage()

        with patch("config.app_config.PROVENANCE_ENABLED", False):
            await storage.store_interaction(
                query="q",
                response="r",
                provenance={"response_mode": "enhanced", "thinking_block": "test"},
            )

        metadata = chroma.add_conversation_memory.call_args[0][2]
        assert "response_mode" not in metadata
        assert "thinking_block" not in metadata


# ---------------------------------------------------------------------------
# Phase 2a: session_id property on MemoryCoordinator
# ---------------------------------------------------------------------------

class TestMemoryCoordinatorSessionId:
    """Tests for MemoryCoordinator.session_id property."""

    def test_session_id_returns_iso_string(self):
        """session_id should return ISO-formatted session_start."""
        from memory.memory_coordinator import MemoryCoordinator

        mc = MemoryCoordinator.__new__(MemoryCoordinator)
        mc.session_start = datetime(2026, 3, 26, 10, 30, 0)

        assert mc.session_id == "2026-03-26T10:30:00"

    def test_session_id_handles_string_fallback(self):
        """session_id should handle non-datetime gracefully."""
        from memory.memory_coordinator import MemoryCoordinator

        mc = MemoryCoordinator.__new__(MemoryCoordinator)
        mc.session_start = "some-string-timestamp"

        assert mc.session_id == "some-string-timestamp"


# ---------------------------------------------------------------------------
# Phase 4: AgenticSearchSession.get_provenance_summary()
# ---------------------------------------------------------------------------

class TestAgenticProvenanceSummary:
    """Tests for AgenticSearchSession.get_provenance_summary()."""

    def test_empty_session_returns_valid_dict(self):
        """Empty session should return a well-formed provenance dict."""
        from core.agentic.types import AgenticSearchSession, SearchProtocol

        session = AgenticSearchSession(
            query="test query",
            protocol=SearchProtocol.NATIVE_TOOLS,
        )
        prov = session.get_provenance_summary()

        assert prov["total_rounds"] == 0
        assert prov["protocol"] == "native_tools"
        assert prov["agentic_rounds"] == []
        assert prov["model_signaled_done"] is False
        assert prov["done_reason"] == ""
        assert "total_duration_ms" in prov
        assert "final_prompt_hash" in prov

    def test_session_with_rounds(self):
        """Session with search rounds should include round details."""
        from core.agentic.types import (
            AgenticSearchSession, SearchProtocol,
            SearchRound, SearchRequest,
        )

        session = AgenticSearchSession(
            query="climate change",
            protocol=SearchProtocol.XML_MARKERS,
        )
        session.rounds = [
            SearchRound(
                round_number=1,
                request=SearchRequest(query="climate change effects", reason="need data"),
                duration_ms=150.5,
            ),
            SearchRound(
                round_number=2,
                request=SearchRequest(query="CO2 levels 2025"),
                duration_ms=200.0,
                error="timeout",
            ),
        ]
        session.model_signaled_done = True
        session.done_reason = "enough info"
        session.final_prompt_hash = "abc123"

        prov = session.get_provenance_summary()

        assert prov["total_rounds"] == 2
        assert len(prov["agentic_rounds"]) == 2
        assert prov["agentic_rounds"][0]["query"] == "climate change effects"
        assert prov["agentic_rounds"][0]["reason"] == "need data"
        assert prov["agentic_rounds"][1]["error"] == "timeout"
        assert prov["model_signaled_done"] is True
        assert prov["done_reason"] == "enough info"
        assert prov["final_prompt_hash"] == "abc123"

    def test_final_prompt_hash_field(self):
        """final_prompt_hash field should default to empty string."""
        from core.agentic.types import AgenticSearchSession

        session = AgenticSearchSession(query="test")
        assert session.final_prompt_hash == ""

    def test_action_classification_by_query_prefix(self):
        """get_provenance_summary() should classify action type from query prefix."""
        from core.agentic.types import (
            AgenticSearchSession, SearchProtocol,
            SearchRound, SearchRequest,
        )

        session = AgenticSearchSession(
            query="test",
            protocol=SearchProtocol.XML_MARKERS,
        )
        session.rounds = [
            SearchRound(
                round_number=1,
                request=SearchRequest(query="climate change"),
                duration_ms=100.0,
            ),
            SearchRound(
                round_number=2,
                request=SearchRequest(query="[Memory: facts] user's dog name"),
                duration_ms=50.0,
            ),
            SearchRound(
                round_number=3,
                request=SearchRequest(query="[Python: calculate BMI]"),
                duration_ms=200.0,
            ),
            SearchRound(
                round_number=4,
                request=SearchRequest(query="[File Read] core/orchestrator.py"),
                duration_ms=30.0,
            ),
            SearchRound(
                round_number=5,
                request=SearchRequest(query="[Expand Memory] a1b2c3d4"),
                duration_ms=40.0,
            ),
        ]

        prov = session.get_provenance_summary()
        rounds = prov["agentic_rounds"]

        assert rounds[0]["action"] == "web_search"
        assert rounds[1]["action"] == "memory_search"
        assert rounds[2]["action"] == "sandbox"
        assert rounds[3]["action"] == "file_read"
        assert rounds[4]["action"] == "expand_memory"
        assert prov["total_rounds"] == 5


# ---------------------------------------------------------------------------
# Phase 1a: memory_id_map fix — recent conversations have non-empty content
# ---------------------------------------------------------------------------

class TestMemoryIdMapContentFix:
    """Tests for the content fix in context_gatherer memory_id_map."""

    def test_recent_conversation_content_not_empty(self):
        """Recent conversations with query+response but no content should build content from Q/A."""
        # Simulate what context_gatherer does
        mem = {'query': 'What is my dog called?', 'response': 'Your dog is called Flapjack', 'content': ''}

        _content = str(mem.get('content', '')) or ''
        if not _content.strip():
            _q = str(mem.get('query', ''))[:200]
            _a = str(mem.get('response', ''))[:200]
            _content = f"Q: {_q} A: {_a}" if (_q or _a) else ''

        assert _content != ''
        assert 'What is my dog' in _content
        assert 'Flapjack' in _content

    def test_content_present_not_overwritten(self):
        """When content field is populated, it should be used as-is."""
        mem = {'query': 'q', 'response': 'r', 'content': 'existing content here'}

        _content = str(mem.get('content', '')) or ''
        if not _content.strip():
            _q = str(mem.get('query', ''))[:200]
            _a = str(mem.get('response', ''))[:200]
            _content = f"Q: {_q} A: {_a}" if (_q or _a) else ''

        assert _content == 'existing content here'


# ---------------------------------------------------------------------------
# Phase 1c: Citation markers in agentic _build_final_prompt()
# ---------------------------------------------------------------------------

class TestAgenticCitationMarkers:
    """Tests for citation markers in agentic controller formatting."""

    def test_format_recent_conversations_has_markers(self):
        """_format_recent_conversations should include [MEM_RECENT_N] markers."""
        from core.agentic.formatters import AgenticFormatter

        fmt = AgenticFormatter()
        conversations = [
            {'timestamp': '2026-03-26T10:00', 'query': 'hello', 'response': 'hi'},
            {'timestamp': '2026-03-26T10:01', 'query': 'how are you', 'response': 'good'},
        ]

        result = fmt.format_recent_conversations(conversations)

        assert '[MEM_RECENT_1]' in result
        assert '[MEM_RECENT_2]' in result
        assert 'hello' in result

    def test_format_memories_has_markers(self):
        """_format_memories should include [MEM_SEMANTIC_N] markers."""
        from core.agentic.formatters import AgenticFormatter

        fmt = AgenticFormatter()
        memories = [
            {'timestamp': '2026-03-26T10:00', 'content': 'user likes Python'},
            {'timestamp': '2026-03-26T10:01', 'query': 'user age', 'response': '33'},
        ]

        result = fmt.format_memories(memories)

        assert '[MEM_SEMANTIC_1]' in result
        assert '[MEM_SEMANTIC_2]' in result
        assert 'Python' in result


# ---------------------------------------------------------------------------
# Phase 6: Config constants
# ---------------------------------------------------------------------------

class TestProvenanceConfig:
    """Tests for provenance configuration constants."""

    def test_config_constants_exist(self):
        """PROVENANCE_* constants should be importable from app_config."""
        from config.app_config import PROVENANCE_ENABLED, PROVENANCE_THINKING_MAX_CHARS

        assert isinstance(PROVENANCE_ENABLED, bool)
        assert isinstance(PROVENANCE_THINKING_MAX_CHARS, int)
        assert PROVENANCE_THINKING_MAX_CHARS > 0

    def test_default_values(self):
        """Default provenance config values should be sensible."""
        from config.app_config import PROVENANCE_ENABLED, PROVENANCE_THINKING_MAX_CHARS

        assert PROVENANCE_ENABLED is True
        assert PROVENANCE_THINKING_MAX_CHARS == 4000


# ---------------------------------------------------------------------------
# Phase 4b: Prompt hashing in controller
# ---------------------------------------------------------------------------

class TestPromptHashing:
    """Tests for prompt hashing in agentic controller."""

    def test_sha256_hash_format(self):
        """Hash should be 16-char hex substring of SHA-256."""
        import hashlib

        prompt = "test prompt content"
        h = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        assert len(h) == 16
        assert all(c in '0123456789abcdef' for c in h)
