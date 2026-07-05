"""
Tests for the API-error storage guard in memory_storage.store_interaction().

Background (2026-07-03): a turn whose LLM call failed with a 402 stored the
literal "[API Error] Error code: 402 - Insufficient credits" sentinel as a
Daemon reply in the conversations collection — a transport failure persisted
as a false memory. store_interaction() now skips any response starting with
an API-error sentinel prefix (as emitted by model_manager._classify_api_error).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


@pytest.fixture
def storage():
    from memory.memory_storage import MemoryStorage
    corpus_manager = MagicMock()
    chroma_store = MagicMock()
    chroma_store.add_conversation_memory.return_value = "doc-123"
    ms = MemoryStorage(
        corpus_manager=corpus_manager,
        chroma_store=chroma_store,
        fact_extractor=MagicMock(),
    )
    return ms, corpus_manager, chroma_store


class TestApiErrorStorageGuard:
    @pytest.mark.asyncio
    async def test_api_error_response_not_stored(self, storage):
        ms, corpus_manager, chroma_store = storage
        result = await ms.store_interaction(
            query="hey how's it going",
            response="[API Error] Error code: 402 - Insufficient credits",
        )
        assert result is None
        assert not corpus_manager.add_entry.called
        assert not chroma_store.add_conversation_memory.called

    @pytest.mark.asyncio
    async def test_credits_exhausted_sentinel_not_stored(self, storage):
        ms, corpus_manager, chroma_store = storage
        result = await ms.store_interaction(
            query="what's the weather",
            response="[CREDITS EXHAUSTED] OpenRouter balance is empty.",
        )
        assert result is None
        assert not corpus_manager.add_entry.called
        assert not chroma_store.add_conversation_memory.called

    @pytest.mark.asyncio
    async def test_all_sentinel_prefixes_blocked(self, storage):
        ms, corpus_manager, _ = storage
        sentinels = [
            "[API Error] boom",
            "[API unavailable] no route",
            "[CREDITS EXHAUSTED] empty",
            "[RATE LIMITED] slow down",
            "[AUTH ERROR] bad key",
            "[MODEL NOT SUPPORTED] nope",
            "[MODEL NOT FOUND] gone",
            "[SERVER ERROR] 500",
        ]
        for s in sentinels:
            result = await ms.store_interaction(query="q", response=s)
            assert result is None, f"sentinel not blocked: {s!r}"
        assert not corpus_manager.add_entry.called

    @pytest.mark.asyncio
    async def test_leading_whitespace_still_blocked(self, storage):
        ms, corpus_manager, _ = storage
        result = await ms.store_interaction(
            query="q",
            response="  \n[API Error] Error code: 502",
        )
        assert result is None
        assert not corpus_manager.add_entry.called

    @pytest.mark.asyncio
    async def test_clean_response_still_stored(self, storage):
        ms, corpus_manager, _ = storage
        await ms.store_interaction(
            query="how's it going",
            response="All good over here — how was the exam?",
        )
        assert corpus_manager.add_entry.called
        stored_response = corpus_manager.add_entry.call_args[0][1]
        assert stored_response.startswith("All good")

    @pytest.mark.asyncio
    async def test_response_mentioning_error_in_body_still_stored(self, storage):
        """Only a sentinel PREFIX is a transport failure — a real answer that
        merely mentions an API error mid-text must persist normally."""
        ms, corpus_manager, _ = storage
        await ms.store_interaction(
            query="why did it fail",
            response="Your log shows [API Error] on the third retry, which means the key expired.",
        )
        assert corpus_manager.add_entry.called
