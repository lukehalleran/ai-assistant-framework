"""Retrieval-time junk-memory guard (2026-07-15).

Historical docs stored BEFORE the 2026-07-03 storage-time guards —
API-error sentinel turns from Feb–March plus bare "test" exchanges — were
ranking in top-10 retrieval. These tests drive THE deployed predicate
(memory.utils.is_junk_conversation_doc) and THE deployed hybrid retrieval
path that now applies it.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from memory.utils import is_junk_conversation_doc
from memory.hybrid_retriever import HybridRetriever


class TestJunkPredicate:
    def test_api_unavailable_doc_content(self):
        assert is_junk_conversation_doc(
            content="User: hey\nAssistant: [API unavailable] Unable to reach the language model."
        )

    def test_api_error_doc_content(self):
        assert is_junk_conversation_doc(
            content="User: what's up\nAssistant: [API Error] 402 Insufficient credits"
        )

    def test_credits_exhausted_response_field(self):
        assert is_junk_conversation_doc(response="[CREDITS EXHAUSTED] add credits")

    def test_bare_test_exchange(self):
        assert is_junk_conversation_doc(content="User: test\nAssistant: Hello! How can I help?")
        assert is_junk_conversation_doc(query="Testing", response="I'm here!")

    def test_real_conversation_passes(self):
        assert not is_junk_conversation_doc(
            content="User: how do I test my API endpoint?\nAssistant: Use pytest with httpx."
        )

    def test_mentioning_error_inside_answer_passes(self):
        # Sentinel text quoted mid-answer is not a sentinel response
        assert not is_junk_conversation_doc(
            content="User: what does [API Error] mean?\nAssistant: It means the call failed."
        )

    def test_summary_shaped_content_passes(self):
        assert not is_junk_conversation_doc(
            content="Weekly summary: user worked on the retrieval pipeline."
        )

    def test_bare_sentinel_content_no_assistant_marker(self):
        assert is_junk_conversation_doc(content="[API unavailable] Unable to reach the model.")

    def test_box_test_battery_is_junk(self):
        # The owner's standing 4-message test battery had accreted 547 stored
        # conversation docs by 2026-07-25 and surfaced as [RELEVANT MEMORIES]
        # inside unrelated emotional conversations. Exact-match only.
        for q in (
            "Hey, I'm working on a Python project",
            "It's a RAG system with vector search",
            "Using ChromaDB for the vector store",
            "Should I add prompt caching?",
        ):
            assert is_junk_conversation_doc(
                content=f"User: {q}\nAssistant: some reply"
            ), q

    def test_similar_but_genuine_message_is_not_junk(self):
        # Near-misses must not match — the filter is exact-string only.
        assert not is_junk_conversation_doc(
            content="User: I'm working on my Python homework\nAssistant: what part?"
        )
        assert not is_junk_conversation_doc(
            content="User: Should I add prompt caching to the wiki path?\nAssistant: ..."
        )

    def test_empty_inputs_pass(self):
        assert not is_junk_conversation_doc()


def _store_result(doc_id, content, relevance=0.8):
    return {
        "id": doc_id,
        "content": content,
        "metadata": {"timestamp": "2026-03-01T12:00:00"},
        "relevance_score": relevance,
        "collection": "conversations",
        "rank": 1,
    }


class TestHybridRetrieverDropsJunk:
    @pytest.mark.asyncio
    async def test_junk_docs_never_reach_ranking(self):
        results = [
            _store_result("good", "User: how's my project going\nAssistant: Making progress on the gate."),
            _store_result("err1", "User: hey\nAssistant: [API unavailable] Unable to reach the language model.", 0.95),
            _store_result("err2", "User: hello\nAssistant: [API Error] 402 Insufficient credits", 0.94),
            _store_result("tst1", "User: test\nAssistant: Hi! I'm here."),
        ]
        r = HybridRetriever.__new__(HybridRetriever)
        r.chroma_store = MagicMock()
        r.chroma_store.query_multiple_collections = AsyncMock(
            return_value={"conversations": results}
        )
        r.semantic_weight = 0.7
        r.keyword_weight = 0.3
        r._fast_mode = False

        out = await r.retrieve("project status", limit=10)
        ids = {m["id"] for m in out}
        assert ids == {"good"}
