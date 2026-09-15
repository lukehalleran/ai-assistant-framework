"""Regression tests for CGR-20260913-008 #101/#102 (F3b,
docs/execution/generalization/failure_outcome_design.md).

Root defects (verified against the deployed source before this batch):
`_keyword_search` (#102) collapses "no matches" and a raised store error
into a bare `[]` (dm18). `get_notes` (#101) does the same for its hybrid
read, and the combine step slices `keyword_results[:keyword_limit]` before
any status is read -- a list-subclass status/reason would silently drop on
that slice, so the fix reads `outcome_status(keyword_results)` immediately
after the keyword call, before the slice.

Drives the DEPLOYED `ObsidianManager` methods directly, never a
re-derivation. Isolation follows test_refdocs_failure_outcomes.py's pattern:
a `MagicMock` store whose `_get_collection()`/`query_collection()` are wired
directly, no real ChromaDB, embedder, vault or network.
"""
from unittest.mock import MagicMock

import pytest

from knowledge.obsidian_manager import ObsidianManager
from utils.retrieval_outcome import OutcomeList

# Distinctive markers so a privacy assertion proves absence, not luck.
PRIVATE_EXC_TEXT = "leaked exception detail EXCZQX9"
PRIVATE_QUERY = "sensitive question about TITLEZQX9 health"


def _mgr_with_collection(coll):
    store = MagicMock()
    store._get_collection = MagicMock(return_value=coll)
    return ObsidianManager(chroma_store=store, vault_path="/nonexistent")


def _mock_collection_returning(payload):
    coll = MagicMock()
    coll.get = MagicMock(return_value=payload)
    return coll


def _mgr_with_raising_get(exc):
    coll = MagicMock()
    coll.get = MagicMock(side_effect=exc)
    return _mgr_with_collection(coll)


def _mgr_with_unavailable_collection():
    store = MagicMock()
    store._get_collection = MagicMock(return_value=None)
    return ObsidianManager(chroma_store=store, vault_path="/nonexistent")


def _mgr_for_hybrid():
    return ObsidianManager(chroma_store=MagicMock(), vault_path="/nonexistent")


MATCHING_PAYLOAD = {
    "ids": ["c1"],
    "documents": ["Notes about the advising meeting and course plan."],
    "metadatas": [{"title": "Advising", "section": "", "tags": "",
                    "file_path": "school/advising.md"}],
}
EMPTY_PAYLOAD = {"ids": [], "documents": [], "metadatas": []}


def _keyword_item(title="Key Note"):
    return {"content": "keyword hit", "metadata": {"title": title, "section": ""},
            "relevance_score": 1.0, "match_type": "keyword"}


def _semantic_item(title="Sem Note"):
    return {"content": "semantic hit", "metadata": {"title": title, "section": ""},
            "relevance_score": 0.5}


def _semantic_item_with_image(title="Sem Note"):
    return {"content": "semantic hit",
            "metadata": {"title": title, "section": "", "images": "photo.png",
                          "file_path": "notes/photo.md"},
            "relevance_score": 0.5}


# #102 -- _keyword_search

class TestKeywordSearchOutcomes:
    def test_raising_collection_get_is_failed_and_empty(self):
        mgr = _mgr_with_raising_get(RuntimeError("boom"))
        result = mgr._keyword_search("advising", limit=10)
        assert isinstance(result, OutcomeList)
        assert result.status == "failed"
        assert result.reason == "RuntimeError"
        assert result == []

    def test_missing_collection_is_unavailable_and_empty(self):
        mgr = _mgr_with_unavailable_collection()
        result = mgr._keyword_search("advising", limit=10)
        assert isinstance(result, OutcomeList)
        assert result.status == "unavailable"
        assert result.reason == "collection_unavailable"
        assert result == []

    def test_control_matching_read_succeeds(self):
        mgr = _mgr_with_collection(_mock_collection_returning(MATCHING_PAYLOAD))
        result = mgr._keyword_search("advising", limit=10)
        assert isinstance(result, OutcomeList)
        assert result.status == "succeeded"
        assert result != []

    def test_control_no_match_is_no_results(self):
        mgr = _mgr_with_collection(_mock_collection_returning(MATCHING_PAYLOAD))
        result = mgr._keyword_search("zzz_no_such_topic_at_all", limit=10)
        assert isinstance(result, OutcomeList)
        assert result.status == "no_results"
        assert result == []


# #101 -- get_notes hybrid keyword + semantic

class TestGetNotesHybrid:
    @pytest.mark.asyncio
    async def test_keyword_failed_items_kept_reason_prefixed(self):
        mgr = _mgr_for_hybrid()
        mgr._keyword_search = MagicMock(return_value=OutcomeList.failed("RuntimeError"))
        mgr.chroma_store.query_collection = MagicMock(return_value=[_semantic_item()])

        result = await mgr.get_notes("q", limit=9)

        assert isinstance(result, OutcomeList)
        assert result.status == "failed"
        assert result.reason.startswith("keyword:")
        assert len(result) == 1  # the semantic item is kept

    @pytest.mark.asyncio
    async def test_semantic_raises_is_failed_and_empty(self):
        mgr = _mgr_for_hybrid()
        mgr._keyword_search = MagicMock(return_value=OutcomeList([_keyword_item()]))
        mgr.chroma_store.query_collection = MagicMock(side_effect=RuntimeError("boom"))

        result = await mgr.get_notes("q", limit=9)

        assert isinstance(result, OutcomeList)
        assert result.status == "failed"
        assert result == []

    @pytest.mark.asyncio
    async def test_keyword_search_raising_hits_outer_except(self):
        # Distinct from the above: here _keyword_search itself raises rather
        # than returning a failed OutcomeList, so the OUTER except in
        # get_notes fires -- reason is the bare exception class name, no
        # "keyword:" prefix.
        mgr = _mgr_for_hybrid()
        mgr._keyword_search = MagicMock(side_effect=RuntimeError("boom"))
        mgr.chroma_store.query_collection = MagicMock(return_value=[])

        result = await mgr.get_notes("q", limit=9)

        assert isinstance(result, OutcomeList)
        assert result.status == "failed"
        assert result.reason == "RuntimeError"
        assert result == []

    @pytest.mark.asyncio
    async def test_control_healthy_hybrid_succeeds(self):
        mgr = _mgr_for_hybrid()
        mgr._keyword_search = MagicMock(return_value=OutcomeList([_keyword_item()]))
        mgr.chroma_store.query_collection = MagicMock(return_value=[_semantic_item()])

        result = await mgr.get_notes("q", limit=9)

        assert isinstance(result, OutcomeList)
        assert result.status == "succeeded"
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_control_healthy_but_empty_is_no_results(self):
        mgr = _mgr_for_hybrid()
        mgr._keyword_search = MagicMock(return_value=OutcomeList([]))
        mgr.chroma_store.query_collection = MagicMock(return_value=[])

        result = await mgr.get_notes("q", limit=9)

        assert isinstance(result, OutcomeList)
        assert result.status == "no_results"
        assert result == []

    @pytest.mark.asyncio
    async def test_status_survives_include_images(self):
        # Contract: image loading runs on final_results BEFORE wrapping, and
        # the wrapped OutcomeList holds the same dicts (with image_data set).
        mgr = _mgr_for_hybrid()
        mgr._keyword_search = MagicMock(return_value=OutcomeList.failed("RuntimeError"))
        mgr.chroma_store.query_collection = MagicMock(
            return_value=[_semantic_item_with_image()])
        mgr.load_images_for_chunk = MagicMock(
            return_value=[{"data": "AAAA", "media_type": "image/png", "filename": "photo.png"}])

        result = await mgr.get_notes("q", limit=9, include_images=True)

        assert isinstance(result, OutcomeList)
        assert result.status == "failed"
        assert result.reason.startswith("keyword:")
        assert len(result) == 1
        assert result[0]["image_data"]
        mgr.load_images_for_chunk.assert_called_once()


# Privacy: no reason string carries the query or exception text

class TestPrivacyNoLeakedText:
    def test_keyword_search_failure_reason_has_no_leaked_text(self):
        mgr = _mgr_with_raising_get(RuntimeError(PRIVATE_EXC_TEXT))
        result = mgr._keyword_search(PRIVATE_QUERY)
        assert PRIVATE_EXC_TEXT not in result.reason
        assert PRIVATE_QUERY not in result.reason
        assert PRIVATE_EXC_TEXT not in str(result.reason)
        assert PRIVATE_QUERY not in str(result.reason)

    @pytest.mark.asyncio
    async def test_get_notes_keyword_failed_reason_has_no_leaked_text(self):
        mgr = _mgr_for_hybrid()
        mgr._keyword_search = MagicMock(return_value=OutcomeList.failed("RuntimeError"))
        mgr.chroma_store.query_collection = MagicMock(side_effect=RuntimeError(PRIVATE_EXC_TEXT))
        result = await mgr.get_notes(PRIVATE_QUERY)
        assert PRIVATE_EXC_TEXT not in result.reason
        assert PRIVATE_QUERY not in result.reason

    @pytest.mark.asyncio
    async def test_get_notes_outer_except_reason_has_no_leaked_text(self):
        mgr = _mgr_for_hybrid()
        mgr._keyword_search = MagicMock(side_effect=RuntimeError(PRIVATE_EXC_TEXT))
        result = await mgr.get_notes(PRIVATE_QUERY)
        assert PRIVATE_EXC_TEXT not in result.reason
        assert PRIVATE_QUERY not in result.reason


# Deployed consumer (read-only): today's behaviour is unchanged until F7 --
# an empty OutcomeList is falsy, so `notes or []` in get_personal_notes still
# collapses a failed read to a bare [].

class TestGathererConsumerUnaffected:
    @pytest.mark.asyncio
    async def test_get_personal_notes_with_failing_manager_returns_empty(self):
        import core.prompt.gatherer_knowledge as gk

        mgr = _mgr_for_hybrid()
        mgr._keyword_search = MagicMock(return_value=OutcomeList.failed("RuntimeError"))
        mgr.chroma_store.query_collection = MagicMock(side_effect=RuntimeError("boom"))

        g = gk.KnowledgeRetrievalMixin.__new__(gk.KnowledgeRetrievalMixin)
        g.obsidian_manager = mgr
        g.memory_id_map = {}

        result = await gk.KnowledgeRetrievalMixin.get_personal_notes(g, "q", 5)

        assert result == []
