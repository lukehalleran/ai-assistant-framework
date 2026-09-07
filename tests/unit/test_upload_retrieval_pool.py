"""Regression tests for the 2026-09-07 upload-retrieval contract (Delegate A,
docs/HANDOFF_20260907_upload_reuse_contracts.md, items A1-A5).

Root cause (R1): `ContextGatherer.get_user_uploads` pooled
`ReferenceDocsManager.get_documents(query, limit=10)` over the WHOLE
`reference_docs` collection (1,644 chunks, 242 uploads) with no type filter,
then filtered to `type == 'user_upload'` AFTER the pool was already capped —
so a fresh, relevant upload could lose outright to unrelated reference_doc
chunks and never even reach the filter. Root cause (R2): the search_memory
tool definition never told the model that reference_docs also holds the
user's uploaded files, so the agentic loop searched everywhere else and
never surfaced upload titles.

Fixes under test (exercising the DEPLOYED functions, no re-derivation):
  A1. `MultiCollectionChromaStore.query_collection(..., where=...)` forwards
      an explicit `where` kwarg to the underlying `.query()` call.
  A2. `ReferenceDocsManager.get_documents`/`_keyword_search` gain a
      keyword-only `doc_type` parameter restricting BOTH hybrid legs.
  A3. `get_user_uploads` calls `get_documents(..., doc_type="user_upload")`
      so the pool is upload-only before any staleness/relevance filtering.
  A4. A fresh-upload title/date roster is attached to the uploads result
      whenever the query carries a document cue or a filename-shaped token,
      even when neither hybrid leg admits anything.
  A5. The search_memory tool definition documents reference_docs as also
      holding uploads, and the agentic memory-search formatter already
      shows the title for reference_docs hits.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from core.agentic.formatters import AgenticFormatter
from core.agentic.types import MEMORY_SEARCH_TOOL_DEFINITION
from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin
from knowledge.reference_docs_manager import ReferenceDocsManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore


# ===========================================================================
# A1: MultiCollectionChromaStore.query_collection `where` forwarding
# ===========================================================================

class _FakeCollection:
    def __init__(self, ids=None, docs=None, metas=None, dists=None):
        self.ids = ids or ["a"]
        self.docs = docs or ["content"]
        self.metas = metas or [{}]
        self.dists = dists or [0.1]
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "ids": [self.ids],
            "documents": [self.docs],
            "metadatas": [self.metas],
            "distances": [self.dists],
        }


def _store_with_fake_collection(fake_coll):
    store = MultiCollectionChromaStore.__new__(MultiCollectionChromaStore)
    store.collections = {"reference_docs": object()}
    store._embedding_cache = {}
    store._get_collection = lambda name: fake_coll
    return store


class TestQueryCollectionWhereForwarding:
    def test_where_forwarded_when_given(self):
        fake = _FakeCollection()
        store = _store_with_fake_collection(fake)
        store.query_collection("reference_docs", "q", n_results=5, where={"type": "user_upload"})
        assert fake.calls[-1].get("where") == {"type": "user_upload"}

    def test_where_absent_when_not_given(self):
        fake = _FakeCollection()
        store = _store_with_fake_collection(fake)
        store.query_collection("reference_docs", "q", n_results=5)
        assert "where" not in fake.calls[-1]

    def test_where_none_explicit_not_forwarded(self):
        # Default None must be byte-identical to omitting the kwarg entirely.
        fake = _FakeCollection()
        store = _store_with_fake_collection(fake)
        store.query_collection("reference_docs", "q", n_results=5, where=None)
        assert "where" not in fake.calls[-1]

    def test_result_shape_unaffected(self):
        fake = _FakeCollection(ids=["x"], docs=["hello"], metas=[{"type": "user_upload"}], dists=[0.2])
        store = _store_with_fake_collection(fake)
        results = store.query_collection("reference_docs", "q", n_results=5, where={"type": "user_upload"})
        assert len(results) == 1
        assert results[0]["content"] == "hello"
        assert results[0]["metadata"] == {"type": "user_upload"}


# ===========================================================================
# A2: ReferenceDocsManager.get_documents / _keyword_search doc_type filter
# ===========================================================================

def _manager_with_docs(docs_and_metas):
    """Same isolation style as tests/unit/test_upload_keyword_score_leak.py:
    a ReferenceDocsManager whose _collection() returns a fake ChromaDB
    collection backed by (document_text, metadata) pairs."""
    mgr = ReferenceDocsManager(chroma_store=MagicMock())
    documents = [d for d, _ in docs_and_metas]
    metadatas = [m for _, m in docs_and_metas]
    fake_collection = MagicMock()
    fake_collection.get = MagicMock(return_value={
        "documents": documents,
        "metadatas": metadatas,
    })
    mgr._collection = MagicMock(return_value=fake_collection)
    return mgr


class TestKeywordSearchDocTypeFilter:
    def test_doc_type_restricts_to_matching_type(self):
        mgr = _manager_with_docs([
            ("Homework instructions for assignment one.",
             {"title": "upload:Homework1-2.pdf", "section": "", "type": "user_upload"}),
            ("Architecture notes about memory scoring.",
             {"title": "Memory Scoring Architecture", "section": "", "type": "reference_doc"}),
        ])
        results = mgr._keyword_search("homework assignment", limit=10, doc_type="user_upload")
        assert len(results) == 1
        assert results[0]["metadata"]["title"] == "upload:Homework1-2.pdf"

    def test_doc_type_none_returns_both_types(self):
        mgr = _manager_with_docs([
            ("Homework instructions for assignment one.",
             {"title": "upload:Homework1-2.pdf", "section": "", "type": "user_upload"}),
            ("Homework assignment reference archive.",
             {"title": "Homework Archive", "section": "", "type": "reference_doc"}),
        ])
        results = mgr._keyword_search("homework assignment", limit=10, doc_type=None)
        titles = {r["metadata"]["title"] for r in results}
        assert titles == {"upload:Homework1-2.pdf", "Homework Archive"}


class TestGetDocumentsDocTypeWiring:
    @pytest.mark.asyncio
    async def test_doc_type_forwarded_to_both_legs(self):
        mgr = ReferenceDocsManager(chroma_store=MagicMock())
        mgr._keyword_search = MagicMock(return_value=[])
        captured = {}

        def fake_query_collection(name, query, n_results=5, where=None, **kw):
            captured["where"] = where
            return []

        mgr.chroma_store.query_collection = MagicMock(side_effect=fake_query_collection)

        await mgr.get_documents("q", limit=9, doc_type="user_upload")

        assert captured["where"] == {"type": "user_upload"}
        _, kwargs = mgr._keyword_search.call_args
        assert kwargs.get("doc_type") == "user_upload"

    @pytest.mark.asyncio
    async def test_doc_type_none_is_byte_identical_to_omitted(self):
        """Regression pin: doc_type=None must forward where=None (never
        added to query_collection's kwargs per A1) and doc_type=None to the
        keyword leg — i.e. every existing caller of get_documents(query,
        limit) keeps working unchanged."""
        mgr = ReferenceDocsManager(chroma_store=MagicMock())
        mgr._keyword_search = MagicMock(return_value=[])
        captured = {}

        def fake_query_collection(name, query, n_results=5, where=None, **kw):
            captured["where"] = where
            return []

        mgr.chroma_store.query_collection = MagicMock(side_effect=fake_query_collection)

        await mgr.get_documents("q", limit=9)

        assert captured["where"] is None
        _, kwargs = mgr._keyword_search.call_args
        assert kwargs.get("doc_type") is None


# ===========================================================================
# A3: get_user_uploads pools ONLY user_upload chunks
# ===========================================================================

class _FakeChromaStoreForPool:
    """Minimal in-memory stand-in for chroma semantic search: without a
    `where` filter, non-upload chunks (there are far more of them) fill the
    entire n_results window and the single upload chunk never appears —
    the PRE-FIX shape (get_documents pooled the whole reference_docs
    collection with no type filter, so a relevant fresh upload could lose
    outright to 1,000+ unrelated chunks before get_user_uploads' own
    type/staleness filters ever ran). With a `where={"type": "user_upload"}`
    filter, only the upload chunk is eligible."""

    def __init__(self, chunks):
        self.chunks = chunks  # list of (id, content, metadata)

    def query_collection(self, collection_name, query_text, n_results=5, where=None, **kwargs):
        pool = self.chunks
        if where:
            key, val = next(iter(where.items()))
            pool = [c for c in pool if c[2].get(key) == val]
        else:
            # Simulate the junk reference_doc chunks "outranking" the upload
            # in an unfiltered pool.
            pool = sorted(pool, key=lambda c: 0 if c[2].get("type") != "user_upload" else 1)
        top = pool[:n_results]
        return [
            {"id": cid, "content": content, "metadata": meta, "relevance_score": 0.9,
             "collection": collection_name, "rank": i + 1}
            for i, (cid, content, meta) in enumerate(top)
        ]


class TestUploadPoolRestriction:
    @pytest.mark.asyncio
    async def test_fresh_upload_admitted_despite_1000_outranking_chunks(self):
        junk_chunks = [
            (f"junk-{i}", f"junk content {i}", {"type": "reference_doc", "title": f"Doc {i}"})
            for i in range(1000)
        ]
        fresh_ts = datetime.now().isoformat()
        upload_chunk = (
            "upload-1",
            "Homework 1 instructions: complete problems 1-10.",
            {"type": "user_upload", "title": "upload:Homework1-2.pdf", "timestamp": fresh_ts},
        )
        fake_store = _FakeChromaStoreForPool(junk_chunks + [upload_chunk])
        mgr = ReferenceDocsManager(chroma_store=fake_store)
        mgr._keyword_search = MagicMock(return_value=[])  # isolate the semantic-leg fix

        fake_coll = MagicMock()
        fake_coll.get = MagicMock(return_value={"ids": ["upload-1"], "metadatas": [upload_chunk[2]]})
        mgr.chroma_store._get_collection = MagicMock(return_value=fake_coll)

        gatherer = KnowledgeRetrievalMixin()
        gatherer.memory_id_map = {}
        gatherer._current_turn_upload_filenames = []
        gatherer.reference_docs_manager = mgr

        uploads = await gatherer.get_user_uploads("pull up the first assignment", limit=5)

        titles = [
            u.get("metadata", {}).get("title") for u in uploads
            if isinstance(u, dict) and u.get("metadata", {}).get("type") == "user_upload"
        ]
        assert "upload:Homework1-2.pdf" in titles


# ===========================================================================
# A4: fresh-upload roster
# ===========================================================================

class _FakeRosterCollection:
    def __init__(self, metadatas):
        self._metadatas = metadatas
        self.calls = []

    def get(self, **kwargs):
        self.calls.append(kwargs)
        return {"ids": ["x"] * len(self._metadatas), "metadatas": self._metadatas}


def _roster_gatherer(metadatas):
    gatherer = KnowledgeRetrievalMixin()
    gatherer.memory_id_map = {}
    gatherer._current_turn_upload_filenames = []
    manager = Mock()
    manager.get_documents = AsyncMock(return_value=[])
    fake_coll = _FakeRosterCollection(metadatas)
    manager.chroma_store = Mock()
    manager.chroma_store._get_collection = Mock(return_value=fake_coll)
    gatherer.reference_docs_manager = manager
    return gatherer, fake_coll


def _upload_meta(title, age_hours=1.0, is_image=False):
    ts = (datetime.now() - timedelta(hours=age_hours)).isoformat()
    meta = {"type": "user_upload", "title": title, "timestamp": ts}
    if is_image:
        meta["is_image"] = True
    return meta


class TestUploadRoster:
    @pytest.mark.asyncio
    async def test_roster_present_on_document_cue(self):
        metas = [_upload_meta("upload:Homework1-2.pdf", age_hours=1)]
        gatherer, _ = _roster_gatherer(metas)
        uploads = await gatherer.get_user_uploads("pull up the first assignment", limit=5)
        assert gatherer._last_upload_roster
        assert gatherer._last_upload_roster[0]["title"] == "Homework1-2.pdf"
        roster_items = [
            u for u in uploads
            if isinstance(u, dict) and u.get("metadata", {}).get("type") == "upload_roster"
        ]
        assert roster_items

    @pytest.mark.asyncio
    async def test_roster_present_on_user_uploads_mention(self):
        metas = [_upload_meta("upload:notes.txt", age_hours=1)]
        gatherer, _ = _roster_gatherer(metas)
        await gatherer.get_user_uploads("look in the user uploads", limit=5)
        assert gatherer._last_upload_roster

    @pytest.mark.asyncio
    async def test_roster_present_on_filename_token(self):
        metas = [_upload_meta("upload:UsedCars2.csv", age_hours=1)]
        gatherer, _ = _roster_gatherer(metas)
        await gatherer.get_user_uploads("what's in UsedCars2.csv", limit=5)
        assert gatherer._last_upload_roster

    @pytest.mark.asyncio
    async def test_roster_absent_without_document_cue(self):
        metas = [_upload_meta("upload:Homework1-2.pdf", age_hours=1)]
        gatherer, _ = _roster_gatherer(metas)
        uploads = await gatherer.get_user_uploads("how are the cats", limit=5)
        assert gatherer._last_upload_roster == []
        assert not any(
            isinstance(u, dict) and u.get("metadata", {}).get("type") == "upload_roster"
            for u in uploads
        )

    @pytest.mark.asyncio
    async def test_roster_capped_at_8_newest_first(self):
        metas = [_upload_meta(f"upload:file{i}.pdf", age_hours=i) for i in range(12)]
        gatherer, _ = _roster_gatherer(metas)
        await gatherer.get_user_uploads("show me my uploads", limit=5)
        roster = gatherer._last_upload_roster
        assert len(roster) == 8
        assert roster[0]["title"] == "file0.pdf"
        assert roster[-1]["title"] == "file7.pdf"

    @pytest.mark.asyncio
    async def test_roster_excludes_image_stubs(self):
        metas = [
            _upload_meta("upload:photo.jpg", age_hours=1, is_image=True),
            _upload_meta("upload:Homework1-2.pdf", age_hours=1),
        ]
        gatherer, _ = _roster_gatherer(metas)
        await gatherer.get_user_uploads("pull up the assignment", limit=5)
        titles = [r["title"] for r in gatherer._last_upload_roster]
        assert "photo.jpg" not in titles
        assert "Homework1-2.pdf" in titles

    @pytest.mark.asyncio
    async def test_roster_excludes_stale_titles(self):
        metas = [
            _upload_meta("upload:old.pdf", age_hours=24 * 30),
            _upload_meta("upload:Homework1-2.pdf", age_hours=1),
        ]
        gatherer, _ = _roster_gatherer(metas)
        await gatherer.get_user_uploads("pull up the assignment", limit=5)
        titles = [r["title"] for r in gatherer._last_upload_roster]
        assert "old.pdf" not in titles
        assert "Homework1-2.pdf" in titles

    @pytest.mark.asyncio
    async def test_roster_fetch_is_metadata_only(self):
        metas = [_upload_meta("upload:Homework1-2.pdf", age_hours=1)]
        gatherer, fake_coll = _roster_gatherer(metas)
        await gatherer.get_user_uploads("pull up the first assignment", limit=5)
        roster_calls = [c for c in fake_coll.calls if c.get("include") == ["metadatas"]]
        assert roster_calls


class TestUploadRosterFormatterRender:
    def _get_formatter(self):
        from core.prompt.formatter import PromptFormatter
        token_mgr = MagicMock()
        token_mgr.count_tokens = MagicMock(return_value=10)
        fmt = PromptFormatter(token_manager=token_mgr, time_manager=None)
        fmt._feature_inventory_cache = None
        return fmt

    def _make_context(self, user_uploads):
        return {
            "recent_conversations": [], "memories": [], "user_profile": "",
            "narrative_state": "", "summaries": [], "reflections": [],
            "dreams": [], "semantic_chunks": [], "wiki": [],
            "personal_notes": [], "reference_docs": [],
            "user_uploads": user_uploads, "git_commits": [],
            "procedural_skills": [], "proposed_features": [],
            "graph_context": [], "unresolved_threads": [],
            "upcoming_schedule": [], "google_calendar": [],
            "proactive_insights": [], "web_search_results": None,
        }

    def test_roster_renders_trailing_line_with_no_other_uploads(self):
        roster_item = {
            "content": "",
            "metadata": {
                "type": "upload_roster",
                "roster": [
                    {"title": "Homework1-2.pdf", "date": "2026-09-05"},
                    {"title": "UsedCars2.csv", "date": "2026-09-05"},
                ],
            },
            "relevance_score": 0.0,
            "match_type": "roster",
        }
        fmt = self._get_formatter()
        ctx = self._make_context([roster_item])
        result = fmt._assemble_prompt(ctx, "pull up the first assignment")
        assert "[USER UPLOADED ITEMS]" in result
        assert "Recently uploaded files" in result
        assert "Homework1-2.pdf (2026-09-05)" in result
        assert "UsedCars2.csv (2026-09-05)" in result
        # No numbered items when only the roster fired.
        assert "[USER UPLOADED ITEMS] n=0" in result

    def test_roster_line_appends_after_real_upload_items(self):
        upload = {
            "content": "Some syllabus text.",
            "metadata": {"title": "upload:syllabus.pdf", "type": "user_upload"},
            "relevance_score": 0.81,
            "match_type": "semantic",
        }
        roster_item = {
            "content": "",
            "metadata": {
                "type": "upload_roster",
                "roster": [{"title": "syllabus.pdf", "date": "2026-09-05"}],
            },
        }
        fmt = self._get_formatter()
        ctx = self._make_context([upload, roster_item])
        result = fmt._assemble_prompt(ctx, "what's the grading policy")
        assert "[USER UPLOADED ITEMS] n=1" in result
        assert "Recently uploaded files" in result

    def test_no_roster_key_renders_as_before(self):
        upload = {
            "content": "Some fresh upload text.",
            "metadata": {"title": "upload:notes.txt", "type": "user_upload"},
            "relevance_score": 0.0,
            "match_type": "",
        }
        fmt = self._get_formatter()
        ctx = self._make_context([upload])
        result = fmt._assemble_prompt(ctx, "unrelated query")
        assert "Recently uploaded files" not in result


# ===========================================================================
# A5: tool doc truth
# ===========================================================================

class TestToolDocTruth:
    def test_memory_search_tool_definition_mentions_uploads(self):
        description = MEMORY_SEARCH_TOOL_DEFINITION["function"]["description"]
        assert "reference_docs" in description
        lowered = description.lower()
        assert "uploaded" in lowered
        assert "get_full_document" in lowered

    def test_formatter_shows_title_for_upload_hit(self):
        fmt = AgenticFormatter()
        results = [{
            "id": "abc123",
            "content": "Homework 1 instructions.",
            "relevance_score": 0.9,
            "metadata": {"title": "upload:Homework1-2.pdf", "section": ""},
        }]
        rendered = fmt.format_memory_results(results, "reference_docs")
        assert "upload:Homework1-2.pdf" in rendered


# ===========================================================================
# Fable referee (2026-09-07): roster marker placement vs the token budget's
# list trim. token_manager._manage_token_budget BREAKS at the first list item
# that does not fit, so a trailing marker would be dropped whenever an
# oversized upload chunk preceded it. The marker therefore goes FIRST and the
# formatter numbers real items with its own counter.
# ===========================================================================

class TestRosterMarkerPlacement:
    @pytest.mark.asyncio
    async def test_roster_marker_is_first_in_returned_list(self):
        metas = [_upload_meta("upload:Homework1-2.pdf", age_hours=1)]
        gatherer, _ = _roster_gatherer(metas)
        uploads = await gatherer.get_user_uploads("pull up the first assignment", limit=5)
        assert uploads and uploads[0].get("metadata", {}).get("type") == "upload_roster"

    def test_real_items_numbered_from_one_after_leading_marker(self):
        roster_item = {
            "content": "",
            "metadata": {"type": "upload_roster",
                         "roster": [{"title": "Homework1-2.pdf", "date": "2026-09-05"}]},
            "relevance_score": 0.0,
            "match_type": "roster",
        }
        real = {
            "content": "Some fresh upload text.",
            "metadata": {"title": "upload:notes.txt", "type": "user_upload"},
            "relevance_score": 0.7,
            "match_type": "semantic",
        }
        fmt = TestUploadRosterFormatterRender._get_formatter(TestUploadRosterFormatterRender())
        ctx = TestUploadRosterFormatterRender._make_context(
            TestUploadRosterFormatterRender(), [roster_item, real])
        result = fmt._assemble_prompt(ctx, "pull up the first assignment")
        assert "[USER UPLOADED ITEMS] n=1" in result
        assert "\n1) " in result
        assert "\n2) " not in result
        assert result.index("1) ") < result.index("Recently uploaded files")


class TestKeywordContentOverlapNotAdmittedByFreshness:
    """Fable referee (2026-09-07): with the pool restricted to uploads, the
    keyword leg surfaces content-word-overlap hits (0.2–0.4) that carry no
    relevance evidence; the freshness+document-cue leg must not admit them."""

    def _doc(self, score, match_type, title="upload:tmp_transcript.txt", age_hours=2):
        from datetime import datetime, timedelta
        return {
            "content": "lecture transcript text",
            "relevance_score": score,
            "match_type": match_type,
            "metadata": {"type": "user_upload", "title": title,
                         "timestamp": (datetime.now() - timedelta(hours=age_hours)).isoformat()},
        }

    def test_weak_keyword_hit_rejected_despite_freshness_and_cue(self):
        from core.prompt.gatherer_knowledge import _upload_is_live
        assert _upload_is_live(self._doc(0.35, "keyword"), query="pull up the first assignment") is False

    def test_title_keyword_match_still_admitted(self):
        from core.prompt.gatherer_knowledge import _upload_is_live
        assert _upload_is_live(self._doc(1.0, "keyword"), query="pull up the first assignment") is True

    def test_named_file_keyword_hit_still_admitted(self):
        from core.prompt.gatherer_knowledge import _upload_is_live
        doc = self._doc(0.3, "keyword", title="upload:Homework1-2.pdf")
        assert _upload_is_live(doc, query="what is in homework1-2.pdf") is True

    def test_fresh_semantic_low_score_with_cue_unchanged(self):
        from core.prompt.gatherer_knowledge import _upload_is_live
        assert _upload_is_live(self._doc(0.5, "semantic"), query="pull up the first assignment") is True
