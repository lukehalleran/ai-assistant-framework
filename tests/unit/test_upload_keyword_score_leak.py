"""Fix 1.1 (2026-09-06): upload gate leak via reference_docs_manager's
keyword-search containment scoring.

Root cause (verified): `ReferenceDocsManager._keyword_search` scored
`query_lower in section or section in query_lower` as 0.9 unconditionally
whenever `section == ''` (single-chunk uploads store `section=''` —
utils/text_chunking.py returns section=None for docs under the chunk
threshold, stored as '' by reference_docs_manager.upload_document). Since
`'' in anything` is always True, every empty-section upload chunk topped
EVERY query at 0.9 — well above `_upload_is_live`'s
`USER_UPLOADS_MIN_RELEVANCE` (0.62) — regardless of actual relevance. A
208-day-old 'upload:Homework4.docx' chunk surfaced this way on an unrelated
medication-rest question.

These tests exercise the DEPLOYED `ReferenceDocsManager._keyword_search`,
`core.prompt.gatherer_knowledge._upload_is_live`, and the formatter's
`[USER UPLOADED ITEMS]` rendering — never a re-derivation.
"""
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from knowledge.reference_docs_manager import ReferenceDocsManager
from core.prompt.gatherer_knowledge import _upload_is_live, USER_UPLOADS_MIN_RELEVANCE


def _manager_with_docs(docs_and_metas):
    """A ReferenceDocsManager whose _collection() returns a fake ChromaDB
    collection backed by the given (document_text, metadata) pairs — no
    real ChromaDB or store required (the chroma_store property is never
    touched since _collection is monkeypatched directly, same isolation
    style as tests/unit/test_refdocs_lazy_collection.py)."""
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


class TestKeywordSearchEmptySectionLeak:
    def test_empty_section_upload_does_not_leak_onto_unrelated_query(self):
        # The live-incident shape: a single-chunk upload with section=''
        # and a title/content carrying zero words (not even stopwords) in
        # common with the query, so only the containment branches under
        # test could produce a hit.
        mgr = _manager_with_docs([
            (
                "Take ibuprofen with food and rest for the afternoon.",
                {"title": "upload:Homework4.docx", "section": "",
                 "type": "user_upload", "timestamp": "2026-02-10T00:00:00"},
            ),
        ])
        results = mgr._keyword_search("purple giraffes juggle bicycles underwater", limit=10)
        assert results == []

    def test_whitespace_only_section_does_not_leak(self):
        mgr = _manager_with_docs([
            (
                "Some upload content unrelated to the query at hand.",
                {"title": "upload:notes.txt", "section": "   ",
                 "type": "user_upload"},
            ),
        ])
        results = mgr._keyword_search("purple giraffes juggle bicycles underwater", limit=10)
        assert results == []

    def test_genuine_multiword_title_match_still_scores(self):
        # A real title/query overlap of >=2 content tokens must still match
        # at the original 1.0 score — the fix must not regress real matches.
        mgr = _manager_with_docs([
            (
                "Architecture notes about the memory scoring pipeline.",
                {"title": "Memory Scoring Architecture", "section": "Overview",
                 "type": "reference_doc"},
            ),
        ])
        results = mgr._keyword_search("memory scoring architecture", limit=10)
        assert len(results) == 1
        assert results[0]["relevance_score"] == pytest.approx(1.0)
        assert results[0]["match_type"] == "keyword"

    def test_genuine_multiword_section_match_still_scores(self):
        mgr = _manager_with_docs([
            (
                "Details on gating thresholds and cross-encoder rerank.",
                {"title": "Retrieval Guide", "section": "gating thresholds explained",
                 "type": "reference_doc"},
            ),
        ])
        results = mgr._keyword_search("gating thresholds explained please", limit=10)
        assert len(results) == 1
        assert results[0]["relevance_score"] == pytest.approx(0.9)

    def test_single_short_word_containment_does_not_match(self):
        # A one-token section/title (e.g. a stray "faq") must not score via
        # bare containment — needs >=2 content tokens to count as a real
        # containment match. It may still fall through to the weaker
        # partial-word-overlap branches below, which is fine.
        mgr = _manager_with_docs([
            (
                "Frequently asked questions about billing.",
                {"title": "faq", "section": "", "type": "reference_doc"},
            ),
        ])
        results = mgr._keyword_search("faq", limit=10)
        # Either no match, or a match scored via the weaker partial-title
        # branch (0.6-0.8), never the 1.0 containment branch.
        if results:
            assert results[0]["relevance_score"] < 1.0


class TestUploadIsLiveIgnoresKeywordRelevance:
    def _doc(self, relevance, match_type, ts=None, content_type=None):
        meta = {"type": "user_upload"}
        if ts is not None:
            meta["timestamp"] = ts.isoformat() if isinstance(ts, datetime) else ts
        if content_type is not None:
            meta["content_type"] = content_type
        doc = {"relevance_score": relevance, "metadata": meta, "match_type": match_type}
        return doc

    def test_stale_keyword_scored_upload_dropped_on_unrelated_query(self):
        old = datetime.now() - timedelta(days=208)
        doc = self._doc(relevance=USER_UPLOADS_MIN_RELEVANCE + 0.28, match_type="keyword", ts=old)
        assert not _upload_is_live(doc, query="should I rest after taking my medication")

    def test_same_score_admitted_when_semantic(self):
        old = datetime.now() - timedelta(days=208)
        doc = self._doc(relevance=USER_UPLOADS_MIN_RELEVANCE + 0.28, match_type="semantic", ts=old)
        assert _upload_is_live(doc, query="should I rest after taking my medication")

    def test_fresh_keyword_hit_still_survives_via_document_cue(self):
        # The freshness/document-cue leg is untouched by this fix — a
        # recently uploaded doc with a document-shaped query still gets in
        # even when its score is keyword-typed (and thus zeroed).
        fresh = datetime.now() - timedelta(days=1)
        doc = self._doc(relevance=0.9, match_type="keyword", ts=fresh)
        assert _upload_is_live(doc, query="what does the syllabus say about grading")


class TestUploadRelevanceMarkerRendering:
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

    def test_semantic_upload_renders_relevance_marker(self):
        fmt = self._get_formatter()
        upload = {
            "content": "Some syllabus text.",
            "metadata": {"title": "upload:syllabus.pdf", "type": "user_upload"},
            "relevance_score": 0.81,
            "match_type": "semantic",
        }
        ctx = self._make_context([upload])
        result = fmt._assemble_prompt(ctx, "what's the grading policy")
        assert "[USER UPLOADED ITEMS]" in result
        assert "[relevance: 0.81" in result
        assert "sem" in result

    def test_keyword_upload_renders_kw_marker(self):
        fmt = self._get_formatter()
        upload = {
            "content": "Homework instructions.",
            "metadata": {"title": "upload:hw.docx", "type": "user_upload"},
            "relevance_score": 0.9,
            "match_type": "keyword",
        }
        ctx = self._make_context([upload])
        result = fmt._assemble_prompt(ctx, "when is this due")
        assert "[relevance: 0.90" in result
        assert "kw" in result

    def test_zero_relevance_upload_omits_marker(self):
        fmt = self._get_formatter()
        upload = {
            "content": "Some fresh upload text.",
            "metadata": {"title": "upload:notes.txt", "type": "user_upload"},
            "relevance_score": 0.0,
        }
        ctx = self._make_context([upload])
        result = fmt._assemble_prompt(ctx, "hello")
        assert "[relevance:" not in result
