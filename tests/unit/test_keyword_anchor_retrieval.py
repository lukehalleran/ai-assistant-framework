"""Tests for the rare-proper-noun keyword-anchor retrieval fallback (2026-08-26).

Incident: "I really think I should try and get appointment scheduled with
Morgan for Friday" retrieved 30 appointment-vibe memories with ZERO Morgan
mentions — a rare name contributes almost nothing to a bge query embedding
and the live memory path had no keyword channel (the corpus keyword scan
existed only inside insight mode). The Obsidian keyword search had the twin
failure: whole-query word-set scoring weighed "Morgan" the same as "not", so
the "Advisor: Morgan Reeves" note lost to date-titled daily notes.

Covers:
  - utils.query_checker.extract_rare_proper_nouns (detector)
  - CorpusManager.search_keyword(include_entry=True)
  - MemoryRetriever._keyword_anchor_memories + get_memories injection
  - ObsidianManager._keyword_search proper-noun floor
"""

import asyncio
from datetime import datetime, timedelta

import pytest

from utils.query_checker import extract_rare_proper_nouns
from memory.corpus_manager import CorpusManager
from memory.memory_retriever import MemoryRetriever

Morgan_QUERY = (
    "I'm not sure. I really think I should try and get appointment "
    "scheduled with Morgan for Friday or before hand"
)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class TestExtractRareProperNouns:
    def test_live_Morgan_query(self):
        # the exact turn-3 query — "Morgan" found, "Friday" stoplisted,
        # sentence-initial "I'm" ignored
        assert extract_rare_proper_nouns(Morgan_QUERY) == ["Morgan"]

    def test_days_and_months_stoplisted(self):
        assert extract_rare_proper_nouns("see you next Friday in March") == []

    def test_sentence_initial_excluded(self):
        # no dictionary distinguishes "Please" from a name at sentence start
        assert extract_rare_proper_nouns("Please be honest with me. Focus on it") == []

    def test_sentence_initial_name_found_when_repeated_mid_sentence(self):
        got = extract_rare_proper_nouns("Morgan said she can check Morgan's schedule")
        assert got == ["Morgan"]

    def test_all_caps_emphasis_excluded(self):
        assert extract_rare_proper_nouns("I have SO much evidence") == []

    def test_adjacent_names_merge_to_phrase(self):
        assert extract_rare_proper_nouns("What did I email Jordan Vale about?") == [
            "Jordan Vale"
        ]

    def test_title_abbreviation_not_a_sentence_boundary(self):
        assert extract_rare_proper_nouns("I saw Dr. Goldsman on Monday") == ["Goldsman"]

    def test_possessive_stripped(self):
        assert extract_rare_proper_nouns("waiting on Morgan's reply") == ["Morgan"]

    def test_daemon_self_reference_stoplisted(self):
        assert extract_rare_proper_nouns("I asked Daemon about it") == []

    def test_max_terms_cap(self):
        got = extract_rare_proper_nouns(
            "met Sam with Diane and Frank and Graham and Alexis", max_terms=3
        )
        assert len(got) == 3

    def test_empty_and_lowercase(self):
        assert extract_rare_proper_nouns("") == []
        assert extract_rare_proper_nouns("yes she is academic advisor") == []


# ---------------------------------------------------------------------------
# Corpus include_entry
# ---------------------------------------------------------------------------

@pytest.fixture
def corpus(tmp_path):
    cm = CorpusManager(corpus_file=str(tmp_path / "corpus.json"))
    now = datetime(2026, 8, 20, 12, 0, 0)
    cm.add_entry(
        "sent this: Hi Morgan, I wanted to reach out about my standing this semester.",
        "Good move — that puts the withdrawal question directly to your advisor.",
        timestamp=now - timedelta(days=60),
    )
    cm.add_entry(
        "Slept badly again.",
        "Poor sleep compounds everything.",
        timestamp=now - timedelta(days=1),
    )
    return cm


class TestSearchKeywordIncludeEntry:
    def test_entry_attached(self, corpus):
        hits = corpus.search_keyword("Morgan", include_entry=True)
        assert hits
        assert all("entry" in h for h in hits)
        assert "Hi Morgan" in hits[0]["entry"]["query"]

    def test_default_shape_unchanged(self, corpus):
        hits = corpus.search_keyword("Morgan")
        assert hits
        assert all("entry" not in h for h in hits)


# ---------------------------------------------------------------------------
# Retriever anchor
# ---------------------------------------------------------------------------

def _mem(content, score=0.8):
    return {"id": f"sem::{abs(hash(content)) % 10**8}", "content": content,
            "relevance_score": score, "timestamp": datetime(2026, 8, 1)}


@pytest.fixture
def retriever(corpus):
    return MemoryRetriever(corpus_manager=corpus, chroma_store=None)


class TestKeywordAnchorMemories:
    def test_anchor_fires_when_name_missing_from_pool(self, retriever):
        pool = [_mem("User: about appointments\nAssistant: sure")]
        anchors = retriever._keyword_anchor_memories(Morgan_QUERY, pool)
        assert anchors, "expected exact-match anchor for Morgan"
        assert all(a.get("keyword_anchor") for a in anchors)
        assert any("Morgan" in a["content"] for a in anchors)

    def test_no_anchor_when_pool_already_covers_name(self, retriever):
        pool = [_mem("User: emailed Morgan today\nAssistant: nice")]
        assert retriever._keyword_anchor_memories(Morgan_QUERY, pool) == []

    def test_no_anchor_without_proper_noun(self, retriever):
        assert retriever._keyword_anchor_memories("how are you today", []) == []

    def test_disabled_via_env(self, retriever, monkeypatch):
        monkeypatch.setenv("KEYWORD_ANCHOR_ENABLED", "0")
        assert retriever._keyword_anchor_memories(Morgan_QUERY, []) == []

    def test_cap_respected(self, corpus, monkeypatch):
        now = datetime(2026, 8, 20, 12, 0, 0)
        for i in range(6):
            corpus.add_entry(
                f"Morgan mentioned thing number {i}",
                "noted",
                timestamp=now - timedelta(days=i + 2),
            )
        r = MemoryRetriever(corpus_manager=corpus, chroma_store=None)
        monkeypatch.setenv("KEYWORD_ANCHOR_MAX_HITS", "2")
        anchors = r._keyword_anchor_memories(Morgan_QUERY, [])
        assert len(anchors) == 2

    def test_corpus_without_search_keyword_is_safe(self):
        class Bare:
            pass
        r = MemoryRetriever(corpus_manager=Bare(), chroma_store=None)
        assert r._keyword_anchor_memories(Morgan_QUERY, []) == []


class TestGetMemoriesInjection:
    """End-to-end through THE deployed get_memories with stubbed I/O layers:
    the anchor must survive the threshold filter into the final slice."""

    def _run(self, retriever, query, semantic_pool):
        async def fake_semantic(q, n_results=0):
            return list(semantic_pool)

        async def fake_combine(**kwargs):
            return list(kwargs.get("semantic") or [])

        async def fake_rerank(mems, q):
            return mems

        retriever._get_semantic_memories = fake_semantic
        retriever._combine_memories = fake_combine
        retriever._maybe_cross_encoder_rerank = fake_rerank
        return asyncio.run(retriever.get_memories(query, limit=5))

    def test_anchor_present_in_final_result(self, retriever):
        semantic_pool = [
            _mem(f"User: appointment talk {i}\nAssistant: ok {i}") for i in range(5)
        ]
        result = self._run(retriever, Morgan_QUERY, semantic_pool)
        assert any(m.get("keyword_anchor") for m in result), (
            "keyword anchor was dropped before the final slice"
        )
        assert any("Morgan" in (m.get("content") or "") for m in result)

    def test_no_anchor_pollution_on_plain_query(self, retriever):
        semantic_pool = [_mem("User: sleep talk\nAssistant: rest up")]
        result = self._run(retriever, "how did I sleep this week", semantic_pool)
        assert not any(m.get("keyword_anchor") for m in result)


# ---------------------------------------------------------------------------
# Obsidian proper-noun floor
# ---------------------------------------------------------------------------

class _FakeCollection:
    def __init__(self, docs, metas):
        self._docs, self._metas = docs, metas

    def get(self, include=None):
        return {
            "documents": self._docs,
            "metadatas": self._metas,
            "ids": [f"id{i}" for i in range(len(self._docs))],
        }


class _FakeStore:
    def __init__(self, collection):
        self._c = collection

    def _get_collection(self, name):
        return self._c


class TestObsidianProperNounFloor:
    def _manager(self, tmp_path, docs_metas):
        from knowledge.obsidian_manager import ObsidianManager
        docs = [d for d, _ in docs_metas]
        metas = [m for _, m in docs_metas]
        return ObsidianManager(
            chroma_store=_FakeStore(_FakeCollection(docs, metas)),
            vault_path=str(tmp_path),
        )

    def test_advisor_note_outranks_generic_overlap(self, tmp_path):
        advisor_doc = (
            "Advisor: Morgan Reeves\nqs: do my summer courses sat reqs -> YES "
            "only hard req is ISYE 6740"
        )
        daily_doc = (
            "Had a doctor appointment scheduled. Not sure I should really try "
            "to get more done before Friday."
        )
        mgr = self._manager(tmp_path, [
            (daily_doc, {"title": "9 1 24 Daily Note", "section": "", "tags": "",
                         "file_path": "daily/9 1 24.md"}),
            (advisor_doc, {"title": "Advising", "section": "", "tags": "",
                           "file_path": "school/advising.md"}),
        ])
        results = mgr._keyword_search(Morgan_QUERY, limit=2)
        assert results, "expected keyword results"
        top = results[0]
        assert "Morgan Reeves" in top["content"]
        assert top["relevance_score"] >= 0.75

    def test_no_floor_without_proper_noun(self, tmp_path):
        doc = "generic note about appointments and scheduling things"
        mgr = self._manager(tmp_path, [
            (doc, {"title": "note", "section": "", "tags": "", "file_path": "n.md"}),
        ])
        results = mgr._keyword_search("should I get an appointment scheduled", limit=2)
        for r in results:
            assert r["relevance_score"] < 0.75
