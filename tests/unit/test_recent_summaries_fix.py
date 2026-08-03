"""
Regression tests for the 2026-07-25 recent-summaries incident.

`MemoryRetriever.get_summaries` ran a SEMANTIC query with an EMPTY string —
"summaries nearest the null embedding" — which systematically surfaced the
junkiest stored docs ("-", truncated Feb fragments) as [RECENT SUMMARIES]
while the collection held same-day summaries. Now it uses the store's
timestamp-sorted get_recent() with a junk filter, and junk summaries are
rejected at storage time (Chroma add_summary + CorpusManager.add_summary).
"""

from unittest.mock import MagicMock

from memory.utils import is_junk_summary, SUMMARY_MIN_CHARS
from memory.memory_retriever import MemoryRetriever


GOOD_SUMMARY = (
    "- The user quit their job and stated a benefit is no longer facing "
    "medication pressure; the user picked up their prescription on the 23rd."
)


class TestIsJunkSummary:
    def test_real_junk_from_live_store(self):
        # Actual docs observed dominating [RECENT SUMMARIES] on 2026-07-25.
        assert is_junk_summary("-")
        assert is_junk_summary("* The user is experiencing sleep")

    def test_empty_and_none(self):
        assert is_junk_summary("")
        assert is_junk_summary(None)
        assert is_junk_summary("   \n  ")

    def test_non_string(self):
        assert is_junk_summary({"content": "x"})

    def test_symbol_only(self):
        assert is_junk_summary("*** --- *** --- *** --- *** --- *** ---")

    def test_real_summary_passes(self):
        assert not is_junk_summary(GOOD_SUMMARY)

    def test_threshold_documented(self):
        assert SUMMARY_MIN_CHARS == 40


def _retriever_with_store(store):
    r = MemoryRetriever.__new__(MemoryRetriever)
    r.chroma_store = store
    r.corpus_manager = MagicMock(spec=[])  # no get_summaries attr → no fallback
    return r


class TestGetSummariesRecency:
    def test_uses_get_recent_not_empty_semantic_query(self):
        store = MagicMock()
        store.get_recent.return_value = [
            {"content": GOOD_SUMMARY, "metadata": {"timestamp": "2026-07-25T14:34:31"}},
        ]
        r = _retriever_with_store(store)
        out = r.get_summaries(limit=3)
        store.get_recent.assert_called_once()
        store.query_collection.assert_not_called()
        assert len(out) == 1 and out[0]["content"] == GOOD_SUMMARY
        assert out[0]["timestamp"] == "2026-07-25T14:34:31"

    def test_junk_filtered_and_limit_respected(self):
        store = MagicMock()
        store.get_recent.return_value = [
            {"content": "-", "metadata": {"timestamp": "2026-07-25T14:00:00"}},
            {"content": "* The user is experiencing sleep", "metadata": {"timestamp": "2026-07-25T13:00:00"}},
            {"content": GOOD_SUMMARY + " A", "metadata": {"timestamp": "2026-07-24T12:00:00"}},
            {"content": GOOD_SUMMARY + " B", "metadata": {"timestamp": "2026-07-23T12:00:00"}},
            {"content": GOOD_SUMMARY + " C", "metadata": {"timestamp": "2026-07-22T12:00:00"}},
        ]
        r = _retriever_with_store(store)
        out = r.get_summaries(limit=2)
        contents = [o["content"] for o in out]
        assert contents == [GOOD_SUMMARY + " A", GOOD_SUMMARY + " B"]

    def test_store_error_falls_back_to_corpus(self):
        store = MagicMock()
        store.get_recent.side_effect = RuntimeError("chroma down")
        r = MemoryRetriever.__new__(MemoryRetriever)
        r.chroma_store = store
        r.corpus_manager = MagicMock()
        r.corpus_manager.get_summaries.return_value = [{"content": GOOD_SUMMARY}]
        out = r.get_summaries(limit=3)
        assert out == [{"content": GOOD_SUMMARY}]


class TestStorageGuards:
    def test_chroma_add_summary_rejects_junk(self):
        from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
        store = MultiCollectionChromaStore.__new__(MultiCollectionChromaStore)
        # No collection access should happen for junk input.
        assert store.add_summary("-", period="session") == ""
        assert store.add_summary("", period="session") == ""

    def test_corpus_add_summary_rejects_junk(self):
        from memory.corpus_manager import CorpusManager
        cm = CorpusManager.__new__(CorpusManager)
        cm.corpus = []
        cm.max_entries = 100
        cm._episodic_cache = object()
        cm.save_corpus = MagicMock()
        cm.add_summary("-")
        assert cm.corpus == []
        cm.save_corpus.assert_not_called()

    def test_corpus_add_summary_accepts_real(self):
        from memory.corpus_manager import CorpusManager
        cm = CorpusManager.__new__(CorpusManager)
        cm.corpus = []
        cm.max_entries = 100
        cm._episodic_cache = object()
        cm.save_corpus = MagicMock()
        cm.add_summary(GOOD_SUMMARY)
        assert len(cm.corpus) == 1
        cm.save_corpus.assert_called_once()
