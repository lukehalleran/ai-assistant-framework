"""F10c: the `get_summaries(limit=)` signature fix.

Design source: docs/execution/generalization/failure_outcome_design.md,
amendment "F10 split and the write error type"; the owner decision to FIX
(not just report) the pre-existing defect pinned by F10b's
``test_production_signature_raises_type_error_reason`` (now renamed and
repaired in tests/unit/test_skill_and_summary_outcomes.py per the FIXTURE
RULE -- see that file for the paired failure control).

Defect (F10b): ``MemoryStorage._get_recent_summaries_by_timespan`` called
``self.corpus_manager.get_summaries(limit=50)``, but the real
``CorpusManager.get_summaries(self, count: int = 5)`` has no ``limit``
keyword, so the call always raised ``TypeError`` against production and
narrative regeneration after consolidation never ran.

Contract (briefs/F10c.md):
1. The count is now passed POSITIONALLY (``get_summaries(50)``), which
   matches any first-positional ``count``/``limit`` parameter name.
2. The final sort reuses the SAME per-row parse/naive-conversion the
   existing filter loop already computes (not a re-derived key), so mixed
   ``str``/``datetime``/tz-aware/naive timestamps sort without raising; a
   row whose timestamp cannot be parsed is still skipped exactly as before.
3. Unchanged: empty corpus -> []; unknown span -> []; a real read failure
   still raises ``RetrievalError(source="corpus_summaries", reason=...)``.

Each test below is written against the PRODUCTION call shape (a positional
count, or the real ``CorpusManager`` itself) so it fails on the unedited
source for the reason the contract item above names, and passes once that
item is fixed.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory.corpus_manager import CorpusManager
from memory.memory_storage import MemoryStorage


def _make_storage(*, corpus_manager=None, consolidator=None):
    """LOCAL copy of the `_make_storage` shape used across this lane's
    memory_storage tests (test_skill_and_summary_outcomes.py,
    test_api_error_storage_guard.py) -- kept local per FIXTURE RULE so this
    new file has no cross-file test dependency."""
    corpus_manager = corpus_manager if corpus_manager is not None else MagicMock()
    ms = MemoryStorage(
        corpus_manager=corpus_manager,
        chroma_store=MagicMock(),
        fact_extractor=MagicMock(),
        consolidator=consolidator,
    )
    return ms, corpus_manager


# --- CONTRACT item 1: the positional call matches the production signature ---


class TestProductionSignatureCallFix:
    def test_production_signature_read_returns_summary(self):
        """A fake with the REAL positional-count signature
        (`get_summaries(self, count=5)`) now returns rows instead of
        raising `TypeError` -- fails today (the deployed `limit=` keyword
        call does not match this signature)."""
        recent_ts = datetime.now() - timedelta(days=3)

        class _ProductionSignatureCorpusManager:
            def get_summaries(self, count=5):
                assert count == 50  # the deployed call passes 50 positionally
                return [{"timestamp": recent_ts, "text": "weekly summary"}]

        ms, _corpus = _make_storage(corpus_manager=_ProductionSignatureCorpusManager())

        result = ms._get_recent_summaries_by_timespan("weekly", limit=4)

        assert [r["text"] for r in result] == ["weekly summary"]


# --- CONTRACT item 2: the sort reuses the normalized per-row key ---


class TestMixedTimestampSort:
    def test_mixed_timestamp_types_sort_most_recent_first(self):
        """A naive datetime, a tz-aware (UTC) datetime and an ISO string,
        all inside the weekly window, must sort most-recent-first without
        raising. Fails today: the old sort key read the RAW stored value,
        so comparing `str` against `datetime` raised `TypeError`."""
        now = datetime.now()
        naive_dt = now - timedelta(days=1)
        aware_dt = (now - timedelta(days=2)).replace(tzinfo=timezone.utc)
        iso_str = (now - timedelta(days=3)).isoformat()

        class _MixedTimestampCorpusManager:
            def get_summaries(self, limit=50):
                # Deliberately unordered, and using the OLD keyword name --
                # isolates the sort fix (item 2) from the call-shape fix
                # (item 1), which item 1's test above already covers.
                return [
                    {"timestamp": iso_str, "text": "three-days-ago-str"},
                    {"timestamp": aware_dt, "text": "two-days-ago-aware"},
                    {"timestamp": naive_dt, "text": "one-day-ago-naive"},
                ]

        ms, _corpus = _make_storage(corpus_manager=_MixedTimestampCorpusManager())

        result = ms._get_recent_summaries_by_timespan("weekly", limit=4)

        assert [r["text"] for r in result] == [
            "one-day-ago-naive",
            "two-days-ago-aware",
            "three-days-ago-str",
        ]

    def test_unparseable_timestamp_row_is_skipped_control(self):
        """Paired control: a row whose timestamp string cannot be parsed is
        still skipped exactly as before the fix -- this already passes on
        the unedited source (a one-row result never triggers the mixed-type
        sort comparison), so it proves the fix changes nothing here."""
        recent_ts = datetime.now() - timedelta(days=1)

        class _CorpusManagerWithJunkRow:
            def get_summaries(self, limit=50):
                return [
                    {"timestamp": "not-a-parseable-timestamp", "text": "junk"},
                    {"timestamp": recent_ts, "text": "good"},
                ]

        ms, _corpus = _make_storage(corpus_manager=_CorpusManagerWithJunkRow())

        result = ms._get_recent_summaries_by_timespan("weekly", limit=4)

        assert [r["text"] for r in result] == ["good"]


# --- End-to-end against the real CorpusManager, tmp_path only ---


class TestRealCorpusManagerTmpPath:
    def test_tmp_path_corpus_manager_weekly_read_returns_summaries(self, tmp_path):
        """Against a REAL `CorpusManager` restricted to a tmp_path corpus
        file (never the default data/ path): two `add_summary` entries are
        returned by the weekly read, most-recent-first. Confirms the fix
        against the actual production class, not only a hand-written fake.

        Safety (DATA NOTE): `CorpusManager.__init__`/`_load_corpus` and
        `add_summary`/`save_corpus` write and read only `self.corpus_file`
        (set here to a tmp_path file); `get_summaries` only filters
        `self.corpus` in memory. `save_narrative_context`, the only method
        on this class that touches a DIFFERENT (default) path
        (`NARRATIVE_CONTEXT_PATH`), is never called by this test.
        """
        corpus_manager = CorpusManager(corpus_file=str(tmp_path / "corpus.json"))
        corpus_manager.add_summary(
            content="First weekly recap: the failing summaries read was diagnosed.",
            timestamp=datetime.now() - timedelta(days=5),
        )
        corpus_manager.add_summary(
            content="Second weekly recap: the positional call fix was confirmed.",
            timestamp=datetime.now() - timedelta(days=1),
        )

        ms, _corpus = _make_storage(corpus_manager=corpus_manager)

        result = ms._get_recent_summaries_by_timespan("weekly", limit=4)

        assert [r["content"] for r in result] == [
            "Second weekly recap: the positional call fix was confirmed.",
            "First weekly recap: the failing summaries read was diagnosed.",
        ]


# --- _maybe_regenerate_narrative reaches generation once the read works ---


class TestNarrativeRegenerationWithProductionSignature:
    @pytest.mark.asyncio
    async def test_regenerates_narrative_once_read_succeeds(self, monkeypatch):
        """Owner-approved behaviour change: with a production-signature
        corpus manager, a healthy weekly read now reaches the consolidator
        and persists the narrative. Fails today: the read raised
        `RetrievalError` and generation/save were both skipped."""
        monkeypatch.setattr("config.app_config.NARRATIVE_CONTEXT_ENABLED", True)
        recent_ts = datetime.now() - timedelta(days=3)

        class _ProductionSignatureCorpusManager:
            def __init__(self):
                self.save_narrative_context = MagicMock()

            def get_summaries(self, count=5):
                return [{"timestamp": recent_ts, "text": "weekly ok"}]

            def get_recent_memories(self, count=3):
                return ["stmt-1"]

        corpus_manager = _ProductionSignatureCorpusManager()
        consolidator = AsyncMock()
        consolidator.generate_narrative_context.return_value = "a narrative"
        ms, _corpus = _make_storage(corpus_manager=corpus_manager, consolidator=consolidator)

        await ms._maybe_regenerate_narrative()

        consolidator.generate_narrative_context.assert_awaited_once()
        corpus_manager.save_narrative_context.assert_called_once_with("a narrative")
