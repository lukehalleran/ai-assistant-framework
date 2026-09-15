"""Regression tests for CGR-20260913-010 anchors #128/#129 and CGR-20260913-009
anchor #130 (F10b).

``store_skill`` used to swallow a collection-unavailable failure (#128) and
any other write failure (#129) into the same ``None`` a deliberate skip
returns; ``_get_recent_summaries_by_timespan`` (#130) did the same for a
failed summaries read, so ``_maybe_regenerate_narrative`` could not tell
"no summaries yet" from "the read broke".

Contract (briefs/F10b.md): disabled/dedup still return None; a
collection-unavailable failure raises ``StoreWriteError(source=
"procedural_skills", reason=f"collection:{type(e).__name__}")``; any other
body failure raises the same type with ``reason=type(e).__name__``,
propagated unchanged if already a ``StoreWriteError``.
``_extract_procedural_skills`` counts a per-skill ``StoreWriteError`` as
``failed`` and continues (a dedup ``None`` is "not kept", never "failed").
``_get_recent_summaries_by_timespan`` raises ``RetrievalError(source=
"corpus_summaries", reason=type(e).__name__)``; empty/unknown-span still
return ``[]``. ``_maybe_regenerate_narrative`` logs one warning and returns
without generating/saving if a summaries read or ``get_recent_memories``
fails; genuinely empty/healthy reads are unchanged.

Fixtures: ``_make_storage`` is a LOCAL copy of
test_api_error_storage_guard.py's ``storage`` fixture (FIXTURE RULE -- that
file is unedited). ``_make_shutdown_processor`` builds ``ShutdownProcessor``
via ``__new__`` with only the attributes ``_extract_procedural_skills``
reads -- no real corpus, Chroma, model, consolidator or profile.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory.memory_storage import MemoryStorage
from memory.shutdown_processor import ShutdownProcessor
from utils.retrieval_outcome import RetrievalError, StoreWriteError

_MARKER = "SYNTH_MARKER_f10b_6c1a9e42"


def _make_storage(*, chroma_store=None, corpus_manager=None, consolidator=None):
    corpus_manager = corpus_manager if corpus_manager is not None else MagicMock()
    chroma_store = chroma_store if chroma_store is not None else MagicMock()
    ms = MemoryStorage(
        corpus_manager=corpus_manager,
        chroma_store=chroma_store,
        fact_extractor=MagicMock(),
        consolidator=consolidator,
    )
    return ms, corpus_manager, chroma_store


def _make_skill(**overrides):
    from memory.procedural_skill import ProceduralSkill, SkillCategory

    defaults = dict(
        trigger="when tests fail intermittently on CI",
        action_pattern="retry once, then inspect logs for a flaky marker before re-running",
        category=SkillCategory.DEBUGGING,
        confidence=0.8,
        tags=["ci"],
        source_session_id="session-1",
    )
    defaults.update(overrides)
    return ProceduralSkill(**defaults)


# --- MemoryStorage.store_skill (#128, #129) ---


class TestStoreSkillOutcomes:
    @pytest.mark.asyncio
    async def test_disabled_returns_none_and_no_write(self, monkeypatch):
        monkeypatch.setattr("config.app_config.PROCEDURAL_SKILLS_ENABLED", False)
        chroma_store = MagicMock()
        ms, _corpus, _chroma = _make_storage(chroma_store=chroma_store)

        result = await ms.store_skill(_make_skill())

        assert result is None
        assert not chroma_store._get_collection.called
        assert not chroma_store.add_to_collection.called

    @pytest.mark.asyncio
    async def test_dedup_match_returns_none_and_no_write(self, monkeypatch):
        monkeypatch.setattr("config.app_config.PROCEDURAL_SKILLS_ENABLED", True)
        monkeypatch.setattr("config.app_config.SKILL_DEDUP_THRESHOLD", 0.85)
        chroma_store = MagicMock()
        coll = MagicMock()
        coll.count.return_value = 1
        chroma_store._get_collection.return_value = coll
        chroma_store.query_collection.return_value = [
            {"relevance_score": 0.9, "metadata": {"trigger": "an old pattern"}}
        ]
        ms, _corpus, _chroma = _make_storage(chroma_store=chroma_store)

        result = await ms.store_skill(_make_skill())

        assert result is None
        assert not chroma_store.add_to_collection.called

    @pytest.mark.asyncio
    async def test_collection_unavailable_raises_store_write_error(self, monkeypatch):
        monkeypatch.setattr("config.app_config.PROCEDURAL_SKILLS_ENABLED", True)
        chroma_store = MagicMock()
        chroma_store._get_collection.side_effect = RuntimeError(f"no such collection {_MARKER}")
        ms, _corpus, _chroma = _make_storage(chroma_store=chroma_store)

        with pytest.raises(StoreWriteError) as exc_info:
            await ms.store_skill(_make_skill())

        # Pre-existing warning line unchanged (kept per contract), still
        # carries {e}; only the new error's str below must be label-only.
        assert exc_info.value.source == "procedural_skills"
        assert exc_info.value.reason == "collection:RuntimeError"
        assert _MARKER not in str(exc_info.value)
        assert not chroma_store.add_to_collection.called

    @pytest.mark.asyncio
    async def test_dedup_query_raising_raises_store_write_error_and_no_write(self, monkeypatch):
        monkeypatch.setattr("config.app_config.PROCEDURAL_SKILLS_ENABLED", True)
        chroma_store = MagicMock()
        coll = MagicMock()
        coll.count.return_value = 1
        chroma_store._get_collection.return_value = coll
        chroma_store.query_collection.side_effect = RuntimeError(f"backend down {_MARKER}")
        ms, _corpus, _chroma = _make_storage(chroma_store=chroma_store)

        with pytest.raises(StoreWriteError) as exc_info:
            await ms.store_skill(_make_skill())

        assert exc_info.value.source == "procedural_skills"
        assert exc_info.value.reason == "RuntimeError"
        assert _MARKER not in str(exc_info.value)
        assert not chroma_store.add_to_collection.called

    @pytest.mark.asyncio
    async def test_add_to_collection_raising_raises_store_write_error(self, monkeypatch):
        monkeypatch.setattr("config.app_config.PROCEDURAL_SKILLS_ENABLED", True)
        chroma_store = MagicMock()
        coll = MagicMock()
        coll.count.return_value = 0
        chroma_store._get_collection.return_value = coll
        chroma_store.add_to_collection.side_effect = RuntimeError(f"disk full {_MARKER}")
        ms, _corpus, _chroma = _make_storage(chroma_store=chroma_store)

        with pytest.raises(StoreWriteError) as exc_info:
            await ms.store_skill(_make_skill())

        assert exc_info.value.source == "procedural_skills"
        assert exc_info.value.reason == "RuntimeError"
        assert _MARKER not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_success_control_returns_id(self, monkeypatch):
        """Positive control paired with the failures above and the skips."""
        monkeypatch.setattr("config.app_config.PROCEDURAL_SKILLS_ENABLED", True)
        chroma_store = MagicMock()
        coll = MagicMock()
        coll.count.return_value = 0
        chroma_store._get_collection.return_value = coll
        chroma_store.add_to_collection.return_value = "skill-doc-1"
        ms, _corpus, _chroma = _make_storage(chroma_store=chroma_store)

        result = await ms.store_skill(_make_skill())

        assert result == "skill-doc-1"


# --- ShutdownProcessor._extract_procedural_skills (shutdown count change) ---


def _make_shutdown_processor(*, storage, model_manager, corpus_manager=None, session_start=None):
    sp = ShutdownProcessor.__new__(ShutdownProcessor)
    sp.model_manager = model_manager
    sp._storage = storage
    sp.corpus_manager = corpus_manager if corpus_manager is not None else MagicMock()
    sp.session_start = session_start if session_start is not None else datetime(2026, 1, 1)
    return sp


def _skill_lines(n):
    """``n`` valid procedural-skill JSON lines, as the LLM's raw output."""
    return "\n".join(
        '{"trigger": "situation number %d calling for a fix", '
        '"action_pattern": "abstract multi-step approach number %d for the fix", '
        '"category": "workflow", "confidence": 0.7, "tags": ["t%d"]}' % (i, i, i)
        for i in range(n)
    )


_SESSION_CONVERSATIONS = [
    {"query": "How do I fix flaky CI?", "response": "Retry once, then inspect logs."},
    {"query": "What about silent config defaults?", "response": "Validate required keys at startup."},
    {"query": "Any other recurring pattern?", "response": "Watch for a swallowed exception."},
]


def _make_model_manager(raw_output):
    from types import SimpleNamespace
    return SimpleNamespace(generate_once=AsyncMock(return_value=raw_output))


class TestExtractProceduralSkillsCounts:
    @pytest.mark.asyncio
    async def test_one_failure_one_success_counts_both_no_exception(self, monkeypatch, caplog):
        monkeypatch.setattr("config.app_config.PROCEDURAL_SKILLS_ENABLED", True)
        model_manager = _make_model_manager(_skill_lines(2))
        storage = MagicMock()
        storage.store_skill = AsyncMock(
            side_effect=[StoreWriteError(source="procedural_skills", reason="RuntimeError"), "doc-2"]
        )
        sp = _make_shutdown_processor(storage=storage, model_manager=model_manager)

        with caplog.at_level("INFO"):
            await sp._extract_procedural_skills(_SESSION_CONVERSATIONS)  # must not raise

        assert storage.store_skill.call_count == 2
        assert "Extracted 1 procedural skill(s), 1 failed" in caplog.text

    @pytest.mark.asyncio
    async def test_dedup_skip_is_not_counted_as_failed(self, monkeypatch, caplog):
        """Control: a dedup skip (None, no raise) is "not kept", never "failed"."""
        monkeypatch.setattr("config.app_config.PROCEDURAL_SKILLS_ENABLED", True)
        model_manager = _make_model_manager(_skill_lines(2))
        storage = MagicMock()
        storage.store_skill = AsyncMock(side_effect=[None, "doc-2"])
        sp = _make_shutdown_processor(storage=storage, model_manager=model_manager)

        with caplog.at_level("INFO"):
            await sp._extract_procedural_skills(_SESSION_CONVERSATIONS)

        assert storage.store_skill.call_count == 2
        assert "Extracted 1 procedural skill(s), 0 failed" in caplog.text

    @pytest.mark.asyncio
    async def test_all_healthy_control_kept_two(self, monkeypatch, caplog):
        monkeypatch.setattr("config.app_config.PROCEDURAL_SKILLS_ENABLED", True)
        model_manager = _make_model_manager(_skill_lines(2))
        storage = MagicMock()
        storage.store_skill = AsyncMock(side_effect=["doc-1", "doc-2"])
        sp = _make_shutdown_processor(storage=storage, model_manager=model_manager)

        with caplog.at_level("INFO"):
            await sp._extract_procedural_skills(_SESSION_CONVERSATIONS)

        assert storage.store_skill.call_count == 2
        assert "Extracted 2 procedural skill(s), 0 failed" in caplog.text


# --- MemoryStorage._get_recent_summaries_by_timespan (#130) ---


class TestSummariesReadOutcomes:
    def test_corpus_manager_raising_raises_retrieval_error(self):
        corpus_manager = MagicMock()
        corpus_manager.get_summaries.side_effect = RuntimeError(f"disk error {_MARKER}")
        ms, _corpus, _chroma = _make_storage(corpus_manager=corpus_manager)

        with pytest.raises(RetrievalError) as exc_info:
            ms._get_recent_summaries_by_timespan("weekly", limit=4)

        # Pre-existing debug line unchanged (kept per contract), still
        # carries {e}; only the new error's str below must be label-only.
        assert exc_info.value.source == "corpus_summaries"
        assert exc_info.value.reason == "RuntimeError"
        assert _MARKER not in str(exc_info.value)

    def test_production_signature_returns_recent_summaries(self):
        """[F10c] Renamed positive production-signature proof: the
        previously-pinned defect is now FIXED (owner decision) -- the real
        ``get_summaries(self, count=5)`` signature is called positionally
        and the read succeeds, returning rows instead of raising
        ``TypeError``. Paired with
        ``test_corpus_manager_raising_raises_retrieval_error`` above as the
        failure control: a genuine read failure still raises."""
        recent_ts = datetime.now() - timedelta(days=3)

        class _ProductionSignatureCorpusManager:
            def get_summaries(self, count=5):
                assert count == 50  # the deployed call passes 50 positionally
                return [{"timestamp": recent_ts, "text": "recent"}]

        ms, _corpus, _chroma = _make_storage(corpus_manager=_ProductionSignatureCorpusManager())

        result = ms._get_recent_summaries_by_timespan("weekly", limit=4)

        assert [r["text"] for r in result] == ["recent"]

    def test_empty_corpus_returns_empty_list(self):
        corpus_manager = MagicMock()
        corpus_manager.get_summaries.return_value = []
        ms, _corpus, _chroma = _make_storage(corpus_manager=corpus_manager)

        assert ms._get_recent_summaries_by_timespan("weekly", limit=4) == []

    def test_healthy_read_returns_filtered_sorted_rows(self):
        """[F10c] A fake accepting the deployed positional ``count``
        argument (the real ``CorpusManager.get_summaries(self, count=5)``
        signature): rows filtered by span, sorted most-recent-first."""
        recent_ts = (datetime.now() - timedelta(days=3)).isoformat()
        old_ts = (datetime.now() - timedelta(days=200)).isoformat()

        class _HealthyCorpusManager:
            def get_summaries(self, count=5):
                return [
                    {"timestamp": old_ts, "text": "old"},
                    {"timestamp": recent_ts, "text": "recent"},
                ]

        ms, _corpus, _chroma = _make_storage(corpus_manager=_HealthyCorpusManager())

        result = ms._get_recent_summaries_by_timespan("weekly", limit=4)

        assert [r["text"] for r in result] == ["recent"]


# --- MemoryStorage._maybe_regenerate_narrative ---


class TestMaybeRegenerateNarrativeOutcomes:
    @pytest.mark.asyncio
    async def test_weekly_raising_aborts_before_monthly_no_generate_no_save(self, monkeypatch, caplog):
        monkeypatch.setattr("config.app_config.NARRATIVE_CONTEXT_ENABLED", True)
        corpus_manager = MagicMock()
        healthy_monthly = [{"timestamp": datetime.now().isoformat(), "text": "monthly ok"}]
        corpus_manager.get_summaries.side_effect = [
            RuntimeError(f"disk error {_MARKER}"),  # weekly
            healthy_monthly,  # would-be monthly call -- never reached
        ]
        consolidator = AsyncMock()
        ms, _corpus, _chroma = _make_storage(corpus_manager=corpus_manager, consolidator=consolidator)

        with caplog.at_level("WARNING"):
            await ms._maybe_regenerate_narrative()  # must not raise

        # Only the weekly read ran -- the monthly read was never attempted.
        assert corpus_manager.get_summaries.call_count == 1
        assert not consolidator.generate_narrative_context.called
        assert not corpus_manager.save_narrative_context.called
        assert _MARKER not in caplog.text

    @pytest.mark.asyncio
    async def test_both_summaries_empty_returns_without_generating(self, monkeypatch):
        """Control: a genuinely empty corpus (unchanged behavior)."""
        monkeypatch.setattr("config.app_config.NARRATIVE_CONTEXT_ENABLED", True)
        corpus_manager = MagicMock()
        corpus_manager.get_summaries.return_value = []
        consolidator = AsyncMock()
        ms, _corpus, _chroma = _make_storage(corpus_manager=corpus_manager, consolidator=consolidator)

        await ms._maybe_regenerate_narrative()

        assert not consolidator.generate_narrative_context.called
        assert not corpus_manager.save_narrative_context.called

    @pytest.mark.asyncio
    async def test_recent_memories_raising_returns_without_generating(self, monkeypatch, caplog):
        monkeypatch.setattr("config.app_config.NARRATIVE_CONTEXT_ENABLED", True)
        corpus_manager = MagicMock()
        recent_ts = (datetime.now() - timedelta(days=3)).isoformat()
        corpus_manager.get_summaries.return_value = [{"timestamp": recent_ts, "text": "weekly ok"}]
        corpus_manager.get_recent_memories.side_effect = RuntimeError(f"disk error {_MARKER}")
        consolidator = AsyncMock()
        ms, _corpus, _chroma = _make_storage(corpus_manager=corpus_manager, consolidator=consolidator)

        with caplog.at_level("WARNING"):
            await ms._maybe_regenerate_narrative()  # must not raise

        assert not consolidator.generate_narrative_context.called
        assert not corpus_manager.save_narrative_context.called
        assert _MARKER not in caplog.text

    @pytest.mark.asyncio
    async def test_healthy_reads_generate_and_save_once_control(self, monkeypatch):
        monkeypatch.setattr("config.app_config.NARRATIVE_CONTEXT_ENABLED", True)
        corpus_manager = MagicMock()
        recent_ts = (datetime.now() - timedelta(days=3)).isoformat()
        corpus_manager.get_summaries.return_value = [{"timestamp": recent_ts, "text": "weekly ok"}]
        corpus_manager.get_recent_memories.return_value = ["stmt-1"]
        consolidator = AsyncMock()
        consolidator.generate_narrative_context.return_value = "a narrative"
        ms, _corpus, _chroma = _make_storage(corpus_manager=corpus_manager, consolidator=consolidator)

        await ms._maybe_regenerate_narrative()

        consolidator.generate_narrative_context.assert_called_once()
        corpus_manager.save_narrative_context.assert_called_once_with("a narrative")
