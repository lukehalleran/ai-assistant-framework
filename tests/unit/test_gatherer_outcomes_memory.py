"""F8a: memory gatherer sections report a typed failure instead of an empty
section (CGR-20260913-007 #85-#91).

Design: docs/execution/generalization/failure_outcome_design.md, "F7 split
and gatherer outcome shape" -> the F8 split. Drives the DEPLOYED
`MemoryRetrievalMixin._get_recent_conversations` (#87),
`_get_semantic_memories` (#88), `get_user_profile_context` (#90) and
`get_upcoming_schedule` (#91) directly through a bare-host pattern (the
`test_sep10_probe_dump_interpretation.py` precedent: `MemoryRetrievalMixin
.__new__` + attributes, never `UserProfile()`/`ContextGatherer()`), plus
the DEPLOYED `UnifiedPromptBuilder.build_prompt`/`_build_lightweight_context`
through the `full_builder`/`retrieval_limits` fake-builder pattern
(`test_independent_prompt_audit.py`, also used by F5/F7a/F7b/F7c). Fakes
only -- no real Chroma, corpus store, UserProfile file or model anywhere.

#85 `get_recent_facts`, #86 `get_facts` and #89 `_get_reflections` are
answered with off-path evidence only (no source change): `TestOffPathAnchorsEvidence`
proves, through the deployed builder, that these three MIXIN methods are
never called even when the analogous summaries/reflections sections run.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.prompt.gatherer_memory import MemoryRetrievalMixin
from utils.retrieval_outcome import OutcomeList, outcome_status
from tests.unit.test_independent_prompt_audit import full_builder, retrieval_limits


MARKER = "F8AMARKQ17_sensitive_detail_must_not_leak"
RECENT = [{"query": "Synthetic question", "response": "Synthetic retained answer."}]


def _G(coordinator=None, user_profile=None):
    """Bare MemoryRetrievalMixin host, bypassing __init__ (the
    test_sep10_probe_dump_interpretation.py `_make_r5_gatherer` precedent).
    Never constructs a real UserProfile() or ContextGatherer()."""
    g = MemoryRetrievalMixin.__new__(MemoryRetrievalMixin)
    g.memory_coordinator = coordinator
    g.memory_id_map = {}
    g.user_profile = user_profile
    return g


def _make_builder(monkeypatch, recent=RECENT):
    return full_builder(monkeypatch, recent, budget=10000)


# ---------------------------------------------------------------------------
# #87 _get_recent_conversations
# ---------------------------------------------------------------------------

class _CorpusManager:
    def __init__(self, items=None, raise_exc=None):
        self._items = list(items or [])
        self._raise = raise_exc

    def get_recent_memories(self, count=15):
        if self._raise:
            raise self._raise
        return list(self._items)


class TestRecentConversationsOutcomes:
    @pytest.mark.asyncio
    async def test_raising_corpus_manager_is_failed_and_empty(self):
        corpus = _CorpusManager(raise_exc=RuntimeError(f"{MARKER} corpus store down"))
        coordinator = SimpleNamespace(corpus_manager=corpus)
        g = _G(coordinator)
        result = await g._get_recent_conversations(limit=5)
        assert outcome_status(result) == ("failed", "RuntimeError")
        assert result == []
        assert MARKER not in result.reason

    @pytest.mark.asyncio
    async def test_healthy_nonempty_returns_todays_items(self):
        items = [
            {"query": "hi", "response": "Sure, happy to help.", "timestamp": "t1"},
            {"query": "again", "response": "Of course.", "timestamp": "t2"},
        ]
        corpus = _CorpusManager(items=items)
        g = _G(SimpleNamespace(corpus_manager=corpus))
        result = await g._get_recent_conversations(limit=2)
        assert outcome_status(result) == ("succeeded", "")
        assert len(result) == 2
        assert result[0]["response"] == "Sure, happy to help."

    @pytest.mark.asyncio
    async def test_healthy_empty_is_no_results(self):
        corpus = _CorpusManager(items=[])
        coordinator = SimpleNamespace(corpus_manager=corpus, get_memories=AsyncMock(return_value=[]))
        g = _G(coordinator)
        result = await g._get_recent_conversations(limit=5)
        assert outcome_status(result) == ("no_results", "")

    @pytest.mark.asyncio
    async def test_fallback_only_swallow_is_unchanged_sibling(self):
        """Contract point 1's sibling: the FALLBACK-only inner try/except
        (line 183-194) stays untouched -- a raising fallback still yields
        whatever the healthy corpus_manager already had, not a failed
        status."""
        corpus = _CorpusManager(items=[{"query": "hi", "response": "Sure.", "timestamp": "t1"}])
        coordinator = SimpleNamespace(
            corpus_manager=corpus,
            get_memories=AsyncMock(side_effect=RuntimeError(f"{MARKER} fallback down")),
        )
        g = _G(coordinator)
        result = await g._get_recent_conversations(limit=5)
        assert outcome_status(result) == ("succeeded", "")
        assert len(result) == 1


class TestLightPathUnchangedByTypedReturn:
    """#87's builder.py:2314 direct caller (`_build_lightweight_context`):
    a typed return -- never a raise -- keeps this caller on its NORMAL
    success dict shape, never the except's DIFFERENT fallback dict
    (builder.py ~2373+, which carries upcoming_schedule/google_calendar/
    relevant_emails keys the normal light path never sets)."""

    @pytest.mark.asyncio
    async def test_build_lightweight_context_stays_on_success_path_when_store_raises(self, monkeypatch):
        builder = _make_builder(monkeypatch)
        corpus = _CorpusManager(raise_exc=RuntimeError(f"{MARKER} corpus store down"))
        coordinator = SimpleNamespace(corpus_manager=corpus, get_memories=AsyncMock(return_value=[]))
        builder.context_gatherer = _G(coordinator)
        result = await builder._build_lightweight_context("thanks")
        # Present only on the SUCCESS path (builder.py:2316-2346).
        assert "web_search_decision" in result
        # Present only on the except's DIFFERENT fallback dict (~2373+).
        assert "upcoming_schedule" not in result
        assert result["recent_conversations"] == []
        assert MARKER not in str(result)


# ---------------------------------------------------------------------------
# #88 _get_semantic_memories
# ---------------------------------------------------------------------------

class _Coordinator88:
    def __init__(self, items=None, raise_exc=None):
        self._items = list(items or [])
        self._raise = raise_exc

    async def get_memories(self, *args, **kwargs):
        if self._raise:
            raise self._raise
        return list(self._items)


class TestSemanticMemoriesOutcomes:
    @pytest.mark.asyncio
    async def test_raising_coordinator_get_memories_is_failed_retrieval_prefixed(self):
        coord = _Coordinator88(raise_exc=ValueError(f"{MARKER} coordinator down"))
        g = _G(coord)
        result = await g._get_semantic_memories("find my keys", limit=5)
        assert outcome_status(result) == ("failed", "retrieval:ValueError")
        assert result == []
        assert MARKER not in result.reason

    @pytest.mark.asyncio
    async def test_exception_outside_inner_try_is_failed_class(self):
        coord = _Coordinator88(items=[{"id": "m1", "content": "stuff", "relevance_score": 0.5}])
        g = _G(coord)

        def _raise_dedup(memories):
            raise RuntimeError(f"{MARKER} dedup exploded")

        g._deduplicate_memories = _raise_dedup
        result = await g._get_semantic_memories("anything", limit=5)
        assert outcome_status(result) == ("failed", "RuntimeError")
        assert result == []
        assert MARKER not in result.reason

    @pytest.mark.asyncio
    async def test_empty_query_is_no_results(self):
        g = _G(_Coordinator88())
        result = await g._get_semantic_memories("", limit=5)
        assert result == []
        assert outcome_status(result) == ("no_results", "")

    @pytest.mark.asyncio
    async def test_healthy_returns_todays_items(self):
        items = [
            {"id": "m1", "content": "note about the trip", "relevance_score": 0.9},
            {"id": "m2", "content": "note about the dog", "relevance_score": 0.8},
        ]
        g = _G(_Coordinator88(items=items))
        result = await g._get_semantic_memories("tell me about my dog", limit=2)
        assert outcome_status(result) == ("succeeded", "")
        assert len(result) == 2
        assert {m["id"] for m in result} == {"m1", "m2"}


# ---------------------------------------------------------------------------
# #90 get_user_profile_context
# ---------------------------------------------------------------------------

class _RaisingProfile:
    def __init__(self, exc):
        self._exc = exc

    def get_context_injection(self, max_tokens, query, facts_per_category):
        raise self._exc


class TestUserProfileOutcomes:
    @pytest.mark.asyncio
    async def test_raising_get_context_injection_reraises(self):
        g = _G(None, user_profile=_RaisingProfile(RuntimeError(f"{MARKER} profile blew up")))
        with pytest.raises(RuntimeError):
            await g.get_user_profile_context("q")

    @pytest.mark.asyncio
    async def test_no_profile_returns_empty_string(self):
        g = _G(None, user_profile=None)
        assert await g.get_user_profile_context("q") == ""

    @pytest.mark.asyncio
    async def test_through_builder_raise_recorded_failed_and_prompt_still_builds(self, monkeypatch):
        builder = _make_builder(monkeypatch)
        profile_gatherer = _G(None, user_profile=_RaisingProfile(RuntimeError(f"{MARKER} profile blew up")))
        builder.context_gatherer.get_user_profile_context = profile_gatherer.get_user_profile_context
        result = await builder.build_prompt("Synthetic question", retrieval_overrides=retrieval_limits())
        assert result["_section_outcomes"]["user_profile"] == {"status": "failed", "reason": "RuntimeError"}
        assert "_build_time" in result, "builder must not fall back to its error path"
        assert not result["user_profile"]
        assert MARKER not in str(result)


# ---------------------------------------------------------------------------
# #91 get_upcoming_schedule
# ---------------------------------------------------------------------------

ALL_DAYS = "monday,tuesday,wednesday,thursday,friday,saturday,sunday"


class _FakeCollection:
    def __init__(self, count_val):
        self._count = count_val

    def count(self):
        return self._count


class _FakeChromaStore91:
    def __init__(self, facts_count=1, query_result=None, raise_exc=None):
        self.collections = {"facts": _FakeCollection(facts_count)}
        self._result = query_result if query_result is not None else []
        self._raise = raise_exc

    def query_collection(self, name, query_text=None, n_results=None):
        if self._raise:
            raise self._raise
        return self._result


class TestUpcomingScheduleOutcomes:
    @pytest.mark.asyncio
    async def test_raising_query_collection_is_failed_class(self, monkeypatch):
        monkeypatch.setattr("config.app_config.SCHEDULE_EXTRACTION_ENABLED", True)
        store = _FakeChromaStore91(facts_count=1, raise_exc=RuntimeError(f"{MARKER} schedule store down"))
        g = _G(SimpleNamespace(chroma_store=store))
        result = await g.get_upcoming_schedule("meetings this week", limit=10)
        assert outcome_status(result) == ("failed", "RuntimeError")
        assert result == []
        assert MARKER not in result.reason

    @pytest.mark.asyncio
    async def test_disabled_flag_is_no_results(self, monkeypatch):
        monkeypatch.setattr("config.app_config.SCHEDULE_EXTRACTION_ENABLED", False)
        store = _FakeChromaStore91(facts_count=1)
        g = _G(SimpleNamespace(chroma_store=store))
        result = await g.get_upcoming_schedule("meetings this week", limit=10)
        assert result == []
        assert outcome_status(result) == ("no_results", "")

    @pytest.mark.asyncio
    async def test_healthy_returns_todays_slice(self, monkeypatch):
        monkeypatch.setattr("config.app_config.SCHEDULE_EXTRACTION_ENABLED", True)
        query_result = [{
            "metadata": {
                "fact_type": "schedule",
                "schedule_scope": "recurring",
                "schedule_kind": "class",
                "schedule_days": ALL_DAYS,
                "schedule_start": "10:00",
                "schedule_end": "11:00",
            }
        }]
        store = _FakeChromaStore91(facts_count=1, query_result=query_result)
        g = _G(SimpleNamespace(chroma_store=store))
        result = await g.get_upcoming_schedule("what's on my schedule", limit=10)
        assert outcome_status(result) == ("succeeded", "")
        assert 1 <= len(result) <= 7
        assert all("display_date" in entry for entry in result)


# ---------------------------------------------------------------------------
# #85 get_recent_facts, #86 get_facts, #89 _get_reflections: off-path
# evidence (contract point 5) -- NO code change, no scanner anchor to
# close. Grep evidence recorded in batches/F8a.md §1; this proves it at
# the deployed-builder level.
# ---------------------------------------------------------------------------

class TestOffPathAnchorsEvidence:
    @pytest.mark.asyncio
    async def test_facts_and_reflections_mixin_methods_never_called_through_builder(self, monkeypatch):
        calls = []

        async def _spy_get_recent_facts(self, limit=15):
            calls.append("get_recent_facts")
            raise AssertionError(f"{MARKER} off-path get_recent_facts invoked")

        async def _spy_get_facts(self, query="", limit=15):
            calls.append("get_facts")
            raise AssertionError(f"{MARKER} off-path get_facts invoked")

        async def _spy_get_reflections(self, query="", limit=10):
            calls.append("_get_reflections")
            raise AssertionError(f"{MARKER} off-path _get_reflections invoked")

        monkeypatch.setattr(MemoryRetrievalMixin, "get_recent_facts", _spy_get_recent_facts)
        monkeypatch.setattr(MemoryRetrievalMixin, "get_facts", _spy_get_facts)
        monkeypatch.setattr(MemoryRetrievalMixin, "_get_reflections", _spy_get_reflections)

        coordinator = SimpleNamespace(
            corpus_manager=_CorpusManager(items=list(RECENT)),
            get_summaries=lambda n: [],
            get_reflections=AsyncMock(return_value=[]),
        )
        gatherer = _G(coordinator)
        # web_search's task is created unconditionally (builder.py:1513) by
        # directly CALLING self.context_gatherer._get_web_search_results(...)
        # to build the coroutine, outside any try -- a bare host missing
        # this attribute crashes build_prompt's own outer except (not this
        # test's target); give it a harmless working stub, matching
        # full_builder's own scaffold.
        gatherer._get_web_search_results = AsyncMock(return_value=[])
        builder = _make_builder(monkeypatch)
        builder.context_gatherer = gatherer
        # Enable the REAL sections the builder actually uses for
        # facts/reflections-shaped content (_get_summaries_separate,
        # _get_reflections_separate) -- neither of those calls the bare,
        # off-path methods this test spies on.
        overrides = {**retrieval_limits(), "max_reflections": 4, "max_summaries": 4}
        result = await builder.build_prompt("Synthetic question", retrieval_overrides=overrides)

        assert "_build_time" in result, "builder must not fall back to its error path"
        assert calls == []
