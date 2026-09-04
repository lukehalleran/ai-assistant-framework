"""2026-09-03 follow-up continuity fixes (three defects from a session review).

1. [THREAD CONTEXT] honesty at session start. The thread block is rendered off
   the PREVIOUS stored turn with no gap check, so a bare greeting ("Hey")
   thirteen hours after the last session produced "This is message #3 in an
   ongoing conversation thread about <yesterday's topic>" plus "Maintain
   conversational continuity." The 07-25 honesty branch stayed silent because
   ``is_fragment_continuation("Hey")`` was True (no greeting vocabulary) and a
   "general" current topic reads as related. Fix: greetings are their own
   shape (never fragments, never acks), ``get_thread_context`` exposes the
   last turn's timestamp, and the orchestrator renders a "New session" line
   when the time manager reports the first message OR the thread's last turn
   is past the hard cutoff.

2. Topic label stability + thread depth. Storage-time ``belongs_to_thread``
   compared topic labels with EXACT equality while the read side used a loose
   predicate, and thread detection ran a SECOND classification instead of
   reusing the turn's label — so a classifier relabel of one continuous
   conversation reset thread depth every turn. Fix: one shared
   ``topics_related`` predicate, label stabilization in ``_extract_topics``
   (a related relabel keeps the previous label), and the turn's label is
   forwarded into thread detection.

3. STM novelty override. The STM prompt biases ``reference_type`` toward
   "recall" when in doubt and the formatter then warns "the current message
   restates an event already in context" — wrong for a message naming an
   entity that appears NOWHERE in the short-term window. Fix: a deterministic
   post-override demotes such a "recall" to "unclear" (mid-sentence rare
   proper nouns always; sentence-initial names only via a graph-backed
   person/pet allow-gate) and the formatter names the missing entities.

All fixtures use synthetic vocabulary (pets Biscuit/Mochi, people Casey/Morgan).
"""
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from tests.unit.test_anaphoric_continuation import _FakeTopicManager, _make_pipeline
from tests.unit.test_process_user_query import _make_bfp_orch, _make_context
from utils.query_checker import (
    analyze_query,
    belongs_to_thread,
    is_casual_acknowledgment,
    is_fragment_continuation,
    is_greeting_opener,
    topics_related,
)


# ---------------------------------------------------------------------------
# 1a. Greeting shape
# ---------------------------------------------------------------------------

class TestGreetingShape:
    @pytest.mark.parametrize("q", [
        "Hey", "hey", "Hi there", "Good morning", "Hello!", "yo", "sup",
        "morning", "heyy",
    ])
    def test_greetings_are_not_fragments(self, q):
        assert is_greeting_opener(q)
        assert not is_fragment_continuation(q)

    def test_greetings_are_not_acks(self):
        # Greetings must keep the FULL builder path (so [UNRESOLVED THREADS]
        # surface at session start) — never the light/ack routing.
        assert not is_casual_acknowledgment("Hey")
        assert analyze_query("Hey").is_small_talk is False

    def test_pinned_fragments_unchanged(self):
        assert is_fragment_continuation("Tactical Taylors")
        assert is_fragment_continuation("the other one")
        assert is_fragment_continuation("classic timeline stuff")

    def test_greeting_opener_underfires(self):
        assert not is_greeting_opener("")
        assert not is_greeting_opener(None)
        assert not is_greeting_opener("goodness me")
        assert not is_greeting_opener("good grief")


# ---------------------------------------------------------------------------
# 1b. [THREAD CONTEXT] session-start honesty
# ---------------------------------------------------------------------------

_THREAD_CTX = {"thread_id": "t1", "thread_depth": 3, "thread_topic": "Bank Communication"}


def _tm(gap: str):
    tm = MagicMock()
    tm.time_since_previous_message = MagicMock(return_value=gap)
    return tm


class TestThreadContextSessionStart:
    @pytest.mark.asyncio
    async def test_first_message_renders_new_session(self):
        ctx = _make_context(
            thread_context=dict(_THREAD_CTX), original_query="Hey", primary_topic="general",
        )
        orch = _make_bfp_orch(time_manager=_tm("N/A (first message in session)"))
        _, system_prompt, _ = await orch.build_full_prompt(ctx, return_raw_context=True)
        assert "[THREAD CONTEXT]" in system_prompt
        assert "New session" in system_prompt
        assert "Bank Communication" in system_prompt
        assert "message #3" not in system_prompt
        assert "Maintain conversational continuity" not in system_prompt

    @pytest.mark.asyncio
    async def test_mid_session_unchanged(self):
        ctx = _make_context(
            thread_context=dict(_THREAD_CTX),
            original_query="did they ever answer about the fees",
            primary_topic="Bank Fees",
        )
        orch = _make_bfp_orch(time_manager=_tm("2 m"))
        _, system_prompt, _ = await orch.build_full_prompt(ctx, return_raw_context=True)
        assert "This is message #3" in system_prompt
        assert "Bank Communication" in system_prompt
        assert "Maintain conversational continuity" in system_prompt
        assert "New session" not in system_prompt

    @pytest.mark.asyncio
    async def test_stale_last_timestamp_without_time_manager(self):
        stale = (datetime.now() - timedelta(hours=13)).isoformat()
        ctx = _make_context(
            thread_context={**_THREAD_CTX, "last_timestamp": stale},
            original_query="Hey", primary_topic="general",
        )
        orch = _make_bfp_orch()  # time_manager is None
        _, system_prompt, _ = await orch.build_full_prompt(ctx, return_raw_context=True)
        assert "New session" in system_prompt
        assert "message #3" not in system_prompt

    @pytest.mark.asyncio
    async def test_fresh_last_timestamp_without_time_manager_keeps_thread(self):
        fresh = (datetime.now() - timedelta(minutes=5)).isoformat()
        ctx = _make_context(
            thread_context={**_THREAD_CTX, "last_timestamp": fresh},
            original_query="did they ever answer about the fees",
            primary_topic="Bank Fees",
        )
        orch = _make_bfp_orch()
        _, system_prompt, _ = await orch.build_full_prompt(ctx, return_raw_context=True)
        assert "This is message #3" in system_prompt
        assert "New session" not in system_prompt

    def test_thread_context_is_stale_underfires(self):
        from core.orchestrator import _thread_context_is_stale
        assert _thread_context_is_stale({}) is False
        assert _thread_context_is_stale({"thread_id": "t1"}) is False
        assert _thread_context_is_stale({"last_timestamp": "not a date"}) is False
        assert _thread_context_is_stale({"last_timestamp": 12345}) is False
        assert _thread_context_is_stale(None) is False
        assert _thread_context_is_stale("garbage") is False
        five_min = (datetime.now() - timedelta(minutes=5)).isoformat()
        assert _thread_context_is_stale({"last_timestamp": five_min}) is False
        three_h = datetime.now() - timedelta(hours=3)
        assert _thread_context_is_stale({"last_timestamp": three_h}) is True
        assert _thread_context_is_stale({"last_timestamp": three_h.isoformat()}) is True
        # "Z"-suffixed ISO strings parse too
        assert _thread_context_is_stale(
            {"last_timestamp": "2020-01-01T00:00:00Z"}
        ) is True

    def test_is_session_start_underfires(self):
        orch = _make_bfp_orch()
        assert orch._is_session_start() is False  # time_manager None
        broken = MagicMock()
        broken.time_since_previous_message = MagicMock(side_effect=RuntimeError("x"))
        orch.time_manager = broken
        assert orch._is_session_start() is False
        orch.time_manager = _tm("N/A (first message in session)")
        assert orch._is_session_start() is True
        orch.time_manager = _tm("2 m")
        assert orch._is_session_start() is False


# ---------------------------------------------------------------------------
# 1c. ThreadManager exposes the last turn's timestamp
# ---------------------------------------------------------------------------

class _FakeCorpus:
    def __init__(self, recent):
        self._recent = recent

    def get_recent_memories(self, count=1):
        return list(self._recent)[:count]


class TestThreadManagerContext:
    def test_last_timestamp_exposed(self):
        from memory.thread_manager import ThreadManager
        tm = ThreadManager(corpus_manager=_FakeCorpus([{
            "thread_id": "t1", "thread_depth": 2, "thread_topic": "Playing Fetch",
            "timestamp": "2026-09-02T20:00:00",
        }]))
        ctx = tm.get_thread_context()
        assert ctx["thread_id"] == "t1"
        assert ctx["last_timestamp"] == "2026-09-02T20:00:00"

    def test_no_recent_returns_none(self):
        from memory.thread_manager import ThreadManager
        assert ThreadManager(corpus_manager=_FakeCorpus([])).get_thread_context() is None


# ---------------------------------------------------------------------------
# 2a. One shared topic-continuity predicate
# ---------------------------------------------------------------------------

class TestTopicsRelatedShared:
    def test_orchestrator_alias_is_the_query_checker_function(self):
        import core.orchestrator as orch_mod
        assert orch_mod._topics_related is topics_related

    def test_predicate_shape(self):
        assert topics_related("Playing Fetch", "Playing Games")
        assert not topics_related("Playing Fetch", "Tax Forms")
        assert topics_related("general", "Tax Forms")  # no signal → related
        assert topics_related("", "Tax Forms")


# ---------------------------------------------------------------------------
# 2b. belongs_to_thread uses the loose predicate (but never "general")
# ---------------------------------------------------------------------------

_LAST_CONV_BASE = {
    "query": "we threw the ball for Mochi in the yard",
    "response": "Sounds like a good session.",
    "is_heavy_topic": False,
    "thread_depth": 2,
}
_LOW_OVERLAP_QUERY = "the board night ran late"


def _last_conv(topic):
    return {
        **_LAST_CONV_BASE,
        "topic": topic,
        "timestamp": (datetime.now() - timedelta(seconds=60)).isoformat(),
    }


class TestBelongsToThreadLooseTopic:
    def test_related_labels_earn_topic_bonus(self):
        assert belongs_to_thread(
            _LOW_OVERLAP_QUERY, _last_conv("Playing Fetch"), current_topic="Playing Games"
        ) is True
        assert belongs_to_thread(
            _LOW_OVERLAP_QUERY, _last_conv("Playing Fetch"), current_topic="Tax Forms"
        ) is False

    def test_general_never_earns_bonus(self):
        # topics_related("general", "general") is True, but the bonus must
        # not fire on a label that carries no signal.
        general = belongs_to_thread(
            _LOW_OVERLAP_QUERY, _last_conv("general"), current_topic="general"
        )
        unrelated = belongs_to_thread(
            _LOW_OVERLAP_QUERY, _last_conv("Playing Fetch"), current_topic="Tax Forms"
        )
        assert general == unrelated is False
        assert belongs_to_thread(
            _LOW_OVERLAP_QUERY, _last_conv("Playing Fetch"), current_topic="general"
        ) is False
        assert belongs_to_thread(
            _LOW_OVERLAP_QUERY, _last_conv("general"), current_topic="Playing Games"
        ) is False


# ---------------------------------------------------------------------------
# 2c. Topic label stabilization in ContextPipeline._extract_topics
# ---------------------------------------------------------------------------

_STANDALONE_QUERY = "we tried a new board game tonight after dinner"


class TestLabelStabilization:
    def test_related_fresh_label_keeps_previous(self):
        tm = _FakeTopicManager(last_topic="Playing Fetch", fresh_topic="Playing Games")
        pipeline = _make_pipeline(tm)
        primary, topics = asyncio.run(pipeline._extract_topics(_STANDALONE_QUERY))
        assert primary == "Playing Fetch"
        assert topics == ["Playing Fetch"]
        assert tm.fresh_calls == 1  # classifier still runs
        assert tm.last_topic == "Playing Fetch"  # anchor stays sticky

    def test_unrelated_fresh_label_passes_through(self):
        tm = _FakeTopicManager(last_topic="Playing Fetch", fresh_topic="Tax Forms")
        pipeline = _make_pipeline(tm)
        primary, _ = asyncio.run(pipeline._extract_topics(_STANDALONE_QUERY))
        assert primary == "Tax Forms"
        assert tm.fresh_calls == 1

    def test_general_fresh_label_not_stabilized(self):
        tm = _FakeTopicManager(last_topic="Playing Fetch", fresh_topic="general")
        pipeline = _make_pipeline(tm)
        primary, _ = asyncio.run(pipeline._extract_topics(_STANDALONE_QUERY))
        assert primary == "general"

    def test_mock_last_topic_skipped(self):
        # A Mock topic manager (non-str last_topic) must never trip the guards.
        tm = MagicMock()
        tm.get_primary_topic = MagicMock(return_value="Playing Games")
        tm.get_entities = MagicMock(return_value=[])
        pipeline = _make_pipeline(tm)
        primary, _ = asyncio.run(pipeline._extract_topics(_STANDALONE_QUERY))
        assert primary == "Playing Games"


# ---------------------------------------------------------------------------
# 2d. Thread detection reuses the turn's label
# ---------------------------------------------------------------------------

class TestThreadManagerUsesPassedTopic:
    def test_passed_topic_wins(self):
        from memory.thread_manager import ThreadManager
        topic_manager = MagicMock()
        tm = ThreadManager(corpus_manager=_FakeCorpus([]), topic_manager=topic_manager)
        info = tm.detect_or_create_thread("we threw the ball", False, current_topic="Playing Fetch")
        assert info["topic"] == "Playing Fetch"
        assert info["depth"] == 1
        topic_manager.update_from_user_input.assert_not_called()
        topic_manager.get_primary_topic.assert_not_called()

    def test_passed_general_used_verbatim(self):
        from memory.thread_manager import ThreadManager
        topic_manager = MagicMock()
        tm = ThreadManager(corpus_manager=_FakeCorpus([]), topic_manager=topic_manager)
        info = tm.detect_or_create_thread("hmm", False, current_topic="general")
        assert info["topic"] == "general"
        topic_manager.update_from_user_input.assert_not_called()

    def test_fallback_when_absent(self):
        from memory.thread_manager import ThreadManager
        topic_manager = MagicMock()
        topic_manager.get_primary_topic = MagicMock(return_value="Cats")
        tm = ThreadManager(corpus_manager=_FakeCorpus([]), topic_manager=topic_manager)
        info = tm.detect_or_create_thread("Biscuit is asleep", False)
        assert info["topic"] == "Cats"
        topic_manager.update_from_user_input.assert_called_once()
        info = tm.detect_or_create_thread("Biscuit is asleep", False, current_topic="   ")
        assert info["topic"] == "Cats"

    def test_storage_forwards_current_topic(self):
        import inspect
        from memory import memory_storage
        src = inspect.getsource(memory_storage)
        assert "self._thread_detect_fn(" in src
        assert "current_topic=self.current_topic" in src

    def test_coordinator_forwards_current_topic(self):
        from memory.memory_coordinator import MemoryCoordinator
        coord = object.__new__(MemoryCoordinator)
        coord.thread_manager = MagicMock()
        coord.thread_manager.detect_or_create_thread = MagicMock(return_value={"thread_id": "t"})
        coord._detect_or_create_thread("q", False, current_topic="Playing Fetch")
        coord.thread_manager.detect_or_create_thread.assert_called_once_with(
            "q", False, current_topic="Playing Fetch"
        )


# ---------------------------------------------------------------------------
# 3. STM novelty override
# ---------------------------------------------------------------------------

class _RecallModel:
    """Scripts the prompt's recall bias: every turn comes back as 'recall'."""

    def __init__(self):
        self.prompt = ""

    async def generate_once(self, prompt, **kwargs):
        self.prompt = prompt
        return json.dumps({
            "topic": "Cat antics",
            "user_question": "User is restating the moth incident",
            "intent": "Share",
            "tone": "casual",
            "reference_type": "recall",
            "temporal_facts": [],
            "open_threads": [],
            "constraints": [],
        })


def _analyzer(notes_text: str = ""):
    from core.stm_analyzer import STMAnalyzer
    analyzer = STMAnalyzer(_RecallModel())
    analyzer._get_recent_daily_notes_text = lambda *a, **k: notes_text
    return analyzer


def _run(analyzer, query, memories=None, last_reply=None, graph_memory=None):
    return asyncio.run(analyzer.analyze(
        recent_memories=memories or [],
        user_query=query,
        last_assistant_response=last_reply,
        graph_memory=graph_memory,
    ))


_MOCHI_WINDOW = [{
    "timestamp": "2026-09-03T09:00:00",
    "query": "Mochi slept on the couch all morning",
    "response": "Sounds like a lazy day for Mochi.",
}]


def _graph_with_biscuit(tmp_path, aliases=()):
    from memory.graph_memory import GraphMemory
    from memory.graph_models import GraphNode
    gm = GraphMemory(persist_path=str(tmp_path / "graph.json"))
    gm.add_entity(GraphNode(
        entity_id="biscuit", display_name="Biscuit", entity_type="animal",
        aliases=list(aliases),
    ))
    return gm


class TestSTMNoveltyOverride:
    def test_mid_sentence_novel_name_demotes_to_unclear(self):
        result = _run(_analyzer(), "I saw Biscuit chase a moth", memories=_MOCHI_WINDOW)
        assert result["reference_type"] == "unclear"
        assert result["novelty_override"] is True
        assert result["novel_entities"] == ["Biscuit"]

    def test_name_in_window_stays_recall(self):
        window = [{
            "timestamp": "2026-09-03T09:00:00",
            "query": "Biscuit chased a moth around the kitchen",
            "response": "Classic Biscuit.",
        }]
        result = _run(_analyzer(), "I saw Biscuit chase a moth", memories=window)
        assert result["reference_type"] == "recall"
        assert "novelty_override" not in result

    def test_name_in_last_assistant_reply_is_not_novel(self):
        result = _run(
            _analyzer(), "I saw Biscuit chase a moth",
            memories=_MOCHI_WINDOW, last_reply="Did Biscuit catch the moth this time?",
        )
        # This is also a direct answer to the immediately preceding question,
        # so the pre-existing continuation override correctly wins.
        assert result["reference_type"] == "clarification"
        assert "novelty_override" not in result

    def test_name_in_daily_notes_stays_recall(self):
        result = _run(
            _analyzer(notes_text="--- Daily note --- Biscuit chased a moth."),
            "I saw Biscuit chase a moth", memories=_MOCHI_WINDOW,
        )
        assert result["reference_type"] == "recall"

    def test_sentence_initial_name_underfires_without_graph(self):
        result = _run(_analyzer(), "Biscuit caught a moth today", memories=_MOCHI_WINDOW)
        assert result["reference_type"] == "recall"
        assert "novelty_override" not in result

    def test_sentence_initial_known_pet_demotes_with_graph(self, tmp_path):
        gm = _graph_with_biscuit(tmp_path)
        result = _run(
            _analyzer(), "Biscuit caught a moth today",
            memories=_MOCHI_WINDOW, graph_memory=gm,
        )
        assert result["reference_type"] == "unclear"
        assert result["novel_entities"] == ["Biscuit"]

    def test_sentence_initial_unknown_word_underfires_with_graph(self, tmp_path):
        gm = _graph_with_biscuit(tmp_path)
        result = _run(_analyzer(), "Please pass the salt", memories=_MOCHI_WINDOW, graph_memory=gm)
        assert result["reference_type"] == "recall"

    def test_alias_in_window_stays_recall(self, tmp_path):
        gm = _graph_with_biscuit(tmp_path, aliases=["coco"])
        window = [{
            "timestamp": "2026-09-03T09:00:00",
            "query": "coco chased a moth around the kitchen",
            "response": "Classic.",
        }]
        result = _run(_analyzer(), "Biscuit caught a moth today", memories=window, graph_memory=gm)
        assert result["reference_type"] == "recall"

    def test_mock_graph_never_raises(self):
        result = _run(
            _analyzer(), "Biscuit caught a moth today",
            memories=_MOCHI_WINDOW, graph_memory=MagicMock(),
        )
        assert result["reference_type"] == "recall"  # Mock resolution → no allow-gate hit

    def test_continuation_answer_override_still_wins(self):
        prior = "Want me to set the vet reminder the day before, or the day of?"
        result = _run(_analyzer(), "Day of please", last_reply=prior)
        assert result["reference_type"] == "clarification"
        assert "novelty_override" not in result

    def test_novel_named_entities_underfires_on_garbage(self):
        from core.stm_analyzer import novel_named_entities
        assert novel_named_entities("", "") == []
        assert novel_named_entities(None, None) == []
        assert novel_named_entities("I saw Biscuit chase a moth", None) == ["Biscuit"]
        assert novel_named_entities("no names here at all", "") == []

    @pytest.mark.asyncio
    async def test_pipeline_passes_graph_memory(self):
        from core.context_pipeline import ContextPipeline
        pipeline = object.__new__(ContextPipeline)
        analyzer = MagicMock()
        captured = {}

        async def _analyze(**kwargs):
            captured.update(kwargs)
            return {"reference_type": "unclear"}

        analyzer.analyze = _analyze
        pipeline.stm_analyzer = analyzer
        pipeline._stm_max_recent = 10
        pipeline.memory_system = MagicMock()
        pipeline.memory_system.graph_memory = "GRAPH"
        pipeline.memory_system.corpus_manager.get_recent_memories = MagicMock(return_value=[])
        await pipeline._analyze_stm("I saw Biscuit chase a moth", conversation_history=[])
        assert captured.get("graph_memory") == "GRAPH"


class TestFormatterNoveltyNote:
    def _render(self, stm_summary):
        from core.prompt.formatter import PromptFormatter
        fmt = PromptFormatter(token_manager=MagicMock())
        return fmt._assemble_prompt(
            context={"stm_summary": stm_summary},
            user_input="I saw Biscuit chase a moth", directives="", system_prompt="",
        )

    def _summary(self, **extra):
        return {
            "topic": "Cat antics", "user_question": "q", "intent": "i", "tone": "casual",
            "reference_type": "unclear", "temporal_facts": [], "open_threads": [],
            "constraints": [], **extra,
        }

    def test_novelty_note_rendered(self):
        out = self._render(self._summary(novelty_override=True, novel_entities=["Biscuit"]))
        assert "Reference Type: unclear" in out
        assert "Reference is ambiguous" in out
        assert (
            "Note: the current message names Biscuit, which do not appear in the "
            "short-term window." in out
        )

    def test_no_note_without_override(self):
        out = self._render(self._summary())
        assert "Reference Type: unclear" in out
        assert "do not appear in the short-term window" not in out

    def test_no_note_with_empty_entities(self):
        out = self._render(self._summary(novelty_override=True, novel_entities=[]))
        assert "do not appear in the short-term window" not in out
