"""Thread depth advances across continuing turns while the turn's own pending copy exists.

Incident date: 2026-10-27 (logs 10-20..10-26: 88 of 88 "Continuing thread None at depth 1",
every [THREAD CONTEXT] "message #1"). Class: BC-58 (read-path view used as stored state), BC-46.

Drives THE deployed ThreadManager.detect_or_create_thread over a real CorpusManager with the
pending-turn registry populated the way gui/handlers._dispatch_storage does it: the turn is
registered as delivered-but-unstored BEFORE store_interaction runs the thread decision.
Fixture text is synthetic.
"""
import pytest

import memory.pending_turns as pending_turns
from memory.corpus_manager import CorpusManager
from memory.thread_manager import ThreadManager


@pytest.fixture
def env(tmp_path):
    pending_turns._pending.clear()
    cm = CorpusManager(str(tmp_path / "corpus.json"))
    yield cm, ThreadManager(corpus_manager=cm)
    pending_turns._pending.clear()


def _store_turn(cm, tm, query, topic="Playing Fetch"):
    """One turn as the deployed path orders it: register pending, decide thread, store, forget."""
    pending_turns.register(cm, query, "an answer", query)
    info = tm.detect_or_create_thread(query, False, current_topic=topic)
    cm.add_entry(
        query, "an answer", tags=["topic:" + topic],
        thread_id=info["thread_id"], thread_depth=info["depth"],
        thread_started=info["started"], thread_topic=info["topic"], topic=topic,
    )
    pending_turns.forget(cm, query)
    return info


def test_three_continuing_turns_advance_depth(env):
    cm, tm = env
    infos = [
        _store_turn(cm, tm, "we threw the ball at the park today"),
        _store_turn(cm, tm, "the ball went into the park pond"),
        _store_turn(cm, tm, "we found the ball near the park pond later"),
    ]
    assert [i["depth"] for i in infos] == [1, 2, 3]
    assert len({i["thread_id"] for i in infos}) == 1
    assert infos[0]["thread_id"]
    stored = [e for e in cm.corpus if e.get("thread_id")]
    assert [e["thread_depth"] for e in stored] == [1, 2, 3]


def test_pending_copy_of_a_previous_unstored_turn_is_not_read_as_thread_state(env):
    cm, tm = env
    first = _store_turn(cm, tm, "we threw the ball at the park today")
    # turn 2 delivered but its storage still in flight when turn 3 is decided
    pending_turns.register(cm, "the ball went into the park pond", "an answer", None)
    third = tm.detect_or_create_thread(
        "we found the ball near the park pond later", False, current_topic="Playing Fetch",
    )
    # decided from the last STORED turn (depth 1) -- never from a thread-less pending copy
    assert third["thread_id"] == first["thread_id"]
    assert third["depth"] == 2
