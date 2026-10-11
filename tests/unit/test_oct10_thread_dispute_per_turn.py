"""A correcting turn disputes an overlapping open thread in the SAME turn
(2026-10-10, class: BC-58, BC-52).

Live 10-08: "Don't think it is standing" at 13:51 left the false "Thursday
check-in with professor" thread open all afternoon -- the typed
DisputedResolution outcome was only reachable from the idle-shutdown pass.
These tests drive the deployed POST_RESPONSE_HOOKS registry (the chokepoint
both the GUI path and process_user_query run), with the real
CorrectionDetector and the real ThreadStore over a fake chroma.
"""
from types import SimpleNamespace

import pytest

import core.orchestrator as orch_mod
from core.correction_detector import CorrectionDetector
from memory.thread_models import OpenThread, ThreadStatus, ThreadType
from memory.thread_store import COLLECTION_NAME, ThreadStore

CORRECTION = "Actually it's not a standing Thursday meeting, the professor check-in is cancelled"
PLAIN = "Thursday check-in with the professor went fine, he was standing right there"


class _Coll:
    def __init__(self, items):
        self._items = items

    def count(self):
        return len(self._items)

    def delete(self, ids=None):
        gone = set(ids or [])
        self._items[:] = [i for i in self._items if i["id"] not in gone]


class _Chroma:
    def __init__(self):
        self._items = []
        self._n = 0
        self.collection = _Coll(self._items)
        self.collections = {COLLECTION_NAME: self.collection}

    def add_to_collection(self, name, text, metadata):
        self._n += 1
        self._items.append({"id": f"d{self._n}", "content": text, "metadata": dict(metadata)})
        return f"d{self._n}"

    def list_all(self, name):
        return list(self._items)

    def create_collection(self, name):
        return self.collection


def _thread(tid, topic):
    return OpenThread(thread_id=tid, topic=topic, summary="s", thread_type=ThreadType.DEADLINE)


def _status(chroma, tid):
    return [i["metadata"]["status"] for i in chroma.list_all(COLLECTION_NAME)
            if i["metadata"]["thread_id"] == tid]


def _run(monkeypatch, store, user_input, history=()):
    """Run ONLY the registered thread_dispute entry, through the registry."""
    registry = dict(orch_mod.POST_RESPONSE_HOOKS)
    assert "thread_dispute" in registry
    monkeypatch.setattr(
        orch_mod, "POST_RESPONSE_HOOKS", [("thread_dispute", registry["thread_dispute"])]
    )
    orch = SimpleNamespace(
        correction_detector=CorrectionDetector(),
        memory_system=SimpleNamespace(
            thread_store=store,
            corpus_manager=SimpleNamespace(get_recent_memories=lambda n: list(history)[:n]),
        ),
    )
    orch_mod.run_post_response_hooks(
        orch_mod.PostResponseHookContext(orchestrator=orch, user_input=user_input)
    )


def _fixture(*threads):
    chroma = _Chroma()
    store = ThreadStore(chroma_store=chroma)
    for t in threads:
        store.store_thread(t)
    return chroma, store


def test_correction_disputes_overlapping_open_thread(monkeypatch):
    chroma, store = _fixture(_thread("t1", "Thursday check-in with professor"))
    _run(monkeypatch, store, CORRECTION)
    assert _status(chroma, "t1") == [ThreadStatus.STALE.value]
    assert store.list_open_threads() == []


def test_non_correction_turn_changes_nothing(monkeypatch):
    chroma, store = _fixture(_thread("t1", "Thursday check-in with professor"))
    _run(monkeypatch, store, PLAIN)
    assert _status(chroma, "t1") == [ThreadStatus.OPEN.value]


def test_correction_without_overlap_changes_nothing(monkeypatch):
    chroma, store = _fixture(_thread("t1", "File taxes by April"))
    _run(monkeypatch, store, CORRECTION)
    assert _status(chroma, "t1") == [ThreadStatus.OPEN.value]


def test_single_keyword_overlap_is_not_enough(monkeypatch):
    chroma, store = _fixture(_thread("t1", "Call the professor about grades"))
    _run(monkeypatch, store, CORRECTION)
    assert _status(chroma, "t1") == [ThreadStatus.OPEN.value]


def test_at_most_two_threads_disputed_per_turn(monkeypatch):
    chroma, store = _fixture(
        _thread("a", "Thursday check-in with professor"),
        _thread("b", "Thursday meeting with professor"),
        _thread("c", "Thursday professor check-in notes"),
    )
    _run(monkeypatch, store, CORRECTION)
    stale = [t for t in "abc" if _status(chroma, t) == [ThreadStatus.STALE.value]]
    assert len(stale) == 2
    assert sum(_status(chroma, t) == [ThreadStatus.OPEN.value] for t in "abc") == 1


def test_hook_never_raises_into_the_turn(monkeypatch, caplog):
    class _Boom:
        def list_open_threads(self):
            raise RuntimeError("store down")

    registry = dict(orch_mod.POST_RESPONSE_HOOKS)
    orch = SimpleNamespace(
        correction_detector=CorrectionDetector(),
        memory_system=SimpleNamespace(thread_store=_Boom()),
    )
    ctx = orch_mod.PostResponseHookContext(orchestrator=orch, user_input=CORRECTION)
    with caplog.at_level("DEBUG"):
        registry["thread_dispute"](ctx)  # direct call: the hook itself swallows
    assert any("[ThreadDispute] skipped" in r.getMessage() for r in caplog.records)


def test_missing_collaborators_are_a_noop(monkeypatch):
    registry = dict(orch_mod.POST_RESPONSE_HOOKS)
    for orch in (None, SimpleNamespace(), SimpleNamespace(correction_detector=CorrectionDetector())):
        registry["thread_dispute"](
            orch_mod.PostResponseHookContext(orchestrator=orch, user_input=CORRECTION)
        )


PREV = {"query": "what is on my calendar", "response":
        "you've got the Prof. Rivera check-in at 2:30 Thursday, which lands in the middle of the haircut"}
LIVE = ("Actually, I don't believe that is correct. I don't think it is standing and "
        "I emailed him yesterday with no response")


def test_live_shape_overlap_comes_from_the_corrected_reply(monkeypatch):
    # Message alone shares no keyword with the thread; the corrected reply does.
    chroma, store = _fixture(_thread("t1", "Thursday check-in with Prof. Rivera at 2:30 PM"))
    own_turn = {"query": LIVE, "response": "Fair, I will drop it."}  # already stored: skipped
    _run(monkeypatch, store, LIVE, history=[own_turn, PREV])
    assert _status(chroma, "t1") == [ThreadStatus.STALE.value]
    # without the previous reply there is no overlap -> untouched
    chroma2, store2 = _fixture(_thread("t1", "Thursday check-in with Prof. Rivera at 2:30 PM"))
    _run(monkeypatch, store2, LIVE, history=[])
    assert _status(chroma2, "t1") == [ThreadStatus.OPEN.value]


def test_correction_about_an_unrelated_reply_changes_nothing(monkeypatch):
    chroma, store = _fixture(_thread("t1", "Thursday check-in with Prof. Rivera at 2:30 PM"))
    other = {"query": "q", "response": "Your lease renewal form is due in March"}
    _run(monkeypatch, store, LIVE, history=[other])
    assert _status(chroma, "t1") == [ThreadStatus.OPEN.value]


def test_non_correction_turn_mentioning_the_thread_changes_nothing(monkeypatch):
    chroma, store = _fixture(_thread("t1", "Thursday check-in with Prof. Rivera at 2:30 PM"))
    _run(monkeypatch, store, "Thanks, the Rivera check-in on Thursday at 2:30 is fine", history=[PREV])
    assert _status(chroma, "t1") == [ThreadStatus.OPEN.value]
