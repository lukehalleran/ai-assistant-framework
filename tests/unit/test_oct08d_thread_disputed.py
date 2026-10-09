"""A user correction can DISPUTE a thread (2026-10-08, class: BC-58, BC-52).

detect_resolutions gains a typed third outcome ("disputed"); the store maps it
to the existing stale operation. Per-turn correction wiring is NOT covered:
no correction verdict reaches MemoryCoordinator.store_interaction (see handoff).
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from memory.thread_extractor import ThreadExtractor
from memory.thread_models import DisputedResolution, OpenThread, ThreadStatus, ThreadType
from memory.thread_store import COLLECTION_NAME, ThreadStore


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


def _thread(tid, topic="Thursday standing meeting"):
    return OpenThread(thread_id=tid, topic=topic, summary="s", thread_type=ThreadType.DEADLINE)


def _detect(threads, payload):
    mm = MagicMock()
    mm.generate_once = AsyncMock(return_value=json.dumps(payload))
    convos = [{"query": "Don't think it is standing", "response": "Fair."}]
    return asyncio.run(ThreadExtractor(model_manager=mm).detect_resolutions(convos, threads))


def _status(chroma, tid):
    return [i["metadata"]["status"] for i in chroma.list_all(COLLECTION_NAME)
            if i["metadata"]["thread_id"] == tid]


def test_disputed_item_is_typed_and_marks_thread_stale():
    chroma = _Chroma()
    store = ThreadStore(chroma_store=chroma)
    t = _thread("t1")
    store.store_thread(t)
    out = _detect([t], [{"thread_id": "t1", "resolution": "user says it is not standing",
                         "outcome": "disputed"}])
    assert len(out) == 1 and isinstance(out[0][1], DisputedResolution)
    # the caller (ShutdownProcessor) applies every tuple through resolve_thread
    assert store.resolve_thread(*out[0]) is True
    assert _status(chroma, "t1") == [ThreadStatus.STALE.value]
    assert store.list_open_threads() == []


def test_done_and_cancelled_outcomes_unchanged():
    chroma = _Chroma()
    store = ThreadStore(chroma_store=chroma)
    a, b = _thread("a", "File taxes"), _thread("b", "Call landlord")
    store.store_thread(a)
    store.store_thread(b)
    out = _detect([a, b], [
        {"thread_id": "a", "resolution": "done", "outcome": "resolved"},
        {"thread_id": "b", "resolution": "cancelled by user"},  # legacy shape, no outcome
    ])
    assert not any(isinstance(r, DisputedResolution) for _, r in out)
    for tid, res in out:
        assert store.resolve_thread(tid, res) is True
    assert _status(chroma, "a") == [ThreadStatus.RESOLVED.value]
    assert _status(chroma, "b") == [ThreadStatus.RESOLVED.value]


def test_resolution_prompt_defines_disputed_outcome():
    mm = MagicMock()
    mm.generate_once = AsyncMock(return_value="[]")
    t = _thread("t1")
    asyncio.run(ThreadExtractor(model_manager=mm).detect_resolutions(
        [{"query": "x", "response": "y"}], [t]))
    prompt = mm.generate_once.call_args.args[0]
    assert "DISPUTED" in prompt and '"outcome"' in prompt
