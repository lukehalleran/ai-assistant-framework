"""Chroma keep-warm touch (incident 2026-10-28, class: BC-41, BC-70).

Opening a collection reads nothing (CHR: 0.5 ms), so warmup step 12 -- which only
opened collections -- left the first turn to pay the cold-cache query. The touch
reads one row per collection, read-only, bounded, and never raises.
"""
import threading
from types import SimpleNamespace

import pytest

from memory.storage import chroma_keepwarm as kw


class _Coll:
    def __init__(self, name, log, fail=None, block=None):
        self.name, self.log, self.fail, self.block = name, log, fail, block

    def query(self, **kwargs):
        self.log.append((self.name, "query", kwargs))
        if self.block is not None:
            self.block.wait(5)
        if self.fail == "query":
            raise RuntimeError("boom")
        return {}

    def get(self, **kwargs):
        self.log.append((self.name, "get", kwargs))
        if self.fail == "get":
            raise RuntimeError("boom")
        return {}

    def add(self, **kw):  # a write would be a bug
        raise AssertionError("keep-warm must be read-only")

    upsert = update = delete = add


@pytest.fixture(autouse=True)
def _no_outstanding_workers(monkeypatch):
    """A timed-out worker from one test must not make another test skip its collection."""
    monkeypatch.setattr(kw, "_OUTSTANDING", {}, raising=False)


def _join_worker(name):
    for t in threading.enumerate():
        if t.name == f"keepwarm-{name}":
            t.join(2)


def _store(names, **per):
    log = []
    colls = {n: _Coll(n, log, **per.get(n, {})) for n in names}
    store = SimpleNamespace(collections={n: None for n in names}, embedding_fn=None)
    store._get_collection = lambda n: colls[n]
    return store, log


def test_touches_each_collection_read_only_with_a_fixed_vector():
    store, log = _store(["conversations", "threads", "reference_docs", "facts"])
    res = kw.touch_collections(store, ["conversations", "threads", "reference_docs", "facts"])
    assert set(res) == {"conversations", "threads", "reference_docs", "facts"}
    assert all(isinstance(v, float) and v >= 0 for v in res.values())
    q = [c for c in log if c[1] == "query"]
    assert len(q) == 4
    for _n, _op, kwargs in q:
        assert "query_texts" not in kwargs  # never embeds
        assert kwargs["n_results"] == 1 and len(kwargs["query_embeddings"][0]) == 384
    gets = {c[0]: c[2] for c in log if c[1] == "get"}
    assert gets["threads"]["where"] == {"status": "open"}
    assert gets["reference_docs"]["where"] == {"type": "user_upload"}
    assert "where" not in gets["conversations"] and gets["conversations"]["limit"] == 1


def test_a_failing_collection_is_reported_and_the_others_still_run():
    store, log = _store(["a", "b", "c"], b={"fail": "query"})
    res = kw.touch_collections(store, ["a", "b", "c"])
    assert res["b"] is None and res["a"] is not None and res["c"] is not None
    assert ("c", "get") in [(n, op) for n, op, _ in log]
    assert ("b", "get") in [(n, op) for n, op, _ in log]  # query failure does not skip the get


def test_a_hung_collection_times_out_without_stopping_the_pass():
    gate = threading.Event()
    store, _log = _store(["a", "b"], a={"block": gate})
    try:
        res = kw.touch_collections(store, ["a", "b"], per_collection_timeout_s=0.05)
    finally:
        gate.set()
    assert res["a"] is None and res["b"] is not None


def test_total_budget_skips_the_rest():
    ticks = iter(range(0, 1000, 10))
    store, _ = _store(["a", "b"])
    res = kw.touch_collections(store, ["a", "b"], total_budget_s=5, clock=lambda: next(ticks))
    assert res["b"] is None


def test_never_raises_on_a_broken_store():
    assert kw.touch_collections(object(), ["x"]) == {"x": None}
    bad = SimpleNamespace(collections={"a": None})
    bad._get_collection = lambda n: (_ for _ in ()).throw(ValueError("nope"))
    assert kw.touch_collections(bad, ["a"]) == {"a": None}


def test_default_list_is_the_turn_path_set_and_excludes_faiss_backed():
    d = set(kw.DEFAULT_COLLECTIONS)
    assert {"conversations", "facts", "summaries", "threads", "obsidian_notes"} <= d
    assert not ({"wiki_knowledge", "visual_memories"} & d)


def test_config_collections_override_is_read_live(monkeypatch):
    from config import app_config
    store, log = _store(["facts", "threads"])
    monkeypatch.setattr(app_config, "CHROMA_KEEPWARM_COLLECTIONS", ["facts"])
    res = kw.touch_from_config(store)
    assert list(res) == ["facts"]


def test_warmup_step_uses_the_touch(monkeypatch):
    from gui import launch

    seen = {}
    monkeypatch.setattr(kw, "touch_from_config",
                        lambda store, extra=(): seen.update(store=store, extra=tuple(extra)) or {})

    class _Inline:
        def __init__(self, target=None, daemon=None):
            self._t = target

        def start(self):
            self._t()

    store = object()
    orch = SimpleNamespace(memory_system=SimpleNamespace(chroma_store=store), model_manager=None)
    import knowledge.clip_manager as clip
    import utils.need_detector as needs
    import utils.tone_detector as tone
    from unittest.mock import MagicMock, patch
    monkeypatch.setattr(clip, "get_clip_manager", MagicMock())
    monkeypatch.setattr(tone, "_get_exemplar_embeddings", MagicMock())
    monkeypatch.setattr(needs, "_get_need_exemplar_embeddings", MagicMock())
    with patch("threading.Thread", _Inline):
        launch._run_model_warmup(orch)
    assert seen["store"] is store
    assert set(launch._TURN1_COLD_COLLECTIONS) <= set(seen["extra"])


def test_a_late_error_from_a_timed_out_worker_is_not_charged_to_the_next_collection():
    # Review finding 2026-10-28: the worker closure read the loop's `box`, so "a" erroring after
    # its timeout wrote into "b"'s box and b was reported failed.
    release = threading.Event()

    class _LateFail(_Coll):
        def query(self, **kwargs):
            release.wait(5)
            raise RuntimeError("late")

    class _Releaser(_Coll):
        def query(self, **kwargs):
            release.set()
            _join_worker("a")  # a's late error is recorded before b's result is read
            return {}

    log = []
    colls = {"a": _LateFail("a", log), "b": _Releaser("b", log)}
    store = SimpleNamespace(collections={"a": None, "b": None}, embedding_fn=None)
    store._get_collection = lambda n: colls[n]
    res = kw.touch_collections(store, ["a", "b"], per_collection_timeout_s=0.05)
    assert res["a"] is None
    assert res["b"] is not None


def test_a_hung_touch_is_not_restarted_until_it_returns():
    gate = threading.Event()
    store, log = _store(["a", "b"], a={"block": gate})
    try:
        first = kw.touch_collections(store, ["a", "b"], per_collection_timeout_s=0.05)
        second = kw.touch_collections(store, ["a", "b"], per_collection_timeout_s=0.05)
    finally:
        gate.set()
    assert first["a"] is None and second["a"] is None and second["b"] is not None
    assert [n for n, op, _ in log if op == "query"].count("a") == 1  # no second worker while the first hangs
    _join_worker("a")
    third = kw.touch_collections(store, ["a", "b"], per_collection_timeout_s=0.5)
    assert third["a"] is not None
