"""B6 image latency regressions against deployed ingestion and startup code."""

import asyncio
import base64
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from knowledge.visual_memory_pipeline import VisualMemoryPipeline


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["hash", "dedup", "clip", "entities", "store", "caption"])
async def test_ingest_blocking_steps_leave_loop_responsive(stage, monkeypatch, tmp_path):
    path = tmp_path / "sample.png"
    path.write_bytes(b"synthetic-image")
    loop_thread = threading.get_ident()
    events = []
    worker_threads = []
    clip = SimpleNamespace(encode_image_from_path=lambda _: [1.0])
    store = SimpleNamespace(has_hash=lambda _: False, add_image=lambda **_: "image-id")

    async def generate_async(*args, **kwargs):
        async def chunks():
            yield "A sample image."
        return chunks()

    pipeline = VisualMemoryPipeline(
        clip, store, model_manager=SimpleNamespace(generate_async=generate_async),
    )
    target, attr = {
        "hash": (pipeline, "_compute_hash"),
        "dedup": (store, "has_hash"),
        "clip": (clip, "encode_image_from_path"),
        "entities": (pipeline, "_extract_entities"),
        "store": (store, "add_image"),
        "caption": (base64, "b64encode"),
    }[stage]
    original = getattr(target, attr)

    def slow_step(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        events.append("started")
        time.sleep(0.2)
        result = original(*args, **kwargs)
        events.append("finished")
        return result

    monkeypatch.setattr(target, attr, slow_step)

    async def heartbeat():
        while "started" not in events:
            await asyncio.sleep(0.001)
        await asyncio.sleep(0.05)
        events.append("heartbeat")

    result, _ = await asyncio.gather(pipeline.ingest_image(str(path)), heartbeat())
    assert result == "image-id"
    assert events.index("heartbeat") < events.index("finished")
    assert len(worker_threads) == 1
    assert worker_threads[0] != loop_thread


@pytest.mark.asyncio
async def test_concurrent_ingests_serialize_store_operations(tmp_path):
    path = tmp_path / "sample.png"
    path.write_bytes(b"synthetic-image")
    active = 0
    peak = 0
    completed = []

    def add_image(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        time.sleep(0.05)
        completed.append(kwargs["source"])
        active -= 1
        return kwargs["source"]

    pipeline = VisualMemoryPipeline(
        SimpleNamespace(encode_image_from_path=lambda _: [1.0]),
        SimpleNamespace(has_hash=lambda _: False, add_image=add_image),
    )
    results = await asyncio.gather(
        pipeline.ingest_image(str(path), source="first"),
        pipeline.ingest_image(str(path), source="second"),
    )
    assert results == ["first", "second"]
    assert sorted(completed) == ["first", "second"]
    assert peak == 1


def test_clip_load_is_serialized_between_warmup_and_ingestion(monkeypatch):
    from knowledge.clip_manager import CLIPManager

    manager = CLIPManager()
    started = threading.Event()
    release = threading.Event()
    create_calls = []

    def create_model(*args, **kwargs):
        create_calls.append(threading.get_ident())
        started.set()
        assert release.wait(2), "test did not release fake model construction"
        return MagicMock(), None, MagicMock()

    open_clip = SimpleNamespace(
        create_model_and_transforms=create_model,
        get_tokenizer=lambda _: MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "open_clip", open_clip)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    warmup = threading.Thread(target=manager.load)
    second_entered = threading.Event()

    def concurrent_load():
        second_entered.set()
        manager.load()

    ingest = threading.Thread(target=concurrent_load)
    try:
        warmup.start()
        assert started.wait(2)
        ingest.start()
        assert second_entered.wait(2)
        # Let the second caller reach the lock (or duplicate construction).
        time.sleep(0.05)
    finally:
        release.set()
        warmup.join(2)
        if ingest.ident is not None:
            ingest.join(2)
    assert not warmup.is_alive() and not ingest.is_alive()
    assert manager.loaded
    assert len(create_calls) == 1


class _InlineThread:
    def __init__(self, target=None, daemon=None):
        self.target = target

    def start(self):
        self.target()


@pytest.fixture
def warmup_fakes(monkeypatch):
    import config.app_config as config
    import core.prompt.gatherer_knowledge as gatherer
    import knowledge.clip_manager as clip
    import knowledge.visual_memory_store as visual_store
    import memory.memory_retriever as retriever
    import utils.need_detector as needs
    import utils.tone_detector as tone
    import utils.web_search_trigger as trigger

    monkeypatch.setattr(threading, "Thread", _InlineThread)
    monkeypatch.setattr(retriever.MemoryRetriever, "_cross_encoder", MagicMock())
    monkeypatch.setattr(trigger, "_get_search_anchors", MagicMock())
    monkeypatch.setattr(tone, "_get_exemplar_embeddings", MagicMock())
    monkeypatch.setattr(needs, "_get_need_exemplar_embeddings", MagicMock())
    # Refuse the wiki warm slot so no executor submits or index loads occur.
    monkeypatch.setattr(gatherer, "_WIKI_SEM_INFLIGHT", SimpleNamespace(acquire=lambda **_: False))
    monkeypatch.setattr(config, "VISUAL_MEMORY_ENABLED", True)
    manager = MagicMock()
    get_manager = MagicMock(return_value=manager)
    store = MagicMock()
    constructor = MagicMock(return_value=store)
    monkeypatch.setattr(clip, "get_clip_manager", get_manager)
    monkeypatch.setattr(visual_store, "VisualMemoryStore", constructor)
    orchestrator = MagicMock()
    orchestrator.memory_system.get_memories = AsyncMock(return_value=[])
    return SimpleNamespace(
        manager=manager, get_manager=get_manager, store=store,
        constructor=constructor, orchestrator=orchestrator, config=config,
    )


def test_warmup_loads_clip_and_opens_visual_store(warmup_fakes):
    from gui.launch import _run_model_warmup

    _run_model_warmup(warmup_fakes.orchestrator)
    warmup_fakes.get_manager.assert_called_once_with()
    warmup_fakes.manager.load.assert_called_once_with()
    warmup_fakes.constructor.assert_called_once()
    warmup_fakes.store.load.assert_called_once_with()
    warmup_fakes.store.save.assert_not_called()


@pytest.mark.parametrize("failure", ["clip", "store"])
def test_warmup_visual_failure_is_guarded(failure, warmup_fakes, capsys):
    from gui.launch import _run_model_warmup

    target = warmup_fakes.manager if failure == "clip" else warmup_fakes.store
    target.load.side_effect = RuntimeError("synthetic warmup failure")
    _run_model_warmup(warmup_fakes.orchestrator)
    output = capsys.readouterr().out
    assert "[Warmup] clip skip: synthetic warmup failure" in output
    assert "[Warmup] Model warmup complete" in output


def test_warmup_skips_visual_models_when_disabled(warmup_fakes, monkeypatch):
    from gui.launch import _run_model_warmup

    monkeypatch.setattr(warmup_fakes.config, "VISUAL_MEMORY_ENABLED", False)
    _run_model_warmup(warmup_fakes.orchestrator)
    warmup_fakes.get_manager.assert_not_called()
    warmup_fakes.constructor.assert_not_called()
