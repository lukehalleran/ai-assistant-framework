"""Event-loop hygiene: coroutines schedule through the RUNNING loop, and the
novelty stage's embedder.encode runs off the loop.

Incident: oct27 wave 2 lane L1 (M-7), 2026-10-27. Class: BC-41 (blocking work on
the event loop), BC-69.  M-7's AST scan found ``embedder.encode`` called
directly inside ``_stage_3_novelty_external`` (a CPU/GPU block on the loop at
shutdown synthesis) and five ``asyncio.get_event_loop()`` calls inside
coroutines.  Drives the deployed methods; the embedder / Tavily client are fakes.
"""
import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import knowledge.semantic_search as semantic_search
import models.model_manager as mm_mod
from knowledge.synthesis_filter import SynthesisFilter


class _Embedder:
    def __init__(self):
        self.threads = []

    def encode(self, texts, normalize_embeddings=True):
        self.threads.append(threading.current_thread())
        return np.array([[1.0, 0.0], [0.0, 1.0]])


@pytest.mark.asyncio
async def test_novelty_stage_encodes_off_the_event_loop(monkeypatch):
    emb = _Embedder()
    monkeypatch.setattr(mm_mod.ModelManager, "_get_cached_embedder", staticmethod(lambda: emb))
    monkeypatch.setattr(semantic_search, "semantic_search_with_neighbors", lambda claim, k=3: [])
    flt = SynthesisFilter.__new__(SynthesisFilter)
    cand = SimpleNamespace(connection_claim="A links to B", concept_a="alpha", concept_b="beta", walk_path=None)
    result = SimpleNamespace(candidate=cand, novelty_score_external=0.0, nearest_known_external="",
                             cooccurrence_similarity=0.0, template_similarity=0.0)
    stage = await flt._stage_3_novelty_external(result)
    assert stage.passed
    assert emb.threads, "embedder was never called"
    assert emb.threads[0] is not threading.main_thread()   # base: runs on the loop thread (main)


@pytest.mark.asyncio
async def test_tavily_calls_use_the_running_loop_not_get_event_loop(monkeypatch):
    from knowledge.web_search_manager import WebSearchManager

    def _boom():
        raise AssertionError("asyncio.get_event_loop() called inside a coroutine")

    monkeypatch.setattr(asyncio, "get_event_loop", _boom)
    client = SimpleNamespace(
        search=lambda **kw: {"results": [{"url": "https://example.org/a", "title": "t", "content": "c"}]},
        extract=lambda urls: {"results": [{"url": urls[0], "title": "t", "raw_content": "body"}]},
    )
    wsm = WebSearchManager.__new__(WebSearchManager)
    wsm._tavily_client = client
    wsm.max_content_chars = 1000
    found = await wsm._tavily_search("quiet question", 3)
    assert [p.url for p in found] == ["https://example.org/a"]
    pages = await wsm._tavily_extract(["https://example.org/a"])
    assert pages[0].content == "body"
