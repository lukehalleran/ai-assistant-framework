"""Blocking calls inside ``async def`` run OFF the event loop (BC-41), batch B.

Companion to ``test_event_loop_blocking_sites.py``. Each test drives the
deployed coroutine with minimal fakes, records the thread the blocking call
actually runs on, and compares it with the event-loop thread. On the pre-fix
code every one of these calls ran ON the loop thread.

Sites: ``UnifiedPromptBuilder.build_prompt`` (query pre-embed),
``MultiCollectionChromaStore.query_multiple_collections`` (query pre-embed),
``KnowledgeRetrievalMixin.get_relevant_emails`` (two embedder encodes),
``ContentHygiene._backfill_recent_conversations`` (embedder encode),
``FileAccessManager.grep_files`` (grep subprocess), ``GET /api/graph``
(graph JSON read).

``build_prompt`` is driven through the same ``full_builder`` harness
``test_prompt_timeout.py`` uses, i.e. the real coroutine.
"""

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from tests.unit.test_independent_prompt_audit import full_builder, retrieval_limits

TIMEOUT_S = 5


def _loop_thread():
    return threading.get_ident()


@pytest.mark.asyncio
async def test_build_prompt_pre_embed_runs_off_loop(monkeypatch):
    builder = full_builder(monkeypatch, [])
    seen = []

    class _Chroma:
        def clear_embedding_cache(self):
            pass

        def _cached_embed(self, text):
            seen.append((text, threading.get_ident()))
            return [0.0]

    builder.memory_coordinator.chroma_store = _Chroma()
    loop_tid = _loop_thread()
    result = await asyncio.wait_for(
        builder.build_prompt("Synthetic question", retrieval_overrides=retrieval_limits()),
        TIMEOUT_S,
    )
    assert "_build_time" in result, "builder must not fall back to its error path"
    pre_embed = [tid for text, tid in seen if text == "Synthetic question"]
    assert pre_embed, "build_prompt must pre-embed the query"
    assert all(tid != loop_tid for tid in pre_embed), "pre-embed ran on the event loop"


@pytest.mark.asyncio
async def test_query_multiple_collections_pre_embed_runs_off_loop():
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

    store = MultiCollectionChromaStore.__new__(MultiCollectionChromaStore)
    store.collections = {}
    seen = []

    def _embed(text):
        seen.append(threading.get_ident())
        return [0.1, 0.2]

    store._cached_embed = _embed
    loop_tid = _loop_thread()
    out = await asyncio.wait_for(
        store.query_multiple_collections(["conversations"], "synthetic query"), TIMEOUT_S
    )
    assert out == {"conversations": []}
    assert seen and all(tid != loop_tid for tid in seen), "pre-embed ran on the event loop"


@pytest.mark.asyncio
async def test_get_relevant_emails_encodes_off_loop(monkeypatch):
    import core.actions.google_contacts as gc
    import core.email.service as svc
    import core.prompt.gatherer_knowledge as gk
    from core.email.provider import EmailMessage
    from models.model_manager import ModelManager

    msg = EmailMessage(
        provider="gmail", message_id="m1", sender="Morgan <morgan@example.com>",
        subject="Fall registration", snippet="You are all set.",
        date="2026-08-28T10:00:00",
    )

    class _Service:
        async def search(self, *a, **k):
            return [msg]

    async def _resolve(name, **k):
        return []

    encode_threads = []

    class _Embedder:
        def encode(self, text, **k):
            encode_threads.append(threading.get_ident())
            return np.array([1.0, 0.0])

    monkeypatch.setattr(svc, "get_email_service", lambda: _Service())
    monkeypatch.setattr(gc, "resolve_contact", _resolve)
    monkeypatch.setattr(ModelManager, "_get_cached_embedder", staticmethod(lambda: _Embedder()))

    gatherer = gk.KnowledgeRetrievalMixin.__new__(gk.KnowledgeRetrievalMixin)
    gatherer._distress_active = False
    loop_tid = _loop_thread()
    out = await asyncio.wait_for(
        gk.KnowledgeRetrievalMixin.get_relevant_emails(gatherer, "check my inbox"), TIMEOUT_S
    )
    assert out and out[0]["subject"] == "Fall registration"
    assert len(encode_threads) == 2, "query + one email text must both be encoded"
    assert all(tid != loop_tid for tid in encode_threads), "encode ran on the event loop"


@pytest.mark.asyncio
async def test_hygiene_backfill_encode_runs_off_loop():
    from core.prompt.hygiene import ContentHygiene

    items = [
        {"query": "alpha question", "response": "alpha answer"},
        {"query": "beta question", "response": "beta answer"},
    ]
    corpus = SimpleNamespace(get_recent_memories=lambda count: items[:count])
    hygiene = ContentHygiene(SimpleNamespace(corpus_manager=corpus), SimpleNamespace())
    encode_threads = []

    class _Embedder:
        def encode(self, text, **k):
            encode_threads.append(threading.get_ident())
            return np.array([1.0, 0.0]) if "alpha" in text else np.array([0.0, 1.0])

    loop_tid = _loop_thread()
    out = await asyncio.wait_for(
        hygiene._backfill_recent_conversations(
            existing_items=[], seen_embeddings=[], seen_content=set(),
            target_count=2, offset=0, embedder=_Embedder(), similarity_threshold=0.9,
        ),
        TIMEOUT_S,
    )
    assert len(out) == 2
    assert len(encode_threads) == 2
    assert all(tid != loop_tid for tid in encode_threads), "encode ran on the event loop"


@pytest.mark.asyncio
async def test_grep_files_subprocess_runs_off_loop(tmp_path, monkeypatch):
    import core.file_access_manager as fam

    (tmp_path / "a.py").write_text("needle here\n")
    manager = fam.FileAccessManager(approved_folders=[str(tmp_path)])
    run_threads = []
    real_run = fam.subprocess.run

    def _run(cmd, **kwargs):
        run_threads.append(threading.get_ident())
        return real_run(cmd, **kwargs)

    monkeypatch.setattr("core.file_access_manager.subprocess.run", _run)
    loop_tid = _loop_thread()
    out = await asyncio.wait_for(manager.grep_files("needle"), TIMEOUT_S)
    assert out["success"] and out["total_matches"] >= 1
    assert run_threads, "grep must go through subprocess.run"
    assert all(tid != loop_tid for tid in run_threads), "grep ran on the event loop"


@pytest.mark.asyncio
async def test_graph_route_reads_json_off_loop(tmp_path, monkeypatch):
    import json as _json

    from api.routes import system

    path = tmp_path / "kg.json"
    path.write_text(_json.dumps({"nodes": {}, "edges": []}))
    monkeypatch.setattr("config.app_config.KNOWLEDGE_GRAPH_PERSIST_PATH", str(path))
    load_threads = []
    real_load = _json.load

    def _load(fh, *a, **k):
        load_threads.append(threading.get_ident())
        return real_load(fh, *a, **k)

    monkeypatch.setattr(system.json, "load", _load)
    loop_tid = _loop_thread()
    out = await asyncio.wait_for(system.graph(request=None, limit=10), TIMEOUT_S)
    assert out == {"nodes": [], "edges": []}
    assert load_threads and all(tid != loop_tid for tid in load_threads), \
        "graph JSON read ran on the event loop"
