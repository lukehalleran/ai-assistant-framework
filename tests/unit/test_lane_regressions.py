"""Focused regressions for memory/eval/gating audit fixes."""

import asyncio
from unittest.mock import MagicMock

import numpy as np

from eval.schema import PersistenceSnapshot, PromptSnapshot, SnapshotLayer
from memory.cross_deduplicator import CrossCollectionDeduplicator
from processing import gate_system


def test_cross_dedup_duplicate_guard_includes_collection_namespace():
    """Equal IDs in separate Chroma collections are independent documents."""
    store = MagicMock()
    dedup = CrossCollectionDeduplicator(store)
    dedup.duplicate_threshold = 0.9
    docs = [
        {"id": "same", "content": "IDENTICAL", "metadata": {}, "collection": "facts"},
        {"id": "same", "content": "IDENTICAL", "metadata": {}, "collection": "summaries"},
        {"id": "same", "content": "IDENTICAL", "metadata": {}, "collection": "reflections"},
    ]
    embeddings = np.ones((3, 2), dtype=np.float32) / np.sqrt(2)

    pairs = dedup._find_cross_duplicates(docs, embeddings)

    # facts→summaries and reflections→summaries are two valid independent
    # duplicate decisions; a bare-ID guard would incorrectly return only one.
    assert len(pairs) == 2


def test_snapshot_layer_from_dict_does_not_mutate_payload():
    section = {
        "key": "memories", "header": "MEMORIES", "structured_content": [],
        "formatted_text": "", "token_count": 0, "source_field": "memories",
        "category": "memory", "eligible_for_ablation": True,
        "structurally_required": False, "assembly_order": 1, "metadata": {},
    }
    payload = {
        "layer_name": "raw_retrieval", "sections": {"memories": section},
        "layer_content_hash": "h", "prompt_text": None,
        "prompt_hash_exact": None, "prompt_hash_normalized": None,
        "capture_timestamp": "now", "metadata": {},
    }

    SnapshotLayer.from_dict(payload)

    assert "sections" in payload
    assert payload["sections"]["memories"] is section


def test_prompt_snapshot_from_dict_does_not_mutate_payload():
    layer = {
        "layer_name": "raw_retrieval", "sections": {},
        "layer_content_hash": "h", "prompt_text": None,
        "prompt_hash_exact": None, "prompt_hash_normalized": None,
        "capture_timestamp": "now", "metadata": {},
    }
    payload = {
        "snapshot_id": "s", "query_text": "q", "query_timestamp": "now",
        "processed_query": "q", "detected_intent": "chat",
        "detected_tone": "neutral",
        "provenance": {
            "model_name": "m", "git_commit_hash": "g",
            "system_prompt_hash": "h",
        },
        "layers": {"raw_retrieval": layer}, "retrieval_metadata": {},
        "assembly_metadata": {},
    }

    PromptSnapshot.from_dict(payload)

    assert "provenance" in payload
    assert "layers" in payload


def test_persistence_snapshot_from_dict_does_not_mutate_payload():
    payload = {
        "snapshot_id": "s", "capture_timestamp": "now",
        "fingerprints": {"facts": {"name": "facts", "kind": "chroma"}},
    }

    PersistenceSnapshot.from_dict(payload)

    assert "fingerprints" in payload


def test_gated_wiki_fetch_fails_soft_on_arbitrary_fetcher_exception(monkeypatch):
    async def broken_fetch(_query):
        raise RuntimeError("injected fetch failure")

    monkeypatch.setattr(gate_system, "fetch_wiki_with_fallbacks", broken_fetch)
    ok, text = asyncio.run(gate_system.gated_wiki_fetch("what is Python"))

    assert (ok, text) == (False, "")
