"""Stage 3 of the synthesis filter runs the sync FAISS wiki search off the event loop.

Incident date: 2026-10-27 (wave-2 follow-up to the stage-2 encode fix).  Class: BC-41
(blocking work on the loop) -- ``semantic_search_with_neighbors`` (IVFPQ over the wiki
index) was called directly inside the coroutine ``_stage_3_novelty_external``.

Drives the deployed coroutine; the search and embedder are fakes that record the thread.
"""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import knowledge.semantic_search as semantic_search
import models.model_manager as mm_mod
from knowledge.synthesis_filter import SynthesisFilter


class _Embedder:
    def encode(self, texts, normalize_embeddings=True):
        return np.array([[1.0, 0.0], [0.0, 1.0]])


@pytest.mark.parametrize("claim", ["A links to B", "  A links\n   to B  "])
async def test_faiss_claim_search_runs_off_the_event_loop(monkeypatch, claim):
    threads = []

    def _search(q, k=8):
        threads.append(threading.current_thread())
        assert k == 3
        return []

    monkeypatch.setattr(mm_mod.ModelManager, "_get_cached_embedder", staticmethod(lambda: _Embedder()))
    monkeypatch.setattr(semantic_search, "semantic_search_with_neighbors", _search)
    flt = SynthesisFilter.__new__(SynthesisFilter)
    cand = SimpleNamespace(connection_claim=claim, concept_a="alpha", concept_b="beta", walk_path=None)
    result = SimpleNamespace(candidate=cand, novelty_score_external=0.0, nearest_known_external="",
                             cooccurrence_similarity=0.0, template_similarity=0.0)
    stage = await flt._stage_3_novelty_external(result)
    assert stage.passed
    assert threads, "claim search was never called"
    assert threads[0] is not threading.main_thread()
