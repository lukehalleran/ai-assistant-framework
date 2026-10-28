"""Behavioural siblings: the four exemplar adopters reuse per-text vectors (source-text rows).

Incident date: 2026-08-21 (tone, need, intent and web-trigger prototypes were keyed on
``store.version`` and re-encoded the WHOLE seed+learned set on every accepted exemplar).
Class: BC-63 (a grep for ``encode_texts_cached`` stands in for "a version bump encodes only
the new text"), BC-82-adjacent cache keyed on the wrong thing.

For each adopter the deployed prototype/anchor builder runs twice against a counting
embedder: once cold (encodes every seed), then after the adaptive store learns ONE new
exemplar (version bump) — the second build must encode exactly that one text. A module
that stopped using the shared per-text cache (or shared one cache across modules) fails
the cold-vs-warm accounting here.
"""
import numpy as np
import pytest

import utils.adaptive_exemplars as ae
from models.model_manager import ModelManager


class CountingEmbedder:
    def __init__(self):
        self.batches = []

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=False, **kw):
        texts = [texts] if isinstance(texts, str) else list(texts)
        self.batches.append(texts)
        out = np.zeros((len(texts), 8), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, sum(map(ord, t)) % 8] = 1.0
        return out

    @property
    def encoded(self):
        return [t for b in self.batches for t in b]


class FakeStore:
    """Minimal adaptive store: a version and per-(domain,label) learned exemplars."""
    def __init__(self):
        self.version = 0
        self.learned = {}

    def get_learned(self, domain, label):
        return list(self.learned.get((domain, label), []))

    def learn(self, domain, label, text):
        self.learned.setdefault((domain, label), []).append(text)
        self.version += 1


@pytest.fixture
def world(monkeypatch):
    emb, store = CountingEmbedder(), FakeStore()
    monkeypatch.setattr(ae, "get_store", lambda: store)
    monkeypatch.setattr(ModelManager, "_get_cached_embedder", staticmethod(lambda: emb))
    return emb, store


def _assert_only_new_text_reencoded(emb, store, build, domain, label):
    build()
    cold = len(emb.encoded)
    assert cold > 5, "cold build encoded the seed set"
    emb.batches.clear()
    build()
    assert emb.encoded == [], "same store version must not re-encode anything"
    store.learn(domain, label, "a freshly learned exemplar sentence")
    build()
    assert emb.encoded == ["a freshly learned exemplar sentence"]


def test_tone_prototypes_reencode_only_the_new_exemplar(world):
    import utils.tone_detector as td
    emb, store = world
    _assert_only_new_text_reencoded(
        emb, store, td._get_exemplar_embeddings, "tone", next(iter(td.CRISIS_EXEMPLARS)))


def test_need_prototypes_reencode_only_the_new_exemplar(world):
    import utils.need_detector as nd
    emb, store = world
    _assert_only_new_text_reencoded(
        emb, store, nd._get_need_exemplar_embeddings, "need", next(iter(nd.NEED_EXEMPLARS)))


def test_intent_prototypes_reencode_only_the_new_exemplar(world):
    import core.intent_classifier as ic
    emb, store = world
    _assert_only_new_text_reencoded(
        emb, store, ic._get_intent_prototypes, "intent", next(iter(ic.INTENT_EXEMPLARS)))


def test_web_trigger_anchors_reencode_only_the_new_phrase(world):
    import utils.web_search_trigger as wst
    emb, store = world
    _assert_only_new_text_reencoded(
        emb, store, wst._get_search_anchors, "web_search", "search_worthy")


def test_adopters_keep_separate_caches(world):
    """One adopter's vectors are never served to another (different embedding spaces)."""
    import core.intent_classifier as ic
    import utils.need_detector as nd
    import utils.tone_detector as td
    import utils.web_search_trigger as wst
    caches = [td._exemplar_text_emb_cache, nd._need_text_emb_cache,
              ic._intent_text_emb_cache, wst._anchor_text_emb_cache]
    assert len({id(c) for c in caches}) == 4
    emb, _ = world
    td._get_exemplar_embeddings()
    assert td._exemplar_text_emb_cache and not nd._need_text_emb_cache
    emb.batches.clear()
    nd._get_need_exemplar_embeddings()
    assert emb.encoded, "need prototypes must encode their own seeds, not reuse tone vectors"
