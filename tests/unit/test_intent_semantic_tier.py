"""
Tests for the intent classifier's semantic tier + adaptive learning
(2026-08-03).

Gap this closes: regex-first classification had NO semantic channel — a
distress vent with no regex hit landed general@0.00 (every 2026-08-02 vent
did), starving the tone-corroborated agentic veto. The semantic tier scores
queries against per-intent exemplar prototypes (seeds + per-user learned)
when regex is unconfident, capped at the 0.60 routing floor. Teachers are
independent channels only: confident regex hits (≥0.85) and STM refinements;
the semantic tier never teaches itself and GENERAL is never learned.
"""

import numpy as np
import pytest
from unittest.mock import patch

import core.intent_classifier as ic
from core.intent_classifier import IntentClassifier, IntentType
from utils.adaptive_exemplars import get_store


@pytest.fixture(autouse=True)
def _reset_prototype_cache():
    ic._intent_prototype_cache = None
    yield
    ic._intent_prototype_cache = None


def _fake_protos(label_vecs):
    """Prototype dict with unit vectors per label."""
    return {
        label: np.array(v, dtype=float) / np.linalg.norm(v)
        for label, v in label_vecs.items()
    }


class TestSemanticTier:
    def test_vent_routes_to_emotional_support(self):
        clf = IntentClassifier()
        protos = _fake_protos({
            "emotional_support": [1.0, 0.0],
            "technical_help": [0.0, 1.0],
        })

        class FakeEmbedder:
            def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
                return np.array([[0.9, 0.1]] * len(texts))

        with patch.object(ic, "_get_intent_prototypes", return_value=protos), \
             patch("models.model_manager.ModelManager._get_cached_embedder",
                   return_value=FakeEmbedder()):
            result = clf.classify(
                "I am embarrassed for how I reacted earlier. I am so unhappy"
            )
        assert result.intent == IntentType.EMOTIONAL_SUPPORT
        assert result.confidence == 0.60
        assert result.source == "semantic"

    def test_confident_regex_bypasses_semantic(self):
        clf = IntentClassifier()
        with patch.object(ic, "_semantic_intent") as sem:
            result = clf.classify("What's my sister's name?")
        assert result.confidence >= 0.50
        sem.assert_not_called()

    def test_below_threshold_similarity_stays_general(self):
        clf = IntentClassifier()
        protos = _fake_protos({
            "emotional_support": [1.0, 0.0],
            "technical_help": [0.0, 1.0],
        })

        class WeakEmbedder:
            def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
                return np.array([[0.3, 0.29]] * len(texts))  # sims < min_sim

        with patch.object(ic, "_get_intent_prototypes", return_value=protos), \
             patch("models.model_manager.ModelManager._get_cached_embedder",
                   return_value=WeakEmbedder()):
            result = clf.classify("mumble mumble nothing in particular")
        assert result.intent == IntentType.GENERAL
        assert result.source != "semantic"

    def test_disable_flag(self):
        clf = IntentClassifier()
        with patch.dict(ic._SEMANTIC_TIER_CONFIG, {"enabled": False}), \
             patch.object(ic, "_semantic_intent") as sem:
            clf.classify("just some words with no pattern")
        sem.assert_not_called()

    def test_semantic_confidence_stays_below_veto_floor(self):
        assert ic._SEMANTIC_TIER_CONFIG["confidence"] < 0.75


class TestIntentLearning:
    def test_confident_regex_teaches(self):
        clf = IntentClassifier()
        result = clf.classify("What's my sister's name?")
        if result.confidence >= 0.85 and result.intent.value in ic.INTENT_EXEMPLARS:
            learned = get_store().get_learned("intent", result.intent.value)
            assert any("sister" in t for t in learned)
        else:
            pytest.skip("query not regex-confident in this config")

    def test_stm_refinement_teaches(self):
        clf = IntentClassifier()
        weak = clf._build_result(IntentType.GENERAL, 0.0)
        refined = clf.refine_with_stm(
            weak, "seeking emotional support and comfort",
            query="honestly today just broke me a little",
        )
        assert refined.source == "stm_refined"
        assert any(
            "broke me" in t
            for t in get_store().get_learned("intent", refined.intent.value)
        )

    def test_semantic_tier_never_teaches_itself(self):
        clf = IntentClassifier()
        protos = _fake_protos({
            "emotional_support": [1.0, 0.0],
            "technical_help": [0.0, 1.0],
        })

        class FakeEmbedder:
            def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
                return np.array([[0.9, 0.1]] * len(texts))

        with patch.object(ic, "_get_intent_prototypes", return_value=protos), \
             patch("models.model_manager.ModelManager._get_cached_embedder",
                   return_value=FakeEmbedder()):
            clf.classify("a nondescript unhappy sentence with no regex hooks")
        assert get_store().get_learned("intent", "emotional_support") == []

    def test_general_never_learned(self):
        ic._learn_intent_exemplar("some text long enough to store", "general", "regex")
        assert get_store().get_learned("intent", "general") == []

    def test_prototypes_merge_learned(self):
        get_store().record("intent", "project_work",
                           "the dedup script needs a dry run pass first", "t")
        seen = []

        class FakeEmbedder:
            def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
                seen.append(list(texts))
                return np.ones((len(texts), 4))

        with patch("models.model_manager.ModelManager._get_cached_embedder",
                   return_value=FakeEmbedder()):
            protos = ic._get_intent_prototypes()
        assert set(protos) == set(ic.INTENT_EXEMPLARS)
        batch = next(b for b in seen if "dry run pass" in " ".join(b))
        assert set(ic.INTENT_EXEMPLARS["project_work"]).issubset(set(batch))


class TestElevatedToneNeverTeaches:
    """2026-08-21 (08-18 audit): crisis vents that hit a confident regex or an
    STM refinement were being taught as intent exemplars — crisis phrasing was
    becoming the learned prototype for ordinary intents (temporal_recall).
    Elevated tone suppresses BOTH teachers; classification itself is unchanged."""

    def test_regex_teacher_suppressed_on_elevated_tone(self):
        clf = IntentClassifier()
        baseline = clf.classify("What's my sister's name?")
        if baseline.confidence < 0.85 or baseline.intent.value not in ic.INTENT_EXEMPLARS:
            pytest.skip("query not regex-confident in this config")
        # store is per-test sandboxed; re-check from clean state per encoding
        for tone in ("MEDIUM", "CONCERN", "crisis_support", "CrisisLevel.CONCERN"):
            result = clf.classify("What's my sister's name?", tone_level=tone)
            assert result.intent == baseline.intent  # routing unchanged
        assert get_store().get_learned("intent", baseline.intent.value) == [
            "What's my sister's name?"
        ] or get_store().get_learned("intent", baseline.intent.value) == []
        # the ONLY entry (if any) must come from the baseline conversational call
        learned = get_store().get_learned("intent", baseline.intent.value)
        assert len(learned) <= 1

    def test_regex_teacher_still_teaches_on_conversational(self):
        clf = IntentClassifier()
        result = clf.classify("What's my sister's name?", tone_level="CONVERSATIONAL")
        if result.confidence < 0.85 or result.intent.value not in ic.INTENT_EXEMPLARS:
            pytest.skip("query not regex-confident in this config")
        assert any(
            "sister" in t
            for t in get_store().get_learned("intent", result.intent.value)
        )

    def test_stm_teacher_suppressed_on_elevated_tone(self):
        clf = IntentClassifier()
        weak = clf._build_result(IntentType.GENERAL, 0.0)
        refined = clf.refine_with_stm(
            weak, "seeking emotional support and comfort",
            query="honestly today just broke me a little",
            tone_level="MEDIUM",
        )
        # refinement still routes THIS turn...
        assert refined.source == "stm_refined"
        # ...but the crisis phrasing is never learned
        assert get_store().get_learned("intent", refined.intent.value) == []

    def test_stm_teacher_teaches_without_tone(self):
        clf = IntentClassifier()
        weak = clf._build_result(IntentType.GENERAL, 0.0)
        refined = clf.refine_with_stm(
            weak, "seeking emotional support and comfort",
            query="honestly today just broke me a little",
            tone_level="CONVERSATIONAL",
        )
        assert refined.source == "stm_refined"
        assert any(
            "broke me" in t
            for t in get_store().get_learned("intent", refined.intent.value)
        )

    def test_tone_elevation_predicate_both_encodings(self):
        for t in ("HIGH", "MEDIUM", "CONCERN", "light_support",
                  "elevated_support", "crisis_support", "CrisisLevel.HIGH"):
            assert ic._tone_is_elevated(t) is True
        for t in (None, "", "CONVERSATIONAL", "conversational"):
            assert ic._tone_is_elevated(t) is False
