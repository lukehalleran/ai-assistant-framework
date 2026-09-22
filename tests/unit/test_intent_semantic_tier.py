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


class TestSTMSelfReportAndCueGuard:
    """2026-09-22 live incident: STM hands refine_with_stm a free-text
    PARAPHRASE of the user's turn, not the user's own words. Two failure
    shapes from the same root cause (paraphrase cue != user's cue):
    (1) a first-person status update ("I looked yesterday just two but I
    will check again...") got paraphrased by STM as "Confirm whether to
    attend..." — the paraphrase's "confirm" hit the FACTUAL_RECALL keyword
    family and the turn was refined+taught as factual_recall, although the
    user was reporting, not asking to recall anything;
    (2) more generally, ANY STM paraphrase can carry a keyword the user
    never wrote, which then taught the learned-exemplar store a prototype
    for a query with no actual intent signal in its own text (class:
    BC-51, BC-52, BC-58)."""

    def test_self_report_never_refines_to_a_recall_target(self):
        # Live query + live (paraphrased) STM intent, verbatim.
        query = (
            "I looked yesterday just two but I will check again before "
            "I meet up with my dad"
        )
        stm_intent = (
            "Confirm whether to attend the in-person career fair based "
            "on employer list."
        )
        # Precondition asserted directly on the deployed shape predicate —
        # this is what makes the query a self-report in the first place.
        from utils.query_checker import is_self_report
        assert is_self_report(query) is True

        clf = IntentClassifier()
        regex_result = clf.classify(query)
        weak = clf._build_result(regex_result.intent, regex_result.confidence)
        with patch.object(ic, "_learn_intent_exemplar") as teach:
            refined = clf.refine_with_stm(weak, stm_intent, query=query)
        # The "confirm" cue lives only in the STM paraphrase; the query
        # itself is a first-person report, never a recall request.
        assert refined.intent != IntentType.FACTUAL_RECALL
        assert refined.intent != IntentType.TEMPORAL_RECALL
        # No other STM keyword family matches this paraphrase either, so
        # refinement falls through entirely and the original (regex)
        # result comes back unchanged.
        if regex_result.intent == IntentType.GENERAL:
            assert refined.source != "stm_refined"
        teach.assert_not_called()

    def test_paraphrase_cue_present_in_query_still_teaches(self):
        # Regex-unconfident by construction (no bare "recall"-pattern hit),
        # but the query DOES carry the "recall" cue itself, so the fix's
        # teaching gate (cue must be in the user's own words) is satisfied.
        query = "hmm, help me recall the cat name real quick"
        clf = IntentClassifier()
        regex_result = clf.classify(query)
        assert regex_result.confidence < 0.50, (
            f"query must be regex-unconfident for this to test STM "
            f"refinement; got {regex_result.confidence}"
        )
        weak = clf._build_result(regex_result.intent, regex_result.confidence)
        refined = clf.refine_with_stm(weak, "Recall the cat's name", query=query)
        assert refined.intent == IntentType.FACTUAL_RECALL
        assert refined.source == "stm_refined"
        assert any(
            "recall the cat" in t
            for t in get_store().get_learned("intent", refined.intent.value)
        )

    def test_paraphrase_only_cue_routes_but_never_teaches(self):
        # "Confirm the earlier decision" (STM paraphrase) hits the same
        # FACTUAL_RECALL "confirm" keyword as the live incident, but this
        # query is a genuine question (not a self-report), so the routing
        # skip added in this batch does NOT apply here — only the
        # teaching-gate cue check should suppress learning, isolating that
        # gate from the self-report skip tested above.
        query = (
            "does that track with what happened before, or am I "
            "misremembering things"
        )
        from utils.query_checker import is_self_report
        assert is_self_report(query) is False
        clf = IntentClassifier()
        regex_result = clf.classify(query)
        weak = clf._build_result(regex_result.intent, regex_result.confidence)
        with patch.object(ic, "_learn_intent_exemplar") as teach:
            refined = clf.refine_with_stm(
                weak, "Confirm the earlier decision", query=query
            )
        # Routes (this turn still gets factual_recall@0.60)...
        assert refined.intent == IntentType.FACTUAL_RECALL
        assert refined.source == "stm_refined"
        # ...but "confirm" never appears in the user's own words, so no
        # exemplar is learned under that label.
        teach.assert_not_called()
        assert get_store().get_learned("intent", "factual_recall") == []
