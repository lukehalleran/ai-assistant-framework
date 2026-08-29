"""Tests for memory/stance_classifier.py — the single-source stance core.

The casey|is|evil case is the SENTINEL for this whole layer: a one-mention
crisis-day value judgment must classify as an appraisal, never an objective
world-fact. If that test breaks, the epistemic layer's reason to exist broke.
"""

import pytest

from memory.stance_classifier import (
    LEGACY_STANCE_DEFAULT,
    StanceResult,
    capture_tone_from_level,
    classify_for_storage,
    classify_triple_stance,
    classify_utterance_stance,
    effective_stance,
    is_evaluative_text,
    scope_unresolved_referent,
)


class TestTripleStance:
    def test_sentinel_casey_is_evil_is_appraisal(self):
        result = classify_triple_stance("casey", "is", "evil")
        assert result.stance == "appraisal"
        assert result.is_appraisal

    def test_ordinary_fact_is_objective(self):
        result = classify_triple_stance("user", "lives_in", "chicago")
        assert result.stance == "objective"
        assert not result.is_appraisal

    def test_assistant_authored_is_inferred(self):
        result = classify_triple_stance(
            "casey", "trained", "user to doubt perceptions", author="assistant"
        )
        assert result.stance == "inferred"
        assert result.is_inferred

    def test_assistant_authored_beats_appraisal(self):
        # assistant-authored evaluative triple: inferred wins (rule order)
        result = classify_triple_stance("casey", "is", "evil", author="assistant")
        assert result.stance == "inferred"

    def test_user_self_appraisal(self):
        result = classify_triple_stance("user", "is", "a failure")
        assert result.stance == "appraisal"

    def test_user_noncopula_self_appraisal(self):
        # user subject + evaluative object triggers even off-copula
        result = classify_triple_stance("user", "feels_like", "a worthless failure")
        assert result.stance == "appraisal"

    def test_reporting_relation_is_reported(self):
        result = classify_triple_stance("casey", "said", "user is worthless")
        assert result.stance == "reported"

    def test_reporting_beats_appraisal(self):
        result = classify_triple_stance("casey", "told_me", "I am pathetic")
        assert result.stance == "reported"

    def test_nonuser_noncopula_evaluative_object_stays_objective(self):
        # conservative: third-party subject with non-copula relation does not
        # auto-appraise ("casey has_dog crazy" style noise)
        result = classify_triple_stance("casey", "owns", "a crazy dog")
        assert result.stance == "objective"

    def test_positive_thick_terms_are_appraisals_too(self):
        result = classify_triple_stance("sam", "is", "wonderful")
        assert result.stance == "appraisal"

    def test_diagnosis_shaped_relation_stays_objective(self):
        # diagnosed_with is not a copula and subject isn't user-evaluative path
        result = classify_triple_stance("casey", "diagnosed_with", "depression")
        assert result.stance == "objective"

    def test_reasons_populated(self):
        result = classify_triple_stance("casey", "is", "evil")
        assert result.reasons and any("copula" in r for r in result.reasons)


class TestEvaluativeText:
    def test_thick_terms_hit(self):
        assert is_evaluative_text("she was completely abusive")
        assert is_evaluative_text("Evil, plain and simple")
        assert is_evaluative_text("I'm such a piece of shit")

    def test_word_boundary_no_substring_hits(self):
        # 'evil' must not match inside 'evildoer-adjacent' ordinary words
        assert not is_evaluative_text("the devil is in the details")
        assert not is_evaluative_text("medieval history")

    def test_thin_terms_excluded(self):
        assert not is_evaluative_text("that was a bad day")
        assert not is_evaluative_text("the weather is cold and the test was hard")

    def test_empty(self):
        assert not is_evaluative_text("")
        assert not is_evaluative_text(None)


class TestUtteranceStance:
    def test_assistant_text_is_inferred(self):
        r = classify_utterance_stance(
            "She was systematically training you to doubt your own perceptions.",
            speaker="assistant",
        )
        assert r.stance == "inferred"

    def test_user_evaluative_is_appraisal(self):
        r = classify_utterance_stance("Casey was evil.", speaker="user")
        assert r.stance == "appraisal"

    def test_user_plain_is_objective(self):
        r = classify_utterance_stance("I moved to Chicago in March.", speaker="user")
        assert r.stance == "objective"


class TestReferentScoping:
    def test_pronoun_subject_scopes_to_user(self):
        scoped = scope_unresolved_referent("she", "abusive")
        assert scoped == "user's unnamed referent"

    def test_role_subject_scopes_with_qualifier(self):
        scoped = scope_unresolved_referent("my last partner", "abusive")
        assert scoped == "user's last partner"

    def test_role_subject_without_qualifier(self):
        scoped = scope_unresolved_referent("my ex", "toxic")
        assert scoped == "user's ex"

    def test_never_returns_named_entity(self):
        # named subject: no rescoping — and critically, never a *different*
        # named entity even if a resolver is supplied
        class FakeResolver:
            def resolve(self, name):
                return "casey"  # a fuzzy binder would return this

        assert scope_unresolved_referent("she", "abusive", entity_resolver=FakeResolver()) == "user's unnamed referent"
        assert scope_unresolved_referent("casey", "evil") is None

    def test_non_evaluative_object_no_scoping(self):
        assert scope_unresolved_referent("she", "a teacher in ohio") is None

    def test_user_subject_no_scoping(self):
        assert scope_unresolved_referent("user", "a failure") is None


class TestStorageAPI:
    def test_classify_for_storage_shape(self):
        d = classify_for_storage("casey", "is", "evil", tone_level="CrisisLevel.MEDIUM")
        assert d == {"stance": "appraisal", "capture_tone": "elevated"}

    def test_capture_tone_mapping(self):
        assert capture_tone_from_level("HIGH") == "elevated"
        assert capture_tone_from_level("concern") == "elevated"
        assert capture_tone_from_level("CONVERSATIONAL") == "non_elevated"
        assert capture_tone_from_level(None) == "unknown"
        assert capture_tone_from_level("weird_value") == "unknown"

    def test_effective_stance_legacy_default(self):
        assert effective_stance(None) == LEGACY_STANCE_DEFAULT == "unknown"
        assert effective_stance({}) == "unknown"
        assert effective_stance({"stance": "appraisal"}) == "appraisal"
        assert effective_stance({"stance": "bogus"}) == "unknown"


class TestStanceResultModel:
    def test_defaults(self):
        r = StanceResult()
        assert r.stance == "objective"
        assert r.reasons == []
