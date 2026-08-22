"""
Regression tests for the 2026-08-02 junk-fact batch.

Live junk stored from emotional turns: `user | likes | my fucking life`
(source: "Feel like I hate my fucking life"), `user | dad_show_up | for a
bit`, `user | dad_show_up | with food`, `user | goal | yesterday`,
`user | unhappiness | unhappy`, `user | eaten | not eaten yet`.

Guards under test (memory/fact_extractor.py):
- _is_junk_object: adverbial/temporal/negation fragment objects dropped
  (wired into _clean_triple)
- _polarity_conflict: positive-preference triples negated by their own
  source text dropped (wired into extract_facts)
- _canonicalize_preferences: the "i like " rewrite requires the object to
  actually follow the trigger (a bare substring hit rewrote ANY triple to
  likes|<object>)
"""

import pytest

from memory.fact_extractor import (
    _canonicalize_preferences,
    _clean_triple,
    _is_junk_object,
    _polarity_conflict,
)


class TestJunkObject:
    @pytest.mark.parametrize("obj", [
        "for a bit", "with food", "for a few hours", "for a while",
        "yesterday", "today", "tomorrow", "not good", "not eaten yet",
        "a bit", "a few days", "unhappy", "with the food", "in a while",
    ])
    def test_live_junk_objects_dropped(self, obj):
        assert _is_junk_object(obj, "dad_show_up")

    @pytest.mark.parametrize("obj", [
        "kavarin", "Vicodin", "opioid addiction", "high dose, 3 times a day",
        "guitar", "computer science", "St. Louis",
    ])
    def test_real_objects_kept(self, obj):
        assert not _is_junk_object(obj, "previous_addiction")

    def test_schedule_relations_allow_temporal_objects(self):
        assert not _is_junk_object("for a few hours", "work_schedule")
        assert not _is_junk_object("tomorrow", "exam_date")

    def test_communication_relations_allow_negation_objects(self):
        # 2026-08-05: the negation-fragment check was junk-killing the exact
        # care-team fact class the vocab addition targets — for these
        # relations the negation IS the content.
        assert not _is_junk_object("no patient portal", "doctor_communication")
        assert not _is_junk_object("not reachable by phone", "pharmacy_status")
        assert not _is_junk_object("no online access", "portal_access")

    def test_negation_still_junk_for_ordinary_relations(self):
        assert _is_junk_object("not good", "dad_show_up")
        assert _is_junk_object("no idea", "goal")

    def test_clean_triple_drops_junk_objects(self):
        assert _clean_triple("user", "dad_show_up", "for a bit") is None
        assert _clean_triple("user", "goal", "yesterday") is None
        assert _clean_triple("user", "previous_addiction", "kavarin") is not None


class TestLLMPathJunkObject:
    """The LLM extractor path had NO junk-object check until 2026-08-03 —
    junk like `dad_show_up=for a bit` entered the profile through
    _normalize_triple while the regex path blocked it."""

    def test_llm_normalize_drops_junk_objects(self):
        from memory.llm_fact_extractor import _normalize_triple
        assert _normalize_triple(
            {"subject": "user", "relation": "dad_show_up", "object": "for a bit"}
        ) is None
        assert _normalize_triple(
            {"subject": "user", "relation": "felt", "object": "not good"}
        ) is None

    def test_llm_normalize_keeps_real_facts(self):
        from memory.llm_fact_extractor import _normalize_triple
        norm = _normalize_triple(
            {"subject": "user", "relation": "takes_medication", "object": "zelphex"}
        )
        assert norm is not None and norm["object"] == "zelphex"

    def test_llm_normalize_schedule_exempt(self):
        from memory.llm_fact_extractor import _normalize_triple
        norm = _normalize_triple(
            {"subject": "user", "relation": "schedule", "object": "every monday at 9am"}
        )
        assert norm is not None


class TestPolarityGuard:
    def test_live_inversion_blocked(self):
        text = "Couldn't do it. Am struggling immensely. Feel like I hate my fucking life"
        assert _polarity_conflict(text, "likes", "my fucking life")

    @pytest.mark.parametrize("neg", [
        "I hate cardio", "I can't stand cardio", "I don't like cardio",
        "I'm sick of cardio", "I am so tired of cardio", "I despise cardio",
    ])
    def test_negative_governors_block_likes(self, neg):
        assert _polarity_conflict(neg, "likes", "cardio")

    def test_genuine_positive_kept(self):
        assert not _polarity_conflict("I really love the gym", "likes", "the gym")
        # Negativity about a DIFFERENT object must not block this one
        assert not _polarity_conflict(
            "I like the gym but I hate cardio", "likes", "the gym"
        )

    def test_non_preference_relations_unaffected(self):
        assert not _polarity_conflict("I hate my job at Acme", "works_at", "acme")


class TestLikeTriggerPrecision:
    def test_rewrite_requires_object_after_trigger(self):
        # "i like " present, but this triple's object never follows it —
        # the old substring check rewrote (user, hates, cardio) to likes|cardio.
        s, r, o = _canonicalize_preferences(
            "I like the gym but cardio is awful", "", "user", "hates", "cardio"
        )
        assert r != "likes"

    def test_rewrite_fires_when_object_follows_trigger(self):
        s, r, o = _canonicalize_preferences(
            "I really like the gym these days", "", "user", "goes_to", "the gym"
        )
        assert (s, r, o) == ("user", "likes", "the gym")

    def test_feel_like_does_not_trigger(self):
        s, r, o = _canonicalize_preferences(
            "Feel like I hate my fucking life", "", "user", "is", "my fucking life"
        )
        assert r != "likes"
