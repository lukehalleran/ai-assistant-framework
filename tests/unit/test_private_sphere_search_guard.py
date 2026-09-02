"""Private-sphere generic search-term guard (2026-09-01).

Live misfire (16:52 turn): asked whether his first assignment "is just a quiz
on lectures or is there deliverable", the trigger LLM said search at conf 0.8
with terms ['Georgia Tech assignment due September 13 2026', ...] — 6 Tavily
credits returned GT football schedules while the answer sat in his own
syllabus/schedule docs. The guard suppresses searches whose EVERY term is a
personal-context anchor + private-sphere-generic nouns + time tokens — one
categorized vocabulary spanning school / work / appointments / hobby domains
(generality doctrine: extend the category lists, never clone the guard).
"""
import asyncio
import json

import pytest

from utils.web_search_trigger import terms_are_private_sphere_generic


GT = ["Georgia Tech"]

LIVE_TERMS = [
    "Georgia Tech assignment due September 13 2026",
    "Georgia Tech first assignment details September 2026",
    "Georgia Tech course quiz or deliverable September 2026",
]


class TestSchoolDomain:
    def test_live_terms_suppressed(self):
        assert terms_are_private_sphere_generic(LIVE_TERMS, GT)

    def test_registrar_public_query_rescued(self):
        # The institution feature's flagship query — "drop" is not
        # private-sphere vocabulary, registrar facts ARE on the web.
        assert not terms_are_private_sphere_generic(
            ["Georgia Tech drop date August 2026"], GT)

    def test_course_code_rescued(self):
        assert not terms_are_private_sphere_generic(
            ["Georgia Tech MGT 6203 first assignment"], GT)

    def test_public_event_word_rescued(self):
        # "game" is deliberately not in the vocabulary (public-event
        # polysemy) and the term has no private-sphere noun at all.
        assert not terms_are_private_sphere_generic(
            ["Georgia Tech game September 2026"], GT)


class TestWorkDomain:
    def test_generic_standup_suppressed(self):
        assert terms_are_private_sphere_generic(
            ["standup meeting schedule September 2026"], [])

    def test_shift_schedule_suppressed(self):
        assert terms_are_private_sphere_generic(
            ["work shift schedule next week"], [])

    def test_named_unknown_employer_rescued(self):
        # An org we have no anchor for: its name is a content word — the
        # search might legitimately find something. Under-fires by design.
        assert not terms_are_private_sphere_generic(
            ["Acme standup meeting time"], GT)


class TestOtherDomains:
    def test_hobby_practice_suppressed(self):
        assert terms_are_private_sphere_generic(
            ["practice session Tuesday next week"], [])

    def test_appointment_suppressed(self):
        assert terms_are_private_sphere_generic(
            ["appointment date first week September"], [])

    def test_topic_word_rescued(self):
        assert not terms_are_private_sphere_generic(
            ["lorvatin appointment questions"], [])


class TestGuardShape:
    def test_empty_terms_never_fire(self):
        assert not terms_are_private_sphere_generic([], GT)
        assert not terms_are_private_sphere_generic(None, GT)

    def test_requires_a_private_sphere_noun(self):
        # Pure anchor+time junk is the gate's temporal-generic guard's job.
        assert not terms_are_private_sphere_generic(
            ["Georgia Tech September 2026"], GT)

    def test_one_specific_term_rescues_the_batch(self):
        assert not terms_are_private_sphere_generic(
            LIVE_TERMS + ["MGT 6203 homework 1 R linear regression"], GT)

    def test_anchor_none_entries_tolerated(self):
        assert terms_are_private_sphere_generic(
            ["assignment due date next week"], [None])


class _StubMM:
    def __init__(self, raw):
        self.raw = raw

    async def generate_once(self, prompt, **k):
        return self.raw


LIVE_RAW = json.dumps({
    "should_search": True,
    "confidence": 0.8,
    "search_terms": LIVE_TERMS,
    "search_depth": "standard",
    "num_searches": 3,
    "reasoning": "assignment details may need verification",
})


class TestTriggerWiring:
    """Drives the deployed _classify_with_llm_unified with the live-turn
    LLM output — the suppression must flip should_search at the parse
    layer so BOTH consumers (agentic gate + builder web path) inherit."""

    def _classify(self, raw, institution="Georgia Tech"):
        import utils.institution_resolver as ir
        import utils.location_resolver as lr
        import utils.web_search_trigger as wst
        orig_inst, orig_loc = ir.get_user_institution, lr.get_user_location
        orig_anchors = ir.get_user_anchors
        ir.get_user_institution = lambda: institution
        # The guard call site reads get_user_anchors() (profile-backed) — patch
        # it too, or this test only passes on a machine whose LIVE profile
        # carries the institution (exactly the owner-data-dependence class).
        ir.get_user_anchors = lambda: [institution] if institution else []
        lr.get_user_location = lambda: None
        try:
            return asyncio.run(wst._classify_with_llm_unified(
                "Alright id verified. Okay. Yes first assignment is due the "
                "13th. Are you saying it's just a quiz on lectures or is "
                "there deliverable",
                _StubMM(raw),
            ))
        finally:
            ir.get_user_institution = orig_inst
            ir.get_user_anchors = orig_anchors
            lr.get_user_location = orig_loc

    def test_live_turn_suppressed_at_parse(self):
        parsed = self._classify(LIVE_RAW)
        assert parsed is not None
        assert parsed.should_search is False
        assert parsed.search_terms == []

    def test_specific_terms_untouched(self):
        raw = json.dumps({
            "should_search": True,
            "confidence": 0.8,
            "search_terms": ["Georgia Tech fall 2026 drop deadline"],
            "search_depth": "quick",
            "num_searches": 1,
            "reasoning": "registrar logistics",
        })
        parsed = self._classify(raw)
        assert parsed is not None
        assert parsed.should_search is True
        assert parsed.search_terms
