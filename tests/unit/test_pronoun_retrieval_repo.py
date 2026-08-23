"""
"Check it out" after a repo/docs turn must route to file tools (2026-08-22).

Live miss: "Alright, check it out now" — after "Pushed yesterdays works docs
are updated too" and the model's own "being able to actually look at the repo
would make these check-ins more useful" — fell through every net:
  * light path swallowed it as a casual ack (ack opener hid the imperative),
  * FILE_RETRIEVAL_PRONOUN_PATTERN knew pull/print/fetch but not check/review,
  * FILE_DOC_CONTEXT_WORDS had no repo/commit/docs vocabulary ('doc ' is not
    a substring of "docs are").
All three fixed; first-person self-reports ("I checked it out, kind of
sucked") stay conversational.
"""

import pytest

from core.agentic.gate import (
    FILE_DOC_CONTEXT_WORDS,
    FILE_RETRIEVAL_PRONOUN_PATTERN,
    evaluate_agentic_gate,
)
from utils.query_checker import is_casual_acknowledgment


class FakeCorpus:
    def __init__(self, turns):
        self._turns = turns

    def get_recent_memories(self, n):
        return self._turns[:n]


REPO_TURNS = [
    {
        "query": "I would like to show you your progress, so I will fix that",
        "response": "That'd be great — being able to actually look at the repo "
                    "would make these check-ins a lot more useful.",
        "response_mode": "enhanced",
    },
    {
        "query": "Pushed yesterdays works docs are updated too, check it out",
        "response": "Nice — pushed and docs updated. I can't actually pull up "
                    "the repo from here.",
        "response_mode": "enhanced",
    },
]

GYM_TURNS = [
    {
        "query": "I checked it out, kind of sucked. Gym is a 50 minute walk",
        "response": "Damn, that sucks — 50 minutes each way is a real barrier.",
        "response_mode": "enhanced",
    },
]


class TestPattern:
    def test_check_and_review_verbs_match(self):
        for q in ("alright, check it out now", "review it when you can",
                  "take a look at it", "read it and tell me what you think"):
            assert FILE_RETRIEVAL_PRONOUN_PATTERN.search(q), q

    def test_repo_vocab_in_context_words(self):
        blob = "pushed yesterdays works docs are updated too, check it out"
        assert any(w in blob for w in FILE_DOC_CONTEXT_WORDS)


class TestGateRouting:
    @pytest.mark.asyncio
    async def test_live_miss_now_routes_to_tools(self):
        d = await evaluate_agentic_gate(
            user_text="Alright, check it out now",
            corpus_manager=FakeCorpus(REPO_TURNS),
        )
        assert d.should_trigger is True
        assert "tools" in d.modes

    @pytest.mark.asyncio
    async def test_self_report_stays_conversational(self):
        # the user NARRATING that they checked something must not trigger,
        # even with repo words nearby
        d = await evaluate_agentic_gate(
            user_text="I checked it out, kind of sucked. The repo docs I mean",
            corpus_manager=FakeCorpus(REPO_TURNS),
        )
        assert "tools" not in d.modes or d.should_trigger is False or \
            "file" not in (d.reason or "").lower()

    @pytest.mark.asyncio
    async def test_no_repo_context_no_routing(self):
        # "check it out" after gym talk: no file/repo context → no tool route
        d = await evaluate_agentic_gate(
            user_text="alright check it out now",
            corpus_manager=FakeCorpus(GYM_TURNS),
        )
        assert d.should_trigger is False


class TestLightPath:
    def test_ack_prefixed_imperative_not_casual(self):
        assert not is_casual_acknowledgment("Alright, check it out now")
        assert not is_casual_acknowledgment("ok run the numbers")

    def test_genuine_acks_still_casual(self):
        assert is_casual_acknowledgment("alright cool")
        assert is_casual_acknowledgment("ok thanks")
        assert is_casual_acknowledgment("hmm not working yet")


AGENTIC_PREV_TURNS = [
    {
        "query": "Ok third time's the charm you should be able to check it out now",
        "response": "Third time's the charm — it worked. Git access is live and I "
                    "can actually see the repo now. Want me to dig into anything "
                    "specific? The retrieval budget changes or the learning-loop "
                    "guards are the ones I'd be curious to read first.",
        "response_mode": "agentic-search",
    },
]


class TestAffirmativeDirectiveContinuation:
    """2026-08-22 turn 2: "Sure check out the learning loop stuff" (7 words —
    one over CONTINUATION_MAX_WORDS, noun phrase instead of pronoun) ran
    enhanced instead of continuing the agentic session. An affirmation-opener
    + request-shaped message after a stored-agentic turn is a continuation."""

    @pytest.mark.asyncio
    async def test_affirmative_directive_continues_agentic(self):
        d = await evaluate_agentic_gate(
            user_text="Sure check out the learning loop stuff",
            corpus_manager=FakeCorpus(AGENTIC_PREV_TURNS),
        )
        assert d.should_trigger is True
        assert "tools" in d.modes

    @pytest.mark.asyncio
    async def test_benzo_guard_holds(self):
        # long ack-opener STATEMENT after an agentic turn — never a continuation
        d = await evaluate_agentic_gate(
            user_text="Yeah they seem like the worst drug to get addicted to",
            corpus_manager=FakeCorpus(AGENTIC_PREV_TURNS),
        )
        assert d.should_trigger is False

    @pytest.mark.asyncio
    async def test_directive_without_agentic_prior_does_not_continue(self):
        d = await evaluate_agentic_gate(
            user_text="Sure check out the learning loop stuff",
            corpus_manager=FakeCorpus(GYM_TURNS),  # prior turn: enhanced gym talk
        )
        # no agentic prior, no file/repo context → no continuation route
        assert "tools" not in d.modes or d.should_trigger is False
