"""2026-09-21 live turn: two defects at the END of one reply.

A casual check-in turn (no action requested anywhere in it) ended:

    "...if you want the rest of the PR rundown later while you're not doing
    anything, it's still queued up.<|close|>think<|sep|><|open|>response<|sep|>
    <|close|>message

    > Heads up — there's no card to approve: nothing was actually queued ..."

1. The chat-template tail was displayed and stored: the edge stripper removed
   only runs of PURE tokens, and this run has a bare channel word after each
   token and ends in one.
2. "it's still queued up" (said of a PR rundown) drew the no-card notice:
   "queued" was listed as an unambiguous card anchor, and it is also the
   grammar's own STATE word, so the match anchored itself; the backstop had no
   "was Daemon expected to act this turn" condition.

Every assertion drives the deployed function with the live text, in its clean
and its client-wrapped form.
"""
from types import SimpleNamespace

import pytest

import gui.handlers as h
from core import action_claim_guard as guard
from core.response_parser import ResponseParser

TAIL = "<|close|>think<|sep|><|open|>response<|sep|><|close|>message"
LAST_PARAGRAPH = (
    "Enjoy the hot shower — sometimes that warm water on the shoulder is half the therapy anyway. "
    "And if you want the rest of the PR rundown later while you're not doing anything, "
    "it's still queued up."
)
LIVE_REPLY = (
    "A semi-off day makes sense after yesterday's output — and the two task items are genuinely "
    "small, which is the right kind of list for a light day. Doing those from your phone counts.\n\n"
    "On the arm — if it's tendon-y and it's been bugging you since yesterday, skipping the gym is "
    "the right call, not a wimp-out.\n\n" + LAST_PARAGRAPH
)
WRAPPED_REPLY = LIVE_REPLY.replace("while you're not doing anything, ", "while you're not doing\n  anything, ")
LIVE_USER = (
    "Not sure yet. I kinda want today to be semi \"off\". I need to upload the resume though we "
    "finished it but never uploaded to website is a 5 min task. I guess I need to check out emails "
    "too and confirm I have access to sim stuff like I think I do. No gym. Went last 2 days and I "
    "think I fucked up a tendon in left arm yesterday made even getting to sleep a bit annoying. "
    "Got back from store, had supplements caffeine and such and now in the bathroom with shower running"
)


# ---------------------------------------------------------------------------
# 1. chat-template tail
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reply", [LIVE_REPLY, WRAPPED_REPLY])
def test_live_tail_is_stripped_for_display_and_storage(reply):
    dirty = reply + TAIL
    assert ResponseParser.strip_stream_special_tokens(dirty) == reply
    stored = ResponseParser.sanitize_for_storage(dirty)
    assert "<|" not in stored and stored.rstrip().endswith("it's still queued up.")


@pytest.mark.parametrize("dirty, clean", [
    ("Answer.<|sep|>", "Answer."),                                   # the 2026-08-21 shape still works
    ("<|sep|>That's fine.", "That's fine."),
    ("<|open|>response<|sep|>Hey there.", "Hey there."),              # leading channel run
    ("Answer.\n<|close|>message\n", "Answer."),
    ("Answer. <|close|>think <|sep|>", "Answer."),
])
def test_channel_runs_at_either_edge(dirty, clean):
    assert ResponseParser.strip_stream_special_tokens(dirty) == clean


def test_a_leading_token_never_eats_the_reply_s_first_word():
    assert ResponseParser.strip_stream_special_tokens("<|sep|>that's fine") == "that's fine"
    assert ResponseParser.strip_stream_special_tokens("<|sep|>message received, on it") == "message received, on it"


def test_a_token_quoted_mid_text_is_left_alone():
    text = "kimi sometimes emits <|sep|> as its first chunk, which we strip."
    assert ResponseParser.strip_stream_special_tokens(text) == text


# ---------------------------------------------------------------------------
# 2. "queued up" is ordinary English
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reply", [LIVE_REPLY, WRAPPED_REPLY, LAST_PARAGRAPH])
def test_the_live_reply_is_only_a_soft_claim(reply):
    assert guard.card_claim_needs_action_context(reply) is True
    assert guard._card_claim_regions(reply, hard_only=True) == []


@pytest.mark.parametrize("reply", [
    "That song is queued up next.",
    "Your order is still pending with the pharmacy, it's queued behind two others.",
])
def test_other_ordinary_queue_wording_is_soft_or_nothing(reply):
    assert not guard._card_claim_regions(reply, hard_only=True)
    assert (not guard.claims_pending_card(reply)) or guard.card_claim_needs_action_context(reply)


@pytest.mark.parametrize("reply", [
    "I've queued it up — approve the card below.",
    "You should see the approval card pop up in a second.",
    "Queued the deletion. Confirm and it's off your calendar.",
    "Hit approve and it should land this time.",
])
def test_a_reply_naming_the_card_or_the_approval_is_a_hard_claim(reply):
    assert guard.claims_pending_card(reply) is True
    assert guard.card_claim_needs_action_context(reply) is False


@pytest.mark.parametrize("reply", [LIVE_REPLY, WRAPPED_REPLY])
def test_read_time_marker_is_not_stamped_on_the_live_reply(reply):
    assert "[unverified action claim]" not in guard.annotate_unverified_action_claim(reply)


def test_read_time_marker_still_fires_on_a_hard_claim():
    out = guard.annotate_unverified_action_claim("I've queued it up — approve the card below.")
    assert out.rstrip().endswith("[unverified action claim]")


# ---------------------------------------------------------------------------
# the deployed backstop
# ---------------------------------------------------------------------------

class _FakeMM:
    async def generate_once(self, *a, **k):
        return "note"

    def get_active_model_name(self):
        return "test-model"


class _FakeCorpus:
    def __init__(self, recent=()):
        self._recent = list(recent)

    def get_recent_memories(self, n):
        return self._recent[-n:]


class _FakeOrch:
    def __init__(self, recent=()):
        self.model_manager = _FakeMM()
        self.memory_system = SimpleNamespace(
            chroma_store=None, corpus_manager=_FakeCorpus(recent), session_id="test-session")


async def _suffix(user_text, reply, recent=()):
    ctx = SimpleNamespace(orchestrator=_FakeOrch(recent), user_text=user_text, handled=False)
    return await h._apply_action_guard(
        ctx, reply, executed_kinds=set(), proposed_kinds=set(), self_repair=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [LIVE_REPLY, WRAPPED_REPLY])
async def test_live_turn_gets_no_notice(reply):
    assert guard.NO_CARD_NOTICE.strip() not in await _suffix(LIVE_USER, reply)


@pytest.mark.asyncio
@pytest.mark.parametrize("user_text", [
    "add a dentist appointment to my calendar tomorrow at 3pm",
    "try again",
])
async def test_the_same_wording_on_an_action_turn_still_gets_the_notice(user_text):
    assert h._turn_expected_an_action(SimpleNamespace(user_text=user_text, orchestrator=_FakeOrch()))
    suffix = await _suffix(user_text, "Done — it's queued up and waiting on your end.")
    assert guard.NO_CARD_NOTICE.strip() in suffix


@pytest.mark.asyncio
async def test_a_hard_claim_needs_no_action_context():
    suffix = await _suffix(LIVE_USER, "Approve the card below and it's on your calendar.")
    assert guard.NO_CARD_NOTICE.strip() in suffix


def test_no_action_context_on_the_live_turn():
    ctx = SimpleNamespace(user_text=LIVE_USER, orchestrator=_FakeOrch())
    assert h._turn_expected_an_action(ctx) is False


def test_a_broken_context_check_errs_toward_the_notice(monkeypatch):
    def boom(_text):
        raise RuntimeError("synthetic")
    monkeypatch.setattr(h.registry, "detect_action_intent", boom)
    assert h._turn_expected_an_action(SimpleNamespace(user_text="x", orchestrator=_FakeOrch())) is True
