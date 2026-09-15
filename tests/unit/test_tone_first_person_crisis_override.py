"""tests/unit/test_tone_first_person_crisis_override.py

T01 (2026-09-13, E-R05-1): `detect_crisis_level`'s Stage 0
(`_check_observational_language`) previously short-circuited to
CONVERSATIONAL before Stage 1's keyword check ever ran, masking explicit
first-person crisis language mixed with news/observational framing —
"people are suffering and I want to die" resolved CONVERSATIONAL.

Fixed by `_has_first_person_high_crisis_hit` (utils/tone_detector.py,
beside `_check_observational_language`): Stage 0 no longer short-circuits
when the message carries a first-person HIGH-crisis keyword hit — either
the matched HIGH phrase itself is first-person (rule a: "kill myself",
"end my life"), or a first-person subject sits within
`FIRST_PERSON_CRISIS_WINDOW_TOKENS` (3) whitespace tokens before the hit,
in the same sentence (rule b). Detection then continues to Stage 1
unchanged; no new vocabulary is introduced (BC-76) — both rules reuse the
module's existing word-bounded `_HISTORY_FIRST_PERSON_RE`.

Scope: HIGH vocabulary only. MEDIUM/CONCERN first-person distress under
observational framing ("people are suffering and I feel hopeless") stays
CONVERSATIONAL — an open sibling recorded for the owner
(docs/execution/generalization/batches/T01.md), addressed by a separate
batch (docs/execution/generalization/T02_design.md), not fixed here.

BC-29 (self-teaching guard): an overridden turn's Stage 1 result does NOT
call `_learn_tone_exemplar` — a message that mixes news framing with
first-person crisis language is not a clean exemplar; teaching it could
make future news discussion score as crisis. A non-observational
first-person crisis message reaching Stage 1 without the override keeps
today's learning.

Deployed `detect_crisis_level(message, model_manager=None)` is exercised
throughout; two tests also call the new helper directly for the
window-boundary/sentence-scoping edge cases, where a precise token-count
assertion is clearer than reasoning back from the routed CrisisLevel.
Every user-text fixture appears in clean and wrapped/indented form (BC-64).
No test depends on the embedder — every row resolves at Stage 0 or Stage 1.
"""

import pytest

import utils.tone_detector as tone_detector
from utils.tone_detector import CrisisLevel, detect_crisis_level


def _wrap(text: str) -> str:
    """Line-wrap/indent a fixture the way a real chat client sends it
    (BC-64), without altering any word — the wrapped form must classify
    the same as the clean form."""
    words = text.split(" ")
    mid = len(words) // 2
    return " ".join(words[:mid]) + "\n  " + " ".join(words[mid:])


# ===========================================================================
# Escalate: a first-person HIGH-crisis hit overrides observational framing.
# Each of these resolves CONVERSATIONAL/"observational_language" on the
# UNPATCHED source (verified against the deployed, unedited
# _check_observational_language) — this is the failing-before proof.
# ===========================================================================

ESCALATE_CASES = [
    # design doc table rows — rule (b): a first-person subject within the
    # window, same sentence as the HIGH hit.
    ("rule_b_want_to_die", "people are suffering and I want to die"),
    ("rule_b_cant_go_on",
     "According to the news, officials say millions are struggling, and "
     "honestly I can't go on"),
    ("rule_b_dont_want_to_be_here",
     "People are dying in the news and I don't want to be here anymore"),
    ("rule_b_ready_to_die",
     "the news is awful, people are suffering, and I'm ready to die"),
    ("rule_b_want_to_die_article",
     "According to the article people are struggling and I really just "
     "want to die"),
    # rule (a) alone: the matched HIGH phrase itself is first-person
    # ("kill myself" contains "myself"). The tokens immediately before the
    # hit ("the", "urge", "to") carry no first-person subject, so rule (b)
    # does not independently apply here — this row isolates rule (a).
    ("rule_a_only_kill_myself",
     "According to officials, millions of people are struggling, and the "
     "urge to kill myself keeps returning"),
]

STAY_CASES = [
    # design doc "stay" list — no first-person HIGH hit at all.
    ("design_suicide_rates",
     "According to officials, suicide rates rose among young people"),
    ("design_suicidal_after_policies",
     "The news article says people are suicidal after the policies"),
    ("design_news_coverage_irresponsible",
     "I think the news coverage of suicide is irresponsible, people are "
     "struggling"),
    ("design_third_person_cant_go_on",
     "They say people are suffering and they can't go on"),
    ("design_millions_want_to_die_war",
     "According to the news, millions of people want to die in that war"),
    ("design_struggling_policies_officials",
     "People are struggling with the new policies, according to officials"),
    # sentence scoping: a first-person token in an earlier sentence must
    # not qualify a later hit. Without sentence scoping, the raw 3-token
    # window before "want to die" would reach back across the "?" into
    # "me" ("wait, is that me? civilians, society want to die" — the
    # naive last-3-tokens window is "me? civilians, society"); scoped to
    # the sentence starting after "?", the window is only "civilians,
    # society", which is not first-person.
    ("sentence_scoping_previous_sentence",
     "Wait, is that me? Civilians, society want to die."),
]

# Window boundary: a first-person subject exactly
# FIRST_PERSON_CRISIS_WINDOW_TOKENS (3) tokens before the hit qualifies; one
# more filler word (4 tokens before) does not. Kept out of the generic
# ESCALATE/STAY clean+wrap loops above: the generic `_wrap` splits a message
# at its WORD MIDPOINT, and for a message this short that midpoint falls
# inside the very 3-token window being probed — the design's own sentence
# rule (`_has_first_person_high_crisis_hit`'s docstring; same in
# `docs/execution/generalization/probes/t01_probe.py`) treats '\n' as a
# sentence boundary equally with '.', '!', '?', so a naive mid-window wrap
# would change what is being tested, not exercise line-wrap robustness.
# The wrapped form here instead wraps BEFORE the window under test, which is
# still a genuine BC-64 check (an early client line-wrap must not perturb a
# later token-count boundary) without that self-contradiction.
WINDOW_BOUNDARY_QUALIFIES = "According to the news, I honestly really want to die"
WINDOW_BOUNDARY_ONE_OVER = "According to the news, I honestly really quite want to die"


def _wrap_before_window(text: str) -> str:
    """Wrap after the first two words — well before the 3-token window
    under test in the two fixtures above — so the newline sentence-boundary
    rule cannot interact with the assertion (see comment above)."""
    words = text.split(" ")
    return " ".join(words[:2]) + "\n  " + " ".join(words[2:])


class TestWindowBoundaryEndToEnd:
    async def test_exact_three_tokens_escalates_clean(self):
        result = await detect_crisis_level(WINDOW_BOUNDARY_QUALIFIES, model_manager=None)
        assert result.level != CrisisLevel.CONVERSATIONAL
        assert result.trigger != "observational_language"

    async def test_exact_three_tokens_escalates_wrapped_before_window(self):
        result = await detect_crisis_level(
            _wrap_before_window(WINDOW_BOUNDARY_QUALIFIES), model_manager=None
        )
        assert result.level != CrisisLevel.CONVERSATIONAL
        assert result.trigger != "observational_language"

    async def test_four_tokens_stays_conversational_clean(self):
        result = await detect_crisis_level(WINDOW_BOUNDARY_ONE_OVER, model_manager=None)
        assert result.level == CrisisLevel.CONVERSATIONAL
        assert result.trigger == "observational_language"

    async def test_four_tokens_stays_conversational_wrapped_before_window(self):
        result = await detect_crisis_level(
            _wrap_before_window(WINDOW_BOUNDARY_ONE_OVER), model_manager=None
        )
        assert result.level == CrisisLevel.CONVERSATIONAL
        assert result.trigger == "observational_language"


@pytest.mark.parametrize("message", [m for _, m in ESCALATE_CASES],
                         ids=[i for i, _ in ESCALATE_CASES])
async def test_escalate_clean(message):
    result = await detect_crisis_level(message, model_manager=None)
    assert result.level != CrisisLevel.CONVERSATIONAL
    assert result.trigger != "observational_language"


@pytest.mark.parametrize("message", [m for _, m in ESCALATE_CASES],
                         ids=[i for i, _ in ESCALATE_CASES])
async def test_escalate_wrapped(message):
    result = await detect_crisis_level(_wrap(message), model_manager=None)
    assert result.level != CrisisLevel.CONVERSATIONAL
    assert result.trigger != "observational_language"


@pytest.mark.parametrize("message", [m for _, m in STAY_CASES],
                         ids=[i for i, _ in STAY_CASES])
async def test_stays_conversational_clean(message):
    result = await detect_crisis_level(message, model_manager=None)
    assert result.level == CrisisLevel.CONVERSATIONAL
    assert result.trigger == "observational_language"


@pytest.mark.parametrize("message", [m for _, m in STAY_CASES],
                         ids=[i for i, _ in STAY_CASES])
async def test_stays_conversational_wrapped(message):
    result = await detect_crisis_level(_wrap(message), model_manager=None)
    assert result.level == CrisisLevel.CONVERSATIONAL
    assert result.trigger == "observational_language"


# ===========================================================================
# Helper-direct: the window boundary and sentence scoping are precise token
# arithmetic, clearer asserted directly on `_has_first_person_high_crisis_hit`
# than reasoned back from the routed CrisisLevel.
# ===========================================================================

class TestHelperWindowAndSentenceBoundary:
    def test_window_boundary_exact_vs_one_over(self):
        from utils.tone_detector import (
            _has_first_person_high_crisis_hit,
            FIRST_PERSON_CRISIS_WINDOW_TOKENS,
        )
        assert FIRST_PERSON_CRISIS_WINDOW_TOKENS == 3
        assert _has_first_person_high_crisis_hit(WINDOW_BOUNDARY_QUALIFIES) is True
        assert _has_first_person_high_crisis_hit(WINDOW_BOUNDARY_ONE_OVER) is False

    def test_sentence_boundary_resets_the_window(self):
        from utils.tone_detector import _has_first_person_high_crisis_hit
        message = "Wait, is that me? Civilians, society want to die."
        assert _has_first_person_high_crisis_hit(message) is False


# ===========================================================================
# BC-29: no self-teaching from a mixed (override) message; a clean
# non-observational first-person crisis message keeps teaching.
# ===========================================================================

class TestLearningGuardBC29:
    async def test_override_turn_does_not_teach(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            tone_detector, "_learn_tone_exemplar",
            lambda *args, **kwargs: calls.append((args, kwargs)),
        )
        message = ESCALATE_CASES[0][1]
        result = await detect_crisis_level(message, model_manager=None)
        assert result.level != CrisisLevel.CONVERSATIONAL
        assert calls == []

    async def test_control_first_person_crisis_still_teaches(self, monkeypatch):
        calls = []

        def spy(message, level_key, source, model_manager=None):
            calls.append((level_key, source))

        monkeypatch.setattr(tone_detector, "_learn_tone_exemplar", spy)
        # Non-observational (no news/third-party framing at all) — this
        # message never engages Stage 0's override path; it is the control
        # showing the override guard does not over-suppress learning.
        control = "I want to die and I can't go on"
        assert tone_detector._check_observational_language(control) is False
        result = await detect_crisis_level(control, model_manager=None)
        assert result.level != CrisisLevel.CONVERSATIONAL
        assert calls, "expected _learn_tone_exemplar to be called for a clean first-person crisis message"
        assert calls[0][1] == "keyword"
