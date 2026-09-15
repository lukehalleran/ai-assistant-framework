"""tests/unit/test_tone_keyword_boundaries.py

CGR-20260913-003 (class-guard request) — anchors #29/#30 in
`utils/tone_detector.py`: `_check_observational_language`'s OBSERVATIONAL_MARKERS
count and `_calculate_harm_score`'s 'overwhelmed' special-case (with its
positive-marker exemption list) each used a raw `marker in message_lower`
substring membership test instead of the chokepoint (`utils.trigger_match`),
so a marker fires inside an unrelated longer word (BC-01) and 'overwhelmed'
fires without regard to negation (BC-02).

Fixed by routing both through `compile_keyword_matcher` (word-boundary for
bare single-word markers, substring unchanged for multi-word phrases).

Parent correction (2026-09-13): 'overwhelmed' is NOT negation-filtered. The
harm score scans all of its keyword vocabularies negation-blind, and the
chokepoint's NEGATION_CUE_RE is request-scoping vocabulary ("stop", "never",
"without", "skip") that inverts on affective statements — "I can't stop
feeling overwhelmed" is intensified distress, not a negation. 'overwhelmed'
follows the same policy as its CONCERN siblings (see CGR-20260913-003-2).

Deployed functions only: `_check_observational_language`,
`_calculate_harm_score`, `detect_crisis_level`. Each user-text fixture
appears in clean and wrapped/indented form (BC-64).
"""

from utils.tone_detector import (
    CrisisLevel,
    _check_observational_language,
    _calculate_harm_score,
    detect_crisis_level,
)


def _wrap(text: str) -> str:
    """Line-wrap/indent a fixture the way a real chat client sends it
    (BC-64), without altering any word — the wrapped form must classify
    the same as the clean form."""
    words = text.split(" ")
    mid = len(words) // 2
    return " ".join(words[:mid]) + "\n  " + " ".join(words[mid:])


# ===========================================================================
# Anchor #29 — OBSERVATIONAL_MARKERS containment (_check_observational_language)
# ===========================================================================

# 'news' contained in "newsletter" (the plan's own containment example) and
# 'article' contained in "particle" ("particle"[1:] == "article") each used
# to count toward the >=2 observational-marker threshold, pre-fix.
CONTAINMENT_MESSAGE = (
    "I read a newsletter about a physics particle experiment, but honestly "
    "I want to die and I can't go on anymore"
)

# Real, non-contained observational markers ("according to", "officials",
# "millions", "people are", "struggling with", "policies") — paired positive
# control: the count must stay >= 2 after the fix.
GENUINE_OBSERVATIONAL_MESSAGE = (
    "According to officials, millions of people are struggling with the new policies"
)


class TestObservationalMarkerContainment:
    def test_containment_does_not_count_as_observational_clean(self):
        assert _check_observational_language(CONTAINMENT_MESSAGE) is False

    def test_containment_does_not_count_as_observational_wrapped(self):
        assert _check_observational_language(_wrap(CONTAINMENT_MESSAGE)) is False

    def test_genuine_observational_control_preserved_clean(self):
        assert _check_observational_language(GENUINE_OBSERVATIONAL_MESSAGE) is True

    def test_genuine_observational_control_preserved_wrapped(self):
        assert _check_observational_language(_wrap(GENUINE_OBSERVATIONAL_MESSAGE)) is True

    # --- BC-28: the resulting tone LEVEL from detect_crisis_level, not just
    # the boolean/score. Pre-fix, the containment false positive short-
    # circuited Stage 0 to CONVERSATIONAL before the two HIGH keyword hits
    # ("want to die", "can't go on") were ever scanned, masking a real
    # crisis signal. Both messages resolve at the deterministic keyword
    # stage (Stage 0/1) — no model_manager/embedder is reached either way.

    async def test_containment_false_positive_reaches_real_crisis_level_clean(self):
        result = await detect_crisis_level(CONTAINMENT_MESSAGE, model_manager=None)
        assert result.level == CrisisLevel.HIGH

    async def test_containment_false_positive_reaches_real_crisis_level_wrapped(self):
        result = await detect_crisis_level(_wrap(CONTAINMENT_MESSAGE), model_manager=None)
        assert result.level == CrisisLevel.HIGH

    async def test_genuine_observational_control_stays_conversational(self):
        result = await detect_crisis_level(GENUINE_OBSERVATIONAL_MESSAGE, model_manager=None)
        assert result.level == CrisisLevel.CONVERSATIONAL
        assert result.trigger == "observational_language"


# ===========================================================================
# Anchor #30 — 'overwhelmed' + positive-marker exemption (_calculate_harm_score)
# ===========================================================================

class TestOverwhelmedBoundaryAndNegation:
    # 'happy' contained in "unhappy" no longer exempts.
    def test_unhappy_no_longer_exempts_overwhelmed_clean(self):
        score, matched, cats = _calculate_harm_score("unhappy and overwhelmed")
        assert score == 2.0
        assert any("overwhelmed" in m for m in matched)
        assert cats["concern"] == 1

    def test_unhappy_no_longer_exempts_overwhelmed_wrapped(self):
        score, matched, _ = _calculate_harm_score(_wrap("unhappy and overwhelmed"))
        assert score == 2.0

    # Paired positive control: a genuine positive marker present as a whole
    # word still exempts.
    def test_genuine_positive_marker_control_still_exempts_clean(self):
        score, matched, _ = _calculate_harm_score("overwhelmed by all the birthday gifts")
        assert score == 0.0
        assert matched == []

    def test_genuine_positive_marker_control_still_exempts_wrapped(self):
        score, _, _ = _calculate_harm_score(_wrap("overwhelmed by all the birthday gifts"))
        assert score == 0.0

    # Paired positive control: a genuinely overwhelmed message with no
    # positive markers keeps its current score.
    def test_genuinely_overwhelmed_control_score_unchanged_clean(self):
        score, matched, cats = _calculate_harm_score("I'm so overwhelmed right now")
        assert score == 2.0
        assert cats["concern"] == 1

    def test_genuinely_overwhelmed_control_score_unchanged_wrapped(self):
        score, _, _ = _calculate_harm_score(_wrap("I'm so overwhelmed right now"))
        assert score == 2.0

    # Parent correction: a negation cue before 'overwhelmed' that scopes
    # something else, or intensifies the distress, must not drop the +2.
    # (Under the chokepoint's is_negated() these all scored 0.0.)
    def test_intensified_distress_still_adds_concern_clean(self):
        for msg in INTENSIFIED_DISTRESS:
            score, matched, cats = _calculate_harm_score(msg)
            assert score == 2.0, f"{msg!r}: {matched}"
            assert cats["concern"] == 1

    def test_intensified_distress_still_adds_concern_wrapped(self):
        for msg in INTENSIFIED_DISTRESS:
            score, matched, _ = _calculate_harm_score(_wrap(msg))
            assert score == 2.0, f"{_wrap(msg)!r}: {matched}"

    # Negation policy parity: 'overwhelmed' scores exactly like its CONCERN
    # sibling 'alone' in the same negated frame, whatever the harm score's
    # (negation-blind) policy is — no one-off rule for this one keyword.
    def test_negation_policy_matches_concern_sibling(self):
        for frame in NEGATED_FRAMES:
            for shape in (lambda s: s, _wrap):
                overwhelmed = _calculate_harm_score(shape(frame.format("overwhelmed")))[0]
                alone = _calculate_harm_score(shape(frame.format("alone")))[0]
                assert overwhelmed == alone, frame

    # BC-28: the exemption fix changes the resulting LEVEL — "unhappy" no
    # longer routes this message differently from the same message without
    # it. Pre-fix, 'happy' ⊂ "unhappy" dropped the +2 and the message took
    # the semantic path (MEDIUM) while its sibling took the keyword route
    # (CONCERN). Whether a score-4 keyword CONCERN should pre-empt a stronger
    # semantic reading is a pre-existing Stage 1 question escalated in
    # R05.md; it is deliberately not asserted as desirable here.
    async def test_exemption_fix_routes_like_the_no_unhappy_sibling(self):
        sibling = await detect_crisis_level(
            "I feel so alone and completely overwhelmed", model_manager=None
        )
        for msg in (UNHAPPY_OVERWHELMED_MESSAGE, _wrap(UNHAPPY_OVERWHELMED_MESSAGE)):
            result = await detect_crisis_level(msg, model_manager=None)
            assert (result.level, result.trigger) == (sibling.level, sibling.trigger)


INTENSIFIED_DISTRESS = [
    "I can't stop feeling overwhelmed",
    "I have never been this overwhelmed",
    "without any help I'm overwhelmed",
]

NEGATED_FRAMES = [
    "I won't be {} today",
    "I don't feel {} anymore",
    "I'm not {}",
    "I'm no longer {}",
]

UNHAPPY_OVERWHELMED_MESSAGE = "I feel so alone and unhappy, completely overwhelmed"
