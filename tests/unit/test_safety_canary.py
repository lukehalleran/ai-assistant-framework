"""
tests/unit/test_safety_canary.py

Tests the log-only runtime canary that flags a sustained negative-affect streak
being classified CONVERSATIONAL (the signature of a tone-flatline miswire).

CGR-20260913-004 (BC-01, review #14): `_is_conversational` used to test
`"conversational" in str(tone).lower()` — raw substring membership, the same
mechanism that let `"crisis" in str(CrisisLevel.CONVERSATIONAL)` float every
session (BC-01's first incident). The parametrized tests below drive the
deployed `SafetyCanary.observe` (not just the helper) across every real tone
encoding — every `CrisisLevel` and `ToneLevel` member, plus plain strings —
and assert on streak/fire OUTCOME: only the conversational level may advance
or fire the streak, and a string that merely CONTAINS the word
("not conversational", "conversational_extra") must not count and must reset
an in-progress streak, since substring containment would wrongly admit it.
"""

import logging

import pytest

from core.safety_canary import SafetyCanary
from core.context_pipeline import ToneLevel
from utils.tone_detector import CrisisLevel

_NEG = "i feel completely hopeless and worthless and utterly alone, no one takes me seriously"
_POS = "worked on the sim project today and it went really well, solid progress"

# Every real tone encoding the deployed system can hand to `observe()`:
# core/orchestrator.py:1599 passes `context.tone_level`, always a `ToneLevel`
# member (core/context_pipeline.py:551, :1172 — the only two `ContextResult`
# producers); `_is_conversational` is also documented to accept `CrisisLevel`
# (utils/tone_detector.py:109) and plain strings, so both families are
# exercised here. Only the CONVERSATIONAL member of each family should ever
# advance or fire the streak.
_ALL_TONE_ENUM_MEMBERS = list(CrisisLevel) + list(ToneLevel)
_CONVERSATIONAL_MEMBERS = {CrisisLevel.CONVERSATIONAL, ToneLevel.CONVERSATIONAL}

# Plain strings that ARE the conversational level under the normalized
# `.strip().lower() == "conversational"` contract (whitespace/case variants).
_CONVERSATIONAL_STRINGS = ["conversational", "CONVERSATIONAL", " Conversational ", "Conversational"]

# Plain strings that are NOT the conversational level, including containment
# counterexamples that a raw `in` substring test would wrongly admit.
_NON_CONVERSATIONAL_STRINGS = [
    "CONCERN", "light_support", "HIGH", "crisis_support", "",
    "not conversational",       # containment: "conversational" ⊂ this string
    "conversational_extra",     # containment: this string ⊃ "conversational"
    "inconversational",         # containment: word runs into a longer word
    "the tone was conversational once",  # containment inside a sentence
]


def test_fires_after_threshold_consecutive_negative_conversational():
    c = SafetyCanary(threshold=4)
    events = [c.observe(_NEG, "conversational") for _ in range(4)]
    assert events[0] is None and events[1] is None and events[2] is None
    assert events[3] is not None
    ev = events[3]
    assert ev["event"] == "SAFETY_CANARY_TONE_FLATLINE"
    assert ev["consecutive"] == 4
    assert ev["turns"] == [1, 2, 3, 4]


def test_non_negative_message_resets_streak():
    c = SafetyCanary(threshold=4)
    c.observe(_NEG, "conversational")
    c.observe(_NEG, "conversational")
    assert c.observe(_POS, "conversational") is None  # positive breaks it
    c.observe(_NEG, "conversational")
    c.observe(_NEG, "conversational")
    # Only 2 in the new streak → still no fire at threshold 4.
    assert c.observe(_NEG, "conversational") is None


def test_non_conversational_tone_resets_streak():
    c = SafetyCanary(threshold=4)
    for _ in range(3):
        c.observe(_NEG, "conversational")
    # A correctly-classified distress turn breaks the flatline streak.
    assert c.observe(_NEG, "CONCERN") is None
    assert c.observe(_NEG, "conversational") is None  # streak restarted


def test_tone_encoding_agnostic():
    # Works for CrisisLevel value, ToneLevel name, or plain string.
    c = SafetyCanary(threshold=2)
    assert c.observe(_NEG, CrisisLevel.CONVERSATIONAL) is None
    assert c.observe(_NEG, ToneLevel.CONVERSATIONAL) is not None


def test_disabled_is_noop():
    c = SafetyCanary(threshold=1, enabled=False)
    assert c.observe(_NEG, "conversational") is None


def test_emits_warning_log(caplog):
    c = SafetyCanary(threshold=2, session_id="sess-xyz")
    with caplog.at_level(logging.WARNING):
        c.observe(_NEG, "conversational")
        c.observe(_NEG, "conversational")
    assert any("SAFETY_CANARY_TONE_FLATLINE" in r.message and "sess-xyz" in r.message
               for r in caplog.records)


def test_positive_affect_never_fires():
    c = SafetyCanary(threshold=2)
    assert c.observe(_POS, "conversational") is None
    assert c.observe(_POS, "conversational") is None


# ---------------------------------------------------------------------------
# CGR-20260913-004 (BC-01, review #14) — deployed-function regression tests.
# Proof is streak/fire OUTCOME through `observe()`, not the helper directly.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tone", _ALL_TONE_ENUM_MEMBERS, ids=lambda t: str(t))
def test_observe_streak_advances_only_for_conversational_enum_member(tone):
    """Every CrisisLevel and ToneLevel member, driven through observe(): the
    streak advances and fires only for the CONVERSATIONAL member of each
    family; every other member (CONCERN/MEDIUM/HIGH, ELEVATED/CRISIS, ...)
    must reset/refuse the streak on a negative-affect turn."""
    c = SafetyCanary(threshold=2)
    first = c.observe(_NEG, tone)
    second = c.observe(_NEG, tone)
    if tone in _CONVERSATIONAL_MEMBERS:
        assert first is None
        assert second is not None
        assert second["consecutive"] == 2
    else:
        assert first is None
        assert second is None


@pytest.mark.parametrize("tone", _CONVERSATIONAL_STRINGS)
def test_observe_streak_advances_for_conversational_plain_strings(tone):
    """Plain-string encodings of the conversational level (case/whitespace
    variants) still advance and fire the streak — equality is normalized,
    not case- or whitespace-sensitive."""
    c = SafetyCanary(threshold=2)
    assert c.observe(_NEG, tone) is None
    assert c.observe(_NEG, tone) is not None


@pytest.mark.parametrize("tone", _NON_CONVERSATIONAL_STRINGS)
def test_observe_never_advances_for_non_conversational_or_containment_strings(tone):
    """The core BC-01 regression: a raw `"conversational" in str(tone).lower()`
    substring test wrongly admits "not conversational", "conversational_extra",
    "inconversational" and a sentence that merely mentions the word. None of
    these may ever advance or fire the streak — repeated turns stay refused."""
    c = SafetyCanary(threshold=2)
    assert c.observe(_NEG, tone) is None
    assert c.observe(_NEG, tone) is None
    assert c.observe(_NEG, tone) is None


def test_containment_counterexample_resets_an_in_progress_streak():
    """A containment string arriving mid-streak must reset it exactly like a
    genuinely non-conversational tone does (test_non_conversational_tone_resets_streak),
    not be silently treated as equivalent to real CONVERSATIONAL turns."""
    c = SafetyCanary(threshold=4)
    assert c.observe(_NEG, "conversational") is None
    assert c.observe(_NEG, "conversational") is None
    assert c.observe(_NEG, "conversational") is None
    # Containment counterexample — must NOT count as the 4th conversational
    # turn and must NOT preserve the streak for what follows.
    assert c.observe(_NEG, "conversational_extra") is None
    # Streak restarted at 1, not continuing from 3 — three more genuine
    # conversational turns are needed to reach threshold=4 again.
    assert c.observe(_NEG, "conversational") is None
    assert c.observe(_NEG, "conversational") is None
    assert c.observe(_NEG, "conversational") is None
    fired = c.observe(_NEG, "conversational")
    assert fired is not None
    assert fired["consecutive"] == 4


def test_observe_never_raises_on_unusual_tone_values():
    """observe() must never raise — proven directly against the deployed
    function for values outside the documented encodings (None, int, a
    bare object with no __str__ override)."""
    c = SafetyCanary(threshold=2)
    for tone in (None, 42, object()):
        assert c.observe(_NEG, tone) is None
