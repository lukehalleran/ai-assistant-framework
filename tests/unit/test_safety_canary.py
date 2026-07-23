"""
tests/unit/test_safety_canary.py

Tests the log-only runtime canary that flags a sustained negative-affect streak
being classified CONVERSATIONAL (the signature of a tone-flatline miswire).
"""

import logging

from core.safety_canary import SafetyCanary

_NEG = "i feel completely hopeless and worthless and utterly alone, no one takes me seriously"
_POS = "worked on the sim project today and it went really well, solid progress"


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
    from utils.tone_detector import CrisisLevel
    from core.context_pipeline import ToneLevel
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
