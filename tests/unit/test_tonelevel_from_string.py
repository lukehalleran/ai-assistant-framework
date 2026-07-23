"""
tests/unit/test_tonelevel_from_string.py

Regression for the CrisisLevel->ToneLevel dead-wire (2026-07-21).

The pipeline maps tone via `ToneLevel.from_string(crisis_level.value)`, where
CrisisLevel.value is "light_support"/"elevated_support"/"crisis_support"/
"conversational" — NOT the "HIGH"/"MEDIUM"/"CONCERN" name-scale. Before the fix,
from_string knew only the name-scale, so EVERY CrisisLevel defaulted to
CONVERSATIONAL and the EscalationTracker was fed CONVERSATIONAL every turn
(GROUNDING/QUIET could never fire in production). The prior test only exercised
the name-scale, so it missed this entirely.

These tests pin BOTH encodings so the dead-wire cannot silently return.
"""

from utils.tone_detector import CrisisLevel
from core.context_pipeline import ToneLevel


def test_crisislevel_value_maps_to_correct_tonelevel():
    # The exact call the pipeline makes: from_string(crisis_level.value)
    expected = {
        CrisisLevel.CONVERSATIONAL: ToneLevel.CONVERSATIONAL,
        CrisisLevel.CONCERN: ToneLevel.CONCERN,
        CrisisLevel.MEDIUM: ToneLevel.ELEVATED,
        CrisisLevel.HIGH: ToneLevel.CRISIS,
    }
    for cl, expected_tl in expected.items():
        assert ToneLevel.from_string(cl.value) == expected_tl, (
            f"{cl.name}.value={cl.value!r} mis-mapped to "
            f"{ToneLevel.from_string(cl.value).name} (expected {expected_tl.name})"
        )


def test_crisislevel_name_still_maps():
    # The name-scale encoding must keep working too.
    assert ToneLevel.from_string("HIGH") == ToneLevel.CRISIS
    assert ToneLevel.from_string("MEDIUM") == ToneLevel.ELEVATED
    assert ToneLevel.from_string("CONCERN") == ToneLevel.CONCERN
    assert ToneLevel.from_string("CONVERSATIONAL") == ToneLevel.CONVERSATIONAL


def test_case_insensitive_and_invalid():
    assert ToneLevel.from_string("crisis_support") == ToneLevel.CRISIS
    assert ToneLevel.from_string("CRISIS_SUPPORT") == ToneLevel.CRISIS
    assert ToneLevel.from_string("light_support") == ToneLevel.CONCERN
    # Genuinely invalid still defaults to CONVERSATIONAL.
    assert ToneLevel.from_string("nonsense") == ToneLevel.CONVERSATIONAL
    assert ToneLevel.from_string("") == ToneLevel.CONVERSATIONAL
    assert ToneLevel.from_string(None) == ToneLevel.CONVERSATIONAL


def test_no_crisislevel_falls_through_to_conversational():
    # Belt-and-suspenders: no NON-conversational CrisisLevel may map to
    # CONVERSATIONAL via either encoding (that was the whole bug).
    for cl in (CrisisLevel.CONCERN, CrisisLevel.MEDIUM, CrisisLevel.HIGH):
        assert ToneLevel.from_string(cl.value) != ToneLevel.CONVERSATIONAL
        assert ToneLevel.from_string(cl.name) != ToneLevel.CONVERSATIONAL
