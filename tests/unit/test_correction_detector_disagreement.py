"""Correction detector: disagreement with the prior assistant claim (lane C1, wave 2).

Incident 2026-10-10..26 (FOLLOWUPS): "I don't believe that is correct" and "Don't
think it is standing" scored 0, so the thread-dispute hook (gate 0.6) fired only
on "Actually, ...". Class: BC-51 (signal never reaches its consumer).

Drives THE deployed ``CorrectionDetector.detect_correction_signal``.
"""
import pytest

from core.correction_detector import _MIN_CONFIDENCE, CorrectionDetector

DET = CorrectionDetector()


def _wrap(text):
    return text.replace(" ", "\n    ", 2)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("msg", [
    "I don't believe that is correct.",
    "Don't think it is standing.",
    "That isn't right.",
    "I do not think this is what Tamsin said.",
])
def test_disagreement_with_prior_claim_reaches_the_threshold(msg, wrapped):
    assert DET.detect_correction_signal(_wrap(msg) if wrapped else msg) >= _MIN_CONFIDENCE


@pytest.mark.parametrize("msg", [
    "I don't think I'll go.",
    "I don't think it's going to rain.",
    "I do think that is correct.",
    "It isn't funny.",
    "I don't think so.",
    "how are you today",
])
def test_negated_belief_about_something_else_does_not_fire(msg):
    assert DET.detect_correction_signal(msg) < _MIN_CONFIDENCE


def test_existing_regex_confidences_unchanged():
    assert DET.detect_correction_signal("actually it's 6 weeks") == 0.80
    assert DET.detect_correction_signal("no, that's wrong") == 0.85


def test_thread_dispute_gate_sees_the_new_signal():
    import core.orchestrator as orch
    assert DET.detect_correction_signal("Don't think it is standing.") >= orch._CORRECTION_MIN_CONFIDENCE
