"""Past-anchored first-person questions are temporal_recall.

Incident 2026-10-15 22:05: "What was I working on in late September, and when
is sim HW 4 due?" classified general@0.00, so temporal_recall weights and
max_wiki=0 never applied (8.4 s wiki task). class: BC-58 (BC-55 negatives).
Drives the deployed IntentClassifier.
"""
import pytest

from core.intent_classifier import IntentClassifier, IntentType


def _intent(q):
    return IntentClassifier().classify(q)


POSITIVES = [
    "What was I working on in late September?",
    "What was I working on in late September, and when is sim HW 4 due?",
    "What did I do last weekend?",
    "When did we last talk about my resume?",
    "What was I stressed about in early August, and is it still a problem?",
    "Where was I living in March 2025?",
    "What were we building in mid October?",
    "Did I finish the lab report 3 weeks ago?",
    "What was I reading during 2024?",
    "Who was I meeting with in July?",
]

NEGATIVES = [
    "What am I doing next week?",
    "When is HW 5 due?",
    "What was Rome doing in 100 BC?",
    "I was working on HW 4 in late September.",
    "Biscuit turned 2 in September.",
    "What will I be doing in late September?",
    "Is the report due in October?",
    "What is the capital of France in March?",
]


@pytest.mark.parametrize("q", POSITIVES)
def test_positive(q):
    r = _intent(q)
    assert r.intent == IntentType.TEMPORAL_RECALL, (q, r.intent, r.confidence)
    assert r.confidence >= 0.85


@pytest.mark.parametrize("q", NEGATIVES)
def test_negative(q):
    from core.intent_classifier import _past_anchored_first_person_question as f
    assert not f(q), q
    r = _intent(q)
    # The semantic tier (pre-existing, 0.60) may still route a future question
    # ("What am I doing next week?" -> temporal_recall on base too); the
    # regex tier must not claim any of these.
    assert r.intent != IntentType.TEMPORAL_RECALL or r.source == "semantic", q


def test_future_second_clause_does_not_veto():
    r = _intent("What was I working on in late September, and when is the next exam?")
    assert r.intent == IntentType.TEMPORAL_RECALL


def test_past_question_without_anchor_not_claimed():
    from core.intent_classifier import _past_anchored_first_person_question as f
    assert not f("What was I saying?")
