"""Ambiguous verbs count as technical/mathematical only near their object.

Incidents 2026-10-18/19, classes BC-01 / BC-73: "its not working i keep getting
angry" and "i dont feel good when im not working" classified technical_help@0.75;
"the Assign Module to differentiate our entities" opened a computation loop
(tier1_computation). One shared helper (utils/anchored_sense) serves the
crashed/broke rule, the not-working rule and the differentiate/integrate rule.
"""

import pytest

from core.agentic.gate import evaluate_agentic_gate
from core.intent_classifier import IntentClassifier, IntentType


def _wrap(s):
    """A client soft line-wrap: break at the first space after word 3."""
    w = s.split(" ")
    return " ".join(w[:3]) + "\n" + " ".join(w[3:])


def _intent(q):
    return IntentClassifier().classify(q).intent


NOT_TECH = [
    "its not working i keep getting angry",
    "i dont feel good when im not working",
    "my job isnt working out and I feel stuck",
    "Rowan said the plan doesn't work for her schedule",
]
TECH = [
    "the script is not working",
    "my wifi isn't working",
    "the build doesn't work after the update",
    "the server isn't working again",
]


@pytest.mark.parametrize("q", NOT_TECH)
def test_everyday_not_working_is_not_technical_help(q):
    for text in (q, _wrap(q)):
        assert _intent(text) != IntentType.TECHNICAL_HELP, text


@pytest.mark.parametrize("q", TECH)
def test_not_working_near_a_tech_noun_is_technical_help(q):
    for text in (q, _wrap(q)):
        assert _intent(text) == IntentType.TECHNICAL_HELP, text


def test_crashed_rule_behaviour_unchanged():
    assert _intent("Yeah I crashed a bit. So tired. Walking to store now") != IntentType.TECHNICAL_HELP
    assert _intent("my app crashed on startup") == IntentType.TECHNICAL_HELP
    assert _intent("the laptop broke yesterday") == IntentType.TECHNICAL_HELP


async def _modes(q):
    return (await evaluate_agentic_gate(q)).modes


@pytest.mark.asyncio
@pytest.mark.parametrize("q", [
    "differentiate x^2 + 3x",
    "integrate f(x) dx from 0 to 1",
    "differentiate the function with respect to t",
])
async def test_math_verbs_with_a_math_object_open_computation(q):
    for text in (q, _wrap(q)):
        assert "computation" in await _modes(text), text


@pytest.mark.asyncio
@pytest.mark.parametrize("q", [
    "the Assign Module to differentiate our entities",
    "integrate the feedback into the essay",
    "how do we differentiate Rowan from Tamsin in the story",
])
async def test_math_verbs_without_a_math_object_do_not(q):
    for text in (q, _wrap(q)):
        assert "computation" not in await _modes(text), text


def test_helper_is_shape_based():
    from utils import anchored_sense
    assert anchored_sense.math_verb_has_math_object("integrate 2x + 1")
    assert not anchored_sense.math_verb_has_math_object("integrate the new hires")
