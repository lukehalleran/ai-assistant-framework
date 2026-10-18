"""Incident 2026-10-17 (class BC-06, over-fire at the extraction chokepoint).

The idle-flush shutdown re-ran FactExtractor on session turns and stored
``exactly | is_a | the difference between ...`` and ``user | goal | edit`` from a
QUESTION, and ``user | is | linux`` from "i am on linux rn" (a state, not an
identity). Driven through THE deployed FactExtractor (``_extract_with_regex``
and ``extract_facts``); every live text is asserted clean AND wrapped/indented.
"""
import asyncio

import pytest

from memory.fact_extractor import FactExtractor, _is_interrogative

Q1 = ("what exactly is the difference between flowchart module and data module. "
      "why would i want to edit data module")
Q1_WRAPPED = ("   what exactly is the difference between flowchart module\n"
              "   and data module. why would i want to edit data module\n")
STATE = "i am on linux rn"
STATE_WRAPPED = "  first, i am on linux rn,\n   so that is where I am"
STATE_Q = "first, i am on linux rn, do i need to be on windows to install it"


@pytest.fixture(scope="module")
def fe():
    return FactExtractor(use_rebel=False, use_regex=True)


def _rx(fe, text):
    return fe._extract_with_regex(text)


def _full(fe, text):
    return asyncio.run(fe.extract_facts(text, ""))


def _contents(nodes):
    return [n.content for n in nodes]


@pytest.mark.parametrize("text", [Q1, Q1_WRAPPED])
def test_incident_question_yields_nothing(fe, text):
    triples = _rx(fe, text)
    assert not [t for t in triples if t[0] == "exactly"]
    assert not [t for t in triples if t[1] == "goal"]
    assert triples == []
    assert _full(fe, text) == []


@pytest.mark.parametrize("text", [
    "do i need to be on windows to install it",
    "   do i need to be on windows\n   to install it",
    "can you send it to Rowan?",
    "  can you send it\n   to Rowan?\n",
])
def test_questions_yield_no_triples(fe, text):
    assert _rx(fe, text) == []
    assert _full(fe, text) == []


@pytest.mark.parametrize("text", [STATE, STATE_WRAPPED, STATE_Q])
def test_prepositional_state_is_not_identity(fe, text):
    assert not [t for t in _rx(fe, text) if t[0] == "user" and t[1] == "is"]
    assert not [c for c in _contents(_full(fe, text)) if c.startswith("user | is |")]


@pytest.mark.parametrize("text", ["I'm in Chicago.", "   I'm in Chicago.\n  "])
def test_location_statement_still_lives_in(fe, text):
    # Base (pristine clone) produced ('user','lives_in','chicago') for this text.
    assert ("user", "lives_in", "chicago", 0.75, "regex") in _rx(fe, text)


def test_live_in_still_lives_in(fe):
    assert ("user", "lives_in", "leeds", 0.75, "regex") in _rx(fe, "I live in Leeds.")


@pytest.mark.parametrize("text,expected", [
    ("I want to edit the essay.", ("user", "goal", "edit the essay", 0.7, "regex")),
    ("   I want to edit the essay.\n", ("user", "goal", "edit the essay", 0.7, "regex")),
    ("My sister is a nurse.", ("user", "sister", "a nurse", 0.6, "regex")),
    ("Tamsin is my manager.", ("tamsin", "is", "my manager", 0.6, "regex")),
    ("What I need is a break.", ("need", "is_a", "a break", 0.6, "regex")),
    ("  What I need is a break.\n", ("need", "is_a", "a break", 0.6, "regex")),
])
def test_statements_unchanged(fe, text, expected):
    got = _rx(fe, text)
    if expected is not None:
        assert expected in got


def test_goal_statement_kept_after_a_question_in_same_line(fe):
    got = _rx(fe, "why would i do that? I want to finish the essay tonight.")
    assert [t for t in got if t[1] == "goal"]


@pytest.mark.parametrize("sentence,expected", [
    ("what exactly is the difference", True),
    ("why would i want to edit data module", True),
    ("how do you do that", True),
    ("do i need to be on windows", True),
    ("can you send it to Rowan?", True),
    ("Rowan is here?", True),
    ("What I need is a break.", False),
    ("How I spent the day: gym and reading.", False),
    ("When I moved to Leeds I started running.", False),
    ("I want to edit the essay.", False),
    ("", False),
])
def test_is_interrogative_closed_grammar(sentence, expected):
    assert _is_interrogative(sentence) is expected
