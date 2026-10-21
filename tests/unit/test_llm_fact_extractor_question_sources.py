"""A question asserts nothing about the asker (shutdown LLM path).

Incident 2026-10-17 (class: BC-51 speech act): shutdown facts
`exactly | is_a | the difference between ...`, `user | goal | edit` and
`user | is | linux` came from QUESTIONS. The per-line regex guard
(``_is_interrogative``) already covers the regex arm; the LLM provenance join
(``_attach_source_excerpts``) and the post-hoc extract_facts gate had none.
Decision: the whole interrogative SENTENCE is dropped, including a statement
embedded in it ("Do you remember I finished the draft?"): the user presupposed
rather than reported it, and a plain statement later restores the fact. A
declarative in a SEPARATE sentence still extracts.
"""
import asyncio

import pytest

from memory.fact_extractor import FactExtractor, _interrogative_conflict
from memory.llm_fact_extractor import LLMFactExtractor


def _kept(message, obj, rel="is"):
    triples = [{"subject": "user", "relation": rel, "object": obj}]
    LLMFactExtractor._attach_source_excerpts(triples, [message])
    return triples


def _forms(t):
    return [t, "   " + t.replace(" ", "\n   ", 3)]


@pytest.mark.parametrize("message,obj", [
    ("Do you remember I finished the draft?", "the draft"),
    ("Am I a linux user?", "linux user"),
    ("Why would I want to edit the data module?", "edit the data module"),
    ("What exactly is the difference between flowchart module and data module?",
     "the difference between flowchart module and data module"),
    ("Is the linux user group still meeting", "linux user group"),   # inversion, no '?'
])
def test_question_sentence_yields_no_llm_fact(message, obj):
    for form in _forms(message):
        assert _kept(form, obj) == [], form


@pytest.mark.parametrize("message,obj", [
    ("Is linux ok on my laptop? I am a linux user.", "linux user"),
    ("I am a linux user. Do you remember?", "linux user"),
    ("What is wrong with it? I finished the draft.", "the draft"),
])
def test_declarative_in_its_own_sentence_still_extracts(message, obj):
    for form in _forms(message):
        got = _kept(form, obj)
        assert len(got) == 1 and obj in got[0]["source_excerpt"], form


@pytest.mark.parametrize("sentence,obj,expected", [
    ("Is Rowan my manager?", "my manager", True),
    ("Rowan is my manager.", "my manager", False),
    ("Who is the manager", "the manager", True),
    ("What I need is a break.", "a break", False),
])
def test_predicate_directly(sentence, obj, expected):
    assert _interrogative_conflict(sentence, obj) is expected


def test_regex_arm_unchanged():
    fe = FactExtractor(use_rebel=False)
    got = [n.content for n in asyncio.run(fe.extract_facts("I want to edit the essay.", ""))]
    assert got  # statement still extracts
    assert asyncio.run(fe.extract_facts("why would i want to edit data module?", "")) == []
