"""A completion relation cannot ride on a subordinate once/when/after/if clause.

Incident 2026-10-20 (oct20b residual 1a), class: BC-58 (BC-51 modality sibling).
The per-turn extractor already drops "Once I have finished X, I'll ..." through
its dependency-parse guard, but the provenance join (``find_supporting_user_span``,
the gate every shutdown-LLM triple passes) judged "finished" a past-tense cue and
supported ``user | finished | the lab``.  Drives THE deployed join; clean AND
wrapped/indented forms.  Past-tense subordinate clauses still report facts.
"""
import pytest

from memory.fact_source import find_supporting_user_span


def _supported(text, relation="finished", obj="the lab"):
    triple = {"subject": "user", "relation": relation, "object": obj}
    return find_supporting_user_span(triple, [{"query": text}]) is not None


def _forms(text):
    return [text, "   " + text.replace(", ", ",\n   ") + "\n"]


@pytest.mark.parametrize("text", [
    "Once I have finished the lab, I'll submit it.",
    "Once I have finished the lab",
    "When I have finished the lab, I will sleep.",
    "After I have finished the lab, I'm going to nap.",
    "If I finished the lab, I'd sleep.",
])
def test_subordinate_clause_completion_is_not_supported(text):
    for form in _forms(text):
        assert not _supported(form), form


@pytest.mark.parametrize("text", [
    "I have finished the lab.",
    "Once I finished the lab I went home.",
    "After I had finished the lab, I went to bed.",
    "I finished the lab yesterday.",
])
def test_reported_completion_is_still_supported(text):
    for form in _forms(text):
        assert _supported(form), form
