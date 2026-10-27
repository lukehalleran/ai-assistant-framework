"""V3 (incident 2026-09-20, measured 2026-10-26; class: BC-75, BC-58).

A resume typed into the message box minted `graduated`, `studies_at` and
`goals` facts: the paste guard only dropped triples whose SUBJECT was not the
user, and resume-derived triples have subject=user. A user-subject triple from
a paste-sized message must also be supported by the user's own typed framing
outside the paste (`paste_shape.authored_framing` + THE provenance join).

Driven through the deployed `_paste_guard_filter` (per-turn regex path) and
`LLMFactExtractor._attach_source_excerpts` (shutdown LLM path).
"""
from __future__ import annotations

import pytest

from memory.llm_fact_extractor import LLMFactExtractor
from memory.memory_storage import _paste_guard_filter

_RESUME = (
    "Tamsin Rowan\nActuarial Analyst, Halden Group, Metro City\n\n"
    "EDUCATION\n- B.S. Mathematics, Larkfield University, graduated 2019\n"
    "- Studies at Larkfield Graduate School, part-time\n\n"
    "OBJECTIVE\nSeeking data science internships for the summer.\n\n"
    + ("Built forecasting models for claims reserving and reported results to "
       "the finance committee every quarter. " * 14)
)


def _wrap(text, width=70):
    import textwrap
    return "\n".join(textwrap.fill(p, width) for p in text.split("\n"))


def _indent(text):
    return "\n".join("    " + ln for ln in text.split("\n"))


def _fact(rel, obj, subj="user"):
    return {"content": f"{subj} | {rel} | {obj}",
            "metadata": {"subject": subj, "relation": rel, "object": obj}}


def _triple(rel, obj, subj="user"):
    return {"subject": subj, "relation": rel, "object": obj, "confidence": 0.9}


FRAMINGS = ["here's my resume\n\n", "here's my resume, can you tighten it up:\n"]


@pytest.mark.parametrize("framing", FRAMINGS)
@pytest.mark.parametrize("shape", [lambda t: t, _wrap, _indent])
def test_per_turn_resume_paste_yields_no_user_triples(framing, shape):
    msg = shape(framing + _RESUME)
    facts = [_fact("graduated", "2019"), _fact("studies_at", "Larkfield Graduate School"),
             _fact("goal", "data science internships")]
    assert _paste_guard_filter(msg, facts) == []


@pytest.mark.parametrize("shape", [lambda t: t, _wrap])
def test_typed_statement_in_the_framing_is_kept(shape):
    msg = shape("I graduated in 2019 and I'm polishing this. Here's my resume:\n\n" + _RESUME)
    kept = _paste_guard_filter(msg, [_fact("graduated", "2019"),
                                     _fact("studies_at", "Larkfield Graduate School")])
    assert [f["metadata"]["relation"] for f in kept] == ["graduated"]


def test_short_typed_statement_outside_any_paste_is_kept():
    facts = [_fact("graduated", "2019")]
    assert _paste_guard_filter("I graduated in 2019", facts) is facts


def test_long_message_without_identifiable_framing_is_unchanged():
    from memory.memory_storage import supported_outside_paste   # new name: keep the other tests collectable on base
    msg = "I have been thinking about school lately. " * 60   # no structural boundary
    assert supported_outside_paste(msg, _triple("graduated", "2019")) is True


@pytest.mark.parametrize("shape", [lambda t: t, _wrap, _indent])
def test_shutdown_llm_path_drops_resume_triples_and_keeps_typed_ones(shape):
    msg = shape("I finished my degree, graduated in 2021. Resume below:\n\n" + _RESUME)
    triples = [_triple("graduated", "2019"), _triple("graduated", "2021"),
               _triple("studies_at", "Larkfield Graduate School")]
    LLMFactExtractor._attach_source_excerpts(triples, [{"query": msg}])
    assert [(t["relation"], t["object"]) for t in triples] == [("graduated", "2021")]
