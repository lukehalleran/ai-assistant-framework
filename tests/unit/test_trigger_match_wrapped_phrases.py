"""Phrase keywords match across a client line-wrap.

Incident 2026-10-17 (lane AGT note) and 2026-10-20 (recall probe stored as
"[test]Do you remember what we talked\\n     about ..."): the client
hard-wraps long messages, and every multi-word keyword in
utils.trigger_match was a literal-space substring, so "did we\\n    talk"
missed the Tier-1 memory arm while the clean form hit it. Each space in a
phrase keyword now matches any whitespace run; hit offsets stay positions in
the text as given. class: BC-58 (wrapped-text sibling; the CM-01 chokepoint).
"""

import pytest

from core.agentic.gate import _MEMORY_HIT, _tier1_keyword_hit
from utils.trigger_match import compile_keyword_matcher, find_hits

CLEAN = "hey did we talk about the garden plan last week"
WRAPPED = "hey did we\n     talk about the garden plan last week"


@pytest.mark.parametrize("text", [CLEAN, WRAPPED, "hey did\twe  talk about it"])
def test_tier1_memory_phrase_hits_clean_and_wrapped(text):
    assert _tier1_keyword_hit(text, _MEMORY_HIT)


def test_wrapped_hit_offsets_point_into_the_original_text():
    matcher = compile_keyword_matcher(["did we talk"])
    (hit,) = find_hits(WRAPPED.lower(), matcher)
    assert WRAPPED[hit.start:hit.end] == "did we\n     talk"


def test_trailing_space_sentinel_accepts_a_line_break():
    matcher = compile_keyword_matcher(["tell me about "])
    assert matcher("tell me about\n  the lighthouse")
    assert not matcher("tell me aboutness")


def test_phrase_still_needs_its_words_in_order():
    matcher = compile_keyword_matcher(["did we talk"])
    assert not matcher("did\n we not talk")
    assert not matcher("we did talk")


def test_negation_still_scopes_a_wrapped_phrase():
    matcher = compile_keyword_matcher(["look up"])
    assert find_hits("please look\n   up the ferry times", matcher)
    assert not find_hits("don't look\n   up the ferry times", matcher)


def test_go_to_http_keeps_substring_semantics():
    matcher = compile_keyword_matcher(["go to http"])
    assert matcher("go to https://example.org")
    assert matcher("go to\n  https://example.org")
