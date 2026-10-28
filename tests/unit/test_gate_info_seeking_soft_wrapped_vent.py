"""A soft-wrapped vent statement is not info-seeking just because a wrapped line starts "how".

Incident date: 2026-10-27 (wave-1/2 follow-up).  Class: BC-06 (a boundary regex that
treats layout as meaning): ``_INFO_SEEKING_SENTENCE_SPLIT_RE`` split on every newline, so
a line wrapped mid-sentence that began with an interrogative opener made the whole vent
"info-seeking" and the tone-statement veto stood down.

Drives THE deployed ``core.agentic.gate._is_info_seeking``.
"""
import textwrap

import pytest

from core.agentic.gate import _is_info_seeking

VENT = "I keep replaying the whole thing with my dad and how it all fell apart again"


def _wrapped(text, width):
    return "\n".join("  " + ln for ln in textwrap.wrap(text, width))


def test_clean_vent_is_not_info_seeking():
    assert _is_info_seeking(VENT) is False


@pytest.mark.parametrize("width", [20, 36, 50])
def test_wrapped_vent_is_not_info_seeking(width):
    wrapped = _wrapped(VENT, width)
    assert _is_info_seeking(wrapped) is False


def test_wrap_exactly_before_how():
    assert _is_info_seeking("I keep replaying the whole thing with my dad and\nhow it all fell apart again") is False


@pytest.mark.parametrize("text", [
    "Rough night. How do I stop replaying it",
    "Rough night.\nHow do I stop replaying it",
    "  Rough night.\n  How do I stop\n  replaying it",
])
def test_real_sentence_openers_still_count(text):
    assert _is_info_seeking(text) is True
