"""Recalled-time anchor brackets are prompt machinery: replies that copy them are cleaned.

Incident 2026-10-19 15:32 (live [test] recall probe, kimi-3, decision-answer reuse):
#90 annotates time words in recalled text with the date they meant ("today
[Sat 2026-10-17]"). The reply copied the style into what the user saw and what was
stored: "Yesterday [Sun 2026-10-18] was a grind ... How are you feeling about it
today [Mon 2026-10-19]?".

Classes: BC-91 (machinery text as content), BC-100 (the anchors themselves).
ONE definition (utils.read_time_markers) serves the emitter's idempotence guard and
the stripper; strip_trailing_stream_artifact carries it to every display/storage path.
Invented content only.
"""
from datetime import datetime

import pytest

import utils.read_time_markers as read_time_markers
import utils.temporal_resolver as temporal_resolver
from core.response_parser import ResponseParser

REPLY = ("Yesterday [Sun 2026-10-18] was a long one for the quasar project. "
         "How is it going today [Mon 2026-10-19]? Next week [week of Sat 2026-10-24] "
         "looks lighter, and next month [November 2026] the review lands.")
CLEAN = ("Yesterday was a long one for the quasar project. How is it going today? "
         "Next week looks lighter, and next month the review lands.")


def _wrap(text):
    return "\n".join("    " + part for part in text.split(". "))


@pytest.mark.parametrize("wrapped", [False, True])
def test_strip_time_anchors_removes_copied_brackets(wrapped):
    src = _wrap(REPLY) if wrapped else REPLY
    want = _wrap(CLEAN) if wrapped else CLEAN
    assert read_time_markers.strip_time_anchors(src) == want


@pytest.mark.parametrize("text", [
    "See [WEB_1] and [MEM_RECENT_2].",
    "Logged on [2026-10-18] in the notes.",
    "The file [Sun 2026-10-18 notes].md is fine.",
    "A [link](https://example.invalid) and [x] a checkbox.",
    "Due Sunday 2026-10-18, as planned.",
    "[Personal-claim check: x]",
])
def test_other_brackets_untouched(text):
    assert read_time_markers.strip_time_anchors(text) == text


def test_strip_is_the_inverse_of_the_emitter():
    said = "Finish the quasar draft by end of day today, review it tomorrow, ship next week."
    anchored = temporal_resolver.anchor_temporal_references(
        said, "2026-10-17T20:00:00", datetime(2026, 10, 19, 9, 0))
    assert anchored != said and "[Sat 2026-10-17]" in anchored
    assert read_time_markers.strip_time_anchors(anchored) == said
    # the emitter's guard uses the same definition: no double annotation
    again = temporal_resolver.anchor_temporal_references(
        anchored, "2026-10-17T20:00:00", datetime(2026, 10, 19, 9, 0))
    assert again == anchored


def test_display_path_strips():
    assert ResponseParser.strip_trailing_stream_artifact(REPLY) == CLEAN


def test_storage_path_strips():
    assert ResponseParser.sanitize_for_storage(REPLY) == CLEAN


def test_strip_machinery_strips():
    assert read_time_markers.strip_machinery(REPLY) == CLEAN


@pytest.mark.parametrize("value", [None, "", 7])
def test_non_text_returned_unchanged(value):
    assert read_time_markers.strip_time_anchors(value) == value
