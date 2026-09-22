"""utils/read_time_markers.py — machinery text is not content.

class: BC-91. Plan: PLAN_20260922_turn_audit_guardfixes.md §3 (B0).
"""

from core.action_claim_guard import UNVERIFIED_CLAIM_MARKER as GUARD_UNVERIFIED_MARKER
from utils.personal_claim_provenance import (
    KEY,
    MARKER as PROVENANCE_MARKER,
    annotate_personal_claim_memory,
    clean_personal_claim_receipt,
)
from utils.read_time_markers import (
    PERSONAL_CLAIM_MARKER,
    READ_TIME_MARKERS,
    UNVERIFIED_CLAIM_MARKER,
    strip_delivery_notices,
    strip_machinery,
    strip_read_time_markers,
)

CALENDAR_NOTICE = (
    "\n\n> ⚠️ I don't see that on your calendar — nothing "
    "was created. Say \"add it\" and I'll queue a card."
)


# --- (a) marker-line stripping, body preserved byte-identical -------------

def test_strip_read_time_markers_drops_only_the_marker_line():
    for marker in READ_TIME_MARKERS:
        text = f"Line one.\n{marker}\nLine two, trailing spaces.  \nLine three."
        result = strip_read_time_markers(text)
        assert marker not in result
        assert result == "Line one.\nLine two, trailing spaces.  \nLine three."


def test_strip_read_time_markers_handles_marker_with_surrounding_whitespace():
    text = f"Body.\n  {PERSONAL_CLAIM_MARKER}  \nMore body."
    result = strip_read_time_markers(text)
    assert PERSONAL_CLAIM_MARKER not in result
    assert result == "Body.\nMore body."


def test_strip_read_time_markers_leaves_text_without_a_marker_untouched():
    text = "Nothing to see here.\nSecond line."
    assert strip_read_time_markers(text) == text


# --- (b) delivery notices: trailing-run + prefix rule ----------------------

def test_strip_delivery_notices_removes_the_live_calendar_notice():
    body = "Sounds like a rough morning — glad you got a little rest."
    text = body + CALENDAR_NOTICE
    assert strip_delivery_notices(text) == body


def test_strip_delivery_notice_preserves_an_authored_quote_before_suffix():
    body = "Answer.\n\n> The assistant quoted a warning from the document."
    text = body + CALENDAR_NOTICE
    assert strip_delivery_notices(text) == body


def test_strip_multiple_registered_notices_preserves_prior_authored_quote():
    body = "Answer.\n\n> A quoted warning that is part of the answer."
    first = CALENDAR_NOTICE
    second = "\n\n> ⚠️ I couldn't run a fresh web search because today's search budget is used up."
    assert strip_delivery_notices(body + first + second) == body


def test_registered_notice_earlier_in_answer_is_not_removed():
    text = (
        "The system once displayed this notice:\n"
        "> ⚠️ I don't see that on your calendar — nothing was created.\n\n"
        "Here is the actual answer that follows it."
    )
    assert strip_delivery_notices(text) == text


def test_strip_delivery_notices_preserves_a_midtext_blockquote():
    text = (
        "Intro paragraph.\n\n"
        "> ⚠️ heads up, mid text\n\n"
        "A normal paragraph follows and is the last line."
    )
    assert strip_delivery_notices(text) == text


def test_strip_delivery_notices_preserves_a_trailing_non_notice_blockquote():
    text = "Body text here.\n\n> just a quote, not a delivery notice"
    assert strip_delivery_notices(text) == text


def test_strip_delivery_notices_non_str_and_empty_are_unchanged():
    assert strip_delivery_notices(None) is None
    assert strip_delivery_notices("") == ""
    assert strip_delivery_notices(123) == 123


# --- (c) strip_machinery: composition, idempotence, non-str -----------------

def test_strip_machinery_removes_marker_then_trailing_notice():
    body = "Here's what I found."
    text = body + CALENDAR_NOTICE + "\n" + PERSONAL_CLAIM_MARKER
    assert strip_machinery(text) == body


def test_strip_machinery_is_idempotent():
    body = "Here's what I found."
    text = body + CALENDAR_NOTICE + "\n" + PERSONAL_CLAIM_MARKER
    once = strip_machinery(text)
    twice = strip_machinery(once)
    assert once == body
    assert once == twice


def test_strip_machinery_non_str_returns_input_unchanged():
    for value in (None, 123, [], {}, ""):
        assert strip_machinery(value) is value


def test_strip_read_time_markers_and_strip_delivery_notices_never_raise_on_odd_input():
    for fn in (strip_read_time_markers, strip_delivery_notices, strip_machinery):
        assert fn(None) is None
        assert fn(42) == 42
        assert fn([1, 2]) == [1, 2]


# --- (d) parity across modules ----------------------------------------------

def test_marker_parity_across_modules():
    assert PROVENANCE_MARKER is PERSONAL_CLAIM_MARKER
    assert GUARD_UNVERIFIED_MARKER == UNVERIFIED_CLAIM_MARKER


# --- (e) through the deployed functions: digest over pre-suffix text -------

_PRE_SUFFIX_REPLY = "Sounds like a rough morning — glad you got a little rest."


def _receipt_for_pre_suffix_reply(**changes):
    value = dict(status="checked", delivery="unchanged", insufficient_count=1)
    value.update(changes)
    return clean_personal_claim_receipt(value, response=_PRE_SUFFIX_REPLY)


def test_annotate_marks_a_response_whose_receipt_was_hashed_pre_suffix():
    receipt = _receipt_for_pre_suffix_reply()
    stored_response = _PRE_SUFFIX_REPLY + CALENDAR_NOTICE
    item = {"response": stored_response, KEY: receipt}
    marked = annotate_personal_claim_memory(item)
    assert marked["response"] == stored_response.rstrip() + "\n" + PERSONAL_CLAIM_MARKER
    # original item is untouched
    assert item["response"] == stored_response


def test_annotate_does_not_mark_a_different_response_body():
    receipt = _receipt_for_pre_suffix_reply()
    different_response = "Something else entirely happened today." + CALENDAR_NOTICE
    item = {"response": different_response, KEY: receipt}
    assert annotate_personal_claim_memory(item) == item


# ---------------------------------------------------------------------------
# Emitter registry (2026-09-22 review): ONE definition of every notice family.
# ---------------------------------------------------------------------------

import ast as _ast
import pathlib as _pathlib

import pytest as _pytest

import utils.read_time_markers as _rtm
from utils.read_time_markers import DELIVERY_NOTICE_TEXTS, delivery_notice

_REPO = _pathlib.Path(__file__).resolve().parents[2]


def test_delivery_notice_composes_one_registered_line_that_its_own_stripper_removes():
    notice = delivery_notice(_rtm.NOTICE_CALENDAR_UNSEEN, " Say \"add it\" and I'll queue a card.")
    assert notice.startswith("\n\n> ⚠️ I don't see that on your calendar")
    assert "\n" not in notice.strip()
    assert _rtm.strip_delivery_notices("Body." + notice) == "Body."
    assert _rtm.strip_delivery_notices("Body." + notice + delivery_notice(_rtm.NOTICE_WEB_BUDGET, " is used up.")) == "Body."


def test_delivery_notice_rejects_an_unregistered_opening():
    with _pytest.raises(ValueError):
        delivery_notice("Heads up — something new", " detail")


def test_every_registered_opening_is_a_module_constant_and_unique():
    constants = {n: v for n, v in vars(_rtm).items() if n.startswith("NOTICE_")}
    assert set(constants.values()) == set(DELIVERY_NOTICE_TEXTS)
    assert len(set(DELIVERY_NOTICE_TEXTS)) == len(DELIVERY_NOTICE_TEXTS)


def test_no_emitter_writes_the_blockquote_literal_outside_the_leaf():
    """Every `> ⚠️` in production code lives in the leaf; emitters compose via
    delivery_notice(NOTICE_*). Mirrors DM-38 rule 4 at unit-lane speed."""
    offenders = []
    for root in ("core", "gui", "utils", "api", "knowledge", "memory"):
        for path in (_REPO / root).rglob("*.py"):
            rel = path.relative_to(_REPO).as_posix()
            if rel == "utils/read_time_markers.py":
                continue
            tree = _ast.parse(path.read_text(encoding="utf-8"))
            for node in _ast.walk(tree):
                if isinstance(node, _ast.Constant) and isinstance(node.value, str) and "> ⚠️" in node.value:
                    offenders.append(f"{rel}:{node.lineno}")
                elif isinstance(node, _ast.Call):
                    tail = node.func.attr if isinstance(node.func, _ast.Attribute) else getattr(node.func, "id", "")
                    if tail == "delivery_notice":
                        first = node.args[0] if node.args else None
                        name = (first.attr if isinstance(first, _ast.Attribute) else getattr(first, "id", "")) if first is not None else ""
                        if not name.startswith("NOTICE_"):
                            offenders.append(f"{rel}:{node.lineno} (unregistered opening)")
    assert offenders == []


def test_deployed_emitters_produce_only_registered_families():
    """The constants the guards actually export strip cleanly — the registry
    and the emitters cannot drift apart because they are the same strings."""
    import core.action_claim_guard as acg
    import utils.web_evidence_receipt as wer
    for notice in (acg.NO_CARD_NOTICE, wer.BUDGET_NOTICE, wer.BUDGET_NOTICE_PARTIAL):
        assert _rtm.strip_delivery_notices("Answer." + notice) == "Answer."
