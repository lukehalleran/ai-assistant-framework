"""Assistant text never corroborates a personal claim; counts need a report per day.

Incident 2026-10-17 (live, invented details): the draft told the user "you were
up late grinding through that module last night" when only the assistant's own
prior turn said so, then counted that line as a night ("the second late night in
a row"). The auditor returned supported for 1/1 and 7/7 claims.
Classes: BC-46 (assistant text as evidence), BC-75 (laundering through the
assistant's own earlier turn), BC-84 (one bad element must not pass the whole).

Drives THE deployed ``audit_personal_claims`` with a misbehaving scripted model
(it always says "supported") over evidence from THE deployed
``build_personal_evidence``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from core.personal_claim_check import (
    audit_personal_claims,
    build_personal_evidence,
    claim_repetition_count,
)


class _Model:
    def __init__(self, payload):
        self.payload = payload

    async def generate_once(self, prompt, **kwargs):
        return json.dumps(self.payload)


def _turn(stamp, query, response, ident):
    return {"id": ident, "timestamp": stamp, "query": query, "response": response}


def _wrap(text, wrapped):
    return text.replace(". ", ".\n    ").replace(" last night", "\n    last night") if wrapped else text


def _ref(evidence, needle, role=None):
    row = next(r for r in evidence if needle in r["text"] and (role is None or r["role"] == role))
    return {"source_id": row["source_id"], "quote": needle}


def _run(draft, evidence, claim):
    payload = {"claims": [claim]}
    return asyncio.run(audit_personal_claims(draft, evidence, _Model(payload)))


ASSIST_LINE = "You were up late grinding through that module last night."


def _evidence(prev_stamp="2026-10-16T09:00:00", extra_ctx=None):
    ctx = [_turn(prev_stamp, "Hey, quick one about Rowan's lab", ASSIST_LINE + " Tamsin would approve.", "t1")]
    if extra_ctx:
        ctx.extend(extra_ctx)
    return build_personal_evidence("I was up til like 5 maybe.", {"recent_conversations": ctx})


@pytest.mark.parametrize("wrapped", [False, True])
def test_assistant_only_state_claim_is_demoted_and_flagged(wrapped):
    draft = _wrap("Welcome back. " + ASSIST_LINE + " Rest up.", wrapped)
    ev = _evidence()
    span = ASSIST_LINE.replace(" last night", "\n    last night") if wrapped else ASSIST_LINE
    claim = {"text": span, "status": "supported", "kind": "state",
             "evidence": [_ref(ev, "grinding through that module", "assistant")]}
    result = _run(draft, ev, claim)
    assert result.claims[0]["status"] == "insufficient"
    assert result.demoted_count == 1
    assert result.receipt()["flagged_spans"]


@pytest.mark.parametrize("wrapped", [False, True])
def test_same_claim_citing_current_user_query_stays_supported(wrapped):
    draft = _wrap("Welcome back. " + ASSIST_LINE + " Rest up.", wrapped)
    ev = _evidence()
    span = ASSIST_LINE.replace(" last night", "\n    last night") if wrapped else ASSIST_LINE
    claim = {"text": span, "status": "supported", "kind": "state",
             "evidence": [_ref(ev, "up til like 5", "user")]}
    assert _run(draft, ev, claim).claims[0]["status"] == "supported"


def test_discussion_kind_on_assistant_row_untouched():
    ev = _evidence()
    claim = {"text": ASSIST_LINE, "status": "supported", "kind": "discussion",
             "evidence": [_ref(ev, "grinding through that module", "assistant")]}
    assert _run(ASSIST_LINE, ev, claim).claims[0]["status"] == "supported"


def test_unknown_role_only_claim_unchanged():
    ev = [{"source_id": "src_current_query", "role": "user", "timestamp": None, "text": "hi"},
          {"source_id": "src_unk", "role": "unknown", "timestamp": None, "text": "slept little"}]
    claim = {"text": "You slept little.", "status": "supported", "kind": "state",
             "evidence": [{"source_id": "src_unk", "quote": "slept little"}]}
    assert _run("You slept little.", ev, claim).claims[0]["status"] == "supported"


COUNT = "That's the second late night in a row."


@pytest.mark.parametrize("wrapped", [False, True])
def test_count_claim_needs_one_user_day_per_count(wrapped):
    draft = COUNT.replace("late night", "late\n    night") if wrapped else COUNT
    span = draft
    only_now = _evidence()
    c1 = {"text": span, "status": "supported", "kind": "pattern",
          "evidence": [_ref(only_now, "up til like 5", "user")]}
    assert _run(draft, only_now, c1).claims[0]["status"] == "insufficient"

    prior = _evidence(extra_ctx=[_turn("2026-10-15T23:30:00", "Up late again, finished the lab", "ok", "t0")])
    c2 = {"text": span, "status": "supported", "kind": "pattern",
          "evidence": [_ref(prior, "up til like 5", "user"), _ref(prior, "Up late again", "user")]}
    assert _run(draft, prior, c2).claims[0]["status"] == "supported"

    same_day = _evidence(extra_ctx=[
        _turn("2026-10-16T08:00:00", "Slept badly", "ok", "ta"),
        _turn("2026-10-16T20:00:00", "Slept badly again today", "ok", "tb")])
    c3 = {"text": span, "status": "supported", "kind": "pattern",
          "evidence": [_ref(same_day, "Slept badly", "user"), _ref(same_day, "Slept badly again", "user")]}
    assert _run(draft, same_day, c3).claims[0]["status"] == "insufficient"


@pytest.mark.parametrize("text,expected", [
    ("second late night in a row", 2),
    ("3rd time this week", 3),
    ("two nights straight", 2),
    ("three late nights in a row", 3),
    ("the second chapter", None),
    ("first late night", None),
    ("again", None),
    ("a late night in a row", None),
    # the time unit, never an ordinal count
    ("wait a second and try again", None),
    ("hang on one second, again", None),
    ("your second exam this week", 2),
])
def test_claim_repetition_count(text, expected):
    assert claim_repetition_count(text) == expected
