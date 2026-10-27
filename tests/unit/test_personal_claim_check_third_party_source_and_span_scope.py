"""Personal-claim check: third-party facts, generic pronouns, auditor span scope (lane C1, wave 2).

Incidents 2026-10-17/18 (FOLLOWUPS): (a) third-party facts from this turn's web
results were labelled completions and flagged; (b) generic "them"/"they" counted as
a person; (c) on terse turns the auditor quoted the USER's message instead of the
draft. Classes: BC-84, BC-46, BC-75.

Drives THE deployed ``build_personal_evidence`` / ``audit_personal_claims``.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from core import personal_claim_check as pcc
from core.personal_claim_check import audit_personal_claims, build_personal_evidence


class _Model:
    def __init__(self, raw):
        self.raw = raw

    async def generate_once(self, prompt, **kwargs):
        return json.dumps(self.raw)


@pytest.fixture(autouse=True)
def _display_name(monkeypatch):
    monkeypatch.setattr(pcc, "get_user_display_name", lambda: "Rowan", raising=False)


def _wrap(text, wrapped):
    return text.replace(" ", "\n    ", 3) if wrapped else text


def _web(text):
    page = SimpleNamespace(title="Results page", content=text, url="https://example.invalid/a")
    return {"web_search_results": SimpleNamespace(pages=[page])}


def _run(draft, evidence, claims):
    return asyncio.run(audit_personal_claims(draft, evidence, _Model({"claims": claims})))


def _source_id(evidence):
    return next(r["source_id"] for r in evidence if r["role"] == "source")


# ---- (a) third-party completion supported by this turn's web source ----------

THIRD = "Tamsin finished the regional marathon in March."
PAGE = "Local news: Tamsin finished the regional marathon in March, placing fourth overall."


@pytest.mark.parametrize("wrapped", [False, True])
def test_web_pages_become_source_rows(wrapped):
    ev = build_personal_evidence("who ran the marathon?", _web(_wrap(PAGE, wrapped)))
    src = [r for r in ev if r["role"] == "source"]
    assert len(src) == 1 and "marathon" in src[0]["text"]


@pytest.mark.parametrize("wrapped", [False, True])
def test_third_party_completion_backed_by_source_stays_supported(wrapped):
    ev = build_personal_evidence("who ran the marathon?", _web(PAGE))
    shown = _wrap(THIRD, wrapped)
    res = _run("Okay. " + shown + " Moving on.", ev, [{
        "text": shown, "status": "supported", "kind": "completion",
        "evidence": [{"source_id": _source_id(ev), "quote": "Tamsin finished the regional marathon in March"}]}])
    assert [c["status"] for c in res.claims] == ["supported"]
    assert res.demoted_count == 0


def test_user_completion_is_not_supported_by_a_web_source():
    ev = build_personal_evidence("hello", _web("Rowan finished the regional marathon in March."))
    res = _run("Okay. You finished the regional marathon in March. Moving on.", ev, [{
        "text": "You finished the regional marathon in March.", "status": "supported", "kind": "completion",
        "evidence": [{"source_id": _source_id(ev), "quote": "Rowan finished the regional marathon in March"}]}])
    assert [c["status"] for c in res.claims] == ["insufficient"]


def test_third_party_completion_without_source_row_is_still_insufficient():
    ctx = {"recent_conversations": [{"query": "earlier", "response": THIRD, "timestamp": "2026-10-20T10:00:00"}]}
    ev = build_personal_evidence("who ran the marathon?", ctx)
    assistant = next(r["source_id"] for r in ev if r["role"] == "assistant")
    res = _run("Okay. " + THIRD + " Moving on.", ev, [{
        "text": THIRD, "status": "supported", "kind": "completion",
        "evidence": [{"source_id": assistant, "quote": "Tamsin finished the regional marathon"}]}])
    assert [c["status"] for c in res.claims] == ["insufficient"]


# ---- (b) generic pronouns are not persons ------------------------------------

GENERIC = "The exam hands them a table and asks them to compute the mean."


@pytest.mark.parametrize("wrapped", [False, True])
def test_generic_them_in_course_prose_is_subject_matter(wrapped):
    shown = _wrap(GENERIC, wrapped)
    ev = build_personal_evidence("explain the exam format", {})
    res = _run("Okay. " + shown + " Moving on.", ev, [
        {"text": shown, "status": "insufficient", "kind": "completion", "evidence": []}])
    assert res.claims == [] and res.subject_drops == 1


@pytest.mark.parametrize("span", ["They submitted the form.", "She dropped off the package."])
def test_third_person_subject_of_a_past_act_stays_personal(span):
    ev = build_personal_evidence("explain the exam format", {})
    res = _run("Okay. " + span + " Moving on.", ev, [
        {"text": span, "status": "insufficient", "kind": "completion", "evidence": []}])
    assert len(res.claims) == 1 and res.flagged_spans == [span]


# ---- (c) the auditor quoted the user's message -------------------------------

USER_MSG = "I finished the HW5 writeup last night"


@pytest.mark.parametrize("wrapped", [False, True])
def test_claim_copied_from_the_user_message_is_dropped_not_relocated(wrapped):
    # the draft echoes the user's words in a longer sentence, so token overlap would relocate onto it
    draft = "Nice, so after you finished the HW5 writeup last night the next step is the review."
    ev = build_personal_evidence(USER_MSG, {})
    res = _run(draft, ev, [{"text": _wrap(USER_MSG, wrapped), "status": "insufficient",
                            "kind": "completion", "evidence": []}])
    assert res.claims == [] and res.dropped_claim_count == 1 and res.flagged_spans == []


def test_paraphrase_of_the_draft_is_still_relocated():
    draft = "Tomorrow's got the cover letter waiting for a fresher brain."
    ev = build_personal_evidence("tired", {})
    res = _run(draft, ev, [{"text": "User has the cover letter waiting for a fresher brain tomorrow.",
                            "status": "insufficient", "kind": "plan", "evidence": []}])
    assert len(res.claims) == 1 and res.relocated_count == 1
