"""Course explanations labelled "completion" are subject matter unless a person did something.

Incident 2026-10-18 (lesson session): 22 of 24 turns had course explanations
flagged as unverified personal claims because the auditor (gpt-4o-mini) labelled
them kind ``completion`` and the subject-matter backstop returned False for every
completion kind; generic tutoring "you" also counted as a person.
Classes: BC-84 (personal-claim support), BC-58 (guard covers only the first path).

Drives THE deployed ``audit_personal_claims`` / ``_is_subject_matter_span``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from core import personal_claim_check as pcc
from core.personal_claim_check import audit_personal_claims, build_personal_evidence

QUERY = "Almost about to have breakfast first"

SUBJECT_SPANS = [
    "TRUE — you can find individual Seize, Delay, and Release modules in the Advanced Process Template.",
    "You don't need the Blocks template to get an infinite-capacity queue.",
    "**Cyclical** – Round-robin style.",
    "The buffer size gets set directly in the Queue block.",
    "You seize a resource and release it in the same block.",
]
PERSONAL_SPANS = [
    "You finished the second lab yesterday.",
    "You've already sent the form.",
    "Finished the reading on Tuesday.",
    "She dropped off the package.",
    "We moved the meeting.",
    "You submitted it.",
    "Rowan has the forms.",
    "Tamsin keeps the spare key.",
]


class _Model:
    def __init__(self, raw):
        self.raw = raw

    async def generate_once(self, prompt, **kwargs):
        return json.dumps(self.raw)


def _wrap(text, wrapped):
    return text.replace(" ", "\n    ", 3) if wrapped else text


def _audit(span, kind="completion", status="insufficient", wrapped=False):
    shown = _wrap(span, wrapped)
    draft = "Okay. " + shown + " Moving on."
    payload = {"claims": [{"text": shown, "status": status, "kind": kind, "evidence": []}]}
    return asyncio.run(audit_personal_claims(draft, build_personal_evidence(QUERY, {}), _Model(payload)))


@pytest.fixture(autouse=True)
def _display_name(monkeypatch):
    monkeypatch.setattr(pcc, "get_user_display_name", lambda: "Tamsin", raising=False)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("span", SUBJECT_SPANS)
def test_course_explanation_labelled_completion_is_dropped(span, wrapped):
    result = _audit(span, wrapped=wrapped)
    assert result.claims == []
    assert result.subject_drops == 1
    assert result.receipt()["flagged_spans"] == []


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("span", PERSONAL_SPANS)
def test_personal_completion_still_flagged(span, wrapped):
    result = _audit(span, wrapped=wrapped)
    assert result.subject_drops == 0
    assert [c["status"] for c in result.claims] == ["insufficient"]
    assert len(result.receipt()["flagged_spans"]) == 1


@pytest.mark.parametrize("span", SUBJECT_SPANS[:3])
def test_plan_kind_untouched(span):
    assert pcc._is_subject_matter_span(span, "plan") is False
    assert pcc._is_subject_matter_span(span, "cancellation") is False


def test_supported_claims_untouched():
    result = _audit("We moved the meeting.", status="supported")
    # supported without evidence is demoted exactly as before; never a subject drop
    assert result.subject_drops == 0


def test_non_completion_rule_unchanged():
    assert pcc._is_subject_matter_span("You don't need the Blocks template.", "other") is False
    assert pcc._is_subject_matter_span("The queue holds entities.", "other") is True


def test_fails_closed_without_parser(monkeypatch):
    monkeypatch.setattr(pcc, "_get_span_nlp", lambda: None)
    assert pcc._is_subject_matter_span(SUBJECT_SPANS[1], "completion") is False


def test_time_reference_keeps_completion_flagged():
    assert pcc._is_subject_matter_span("The queue was emptied at 3pm.", "completion") is False


def test_event_loop_is_not_blocked_by_parsing(monkeypatch):
    import threading
    seen = []
    real = pcc._validate_claims

    def spy(*a, **k):
        seen.append(threading.current_thread() is threading.main_thread())
        return real(*a, **k)

    monkeypatch.setattr(pcc, "_validate_claims", spy)
    _audit(SUBJECT_SPANS[1])
    assert seen == [False]
