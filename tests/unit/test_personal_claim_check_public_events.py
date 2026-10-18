"""Public, organisational and software events are subject matter, not personal claims.

Incident 2026-10-18 00:37 ([test] web-search probe): a reply summarising web
sources about a software release got ``completion:insufficient`` x7 because the
auditor prompt defined "completion" as "the user or another person did
something", so a software project shipping a release read as "another person".
The receipt's flagged spans were then re-rendered into later prompts as "do not
repeat it as fact". Fix is in the prompt (who counts as another person); the
elided-subject backstop for completion kinds is deliberately unchanged.
Classes: BC-84 (personal-claim auditor precision), BC-75.

Drives THE deployed ``audit_personal_claims`` with scripted auditor fakes.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from core import personal_claim_check as pcc
from core.personal_claim_check import audit_personal_claims, build_personal_evidence

QUERY = "Search the web for what changed in the free-threaded build"

SPANS = [
    "Python 3.13 shipped an experimental build with the Global Interpreter Lock (GIL) disabled, per PEP 703.",
    "- It's a **separate executable**, not the default — installed as `python3.13t` alongside the regular build.",
]


class _Model:
    def __init__(self, raw):
        self.raw = raw

    async def generate_once(self, prompt, **kwargs):
        return json.dumps(self.raw)


def _wrap(text, wrapped):
    return text.replace(" ", "\n    ", 3) if wrapped else text


def _audit(span, kind, status="insufficient", wrapped=False):
    shown = _wrap(span, wrapped)
    draft = "Here is the summary. " + shown + " That is all."
    evidence = build_personal_evidence(QUERY, {})
    raw = {"claims": [{"text": shown, "status": status, "kind": kind, "evidence": []}]}
    return asyncio.run(audit_personal_claims(draft, evidence, _Model(raw)))


@pytest.fixture(autouse=True)
def _display_name(monkeypatch):
    monkeypatch.setattr(pcc, "get_user_display_name", lambda: "Tamsin", raising=False)


def test_prompt_scopes_another_person_to_the_users_life():
    p = " ".join(pcc._SYSTEM_PROMPT.split())
    assert "the user or another person did something" not in p
    assert "a person in the user's own life" in p
    for noun in ("organisations", "software and its releases", "public or historical events"):
        assert noun in p
    assert 'kind "subject"' in p


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("span", SPANS)
def test_subject_kind_public_event_is_not_flagged(span, wrapped):
    result = _audit(span, "subject", wrapped=wrapped)
    assert result.receipt()["flagged_spans"] == []
    assert result.dropped_claim_count == 0


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("span", SPANS)
def test_misbehaving_model_completion_kind_stays_flagged(span, wrapped):
    # Documents that the prompt, not a backstop, is the fix.
    result = _audit(span, "completion", wrapped=wrapped)
    assert [" ".join(f.split()) for f in result.receipt()["flagged_spans"]] == [span]


@pytest.mark.parametrize("wrapped", [False, True])
def test_elided_subject_user_completion_stays_flagged(wrapped):
    result = _audit("HW4 submitted.", "completion", wrapped=wrapped)
    assert [" ".join(f.split()) for f in result.receipt()["flagged_spans"]] == ["HW4 submitted."]
