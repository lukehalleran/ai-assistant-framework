"""A "contradicted" verdict needs a surviving exact evidence quote, like "supported".

Incident 2026-10-18 14:04 (live [test] recall probe): the auditor marked all 5
claims of a recap "completion:contradicted"; 3 evidence quotes were dropped as
not exact, yet 0 claims were demoted, because only "supported" demoted when its
evidence was gone. A replay of the same draft returned all 5 "supported". The
flag is the same either way (both statuses are unsupported); the receipt's
contradicted/insufficient counts are what this fixes.

Class: BC-84 (personal-claim auditor precision). Drives THE deployed
audit_personal_claims with scripted auditor fakes; invented values only.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from core import personal_claim_check as pcc
from core.personal_claim_check import audit_personal_claims, build_personal_evidence

QUERY = "Do you remember what we did about the quasar report on Friday?"
USER_ROW = {"role": "user", "timestamp": "2026-10-16T20:00:00", "text": "I sent the quasar report to Rowan tonight."}
SPAN = "You sent the quasar report to Rowan on Friday."


class _Model:
    def __init__(self, raw):
        self.raw = raw

    async def generate_once(self, prompt, **kwargs):
        return json.dumps(self.raw)


@pytest.fixture(autouse=True)
def _display_name(monkeypatch):
    monkeypatch.setattr(pcc, "get_user_display_name", lambda: "Tamsin", raising=False)


def _audit(status, quote, wrapped=False):
    span = SPAN.replace(" ", "\n    ", 2) if wrapped else SPAN
    draft = "Yes. " + span + " Anything else?"
    evidence = build_personal_evidence(QUERY, {"recent_conversations": [USER_ROW]})
    src = next(r["source_id"] for r in evidence if r["role"] == "user" and r["text"] == USER_ROW["text"])
    raw = {"claims": [{"text": span, "status": status, "kind": "completion",
                       "evidence": [{"source_id": src, "quote": quote}]}]}
    return asyncio.run(audit_personal_claims(draft, evidence, _Model(raw)))


@pytest.mark.parametrize("wrapped", [False, True])
def test_contradicted_with_inexact_quote_demotes_to_insufficient(wrapped):
    res = _audit("contradicted", "I never sent any quasar report", wrapped)
    assert [c["status"] for c in res.claims] == ["insufficient"]
    assert res.demoted_count == 1
    assert res.receipt()["contradicted_count"] == 0


@pytest.mark.parametrize("wrapped", [False, True])
def test_contradicted_with_exact_quote_stays_contradicted(wrapped):
    res = _audit("contradicted", "I sent the quasar report to Rowan tonight", wrapped)
    assert [c["status"] for c in res.claims] == ["contradicted"]
    assert res.demoted_count == 0


def test_supported_with_inexact_quote_still_demotes():
    res = _audit("supported", "I mailed a quasar report")
    assert [c["status"] for c in res.claims] == ["insufficient"]
