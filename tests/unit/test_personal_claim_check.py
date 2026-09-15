"""Focused contract tests for the personal-claim evidence boundary.

Scripted providers exercise transport and source-span validation.  They do not
measure semantic model accuracy.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from core.personal_claim_check import (
    PersonalClaimResult,
    audit_personal_claims,
    build_personal_evidence,
    omit_unsupported_claims,
)


QUERY = "No idea how it's almost 5, feel like I've thunk a total of like 3 thoughts today"
REPLY = (
    "That's the sleep-deprivation time warp — 6.5 hours on a 3am crash doesn't feel like a day, "
    "it feels like a blur with occasional horizontal breaks. And honestly, you did think more "
    "than 3 thoughts: you reworked the resume enough to call it done, uploaded it, and had a "
    "whole tripwire-framework conversation. Fried brains just don't log those as \"thinking.\"\n\n"
    "Low bar for the rest of the evening. Nothing left today that can't survive until tomorrow."
)


class ScriptedModel:
    def __init__(self, payload=None, *, error=None, delay=0):
        self.payload = payload
        self.error = error
        self.delay = delay
        self.calls = []

    async def generate_once(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return json.dumps(self.payload, ensure_ascii=False)


def _evidence():
    return build_personal_evidence(
        QUERY,
        {
            "recent_conversations": [
                {
                    "id": "turn-user",
                    "role": "user",
                    "timestamp": "2026-09-15T16:40:00-05:00",
                    "text": "I could just upload what I have to get something up, but it needs rework.",
                },
                {
                    "id": "turn-assistant",
                    "role": "assistant",
                    "timestamp": "2026-09-15T16:41:00-05:00",
                    "text": "You could upload it as-is; then the one task is upload-as-is.",
                },
                {
                    "id": "turn-correction",
                    "role": "user",
                    "timestamp": "2026-09-15T16:42:00-05:00",
                    "text": "Actually, I did not upload it; I cancelled that plan.",
                },
            ]
        },
    )


def _ref(evidence, needle):
    row = next(row for row in evidence if needle in row["text"])
    return {"source_id": row["source_id"], "quote": needle}


def test_build_preserves_roles_timestamps_and_latest_correction_with_opaque_ids():
    evidence = _evidence()
    assert evidence[0] == {"source_id": "src_current_query", "role": "user", "timestamp": None, "text": QUERY}
    assert [row["role"] for row in evidence[1:]] == ["user", "assistant", "user"]
    correction = next(row for row in evidence if row["text"].startswith("Actually"))
    assert correction["timestamp"] == "2026-09-15T16:42:00-05:00"
    assert all("turn-" not in row["source_id"] for row in evidence)


def test_explicit_empty_user_text_does_not_fallback_to_embedded_or_generic_text():
    evidence = build_personal_evidence(
        "current",
        {
            "recent_conversations": [
                {"id": "empty", "user_text": "", "text": "I completed the task", "assistant_text": "I suggested doing it."},
                {"id": "wrapped", "role": "assistant", "text": "User: I completed the task\nAssistant: Great."},
            ]
        },
    )
    assert not any("I completed the task" == row["text"] and row["role"] == "user" for row in evidence)
    assert any(row["role"] == "assistant" and row["text"] == "I suggested doing it." for row in evidence)
    assert sum("User:" in row["text"] for row in evidence) == 1
    assert all(row["role"] != "user" for row in evidence[1:])


@pytest.mark.asyncio
async def test_exact_audit_fixture_works_for_clean_and_wrapped_reply():
    for reply in (REPLY, "\n  " + REPLY.replace("\n\n", "\n  \n  ")):
        evidence = _evidence()
        payload = {
            "claims": [
                {
                    "text": "you reworked the resume enough to call it done",
                    "status": "insufficient",
                    "kind": "personal_completion",
                    "evidence": [_ref(evidence, "I could just upload")],
                },
                {
                    "text": "uploaded it",
                    "status": "contradicted",
                    "kind": "personal_completion",
                    "evidence": [_ref(evidence, "Actually, I did not upload it")],
                },
                {
                    "text": "had a whole tripwire-framework conversation",
                    "status": "supported",
                    "kind": "discussion",
                    "evidence": [_ref(evidence, "You could upload it as-is")],
                },
            ]
        }
        model = ScriptedModel(payload)
        result = await audit_personal_claims(reply, evidence, model, model_name="scripted")
        assert result.status == "checked"
        assert [claim["status"] for claim in result.claims] == ["insufficient", "contradicted", "supported"]
        assert result.receipt()["candidate_count"] == 3
        assert "I could just upload" not in json.dumps(result.receipt())


@pytest.mark.asyncio
async def test_assistant_advice_cannot_support_personal_completion():
    evidence = _evidence()
    assistant_ref = _ref(evidence, "You could upload it as-is")
    model = ScriptedModel(
        {"claims": [{"text": "uploaded it", "status": "supported", "kind": "completion", "evidence": [assistant_ref]}]}
    )
    result = await audit_personal_claims("You uploaded it.", evidence, model)
    assert result.status == "checked"
    assert result.claims[0]["status"] == "insufficient"


@pytest.mark.asyncio
async def test_no_claims_is_a_checked_clean_result():
    model = ScriptedModel({"claims": []})
    result = await audit_personal_claims("That sounds frustrating.", _evidence(), model)
    assert result.status == "checked"
    assert result.claims == []
    assert result.receipt()["candidate_count"] == 0


@pytest.mark.asyncio
async def test_unparseable_output_is_invalid_json_and_preserves_text():
    evidence = _evidence()
    bad_json = ScriptedModel({"claims": [{"text": "uploaded it"}]})
    invalid = await audit_personal_claims("You uploaded it.", evidence, bad_json)
    assert invalid.status == "failed"
    assert invalid.reason == "invalid_verdict"  # parsed, but no usable claim survived
    assert invalid.dropped_claim_count == 1
    assert omit_unsupported_claims("You uploaded it.", invalid) == "You uploaded it."

    class Fenced(ScriptedModel):
        async def generate_once(self, prompt, **kwargs):
            return "```json\n{\"claims\": []}\n```"

    fenced = await audit_personal_claims("You uploaded it.", evidence, Fenced())
    assert (fenced.status, fenced.reason) == ("failed", "invalid_json")


@pytest.mark.asyncio
async def test_bad_reference_is_dropped_not_fatal_and_only_demotes():
    """BC-84: one inexact quote must not discard the whole audit. A dropped
    reference can only move a claim toward insufficient."""
    evidence = _evidence()
    real_source = _ref(evidence, "I could just upload")["source_id"]
    inexact = {"source_id": real_source, "quote": "not present"}
    unknown = {"source_id": "src_nope", "quote": "uploaded"}
    model = ScriptedModel({"claims": [
        # contradicted needs no evidence: survives with its bad refs dropped
        {"text": "uploaded it", "status": "contradicted", "kind": "completion", "evidence": [inexact, unknown]},
        # supported with ONLY bad evidence: demoted, never kept as supported
        {"text": "You", "status": "supported", "kind": "discussion", "evidence": [inexact]},
    ]})
    result = await audit_personal_claims("You uploaded it.", evidence, model)
    assert result.status == "checked" and result.reason == "ok"
    assert [c["status"] for c in result.claims] == ["contradicted", "insufficient"]
    assert result.claims[0]["evidence"] == []
    assert result.dropped_evidence_count == 3
    assert result.demoted_count == 1
    receipt = result.receipt()
    assert receipt["dropped_evidence_count"] == 3 and receipt["demoted_count"] == 1
    assert receipt["source_ids"] == []


@pytest.mark.asyncio
async def test_non_span_claim_is_dropped_while_the_rest_survive():
    evidence = _evidence()
    model = ScriptedModel({"claims": [
        {"text": "you paraphrased this", "status": "insufficient", "kind": "other", "evidence": []},
        {"text": "uploaded it", "status": "insufficient", "kind": "completion", "evidence": []},
        {"text": "uploaded it", "status": "insufficient", "kind": "completion", "evidence": []},  # duplicate
        {"text": "uploaded it", "status": "maybe", "kind": "completion"},  # bad shape
    ]})
    result = await audit_personal_claims("You uploaded it.", evidence, model)
    assert result.status == "checked"
    assert [c["text"] for c in result.claims] == ["uploaded it"]
    assert result.dropped_claim_count == 3
    assert omit_unsupported_claims("You uploaded it.", result) == "I don't have enough context to verify those personal details."


@pytest.mark.asyncio
async def test_timeout_is_explicit_and_fail_open():
    model = ScriptedModel({"claims": []}, delay=0.05)
    result = await audit_personal_claims("You paid the bill.", _evidence(), model, timeout_s=0.001)
    assert result.status == "unavailable"
    assert result.reason == "timeout"


def test_omission_removes_whole_unsupported_sentences_and_preserves_supported_text():
    response = "You finished the repair. You might have sent the message. The appointment was discussed."
    result = PersonalClaimResult(
        "checked",
        "ok",
        claims=[
            {"text": "finished the repair", "status": "supported", "kind": "completion", "evidence": []},
            {"text": "might have sent the message", "status": "insufficient", "kind": "plan", "evidence": []},
            {"text": "appointment was discussed", "status": "contradicted", "kind": "completion", "evidence": []},
        ],
    )
    assert omit_unsupported_claims(response, result) == "You finished the repair."


def test_omission_all_removed_uses_neutral_fallback_and_failures_preserve_text():
    response = "You completed the repair."
    result = PersonalClaimResult(
        "checked", "ok",
        claims=[{"text": "completed the repair", "status": "insufficient", "kind": "completion", "evidence": []}],
    )
    assert omit_unsupported_claims(response, result) == "I don't have enough context to verify those personal details."
    failed = PersonalClaimResult("failed", "invalid_json")
    assert omit_unsupported_claims(response, failed) == response
