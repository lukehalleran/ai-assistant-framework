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
    assert invalid.reason == "invalid_verdict"  # parsed, but the only claim broke the schema
    assert invalid.dropped_claim_count == 1
    assert omit_unsupported_claims("You uploaded it.", invalid) == "You uploaded it."

    class Fenced(ScriptedModel):
        async def generate_once(self, prompt, **kwargs):
            return "```json\n{\"claims\": []}\n```"

    # A surrounding code fence is presentation (live 2026-09-15 gpt-4o-mini shape).
    fenced = await audit_personal_claims("You uploaded it.", evidence, Fenced())
    assert (fenced.status, fenced.reason) == ("checked", "ok")

    class Prose(ScriptedModel):
        async def generate_once(self, prompt, **kwargs):
            return "The draft makes no personal claims. {\"claims\": []}"

    prose = await audit_personal_claims("You uploaded it.", evidence, Prose())
    assert (prose.status, prose.reason) == ("failed", "invalid_json")


def test_history_duplicates_of_recent_conversations_collapse_to_one_source():
    rows = [
        {"id": "c1", "timestamp": "2026-09-15T16:41:00-05:00",
         "query": "I could just upload what I have.", "response": "You could upload it as-is."},
    ]
    history = [
        {"role": "user", "content": "I could just upload what I have."},
        {"role": "assistant", "content": "You could upload it as-is."},
    ]
    evidence = build_personal_evidence("current", {"recent_conversations": rows}, history=history)
    texts = [(r["role"], r["text"]) for r in evidence[1:]]
    assert texts == [("user", "I could just upload what I have."), ("assistant", "You could upload it as-is.")]
    # the timestamped corpus copy is the one kept
    assert all(r["timestamp"] == "2026-09-15T16:41:00-05:00" for r in evidence[1:])


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
    # non-span + bad shape are dropped; the duplicate span is MERGED, not dropped
    assert result.dropped_claim_count == 2
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


# ---------------------------------------------------------------------------
# Paraphrased claims (live 2026-09-15 19:51: gpt-4o-mini restated every claim)
# ---------------------------------------------------------------------------

# Shape of the live 19:51 reply with the medication names genericized (the
# committed tree carries no personal vocabulary — config/privacy_terms.local.txt).
LIVE_REPLY = (
    "Fingers crossed it lines up — after today, an early night is exactly what the doctor ordered. "
    "One thing worth checking though: if\"meds\" means the morning stimulant, taking it this late works against "
    "the early bedtime. If it's the evening stuff (the usual two), then yeah, let it do its job "
    "and don't fight it with screens. Tomorrow's got the cover letter waiting for a fresher brain."
)


@pytest.mark.asyncio
async def test_paraphrased_claim_relocates_to_the_exact_draft_sentence():
    """The exact live model output for the 19:51 turn: three paraphrases, one
    of which maps onto a draft sentence; the two restatements of the USER's
    own message have no draft sentence and are dropped."""
    evidence = build_personal_evidence("Ya took meds just recently hoping I can go to bed early", {})
    model = ScriptedModel({"claims": [
        {"text": "User took meds just recently hoping to go to bed early.", "status": "supported",
         "kind": "completed action", "evidence": []},
        {"text": "User is checking if the meds are the stimulant or evening stuff.", "status": "insufficient",
         "kind": "plan/suggestion", "evidence": []},
        {"text": "User has a cover letter waiting for a fresher brain tomorrow.", "status": "supported",
         "kind": "future plan", "evidence": []},
    ]})
    result = await audit_personal_claims(LIVE_REPLY, evidence, model)
    assert result.status == "checked"
    assert [c["text"] for c in result.claims] == ["Tomorrow's got the cover letter waiting for a fresher brain."]
    assert result.claims[0]["status"] == "insufficient"  # supported with no evidence → demoted
    assert result.relocated_count == 1 and result.dropped_claim_count == 2 and result.demoted_count == 1
    assert omit_unsupported_claims(LIVE_REPLY, result).endswith("don't fight it with screens.")


@pytest.mark.asyncio
async def test_two_paraphrases_on_one_sentence_merge_in_the_conservative_direction():
    evidence = _evidence()
    model = ScriptedModel({"claims": [
        {"text": "The user has a cover letter waiting for tomorrow.", "status": "supported",
         "kind": "plan", "evidence": [_ref(evidence, "I could just upload")]},
        {"text": "The cover letter is waiting for a fresher brain tomorrow.", "status": "contradicted",
         "kind": "plan", "evidence": []},
    ]})
    result = await audit_personal_claims(LIVE_REPLY, evidence, model)
    assert len(result.claims) == 1
    assert result.claims[0]["status"] == "contradicted"
    assert result.relocated_count == 2


@pytest.mark.asyncio
async def test_ambiguous_paraphrase_is_dropped_not_guessed():
    reply = "You sent the email to your advisor today. You sent the email to your professor today."
    model = ScriptedModel({"claims": [
        {"text": "The user sent the email today.", "status": "supported", "kind": "completion", "evidence": []},
    ]})
    result = await audit_personal_claims(reply, _evidence(), model)
    # Well-formed but unlocatable: a CHECKED audit with no candidates, never a failure.
    assert result.status == "checked" and result.reason == "no_claims"
    assert result.claims == [] and result.dropped_claim_count == 1
    assert omit_unsupported_claims(reply, result) == reply


@pytest.mark.asyncio
async def test_restatement_of_the_user_message_is_a_checked_audit_with_no_claims():
    """Live 2026-09-15 20:29: the only claim was the user's own message restated;
    it has no draft sentence. Dropping it is right; calling the audit failed is not."""
    reply = "Looks like that came through as a test — same message as a few minutes ago. My answer stands."
    model = ScriptedModel({"claims": [
        {"text": "User took meds recently and hopes to go to bed early.", "status": "supported",
         "kind": "completed action", "evidence": [{"source_id": "src_current_query", "quote": "took meds"}]},
    ]})
    result = await audit_personal_claims(reply, build_personal_evidence("Ya took meds just recently hoping I can go to bed early", {}), model)
    assert (result.status, result.reason) == ("checked", "no_claims")
    assert result.dropped_claim_count == 1 and result.claims == []
    assert result.receipt()["candidate_count"] == 0
