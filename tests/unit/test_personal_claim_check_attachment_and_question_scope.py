"""The personal-claim auditor sees attachments, and questions are never claims.

Incident 2026-10-17 (live, three turns after #83; invented details): 19 of 20
audited claims came back `insufficient`. Flagged spans included a question
("What's 6 covering?") and descriptions of a homework PDF the user attached
that turn ("It's 20 questions, all 1 point each"): the attachment text never
reached the evidence, so every claim about the document was unsupported.
Classes: BC-84 (personal-claim boundary precision), BC-15 (attachment content
never reached a consumer).

Drives THE deployed ``build_personal_evidence`` / ``audit_personal_claims`` /
``handlers._apply_personal_claim_check`` with scripted auditor fakes.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

import config.app_config as config
from core.personal_claim_check import audit_personal_claims, build_personal_evidence
from gui import handlers

DOC_NAME = "rowan_hw6.pdf"
DOC_TEXT = "Homework 6. There are 20 questions, all 1 point each. Question 3 covers sampling."
QUERY = "Here's the file, probably starting around 230 to 3."


class _Model:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    async def generate_once(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return json.dumps(self.payload)


def _wrap(text, wrapped):
    return text.replace(" ", "\n    ", 3) if wrapped else text


def _evidence(**kw):
    return build_personal_evidence(QUERY, {}, attachments=[(DOC_NAME, DOC_TEXT)], **kw)


def _audit(draft, evidence, claim):
    return asyncio.run(audit_personal_claims(draft, evidence, _Model({"claims": [claim]})))


def test_attachment_row_follows_current_query_with_role_attachment():
    ev = _evidence()
    assert ev[0]["source_id"] == "src_current_query"
    assert ev[1]["role"] == "attachment" and ev[1]["source_id"] == "src_attachment_1"
    assert ev[1]["timestamp"] is None and ev[1]["text"] == DOC_TEXT


def test_attachments_bounded_to_forty_percent_across_all_documents():
    big = "x" * 50000
    ev = build_personal_evidence(
        QUERY, {}, max_chars=1000, attachments=[("a.txt", big), ("b.txt", big)])
    rows = [r for r in ev if r["role"] == "attachment"]
    assert [r["source_id"] for r in rows] == ["src_attachment_1", "src_attachment_2"]
    assert sum(len(r["text"]) for r in rows) <= 400
    assert all(r["text"].endswith("[...truncated]") for r in rows)


def test_no_attachments_leaves_evidence_unchanged():
    assert build_personal_evidence(QUERY, {}) == build_personal_evidence(QUERY, {}, attachments=())


@pytest.mark.parametrize("wrapped", [False, True])
def test_document_content_claim_citing_attachment_stays_supported(wrapped):
    span = "It's 20 questions, all 1 point each."
    draft = "HW 6 is short. " + _wrap(span, wrapped) + " Good luck."
    claim_text = _wrap(span, wrapped)
    ev = _evidence()
    claim = {"text": claim_text, "status": "supported", "kind": "discussion",
             "evidence": [{"source_id": "src_attachment_1", "quote": "20 questions"}]}
    result = _audit(draft, ev, claim)
    assert result.claims and result.claims[0]["status"] == "supported"


@pytest.mark.parametrize("wrapped", [False, True])
def test_completion_claim_citing_only_attachment_is_insufficient(wrapped):
    span = "You finished question 3 already."
    draft = "Nice. " + _wrap(span, wrapped) + " Keep going."
    ev = _evidence()
    claim = {"text": _wrap(span, wrapped), "status": "supported", "kind": "personal_completion",
             "evidence": [{"source_id": "src_attachment_1", "quote": "Question 3"}]}
    result = _audit(draft, ev, claim)
    assert result.claims[0]["status"] == "insufficient" and result.demoted_count == 1


def test_other_kind_document_claim_citing_attachment_stays_supported():
    # Referee 2026-10-17: rule 1 (all-assistant evidence) is NOT widened to
    # attachments -- the auditor labels document descriptions "other" too, and
    # demoting them would re-create the over-flagging this lane fixes. Only
    # completion claims need user evidence (test above).
    span = "HW 6 looks pretty manageable."
    ev = _evidence()
    claim = {"text": span, "status": "supported", "kind": "other",
             "evidence": [{"source_id": "src_attachment_1", "quote": "Homework 6"}]}
    result = _audit("Okay. " + span + " Rest.", ev, claim)
    assert result.claims[0]["status"] == "supported"


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("tail", ["?", '?"', "?)", "? "])
def test_question_span_is_dropped_and_counted(wrapped, tail):
    span = "What's 6 covering" + tail
    draft = "Sure. " + _wrap(span, wrapped) + " Tell me more."
    ev = _evidence()
    claim = {"text": _wrap(span.strip(), wrapped), "status": "insufficient",
             "kind": "personal_state", "evidence": []}
    result = _audit(draft, ev, claim)
    assert result.claims == []
    assert result.question_drops == 1
    assert result.dropped_claim_count == 0
    assert result.receipt()["question_drops"] == 1


def test_non_question_claim_is_not_counted():
    span = "You started around three."
    claim = {"text": span, "status": "insufficient", "kind": "plan", "evidence": []}
    result = _audit("Okay. " + span + " Fine.", _evidence(), claim)
    assert len(result.claims) == 1 and result.question_drops == 0


def test_question_drops_survives_receipt_cleaning():
    from utils.personal_claim_provenance import clean_personal_claim_receipt
    receipt = _audit("Hm. What's 6 covering? Ok.", _evidence(), {
        "text": "What's 6 covering?", "status": "insufficient", "kind": "state",
        "evidence": []}).receipt()
    assert clean_personal_claim_receipt(receipt)["question_drops"] == 1


def test_system_prompt_names_attachment_rows_and_questions():
    from core.personal_claim_check import _SYSTEM_PROMPT
    assert '"attachment"' in _SYSTEM_PROMPT
    assert "not a claim" in _SYSTEM_PROMPT


async def test_handler_passes_turn_attachments_into_model_evidence(monkeypatch):
    monkeypatch.setattr(config, "PERSONAL_CLAIM_CHECK_ENABLED", True)
    model = _Model({"claims": []})
    ctx = SimpleNamespace(
        user_text=QUERY, telemetry={}, debug_record={},
        orchestrator=SimpleNamespace(model_manager=model),
        raw_context={}, history=(), personal_claim_mode="log_only",
        personal_claim_pending=None, personal_claim_task=None, personal_claim_receipt=None,
        turn_attachments=[{"name": DOC_NAME, "path": None, "extension": ".pdf", "text": DOC_TEXT}],
    )
    await handlers._apply_personal_claim_check(ctx, "HW 6 looks pretty manageable.", mode="log_only")
    evidence = json.loads(model.prompts[0].split("EVIDENCE:\n", 1)[1].split("\n\nDRAFT:\n", 1)[0])
    assert [r["role"] for r in evidence] == ["user", "attachment"]
    assert evidence[1]["text"] == DOC_TEXT


def test_truncated_evidence_is_reported_in_the_receipt():
    # Referee 2026-10-17: builder and detector now share TRUNCATION_MARKER.
    from core.personal_claim_check import TRUNCATION_MARKER
    ev = build_personal_evidence(QUERY, {}, attachments=[(DOC_NAME, DOC_TEXT + " filler" * 400)],
                                 max_chars=400)
    assert any(TRUNCATION_MARKER.strip() in r["text"] for r in ev)
    span = "It's 20 questions, all 1 point each."
    claim = {"text": span, "status": "supported", "kind": "discussion",
             "evidence": [{"source_id": "src_attachment_1", "quote": "20"}]}
    result = _audit("HW 6 is short. " + span, ev, claim)
    assert result.receipt().get("evidence_truncated") is True
