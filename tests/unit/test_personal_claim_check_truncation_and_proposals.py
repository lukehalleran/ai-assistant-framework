"""The personal-claim auditor reports truncation, and assistant proposals are not claims.

Incident 2026-10-17 (live, invented names). (A) An audit of a long reply came
back status=failed reason=invalid_json: completion tokens == max_tokens (900)
and the decode error sat at the very end of the text -- the JSON was cut off.
(B) A reply ending "when you're back we can hit the recall questions
rapid-fire and save the Arena builds for after" was audited as a `plan` claim,
flagged insufficient, and the read-time marker then quoted it into later
prompts as an unsupported user claim.
Classes: BC-89 (small-budget LLM call), BC-84 (personal-claim boundary precision).

Drives THE deployed ``audit_personal_claims`` / ``handlers._apply_personal_claim_check``
with scripted (misbehaving) auditor fakes.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

import config.app_config as config
from core.personal_claim_check import audit_personal_claims, build_personal_evidence
from gui import handlers
from utils.personal_claim_provenance import clean_personal_claim_receipt

QUERY = "Almost about to have breakfast first"
SPAN = "when you're back we can hit the recall questions rapid-fire and save the Arena builds for after"
DRAFT = ("Fair enough — fuel first, then homework. Either way, " + SPAN + ".")


class _Model:
    def __init__(self, raw):
        self.raw = raw
        self.prompts = []
        self.kwargs = []

    async def generate_once(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        return self.raw if isinstance(self.raw, str) else json.dumps(self.raw)


def _wrap(text, wrapped):
    return text.replace(" ", "\n    ", 3) if wrapped else text


def _evidence():
    return build_personal_evidence(QUERY, {})


def _audit(draft, raw):
    return asyncio.run(audit_personal_claims(draft, _evidence(), _Model(raw)))


def _claim(span, status="insufficient", kind="plan"):
    return {"claims": [{"text": span, "status": status, "kind": kind, "evidence": []}]}


# ---- Incident A: truncation ----

def test_json_cut_mid_object_reports_output_truncated():
    full = json.dumps(_claim("Fair enough."))
    result = _audit(DRAFT, full[: len(full) - 9])
    assert result.status == "failed" and result.reason == "output_truncated"
    assert result.receipt()["reason"] == "output_truncated"


def test_prose_output_is_still_invalid_json():
    result = _audit(DRAFT, "Sure, here is my review of the draft.")
    assert result.status == "failed" and result.reason == "invalid_json"


def test_mid_text_syntax_error_is_not_called_truncation():
    result = _audit(DRAFT, '{"claims": [}, {"a": 1}] and then a lot more trailing text after the error')
    assert result.status == "failed" and result.reason == "invalid_json"


def test_truncated_reason_survives_receipt_cleaning():
    receipt = _audit(DRAFT, '{"claims":[{"text":"x"').receipt()
    assert clean_personal_claim_receipt(receipt)["reason"] == "output_truncated"


def test_shipped_yaml_budget_reaches_the_model_call(monkeypatch):
    from config.app_config import load_yaml_config
    cfg = load_yaml_config("config.yaml")["personal_claim_check"]
    assert cfg["max_tokens"] == 1600 and cfg["timeout_s"] == 18.0
    assert config.PERSONAL_CLAIM_MAX_TOKENS == 1600
    monkeypatch.setattr(config, "PERSONAL_CLAIM_CHECK_ENABLED", True)
    model = _Model({"claims": []})
    ctx = SimpleNamespace(
        user_text=QUERY, telemetry={}, debug_record={},
        orchestrator=SimpleNamespace(model_manager=model),
        raw_context={}, history=(), personal_claim_mode="log_only",
        personal_claim_pending=None, personal_claim_task=None, personal_claim_receipt=None,
        turn_attachments=[],
    )
    asyncio.run(handlers._apply_personal_claim_check(ctx, DRAFT, mode="log_only"))
    assert model.kwargs[0]["max_tokens"] == config.PERSONAL_CLAIM_MAX_TOKENS


def test_prompt_carries_length_rule_and_kind_list():
    from core.personal_claim_check import _SYSTEM_PROMPT
    assert "at most 8 claims" in _SYSTEM_PROMPT and "25 words" in _SYSTEM_PROMPT
    assert '"suggestion" (something the ASSISTANT proposes' in _SYSTEM_PROMPT


# ---- Incident B: assistant proposals ----

@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("kind", ["plan", "suggestion", "other"])
def test_assistant_proposed_next_step_is_dropped_and_counted(wrapped, kind):
    draft = DRAFT.replace(SPAN, _wrap(SPAN, wrapped))
    result = _audit(draft, _claim(_wrap(SPAN, wrapped), kind=kind))
    assert result.claims == [] and result.proposal_drops == 1
    assert result.receipt()["proposal_drops"] == 1
    assert result.receipt()["flagged_spans"] == []
    assert result.dropped_claim_count == 0


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("span", [
    "Let's start with the recall questions.",
    "I'll send the outline once you're back.",
    "We could split the set in two.",
])
def test_other_first_person_proposals_are_dropped(span, wrapped):
    draft = "Okay. " + _wrap(span, wrapped) + " Sound good?"
    result = _audit(draft, _claim(_wrap(span, wrapped)))
    assert result.proposal_drops == 1 and result.claims == []


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("span,kind", [
    ("You finished the report.", "completion"),
    ("You said you'd start at 3.", "plan"),
    ("We talked about it yesterday.", "discussion"),
    ("I saw you uploaded it.", "completion"),
    ("We couldn't find the file.", "other"),
    # Referee (Opus): a proposal LATER in the span still carries a user claim.
    ("Since you uploaded the resume, I can tailor the cover letter.", "completion"),
    ("You finished the report, so we can move on.", "completion"),
    # "would"/"'d" is an assessment hedge, not an offer.
    ("I'd say you finished the hard part.", "completion"),
])
def test_user_claims_and_non_modal_clauses_are_kept(span, kind, wrapped):
    draft = "Okay. " + _wrap(span, wrapped) + " Moving on."
    result = _audit(draft, _claim(span if not wrapped else _wrap(span, wrapped), kind=kind))
    assert len(result.claims) == 1 and result.proposal_drops == 0
    assert result.claims[0]["status"] == "insufficient"


def test_proposal_drops_reach_turn_telemetry_and_debug_record(monkeypatch):
    monkeypatch.setattr(config, "PERSONAL_CLAIM_CHECK_ENABLED", True)
    model = _Model(_claim(SPAN))
    ctx = SimpleNamespace(
        user_text=QUERY, telemetry={}, debug_record={},
        orchestrator=SimpleNamespace(model_manager=model),
        raw_context={}, history=(), personal_claim_mode="log_only",
        personal_claim_pending=None, personal_claim_task=None, personal_claim_receipt=None,
        turn_attachments=[],
    )
    asyncio.run(handlers._apply_personal_claim_check(ctx, DRAFT, mode="log_only"))
    assert ctx.telemetry["personal_claim_proposal_drops"] == 1
    assert ctx.debug_record["personal_claim_proposal_drops"] == 1
    assert clean_personal_claim_receipt(ctx.personal_claim_receipt)["proposal_drops"] == 1


def test_json_cut_after_a_complete_token_reports_output_truncated():
    full = json.dumps(_claim("Fair enough."))
    result = _audit(DRAFT, full[: full.index('"status"') + 8])
    assert result.reason == "output_truncated"
