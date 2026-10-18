"""Subject-matter statements are not personal claims.

Incident 2026-10-17 (evening homework session): the personal-claim auditor
returned 52 claims over 7 audits, 43 insufficient, and 33 distinct spans were
re-rendered into later prompts as "not found in the user's own words; do not
repeat it as fact" -- almost all were explanations of course material.
Classes: BC-84 (personal-claim scope), BC-75.

Drives THE deployed ``audit_personal_claims`` with scripted auditor fakes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

import config.app_config as config
from core import personal_claim_check as pcc
from core.personal_claim_check import audit_personal_claims, build_personal_evidence
from gui import handlers
from utils.personal_claim_provenance import clean_personal_claim_receipt

QUERY = "Almost about to have breakfast first"


class _Model:
    def __init__(self, raw):
        self.raw = raw

    async def generate_once(self, prompt, **kwargs):
        return self.raw if isinstance(self.raw, str) else json.dumps(self.raw)


def _wrap(text, wrapped):
    return text.replace(" ", "\n    ", 3) if wrapped else text


def _claim(span, status="insufficient", kind="other"):
    return {"claims": [{"text": span, "status": status, "kind": kind, "evidence": []}]}


def _audit(span, kind="other", wrapped=False):
    draft = "Okay. " + _wrap(span, wrapped) + " Moving on."
    evidence = build_personal_evidence(QUERY, {})
    return asyncio.run(audit_personal_claims(draft, evidence, _Model(_claim(_wrap(span, wrapped), kind=kind))))


@pytest.fixture(autouse=True)
def _display_name(monkeypatch):
    monkeypatch.setattr(pcc, "get_user_display_name", lambda: "Tamsin", raising=False)


INCIDENT_SPANS = [
    "**Flowchart modules** (Create, Process, Dispose, Decide, Assign...) live on the model canvas as blocks connected by arrows.",
    "Create-Process-Dispose is the whole answer, and the transcripts back that up exactly.",
    'Seize-Delay-Release is *inside* the Process module (a dropdown option for the "type of action"), not a separate module in the sequence — that\'s why option a is the trap answer.',
    "HW 6 looks pretty manageable.",
]
# Contain a personal pronoun ("them", "you"): the closed pronoun rule keeps
# them (brief item 2), so only the prompt's "subject" kind covers these two.
QUEUE_SPAN = ("The queueing behavior is just what *emerges* when entities arrive faster than a resource can "
              "process them — it's an output of the model, not the thing Arena is built around.")
PRONOUN_SPAN = ("Simulation languages have been getting built since the late 1950s, and the cumulative list "
                "runs well into the hundreds once you include the classics (GPSS, SIMSCRIPT, GASP, SLAM, …).")


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("span", INCIDENT_SPANS)
def test_incident_subject_spans_are_dropped_not_flagged(span, wrapped):
    result = _audit(span, wrapped=wrapped)
    assert result.claims == [] and result.subject_drops == 1
    assert result.receipt()["flagged_spans"] == []
    assert result.receipt()["subject_drops"] == 1


@pytest.mark.parametrize("wrapped", [False, True])
def test_subject_kind_is_non_personal_and_not_counted_as_drop(wrapped):
    result = _audit(INCIDENT_SPANS[0], kind="subject", wrapped=wrapped)
    assert len(result.claims) == 1 and result.subject_drops == 0
    assert result.receipt()["flagged_spans"] == []


@pytest.mark.parametrize("wrapped", [False, True])
def test_supported_subject_claims_are_untouched(wrapped):
    span = "HW 6 looks pretty manageable."
    draft = "Okay. " + _wrap(span, wrapped) + " Rest."
    evidence = build_personal_evidence(QUERY, {})
    raw = {"claims": [{"text": _wrap(span, wrapped), "status": "supported", "kind": "other",
                       "evidence": [{"source_id": evidence[0]["source_id"], "quote": "breakfast first"}]}]}
    result = asyncio.run(audit_personal_claims(draft, evidence, _Model(raw)))
    assert result.subject_drops == 0 and len(result.claims) == 1


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("span,kind", [
    ("Nothing left today that can't wait.", "other"),
    ("Sim HW 4 is due the 25th.", "other"),
    ("The dentist moved to Thursday.", "other"),
    ("The appointment is at 3.", "plan"),
    ("He already sent the forms.", "other"),
    ("Tamsin's deadline moved up.", "other"),
    ("You're not underthinking it.", "other"),
])
def test_personal_or_timed_claims_stay_flagged(span, kind, wrapped):
    result = _audit(span, kind=kind, wrapped=wrapped)
    assert len(result.claims) == 1 and result.subject_drops == 0
    assert [" ".join(f.split()) for f in result.receipt()["flagged_spans"]] == [span]


@pytest.mark.parametrize("span", [PRONOUN_SPAN, QUEUE_SPAN])
def test_pronoun_spans_are_kept_by_the_closed_rule(span):
    result = _audit(span)
    assert result.subject_drops == 0 and len(result.claims) == 1


@pytest.mark.parametrize("wrapped", [False, True])
def test_documented_residual_bare_third_party_name(wrapped):
    # A name that is not the display name, no pronoun, no time: dropped when
    # kind is "other" (deliberate residual); kept when the kind is completion.
    assert _audit("Rowan has the forms.", wrapped=wrapped).subject_drops == 1
    kept = _audit("Rowan has the forms.", kind="completion", wrapped=wrapped)
    assert kept.subject_drops == 0 and len(kept.claims) == 1


@pytest.mark.parametrize("wrapped", [False, True])
def test_completion_without_person_or_time_is_kept(wrapped):
    result = _audit("Finished the report and sent it.", kind="completion", wrapped=wrapped)
    assert result.subject_drops == 0 and len(result.claims) == 1


def test_subject_drops_reach_receipt_telemetry_and_debug_record(monkeypatch):
    monkeypatch.setattr(config, "PERSONAL_CLAIM_CHECK_ENABLED", True)
    span = INCIDENT_SPANS[1]
    model = _Model(_claim(span))
    ctx = SimpleNamespace(
        user_text=QUERY, telemetry={}, debug_record={},
        orchestrator=SimpleNamespace(model_manager=model),
        raw_context={}, history=(), personal_claim_mode="log_only",
        personal_claim_pending=None, personal_claim_task=None, personal_claim_receipt=None,
        turn_attachments=[],
    )
    asyncio.run(handlers._apply_personal_claim_check(ctx, "Okay. " + span, mode="log_only"))
    assert ctx.telemetry["personal_claim_subject_drops"] == 1
    assert ctx.debug_record["personal_claim_subject_drops"] == 1
    assert clean_personal_claim_receipt(ctx.personal_claim_receipt)["subject_drops"] == 1


def test_prompt_declares_subject_kind():
    assert '"subject"' in pcc._SYSTEM_PROMPT
    assert "subject" in pcc._NON_PERSONAL_KINDS
    assert pcc._omission_eligible_kind("other") and not pcc._omission_eligible_kind("subject")


def test_kinds_debug_line_has_counts_and_no_span_text(caplog):
    span = "Tamsin's deadline moved up."
    with caplog.at_level(logging.DEBUG, logger=pcc.logger.name):
        _audit(span, kind="Other")
    lines = [r.getMessage() for r in caplog.records if "[PersonalClaim] kinds:" in r.getMessage()]
    assert len(lines) == 1
    assert json.loads(lines[0].split("kinds: ", 1)[1]) == {"other:insufficient": 1}
    assert "deadline" not in lines[0]
