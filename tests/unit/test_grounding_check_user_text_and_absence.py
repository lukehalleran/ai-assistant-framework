"""Grounding verifier: the user's own words and unverifiable absence claims (lane C1, wave 2).

Incident 2026-10-25 (oct26b M2 labelled sheet): 0/7 judgeable verifier flags were
correct. Rows 13/8 flagged the USER's own sentence quoted in the reply; rows A1/A2
flagged "the late policy isn't in this document" although the verifier never
received the document. Classes: BC-50, BC-84.

Drives THE deployed ``verify_grounding`` with a misbehaving fake verifier model.
"""
import asyncio
import json

import pytest

from core import grounding_check as gc

CLOCK = "[AUTHORITATIVE RUNTIME CLOCK]\nCurrent time: Sunday, 2026-10-25 19:10:30 CDT"


class _Model:
    def __init__(self, claim, why="The assistant is wrong.", correction="The real policy is different."):
        self.payload = {"false_claim_present": True, "claim": claim, "why_false": why,
                        "confidence": 0.9, "correction": correction}

    async def generate_once(self, prompt, **kw):
        return json.dumps(self.payload)


@pytest.fixture(autouse=True)
def _no_disk(monkeypatch):
    rows = []
    monkeypatch.setattr(gc.checker_findings, "record_finding",
                        lambda checker, **kw: rows.append(kw) or True)
    return rows


def _verify(claim, query, source="", response="reply text"):
    return asyncio.run(gc.verify_grounding(query, response, _Model(claim), source_material=source))


def _wrap(text):
    return text.replace(" ", "\n    ", 3)


USER_Q = "i think you lose ten percent per day up to three days, can you confirm? not sure"


@pytest.mark.parametrize("wrapped", [False, True])
def test_users_own_sentence_is_not_an_assistant_claim(wrapped, _no_disk):
    claim = "you lose ten percent per day i think up to three days"
    assert _verify(_wrap(claim) if wrapped else claim, USER_Q) is None
    assert _no_disk[-1]["verdict"] == "demoted:claim_user_stated"


def test_swapped_number_in_users_sentence_still_flags(_no_disk):
    verdict = _verify("you lose twenty percent per day up to three days", USER_Q)
    assert verdict is not None and _no_disk[-1]["verdict"] == "flagged"


def test_negation_flip_of_users_sentence_still_flags(_no_disk):
    # "not sure" in the user's message must not launder a claim that adds its own negation
    assert _verify("you do not lose ten percent per day up to three days", USER_Q) is not None


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("claim", [
    "the late penalty policy isn't in this document.",
    "the 10%/day policy is *not* in the syllabus you sent me",
])
def test_absence_claim_about_unreceived_document_abstains(claim, wrapped, _no_disk):
    q = "what is the late policy for the course syllabus?"
    assert _verify(_wrap(claim) if wrapped else claim, q, source=CLOCK) is None
    assert _no_disk[-1]["verdict"] == "demoted:absence_unverifiable"


def test_absence_claim_flags_when_document_text_is_in_evidence(_no_disk):
    doc = "COURSE SYLLABUS\n" + "Late work is accepted for three days with a ten percent penalty per day. " * 6
    verdict = _verify("the late penalty policy isn't in this document.", "what is the late policy?",
                      source=CLOCK + "\n\n" + doc)
    assert verdict is not None and _no_disk[-1]["verdict"] == "flagged"


def test_world_fact_negation_is_still_checked(_no_disk):
    verdict = _verify("Insulin isn't produced in the pancreas of healthy adults.", "tell me about insulin")
    assert verdict is not None and _no_disk[-1]["verdict"] == "flagged"


def test_real_wrong_factual_claim_still_flags(_no_disk):
    verdict = _verify("Vaccines cause autism in toddlers.", "is that true?", source=CLOCK)
    assert verdict is not None and _no_disk[-1]["verdict"] == "flagged"
