"""Tests for core/insight/assessor.py + types.Assessment — fail-honest doctrine."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from core.insight.assessor import assess
from core.insight.types import Assessment, ClaimAssessment, EvidenceItem


def _mm(response):
    mm = MagicMock()
    mm.generate_once = AsyncMock(return_value=response)
    return mm


def _run(claims, response, evidence=None):
    return asyncio.run(assess(
        claims,
        evidence or [EvidenceItem(text="e", collection="corpus", speaker="user")],
        _mm(response),
    ))


class TestAssessor:
    def test_parse_failure_is_insufficient(self):
        result = _run(["my problem is avoidance"], "I think the user is right!")
        assert result.overall == "insufficient"

    def test_llm_exception_is_insufficient(self):
        mm = MagicMock()
        mm.generate_once = AsyncMock(side_effect=RuntimeError("api down"))
        result = asyncio.run(assess(["c"], [], mm))
        assert result.overall == "insufficient"

    def test_empty_response_is_insufficient(self):
        assert _run(["c"], "").overall == "insufficient"
        assert _run(["c"], None).overall == "insufficient"

    def test_unknown_verdict_coerces_to_insufficient(self):
        payload = json.dumps({"claims": [
            {"claim": "c", "verdict": "totally-right", "support": [], "refute": []},
        ]})
        result = _run(["c"], payload)
        assert result.overall == "insufficient"

    def test_refute_heavy_disagree(self):
        payload = json.dumps({"claims": [
            {"claim": "c1", "verdict": "agree", "support": ["E1"], "refute": []},
            {"claim": "c2", "verdict": "disagree", "support": [], "refute": ["E2", "E3"]},
        ]})
        result = _run(["c1", "c2"], payload)
        assert result.overall == "disagree"  # worst-of

    def test_agree_when_all_agree(self):
        payload = json.dumps({"claims": [
            {"claim": "c", "verdict": "agree", "support": ["E1", "E2"], "refute": [],
             "notes": "consistent across months"},
        ]})
        result = _run(["c"], payload)
        assert result.overall == "agree"
        assert result.claims[0].support == ["E1", "E2"]

    def test_no_claims_is_insufficient(self):
        result = asyncio.run(assess([], [], _mm("{}")))
        assert result.overall == "insufficient"

    def test_json_with_surrounding_prose_still_parses(self):
        payload = ("Here is my assessment:\n"
                   + json.dumps({"claims": [{"claim": "c", "verdict": "partial"}]})
                   + "\nHope that helps.")
        assert _run(["c"], payload).overall == "partial"

    def test_adversarial_prompt_contains_evidence_and_caveat(self):
        mm = _mm(json.dumps({"claims": [{"claim": "c", "verdict": "agree"}]}))
        asyncio.run(assess(
            ["c"],
            [EvidenceItem(text="she wasn't abusive", collection="corpus",
                          speaker="user", date="2026-08-18")],
            mm,
        ))
        prompt = mm.generate_once.call_args.args[0]
        assert "over-samples" in prompt
        assert "she wasn't abusive" in prompt
        system = mm.generate_once.call_args.kwargs["system_prompt"]
        assert "refuting" in system.lower()


class TestDocumentGating:
    def test_doc_only_on_agree_or_partial(self):
        assert Assessment(overall="agree").allows_document
        assert Assessment(overall="partial").allows_document
        assert not Assessment(overall="insufficient").allows_document
        assert not Assessment(overall="disagree").allows_document

    def test_worst_of_ordering(self):
        def mk(*verdicts):
            return Assessment.from_claims(
                [ClaimAssessment(claim="c", verdict=v) for v in verdicts]
            ).overall

        assert mk("agree", "agree") == "agree"
        assert mk("agree", "partial") == "partial"
        assert mk("agree", "insufficient", "partial") == "insufficient"
        assert mk("agree", "disagree", "partial") == "disagree"
        assert Assessment.from_claims([]).overall == "insufficient"
