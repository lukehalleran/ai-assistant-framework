"""Findings of the log-only checkers are persisted for precision labelling.

Lane MEAS2, 2026-10-25. class: BC-58 (unmeasured guard: a guard that records
nothing it found can never be measured or promoted). Drives the DEPLOYED
checker entries (review gate in handle_submit, verify_grounding,
audit_personal_claims) with fakes; invented values only.
"""
from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core import grounding_check as gc
from core import personal_claim_check as pcc
from utils import checker_findings as cf
from utils import log_rotation

QUERY = "Did I send the quasar report to Rowan on Friday?"
USER_ROW = {"role": "user", "timestamp": "2026-10-16T20:00:00", "text": "I sent the quasar report to Rowan tonight."}
SPAN = "You sent the quasar report to Rowan on Friday."


def _lines():
    path = cf.findings_path()
    if not os.path.isfile(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


@pytest.fixture(autouse=True)
def _isolated_findings_file(monkeypatch, tmp_path):
    """Each test writes/reads its OWN findings file (no cross-test or run-order leakage)."""
    target = str(tmp_path / "findings" / cf.FILENAME)
    monkeypatch.setattr(cf, "findings_path", lambda: target)


@pytest.fixture(autouse=True)
def _name(monkeypatch):
    monkeypatch.setattr(pcc, "get_user_display_name", lambda: "Tamsin", raising=False)


class _PCModel:
    def __init__(self, status="insufficient"):
        self.status = status

    async def generate_once(self, prompt, **kw):
        return json.dumps({"claims": [{"text": SPAN, "status": self.status, "kind": "completion", "evidence": []}]})


def _audit(draft, query=QUERY, status="insufficient"):
    ev = pcc.build_personal_evidence(query, {})
    return asyncio.run(pcc.audit_personal_claims(draft, ev, _PCModel(status)))


@pytest.mark.parametrize("wrapped", [False, True])
def test_personal_claim_finding_written_with_bounded_fields(wrapped):
    reply = ("Okay. " + ("filler words. " * 80)) + (f"  {SPAN}\n    " if wrapped else SPAN) + " Moving on."
    _audit(reply)
    rows = [r for r in _lines() if r["checker"] == "personal_claim"]
    assert len(rows) == 1
    row = rows[0]
    assert row["verdict"] == "insufficient" and row["claim_span"].strip() == SPAN
    assert len(row["claim_span"]) <= cf.SPAN_MAX and len(row["reply_excerpt"]) <= cf.EXCERPT_MAX
    assert "quasar report" in row["reply_excerpt"] and row["evidence_excerpt"]


def test_personal_claim_probe_excluded():
    _audit("Fine. " + SPAN, query="[test] " + QUERY + " [/test]")
    assert [r for r in _lines() if r["checker"] == "personal_claim"] == []


class _GModel:
    async def generate_once(self, prompt, **kw):
        return json.dumps({"false_claim_present": True, "claim": "The report went out on Friday",
                           "why_false": "source says Thursday", "confidence": 0.9,
                           "correction": "It went out on Thursday."})


@pytest.mark.parametrize("wrapped", [False, True])
def test_grounding_finding_written_with_claim_and_source(wrapped):
    reply = "Sure. " + ("pad " * 400) + ("\n   The report went out on Friday.\n" if wrapped else "The report went out on Friday.")
    verdict = asyncio.run(gc.verify_grounding("When did it go out?", reply, _GModel(),
                                              source_material="SOURCE: " + "z" * 2000))
    assert verdict is not None and verdict.false_claim_present
    row = [r for r in _lines() if r["checker"] == "grounding"][0]
    assert row["verdict"] == "flagged" and row["claim_span"] == "The report went out on Friday"
    assert row["evidence_excerpt"].startswith("SOURCE:") and len(row["evidence_excerpt"]) <= cf.EXCERPT_MAX
    assert "Friday" in row["reply_excerpt"] and len(row["reply_excerpt"]) <= cf.EXCERPT_MAX


def test_grounding_probe_excluded():
    asyncio.run(gc.verify_grounding("[test] when? [/test]", "The report went out on Friday.", _GModel()))
    assert [r for r in _lines() if r["checker"] == "grounding"] == []


def test_review_finding_written_by_handle_submit():
    from tests.unit import test_handle_submit as hs
    resp = "The cat has some fur. " * 8
    orch = hs._make_orchestrator(agentic_enabled=True, streaming_chunks=[resp])
    planner = MagicMock()
    rv = MagicMock(passes=False, confidence=0.95, issues=["missing detail", "x" * 900])
    planner.review_answer = AsyncMock(return_value=rv)
    orch.response_planner = planner
    orch._current_response_plan = MagicMock()
    extra = [
        patch("config.app_config.RESPONSE_REVIEW_ENABLED", True),
        patch("config.app_config.RESPONSE_REVIEW_CONFIDENCE_THRESHOLD", 0.9),
        patch("config.app_config.UNCERTAINTY_FALLBACK_ENABLED", False),
        patch("utils.web_search_trigger.analyze_for_web_search_llm", new_callable=AsyncMock,
              return_value=hs._no_trigger_decision()),
        patch("gui.handlers._silent_agentic_retry", AsyncMock()),
    ]
    asyncio.run(hs._run_submit(hs.TestReviewGate._QUERY if hasattr(hs, "TestReviewGate") else
                               "Can you describe the color of my cat's fur and general appearance in some detail?",
                               orch, extra_patches=extra))
    rows = [r for r in _lines() if r["checker"] == "review"]
    assert len(rows) == 1 and rows[0]["verdict"] == "fail" and rows[0]["mode"] == "log_only"
    assert "missing detail" in rows[0]["claim_span"] and len(rows[0]["claim_span"]) <= cf.SPAN_MAX
    assert "cat has some fur" in rows[0]["reply_excerpt"]


def test_disabled_flag_writes_nothing(monkeypatch):
    import config.app_config as ac
    monkeypatch.setattr(ac, "TURN_TELEMETRY_CFG", {"checker_findings_enabled": False}, raising=False)
    assert cf.record_finding("review", verdict="fail", claim_span="x") is False
    _audit("Fine. " + SPAN)
    assert _lines() == []
    monkeypatch.setattr(ac, "TURN_TELEMETRY_CFG", {"checker_findings_enabled": True}, raising=False)
    assert cf.record_finding("review", verdict="fail", claim_span="x") is True  # same process, flag read live


def test_write_failure_does_not_raise(monkeypatch):
    monkeypatch.setattr(cf.os, "makedirs", MagicMock(side_effect=OSError("disk full")))
    monkeypatch.setattr(cf, "findings_path", lambda: "/nonexistent_dir_xyz/checker_findings.jsonl")  # overrides the per-test path
    assert cf.record_finding("grounding", verdict="flagged") is False
    res = _audit("Fine. " + SPAN)  # the deployed checker still returns its result
    assert res.status == "checked"


def test_rotation_entry_present(tmp_path, monkeypatch):
    import config.app_config as ac
    monkeypatch.setattr(ac, "TURN_TELEMETRY_PATH", str(tmp_path / "turn_records.jsonl"), raising=False)
    monkeypatch.setattr(ac, "LOG_MAINTENANCE_TURN_RECORDS_MAX_MB", 0.0001, raising=False)
    monkeypatch.setattr(ac, "LOG_MAINTENANCE_ENABLED", True, raising=False)
    big = tmp_path / "checker_findings.jsonl"
    big.write_text("x" * 5000)
    log_rotation.run_startup_log_maintenance()
    assert (tmp_path / "checker_findings.jsonl.1").exists() and not big.exists()


def test_no_share_surface_reads_findings_file():
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    surfaces = ["scripts/diagnostic_rollup.py", "api/routes/debug.py", "utils/privacy_redaction.py"]
    for rel in surfaces:
        assert "checker_findings" not in open(os.path.join(root, rel), encoding="utf-8").read(), rel
    readers = []
    for d in ("api", "core", "gui", "utils", "memory", "knowledge"):
        for dirpath, _dn, files in os.walk(os.path.join(root, d)):
            for f in files:
                if f.endswith(".py") and "checker_findings" in open(os.path.join(dirpath, f), encoding="utf-8").read():
                    readers.append(os.path.relpath(os.path.join(dirpath, f), root))
    assert sorted(readers) == ["core/grounding_check.py", "core/personal_claim_check.py",
                               "gui/handlers.py", "utils/checker_findings.py", "utils/log_rotation.py"]
