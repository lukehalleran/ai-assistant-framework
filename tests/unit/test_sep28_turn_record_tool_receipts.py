"""Turn records carry tool-call receipts (2026-09-28, BC-72).

The revoked Gmail token (09-23..09-27) was visible only in daemon_debug.log;
turn_records.jsonl had modes but no tool names/outcomes. Each record now has
`tool_calls: [{tool, status, detail}]` (always present) and, when email
coverage reported failures, `providers_failed`.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import config.app_config as app_config
from core.agentic.controller import AgenticSearchController
from core.agentic.types import AgenticSearchSession, SearchDecision
from utils import turn_telemetry


@pytest.fixture
def rec_path(tmp_path, monkeypatch):
    p = tmp_path / "turn_records.jsonl"
    monkeypatch.setattr(app_config, "TURN_TELEMETRY_PATH", str(p), raising=False)
    monkeypatch.setattr(app_config, "TURN_TELEMETRY_ENABLED", True, raising=False)
    return p


def _last(p):
    return json.loads(p.read_text().strip().splitlines()[-1])


@pytest.fixture
def controller():
    manager = MagicMock()
    manager.api_models = {}
    return AgenticSearchController(model_manager=manager, web_search_manager=MagicMock())


class _FailedService:
    providers = []

    async def search(self, *a, **k):
        return []

    async def recent(self, *a, **k):
        return []


class TestToolReceipts:
    @pytest.mark.asyncio
    async def test_failed_email_search_is_receipted(self, controller, rec_path):
        session = AgenticSearchSession(query="q", max_rounds=3, protocol="native")
        cov = {"searched": [], "unconnected": {}, "failed": {"gmail": "token revoked - reauth"}}
        with patch("core.email.service.get_email_service", return_value=_FailedService()), \
             patch("core.email.registry.provider_coverage", return_value=cov), \
             patch("core.email.registry.coverage_note", return_value=""):
            decision = SearchDecision(wants_email_search=True, email_query="SECRET query", email_window_days=30)
            await controller._dispatch_single_inner(decision, 1, session, None, None)

        assert session.tool_receipts[0]["tool"] == "email_search"
        assert session.tool_receipts[0]["status"] == "failed"
        assert session.providers_failed == {"gmail": "token revoked - reauth"}

        turn_telemetry.record_turn({
            "mode": "agentic-search",
            "tool_calls": session.tool_receipts,
            "providers_failed": session.providers_failed,
        })
        row = _last(rec_path)
        assert row["tool_calls"][0]["status"] == "failed"
        assert row["providers_failed"] == {"gmail": "token revoked - reauth"}
        assert "SECRET" not in json.dumps(row)  # no query text in receipts

    @pytest.mark.asyncio
    async def test_empty_and_raising_tools(self, controller):
        session = AgenticSearchSession(query="q", max_rounds=3, protocol="native")

        async def boom(decision, round_number):
            raise RuntimeError("kaboom with details")
        controller._dispatch_memory_search = boom
        decision = SearchDecision(wants_memory_search=True, memory_query="x", memory_collection="facts")
        with pytest.raises(RuntimeError):
            await controller._dispatch_single_inner(decision, 1, session, None, None)
        assert session.tool_receipts == [
            {"tool": "search_memory", "status": "failed", "detail": "RuntimeError"}
        ]


class TestRecordShape:
    def test_non_agentic_turn_has_empty_tool_calls(self, rec_path):
        turn_telemetry.record_turn({"mode": "enhanced"})
        row = _last(rec_path)
        assert row["tool_calls"] == []
        assert "providers_failed" not in row

    def test_sanitised_bounded_and_serialisable(self, rec_path):
        many = [{"tool": "t" * 500, "status": "weird", "detail": "d" * 900, "query": "PII"}] * 100
        turn_telemetry.record_turn({"mode": "agentic-search", "tool_calls": many})
        row = _last(rec_path)
        assert len(row["tool_calls"]) == 30
        c = row["tool_calls"][0]
        assert set(c) == {"tool", "status", "detail"}
        assert len(c["detail"]) == 120 and len(c["tool"]) == 60
        assert c["status"] in {"ok", "empty", "failed", "unavailable"}
        assert "PII" not in json.dumps(row)
        assert len(json.dumps(row)) < 20000

    def test_test_mode_never_touches_real_log(self, rec_path):
        import os
        assert os.getenv("DAEMON_TEST_MODE")
        turn_telemetry.record_turn({"mode": "x"})
        assert rec_path.exists()
        assert _last(rec_path)["test_env"] is True
