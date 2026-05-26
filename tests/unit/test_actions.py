"""
Unit tests for the Internet Actions subsystem.

Tests: ActionType, ActionProposal, PendingActionsStore, ActionAuditLog,
       ActionExecutorRegistry, protocol parsing, and dispatch.
"""

import asyncio
import json
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.actions.types import (
    ActionType,
    ActionProposal,
    ActionResult,
    PendingActionsStore,
    CONFIRMATION_REQUIRED,
)
from core.actions.audit import ActionAuditLog


# ---------------------------------------------------------------------------
# ActionType tests
# ---------------------------------------------------------------------------

class TestActionType:
    def test_all_types_are_strings(self):
        for t in ActionType:
            assert isinstance(t.value, str)

    def test_confirmation_required_set(self):
        assert ActionType.SEND_TELEGRAM in CONFIRMATION_REQUIRED
        assert ActionType.SEND_DISCORD in CONFIRMATION_REQUIRED
        assert ActionType.SEND_EMAIL in CONFIRMATION_REQUIRED
        assert ActionType.GITHUB_CREATE_ISSUE in CONFIRMATION_REQUIRED
        # Tier 1 tools don't require confirmation
        assert ActionType.BROWSER_FETCH not in CONFIRMATION_REQUIRED
        assert ActionType.RSS_CHECK not in CONFIRMATION_REQUIRED

    def test_from_string(self):
        assert ActionType("send_telegram") == ActionType.SEND_TELEGRAM
        assert ActionType("send_discord") == ActionType.SEND_DISCORD


# ---------------------------------------------------------------------------
# ActionProposal tests
# ---------------------------------------------------------------------------

class TestActionProposal:
    def test_defaults(self):
        p = ActionProposal(action_type=ActionType.SEND_TELEGRAM)
        assert p.action_id  # UUID generated
        assert p.status == "pending"
        assert p.params == {}
        assert p.reversible is True
        assert p.proposed_at <= datetime.now(timezone.utc)

    def test_custom_fields(self):
        p = ActionProposal(
            action_type=ActionType.SEND_EMAIL,
            params={"recipient": "test@example.com", "message": "Hello"},
            summary="Email to test@example.com",
            reasoning="User asked to send reminder",
        )
        assert p.action_type == ActionType.SEND_EMAIL
        assert p.params["recipient"] == "test@example.com"
        assert "Email" in p.summary


# ---------------------------------------------------------------------------
# PendingActionsStore tests
# ---------------------------------------------------------------------------

class TestPendingActionsStore:
    def test_propose_and_get(self):
        store = PendingActionsStore(ttl_seconds=60, max_pending=5)
        p = ActionProposal(action_type=ActionType.SEND_TELEGRAM, summary="test")
        assert store.propose(p) is True
        assert store.get(p.action_id) == p
        assert store.pending_count() == 1

    def test_max_pending_enforced(self):
        store = PendingActionsStore(ttl_seconds=60, max_pending=2)
        p1 = ActionProposal(action_type=ActionType.SEND_TELEGRAM)
        p2 = ActionProposal(action_type=ActionType.SEND_DISCORD)
        p3 = ActionProposal(action_type=ActionType.SEND_EMAIL)
        assert store.propose(p1) is True
        assert store.propose(p2) is True
        assert store.propose(p3) is False  # At capacity
        assert store.pending_count() == 2

    def test_ttl_expiry(self):
        store = PendingActionsStore(ttl_seconds=1, max_pending=5)
        p = ActionProposal(
            action_type=ActionType.SEND_TELEGRAM,
            proposed_at=datetime.now(timezone.utc) - timedelta(seconds=10),
        )
        store.propose(p)
        # After prune, should be expired
        assert store.get(p.action_id) is None
        assert store.pending_count() == 0

    def test_approve(self):
        store = PendingActionsStore(ttl_seconds=60, max_pending=5)
        p = ActionProposal(action_type=ActionType.SEND_TELEGRAM)
        store.propose(p)
        approved = store.approve(p.action_id)
        assert approved is not None
        assert approved.status == "approved"
        # Can't approve twice
        assert store.approve(p.action_id) is None

    def test_reject(self):
        store = PendingActionsStore(ttl_seconds=60, max_pending=5)
        p = ActionProposal(action_type=ActionType.SEND_TELEGRAM)
        store.propose(p)
        rejected = store.reject(p.action_id)
        assert rejected is not None
        assert rejected.status == "rejected"

    def test_mark_executed(self):
        store = PendingActionsStore(ttl_seconds=60, max_pending=5)
        p = ActionProposal(action_type=ActionType.SEND_TELEGRAM)
        store.propose(p)
        store.mark_executed(p.action_id, "sent ok")
        assert store._store[p.action_id].status == "executed"
        assert store._store[p.action_id].result == "sent ok"

    def test_mark_failed(self):
        store = PendingActionsStore(ttl_seconds=60, max_pending=5)
        p = ActionProposal(action_type=ActionType.SEND_TELEGRAM)
        store.propose(p)
        store.mark_failed(p.action_id, "network error")
        assert store._store[p.action_id].status == "failed"
        assert store._store[p.action_id].error == "network error"

    def test_get_pending_returns_most_recent(self):
        store = PendingActionsStore(ttl_seconds=60, max_pending=5)
        p1 = ActionProposal(
            action_type=ActionType.SEND_TELEGRAM,
            proposed_at=datetime.now(timezone.utc) - timedelta(seconds=5),
        )
        p2 = ActionProposal(
            action_type=ActionType.SEND_DISCORD,
            proposed_at=datetime.now(timezone.utc),
        )
        store.propose(p1)
        store.propose(p2)
        assert store.get_pending().action_id == p2.action_id

    def test_clear(self):
        store = PendingActionsStore(ttl_seconds=60, max_pending=5)
        p = ActionProposal(action_type=ActionType.SEND_TELEGRAM)
        store.propose(p)
        store.clear()
        assert store.pending_count() == 0

    def test_expired_marked_as_failed(self):
        store = PendingActionsStore(ttl_seconds=1, max_pending=5)
        p = ActionProposal(
            action_type=ActionType.SEND_TELEGRAM,
            proposed_at=datetime.now(timezone.utc) - timedelta(seconds=10),
        )
        store._store[p.action_id] = p
        p.expires_at = p.proposed_at + timedelta(seconds=1)
        store._prune_expired()
        assert store._store[p.action_id].status == "failed"
        assert store._store[p.action_id].error == "expired"


# ---------------------------------------------------------------------------
# ActionAuditLog tests
# ---------------------------------------------------------------------------

class TestActionAuditLog:
    def test_log_proposal(self, tmp_path):
        log_path = str(tmp_path / "audit.jsonl")
        audit = ActionAuditLog(log_path)
        p = ActionProposal(
            action_type=ActionType.SEND_TELEGRAM,
            params={"message": "hello"},
            summary="test proposal",
        )
        audit.log_proposal(p)

        entries = audit.get_history()
        assert len(entries) == 1
        assert entries[0]["event"] == "proposed"
        assert entries[0]["type"] == "send_telegram"
        assert entries[0]["summary"] == "test proposal"

    def test_log_decision(self, tmp_path):
        log_path = str(tmp_path / "audit.jsonl")
        audit = ActionAuditLog(log_path)
        audit.log_decision("abc123", approved=True)
        audit.log_decision("def456", approved=False, reason="changed mind")

        entries = audit.get_history()
        assert len(entries) == 2
        assert entries[0]["event"] == "approved"
        assert entries[1]["event"] == "rejected"
        assert entries[1]["reason"] == "changed mind"

    def test_log_execution(self, tmp_path):
        log_path = str(tmp_path / "audit.jsonl")
        audit = ActionAuditLog(log_path)
        result = ActionResult(action_id="abc", success=True, message="sent ok")
        audit.log_execution("abc", result)

        entries = audit.get_history()
        assert len(entries) == 1
        assert entries[0]["event"] == "executed"
        assert entries[0]["success"] is True

    def test_get_history_limit(self, tmp_path):
        log_path = str(tmp_path / "audit.jsonl")
        audit = ActionAuditLog(log_path)
        for i in range(10):
            audit.log_decision(f"id_{i}", approved=True)
        entries = audit.get_history(limit=3)
        assert len(entries) == 3

    def test_empty_log(self, tmp_path):
        log_path = str(tmp_path / "nonexistent.jsonl")
        audit = ActionAuditLog(log_path)
        assert audit.get_history() == []


# ---------------------------------------------------------------------------
# ActionExecutorRegistry tests
# ---------------------------------------------------------------------------

class TestActionExecutorRegistry:
    @pytest.mark.asyncio
    async def test_unknown_action_type(self):
        from core.actions.executors import ActionExecutorRegistry
        registry = ActionExecutorRegistry()
        p = ActionProposal(action_type=ActionType.BROWSER_FETCH)
        result = await registry.execute(p)
        assert result.success is False
        assert "No executor" in result.message

    @pytest.mark.asyncio
    async def test_telegram_dispatch(self):
        from core.actions.executors import ActionExecutorRegistry
        registry = ActionExecutorRegistry()
        p = ActionProposal(
            action_type=ActionType.SEND_TELEGRAM,
            params={"message": "test", "recipient": "12345"},
        )
        with patch("core.actions.telegram.send_telegram_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = ActionResult(
                action_id=p.action_id, success=True, message="sent"
            )
            result = await registry.execute(p)
            assert result.success is True
            mock_send.assert_called_once_with(p)

    @pytest.mark.asyncio
    async def test_executor_exception_caught(self):
        from core.actions.executors import ActionExecutorRegistry
        registry = ActionExecutorRegistry()
        p = ActionProposal(
            action_type=ActionType.SEND_TELEGRAM,
            params={"message": "test"},
        )
        with patch("core.actions.telegram.send_telegram_message", new_callable=AsyncMock) as mock_send:
            mock_send.side_effect = RuntimeError("network down")
            result = await registry.execute(p)
            assert result.success is False
            assert "network down" in result.message


# ---------------------------------------------------------------------------
# Protocol parsing tests (native tools)
# ---------------------------------------------------------------------------

class TestProtocolParsing:
    def _make_native_handler(self):
        from core.agentic.protocols import NativeToolsHandler
        return NativeToolsHandler(actions_available=True)

    def test_parse_propose_action_native(self):
        handler = self._make_native_handler()
        # Simulate a tool call response
        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "propose_action"
        mock_tool_call.function.arguments = json.dumps({
            "action_type": "send_telegram",
            "recipient": "@luke",
            "message": "Reminder about deadline",
            "reason": "User mentioned deadline",
        })
        mock_response.tool_calls = [mock_tool_call]
        mock_response.content = None

        decisions = handler.parse_response(mock_response)
        assert len(decisions) == 1
        d = decisions[0]
        assert d.wants_action is True
        assert d.action_type == "send_telegram"
        assert d.action_params["message"] == "Reminder about deadline"
        assert d.action_params["recipient"] == "@luke"
        assert "send_telegram to @luke" in d.action_summary

    def test_parse_propose_action_xml(self):
        from core.agentic.protocols import XMLMarkerHandler
        handler = XMLMarkerHandler()
        text = '<action type="send_telegram" recipient="@luke" reason="deadline reminder">Don\'t forget your ISYE deadline!</action>'
        decisions = handler.parse_response(text)
        assert len(decisions) == 1
        d = decisions[0]
        assert d.wants_action is True
        assert d.action_type == "send_telegram"
        assert d.action_params["message"] == "Don't forget your ISYE deadline!"
        assert d.action_params["recipient"] == "@luke"


# ---------------------------------------------------------------------------
# SearchDecision fields test
# ---------------------------------------------------------------------------

class TestSearchDecisionActionFields:
    def test_action_fields_exist(self):
        from core.agentic.types import SearchDecision
        d = SearchDecision(
            wants_action=True,
            action_type="send_telegram",
            action_params={"message": "hello"},
            action_summary="send telegram: hello",
            action_reason="user asked",
        )
        assert d.wants_action is True
        assert d.action_type == "send_telegram"
        assert d.action_params == {"message": "hello"}


# ---------------------------------------------------------------------------
# Telegram executor tests
# ---------------------------------------------------------------------------

class TestTelegramExecutor:
    @pytest.mark.asyncio
    async def test_missing_token(self):
        from core.actions.telegram import send_telegram_message
        p = ActionProposal(
            action_type=ActionType.SEND_TELEGRAM,
            params={"message": "test", "recipient": "12345"},
        )
        with patch("config.app_config.INTERNET_ACTIONS_TELEGRAM_BOT_TOKEN", ""):
            result = await send_telegram_message(p)
        assert result.success is False
        assert "not configured" in result.message

    @pytest.mark.asyncio
    async def test_missing_chat_id(self):
        from core.actions.telegram import send_telegram_message
        p = ActionProposal(
            action_type=ActionType.SEND_TELEGRAM,
            params={"message": "test"},  # No recipient
        )
        with patch("config.app_config.INTERNET_ACTIONS_TELEGRAM_BOT_TOKEN", "fake_token"), \
             patch("config.app_config.INTERNET_ACTIONS_TELEGRAM_CHAT_ID", ""):
            result = await send_telegram_message(p)
        assert result.success is False
        assert "chat ID" in result.message

    @pytest.mark.asyncio
    async def test_successful_send(self):
        from core.actions.telegram import send_telegram_message
        p = ActionProposal(
            action_type=ActionType.SEND_TELEGRAM,
            params={"message": "hello world", "recipient": "12345"},
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True, "result": {"message_id": 42}}

        with patch("config.app_config.INTERNET_ACTIONS_TELEGRAM_BOT_TOKEN", "fake_token"), \
             patch("config.app_config.INTERNET_ACTIONS_TELEGRAM_CHAT_ID", "default_chat"):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.post.return_value = mock_response
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                result = await send_telegram_message(p)

        assert result.success is True
        assert "msg_id: 42" in result.message
