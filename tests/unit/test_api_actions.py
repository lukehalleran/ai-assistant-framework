"""Tests for pending-action approve/reject: core functions, API routes, Gradio wrappers.

The core functions are THE deployed decision path (audit + executor registry);
the Gradio wrappers must keep their legacy (chat_history, gr.update, gr.update)
tuple so the /admin buttons and existing wiring stay intact.
"""

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from api.app import create_app
from core.actions.types import ActionOutcome
from tests.unit.helpers_orchestrator import _make_orchestrator


def _make_proposal(action_type="send_email", summary="Send email to Bob"):
    proposal = MagicMock()
    proposal.action_type.value = action_type
    proposal.summary = summary
    return proposal


def _patches(store, execute_result=None, execute_error=None):
    """Patch the pending-actions store, audit log, and executor registry."""
    audit = MagicMock()
    registry = MagicMock()
    if execute_error is not None:
        registry.execute = AsyncMock(side_effect=execute_error)
    else:
        registry.execute = AsyncMock(return_value=execute_result)
    return [
        patch("core.agentic.tools.ToolExecutor._get_pending_actions_store",
              return_value=store),
        patch("core.actions.audit.ActionAuditLog", return_value=audit),
        patch("core.actions.executors.ActionExecutorRegistry", return_value=registry),
    ], audit, registry


class TestExecuteCore:
    @pytest.mark.asyncio
    async def test_approve_success(self):
        from gui.handlers import execute_pending_action_core

        proposal = _make_proposal()
        store = MagicMock()
        store.approve.return_value = proposal
        result = MagicMock(success=True, message="Email sent to Bob")

        patches, audit, registry = _patches(store, execute_result=result)
        for p in patches:
            p.start()
        try:
            outcome = await execute_pending_action_core("act-1")
        finally:
            patch.stopall()

        assert isinstance(outcome, ActionOutcome)
        assert outcome.status == "executed"
        assert "[ACTION EXECUTED: send_email]" in outcome.message
        store.mark_executed.assert_called_once_with("act-1", "Email sent to Bob")
        audit.log_decision.assert_called_once_with("act-1", approved=True)
        audit.log_execution.assert_called_once()

    @pytest.mark.asyncio
    async def test_approve_executor_failure(self):
        from gui.handlers import execute_pending_action_core

        store = MagicMock()
        store.approve.return_value = _make_proposal()
        result = MagicMock(success=False, message="SMTP down")

        patches, audit, _ = _patches(store, execute_result=result)
        for p in patches:
            p.start()
        try:
            outcome = await execute_pending_action_core("act-1")
        finally:
            patch.stopall()

        assert outcome.status == "failed"
        assert "SMTP down" in outcome.message
        store.mark_failed.assert_called_once_with("act-1", "SMTP down")

    @pytest.mark.asyncio
    async def test_approve_executor_exception(self):
        from gui.handlers import execute_pending_action_core

        store = MagicMock()
        store.approve.return_value = _make_proposal()

        patches, _, _ = _patches(store, execute_error=RuntimeError("boom"))
        for p in patches:
            p.start()
        try:
            outcome = await execute_pending_action_core("act-1")
        finally:
            patch.stopall()

        assert outcome.status == "failed"
        assert "boom" in outcome.message
        store.mark_failed.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_action_id(self):
        from gui.handlers import execute_pending_action_core

        store = MagicMock()
        store.approve.return_value = None

        patches, _, _ = _patches(store)
        for p in patches:
            p.start()
        try:
            outcome = await execute_pending_action_core("nope")
        finally:
            patch.stopall()

        assert outcome.status == "not_found"

    @pytest.mark.asyncio
    async def test_empty_action_id_short_circuits(self):
        from gui.handlers import execute_pending_action_core
        outcome = await execute_pending_action_core("")
        assert outcome.status == "not_found"


class TestRejectCore:
    @pytest.mark.asyncio
    async def test_reject(self):
        from gui.handlers import reject_pending_action_core

        store = MagicMock()
        store.reject.return_value = _make_proposal(summary="Send email to Bob")

        patches, audit, _ = _patches(store)
        for p in patches:
            p.start()
        try:
            outcome = await reject_pending_action_core("act-1")
        finally:
            patch.stopall()

        assert outcome.status == "rejected"
        assert "Send email to Bob" in outcome.message
        audit.log_decision.assert_called_once_with("act-1", approved=False)


class TestGradioWrappers:
    """The legacy wrappers must keep the (history, gr.update, gr.update) contract."""

    @pytest.mark.asyncio
    async def test_wrapper_returns_three_tuple_and_appends(self):
        from gui.handlers import execute_pending_action

        store = MagicMock()
        store.approve.return_value = _make_proposal()
        result = MagicMock(success=True, message="done")

        patches, _, _ = _patches(store, execute_result=result)
        for p in patches:
            p.start()
        try:
            history = []
            out = await execute_pending_action("act-1", history)
        finally:
            patch.stopall()

        assert len(out) == 3
        assert out[0] is history
        assert history[-1]["role"] == "assistant"
        assert "[ACTION EXECUTED" in history[-1]["content"]

    @pytest.mark.asyncio
    async def test_wrapper_empty_id_appends_nothing(self):
        from gui.handlers import reject_pending_action
        history = []
        out = await reject_pending_action("", history)
        assert len(out) == 3
        assert history == []


class TestActionRoutes:
    def _app(self):
        return create_app(_make_orchestrator(), start_background=False)

    @pytest.mark.asyncio
    async def test_approve_route(self):
        app = self._app()
        session = app.state.daemon.session
        session.pending_action_id = "act-1"

        store = MagicMock()
        store.approve.return_value = _make_proposal()
        result = MagicMock(success=True, message="sent")

        patches, audit, _ = _patches(store, execute_result=result)
        for p in patches:
            p.start()
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
                resp = await client.post("/api/actions/act-1/approve")
        finally:
            patch.stopall()

        assert resp.status_code == 200
        body = resp.json()
        assert body["outcome"]["status"] == "executed"
        assert body["message"]["role"] == "assistant"
        # Route appends the chat line and clears the pending id
        assert session.history[-1]["content"] == body["message"]["content"]
        assert session.pending_action_id is None
        audit.log_decision.assert_called_once_with("act-1", approved=True)

    @pytest.mark.asyncio
    async def test_reject_route(self):
        app = self._app()
        store = MagicMock()
        store.reject.return_value = _make_proposal()

        patches, _, _ = _patches(store)
        for p in patches:
            p.start()
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
                resp = await client.post("/api/actions/act-1/reject")
        finally:
            patch.stopall()

        assert resp.status_code == 200
        assert resp.json()["outcome"]["status"] == "rejected"
