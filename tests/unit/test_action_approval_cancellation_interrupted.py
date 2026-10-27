"""Cancelled action approval ends in an explicit "interrupted" state.

Incident date: 2026-10-23 (audit A07). Class: BC-30 / BC-40 / BC-47.
A CancelledError during execution used to leave the proposal "approved"
forever (slot consumed, not listed/rejectable/prunable). Fake executors only.
"""
import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from core.actions import types as atypes
from core.actions.types import ActionProposal, ActionType, PendingActionsStore
from core.agentic.tools import ToolExecutor
import gui.handlers as handlers
from config import app_config


def _proposal(summary="Send a note to Rowan"):
    return ActionProposal(action_type=ActionType.SEND_TELEGRAM,
                          params={"text": "hi"}, summary=summary)


@pytest.fixture
def env(tmp_path, monkeypatch):
    store = PendingActionsStore(max_pending=1, persist=False)
    monkeypatch.setattr(ToolExecutor, "_pending_actions_store", store, raising=False)
    log = tmp_path / "audit.jsonl"
    monkeypatch.setattr(app_config, "INTERNET_ACTIONS_AUDIT_LOG", str(log))
    effects = []

    def install(after_effect):
        class FakeRegistry:
            async def execute(self, proposal):
                if after_effect:
                    effects.append(proposal.action_id)  # simulated external side effect
                raise asyncio.CancelledError()
        monkeypatch.setattr(handlers.executors, "ActionExecutorRegistry", FakeRegistry)

    return store, log, effects, install


@pytest.mark.parametrize("after_effect", [False, True])
def test_cancel_marks_interrupted_frees_slot_and_propagates(env, after_effect):
    store, log, effects, install = env
    install(after_effect)
    p = _proposal()
    assert store.propose(p)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(handlers.execute_pending_action_core(p.action_id))
    assert store._store[p.action_id].status == "interrupted"
    assert "unknown" in store._store[p.action_id].error
    # capacity freed: a new proposal is accepted at max_pending=1
    assert store.propose(_proposal("second"))
    # never auto-retried
    assert store.most_recent_failed() is None
    assert store.approve(p.action_id) is None and store.reject(p.action_id) is None
    assert bool(effects) is after_effect  # executor ran exactly once, no retry
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert lines[-1]["action_id"] == p.action_id
    assert "INTERRUPTED" in lines[-1]["message"] and lines[-1]["success"] is False


def test_reapproving_interrupted_reports_interrupted(env):
    store, _, _, install = env
    install(False)
    p = _proposal()
    store.propose(p)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(handlers.execute_pending_action_core(p.action_id))
    out = asyncio.run(handlers.execute_pending_action_core(p.action_id))
    assert out.status == "interrupted"
    assert "check whether it happened" in out.message


def test_stuck_approved_recovered_after_bound():
    store = PendingActionsStore(max_pending=1, persist=False)
    p = _proposal()
    store.propose(p)
    store.approve(p.action_id)
    assert not store.propose(_proposal("blocked"))  # genuinely executing: counts
    p.approved_at = datetime.now(timezone.utc) - timedelta(
        seconds=atypes.APPROVED_STUCK_SECONDS + 5)
    assert store.propose(_proposal("after recovery"))
    assert p.status == "interrupted"


def test_persisted_approved_reconciled_on_load(tmp_path):
    path = str(tmp_path / "pending.json")
    s1 = PendingActionsStore(max_pending=2, path=path)
    p = _proposal()
    s1.propose(p)
    s1.approve(p.action_id)
    s2 = PendingActionsStore(max_pending=1, path=path)
    assert s2._store[p.action_id].status == "interrupted"
    assert s2.propose(_proposal("new"))
    on_disk = json.load(open(path))["proposals"]
    assert any(d["status"] == "interrupted" for d in on_disk)
    s3 = PendingActionsStore(max_pending=2, path=path)
    assert s3._store[p.action_id].status == "interrupted"  # kept across loads


# --- 2026-10-26 (lane H1, #101 follow-up, class: BC-58) ---------------------
def test_cancel_audit_event_is_named_interrupted_not_failed(env):
    store, log, _, install = env
    install(False)
    p = _proposal()
    store.propose(p)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(handlers.execute_pending_action_core(p.action_id))
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert lines[-1]["event"] == "interrupted"
    assert lines[-1]["success"] is False


def test_genuine_failure_still_audits_as_failed(tmp_path):
    from core.actions.audit import ActionAuditLog
    path = tmp_path / "a.jsonl"
    audit = ActionAuditLog(str(path))
    audit.log_execution("x", atypes.ActionResult(action_id="x", success=False, message="boom"))
    assert json.loads(path.read_text().splitlines()[-1])["event"] == "failed"


def test_interrupted_record_ages_out_but_survives_before_the_bound(tmp_path):
    path = str(tmp_path / "pending.json")
    s1 = PendingActionsStore(max_pending=2, path=path)
    p = _proposal()
    s1.propose(p)
    s1.approve(p.action_id)
    s1.mark_interrupted(p.action_id)
    assert s1._store[p.action_id].interrupted_at is not None
    # inside the window: kept across a reload (existing visibility contract)
    assert PendingActionsStore(max_pending=2, path=path)._store[p.action_id].status == "interrupted"
    # past the window: pruned from the store AND the file on the next prune
    s1._store[p.action_id].interrupted_at = datetime.now(timezone.utc) - timedelta(
        seconds=atypes.INTERRUPTED_RETENTION_SECONDS + 5)
    s1._prune_expired()
    assert p.action_id not in s1._store
    assert p.action_id not in {d["action_id"] for d in json.load(open(path))["proposals"]}
    assert p.action_id not in PendingActionsStore(max_pending=2, path=path)._store


def test_legacy_interrupted_record_without_timestamp_ages_from_proposed_at(tmp_path):
    path = str(tmp_path / "pending.json")
    s1 = PendingActionsStore(max_pending=2, path=path)
    p = _proposal()
    p.proposed_at = datetime.now(timezone.utc) - timedelta(
        seconds=atypes.INTERRUPTED_RETENTION_SECONDS + 60)
    p.status = "interrupted"
    s1._store[p.action_id] = p
    s1._save()
    assert p.action_id not in PendingActionsStore(max_pending=2, path=path)._store
