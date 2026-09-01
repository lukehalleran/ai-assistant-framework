"""
# core/actions/types.py

Module Contract
- Purpose: Data models for the Internet Actions subsystem.
- Public interface:
  - ActionType (str enum): available action types
  - ActionProposal (Pydantic BaseModel): a proposed write action awaiting confirmation
  - ActionResult (Pydantic BaseModel): outcome of an executed action
  - PendingActionsStore: in-memory store of pending proposals with TTL expiry
- Dependencies: None (pure data models + lightweight store)
- Side effects: None
"""

import uuid
import json
import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from utils.safe_json import atomic_write_json

try:
    from config.app_config import PENDING_ACTIONS_STORE_PATH as _CFG_STORE_PATH
except Exception:
    _CFG_STORE_PATH = "data/pending_actions.json"
_STORE_PATH = _CFG_STORE_PATH

logger = logging.getLogger("actions")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Available internet action types."""
    # Tier 2: Write actions (require confirmation)
    SEND_TELEGRAM = "send_telegram"
    SEND_DISCORD = "send_discord"
    SEND_EMAIL = "send_email"
    GITHUB_CREATE_ISSUE = "github_create_issue"
    GITHUB_COMMENT_PR = "github_comment_pr"
    CALENDAR_CREATE_EVENT = "calendar_create_event"
    CALENDAR_UPDATE_EVENT = "calendar_update_event"
    CALENDAR_DELETE_EVENT = "calendar_delete_event"
    # Tier 1: Read-only enhanced (no confirmation needed)
    BROWSER_FETCH = "browser_fetch"
    RSS_CHECK = "rss_check"


# Actions that require user confirmation before execution
CONFIRMATION_REQUIRED = {
    ActionType.SEND_TELEGRAM,
    ActionType.SEND_DISCORD,
    ActionType.SEND_EMAIL,
    ActionType.GITHUB_CREATE_ISSUE,
    ActionType.GITHUB_COMMENT_PR,
    ActionType.CALENDAR_CREATE_EVENT,
    ActionType.CALENDAR_UPDATE_EVENT,
    ActionType.CALENDAR_DELETE_EVENT,
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ActionProposal(BaseModel):
    """A proposed write action awaiting user confirmation."""
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action_type: ActionType
    params: Dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    reasoning: str = ""
    reversible: bool = True
    proposed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(seconds=300))
    status: Literal["pending", "approved", "rejected", "executed", "failed"] = "pending"
    result: Optional[str] = None
    error: Optional[str] = None


class ActionResult(BaseModel):
    """Outcome of an executed action."""
    action_id: str
    success: bool
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ActionOutcome(BaseModel):
    """Transport-agnostic result of an approve/reject decision on a pending action.

    Returned by gui.handlers.execute_pending_action_core / reject_pending_action_core;
    consumed by both the Gradio button wrappers and the FastAPI action routes.
    `message` is the assistant-styled chat line to append to the conversation.
    """
    status: Literal["executed", "failed", "rejected", "not_found"]
    message: str
    action_type: Optional[str] = None
    summary: Optional[str] = None
    # Approval chaining (2026-09-01): when another proposal is still pending
    # after this decision, surface it so multi-proposal turns (e.g.
    # delete + create) can be approved one after another instead of the
    # older proposal silently expiring unseen.
    next_action_id: Optional[str] = None
    next_summary: Optional[str] = None


# ---------------------------------------------------------------------------
# Pending Actions Store
# ---------------------------------------------------------------------------

class PendingActionsStore:
    """JSON-backed store of pending action proposals, keyed by action_id.

    TTL remains 300 seconds by default, so only quick restarts preserve a live
    proposal. Slow-restart losses are made visible; fast fix-and-restart cycles
    keep the proposal approvable. Persistence is best-effort and lenient.
    """

    def __init__(self, ttl_seconds: int = 300, max_pending: int = 5,
                 path: Optional[str] = None, persist: bool = True):
        self._store: Dict[str, ActionProposal] = {}
        self._ttl_seconds = ttl_seconds
        self._max_pending = max_pending
        self._path = path or _STORE_PATH
        self._persist = persist
        if persist:
            self._load()

    def _save(self) -> None:
        if not self._persist:
            return
        try:
            atomic_write_json(self._path, {
                "schema_version": 1,
                "proposals": [p.model_dump(mode="json") for p in self._store.values()],
            })
        except Exception as exc:
            logger.warning(f"[Actions] Could not persist pending actions: {exc}")

    def _load(self) -> None:
        try:
            with open(self._path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return
        except Exception as exc:
            logger.warning(f"[Actions] Could not load pending actions; cold start: {exc}")
            return
        restored = 0
        for data in payload.get("proposals", []) if isinstance(payload, dict) else []:
            try:
                proposal = ActionProposal.model_validate(data)
                if proposal.status == "pending":
                    self._store[proposal.action_id] = proposal
                    restored += 1
            except Exception as exc:
                logger.warning(f"[Actions] Ignoring malformed persisted proposal: {exc}")
        self._prune_expired(restart=True)
        if restored:
            logger.info(f"[Actions] Restored {restored} pending proposal(s) from disk")

    def propose(self, proposal: ActionProposal) -> bool:
        """Store a new proposal. Returns False if at capacity (after pruning expired)."""
        self._prune_expired()
        # Audit F14 (2026-08-31): only ACTIVE proposals occupy slots —
        # counting terminal ones (executed/failed/rejected, incl. expired
        # restores) rejected every new proposal after 5 lifetime outcomes
        # until restart.
        active = sum(
            1 for p in self._store.values() if p.status in ("pending", "approved")
        )
        if active >= self._max_pending:
            logger.warning(
                f"[Actions] Pending store at capacity ({self._max_pending}), "
                f"rejecting proposal {proposal.action_id}"
            )
            return False
        # Set expiry based on store TTL
        proposal.expires_at = proposal.proposed_at + timedelta(seconds=self._ttl_seconds)
        self._store[proposal.action_id] = proposal
        self._save()
        logger.info(f"[Actions] Proposal stored: {proposal.action_id} ({proposal.action_type.value})")
        return True

    def get(self, action_id: str) -> Optional[ActionProposal]:
        """Get a proposal by ID. Returns None if expired, failed, or not found."""
        self._prune_expired()
        proposal = self._store.get(action_id)
        if proposal and proposal.status in ("pending", "approved", "executed"):
            return proposal
        return None

    def approve(self, action_id: str) -> Optional[ActionProposal]:
        """Mark a proposal as approved. Returns the proposal or None if expired/missing."""
        proposal = self.get(action_id)
        if proposal and proposal.status == "pending":
            proposal.status = "approved"
            self._save()
            return proposal
        return None

    def reject(self, action_id: str) -> Optional[ActionProposal]:
        """Mark a proposal as rejected. Returns the proposal or None if expired/missing."""
        proposal = self.get(action_id)
        if proposal and proposal.status == "pending":
            proposal.status = "rejected"
            self._save()
            return proposal
        return None

    def mark_executed(self, action_id: str, result: str) -> None:
        """Mark a proposal as successfully executed."""
        proposal = self._store.get(action_id)
        if proposal:
            proposal.status = "executed"
            proposal.result = result
            self._save()

    def mark_failed(self, action_id: str, error: str) -> None:
        """Mark a proposal as failed."""
        proposal = self._store.get(action_id)
        if proposal:
            proposal.status = "failed"
            proposal.error = error
            self._save()

    def pending_count(self) -> int:
        """Number of currently pending (non-expired) proposals."""
        self._prune_expired()
        return sum(1 for p in self._store.values() if p.status == "pending")

    def get_pending(self) -> Optional[ActionProposal]:
        """Get the most recent pending proposal, or None."""
        self._prune_expired()
        pending = [p for p in self._store.values() if p.status == "pending"]
        if not pending:
            return None
        return max(pending, key=lambda p: p.proposed_at)

    def get_all_pending(self) -> List["ActionProposal"]:
        """All pending proposals, oldest first (2026-09-01: a delete+create
        turn orphaned the older proposal — its card never rendered and it
        could never be approved)."""
        self._prune_expired()
        return sorted(
            (p for p in self._store.values() if p.status == "pending"),
            key=lambda p: p.proposed_at,
        )

    def clear(self) -> None:
        """Clear all proposals."""
        self._store.clear()
        self._save()

    def _prune_expired(self, restart: bool = False) -> None:
        """Remove expired proposals from the store."""
        now = datetime.now(timezone.utc)
        expired = [
            aid for aid, p in self._store.items()
            if p.status == "pending" and p.expires_at <= now
        ]
        for aid in expired:
            self._store[aid].status = "failed"
            self._store[aid].error = "expired"
            if restart:
                logger.warning(f"[Actions] Proposal {aid} expired during restart")
            else:
                logger.debug(f"[Actions] Proposal {aid} expired")
        if expired:
            self._save()
