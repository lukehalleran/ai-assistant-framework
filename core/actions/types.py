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
import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

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


# ---------------------------------------------------------------------------
# Pending Actions Store
# ---------------------------------------------------------------------------

class PendingActionsStore:
    """In-memory store of pending action proposals, keyed by action_id, with TTL expiry.

    Thread-safe for single-process async use (no concurrent writes from threads).
    """

    def __init__(self, ttl_seconds: int = 300, max_pending: int = 5):
        self._store: Dict[str, ActionProposal] = {}
        self._ttl_seconds = ttl_seconds
        self._max_pending = max_pending

    def propose(self, proposal: ActionProposal) -> bool:
        """Store a new proposal. Returns False if at capacity (after pruning expired)."""
        self._prune_expired()
        if len(self._store) >= self._max_pending:
            logger.warning(
                f"[Actions] Pending store at capacity ({self._max_pending}), "
                f"rejecting proposal {proposal.action_id}"
            )
            return False
        # Set expiry based on store TTL
        proposal.expires_at = proposal.proposed_at + timedelta(seconds=self._ttl_seconds)
        self._store[proposal.action_id] = proposal
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
            return proposal
        return None

    def reject(self, action_id: str) -> Optional[ActionProposal]:
        """Mark a proposal as rejected. Returns the proposal or None if expired/missing."""
        proposal = self.get(action_id)
        if proposal and proposal.status == "pending":
            proposal.status = "rejected"
            return proposal
        return None

    def mark_executed(self, action_id: str, result: str) -> None:
        """Mark a proposal as successfully executed."""
        proposal = self._store.get(action_id)
        if proposal:
            proposal.status = "executed"
            proposal.result = result

    def mark_failed(self, action_id: str, error: str) -> None:
        """Mark a proposal as failed."""
        proposal = self._store.get(action_id)
        if proposal:
            proposal.status = "failed"
            proposal.error = error

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

    def clear(self) -> None:
        """Clear all proposals."""
        self._store.clear()

    def _prune_expired(self) -> None:
        """Remove expired proposals from the store."""
        now = datetime.now(timezone.utc)
        expired = [
            aid for aid, p in self._store.items()
            if p.status == "pending" and p.expires_at <= now
        ]
        for aid in expired:
            self._store[aid].status = "failed"
            self._store[aid].error = "expired"
            logger.debug(f"[Actions] Proposal {aid} expired")
