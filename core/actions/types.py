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
from utils.bootstrap import store_path
from utils.safe_json import atomic_write_json

try:
    from config.app_config import PENDING_ACTIONS_STORE_PATH as _CFG_STORE_PATH
except Exception:
    _CFG_STORE_PATH = store_path("data/pending_actions.json")
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
    status: Literal["pending", "approved", "rejected", "executed", "failed", "interrupted"] = "pending"
    result: Optional[str] = None
    error: Optional[str] = None
    # When approval happened (UTC); lets a proposal stuck in "approved" with no
    # live execution be recovered to "interrupted" (2026-10-23, audit A07).
    approved_at: Optional[datetime] = None
    # When the record became "interrupted" (UTC); drives its aging. Legacy
    # records without it age from approved_at / proposed_at.
    interrupted_at: Optional[datetime] = None


class ActionResult(BaseModel):
    """Outcome of an executed action."""
    action_id: str
    success: bool
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # True when execution was cancelled mid-flight and the outcome is UNKNOWN:
    # the audit log names the event "interrupted", not "failed" (2026-10-26).
    interrupted: bool = False


class ActionOutcome(BaseModel):
    """Transport-agnostic result of an approve/reject decision on a pending action.

    Returned by gui.handlers.execute_pending_action_core / reject_pending_action_core;
    consumed by both the Gradio button wrappers and the FastAPI action routes.
    `message` is the assistant-styled chat line to append to the conversation.
    """
    status: Literal["executed", "failed", "rejected", "not_found", "interrupted"]
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

# An approved proposal with no outcome after this long has no live execution
# (a single executor call is one API round-trip, seconds; 15 min is generous)
# and is reconciled to "interrupted" rather than holding a capacity slot forever.
APPROVED_STUCK_SECONDS = 900

# An interrupted record stays visible (re-approval answers "interrupted", it
# survives restarts) for this long, then ages out like any other terminal
# record instead of living in the store and the JSON file forever.
INTERRUPTED_RETENTION_SECONDS = 24 * 3600

INTERRUPTED_REASON = (
    "execution was interrupted before an outcome was recorded; "
    "whether it happened is unknown"
)

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
        reconciled = 0
        for data in payload.get("proposals", []) if isinstance(payload, dict) else []:
            try:
                proposal = ActionProposal.model_validate(data)
                if proposal.status == "approved":
                    # No live execution can survive a restart: the outcome is
                    # unknown. Keep the record (visibility/audit), never retry.
                    proposal.status = "interrupted"
                    proposal.error = INTERRUPTED_REASON
                    proposal.interrupted_at = datetime.now(timezone.utc)
                    reconciled += 1
                    self._store[proposal.action_id] = proposal
                elif proposal.status in ("pending", "interrupted"):
                    self._store[proposal.action_id] = proposal
                    restored += 1
            except Exception as exc:
                logger.warning(f"[Actions] Ignoring malformed persisted proposal: {exc}")
        self._prune_expired(restart=True)
        if reconciled:
            logger.warning(
                f"[Actions] {reconciled} approved proposal(s) found at load; "
                "marked interrupted (outcome unknown, not retried)"
            )
            self._save()
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
            proposal.approved_at = datetime.now(timezone.utc)
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

    def mark_interrupted(self, action_id: str, reason: str = INTERRUPTED_REASON) -> None:
        """Mark a proposal whose execution was cancelled mid-flight.

        Terminal and NEVER auto-retried: an external side effect may already
        have happened. Frees the capacity slot (only pending/approved count).
        """
        proposal = self._store.get(action_id)
        if proposal and proposal.status in ("approved", "pending"):
            proposal.status = "interrupted"
            proposal.error = reason
            proposal.interrupted_at = datetime.now(timezone.utc)
            self._save()

    def peek_status(self, action_id: str) -> Optional[str]:
        """Status of a stored proposal regardless of state (None if unknown)."""
        self._prune_expired()
        proposal = self._store.get(action_id)
        return proposal.status if proposal else None

    def mark_failed(self, action_id: str, error: str) -> None:
        """Mark a proposal as failed."""
        proposal = self._store.get(action_id)
        if proposal:
            proposal.status = "failed"
            proposal.error = error
            self._save()

    def most_recent_failed(self, max_age_seconds: int = 1800) -> Optional["ActionProposal"]:
        """The newest FAILED proposal (executor error or expired unseen) proposed
        within ``max_age_seconds``, or None.

        2026-09-07: "good now, please try again" after a token-refresh failure
        had no route — the failed proposal, with the exact params the user had
        already approved once, was sitting here. The retry path re-queues it
        deterministically (new card, human approval again) instead of asking
        the model to re-derive the event.
        """
        self._prune_expired()
        now = datetime.now(timezone.utc)
        candidates = [
            p for p in self._store.values()
            if p.status == "failed"
            and (now - p.proposed_at).total_seconds() <= max_age_seconds
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.proposed_at)

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
        stuck = [
            aid for aid, p in self._store.items()
            if p.status == "approved"
            and (p.approved_at is None
                 or (now - p.approved_at).total_seconds() > APPROVED_STUCK_SECONDS)
        ]
        for aid in stuck:
            self._store[aid].status = "interrupted"
            self._store[aid].error = INTERRUPTED_REASON
            self._store[aid].interrupted_at = now
            logger.warning(f"[Actions] Proposal {aid} stuck in approved; marked interrupted")
        aged = [
            aid for aid, p in self._store.items()
            if p.status == "interrupted"
            and (now - (p.interrupted_at or p.approved_at or p.proposed_at)).total_seconds()
            > INTERRUPTED_RETENTION_SECONDS
        ]
        for aid in aged:
            del self._store[aid]
            logger.info(f"[Actions] Interrupted proposal {aid} aged out")
        if expired or stuck or aged:
            self._save()
