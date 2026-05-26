"""
# core/actions/audit.py

Module Contract
- Purpose: Append-only JSONL audit log for all action proposals, decisions, and executions.
- Public interface:
  - ActionAuditLog: class with log_proposal(), log_decision(), log_execution(), get_history()
- Dependencies: core.actions.types (ActionProposal, ActionResult)
- Side effects: Writes to audit JSONL file on disk
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.actions.types import ActionProposal, ActionResult

logger = logging.getLogger("actions_audit")


class ActionAuditLog:
    """Append-only JSONL audit log for internet actions.

    Each line is a JSON object with:
        ts: ISO timestamp
        event: "proposed" | "approved" | "rejected" | "executed" | "failed"
        action_id: UUID of the action
        + event-specific fields
    """

    def __init__(self, log_path: str = "logs/actions_audit.jsonl"):
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_proposal(self, proposal: ActionProposal) -> None:
        """Log that an action was proposed."""
        self._append({
            "event": "proposed",
            "action_id": proposal.action_id,
            "type": proposal.action_type.value,
            "params": proposal.params,
            "summary": proposal.summary,
            "reasoning": proposal.reasoning,
        })

    def log_decision(self, action_id: str, approved: bool, reason: str = "") -> None:
        """Log that the user approved or rejected an action."""
        self._append({
            "event": "approved" if approved else "rejected",
            "action_id": action_id,
            "reason": reason,
        })

    def log_execution(self, action_id: str, result: ActionResult) -> None:
        """Log the outcome of action execution."""
        self._append({
            "event": "executed" if result.success else "failed",
            "action_id": action_id,
            "success": result.success,
            "message": result.message,
        })

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Read the most recent N audit entries."""
        if not self._log_path.exists():
            return []
        entries: List[Dict[str, Any]] = []
        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except OSError as e:
            logger.warning(f"[ActionAudit] Could not read log: {e}")
            return []
        return entries[-limit:]

    def _append(self, record: Dict[str, Any]) -> None:
        """Append a single record to the audit log."""
        record["ts"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError as e:
            logger.error(f"[ActionAudit] Failed to write audit entry: {e}")
