"""
# core/actions/__init__.py

Internet Actions subsystem — propose, confirm, and execute write actions
on external services (Telegram, Discord, email, GitHub, calendar).

Public exports:
  - ActionType, ActionProposal, ActionResult, PendingActionsStore (types)
  - ActionAuditLog (audit)
  - ActionExecutorRegistry (executors)
"""

from core.actions.types import (
    ActionType,
    ActionProposal,
    ActionResult,
    PendingActionsStore,
)
from core.actions.audit import ActionAuditLog

__all__ = [
    "ActionType",
    "ActionProposal",
    "ActionResult",
    "PendingActionsStore",
    "ActionAuditLog",
]
