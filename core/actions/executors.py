"""
# core/actions/executors.py

Module Contract
- Purpose: Registry that routes ActionProposal execution to type-specific executors.
- Public interface:
  - ActionExecutorRegistry: class with execute(proposal) -> ActionResult
- Dependencies:
  - core.actions.types (ActionProposal, ActionResult)
  - core.actions.registry (ACTION_SPECS — the action→executor source of truth)
- Side effects: Executes internet write actions (sends messages, creates issues, calendar events, etc.)
"""

import logging

from core.actions.types import ActionProposal, ActionResult

logger = logging.getLogger("actions")


class ActionExecutorRegistry:
    """Routes action execution to type-specific async handlers."""

    async def execute(self, proposal: ActionProposal) -> ActionResult:
        """Execute an approved action proposal by routing to its registered executor.

        The action→executor mapping lives in core.actions.registry (ACTION_SPECS) — the single
        source of truth. Adding an action there makes it executable here automatically.
        """
        from core.actions.registry import ACTION_SPECS

        spec = ACTION_SPECS.get(proposal.action_type)
        if spec is None:
            return ActionResult(
                action_id=proposal.action_id,
                success=False,
                message=f"No executor registered for action type: {proposal.action_type.value}",
            )

        try:
            return await spec.resolve_executor()(proposal)
        except Exception as e:
            logger.error(f"[Actions] Executor failed for {proposal.action_type.value}: {e}")
            return ActionResult(
                action_id=proposal.action_id,
                success=False,
                message=f"Execution error: {e}",
            )
