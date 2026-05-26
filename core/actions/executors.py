"""
# core/actions/executors.py

Module Contract
- Purpose: Registry that routes ActionProposal execution to type-specific executors.
- Public interface:
  - ActionExecutorRegistry: class with execute(proposal) -> ActionResult
- Dependencies:
  - core.actions.types (ActionProposal, ActionResult, ActionType)
  - Individual executor modules (telegram, discord, email)
- Side effects: Executes internet write actions (sends messages, creates issues, etc.)
"""

import logging
from typing import Dict, Callable, Awaitable

from core.actions.types import ActionProposal, ActionResult, ActionType

logger = logging.getLogger("actions")


class ActionExecutorRegistry:
    """Routes action execution to type-specific async handlers."""

    async def execute(self, proposal: ActionProposal) -> ActionResult:
        """Execute an approved action proposal.

        Args:
            proposal: The approved ActionProposal to execute.

        Returns:
            ActionResult with success/failure and message.
        """
        executor_map: Dict[ActionType, Callable[[ActionProposal], Awaitable[ActionResult]]] = {
            ActionType.SEND_TELEGRAM: self._execute_telegram,
            ActionType.SEND_DISCORD: self._execute_discord,
            ActionType.SEND_EMAIL: self._execute_email,
            ActionType.GITHUB_CREATE_ISSUE: self._execute_github_create_issue,
            ActionType.GITHUB_COMMENT_PR: self._execute_github_comment_pr,
            ActionType.CALENDAR_CREATE_EVENT: self._execute_calendar_create_event,
        }

        executor = executor_map.get(proposal.action_type)
        if not executor:
            return ActionResult(
                action_id=proposal.action_id,
                success=False,
                message=f"No executor registered for action type: {proposal.action_type.value}",
            )

        try:
            return await executor(proposal)
        except Exception as e:
            logger.error(f"[Actions] Executor failed for {proposal.action_type.value}: {e}")
            return ActionResult(
                action_id=proposal.action_id,
                success=False,
                message=f"Execution error: {e}",
            )

    async def _execute_telegram(self, proposal: ActionProposal) -> ActionResult:
        """Send a Telegram message via Bot API."""
        from core.actions.telegram import send_telegram_message
        return await send_telegram_message(proposal)

    async def _execute_discord(self, proposal: ActionProposal) -> ActionResult:
        """Send a Discord message via webhook."""
        from core.actions.discord import send_discord_message
        return await send_discord_message(proposal)

    async def _execute_email(self, proposal: ActionProposal) -> ActionResult:
        """Send an email via SMTP."""
        from core.actions.email import send_email
        return await send_email(proposal)

    async def _execute_github_create_issue(self, proposal: ActionProposal) -> ActionResult:
        """Create a GitHub issue via gh CLI."""
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="GitHub issue creation not yet implemented.",
        )

    async def _execute_github_comment_pr(self, proposal: ActionProposal) -> ActionResult:
        """Comment on a GitHub PR via gh CLI."""
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="GitHub PR commenting not yet implemented.",
        )

    async def _execute_calendar_create_event(self, proposal: ActionProposal) -> ActionResult:
        """Create a Google Calendar event."""
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="Calendar event creation not yet implemented.",
        )
