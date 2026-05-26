"""
# core/actions/discord.py

Module Contract
- Purpose: Send messages via Discord webhook.
- Public interface:
  - send_discord_message(proposal: ActionProposal) -> ActionResult
- Dependencies: httpx
- Side effects: Sends a message to a Discord channel via webhook POST.
"""

import logging

from core.actions.types import ActionProposal, ActionResult

logger = logging.getLogger("actions_discord")


async def send_discord_message(proposal: ActionProposal) -> ActionResult:
    """Send a message via Discord webhook.

    Expects proposal.params to contain:
        - message (str): The message text.
        - recipient (str, optional): Webhook URL override. Falls back to config default.
    """
    from config.app_config import INTERNET_ACTIONS_DISCORD_WEBHOOK_URL

    webhook_url = proposal.params.get("recipient") or INTERNET_ACTIONS_DISCORD_WEBHOOK_URL
    if not webhook_url:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="Discord webhook URL not configured. Set discord_webhook_url in config or DISCORD_WEBHOOK_URL env var.",
        )

    message_text = proposal.params.get("message", "")
    if not message_text:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="No message content to send.",
        )

    payload = {"content": message_text}

    try:
        import httpx
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(webhook_url, json=payload)

        if response.status_code in (200, 204):
            logger.info("[Discord] Message sent successfully via webhook")
            return ActionResult(
                action_id=proposal.action_id,
                success=True,
                message="Sent Discord message via webhook.",
            )
        else:
            logger.warning(f"[Discord] HTTP {response.status_code}: {response.text[:200]}")
            return ActionResult(
                action_id=proposal.action_id,
                success=False,
                message=f"Discord HTTP error {response.status_code}: {response.text[:100]}",
            )

    except ImportError:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="httpx not installed. Run: pip install httpx",
        )
    except Exception as e:
        logger.error(f"[Discord] Send failed: {e}")
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message=f"Failed to send Discord message: {e}",
        )
