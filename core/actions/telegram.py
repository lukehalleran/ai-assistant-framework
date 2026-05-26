"""
# core/actions/telegram.py

Module Contract
- Purpose: Send messages via Telegram Bot API.
- Public interface:
  - send_telegram_message(proposal: ActionProposal) -> ActionResult
- Dependencies:
  - httpx (async HTTP client)
  - config.app_config (INTERNET_ACTIONS_TELEGRAM_BOT_TOKEN, INTERNET_ACTIONS_TELEGRAM_CHAT_ID)
- Side effects: Sends a message to a Telegram chat via Bot API HTTP POST.
"""

import logging

from core.actions.types import ActionProposal, ActionResult

logger = logging.getLogger("actions_telegram")


async def send_telegram_message(proposal: ActionProposal) -> ActionResult:
    """Send a message via Telegram Bot API.

    Expects proposal.params to contain:
        - message (str): The message text to send.
        - recipient (str, optional): Chat ID override. Falls back to config default.

    Returns:
        ActionResult with success status and delivery info.
    """
    from config.app_config import (
        INTERNET_ACTIONS_TELEGRAM_BOT_TOKEN,
        INTERNET_ACTIONS_TELEGRAM_CHAT_ID,
    )

    bot_token = INTERNET_ACTIONS_TELEGRAM_BOT_TOKEN
    if not bot_token:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="Telegram bot token not configured. Set telegram_bot_token in config or TELEGRAM_BOT_TOKEN env var.",
        )

    chat_id = proposal.params.get("recipient") or INTERNET_ACTIONS_TELEGRAM_CHAT_ID
    if not chat_id:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="No Telegram chat ID specified. Set telegram_default_chat_id in config or provide a recipient.",
        )

    message_text = proposal.params.get("message", "")
    if not message_text:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="No message content to send.",
        )

    # Send via Bot API
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message_text,
        "parse_mode": "Markdown",
    }

    try:
        import httpx
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, json=payload)

        if response.status_code == 200:
            data = response.json()
            if data.get("ok"):
                msg_id = data.get("result", {}).get("message_id", "?")
                logger.info(f"[Telegram] Message sent successfully (msg_id={msg_id}, chat={chat_id})")
                return ActionResult(
                    action_id=proposal.action_id,
                    success=True,
                    message=f"Sent Telegram message to chat {chat_id} (msg_id: {msg_id})",
                )
            else:
                desc = data.get("description", "Unknown error")
                logger.warning(f"[Telegram] API error: {desc}")
                return ActionResult(
                    action_id=proposal.action_id,
                    success=False,
                    message=f"Telegram API error: {desc}",
                )
        else:
            logger.warning(f"[Telegram] HTTP {response.status_code}: {response.text[:200]}")
            return ActionResult(
                action_id=proposal.action_id,
                success=False,
                message=f"Telegram HTTP error {response.status_code}: {response.text[:100]}",
            )

    except ImportError:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="httpx not installed. Run: pip install httpx",
        )
    except Exception as e:
        logger.error(f"[Telegram] Send failed: {e}")
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message=f"Failed to send Telegram message: {e}",
        )
