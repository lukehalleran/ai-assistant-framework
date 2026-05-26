"""
# core/actions/email.py

Module Contract
- Purpose: Send emails via SMTP.
- Public interface:
  - send_email(proposal: ActionProposal) -> ActionResult
- Dependencies: aiosmtplib (async SMTP), or fallback to stdlib smtplib in thread
- Side effects: Sends an email via configured SMTP server.
"""

import logging

from core.actions.types import ActionProposal, ActionResult

logger = logging.getLogger("actions_email")


async def send_email(proposal: ActionProposal) -> ActionResult:
    """Send an email via SMTP.

    Expects proposal.params to contain:
        - message (str): Email body text.
        - recipient (str): Recipient email address.
        - subject (str, optional): Email subject line.
    """
    from config.app_config import (
        INTERNET_ACTIONS_SMTP_HOST,
        INTERNET_ACTIONS_SMTP_PORT,
        INTERNET_ACTIONS_SMTP_USER,
        INTERNET_ACTIONS_SMTP_PASSWORD,
        INTERNET_ACTIONS_SMTP_FROM,
    )

    if not INTERNET_ACTIONS_SMTP_HOST:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="SMTP not configured. Set smtp_host in config.",
        )

    recipient = proposal.params.get("recipient", "")
    if not recipient or "@" not in recipient:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message=f"Invalid or missing recipient email address: {recipient!r}",
        )

    message_text = proposal.params.get("message", "")
    if not message_text:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="No message content to send.",
        )

    subject = proposal.params.get("subject", "Message from Daemon")
    from_addr = INTERNET_ACTIONS_SMTP_FROM or INTERNET_ACTIONS_SMTP_USER

    # Build email
    import email.message
    msg = email.message.EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = recipient
    msg.set_content(message_text)

    try:
        import asyncio
        import smtplib

        # Use stdlib smtplib in a thread (aiosmtplib may not be installed)
        def _send():
            with smtplib.SMTP(INTERNET_ACTIONS_SMTP_HOST, INTERNET_ACTIONS_SMTP_PORT) as server:
                server.starttls()
                if INTERNET_ACTIONS_SMTP_USER and INTERNET_ACTIONS_SMTP_PASSWORD:
                    server.login(INTERNET_ACTIONS_SMTP_USER, INTERNET_ACTIONS_SMTP_PASSWORD)
                server.send_message(msg)

        await asyncio.get_event_loop().run_in_executor(None, _send)
        logger.info(f"[Email] Sent to {recipient}: {subject}")
        return ActionResult(
            action_id=proposal.action_id,
            success=True,
            message=f"Email sent to {recipient} with subject: {subject}",
        )

    except Exception as e:
        logger.error(f"[Email] Send failed: {e}")
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message=f"Failed to send email: {e}",
        )
