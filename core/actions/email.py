"""
# core/actions/email.py

Module Contract
- Purpose: Send emails via Gmail API (preferred) or SMTP (fallback).
  Resolves recipient names to emails via Google Contacts when recipient lacks '@'.
- Public interface:
  - send_email(proposal: ActionProposal) -> ActionResult
  - _resolve_recipient(name: str) -> tuple[Optional[str], str]
  - _try_gmail_send(proposal, recipient, message) -> ActionResult | None
  - _smtp_send(proposal, recipient, message) -> ActionResult
- Dependencies: httpx (Gmail API), smtplib (SMTP fallback), core.actions.google_auth,
  core.actions.google_contacts (for name resolution)
- Side effects: Sends an email. Gmail API attempted first; SMTP only if Gmail unconfigured.
  No SMTP fallback after a Gmail API attempt (prevents duplicate sends).
"""

import logging
from typing import Optional

from core.actions.types import ActionProposal, ActionResult

logger = logging.getLogger("actions_email")


async def send_email(proposal: ActionProposal) -> ActionResult:
    """Send an email via Gmail API or SMTP fallback.

    Tries Gmail API first. Falls back to SMTP only if Gmail OAuth is
    unconfigured or unauthenticated.  If Gmail API was attempted but
    failed, does NOT fall back to SMTP (prevents duplicate sends).

    Expects proposal.params to contain:
        - message (str): Email body text.
        - recipient (str): Recipient email address.
        - subject (str, optional): Email subject line.
    """
    recipient = proposal.params.get("recipient", "")
    if not recipient:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="Missing recipient.",
        )

    # If recipient lacks '@', try to resolve name → email (defense in depth;
    # the agentic proposal path already resolves at proposal time)
    if "@" not in recipient:
        resolved, resolve_msg = await _resolve_recipient(recipient)
        if resolved:
            recipient = resolved
            logger.info(f"[Email] Resolved '{proposal.params.get('recipient')}' -> {recipient}")
        else:
            return ActionResult(
                action_id=proposal.action_id,
                success=False,
                message=resolve_msg,
            )

    message_text = proposal.params.get("message", "")
    if not message_text:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="No message content to send.",
        )

    # Try Gmail API first
    gmail_result = await _try_gmail_send(proposal, recipient, message_text)
    if gmail_result is not None:
        # Gmail was attempted — return result regardless of success/failure
        return gmail_result

    # Gmail not configured/authenticated — fall back to SMTP
    return await _smtp_send(proposal, recipient, message_text)


async def _try_gmail_send(
    proposal: ActionProposal,
    recipient: str,
    message: str,
) -> Optional[ActionResult]:
    """Try sending via Gmail API.

    Returns:
        ActionResult(success=True)  — Gmail sent successfully.
        ActionResult(success=False) — Gmail attempted but failed (no SMTP fallback).
        None                        — Gmail not configured; caller should try SMTP.
    """
    from config.app_config import (
        INTERNET_ACTIONS_GOOGLE_CLIENT_ID,
        INTERNET_ACTIONS_GOOGLE_CLIENT_SECRET,
        INTERNET_ACTIONS_GOOGLE_TOKEN_PATH,
        INTERNET_ACTIONS_SMTP_FROM,
        INTERNET_ACTIONS_SMTP_USER,
    )

    # Not configured → return None so caller falls back to SMTP
    if not INTERNET_ACTIONS_GOOGLE_CLIENT_ID or not INTERNET_ACTIONS_GOOGLE_CLIENT_SECRET:
        return None

    from core.actions.google_auth import GoogleAuthManager

    auth = GoogleAuthManager(
        client_id=INTERNET_ACTIONS_GOOGLE_CLIENT_ID,
        client_secret=INTERNET_ACTIONS_GOOGLE_CLIENT_SECRET,
        token_path=INTERNET_ACTIONS_GOOGLE_TOKEN_PATH,
    )

    if not auth.is_authenticated:
        return None

    creds = auth.get_credentials()
    if not creds:
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message="Gmail token refresh failed.",
        )

    # Build RFC 5322 message
    import base64
    import email.message

    from_addr = INTERNET_ACTIONS_SMTP_FROM or INTERNET_ACTIONS_SMTP_USER
    subject = proposal.params.get("subject", "Message from Daemon")

    msg = email.message.EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = recipient
    msg.set_content(message)

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

    try:
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
                headers={"Authorization": f"Bearer {creds.token}"},
                json={"raw": raw},
                timeout=15.0,
            )

        if resp.status_code == 200:
            logger.info(f"[Gmail] Sent to {recipient}: {subject}")
            return ActionResult(
                action_id=proposal.action_id,
                success=True,
                message=f"Gmail sent to {recipient} with subject: {subject}",
            )

        logger.warning(f"[Gmail] API error: HTTP {resp.status_code}")
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message=f"Gmail API error: HTTP {resp.status_code}",
        )

    except Exception as e:
        logger.error(f"[Gmail] Send failed: {e}")
        return ActionResult(
            action_id=proposal.action_id,
            success=False,
            message=f"Gmail send failed: {e}",
        )


async def _smtp_send(
    proposal: ActionProposal,
    recipient: str,
    message: str,
) -> ActionResult:
    """Send via SMTP. Used as fallback when Gmail is not configured."""
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

    subject = proposal.params.get("subject", "Message from Daemon")
    from_addr = INTERNET_ACTIONS_SMTP_FROM or INTERNET_ACTIONS_SMTP_USER

    import email.message
    msg = email.message.EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = recipient
    msg.set_content(message)

    try:
        import asyncio
        import smtplib

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


async def _resolve_recipient(name: str) -> tuple:
    """Resolve a name to an email address using Google Contacts.

    Returns:
        (email, "") on single match — auto-resolve.
        (None, message) on zero or multiple matches.
    """
    try:
        from core.actions.google_contacts import resolve_contact
        matches = await resolve_contact(name, max_results=10)
    except Exception as e:
        logger.debug(f"[Email] Contact resolution failed: {e}")
        return None, f"Could not resolve '{name}' to an email address: {e}"

    if not matches:
        return None, (
            f"Could not resolve '{name}' to an email address. "
            f"No matches found in Google Contacts."
        )

    if len(matches) == 1:
        return matches[0]["email"], ""

    # Multiple matches — list them so caller can specify
    match_lines = [f"  - {m['name']} <{m['email']}> ({m['source']})" for m in matches[:10]]
    match_str = "\n".join(match_lines)
    return None, (
        f"Multiple contacts found for '{name}':\n{match_str}\n"
        f"Please specify the exact email address."
    )
