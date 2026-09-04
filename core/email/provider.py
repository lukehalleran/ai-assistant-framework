"""
# core/email/provider.py

Module Contract
- Purpose: provider-agnostic email types (2026-09-01, docs/EMAIL_INTEGRATION_DESIGN.md).
  EmailMessage is THE interchange shape every provider adapter returns and every
  consumer (agentic email_search tool, passive [RELEVANT EMAILS] gatherer,
  pattern-engine email dimension) reads. EmailProvider is the Protocol each
  adapter implements; the registry (registry.py) maps provider names to
  factories so adding a provider = one adapter + one registry row.
- Doctrine: read-only, metadata-first (headers + provider snippet only — never
  full bodies in v1), live-fetch only (results are NEVER persisted to
  chroma/corpus; 5-min TTL in-memory cache lives in service.py).
- Dependencies: stdlib only. Adapters depend on httpx + their auth modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Protocol, runtime_checkable


def message_timestamp(message: "EmailMessage") -> float:
    """Comparable instant for provider dates; malformed dates sort last."""
    value = (message.date or "").strip()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, OverflowError):
        return float("-inf")


@dataclass
class EmailMessage:
    """One email, provider-agnostic. All fields are plain strings so the
    formatter/tool layers never need provider-specific handling."""

    provider: str            # "gmail" | "outlook" | future registry names
    message_id: str
    thread_id: str = ""
    sender: str = ""         # display form: 'Name <addr>' when available
    to: str = ""
    subject: str = ""
    snippet: str = ""        # plain-text provider snippet / bodyPreview
    date: str = ""           # ISO 8601 (provider's received timestamp)
    unread: bool = False
    web_link: str = ""       # deep link to the message when the provider has one
    extra: dict = field(default_factory=dict)


@runtime_checkable
class EmailProvider(Protocol):
    """Contract every provider adapter implements. Adapters must be
    fail-soft: any auth/config/network problem returns [] (and health()
    explains why) — never an exception to the caller."""

    name: str

    def is_configured(self) -> bool:
        """Cheap, no-network: credentials/config present for this install."""
        ...

    async def health(self) -> dict:
        """{"available": bool, "detail": str} — cheap check (token presence /
        expiry / enable flags), no message fetch. Consumed by tool-health."""
        ...

    async def search(self, query: str, *, window_days: int = 30,
                     limit: int = 20) -> List[EmailMessage]:
        """Provider-native search over the window, newest first."""
        ...

    async def recent(self, *, window_days: int = 7,
                     limit: int = 25) -> List[EmailMessage]:
        """Recent messages in the window, newest first (no query)."""
        ...
