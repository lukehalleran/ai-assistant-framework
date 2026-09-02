"""
# core/email/registry.py

Module Contract
- Purpose: Provider registry — mapping of provider names to factories and
  enable conditions (2026-09-01, docs/EMAIL_INTEGRATION_DESIGN.md).
  Adding a provider = one adapter module + one row here; service/tool/health
  consumers inherit automatically (ACTION_SPECS doctrine).
- Public interface:
  - build_enabled_providers() -> List[EmailProvider]: instantiate configured providers.
- Dependencies: core/email/{provider,gmail_provider,outlook_provider}
"""

from typing import Callable, Dict, List

from core.email.provider import EmailProvider
from utils.logging_utils import get_logger

logger = get_logger("email_registry")


def _build_gmail() -> EmailProvider:
    """Factory for Gmail provider."""
    from core.email.gmail_provider import GmailProvider

    return GmailProvider()


def _build_outlook() -> EmailProvider:
    """Factory for Outlook provider."""
    from core.email.outlook_provider import OutlookProvider

    return OutlookProvider()


def _gmail_enabled() -> bool:
    """Check if Gmail provider should be instantiated.

    Uses function-body import for live-config doctrine.
    """
    try:
        from config.app_config import (
            EMAIL_INTEGRATION_ENABLED,
            EMAIL_GMAIL_ENABLED,
        )
    except ImportError:
        return False
    return EMAIL_INTEGRATION_ENABLED and EMAIL_GMAIL_ENABLED


def _outlook_enabled() -> bool:
    """Check if Outlook provider should be instantiated.

    Uses function-body import for live-config doctrine.
    """
    try:
        from config.app_config import (
            EMAIL_INTEGRATION_ENABLED,
            EMAIL_OUTLOOK_ENABLED,
        )
    except ImportError:
        return False
    return EMAIL_INTEGRATION_ENABLED and EMAIL_OUTLOOK_ENABLED


# Provider registry: name -> {factory: callable, enabled: callable}
PROVIDERS: Dict[str, Dict[str, Callable]] = {
    "gmail": {
        "factory": _build_gmail,
        "enabled": _gmail_enabled,
    },
    "outlook": {
        "factory": _build_outlook,
        "enabled": _outlook_enabled,
    },
}


def provider_coverage() -> Dict[str, object]:
    """Which providers a search actually covers — {'searched': [names],
    'unconnected': {name: reason}}. Deterministic and cheap (config flags +
    is_configured disk checks, no network). Consumers append this to email
    answers so a NEGATIVE result is honestly scoped: "no reply from Morgan"
    means nothing when her mail lives in a provider we never looked at
    (2026-09-01 live finding — advisor emails in unconnected Outlook)."""
    searched: List[str] = []
    unconnected: Dict[str, str] = {}
    for provider_name, registry_row in PROVIDERS.items():
        try:
            if not registry_row["enabled"]():
                unconnected[provider_name] = "disabled"
                continue
            provider = registry_row["factory"]()
            if provider.is_configured():
                searched.append(provider_name)
            else:
                unconnected[provider_name] = "not connected"
        except Exception:
            unconnected[provider_name] = "unavailable"
    return {"searched": searched, "unconnected": unconnected}


def coverage_note() -> str:
    """One-line human-readable coverage disclosure, '' when nothing useful
    to say (no providers registered at all)."""
    cov = provider_coverage()
    searched = cov["searched"]
    unconnected = cov["unconnected"]
    parts = []
    if searched:
        parts.append("Searched: " + ", ".join(n.capitalize() for n in searched))
    else:
        parts.append("No email accounts connected")
    if unconnected:
        parts.append("; ".join(
            f"{n.capitalize()} {reason}" for n, reason in unconnected.items()))
    return ". ".join(parts) + "."


def build_enabled_providers() -> List[EmailProvider]:
    """Instantiate all enabled providers.

    Returns list of configured EmailProvider instances. A provider that
    raises during construction is logged and skipped (never fatal).

    This is the entry point consumed by core/email/service.py at startup.
    """
    providers: List[EmailProvider] = []

    for provider_name, registry_row in PROVIDERS.items():
        enabled_fn = registry_row["enabled"]
        factory_fn = registry_row["factory"]

        try:
            # Check enabled flag at call time (live-config doctrine)
            if not enabled_fn():
                logger.debug(f"[EmailRegistry] {provider_name} disabled")
                continue

            # Instantiate
            provider = factory_fn()
            if not provider.is_configured():
                logger.debug(f"[EmailRegistry] {provider_name} not configured")
                continue

            providers.append(provider)
            logger.info(f"[EmailRegistry] loaded {provider_name}")

        except Exception as e:
            logger.warning(
                f"[EmailRegistry] Failed to load {provider_name}: {e}"
            )
            continue

    return providers
