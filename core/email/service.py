"""
# core/email/service.py

Module Contract
- Purpose: EmailService — the single consumer-facing surface over all enabled
  email providers (docs/EMAIL_INTEGRATION_DESIGN.md). Fans a search/recent
  call out across the registry's enabled providers concurrently, merges
  newest-first, caches results (in-memory, TTL — the google_calendar
  pattern). Consumers: agentic email_search tool, passive [RELEVANT EMAILS]
  gatherer, pattern-engine email dimension.
- Key symbols (FROZEN contract — consumers import exactly these):
  - get_email_service() -> EmailService (module singleton)
  - EmailService.search(query, *, window_days=30, limit=20) -> list[EmailMessage]
  - EmailService.recent(*, window_days=7, limit=25) -> list[EmailMessage]
  - EmailService.health() -> dict[str, dict]  (provider name -> health dict)
- Fail-soft doctrine: a provider that raises is logged and contributes [] —
  one broken provider never breaks the fan-out.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional

from core.email.provider import EmailMessage, EmailProvider
from utils.logging_utils import get_logger

logger = get_logger("email_service")


class EmailService:
    """Fan-out over enabled providers with a small TTL cache."""

    def __init__(self, providers: Optional[List[EmailProvider]] = None,
                 cache_ttl_seconds: Optional[float] = None):
        if providers is None:
            # lazy import: cycle (registry imports adapters which may import
            # config at call time)
            from core.email.registry import build_enabled_providers
            providers = build_enabled_providers()
        self.providers: List[EmailProvider] = providers
        if cache_ttl_seconds is None:
            try:
                from config.app_config import EMAIL_CACHE_TTL_SECONDS
                cache_ttl_seconds = float(EMAIL_CACHE_TTL_SECONDS)
            except Exception:
                cache_ttl_seconds = 300.0
        self._cache_ttl = cache_ttl_seconds
        self._cache: Dict[tuple, tuple] = {}  # key -> (ts, results)

    async def _fan_out(self, coros) -> List[EmailMessage]:
        results = await asyncio.gather(*coros, return_exceptions=True)
        merged: List[EmailMessage] = []
        for provider, res in zip(self.providers, results):
            if isinstance(res, Exception):
                logger.warning(
                    f"[EmailService] provider {getattr(provider, 'name', '?')} "
                    f"failed: {res}")
                continue
            merged.extend(res or [])
        merged.sort(key=lambda m: m.date or "", reverse=True)
        return merged

    def _cached(self, key: tuple) -> Optional[List[EmailMessage]]:
        hit = self._cache.get(key)
        if hit and (time.monotonic() - hit[0]) < self._cache_ttl:
            return hit[1]
        return None

    async def search(self, query: str, *, window_days: int = 30,
                     limit: int = 20) -> List[EmailMessage]:
        key = ("search", (query or "").strip().lower(), window_days, limit)
        cached = self._cached(key)
        if cached is not None:
            return cached
        merged = await self._fan_out([
            p.search(query, window_days=window_days, limit=limit)
            for p in self.providers
        ])
        merged = merged[:limit]
        self._cache[key] = (time.monotonic(), merged)
        return merged

    async def recent(self, *, window_days: int = 7,
                     limit: int = 25) -> List[EmailMessage]:
        key = ("recent", window_days, limit)
        cached = self._cached(key)
        if cached is not None:
            return cached
        merged = await self._fan_out([
            p.recent(window_days=window_days, limit=limit)
            for p in self.providers
        ])
        merged = merged[:limit]
        self._cache[key] = (time.monotonic(), merged)
        return merged

    async def health(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for p in self.providers:
            try:
                out[p.name] = await p.health()
            except Exception as e:
                out[p.name] = {"available": False, "detail": f"health check failed: {e}"}
        return out


_service: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """Module singleton (reset by tests via `core.email.service._service = None`)."""
    global _service
    if _service is None:
        _service = EmailService()
    return _service
