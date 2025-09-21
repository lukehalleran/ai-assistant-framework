#intergrations/wikipedia_api.py
"""Compatibility wrapper providing an async Wikipedia fetch interface."""

from __future__ import annotations

import asyncio
from typing import Tuple

from knowledge.WikiManager import WikiManager


class WikipediaAPI:
    """Lightweight async facade for the existing WikiManager.

    The gate system expects an object exposing ``fetch_article_text`` that
    returns ``(ok, text)`` when awaited.  WikiManager already resolves topics
    and returns summaries, so we simply delegate to it in a background thread.
    """

    def __init__(self, lang: str = "en") -> None:
        self._manager = WikiManager(lang=lang)

    async def fetch_article_text(self, title: str) -> Tuple[bool, str]:
        """Return (ok, article_text) for a resolved Wikipedia title."""

        if not title:
            return False, ""

        loop = asyncio.get_running_loop()

        def _resolve() -> Tuple[bool, str]:
            try:
                page = self._manager.resolve_and_fetch(title)
            except Exception:
                return False, ""
            if page and getattr(page, "summary", None):
                return True, page.summary
            return False, ""

        return await loop.run_in_executor(None, _resolve)


__all__ = ["WikipediaAPI"]

