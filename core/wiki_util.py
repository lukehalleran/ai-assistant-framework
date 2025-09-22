#core/wiki_util
"""Shared Wikipedia helpers used by the prompt builder."""

from __future__ import annotations

from functools import lru_cache

from knowledge.WikiManager import WikiManager, _clean_query as _wiki_clean_query


@lru_cache(maxsize=1)
def _get_manager() -> WikiManager:
    return WikiManager()


def clean_query(query: str) -> str:
    return _wiki_clean_query(query)


def get_wiki_snippet(query: str) -> str:
    if not query:
        return ""
    try:
        page = _get_manager().resolve_and_fetch(query)
    except Exception:
        return ""
    # Drop disambiguation pages entirely (e.g., "Luke may refer to:")
    if not page or page.is_disambiguation:
        return ""
    if page.summary:
        return page.summary
    return ""


__all__ = ["get_wiki_snippet", "clean_query"]
