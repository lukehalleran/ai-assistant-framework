#core/wiki_util
"""Shared Wikipedia helpers used by the prompt builder."""

from __future__ import annotations

import re
from functools import lru_cache

from knowledge.WikiManager import WikiManager, _clean_query as _wiki_clean_query

# Disambiguation-page lead shape: "<Title> may (also) refer to:" and variants.
# Only tested against the OPENING of a chunk (see looks_like_disambiguation_text)
# so an article that merely discusses ambiguity mid-text never matches.
_DISAMBIGUATION_RE = re.compile(
    r"\b(?:may|might|can|commonly|most\s+commonly)\s+(?:also\s+)?refers?\s+to\b",
    re.IGNORECASE,
)


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
        # Fix 1.6 (2026-09-06): the live-API fallback
        # (WikiManager._fetch_extract_action_api) hardcodes
        # is_disambiguation=False, so a text-shaped stub ("Give may refer
        # to: ...") sailed past the flag check above and reached
        # [BACKGROUND KNOWLEDGE] — check the shape here too, regardless of
        # which fetch path populated `page`.
        if looks_like_disambiguation_text(page.summary, getattr(page, "title", "")):
            return ""
        return page.summary
    return ""


def looks_like_disambiguation_text(text: str, title: str = "") -> bool:
    """Text-shape test for wiki disambiguation-page chunks.

    The live-API path drops disambiguation pages via ``page.is_disambiguation``,
    but the embedded wiki corpus (wiki_knowledge collection + FAISS semantic
    chunks) contains them as plain text — "Feel may refer to:\\n\\nFeeling"
    reached [BACKGROUND KNOWLEDGE] on an emotional turn (2026-08-28). Those
    chunks carry no useful content on ANY query, so a deterministic drop is
    always safe. Conservative: only the first ~100 normalized chars are
    tested, plus an explicit "(disambiguation)" title suffix.
    """
    if title and title.strip().lower().endswith("(disambiguation)"):
        return True
    if not text:
        return False
    head = " ".join(text.strip().split())[:100]
    return bool(_DISAMBIGUATION_RE.search(head))


__all__ = ["get_wiki_snippet", "clean_query", "looks_like_disambiguation_text"]
