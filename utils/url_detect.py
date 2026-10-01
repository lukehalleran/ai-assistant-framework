"""Shared URL-presence check (2026-09-28, class BC-01).

One compiled matcher replaces bare ``'http://' in text`` substring tests, which
fire on any occurrence of the characters regardless of token boundary.

Semantics: a URL is an ``http``/``https`` scheme followed by ``://`` and at
least one non-space character, starting at a token boundary (not glued to a
preceding word character or slash). ``www.`` and bare domains do NOT count --
this keeps the scheme-only semantics the agentic gate had before.
"""
import re

_URL_RE = re.compile(r"(?<![\w/])https?://\S", re.IGNORECASE)


def contains_url(text) -> bool:
    """True when ``text`` contains an http(s):// URL at a token boundary."""
    if not text:
        return False
    return _URL_RE.search(text) is not None
