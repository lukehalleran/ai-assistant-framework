"""Structural separation of a user's own REQUEST from a pasted document block.

Leaf module (stdlib only). A message is "request + paste" when a contiguous
middle block is both long (> PASTE_BLOCK_MIN_CHARS) and document-shaped
(many lines, a large share of them short — headings, nav/reference chrome),
flanked by a short request at the head and/or tail. Routing and tone
detectors judge only the request span, so third-party text cannot set the
user's intent, tone or distress floor (BC-58).

Conservative by construction: a long first-person message (few long lines,
no short-line document structure) is NEVER split — the whole text returns.
"""
from __future__ import annotations

# Same threshold as memory_storage's paste guard (FACT_EXTRACT_PASTE_CHARS).
PASTE_BLOCK_MIN_CHARS = 1500
REQUEST_MAX_CHARS = 300
_BLOCK_MIN_LINES = 12
_SHORT_LINE_CHARS = 60
_SHORT_LINE_SHARE = 0.25


def _is_document_block(block: str) -> bool:
    if len(block) <= PASTE_BLOCK_MIN_CHARS:
        return False
    lines = [ln for ln in block.split("\n") if ln.strip()]
    if len(lines) < _BLOCK_MIN_LINES:
        return False
    short = sum(1 for ln in lines if len(ln.strip()) <= _SHORT_LINE_CHARS)
    return short / len(lines) >= _SHORT_LINE_SHARE


def request_span(text: str) -> str:
    """Return the user's own request text, excluding a pasted document block.

    Returns `text` unchanged unless a document-shaped block is flanked by a
    short head and/or tail request.
    """
    if not text or len(text) <= PASTE_BLOCK_MIN_CHARS:
        return text
    stripped = text.strip()
    head, sep, rest = stripped.partition("\n")
    head = head.strip()
    if not sep or not head or len(head) > REQUEST_MAX_CHARS:
        head, rest = "", stripped
    tail = ""
    body = rest
    idx = rest.rstrip().rfind("\n\n")
    if idx != -1:
        cand = rest[idx:].strip()
        if cand and len(cand) <= REQUEST_MAX_CHARS and cand.count("\n") <= 1:
            tail, body = cand, rest[:idx]
    if not _is_document_block(body):
        if tail and _is_document_block(rest):
            return head if head else text
        return text
    if not head and not tail:
        return text
    return "\n".join(p for p in (head, tail) if p)
