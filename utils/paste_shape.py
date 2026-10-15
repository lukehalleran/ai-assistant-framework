"""Structural helpers for messages that carry an inline paste (leaf module).

No vocabulary: the boundary of the user's own words is found from layout alone
(a line-ending colon, a blank line, the first list item / heading). utils/ must
not import memory/, so memory_storage.is_paste_sized is deliberately not reused.
"""
import re

AUTHORED_HEAD_MIN_MESSAGE_WORDS = 120
AUTHORED_HEAD_MAX_WORDS = 60
AUTHORED_HEAD_MIN_WORDS = 4

_LIST_OR_HEADING_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|#)\s", re.MULTILINE)
_BLANK_LINE = re.compile(r"\n[ \t]*\n")
_COLON_BOUNDARY = re.compile(r":[ \t]*(?:\n|$)")


def authored_head(text: str) -> str:
    """The user's own opening words before an inline paste, or "".

    "" unless the message has more than 120 words. Otherwise the prefix up to
    the FIRST structural boundary (a colon ending a line, a blank line, or the
    start of a list-item/heading line), capped at 60 words; "" when the prefix
    has fewer than 4 words.
    """
    if not text or len(text.split()) <= AUTHORED_HEAD_MIN_MESSAGE_WORDS:
        return ""
    cut = len(text)
    m = _COLON_BOUNDARY.search(text)
    if m:
        cut = min(cut, m.end())
    m = _BLANK_LINE.search(text)
    if m:
        cut = min(cut, m.start())
    for m in _LIST_OR_HEADING_LINE.finditer(text):
        cut = min(cut, m.start())
        break
    head = text[:cut].strip()
    words = head.split()
    if len(words) > AUTHORED_HEAD_MAX_WORDS:
        head = " ".join(words[:AUTHORED_HEAD_MAX_WORDS])
        words = words[:AUTHORED_HEAD_MAX_WORDS]
    if len(words) < AUTHORED_HEAD_MIN_WORDS:
        return ""
    return head
