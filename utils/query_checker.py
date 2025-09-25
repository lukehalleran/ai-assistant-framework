
"""
Utilities for quick query analysis and gating hints.

This module began as a small deictic checker; it now provides a few
lightweight heuristics that help the orchestrator and gate system make
fast decisions without model calls.
"""

from dataclasses import dataclass
from typing import List, Optional, Set


DEICTIC_HINTS: tuple[str, ...] = (
    "explain", "that", "it", "this", "again", "another way",
    "different way", "more", "elaborate", "clarify", "what about",
    "those", "these", "there", "former", "latter"
)

QUESTION_LEADS: tuple[str, ...] = (
    "what", "who", "when", "where", "why", "how", "which"
)

COMMAND_SIGNS: tuple[str, ...] = (
    "/", "please ", "do ", "tell me to ", "create ", "generate ", "write ", "summarize ",
)


def _normalize(q: str) -> str:
    return (q or "").strip().lower()


def is_deictic(query: str) -> bool:
    """True if the query likely refers to earlier context (anaphora)."""
    if not query:
        return False
    ql = _normalize(query)

    # Short follow-ups with hints are often deictic
    if len(ql.split()) <= 6 and any(h in ql for h in DEICTIC_HINTS):
        return True

    # Pronouns/markers at beginning suggest reference
    if ql.startswith(("that", "this", "it", "they", "those", "these", "so", "and", "then")):
        return True

    return False


def is_deictic_followup(q: str) -> bool:
    """Softer check for follow-up phrasing used by wiki gating."""
    ql = _normalize(q)
    return any(h in ql for h in DEICTIC_HINTS)


def is_question(q: str) -> bool:
    ql = _normalize(q)
    return ql.endswith("?") or ql.startswith(QUESTION_LEADS)


def is_command(q: str) -> bool:
    ql = _normalize(q)
    return ql.startswith(COMMAND_SIGNS)


def keyword_tokens(q: str, min_len: int = 3) -> List[str]:
    ql = _normalize(q)
    return [t for t in ql.split() if len(t) >= min_len]


@dataclass
class QueryAnalysis:
    text: str
    tokens: List[str]
    is_question: bool
    is_command: bool
    is_deictic: bool
    is_followup: bool
    token_count: int
    char_count: int
    intents: Set[str]


def analyze_query(q: str) -> QueryAnalysis:
    tokens = keyword_tokens(q)
    intents: Set[str] = set()
    q_is_question = is_question(q)
    q_is_command = is_command(q)
    q_is_deictic = is_deictic(q)
    q_is_follow = is_deictic_followup(q)

    if q_is_question:
        intents.add("question")
    if q_is_command:
        intents.add("command")
    if not intents:
        intents.add("statement")

    return QueryAnalysis(
        text=q or "",
        tokens=tokens,
        is_question=q_is_question,
        is_command=q_is_command,
        is_deictic=q_is_deictic,
        is_followup=q_is_follow,
        token_count=len(tokens),
        char_count=len(q or ""),
        intents=intents,
    )
