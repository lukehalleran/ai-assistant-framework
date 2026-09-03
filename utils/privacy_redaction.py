"""Deterministic redaction for user-shareable debug artifacts.

This module is intentionally dependency-free so API routes and the legacy GUI
can use the same policy.  It targets structured, high-confidence PII rather
than attempting to infer people's names from ordinary prose.
"""

from __future__ import annotations

import copy
import re
from typing import Any


_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Credentials are not strictly PII, but a prompt export is an especially
    # dangerous place to leave a bearer token or labelled password intact.
    (
        re.compile(
            r"(?i)\b(api[_ -]?key|access[_ -]?token|bearer|password|passwd)"
            r"(\s*[:=]\s*)([^\s,;]+)"
        ),
        r"\1\2[REDACTED CREDENTIAL]",
    ),
    (
        re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{16,}\b"),
        "[REDACTED CREDENTIAL]",
    ),
    (
        re.compile(r"(?i)\b(?:home|mailing|street)?\s*address\s*[:=]\s*[^\r\n]+"),
        "address: [REDACTED ADDRESS]",
    ),
    (
        re.compile(r"(?i)\b(?:date\s+of\s+birth|dob)\s*[:=]\s*[^\r\n,;]+"),
        "date of birth: [REDACTED DOB]",
    ),
    (
        re.compile(r"(?i)\b(gtid|student\s*id)(\s*[:#=-]?\s*)\d{5,12}\b"),
        r"\1\2[REDACTED ID]",
    ),
    (
        re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])", re.I),
        "[REDACTED EMAIL]",
    ),
    (
        re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
        "[REDACTED SSN]",
    ),
    # North-American numbers, with an optional country code and common
    # separators.  Word boundaries alone are insufficient around parentheses.
    (
        re.compile(
            r"(?<![\w\d])(?:\+?1[\s.()-]*)?"
            r"\(?\d{3}\)?[\s.-]*\d{3}[\s.-]*\d{4}(?![\w\d])"
        ),
        "[REDACTED PHONE]",
    ),
    # Explicit international numbers. Requiring '+' keeps ordinary long debug
    # counters and timestamps from being swallowed by this broader pattern.
    (
        re.compile(r"(?<![\w\d])\+\d{1,3}(?:[\s.-]?\(?\d{1,4}\)?){2,5}(?![\w\d])"),
        "[REDACTED PHONE]",
    ),
    # GTIDs are nine digits and are frequently pasted without a label.  In a
    # shareable export it is safer to redact any standalone nine-digit ID.
    (
        re.compile(r"(?<!\d)\d{9}(?!\d)"),
        "[REDACTED ID]",
    ),
)


def redact_text(value: Any) -> str:
    """Return text with high-confidence PII and credentials replaced.

    The operation is deterministic and idempotent. Non-string inputs are
    stringified to make it safe at export boundaries.
    """

    text = value if isinstance(value, str) else str(value)
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def redact_data(value: Any) -> Any:
    """Deep-copy a JSON-like value while redacting every string leaf."""

    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {key: redact_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_data(item) for item in value)
    return copy.deepcopy(value)


def build_redacted_prompt_export(record: dict[str, Any], *, include_system: bool) -> str:
    """Build the privacy-safe text used by both prompt-download surfaces."""

    lines = [
        "=" * 80,
        "DAEMON RAG AGENT - FULL PROMPT EXPORT",
        "=" * 80,
        f"Mode: {record.get('mode', 'unknown')}",
        f"Model: {record.get('model', 'unknown')}",
        "=" * 80,
        "",
    ]
    system_prompt = record.get("system_prompt", "")
    if system_prompt and include_system:
        lines += [
            "[SYSTEM PROMPT]", "-" * 80, redact_text(system_prompt),
            "", "=" * 80, "",
        ]
    lines += [
        "[USER QUERY]", "-" * 80, redact_text(record.get("query", "")),
        "", "=" * 80, "",
        "[FULL CONTEXT PROMPT]", "-" * 80,
        redact_text(record.get("prompt", "")), "", "=" * 80,
    ]
    return "\n".join(lines)
