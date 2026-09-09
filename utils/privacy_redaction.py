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
    # Structured profile-field rules (2026-09-08, F11): a rendered profile
    # fact like `birthday=1990-01-02` or `lives_in: Springfield` is not
    # covered by the labelled "date of birth"/"address" rules above.
    (
        re.compile(r"(?i)\b(birthday|birth\s*date|born)\s*[:=]\s*[^\r\n,;]+"),
        r"\1: [REDACTED DOB]",
    ),
    (
        re.compile(
            r"(?i)\b(lives?_in|lives\s+in|location|home\s*town|hometown)"
            r"\s*[:=]\s*[^\r\n,;]+"
        ),
        r"\1: [REDACTED LOCATION]",
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
    # Separators are a SINGLE non-newline char (space/dot/hyphen/paren), not
    # `\s*` (2026-09-04, homework-attachment turn audit item 5): `\s` matches
    # newlines, so an unrelated CSV table's adjacent numeric cells across a
    # line break ("1085\n999  1000") could accidentally line up into a
    # phone-shaped 3-3-4 digit run and get redacted as a real phone number.
    (
        re.compile(
            r"(?<![\w\d])(?:\+?1[ .()-]?)?"
            r"\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?![\w\d])"
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
    # Boundary is alphanumeric, not just digit (2026-09-08, F11): the old
    # `(?<!\d)...(?!\d)` form only excluded adjacent DIGITS, so a nine-digit
    # run embedded in a hex string (a plan_sha256 hash, a git SHA) still
    # matched — `redact_text('abc123456789def')` became
    # `abc[REDACTED ID]def`, corrupting reproducible planner receipts. A
    # bare ID is a STANDALONE TOKEN — bounding on `[0-9A-Za-z]` is a format
    # constraint on the ID itself, not a hash exemption.
    (
        re.compile(r"(?<![0-9A-Za-z])\d{9}(?![0-9A-Za-z])"),
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
    """Build the privacy-safe text used by both prompt-download surfaces.

    F7 (2026-09-08): an agentic record may carry `answer_prompt` — the exact
    prompt the ANSWERING call saw (`controller._build_final_prompt`), which
    can differ from the base retrieval `prompt` this export otherwise shows
    (record 40, 2026-09-08 dump: the base prompt rendered sections the
    answering call's own receipt said were omitted). When present, it is
    rendered FIRST under an "ANSWERING CALL" heading, then the existing
    fields follow under "BASE RETRIEVAL PROMPT". A legacy record without the
    field exports exactly as before.
    """

    lines = [
        "=" * 80,
        "DAEMON RAG AGENT - FULL PROMPT EXPORT",
        "=" * 80,
        f"Mode: {record.get('mode', 'unknown')}",
        f"Model: {record.get('model', 'unknown')}",
        "=" * 80,
        "",
    ]
    answer_prompt = record.get("answer_prompt")
    if answer_prompt:
        answer_call = record.get("answer_call") or "unknown"
        lines += [f"[ANSWERING CALL ({answer_call})]", "-" * 80]
        answer_system_prompt = record.get("answer_system_prompt", "")
        if answer_system_prompt and include_system:
            lines += [
                "[ANSWERING CALL SYSTEM PROMPT]", "-" * 80,
                redact_text(answer_system_prompt),
                "", "-" * 80, "",
            ]
        lines += [
            redact_text(answer_prompt),
            "", "=" * 80, "",
            "[BASE RETRIEVAL PROMPT]",
            "-" * 80,
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
