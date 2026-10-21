"""Evidence boundary for claims about what a person did.

This module deliberately keeps two concerns separate.  ``build_personal_evidence``
normalises the conversation records without changing their speaker, while
``audit_personal_claims`` asks a model to make the open ended semantic
judgement.  The model's source references are then checked mechanically before
they are allowed to affect delivery.

The checker is fail open: an unavailable or invalid review leaves the draft
alone, but its result is explicitly marked unavailable/failed.  In particular,
an assistant suggestion, a quote, a summary, or a tool result is not evidence
that the user completed an outside task.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

# Shared claim-location machinery (2026-09-15): the grounding verifier already
# solved "the model paraphrased the claim" with exact-then-overlap sentence
# location over chunks that concatenate back to the input byte-for-byte.
# One implementation, two consumers (no cycle: grounding_check never imports
# this module).
from core.grounding_check import _locate_claim_sentence, _sentence_chunks
import memory.graph_utils as graph_utils
from utils.read_time_markers import strip_machinery
from utils.trigger_match import normalize_ws
from utils.user_identity import get_user_display_name

logger = logging.getLogger(__name__)

# Same defaults as grounding_check.fallback_claim_overlap_threshold /
# fallback_min_claim_tokens (config.yaml); explicit kwargs keep this pure.
DEFAULT_LOCATE_OVERLAP = 0.8
DEFAULT_LOCATE_MIN_TOKENS = 3


_VALID_STATUSES = frozenset({"supported", "contradicted", "insufficient"})
_RESULT_STATUSES = frozenset({"checked", "unavailable", "failed", "skipped"})
_UNSUPPORTED = frozenset({"contradicted", "insufficient"})

_KINDS = frozenset({
    "completion", "personal_completion", "event", "personal_event",
    "discussion", "plan", "suggestion", "conditional", "negation",
    "partial", "cancellation", "quote", "causal", "obligation", "other",
})


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _timestamp(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except Exception:
            pass
    return _as_text(value)


def _opaque_id(identity: Any, role: str, text: str, ordinal: int = 0) -> str:
    """Return a stable, non-sensitive source identifier.

    Conversation IDs can be human labels in tests or installations.  Hashing
    them keeps the receipt useful for joining records while ensuring it cannot
    disclose a title, path, or private source text.
    """

    # ``ordinal`` remains an argument for compatibility with callers, but the
    # ID is content/source based so reordering a bounded window does not alter
    # provenance.
    raw = f"{_as_text(identity)}\x1f{role}\x1f{text}"
    return "src_" + hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:20]


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, Mapping) else {}


def _row_identity(row: Mapping[str, Any]) -> Any:
    meta = _metadata(row)
    return (
        row.get("source_id")
        if row.get("source_id") not in (None, "")
        else row.get("id")
        if row.get("id") not in (None, "")
        else meta.get("source_id")
        if meta.get("source_id") not in (None, "")
        else meta.get("id")
    )


def _row_timestamp(row: Mapping[str, Any]) -> str | None:
    meta = _metadata(row)
    return _timestamp(
        row.get("timestamp")
        if row.get("timestamp") not in (None, "")
        else row.get("created_at")
        if row.get("created_at") not in (None, "")
        else row.get("date")
        if row.get("date") not in (None, "")
        else meta.get("timestamp")
    )


def _canonical_rows(value: Any) -> list[tuple[str, str | None, str, Any]]:
    """Extract canonical role/text rows from one context value.

    A record with ``user_text`` is authoritative even when it is empty; its
    generic ``text`` field is never recursively parsed as a transcript.  This
    avoids turning pasted ``User:``/``Assistant:`` labels into provenance.
    """

    rows: list[tuple[str, str | None, str, Any]] = []
    if isinstance(value, Mapping):
        identity = _row_identity(value)
        timestamp = _row_timestamp(value)
        meta = _metadata(value)

        # Corpus records commonly store a complete turn as {query, response};
        # semantic retrieval records put those fields under metadata.  A
        # semantic record's ``content`` is a merged/rendered view, so never
        # parse its embedded speaker labels recursively.
        pair = value if ("query" in value or "response" in value) else meta
        if isinstance(pair, Mapping) and ("query" in pair or "response" in pair):
            has_explicit_user = "user_text" in value or "user_text" in meta
            user_value = value.get("user_text") if "user_text" in value else meta.get("user_text")
            if not has_explicit_user:
                user_value = pair.get("query")
            if _as_text(user_value).strip():
                rows.append(("user", timestamp, _as_text(user_value), identity))
            response_value = pair.get("response")
            if response_value is None:
                response_value = value.get("assistant_text", meta.get("assistant_text"))
            if _as_text(response_value).strip():
                rows.append(("assistant", timestamp, _as_text(response_value), identity))
            return rows

        has_user_text = "user_text" in value
        has_assistant_text = "assistant_text" in value
        if has_user_text or has_assistant_text:
            if has_user_text:
                text = _as_text(value.get("user_text"))
                if text.strip():
                    rows.append(("user", timestamp, text, identity))
            if has_assistant_text:
                text = _as_text(value.get("assistant_text"))
                if text.strip():
                    rows.append(("assistant", timestamp, text, identity))
            return rows

        role = value.get("role")
        if role in (None, ""):
            role = value.get("speaker", value.get("author", meta.get("role")))
        role = _as_text(role).lower().strip()
        if role in {"human", "client", "customer"}:
            role = "user"
        if role in {"bot", "daemon", "ai", "model"}:
            role = "assistant"
        text_value = value.get("text") if "text" in value else value.get("content")
        text = _as_text(text_value)
        # A canonical merged conversation row must identify its speaker.  A
        # random memory/profile dict with just a ``text`` field is not evidence.
        if text.strip() and role in {"user", "assistant"}:
            rows.append((role, timestamp, text, identity))
        return rows

    # Plain strings may occur in a legacy context.  Keep them visible as
    # unknown context, where the auditor cannot use them to prove a completion.
    if isinstance(value, str) and value.strip():
        rows.append(("unknown", None, value, None))
    return rows


def _iter_context(context: Any, history: Iterable[Any]) -> list[Any]:
    values: list[Any] = []
    if isinstance(context, Mapping):
        # Only conversation-shaped sections are admitted.  Profiles, facts,
        # summaries, narratives, and tool output are intentionally excluded.
        keys = (
            "recent_conversations",
            "recent",
            "relevant_conversations",
            "relevant_memories",
            "memories",
            "conversation_history",
            "history",
        )
        for key in keys:
            section = context.get(key)
            if isinstance(section, (list, tuple)):
                values.extend(section)
            elif isinstance(section, Mapping) or isinstance(section, str):
                values.append(section)
    elif isinstance(context, (list, tuple)):
        values.extend(context)
    elif context is not None:
        values.append(context)
    if history:
        values.extend(list(history))
    return values


ATTACHMENT_EVIDENCE_SHARE = 0.4


# ONE definition: the builder writes it and audit_personal_claims detects it
# (2026-10-17 referee: the detector looked for "[…truncated]" with a Unicode
# ellipsis, so evidence_truncated never fired).
TRUNCATION_MARKER = " [...truncated]"


def build_personal_evidence(
    query: str,
    context: Any,
    history: Iterable[Any] = (),
    *,
    max_chars: int = 12000,
    attachments: Sequence[tuple[str, str]] = (),
) -> list[dict]:
    """Build bounded, role-preserved evidence for one draft.

    ``attachments`` are this turn's (display name, text) documents; each
    becomes a role ``"attachment"`` row (``src_attachment_<n>``) capped to a
    share of at most ``ATTACHMENT_EVIDENCE_SHARE`` of ``max_chars`` across all
    of them. An attachment is document text, never user evidence.

    The current query is always the first entry.  Entries retain unknown
    timestamps as ``None`` and are selected in a stable order, with the latest
    recognisable user correction promoted ahead of less relevant history.
    """

    budget = max(int(max_chars), 1)
    current = _as_text(query)
    entries: list[dict] = []

    def add(role: str, timestamp: str | None, text: str, identity: Any, ordinal: int) -> None:
        nonlocal budget
        if not text.strip() or budget <= 0:
            return
        # Keep a truthful marker when a single query/record exceeds the bound.
        if len(text) > budget:
            marker = TRUNCATION_MARKER
            text = text[: max(1, budget - len(marker))] + marker
        if len(text) > budget:
            return
        if identity == "__current_query__":
            source_id = "src_current_query"
        elif isinstance(identity, str) and identity.startswith("__attachment_"):
            source_id = "src_attachment_" + identity[len("__attachment_"):].rstrip("_")
        else:
            source_id = _opaque_id(identity, role, text, ordinal)
        item = {"source_id": source_id, "role": role, "timestamp": timestamp, "text": text}
        entries.append(item)
        budget -= len(text)

    add("user", None, current, "__current_query__", 0)
    if budget <= 0:
        return entries

    docs = [(str(n or ""), _as_text(t)) for n, t in (attachments or ()) if _as_text(t).strip()]
    if docs:
        share = max(int(int(max_chars) * ATTACHMENT_EVIDENCE_SHARE), 1)
        share = min(share, budget)
        per_doc = max(share // len(docs), 1)
        for number, (name, doc_text) in enumerate(docs, 1):
            before = len(entries)
            if len(doc_text) > per_doc:
                marker = TRUNCATION_MARKER
                doc_text = doc_text[: max(1, per_doc - len(marker))] + marker
            add("attachment", None, doc_text, f"__attachment_{number}__", number)
            if name and len(entries) > before:
                entries[-1]["name"] = name

    candidates: list[tuple[int, int, str, str | None, str, Any]] = []
    rows = _iter_context(context, history)
    correction_indexes: list[int] = []
    flattened: list[tuple[str, str | None, str, Any]] = []
    for value in rows:
        flattened.extend(_canonical_rows(value))
    # One speaker saying one text is ONE source, whatever record carried it:
    # the API chat history repeats the corpus rows without ids/timestamps.
    # Rows are pre-sorted so a timestamped copy wins over a bare one.
    seen: set[tuple[str, str]] = set()
    ordered = sorted(enumerate(flattened), key=lambda item: (item[1][1] is None, item[0]))
    kept: list[tuple[int, str, str | None, str, Any]] = []
    for index, (role, stamp, text, identity) in ordered:
        key = (role, text.strip())
        if key in seen:
            continue
        seen.add(key)
        kept.append((index, role, stamp, text, identity))
    kept.sort(key=lambda item: item[0])
    for index, role, stamp, text, identity in kept:
        candidates.append((0, index, role, stamp, text, identity))
    # Records from retrieval are often newest-first.  Select newest timestamped
    # rows first to keep later corrections within the budget, then render the
    # retained set chronologically so episode order remains intelligible.
    selection_order = sorted(
        candidates,
        key=lambda item: (item[3] is not None, item[3] or "", item[1]),
        reverse=True,
    )
    selected: list[tuple[int, int, str, str | None, str, Any]] = []
    remaining = budget
    for candidate in selection_order:
        text = candidate[4]
        if remaining <= 0:
            break
        if len(text) <= remaining or len(selected) == 0:
            selected.append(candidate)
            remaining -= min(len(text), remaining)
    selected.sort(key=lambda item: (item[3] is not None, item[3] or "", item[1]))
    for ordinal, (_priority, _index, role, stamp, text, identity) in enumerate(selected, 1):
        add(role, stamp, text, identity, ordinal)
        if budget <= 0:
            break
    return entries


_SYSTEM_PROMPT = """You audit personal claims in an assistant draft against supplied conversation evidence.
Return one strict JSON object with exactly this shape: {\"claims\":[{\"text\":string,\"status\":\"supported\"|\"contradicted\"|\"insufficient\",\"kind\":string,\"evidence\":[{\"source_id\":string,\"quote\":string}]}]}.
Find every material claim IN THE DRAFT about what the user or a person in the user's own life did, including novel actions; do not use a closed action-verb list. Also classify plans, conditional/partial/negated/cancelled claims, quoted claims, discussion claims, and claims about another episode or object. The "text" field MUST be a character-exact copy of a contiguous span of the DRAFT: never paraphrase, never prefix it with "User", never restate the user's own message as a claim (the user's message is evidence, not a claim). If you cannot copy the span exactly, omit the claim.
The "kind" field is one of: "completion" (the user or a person in the user's own life did something), "plan" (a plan the USER or a person in the user's own life holds or stated), "suggestion" (something the ASSISTANT proposes, offers or recommends doing next, including "we can...", "let's...", "you could...", "when you're back we..."), "discussion" (what was said or covered in the conversation), "quote", "causal", "subject" (a statement about a subject, a document, software, course material or the world that says nothing about what the user or another person did, holds, plans or is), "other". A "person in the user's own life" is someone the user or the conversation refers to as known to them; organisations, companies, projects, software and its releases, authors of cited sources, public figures, and public or historical events are subject matter, so claims about what they did are kind "subject".
Evidence is source-backed context, not a substitute for semantic entailment. A user report can support a user's completion. An assistant suggestion, assistant summary, quote, generated narrative, profile/fact, or tool text cannot establish that the user completed an external task. Assistant-origin evidence may support only a claim about the discussion itself. Prefer a newer direct user correction over an earlier report. Evidence rows with role "attachment" are the text of files the user attached this turn: they support claims about that document's content, never claims about what the user did. A question or an offer in the draft is not a claim; do not return it. Subject-matter statements are not personal claims and should not be returned; if one is returned it must have kind "subject". An assistant-proposed next step may be returned only with kind "suggestion".
Return at most 8 claims (the most material), and keep each evidence quote to at most 25 words.
Ambiguous, missing, or incomplete evidence is insufficient, never a negative fact. Cite exact contiguous source quotes and their source IDs; cite no source for an insufficient claim when no exact span supports it.
"""


def _review_prompt(response: str, evidence: Sequence[Mapping[str, Any]]) -> str:
    evidence_json = json.dumps(list(evidence), ensure_ascii=False, separators=(",", ":"))
    return (
        "Review this assistant draft. Treat evidence entries as records with preserved roles.\n\n"
        "EVIDENCE:\n" + evidence_json + "\n\nDRAFT:\n" + response
    )


_FENCE_RE = re.compile(r"\A```[A-Za-z0-9_-]*[ \t]*\r?\n(.*?)\r?\n?```\Z", re.DOTALL)


def _unfence(text: str) -> str:
    """Strip ONE surrounding Markdown code fence; anything else is untouched.

    gpt-4o-mini wraps strict-JSON answers in ```json fences on some prompts
    (live 2026-09-15: 78 completion tokens rejected as "not strict JSON" while
    the offline replay returned bare JSON). A fence is presentation, not
    content; the schema check below is still exact. Prose around JSON is
    still rejected -- this is not a "find the first brace" extraction.
    """
    match = _FENCE_RE.match(text)
    return match.group(1).strip() if match else text


def _strict_json(raw: Any) -> dict:
    if not isinstance(raw, str):
        raise ValueError("shape=not_text")
    text = raw.strip()
    if not text:
        raise ValueError("shape=empty")
    fenced = text.startswith("```")
    text = _unfence(text)
    if text.startswith("```"):
        raise ValueError(f"shape=fenced_unclosed len={len(raw)}")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        shape = "fenced_not_json" if fenced else ("prose" if text[:1] not in "{[" else "invalid_json")
        if shape == "invalid_json" and (
            exc.pos >= len(text) - 2 or exc.msg.startswith("Unterminated string")
        ):
            # The decode error sits at the very end -- or is a string that ran
            # to the end of the text (the parser reports the string's START):
            # the output was cut off (completion == max_tokens in the
            # 2026-10-17 live audit).
            shape = "output_truncated"
        raise ValueError(f"shape={shape} len={len(raw)} at={exc.pos}") from None
    if not isinstance(parsed, dict) or set(parsed) != {"claims"}:
        keys = sorted(parsed)[:6] if isinstance(parsed, dict) else type(parsed).__name__
        raise ValueError(f"shape=wrong_keys keys={keys}")
    if not isinstance(parsed["claims"], list):
        raise ValueError("shape=claims_not_list")
    return parsed


def _is_completion_kind(kind: str) -> bool:
    lowered = kind.lower().replace("-", "_")
    return (
        "completion" in lowered
        or "completed" in lowered
        or lowered in {"event", "personal_event", "user_event", "achievement"}
    )


# Omission scope (2026-09-27, BC-46/BC-84; S10 roll-up): claims that assert
# nothing about what the user did or has — discussion, advice, quotes, causal
# explanation — are recorded but never excised. Everything else (completion,
# event, plan, cancellation, …: statements about the user's own state) stays
# omission-eligible — the 09-15 incident was a PLAN claim ("a cover letter
# waiting for tomorrow"). "other" stays eligible: the auditor uses it for
# personal-state sentences too ("nothing left today that can't wait").
# "subject" (2026-10-17, BC-84): course-material / software / world statements.
_NON_PERSONAL_KINDS = frozenset({"discussion", "suggestion", "quote", "causal", "subject"})


def _omission_eligible_kind(kind: str) -> bool:
    tokens = [t for t in re.split(r"[\s/_,-]+", kind.lower()) if t]
    return bool(tokens) and not all(t in _NON_PERSONAL_KINDS for t in tokens)


# Repetition-count grammar (2026-10-17, BC-46/BC-75/BC-84). These tables are
# closed grammatical categories of English (number words and the frames that
# make a number a repetition count), not vocabulary from any incident.
_ORDINAL_WORDS = {
    "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
_CARDINAL_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_ORDINAL_FRAMES = ("in a row", "straight", "consecutive", "running", "this week", "again")
_CARDINAL_FRAMES = ("in a row", "straight", "consecutive", "running")
# "a second" / "one second" / "per second" is the time unit (or "another"),
# never an ordinal count.
_ORDINAL_COUNT_RE = re.compile(
    r"(?<!\ba )(?<!\bone )(?<!\bper )"
    r"\b(" + "|".join(_ORDINAL_WORDS) + r"|(?:[2-9]|10)(?:nd|rd|th))\b", re.IGNORECASE)
_CARDINAL_COUNT_RE = re.compile(
    r"\b(" + "|".join(_CARDINAL_WORDS) + r"|(?:[2-9]|10))\s+(?:[a-z'-]+\s+){0,2}?[a-z'-]+s\s+(?:"
    + "|".join(re.escape(f) for f in _CARDINAL_FRAMES) + r")\b", re.IGNORECASE)


def claim_repetition_count(text: str) -> int | None:
    """N (>= 2) when ``text`` asserts a repetition count, else None."""
    lowered = _as_text(text).lower()
    found: list[int] = []
    if any(re.search(r"\b" + re.escape(f) + r"\b", lowered) for f in _ORDINAL_FRAMES):
        for match in _ORDINAL_COUNT_RE.finditer(lowered):
            token = match.group(1)
            found.append(_ORDINAL_WORDS.get(token) or int(token[:-2]))
    for match in _CARDINAL_COUNT_RE.finditer(lowered):
        token = match.group(1)
        found.append(_CARDINAL_WORDS.get(token) or int(token))
    return max(found) if found else None


_DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _distinct_user_days(checked_refs: Sequence[Mapping[str, Any]],
                        sources: Mapping[str, list[Mapping[str, Any]]]) -> int:
    """Distinct calendar days among the cited USER-role sources.

    The current query is today (one day); a timestamp-less user row is no day.
    """
    days: set[str] = set()
    for ref in checked_refs:
        source_id = ref["source_id"]
        for row in sources.get(source_id, ()):
            if row.get("role") != "user":
                continue
            if source_id == "src_current_query":
                days.add("__today__")
                continue
            match = _DATE_PREFIX_RE.match(_as_text(row.get("timestamp")))
            if match:
                days.add(match.group(0))
    return len(days)


class _DropClaim(ValueError):
    """This claim cannot be used; the rest of the audit still can."""


class _MalformedClaim(_DropClaim):
    """The model broke the schema for this claim (shape/enum/empty text)."""


_STATUS_RANK = {"supported": 0, "insufficient": 1, "contradicted": 2}


def _locate_span(response: str, text: str, *, overlap_threshold: float,
                 min_claim_tokens: int) -> tuple[str | None, str]:
    """Map a model claim onto an EXACT span of the draft.

    Exact substring first. Otherwise the grounding verifier's locator finds
    the single sentence whose content tokens cover ``overlap_threshold`` of
    the claim's (live 2026-09-15: gpt-4o-mini wrote "User has a cover letter
    waiting for a fresher brain tomorrow." for the draft's "Tomorrow's got
    the cover letter waiting for a fresher brain."). Ambiguous or missing →
    ``None``: a restatement of the user's own message has no draft sentence
    and is dropped, which is the correct outcome.
    """
    if text in response:
        return text, "exact"
    chunks = _sentence_chunks(response)
    index, reason = _locate_claim_sentence(
        chunks, text, overlap_threshold=overlap_threshold, min_claim_tokens=min_claim_tokens,
    )
    if index is None:
        return None, reason
    span = chunks[index].strip()
    if not span or span not in response:
        return None, "claim_not_located"
    return span, "relocated"


# Diagnostics only (2026-10-18/19/20, BC-84): WHY a cited quote is not an exact
# span. One small fold table: typographic quotes/dashes/ellipsis -> ASCII.
_TYPOGRAPHY_FOLD = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u2026": "...", "\u00a0": " ",
})
QUOTE_MISS_PREFIX = "source quote is not exact"
QUOTE_MISS_CATEGORIES = ("marker", "whitespace_typography", "case", "other_source", "absent")


def _fold_typography(text: str) -> str:
    return normalize_ws(text.translate(_TYPOGRAPHY_FOLD))


def _classify_quote_miss(
    quote: str, source_id: str, sources: Mapping[str, list[Mapping[str, Any]]],
) -> str:
    """Category of a quote that is NOT an exact span of its cited rows.

    Pure diagnostics: the caller drops the reference either way. First
    applicable category wins; see QUOTE_MISS_CATEGORIES.
    """
    rows = [_as_text(row.get("text")) for row in sources.get(source_id, ())]
    folded_quote = _fold_typography(quote)
    folded_rows = [_fold_typography(r) for r in rows]
    stripped_rows = [strip_machinery(r) for r in rows]
    if strip_machinery(quote) != quote or any(quote in s for s in stripped_rows):
        return "marker"
    if folded_quote and not any(folded_quote in r for r in folded_rows) and any(
            folded_quote in _fold_typography(s) for s in stripped_rows):
        return "marker"  # machinery in the row AND a wrapped/typographic quote
    if folded_quote and any(folded_quote in r for r in folded_rows):
        return "whitespace_typography"
    if folded_quote and any(folded_quote.casefold() in r.casefold() for r in folded_rows):
        return "case"
    for other_id, other_rows in sources.items():
        if other_id != source_id and any(quote in _as_text(r.get("text")) for r in other_rows):
            return "other_source"
    return "absent"


def quote_miss_summary(drops: Iterable[str]) -> dict[str, int]:
    """Count quote-miss categories in a drop-reason list (no quote text)."""
    tally: dict[str, int] = {}
    for reason in drops:
        if not isinstance(reason, str) or not reason.startswith(QUOTE_MISS_PREFIX + " ("):
            continue
        category = reason[len(QUOTE_MISS_PREFIX) + 2:].rstrip(")")
        tally[category] = tally.get(category, 0) + 1
    return tally


def _checked_references(
    refs: Any, sources: Mapping[str, list[Mapping[str, Any]]], drops: list[str],
) -> list[dict]:
    """Keep only references whose quote is an exact span of the cited source.

    A bad reference is DROPPED, never repaired -- dropping evidence can only
    move a claim toward ``insufficient``, so this is the conservative
    direction (BC-84: one invalid element must not discard the valid ones).
    """
    if not isinstance(refs, list):
        drops.append("evidence is not a list")
        return []
    checked: list[dict] = []
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {"source_id", "quote"}:
            drops.append("unexpected evidence shape")
            continue
        source_id, quote = ref["source_id"], ref["quote"]
        if not isinstance(source_id, str) or not isinstance(quote, str) or not quote:
            drops.append("invalid source reference")
            continue
        rows = sources.get(source_id)
        if not rows:
            drops.append("unknown source id")
            continue
        if not any(quote in _as_text(row.get("text")) for row in rows):
            drops.append(f"{QUOTE_MISS_PREFIX} ({_classify_quote_miss(quote, source_id, sources)})")
            continue
        checked.append({"source_id": source_id, "quote": quote})
    return checked


_TRAILING_CLOSERS = " \t\r\n\"'\u201d\u2019)]}>*_`"

# Assistant-proposal grammar (2026-10-17, BC-84). Closed grammatical categories
# of English, not vocabulary from any incident: the span must OPEN with a
# first-person subject governed by a future/ability/deontic modal, or with the
# "let's" hortative, optionally after one leading temporal/conditional clause
# ("when you're back we can ..."). Anchored at the span start on purpose: a
# proposal later in the span ("Since you uploaded the resume, I can tailor the
# letter") still carries a claim about the user and must stay audited.
# "would"/"'d" are excluded: "I'd say you finished it" is an assessment, not an
# offer. Subject "you", and first-person clauses with no modal, never match.
_PROPOSAL_LEAD = (
    r"(?:(?:when|whenever|if|once|after|before|until|while|as\s+soon\s+as)\b"
    r"[^.;:!?]*?[,\s]\s*)?"
)
_PROPOSAL_HEAD = (
    r"(?:let['\u2019]?s\b|let\s+us\b"
    r"|(?:we|i)\s+(?:[a-z]+ly\s+)?(?:can|could|will|should|might|may|shall)\b(?!\s*n['\u2019]t)"
    r"|(?:we|i)['\u2019]ll\b)"
)
_PROPOSAL_RE = re.compile(r"\A[\s\"'\u201c\u2018(*_]*" + _PROPOSAL_LEAD + _PROPOSAL_HEAD, re.IGNORECASE)


def _is_assistant_proposal(span: str) -> bool:
    return _PROPOSAL_RE.match(span) is not None


# Subject-matter backstop (2026-10-17, BC-84/BC-75). Closed grammatical
# categories of English -- personal pronouns/possessives and temporal
# reference -- not vocabulary from any incident.
_PERSON_REF_RE = re.compile(
    r"\b(?:i|me|my|mine|myself|we|us|our|ours|ourselves|you|your|yours|yourself|"
    r"yourselves|he|him|his|himself|she|her|hers|herself|they|them|their|theirs|"
    r"themselves|user)\b",
    re.IGNORECASE,
)
_CLOCK_TIME_RES = (
    re.compile(r"\b\d{1,2}:\d{2}\b"),
    re.compile(r"\b\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)\b", re.IGNORECASE),
    re.compile(r"\b(?:at|by|until|till|from|around|before|after)\s+\d{1,2}\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}(?:st|nd|rd|th)\b", re.IGNORECASE),
)
_TIME_WINDOW_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _has_time_reference(span: str) -> bool:
    if any(rx.search(span) for rx in _CLOCK_TIME_RES):
        return True
    tokens = _TIME_WINDOW_TOKEN_RE.findall(span)
    for size in (1, 2, 3):
        for start in range(len(tokens) - size + 1):
            if graph_utils.is_temporal_deictic(" ".join(tokens[start:start + size])):
                return True
    return False


def _is_subject_matter_span(span: str, kind: str) -> bool:
    """True when a non-completion/plan claim names no person and no time."""
    if _is_completion_kind(kind):
        return False
    tokens = [t for t in re.split(r"[\s/_,-]+", kind.lower()) if t]
    if any(t in ("plan", "cancel") or t.startswith("cancel") for t in tokens):
        return False
    if _PERSON_REF_RE.search(span):
        return False
    name = (get_user_display_name() or "").strip()
    if name and re.search(r"\b" + re.escape(name) + r"\b", span, re.IGNORECASE):
        return False
    return not _has_time_reference(span)


def _kind_status_summary(claims: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    tally: dict[str, int] = {}
    for claim in claims:
        key = f"{_as_text(claim.get('kind')).strip().lower()}:{claim.get('status')}"
        tally[key] = tally.get(key, 0) + 1
    return tally


def _validate_claims(
    response: str, payload: dict, evidence: Sequence[Mapping[str, Any]], *,
    overlap_threshold: float = DEFAULT_LOCATE_OVERLAP,
    min_claim_tokens: int = DEFAULT_LOCATE_MIN_TOKENS,
) -> tuple[list[dict], dict[str, int], list[str]]:
    """Validate the model's claims one at a time.

    Returns ``(claims, counts, drop_reasons)``. A claim that is not an exact
    response span (or is malformed/duplicated) is dropped and counted; a bad
    evidence reference is dropped and counted; a ``supported`` or
    ``contradicted`` claim left without evidence -- or a supported
    completion-kind claim without USER-role evidence -- is demoted to
    ``insufficient`` and counted. Nothing is ever promoted.
    """
    sources: dict[str, list[Mapping[str, Any]]] = {}
    for row in evidence:
        source_id = row.get("source_id")
        if isinstance(source_id, str):
            sources.setdefault(source_id, []).append(row)
    claims: list[dict] = []
    by_span: dict[str, dict] = {}
    counts = {"dropped_claims": 0, "dropped_evidence": 0, "demoted": 0, "relocated": 0,
              "malformed": 0, "question_drops": 0, "proposal_drops": 0,
              "subject_drops": 0}
    drops: list[str] = []
    for raw_claim in payload["claims"]:
        try:
            if not isinstance(raw_claim, dict) or set(raw_claim) != {"text", "status", "kind", "evidence"}:
                raise _MalformedClaim("unexpected claim shape")
            text, status, kind = raw_claim["text"], raw_claim["status"], raw_claim["kind"]
            if status not in _VALID_STATUSES or not isinstance(kind, str) or not kind.strip():
                raise _MalformedClaim("invalid claim enum")
            if not isinstance(text, str) or not text.strip():
                raise _MalformedClaim("empty claim text")
            located, how = _locate_span(
                response, text, overlap_threshold=overlap_threshold,
                min_claim_tokens=min_claim_tokens,
            )
            if located is None:
                # A semantic miss, not a broken verdict: the model audited
                # something that is not in the draft (live 2026-09-15 20:29:
                # it restated the user's own message). Dropping it IS the
                # correct verdict for that claim.
                raise _DropClaim(f"claim is not an exact response span ({how})")
        except _MalformedClaim as exc:
            counts["dropped_claims"] += 1
            counts["malformed"] += 1
            drops.append(str(exc))
            continue
        except _DropClaim as exc:
            counts["dropped_claims"] += 1
            drops.append(str(exc))
            continue
        if how == "relocated":
            counts["relocated"] += 1
        text = located
        if text.rstrip(_TRAILING_CLOSERS).endswith("?"):
            # A question is not a claim (live 2026-10-17: "What's 6 covering?"
            # was audited as insufficient). Mechanical, model-independent.
            counts["question_drops"] += 1
            drops.append("claim span is a question")
            continue
        if _is_assistant_proposal(text):
            # An assistant-proposed next step ("when you're back we can ...")
            # is not a claim about the user, whatever kind the auditor gave it.
            counts["proposal_drops"] += 1
            drops.append("claim span is an assistant proposal")
            continue
        if text in by_span:
            # Two claims landed on one sentence: keep ONE claim per span and
            # let the status move only in the conservative direction.
            existing = by_span[text]
            if _STATUS_RANK[status] > _STATUS_RANK[existing["status"]]:
                existing["status"] = status
                existing["kind"] = kind.strip()
            drops.append("duplicate claim span (merged)")
            continue
        ref_drops: list[str] = []
        checked_refs = _checked_references(raw_claim["evidence"], sources, ref_drops)
        counts["dropped_evidence"] += len(ref_drops)
        drops.extend(ref_drops)
        if status in ("supported", "contradicted") and not checked_refs:
            # A verdict either way needs a surviving exact quote. A
            # "contradicted" with none stayed contradicted until 2026-10-18
            # (live: 5 recall claims contradicted, 0 demoted, their quotes
            # dropped as inexact; a replay of the same draft returned all
            # 5 "supported"). Same flag either way; the receipt now tells
            # the truth about what the evidence showed.
            status = "insufficient"
            counts["demoted"] += 1
        elif status == "supported" and _is_completion_kind(kind):
            user_sources = {
                ref["source_id"]
                for ref in checked_refs
                if any(row.get("role") == "user" for row in sources.get(ref["source_id"], ()))
            }
            if not user_sources:
                # Assistant discussion/advice cannot corroborate a user event.
                status = "insufficient"
                counts["demoted"] += 1
        if status == "supported" and _omission_eligible_kind(kind):
            # Rule 1: a personal claim resting ONLY on assistant-role rows is
            # the assistant's own earlier line quoted back (BC-75). Only the
            # "assistant" role counts: profile/summary/unknown rows stay.
            rows = [row for ref in checked_refs for row in sources.get(ref["source_id"], ())]
            if rows and all(row.get("role") == "assistant" for row in rows):
                status = "insufficient"
                counts["demoted"] += 1
            else:
                # Rule 2: "the Nth ... in a row" needs N distinct user days.
                repeated = claim_repetition_count(text)
                if repeated is not None and _distinct_user_days(checked_refs, sources) < repeated:
                    status = "insufficient"
                    counts["demoted"] += 1
        if (status in _UNSUPPORTED and _omission_eligible_kind(kind)
                and _is_subject_matter_span(text, kind)):
            # A would-be-flagged claim naming no person and no time is about a
            # subject, not a person (live 2026-10-17: course-material
            # explanations were flagged and re-rendered as "do not repeat it
            # as fact"). Supported / non-personal-kind claims are untouched:
            # nothing is flagged for them, and receipts keep their counts.
            counts["subject_drops"] += 1
            drops.append("claim span names no person and no time")
            continue
        claim = {"text": text, "status": status, "kind": kind.strip(), "evidence": checked_refs}
        by_span[text] = claim
        claims.append(claim)
    return claims, counts, drops


FLAGGED_SPAN_MAX_CHARS = 240
FLAGGED_SPAN_MAX_COUNT = 3


def _flagged_spans(response: str, claims: Sequence[Mapping[str, Any]]) -> list[str]:
    """Span text of each contradicted/insufficient personal-kind claim.

    Position-ordered, newline-normalised, cut (at whitespace) to a prefix of
    at most FLAGGED_SPAN_MAX_CHARS, deduped, at most FLAGGED_SPAN_MAX_COUNT.
    The cut span is still a substring of the normalised response.
    """
    found: list[tuple[int, str]] = []
    for claim in claims:
        if claim.get("status") not in _UNSUPPORTED:
            continue
        if not _omission_eligible_kind(_as_text(claim.get("kind"))):
            continue
        text = claim.get("text")
        if not isinstance(text, str):
            continue
        position = response.find(text)
        span = text.replace("\n", " ").strip()
        if len(span) > FLAGGED_SPAN_MAX_CHARS:
            cut = span.rfind(" ", 0, FLAGGED_SPAN_MAX_CHARS)
            span = span[:cut if cut > 0 else FLAGGED_SPAN_MAX_CHARS].rstrip()
        if span:
            found.append((position if position >= 0 else len(response), span))
    ordered: list[str] = []
    for _, span in sorted(found, key=lambda item: item[0]):
        if span not in ordered:
            ordered.append(span)
    return ordered[:FLAGGED_SPAN_MAX_COUNT]


@dataclass
class PersonalClaimResult:
    status: str
    reason: str
    claims: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0
    evidence_truncated: bool = False
    dropped_claim_count: int = 0
    dropped_evidence_count: int = 0
    demoted_count: int = 0
    relocated_count: int = 0
    question_drops: int = 0
    proposal_drops: int = 0
    subject_drops: int = 0
    # Located response spans of the unsupported PERSONAL-kind claims, ordered by
    # position (class: BC-97): the receipt names WHICH claim was flagged.
    # None = not computed by the caller (a result built outside
    # audit_personal_claims): receipt() derives the spans from ``claims`` in
    # claim order rather than emitting [] -- an empty list MEANS "nothing
    # personal was flagged" and suppresses the marker (referee 2026-10-14).
    flagged_spans: list[str] | None = None

    def __post_init__(self) -> None:
        if self.status not in _RESULT_STATUSES:
            raise ValueError(f"unknown result status: {self.status}")

    def receipt(self) -> dict:
        referenced = []
        seen: set[str] = set()
        for claim in self.claims:
            for ref in claim.get("evidence", ()):
                source_id = ref.get("source_id") if isinstance(ref, dict) else None
                if isinstance(source_id, str) and source_id not in seen:
                    seen.add(source_id)
                    referenced.append(source_id)
        counts = {status: sum(1 for c in self.claims if c.get("status") == status) for status in _VALID_STATUSES}
        result = {
            "status": self.status,
            "reason": self.reason,
            "candidate_count": len(self.claims),
            "supported_count": counts["supported"],
            "contradicted_count": counts["contradicted"],
            "insufficient_count": counts["insufficient"],
            "source_ids": referenced,
            "elapsed_s": round(float(self.elapsed_s), 3),
            "dropped_claim_count": int(self.dropped_claim_count),
            "dropped_evidence_count": int(self.dropped_evidence_count),
            "demoted_count": int(self.demoted_count),
            "relocated_count": int(self.relocated_count),
            "question_drops": int(self.question_drops),
            "proposal_drops": int(self.proposal_drops),
            "subject_drops": int(self.subject_drops),
        }
        if self.evidence_truncated:
            result["evidence_truncated"] = True
        if self.status == "checked":
            spans = self.flagged_spans
            if spans is None:
                spans = _flagged_spans("", self.claims)
            result["flagged_spans"] = list(spans)
        return result


async def audit_personal_claims(
    response: str,
    evidence: Sequence[Mapping[str, Any]],
    model_manager: Any,
    *,
    model_name: str | None = None,
    # 2026-09-27 (BC-47): matches config.yaml personal_claim_check.timeout_s,
    # which callers always pass explicitly -- this default only documents the
    # measured budget for a direct call. S10/personal_claim_rollup.md (378
    # production checks against the code's actual model default, gpt-4o-mini)
    # found the checked-only p90 was already 4.711s against the old 5.0s
    # ceiling -- not a long tail, the typical successful call -- while 37.3%
    # of ALL checks timed out outright (pure waste, zero signal). 6.5s is
    # that measured p90 plus a ~1.8s margin.
    timeout_s: float = 6.5,
    max_tokens: int = 1600,
    overlap_threshold: float = DEFAULT_LOCATE_OVERLAP,
    min_claim_tokens: int = DEFAULT_LOCATE_MIN_TOKENS,
) -> PersonalClaimResult:
    """Semantically review ``response`` and mechanically validate its JSON."""

    started = perf_counter()
    if not response or not response.strip():
        return PersonalClaimResult("skipped", "empty_response", elapsed_s=perf_counter() - started)
    if not evidence:
        return PersonalClaimResult("unavailable", "no_evidence", elapsed_s=perf_counter() - started)
    if model_manager is None or not callable(getattr(model_manager, "generate_once", None)):
        return PersonalClaimResult("unavailable", "no_model", elapsed_s=perf_counter() - started)
    truncated = any(TRUNCATION_MARKER.strip() in _as_text(row.get("text")) for row in evidence)
    try:
        raw = await asyncio.wait_for(
            model_manager.generate_once(
                _review_prompt(response, evidence),
                model_name=model_name,
                system_prompt=_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=0.0,
                disable_reasoning=True,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        return PersonalClaimResult("unavailable", "timeout", elapsed_s=perf_counter() - started,
                                   evidence_truncated=truncated)
    except asyncio.CancelledError:
        raise
    except Exception:
        return PersonalClaimResult("failed", "provider_error", elapsed_s=perf_counter() - started,
                                   evidence_truncated=truncated)
    try:
        payload = _strict_json(raw)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        # Constant-string reason only; the raw output is never logged.
        truncated_output = "shape=output_truncated" in str(exc)
        logger.debug(f"[PersonalClaim] {'output_truncated' if truncated_output else 'invalid_json'}: {exc}")
        return PersonalClaimResult("failed", "output_truncated" if truncated_output else "invalid_json", elapsed_s=perf_counter() - started,
                                   evidence_truncated=truncated)
    claims, counts, drops = _validate_claims(
        response, payload, evidence,
        overlap_threshold=overlap_threshold, min_claim_tokens=min_claim_tokens,
    )
    if drops:
        logger.debug(f"[PersonalClaim] dropped {counts['dropped_claims']} claim(s), "
                     f"{counts['dropped_evidence']} reference(s): {sorted(set(drops))}")
    misses = quote_miss_summary(drops)
    if misses:
        logger.debug("[PersonalClaim] quote misses: " + json.dumps(misses, sort_keys=True))
    logger.debug("[PersonalClaim] kinds: " + json.dumps(_kind_status_summary(claims), sort_keys=True))
    if payload["claims"] and not claims and counts["malformed"]:
        # The model broke the schema and nothing survived: nothing was checked.
        return PersonalClaimResult("failed", "invalid_verdict", elapsed_s=perf_counter() - started,
                                   evidence_truncated=truncated,
                                   dropped_claim_count=counts["dropped_claims"],
                                   dropped_evidence_count=counts["dropped_evidence"])
    # Well-formed claims that were all semantic misses (restatements of the
    # user's message, nothing in the draft) are a CHECKED audit with no
    # candidates -- the counts say what was dropped.
    reason = "ok" if claims or not payload["claims"] else "no_claims"
    return PersonalClaimResult("checked", reason, claims=claims, elapsed_s=perf_counter() - started,
                               evidence_truncated=truncated,
                               dropped_claim_count=counts["dropped_claims"],
                               dropped_evidence_count=counts["dropped_evidence"],
                               demoted_count=counts["demoted"],
                               relocated_count=counts["relocated"],
                               question_drops=counts["question_drops"],
                               proposal_drops=counts["proposal_drops"],
                               subject_drops=counts["subject_drops"],
                               flagged_spans=_flagged_spans(response, claims))


_SENTENCE_END_RE = re.compile(r"[.!?]+(?:[\"'”’»\)\]]+)?(?:\s+|$)|\n{2,}")


def _sentence_span(text: str, start: int, end: int) -> tuple[int, int]:
    left_match = list(_SENTENCE_END_RE.finditer(text, 0, start))
    left = left_match[-1].end() if left_match else 0
    right_match = _SENTENCE_END_RE.search(text, end)
    right = right_match.end() if right_match else len(text)
    return left, right


def omit_unsupported_claims(response: str, result: PersonalClaimResult) -> str:
    """Remove whole sentences containing exact unsupported personal claim spans.

    Omission scope is completion-kind only (2026-09-27, BC-46/BC-84/BC-47;
    S10/personal_claim_rollup.md, 378 production checks): 79% of successfully
    checked replies had >=1 unsupported claim, but a 10-sample manual review
    found 7/10 were discussion/advice/explanatory sentences (a news-story
    opinion, interview-prep advice, a statistics explanation) that the
    auditor's necessarily broad system prompt classifies as claims about
    "another person" or "another episode" even though they assert nothing
    about what the user did -- ``_is_completion_kind`` already isolates
    exactly this distinction for the demotion path above, so it gates
    omission too. A discussion/plan/suggestion/other-kind claim is still
    recorded on the receipt (contradicted/insufficient counts, never
    silently dropped) -- it is just never physically removed from the
    response text, because the checker cannot fabricate a replacement for
    a sentence that was never an unverifiable personal-completion claim.

    Callers opt into this function only in correction mode.  Checker failures
    are fail-open and preserve the response.  No negation or replacement event
    is invented; if every sentence is removed a neutral context statement is
    returned.
    """

    if result.status != "checked" or not response:
        return response
    removals: list[tuple[int, int]] = []
    for claim in result.claims:
        if claim.get("status") not in _UNSUPPORTED:
            continue
        if not _omission_eligible_kind(_as_text(claim.get("kind"))):
            continue
        claim_text = claim.get("text")
        if not isinstance(claim_text, str):
            continue
        start = response.find(claim_text)
        if start < 0:
            continue
        removals.append(_sentence_span(response, start, start + len(claim_text)))
    if not removals:
        return response
    merged: list[list[int]] = []
    for start, end in sorted(removals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    cleaned = response
    for start, end in reversed(merged):
        cleaned = cleaned[:start] + cleaned[end:]
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or "I don't have enough context to verify those personal details."


__all__ = [
    "PersonalClaimResult",
    "audit_personal_claims",
    "build_personal_evidence",
    "omit_unsupported_claims",
]
