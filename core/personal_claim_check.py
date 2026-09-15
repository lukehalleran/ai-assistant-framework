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
import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence


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


def build_personal_evidence(
    query: str,
    context: Any,
    history: Iterable[Any] = (),
    *,
    max_chars: int = 12000,
) -> list[dict]:
    """Build bounded, role-preserved evidence for one draft.

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
            marker = " [...truncated]"
            text = text[: max(1, budget - len(marker))] + marker
        if len(text) > budget:
            return
        source_id = (
            "src_current_query"
            if identity == "__current_query__"
            else _opaque_id(identity, role, text, ordinal)
        )
        item = {"source_id": source_id, "role": role, "timestamp": timestamp, "text": text}
        entries.append(item)
        budget -= len(text)

    add("user", None, current, "__current_query__", 0)
    if budget <= 0:
        return entries

    candidates: list[tuple[int, int, str, str | None, str, Any]] = []
    rows = _iter_context(context, history)
    correction_indexes: list[int] = []
    flattened: list[tuple[str, str | None, str, Any]] = []
    for value in rows:
        flattened.extend(_canonical_rows(value))
    seen: set[tuple[str, str, str]] = set()
    for index, (role, stamp, text, identity) in enumerate(flattened):
        key = (role, stamp or "", text, _as_text(identity))
        if key in seen:
            continue
        seen.add(key)
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
Find every material claim about what the user or another person did, including novel actions; do not use a closed action-verb list. Also classify plans, suggestions, conditional/partial/negated/cancelled claims, quoted claims, discussion claims, and claims about another episode or object. The draft span must be copied exactly.
Evidence is source-backed context, not a substitute for semantic entailment. A user report can support a user's completion. An assistant suggestion, assistant summary, quote, generated narrative, profile/fact, or tool text cannot establish that the user completed an external task. Assistant-origin evidence may support only a claim about the discussion itself. Prefer a newer direct user correction over an earlier report. Ambiguous, missing, or incomplete evidence is insufficient, never a negative fact. Cite exact contiguous source quotes and their source IDs; cite no source for an insufficient claim when no exact span supports it.
"""


def _review_prompt(response: str, evidence: Sequence[Mapping[str, Any]]) -> str:
    evidence_json = json.dumps(list(evidence), ensure_ascii=False, separators=(",", ":"))
    return (
        "Review this assistant draft. Treat evidence entries as records with preserved roles.\n\n"
        "EVIDENCE:\n" + evidence_json + "\n\nDRAFT:\n" + response
    )


def _strict_json(raw: Any) -> dict:
    if not isinstance(raw, str):
        raise ValueError("response is not text")
    text = raw.strip()
    if not text or text.startswith("``"):
        raise ValueError("not strict JSON")
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or set(parsed) != {"claims"}:
        raise ValueError("unexpected JSON shape")
    if not isinstance(parsed["claims"], list):
        raise ValueError("claims is not a list")
    return parsed


def _is_completion_kind(kind: str) -> bool:
    lowered = kind.lower().replace("-", "_")
    return (
        "completion" in lowered
        or "completed" in lowered
        or lowered in {"event", "personal_event", "user_event", "achievement"}
    )


def _validate_claims(response: str, payload: dict, evidence: Sequence[Mapping[str, Any]]) -> list[dict]:
    sources: dict[str, list[Mapping[str, Any]]] = {}
    for row in evidence:
        source_id = row.get("source_id")
        if isinstance(source_id, str):
            sources.setdefault(source_id, []).append(row)
    claims: list[dict] = []
    spans: set[str] = set()
    for raw_claim in payload["claims"]:
        if not isinstance(raw_claim, dict) or set(raw_claim) != {"text", "status", "kind", "evidence"}:
            raise ValueError("unexpected claim shape")
        text = raw_claim["text"]
        status = raw_claim["status"]
        kind = raw_claim["kind"]
        refs = raw_claim["evidence"]
        if not isinstance(text, str) or not text or text not in response:
            raise ValueError("claim is not an exact response span")
        if text in spans:
            raise ValueError("duplicate claim span")
        spans.add(text)
        if status not in _VALID_STATUSES or not isinstance(kind, str) or not kind.strip():
            raise ValueError("invalid claim enum")
        if not isinstance(refs, list):
            raise ValueError("evidence is not a list")
        checked_refs: list[dict] = []
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) != {"source_id", "quote"}:
                raise ValueError("unexpected evidence shape")
            source_id, quote = ref["source_id"], ref["quote"]
            if not isinstance(source_id, str) or not isinstance(quote, str) or not quote:
                raise ValueError("invalid source reference")
            rows = sources.get(source_id)
            if not rows or not any(quote in _as_text(row.get("text")) for row in rows):
                raise ValueError("source quote is not exact")
            checked_refs.append({"source_id": source_id, "quote": quote})
        if status == "supported" and not checked_refs:
            raise ValueError("supported claim lacks evidence")
        if status == "supported" and _is_completion_kind(kind):
            user_sources = {
                ref["source_id"]
                for ref in checked_refs
                if any(row.get("role") == "user" for row in sources.get(ref["source_id"], ()))
            }
            if not user_sources:
                # Assistant discussion/advice cannot corroborate a user event.
                status = "insufficient"
        claims.append({"text": text, "status": status, "kind": kind.strip(), "evidence": checked_refs})
    return claims


@dataclass
class PersonalClaimResult:
    status: str
    reason: str
    claims: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0
    evidence_truncated: bool = False

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
        }
        if self.evidence_truncated:
            result["evidence_truncated"] = True
        return result


async def audit_personal_claims(
    response: str,
    evidence: Sequence[Mapping[str, Any]],
    model_manager: Any,
    *,
    model_name: str | None = None,
    timeout_s: float = 5.0,
    max_tokens: int = 900,
) -> PersonalClaimResult:
    """Semantically review ``response`` and mechanically validate its JSON."""

    started = perf_counter()
    if not response or not response.strip():
        return PersonalClaimResult("skipped", "empty_response", elapsed_s=perf_counter() - started)
    if not evidence:
        return PersonalClaimResult("unavailable", "no_evidence", elapsed_s=perf_counter() - started)
    if model_manager is None or not callable(getattr(model_manager, "generate_once", None)):
        return PersonalClaimResult("unavailable", "no_model", elapsed_s=perf_counter() - started)
    truncated = any("[…truncated]" in _as_text(row.get("text")) for row in evidence)
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
        claims = _validate_claims(response, payload, evidence)
    except (ValueError, TypeError, json.JSONDecodeError):
        return PersonalClaimResult("failed", "invalid_json", elapsed_s=perf_counter() - started,
                                   evidence_truncated=truncated)
    return PersonalClaimResult("checked", "ok", claims=claims, elapsed_s=perf_counter() - started,
                               evidence_truncated=truncated)


_SENTENCE_END_RE = re.compile(r"[.!?]+(?:[\"'”’»\)\]]+)?(?:\s+|$)|\n{2,}")


def _sentence_span(text: str, start: int, end: int) -> tuple[int, int]:
    left_match = list(_SENTENCE_END_RE.finditer(text, 0, start))
    left = left_match[-1].end() if left_match else 0
    right_match = _SENTENCE_END_RE.search(text, end)
    right = right_match.end() if right_match else len(text)
    return left, right


def omit_unsupported_claims(response: str, result: PersonalClaimResult) -> str:
    """Remove whole sentences containing exact unsupported claim spans.

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
