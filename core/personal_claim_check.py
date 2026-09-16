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
Find every material claim IN THE DRAFT about what the user or another person did, including novel actions; do not use a closed action-verb list. Also classify plans, suggestions, conditional/partial/negated/cancelled claims, quoted claims, discussion claims, and claims about another episode or object. The "text" field MUST be a character-exact copy of a contiguous span of the DRAFT: never paraphrase, never prefix it with "User", never restate the user's own message as a claim (the user's message is evidence, not a claim). If you cannot copy the span exactly, omit the claim.
Evidence is source-backed context, not a substitute for semantic entailment. A user report can support a user's completion. An assistant suggestion, assistant summary, quote, generated narrative, profile/fact, or tool text cannot establish that the user completed an external task. Assistant-origin evidence may support only a claim about the discussion itself. Prefer a newer direct user correction over an earlier report. Ambiguous, missing, or incomplete evidence is insufficient, never a negative fact. Cite exact contiguous source quotes and their source IDs; cite no source for an insufficient claim when no exact span supports it.
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
            drops.append("source quote is not exact")
            continue
        checked.append({"source_id": source_id, "quote": quote})
    return checked


def _validate_claims(
    response: str, payload: dict, evidence: Sequence[Mapping[str, Any]], *,
    overlap_threshold: float = DEFAULT_LOCATE_OVERLAP,
    min_claim_tokens: int = DEFAULT_LOCATE_MIN_TOKENS,
) -> tuple[list[dict], dict[str, int], list[str]]:
    """Validate the model's claims one at a time.

    Returns ``(claims, counts, drop_reasons)``. A claim that is not an exact
    response span (or is malformed/duplicated) is dropped and counted; a bad
    evidence reference is dropped and counted; a ``supported`` claim left
    without evidence -- or a completion-kind claim without USER-role evidence
    -- is demoted to ``insufficient`` and counted. Nothing is ever promoted.
    """
    sources: dict[str, list[Mapping[str, Any]]] = {}
    for row in evidence:
        source_id = row.get("source_id")
        if isinstance(source_id, str):
            sources.setdefault(source_id, []).append(row)
    claims: list[dict] = []
    by_span: dict[str, dict] = {}
    counts = {"dropped_claims": 0, "dropped_evidence": 0, "demoted": 0, "relocated": 0,
              "malformed": 0}
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
        if status == "supported" and not checked_refs:
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
        claim = {"text": text, "status": status, "kind": kind.strip(), "evidence": checked_refs}
        by_span[text] = claim
        claims.append(claim)
    return claims, counts, drops


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
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        # Constant-string reason only; the raw output is never logged.
        logger.debug(f"[PersonalClaim] invalid_json: {exc}")
        return PersonalClaimResult("failed", "invalid_json", elapsed_s=perf_counter() - started,
                                   evidence_truncated=truncated)
    claims, counts, drops = _validate_claims(
        response, payload, evidence,
        overlap_threshold=overlap_threshold, min_claim_tokens=min_claim_tokens,
    )
    if drops:
        logger.debug(f"[PersonalClaim] dropped {counts['dropped_claims']} claim(s), "
                     f"{counts['dropped_evidence']} reference(s): {sorted(set(drops))}")
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
                               relocated_count=counts["relocated"])


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
