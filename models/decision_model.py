"""Decision-model client: the ONE place that builds a System One request or parses its reply.

Jev (TypeSafe) answers typed questions (noul / choice / score) over OpenRouter's System One
endpoint. Feature code builds Questions with ``noul`` / ``choice`` / ``score``, calls
``evaluate`` and reads a typed ``DecisionOutcome``; it never assembles JSON (BC-15).

Contract (docs/execution/decision_model/PLAN_20261008.md, D1-D5, D11-D14):
- the destination is the code constant ``SYSTEM_ONE_URL``, checked against the manager's base
  URL before any credential is touched; transport and key stay inside ``ModelManager`` (D5b);
- ``evaluate`` never raises operational/model/transport errors and never fakes a negative:
  every failure is a status + reason. ``asyncio.CancelledError`` always propagates;
- every required field of every answer is validated on its own (D4, BC-84); the served model
  and provider must be in the pinned allowlist (D14);
- no truncation: state over the role's ``max_state_chars`` is ``unavailable/state_too_large``;
- one retry (429, 529, timeout, transport) only if it fits inside ``timeout_s`` (D9);
- no environment reads; nothing logged but status/reason/latency/retried (BC-72).
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlsplit

import httpx

from utils.async_results import classify_gather_results
from utils.logging_utils import get_logger

logger = get_logger("decision_model")

SYSTEM_ONE_URL = "https://openrouter.ai/api/v1/systemone"  # D1, D11: code, not config

REASONS = frozenset({
    "timeout", "http_400", "http_401", "http_402", "http_422", "http_429", "http_529",
    "http_other", "transport", "parse", "schema_mismatch", "model_mismatch",
    "state_too_large", "not_configured",
})
_RETRY_STATUS = {429: "http_429", 529: "http_529"}
_HARD_STATUS = {401: "http_401", 402: "http_402"}
_INVALID_STATUS = {400: "http_400", 422: "http_422"}
_RETRY_BACKOFF_S = 0.1   # used when a 429/529 carries no usable Retry-After
_MIN_ATTEMPT_S = 0.1     # a retry must leave at least this much of the deadline
_SUM_TOL = 0.02
_SCORE_TOL = 0.05


@dataclass(frozen=True)
class Question:
    qid: str
    kind: str  # noul | choice | score
    instructions: str
    criteria: object = None


def _check_qid(qid: str, instructions: str) -> None:
    if not isinstance(qid, str) or not qid or not isinstance(instructions, str) or not instructions:
        raise ValueError("question needs a non-empty qid and instructions")


def noul(qid: str, instructions: str, true_desc: Optional[str] = None,
         false_desc: Optional[str] = None) -> Question:
    _check_qid(qid, instructions)
    criteria = {k: v for k, v in (("true", true_desc), ("false", false_desc)) if v}
    return Question(qid, "noul", instructions, criteria or None)


def choice(qid: str, instructions: str, options: Mapping[str, str]) -> Question:
    _check_qid(qid, instructions)
    if not isinstance(options, Mapping) or not 2 <= len(options) <= 255:
        raise ValueError("choice needs 2..255 options (option -> description)")
    if not all(isinstance(k, str) and k and isinstance(v, str) for k, v in options.items()):
        raise ValueError("choice options must map non-empty strings to strings")
    return Question(qid, "choice", instructions, dict(options))


def score(qid: str, instructions: str, levels: Sequence[str]) -> Question:
    _check_qid(qid, instructions)
    if isinstance(levels, (str, bytes)) or not 2 <= len(levels) <= 10 or not all(
            isinstance(v, str) and v for v in levels):
        raise ValueError("score needs 2..10 non-empty level descriptions, low to high")
    return Question(qid, "score", instructions, list(levels))


@dataclass(frozen=True)
class Answer:
    qid: str
    kind: str
    valid: bool
    invalid_reason: Optional[str] = None
    choice: Optional[str] = None
    score: Optional[float] = None
    noul: Optional[float] = None
    probabilities: Optional[dict] = None
    confidence: Optional[float] = None
    legend: Optional[dict] = None


@dataclass(frozen=True)
class DecisionOutcome:
    status: str  # ok | unavailable | invalid | disabled
    reason: Optional[str] = None
    answers: dict = field(default_factory=dict)
    served_model: Optional[str] = None
    provider: Optional[str] = None
    latency_ms: float = 0.0
    input_tokens: Optional[int] = None
    cost_usd: Optional[float] = None  # None when the route reports no cost (BC-47)
    retried: bool = False


def build_request(state: Any, questions: Sequence[Question], *, model: str) -> dict:
    """Pure: the System One JSON body."""
    qs: dict = {}
    for q in questions:
        if q.qid in qs:
            raise ValueError(f"duplicate question id {q.qid!r}")
        item: dict = {"type": q.kind, "instructions": q.instructions}
        if q.criteria is not None:
            item["criteria"] = q.criteria
        qs[q.qid] = item
    if not qs:
        raise ValueError("at least one question is required")
    return {"model": model, "state": state, "questions": qs}


def _num(x: Any, lo: float, hi: float) -> bool:
    """Finite number in [lo, hi]; False (never raises) for bools, NaN, or an int too big for float."""
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return False
    try:
        return math.isfinite(x) and lo <= x <= hi
    except (OverflowError, TypeError, ValueError):
        return False


def _probs_problem(probs: Any, keys: set) -> Optional[str]:
    if not isinstance(probs, dict) or set(probs) != keys:
        return "probability_keys"
    if not all(_num(v, 0.0, 1.0) for v in probs.values()):
        return "probability_value"
    if abs(sum(probs.values()) - 1.0) > _SUM_TOL:
        return "probability_sum"
    return None


def _validate_answer(q: Question, raw: Any) -> Answer:
    def bad(reason: str) -> Answer:
        return Answer(q.qid, q.kind, False, reason)

    if not isinstance(raw, dict):
        return bad("answer_missing")
    if raw.get("type") != q.kind:
        return bad("type_mismatch")
    if q.kind == "noul":
        v = raw.get("noul")
        return Answer(q.qid, q.kind, True, noul=float(v)) if _num(v, 0.0, 1.0) else bad("noul_range")
    conf, probs = raw.get("confidence"), raw.get("probabilities")
    if q.kind == "choice":
        crit = q.criteria or {}
        picked = raw.get("choice")
        if not isinstance(picked, str) or picked not in crit:
            return bad("choice_not_in_criteria")
        problem = _probs_problem(probs, set(crit))
        if problem:
            return bad(problem)
        if not _num(conf, 0.0, 1.0):
            return bad("confidence_range")
        return Answer(q.qid, q.kind, True, choice=picked, probabilities=dict(probs), confidence=float(conf))
    levels = list(q.criteria or [])
    keys = {str(i) for i in range(len(levels))}
    legend = raw.get("legend")
    if not isinstance(legend, dict) or set(legend) != keys:
        return bad("legend_keys")
    if any(legend[str(i)] != d for i, d in enumerate(levels)):
        return bad("legend_description")
    problem = _probs_problem(probs, keys)
    if problem:
        return bad(problem)
    if not _num(conf, 0.0, 1.0):
        return bad("confidence_range")
    sc = raw.get("score")
    if not _num(sc, 0.0, len(levels) - 1):
        return bad("score_range")
    if abs(sc - sum(i * probs[str(i)] for i in range(len(levels)))) > _SCORE_TOL:
        return bad("score_inconsistent")
    return Answer(q.qid, q.kind, True, score=float(sc), probabilities=dict(probs),
                  confidence=float(conf), legend=dict(legend))


def parse_response(body: Any, questions: Sequence[Question], *, served_models: Sequence[str],
                   provider: str) -> tuple:
    """Pure (D4). Returns ``(answers, meta)``: ``answers`` maps every asked qid to an Answer
    judged on its own; ``meta`` has served_model, provider, input_tokens, cost_usd and
    ``reason`` (None | "parse" | "model_mismatch" | "schema_mismatch"). Unknown top-level
    keys are ignored; a missing answer is an invalid Answer, never a skipped one."""
    meta: dict = {"served_model": None, "provider": None, "input_tokens": None,
                  "cost_usd": None, "reason": None}
    if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
        meta["reason"] = "parse"
        return {q.qid: Answer(q.qid, q.kind, False, "parse") for q in questions}, meta
    served, prov = body.get("model"), body.get("provider")
    # Only pinned-allowlist values survive; a rejected raw string never reaches the outcome (BC-72).
    meta["served_model"] = served if isinstance(served, str) and served in served_models else None
    meta["provider"] = prov if isinstance(prov, str) and prov == provider else None
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    tokens, cost = usage.get("input_tokens"), usage.get("cost")
    if isinstance(tokens, int) and not isinstance(tokens, bool):
        meta["input_tokens"] = tokens
    if _num(cost, 0.0, 1e12):  # optional metadata: absent or unparseable -> None, never an error
        meta["cost_usd"] = float(cost)
    if meta["served_model"] is None or meta["provider"] is None:
        meta["served_model"] = meta["provider"] = None  # a mismatched response reports neither
        meta["reason"] = "model_mismatch"
        return {q.qid: Answer(q.qid, q.kind, False, "model_mismatch") for q in questions}, meta
    answers = {q.qid: _validate_answer(q, body["answers"].get(q.qid)) for q in questions}
    if not all(a.valid for a in answers.values()):
        meta["reason"] = "schema_mismatch"
    return answers, meta


def _retry_after_s(header: Optional[str]) -> float:
    """Delay in seconds from Retry-After: delay-seconds or HTTP-date; default backoff if absent/garbage."""
    if header is None:
        return _RETRY_BACKOFF_S
    try:
        v = float(header)
        return max(0.0, v) if math.isfinite(v) else _RETRY_BACKOFF_S
    except ValueError:
        pass  # degrades: not delay-seconds, so the HTTP-date form is tried next
    try:
        when = parsedate_to_datetime(header)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, IndexError, OverflowError):
        return _RETRY_BACKOFF_S  # degrades: an unreadable Retry-After falls back to the default backoff


def _same_endpoint(a: str, b: str) -> bool:
    ua, ub = urlsplit(a), urlsplit(b)
    return ((ua.scheme.lower(), ua.netloc.lower(), ua.path) == (ub.scheme.lower(), ub.netloc.lower(), ub.path)
            and not ua.query and not ub.query and not ua.fragment and not ub.fragment)


async def evaluate(model_manager: Any, state: Any, questions: Sequence[Question], *,
                   role: str) -> DecisionOutcome:
    """Ask the decision model; never raises except CancelledError (D3)."""
    from config import app_config  # lazy import: live-config (Settings/tests mutate these at call time)

    t0 = time.monotonic()

    def done(status, reason=None, answers=None, meta=None, retried=False) -> DecisionOutcome:
        meta = meta or {}
        ms = round((time.monotonic() - t0) * 1000, 1)
        logger.debug("[DecisionModel] role=%s status=%s reason=%s latency_ms=%s retried=%s",
                     role, status, reason, ms, retried)
        return DecisionOutcome(status, reason, answers or {}, meta.get("served_model"),
                               meta.get("provider"), ms, meta.get("input_tokens"),
                               meta.get("cost_usd"), retried)

    if app_config.decision_model_mode(role) == "off":
        return done("disabled")
    limit = (app_config.DECISION_MODEL_MAX_STATE_CHARS or {}).get(role)
    post = getattr(model_manager, "post_system_one", None)
    base = getattr(model_manager, "base_url", None)
    if (post is None or not isinstance(limit, int) or not isinstance(base, str)
            or not _same_endpoint(SYSTEM_ONE_URL, f"{base}/systemone")):
        return done("unavailable", "not_configured")
    size = len(state) if isinstance(state, str) else len(json.dumps(state, default=str))
    if size > limit:
        return done("unavailable", "state_too_large")
    try:
        payload = build_request(state, questions, model=app_config.DECISION_MODEL_SLUG)
    except ValueError:
        return done("invalid", "schema_mismatch")  # a caller bug, reported not raised

    timeout_s = float(app_config.DECISION_MODEL_TIMEOUT_S)
    retried = False
    reason = "transport"
    while True:
        remaining = timeout_s - (time.monotonic() - t0)
        if remaining <= 0:
            return done("unavailable", reason or "timeout", retried=retried)
        wait = 0.0
        try:
            result = await asyncio.wait_for(post(SYSTEM_ONE_URL, payload, timeout_s=remaining), remaining)
        except (httpx.TimeoutException, asyncio.TimeoutError):
            reason = "timeout"
        except httpx.TransportError:
            reason = "transport"
        except ValueError:
            return done("unavailable", "not_configured", retried=retried)
        except Exception as exc:  # degrades: this decision is skipped, caller takes its legacy path
            logger.warning("[DecisionModel] unexpected %s", type(exc).__name__)
            return done("unavailable", "transport", retried=retried)
        else:
            if result is None:
                return done("unavailable", "not_configured", retried=retried)
            status, retry_after, text = result
            if status == 200:
                stage = "parse"  # ONE boundary for decode + validation; CancelledError is never caught
                try:
                    body = json.loads(text)
                    stage = "schema_mismatch"
                    answers, meta = parse_response(
                        body, questions, served_models=app_config.DECISION_MODEL_SERVED_MODELS,
                        provider=app_config.DECISION_MODEL_PROVIDER)
                except (RecursionError, OverflowError, ValueError, TypeError, KeyError, AttributeError) as exc:
                    logger.warning("[DecisionModel] response rejected at %s: %s", stage, type(exc).__name__)
                    return done("invalid", stage, retried=retried)
                if meta["reason"]:
                    return done("invalid", meta["reason"], answers, meta, retried)
                return done("ok", None, answers, meta, retried)
            if status in _INVALID_STATUS:
                return done("invalid", _INVALID_STATUS[status], retried=retried)
            if status in _HARD_STATUS:
                return done("unavailable", _HARD_STATUS[status], retried=retried)
            if status not in _RETRY_STATUS:
                return done("unavailable", "http_other", retried=retried)
            reason, wait = _RETRY_STATUS[status], _retry_after_s(retry_after)
        left = timeout_s - (time.monotonic() - t0) - wait
        if retried or left < _MIN_ATTEMPT_S:
            return done("unavailable", reason, retried=retried)
        retried = True
        if wait:
            await asyncio.sleep(wait)


async def run_with_shadow(primary: Any, shadow: Any) -> tuple:
    """Run two awaitables concurrently; return ``(primary_outcome, shadow_outcome)`` as
    ``utils.async_results.GatherOutcome`` (value or error per side, so a failing shadow can
    never break the primary). Whatever is still pending is cancelled AND awaited in
    ``finally``; a cancelled caller sees ``CancelledError`` (D3, BC-87)."""
    tasks: list = []
    try:
        tasks.append(asyncio.ensure_future(primary))
        tasks.append(asyncio.ensure_future(shadow))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        first, second = classify_gather_results(results)
        return first, second
    finally:
        pending = [t for t in tasks if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
