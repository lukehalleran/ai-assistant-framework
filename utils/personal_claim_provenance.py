"""Bounded personal-claim receipts at storage and conversation read boundaries.

A checker receipt describes a particular assistant response. It never turns
that response into user-authored evidence, and shadow findings never delete it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

KEY = "personal_claim_support"
MARKER = (
    "[Personal-claim check: this assistant response contains claims with "
    "insufficient or conflicting user evidence. Treat it as assistant "
    "interpretation, not a user completion report.]"
)
_COUNTS = ("candidate_count", "supported_count", "contradicted_count", "insufficient_count",
           "dropped_claim_count", "dropped_evidence_count", "demoted_count",
           "relocated_count")
_STATUSES = {"checked", "unavailable", "failed", "skipped"}
_REASONS = {
    "ok", "checked", "no_claims", "empty_response", "disabled", "no_model",
    "no_evidence", "timeout", "cancelled", "provider_error", "invalid_json",
    "invalid_response", "invalid_verdict", "invalid_evidence", "response_too_long",
    "check_error",
}


def response_digest(response: str) -> str:
    return hashlib.sha256(response.strip().encode("utf-8")).hexdigest()


def clean_personal_claim_receipt(value: Any, *, response: str | None = None) -> dict:
    """Accept only bounded receipt fields; drop excerpts and model-generated prose."""
    if isinstance(value, str):
        if len(value) > 16000:
            return {}
        try:
            value = json.loads(value)
        except (ValueError, RecursionError):
            return {}
    if (not isinstance(value, dict) or not isinstance(value.get("status"), str)
            or value["status"] not in _STATUSES):
        return {}
    reason = value.get("reason")
    receipt = {
        "status": value["status"],
        "reason": reason if isinstance(reason, str) and reason in _REASONS else "other",
    }
    for key in _COUNTS:
        count = value.get(key)
        if type(count) is int and 0 <= count <= 100:
            receipt[key] = count
    elapsed = value.get("elapsed_s")
    if type(elapsed) in (float, int) and math.isfinite(elapsed) and 0 <= elapsed <= 3600:
        receipt["elapsed_s"] = round(elapsed, 3)
    ids = value.get("source_ids")
    if isinstance(ids, list):
        receipt["source_ids"] = [
            item for item in ids[:64]
            if isinstance(item, str) and re.fullmatch(r"[A-Za-z0-9:_-]{1,96}", item)
        ]
    if type(value.get("evidence_truncated")) is bool:
        receipt["evidence_truncated"] = value["evidence_truncated"]
    delivery = value.get("delivery")
    if isinstance(delivery, str) and delivery in {"unchanged", "omitted", "failed_open"}:
        receipt["delivery"] = delivery
    digest = response_digest(response) if response is not None else value.get("response_sha256")
    if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
        receipt["response_sha256"] = digest
    return receipt


def _receipt_for(item: dict) -> dict:
    for container in (item, item.get("metadata"), item.get("provenance")):
        if isinstance(container, dict):
            receipt = clean_personal_claim_receipt(container.get(KEY))
            if receipt:
                return receipt
    return {}


def annotate_personal_claim_memory(item: Any) -> Any:
    """Mark the assistant segment only, and only for the exact checked response.

    Supports corpus query/response and semantic content+metadata shapes. No
    role guessing from embedded transcript labels, no mutations, no model calls.
    Unknown/legacy receipts remain unknown; correction receipts do not label
    the revised response with findings about the discarded draft.
    """
    if not isinstance(item, dict):
        return item
    receipt = _receipt_for(item)
    if (receipt.get("status") != "checked"
            or receipt.get("delivery") != "unchanged"
            or not any(receipt.get(key, 0) for key in ("contradicted_count", "insufficient_count"))):
        return item
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    response = item.get("response", item.get("a", metadata.get("response")))
    if not isinstance(response, str) or not response or MARKER in response:
        return item
    if receipt.get("response_sha256") != response_digest(response):
        return item
    annotated = response.rstrip() + "\n" + MARKER
    result = dict(item)
    for key in ("response", "a"):
        if result.get(key) == response:
            result[key] = annotated
    # The known stored assistant response must occupy the suffix. A matching
    # string quoted in the user's text cannot become an assistant boundary.
    for key in ("content", "text", "formatted"):
        content = result.get(key)
        if not isinstance(content, str) or MARKER in content:
            continue
        for label in ("\nAssistant: ", "\nDaemon: "):
            suffix = label + response.strip()
            if content.rstrip().endswith(suffix):
                result[key] = content.rstrip() + "\n" + MARKER
                break
    return result
