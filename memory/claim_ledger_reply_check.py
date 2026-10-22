"""Log-only reply check against the user-claim ledger (design 7a.4).

Incident 2026-10-20; class: BC-75, BC-52, BC-89. After a reply is DELIVERED, one
background call (same model as the live write, reasoning off, bounded by
``write_timeout_s``) asks which statements of the reply restate one of the user's
STANDING ledger subjects and with what modality; the deterministic validator in
``claim_ledger`` admits and classifies them. Only the LIVE classes are recorded
(``claim_ledger.REPLY_CHECK_LIVE_CLASSES``: a conditional or an estimate stated as
plain fact). The stale-value classes were measured at 0 confirmed drift of 22
flags and are never recorded here.

This module only READS the ledger and never edits the reply: it returns a receipt.

Receipt (labels and counts only -- no value, no quote):
  claim_ledger_reply_findings  {class: count}
  claim_ledger_reply_status    ok | skipped:<reason> | timeout | error
  claim_ledger_reply_subjects  sorted subject keys (debug record only; the handler
                               keeps it off the turn record)
"""
from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any, Optional

from config import app_config
from memory import claim_ledger
from memory import claim_ledger_store as store
from utils.logging_utils import get_logger
from utils.safe_json import CorruptStoreError, StoreVersionError

logger = get_logger("claim_ledger_reply_check")


def enabled() -> bool:
    return bool(app_config.CLAIM_LEDGER_ENABLED and app_config.CLAIM_LEDGER_REPLY_CHECK == "log_only")


async def _check(reply_text: str, said_at: Any, model_manager: Any):
    ledger = store.get_ledger()      # Corrupt/Version errors propagate; the ledger is only read

    async def call_model(system: str, user: str) -> str:
        raw = await claim_ledger.generate_for_ledger(
            model_manager, app_config.CLAIM_LEDGER_MODEL, system, user, claim_ledger.REPLY_MAX_TOKENS)
        from models.model_manager import API_ERROR_PREFIXES  # lazy import: startup-cost (openai/httpx stack)
        if not isinstance(raw, str) or raw.lstrip().startswith(tuple(API_ERROR_PREFIXES)):
            raise RuntimeError("model error sentinel")
        return raw

    return await claim_ledger.check_reply(reply_text, said_at, ledger, call_model)


async def run(reply_text: Optional[str], said_at: Any, model_manager: Any) -> dict:
    """Receipt for one delivered reply; ``{}`` when the check is not enabled. Never raises
    (CancelledError propagates)."""
    if not enabled():
        return {}
    status, counts, subjects = "error", {}, []
    try:
        findings, _drops, skip = await asyncio.wait_for(
            _check(reply_text or "", said_at, model_manager),
            timeout=app_config.CLAIM_LEDGER_WRITE_TIMEOUT_S)
        if skip:
            status = f"skipped:{skip}"
        else:
            live = [f for f in findings if f.classification in claim_ledger.REPLY_CHECK_LIVE_CLASSES]
            counts = dict(Counter(f.classification for f in live))
            subjects = sorted({f.subject_key for f in live})
            status = "ok"
    except asyncio.TimeoutError:
        status = "timeout"
    except (CorruptStoreError, StoreVersionError) as exc:
        logger.error("[ClaimLedger] reply check: store unreadable: %s", exc)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # degrades: no reply-check receipt for this turn
        logger.warning("[ClaimLedger] reply check failed: %s", type(exc).__name__)
    if counts:
        logger.warning("[ClaimLedger] reply check findings: %s", counts)
    return {"claim_ledger_reply_findings": counts, "claim_ledger_reply_status": status,
            "claim_ledger_reply_subjects": subjects}
