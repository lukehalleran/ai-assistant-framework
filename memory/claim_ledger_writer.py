"""Live write of the user-claim ledger: one bounded background call per turn.

Incident 2026-10-20; class: BC-75, BC-52, BC-89. ``record_turn`` is called by
``MemoryStorage.store_interaction`` AFTER the turn is persisted. It never raises
into storage: every failure degrades to a status label.

Flow: flag off -> return at once (no model call, no keys); probe turn / pasted or
inadmissible text -> skipped (``claim_ledger.prepare_user_text``, the same
chokepoint the fact extractors use); search-credit budget known and <= 0 ->
skipped; otherwise THE deployed ``generate_once`` (temperature 0, reasoning off,
small ``max_tokens`` -- ``claim_ledger.generate_for_ledger``), deterministic
validation, ``ledger.add``, atomic save. The whole call runs under
``asyncio.wait_for(write_timeout_s)``.

Receipt (turn-record keys, labels and counts only -- never a value or a quote):
``claim_ledger_written`` (int) and ``claim_ledger_status``
(``ok`` | ``skipped:<reason>`` | ``timeout`` | ``error``).
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from config import app_config
from memory import claim_ledger
from memory import claim_ledger_store as store
from utils.logging_utils import get_logger
from utils.safe_json import CorruptStoreError, StoreVersionError

logger = get_logger("claim_ledger_writer")


def _is_error_sentinel(text: Any) -> bool:
    from models.model_manager import API_ERROR_PREFIXES  # lazy import: startup-cost (openai/httpx stack)
    return not isinstance(text, str) or text.lstrip().startswith(tuple(API_ERROR_PREFIXES))


def _credits_exhausted() -> bool:
    try:
        from knowledge.web_search_manager import live_remaining_credits  # lazy import: patch-point (tests stub the credit source) + layering
        remaining = live_remaining_credits()
    except Exception:  # degrades: credit lookup unavailable -> unknown -> proceed
        return False
    return remaining is not None and remaining <= 0       # None = unknown = proceed


async def _write(user_text: str, said_at: Any, turn_id: str, model_manager: Any) -> tuple[int, str]:
    reason, text = claim_ledger.prepare_user_text(user_text)
    if reason:
        return 0, f"skipped:{reason}"
    if _credits_exhausted():
        return 0, "skipped:no_credits"
    ledger = store.get_ledger()      # Corrupt/Version errors propagate: never an empty stand-in
    subjects = ledger.open_subjects(claim_ledger.OPEN_SUBJECTS_CAP, as_of=said_at)
    system, user = claim_ledger.build_extraction_prompt(text, said_at, subjects)
    raw = await claim_ledger.generate_for_ledger(
        model_manager, app_config.CLAIM_LEDGER_MODEL, system, user, claim_ledger.EXTRACT_MAX_TOKENS)
    if _is_error_sentinel(raw):
        return 0, "error"
    claims, drops = claim_ledger.validate_claims(raw, text, said_at, subjects, turn_id=turn_id)
    if drops:
        logger.debug("[ClaimLedger] dropped %d claim(s): %s", len(drops),
                     sorted({str(d.get('reason')).split(' ')[0] for d in drops}))
    if not claims:
        return 0, "ok"
    async with store.write_lock():
        ledger = store.get_ledger()
        for c in claims:
            ledger.add(c)
        store.save_ledger()
    return len(claims), "ok"


async def record_turn(user_text: Optional[str], said_at: Any, turn_id: str, model_manager: Any,
                      *, receipt: Optional[dict] = None) -> dict:
    """Extract and store the claims in ONE turn's user-authored text. Returns the
    receipt ``{"claim_ledger_written", "claim_ledger_status"}`` (empty when the
    ledger is disabled); also merged into ``receipt`` when one is passed."""
    if not app_config.CLAIM_LEDGER_ENABLED:
        return {}
    written, status = 0, "error"
    try:
        written, status = await asyncio.wait_for(
            _write(user_text or "", said_at, turn_id, model_manager),
            timeout=app_config.CLAIM_LEDGER_WRITE_TIMEOUT_S)
    except asyncio.TimeoutError:
        status = "timeout"
        logger.warning("[ClaimLedger] write timed out after %.1fs", app_config.CLAIM_LEDGER_WRITE_TIMEOUT_S)
    except (CorruptStoreError, StoreVersionError) as exc:
        status = "error"       # degrades: this turn's claims are not recorded; the store file is left untouched
        logger.error("[ClaimLedger] store unreadable, nothing written: %s", exc)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # degrades: this turn's claims are not recorded; storage is unaffected
        status = "error"
        logger.warning("[ClaimLedger] write failed: %s", type(exc).__name__)
    out = {"claim_ledger_written": written, "claim_ledger_status": status}
    if isinstance(receipt, dict):
        receipt.update(out)
    return out
