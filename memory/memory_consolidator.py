"""
# memory/memory_consolidator.py

Module Contract
- Purpose: Generate concise conversation summaries from recent exchanges. Used by shutdown summarizer and optionally mid‑session consolidation.
- Inputs:
  - consolidation_threshold (N): target block size for summaries
  - maybe_consolidate(corpus_manager): returns True if a new summary node is stored
- Outputs:
  - Creates a new summary node via corpus_manager.add_summary when due.
- Dependencies:
  - models.model_manager for generate_once
- Side effects:
  - Writes summary nodes to corpus (and by caller pathways, to Chroma).
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from utils.logging_utils import get_logger

logger = get_logger("memory_consolidator")

SUMMARY_EVERY_N = int(os.getenv("SUMMARY_EVERY_N", "20"))
SUMMARY_LOOKBACK = int(os.getenv("SUMMARY_LOOKBACK", "40"))
SUMMARY_MIN_GAP_MIN = int(os.getenv("SUMMARY_MIN_GAP_MIN", "20"))
SUMMARY_MODEL_ALIAS = os.getenv("LLM_SUMMARY_ALIAS", "gpt-4o-mini")
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "220"))


def _format_recent_for_summary(recent: List[Dict[str, Any]],
                               q_max: int = 240,
                               a_max: int = 300) -> List[str]:
    """
    Build short conversation slices from recent corpus entries for summarization.
    Each slice is of the form:
      "User: <q>\nAssistant: <a>"
    with conservative clipping to avoid overly long prompts.
    """
    out: List[str] = []
    if not recent:
        return out
    for e in recent:
        try:
            q = (e.get("query") or "").strip()
            a = (e.get("response") or "").strip()
            if not (q or a):
                continue
            if len(q) > q_max:
                q = q[:q_max]
            if len(a) > a_max:
                a = a[:a_max]
            out.append(f"User: {q}\nAssistant: {a}")
        except Exception:
            # best-effort; skip malformed entries
            continue
    return out

class MemoryConsolidator:
    def __init__(
        self,
        model_manager,
        *,
        consolidation_threshold: int | None = None,
        lookback: int | None = None,
        min_gap_minutes: int | None = None,
        **_unused,  # tolerate extra kwargs from older callers
    ):
        """
        consolidation_threshold: how many exchanges between stored summaries (default from env SUMMARY_EVERY_N)
        lookback:                how many recent exchanges to compress (default from env SUMMARY_LOOKBACK)
        min_gap_minutes:         minimum minutes between stored summaries (default from env SUMMARY_MIN_GAP_MIN)
        """
        self.model_manager = model_manager

        # Prefer explicit args from caller; fall back to env defaults
        self.consolidation_threshold = consolidation_threshold or SUMMARY_EVERY_N
        self.lookback = lookback or SUMMARY_LOOKBACK
        self.min_gap_minutes = min_gap_minutes or SUMMARY_MIN_GAP_MIN

        logger.debug(
            "[Consolidator] init threshold=%s lookback=%s min_gap_min=%s",
            self.consolidation_threshold, self.lookback, self.min_gap_minutes
        )

    async def maybe_consolidate(self, corpus_manager) -> bool:
        """
        Check counters/time and, if due, generate+store a summary memory node.
        Returns True if a new summary was stored.
        """
        try:
            # 1) Throttle by time
            last = getattr(corpus_manager, "get_last_summary_meta", lambda: None)()
            if last and isinstance(last, dict):
                ts = last.get("timestamp")
                if ts and isinstance(ts, datetime):
                    if datetime.now() - ts < timedelta(minutes=self.min_gap_minutes):
                        logger.debug("[Consolidation] Skipped (min gap not elapsed)")
                        return False

            # 2) Count since last summary
            num_since_fn = getattr(corpus_manager, "count_exchanges_since_last_summary", None)
            if callable(num_since_fn):
                since = num_since_fn()
                if since < self.consolidation_threshold:
                    logger.debug(
                        "[Consolidation] Skipped (%s/%s since last summary)",
                        since, self.consolidation_threshold
                    )
                    return False
            else:
                # Fallback: use recent length as proxy
                recent_probe = corpus_manager.get_recent_memories(self.lookback)
                if len(recent_probe) < self.consolidation_threshold:
                    logger.debug(
                        "[Consolidation] Skipped (proxy %s/%s)",
                        len(recent_probe), self.consolidation_threshold
                    )
                    return False

            # 3) Gather the recent exchanges to compress
            recent = corpus_manager.get_recent_memories(self.lookback)
            convo_slices = _format_recent_for_summary(recent)
            if not convo_slices:
                logger.debug("[Consolidation] No material to summarize")
                return False

            # 4) LLM summarize (bounded by your model_manager’s timeout policies)
            excerpts = "\n\n".join(convo_slices)
            prompt = (
                "You are an extractive note-taker. Using ONLY the EXCERPTS below, write 3–5 factual bullets. "
                "Do NOT infer or invent anything not present. If information is minimal, output 1–2 bullets that "
                "quote or paraphrase the text. No headers, just bullets.\n\nEXCERPTS:\n" + excerpts + "\n\nBullets:"
            )
            if not hasattr(self.model_manager, "generate_once"):
                logger.debug("[Consolidation] No generate_once on model_manager")
                return False

            text = await self.model_manager.generate_once(prompt, max_tokens=SUMMARY_MAX_TOKENS)
            text = (text or "").strip()
            if not text:
                logger.debug("[Consolidation] LLM returned empty text")
                return False

            # 5) Persist as a summary node
            add_summary = getattr(corpus_manager, "add_summary", None)
            if not callable(add_summary):
                logger.debug("[Consolidation] corpus_manager.add_summary not available")
                return False

            node = {
                "content": text,
                "timestamp": datetime.now(),
                "type": "summary",
                "tags": ["summary:consolidated", "source:consolidator"],
            }
            add_summary(node)
            logger.info("[Consolidation] Stored new summary node")
            return True

        except Exception as e:
            logger.debug(f"[Consolidation] Error: {e}")
            return False
