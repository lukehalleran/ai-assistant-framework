# memory/memory_consolidator.py

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
            prompt = (
                "You are a neutral note-taker. Create a concise recap of the recent conversation "
                "in 3–5 bullet points, third-person, factual, no praise, no instructions.\n\n"
                "EXCERPTS:\n" + "\n\n".join(convo_slices)
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
