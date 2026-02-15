"""
# memory/truth_scorer.py

Module Contract
- Purpose: Evidence-based truth scoring with time decay. Replaces the old
  access-count echo chamber with decay-toward-uncertainty. Facts start at a
  source-dependent initial score, gain truth through user confirmations, lose
  truth through corrections/contradictions, and decay toward a floor when
  unconfirmed.
- Inputs:
  - Metadata dicts from ChromaDB documents or user_profile facts
- Outputs:
  - Float truth scores in [0.0, 1.0]
- Key behaviors:
  - Stateless utility: all state lives in document metadata
  - Time decay is read-only (computed at retrieval, not written back)
  - Confirmation resets the decay clock
  - Corrections apply a sharp penalty; contradictions a milder one
- Side effects:
  - None (pure computation)
"""

from datetime import datetime
from typing import Optional

from utils.logging_utils import get_logger
from config.app_config import (
    TRUTH_SCORER_ENABLED,
    TRUTH_SCORER_INITIAL_SCORE,
    TRUTH_SCORER_CONFIRMED_BOOST,
    TRUTH_SCORER_CORRECTION_PENALTY,
    TRUTH_SCORER_CONTRADICTION_PENALTY,
    TRUTH_SCORER_DECAY_RATE,
    TRUTH_SCORER_DECAY_FLOOR,
    TRUTH_SCORER_SOURCE_SCORES,
)

logger = get_logger("truth_scorer")


class TruthScorer:
    """Stateless truth scoring engine.

    All constants are read from config at import time.  Every method is a
    pure function over its arguments — no instance state is mutated.
    """

    # ------------------------------------------------------------------
    # Initial scoring
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_initial_score(source: str = "llm_extracted") -> float:
        """Return the initial truth score for a given fact source.

        Args:
            source: One of "user_stated", "corrected", "llm_extracted", "inferred".

        Returns:
            Float initial score (falls back to TRUTH_SCORER_INITIAL_SCORE).
        """
        return float(
            TRUTH_SCORER_SOURCE_SCORES.get(source, TRUTH_SCORER_INITIAL_SCORE)
        )

    # ------------------------------------------------------------------
    # Score adjustments
    # ------------------------------------------------------------------

    @staticmethod
    def apply_confirmation(current_score: float) -> float:
        """Boost truth when the user re-states or confirms a fact."""
        return min(1.0, current_score + TRUTH_SCORER_CONFIRMED_BOOST)

    @staticmethod
    def apply_correction(current_score: float) -> float:
        """Penalize truth when the user explicitly corrects a fact."""
        return max(0.0, current_score - TRUTH_SCORER_CORRECTION_PENALTY)

    @staticmethod
    def apply_contradiction(current_score: float) -> float:
        """Mild penalty when a cross-collection contradiction is detected."""
        return max(0.0, current_score - TRUTH_SCORER_CONTRADICTION_PENALTY)

    # ------------------------------------------------------------------
    # Time decay
    # ------------------------------------------------------------------

    @staticmethod
    def apply_time_decay(
        current_score: float,
        last_confirmed_at: Optional[datetime] = None,
    ) -> float:
        """Decay truth toward the floor based on time since last confirmation.

        The decay is linear per week:
            decayed = current - (weeks_since_confirmed * DECAY_RATE)
            clamped to [DECAY_FLOOR, current]

        Args:
            current_score: The stored truth_score.
            last_confirmed_at: Timestamp of last confirmation/creation.

        Returns:
            Effective truth score after decay (read-only, not persisted).
        """
        if last_confirmed_at is None:
            return current_score

        now = datetime.now()
        try:
            if isinstance(last_confirmed_at, str):
                last_confirmed_at = datetime.fromisoformat(last_confirmed_at)
            elapsed_weeks = max(
                0.0, (now - last_confirmed_at).total_seconds() / (7 * 24 * 3600)
            )
        except (ValueError, TypeError):
            return current_score

        decayed = current_score - (elapsed_weeks * TRUTH_SCORER_DECAY_RATE)
        return max(TRUTH_SCORER_DECAY_FLOOR, min(current_score, decayed))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @staticmethod
    def compute_effective_truth(metadata: dict) -> float:
        """Compute the effective truth score for a document at read time.

        Reads ``truth_score`` and ``last_confirmed_at`` from metadata,
        applies time decay, and returns the result.  If truth scoring is
        disabled or metadata lacks a truth_score, falls back to the
        legacy ``truth_score`` field or a default of 0.6.

        This is a read-only operation — the returned value should be used
        for ranking but NOT written back to ChromaDB (decay is transient).
        """
        if not TRUTH_SCORER_ENABLED:
            # Fallback: use stored truth_score or default
            return float(metadata.get("truth_score", 0.6))

        stored = float(metadata.get("truth_score", TRUTH_SCORER_INITIAL_SCORE))
        last_confirmed = metadata.get("last_confirmed_at")

        if last_confirmed:
            return TruthScorer.apply_time_decay(stored, last_confirmed)
        else:
            # No confirmation timestamp — use creation timestamp as anchor
            created = metadata.get("timestamp")
            if created:
                return TruthScorer.apply_time_decay(stored, created)
            return stored
