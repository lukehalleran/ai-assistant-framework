"""Tests for memory/truth_scorer.py — evidence-based truth scoring."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from memory.truth_scorer import TruthScorer


class TestInitialScores:
    """Test source-based initial scoring."""

    def test_user_stated_gets_high_score(self):
        score = TruthScorer.calculate_initial_score("user_stated")
        assert score == 0.8

    def test_corrected_gets_highest_score(self):
        score = TruthScorer.calculate_initial_score("corrected")
        assert score == 0.85

    def test_llm_extracted_gets_default(self):
        score = TruthScorer.calculate_initial_score("llm_extracted")
        assert score == 0.7

    def test_inferred_gets_low_score(self):
        score = TruthScorer.calculate_initial_score("inferred")
        assert score == 0.5

    def test_unknown_source_falls_back_to_initial(self):
        score = TruthScorer.calculate_initial_score("unknown_source_xyz")
        assert score == 0.7  # TRUTH_SCORER_INITIAL_SCORE default

    def test_initial_scores_by_source(self):
        """Verify all defined sources return distinct values."""
        sources = ["user_stated", "corrected", "llm_extracted", "inferred"]
        scores = [TruthScorer.calculate_initial_score(s) for s in sources]
        # All scores should be in valid range
        assert all(0.0 <= s <= 1.0 for s in scores)
        # corrected > user_stated > llm_extracted > inferred
        assert scores[1] > scores[0] > scores[2] > scores[3]


class TestConfirmation:
    """Test confirmation boost."""

    def test_confirmation_increases_score(self):
        result = TruthScorer.apply_confirmation(0.7)
        assert result > 0.7

    def test_confirmation_capped_at_1(self):
        result = TruthScorer.apply_confirmation(0.98)
        assert result <= 1.0

    def test_confirmation_resets_decay_clock(self):
        """Confirming a fact should reset the decay anchor.

        When confirmation happens, the caller updates last_confirmed_at.
        Subsequent decay should be measured from that new timestamp,
        not the original creation time.
        """
        # Old fact with decayed score
        old_confirmed = (datetime.now() - timedelta(weeks=10)).isoformat()
        metadata_old = {"truth_score": 0.7, "last_confirmed_at": old_confirmed}
        decayed_truth = TruthScorer.compute_effective_truth(metadata_old)
        assert decayed_truth < 0.7  # Should have decayed

        # After confirmation: truth boosted + new timestamp
        new_truth = TruthScorer.apply_confirmation(0.7)
        metadata_new = {
            "truth_score": new_truth,
            "last_confirmed_at": datetime.now().isoformat(),
        }
        fresh_truth = TruthScorer.compute_effective_truth(metadata_new)
        # Fresh confirmation should be higher than the decayed version
        assert fresh_truth > decayed_truth
        # And close to the boosted score (minimal decay since just confirmed)
        assert abs(fresh_truth - new_truth) < 0.01


class TestCorrection:
    """Test correction penalty."""

    def test_correction_penalty_applied(self):
        result = TruthScorer.apply_correction(0.7)
        assert result < 0.7
        assert result == pytest.approx(0.45, abs=0.01)  # 0.7 - 0.25

    def test_correction_floor_at_zero(self):
        result = TruthScorer.apply_correction(0.1)
        assert result >= 0.0


class TestContradiction:
    """Test contradiction penalty."""

    def test_contradiction_penalty_less_than_correction(self):
        corr = TruthScorer.apply_correction(0.7)
        cont = TruthScorer.apply_contradiction(0.7)
        # Contradiction should be less severe than correction
        assert cont > corr

    def test_contradiction_reduces_score(self):
        result = TruthScorer.apply_contradiction(0.7)
        assert result < 0.7

    def test_contradiction_floor_at_zero(self):
        result = TruthScorer.apply_contradiction(0.05)
        assert result >= 0.0


class TestTimeDecay:
    """Test time-based decay."""

    def test_recent_fact_no_decay(self):
        now = datetime.now()
        result = TruthScorer.apply_time_decay(0.7, now)
        assert abs(result - 0.7) < 0.001

    def test_old_fact_decays(self):
        ten_weeks_ago = datetime.now() - timedelta(weeks=10)
        result = TruthScorer.apply_time_decay(0.7, ten_weeks_ago)
        assert result < 0.7

    def test_decay_floor(self):
        """Score should never decay below the configured floor."""
        very_old = datetime.now() - timedelta(weeks=100)
        result = TruthScorer.apply_time_decay(0.7, very_old)
        assert result >= 0.3  # TRUTH_SCORER_DECAY_FLOOR default

    def test_decay_with_string_timestamp(self):
        ts = (datetime.now() - timedelta(weeks=5)).isoformat()
        result = TruthScorer.apply_time_decay(0.7, ts)
        assert result < 0.7
        assert result >= 0.3

    def test_decay_with_none_timestamp(self):
        result = TruthScorer.apply_time_decay(0.7, None)
        assert result == 0.7  # No decay when no timestamp

    def test_decay_independent_of_retrieval(self):
        """Decay should be the same regardless of how many times a fact is retrieved.

        This is the anti-echo-chamber test: accessing a memory should NOT
        increase its truth score. Decay is purely time-based.
        """
        five_weeks_ago = datetime.now() - timedelta(weeks=5)
        metadata = {
            "truth_score": 0.7,
            "last_confirmed_at": five_weeks_ago.isoformat(),
            "access_count": 0,
        }

        # First "retrieval"
        score_first = TruthScorer.compute_effective_truth(metadata)

        # Simulate 100 accesses (old system would boost truth)
        metadata["access_count"] = 100
        score_after_many_accesses = TruthScorer.compute_effective_truth(metadata)

        # Score should be essentially identical — access count is irrelevant
        # (tiny float diff from datetime.now() advancing between calls)
        assert score_first == pytest.approx(score_after_many_accesses, abs=1e-6)


class TestComputeEffectiveTruth:
    """Test the main entry point."""

    def test_with_full_metadata(self):
        metadata = {
            "truth_score": 0.8,
            "last_confirmed_at": datetime.now().isoformat(),
        }
        result = TruthScorer.compute_effective_truth(metadata)
        assert 0.79 <= result <= 0.81

    def test_with_no_truth_score(self):
        metadata = {"timestamp": datetime.now().isoformat()}
        result = TruthScorer.compute_effective_truth(metadata)
        # Falls back to TRUTH_SCORER_INITIAL_SCORE (0.7)
        assert 0.69 <= result <= 0.71

    def test_with_empty_metadata(self):
        result = TruthScorer.compute_effective_truth({})
        assert 0.0 <= result <= 1.0

    def test_disabled_falls_back_to_stored(self):
        """When TRUTH_SCORER_ENABLED is False, use stored truth_score directly."""
        metadata = {"truth_score": 0.42}
        with patch("memory.truth_scorer.TRUTH_SCORER_ENABLED", False):
            result = TruthScorer.compute_effective_truth(metadata)
        assert result == pytest.approx(0.42)

    def test_echo_chamber_prevention(self):
        """Frequently retrieved facts should still decay — the core safeguard.

        In the old system, every retrieval boosted truth_score, creating
        an echo chamber where popular memories became artificially "true".
        The new system must ensure decay happens regardless of access patterns.
        """
        # Fact created 8 weeks ago, accessed frequently
        eight_weeks_ago = datetime.now() - timedelta(weeks=8)
        metadata = {
            "truth_score": 0.7,
            "last_confirmed_at": eight_weeks_ago.isoformat(),
            "access_count": 50,  # Old system would boost this
            "last_accessed": datetime.now().isoformat(),
        }

        result = TruthScorer.compute_effective_truth(metadata)

        # Must be lower than stored score despite frequent access
        assert result < 0.7
        # Must be above floor
        assert result >= 0.3
