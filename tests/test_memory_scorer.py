"""
Tests for memory_scorer.py — standalone memory scoring and ranking engine.

Module Contract
- Purpose: Verify the 12-step scoring pipeline, TruthScorer, ScorerConfig,
  temporal decay, graph boosting, staleness penalties, and CLI entry points.
- Inputs: Synthetic memory dicts, config overrides, mock graph scorers.
- Outputs: Pass/fail assertions on scores, rankings, and debug dicts.
- Dependencies: pytest, standard library only. No external fixtures or conftest.
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Set
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import memory_scorer
from memory_scorer import (
    GraphScorerProtocol,
    MemoryScorer,
    ScorerConfig,
    TruthScorer,
    _analogy_markers,
    _build_anchor_tokens,
    _is_deictic_followup,
    _num_op_density,
    _salient_tokens,
    _stem,
)


# ============================================================================
# Fixtures
# ============================================================================

NOW = datetime(2026, 5, 23, 12, 0, 0)


def _make_memory(
    query="test query",
    response="test response",
    relevance=0.5,
    hours_ago=1,
    collection="conversations",
    truth_score=0.7,
    **extra_metadata,
):
    """Build a synthetic memory dict for testing."""
    ts = (NOW - timedelta(hours=hours_ago)).isoformat()
    md = {"truth_score": truth_score, **extra_metadata}
    return {
        "query": query,
        "response": response,
        "relevance_score": relevance,
        "timestamp": ts,
        "collection": collection,
        "metadata": md,
    }


def _scorer(**kwargs):
    """Build a MemoryScorer pinned to NOW."""
    kwargs.setdefault("time_fn", lambda: NOW)
    return MemoryScorer(**kwargs)


# ============================================================================
# TruthScorer tests
# ============================================================================


class TestTruthScorer:
    """Tests for the stateless TruthScorer engine."""

    def test_initial_score_user_stated(self):
        ts = TruthScorer()
        assert ts.calculate_initial_score("user_stated") == 0.8

    def test_initial_score_llm_extracted(self):
        ts = TruthScorer()
        assert ts.calculate_initial_score("llm_extracted") == 0.7

    def test_initial_score_inferred(self):
        ts = TruthScorer()
        assert ts.calculate_initial_score("inferred") == 0.5

    def test_initial_score_unknown_source(self):
        ts = TruthScorer()
        score = ts.calculate_initial_score("some_unknown_source")
        assert score == 0.7  # falls back to initial_score default

    def test_initial_score_custom_config(self):
        cfg = ScorerConfig(truth_source_scores={"custom": 0.99})
        ts = TruthScorer(cfg)
        assert ts.calculate_initial_score("custom") == 0.99

    def test_apply_confirmation(self):
        ts = TruthScorer()
        assert ts.apply_confirmation(0.7) == pytest.approx(0.85)

    def test_apply_confirmation_caps_at_1(self):
        ts = TruthScorer()
        assert ts.apply_confirmation(0.95) == 1.0

    def test_apply_correction(self):
        ts = TruthScorer()
        assert ts.apply_correction(0.7) == pytest.approx(0.45)

    def test_apply_correction_floors_at_0(self):
        ts = TruthScorer()
        assert ts.apply_correction(0.1) == 0.0

    def test_apply_contradiction(self):
        ts = TruthScorer()
        assert ts.apply_contradiction(0.7) == pytest.approx(0.55)

    def test_apply_contradiction_floors_at_0(self):
        ts = TruthScorer()
        assert ts.apply_contradiction(0.05) == 0.0

    def test_time_decay_no_timestamp(self):
        ts = TruthScorer()
        assert ts.apply_time_decay(0.8) == 0.8

    def test_time_decay_recent(self):
        ts = TruthScorer()
        result = ts.apply_time_decay(0.8, NOW - timedelta(hours=1), now=NOW)
        assert result == pytest.approx(0.8, abs=0.01)  # almost no decay

    def test_time_decay_old(self):
        ts = TruthScorer()
        result = ts.apply_time_decay(0.8, NOW - timedelta(weeks=10), now=NOW)
        assert result < 0.8
        assert result >= 0.3  # above floor

    def test_time_decay_very_old_hits_floor(self):
        ts = TruthScorer()
        result = ts.apply_time_decay(0.8, NOW - timedelta(weeks=100), now=NOW)
        assert result == pytest.approx(0.3)

    def test_time_decay_string_timestamp(self):
        ts = TruthScorer()
        ts_str = (NOW - timedelta(weeks=5)).isoformat()
        result = ts.apply_time_decay(0.8, ts_str, now=NOW)
        assert result < 0.8

    def test_time_decay_invalid_timestamp(self):
        ts = TruthScorer()
        assert ts.apply_time_decay(0.8, "not-a-date", now=NOW) == 0.8

    def test_compute_effective_truth_basic(self):
        ts = TruthScorer()
        md = {"truth_score": 0.8, "last_confirmed_at": NOW.isoformat()}
        result = ts.compute_effective_truth(md)
        assert result == pytest.approx(0.8, abs=0.01)

    def test_compute_effective_truth_no_score(self):
        ts = TruthScorer()
        result = ts.compute_effective_truth({})
        assert result == 0.7  # initial_score default

    def test_compute_effective_truth_disabled(self):
        cfg = ScorerConfig(truth_enabled=False)
        ts = TruthScorer(cfg)
        result = ts.compute_effective_truth({"truth_score": 0.9})
        assert result == 0.9

    def test_compute_effective_truth_disabled_default(self):
        cfg = ScorerConfig(truth_enabled=False)
        ts = TruthScorer(cfg)
        result = ts.compute_effective_truth({})
        assert result == 0.6  # legacy fallback

    def test_compute_effective_truth_falls_back_to_timestamp(self):
        ts = TruthScorer()
        md = {"truth_score": 0.8, "timestamp": (NOW - timedelta(weeks=5)).isoformat()}
        result = ts.compute_effective_truth(md)
        assert result < 0.8


# ============================================================================
# Text helper tests
# ============================================================================


class TestTextHelpers:
    """Tests for stemming, tokenization, and density helpers."""

    def test_stem_short_word(self):
        assert _stem("cat") == "cat"

    def test_stem_suffix_strip(self):
        assert _stem("features") == "featur"
        assert _stem("deployment") == "deploy"
        assert _stem("anxious") == "anx"

    def test_stem_no_strip(self):
        assert _stem("asyncio") == "asyncio"

    def test_salient_tokens(self):
        tokens = _salient_tokens("The quick brown fox jumps over the lazy dog")
        assert "quick" in tokens
        assert "the" not in tokens  # stopword

    def test_salient_tokens_empty(self):
        assert _salient_tokens("") == set()

    def test_num_op_density_math(self):
        d = _num_op_density("3 + 4 = 7")
        assert d > 0.3

    def test_num_op_density_text(self):
        d = _num_op_density("hello world how are you")
        assert d == 0.0

    def test_num_op_density_empty(self):
        assert _num_op_density("") == 0.0

    def test_analogy_markers(self):
        assert _analogy_markers("It's like a river flowing") >= 1
        assert _analogy_markers("The sky is blue") == 0

    def test_is_deictic_followup(self):
        assert _is_deictic_followup("explain that again")
        assert not _is_deictic_followup("what is 2+2")

    def test_build_anchor_tokens(self):
        ctx = [{"query": "what is f(x)", "response": "f(x) = 2x + 3"}]
        anchors = _build_anchor_tokens(ctx)
        assert len(anchors) > 0

    def test_build_anchor_tokens_empty(self):
        assert _build_anchor_tokens([]) == set()


# ============================================================================
# ScorerConfig tests
# ============================================================================


class TestScorerConfig:
    """Tests for configuration dataclass."""

    def test_default_weights_sum(self):
        cfg = ScorerConfig()
        total = sum(cfg.score_weights.values())
        assert total == pytest.approx(1.0)

    def test_custom_config(self):
        cfg = ScorerConfig(recency_decay_rate=0.1, staleness_enabled=False)
        assert cfg.recency_decay_rate == 0.1
        assert cfg.staleness_enabled is False

    def test_debug_default_off(self):
        assert ScorerConfig().debug is False


# ============================================================================
# MemoryScorer tests
# ============================================================================


class TestMemoryScorerBasic:
    """Basic MemoryScorer tests."""

    def test_empty_memories(self):
        scorer = _scorer()
        assert scorer.rank_memories([], "test") == []

    def test_single_memory_gets_score(self):
        scorer = _scorer()
        m = _make_memory()
        result = scorer.rank_memories([m], "test query")
        assert len(result) == 1
        assert "final_score" in result[0]
        assert result[0]["final_score"] > 0

    def test_higher_relevance_ranks_higher(self):
        scorer = _scorer()
        low = _make_memory(relevance=0.3, query="low rel")
        high = _make_memory(relevance=0.9, query="high rel")
        result = scorer.rank_memories([low, high], "test")
        assert result[0]["relevance_score"] == 0.9

    def test_more_recent_ranks_higher(self):
        scorer = _scorer()
        old = _make_memory(hours_ago=100, relevance=0.7, query="old")
        new = _make_memory(hours_ago=1, relevance=0.7, query="new")
        result = scorer.rank_memories([old, new], "test")
        assert result[0] is new

    def test_collection_boost(self):
        scorer = _scorer()
        fact = _make_memory(collection="facts", relevance=0.5)
        conv = _make_memory(collection="conversations", relevance=0.5)
        result = scorer.rank_memories([conv, fact], "test")
        assert result[0]["collection"] == "facts"

    def test_custom_collection_boosts(self):
        cfg = ScorerConfig(collection_boosts={"custom": 0.5})
        scorer = _scorer(config=cfg)
        m = _make_memory(collection="custom", relevance=0.3)
        result = scorer.rank_memories([m], "test")
        assert result[0]["final_score"] > 0

    def test_debug_dict_when_enabled(self):
        cfg = ScorerConfig(debug=True)
        scorer = _scorer(config=cfg)
        m = _make_memory()
        result = scorer.rank_memories([m], "test")
        assert "debug" in result[0]
        dbg = result[0]["debug"]
        assert "rel" in dbg
        assert "recency" in dbg
        assert "truth" in dbg

    def test_no_debug_dict_by_default(self):
        scorer = _scorer()
        # Ensure logger is NOT at DEBUG level
        memory_scorer.logger.setLevel(logging.WARNING)
        m = _make_memory()
        result = scorer.rank_memories([m], "test")
        assert "debug" not in result[0]

    def test_debug_via_logger_level(self):
        scorer = _scorer()
        memory_scorer.logger.setLevel(logging.DEBUG)
        try:
            m = _make_memory()
            result = scorer.rank_memories([m], "test")
            assert "debug" in result[0]
        finally:
            memory_scorer.logger.setLevel(logging.WARNING)


class TestMemoryScorerWeights:
    """Tests for weight overrides and intent-driven scoring."""

    def test_weight_overrides_parameter(self):
        scorer = _scorer()
        m = _make_memory(relevance=0.9, hours_ago=100)
        # Heavy recency weight with old memory should lower score
        result_default = scorer.rank_memories(
            [_make_memory(relevance=0.9, hours_ago=100)], "test"
        )
        result_recency = scorer.rank_memories(
            [m], "test", weight_overrides={"recency": 0.80, "relevance": 0.10}
        )
        # The recency-heavy score should be lower for an old memory
        assert result_recency[0]["final_score"] < result_default[0]["final_score"]

    def test_intent_weight_overrides_attribute(self):
        scorer = _scorer()
        scorer._intent_weight_overrides = {"relevance": 0.90, "recency": 0.05}
        m = _make_memory(relevance=0.9, hours_ago=100)
        result = scorer.rank_memories([m], "test")
        assert result[0]["final_score"] > 0

    def test_explicit_overrides_beat_intent(self):
        scorer = _scorer()
        scorer._intent_weight_overrides = {"relevance": 0.10}
        m1 = _make_memory(relevance=0.9)
        m2 = _make_memory(relevance=0.9)
        # Explicit parameter should take precedence
        r1 = scorer.rank_memories([m1], "test")
        r2 = scorer.rank_memories(
            [m2], "test", weight_overrides={"relevance": 0.90}
        )
        # r2 has much higher relevance weight, so should score higher
        assert r2[0]["final_score"] > r1[0]["final_score"]


class TestTemporalAnchor:
    """Tests for temporal-anchor recency decay."""

    def test_small_anchor_in_window(self):
        scorer = _scorer()
        m = _make_memory(hours_ago=6)
        result = scorer.rank_memories(
            [m], "test",
            weight_overrides={"_temporal_anchor_hours": 24},
        )
        # Within window, recency should be high (~0.96)
        assert result[0]["final_score"] > 0

    def test_small_anchor_past_grace(self):
        scorer = _scorer()
        m_in = _make_memory(hours_ago=6, query="in window")
        m_out = _make_memory(hours_ago=48, query="past grace")
        result = scorer.rank_memories(
            [m_in, m_out], "test",
            weight_overrides={"_temporal_anchor_hours": 24},
        )
        assert result[0] is m_in

    def test_large_anchor_sqrt_ramp(self):
        cfg = ScorerConfig(debug=True)
        scorer = _scorer(config=cfg)
        m = _make_memory(hours_ago=100)
        result = scorer.rank_memories(
            [m], "test",
            weight_overrides={"_temporal_anchor_hours": 168},  # 1 week
        )
        dbg = result[0].get("debug", {})
        # Should be on the sqrt ramp
        assert dbg.get("recency", 0) > 0.5

    def test_corpus_cap_on_large_anchor(self):
        cfg = ScorerConfig(debug=True)
        scorer = _scorer(config=cfg)
        m = _make_memory(relevance=0.9)
        m["source"] = "corpus"
        result = scorer.rank_memories(
            [m], "test",
            weight_overrides={"_temporal_anchor_hours": 72},
        )
        # Corpus relevance should be capped at 0.5
        assert result[0]["debug"]["rel"] <= 0.5


class TestTimelineMode:
    """Tests for _timeline_mode bonus."""

    def test_timeline_boosts_summaries(self):
        cfg = ScorerConfig(debug=True)
        scorer = _scorer(config=cfg)
        summary = _make_memory(collection="summaries", relevance=0.5)
        conv = _make_memory(collection="conversations", relevance=0.5)
        result = scorer.rank_memories(
            [conv, summary], "how has my mood changed over time",
            weight_overrides={"_timeline_mode": True},
        )
        assert result[0]["collection"] == "summaries"


class TestStaleness:
    """Tests for staleness penalty."""

    def test_staleness_penalty_applied(self):
        cfg = ScorerConfig(debug=True)
        scorer = _scorer(config=cfg)
        stale = _make_memory(relevance=0.8, staleness_ratio=0.9)
        fresh = _make_memory(relevance=0.8)
        result = scorer.rank_memories([stale, fresh], "test")
        assert result[0] is fresh

    def test_staleness_steep_multiplier(self):
        cfg = ScorerConfig(debug=True)
        scorer = _scorer(config=cfg)
        # Above steep threshold
        m = _make_memory(relevance=0.8, staleness_ratio=0.9)
        result = scorer.rank_memories([m], "test")
        dbg = result[0]["debug"]
        # Penalty should be > weight * ratio (due to 2x multiplier)
        assert dbg["staleness_penalty"] < -0.15 * 0.9

    def test_staleness_reflection_discount(self):
        cfg = ScorerConfig(debug=True)
        scorer = _scorer(config=cfg)
        refl = _make_memory(collection="reflections", relevance=0.8, staleness_ratio=0.5)
        fact = _make_memory(collection="facts", relevance=0.8, staleness_ratio=0.5)
        result = scorer.rank_memories([refl, fact], "test")
        refl_penalty = abs(result[0]["debug"]["staleness_penalty"] if result[0]["collection"] == "reflections" else result[1]["debug"]["staleness_penalty"])
        fact_penalty = abs(result[0]["debug"]["staleness_penalty"] if result[0]["collection"] == "facts" else result[1]["debug"]["staleness_penalty"])
        assert refl_penalty < fact_penalty

    def test_staleness_disabled(self):
        cfg = ScorerConfig(staleness_enabled=False, debug=True)
        scorer = _scorer(config=cfg)
        m = _make_memory(staleness_ratio=0.9)
        result = scorer.rank_memories([m], "test")
        assert result[0]["debug"]["staleness_penalty"] == 0.0


class TestSizePenalty:
    """Tests for large document size penalty."""

    def test_small_doc_no_penalty(self):
        cfg = ScorerConfig(debug=True)
        scorer = _scorer(config=cfg)
        m = _make_memory(response="short text")
        result = scorer.rank_memories([m], "test")
        assert result[0]["debug"]["size_penalty"] == 0.0

    def test_large_doc_with_keywords_no_penalty(self):
        scorer = _scorer(config=ScorerConfig(debug=True))
        m = _make_memory(response="x" * 20_000)
        m["keyword_score"] = 0.5
        result = scorer.rank_memories([m], "test")
        assert result[0]["debug"]["size_penalty"] == 0.0

    def test_large_doc_without_keywords_penalized(self):
        scorer = _scorer(config=ScorerConfig(debug=True))
        m = _make_memory(response="x" * 20_000)
        m["keyword_score"] = 0.1
        result = scorer.rank_memories([m], "test")
        assert result[0]["debug"]["size_penalty"] < 0


class TestDeictic:
    """Tests for deictic follow-up handling."""

    def test_deictic_with_anchor_overlap(self):
        ctx = [{"query": "tell me about calculus", "response": "Calculus is the study of change"}]
        scorer = _scorer(conversation_context=ctx)
        m = _make_memory(response="Calculus involves derivatives and integrals")
        result = scorer.rank_memories([m], "explain that again")
        assert result[0]["final_score"] > 0

    def test_deictic_drift_guardrail(self):
        cfg = ScorerConfig(debug=True)
        scorer = _scorer(config=cfg)
        # Memory with no overlap to previous context
        m = _make_memory(
            response="completely unrelated topic about cooking",
            hours_ago=48,
        )
        result = scorer.rank_memories([m], "explain that again")
        # Score should be reduced by deictic drift guardrail
        assert result[0]["final_score"] > 0  # still positive


class TestMetaConversational:
    """Tests for meta-conversational bonus."""

    def test_meta_boosts_episodic(self):
        cfg = ScorerConfig(debug=True)
        scorer = _scorer(config=cfg)
        episodic = _make_memory(collection="episodic", relevance=0.5)
        episodic["memory_type"] = "EPISODIC"
        conv = _make_memory(collection="conversations", relevance=0.5)
        result = scorer.rank_memories(
            [conv, episodic], "what have we talked about",
            is_meta_conversational=True,
        )
        ep = [m for m in result if m.get("memory_type") == "EPISODIC"][0]
        assert ep["debug"]["meta_bonus"] == 0.15


class TestTopicMatch:
    """Tests for topic alignment scoring."""

    def test_exact_topic_match(self):
        scorer = _scorer()
        m = _make_memory()
        m["metadata"]["topics"] = ["cooking"]
        score = scorer._calculate_topic_match(m, "cooking")
        assert score == 1.0

    def test_no_topic_info(self):
        scorer = _scorer()
        score = scorer._calculate_topic_match({}, None)
        assert score == 0.5

    def test_different_topic(self):
        scorer = _scorer()
        m = _make_memory()
        m["metadata"]["topics"] = ["cooking"]
        score = scorer._calculate_topic_match(m, "sports")
        assert score == 0.2

    def test_topic_from_tags(self):
        scorer = _scorer()
        m = {"tags": "topic:cooking,topic:food", "metadata": {}}
        score = scorer._calculate_topic_match(m, "cooking")
        assert score == 1.0


class TestToneAdjustment:
    """Tests for tone-based truth reduction."""

    def test_rude_response_truth_penalty(self):
        cfg = ScorerConfig(debug=True)
        scorer = _scorer(config=cfg)
        m = _make_memory(response="you are an idiot for asking this", truth_score=0.8)
        result = scorer.rank_memories([m], "test")
        assert result[0]["debug"]["truth"] < 0.8


class TestGraphScoring:
    """Tests for optional graph-boosted scoring."""

    def test_graph_boost_applied(self):
        class MockGraph:
            def get_related_names(self, query: str) -> Set[str]:
                return {"flapjack", "cooking"}

        cfg = ScorerConfig(debug=True)
        scorer = _scorer(config=cfg)
        scorer.graph_scorer = MockGraph()
        m = _make_memory(response="Flapjack is a great cat who loves cooking")
        result = scorer.rank_memories([m], "tell me about my cat")
        assert result[0]["debug"]["graph_bonus"] > 0

    def test_graph_boost_capped(self):
        class MockGraph:
            def get_related_names(self, query: str) -> Set[str]:
                return {"a", "b", "c", "d", "e"}

        cfg = ScorerConfig(debug=True)
        scorer = _scorer(config=cfg)
        scorer.graph_scorer = MockGraph()
        m = _make_memory(response="a b c d e all mentioned")
        result = scorer.rank_memories([m], "test")
        assert result[0]["debug"]["graph_bonus"] <= 0.15

    def test_no_graph_scorer(self):
        cfg = ScorerConfig(debug=True)
        scorer = _scorer(config=cfg)
        m = _make_memory()
        result = scorer.rank_memories([m], "test")
        assert result[0]["debug"]["graph_bonus"] == 0.0

    def test_graph_protocol_check(self):
        class ValidGraph:
            def get_related_names(self, query: str) -> Set[str]:
                return set()

        assert isinstance(ValidGraph(), GraphScorerProtocol)

    def test_graph_error_graceful(self):
        class BrokenGraph:
            def get_related_names(self, query: str) -> Set[str]:
                raise RuntimeError("graph unavailable")

        scorer = _scorer(config=ScorerConfig(debug=True))
        scorer.graph_scorer = BrokenGraph()
        m = _make_memory()
        result = scorer.rank_memories([m], "test")
        assert result[0]["debug"]["graph_bonus"] == 0.0


class TestConvenienceScorers:
    """Tests for calculate_truth_score and calculate_importance_score."""

    def test_truth_score_basic(self):
        scorer = _scorer()
        score = scorer.calculate_truth_score("what is 2+2?", "The answer is 4")
        assert 0 < score <= 1.0

    def test_truth_score_with_confirmation(self):
        scorer = _scorer()
        s1 = scorer.calculate_truth_score("test", "short")
        s2 = scorer.calculate_truth_score("test", "yes exactly, that makes sense and is correct")
        assert s2 > s1

    def test_truth_score_with_context(self):
        ctx = [{"query": "hello", "response": "hello there"}]
        scorer = _scorer(conversation_context=ctx)
        score = scorer.calculate_truth_score("hello again", "response")
        assert score > 0.5  # continuity bonus

    def test_importance_basic(self):
        scorer = _scorer()
        assert scorer.calculate_importance_score("short") == 0.5

    def test_importance_with_keywords(self):
        scorer = _scorer()
        score = scorer.calculate_importance_score(
            "This is critical and important to remember"
        )
        assert score > 0.5

    def test_importance_caps_at_1(self):
        scorer = _scorer()
        score = scorer.calculate_importance_score(
            "?" * 300 + " important critical essential"
        )
        assert score <= 1.0


class TestApplyTemporalDecay:
    """Tests for the simplified apply_temporal_decay method."""

    def test_recent_memory_high_score(self):
        scorer = _scorer()
        m = _make_memory(hours_ago=1)
        m["relevance_score"] = 0.8
        m["truth_score"] = 0.8
        result = scorer.apply_temporal_decay([m])
        assert result[0]["final_score"] > 0

    def test_old_memory_lower_score(self):
        scorer = _scorer()
        new = _make_memory(hours_ago=1)
        old = _make_memory(hours_ago=1000)
        for m in [new, old]:
            m["relevance_score"] = 0.8
            m["truth_score"] = 0.8
        scorer.apply_temporal_decay([new, old])
        assert new["final_score"] > old["final_score"]

    def test_invalid_timestamp_skipped(self):
        scorer = _scorer()
        m = {"timestamp": "not-a-date", "relevance_score": 0.5}
        result = scorer.apply_temporal_decay([m])
        assert "final_score" not in result[0]


class TestTimestampParsing:
    """Tests for timestamp handling in rank_memories."""

    def test_string_timestamp(self):
        scorer = _scorer()
        m = _make_memory(hours_ago=1)
        # timestamp is already a string from _make_memory
        result = scorer.rank_memories([m], "test")
        assert result[0]["final_score"] > 0

    def test_datetime_timestamp(self):
        scorer = _scorer()
        m = _make_memory()
        m["timestamp"] = NOW - timedelta(hours=1)
        result = scorer.rank_memories([m], "test")
        assert result[0]["final_score"] > 0

    def test_invalid_timestamp_defaults_to_now(self):
        scorer = _scorer()
        m = _make_memory()
        m["timestamp"] = 12345  # not a string or datetime
        result = scorer.rank_memories([m], "test")
        assert result[0]["final_score"] > 0

    def test_bad_string_timestamp(self):
        scorer = _scorer()
        m = _make_memory()
        m["timestamp"] = "not-a-date"
        result = scorer.rank_memories([m], "test")
        assert result[0]["final_score"] > 0  # defaults to now


# ============================================================================
# CLI tests
# ============================================================================


class TestCLI:
    """Tests for CLI entry points."""

    def test_cli_demo(self, capsys):
        memory_scorer._cli_demo()
        captured = capsys.readouterr()
        assert "Memory Scorer Demo" in captured.out
        assert "final_score" not in captured.out or "score=" in captured.out

    def test_cli_truth(self, capsys):
        md = json.dumps({"truth_score": 0.8, "last_confirmed_at": NOW.isoformat()})
        memory_scorer._cli_truth([md])
        captured = capsys.readouterr()
        assert "effective_truth:" in captured.out

    def test_cli_score(self, capsys):
        m = json.dumps({
            "relevance_score": 0.8,
            "timestamp": NOW.isoformat(),
            "metadata": {"truth_score": 0.7},
        })
        memory_scorer._cli_score([m, "--query", "test query"])
        captured = capsys.readouterr()
        assert "final_score:" in captured.out

    def test_cli_score_no_args(self):
        with pytest.raises(SystemExit):
            memory_scorer._cli_score([])

    def test_cli_truth_bad_json(self):
        with pytest.raises(SystemExit):
            memory_scorer._cli_truth(["not json"])

    def test_cli_main_unknown_command(self):
        with patch.object(sys, "argv", ["memory_scorer.py", "unknown"]):
            with pytest.raises(SystemExit):
                memory_scorer.main()

    def test_cli_main_no_args(self):
        with patch.object(sys, "argv", ["memory_scorer.py"]):
            with pytest.raises(SystemExit):
                memory_scorer.main()
