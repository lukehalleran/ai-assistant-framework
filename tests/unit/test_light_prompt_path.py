"""Light-prompt path for casual acknowledgments (2026-07-15).

QueryAnalysis.is_small_talk was DEAD wiring: the builder read it via
getattr(..., False) but no code ever set the field, so the lightweight
context path never fired — a 7-word "Hmm not working yet" pulled a
23K-token full-apparatus prompt (30 memories, git history, docs, wiki).

Tests drive THE deployed functions: query_checker.is_casual_acknowledgment,
analyze_query (field wiring + heavy-topic exclusion), and the builder's
_should_use_light_path routing incl. the two crisis_level encodings.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from utils.query_checker import analyze_query, is_casual_acknowledgment
from core.prompt.builder import UnifiedPromptBuilder


class TestCasualAcknowledgment:
    @pytest.mark.parametrize("q", [
        "Hmm not working yet",   # the motivating 23K-token turn
        "ok",
        "ok cool",
        "thanks!",
        "yeah makes sense",
        "lol nice",
        "alright, on it",
        "hmm still broken",
        "nah all good",
    ])
    def test_acks_detected(self, q):
        assert is_casual_acknowledgment(q)

    @pytest.mark.parametrize("q", [
        "ok how do i fix this",              # interrogative anywhere
        "what's the weather in chicago",     # question lead
        "can you check my email",            # request marker
        "ok can you rerun it",               # ack opener but a request
        "please save that as a doc",         # command
        "did we talk about this yesterday?", # meta-conversational
        "yeah so I was thinking about the gate design and whether "
        "the threshold should be quantile matched across spaces",  # too long
        "tell me more",
        "",
    ])
    def test_non_acks_rejected(self, q):
        assert not is_casual_acknowledgment(q)

    def test_analyze_query_sets_field(self):
        assert analyze_query("Hmm not working yet").is_small_talk
        assert not analyze_query("how does the memory gate work?").is_small_talk

    def test_heavy_topic_never_small_talk(self):
        # Ack-shaped opener + crisis content must keep the full apparatus
        res = analyze_query("ugh i want to kill myself")
        assert res.is_heavy_topic
        assert not res.is_small_talk


class TestBuilderRouting:
    def _builder(self):
        return UnifiedPromptBuilder.__new__(UnifiedPromptBuilder)

    def test_light_path_for_ack(self):
        b = self._builder()
        qa = analyze_query("ok cool")
        assert b._should_use_light_path(qa, None)
        assert b._should_use_light_path(qa, "conversational")

    @pytest.mark.parametrize("crisis", [
        "CrisisLevel.HIGH",     # str(enum) — orchestrator encoding
        "CrisisLevel.MEDIUM",
        "CrisisLevel.CONCERN",
        "crisis_support",       # enum value — crisis_level_str encoding
        "elevated_support",
        "light_support",
    ])
    def test_elevated_tone_forces_full_context(self, crisis):
        b = self._builder()
        qa = analyze_query("ok cool")
        assert not b._should_use_light_path(qa, crisis)

    def test_config_kill_switch(self):
        b = self._builder()
        qa = analyze_query("ok cool")
        with patch("config.app_config.LIGHT_PROMPT_ENABLED", False):
            assert not b._should_use_light_path(qa, None)

    def test_non_ack_full_context(self):
        b = self._builder()
        qa = analyze_query("how does the memory gate work?")
        assert not b._should_use_light_path(qa, None)

    @pytest.mark.asyncio
    async def test_build_prompt_returns_lightweight_context(self):
        """End-to-end through THE deployed build_prompt entry: an ack turn
        must return the lightweight context without touching retrieval."""
        b = self._builder()
        b.context_gatherer = MagicMock()
        b.time_manager = None
        sentinel = {"recent_conversations": [], "memories": [], "light": True}
        b._build_lightweight_context = AsyncMock(return_value=sentinel)

        out = await b.build_prompt("Hmm not working yet")
        assert out is sentinel
        b._build_lightweight_context.assert_awaited_once()
