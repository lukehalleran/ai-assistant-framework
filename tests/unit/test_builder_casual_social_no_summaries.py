"""A casual_social turn renders no [RECENT SUMMARIES]; other intents keep their floor.

Incident date: 2026-10-27 (M1 V5: 6 of 7 casual_social prompts since 10-12 rendered
`[RECENT SUMMARIES] n=1`; owner intent, FOLLOWUPS 09-21: zero on small talk). Class: BC-24
(floor / reserve vs. the intent cap), BC-58.

Drives THE deployed `UnifiedPromptBuilder.build_prompt` with the REAL casual_social retrieval
overrides from the intent profile table and a store that HAS summaries (so the floor top-up
could fire if it were allowed to). Fixture text is synthetic.
"""
from unittest.mock import AsyncMock

import pytest

from core.intent_classifier import IntentType, _PROFILES
from tests.unit.test_independent_prompt_audit import full_builder, retrieval_limits

RECENT = [{"query": "Synthetic hello", "response": "Short reply."}]
SUMMARIES = [
    {"content": "Synthetic summary one about gardening.", "timestamp": "2026-10-20T10:00:00"},
    {"content": "Synthetic summary two about pottery.", "timestamp": "2026-10-21T10:00:00"},
    {"content": "Synthetic summary three about hiking.", "timestamp": "2026-10-22T10:00:00"},
]


def _builder(monkeypatch):
    builder = full_builder(monkeypatch, RECENT, budget=10000)
    builder.memory_coordinator.get_summaries = lambda count: list(SUMMARIES)
    builder.context_gatherer._get_summaries_separate = AsyncMock(
        return_value=([], [])
    )
    return builder


def _overrides(intent):
    """The harness's all-off limits with the intent profile's REAL recent/summaries caps."""
    prof = _PROFILES[intent]["retrieval"]
    ro = retrieval_limits()
    ro["max_recent"] = prof["max_recent"]
    ro["max_summaries"] = prof["max_summaries"]
    return ro


def test_casual_social_profile_cap_is_zero():
    assert _PROFILES[IntentType.CASUAL_SOCIAL]["retrieval"]["max_summaries"] == 0


@pytest.mark.asyncio
async def test_casual_social_turn_renders_no_summaries(monkeypatch):
    builder = _builder(monkeypatch)
    result = await builder.build_prompt("hey there", retrieval_overrides=_overrides(IntentType.CASUAL_SOCIAL))
    assert "_build_time" in result, "builder returned its error fallback"
    assert not result.get("recent_summaries")
    assert not result.get("semantic_summaries")
    builder.context_gatherer._get_summaries_separate.assert_not_called()


@pytest.mark.asyncio
async def test_other_intent_keeps_its_summary_floor(monkeypatch):
    builder = _builder(monkeypatch)
    ro = _overrides(IntentType.CASUAL_SOCIAL)
    ro["max_summaries"] = 4          # an intent that wants summaries (not casual_social)
    result = await builder.build_prompt("hey there", retrieval_overrides=ro)
    assert "_build_time" in result
    assert len(result.get("recent_summaries") or []) == 2   # PROMPT_MIN_SUMMARIES_FLOOR


@pytest.mark.asyncio
async def test_only_casual_social_changed_in_the_profile_table():
    for intent, prof in _PROFILES.items():
        if intent is IntentType.CASUAL_SOCIAL:
            continue
        cap = prof["retrieval"].get("max_summaries")
        assert cap is None or cap >= 1, intent
