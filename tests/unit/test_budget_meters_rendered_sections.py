"""Token budget must meter the sections the formatter RENDERS (2026-08-14).

Live incident: a turn's true prompt hit 17.5K tokens against the 10K budget.
Root cause: the formatter renders the SPLIT summary/reflection keys
(recent_summaries / semantic_summaries / recent_reflections /
semantic_reflections) but PRIORITY_ORDER metered the combined
"summaries"/"reflections" keys — which nothing renders. So the four rendered
sections were invisible to the budget AND untrimmable, and the builder's
floors were topping up the dead keys with retrieval calls whose output never
reached the prompt.
"""

import re
from pathlib import Path

import pytest

from core.prompt.token_manager import (
    PRIORITY_ORDER,
    UNRENDERED_CONTEXT_KEYS,
    TokenManager,
)


_PRIORITY_NAMES = {name for name, _ in PRIORITY_ORDER}


class TestPriorityOrderCoversRenderedKeys:
    def test_split_summary_reflection_keys_are_metered(self):
        for key in (
            "recent_summaries",
            "semantic_summaries",
            "recent_reflections",
            "semantic_reflections",
        ):
            assert key in _PRIORITY_NAMES, f"rendered key {key} missing from PRIORITY_ORDER"

    def test_dead_combined_keys_are_not_metered(self):
        # The combined keys are inputs nothing renders — metering them would
        # spend budget on invisible content.
        assert "summaries" not in _PRIORITY_NAMES
        assert "reflections" not in _PRIORITY_NAMES
        assert "summaries" in UNRENDERED_CONTEXT_KEYS
        assert "reflections" in UNRENDERED_CONTEXT_KEYS

    def test_calendar_and_schedule_are_metered(self):
        assert "google_calendar" in _PRIORITY_NAMES
        assert "upcoming_schedule" in _PRIORITY_NAMES

    def test_formatter_rendered_keys_parity(self):
        """Every list/str context key the formatter reads must be metered (or
        be a known non-text exception). Fails loudly when a new rendered
        section is added without a PRIORITY_ORDER row — the silent-unmetered
        failure mode behind the 17.5K prompt."""
        formatter_src = Path("core/prompt/formatter.py").read_text()
        rendered_keys = set(re.findall(r'context\.get\("(\w+)"', formatter_src))
        # Non-text or metadata keys the budget deliberately does not meter:
        # images (note_images, visual_memories), STM metadata, citation map.
        exceptions = {
            "note_images", "visual_memories", "stm_summary", "memory_id_map",
        }
        unmetered = rendered_keys - _PRIORITY_NAMES - exceptions
        assert not unmetered, (
            f"Formatter renders unmetered context keys: {sorted(unmetered)} — "
            "add PRIORITY_ORDER rows (or an explicit exception here)"
        )


class _FakeModelManager:
    def get_active_model_name(self):
        return "test-model"


class _FakeTokenizerManager:
    def count_tokens(self, text, model_name=None):
        return len((text or "").split())


def _mk_items(n, words_per_item=50, tag="item"):
    return [{"content": " ".join([f"{tag}{i}"] * words_per_item)} for i in range(n)]


class TestBudgetTrimsRenderedSections:
    def _manager(self, budget):
        return TokenManager(_FakeModelManager(), _FakeTokenizerManager(), budget)

    def test_over_budget_context_trims_semantic_reflections(self):
        # Lowest-priority rendered content must actually shrink now that the
        # split keys are metered (pre-fix: untouched at any budget).
        ctx = {
            "recent_conversations": _mk_items(4, tag="conv"),
            "recent_summaries": _mk_items(5, tag="rsum"),
            "semantic_summaries": _mk_items(5, tag="ssum"),
            "recent_reflections": _mk_items(4, tag="rref"),
            "semantic_reflections": _mk_items(4, tag="sref"),
        }
        tm = self._manager(budget=600)
        trimmed = tm._manage_token_budget(ctx)
        total = sum(
            len(it["content"].split())
            for key in ctx
            for it in trimmed.get(key, [])
        )
        assert total <= 600
        kept_refl = len(trimmed.get("recent_reflections", [])) + len(
            trimmed.get("semantic_reflections", [])
        )
        assert kept_refl < 8, "reflections were not trimmed at all"

    def test_under_budget_context_untouched(self):
        ctx = {
            "recent_conversations": _mk_items(2, words_per_item=10),
            "recent_summaries": _mk_items(2, words_per_item=10),
        }
        tm = self._manager(budget=10_000)
        trimmed = tm._manage_token_budget(ctx)
        assert len(trimmed["recent_conversations"]) == 2
        assert len(trimmed["recent_summaries"]) == 2

    def test_dead_combined_keys_do_not_consume_budget(self):
        # A fat combined "summaries" key (unrendered input) must not push
        # rendered content out of the budget.
        ctx = {
            "summaries": _mk_items(50, words_per_item=100, tag="dead"),
            "recent_conversations": _mk_items(3, words_per_item=50, tag="conv"),
        }
        tm = self._manager(budget=200)
        trimmed = tm._manage_token_budget(ctx)
        assert len(trimmed["recent_conversations"]) == 3
        # The dead key passes through untouched (back-compat), just unmetered.
        assert len(trimmed["summaries"]) == 50
