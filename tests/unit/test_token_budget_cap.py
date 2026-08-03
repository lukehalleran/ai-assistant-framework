"""
Regression tests for the 2026-07-25 token-budget incident.

The 2026-07-15 preregistered budget experiment adopted `token_budget.default:
10000`, but `_compute_token_budget`'s model-aware path (ctx * fraction, capped
only by the 16K ceiling) overrode it for every API model with ctx >= ~84K —
prod silently ran budget=15360, the experiment's LOSING 15K arm. Compounding
it, `get_context_limit()` hardcoded 128000 for ALL API models ("Default GPT-4
Turbo context") instead of consulting the model registry.

Fixes under test:
- MODEL_CONTEXT_LIMITS registry (full-slug keyed, DEFAULT_API_CONTEXT_LIMIT
  fallback) consulted by get_context_limit().
- API budgets cap at PROMPT_TOKEN_BUDGET_DEFAULT — the context fraction can
  only lower the budget (small-ctx models), never raise it above the
  experiment-adopted value.
"""

from unittest.mock import MagicMock, patch

import core.prompt.builder as builder_mod
from core.prompt.builder import _compute_token_budget
from models.model_manager import (
    DEFAULT_API_CONTEXT_LIMIT,
    MODEL_CONTEXT_LIMITS,
    ModelManager,
)


def _mock_mm(ctx_limit, is_local=False, name="some-model"):
    mm = MagicMock()
    mm.get_context_limit.return_value = ctx_limit
    mm.is_api_model.return_value = not is_local
    mm.get_active_model_name.return_value = name
    return mm


def _budget_constants(**overrides):
    base = {
        "PROMPT_TOKEN_BUDGET_OVERRIDE": None,
        "PROMPT_TOKEN_BUDGET_DEFAULT": 10000,
        "PROMPT_TOKEN_BUDGET_LOCAL": 12000,
        "PROMPT_TOKEN_BUDGET_FLOOR": 8000,
        "PROMPT_TOKEN_BUDGET_CEILING": 16000,
        "PROMPT_TOKEN_BUDGET_CONTEXT_FRACTION": 0.12,
    }
    base.update(overrides)
    return patch.multiple(builder_mod, **base)


class TestComputeTokenBudget:
    def test_large_ctx_api_model_caps_at_experiment_default(self):
        # Kimi K3: 1.05M ctx -> raw 126000 -> must cap at 10000, NOT 16000.
        with _budget_constants():
            assert _compute_token_budget(_mock_mm(1_050_000)) == 10000

    def test_128k_api_model_caps_at_experiment_default(self):
        # The old live bug: 128000 * 0.12 = 15360 (the losing 15K arm).
        with _budget_constants():
            assert _compute_token_budget(_mock_mm(128_000)) == 10000

    def test_small_ctx_api_model_scales_below_default(self):
        # 70K ctx -> raw 8400: fraction may lower the budget below default.
        with _budget_constants():
            assert _compute_token_budget(_mock_mm(70_000)) == 8400

    def test_tiny_ctx_api_model_hits_floor(self):
        with _budget_constants():
            assert _compute_token_budget(_mock_mm(30_000)) == 8000

    def test_local_model_behavior_unchanged(self):
        # Local path: min(raw, LOCAL) with floor — pre-existing semantics.
        with _budget_constants():
            assert _compute_token_budget(_mock_mm(128_000, is_local=True)) == 12000

    def test_env_override_still_wins(self):
        with _budget_constants(PROMPT_TOKEN_BUDGET_OVERRIDE=15000):
            assert _compute_token_budget(_mock_mm(1_050_000)) == 15000

    def test_no_model_manager_uses_default(self):
        with _budget_constants():
            assert _compute_token_budget(None) == 10000


class TestGetContextLimit:
    def _mm_self(self, active_name, api_models):
        m = MagicMock()
        m.get_active_model_name.return_value = active_name
        m.is_api_model.return_value = True
        m.api_models = api_models
        return m

    def test_registered_model_resolves_via_alias(self):
        m = self._mm_self("kimi-3", {"kimi-3": "moonshotai/kimi-k3"})
        assert ModelManager.get_context_limit(m) == 1_050_000

    def test_already_resolved_slug(self):
        m = self._mm_self("moonshotai/kimi-k3", {})
        assert ModelManager.get_context_limit(m) == 1_050_000

    def test_unknown_slug_falls_back(self):
        m = self._mm_self("provider/some-new-model", {})
        assert ModelManager.get_context_limit(m) == DEFAULT_API_CONTEXT_LIMIT

    def test_claude_models_have_context_rows(self):
        # Every Claude row in the registry carries the 200K standard limit.
        claude = [s for s in MODEL_CONTEXT_LIMITS if s.startswith("anthropic/claude")]
        assert claude, "expected Claude entries in MODEL_CONTEXT_LIMITS"
        assert all(MODEL_CONTEXT_LIMITS[s] == 200_000 for s in claude)

    def test_registry_values_are_positive_ints(self):
        for slug, ctx in MODEL_CONTEXT_LIMITS.items():
            assert isinstance(ctx, int) and ctx > 0, slug
