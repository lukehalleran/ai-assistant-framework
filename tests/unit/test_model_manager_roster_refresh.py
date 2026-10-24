"""
Model roster refresh against the live OpenRouter catalog (2026-10-23).

Registers the 5.5-generation Claude, GPT-6.1 Sol / 6 Luna, Gemini 4 Argon,
Kimi K3.1 Pro and DeepSeek V4.1 Flash 1017 routes with their catalog facts
(tools/reasoning/image + context length), retires the models nothing references,
keeps the ones still referenced by code defaults, and pins that NO request-shape
constraint is declared for a route that has not been probed (the shape is set
after the paid probe — declaring one blindly is how fable-5 400'd for months).

class: BC-10 (a registered model is reachable), BC-82/BC-83 (request shapes).
"""
import pytest

from models.model_manager import (
    API_MODEL_ALIASES,
    MODEL_CAPABILITIES,
    MODEL_CONTEXT_LIMITS,
    _slug_supports_prompt_caching,
    _slug_supports_reasoning,
    _slug_supports_tools,
    _slug_supports_vision,
)
from models.tokenizer_manager import _choose_tiktoken_encoding

NEW = {
    "claude-opus-5.5": ("anthropic/claude-opus-5.5", 1_000_000),
    "opus-5.5": ("anthropic/claude-opus-5.5", 1_000_000),
    "claude-opus": ("anthropic/claude-opus-5.5", 1_000_000),
    "sonnet-5.5": ("anthropic/claude-sonnet-5.5", 1_000_000),
    "haiku-5.5": ("anthropic/claude-haiku-5.5", 1_000_000),
    "gpt-6.1": ("openai/gpt-6.1-sol", 1_050_000),
    "gpt-6.1-sol": ("openai/gpt-6.1-sol", 1_050_000),
    "gpt-6-luna": ("openai/gpt-6-luna", 1_050_000),
    "gemini-4-argon": ("google/gemini-4-argon", 1_048_576),
    "kimi-k3.1-pro": ("moonshotai/kimi-k3.1-pro", 1_048_576),
    "kimi-3.1": ("moonshotai/kimi-k3.1-pro", 1_048_576),
    "deepseek-v4.1-flash-1017": ("deepseek/deepseek-v4.1-flash-1017", 1_048_576),
}
# Probed 2026-10-23 (scripts/probe_model_request_shapes.py --run): the declared shapes below
# are the live result; every other new route accepted all three shapes.
PROBED_SHAPES = {
    "anthropic/claude-opus-5.5": {"reasoning_mandatory": True, "forced_tool_choice": False},
    "anthropic/claude-sonnet-5.5": {"reasoning_mandatory": True, "forced_tool_choice": False},
    "openai/gpt-6.1-sol": {"reasoning_mandatory": True},
    "google/gemini-4-argon": {"reasoning_mandatory": True},
}
UNPROBED = sorted({slug for slug, _ in NEW.values()} - set(PROBED_SHAPES))


@pytest.mark.parametrize("slug", sorted(PROBED_SHAPES))
def test_probed_shapes_are_declared(slug):
    row = MODEL_CAPABILITIES[slug]
    assert {k: row[k] for k in SHAPE_KEYS if k in row} == PROBED_SHAPES[slug]
SHAPE_KEYS = ("reasoning_mandatory", "forced_tool_choice", "forced_top_p", "sampling_params")
REMOVED = ["claude-opus-4.6", "claude-opus-4.7", "sonnet-4.6", "gpt-4.1", "gpt-5.1", "kimi-k2-thinking",
           "kimi-k2.6", "glm-4.7", "glm-5", "glm-5-turbo"]
# Still referenced by a code default (utils/notes_common.FALLBACK_MODELS, utils/tag_generator,
# agent_branch workers, main.py) or required to stay: never removed by the refresh.
KEPT = ["gpt-4o", "gpt-4o-mini", "sonnet-4.5", "claude-opus-4.8", "gpt-5.5", "gpt-6-astra", "fable-5",
        "fable-5.1", "deepseek-v4", "deepseek-v4-flash", "deepseek-v4.1-flash", "kimi-k3", "kimi-3",
        "glm-5.2", "claude-opus-4.5", "haiku-4.5", "gpt-5", "deepseek-v3.1", "deepseek-r1", "glm-4.6",
        "gemini-3-pro"]


@pytest.mark.parametrize("alias", sorted(NEW))
def test_alias_resolves_and_has_catalog_context(alias):
    slug, ctx = NEW[alias]
    assert API_MODEL_ALIASES[alias] == slug
    assert slug in MODEL_CAPABILITIES
    assert MODEL_CONTEXT_LIMITS[slug] == ctx  # not the 200K Claude comprehension default


@pytest.mark.parametrize("slug", UNPROBED)
def test_catalog_capabilities_and_no_unprobed_shape(slug):
    # Catalog 2026-10-23: every new route lists tools + reasoning + image input.
    assert _slug_supports_tools(slug) and _slug_supports_vision(slug) and _slug_supports_reasoning(slug)
    row = MODEL_CAPABILITIES[slug]
    assert [k for k in SHAPE_KEYS if k in row] == [], f"{slug} declares an unprobed shape"


def test_caching_modes():
    assert _slug_supports_prompt_caching("anthropic/claude-sonnet-5.5")      # explicit markers
    assert not _slug_supports_prompt_caching("openai/gpt-6.1-sol")           # implicit, like astra
    assert not _slug_supports_prompt_caching("moonshotai/kimi-k3.1-pro")


@pytest.mark.parametrize("alias", REMOVED)
def test_retired_aliases_are_gone_with_their_rows(alias):
    assert alias not in API_MODEL_ALIASES


def test_retired_slugs_have_no_orphan_rows():
    live = set(API_MODEL_ALIASES.values())
    assert set(MODEL_CAPABILITIES) == live
    assert set(MODEL_CONTEXT_LIMITS) <= live


@pytest.mark.parametrize("alias", KEPT)
def test_kept_aliases_still_registered(alias):
    assert alias in API_MODEL_ALIASES


def test_kept_shape_constraints_untouched():
    assert MODEL_CAPABILITIES["moonshotai/kimi-k3"]["forced_top_p"] == 0.95
    assert MODEL_CAPABILITIES["anthropic/claude-fable-5.1"]["forced_tool_choice"] is False
    assert MODEL_CAPABILITIES["openai/gpt-6-astra"]["reasoning_mandatory"] is True


@pytest.mark.parametrize("slug,expected", [
    ("openai/gpt-6.1-sol", "o200k_base"), ("openai/gpt-6-luna", "o200k_base"),
    ("openai/gpt-6-astra", "o200k_base"), ("openai/gpt-4o", "o200k_base"),
    ("moonshotai/kimi-k3.1-pro", "cl100k_base"),
])
def test_tokenizer_family_for_new_gpt_slugs(slug, expected):
    assert _choose_tiktoken_encoding(slug) == expected


def test_best_of_lists_reference_registered_models():
    from config.app_config import BEST_OF_GENERATOR_MODELS, BEST_OF_SELECTOR_MODELS
    for name in list(BEST_OF_GENERATOR_MODELS) + list(BEST_OF_SELECTOR_MODELS):
        assert name in API_MODEL_ALIASES, name
