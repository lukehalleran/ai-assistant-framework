"""
Per-route request-shape constraints (2026-09-21).

Two provider refusals, both observed live against OpenRouter and both HTTP 400
(a refusal, not a silently ignored parameter):

  * `reasoning: {"enabled": false}` on a route where reasoning is mandatory
    ("Reasoning is mandatory for this endpoint and cannot be disabled") —
    anthropic/claude-fable-5 (registered since 2026-06, so every
    disable_reasoning call on it had been failing), anthropic/claude-fable-5.1
    and openai/gpt-6-astra.
  * a named tool_choice on anthropic/claude-fable-5.1 ("tool_choice: type
    "tool" and "any" are not supported for this model"), which is what the
    agentic forced-action round sends.

The constraints are declared per row in MODEL_CAPABILITIES and applied by
reasoning_request_config() / resolve_tool_choice(). These tests drive THE
deployed generate paths with a fake transport that captures the request kwargs.
"""
from types import SimpleNamespace

import pytest

from models.model_manager import (
    API_MODEL_ALIASES,
    MODEL_CAPABILITIES,
    MODEL_CONTEXT_LIMITS,
    ModelManager,
    _slug_supports_tools,
    _slug_supports_vision,
    reasoning_request_config,
    resolve_tool_choice,
)

MANDATORY_ALIASES = ("claude-fable-5", "claude-fable-5.1", "fable-5.1", "gpt-6-astra", "gpt-6")
SWITCHABLE_ALIASES = ("deepseek-v4.1-flash", "deepseek-v4", "kimi-3", "claude-opus-4.8")
FORCED = {"type": "function", "function": {"name": "propose_action"}}
TOOLS = [{"type": "function", "function": {"name": "propose_action", "parameters": {"type": "object"}}}]


def _fake_response():
    msg = SimpleNamespace(content="ok", tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


class _FakeAsyncClient:
    def __init__(self, captured):
        async def _create(**kwargs):
            captured.append(kwargs)
            if kwargs.get("stream"):
                async def _stream():
                    return
                    yield  # pragma: no cover
                return _stream()
            return _fake_response()

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(ModelManager, "_get_cached_embedder", staticmethod(lambda: None))
    mm = ModelManager(api_key="test-key")
    mm.captured = []
    mm.async_client = _FakeAsyncClient(mm.captured)
    return mm


def _reasoning_sent(kwargs):
    return (kwargs.get("extra_body") or {}).get("reasoning")


# ---------------------------------------------------------------------------
# The three requested models are registered and truthfully described
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alias, slug", [
    ("gpt-6-astra", "openai/gpt-6-astra"),
    ("gpt-6", "openai/gpt-6-astra"),
    ("claude-fable-5.1", "anthropic/claude-fable-5.1"),
    ("fable-5.1", "anthropic/claude-fable-5.1"),
    ("deepseek-v4.1-flash", "deepseek/deepseek-v4.1-flash"),
])
def test_new_aliases_resolve_and_are_fully_capable(alias, slug, manager):
    assert API_MODEL_ALIASES[alias] == slug
    assert manager.is_api_model(alias)
    assert (manager.supports_reasoning(alias), manager.supports_vision(alias),
            manager.supports_tools(alias)) == (True, True, True)


def test_context_limits_for_the_new_routes(manager):
    assert MODEL_CONTEXT_LIMITS["openai/gpt-6-astra"] == 1_050_000
    assert MODEL_CONTEXT_LIMITS["deepseek/deepseek-v4.1-flash"] == 1_048_576
    manager.switch_model("fable-5.1")
    assert manager.get_context_limit() == 200_000


def test_explicit_cache_markers_only_where_the_route_needs_them(manager):
    assert manager.supports_prompt_caching("fable-5.1") is True
    # Astra and DeepSeek cache server-side; a marker is never injected.
    assert manager.supports_prompt_caching("gpt-6-astra") is False
    assert manager.supports_prompt_caching("deepseek-v4.1-flash") is False


# ---------------------------------------------------------------------------
# A registered row outranks the family heuristics
# ---------------------------------------------------------------------------

def test_row_beats_the_text_only_family_rule():
    # The heuristic says "deepseek => no vision"; V4.1 Flash has image input.
    assert _slug_supports_vision("deepseek/deepseek-v4.1-flash") is True
    assert _slug_supports_vision("deepseek/deepseek-v4-flash") is False


def test_row_answers_for_a_generation_the_heuristics_never_listed():
    # "gpt-4"/"gpt-5" substrings do not match gpt-6; the row does.
    assert _slug_supports_tools("openai/gpt-6-astra") is True
    assert _slug_supports_vision("openai/gpt-6-astra") is True


def test_a_declared_false_is_not_overridden_by_a_matching_heuristic(monkeypatch):
    import models.model_manager as mm_module
    slug = "openai/gpt-5-synthetic-no-tools"
    monkeypatch.setitem(mm_module.MODEL_CAPABILITIES, slug,
                        {"reasoning": False, "vision": False, "tools": False, "caching": None})
    assert _slug_supports_tools(slug) is False
    assert _slug_supports_vision(slug) is False


# ---------------------------------------------------------------------------
# reasoning_request_config
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("slug", sorted(MODEL_CAPABILITIES))
def test_mandatory_routes_never_receive_the_off_switch(slug):
    config = reasoning_request_config(slug, disable_reasoning=True)
    if MODEL_CAPABILITIES[slug].get("reasoning_mandatory"):
        assert "enabled" not in config
        assert config == {"effort": "low", "exclude": True}
    else:
        assert config == {"enabled": False}
    assert reasoning_request_config(slug) == {"effort": "medium"}


def test_mandatory_flag_is_declared_for_the_probed_routes():
    for slug in ("anthropic/claude-fable-5", "anthropic/claude-fable-5.1", "openai/gpt-6-astra"):
        assert MODEL_CAPABILITIES[slug].get("reasoning_mandatory") is True, slug
    # Low effort + exclude burned a 16-token budget on this route; it must keep
    # the real off-switch.
    assert not MODEL_CAPABILITIES["deepseek/deepseek-v4.1-flash"].get("reasoning_mandatory")


# ---------------------------------------------------------------------------
# Deployed request paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("alias", MANDATORY_ALIASES)
async def test_generate_once_disable_reasoning_on_a_mandatory_route(manager, alias):
    await manager.generate_once("hi", model_name=alias, disable_reasoning=True, max_tokens=16)
    assert _reasoning_sent(manager.captured[0]) == {"effort": "low", "exclude": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", SWITCHABLE_ALIASES)
async def test_generate_once_disable_reasoning_on_a_switchable_route(manager, alias):
    await manager.generate_once("hi", model_name=alias, disable_reasoning=True)
    assert _reasoning_sent(manager.captured[0]) == {"enabled": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", MANDATORY_ALIASES + SWITCHABLE_ALIASES)
async def test_generate_once_default_requests_medium_effort(manager, alias):
    await manager.generate_once("hi", model_name=alias)
    assert _reasoning_sent(manager.captured[0]) == {"effort": "medium"}


@pytest.mark.asyncio
@pytest.mark.parametrize("alias, expected", [
    ("fable-5.1", {"effort": "low", "exclude": True}),
    ("gpt-6-astra", {"effort": "low", "exclude": True}),
    ("deepseek-v4.1-flash", {"enabled": False}),
])
async def test_streaming_path_disable_reasoning(manager, alias, expected):
    manager.switch_model(alias)
    await manager.generate_async("hi", disable_reasoning=True)
    assert manager.captured[0]["stream"] is True
    assert _reasoning_sent(manager.captured[0]) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("alias, expected", [
    ("claude-fable-5", {"effort": "low", "exclude": True}),
    ("fable-5.1", {"effort": "low", "exclude": True}),
    ("gpt-6-astra", {"effort": "low", "exclude": True}),
    ("deepseek-v4.1-flash", {"enabled": False}),
])
async def test_tools_path_disable_reasoning(manager, alias, expected):
    await manager.generate_once_with_tools(
        "save a note", model_name=alias, tools=TOOLS, disable_reasoning=True)
    assert _reasoning_sent(manager.captured[0]) == expected


@pytest.mark.asyncio
async def test_forced_tool_choice_is_sent_as_auto_where_the_route_refuses_it(manager):
    await manager.generate_once_with_tools(
        "save a note", model_name="fable-5.1", tools=TOOLS, tool_choice=FORCED)
    sent = manager.captured[0]
    assert sent["tool_choice"] == "auto"
    assert sent["tools"] == TOOLS  # the narrowed tool list still reaches the model


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ("gpt-6-astra", "deepseek-v4.1-flash", "claude-fable-5", "kimi-3"))
async def test_forced_tool_choice_passes_through_elsewhere(manager, alias):
    await manager.generate_once_with_tools(
        "save a note", model_name=alias, tools=TOOLS, tool_choice=FORCED)
    assert manager.captured[0]["tool_choice"] == FORCED


def test_resolve_tool_choice_leaves_unforced_selectors_alone():
    slug = "anthropic/claude-fable-5.1"
    assert MODEL_CAPABILITIES[slug]["forced_tool_choice"] is False
    for selector in ("auto", "none", None):
        assert resolve_tool_choice(slug, selector) == selector
    assert resolve_tool_choice(slug, "required") == "auto"
    assert resolve_tool_choice("vendor/unregistered", FORCED) == FORCED
