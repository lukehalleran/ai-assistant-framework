"""
Model capability wiring parity tests.

These turn "silently disabled feature for a model" into a loud red test at commit
time. Registering a model (adding it to api_models) used to require also touching
four independent substring allowlists — supports_reasoning / supports_vision /
supports_tools / supports_prompt_caching. Forgetting one silently disabled that
feature for the model with NO error: e.g. Kimi K3 was registered but omitted from
the tool-calling allowlist, which would have silently killed the entire agentic
loop (it could only narrate, never call tools). Building the capability table also
surfaced that claude-fable-5 — a flagship Claude model — was returning vision=False
AND tools=False because none of the per-name substrings ("claude-3/opus/sonnet/
haiku") matched "fable".

MODEL_CAPABILITIES is now the single source of truth. If you add a model and one
of these fails, you forgot to declare a capability, or a substring list drifted
from the declared intent — the failure message tells you which.
"""
import pytest

from models.model_manager import (
    API_MODEL_ALIASES,
    MODEL_CAPABILITIES,
    _slug_supports_reasoning,
    _slug_supports_vision,
    _slug_supports_tools,
    _slug_supports_prompt_caching,
)


REGISTERED_SLUGS = sorted(set(API_MODEL_ALIASES.values()))


def test_every_registered_model_declares_capabilities():
    """Every slug reachable via an alias must have a MODEL_CAPABILITIES row.

    This is the guard that would have caught Kimi K3 being registered without a
    tool-calling declaration.
    """
    missing = [slug for slug in REGISTERED_SLUGS if slug not in MODEL_CAPABILITIES]
    assert not missing, (
        f"Registered models missing from MODEL_CAPABILITIES: {missing}. "
        f"Declare reasoning/vision/tools/caching for each."
    )


def test_no_orphan_capability_rows():
    """Every MODEL_CAPABILITIES row should correspond to a registered model."""
    registered = set(API_MODEL_ALIASES.values())
    orphans = [slug for slug in MODEL_CAPABILITIES if slug not in registered]
    assert not orphans, (
        f"MODEL_CAPABILITIES rows with no registered alias: {orphans}."
    )


def test_capability_rows_have_all_keys():
    required = {"reasoning", "vision", "tools", "caching"}
    for slug, caps in MODEL_CAPABILITIES.items():
        assert required <= set(caps), (
            f"{slug} missing capability keys: {required - set(caps)}"
        )
        assert caps["caching"] in (None, "explicit", "implicit"), (
            f"{slug} has invalid caching value {caps['caching']!r}"
        )


@pytest.mark.parametrize("slug", MODEL_CAPABILITIES.keys())
def test_classifiers_agree_with_declared_capabilities(slug):
    """Each pure classifier must match the declared capability for every slug.

    Catches drift between a substring allowlist and the declared intent.
    """
    caps = MODEL_CAPABILITIES[slug]
    assert _slug_supports_reasoning(slug) == caps["reasoning"], (
        f"supports_reasoning({slug}) disagrees with declared {caps['reasoning']}"
    )
    assert _slug_supports_vision(slug) == caps["vision"], (
        f"supports_vision({slug}) disagrees with declared {caps['vision']}"
    )
    assert _slug_supports_tools(slug) == caps["tools"], (
        f"supports_tools({slug}) disagrees with declared {caps['tools']}"
    )
    # caching classifier is about EXPLICIT cache_control injection only.
    assert _slug_supports_prompt_caching(slug) == (caps["caching"] == "explicit"), (
        f"supports_prompt_caching({slug}) disagrees with declared caching="
        f"{caps['caching']!r}"
    )


def test_kimi_k3_fully_wired():
    """Regression: the model this whole exercise started from."""
    slug = "moonshotai/kimi-k3"
    assert API_MODEL_ALIASES["kimi-k3"] == slug
    assert API_MODEL_ALIASES["kimi-3"] == slug
    assert _slug_supports_reasoning(slug) is True
    assert _slug_supports_vision(slug) is True
    assert _slug_supports_tools(slug) is True
    # Kimi caches server-side implicitly — we must NOT inject cache_control.
    assert _slug_supports_prompt_caching(slug) is False
    assert MODEL_CAPABILITIES[slug]["caching"] == "implicit"


def test_fable5_vision_and_tools_fixed():
    """Regression: fable-5 previously fell through the per-name substring lists.

    Verified against OpenRouter (input_modalities=[text,image,file], tools=True).
    """
    slug = "anthropic/claude-fable-5"
    assert _slug_supports_vision(slug) is True
    assert _slug_supports_tools(slug) is True
    assert _slug_supports_reasoning(slug) is True


def test_deepseek_r1_tools_enabled():
    """Regression: R1 tool-calling verified present in OpenRouter supported_parameters."""
    slug = "deepseek/deepseek-r1-0528"
    assert _slug_supports_tools(slug) is True
    assert _slug_supports_reasoning(slug) is True
    assert _slug_supports_vision(slug) is False  # R1 is text-only
