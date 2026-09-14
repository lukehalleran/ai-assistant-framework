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

Strengthened 2026-09-13: every ALIAS is also checked through the public
ModelManager classifiers and the protocol chooser (the path production uses),
registry parity lives in one helper with its own red controls (a new alias
with no row, an orphan row, a missing or mistyped capability), and no test
copies a classifier's decision list.
"""
from types import SimpleNamespace

import pytest

from models.model_manager import (
    API_MODEL_ALIASES,
    MODEL_CAPABILITIES,
    ModelManager,
    _slug_supports_reasoning,
    _slug_supports_vision,
    _slug_supports_tools,
    _slug_supports_prompt_caching,
)


REGISTERED_SLUGS = sorted(set(API_MODEL_ALIASES.values()))
ALIASES = sorted(API_MODEL_ALIASES)
REQUIRED_KEYS = frozenset({"reasoning", "vision", "tools", "caching"})
OPTIONAL_KEYS = frozenset({"forced_top_p"})
CACHING_VALUES = (None, "explicit", "implicit")


def registry_problems(aliases, capabilities) -> list:
    """Every wiring problem between an alias table and a capability table."""
    problems = []
    registered = set(aliases.values())
    for alias, slug in sorted(aliases.items()):
        if slug not in capabilities:
            problems.append(f"alias {alias!r} -> {slug!r} has no MODEL_CAPABILITIES row")
    for slug in sorted(capabilities):
        if slug not in registered:
            problems.append(f"MODEL_CAPABILITIES row {slug!r} has no registered alias")
    for slug, caps in sorted(capabilities.items()):
        if not isinstance(caps, dict):
            problems.append(f"{slug}: capability row is not a mapping")
            continue
        missing = REQUIRED_KEYS - set(caps)
        unknown = set(caps) - REQUIRED_KEYS - OPTIONAL_KEYS
        if missing:
            problems.append(f"{slug}: missing capability keys {sorted(missing)}")
        if unknown:
            problems.append(f"{slug}: unknown capability keys {sorted(unknown)}")
        for key in ("reasoning", "vision", "tools"):
            if key in caps and type(caps[key]) is not bool:
                problems.append(f"{slug}: {key} must be True or False, got {caps[key]!r}")
        if caps.get("caching") not in CACHING_VALUES:
            problems.append(f"{slug}: invalid caching value {caps['caching']!r}")
        if "forced_top_p" in caps:
            value = caps["forced_top_p"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
                problems.append(f"{slug}: forced_top_p must be a number in (0, 1], got {value!r}")
    return problems


def _public_manager() -> ModelManager:
    """The deployed classifiers without a heavyweight constructor."""
    manager = ModelManager.__new__(ModelManager)
    manager.api_models = dict(API_MODEL_ALIASES)
    manager.models = {}
    manager.active_model_name = None
    return manager


def test_live_registry_has_no_wiring_problems():
    """Every alias has a row, every row has an alias, every row is well formed."""
    problems = registry_problems(API_MODEL_ALIASES, MODEL_CAPABILITIES)
    assert not problems, "Model capability wiring problems:\n  " + "\n  ".join(problems)


class TestRegistryHelperControls:
    GOOD = {"reasoning": True, "vision": False, "tools": True, "caching": None}

    def test_new_alias_without_a_row_is_reported(self):
        problems = registry_problems({"new": "vendor/new-model"}, {})
        assert problems == ["alias 'new' -> 'vendor/new-model' has no MODEL_CAPABILITIES row"]

    def test_orphan_row_is_reported(self):
        problems = registry_problems({}, {"vendor/orphan": dict(self.GOOD)})
        assert problems == ["MODEL_CAPABILITIES row 'vendor/orphan' has no registered alias"]

    @pytest.mark.parametrize(
        "row, needle",
        [
            ({"reasoning": True, "vision": False, "tools": True}, "missing capability keys"),
            ({**GOOD, "streaming": True}, "unknown capability keys"),
            ({**GOOD, "tools": "yes"}, "tools must be True or False"),
            ({**GOOD, "caching": "sometimes"}, "invalid caching value"),
            ({**GOOD, "forced_top_p": 1.5}, "forced_top_p must be a number"),
            ({**GOOD, "forced_top_p": True}, "forced_top_p must be a number"),
        ],
    )
    def test_malformed_row_is_reported(self, row, needle):
        problems = registry_problems({"m": "vendor/m"}, {"vendor/m": row})
        assert any(needle in problem for problem in problems), problems


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


@pytest.mark.parametrize("alias", ALIASES)
def test_public_classifiers_agree_for_every_alias(alias):
    """The ModelManager methods production calls, driven by alias, match the row."""
    caps = MODEL_CAPABILITIES[API_MODEL_ALIASES[alias]]
    manager = _public_manager()
    observed = {
        "reasoning": manager.supports_reasoning(alias),
        "vision": manager.supports_vision(alias),
        "tools": manager.supports_tools(alias),
        "explicit_caching": manager.supports_prompt_caching(alias),
    }
    declared = {
        "reasoning": caps["reasoning"],
        "vision": caps["vision"],
        "tools": caps["tools"],
        "explicit_caching": caps["caching"] == "explicit",
    }
    assert observed == declared, f"{alias} -> {API_MODEL_ALIASES[alias]}"


@pytest.mark.parametrize("slug", REGISTERED_SLUGS)
def test_public_tool_classifier_accepts_an_already_resolved_slug(slug):
    """The agentic path passes the resolved slug; tools must not drop for it."""
    assert _public_manager().supports_tools(slug) == MODEL_CAPABILITIES[slug]["tools"]


def test_unregistered_name_claims_no_capability_through_the_public_api():
    manager = _public_manager()
    name = "vendor/not-a-registered-model"
    assert (manager.supports_reasoning(name), manager.supports_vision(name),
            manager.supports_prompt_caching(name)) == (False, False, False)


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


@pytest.mark.parametrize("slug", MODEL_CAPABILITIES.keys())
def test_detect_protocol_agrees_with_declared_tools(slug):
    """core.agentic.protocols.detect_protocol must agree with MODEL_CAPABILITIES.

    Fix 1.5 (2026-09-06): NATIVE_TOOL_MODELS was a hand-curated substring list
    that drifted from the capability registry — moonshotai/kimi-k3 (the active
    model) declares tools: True but no substring ever matched it, so every
    agentic decision round on that model silently used the XML-marker protocol
    instead of native tool calling. This asserts the two can never drift again:
    every tools: True slug is detected as native-tools, and (by the same
    parametrization) any future tools: False slug would be caught detecting
    native-tools when it shouldn't.
    """
    from core.agentic.protocols import detect_protocol
    from core.agentic.types import SearchProtocol

    caps = MODEL_CAPABILITIES[slug]
    protocol = detect_protocol(slug)
    if caps["tools"]:
        assert protocol == SearchProtocol.NATIVE_TOOLS, (
            f"{slug} declares tools: True but detect_protocol chose {protocol}"
        )
    else:
        assert protocol != SearchProtocol.NATIVE_TOOLS, (
            f"{slug} declares tools: False but detect_protocol chose NATIVE_TOOLS"
        )


@pytest.mark.parametrize("alias", ALIASES)
def test_detect_protocol_agrees_for_every_alias(alias):
    """The alias + api_models resolution path production uses picks the declared protocol."""
    from core.agentic.protocols import detect_protocol
    from core.agentic.types import SearchProtocol

    tools = MODEL_CAPABILITIES[API_MODEL_ALIASES[alias]]["tools"]
    protocol = detect_protocol(alias, api_models=API_MODEL_ALIASES)
    assert (protocol == SearchProtocol.NATIVE_TOOLS) == tools, f"{alias}: {protocol}"


def test_kimi_k3_detected_as_native_tools():
    """Regression: the exact slug fix 1.5 fixes."""
    from core.agentic.protocols import SearchProtocol, detect_protocol

    assert detect_protocol("moonshotai/kimi-k3") == SearchProtocol.NATIVE_TOOLS
    # Also via the short alias + api_models resolution path used in prod.
    assert detect_protocol("kimi-3", api_models=API_MODEL_ALIASES) == SearchProtocol.NATIVE_TOOLS


def test_public_manager_stand_in_uses_the_real_methods():
    """The stand-in is only an instance without __init__; the methods are the class's own."""
    manager = _public_manager()
    for name in ("supports_reasoning", "supports_vision", "supports_tools", "supports_prompt_caching"):
        assert getattr(type(manager), name) is getattr(ModelManager, name)
    assert isinstance(SimpleNamespace, type)
