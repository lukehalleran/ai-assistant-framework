"""decision_model config (plan 2026-10-08 B1): committed defaults are all OFF, the schema
refuses unreachable "active" states and endpoint keys, and the pure resolver (BC-63: a
dict argument, never patched constants) agrees."""
import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import config.app_config as ac
from config.schema import DaemonConfig, DecisionModelSection

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))["decision_model"]
ROLES = ("tone_arbiter", "heavy_topic")
PREREQ = {"tone_arbiter": {"tone_policy": "argmax"}, "heavy_topic": {"heavy_topic_threshold": 0.4}}


def cfg(roles=None, **over):
    c = copy.deepcopy(COMMITTED)
    c["roles"] = roles if not isinstance(roles, dict) else {**c["roles"], **roles}
    c.update(over)
    return c


def test_committed_section_validates_pinned_and_entirely_off():
    DecisionModelSection(**COMMITTED)
    DaemonConfig(decision_model=COMMITTED)
    assert COMMITTED["enabled"] is False
    assert COMMITTED["roles"] == {"tone_arbiter": "off", "heavy_topic": "off"}
    assert COMMITTED["tone_policy"] == "unset" and COMMITTED["heavy_topic_threshold"] is None
    assert [ac.resolve_decision_mode(COMMITTED, r) for r in ROLES] == ["off", "off"]
    assert (COMMITTED["model"], COMMITTED["provider"], COMMITTED["timeout_s"]) == ("typesafe/jev-1.13", "TypeSafe", 1.5)
    assert COMMITTED["served_models"] == ["typesafe/jev-1.13-20260917"]
    assert COMMITTED["max_state_chars"] == {"tone_arbiter": 80000, "heavy_topic": 80000}
    assert isinstance(ac.DECISION_MODEL_SERVED_MODELS, tuple) and ac.DECISION_MODEL_SERVED_MODELS


@pytest.mark.parametrize("role", ROLES)
def test_master_switch_off_forces_off(role):
    c = cfg({role: "active"}, tone_policy="argmax", heavy_topic_threshold=0.5)
    assert c["enabled"] is False and ac.resolve_decision_mode(c, role) == "off"


@pytest.mark.parametrize("role", ROLES)
def test_active_requires_prerequisite(role, caplog):
    bare = cfg({role: "active"}, enabled=True)
    with pytest.raises(ValidationError):
        DecisionModelSection(**bare)
    with caplog.at_level("WARNING"):
        assert ac.resolve_decision_mode(bare, role) == "off"
    assert "resolving to off" in caplog.text
    ok = cfg({role: "active"}, enabled=True, **PREREQ[role])
    DecisionModelSection(**ok)
    assert ac.resolve_decision_mode(ok, role) == "active"
    assert ac.resolve_decision_mode(cfg({role: "shadow"}, enabled=True), role) == "shadow"


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_threshold_out_of_range(bad):
    c = cfg({"heavy_topic": "active"}, enabled=True, heavy_topic_threshold=bad)
    with pytest.raises(ValidationError):
        DecisionModelSection(**c)
    assert ac.resolve_decision_mode(c, "heavy_topic") == "off"


@pytest.mark.parametrize("value", ["maybe", True, None, 3, ""])
def test_unknown_mode_values_resolve_off(value):
    assert ac.resolve_decision_mode(cfg({"tone_arbiter": value}, enabled=True), "tone_arbiter") == "off"
    assert ac.resolve_decision_mode(cfg(enabled=True), "web_search") == "off"
    assert ac.resolve_decision_mode(cfg("shadow", enabled=True), "tone_arbiter") == "off"


def test_bare_yaml_off_is_accepted_as_off():
    raw = yaml.safe_load("enabled: true\nroles:\n  tone_arbiter: off\n  heavy_topic: off\n")
    assert raw["roles"]["tone_arbiter"] is False  # the YAML 1.1 footgun this guards
    assert DecisionModelSection(**raw).roles.tone_arbiter == "off"
    assert ac.resolve_decision_mode(raw, "tone_arbiter") == "off"
    with pytest.raises(ValidationError):
        DecisionModelSection(roles={"tone_arbiter": True})


@pytest.mark.parametrize("key", ["endpoint", "base_url", "url", "api_key"])
def test_endpoint_style_keys_are_rejected(key):
    with pytest.raises(ValidationError):
        DecisionModelSection(**cfg(**{key: "https://example.com/x"}))
    with pytest.raises(ValidationError):
        DecisionModelSection(**cfg({key: "off"}))


@pytest.mark.parametrize("field,bad", [("timeout_s", 0), ("timeout_s", 6), ("served_models", []), ("provider", ""),
                                       ("max_state_chars", {"tone_arbiter": 10}), ("tone_policy", "median")])
def test_field_bounds(field, bad):
    with pytest.raises(ValidationError):
        DecisionModelSection(**cfg(**{field: bad}))


def test_section_stays_out_of_settings_ui():
    for rel in ("gui/settings_core.py", "api/routes/settings.py"):
        assert "decision_model" not in (ROOT / rel).read_text(encoding="utf-8"), rel


# B3-G6-2: tone_policy_params validated per policy (mirrored at runtime by tone_detector._policy_bounds_ok)
@pytest.mark.parametrize("policy,params", [
    ("unset", {}), ("argmax", {}),
    ("weighted", {"cuts": [0.5, 1.5, 2.5]}), ("weighted", {"cuts": [0, 1, 3]}), ("weighted", {"cuts": [0.1, 0.2, 0.3]}),
    ("cumulative", {"taus": [0.5, 0.5, 0.5]}), ("cumulative", {"taus": [0.01, 0.9, 1]}),
])
def test_tone_policy_params_accepted(policy, params):
    DecisionModelSection(**cfg({}, tone_policy=policy, tone_policy_params=params))


@pytest.mark.parametrize("policy,params", [
    ("unset", {"cuts": [0.5, 1.5, 2.5]}), ("argmax", {"taus": [0.5, 0.5, 0.5]}), ("argmax", {"x": 1}),
    ("weighted", {}), ("weighted", {"taus": [0.5, 0.5, 0.5]}), ("weighted", {"cuts": [0.5, 1.5, 2.5], "extra": 1}),
    ("weighted", {"cuts": [0.5, 1.5]}), ("weighted", {"cuts": [0.5, 1.5, 2.5, 2.9]}), ("weighted", "cuts"),
    ("weighted", {"cuts": [1.5, 1.5, 2.5]}), ("weighted", {"cuts": [2.5, 1.5, 0.5]}),
    ("weighted", {"cuts": [-0.1, 1.5, 2.5]}), ("weighted", {"cuts": [0.5, 1.5, 3.1]}),
    ("weighted", {"cuts": [10 ** 30, 10 ** 31, 10 ** 32]}), ("weighted", {"cuts": [10 ** 400, 1.5, 2.5]}),
    ("weighted", {"cuts": [float("nan"), 1.5, 2.5]}), ("weighted", {"cuts": [0.5, float("inf"), 2.5]}),
    ("weighted", {"cuts": [True, 1.5, 2.5]}), ("weighted", {"cuts": ["0.5", 1.5, 2.5]}),
    ("cumulative", {}), ("cumulative", {"cuts": [0.5, 1.5, 2.5]}), ("cumulative", {"taus": [0, 0.5, 0.5]}),
    ("cumulative", {"taus": [0.5, 1.1, 0.5]}), ("cumulative", {"taus": [0.5, 0.5]}),
    ("cumulative", {"taus": [float("nan"), 0.5, 0.5]}), ("cumulative", {"taus": [None, 0.5, 0.5]}),
])
def test_tone_policy_params_rejected(policy, params):
    with pytest.raises(ValidationError):
        DecisionModelSection(**cfg({}, tone_policy=policy, tone_policy_params=params))
