"""Behavioural siblings: the Gradio Settings tab and the API apply THE SAME settings (source-text rows).

Incident dates: 2026-07-14 (settings apply logic extracted from the Gradio tab's
closures into gui/settings_core so both UIs share it), 2026-10-23 (model choosers
read the registry, not a hand-typed list). Class: BC-63 (a grep for
``settings_core.apply_*`` stands in for "both UIs change the same live value"),
BC-10 (registered-but-unreachable model).

For each of the eight settings the same logical input is sent through (a) the real
API route and (b) the real Gradio handler (the closure registered on the tab's
button, found in the built Blocks), each on its own fresh orchestrator with an
in-memory settings store. Persisted dict, runtime orchestrator state and the
``app_config`` constants must end identical. A route or handler that stopped
calling ``settings_core`` (or called it with different arguments) diverges here.
"""
import copy
from types import SimpleNamespace

import gradio as gr
import pytest

import config.app_config as app_cfg
import gui.settings_core as sc
import models.model_manager as mm_module
from gui.tabs.settings import build_settings_tab
from tests.unit.api_launch_auth_client import authed_client as _client
from tests.unit.api_launch_auth_client import make_test_app as create_app
from tests.unit.helpers_orchestrator import _make_orchestrator

# (api path, api body, gradio handler name, gradio positional args)
CASES = {
    "streaming": ("/api/settings/streaming",
                  {"disable_best_of": True, "disable_query_rewrite": True, "best_of_latency_budget_s": 5.0},
                  "_apply_fast", (True, True, 5.0)),
    "web_search": ("/api/settings/web-search",
                   {"enabled": False, "daily_credit_limit": 40},
                   "_apply_web_search", (False, 40)),
    "duel": ("/api/settings/duel",
             {"enabled": True, "model_1": "m-a", "model_2": "m-b"},
             "_apply_duel_settings", (True, "m-a", "m-b")),
    "tokens": ("/api/settings/tokens",
               {"best_of_max_tokens": 900, "judge_max_tokens": 300, "streaming_max_tokens": 2500},
               "_apply_tokens", (900, 300, 2500)),
    "temperature": ("/api/settings/temperature", {"temperature": 0.4},
                    "_apply_temperature", (0.4,)),
    "summary_cadence": ("/api/settings/summary-cadence", {"every_n": 25},
                        "_apply_summary_n", (25,)),
    "synthesis": ("/api/settings/synthesis", {"enabled": False, "candidates_per_session": 4},
                  "_apply_synthesis", (False, 4)),
    "proposals": ("/api/settings/proposals", {"enabled": False, "max_per_session": 2},
                  "_apply_proposals", (False, 2)),
}

APP_CFG_KEYS = (
    "SYNTHESIS_POOLED_ENABLED", "SYNTHESIS_POOLED_CANDIDATES_PER_SESSION",
    "SYNTHESIS_GENERATOR_ENABLED", "CODE_PROPOSALS_ENABLED", "CODE_PROPOSALS_MAX_PER_SESSION",
    "WEB_SEARCH_ENABLED", "WEB_SEARCH_DAILY_CREDIT_LIMIT",
)


def _orchestrator():
    orch = _make_orchestrator()
    orch.config = {"features": {}, "web_search": {}, "models": {}}
    orch.model_manager.api_models = {"m-a": {}, "m-b": {}}
    orch.model_manager.models = {}
    orch.model_manager.default_max_tokens = 2048
    orch.model_manager.default_temperature = 0.7
    orch.memory_system.consolidator = SimpleNamespace(consolidation_threshold=20)
    orch.prompt_builder.consolidator = SimpleNamespace(consolidation_threshold=20)
    return orch


class _Store:
    """In-memory settings store; ``save`` mirrors settings_core.save_settings."""
    def __init__(self):
        self.data = {}

    def load(self):
        return copy.deepcopy(self.data)

    def save(self, updater):
        try:
            updater(self.data)
            return True, None
        except Exception as e:  # pragma: no cover - a failing updater is the finding
            return False, str(e)


def _state(orch, store):
    mm = orch.model_manager
    return {
        "persisted": copy.deepcopy(store.data),
        "config": copy.deepcopy(orch.config),
        "mm": (mm.default_max_tokens, mm.default_temperature),
        "cadence": (orch.memory_system.consolidator.consolidation_threshold,
                    orch.prompt_builder.consolidator.consolidation_threshold),
        "app_cfg": {k: getattr(app_cfg, k, None) for k in APP_CFG_KEYS},
    }


def _gradio_handlers(orch, store):
    with gr.Blocks() as demo:
        with gr.Tabs():
            build_settings_tab(orch, store.load, store.save)
    return {b.fn.__name__: b.fn for b in demo.fns.values() if getattr(b, "fn", None)}


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # The setters assign live app_config constants (web search, synthesis, proposals...). Snapshot
    # EVERY upper-case constant so none leaks into later tests (a leaked WEB_SEARCH_ENABLED=False
    # silently disabled search for every web test that ran after this file).
    for k in [n for n in dir(app_cfg) if n.isupper()]:
        monkeypatch.setattr(app_cfg, k, getattr(app_cfg, k))
    monkeypatch.setenv("SUMMARY_EVERY_N", "20")
    return monkeypatch


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(CASES))
async def test_api_route_and_gradio_handler_leave_identical_state(isolated, name):
    path, body, handler_name, args = CASES[name]
    baseline = {k: getattr(app_cfg, k, None) for k in APP_CFG_KEYS}

    # --- API route (settings_core.save_settings patched to an in-memory store)
    api_store = _Store()
    isolated.setattr(sc, "save_settings", api_store.save)
    api_orch = _orchestrator()
    app = create_app(api_orch, start_background=False)
    async with _client(app) as client:
        resp = await client.put(path, json=body)
    assert resp.status_code == 200 and resp.json()["ok"] is True, resp.text
    api_state = _state(api_orch, api_store)
    assert api_state["persisted"], "API route persisted nothing"

    # --- Gradio handler on a fresh orchestrator + store, from the SAME pre-API baseline (restoring
    # the API run's values instead would let a Gradio handler that sets nothing still match)
    for k, v in baseline.items():
        isolated.setattr(app_cfg, k, v, raising=False)
    gr_store = _Store()
    gr_orch = _orchestrator()
    handlers = _gradio_handlers(gr_orch, gr_store)
    assert handler_name in handlers, sorted(handlers)
    message = handlers[handler_name](*args)
    assert isinstance(message, str) and message
    assert _state(gr_orch, gr_store) == api_state


def test_every_setting_has_a_gradio_handler_and_an_api_route():
    """No setting exists in only one UI: each CASES row resolved above, and the tab registers no
    apply handler that the API lacks (handlers carry the _apply prefix)."""
    handlers = _gradio_handlers(_orchestrator(), _Store())
    gradio_apply = {n for n in handlers if n.startswith("_apply")}
    assert gradio_apply == {c[2] for c in CASES.values()}


# ------------------------------------------------ model choosers read the registry
def _duel_dropdown_choices(orch):
    with gr.Blocks() as demo:
        with gr.Tabs():
            build_settings_tab(orch, lambda: {}, lambda updater: (True, None))
    dropdowns = [b for b in demo.blocks.values()
                 if isinstance(b, gr.Dropdown) and b.label in ("Model 1", "Model 2")]
    assert len(dropdowns) == 2
    return [[c[1] if isinstance(c, (tuple, list)) else c for c in d.choices] for d in dropdowns]


def test_settings_tab_duel_choosers_offer_a_newly_registered_model(monkeypatch):
    monkeypatch.setitem(mm_module.API_MODEL_ALIASES, "brand-new-model", "vendor/brand-new-model")
    orch = SimpleNamespace(model_manager=SimpleNamespace(
        api_models={}, models={}, default_max_tokens=2048, default_temperature=0.7))
    for choices in _duel_dropdown_choices(orch):  # live manager lists nothing -> registry
        assert "brand-new-model" in choices
        assert "gpt-5.1" not in choices  # the retired hardcoded entry


def _chat_model_dropdown(orch):
    import gui.launch as launch
    demo = launch.build_demo(orch, dev_tabs=False)
    found = [b for b in demo.blocks.values() if isinstance(b, gr.Dropdown) and b.label == "Model"]
    assert len(found) == 1
    return [c[1] if isinstance(c, (tuple, list)) else c for c in found[0].choices], found[0].value


def test_legacy_chat_dropdown_reads_the_live_manager_first():
    from unittest.mock import MagicMock
    orch = MagicMock()
    orch.model_manager.api_models = {"m-a": {}}
    orch.model_manager.models = {}
    orch.model_manager.get_active_model_name.return_value = "m-a"
    assert _chat_model_dropdown(orch) == (["m-a"], "m-a")


def test_legacy_chat_dropdown_falls_back_to_the_registry_not_a_hardcoded_list(monkeypatch):
    from unittest.mock import MagicMock
    monkeypatch.setitem(mm_module.API_MODEL_ALIASES, "brand-new-model", "vendor/brand-new-model")
    monkeypatch.setattr(sc, "load_settings", lambda: {"models": {"active": "brand-new-model"}})
    orch = MagicMock()
    orch.model_manager.api_models = {}
    orch.model_manager.models = {}
    orch.model_manager.default_max_tokens = 2048
    orch.model_manager.default_temperature = 0.7
    orch.model_manager.get_active_model_name.side_effect = AttributeError  # live manager unreachable
    choices, value = _chat_model_dropdown(orch)
    assert "brand-new-model" in choices and value == "brand-new-model"
    assert "gpt-5.1" not in choices
