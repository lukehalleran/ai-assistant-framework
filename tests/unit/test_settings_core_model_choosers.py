"""
Model choosers read the registry, not a hand-typed list (2026-10-23).

The legacy Gradio chat dropdown (gui/launch.py) and the Settings duel dropdowns
(gui/tabs/settings.py) kept hardcoded fallback lists (gpt-5.1, claude-opus-4.5 ...)
that drifted from the roster, so a newly registered model was unreachable from
them whenever the live ModelManager could not be read. Both now go through
gui/settings_core (chat_model_selection / registry_model_choices).

class: BC-10 (registered-but-unreachable).
"""
import inspect
from types import SimpleNamespace

import gui.settings_core as sc
import models.model_manager as mm_module


def test_registry_choices_contain_every_registered_alias(monkeypatch):
    monkeypatch.setitem(mm_module.API_MODEL_ALIASES, "brand-new-model", "vendor/brand-new-model")
    assert "brand-new-model" in sc.registry_model_choices()
    assert "kimi-k3.1-pro" in sc.registry_model_choices()


def test_chat_selection_prefers_the_live_manager():
    orch = SimpleNamespace(model_manager=SimpleNamespace(
        api_models={"m-a": "x/a"}, models={"loc": object()}, get_active_model_name=lambda: "m-a"))
    assert sc.chat_model_selection(orch) == (["loc", "m-a"], "m-a")


def test_chat_selection_fallback_is_registry_and_persisted_active(monkeypatch):
    monkeypatch.setitem(mm_module.API_MODEL_ALIASES, "brand-new-model", "vendor/brand-new-model")
    monkeypatch.setattr(sc, "load_settings", lambda: {"models": {"active": "brand-new-model"}})
    choices, active = sc.chat_model_selection(SimpleNamespace())  # no model_manager attribute
    assert "brand-new-model" in choices and active == "brand-new-model"
    assert "gpt-5.1" not in choices  # the retired hardcoded entry


def test_chat_selection_fallback_ignores_unregistered_persisted_active(monkeypatch):
    monkeypatch.setattr(sc, "load_settings", lambda: {"models": {"active": "gone-model"}})
    choices, active = sc.chat_model_selection(SimpleNamespace())
    assert active == choices[0] and active != "gone-model"


def test_both_gradio_builders_use_the_shared_functions():
    import gui.launch as launch
    import gui.tabs.settings as settings_tab
    launch_src = inspect.getsource(launch.build_demo)
    assert "settings_core.chat_model_selection(orchestrator)" in launch_src
    assert "'gpt-5.1'" not in launch_src
    tab_src = inspect.getsource(settings_tab.build_settings_tab)
    assert "settings_core.registry_model_choices()" in tab_src
    assert "'gpt-5.1'" not in tab_src
