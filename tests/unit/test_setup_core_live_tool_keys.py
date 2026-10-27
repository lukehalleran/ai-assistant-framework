"""A04 (2026-10-23): saving a Wolfram / E2B key reaches the ALREADY-CREATED
consumers (same instance before/after), or the result says restart_required.

class: BC-11, BC-58 (settings write that never reaches the live consumer).
Synthetic keys; no provider is contacted.
"""
import config.app_config as cfg
import knowledge.sandbox_manager as sandbox_mod
from core.agentic.tools import ToolExecutor
from gui import setup_core
from knowledge.sandbox_manager import SandboxManager
from knowledge.wolfram_manager import WolframManager

WOLF = "WOLF-SYNTH1"
E2B = "e2b_" + "a" * 24


class _Ctrl:
    """Stands in for the built AgenticSearchController: holds its own slots."""

    def __init__(self, wolfram=None, sandbox=None):
        self.wolfram_manager = wolfram
        self.sandbox_manager = sandbox
        self._tool_executor = ToolExecutor(model_manager=None, web_search_manager=None, formatter=None,
                                           wolfram_manager=wolfram, sandbox_manager=sandbox)


def _orch(ctrl):
    return type("O", (), {"_agentic_controller": ctrl})()


def _prep(monkeypatch):
    monkeypatch.setattr(cfg, "WOLFRAM_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "WOLFRAM_APP_ID", "", raising=False)
    monkeypatch.setattr(cfg, "SANDBOX_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "SANDBOX_API_KEY", "", raising=False)


def test_wolfram_save_refreshes_existing_manager(monkeypatch):
    _prep(monkeypatch)
    mgr = WolframManager()
    ctrl = _Ctrl(wolfram=mgr)
    assert mgr.is_available() is False
    res = setup_core.save_optional_key(_orch(ctrl), "wolfram", WOLF)
    assert res["ok"] is True and "restart_required" not in res
    assert ctrl.wolfram_manager is mgr and mgr.is_available() is True
    assert ctrl._tool_executor.wolfram_manager is mgr


def test_wolfram_slot_cached_as_none_is_filled(monkeypatch):
    _prep(monkeypatch)
    ctrl = _Ctrl()
    res = setup_core.save_optional_key(_orch(ctrl), "wolfram", WOLF)
    assert res["ok"] is True and "restart_required" not in res
    assert ctrl.wolfram_manager is not None and ctrl.wolfram_manager.is_available()
    assert ctrl._tool_executor.wolfram_manager is ctrl.wolfram_manager


def test_sandbox_save_refreshes_singleton_and_fills_slots(monkeypatch):
    _prep(monkeypatch)
    singleton = SandboxManager()
    monkeypatch.setattr(sandbox_mod, "_manager", singleton)
    ctrl = _Ctrl()  # sandbox slot cached as None (unavailable at construction)
    assert singleton.is_available() is False
    res = setup_core.save_optional_key(_orch(ctrl), "e2b", E2B)
    assert res["ok"] is True and "restart_required" not in res
    assert singleton.is_available() is True
    assert ctrl.sandbox_manager is singleton and ctrl._tool_executor.sandbox_manager is singleton


def test_tool_disabled_in_config_says_so_not_restart(monkeypatch):
    # 2026-10-26 (#101 follow-up, BC-58): a restart does not enable a tool that
    # config switches off, so the result must not promise one.
    _prep(monkeypatch)
    monkeypatch.setattr(cfg, "WOLFRAM_ENABLED", False, raising=False)
    ctrl = _Ctrl()
    res = setup_core.save_optional_key(_orch(ctrl), "wolfram", WOLF)
    assert res["ok"] is True and "restart_required" not in res
    assert res["disabled_in_config"] is True
    assert "disabled in your config" in res["notice"]
    assert "will not enable" in res["notice"]


def test_sandbox_disabled_in_config_says_so_not_restart(monkeypatch):
    _prep(monkeypatch)
    singleton = SandboxManager()
    singleton._config_enabled = False
    monkeypatch.setattr(sandbox_mod, "_manager", singleton)
    res = setup_core.save_optional_key(_orch(_Ctrl()), "e2b", E2B)
    assert res["ok"] is True and "restart_required" not in res
    assert res["disabled_in_config"] is True


def test_unrefreshable_consumer_still_reports_restart_required(monkeypatch):
    # The genuine restart case (tool enabled, consumer cannot be refreshed).
    _prep(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("synthetic")

    monkeypatch.setattr(setup_core, "_refresh_wolfram", boom)
    res = setup_core.save_optional_key(_orch(_Ctrl()), "wolfram", WOLF)
    assert res["ok"] is True and res["restart_required"] is True
    assert "disabled_in_config" not in res


def test_refresh_exception_reports_restart_required(monkeypatch):
    _prep(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("synthetic")

    monkeypatch.setattr(setup_core, "_refresh_sandbox", boom)
    res = setup_core.save_optional_key(_orch(_Ctrl()), "e2b", E2B)
    assert res["ok"] is True and res["restart_required"] is True


def test_no_controller_yet_needs_no_restart(monkeypatch):
    _prep(monkeypatch)
    res = setup_core.save_optional_key(_orch(None), "wolfram", WOLF)
    assert res["ok"] is True and "restart_required" not in res
