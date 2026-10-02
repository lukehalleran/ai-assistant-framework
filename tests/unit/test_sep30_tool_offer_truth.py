"""2026-09-30 (BC-46, BC-47, BC-58, BC-72): a tool is OFFERED only when it can
work right now, and an unavailable backend is "not configured", never "no
results". Drives the deployed handler / registry / ToolExecutor functions."""
import asyncio
from types import SimpleNamespace

import pytest

from core.agentic.protocols import NativeToolsHandler
from core.agentic.tools import ToolExecutor
from core.agentic.types import SearchDecision
from core.agentic.formatters import AgenticFormatter
from core.actions import registry
from core.actions.types import ActionType


def _names(tools):
    return [t["function"]["name"] for t in tools]


class _WS:
    def __init__(self, available):
        self._a = available

    def is_available(self):
        return self._a

    def is_enabled(self):
        return True

    async def search(self, **kw):  # pragma: no cover - must not be reached
        raise AssertionError("search dispatched while unavailable")


def _executor(ws):
    return ToolExecutor(model_manager=None, web_search_manager=ws, formatter=AgenticFormatter())


# ---- D1a
def test_web_search_not_offered_when_unavailable():
    assert "web_search" not in _names(NativeToolsHandler(web_search_available=False).get_tools())
    assert "web_search" in _names(NativeToolsHandler(web_search_available=True).get_tools())
    # default keeps legacy constructors unchanged
    assert "web_search" in _names(NativeToolsHandler().get_tools())


def test_web_dispatch_while_unavailable_says_not_configured():
    ex = _executor(_WS(False))
    dec = SearchDecision(wants_search=True, search_query="weather")
    tr = asyncio.run(ex._dispatch_web_search(dec, 1, None))
    assert "Web search unavailable — not configured" in tr.formatted_context
    assert "No results found" not in tr.formatted_context


def test_real_empty_search_still_no_results():
    ex = _executor(_WS(True))

    async def _empty(*a, **k):
        return SimpleNamespace(pages=[])

    ex._execute_search = _empty
    dec = SearchDecision(wants_search=True, search_query="x")
    tr = asyncio.run(ex._dispatch_web_search(dec, 1, None))
    assert "No results found" in tr.formatted_context


# ---- D1b
def test_fetch_url_offered_without_key_when_direct_fetch_exists():
    # handler level: fetch_url_available flag drives the offer
    assert "fetch_url" in _names(NativeToolsHandler(fetch_url_available=True,
                                                    web_search_available=False).get_tools())


# ---- D1c
def _cfg(monkeypatch, **vals):
    import config.app_config as cfg
    base = dict(
        INTERNET_ACTIONS_ENABLED=True, INTERNET_ACTIONS_GITHUB_WRITE_ENABLED=False,
        GOOGLE_CALENDAR_ENABLED=False, INTERNET_ACTIONS_SMTP_HOST="", INTERNET_ACTIONS_SMTP_USER="",
        INTERNET_ACTIONS_TELEGRAM_BOT_TOKEN="", INTERNET_ACTIONS_TELEGRAM_CHAT_ID="",
        INTERNET_ACTIONS_DISCORD_WEBHOOK_URL="",
    )
    base.update(vals)
    for k, v in base.items():
        monkeypatch.setattr(cfg, k, v, raising=False)
    monkeypatch.setattr("core.actions.google_auth.get_google_auth", lambda: None)


def test_nothing_configured_offers_no_propose_action(monkeypatch):
    _cfg(monkeypatch)
    assert registry.configured_action_types() == []
    h = NativeToolsHandler(actions_available=bool(registry.configured_action_types()),
                           configured_action_types=[])
    assert "propose_action" not in _names(h.get_tools())
    assert "NOT OFFERED" in registry.get_runtime_action_health()


def test_telegram_only_narrows_enum_and_health(monkeypatch):
    _cfg(monkeypatch, INTERNET_ACTIONS_TELEGRAM_BOT_TOKEN="t", INTERNET_ACTIONS_TELEGRAM_CHAT_ID="1")
    types = registry.configured_action_types()
    assert types == [ActionType.SEND_TELEGRAM.value]
    h = NativeToolsHandler(actions_available=True, configured_action_types=types)
    tool = [t for t in h.get_tools() if t["function"]["name"] == "propose_action"][0]
    assert tool["function"]["parameters"]["properties"]["action_type"]["enum"] == ["send_telegram"]
    from core.agentic.types import PROPOSE_ACTION_TOOL_DEFINITION as C
    assert len(C["function"]["parameters"]["properties"]["action_type"]["enum"]) == 8  # constant untouched
    health = registry.get_runtime_action_health()
    assert "AVAILABLE (send_telegram" in health and "not configured:" in health


# ---- D1d
def test_lookup_contact_unauthenticated(monkeypatch):
    import config.app_config as cfg
    monkeypatch.setattr(cfg, "GOOGLE_CONTACTS_ENABLED", True, raising=False)
    monkeypatch.setattr("core.actions.google_auth.get_google_auth", lambda: None)
    out = asyncio.run(_executor(_WS(True))._execute_lookup_contact("Maren"))
    assert out == "[Contact lookup unavailable — Google account not connected]"


# ---- D1e
def test_health_text_truth(monkeypatch):
    import config.app_config as cfg
    monkeypatch.setattr(cfg, "VISUAL_MEMORY_ENABLED", True, raising=False)
    monkeypatch.setattr("knowledge.semantic_search.is_faiss_available", lambda: False)
    monkeypatch.setattr(registry, "get_runtime_action_health", lambda: "propose_action: DISABLED")
    health = _executor(_WS(True)).get_tool_health()
    assert "drive may be disconnected" not in health
    assert "wiki index not installed" in health
    assert "recall_image: READY" not in health and "recall_image: NOT OFFERED" in health
