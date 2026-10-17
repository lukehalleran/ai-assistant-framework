"""B5 (2026-10-12): placeless local-conditions query with no known location.
Beta turn: "What's the weather this weekend?" -> Tavily got a placeless query and
ranked a weatherapi page titled "Current, Harbour Island, Bahamas". Placeholder
phrases are never places; with no location the search is skipped and the prompt
carries a deterministic, metered note (class: BC-04).

Original batch label: oct12_b5 (renamed 2026-10-16; old path in tests/RENAMES.tsv)."""
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

import utils.location_resolver as lr
from core.prompt.gatherer_web import WebSearchMixin
from core.prompt.formatter import PromptFormatter
from core.prompt.token_manager import TokenManager, _meter_web_search_results
from knowledge.web_search_manager import (
    LOCATION_UNKNOWN_NOTE,
    WebSearchDepth,
    WebSearchManager,
)


@pytest.fixture(autouse=True)
def _web_on(monkeypatch, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr("config.app_config.WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr("config.app_config.OBSIDIAN_VAULT_PATH", str(vault))


def _mgr(location):
    m = WebSearchManager.__new__(WebSearchManager)
    m._get_user_location = lambda: location
    return m


class TestPlaceholderLocation:
    @pytest.mark.parametrize("v", ["current location", "Current", "my area", "here", "near me",
                                   "My current location", "unknown", "", None])
    def test_placeholders(self, v):
        assert lr.is_placeholder_location(v) is True

    @pytest.mark.parametrize("v", ["Chicago, IL", "Harbour Island, Bahamas", "Springfield"])
    def test_places(self, v):
        assert lr.is_placeholder_location(v) is False

    def test_override_placeholder_does_not_resolve(self, monkeypatch, tmp_path):
        monkeypatch.setattr(lr, "LOCATION_ENABLED", True)
        monkeypatch.setattr(lr, "LOCATION_IP_LOOKUP_ENABLED", False)
        monkeypatch.setattr(lr, "LOCATION_OVERRIDE", "current location")
        r = lr.LocationResolver(profile_path=str(tmp_path / "none.json"))
        assert r.get_location() is None


class TestManagerChokepoint:
    def test_placeholder_phrase_never_sent_as_place(self):
        out = _mgr(None)._localize_query("weather in my current location")
        assert "current location" not in out.lower()

    def test_known_location_replaces_placeholder(self):
        out = _mgr("Chicago, IL")._localize_query("weather in current location")
        assert "Chicago, IL" in out and "current location" not in out.lower()

    def test_unknown_for_placeless_weather(self):
        m = _mgr(None)
        assert m.location_unknown_for(["What's the weather this weekend?"]) is True
        assert m.location_unknown_for(["weather in current location"]) is True

    def test_not_unknown_when_place_named_or_location_known_or_not_local(self):
        assert _mgr(None).location_unknown_for(["weather in Tokyo"]) is False
        assert _mgr("Chicago, IL").location_unknown_for(["weather this weekend"]) is False
        assert _mgr(None).location_unknown_for(["history of the roman empire"]) is False


def _host(manager):
    g = WebSearchMixin.__new__(WebSearchMixin)
    g.web_search_manager = manager
    g.web_search_trigger_llm = AsyncMock(return_value=NS(
        should_search=True, source="test", reason="r", confidence=0.9,
        depth=NS(value="quick"),
        search_terms=["weather forecast October 14-15 2026", "weekend weather report October 14-15 2026"],
        evidence_needed=False, blocked_reason=None, budget_remaining=100.0))
    g.web_search_trigger = None
    g.model_manager = object()
    g.memory_id_map = {}
    return g


class TestGathererSkipsAndNotes:
    @pytest.mark.asyncio
    async def test_unknown_location_skips_provider_and_returns_note(self):
        real = _mgr(None)
        manager = MagicMock()
        manager.is_available.return_value = True
        manager.rate_limiter = None
        manager.multi_search = AsyncMock()
        manager.location_unknown_for = real.location_unknown_for
        host = _host(manager)
        res = await host._get_web_search_results("What's the weather this weekend?")
        manager.multi_search.assert_not_called()
        assert res.note == LOCATION_UNKNOWN_NOTE and res.blocked == "location_unknown"
        assert host.last_web_decision["blocked"] == "location_unknown"

    @pytest.mark.asyncio
    async def test_known_location_still_searches(self):
        real = _mgr("Chicago, IL")
        manager = MagicMock()
        manager.is_available.return_value = True
        manager.rate_limiter = None
        manager.location_unknown_for = real.location_unknown_for
        manager.multi_search = AsyncMock(return_value=NS(
            has_results=False, pages=[], error=None, blocked=None))
        host = _host(manager)
        await host._get_web_search_results("What's the weather this weekend?")
        manager.multi_search.assert_called_once()


class TestNoteIsRenderedAndMetered:
    def _result(self):
        from knowledge.web_search_manager import WebSearchResult
        return WebSearchResult(query="q", search_depth=WebSearchDepth.QUICK,
                               error="x", blocked="location_unknown", note=LOCATION_UNKNOWN_NOTE)

    def test_formatter_renders_note(self):
        tm = MagicMock()
        tm.count_tokens = MagicMock(return_value=10)
        fmt = PromptFormatter(token_manager=tm, time_manager=None)
        fmt._feature_inventory_cache = None
        out = fmt._assemble_prompt({"web_search_results": self._result()}, "What's the weather this weekend?")
        assert "[LOCATION UNKNOWN]" in out

    def test_note_is_metered_and_survives_budget_manager(self):
        res = self._result()
        assert _meter_web_search_results(res, lambda t: len(t.split())) > 0

        class _MM:
            def get_active_model_name(self):
                return "m"

        class _TK:
            def count_tokens(self, text, model_name=None):
                return len((text or "").split())

        out = TokenManager(_MM(), _TK(), 10_000)._manage_token_budget({"web_search_results": res})
        assert out["web_search_results"].note == LOCATION_UNKNOWN_NOTE
