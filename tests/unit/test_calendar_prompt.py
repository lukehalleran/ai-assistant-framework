"""Tests for Google Calendar prompt injection."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


# --- Gatherer tests ---

class TestGetGoogleCalendarEvents:
    """Test KnowledgeRetrievalMixin.get_google_calendar_events()."""

    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self):
        from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin

        mixin = KnowledgeRetrievalMixin.__new__(KnowledgeRetrievalMixin)
        mixin.memory_coordinator = None

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", False):
            result = await mixin.get_google_calendar_events()
        assert result == []

    @pytest.mark.asyncio
    async def test_enabled_returns_events(self):
        from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin

        mixin = KnowledgeRetrievalMixin.__new__(KnowledgeRetrievalMixin)
        mixin.memory_coordinator = None

        events = [
            {"summary": "Standup", "start": "2026-05-27T10:00:00-05:00",
             "end": "2026-05-27T11:00:00-05:00", "all_day": False, "location": ""},
        ]

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_calendar.fetch_upcoming_events",
                   new_callable=AsyncMock, return_value=events):
            result = await mixin.get_google_calendar_events()

        assert len(result) == 1
        assert result[0]["summary"] == "Standup"

    @pytest.mark.asyncio
    async def test_api_error_returns_empty(self):
        from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin

        mixin = KnowledgeRetrievalMixin.__new__(KnowledgeRetrievalMixin)
        mixin.memory_coordinator = None

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_calendar.fetch_upcoming_events",
                   new_callable=AsyncMock, side_effect=Exception("timeout")):
            result = await mixin.get_google_calendar_events()

        assert result == []

    @pytest.mark.asyncio
    async def test_max_events_passed(self):
        from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin

        mixin = KnowledgeRetrievalMixin.__new__(KnowledgeRetrievalMixin)
        mixin.memory_coordinator = None

        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_calendar.fetch_upcoming_events",
                   new_callable=AsyncMock, return_value=[]) as mock_fetch:
            await mixin.get_google_calendar_events(max_events=5)

        mock_fetch.assert_called_once_with(max_events=5)


# --- Formatter tests ---

class TestGoogleCalendarFormatter:
    """Test [GOOGLE CALENDAR] section formatting in _assemble_prompt."""

    def _make_context(self, google_calendar=None, **overrides):
        ctx = {
            "recent_conversations": [],
            "memories": [],
            "user_profile": "",
            "narrative_state": "",
            "summaries": [],
            "reflections": [],
            "dreams": [],
            "semantic_chunks": [],
            "wiki": [],
            "personal_notes": [],
            "reference_docs": [],
            "user_uploads": [],
            "git_commits": [],
            "procedural_skills": [],
            "proposed_features": [],
            "graph_context": [],
            "unresolved_threads": [],
            "upcoming_schedule": [],
            "google_calendar": google_calendar or [],
            "proactive_insights": [],
            "web_search_results": None,
        }
        ctx.update(overrides)
        return ctx

    def _get_formatter(self):
        from core.prompt.formatter import PromptFormatter
        token_mgr = MagicMock()
        token_mgr.count_tokens = MagicMock(return_value=10)
        fmt = PromptFormatter(token_manager=token_mgr, time_manager=None)
        fmt._feature_inventory_cache = None
        return fmt

    def test_no_events_no_section(self):
        fmt = self._get_formatter()
        ctx = self._make_context(google_calendar=[])
        result = fmt._assemble_prompt(ctx, "hello")
        assert "[GOOGLE CALENDAR]" not in result

    def test_timed_event_formatted(self):
        fmt = self._get_formatter()
        events = [
            {"summary": "Standup", "start": "2026-05-27T10:00:00-05:00",
             "end": "2026-05-27T11:00:00-05:00", "all_day": False, "location": ""},
        ]
        ctx = self._make_context(google_calendar=events)
        result = fmt._assemble_prompt(ctx, "hello")
        assert "[GOOGLE CALENDAR] n=1" in result
        assert "Standup" in result

    def test_all_day_event_formatted(self):
        fmt = self._get_formatter()
        events = [
            {"summary": "PTO", "start": "2026-05-28", "end": "2026-05-29",
             "all_day": True, "location": ""},
        ]
        ctx = self._make_context(google_calendar=events)
        result = fmt._assemble_prompt(ctx, "hello")
        assert "[all day]" in result
        assert "PTO" in result

    def test_location_included_when_present(self):
        fmt = self._get_formatter()
        events = [
            {"summary": "Dentist", "start": "2026-05-29T14:00:00-05:00",
             "end": "2026-05-29T15:00:00-05:00", "all_day": False,
             "location": "123 Main St"},
        ]
        ctx = self._make_context(google_calendar=events)
        result = fmt._assemble_prompt(ctx, "hello")
        assert "(123 Main St)" in result

    def test_location_omitted_when_empty(self):
        fmt = self._get_formatter()
        events = [
            {"summary": "Call", "start": "2026-05-27T14:00:00-05:00",
             "end": "2026-05-27T15:00:00-05:00", "all_day": False, "location": ""},
        ]
        ctx = self._make_context(google_calendar=events)
        result = fmt._assemble_prompt(ctx, "hello")
        assert "Call" in result
        # No parenthetical location
        lines = [l for l in result.split("\n") if "Call" in l]
        assert all("()" not in l for l in lines)

    def test_multiple_events_numbered(self):
        fmt = self._get_formatter()
        events = [
            {"summary": "Meeting A", "start": "2026-05-27T09:00:00",
             "end": "2026-05-27T10:00:00", "all_day": False, "location": ""},
            {"summary": "Meeting B", "start": "2026-05-27T11:00:00",
             "end": "2026-05-27T12:00:00", "all_day": False, "location": ""},
        ]
        ctx = self._make_context(google_calendar=events)
        result = fmt._assemble_prompt(ctx, "hello")
        assert "[GOOGLE CALENDAR] n=2" in result
        assert "1)" in result
        assert "2)" in result


# --- Section registry tests ---

class TestSectionRegistry:
    """Test google_calendar and upcoming_schedule are in the section registry."""

    def test_google_calendar_registered(self):
        from eval.section_registry import SECTION_REGISTRY
        assert "google_calendar" in SECTION_REGISTRY
        entry = SECTION_REGISTRY["google_calendar"]
        assert entry.header == "[GOOGLE CALENDAR]"
        assert entry.source_field == "google_calendar"

    def test_upcoming_schedule_registered(self):
        from eval.section_registry import SECTION_REGISTRY
        assert "upcoming_schedule" in SECTION_REGISTRY

    def test_google_calendar_after_upcoming_schedule(self):
        from eval.section_registry import SECTION_REGISTRY
        sched = SECTION_REGISTRY["upcoming_schedule"]
        cal = SECTION_REGISTRY["google_calendar"]
        assert cal.assembly_order > sched.assembly_order
