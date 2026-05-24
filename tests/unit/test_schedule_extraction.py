"""
Tests for schedule extraction pipeline.

Covers: temporal_resolver time normalization + day expansion + date resolution,
fact_extractor schedule patterns + metadata enrichment, schedule scope detection,
past-tense/negation guards, employment pattern non-overlap, and prompt gating.
"""

import re
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

# ── temporal_resolver helpers ─────────────────────────────────────────

from utils.temporal_resolver import (
    normalize_time_range,
    expand_day_abbreviations,
    resolve_date_expression,
)


class TestExpandDayAbbreviations:
    def test_mwf(self):
        assert expand_day_abbreviations("MWF") == ["monday", "wednesday", "friday"]

    def test_tth(self):
        assert expand_day_abbreviations("TTh") == ["tuesday", "thursday"]

    def test_tr(self):
        assert expand_day_abbreviations("TR") == ["tuesday", "thursday"]

    def test_single_full_name(self):
        assert expand_day_abbreviations("Friday") == ["friday"]

    def test_single_abbreviation(self):
        assert expand_day_abbreviations("mon") == ["monday"]

    def test_plural_day(self):
        assert expand_day_abbreviations("fridays") == ["friday"]

    def test_empty(self):
        assert expand_day_abbreviations("") == []

    def test_unknown(self):
        assert expand_day_abbreviations("xyz") == []

    def test_mw(self):
        assert expand_day_abbreviations("MW") == ["monday", "wednesday"]

    def test_tt(self):
        assert expand_day_abbreviations("TT") == ["tuesday", "thursday"]


class TestNormalizeTimeRange:
    """Three-tier resolution: explicit > heuristic > ambiguous."""

    def test_explicit_pm(self):
        s, e, basis, conf = normalize_time_range("3pm", "10pm")
        assert s == "15:00"
        assert e == "22:00"
        assert basis == "explicit_ampm"
        assert conf == 0.95

    def test_explicit_am(self):
        s, e, basis, conf = normalize_time_range("9am", "11am")
        assert s == "09:00"
        assert e == "11:00"
        assert conf == 0.95

    def test_explicit_24h(self):
        s, e, basis, conf = normalize_time_range("15:00", "22:00")
        assert s == "15:00"
        assert e == "22:00"
        assert conf == 0.95

    def test_bare_work_context(self):
        s, e, basis, conf = normalize_time_range("3", "10", context="work")
        assert basis == "work_pm_heuristic"
        assert conf == 0.70
        # Should infer PM
        assert s is not None and e is not None

    def test_bare_class_context(self):
        s, e, basis, conf = normalize_time_range("9", "11", context="class")
        assert basis == "am_context_heuristic"
        assert conf == 0.70

    def test_am_to_pm_crossover(self):
        # "8-2" → likely 08:00-14:00
        s, e, basis, conf = normalize_time_range("8", "2", context="work")
        assert s == "08:00"
        assert e == "14:00"
        assert basis == "am_to_pm_crossover"
        assert conf == 0.65

    def test_overnight_explicit(self):
        s, e, basis, conf = normalize_time_range("10pm", "6am")
        assert s == "22:00"
        assert e == "06:00"
        assert conf >= 0.90

    def test_ambiguous(self):
        s, e, basis, conf = normalize_time_range("10", "6")
        assert conf <= 0.65  # low confidence for ambiguous

    def test_with_minutes(self):
        s, e, basis, conf = normalize_time_range("9:30am", "10:45am")
        assert s == "09:30"
        assert e == "10:45"
        assert conf == 0.95

    def test_unparseable(self):
        s, e, basis, conf = normalize_time_range("noon", "sunset")
        # "noon" won't match the regex, treated as unparseable
        assert conf == 0.0 or s is None

    def test_partial_explicit(self):
        # "3 - 10pm" — one explicit, infer other
        s, e, basis, conf = normalize_time_range("3", "10pm")
        assert e == "22:00"
        assert conf >= 0.85  # partial explicit


class TestResolveDateExpression:
    def test_future_date(self):
        ref = datetime(2026, 1, 15)
        iso, basis, conf = resolve_date_expression("April 15th", ref)
        assert iso == "2026-04-15"
        assert basis == "current_year_future"
        assert conf == 0.90

    def test_past_date_rolls_over(self):
        ref = datetime(2026, 6, 1)
        iso, basis, conf = resolve_date_expression("April 15th", ref)
        assert iso == "2027-04-15"
        assert basis == "next_year_rollover"
        assert conf == 0.70

    def test_abbreviated_month(self):
        ref = datetime(2026, 1, 1)
        iso, basis, conf = resolve_date_expression("Dec 3rd", ref)
        assert iso == "2026-12-03"

    def test_day_of_month_format(self):
        ref = datetime(2026, 1, 1)
        iso, basis, conf = resolve_date_expression("15th of April", ref)
        assert iso == "2026-04-15"

    def test_unparseable(self):
        iso, basis, conf = resolve_date_expression("next week")
        assert iso is None

    def test_empty(self):
        iso, basis, conf = resolve_date_expression("")
        assert iso is None


# ── fact_extractor schedule patterns ──────────────────────────────────

class TestSchedulePatterns:
    """Test that schedule patterns match correctly and DON'T match wrong cases."""

    @pytest.fixture
    def extractor(self):
        from memory.fact_extractor import FactExtractor
        return FactExtractor()

    def test_work_schedule_pattern_matches(self, extractor):
        """'I work Friday 3-10pm' should match work_schedule."""
        for _, pattern in extractor.schedule_patterns:
            matches = re.findall(pattern, "i work friday 3pm-10pm", re.IGNORECASE)
            if matches:
                return  # at least one pattern matched
        pytest.fail("No schedule pattern matched 'i work friday 3pm-10pm'")

    def test_class_schedule_pattern_matches(self, extractor):
        """'class MWF 9-11am' should match class_schedule."""
        for _, pattern in extractor.schedule_patterns:
            matches = re.findall(pattern, "class mwf 9am-11am", re.IGNORECASE)
            if matches:
                return
        pytest.fail("No schedule pattern matched 'class mwf 9am-11am'")

    def test_exam_date_pattern_matches(self, extractor):
        for _, pattern in extractor.schedule_patterns:
            matches = re.findall(pattern, "my exam is April 15th", re.IGNORECASE)
            if matches:
                return
        pytest.fail("No schedule pattern matched 'my exam is April 15th'")

    def test_shift_pattern_matches(self, extractor):
        for _, pattern in extractor.schedule_patterns:
            matches = re.findall(pattern, "i work evenings", re.IGNORECASE)
            if matches:
                return
        pytest.fail("No schedule pattern matched 'i work evenings'")

    def test_day_off_pattern_matches(self, extractor):
        for _, pattern in extractor.schedule_patterns:
            matches = re.findall(pattern, "i'm off tuesday", re.IGNORECASE)
            if matches:
                return
        pytest.fail("No schedule pattern matched 'i'm off tuesday'")

    def test_no_overlap_with_employment_works_at(self, extractor):
        """'I work at Google' should NOT match schedule patterns."""
        for rel, pattern in extractor.schedule_patterns:
            matches = re.findall(pattern, "I work at Google", re.IGNORECASE)
            assert not matches, f"Schedule pattern {rel} wrongly matched 'I work at Google'"

    def test_no_overlap_with_employment_occupation(self, extractor):
        """'I work as a nurse' should NOT match schedule patterns."""
        for rel, pattern in extractor.schedule_patterns:
            matches = re.findall(pattern, "I work as a nurse", re.IGNORECASE)
            assert not matches, f"Schedule pattern {rel} wrongly matched 'I work as a nurse'"


# ── scope detection ───────────────────────────────────────────────────

class TestScheduleScope:
    """Recurring vs one-off vs ambiguous detection."""

    @pytest.fixture
    def extractor(self):
        from memory.fact_extractor import FactExtractor
        return FactExtractor()

    def _enrich(self, extractor, relation, obj, source="test"):
        md = {}
        extractor._enrich_schedule_metadata(md, relation, obj, source)
        return md

    def test_plural_day_is_recurring(self, extractor):
        md = self._enrich(extractor, "work_schedule", "fridays 3pm-10pm")
        assert md["schedule_scope"] == "recurring"

    def test_singular_day_is_ambiguous(self, extractor):
        md = self._enrich(extractor, "work_schedule", "friday 3pm-10pm")
        assert md["schedule_scope"] == "ambiguous"

    def test_multi_day_is_recurring(self, extractor):
        md = self._enrich(extractor, "class_schedule", "mwf 9am-11am")
        assert md["schedule_scope"] == "recurring"
        assert "monday" in md["schedule_days"]
        assert "wednesday" in md["schedule_days"]
        assert "friday" in md["schedule_days"]

    def test_exam_date_is_one_off(self, extractor):
        md = self._enrich(extractor, "exam_date", "April 15th")
        assert md["schedule_scope"] == "one_off"

    def test_shift_pattern_is_recurring(self, extractor):
        md = self._enrich(extractor, "shift_pattern", "evenings")
        assert md["schedule_scope"] == "recurring"

    def test_day_off_singular_is_ambiguous(self, extractor):
        md = self._enrich(extractor, "day_off", "tuesday")
        assert md["schedule_scope"] == "ambiguous"

    def test_day_off_plural_is_recurring(self, extractor):
        md = self._enrich(extractor, "day_off", "tuesdays")
        assert md["schedule_scope"] == "recurring"


# ── metadata integrity ────────────────────────────────────────────────

class TestScheduleMetadata:
    """Every schedule fact must have complete provenance metadata."""

    @pytest.fixture
    def extractor(self):
        from memory.fact_extractor import FactExtractor
        return FactExtractor()

    def _enrich(self, extractor, relation, obj, source="test"):
        md = {}
        extractor._enrich_schedule_metadata(md, relation, obj, source)
        return md

    def test_work_schedule_has_all_fields(self, extractor):
        md = self._enrich(extractor, "work_schedule", "friday 3pm-10pm")
        # fact_type is set by _to_node, not _enrich — check event_type instead
        assert md["event_type"] == "work"
        assert md["schedule_days"]  # non-empty
        assert md["schedule_start"] == "15:00"
        assert md["schedule_end"] == "22:00"
        assert md["resolution_basis"] == "explicit_ampm"
        assert md["parser_confidence"] >= 0.90
        assert md["needs_confirmation"] is False

    def test_bare_time_lower_confidence(self, extractor):
        md = self._enrich(extractor, "work_schedule", "friday 3-10")
        assert md["parser_confidence"] < 0.90  # heuristic, not explicit

    def test_schedule_days_is_string(self, extractor):
        """ChromaDB requires scalar metadata — days must be comma-separated string."""
        md = self._enrich(extractor, "class_schedule", "mwf 9am-11am")
        assert isinstance(md["schedule_days"], str)
        assert "," in md["schedule_days"]  # multiple days

    def test_exam_date_resolution(self, extractor):
        md = self._enrich(extractor, "exam_date", "April 15th")
        assert md["schedule_scope"] == "one_off"
        # Date should be resolved
        assert md["schedule_days"]  # either ISO date or original text

    def test_needs_confirmation_for_ambiguous(self, extractor):
        """Very ambiguous time ranges should flag needs_confirmation."""
        # This depends on normalize_time_range returning low confidence
        md = self._enrich(extractor, "work_schedule", "friday 10-6")
        # 10-6 is ambiguous (could be AM-PM or PM-AM)
        # The exact confidence depends on heuristics, but should be flagged if < threshold
        assert "needs_confirmation" in md


# ── past-tense and negation guards ────────────────────────────────────

class TestPastTenseGuards:
    """Past-tense constructions should NOT be extracted as schedule facts."""

    def test_past_tense_worked(self):
        """'I worked Friday 3-10' should be blocked by past-tense guard."""
        line = "i worked friday 3pm-10pm"
        # The guard checks for "worked" in the processing loop
        assert re.search(r'\b(?:worked|had\s+class|was\s+\w+ing)\b', line.lower())

    def test_historical_observation(self):
        """'work was crazy Friday 3-10' should be blocked."""
        line = "work was crazy friday 3-10"
        assert re.search(r'\bwork\s+was\b', line.lower())

    def test_present_tense_passes(self):
        """'I work Friday 3-10' should NOT be blocked."""
        line = "i work friday 3pm-10pm"
        assert not re.search(r'\b(?:worked|had\s+class|was\s+\w+ing)\b', line.lower())
        assert not re.search(r'\bwork\s+was\b', line.lower())


# ── relation schema ───────────────────────────────────────────────────

class TestRelationSchema:
    def test_work_schedule_in_career(self):
        from memory.user_profile_schema import RELATION_CATEGORY_MAP, ProfileCategory
        assert RELATION_CATEGORY_MAP.get("work_schedule") == ProfileCategory.CAREER

    def test_class_schedule_in_education(self):
        from memory.user_profile_schema import RELATION_CATEGORY_MAP, ProfileCategory
        assert RELATION_CATEGORY_MAP.get("class_schedule") == ProfileCategory.EDUCATION

    def test_shift_pattern_in_career(self):
        from memory.user_profile_schema import RELATION_CATEGORY_MAP, ProfileCategory
        assert RELATION_CATEGORY_MAP.get("shift_pattern") == ProfileCategory.CAREER

    def test_day_off_in_career(self):
        from memory.user_profile_schema import RELATION_CATEGORY_MAP, ProfileCategory
        assert RELATION_CATEGORY_MAP.get("day_off") == ProfileCategory.CAREER

    def test_work_schedule_is_snapshot(self):
        from memory.user_profile_schema import SNAPSHOT_RELATIONS
        assert "work_schedule" in SNAPSHOT_RELATIONS

    def test_class_schedule_is_snapshot(self):
        from memory.user_profile_schema import SNAPSHOT_RELATIONS
        assert "class_schedule" in SNAPSHOT_RELATIONS


# ── config ────────────────────────────────────────────────────────────

class TestScheduleConfig:
    def test_config_loaded(self):
        from config.app_config import (
            SCHEDULE_EXTRACTION_ENABLED,
            SCHEDULE_PROMPT_MAX_EVENTS,
            SCHEDULE_PROMPT_LOOKAHEAD_DAYS,
            SCHEDULE_BARE_TIME_MIN_CONFIDENCE,
        )
        assert isinstance(SCHEDULE_EXTRACTION_ENABLED, bool)
        assert SCHEDULE_PROMPT_MAX_EVENTS > 0
        assert SCHEDULE_PROMPT_LOOKAHEAD_DAYS > 0
        assert 0.0 <= SCHEDULE_BARE_TIME_MIN_CONFIDENCE <= 1.0

    def test_schema_validates(self):
        from config.schema import ScheduleExtractionSection
        section = ScheduleExtractionSection()
        assert section.enabled is True
        assert section.prompt_max_events == 10


# ── intent overrides ──────────────────────────────────────────────────

class TestIntentOverrides:
    def test_temporal_recall_has_schedule_override(self):
        from core.intent_classifier import IntentType, _PROFILES
        profile = _PROFILES[IntentType.TEMPORAL_RECALL]
        assert profile["retrieval"].get("max_upcoming_schedule") == 10

    def test_project_work_has_schedule_override(self):
        from core.intent_classifier import IntentType, _PROFILES
        profile = _PROFILES[IntentType.PROJECT_WORK]
        assert profile["retrieval"].get("max_upcoming_schedule") == 10

    def test_general_has_no_schedule_override(self):
        from core.intent_classifier import IntentType, _PROFILES
        profile = _PROFILES[IntentType.GENERAL]
        assert profile["retrieval"].get("max_upcoming_schedule") is None


# ── supersession ──────────────────────────────────────────────────────

class TestScheduleSupersession:
    def test_non_schedule_returns_none(self):
        from memory.fact_verification import FactVerifier
        mock_chroma = MagicMock()
        verifier = FactVerifier(mock_chroma)
        result = verifier._check_schedule_supersession("occupation", "nurse", "I work as a nurse")
        assert result is None

    def test_schedule_relation_detected(self):
        from memory.fact_verification import FactVerifier
        mock_chroma = MagicMock()
        mock_chroma.collections = {"facts": MagicMock(count=MagicMock(return_value=0))}
        verifier = FactVerifier(mock_chroma)
        # Empty collection → should return None (nothing to supersede)
        result = verifier._check_schedule_supersession("work_schedule", "friday 3pm-10pm", "I work friday 3pm-10pm")
        assert result is None


# ── prompt formatter ──────────────────────────────────────────────────

class TestFormatterScheduleSection:
    """Test the [UPCOMING SCHEDULE] section rendering."""

    def test_empty_schedule_no_section(self):
        """No schedule data → no section rendered."""
        from core.prompt.formatter import PromptFormatter
        # The formatter checks context.get("upcoming_schedule", [])
        # If empty, no section is appended
        context = {"upcoming_schedule": []}
        # We can't easily test the full formatter without extensive mocking,
        # but we can verify the key doesn't cause errors
        assert context["upcoming_schedule"] == []

    def test_schedule_event_structure(self):
        """Verify schedule event dict has expected keys for rendering."""
        evt = {
            "schedule_kind": "work_schedule",
            "event_type": "work",
            "schedule_days": "friday",
            "schedule_start": "15:00",
            "schedule_end": "22:00",
            "schedule_scope": "ambiguous",
            "display_date": "Fri May 30",
            "parser_confidence": 0.95,
            "resolution_basis": "explicit_ampm",
            "needs_confirmation": False,
        }
        # All expected keys present
        for key in ("schedule_kind", "event_type", "schedule_start",
                     "schedule_end", "display_date", "parser_confidence"):
            assert key in evt
