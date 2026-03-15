"""Tests for utils/temporal_resolver.py — relative date resolution in fact values."""

import re
import pytest
from datetime import datetime
from utils.temporal_resolver import (
    has_temporal_reference,
    resolve_temporal_references,
)

# Fixed reference date: Thursday, 2026-03-12
# (March 13 is Friday — confirmed from production logs)
REF = datetime(2026, 3, 12, 15, 36, 0)


# ── Detection ──────────────────────────────────────────────────────────────

class TestHasTemporalReference:
    def test_tomorrow(self):
        assert has_temporal_reference("work tomorrow")

    def test_today(self):
        assert has_temporal_reference("working today")

    def test_tonight(self):
        assert has_temporal_reference("shift tonight")

    def test_yesterday(self):
        assert has_temporal_reference("worked yesterday")

    def test_following_day(self):
        assert has_temporal_reference("work tomorrow and the following day")

    def test_the_day_after(self):
        assert has_temporal_reference("free the day after")

    def test_next_day_name(self):
        assert has_temporal_reference("starts next Monday")

    def test_next_week(self):
        assert has_temporal_reference("vacation next week")

    def test_next_month(self):
        assert has_temporal_reference("starting next month")

    def test_in_n_days(self):
        assert has_temporal_reference("due in 3 days")

    def test_n_days_from_now(self):
        assert has_temporal_reference("ready 5 days from now")

    def test_no_temporal(self):
        assert not has_temporal_reference("works at brewery")

    def test_no_temporal_static_date(self):
        assert not has_temporal_reference("born on 1993-04-15")

    def test_empty(self):
        assert not has_temporal_reference("")

    def test_none(self):
        assert not has_temporal_reference(None)


# ── Resolution ─────────────────────────────────────────────────────────────

class TestResolveTomorrow:
    def test_basic(self):
        result = resolve_temporal_references("work tomorrow", REF)
        assert "Fri 2026-03-13" in result
        assert "tomorrow" not in result.lower()

    def test_with_surrounding_text(self):
        result = resolve_temporal_references("I have work tomorrow from 3 to 10", REF)
        assert "Fri 2026-03-13" in result
        assert "from 3 to 10" in result


class TestResolveFollowingDay:
    def test_tomorrow_and_following_day(self):
        """'the following day' after 'tomorrow' should be +2 days."""
        result = resolve_temporal_references(
            "work tomorrow and the following day", REF
        )
        assert "Fri 2026-03-13" in result   # tomorrow (Thu+1 = Fri)
        assert "Sat 2026-03-14" in result   # the following day (Thu+2 = Sat)

    def test_following_day_alone(self):
        """'the following day' without 'tomorrow' = +1 day."""
        result = resolve_temporal_references("free the following day", REF)
        assert "Fri 2026-03-13" in result

    def test_the_day_after(self):
        result = resolve_temporal_references(
            "busy tomorrow and the day after", REF
        )
        assert "Fri 2026-03-13" in result   # tomorrow
        assert "Sat 2026-03-14" in result   # the day after


class TestResolveTodayTonight:
    def test_today(self):
        result = resolve_temporal_references("free today", REF)
        assert "Thu 2026-03-12" in result

    def test_tonight(self):
        result = resolve_temporal_references("shift tonight", REF)
        assert "Thu 2026-03-12" in result


class TestResolveYesterday:
    def test_basic(self):
        result = resolve_temporal_references("worked yesterday", REF)
        assert "Wed 2026-03-11" in result


class TestResolveNextDayName:
    def test_next_monday(self):
        # REF is Thursday Mar 12 → next Monday is Mar 16
        result = resolve_temporal_references("starts next Monday", REF)
        assert "Mon 2026-03-16" in result

    def test_next_friday(self):
        # REF is Thursday Mar 12 → next Friday is Mar 13
        result = resolve_temporal_references("off next Friday", REF)
        assert "Fri 2026-03-13" in result

    def test_next_wednesday(self):
        # REF is Thursday Mar 12 → next Wednesday is Mar 18
        result = resolve_temporal_references("meeting next Wednesday", REF)
        assert "Wed 2026-03-18" in result

    def test_next_thursday(self):
        # REF is Thursday Mar 12 → next Thursday is +7 = Mar 19
        result = resolve_temporal_references("event next Thursday", REF)
        assert "Thu 2026-03-19" in result

    def test_abbreviated_day(self):
        # "next thu" from Thursday → +7 = Mar 19
        result = resolve_temporal_references("event next thu", REF)
        assert "Thu 2026-03-19" in result


class TestResolveNextWeekMonth:
    def test_next_week(self):
        result = resolve_temporal_references("vacation next week", REF)
        assert "week of Thu 2026-03-19" in result

    def test_next_month(self):
        result = resolve_temporal_references("starting next month", REF)
        assert "April 2026" in result


class TestResolveInNDays:
    def test_in_3_days(self):
        result = resolve_temporal_references("due in 3 days", REF)
        assert "Sun 2026-03-15" in result

    def test_in_1_day(self):
        result = resolve_temporal_references("arriving in 1 day", REF)
        assert "Fri 2026-03-13" in result

    def test_days_from_now(self):
        result = resolve_temporal_references("ready 5 days from now", REF)
        assert "Tue 2026-03-17" in result


class TestNoChange:
    def test_no_temporal_keywords(self):
        text = "works at brewery 3 to 10"
        assert resolve_temporal_references(text, REF) == text

    def test_empty_string(self):
        assert resolve_temporal_references("", REF) == ""

    def test_none(self):
        assert resolve_temporal_references(None, REF) is None


class TestDefaultReferenceDate:
    def test_uses_now_when_no_ref(self):
        """Should not crash when reference_date is None (uses datetime.now)."""
        result = resolve_temporal_references("work tomorrow")
        assert "tomorrow" not in result.lower()
        # Can't assert exact date since it depends on when test runs,
        # but it should have a date format like "Mon 2026-..."
        assert re.search(r"\w{3} \d{4}-\d{2}-\d{2}", result)


class TestCaseInsensitivity:
    def test_tomorrow_uppercase(self):
        result = resolve_temporal_references("WORK TOMORROW", REF)
        assert "Fri 2026-03-13" in result

    def test_next_monday_mixed(self):
        result = resolve_temporal_references("Next Monday", REF)
        assert "Mon 2026-03-16" in result


class TestRealWorldExamples:
    """Test cases modeled on actual user messages."""

    def test_work_schedule_statement(self):
        """The original bug: 'I work tomorrow and the following day'."""
        result = resolve_temporal_references(
            "work tomorrow and the following day", REF
        )
        assert "Fri 2026-03-13" in result
        assert "Sat 2026-03-14" in result
        assert "tomorrow" not in result.lower()
        assert "following" not in result.lower()

    def test_gym_tomorrow(self):
        result = resolve_temporal_references("gym tomorrow morning", REF)
        assert "Fri 2026-03-13" in result
        assert "morning" in result

    def test_exam_in_3_days(self):
        result = resolve_temporal_references("exam in 3 days", REF)
        assert "Sun 2026-03-15" in result

    def test_appointment_next_tuesday(self):
        result = resolve_temporal_references("dentist next Tuesday", REF)
        assert "Tue 2026-03-17" in result
