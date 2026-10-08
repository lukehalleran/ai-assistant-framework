"""2026-10-08 (class: BC-52, BC-04): claim-time classification of past episodes.

Live shape (05:21 shutdown, paraphrased synthetically here): a verbless
weekday-run episode ("But Saturday Sunday Monday awful migraines etc ...",
observed Thu 2026-10-08) was stored as claim_kind "unknown" and rendered as
current state; a past-tense report with no anchor ("Started getting anxious
... needed to get up early ...") came out "plan" although classify_claim_time documents a past-tense cue with no anchor
as "unknown".
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from memory.fact_source import classify_claim_time
from memory.user_profile import UserProfile
from memory.user_profile_schema import ProfileCategory

OBSERVED = datetime(2026, 10, 8, 5, 21, 0)  # a Thursday
EPISODE = ("But Saturday Sunday Monday awful migraines etc one cracker a day "
           "barely off the couch.")


class TestWeekdaySequenceEpisode:
    def test_live_episode_is_event_ending_last_named_weekday(self):
        ct = classify_claim_time(EPISODE, observed_at=OBSERVED)
        assert ct.kind == "event"
        assert ct.event_date == date(2026, 10, 5)
        assert ct.event_date_source == "relative"

    def test_event_date_is_strictly_before_observed_day(self):
        # observed Thursday, last weekday named Thursday -> a week ago, not today
        ct = classify_claim_time("Tuesday Wednesday Thursday were rough, no energy",
                                 observed_at=OBSERVED)
        assert ct.kind == "event"
        assert ct.event_date == date(2026, 10, 1)

    def test_three_letter_forms(self):
        ct = classify_claim_time("Mon Tue rough, no energy", observed_at=OBSERVED)
        assert ct.kind == "event" and ct.event_date == date(2026, 10, 6)

    def test_ambiguous_abbreviations_need_capital(self):
        # "sat in the sun" is not two weekdays
        ct = classify_claim_time("I sat in the sun all afternoon", observed_at=OBSERVED)
        assert ct.kind != "event"

    def test_single_weekday_is_not_a_sequence(self):
        ct = classify_claim_time("Monday was rough, no energy", observed_at=OBSERVED)
        assert ct.kind == "unknown"

    @pytest.mark.parametrize("span", [
        "I will go Saturday and Sunday",
        "I'll go Saturday and Sunday",
        "Going to rest Saturday and Sunday",
        "Next Saturday Sunday are free",
        "planning Saturday Sunday hikes",
    ])
    def test_future_cue_is_not_event(self, span):
        assert classify_claim_time(span, observed_at=OBSERVED).kind != "event"


class TestPastTenseWithoutAnchor:
    def test_started_getting_stressed_is_unknown_not_plan(self):
        ct = classify_claim_time(
            "Started getting anxious about the exam cuz needed to get up early for "
            "the bus didn't want to go",
            observed_at=OBSERVED)
        assert ct.kind == "unknown"
        assert ct.event_date is None

    def test_past_tense_with_anchor_still_event(self):
        ct = classify_claim_time("Started feeling worse yesterday", observed_at=OBSERVED)
        assert ct.kind == "event" and ct.event_date == date(2026, 10, 7)

    def test_pure_plan_without_past_cue_still_plan(self):
        assert classify_claim_time("Going to take it tomorrow.", observed_at=OBSERVED).kind == "plan"


class TestHabitPrecedence:
    def test_every_marker_beats_weekday_sequence(self):
        ct = classify_claim_time("I go to the gym every Monday Wednesday Friday",
                                 observed_at=OBSERVED)
        assert ct.kind == "habit"

    def test_spread_weekday_schedule_is_not_an_episode(self):
        """Referee: a spread list is a schedule; it must not be dated into the
        past (get_category would hide it)."""
        ct = classify_claim_time("I go to the gym Monday Wednesday Friday",
                                 observed_at=OBSERVED)
        assert ct.kind == "habit"

    def test_first_person_present_spread_two_days_is_habit(self):
        ct = classify_claim_time("I go to the gym Monday and Friday", observed_at=OBSERVED)
        assert ct.kind == "habit"

    def test_first_person_past_spread_list_is_not_habit(self):
        ct = classify_claim_time("I worked Monday Wednesday Friday", observed_at=OBSERVED)
        assert ct.kind != "habit"

    def test_first_person_present_consecutive_run_is_not_an_episode(self):
        ct = classify_claim_time("I work Monday Tuesday Wednesday",
                                 observed_at=OBSERVED)
        assert ct.kind != "event"

    def test_first_person_present_consecutive_run_is_neither_event_nor_habit(self):
        ct = classify_claim_time("I work Monday Tuesday Wednesday",
                                 observed_at=OBSERVED)
        assert ct.kind not in ("event", "habit")

    def test_first_person_past_consecutive_run_is_an_episode(self):
        ct = classify_claim_time("I was wrecked Saturday Sunday Monday",
                                 observed_at=OBSERVED)
        assert ct.kind == "event"
        assert str(ct.event_date) == "2026-10-05"


class TestDownstreamDeployedPath:
    """THE deployed chain: classify_claim_time -> add_fact(claim_kind,
    event_date) -> get_category."""

    def test_episode_facts_not_listed_as_current(self, tmp_path):
        profile = UserProfile(profile_path=str(tmp_path / "p.json"))
        ct = classify_claim_time(EPISODE, observed_at=OBSERVED)
        for rel, val in (("condition", "awful migraines"), ("eats", "one cracker a day")):
            assert profile.add_fact(rel, val, 0.7, EPISODE, claim_kind=ct.kind,
                                    event_date=ct.event_date.isoformat())
        listed = []
        for cat in ProfileCategory:
            listed += [f["value"] for f in profile.get_category(cat)]
        assert "awful migraines" not in listed
        assert "one cracker a day" not in listed
        # still retained as history
        hist = []
        for cat in ProfileCategory:
            hist += [f["value"] for f in profile.get_category(cat, include_historical=True)]
        assert "awful migraines" in hist
