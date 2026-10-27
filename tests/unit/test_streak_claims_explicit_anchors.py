"""Explicit start dates, week counts and ended episodes in the day-count anchors (pure + store).

Wave 3b lane DAY, 2026-10-27; class: BC-103 (supply the computed quantity), BC-100 (dated claims),
BC-75 (user text only), BC-52 (a stale count must not keep running). Follow-up to the 2026-10-27 S1
extraction, whose handoff listed "stopped on July 6", "week 3 of X" and episode end as NOT DONE.
Invented activities only; every live-text shape is asserted clean AND wrapped/indented.
"""
from datetime import date

import pytest

import memory.day_count_anchors as dca
from utils.streak_claims import (
    DayCountClaim, day_count_lines, episode_anchors, extract_day_count_claims, stale_day_count_claims,
)

D = date(2026, 10, 27)


def wrap(t):
    return "\n    " + t.replace(" ", "  ", 1) + "\n    (pasted)"


FORMS = ["clean", "wrapped"]


def shape(text, form, on=D, **kw):
    return extract_day_count_claims(text if form == "clean" else wrap(text), on, **kw)


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("text,anchor", [
    ("I started juggling practice on July 6", date(2026, 7, 6)),
    ("I began the kayak routine on 3 October", date(2026, 10, 3)),
    ("I started the pottery class on the 3rd", date(2026, 10, 3)),
    ("Since October 1 I've been doing the pottery class", date(2026, 10, 1)),
])
def test_explicit_start_date_becomes_an_anchor(text, anchor, form):
    starts = [c for c in shape(text, form) if c.kind == "start"]
    assert len(starts) == 1 and starts[0].anchor_date == anchor
    assert starts[0].start == anchor and starts[0].count == (D - anchor).days + 1


def test_start_date_is_resolved_in_the_past_not_next_year():
    # temporal_resolver is future-biased; "July 6" said in October is THIS July.
    c = [c for c in extract_day_count_claims("I started juggling practice on July 6", D) if c.kind == "start"][0]
    assert c.anchor_date.year == 2026
    jan = extract_day_count_claims("I started juggling practice on December 20", date(2026, 1, 5))
    assert [x.anchor_date for x in jan] == [date(2025, 12, 20)]


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("text", [
    "if I started juggling practice on July 6 I would be ahead",
    "I didn't start juggling practice on July 6",
    "I plan to start juggling practice on July 6",
    "She said she started juggling practice on July 6",
    "Did I start juggling practice on July 6?",
    "I start juggling practice on July 6",
    "I started juggling practice and then on July 6 my knee hurt",
])
def test_plans_negations_questions_quotes_make_no_anchor(text, form):
    assert shape(text, form) == []


@pytest.mark.parametrize("form", FORMS)
def test_weeks_are_extracted_and_start_is_stated_minus_seven_n_minus_one(form):
    for text in ("week 3 of kayak physio", "I'm 3 weeks into kayak physio"):
        (c,) = shape(text, form)
        assert (c.unit, c.count, c.start) == ("week", 15, date(2026, 10, 13))
    assert shape("I did 3 weeks of kayak physio", form) == []     # a duration, not a position


def test_week_rendering_in_weeks_and_days():
    a = episode_anchors(extract_day_count_claims("week 3 of kayak physio", D))
    line = day_count_lines(a, date(2026, 11, 4), "how is kayak physio going")[0]
    assert "week 4 today" in line and "3 weeks 1 day in" in line and "day 1 ≈ 2026-10-13" in line


@pytest.mark.parametrize("form", FORMS)
def test_explicit_start_date_beats_a_count_for_the_same_episode(form):
    count = extract_day_count_claims("day 5 of juggling practice", date(2026, 9, 1))        # start 08-28
    explicit = shape("I started juggling practice on July 6", form, on=date(2026, 8, 1))    # older statement
    a = episode_anchors(count + explicit)[0]
    assert a.start == date(2026, 7, 6) and a.source_claim.kind == "start"
    assert a.projected(date(2026, 10, 27)) == 114
    later_count = extract_day_count_claims("day 60 of juggling practice", date(2026, 10, 1))
    assert episode_anchors(count + explicit + later_count)[0].start == date(2026, 7, 6)


@pytest.mark.parametrize("form", FORMS)
def test_stop_ends_the_episode_and_the_count_stops(form):
    count = extract_day_count_claims("day 10 of juggling practice", date(2026, 7, 10))     # start 07-01
    end = shape("I stopped juggling practice on July 20", form, on=date(2026, 7, 25))
    assert [c.kind for c in end] == ["end"]
    (a,) = episode_anchors(count + end)
    assert a.ended == date(2026, 7, 20) and a.projected(D) == 20
    (line,) = day_count_lines([a], D, "juggling practice")
    assert "ended 2026-07-20" in line and "ran about 20 days" in line and "day 2" not in line.split("—")[0]
    assert stale_day_count_claims("You're 119 days into juggling practice.", [a], D)[0].current_count == 20


def test_dateless_stop_ends_on_the_day_said_and_yesterday_shifts():
    count = extract_day_count_claims("day 10 of juggling practice", date(2026, 7, 10))
    (a,) = episode_anchors(count + extract_day_count_claims("I stopped juggling practice", date(2026, 7, 25)))
    assert a.ended == date(2026, 7, 25)
    (b,) = episode_anchors(count + extract_day_count_claims("I stopped juggling practice yesterday", date(2026, 7, 25)))
    assert b.ended == date(2026, 7, 24)
    assert extract_day_count_claims("I stopped by the shop", D) == []


def test_a_count_after_the_stop_resumes_the_episode():
    count = extract_day_count_claims("day 10 of juggling practice", date(2026, 7, 10))
    end = extract_day_count_claims("I stopped juggling practice on July 20", date(2026, 7, 25))
    again = extract_day_count_claims("day 3 of juggling practice", date(2026, 9, 3))
    (a,) = episode_anchors(count + end + again)
    assert a.ended is None
    # a count stated on the same day as the stop statement does not resume it
    same = extract_day_count_claims("day 11 of juggling practice", date(2026, 7, 25))
    assert episode_anchors(count + end + same)[0].ended == date(2026, 7, 20)


def test_end_with_nothing_to_end_and_ambiguous_unattributed_end_are_dropped():
    end_only = extract_day_count_claims("I stopped juggling practice on July 20", D)
    assert episode_anchors(end_only) == []
    unattributed = extract_day_count_claims("I stopped on July 20", D)
    assert [c.lemmas for c in unattributed] == [frozenset()]
    one = extract_day_count_claims("day 10 of juggling practice", date(2026, 7, 10))
    two = extract_day_count_claims("day 4 of pottery class", date(2026, 7, 10))
    assert episode_anchors(one + unattributed)[0].ended == date(2026, 7, 20)
    assert all(a.ended is None for a in episode_anchors(one + two + unattributed))


def test_claim_round_trip_keeps_kind_unit_and_anchor():
    for c in (extract_day_count_claims("I started juggling practice on July 6", D)
              + extract_day_count_claims("week 3 of kayak physio", D)
              + extract_day_count_claims("I stopped juggling practice on July 20", D)):
        back = DayCountClaim.from_dict(c.to_dict())
        assert (back.kind, back.unit, back.anchor_date, back.count) == (c.kind, c.unit, c.anchor_date, c.count)
    legacy = DayCountClaim.from_dict({"count": 16, "stated_on": "2026-07-22", "subject": "x", "lemmas": ["x"]})
    assert (legacy.kind, legacy.unit, legacy.anchor_date) == ("count", "day", None)
    assert DayCountClaim.from_dict({"count": 3, "stated_on": "2026-07-22", "kind": "start"}) is None


@pytest.mark.parametrize("form", FORMS)
def test_store_hook_records_explicit_anchors_and_supplies_ended_line(form):
    day = date(2026, 7, 25)
    assert dca.record_from_user_text(wrap("day 10 of juggling practice") if form == "wrapped"
                                     else "day 10 of juggling practice", date(2026, 7, 10)) == 1
    assert dca.record_from_user_text("I stopped juggling practice on July 20", day) == 1
    dca.reset_for_tests(dca.store_path())                       # strict reload from disk
    from datetime import datetime
    (line,) = dca.supply_lines("how is juggling practice", now=datetime(2026, 10, 27, 9, 0))
    assert "ended 2026-07-20" in line
    assert dca.record_from_user_text("nothing countable here", day) == 0


def test_test_marker_and_assistant_text_have_no_path_in():
    # The extractor is speaker-blind by contract: the orchestrator hook passes user_input only and skips
    # [test] turns (tests/unit/test_orchestrator_day_count_hook.py). Quoted/draft text is rejected here.
    assert extract_day_count_claims("Dear Sam, I started juggling practice on July 6", D) == []
