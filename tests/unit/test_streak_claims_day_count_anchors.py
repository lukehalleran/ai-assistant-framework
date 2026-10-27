"""Dated day counts: extraction, episode anchors, projection, stale check (pure).

Incident 2026-09-11..20 (M3 trace): a note titled "... Day 23" was read as TODAY's count and
echoed forward ("23 days into <activity>", then "24") for nine days; no input carried the start
date, and utils/streak_claims.py extracted none of "N days into X", "day N of X", "(Day N)" or
"I am on like day 16?". Class: BC-103 (supply the computed quantity), BC-100, BC-75, BC-52.
Fixtures use an invented activity; every live-text shape is asserted clean AND wrapped/indented.
"""
from datetime import date

import pytest

from utils.streak_claims import (
    day_count_lines, episode_anchors, extract_day_count_claims, stale_day_count_claims,
)

D = date(2026, 7, 22)
CLEAN = [
    ("I'm 16 days into the cold-shower habit and it is fine.", 16),
    ("day 16 of the cold-shower habit", 16),
    ("16-day cold... hmm. 16 days into cold showers", 16),
]


def wrap(t):
    return "\n    " + t.replace(" ", "  ", 1) + "\n    (pasted)"


@pytest.mark.parametrize("text,n", CLEAN)
@pytest.mark.parametrize("form", ["clean", "wrapped"])
def test_attributed_claims_are_extracted(text, n, form):
    t = text if form == "clean" else wrap(text)
    claims = extract_day_count_claims(t, D)
    attributed = [c for c in claims if c.lemmas]
    assert attributed and attributed[0].count == n
    assert attributed[0].start == date(2026, 7, 7)           # day 1 = stated_on - (N-1)
    assert {"shower"} & attributed[0].lemmas or {"habit"} & attributed[0].lemmas


@pytest.mark.parametrize("form", ["clean", "wrapped"])
def test_bare_day_is_unattributed(form):
    t = "I am on like day 16?"
    claims = extract_day_count_claims(t if form == "clean" else wrap(t), D)
    assert [(c.count, c.lemmas) for c in claims] == [(16, frozenset())]


@pytest.mark.parametrize("text", [
    "if I make it to day 23 I win",
    "Is it day 23?",
    "She said 23 days into the cold-shower habit",
    "I was sick for 3 days",
    "I did 3 days in a row of cold showers",
    "day 900 of the cold-shower habit",
])
def test_non_claims_are_rejected(text):
    assert extract_day_count_claims(text, D) == [] or all(c.count <= 400 for c in extract_day_count_claims(text, D))
    assert [c for c in extract_day_count_claims(text, D) if c.lemmas or c.count == 23] == []


def test_assistant_text_is_never_passed_by_the_hook_contract():
    # The extractor has no notion of speaker: callers pass user text only. The hook and the
    # note path are asserted in test_orchestrator_day_count_hook.py / test_day_count_anchors_supply.py.
    assert extract_day_count_claims("", D) == [] and extract_day_count_claims(None, D) == []


def test_incident_replay_message_plus_dated_note_project_to_today():
    msg = extract_day_count_claims("I'm on day 16 of the cold-shower habit", D)
    note = extract_day_count_claims("7 28 26 Daily Note (Day 23)", date(2026, 7, 28), source="note")
    anchors = episode_anchors(msg + note)
    assert len(anchors) == 1
    a = anchors[0]
    assert a.start == date(2026, 7, 6) and a.newest.source == "note" and a.spread_days == 1
    assert a.projected(date(2026, 9, 20)) == 77
    lines = day_count_lines(anchors, date(2026, 9, 20), "how is the cold shower habit going", max_lines=3)
    assert len(lines) == 1 and "day 77 today (2026-09-20)" in lines[0] and "2026-07-28" in lines[0]
    # 60 days after the anchor turn: the count is projected, not copied
    assert "day 76" in day_count_lines(anchors, date(2026, 9, 19), "cold showers", max_lines=3)[0]


def test_unattributed_with_disagreeing_start_is_dropped_not_guessed():
    msg = extract_day_count_claims("day 16 of the cold-shower habit", D)
    far = extract_day_count_claims("Daily note Day 5", date(2026, 7, 28), source="note")   # start 07-24
    assert [a.spread_days for a in episode_anchors(msg + far)] == [0]
    assert episode_anchors(far) == []                 # unattributed alone never becomes an anchor


def test_newest_statement_wins_and_message_beats_note_on_the_same_day():
    older = extract_day_count_claims("day 5 of the cold-shower habit", date(2026, 7, 10))
    newer = extract_day_count_claims("day 30 of the cold-shower habit", date(2026, 8, 4))
    a = episode_anchors(older + newer)[0]
    assert a.newest.count == 30 and a.start == date(2026, 7, 6)
    note = extract_day_count_claims("Day 31", date(2026, 8, 4), source="note")
    msg = extract_day_count_claims("day 30 of the cold-shower habit", date(2026, 8, 4))
    assert episode_anchors(note + msg)[0].newest.source == "message"


def test_relevance_gate_and_age_cap():
    a = episode_anchors(extract_day_count_claims("day 16 of the cold-shower habit", D))
    assert day_count_lines(a, date(2026, 9, 20), "what's for dinner") == []
    assert day_count_lines(a, date(2026, 9, 20), "") == []
    assert day_count_lines(a, date(2027, 9, 20), "cold showers", max_age_days=400) == []
    assert day_count_lines(a, date(2026, 9, 20), "cold showers", max_lines=0) == []


def test_stale_day_count_claims_flags_a_copied_old_count_only():
    a = episode_anchors(extract_day_count_claims("day 16 of the cold-shower habit", D))
    as_of = date(2026, 9, 20)                                  # true count: 16 + 60 = 76
    bad = stale_day_count_claims("You're 23 days into the cold-shower habit now.", a, as_of)
    assert [(s.stated_count, s.current_count) for s in bad] == [(23, 76)]
    assert stale_day_count_claims("You're 76 days into the cold-shower habit.", a, as_of) == []
    assert stale_day_count_claims("On 2026-07-22 you were 16 days into the cold-shower habit.", a, as_of) == []
    assert stale_day_count_claims("You're 23 days into the lease.", a, as_of) == []
