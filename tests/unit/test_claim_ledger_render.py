"""render_stated_figures: the user's own figures as prompt lines (memory/claim_ledger.py).

Incident 2026-10-20: Daemon restated the user's 10-17 CONDITIONAL ("if I can
get 6 done today then I am only 6 days behind") as plain fact and the user's
10-18 ESTIMATE ("12 days i think?") after a worked-out ~15 days replaced it. The
block is the structural input (an instruction alone does not hold -- BC-46):
each figure rendered with its date and modality, deterministically, no LLM.
Classes: BC-75, BC-52, BC-46, BC-55 (weekday + date).

Drives the deployed renderer; every value and wording is invented.
"""
from __future__ import annotations

import pytest

from memory import claim_ledger as cl

NOW = "2026-10-20T20:30:00"      # a Tuesday
KW = dict(max_lines=12, standing_window_days=30, deadline_window_days=14)


def lc(key, value, modality="asserted", at="2026-10-20T09:00:00", kind="duration", unit="days",
       desc="subject", quote=None, scope="standing", occ=""):
    quote = quote if quote is not None else f"{value} {unit}"
    return cl.LedgerClaim(
        id=cl.make_claim_id(key, kind, at, value, quote + modality + occ), subject_key=key,
        description=desc, kind=kind, value=value, unit=unit, modality=modality, stated_at=at,
        quote=quote, scope=scope, occurrence_date=occ)


def render(*claims, now=NOW, **over):
    ledger = cl.ClaimLedger()
    for c in claims:
        ledger.add(c)
    return cl.render_stated_figures(ledger, now, **{**KW, **over})


# --- wording --------------------------------------------------------------------------------------

def test_conditional_line_says_conditional_and_not_a_fact():
    lines = render(lc("days_behind", "6", "conditional", "2026-10-17T10:00:00", desc="days behind the class",
                      quote="if I can get 6 done today then I am only 6 days behind"))
    assert lines == ['- days behind the class: 6 days — CONDITIONAL, you said "if I can get 6 done today '
                     'then I am only 6 days behind" (Sat 10-17); not a fact']


def test_estimate_line_keeps_its_hedge():
    lines = render(lc("days_spent", "12", "estimate", "2026-10-18T09:00:00",
                      desc="days spent on the material", quote="12 days i think?"))
    assert lines == ['- days spent on the material: ~12 days — your ESTIMATE ("12 days i think?", Sun 10-18)']


def test_asserted_line_shows_value_date_and_earlier_none():
    lines = render(lc("loan_deadline", "2026-09-28", at="2026-09-27T09:00:00", kind="deadline", unit="",
                      desc="loan deadline", quote="tomorrow"))
    assert lines == ['- loan deadline: 2026-09-28 — you said "tomorrow" (Sun 09-27); earlier: none']


def test_superseded_values_are_listed_as_earlier_max_two_newest_first():
    claims = [lc("sprint", v, at=f"2026-10-{d}T09:00:00", desc="sprint length", quote=f"{v} days")
              for v, d in (("9", "14"), ("10", "15"), ("12", "16"), ("15", "20"))]
    (line,) = render(*claims)
    assert line == ('- sprint length: 15 days — you said "15 days" (Tue 10-20); '
                    'earlier: 12 days (Fri 10-16), 10 days (Thu 10-15)')


def test_status_value_reads_as_words():
    (line,) = render(lc("hw_done", "not_done", kind="status", unit="", desc="homework 6", quote="not done yet"))
    assert line.startswith("- homework 6: not done — you said")


# --- quote handling -------------------------------------------------------------------------------

def test_quote_is_cut_to_90_chars_and_double_quotes_are_neutralised():
    long = 'I said "' + "x" * 200 + '" and then 6 days'
    (line,) = render(lc("days_behind", "6", desc="days behind", quote=long))
    quoted = line.split('you said "', 1)[1].rsplit('" (', 1)[0]
    assert len(quoted) <= cl.MAX_RENDER_QUOTE_CHARS and quoted.endswith("…")
    assert '"' not in quoted


def test_quote_passes_through_strip_machinery():
    from utils import read_time_markers
    marker = read_time_markers.delivery_notice(read_time_markers.NOTICE_NOTE_PARTIAL, " x", separator="\n")
    (line,) = render(lc("days_behind", "6", desc="days behind", quote="6 days behind" + marker))
    assert read_time_markers.strip_machinery(line) == line      # nothing machinery-shaped survives
    assert "6 days behind" in line


# --- selection ------------------------------------------------------------------------------------

def test_standing_window_excludes_old_statements():
    lines = render(lc("old_total", "40", at="2026-08-01T09:00:00", desc="old total", unit="pages"),
                   lc("fresh", "6", at="2026-10-10T09:00:00", desc="fresh figure"))
    assert [l.split(":")[0] for l in lines] == ["- fresh figure"]
    # control: a wider window admits the old one
    assert len(render(lc("old_total", "40", at="2026-08-01T09:00:00", desc="old total", unit="pages"),
                      standing_window_days=90)) == 1


def test_deadline_within_the_window_is_shown_even_when_stated_long_ago():
    due_soon = lc("exam", "2026-10-25", at="2026-08-01T09:00:00", kind="deadline", unit="", desc="exam date")
    far = lc("thesis", "2027-03-01", at="2026-08-01T09:00:00", kind="deadline", unit="", desc="thesis due")
    lines = render(due_soon, far)
    assert [l.split(":")[0] for l in lines] == ["- exam date"]
    assert len(render(due_soon, far, deadline_window_days=200)) == 2


def test_occurrence_claims_only_for_today_and_yesterday():
    lines = render(
        lc("sleep_hours", "5", at="2026-10-20T08:00:00", unit="hours", desc="hours slept",
           scope="occurrence", occ="2026-10-19", quote="slept 5 hours last night"),
        lc("session_hours", "2", at="2026-10-17T20:00:00", unit="hours", desc="session length",
           scope="occurrence", occ="2026-10-17", quote="2 hours today"))
    assert len(lines) == 1 and lines[0].startswith("- hours slept: 5 hours")
    assert lines[0].endswith("applies to Mon 10-19 only")
    assert render(lc("sleep_hours", "5", at="2026-10-20T08:00:00", unit="hours", desc="hours slept",
                     scope="occurrence", occ="2026-10-19"), now="2026-10-22T09:00:00") == []


def test_occurrence_dates_of_one_subject_are_separate_lines():
    lines = render(
        lc("sleep_hours", "5", at="2026-10-19T08:00:00", unit="hours", desc="hours slept",
           scope="occurrence", occ="2026-10-19"),
        lc("sleep_hours", "7", at="2026-10-20T08:00:00", unit="hours", desc="hours slept",
           scope="occurrence", occ="2026-10-20"))
    assert len(lines) == 2 and "applies to Tue 10-20 only" in lines[0] and "applies to Mon 10-19 only" in lines[1]
    assert "earlier: none" in lines[0]     # a different date is not an earlier value of this one


def test_newest_first_and_capped():
    claims = [lc(f"k{i}", str(i + 1), at=f"2026-10-{10 + i:02d}T09:00:00", desc=f"figure {i}")
              for i in range(6)]
    lines = render(*claims, max_lines=3)
    assert [l.split(":")[0] for l in lines] == ["- figure 5", "- figure 4", "- figure 3"]
    assert render(*claims, max_lines=0) == [] and len(render(*claims, max_lines=99)) == 6


def test_a_current_asserted_value_hides_a_conditional_of_the_same_subject():
    lines = render(lc("days_behind", "6", "conditional", "2026-10-17T10:00:00", desc="days behind"),
                   lc("days_behind", "3", "asserted", "2026-10-19T10:00:00", desc="days behind"))
    assert len(lines) == 1 and ": 3 days" in lines[0] and "CONDITIONAL" not in lines[0]


def test_unusable_now_renders_nothing():
    assert render(lc("k", "1"), now="not a time") == []


def test_wording_is_deterministic():
    claims = [lc("days_behind", "6", "conditional", "2026-10-17T10:00:00", desc="days behind"),
              lc("sprint", "15", at="2026-10-20T02:47:00", desc="sprint length")]
    assert render(*claims) == render(*reversed(claims))


# --- size -----------------------------------------------------------------------------------------

def test_twelve_line_block_stays_inside_a_few_hundred_tokens():
    claims = [lc(f"subject_{i}", str(i + 2), at=f"2026-10-{8 + i:02d}T09:00:00", desc=f"a figure number {i}",
                 quote=f"my figure number {i} is {i + 2} days right now ok") for i in range(12)]
    lines = render(*claims)
    assert len(lines) == 12
    assert len("\n".join(lines)) / 4 < 600      # chars/4; the config cap (max_block_tokens) trims further


@pytest.mark.parametrize("quote", ["I am only 6 days behind", "I am only 6 days\n    behind"],
                         ids=["clean", "wrapped"])
def test_wrapped_stored_quote_renders_on_one_line(quote):
    (line,) = render(lc("days_behind", "6", desc="days behind", quote=quote))
    assert "\n" not in line and "only 6 days behind" in line
