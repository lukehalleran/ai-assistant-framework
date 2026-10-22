"""Claim-ledger reply check (memory/claim_ledger.py).

Incident 2026-10-20 20:26: Daemon replied "after being six days behind" (the
user's 10-17 CONDITIONAL) and "roughly twelve days" (the 10-18 ESTIMATE the user
superseded on 10-20 with ~fifteen). Classes: BC-75, BC-52.

Fake model JSON only; replies tested clean and wrapped/indented.
"""
from __future__ import annotations

import json

import pytest

from memory import claim_ledger as cl


def lc(key, value, modality, at, kind="duration", unit="days", desc="subject"):
    return cl.LedgerClaim(
        id=cl.make_claim_id(key, kind, at, value, modality), subject_key=key, description=desc, kind=kind,
        value=value, unit=unit, modality=modality, stated_at=at, quote=f"{value} {unit}")


def finding(key, value, quote, modality="asserted", kind="duration"):
    return {"subject_key": key, "kind": kind, "value": value, "unit": "days", "modality": modality,
            "quote": quote}


def raw(*items):
    return json.dumps({"findings": list(items)})


@pytest.fixture()
def conditional_ledger():
    ledger = cl.ClaimLedger()
    ledger.add(lc("days_behind", "6", "conditional", "2026-10-17T10:00:00", desc="days behind schedule"))
    return ledger


@pytest.fixture()
def sprint_ledger():
    ledger = cl.ClaimLedger()
    ledger.add(lc("sprint_span", "12", "estimate", "2026-10-18T09:00:00", desc="length of the sprint"))
    ledger.add(lc("sprint_span", "15", "asserted", "2026-10-20T02:47:00", desc="length of the sprint"))
    return ledger


REPLY_COND = ["Good luck with the push, especially after being six days behind last week.",
              "Good luck with the push,\n    especially after being six days\n    behind last week."]
REPLY_STALE = ["That puts the whole sprint at roughly twelve days of work.",
               "That puts the whole sprint\n    at roughly twelve days\n    of work."]
REPLY_OK = ["That puts the whole sprint at about fifteen days of work.",
            "That puts the whole sprint\n   at about fifteen days of work."]


@pytest.mark.parametrize("reply", REPLY_COND, ids=["clean", "wrapped"])
def test_conditional_restated_as_fact(conditional_ledger, reply):
    findings, drops = cl.validate_reply_findings(
        raw(finding("days_behind", "6", "after being six days behind")), reply, "2026-10-20T20:26:00",
        conditional_ledger)
    assert drops == []
    assert [(f.classification, f.claim_modality) for f in findings] == [("conditional_as_fact", "conditional")]


@pytest.mark.parametrize("reply", REPLY_STALE, ids=["clean", "wrapped"])
def test_superseded_estimate_is_stale(sprint_ledger, reply):
    findings, drops = cl.validate_reply_findings(
        raw(finding("sprint_span", "12", "roughly twelve days of work", modality="estimate")), reply,
        "2026-10-20T20:26:00", sprint_ledger)
    assert drops == []
    f = findings[0]
    assert (f.classification, f.claim_value, f.matched_superseded) == ("stale", "15", True)


@pytest.mark.parametrize("reply", REPLY_OK, ids=["clean", "wrapped"])
def test_current_value_matches(sprint_ledger, reply):
    findings, _ = cl.validate_reply_findings(
        raw(finding("sprint_span", "15", "about fifteen days of work", modality="estimate")), reply,
        "2026-10-20T20:26:00", sprint_ledger)
    assert [f.classification for f in findings] == ["matches_current"]


def test_check_is_as_of_the_reply_time(sprint_ledger):
    # On 10-19 the estimate (12) was still the user's latest word: restating it is correct.
    findings, _ = cl.validate_reply_findings(
        raw(finding("sprint_span", "12", "roughly twelve days of work", modality="estimate")),
        REPLY_STALE[0], "2026-10-19T12:00:00", sprint_ledger)
    assert [f.classification for f in findings] == ["matches_current"]
    # ... but stating that estimate as plain fact is estimate_as_fact
    findings, _ = cl.validate_reply_findings(
        raw(finding("sprint_span", "12", "twelve days of work")), "That is twelve days of work.",
        "2026-10-19T12:00:00", sprint_ledger)
    assert [f.classification for f in findings] == ["estimate_as_fact"]


def test_hedge_cross_check_downgrades_the_reply_modality_never_upgrades(sprint_ledger):
    findings, _ = cl.validate_reply_findings(
        raw(finding("sprint_span", "15", "about fifteen days of work", modality="asserted")), REPLY_OK[0],
        "2026-10-20T20:26:00", sprint_ledger)
    assert findings[0].reply_modality == "estimate"
    findings, _ = cl.validate_reply_findings(
        raw(finding("sprint_span", "15", "fifteen days of work", modality="estimate")),
        "It is fifteen days of work.", "2026-10-20T20:26:00", sprint_ledger)
    assert findings[0].reply_modality == "estimate"


def test_a_conditional_reply_of_a_conditional_claim_is_not_a_finding(conditional_ledger):
    reply = "If you get six days of reading done you would be only six days behind."
    findings, drops = cl.validate_reply_findings(
        raw(finding("days_behind", "6", "only six days behind", modality="asserted")), reply,
        "2026-10-20T20:26:00", conditional_ledger)
    assert findings == [] and [d["reason"] for d in drops] == ["conditional_stated_as_conditional"]


def test_unknown_subject_quote_and_value_are_dropped_not_repaired(sprint_ledger):
    reply = "That puts the whole sprint at about fifteen days of work."
    findings, drops = cl.validate_reply_findings(
        raw(finding("no_such", "15", "about fifteen days of work"),
            finding("sprint_span", "15", "about sixteen days of work"),
            finding("sprint_span", "9", "about fifteen days of work"),
            finding("sprint_span", "15", "about fifteen days of work", modality="estimate"),
            finding("sprint_span", "15", "about fifteen days of work", kind="count")),
        reply, "2026-10-20T20:26:00", sprint_ledger)
    assert [f.classification for f in findings] == ["matches_current"]
    assert sorted(d["reason"] for d in drops) == [
        "kind_mismatch", "quote_not_in_reply", "unknown_subject", "value_not_in_quote"]


def test_future_claims_are_invisible_to_an_earlier_reply(sprint_ledger):
    findings, drops = cl.validate_reply_findings(
        raw(finding("sprint_span", "12", "roughly twelve days of work", modality="estimate")),
        REPLY_STALE[0], "2026-10-17T00:00:00", sprint_ledger)
    assert findings == [] and drops[0]["reason"] == "unknown_subject"


def test_reply_prompt_lists_the_ledger_subjects(sprint_ledger):
    subjects = sprint_ledger.open_subjects(as_of="2026-10-20T20:26:00")
    system, user = cl.build_reply_check_prompt(REPLY_OK[0], "2026-10-20T20:26:00", subjects)
    assert '{"findings"' in system
    assert '"key":"sprint_span"' in user and '"what":"length of the sprint"' in user
    assert "REPLY DATE: 2026-10-20" in user


# --- AMEND1 A4/A5 + reply-side A1 (2026-10-20 trial) --------------------------------------------------------


def wrap(text: str) -> str:
    words = text.split(" ")
    return "\n    ".join(" ".join(words[i:i + 3]) for i in range(0, len(words), 3))


def both(text):
    return [text, wrap(text)]


def cnt(key, value, unit, modality, at, kind="count", desc="subject"):
    return cl.LedgerClaim(
        id=cl.make_claim_id(key, kind, at, value, modality), subject_key=key, description=desc, kind=kind,
        value=value, unit=unit, modality=modality, stated_at=at, quote=f"{value} {unit}",
        unit_norm=cl.norm_unit(unit))


def f(key, value, quote, modality="asserted", kind="count", unit="assignments"):
    return {"subject_key": key, "kind": kind, "value": value, "unit": unit, "modality": modality, "quote": quote}


@pytest.fixture()
def assignments_ledger():
    ledger = cl.ClaimLedger()
    ledger.add(cnt("assignments_finished", "5", "assignments", "asserted", "2026-10-19T09:00:00"))
    return ledger


@pytest.mark.parametrize("reply", both("Answer all five criteria from the question first."))
def test_a4_a_figure_in_another_unit_is_not_the_claims_figure(assignments_ledger, reply):
    findings, drops = cl.validate_reply_findings(
        raw(f("assignments_finished", "5", "all five criteria")), reply, "2026-10-20T10:00:00",
        assignments_ledger)
    assert findings == [] and [d["reason"] for d in drops] == ["unit_mismatch"]


@pytest.mark.parametrize("reply", both("That means all five assignments are finished by now."))
def test_a4_the_claims_own_unit_after_the_number_is_a_match(assignments_ledger, reply):
    findings, drops = cl.validate_reply_findings(
        raw(f("assignments_finished", "5", "all five assignments are finished")), reply, "2026-10-20T10:00:00",
        assignments_ledger)
    assert drops == [] and [x.classification for x in findings] == ["matches_current"]


def test_a4_the_unit_must_follow_the_number(assignments_ledger):
    reply = "Assignments matter, and you have five of them to track."
    findings, drops = cl.validate_reply_findings(
        raw(f("assignments_finished", "5", "Assignments matter, and you have five of them")), reply,
        "2026-10-20T10:00:00", assignments_ledger)
    assert findings == [] and drops[0]["reason"] == "unit_mismatch"


def test_a4_same_key_two_units_picks_the_group_the_reply_names():
    ledger = cl.ClaimLedger()
    ledger.add(cnt("days_behind", "6", "days", "asserted", "2026-10-17T10:00:00", kind="duration"))
    ledger.add(cnt("days_behind", "5", "assignments", "asserted", "2026-10-17T11:00:00", kind="duration"))
    findings, _ = cl.validate_reply_findings(
        raw(f("days_behind", "5", "five assignments behind", kind="duration")),
        "You said five assignments behind.", "2026-10-20T10:00:00", ledger)
    assert [(x.classification, x.claim_value) for x in findings] == [("matches_current", "5")]
    findings, _ = cl.validate_reply_findings(
        raw(f("days_behind", "5", "five days behind", kind="duration", unit="days")),
        "You said five days behind.", "2026-10-20T10:00:00", ledger)
    assert [(x.classification, x.claim_value) for x in findings] == [("stale", "6")]


@pytest.mark.parametrize("reply", both("HW 4 is actually done."))
def test_a1_reply_a_label_is_not_a_count(reply):
    ledger = cl.ClaimLedger()
    ledger.add(cnt("assignments_finished", "4", "assignments", "asserted", "2026-10-19T09:00:00"))
    findings, drops = cl.validate_reply_findings(
        raw(f("assignments_finished", "4", "HW 4 is actually done")), reply, "2026-10-20T10:00:00", ledger)
    assert findings == [] and [d["reason"] for d in drops] == ["identifier_number"]
    findings, _ = cl.validate_reply_findings(
        raw(f("assignments_finished", "4", "4 assignments are actually done")),
        "4 assignments are actually done.", "2026-10-20T10:00:00", ledger)
    assert [x.classification for x in findings] == ["matches_current"]


@pytest.fixture()
def planned_ledger():
    ledger = cl.ClaimLedger()
    ledger.add(cnt("problems_planned_today", "20", "problems", "conditional", "2026-10-19T09:00:00"))
    return ledger


@pytest.mark.parametrize("reply", both("Twenty problems today is a huge lift for one evening."))
def test_a5_plan_discussion_is_never_a_flag_but_asserted_would_be(planned_ledger, reply):
    quote = "Twenty problems today is a huge lift"
    findings, drops = cl.validate_reply_findings(
        raw(f("problems_planned_today", "20", quote, modality="plan_discussion", unit="problems")), reply,
        "2026-10-20T10:00:00", planned_ledger)
    assert findings == [] and [d["reason"] for d in drops] == ["plan_discussion"]
    # the same finding reported as plain fact IS the trial's false positive
    findings, _ = cl.validate_reply_findings(
        raw(f("problems_planned_today", "20", quote, modality="asserted", unit="problems")), reply,
        "2026-10-20T10:00:00", planned_ledger)
    assert [x.classification for x in findings] == ["conditional_as_fact"]


@pytest.fixture()
def behind_ledger():
    ledger = cl.ClaimLedger()
    ledger.add(cnt("days_behind_class", "6", "days", "conditional", "2026-10-17T10:00:00", kind="duration"))
    return ledger


@pytest.mark.parametrize("reply", both("You were six days behind and now you're caught up with the reading."))
def test_a5_the_0209_shape_conditional_only_subject_restated_as_realized(behind_ledger, reply):
    findings, drops = cl.validate_reply_findings(
        raw(f("days_behind_class", "6", "You were six days behind", kind="duration", unit="days")), reply,
        "2026-10-20T02:09:00", behind_ledger)
    assert drops == []
    assert [(x.classification, x.claim_modality) for x in findings] == [("conditional_as_fact", "conditional")]


@pytest.mark.parametrize("reply", [
    "At this rate you will be six days behind by Friday.",
    "At this rate you're going to be six days behind by Friday.",
])
def test_a5_a_future_or_modal_clause_is_not_a_realized_state(behind_ledger, reply):
    findings, drops = cl.validate_reply_findings(
        raw(f("days_behind_class", "6", "six days behind", kind="duration", unit="days")), reply,
        "2026-10-20T02:09:00", behind_ledger)
    assert findings == [] and [d["reason"] for d in drops] == ["not_realized"]


@pytest.mark.parametrize("reply,realized", [
    ("You were six days behind last week.", True),
    ("You have been six days behind all week.", True),
    ("You are six days behind right now.", True),
    ("You will be six days behind.", False),
    ("If it slips you are six days behind.", False),
    ("Six days behind would be a lot.", False),
    ("You have six days behind in mind.", False),
])
def test_a5_reply_states_realized_structure(reply, realized):
    off = reply.lower().index("six")
    assert cl.reply_states_realized(reply, off) is realized


def test_a5_reply_prompt_describes_plan_discussion_and_the_unit_rule():
    system, _ = cl.build_reply_check_prompt("x", "2026-10-20T10:00:00", [])
    assert "plan_discussion" in system and "huge lift" in system and "unit" in system


# --- AMEND2 B1/B2 (2026-10-20 trial 2) ------------------------------------------------------------------------


def test_b1_reply_prompt_carries_no_current_values():
    ledger = cl.ClaimLedger()
    ledger.add(lc("sprint_span", "15", "asserted", "2026-10-20T02:47:00", desc="length of the sprint"))
    subjects = ledger.open_subjects(as_of="2026-10-20T20:26:00")
    _, user = cl.build_reply_check_prompt(REPLY_OK[0], "2026-10-20T20:26:00", subjects)
    catalogue = user.split("<<<")[0]
    assert "15" not in catalogue.split("CATALOGUE", 1)[1] and "=" not in catalogue.split("CATALOGUE", 1)[1]
    # an echoed catalogue line as the quote is dropped, not admitted
    findings, drops = cl.validate_reply_findings(
        raw(finding("sprint_span", "15", "length of the sprint = 15 days")), REPLY_OK[0],
        "2026-10-20T20:26:00", ledger)
    assert findings == [] and [d["reason"] for d in drops] == ["quote_not_in_reply"]


@pytest.fixture()
def estimate_only_ledger():
    ledger = cl.ClaimLedger()
    ledger.add(lc("sprint_span", "12", "estimate", "2026-10-18T09:00:00", desc="length of the sprint"))
    return ledger


@pytest.mark.parametrize("reply", both("That puts the whole sprint at fifteen days of work."))
def test_b2_a_value_that_differs_from_a_current_estimate_is_its_own_class(estimate_only_ledger, reply):
    findings, drops = cl.validate_reply_findings(
        raw(finding("sprint_span", "15", "fifteen days of work")), reply, "2026-10-20T20:26:00",
        estimate_only_ledger)
    assert drops == []
    assert [(x.classification, x.claim_modality) for x in findings] == [("differs_from_estimate", "estimate")]


def test_b2_same_difference_against_an_asserted_current_is_stale():
    ledger = cl.ClaimLedger()
    ledger.add(lc("sprint_span", "12", "asserted", "2026-10-18T09:00:00", desc="length of the sprint"))
    findings, _ = cl.validate_reply_findings(
        raw(finding("sprint_span", "15", "fifteen days of work")), "That is fifteen days of work.",
        "2026-10-20T20:26:00", ledger)
    assert [x.classification for x in findings] == ["stale"]


def test_b2_a_superseded_value_is_stale_even_when_current_is_an_estimate():
    ledger = cl.ClaimLedger()
    ledger.add(lc("sprint_span", "9", "asserted", "2026-10-10T09:00:00", desc="length of the sprint"))
    ledger.add(lc("sprint_span", "12", "estimate", "2026-10-18T09:00:00", desc="length of the sprint"))
    findings, _ = cl.validate_reply_findings(
        raw(finding("sprint_span", "9", "nine days of work")), "That is nine days of work.",
        "2026-10-20T20:26:00", ledger)
    assert [(x.classification, x.matched_superseded) for x in findings] == [("stale", True)]
