"""Claim-ledger scope: standing vs occurrence claims (memory/claim_ledger.py).

Incident 2026-10-20: the paid baseline flagged 15 "stale" replies and found 0
real drift -- last night's sleep (x5) and one day's session length were treated
as ONE current value, so each new night "superseded" the last and the reply
check compared a reply with the wrong night. A figure bound to one date
(occurrence) must supersede and be compared only within its own date.
Classes: BC-75 (self-authored artifact laundered into evidence), BC-52 (stale
claim resurfaces).

Drives the deployed validator / ClaimLedger; every value and wording is invented.
Texts are asserted in BOTH clean and wrapped/indented form.
"""
from __future__ import annotations

import json

import pytest

from memory import claim_ledger as cl
from utils import safe_json

SLEEP_CLEAN = "Rough night. I slept 5 hours last night and I feel it."
SLEEP_WRAPPED = "Rough night. I slept 5 hours\n    last night\n    and I feel it."
SAID = "2026-10-20T09:00:00"


def payload(*claims):
    return json.dumps({"claims": list(claims)})


def raw(kind="duration", key="sleep_hours", value="5", unit="hours", modality="asserted",
        quote="I slept 5 hours last night", description="hours slept", **extra):
    item = {"kind": kind, "subject_key": key, "new_subject": True, "description": description,
            "value": value, "unit": unit, "modality": modality, "quote": quote}
    item.update(extra)
    return item


def lc(key="sleep_hours", value="5", at="2026-10-20T09:00:00", scope="standing", occ="",
       modality="asserted", kind="duration", unit="hours", desc="hours slept"):
    quote = f"{value} {unit}"
    return cl.LedgerClaim(
        id=cl.make_claim_id(key, kind, at, value, quote + modality + occ), subject_key=key,
        description=desc, kind=kind, value=value, unit=unit, modality=modality, stated_at=at,
        quote=quote, scope=scope, occurrence_date=occ)


# --- validator: scope + occurrence_date ---------------------------------------------------------

@pytest.mark.parametrize("text", [SLEEP_CLEAN, SLEEP_WRAPPED], ids=["clean", "wrapped"])
def test_occurrence_date_resolves_a_relative_phrase_against_said_on(text):
    claims, drops = cl.validate_claims(
        payload(raw(scope="occurrence", occurrence_date="2030-01-01")), text, SAID)
    assert drops == []
    # last night, said on 10-20 -> 10-19; the model's own (wrong) date is not trusted
    assert [(c.scope, c.occurrence_date) for c in claims] == [("occurrence", "2026-10-19")]


@pytest.mark.parametrize("text", ["I got 4 problems done.", "I got 4\n    problems done."],
                         ids=["clean", "wrapped"])
def test_occurrence_without_a_date_phrase_is_the_day_it_was_said(text):
    claims, drops = cl.validate_claims(
        payload(raw(kind="count", key="problems_done", value="4", unit="problems",
                    quote="I got 4 problems done", scope="occurrence")), text, SAID)
    assert drops == []
    assert [(c.scope, c.occurrence_date) for c in claims] == [("occurrence", "2026-10-20")]


@pytest.mark.parametrize("text", ["I slept 5 hours on October 17 and it was fine.",
                                  "I slept 5 hours on\n    October 17 and it was fine."],
                         ids=["clean", "wrapped"])
def test_occurrence_month_day_phrase_is_a_past_date_not_next_year(text):
    claims, _ = cl.validate_claims(
        payload(raw(quote="I slept 5 hours on October 17", scope="occurrence")), text, SAID)
    assert [c.occurrence_date for c in claims] == ["2026-10-17"]


def test_occurrence_naming_two_dates_is_dropped_not_guessed():
    text = "I slept 5 hours yesterday and today I slept 7 hours."
    claims, drops = cl.validate_claims(
        payload(raw(quote="I slept 5 hours yesterday and today", scope="occurrence")), text, SAID)
    assert claims == [] and [d["reason"] for d in drops] == ["occurrence_date_ambiguous"]


def test_standing_claim_carries_no_occurrence_date_even_if_the_model_sent_one():
    claims, _ = cl.validate_claims(
        payload(raw(kind="duration", key="days_behind", value="6", unit="days",
                    quote="I am 6 days behind", scope="standing", occurrence_date="2026-10-19")),
        "I am 6 days behind on the reading.", SAID)
    assert [(c.scope, c.occurrence_date) for c in claims] == [("standing", "")]


def test_bad_scope_drops_only_that_claim():
    text = "I am 6 days behind on the reading. I slept 5 hours last night."
    claims, drops = cl.validate_claims(
        payload(raw(key="days_behind", value="6", unit="days", quote="I am 6 days behind", scope="forever"),
                raw(scope="occurrence")), text, SAID)
    assert [c.subject_key for c in claims] == ["sleep_hours"]
    assert [d["reason"] for d in drops] == ["bad_scope"]


def test_missing_scope_is_a_standing_claim():
    claims, drops = cl.validate_claims(
        payload(raw(key="days_behind", value="6", unit="days", quote="I am 6 days behind")),
        "I am 6 days behind on the reading.", SAID)
    assert drops == [] and [(c.scope, c.occurrence_date) for c in claims] == [("standing", "")]


def test_prompt_asks_for_scope_and_occurrence_date():
    system, _ = cl.build_extraction_prompt("x", SAID, [])
    assert '"scope":"standing|occurrence"' in system and '"occurrence_date"' in system
    assert "ONE date" in system and "holds until it changes" in system


# --- ledger: supersession within an occurrence date only -----------------------------------------

def test_occurrence_supersedes_only_within_the_same_date():
    ledger = cl.ClaimLedger()
    mon = lc(value="5", at="2026-10-19T09:00:00", scope="occurrence", occ="2026-10-18")
    tue = lc(value="7", at="2026-10-20T09:00:00", scope="occurrence", occ="2026-10-19")
    tue_fix = lc(value="6", at="2026-10-20T11:00:00", scope="occurrence", occ="2026-10-19")
    ledger.add(mon)
    assert ledger.add(tue) == []                      # a different night is not a restatement
    assert ledger.add(tue_fix) == [tue.id]            # the same night, restated: newest wins
    assert mon.superseded_by is None and tue.superseded_by == tue_fix.id


def test_standing_and_occurrence_claims_of_one_key_never_touch_each_other():
    ledger = cl.ClaimLedger()
    standing = lc(value="8", at="2026-10-18T09:00:00")
    occ = lc(value="5", at="2026-10-20T09:00:00", scope="occurrence", occ="2026-10-19")
    ledger.add(standing)
    assert ledger.add(occ) == []
    assert standing.superseded_by is None and occ.superseded_by is None
    assert ledger.current("sleep_hours", scope="standing").value == "8"
    assert ledger.current("sleep_hours", scope="occurrence").value == "5"


def test_scope_alone_keeps_an_undated_occurrence_apart_from_a_standing_claim():
    # defense in depth: even a hand-built occurrence claim with NO date never supersedes a standing one
    ledger = cl.ClaimLedger()
    standing = lc(value="8", at="2026-10-18T09:00:00")
    undated = lc(value="5", at="2026-10-20T09:00:00", scope="occurrence", occ="")
    ledger.add(standing)
    assert ledger.add(undated) == [] and standing.superseded_by is None
    # ... and the out-of-order direction: an older standing claim added AFTER a newer undated occurrence
    ledger = cl.ClaimLedger()
    ledger.add(lc(value="5", at="2026-10-20T09:00:00", scope="occurrence", occ=""))
    older_standing = lc(value="8", at="2026-10-18T09:00:00")
    ledger.add(older_standing)
    assert older_standing.superseded_by is None


def test_out_of_order_occurrence_add_marks_the_older_one_superseded_in_its_own_date_only():
    ledger = cl.ClaimLedger()
    newer = lc(value="6", at="2026-10-20T11:00:00", scope="occurrence", occ="2026-10-19")
    older_same = lc(value="7", at="2026-10-20T09:00:00", scope="occurrence", occ="2026-10-19")
    older_other = lc(value="4", at="2026-10-19T09:00:00", scope="occurrence", occ="2026-10-18")
    for c in (newer, older_same, older_other):
        ledger.add(c)
    assert older_same.superseded_by == newer.id and older_other.superseded_by is None


def test_open_subjects_lists_an_occurrence_subject_once_by_key():
    ledger = cl.ClaimLedger()
    for day, value in (("2026-10-17", "5"), ("2026-10-18", "6"), ("2026-10-19", "7")):
        ledger.add(lc(value=value, at=f"{day[:8]}{int(day[8:]) + 1:02d}T09:00:00",
                      scope="occurrence", occ=day))
    ledger.add(lc(key="days_behind", value="6", unit="days", at="2026-10-20T08:00:00", desc="days behind"))
    rows = ledger.open_subjects()
    assert sorted(r["subject_key"] for r in rows) == ["days_behind", "sleep_hours"]
    sleep = next(r for r in rows if r["subject_key"] == "sleep_hours")
    assert (sleep["scope"], sleep["occurrence_date"], sleep["value"]) == ("occurrence", "2026-10-19", "7")
    assert [r["subject_key"] for r in ledger.open_subjects(scope="standing")] == ["days_behind"]


def test_catalogue_marks_occurrence_subjects_and_keeps_the_standing_shape():
    ledger = cl.ClaimLedger()
    ledger.add(lc(scope="occurrence", occ="2026-10-19"))
    ledger.add(lc(key="days_behind", value="6", unit="days", desc="days behind"))
    rows = json.loads(cl._render_open_subjects(ledger.open_subjects()))
    by_key = {r["key"]: r for r in rows}
    assert sorted(by_key["days_behind"]) == ["key", "kind", "unit", "what"]
    assert by_key["sleep_hours"]["scope"] == "occurrence"


# --- persistence: legacy files without scope ------------------------------------------------------

def test_legacy_claim_without_scope_loads_as_standing(tmp_path):
    path = tmp_path / "ledger.json"
    legacy = lc().to_dict()
    del legacy["scope"], legacy["occurrence_date"]
    safe_json.atomic_write_json(str(path), {"schema_version": 1, "claims": [legacy]})
    ledger = cl.ClaimLedger.load(str(path))
    assert [(c.scope, c.occurrence_date) for c in ledger.claims] == [("standing", "")]


def test_scope_round_trips_and_a_bad_stored_scope_is_a_corrupt_store(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = cl.ClaimLedger([lc(scope="occurrence", occ="2026-10-19")])
    ledger.save(str(path))
    assert [(c.scope, c.occurrence_date) for c in cl.ClaimLedger.load(str(path)).claims] == [
        ("occurrence", "2026-10-19")]
    bad = lc().to_dict()
    bad["scope"] = "forever"
    safe_json.atomic_write_json(str(path), {"schema_version": 1, "claims": [bad]})
    with pytest.raises(safe_json.CorruptStoreError):
        cl.ClaimLedger.load(str(path))


# --- reply check: standing claims only; modality classes split -------------------------------------

def _finding(key, value, quote, unit="hours", kind="duration", modality="asserted"):
    return {"subject_key": key, "kind": kind, "value": value, "unit": unit, "modality": modality,
            "quote": quote}


def test_reply_check_never_compares_with_an_occurrence_claim():
    ledger = cl.ClaimLedger([lc(value="5", scope="occurrence", occ="2026-10-19")])
    reply = "You said you got eight hours of sleep."
    findings, drops = cl.validate_reply_findings(
        json.dumps({"findings": [_finding("sleep_hours", "8", "eight hours of sleep")]}),
        reply, "2026-10-20T20:00:00", ledger)
    assert findings == [] and [d["reason"] for d in drops] == ["unknown_subject"]
    # the same finding against a STANDING claim of that key is admitted (control)
    ledger_standing = cl.ClaimLedger([lc(value="5")])
    findings, _ = cl.validate_reply_findings(
        json.dumps({"findings": [_finding("sleep_hours", "8", "eight hours of sleep")]}),
        reply, "2026-10-20T20:00:00", ledger_standing)
    assert [f.classification for f in findings] == ["stale"]


def test_check_reply_catalogue_lists_standing_subjects_only():
    ledger = cl.ClaimLedger([lc(scope="occurrence", occ="2026-10-19"),
                             lc(key="days_behind", value="6", unit="days", desc="days behind")])
    seen = {}

    async def call(system, user):
        seen["user"] = user
        return json.dumps({"findings": []})

    import asyncio
    asyncio.run(cl.check_reply("You are six days behind.", "2026-10-20T20:00:00", ledger, call))
    assert '"key":"days_behind"' in seen["user"] and "sleep_hours" not in seen["user"]


REPLY_PLAIN = ["That means you were six days behind.", "That means you were\n    six days behind."]


@pytest.mark.parametrize("reply", REPLY_PLAIN, ids=["clean", "wrapped"])
def test_conditional_claim_restated_as_fact_is_conditional_as_fact(reply):
    ledger = cl.ClaimLedger([lc(key="days_behind", value="6", unit="days", modality="conditional",
                                at="2026-10-17T10:00:00", desc="days behind")])
    findings, _ = cl.validate_reply_findings(
        json.dumps({"findings": [_finding("days_behind", "6", "were six days behind", unit="days")]}),
        reply, "2026-10-20T20:00:00", ledger)
    assert [f.classification for f in findings] == ["conditional_as_fact"]


@pytest.mark.parametrize("reply", REPLY_PLAIN, ids=["clean", "wrapped"])
def test_estimate_claim_restated_as_fact_is_estimate_as_fact(reply):
    ledger = cl.ClaimLedger([lc(key="days_behind", value="6", unit="days", modality="estimate",
                                at="2026-10-17T10:00:00", desc="days behind")])
    findings, _ = cl.validate_reply_findings(
        json.dumps({"findings": [_finding("days_behind", "6", "were six days behind", unit="days")]}),
        reply, "2026-10-20T20:00:00", ledger)
    assert [f.classification for f in findings] == ["estimate_as_fact"]


def test_live_classes_are_exactly_the_two_modality_classes():
    assert cl.REPLY_CHECK_LIVE_CLASSES == ("conditional_as_fact", "estimate_as_fact")
    assert set(cl.REPLY_CHECK_LIVE_CLASSES) <= set(cl.CLASSES)
    # the offline-only classes remain in the vocabulary for the baseline script
    assert {"stale", "differs_from_estimate"} <= set(cl.CLASSES) - set(cl.REPLY_CHECK_LIVE_CLASSES)
