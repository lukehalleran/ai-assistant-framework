"""Claim-ledger subject keys (measured 2026-10-26; wave 2 2026-10-27; class: BC-55, BC-52).

The model invents snake_case subject keys; an awkward one
("weeks_4_through_7_set_with_it") fragmented supersession: a restatement under
a cleaner key ("weeks_set") never superseded it. Supersession now matches on
`subject_match_key` (numerals and function words dropped, content lemmas kept);
the stored key is never rewritten. Content words are kept (not collapsed to a
head noun) because the extraction prompt requires "days_behind_reading" and
"days_to_catch_up" to stay different subjects.
"""
from __future__ import annotations

from memory.claim_ledger import ClaimLedger, LedgerClaim, make_claim_id


def claim(key, value, at, kind="count", unit="weeks"):
    quote = f"{value} {unit}"
    return LedgerClaim(
        id=make_claim_id(key, kind, at, value, quote), subject_key=key,
        description="d", kind=kind, value=value, unit=unit, unit_norm="week",
        modality="asserted", stated_at=at, quote=quote)


def test_awkward_and_clean_keys_for_one_subject_supersede():
    ledger = ClaimLedger()
    old = claim("weeks_4_through_7_set_with_it", "3", "2026-10-20T09:00:00")
    new = claim("weeks_set", "5", "2026-10-22T09:00:00")
    ledger.add(old)
    assert ledger.add(new) == [old.id]
    assert old.superseded_by == new.id
    assert old.subject_key == "weeks_4_through_7_set_with_it"      # stored key untouched


def test_out_of_order_add_marks_the_older_awkward_claim_superseded():
    ledger = ClaimLedger()
    newer = claim("weeks_set", "5", "2026-10-22T09:00:00")
    older = claim("weeks_4_through_7_set_with_it", "3", "2026-10-20T09:00:00")
    ledger.add(newer)
    ledger.add(older)
    assert older.superseded_by == newer.id


def test_distinct_subjects_of_one_unit_stay_distinct():
    ledger = ClaimLedger()
    a = claim("days_behind_reading", "6", "2026-10-20T09:00:00", unit="days")
    b = claim("days_to_catch_up", "3", "2026-10-21T09:00:00", unit="days")
    ledger.add(a)
    assert ledger.add(b) == []
    assert a.superseded_by is None


def test_match_key_ignores_numerals_function_words_and_plurals():
    from memory.claim_ledger import subject_match_key   # new name: keep the other tests collectable on base
    assert (subject_match_key("weeks_4_through_7_set_with_it")
            == subject_match_key("weeks_set") == subject_match_key("week_set"))
    assert subject_match_key("it_with") == "it_with"      # nothing content-bearing: raw key


def test_lone_numerals_identify_the_subject_and_are_kept():
    from memory.claim_ledger import subject_match_key
    assert subject_match_key("assignment_7") != subject_match_key("assignment_8")
    ledger = ClaimLedger()
    a = claim("assignment_7", "2", "2026-10-20T09:00:00", unit="days")
    b = claim("assignment_8", "4", "2026-10-21T09:00:00", unit="days")
    ledger.add(a)
    assert ledger.add(b) == []
