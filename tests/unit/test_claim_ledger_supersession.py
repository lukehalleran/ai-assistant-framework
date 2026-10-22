"""Claim-ledger supersession rules (memory/claim_ledger.py).

Incident 2026-10-20: a user's ESTIMATE ("12 days i think?") outlived its
worked-out replacement (~15 days) and a CONDITIONAL ("if I get 6 done...")
came back as fact. Classes: BC-75 (self-authored artifact laundered into
evidence), BC-52 (lossy multi-hop derivation, stale claim resurfaces).

Drives the deployed ClaimLedger; every value is invented.
"""
from __future__ import annotations

import pytest

from memory.claim_ledger import ClaimLedger, LedgerClaim, make_claim_id
from utils import safe_json


def claim(key="sprint_span", value="12", modality="asserted", at="2026-10-18T09:00:00",
          kind="duration", unit="days", desc="length of the sprint"):
    quote = f"{value} {unit}"
    return LedgerClaim(
        id=make_claim_id(key, kind, at, value, quote + modality), subject_key=key,
        description=desc, kind=kind, value=value, unit=unit, modality=modality,
        stated_at=at, quote=quote)


def test_asserted_supersedes_older_asserted():
    ledger = ClaimLedger()
    first = claim(value="12", at="2026-10-18T09:00:00")
    second = claim(value="15", at="2026-10-20T02:47:00")
    assert ledger.add(first) == []
    assert ledger.add(second) == [first.id]
    assert first.superseded_by == second.id
    assert ledger.current("sprint_span").value == "15"


def test_estimate_same_day_does_not_supersede_asserted():
    ledger = ClaimLedger()
    asserted = claim(value="15", at="2026-10-20T09:00:00")
    estimate = claim(value="12", modality="estimate", at="2026-10-20T15:00:00")
    ledger.add(asserted)
    assert ledger.add(estimate) == []
    assert asserted.superseded_by is None
    assert ledger.current("sprint_span").value == "15"   # asserted outranks the same-day estimate


def test_estimate_next_day_does_supersede_asserted():
    ledger = ClaimLedger()
    asserted = claim(value="15", at="2026-10-20T09:00:00")
    estimate = claim(value="12", modality="estimate", at="2026-10-21T09:00:00")
    ledger.add(asserted)
    assert ledger.add(estimate) == [asserted.id]
    assert ledger.current("sprint_span").value == "12"


def test_correction_supersedes_even_a_same_day_asserted_of_another_kind():
    ledger = ClaimLedger()
    asserted = claim(value="15", at="2026-10-20T09:00:00", kind="count")
    fix = claim(value="18", modality="correction", at="2026-10-20T09:30:00", kind="duration")
    ledger.add(asserted)
    assert ledger.add(fix) == [asserted.id]
    assert ledger.current("sprint_span").value == "18"


def test_conditional_is_recorded_but_never_current_and_never_supersedes():
    ledger = ClaimLedger()
    cond = claim(key="days_behind", value="6", modality="conditional", at="2026-10-17T10:00:00",
                 desc="days behind schedule")
    assert ledger.add(cond) == []
    assert ledger.current("days_behind") is None
    assert len(ledger.claims) == 1
    # the subject is still listed (so its key can be reused) but flagged conditional
    listed = ledger.open_subjects()
    assert [(s["subject_key"], s["modality"]) for s in listed] == [("days_behind", "conditional")]
    # an asserted claim becomes current; a LATER conditional neither supersedes it nor replaces it
    asserted = claim(key="days_behind", value="9", at="2026-10-18T10:00:00", desc="days behind schedule")
    ledger.add(asserted)
    later_cond = claim(key="days_behind", value="2", modality="conditional", at="2026-10-19T10:00:00",
                       desc="days behind schedule")
    assert ledger.add(later_cond) == []
    assert asserted.superseded_by is None
    assert ledger.current("days_behind").value == "9"
    assert cond.superseded_by is None


def test_out_of_order_add_marks_the_older_claim_superseded():
    ledger = ClaimLedger()
    newer = claim(value="15", at="2026-10-20T02:47:00")
    older = claim(value="12", at="2026-10-18T09:00:00")
    ledger.add(newer)
    assert ledger.add(older) == []
    assert older.superseded_by == newer.id
    assert ledger.current("sprint_span").value == "15"


def test_as_of_replays_supersession_at_the_earlier_time():
    ledger = ClaimLedger()
    ledger.add(claim(value="12", at="2026-10-18T09:00:00"))
    ledger.add(claim(value="15", at="2026-10-20T02:47:00"))
    assert ledger.current("sprint_span", as_of="2026-10-19T12:00:00").value == "12"
    assert ledger.current("sprint_span", as_of="2026-10-20T03:00:00").value == "15"
    assert ledger.current("sprint_span", as_of="2026-10-17T00:00:00") is None
    # as_of returns copies: replaying history never mutates the ledger
    assert ledger.claims[0].superseded_by is not None


def test_duplicate_id_is_ignored():
    ledger = ClaimLedger()
    c = claim()
    ledger.add(c)
    assert ledger.add(LedgerClaim(**c.to_dict())) == []
    assert len(ledger.claims) == 1


def test_open_subjects_newest_first_and_capped():
    ledger = ClaimLedger()
    for i in range(45):
        ledger.add(claim(key=f"subject_{i:02d}", value=str(i + 1),
                         at=f"2026-09-{(i % 28) + 1:02d}T08:00:{i:02d}"))
    rows = ledger.open_subjects()
    assert len(rows) == 40
    stamps = [r["stated_at"] for r in rows]
    assert stamps == sorted(stamps, reverse=True)
    assert len(ledger.open_subjects(n=3)) == 3


def test_save_load_roundtrip_and_corrupt_store_is_quarantined(tmp_path):
    ledger = ClaimLedger()
    ledger.add(claim(value="12", at="2026-10-18T09:00:00"))
    ledger.add(claim(value="15", at="2026-10-20T02:47:00"))
    path = tmp_path / "ledger.json"
    ledger.save(str(path))
    again = ClaimLedger.load(str(path))
    assert [c.to_dict() for c in again.claims] == [c.to_dict() for c in ledger.claims]
    assert ClaimLedger.load(str(tmp_path / "missing.json")).claims == []
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(safe_json.CorruptStoreError):
        ClaimLedger.load(str(path))
    assert any(".corrupt-" in p.name for p in tmp_path.iterdir())
    path.write_text('{"schema_version": 1, "claims": [{"id": 3}]}', encoding="utf-8")
    with pytest.raises(safe_json.CorruptStoreError):
        ClaimLedger.load(str(path))


# --- AMEND1 A2 (2026-10-20 trial): the subject is (subject_key, kind, unit_norm) --------------------------


def test_a2_different_units_are_separate_subjects_and_never_supersede():
    ledger = ClaimLedger()
    days = claim(key="days_behind", value="6", unit="days", at="2026-10-17T10:00:00")
    items = claim(key="days_behind", value="5", unit="assignments", at="2026-10-18T10:00:00")
    ledger.add(days)
    assert ledger.add(items) == []
    assert days.superseded_by is None and items.superseded_by is None
    assert ledger.current("days_behind", unit="days").value == "6"
    assert ledger.current("days_behind", unit="assignments").value == "5"
    assert ledger.current("days_behind").value == "5"            # no unit given: newest wins
    rows = ledger.open_subjects()
    assert sorted((r["subject_key"], r["unit"], r["value"]) for r in rows) == [
        ("days_behind", "assignments", "5"), ("days_behind", "days", "6")]


def test_a2_same_unit_still_supersedes():
    ledger = ClaimLedger()
    first = claim(key="days_behind", value="6", unit="days", at="2026-10-17T10:00:00")
    second = claim(key="days_behind", value="5", unit="days", at="2026-10-18T10:00:00")
    ledger.add(first)
    assert ledger.add(second) == [first.id]


def test_a2_unit_norm_defaults_to_the_lowercased_unit_and_roundtrips(tmp_path):
    c = claim(unit="Days")
    assert c.unit_norm == "days"
    ledger = ClaimLedger()
    c2 = LedgerClaim(**{**claim(value="3", unit="days").to_dict(), "unit_norm": "day"})
    ledger.add(c2)
    path = tmp_path / "l.json"
    ledger.save(str(path))
    assert ClaimLedger.load(str(path)).claims[0].unit_norm == "day"
    old = ledger.to_payload()
    del old["claims"][0]["unit_norm"]                            # a pre-AMEND1 file still loads
    path.write_text(__import__("json").dumps(old), encoding="utf-8")
    assert ClaimLedger.load(str(path)).claims[0].unit_norm == "days"
