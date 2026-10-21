"""A dropped auditor quote says WHY it was not an exact span (diagnostics only).

Incidents 2026-10-18/19/20: recall-answer audits flipped between
``completion:insufficient`` x4 and ``supported``/``discussion:insufficient``
with ``dropped 0 claim(s), 1 reference(s): ['source quote is not exact']``.
The design choice (relocate quotes in the cited row? stronger model?) needs
the cause of the misses. Class: BC-84 (one invalid element ...) -- measurement
step; the validator stays strict: the same references survive, with the same
statuses and counts, as before the categories existed.

Drives THE deployed ``_checked_references`` / ``_validate_claims`` /
``audit_personal_claims``. Invented names and rows only.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from core import personal_claim_check as pcc

ROW_USER = "Rowan: I finished the lab write-up yesterday and sent it to Tamsin."
ROW_ANCHORED = "I finished the write-up yesterday [Sat 2026-10-17] and slept badly."
ROW_ASSISTANT = "Daemon: We could review the \"Quarterly Plan\" together."
ROW_OTHER = "Tamsin owes me the budget sheet by Friday."


def _sources():
    return {
        "src_a": [{"source_id": "src_a", "role": "user", "text": ROW_USER}],
        "src_b": [{"source_id": "src_b", "role": "user", "text": ROW_ANCHORED}],
        "src_c": [{"source_id": "src_c", "role": "assistant", "text": ROW_ASSISTANT}],
        "src_d": [{"source_id": "src_d", "role": "user", "text": ROW_OTHER}],
    }


def _wrap(text, wrapped):
    return text.replace(" ", "\n    ", 3) if wrapped else text


def _miss(source_id, quote):
    drops: list[str] = []
    kept = pcc._checked_references([{"source_id": source_id, "quote": quote}], _sources(), drops)
    return kept, drops


CASES = [
    # (id, source, quote, category)
    ("marker_row_anchor", "src_b", "I finished the write-up yesterday and slept badly.", "marker"),
    ("marker_quote_anchor", "src_a", "I finished the lab write-up yesterday [Sat 2026-10-17]", "marker"),
    ("typography_apostrophe", "src_c", "We could review the “Quarterly Plan” together.", "whitespace_typography"),
    ("typography_dash", "src_a", "the lab write-up—yesterday", "absent"),
    ("case", "src_a", "i finished the LAB write-up yesterday", "case"),
    ("other_source", "src_a", "Tamsin owes me the budget sheet by Friday.", "other_source"),
    ("absent", "src_a", "I never mentioned a dentist.", "absent"),
]


@pytest.mark.parametrize("wrapped", [False, True], ids=["clean", "wrapped"])
@pytest.mark.parametrize("name,source_id,quote,category", CASES, ids=[c[0] for c in CASES])
def test_miss_is_still_dropped_and_names_category(name, source_id, quote, category, wrapped):
    # Wrapping the QUOTE (line breaks + indentation the model may copy) must not
    # change the category for the whitespace-insensitive ones.
    q = _wrap(quote, wrapped) if category in ("whitespace_typography", "case") else quote
    kept, drops = _miss(source_id, q)
    assert kept == []                                  # strictness unchanged
    assert drops == [f"source quote is not exact ({category})"]
    assert quote not in " ".join(drops)                # excerpt-free


def test_whitespace_category_for_wrapped_quote():
    kept, drops = _miss("src_a", "I finished the lab\n      write-up   yesterday")
    assert kept == []
    assert drops == ["source quote is not exact (whitespace_typography)"]


def test_exact_quote_survives_and_no_reason():
    kept, drops = _miss("src_a", "finished the lab write-up yesterday")
    assert kept == [{"source_id": "src_a", "quote": "finished the lab write-up yesterday"}]
    assert drops == []


def test_other_reasons_unchanged():
    drops: list[str] = []
    assert pcc._checked_references([{"source_id": "nope", "quote": "x"}], _sources(), drops) == []
    assert drops == ["unknown source id"]


def _payload(status, quote, source_id="src_b"):
    return {"claims": [{"text": "You finished the write-up.", "status": status,
                        "kind": "completion",
                        "evidence": [{"source_id": source_id, "quote": quote}]}]}


EVIDENCE = [dict(rows[0]) for rows in _sources().values()]
DRAFT = "You finished the write-up. Nice."


def test_validate_claims_statuses_and_counts_identical_to_base():
    # Literal expectations captured from the pre-change tree for this input.
    claims, counts, drops = pcc._validate_claims(
        DRAFT, _payload("supported", "I finished the write-up yesterday and slept badly."), EVIDENCE)
    assert [(c["status"], c["evidence"]) for c in claims] == [("insufficient", [])]
    assert counts == {"dropped_claims": 0, "dropped_evidence": 1, "demoted": 1, "relocated": 0,
                      "malformed": 0, "question_drops": 0, "proposal_drops": 0, "subject_drops": 0}
    assert drops == ["source quote is not exact (marker)"]


def test_validate_claims_contradicted_demotes_same_as_base():
    claims, counts, _ = pcc._validate_claims(DRAFT, _payload("contradicted", "no such text"), EVIDENCE)
    assert claims[0]["status"] == "insufficient" and counts["demoted"] == 1


def test_quote_miss_summary_aggregates_without_text():
    drops = ["source quote is not exact (marker)", "source quote is not exact (marker)",
             "source quote is not exact (absent)", "unknown source id",
             "source quote is not exact"]
    assert pcc.quote_miss_summary(drops) == {"marker": 2, "absent": 1}
    assert pcc.quote_miss_summary(["unknown source id"]) == {}


class _Model:
    def __init__(self, payload):
        self.payload = payload

    async def generate_once(self, prompt, **kwargs):
        return json.dumps(self.payload)


@pytest.mark.parametrize("wrapped", [False, True], ids=["clean", "wrapped"])
def test_audit_logs_one_aggregate_line(wrapped, caplog):
    quote = _wrap("I finished the write-up yesterday and slept badly.", wrapped)
    payload = _payload("supported", quote)
    with caplog.at_level(logging.DEBUG):
        result = asyncio.run(pcc.audit_personal_claims(DRAFT, EVIDENCE, _Model(payload)))
    assert result.status == "checked"
    lines = [r.getMessage() for r in caplog.records if "quote misses" in r.getMessage()]
    assert len(lines) == 1
    assert lines[0].endswith('quote misses: {"marker": 1}') or '"marker": 1' in lines[0]
    assert "write-up" not in lines[0]
    assert result.dropped_evidence_count == 1 and result.demoted_count == 1


def test_audit_no_aggregate_line_without_misses(caplog):
    payload = _payload("supported", "finished the write-up yesterday")
    with caplog.at_level(logging.DEBUG):
        asyncio.run(pcc.audit_personal_claims(DRAFT, EVIDENCE, _Model(payload)))
    assert not [r for r in caplog.records if "quote misses" in r.getMessage()]
