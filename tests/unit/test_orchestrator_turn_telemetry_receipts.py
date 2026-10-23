"""Turn rows carry the deferred claim-ledger receipts and an explicit probe flag.

Incident 2026-10-22. (1) class BC-25 (dead wiring): ``_hook_turn_telemetry`` copied
background-callback receipts into the deferred row through a three-literal prefix
allow-list, so the 2026-10-20 ``claim_ledger_*`` receipts reached ``ctx.telemetry``
(where their tests looked) but never ``logs/turn_records.jsonl``. (2) class BC-58
(test-state isolation): consumers recognised a probe by its ``[test]...[/test]``
envelope inside the 300-char ``query``; a probe whose closing tag lay past char 300
counted as a real turn. Rows now carry ``test_probe`` computed from the FULL text by
THE corpus predicate. Invented content only; envelope forms clean and wrapped.
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import orchestrator as orch_mod
from core.orchestrator import (
    DEFERRED_RECEIPT_PREFIXES,
    PostResponseHookContext,
    _hook_turn_telemetry,
)
from memory.pattern_engine import PatternQuery, run_pattern_query

NOW = datetime(2026, 10, 22, 15, 0, 0)


@pytest.fixture
def captured(monkeypatch):
    rows = []
    import utils.turn_telemetry as tt
    monkeypatch.setattr(tt, "record_turn", lambda rec: rows.append(dict(rec)) or True)
    return rows


def _ctx(user_input="Tamsin moved the standup.", telemetry=None, task=None):
    return PostResponseHookContext(
        orchestrator=SimpleNamespace(_last_turn_signals={"tone_level": "CONVERSATIONAL"}),
        user_input=user_input, telemetry=telemetry if telemetry is not None else {},
        telemetry_task=task, mode="enhanced")


# ---------------------------------------------------------------- item 1

def test_deferred_path_writes_claim_ledger_receipts(captured):
    async def run():
        async def work():
            await asyncio.sleep(0)
        task = asyncio.create_task(work())
        ctx = _ctx(task=task)
        _hook_turn_telemetry(ctx)
        assert captured == []                       # nothing written until the task ends
        # background callbacks land AFTER the hook ran, as in gui/handlers.py
        ctx.telemetry.update({"claim_ledger_written": 2, "claim_ledger_status": "ok"})
        ctx.telemetry.update({"claim_ledger_reply_findings": {"a": 1},
                              "claim_ledger_reply_status": "complete"})
        ctx.telemetry["storage_failed"] = "x"
        ctx.telemetry["grounding_status"] = "complete"
        ctx.telemetry["personal_claim_status"] = "ok"
        ctx.telemetry["unrelated_key"] = "must not leak"
        await task
        await asyncio.sleep(0)
        return ctx
    asyncio.run(run())
    assert len(captured) == 1
    row = captured[0]
    for k in ("claim_ledger_written", "claim_ledger_status", "claim_ledger_reply_findings",
              "claim_ledger_reply_status", "storage_failed", "grounding_status",
              "personal_claim_status"):
        assert k in row, k
    assert "unrelated_key" not in row


def test_synchronous_path_unchanged(captured):
    _hook_turn_telemetry(_ctx(telemetry={"claim_ledger_written": 1, "foo": "bar"}))
    assert captured[0]["claim_ledger_written"] == 1 and captured[0]["foo"] == "bar"


def test_prefix_tuple_covers_every_receipt_key_handlers_write():
    src = Path("gui/handlers.py").read_text()
    # Keys written inside the two background done-callbacks (receipt producers) --
    # explicit expected set; asserted against the source so a rename is noticed.
    expected = {"claim_ledger_reply_findings", "claim_ledger_reply_status",
                "storage_failed", "grounding_status"}
    for k in expected:
        assert k in src, k
    from memory import claim_ledger_writer
    writer_src = Path(claim_ledger_writer.__file__).read_text()
    expected |= {"claim_ledger_written", "claim_ledger_status"}
    for k in ("claim_ledger_written", "claim_ledger_status"):
        assert k in writer_src
    for k in expected:
        assert k.startswith(DEFERRED_RECEIPT_PREFIXES), k
    # and the copy reads the tuple (no second literal list)
    body = Path(orch_mod.__file__).read_text()
    assert "k.startswith(DEFERRED_RECEIPT_PREFIXES)" in body


# ---------------------------------------------------------------- item 2

BODY = "Rowan says the budget sheet needs a second tab. " * 8   # ~380 chars


@pytest.mark.parametrize("wrap", [
    "[test]{b}[/test]",
    "   [test]{b}[/test]   ",
    "[test]\n{b}\n[/test]",
    "  [test]\n  {b}\n  [/test]",
], ids=["inline", "wrapped", "whole_line", "indented"])
def test_long_probe_past_300_chars_is_flagged(captured, wrap):
    text = wrap.format(b=BODY)
    assert text.find("[/test]") > 300
    _hook_turn_telemetry(_ctx(user_input=text))
    row = captured[0]
    assert row["test_probe"] is True
    assert "[/test]" not in row["query"]            # the old envelope parse would miss it


def test_normal_turn_is_not_flagged(captured):
    _hook_turn_telemetry(_ctx(user_input="Tamsin moved the standup to ten."))
    assert captured[0]["test_probe"] is False


def _tone_rows(tmp_path, rows):
    path = tmp_path / "turns.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return run_pattern_query(PatternQuery(dimension="tone", window_days=7, now=NOW),
                             telemetry_path=str(path))


def test_pattern_tone_excludes_flagged_row_with_truncated_query(tmp_path):
    rows = [
        {"ts": (NOW - timedelta(days=2)).isoformat(), "tone_level": "CONCERN",
         "query": "I am worn out today", "test_probe": False},
        {"ts": (NOW - timedelta(days=1)).isoformat(), "tone_level": "HIGH",
         "query": ("[test]" + BODY)[:300], "test_probe": True},
    ]
    res = _tone_rows(tmp_path, rows)
    assert res.total == 1
    assert {v for b in res.buckets for v in b.values} == {"concern"}


def test_old_row_without_flag_still_uses_envelope_parse(tmp_path):
    rows = [
        {"ts": (NOW - timedelta(days=2)).isoformat(), "tone_level": "CONCERN",
         "query": "I am worn out today"},
        {"ts": (NOW - timedelta(days=1)).isoformat(), "tone_level": "HIGH",
         "query": "[test]short probe[/test]"},
    ]
    assert _tone_rows(tmp_path, rows).total == 1


def test_flag_false_row_counts_even_without_envelope(tmp_path):
    rows = [{"ts": (NOW - timedelta(days=1)).isoformat(), "tone_level": "HIGH",
             "query": "real turn", "test_probe": False}]
    assert _tone_rows(tmp_path, rows).total == 1
