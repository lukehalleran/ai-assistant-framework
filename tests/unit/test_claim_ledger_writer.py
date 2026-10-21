"""Live write of the user-claim ledger (memory/claim_ledger_writer.py).

Incident 2026-10-20: Daemon restated the user's CONDITIONAL ("if I can get 6
done today then I am only 6 days behind") as fact and quoted a superseded
ESTIMATE. The ledger records the user's own figures at turn storage. Classes:
BC-75, BC-52, BC-89 (small-budget call -> reasoning off), BC-46 (the stored
ledger, not an instruction, is the structural input).

Drives the deployed ``record_turn`` with a fake model manager (never a paid
call). Texts are invented and asserted clean AND wrapped.
"""
from __future__ import annotations

import asyncio
import json
import os

import pytest

import memory.claim_ledger_store as store
import memory.claim_ledger_writer as writer
from config import app_config
from memory import claim_ledger as cl
from utils import safe_json

SAID = "2026-10-20T09:00:00"
CLEAN = "Quick update. I am only 6 days behind on the reading."
WRAPPED = "Quick update. I am only 6\n    days behind\n    on the reading."


class FakeMM:
    """Records every generate_once call; ``reply`` is the raw model text."""

    def __init__(self, reply=None, delay=0.0):
        self.calls = []
        self.delay = delay
        self.reply = reply if reply is not None else json.dumps({"claims": [{
            "kind": "duration", "subject_key": "days_behind", "new_subject": True,
            "description": "days behind on the reading", "value": "6", "unit": "days",
            "modality": "asserted", "scope": "standing", "quote": "I am only 6 days behind"}]})

    async def generate_once(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.reply


class ExplodingMM:
    async def generate_once(self, *a, **k):
        raise AssertionError("no model call was expected")


@pytest.fixture()
def enabled(monkeypatch):
    import knowledge.web_search_manager as wsm
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_ENABLED", True)
    monkeypatch.setattr(wsm, "live_remaining_credits", lambda: None)      # unknown = proceed (process-global limiters)


def run(coro):
    return asyncio.run(coro)


def test_flag_off_returns_at_once_with_zero_model_calls():
    receipt = {}
    out = run(writer.record_turn(CLEAN, SAID, "t1", ExplodingMM(), receipt=receipt))
    assert out == {} and receipt == {}                       # no keys at all when disabled
    assert not os.path.exists(store.store_path())


def test_flag_off_is_a_live_read_same_instance(monkeypatch):
    import knowledge.web_search_manager as wsm
    monkeypatch.setattr(wsm, "live_remaining_credits", lambda: None)
    mm = FakeMM()
    assert run(writer.record_turn(CLEAN, SAID, "t1", mm)) == {} and mm.calls == []
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_ENABLED", True)         # flip on the SAME module
    assert run(writer.record_turn(CLEAN, SAID, "t1", mm))["claim_ledger_status"] == "ok"
    assert len(mm.calls) == 1


@pytest.mark.parametrize("text", ["[test]\nI am only 6 days behind on the reading.\n[/test]",
                                  "  [test]\n    I am only 6 days\n    behind[/test]"], ids=["clean", "wrapped"])
def test_probe_turn_makes_zero_calls(enabled, text):
    out = run(writer.record_turn(text, SAID, "t1", ExplodingMM()))
    assert out == {"claim_ledger_written": 0, "claim_ledger_status": "skipped:test_probe"}


@pytest.mark.parametrize("text", ["", "   \n  "])
def test_empty_authored_text_makes_zero_calls(enabled, text):
    out = run(writer.record_turn(text, SAID, "t1", ExplodingMM()))
    assert out["claim_ledger_status"] == "skipped:empty"


@pytest.mark.parametrize("remaining,calls", [(0, 0), (-3.0, 0), (None, 1), (25, 1)])
def test_credits_at_or_below_zero_skip_and_unknown_proceeds(enabled, monkeypatch, remaining, calls):
    import knowledge.web_search_manager as wsm
    monkeypatch.setattr(wsm, "live_remaining_credits", lambda: remaining)
    mm = FakeMM()
    out = run(writer.record_turn(CLEAN, SAID, "t1", mm))
    assert len(mm.calls) == calls
    assert out["claim_ledger_status"] == ("skipped:no_credits" if calls == 0 else "ok")


@pytest.mark.parametrize("text", [CLEAN, WRAPPED], ids=["clean", "wrapped"])
def test_success_adds_the_claim_saves_and_uses_the_deployed_call_shape(enabled, text):
    mm = FakeMM()
    receipt = {}
    out = run(writer.record_turn(text, SAID, "turn-9", mm, receipt=receipt))
    assert out == {"claim_ledger_written": 1, "claim_ledger_status": "ok"} == receipt
    (call,) = mm.calls
    assert call["model_name"] == app_config.CLAIM_LEDGER_MODEL
    assert call["temperature"] == 0.0 and call["disable_reasoning"] is True       # BC-89
    assert call["max_tokens"] == cl.EXTRACT_MAX_TOKENS
    saved = json.loads(open(store.store_path()).read())
    assert saved["schema_version"] == 1
    (c,) = saved["claims"]
    assert (c["subject_key"], c["value"], c["scope"], c["turn_id"]) == ("days_behind", "6", "standing", "turn-9")


def test_open_subjects_from_the_store_reach_the_next_prompt(enabled):
    run(writer.record_turn(CLEAN, SAID, "t1", FakeMM()))
    mm = FakeMM(reply=json.dumps({"claims": []}))
    run(writer.record_turn("Another update: nothing new.", "2026-10-21T09:00:00", "t2", mm))
    assert '"key":"days_behind"' in mm.calls[0]["prompt"]


def test_receipt_carries_labels_and_a_count_never_a_value_or_quote(enabled):
    out = run(writer.record_turn(CLEAN, SAID, "t1", FakeMM()))
    assert sorted(out) == ["claim_ledger_status", "claim_ledger_written"]
    blob = json.dumps(out)
    assert "days" not in blob and "behind" not in blob


def test_a_validator_drop_is_not_an_error_and_writes_nothing(enabled):
    bad = json.dumps({"claims": [{"kind": "duration", "subject_key": "days_behind", "new_subject": True,
                                  "description": "d", "value": "6", "unit": "days", "modality": "asserted",
                                  "quote": "a sentence the user never wrote"}]})
    out = run(writer.record_turn(CLEAN, SAID, "t1", FakeMM(reply=bad)))
    assert out == {"claim_ledger_written": 0, "claim_ledger_status": "ok"}
    assert not os.path.exists(store.store_path())


@pytest.mark.parametrize("reply", ["[API Error] boom", "[CREDITS EXHAUSTED] empty", "[RATE LIMITED] slow"])
def test_provider_error_sentinel_is_status_error_and_writes_nothing(enabled, reply):
    out = run(writer.record_turn(CLEAN, SAID, "t1", FakeMM(reply=reply)))
    assert out == {"claim_ledger_written": 0, "claim_ledger_status": "error"}
    assert not os.path.exists(store.store_path())


def test_timeout_is_status_timeout_and_the_store_is_unaffected(enabled, monkeypatch):
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_WRITE_TIMEOUT_S", 0.05)
    out = run(writer.record_turn(CLEAN, SAID, "t1", FakeMM(delay=1.0)))
    assert out == {"claim_ledger_written": 0, "claim_ledger_status": "timeout"}
    assert not os.path.exists(store.store_path())


def test_a_raising_model_degrades_to_error_never_raises(enabled):
    class Boom:
        async def generate_once(self, *a, **k):
            raise RuntimeError("provider down")

    out = run(writer.record_turn(CLEAN, SAID, "t1", Boom()))
    assert out["claim_ledger_status"] == "error"


def test_corrupt_store_is_never_replaced_by_an_empty_ledger(enabled):
    path = store.store_path()
    open(path, "w").write("{ broken")
    mm = FakeMM()
    out = run(writer.record_turn(CLEAN, SAID, "t1", mm))
    assert out["claim_ledger_status"] == "error" and mm.calls == []        # failed before any spend
    assert open(path).read() == "{ broken"                                # the original is untouched
    assert any(".corrupt-" in n for n in os.listdir(os.path.dirname(path)))


def test_newer_schema_is_error_and_untouched(enabled):
    body = json.dumps({"schema_version": 99, "claims": []})
    open(store.store_path(), "w").write(body)
    out = run(writer.record_turn(CLEAN, SAID, "t1", FakeMM()))
    assert out["claim_ledger_status"] == "error" and open(store.store_path()).read() == body


def test_concurrent_turns_both_land_and_the_file_has_both(enabled):
    def reply(value, quote):
        return json.dumps({"claims": [{
            "kind": "duration", "subject_key": f"k{value}", "new_subject": True, "description": "d",
            "value": value, "unit": "days", "modality": "asserted", "scope": "standing", "quote": quote}]})

    async def scenario():
        a = writer.record_turn("I am 6 days behind.", SAID, "a", FakeMM(reply("6", "I am 6 days behind"), 0.03))
        b = writer.record_turn("I have 9 days left.", SAID, "b", FakeMM(reply("9", "I have 9 days left"), 0.01))
        return await asyncio.gather(a, b)

    outs = run(scenario())
    assert [o["claim_ledger_written"] for o in outs] == [1, 1]
    saved = json.loads(open(store.store_path()).read())
    assert sorted(c["value"] for c in saved["claims"]) == ["6", "9"]


def test_cancellation_propagates_it_is_not_swallowed(enabled):
    async def scenario():
        task = asyncio.create_task(writer.record_turn(CLEAN, SAID, "t1", FakeMM(delay=5.0)))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(scenario())
