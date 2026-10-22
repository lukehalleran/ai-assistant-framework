"""Log-only reply check of a DELIVERED reply against the user-claim ledger.

Incident 2026-10-20: Daemon replied "after being six days behind" (the user's
CONDITIONAL "if I can get 6 done today then I am only 6 days behind") and quoted a
superseded ESTIMATE. The live check records only modality findings
(conditional_as_fact / estimate_as_fact) on STANDING claims, after delivery, as
counts. Classes: BC-75, BC-52, BC-89 (reasoning off, bounded), BC-91 (receipts carry
no values or quotes on share surfaces).

Drives the deployed memory.claim_ledger_reply_check.run and the handlers wiring
(_start_background_claim_reply_check / _write_turn_telemetry) with a fake model.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

import gui.handlers as handlers
import memory.claim_ledger_reply_check as rc
import memory.claim_ledger_store as store
from config import app_config
from memory import claim_ledger as cl

SAID = "2026-10-20T20:26:00"
REPLY = "Good luck with the push, especially after being six days behind last week."
REPLY_WRAPPED = "Good luck with the push,\n    especially after being six days\n    behind last week."


class FakeMM:
    def __init__(self, findings=None, delay=0.0, reply=None):
        self.calls, self.delay = [], delay
        self.reply = reply if reply is not None else json.dumps({"findings": findings or []})

    async def generate_once(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.reply


class ExplodingMM:
    async def generate_once(self, *a, **k):
        raise AssertionError("no model call expected")


def finding(key, value, quote, kind="duration", modality="asserted"):
    return {"subject_key": key, "kind": kind, "value": value, "unit": "days", "modality": modality, "quote": quote}


def seed(key="days_behind", value="6", modality="conditional", at="2026-10-17T10:00:00", scope="standing"):
    ledger = store.get_ledger()
    ledger.add(cl.LedgerClaim(
        id=cl.make_claim_id(key, "duration", at, value, modality + scope), subject_key=key,
        description="days behind", kind="duration", value=value, unit="days", modality=modality,
        stated_at=at, quote=f"{value} days", scope=scope,
        occurrence_date="2026-10-19" if scope == "occurrence" else ""))


@pytest.fixture()
def on(monkeypatch):
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_ENABLED", True)
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_REPLY_CHECK", "log_only")


def run(coro):
    return asyncio.run(coro)


# --- off / skip ------------------------------------------------------------------------------------

@pytest.mark.parametrize("enabled,mode", [(False, "log_only"), (True, "off"), (False, "off")])
def test_off_makes_zero_calls_and_returns_nothing(monkeypatch, enabled, mode):
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_ENABLED", enabled)
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_REPLY_CHECK", mode)
    seed()
    assert run(rc.run(REPLY, SAID, ExplodingMM())) == {}


def test_no_standing_subjects_skips_without_a_call(on):
    out = run(rc.run(REPLY, SAID, ExplodingMM()))
    assert out["claim_ledger_reply_status"] == "skipped:no_subjects" and out["claim_ledger_reply_findings"] == {}


def test_only_occurrence_claims_means_no_call(on):
    seed(scope="occurrence", modality="asserted")
    assert run(rc.run(REPLY, SAID, ExplodingMM()))["claim_ledger_reply_status"] == "skipped:no_subjects"


# --- findings ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("reply", [REPLY, REPLY_WRAPPED], ids=["clean", "wrapped"])
def test_log_only_records_counts_and_subject_keys_only(on, reply):
    seed()
    mm = FakeMM([finding("days_behind", "6", "after being six days behind")])
    out = run(rc.run(reply, SAID, mm))
    assert out == {"claim_ledger_reply_findings": {"conditional_as_fact": 1},
                   "claim_ledger_reply_status": "ok", "claim_ledger_reply_subjects": ["days_behind"]}
    (call,) = mm.calls
    assert call["model_name"] == app_config.CLAIM_LEDGER_MODEL
    assert call["disable_reasoning"] is True and call["temperature"] == 0.0
    assert call["max_tokens"] == cl.REPLY_MAX_TOKENS
    blob = json.dumps(out)
    assert "six" not in blob and "after being" not in blob          # no quote, no value text


def test_estimate_as_fact_counts(on):
    seed(key="sprint", value="12", modality="estimate", at="2026-10-18T09:00:00")
    mm = FakeMM([finding("sprint", "12", "twelve days of work")])
    out = run(rc.run("That is twelve days of work.", SAID, mm))
    assert out["claim_ledger_reply_findings"] == {"estimate_as_fact": 1}


def test_stale_and_differs_classes_are_never_recorded(on):
    seed(key="sprint", value="15", modality="asserted", at="2026-10-18T09:00:00")
    mm = FakeMM([finding("sprint", "12", "twelve days of work")])           # would classify as `stale`
    out = run(rc.run("That is twelve days of work.", SAID, mm))
    assert out["claim_ledger_reply_status"] == "ok" and out["claim_ledger_reply_findings"] == {}


def test_the_ledger_is_read_only(on):
    seed()
    before = json.dumps(store.get_ledger().to_payload())
    run(rc.run(REPLY, SAID, FakeMM([finding("days_behind", "6", "after being six days behind")])))
    assert json.dumps(store.get_ledger().to_payload()) == before


@pytest.mark.parametrize("raw", ["[API Error] boom", "[CREDITS EXHAUSTED] empty", "not json at all"])
def test_provider_failure_or_junk_is_error_or_empty_never_a_finding(on, raw):
    seed()
    out = run(rc.run(REPLY, SAID, FakeMM(reply=raw)))
    assert out["claim_ledger_reply_findings"] == {}
    assert out["claim_ledger_reply_status"] in ("error", "ok")


def test_timeout_status(on, monkeypatch):
    seed()
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_WRITE_TIMEOUT_S", 0.05)
    out = run(rc.run(REPLY, SAID, FakeMM(delay=1.0)))
    assert out["claim_ledger_reply_status"] == "timeout"


def test_corrupt_store_is_error_and_untouched(on):
    open(store.store_path(), "w").write("{ broken")
    out = run(rc.run(REPLY, SAID, ExplodingMM()))
    assert out["claim_ledger_reply_status"] == "error" and open(store.store_path()).read() == "{ broken"


# --- handlers wiring --------------------------------------------------------------------------------------

def make_ctx(mm):
    return SimpleNamespace(orchestrator=SimpleNamespace(model_manager=mm), telemetry={}, debug_record={})


@pytest.mark.asyncio
async def test_background_check_fills_turn_record_and_debug_record_and_leaves_the_reply_alone(on):
    seed()
    ctx = make_ctx(FakeMM([finding("days_behind", "6", "after being six days behind")]))
    reply = REPLY
    handlers._start_background_claim_reply_check(ctx, reply)
    task = ctx.claim_reply_task
    assert task in handlers._pending_storage_tasks                            # drained at shutdown
    await task
    await asyncio.sleep(0)
    assert ctx.telemetry == {"claim_ledger_reply_findings": {"conditional_as_fact": 1},
                             "claim_ledger_reply_status": "ok"}              # no subject keys on the turn row
    assert ctx.debug_record["claim_ledger_reply_subjects"] == ["days_behind"]
    assert ctx.debug_record["claim_ledger_reply_findings"] == {"conditional_as_fact": 1}
    assert reply == REPLY and task not in handlers._pending_storage_tasks


@pytest.mark.asyncio
async def test_off_schedules_nothing_same_instance(monkeypatch):
    seed()
    ctx = make_ctx(ExplodingMM())
    handlers._start_background_claim_reply_check(ctx, REPLY)
    assert getattr(ctx, "claim_reply_task", None) is None
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_ENABLED", True)
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_REPLY_CHECK", "log_only")
    ctx2 = make_ctx(FakeMM())
    handlers._start_background_claim_reply_check(ctx2, REPLY)
    assert ctx2.claim_reply_task is not None
    await ctx2.claim_reply_task


@pytest.mark.asyncio
async def test_does_not_block_delivery_the_scheduler_returns_before_the_model_finishes(on):
    seed()
    ctx = make_ctx(FakeMM(delay=0.3))
    handlers._start_background_claim_reply_check(ctx, REPLY)
    assert not ctx.claim_reply_task.done() and ctx.telemetry == {}
    await ctx.claim_reply_task


@pytest.mark.asyncio
async def test_cancellation_is_recorded_and_propagates(on):
    seed()
    ctx = make_ctx(FakeMM(delay=5.0))
    handlers._start_background_claim_reply_check(ctx, REPLY)
    await asyncio.sleep(0.05)
    ctx.claim_reply_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await ctx.claim_reply_task
    await asyncio.sleep(0)
    assert ctx.telemetry["claim_ledger_reply_status"] == "cancelled"
    assert ctx.claim_reply_task not in handlers._pending_storage_tasks


@pytest.mark.asyncio
async def test_one_check_per_turn(on):
    seed()
    mm = FakeMM()
    ctx = make_ctx(mm)
    handlers._start_background_claim_reply_check(ctx, REPLY)
    handlers._start_background_claim_reply_check(ctx, REPLY)
    await ctx.claim_reply_task
    assert len(mm.calls) == 1


@pytest.mark.asyncio
async def test_empty_reply_or_missing_model_schedules_nothing(on):
    seed()
    for ctx, text in ((make_ctx(FakeMM()), "  "), (SimpleNamespace(telemetry={}), REPLY)):
        handlers._start_background_claim_reply_check(ctx, text)
        assert getattr(ctx, "claim_reply_task", None) is None


@pytest.mark.asyncio
async def test_turn_telemetry_hook_starts_the_check_and_the_row_waits_for_it(on, monkeypatch):
    import core.orchestrator as orch
    seed()
    captured = {}
    monkeypatch.setattr(orch, "run_post_response_hooks", lambda hook_ctx: captured.setdefault("hook", hook_ctx))
    ctx = make_ctx(FakeMM([finding("days_behind", "6", "after being six days behind")]))
    ctx.user_text = "update"
    handlers._write_turn_telemetry(ctx, "enhanced", "s1", "m", len(REPLY), response_text=REPLY)
    assert ctx.claim_reply_task is not None
    waiter = captured["hook"].telemetry_task
    assert waiter is ctx.claim_reply_task                                     # the row waits for the receipt
    await waiter
    await asyncio.sleep(0)
    assert ctx.telemetry["claim_ledger_reply_findings"] == {"conditional_as_fact": 1}
