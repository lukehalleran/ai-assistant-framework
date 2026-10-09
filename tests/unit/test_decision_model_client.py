"""models/decision_model.py through THE deployed transport seam (D5b).

Every call goes ModelManager.post_system_one -> httpx.MockTransport (no sockets). Fixtures are
real System One bodies from the Phase 0 capture (ids/users replaced); the choice body and the
402/429/529 responses are synthesized from the plan's V-rows. Every await sits under wait_for(2).
"""
import asyncio
import copy
import json
import logging
import math
import threading
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest

from config import app_config
from models import decision_model as dm
from models.decision_model import evaluate, noul, run_with_shadow, score, choice
from models.model_manager import API_MODEL_ALIASES, MODEL_CAPABILITIES, ModelManager

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "decision_model"
REPO = Path(__file__).resolve().parents[2]
SERVED, PROVIDER = "typesafe/jev-1.13-20260917", "TypeSafe"


def fx(name):
    return json.loads((FIX / name).read_text())


def T(coro, t=2):
    return asyncio.wait_for(coro, t)


TONE = fx("tone_score_ok.json")
LEVELS = [TONE["answers"]["tone"]["legend"][str(i)] for i in range(4)]
TONE_Q = score("tone", "judge the author's own state", LEVELS)
HEAVY_Q = noul("heavy", "is it heavy?", "heavy things", "normal things")
OPTS = {"a": "first option", "b": "second option", "c": "third option"}
CHOICE_Q = choice("pick", "choose one", OPTS)


def choice_body(**over):
    ans = {"type": "choice", "choice": "b", "probabilities": {"a": 0.1, "b": 0.8, "c": 0.1}, "confidence": 0.7}
    ans.update(over)
    return {"model": SERVED, "provider": PROVIDER, "answers": {"pick": ans}, "usage": {"input_tokens": 5}}


@pytest.fixture
def mm(monkeypatch):
    monkeypatch.setattr(ModelManager, "_get_cached_embedder", staticmethod(lambda: None))
    return ModelManager(api_key="test-key")


@pytest.fixture(autouse=True)
def live_cfg(monkeypatch):
    cfg = {"enabled": True, "roles": {"tone_arbiter": "shadow", "heavy_topic": "shadow"}}
    for k, v in {"DECISION_MODEL_CFG": cfg, "DECISION_MODEL_SERVED_MODELS": (SERVED,),
                 "DECISION_MODEL_PROVIDER": PROVIDER, "DECISION_MODEL_TIMEOUT_S": 1.5,
                 "DECISION_MODEL_SLUG": "typesafe/jev-1.13",
                 "DECISION_MODEL_MAX_STATE_CHARS": {"tone_arbiter": 100, "heavy_topic": 100}}.items():
        monkeypatch.setattr(app_config, k, v)


class RecTransport(httpx.MockTransport):
    """MockTransport that records aclose(): proves the transport was REALLY closed."""

    def __init__(self, handler, close_delay=0.0):
        super().__init__(handler)
        self.closed, self.close_delay = 0, close_delay

    async def aclose(self):
        await asyncio.sleep(self.close_delay)
        self.closed += 1


class Wire:
    """MockTransport handler: items are zero-arg callables returning a Response or an Exception."""

    def __init__(self, *items):
        self.items, self.requests = list(items), []

    def __call__(self, request):
        self.requests.append(request)
        item = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        out = item()
        if isinstance(out, Exception):
            raise out
        return out

    def install(self, mm):
        mm.system_one_transport = RecTransport(self)
        return self


def ok(body, **kw):
    return lambda: httpx.Response(200, json=body, **kw)


def status(code, body=None, headers=None):
    return lambda: httpx.Response(code, json=body or {"error": {"code": code}}, headers=headers)


async def tone(mm, state="hello there"):
    return await T(evaluate(mm, state, [TONE_Q], role="tone_arbiter"))


# --- happy path ----------------------------------------------------------------------------
async def test_tone_score_fixture_ok_and_request_shape(mm):
    wire = Wire(ok(TONE)).install(mm)
    out = await tone(mm, "some state")
    assert (out.status, out.reason, out.retried) == ("ok", None, False)
    a = out.answers["tone"]
    assert a.valid and a.score == 1 and a.probabilities["1"] == 1 and a.legend["0"] == LEVELS[0]
    assert (out.served_model, out.provider, out.input_tokens, out.cost_usd) == (SERVED, PROVIDER, 459, 1.9278e-05)
    req = wire.requests[0]
    assert str(req.url) == "https://openrouter.ai/api/v1/systemone"
    assert req.headers["authorization"] == "Bearer test-key"
    sent = json.loads(req.content)
    assert sent == {"model": "typesafe/jev-1.13", "state": "some state",
                    "questions": {"tone": {"type": "score", "instructions": "judge the author's own state",
                                           "criteria": LEVELS}}}


async def test_heavy_noul_fixture_ok(mm):
    Wire(ok(fx("heavy_noul_ok.json"))).install(mm)
    out = await T(evaluate(mm, "x", [HEAVY_Q], role="heavy_topic"))
    assert out.status == "ok" and out.answers["heavy"].noul == 0.81


async def test_unknown_top_level_keys_ignored_and_missing_cost_is_none(mm):
    body = copy.deepcopy(TONE)
    body["surprise"] = {"x": 1}
    del body["usage"]["cost"]
    Wire(ok(body)).install(mm)
    out = await tone(mm)
    assert out.status == "ok" and out.cost_usd is None  # BC-47: absent is not zero


# --- errors --------------------------------------------------------------------------------
@pytest.mark.parametrize("resp,want_status,want_reason,n_requests", [
    (status(401, fx("err_401.json")), "unavailable", "http_401", 1),
    (status(402), "unavailable", "http_402", 1),
    (status(400, fx("err_400.json")), "invalid", "http_400", 1),
    (status(422, fx("err_422.json")), "invalid", "http_422", 1),
    (status(429), "unavailable", "http_429", 2),   # no Retry-After: one retry, still 429
    (status(529), "unavailable", "http_529", 2),
    (status(500), "unavailable", "http_other", 1),
])
async def test_http_status_mapping(mm, resp, want_status, want_reason, n_requests):
    wire = Wire(resp).install(mm)
    out = await tone(mm)
    assert (out.status, out.reason) == (want_status, want_reason)
    assert len(wire.requests) == n_requests and out.answers == {}


async def test_429_then_200_retries_and_ok(mm):
    wire = Wire(status(429, headers={"Retry-After": "0"}), ok(TONE)).install(mm)
    out = await tone(mm)
    assert out.status == "ok" and out.retried is True and len(wire.requests) == 2


async def test_retry_after_longer_than_deadline_does_not_retry(mm):
    wire = Wire(status(429, headers={"Retry-After": "5"}), ok(TONE)).install(mm)
    out = await tone(mm)
    assert (out.status, out.reason, out.retried) == ("unavailable", "http_429", False)
    assert len(wire.requests) == 1


async def test_timeout_then_timeout_within_deadline(mm, monkeypatch):
    monkeypatch.setattr(app_config, "DECISION_MODEL_TIMEOUT_S", 0.5)
    wire = Wire(lambda: httpx.ReadTimeout("slow")).install(mm)
    t0 = time.monotonic()
    out = await tone(mm)
    assert (out.status, out.reason, out.retried) == ("unavailable", "timeout", True)
    assert len(wire.requests) == 2 and time.monotonic() - t0 <= 0.5 + 0.2


async def test_slow_server_is_cut_at_the_whole_deadline(mm, monkeypatch):
    monkeypatch.setattr(app_config, "DECISION_MODEL_TIMEOUT_S", 0.3)

    async def slow(request):
        await asyncio.sleep(5)
        return httpx.Response(200, json=TONE)

    mm.system_one_transport = httpx.MockTransport(slow)
    t0 = time.monotonic()
    out = await tone(mm)
    assert (out.status, out.reason) == ("unavailable", "timeout") and time.monotonic() - t0 <= 0.3 + 0.2


async def test_transport_error_retries_then_unavailable(mm):
    wire = Wire(lambda: httpx.ConnectError("boom")).install(mm)
    out = await tone(mm)
    assert (out.status, out.reason, out.retried) == ("unavailable", "transport", True) and len(wire.requests) == 2


async def test_unexpected_exception_never_escapes(mm):
    Wire(lambda: RuntimeError("bug")).install(mm)
    out = await tone(mm)
    assert (out.status, out.reason) == ("unavailable", "transport")


@pytest.mark.parametrize("text", ["{not json", "[1, 2]", '{"model": "x"}', '{"answers": []}'])
async def test_malformed_body_is_invalid_parse(mm, text):
    Wire(lambda: httpx.Response(200, text=text)).install(mm)
    out = await tone(mm)
    assert (out.status, out.reason) == ("invalid", "parse")


# --- D4: one mutation per rule, each on a copy of an ok fixture (BC-64) ----------------------
def _mut(fn):
    b = copy.deepcopy(TONE)
    fn(b)
    return b


def _set(path, value):
    def f(b):
        node = b
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
    return f


def _drop_legend(b):
    del b["answers"]["tone"]["legend"]["3"]


SCORE_MUTATIONS = {
    "legend_keys": _drop_legend,
    "legend_description": _set(["answers", "tone", "legend", "2"], "MEDIUM: something else entirely"),
    "score_range": _set(["answers", "tone", "score"], 4),
    "score_inconsistent": _set(["answers", "tone", "score"], 3),
    "confidence_range": _set(["answers", "tone", "confidence"], 1.2),
    "probability_value": _set(["answers", "tone", "probabilities", "1"], float("nan")),
    "probability_sum": _set(["answers", "tone", "probabilities", "1"], 0.9),
    "probability_keys": _set(["answers", "tone", "probabilities", "7"], 0.0),
    "type_mismatch": _set(["answers", "tone", "type"], "choice"),
    "answer_missing": lambda b: b["answers"].pop("tone"),
}


def _parse(body, q=TONE_Q):
    return dm.parse_response(body, [q], served_models=(SERVED,), provider=PROVIDER)


def test_unmutated_fixture_is_valid_positive_control():
    answers, meta = _parse(TONE)
    assert answers["tone"].valid and meta["reason"] is None


@pytest.mark.parametrize("reason", sorted(SCORE_MUTATIONS))
def test_score_rule_mutation(reason):
    answers, meta = _parse(_mut(SCORE_MUTATIONS[reason]))
    assert not answers["tone"].valid and answers["tone"].invalid_reason == reason
    assert meta["reason"] == "schema_mismatch"


@pytest.mark.parametrize("path,value", [(["model"], "typesafe/jev-9.9-other"), (["provider"], "OtherProvider"),
                                        (["provider"], None)])
async def test_unknown_model_or_provider_is_model_mismatch(mm, path, value):
    Wire(ok(_mut(_set(path, value)))).install(mm)
    out = await tone(mm)
    assert (out.status, out.reason) == ("invalid", "model_mismatch")
    assert out.answers["tone"].valid is False


def test_noul_range_and_choice_rules():
    heavy = fx("heavy_noul_ok.json")
    assert _parse(heavy, HEAVY_Q)[0]["heavy"].valid
    bad = copy.deepcopy(heavy)
    bad["answers"]["heavy"]["noul"] = 1.5
    assert _parse(bad, HEAVY_Q)[0]["heavy"].invalid_reason == "noul_range"
    assert _parse(choice_body(), CHOICE_Q)[0]["pick"].choice == "b"
    for over, reason in [({"choice": "z"}, "choice_not_in_criteria"),
                         ({"probabilities": {"a": 0.1, "b": 0.8, "q": 0.1}}, "probability_keys"),
                         ({"probabilities": {"a": 0.5, "b": 0.8, "c": 0.1}}, "probability_sum"),
                         ({"confidence": -0.1}, "confidence_range")]:
        assert _parse(choice_body(**over), CHOICE_Q)[0]["pick"].invalid_reason == reason


async def test_misbehaving_model_plausible_json_violating_d4_is_not_ok(mm):
    # Structurally perfect and plausible: probabilities peak at level 1 and sum to 1, but the score
    # claims level 2.4 -- a model that answers confidently and inconsistently (BC-84).
    bad = _mut(lambda b: b["answers"]["tone"].update(score=2.4, probabilities={"0": 0.0, "1": 0.9, "2": 0.1, "3": 0.0},
                                                      confidence=0.85))
    Wire(ok(bad)).install(mm)
    out = await tone(mm)
    assert out.status == "invalid" and out.reason == "schema_mismatch"
    assert out.answers["tone"].valid is False and out.answers["tone"].score is None


def test_one_bad_answer_does_not_invalidate_its_sibling():
    body = copy.deepcopy(TONE)
    body["answers"]["heavy"] = {"type": "noul", "noul": 7}
    answers, meta = dm.parse_response(body, [TONE_Q, HEAVY_Q], served_models=(SERVED,), provider=PROVIDER)
    assert answers["tone"].valid and not answers["heavy"].valid and meta["reason"] == "schema_mismatch"


def test_builders_validate_shapes():
    with pytest.raises(ValueError):
        score("q", "i", ["only one"])
    with pytest.raises(ValueError):
        score("q", "i", [str(n) for n in range(11)])
    with pytest.raises(ValueError):
        choice("q", "i", {"a": "x"})
    with pytest.raises(ValueError):
        dm.build_request("s", [TONE_Q, TONE_Q], model="m")
    assert "criteria" not in dm.build_request("s", [noul("n", "i")], model="m")["questions"]["n"]


# --- state size, switch, URL ----------------------------------------------------------------
async def test_state_one_over_limit_sends_nothing_and_at_limit_sends(mm):
    wire = Wire(ok(TONE)).install(mm)
    out = await tone(mm, "x" * 101)
    assert (out.status, out.reason) == ("unavailable", "state_too_large") and wire.requests == []
    assert (await tone(mm, "x" * 100)).status == "ok" and len(wire.requests) == 1


async def test_master_switch_off_is_disabled_and_sends_nothing(mm, monkeypatch):
    monkeypatch.setattr(app_config, "DECISION_MODEL_CFG", {"enabled": False, "roles": {"tone_arbiter": "shadow"}})
    wire = Wire(ok(TONE)).install(mm)
    out = await tone(mm)
    assert out.status == "disabled" and wire.requests == []


async def test_no_api_key_is_not_configured(mm):
    wire = Wire(ok(TONE)).install(mm)
    mm.api_key = None
    out = await tone(mm)
    assert (out.status, out.reason) == ("unavailable", "not_configured") and wire.requests == []


class _SpyManager(ModelManager):
    reads = 0

    @property
    def api_key(self):
        self.reads += 1
        return self._k

    @api_key.setter
    def api_key(self, v):
        self._k = v


@pytest.fixture
def spy(monkeypatch):
    monkeypatch.setattr(ModelManager, "_get_cached_embedder", staticmethod(lambda: None))
    m = _SpyManager(api_key="test-key")
    m.reads = 0
    return m


async def test_foreign_url_is_refused_before_the_credential_is_read(spy, monkeypatch):
    wire = Wire(ok(TONE)).install(spy)
    monkeypatch.setattr(dm, "SYSTEM_ONE_URL", "https://example.com/x")
    out = await tone(spy)
    assert (out.status, out.reason) == ("unavailable", "not_configured")
    assert wire.requests == [] and spy.reads == 0
    with pytest.raises(ValueError):  # the transport refuses on its own, too
        await T(spy.post_system_one("https://example.com/x", {}, timeout_s=1))
    assert spy.reads == 0


async def test_changed_base_url_is_refused_and_a_normal_call_does_read_the_key(spy):
    wire = Wire(ok(TONE)).install(spy)
    assert (await tone(spy)).status == "ok" and spy.reads > 0   # the spy can fail
    spy.base_url = "https://elsewhere.example/api/v1"
    spy.reads = 0
    out = await tone(spy)
    assert out.reason == "not_configured" and len(wire.requests) == 1 and spy.reads == 0


async def test_nothing_sensitive_is_logged(mm, caplog):
    caplog.set_level(logging.DEBUG)
    Wire(ok(TONE)).install(mm)
    await tone(mm, "very private words")
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "[DecisionModel]" in text and "very private words" not in text and "test-key" not in text


# --- cancellation, shadow runner -------------------------------------------------------------
async def _no_stragglers():
    await asyncio.sleep(0)
    return [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]


async def test_cancelled_request_raises_and_leaves_no_task(mm):
    started = asyncio.Event()

    async def hang(request):
        started.set()
        await asyncio.sleep(10)

    mm.system_one_transport = httpx.MockTransport(hang)
    task = asyncio.ensure_future(evaluate(mm, "x", [TONE_Q], role="tone_arbiter"))
    await T(started.wait())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await T(task)
    assert await _no_stragglers() == []


async def test_run_with_shadow_returns_both_and_isolates_a_failing_shadow():
    async def fast():
        return "llm"

    async def broken():
        raise RuntimeError("shadow failed")

    p, s = await T(run_with_shadow(fast(), broken()))
    assert (p.value, p.error) == ("llm", None) and isinstance(s.error, RuntimeError)


async def test_run_with_shadow_runs_concurrently():
    async def nap(v):
        await asyncio.sleep(0.2)
        return v

    t0 = time.monotonic()
    p, s = await T(run_with_shadow(nap(1), nap(2)))
    assert (p.value, s.value) == (1, 2) and time.monotonic() - t0 < 0.35


async def test_run_with_shadow_cancels_and_awaits_both_when_caller_cancelled():
    seen = []
    started = asyncio.Event()

    async def hang(name):
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            seen.append(name)
            raise

    task = asyncio.ensure_future(run_with_shadow(hang("p"), hang("s")))
    await T(started.wait())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await T(task)
    assert sorted(seen) == ["p", "s"] and await _no_stragglers() == []


# --- client lifecycle (D5) --------------------------------------------------------------------
async def test_reinitialize_clients_retires_the_client_and_uses_the_new_key(mm):
    wire = Wire(ok(TONE)).install(mm)
    await tone(mm)
    old = mm._system_one_client
    assert old is not None and mm.reinitialize_clients("new-key") is True
    assert mm._system_one_client is None
    await until(lambda: old.is_closed)
    await tone(mm)
    new = mm._system_one_client
    assert new is not old and not new.is_closed
    assert wire.requests[-1].headers["authorization"] == "Bearer new-key"


def test_a_client_from_another_loop_is_dropped_never_awaited(mm):
    Wire(ok(TONE)).install(mm)
    asyncio.run(tone(mm))
    first, first_loop = mm._system_one_client, mm._system_one_loop
    out = asyncio.run(tone(mm))
    assert out.status == "ok" and mm._system_one_client is not first and mm._system_one_loop is not first_loop


async def until(pred):
    async def spin():
        while not pred():
            await asyncio.sleep(0.01)
    await T(spin())


async def test_aclose_closes_the_transport(mm):
    Wire(ok(TONE)).install(mm)
    await tone(mm)
    client = mm._system_one_client
    await T(mm.aclose())
    assert client.is_closed and mm.system_one_transport.closed == 1 and mm._system_one_client is None


async def test_reinitialize_on_owner_loop_closes_the_transport(mm):
    Wire(ok(TONE)).install(mm)
    await tone(mm)
    assert mm.reinitialize_clients("new-key") is True
    await until(lambda: mm.system_one_transport.closed == 1)
    assert mm._system_one_closing == set() or all(t.done() for t in mm._system_one_closing)


async def test_sync_close_from_a_worker_thread_closes_on_the_owner_loop(mm):
    Wire(ok(TONE)).install(mm)
    await tone(mm)
    await T(asyncio.to_thread(mm.close))   # no running loop in the worker; owner loop keeps running here
    await until(lambda: mm.system_one_transport.closed == 1)
    assert mm._system_one_client is None and mm._system_one_loop is None


async def test_second_loop_recreating_the_client_closes_the_old_one_on_its_own_loop(mm):
    Wire(ok(TONE)).install(mm)
    loop_a = asyncio.new_event_loop()
    th = threading.Thread(target=loop_a.run_forever, daemon=True)
    th.start()
    try:
        first = await T(asyncio.wrap_future(asyncio.run_coroutine_threadsafe(tone(mm), loop_a)))
        old = mm._system_one_client
        assert first.status == "ok" and mm._system_one_loop is loop_a
        assert (await tone(mm)).status == "ok"      # a second loop (this one) posts
        await until(lambda: mm.system_one_transport.closed == 1)
        assert old.is_closed and mm._system_one_client is not old and not mm._system_one_client.is_closed
    finally:
        loop_a.call_soon_threadsafe(loop_a.stop)
        th.join(2)
        loop_a.close()


class FlakyTransport(RecTransport):
    """aclose() raises ``exc`` until it is cleared (the reviewer's failing-transport scenario)."""
    exc = None

    async def aclose(self):
        if self.exc is not None:
            raise self.exc
        await super().aclose()


def _two_retained(mm, first_exc):
    """Two clients, each owned by its own STOPPED-but-usable loop, both retained (a loop is running
    in the thread that retires them). Returns (flaky_first, second, loops)."""
    handler = Wire(ok(TONE))
    first, second = FlakyTransport(handler), RecTransport(handler)
    first.exc = first_exc
    loop_b, loop_c = asyncio.new_event_loop(), asyncio.new_event_loop()
    mm.system_one_transport = first
    assert loop_b.run_until_complete(tone(mm)).status == "ok"
    mm.system_one_transport = second
    assert loop_c.run_until_complete(tone(mm)).status == "ok"   # retires client 1 from inside loop_c
    assert len(mm._system_one_retained) == 1
    return first, second, (loop_b, loop_c)


def test_one_failing_close_does_not_lose_or_block_the_other_retained_clients(mm):
    first, second, loops = _two_retained(mm, OSError("boom"))
    try:
        mm.close()   # must not raise; client 2 closes; client 1 stays retained (its loop is usable)
        assert second.closed == 1 and first.closed == 0 and len(mm._system_one_retained) == 1
        first.exc = None   # the transport stops failing
        mm.close()
        assert first.closed == 1 and mm._system_one_retained == []
    finally:
        for lp in loops:
            lp.close()


def test_cancellation_propagates_and_every_unprocessed_pair_is_restored(mm):
    first, second, loops = _two_retained(mm, asyncio.CancelledError())
    try:
        with pytest.raises(asyncio.CancelledError):
            mm.close()
        assert second.closed == 0 and len(mm._system_one_retained) == 2   # nothing lost
        first.exc = None
        mm.close()
        assert first.closed == 1 and second.closed == 1 and mm._system_one_retained == []
    finally:
        for lp in loops:
            lp.close()


@pytest.mark.parametrize("stop_only,expected_closes", [(False, 0), (True, 1)])
def test_dead_owner_loop_is_dropped_but_a_stopped_usable_loop_is_closed(mm, stop_only, expected_closes):
    Wire(ok(TONE)).install(mm)
    loop = asyncio.new_event_loop()
    assert loop.run_until_complete(tone(mm)).status == "ok"
    if not stop_only:
        loop.close()
    mm.close()   # closed loop: dropped without raising; stopped-but-usable loop: closed before close() returns
    assert mm._system_one_client is None and mm.system_one_transport.closed == expected_closes
    assert mm._system_one_retained == []
    loop.close()


async def test_cross_loop_aclose_returns_only_after_the_slow_close_completed(mm):
    Wire(ok(TONE)).install(mm)
    mm.system_one_transport = RecTransport(mm.system_one_transport.handler, close_delay=0.2)
    loop_a = asyncio.new_event_loop()
    th = threading.Thread(target=loop_a.run_forever, daemon=True)
    th.start()
    try:
        await T(asyncio.wrap_future(asyncio.run_coroutine_threadsafe(tone(mm), loop_a)))
        await T(mm.aclose())   # this loop is not the owner
        assert mm.system_one_transport.closed == 1   # completed at return, not eventually
    finally:
        loop_a.call_soon_threadsafe(loop_a.stop)
        th.join(2)
        loop_a.close()


async def test_stopped_owner_while_another_loop_runs_is_retained_and_closed_on_the_next_call(mm):
    Wire(ok(TONE)).install(mm)
    loop_b = asyncio.new_event_loop()
    try:
        out = await T(asyncio.to_thread(lambda: loop_b.run_until_complete(tone(mm))))
        assert out.status == "ok" and not loop_b.is_running()
        mm.close()   # runs inside this loop (a different loop IS running here): cannot close, must retain
        assert mm.system_one_transport.closed == 0 and len(mm._system_one_retained) == 1
        await T(asyncio.to_thread(mm.close))   # next call, no loop in this thread: retried and completed
        assert mm.system_one_transport.closed == 1 and mm._system_one_retained == []
    finally:
        loop_b.close()


# --- second-referee findings (GPT-6 B2-G6-1, -3, -4) -----------------------------------------
HUGE = 10 ** 400  # a valid JSON integer too large for float()


@pytest.mark.parametrize("path", [["answers", "tone", "score"], ["answers", "tone", "confidence"],
                                  ["answers", "tone", "probabilities", "1"]])
async def test_oversized_integer_in_a_validated_field_is_invalid_not_a_raise(mm, path):
    Wire(ok(_mut(_set(path, HUGE)))).install(mm)
    out = await tone(mm)
    assert (out.status, out.reason) == ("invalid", "schema_mismatch") and out.answers["tone"].valid is False


async def test_oversized_integer_in_optional_cost_stays_ok_with_cost_none(mm):
    Wire(ok(_mut(_set(["usage", "cost"], HUGE)))).install(mm)
    out = await tone(mm)
    assert out.status == "ok" and out.cost_usd is None


async def test_deeply_nested_json_is_invalid_parse(mm):
    Wire(lambda: httpx.Response(200, text="[" * 2000 + "0" + "]" * 2000)).install(mm)
    out = await tone(mm)
    assert (out.status, out.reason) == ("invalid", "parse")


async def test_cancellation_still_propagates_through_the_response_boundary(mm, monkeypatch):
    def boom(*a, **k):
        raise asyncio.CancelledError()
    Wire(ok(TONE)).install(mm)
    monkeypatch.setattr(dm, "parse_response", boom)
    with pytest.raises(asyncio.CancelledError):
        await tone(mm)


def _http_date(delta):
    return format_datetime(datetime.now(timezone.utc) + delta, usegmt=True)


async def test_future_http_date_retry_after_is_honoured_no_retry(mm):
    wire = Wire(status(429, headers={"Retry-After": _http_date(timedelta(hours=1))}), ok(TONE)).install(mm)
    out = await tone(mm)
    assert (out.status, out.reason, out.retried) == ("unavailable", "http_429", False) and len(wire.requests) == 1


async def test_expired_http_date_retry_after_retries(mm):
    wire = Wire(status(429, headers={"Retry-After": _http_date(-timedelta(hours=1))}), ok(TONE)).install(mm)
    out = await tone(mm)
    assert out.status == "ok" and out.retried is True and len(wire.requests) == 2


async def test_garbage_retry_after_falls_back_to_default_backoff(mm):
    wire = Wire(status(529, headers={"Retry-After": "soonish"}), ok(TONE)).install(mm)
    assert (await tone(mm)).status == "ok" and len(wire.requests) == 2


async def test_transport_pins_the_fixed_endpoint_even_when_base_url_is_changed(spy):
    wire = Wire(ok(TONE)).install(spy)
    spy.base_url = "https://example.invalid/api/v1"
    spy.reads = 0
    with pytest.raises(ValueError):
        await T(spy.post_system_one(spy.base_url + "/systemone", {}, timeout_s=1))
    assert spy.reads == 0 and wire.requests == []


# --- GPT-6 round 2: B3-G6-1 (no rejected raw string survives in the outcome) ---------------
SYNTH_MSG = "zebra quokka lantern vortex marmalade sundial"
BIG = ("quartz-lantern " * 400)[:5000]


def _has_six_run(text, hay):
    return any(text[i:i + 6] in hay for i in range(len(text) - 5))


@pytest.mark.parametrize("field", ["model", "provider"])
@pytest.mark.parametrize("echo", [SYNTH_MSG, BIG])
async def test_rejected_model_or_provider_text_never_survives_in_the_outcome(mm, field, echo):
    Wire(ok(_mut(_set([field], echo)))).install(mm)
    out = await tone(mm, SYNTH_MSG)
    assert (out.status, out.reason) == ("invalid", "model_mismatch")
    assert not _has_six_run(echo, repr(out))
    assert out.served_model is None and out.provider is None


async def test_accepted_values_are_still_reported(mm):
    Wire(ok(TONE)).install(mm)
    out = await tone(mm)
    assert (out.served_model, out.provider) == (SERVED, PROVIDER)


# --- D12 --------------------------------------------------------------------------------------
def test_jev_is_not_a_chat_model_and_systemone_lives_in_two_places():
    slugs = ("jev", "typesafe")
    for table in (API_MODEL_ALIASES, MODEL_CAPABILITIES):
        assert not [k for k in table if any(s in str(k).lower() for s in slugs)]
    assert not [v for v in API_MODEL_ALIASES.values() if any(s in str(v).lower() for s in slugs)]
    hits = set()
    for d in ("core", "utils", "models", "memory", "processing", "knowledge", "api", "gui", "config"):
        for p in (REPO / d).rglob("*.py"):
            if "__pycache__" not in p.parts and "systemone" in p.read_text(errors="ignore").lower():
                hits.add(p.relative_to(REPO).as_posix())
    assert hits == {"models/decision_model.py", "models/model_manager.py"}


def test_nan_is_not_finite_helper():
    assert not dm._num(float("nan"), 0, 1) and not dm._num(True, 0, 1) and dm._num(math.pi - 3, 0, 1)
