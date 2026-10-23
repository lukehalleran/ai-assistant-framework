"""
scripts/model_bakeoff.py — plan/estimate arithmetic, spend cap, 402 stop, balance
refusal, resume, report aggregation (2026-10-23). No network, no models: fake
build/generator/judge/transport drive THE deployed phase functions.

class: BC-89-adjacent cost-safety (a paid script must be unable to overspend); BC-10
(candidate models are reachable through the registry aliases).
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

import scripts.model_bakeoff as bo

PRICES = {  # usd per TOKEN
    "moonshotai/kimi-k3": (0.24e-6, 10.8e-6),
    "openai/gpt-6.1-sol": (2e-6, 10e-6),
    "anthropic/claude-opus-4.8": (5e-6, 25e-6),
}
QUERIES = [{"query_id": f"q{i}", "query_text": f"question {i}", "intent": "x"} for i in range(3)]


def run(coro):
    return asyncio.run(coro)


# --- plan / estimate -------------------------------------------------------

def test_catalog_prices_parses_per_token_strings_and_skips_unpriced():
    cat = {"data": [{"id": "a/b", "pricing": {"prompt": "0.000002", "completion": "0.00001"}},
                    {"id": "c/d", "pricing": {"prompt": "-1"}}, {"id": "e/f"}]}
    out = bo.catalog_prices(cat)
    assert out["a/b"] == (2e-6, 1e-5) and "e/f" not in out and "c/d" not in out


def test_plan_arithmetic_with_default_tokens():
    plan = bo.build_plan(9, ["kimi-3", "gpt-6.1"], "claude-opus-4.8", PRICES)
    gen_kimi = 9 * (16_000 * 0.24e-6 + 600 * 10.8e-6)
    gen_gpt = 9 * (16_000 * 2e-6 + 600 * 10e-6)
    judge = 9 * 1 * (4_000 * 5e-6 + 400 * 25e-6)
    assert plan["generations"] == 18 and plan["judgments"] == 9
    assert plan["estimate_usd"] == pytest.approx(gen_kimi + gen_gpt + judge)


def test_plan_uses_measured_tokens_from_a_prior_run(tmp_path):
    (tmp_path / "generations.jsonl").write_text(json.dumps(
        {"model": "kimi-3", "prompt_tokens": 10_000, "response_tokens": 200, "api_error": False}) + "\n")
    measured = bo.measured_tokens(tmp_path)
    plan = bo.build_plan(2, ["kimi-3"], "claude-opus-4.8", PRICES, measured)
    assert plan["tokens_measured"] is True
    assert plan["generation_usd"]["kimi-3"] == pytest.approx(2 * (10_000 * 0.24e-6 + 200 * 10.8e-6))


def test_unpriced_model_is_refused():
    with pytest.raises(bo.StopRun) as e:
        bo.build_plan(9, ["kimi-3", "mystery-model"], "claude-opus-4.8", PRICES)
    assert e.value.reason == "no_price"


def test_select_queries_is_deterministic_per_intent():
    corpus = [{"query_id": "b2", "intent": "b"}, {"query_id": "a2", "intent": "a"},
              {"query_id": "a1", "intent": "a"}, {"query_id": "b1", "intent": "b"}]
    assert [q["query_id"] for q in bo.select_queries(corpus, 1)] == ["a1", "b1"]
    assert [q["query_id"] for q in bo.select_queries(corpus, 2)] == ["a1", "a2", "b1", "b2"]


def test_default_selection_is_nine_queries_one_per_intent():
    from eval.corpus import SEED_CORPUS
    assert len(bo.select_queries(SEED_CORPUS, 1)) == 9


# --- balance ---------------------------------------------------------------

def test_balance_refusal_needs_estimate_plus_margin():
    with pytest.raises(bo.StopRun) as e:
        bo.check_balance(2.30, 1.40)  # 2.30 < 1.40 + 1.00
    assert e.value.reason == "balance"
    bo.check_balance(2.41, 1.40)


def test_fetch_balance_uses_bearer_header_and_never_returns_the_key():
    seen = {}

    def http_get(url, headers):
        seen.update(url=url, headers=headers)
        return {"data": {"total_credits": 10.0, "total_usage": 3.5}}

    assert bo.fetch_balance("sk-fake", http_get) == pytest.approx(6.5)
    assert seen["url"].endswith("/api/v1/credits") and seen["headers"]["Authorization"] == "Bearer sk-fake"


# --- phases ----------------------------------------------------------------

class FakeGen:
    """EvalGenerator stand-in + MeteredGenerator surface (.last)."""

    def __init__(self, responses=None, cost=0.10, ms=100):
        self.calls, self.last, self.cost, self.ms = [], None, cost, ms
        self.responses = responses or {}

    async def generate(self, **kw):
        self.calls.append(kw["model"])
        text = self.responses.get(kw["model"], f"answer from {kw['model']}")
        self.last = {"prompt_tokens": 1000, "response_tokens": 100, "cost_usd": self.cost, "from_usage": True}
        return SimpleNamespace(response_text=text, generation_time_ms=self.ms)


async def _build(text):
    return f"PROMPT[{text}]", "SYSTEM"


def test_max_usd_stops_cleanly_before_exceeding_and_resume_finishes(tmp_path):
    gen = FakeGen(cost=0.10)
    spend = bo.Spend(0.25)
    with pytest.raises(bo.StopRun) as e:
        run(bo.generate_phase(tmp_path, QUERIES, ["kimi-3", "gpt-6.1"], _build, gen, spend, PRICES, 10_000))
    assert e.value.reason == "max_usd" and spend.total <= 0.25
    done = bo._bx._load_jsonl(tmp_path / "generations.jsonl")
    assert len(done) == len(gen.calls) == 2          # nothing half-written
    # resume with room: only the missing 4 of 6 generations run
    gen2 = FakeGen(cost=0.10)
    run(bo.generate_phase(tmp_path, QUERIES, ["kimi-3", "gpt-6.1"], _build, gen2,
                          bo.Spend(5.0, already=bo.spent_so_far(tmp_path)), PRICES, 64))
    assert len(gen2.calls) == 4
    assert len(bo._bx._load_jsonl(tmp_path / "generations.jsonl")) == 6


def test_worst_case_precheck_blocks_a_call_that_could_overshoot(tmp_path):
    # max_tokens=100000 on gpt-6.1: worst case $1.00+ > cap even though spend is 0
    gen = FakeGen()
    with pytest.raises(bo.StopRun) as e:
        run(bo.generate_phase(tmp_path, QUERIES, ["gpt-6.1"], _build, gen, bo.Spend(0.50), PRICES, 100_000))
    assert e.value.reason == "max_usd" and gen.calls == []


def test_402_text_stops_immediately_and_is_not_recorded(tmp_path):
    gen = FakeGen(responses={"gpt-6.1": "[CREDITS EXHAUSTED] The provider reports insufficient credits"})
    with pytest.raises(bo.StopRun) as e:
        run(bo.generate_phase(tmp_path, QUERIES, ["kimi-3", "gpt-6.1"], _build, gen, bo.Spend(9.0), PRICES, 64))
    assert e.value.reason == "http_402"
    recorded = bo._bx._load_jsonl(tmp_path / "generations.jsonl")
    assert [r["model"] for r in recorded] == ["kimi-3"] and len(gen.calls) == 2


def test_prompt_is_built_once_per_query_for_all_models(tmp_path):
    built = []

    async def build(text):
        built.append(text)
        return await _build(text)

    run(bo.generate_phase(tmp_path, QUERIES[:2], ["kimi-3", "gpt-6.1"], build, FakeGen(), bo.Spend(9), PRICES, 64))
    assert built == ["question 0", "question 1"]


def test_resume_skips_queries_fully_done_without_rebuilding(tmp_path):
    run(bo.generate_phase(tmp_path, QUERIES, ["kimi-3"], _build, FakeGen(), bo.Spend(9), PRICES, 64))
    built = []

    async def build(text):
        built.append(text)
        return await _build(text)

    gen = FakeGen()
    run(bo.generate_phase(tmp_path, QUERIES, ["kimi-3"], build, gen, bo.Spend(9), PRICES, 64))
    assert built == [] and gen.calls == []


class FakeJudge:
    def __init__(self, gen, winner_is_baseline=False, cost=0.012):
        self.gen, self.calls, self.win, self.cost = gen, [], winner_is_baseline, cost

    async def judge_pair(self, **kw):
        self.calls.append((kw["snapshot_id"], kw["variant_id"]))
        self.gen.last = {"prompt_tokens": 500, "response_tokens": 50, "cost_usd": self.cost, "from_usage": True}
        from eval.judge import JudgeVerdict
        return JudgeVerdict(
            snapshot_id=kw["snapshot_id"], variant_id=kw["variant_id"], strategy=kw["strategy"],
            sections_removed=[], sections_added=[], query_text=kw["query"], winner="A",
            winner_is_baseline=self.win, confidence=0.9, explanation="x",
            scores_a={c: 4 for c in bo.CRITERIA}, scores_b={c: 3 for c in bo.CRITERIA},
            baseline_position="B" if not self.win else "A", judge_model="j", judge_time_ms=1,
            timestamp="t", raw_judge_output="")


def test_judge_phase_skips_done_and_respects_cap(tmp_path):
    models = ["kimi-3", "gpt-6.1"]
    run(bo.generate_phase(tmp_path, QUERIES, models, _build, FakeGen(cost=0.0), bo.Spend(9), PRICES, 64))
    jg = FakeGen()
    judge = FakeJudge(jg)
    spend = bo.Spend(0.03)
    with pytest.raises(bo.StopRun) as e:
        run(bo.judge_phase(tmp_path, QUERIES, models, judge, jg, spend, PRICES, "claude-opus-4.8"))
    # each judgment really costs 0.012 and its worst case is ~0.016: two fit, a third could not
    assert e.value.reason == "max_usd" and len(judge.calls) == 2 and spend.total <= 0.03
    judge2 = FakeJudge(jg)
    run(bo.judge_phase(tmp_path, QUERIES, models, judge2, jg, bo.Spend(9), PRICES, "claude-opus-4.8"))
    assert len(judge2.calls) == 1  # only the one not yet judged


def test_api_error_generations_are_not_judged(tmp_path):
    gen = FakeGen(responses={"gpt-6.1": "[API Error] boom"})
    models = ["kimi-3", "gpt-6.1"]
    run(bo.generate_phase(tmp_path, QUERIES[:1], models, _build, gen, bo.Spend(9), PRICES, 64))
    judge = FakeJudge(gen)
    run(bo.judge_phase(tmp_path, QUERIES[:1], models, judge, gen, bo.Spend(9), PRICES, "claude-opus-4.8"))
    assert judge.calls == []


# --- metering --------------------------------------------------------------

def test_metered_generator_prefers_billed_cost_then_usage_then_estimate():
    class Inner:
        async def generate(self, **kw):
            return SimpleNamespace(response_text="r", prompt_token_count=4000, response_token_count=40,
                                   generation_time_ms=1)

    tap = bo.UsageTap()
    mg = bo.MeteredGenerator(Inner(), tap, PRICES)
    run(mg.generate(model="gpt-6.1"))                       # no usage: char estimate x catalog price
    assert mg.last["cost_usd"] == pytest.approx(4000 * 2e-6 + 40 * 10e-6) and not mg.last["from_usage"]

    class TapInner(Inner):
        async def generate(self, **kw):
            tap.last = {"prompt_tokens": 1000, "completion_tokens": 100}
            return await super().generate(**kw)

    mg2 = bo.MeteredGenerator(TapInner(), tap, PRICES)
    run(mg2.generate(model="gpt-6.1"))                       # usage tokens x price
    assert mg2.last["cost_usd"] == pytest.approx(1000 * 2e-6 + 100 * 10e-6) and mg2.last["from_usage"]

    class CostInner(Inner):
        async def generate(self, **kw):
            tap.last = {"prompt_tokens": 1000, "completion_tokens": 100, "cost": 0.123}
            return await super().generate(**kw)

    mg3 = bo.MeteredGenerator(CostInner(), tap, PRICES)
    run(mg3.generate(model="gpt-6.1"))                       # billed cost wins
    assert mg3.last["cost_usd"] == pytest.approx(0.123)


def test_usage_tap_observes_without_replacing_the_original():
    seen = []
    mm = SimpleNamespace(_log_cache_usage=lambda usage, **kw: seen.append(usage))
    tap = bo.UsageTap(mm)
    mm._log_cache_usage({"prompt_tokens": 7}, where="t")
    assert tap.last == {"prompt_tokens": 7} and seen == [{"prompt_tokens": 7}]


# --- report ----------------------------------------------------------------

def test_report_aggregation(tmp_path):
    models = ["kimi-3", "gpt-6.1"]
    gens = [{"model": m, "generation_ms": ms, "cost_usd": c, "api_error": False}
            for m, ms, c in [("kimi-3", 100, 0.01), ("kimi-3", 300, 0.03), ("gpt-6.1", 500, 0.10)]]
    base = {c: 3 for c in bo.CRITERIA}
    better = {c: 4 for c in bo.CRITERIA}
    judgments = [
        {"variant_id": "gpt-6.1", "winner_is_baseline": False, "baseline_position": "A",
         "scores_a": base, "scores_b": better},
        {"variant_id": "gpt-6.1", "winner_is_baseline": True, "baseline_position": "B",
         "scores_a": {c: 3 for c in bo.CRITERIA}, "scores_b": {c: 2 for c in bo.CRITERIA}},
        {"variant_id": "gpt-6.1", "winner_is_baseline": None, "baseline_position": "A",
         "scores_a": base, "scores_b": base},
    ]
    report = bo.build_report(gens, judgments, models, "claude-opus-4.8")
    assert "| kimi-3 | 2 | 200 | $0.0200 | $1.00 | 0 |" in report      # median 200ms, mean $0.02, x50
    assert "| gpt-6.1 | 1 | 500 | $0.1000 | $5.00 | 0 |" in report
    assert "candidate wins 1 | ties 1 | candidate losses 1" in report
    # deltas: +1, +1 (baseline in B: cand scores_a=3 vs base 2), 0 -> mean 0.667/criterion x5
    assert "mean score delta (candidate - baseline, out of 25): +3.33" in report


def test_dry_run_never_spends_or_boots(monkeypatch, capsys):
    monkeypatch.setattr(bo, "_fetch_catalog", lambda: {"data": [
        {"id": s, "pricing": {"prompt": str(p[0]), "completion": str(p[1])}} for s, p in PRICES.items()]})
    monkeypatch.setattr(bo, "_run", lambda *a, **k: pytest.fail("dry run must not execute"))
    monkeypatch.setattr(bo._bx, "_daemon_running", lambda: pytest.fail("dry run must not probe"))
    assert bo.main(["--models", "kimi-3", "gpt-6.1"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "ESTIMATE" in out and "9 queries x 2 models" in out
