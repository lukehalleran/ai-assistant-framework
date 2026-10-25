"""
scripts/model_bakeoff.py: .env loading in `_run` and the deployed token cap (incident 2026-10-25).

* B1: `--run` failed "OPENROUTER_API_KEY is not set" because the key check ran before
  `from main import build_orchestrator` (the only place .env was loaded). `_run` now loads
  the project's one .env authority first; importing the script never touches os.environ.
* B2: `--max-tokens` defaulted to 1024 while Daemon answers at models.default_max_tokens, so
  reasoning models spent the cap thinking and EMPTY replies were scored as judged losses.
  Empty replies are now their own count, never judged.

class: BC-83 (one .env authority), BC-89 (small-budget call on a reasoning model).
"""
import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

import scripts.model_bakeoff as bo
from tests.unit.test_model_bakeoff_cost_safety import FakeGen, FakeJudge, PRICES, QUERIES, _build


def run(coro):
    return asyncio.run(coro)


# --- B1 --------------------------------------------------------------------

def test_run_preflight_sees_key_from_project_env(tmp_path, monkeypatch, capsys):
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=sk-invented-0000\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder")  # registers teardown restore ...
    monkeypatch.delenv("OPENROUTER_API_KEY")                  # ... to "unset" after load_dotenv writes it
    monkeypatch.setattr("utils.bootstrap.resolve_env_path", lambda: env)
    seen = {}

    def fake_balance(key, http_get):
        seen["key"] = key
        raise bo.StopRun("balance", "stop here (test)")

    monkeypatch.setattr(bo, "fetch_balance", fake_balance)
    args = SimpleNamespace(models=["kimi-3"], max_tokens=64, max_usd=1.0, judge_model="j")
    code = run(bo._run(args, tmp_path, [], PRICES, {"estimate_usd": 0.5}))
    out = capsys.readouterr().out
    assert "is not set" not in out and seen["key"] == "sk-invented-0000" and code == 2


def test_run_without_any_key_still_refuses(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("utils.bootstrap.resolve_env_path", lambda: tmp_path / "missing.env")
    args = SimpleNamespace(models=["kimi-3"], max_tokens=64, max_usd=1.0, judge_model="j")
    assert run(bo._run(args, tmp_path, [], PRICES, {"estimate_usd": 0.5})) == 2
    assert "OPENROUTER_API_KEY is not set" in capsys.readouterr().out


def test_import_alone_leaves_environ_unchanged_even_with_a_dotenv(tmp_path):
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-invented-1111\n")
    code = (
        "import os, sys; sys.path.insert(0, '.'); os.environ.pop('OPENROUTER_API_KEY', None); "
        "before = dict(os.environ); import scripts.model_bakeoff; "
        "assert dict(os.environ) == before, set(os.environ) ^ set(before)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]


# --- B2 --------------------------------------------------------------------

def test_max_tokens_default_is_the_deployed_constant_read_at_call_time(monkeypatch):
    import config.app_config as ac
    monkeypatch.setattr(ac, "DEFAULT_MAX_TOKENS", 4321)
    assert bo.deployed_max_tokens() == 4321
    monkeypatch.setattr(ac, "DEFAULT_MAX_TOKENS", 77)
    assert bo.deployed_max_tokens() == 77


def test_main_passes_deployed_cap_unless_overridden(monkeypatch, tmp_path):
    import config.app_config as ac
    monkeypatch.setattr(ac, "DEFAULT_MAX_TOKENS", 4321)
    monkeypatch.setattr(bo, "_fetch_catalog", lambda: {"data": [
        {"id": s, "pricing": {"prompt": str(p[0]), "completion": str(p[1])}} for s, p in PRICES.items()]})
    got = {}

    async def fake_run(args, *a, **k):
        got["mt"] = args.max_tokens
        return 0

    monkeypatch.setattr(bo, "_run", fake_run)
    monkeypatch.setattr(bo._bx, "_daemon_running", lambda: False)
    monkeypatch.setattr(bo.sys, "argv", ["x"])
    out = str(tmp_path)
    assert bo.main(["--run", "--models", "kimi-3", "--output-dir", out]) == 0 and got["mt"] == 4321
    assert bo.main(["--run", "--models", "kimi-3", "--output-dir", out, "--max-tokens", "512"]) == 0
    assert got["mt"] == 512


MODELS = ["kimi-3", "gpt-6.1"]


def test_empty_reply_is_flagged_not_judged_and_not_a_loss(tmp_path):
    gen = FakeGen(responses={"gpt-6.1": "   \n"})  # misbehaving model: spends the cap thinking, no text
    run(bo.generate_phase(tmp_path, QUERIES, MODELS, _build, gen, bo.Spend(9), PRICES, 64))
    gens = bo._bx._load_jsonl(tmp_path / "generations.jsonl")
    assert all(g["empty"] for g in gens if g["model"] == "gpt-6.1")
    assert not any(g["empty"] for g in gens if g["model"] == "kimi-3")
    judge = FakeJudge(gen)
    run(bo.judge_phase(tmp_path, QUERIES, MODELS, judge, gen, bo.Spend(9), PRICES, "claude-opus-4.8"))
    assert judge.calls == []                                   # nothing reached the judge
    assert bo._bx._load_jsonl(tmp_path / "judgments.jsonl") == []


def test_empty_baseline_is_not_judged_either(tmp_path):
    gen = FakeGen(responses={"kimi-3": ""})
    run(bo.generate_phase(tmp_path, QUERIES[:1], MODELS, _build, gen, bo.Spend(9), PRICES, 64))
    judge = FakeJudge(gen)
    run(bo.judge_phase(tmp_path, QUERIES[:1], MODELS, judge, gen, bo.Spend(9), PRICES, "claude-opus-4.8"))
    assert judge.calls == []


def test_report_and_json_count_empty_separately(tmp_path):
    gen = FakeGen(responses={"gpt-6.1": ""})
    # query 0: both answer; make q1/q2 candidate empty -> one judged pair needs a non-empty candidate
    run(bo.generate_phase(tmp_path, QUERIES[:1], MODELS, _build, FakeGen(), bo.Spend(9), PRICES, 64))
    run(bo.generate_phase(tmp_path, QUERIES[1:], MODELS, _build, gen, bo.Spend(9), PRICES, 64))
    jg = FakeGen()
    run(bo.judge_phase(tmp_path, QUERIES, MODELS, FakeJudge(jg, winner_is_baseline=True), jg,
                       bo.Spend(9), PRICES, "claude-opus-4.8"))
    report = bo.write_report(tmp_path, MODELS, "claude-opus-4.8")
    assert "| gpt-6.1 | 3 |" in report and "| 0 | 2 |" in report            # api errors 0, empty 2
    assert "candidate wins 0 | ties 0 | candidate losses 1" in report      # only the one real pair
    assert "empty replies (not judged, not losses): 2" in report
    summary = json.loads((tmp_path / "report.json").read_text())["models"]["gpt-6.1"]
    assert summary["empty"] == 2 and summary["losses"] == 1 and summary["wins"] == 0
    assert summary["generations"] == 3


def test_legacy_rows_without_empty_key_are_derived_from_text():
    gens = [{"model": "kimi-3", "generation_ms": 1, "cost_usd": 0.0, "api_error": False, "response": ""},
            {"model": "kimi-3", "generation_ms": 1, "cost_usd": 0.0, "api_error": False, "response": "hi"}]
    assert bo.summarize(gens, [], ["kimi-3"])["kimi-3"]["empty"] == 1
