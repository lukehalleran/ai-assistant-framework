"""Blind pairwise bake-off of daily-driver model candidates on the deployed prompt path.

Question answered: is candidate model X at least as good as the current daily
driver on the prompts Daemon really builds, and what does each turn cost?

Method (reuses scripts/budget_experiment.py's machinery — nothing forked):
  1. Pick queries from the eval seed corpus (`--per-intent N`, deterministic).
  2. Build each prompt ONCE through THE deployed orchestrator.prepare_prompt() at
     the deployed token budget (no budget is set here), web search and proactive
     surfacing off, inside eval.persistence_guard.
  3. Generate with every `--models` entry (FIRST = BASELINE, default kimi-3) via
     eval.no_store_generation.EvalGenerator (the deployed generate_once).
  4. Judge baseline vs each candidate with eval.judge.PairwiseJudge
     (position-randomized, 5 criteria, `--judge-model`).
  5. Report win/tie/loss, mean score delta (out of 25), median latency and the
     MEASURED $/turn and $/50 turns per model.

COST SAFETY (every rule is enforced by a test):
  * Dry run is the default: it prints the plan and a price estimate from the
    LIVE OpenRouter catalog (free GET). Spending needs `--run`.
  * `--max-usd` (default 2.00): actual spend (response usage x catalog price) is
    tracked across the whole run dir; before each call the worst-case cost of
    THAT call is added, and the run stops cleanly (resumable) rather than exceed it.
  * `--run` first reads the OpenRouter balance (GET /api/v1/credits, key from the
    environment, never printed) and refuses if balance < estimate + 1.00.
  * A model with no catalog price is refused (the cap could not be enforced).
  * HTTP 402 / "[CREDITS EXHAUSTED]" stops the run immediately (resumable).
  * Refuses to run while a live Daemon is running (shared ChromaDB).

    python scripts/model_bakeoff.py --models kimi-3 gpt-6.1 sonnet-5.5          # plan only
    python scripts/model_bakeoff.py --run --models kimi-3 kimi-k3.1-pro
    python scripts/model_bakeoff.py --run --resume eval/runs/model_bakeoff_<ts> --models ...
Run under a memory cap:  systemd-run --user --scope -p MemoryMax=9G python scripts/model_bakeoff.py --run ...
"""

import os
import sys

# MUST precede any repo import — app_config reads env at import time.
os.environ["WEB_SEARCH_ENABLED"] = "0"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import asyncio
import json
import statistics
import urllib.request
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from models.model_manager import API_MODEL_ALIASES
from scripts import budget_experiment as _bx

CATALOG_URL = "https://openrouter.ai/api/v1/models"
CREDITS_URL = "https://openrouter.ai/api/v1/credits"

# Token assumptions for an estimate when no earlier run measured them.
DEFAULT_GEN_IN, DEFAULT_GEN_OUT = 16_000, 600
DEFAULT_JUDGE_IN, DEFAULT_JUDGE_OUT = 4_000, 400
BALANCE_MARGIN_USD = 1.00
JUDGE_MAX_TOKENS = 512  # eval.judge.PairwiseJudge default
CRITERIA = ["accuracy", "helpfulness", "conciseness", "tone", "grounding"]


class StopRun(Exception):
    """A clean, resumable stop (spend cap, 402, ...). `.reason` says which."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# ---------------------------------------------------------------------------
# Pure helpers (plan / estimate / report) — no network, no models
# ---------------------------------------------------------------------------

def resolve_slug(name: str) -> str:
    return API_MODEL_ALIASES.get(name, name)


def select_queries(corpus, per_intent: int) -> list:
    """Deterministic: intents alphabetical, queries by id within each intent, first N."""
    by_intent = {}
    for q in corpus:
        by_intent.setdefault(str(q.get("intent", "")), []).append(q)
    picked = []
    for intent in sorted(by_intent):
        picked.extend(sorted(by_intent[intent], key=lambda q: q["query_id"])[:per_intent])
    return picked


def catalog_prices(catalog: dict) -> dict:
    """slug -> (usd_per_input_token, usd_per_output_token) from a /models payload.

    OpenRouter publishes pricing as per-TOKEN decimal strings. A model with a
    missing/unparseable price is omitted (the caller refuses to spend on it).
    """
    out = {}
    for m in (catalog or {}).get("data", []):
        pricing = m.get("pricing") or {}
        try:
            out[m["id"]] = (float(pricing["prompt"]), float(pricing["completion"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def call_cost(prices: dict, slug: str, prompt_tokens: int, completion_tokens: int) -> float:
    p_in, p_out = prices[slug]
    return prompt_tokens * p_in + completion_tokens * p_out


def measured_tokens(run_dir) -> dict:
    """Mean tokens per call from an earlier run's jsonl files (empty when none)."""
    out = {}
    if not run_dir:
        return out
    gens = _bx._load_jsonl(Path(run_dir) / "generations.jsonl")
    for model in {g["model"] for g in gens}:
        rows = [g for g in gens if g["model"] == model and not g.get("api_error")]
        if rows:
            out[model] = (statistics.mean(r["prompt_tokens"] for r in rows),
                          statistics.mean(r["response_tokens"] for r in rows))
    judges = _bx._load_jsonl(Path(run_dir) / "judgments.jsonl")
    rows = [j for j in judges if "prompt_tokens" in j]
    if rows:
        out["__judge__"] = (statistics.mean(r["prompt_tokens"] for r in rows),
                            statistics.mean(r["response_tokens"] for r in rows))
    return out


def build_plan(n_queries: int, models: list, judge_model: str, prices: dict,
               measured: dict = None) -> dict:
    """The dry-run plan: call counts and a USD estimate. Raises if a price is missing."""
    measured = measured or {}
    missing = [m for m in list(models) + [judge_model] if resolve_slug(m) not in prices]
    if missing:
        raise StopRun("no_price", f"no catalog price for {missing}; the --max-usd cap could not be enforced")
    per_model = {}
    for m in models:
        t_in, t_out = measured.get(m, (DEFAULT_GEN_IN, DEFAULT_GEN_OUT))
        per_model[m] = n_queries * call_cost(prices, resolve_slug(m), t_in, t_out)
    j_in, j_out = measured.get("__judge__", (DEFAULT_JUDGE_IN, DEFAULT_JUDGE_OUT))
    n_judgments = n_queries * max(0, len(models) - 1)
    judge_usd = n_judgments * call_cost(prices, resolve_slug(judge_model), j_in, j_out)
    return {
        "queries": n_queries,
        "models": list(models),
        "judge_model": judge_model,
        "generations": n_queries * len(models),
        "judgments": n_judgments,
        "generation_usd": per_model,
        "judge_usd": judge_usd,
        "estimate_usd": sum(per_model.values()) + judge_usd,
        "tokens_measured": bool(measured),
    }


def format_plan(plan: dict, max_usd: float) -> str:
    lines = [
        f"PLAN  {plan['queries']} queries x {len(plan['models'])} models "
        f"(baseline = {plan['models'][0]}) -> {plan['generations']} generations, "
        f"{plan['judgments']} judgments by {plan['judge_model']}",
        "      tokens: " + ("measured from the earlier run" if plan["tokens_measured"] else
                           f"assumed {DEFAULT_GEN_IN}/{DEFAULT_GEN_OUT} per generation, "
                           f"{DEFAULT_JUDGE_IN}/{DEFAULT_JUDGE_OUT} per judgment"),
    ]
    for m, usd in plan["generation_usd"].items():
        lines.append(f"      generation {m:<24} ~${usd:.3f}")
    lines.append(f"      judging    {plan['judge_model']:<24} ~${plan['judge_usd']:.3f}")
    lines.append(f"      ESTIMATE ~${plan['estimate_usd']:.3f}   cap --max-usd ${max_usd:.2f}"
                 + ("   ** estimate exceeds the cap; the run will stop early (resumable) **"
                    if plan["estimate_usd"] > max_usd else ""))
    return "\n".join(lines)


def check_balance(balance_usd: float, estimate_usd: float) -> None:
    if balance_usd < estimate_usd + BALANCE_MARGIN_USD:
        raise StopRun(
            "balance",
            f"OpenRouter balance ${balance_usd:.2f} < estimate ${estimate_usd:.2f} + ${BALANCE_MARGIN_USD:.2f}")


def fetch_balance(api_key: str, http_get) -> float:
    """OpenRouter credits remaining. `http_get(url, headers) -> dict` is injectable."""
    data = http_get(CREDITS_URL, {"Authorization": f"Bearer {api_key}"}).get("data", {})
    return float(data["total_credits"]) - float(data["total_usage"])


def _urllib_get(url: str, headers: dict) -> dict:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_report(generations: list, judgments: list, models: list, judge_model: str) -> str:
    baseline = models[0]
    lines = ["# Model Bake-off Report", "",
             f"Generated: {datetime.now().isoformat(timespec='seconds')}",
             f"Baseline: {baseline} | judge: {judge_model}", "",
             "## Generation cost and latency", "",
             "| model | n | median gen ms | $/turn | $/50 turns | api errors |",
             "|---|---|---|---|---|---|"]
    for m in models:
        rs = [g for g in generations if g["model"] == m]
        clean = [g for g in rs if not g.get("api_error")]
        if not rs:
            continue
        med = statistics.median(g["generation_ms"] for g in clean) if clean else 0
        per_turn = statistics.mean(g["cost_usd"] for g in clean) if clean else 0.0
        lines.append(f"| {m} | {len(rs)} | {med:.0f} | ${per_turn:.4f} | ${per_turn * 50:.2f} "
                     f"| {len(rs) - len(clean)} |")
    lines += ["", "## Quality vs baseline (position-randomized pairwise judge)"]
    for m in models[1:]:
        vs = [j for j in judgments if j["variant_id"] == m]
        if not vs:
            lines += ["", f"### {m}: no judgments yet"]
            continue
        wins = sum(1 for v in vs if v["winner_is_baseline"] is False)
        losses = sum(1 for v in vs if v["winner_is_baseline"] is True)
        ties = len(vs) - wins - losses
        deltas = {c: [] for c in CRITERIA}
        for v in vs:
            base = v["scores_a"] if v["baseline_position"] == "A" else v["scores_b"]
            cand = v["scores_b"] if v["baseline_position"] == "A" else v["scores_a"]
            for c in CRITERIA:
                if c in base and c in cand:
                    deltas[c].append(cand[c] - base[c])
        mean_d = {c: (sum(d) / len(d) if d else 0.0) for c, d in deltas.items()}
        lines += ["", f"### {m} vs {baseline} (n={len(vs)})", "",
                  f"- candidate wins {wins} | ties {ties} | candidate losses {losses}",
                  f"- mean score delta (candidate - baseline, out of 25): {sum(mean_d.values()):+.2f}",
                  "- per criterion: " + ", ".join(f"{c} {mean_d[c]:+.2f}" for c in CRITERIA)]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Metering: real usage from the response, priced from the catalog
# ---------------------------------------------------------------------------

class UsageTap:
    """Records the `usage` object the deployed model manager sees on each response.

    Wraps ModelManager._log_cache_usage (called by every API path with the response
    usage). The wrapper only observes; the original still runs.
    """

    def __init__(self, model_manager=None):
        self.last = None
        if model_manager is not None:
            original = model_manager._log_cache_usage

            def tapped(usage, **kwargs):
                self.last = _usage_dict(usage)
                return original(usage, **kwargs)

            model_manager._log_cache_usage = tapped

    def reset(self):
        self.last = None


def _usage_dict(usage) -> dict:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return dict(usage) if isinstance(usage, dict) else {}


class MeteredGenerator:
    """EvalGenerator wrapper: same .generate() contract, plus the last call's usage/cost."""

    def __init__(self, inner, tap, prices: dict):
        self._inner, self._tap, self._prices = inner, tap, prices
        self.last = None

    async def generate(self, **kwargs):
        self._tap.reset()
        result = await self._inner.generate(**kwargs)
        usage = self._tap.last or {}
        pt = int(usage.get("prompt_tokens") or result.prompt_token_count)
        ct = int(usage.get("completion_tokens") or result.response_token_count)
        slug = resolve_slug(kwargs["model"])
        # OpenRouter reports the billed amount as usage.cost when usage accounting is on.
        usd = usage.get("cost")
        usd = float(usd) if isinstance(usd, (int, float)) else call_cost(self._prices, slug, pt, ct)
        self.last = {"prompt_tokens": pt, "response_tokens": ct, "cost_usd": usd,
                     "from_usage": bool(self._tap.last)}
        return result


class Spend:
    """Cumulative USD for the run dir (survives --resume) against the cap."""

    def __init__(self, max_usd: float, already: float = 0.0):
        self.max_usd, self.total = max_usd, already

    def check(self, worst_case_usd: float) -> None:
        if self.total + worst_case_usd > self.max_usd:
            raise StopRun("max_usd", f"spent ${self.total:.3f} + next call up to ${worst_case_usd:.3f} "
                                     f"would exceed --max-usd ${self.max_usd:.2f}")

    def add(self, usd: float) -> None:
        self.total += usd


def _is_credit_stop(text: str) -> bool:
    return (text or "").lstrip().startswith("[CREDITS EXHAUSTED]")


# ---------------------------------------------------------------------------
# Phases (take injected build/generator/judge so tests run with fakes)
# ---------------------------------------------------------------------------

async def generate_phase(outdir: Path, queries: list, models: list, build_prompt,
                         generator, spend: Spend, prices: dict, max_tokens: int) -> None:
    path = outdir / "generations.jsonl"
    done = {(r["query_id"], r["model"]) for r in _bx._load_jsonl(path)}
    for q in queries:
        todo = [m for m in models if (q["query_id"], m) not in done]
        if not todo:
            continue
        full_prompt, system_prompt = await build_prompt(q["query_text"])  # ONCE per query
        for m in todo:
            est_in = max(1, len(full_prompt) // 4)
            spend.check(call_cost(prices, resolve_slug(m), est_in, max_tokens))
            r = await generator.generate(assembled_prompt=full_prompt, model=m, temperature=0.3,
                                         max_tokens=max_tokens, system_message=system_prompt)
            if _is_credit_stop(r.response_text):
                raise StopRun("http_402", f"{m}: provider reports insufficient credits")
            meter = generator.last
            spend.add(meter["cost_usd"])
            rec = {"query_id": q["query_id"], "query_text": q["query_text"],
                   "intent": q.get("intent", ""), "model": m,
                   "prompt_tokens": meter["prompt_tokens"], "response_tokens": meter["response_tokens"],
                   "generation_ms": r.generation_time_ms, "cost_usd": meter["cost_usd"],
                   "response": r.response_text, "api_error": _bx._is_api_error(r.response_text),
                   "ts": datetime.now().isoformat(timespec="seconds")}
            _bx._append_jsonl(path, rec)
            done.add((q["query_id"], m))
            print(f"[bakeoff] {q['query_id']} {m}: {meter['prompt_tokens']}in/{meter['response_tokens']}out "
                  f"${meter['cost_usd']:.4f} {r.generation_time_ms}ms"
                  + (" API-ERROR" if rec["api_error"] else ""))


async def judge_phase(outdir: Path, queries: list, models: list, judge, judge_generator,
                      spend: Spend, prices: dict, judge_model: str) -> None:
    gens = {(g["query_id"], g["model"]): g for g in _bx._load_jsonl(outdir / "generations.jsonl")}
    path = outdir / "judgments.jsonl"
    done = {(v["snapshot_id"], v["variant_id"]) for v in _bx._load_jsonl(path)}
    baseline = models[0]
    for q in queries:
        base = gens.get((q["query_id"], baseline))
        if not base or base["api_error"]:
            print(f"[bakeoff] judge: skipping {q['query_id']} (no clean baseline)")
            continue
        for m in models[1:]:
            cand = gens.get((q["query_id"], m))
            if (q["query_id"], m) in done or not cand or cand["api_error"]:
                continue
            est_in = max(1, (len(base["response"]) + len(cand["response"]) + len(q["query_text"])) // 4 + 600)
            spend.check(call_cost(prices, resolve_slug(judge_model), est_in, JUDGE_MAX_TOKENS))
            v = await judge.judge_pair(query=q["query_text"], baseline_response=base["response"],
                                       variant_response=cand["response"], snapshot_id=q["query_id"],
                                       variant_id=m, strategy="model_bakeoff",
                                       sections_removed=[], sections_added=[f"model:{baseline}->{m}"])
            meter = judge_generator.last
            spend.add(meter["cost_usd"])
            rec = asdict(v)
            rec.update(prompt_tokens=meter["prompt_tokens"], response_tokens=meter["response_tokens"],
                       cost_usd=meter["cost_usd"])
            _bx._append_jsonl(path, rec)
            print(f"[bakeoff] judged {q['query_id']} {m}: winner_is_baseline={v.winner_is_baseline} "
                  f"${meter['cost_usd']:.4f}")


def spent_so_far(outdir: Path) -> float:
    total = 0.0
    for name in ("generations.jsonl", "judgments.jsonl"):
        total += sum(float(r.get("cost_usd", 0.0)) for r in _bx._load_jsonl(outdir / name))
    return total


def write_report(outdir: Path, models: list, judge_model: str) -> str:
    report = build_report(_bx._load_jsonl(outdir / "generations.jsonl"),
                          _bx._load_jsonl(outdir / "judgments.jsonl"), models, judge_model)
    (outdir / "report.md").write_text(report, encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fetch_catalog() -> dict:
    return _urllib_get(CATALOG_URL, {})


async def _run(args, outdir: Path, queries: list, prices: dict, plan: dict) -> int:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        print("ERROR: OPENROUTER_API_KEY is not set")
        return 2
    try:
        check_balance(fetch_balance(key, _urllib_get), plan["estimate_usd"])
    except StopRun as stop:
        print(f"REFUSED ({stop.reason}): {stop.detail}")
        return 2

    from main import build_orchestrator
    from eval.no_store_generation import EvalGenerator
    from eval.persistence_guard import PersistenceGuard
    from eval.judge import PairwiseJudge
    import config.app_config as _app_config
    _app_config.PROACTIVE_SURFACING_ENABLED = False  # writes surfacing_history.json + leaks state across models

    print("[bakeoff] Booting orchestrator (models + ChromaDB)...")
    orch = build_orchestrator()
    mm = orch.model_manager
    tap = UsageTap(mm)
    gen = MeteredGenerator(EvalGenerator(model_manager=mm), tap, prices)
    judge_gen = MeteredGenerator(EvalGenerator(model_manager=mm), tap, prices)
    judge = PairwiseJudge(judge_gen, judge_model=args.judge_model)
    spend = Spend(args.max_usd, already=spent_so_far(outdir))

    async def build_prompt(text):
        full_prompt, system_prompt, _ = await orch.prepare_prompt(user_input=text, return_context=True)
        return full_prompt, system_prompt

    guard = PersistenceGuard(chromadb_client=getattr(orch.memory_system.chroma_store, "client", None))
    before = guard.capture()
    code = 0
    try:
        await generate_phase(outdir, queries, args.models, build_prompt, gen, spend, prices, args.max_tokens)
        before.assert_same_as(guard.capture())  # raises if build/generate wrote anything
        print("[bakeoff] Persistence guard: no store mutations.")
        await judge_phase(outdir, queries, args.models, judge, judge_gen, spend, prices, args.judge_model)
    except StopRun as stop:
        print(f"[bakeoff] STOPPED ({stop.reason}): {stop.detail}\n"
              f"[bakeoff] Resume with: --resume {outdir} --models {' '.join(args.models)}")
        code = 3
    print(f"[bakeoff] Spent ${spend.total:.3f} of ${args.max_usd:.2f}")
    print(write_report(outdir, args.models, args.judge_model))
    return code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=["kimi-3"],
                    help="aliases or slugs; FIRST is the baseline (default: kimi-3)")
    ap.add_argument("--judge-model", default="claude-opus-4.8")
    ap.add_argument("--per-intent", type=int, default=1, help="queries per intent (default 1 -> 9)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-usd", type=float, default=2.00)
    ap.add_argument("--output-dir", default="eval/runs")
    ap.add_argument("--resume", default=None, help="existing run dir")
    ap.add_argument("--run", action="store_true", help="spend credits (default: dry run)")
    args = ap.parse_args(argv)

    from eval.corpus import SEED_CORPUS
    queries = select_queries(SEED_CORPUS, args.per_intent)

    try:
        prices = catalog_prices(_fetch_catalog())  # free GET
        plan = build_plan(len(queries), args.models, args.judge_model, prices,
                          measured_tokens(args.resume))
    except StopRun as stop:
        print(f"REFUSED ({stop.reason}): {stop.detail}")
        return 2
    print(format_plan(plan, args.max_usd))
    if not args.run:
        print("\nDRY RUN — nothing was spent. Add --run to execute.")
        return 0

    if _bx._daemon_running():
        print("REFUSED: a live Daemon main.py process is running (shared ChromaDB). Close it first.")
        return 2
    if args.resume:
        outdir = Path(args.resume)
        if not outdir.exists():
            print(f"--resume dir not found: {outdir}")
            return 2
    else:
        outdir = Path(args.output_dir) / ("model_bakeoff_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps({
        "models": args.models, "judge_model": args.judge_model, "per_intent": args.per_intent,
        "max_usd": args.max_usd, "max_tokens": args.max_tokens, "web_search_disabled": True,
        "started": datetime.now().isoformat(timespec="seconds")}, indent=2))
    return asyncio.run(_run(args, outdir, queries, prices, plan))


if __name__ == "__main__":
    sys.exit(main())
