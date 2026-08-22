"""Token-budget tradeoff experiment: is a smaller prompt budget worth it?

Context (2026-07-15): every main-model call carries ~28K prompt tokens
(15.3K budgeted sections + ~6K assembly overhead + ~7K system prompt) and
DeepSeek's server-side cache almost never hits at conversation pacing, so
prefill is paid nearly in full every call. Lowering token_budget.default
15000 → 12000/10000 is the biggest remaining TTFB + cost lever, but it
trades retrieved context for speed. This script measures that tradeoff.

Method
------
For each seed-corpus query (eval/corpus.py, 27 queries, 3 per intent) and
each budget condition (first listed = BASELINE):
  1. Set the budget on THE deployed builder (same attribute the config
     sets — no re-derivation, per the validation rule in CLAUDE.md).
  2. Build the prompt through THE deployed orchestrator.prepare_prompt().
  3. Generate side-effect-free via eval.no_store_generation.EvalGenerator
     (wraps the deployed model_manager.generate_once).
Then judge baseline vs each candidate per query with the existing
position-randomized 5-criterion PairwiseJudge (eval/judge.py), and report
win/tie/loss + score deltas next to prompt-token and latency savings.

Safety
------
- WEB_SEARCH_ENABLED=0 is forced before any repo import (no Tavily credits,
  no cross-condition nondeterminism from live results).
- PROACTIVE_SURFACING_ENABLED is patched off for the run — it writes
  surfacing_history.json at prepare time AND its cooldown would leak state
  between conditions.
- eval.persistence_guard fingerprints the data stores before/after the
  build+generate phase and aborts loudly if anything was written.
- Refuses to run when a live Daemon main.py process is detected (shared
  ChromaDB), unless --force.

PREREGISTERED DECISION RULE (set before looking at any results):
  Adopt the LOWEST candidate budget for which, against baseline 15000:
    (a) baseline wins ≤ 1/3 of non-tie-inclusive judgments
        (baseline_win_rate = baseline_wins / n_judged ≤ 0.33), AND
    (b) mean total-score delta (candidate − baseline, out of 25) ≥ −0.3.
  Otherwise keep 15000.

Usage
-----
  python scripts/budget_experiment.py --smoke          # 2 queries, 15000 vs 10000
  python scripts/budget_experiment.py                  # full 27 × [15000,12000,10000]
  python scripts/budget_experiment.py --resume eval/runs/budget_expt_<ts>
Run under a memory cap:
  systemd-run --user --scope -p MemoryMax=9G python scripts/budget_experiment.py
"""

import os
import sys

# MUST precede any repo import — app_config reads env at import time.
os.environ["WEB_SEARCH_ENABLED"] = "0"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import asyncio
import json
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


def _daemon_running() -> bool:
    try:
        from utils.daemon_guard import daemon_running
        return daemon_running()
    except Exception:
        pass
    # Fallback (guard module unavailable): old cmdline heuristic. Known hole:
    # a relative-path launch has no repo name in its cmdline (2026-08-21).
    try:
        out = subprocess.run(
            ["pgrep", "-af", "main.py"], capture_output=True, text=True
        ).stdout
        return any("Daemon_v1" in line for line in out.splitlines())
    except Exception:
        return False


def _set_budget(builder, budget: int) -> None:
    """Point THE deployed trimming code at a different budget value."""
    builder.token_budget = budget
    builder.token_manager.token_budget = budget


def _is_api_error(text: str) -> bool:
    t = (text or "").strip()
    return t.startswith("[API Error") or t.startswith("[CREDITS")


def _append_jsonl(path: Path, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


async def _generate_phase(args, outdir, queries, budgets):
    from main import build_orchestrator
    from eval.no_store_generation import EvalGenerator
    from eval.persistence_guard import PersistenceGuard

    # Proactive insight surfacing WRITES data/surfacing_history.json during
    # prepare_prompt (cooldown bookkeeping) — a store mutation the guard
    # rightly rejects, and a comparability bug: an insight surfaced in the
    # baseline condition goes on cooldown and disappears from the candidate
    # conditions. memory_coordinator imports the flag lazily per call, so
    # patching the module attribute disables it for the whole run.
    import config.app_config as _app_config
    _app_config.PROACTIVE_SURFACING_ENABLED = False
    print("[budget-expt] Proactive surfacing disabled for the run.")

    print("[budget-expt] Booting orchestrator (models + ChromaDB)...")
    orch = build_orchestrator()
    builder = orch.prompt_builder
    mm = orch.model_manager
    gen_model = args.model or getattr(mm, "active_model_name", None) or "gpt-4o-mini"
    print(f"[budget-expt] Generation model: {gen_model}")

    generator = EvalGenerator(model_manager=mm)
    guard = PersistenceGuard(
        chromadb_client=getattr(orch.memory_system.chroma_store, "client", None)
    )
    before = guard.capture()

    results_path = outdir / "results.jsonl"
    done = {(r["query_id"], r["budget"]) for r in _load_jsonl(results_path)}
    original_budget = builder.token_budget
    total = len(queries) * len(budgets)
    n = 0

    try:
        for q in queries:
            qid = q["query_id"]
            for budget in budgets:
                n += 1
                if (qid, budget) in done:
                    print(f"[budget-expt] ({n}/{total}) {qid} @ {budget}: already done, skipping")
                    continue
                _set_budget(builder, budget)

                t0 = time.perf_counter()
                full_prompt, system_prompt, _ = await orch.prepare_prompt(
                    user_input=q["query_text"], return_context=True
                )
                build_s = time.perf_counter() - t0

                r = await generator.generate(
                    assembled_prompt=full_prompt,
                    model=gen_model,
                    temperature=0.3,
                    max_tokens=args.max_tokens,
                    system_message=system_prompt,
                )
                rec = {
                    "query_id": qid,
                    "query_text": q["query_text"],
                    "intent": q.get("intent", ""),
                    "budget": budget,
                    "model": gen_model,
                    "prompt_chars": len(full_prompt),
                    "prompt_tokens": r.prompt_token_count,
                    "response_tokens": r.response_token_count,
                    "build_s": round(build_s, 2),
                    "generation_ms": r.generation_time_ms,
                    "response": r.response_text,
                    "api_error": _is_api_error(r.response_text),
                    "ts": datetime.now().isoformat(timespec="seconds"),
                }
                _append_jsonl(results_path, rec)
                print(
                    f"[budget-expt] ({n}/{total}) {qid} @ {budget}: "
                    f"prompt={rec['prompt_tokens']}tok build={build_s:.1f}s "
                    f"gen={r.generation_time_ms}ms"
                    + (" API-ERROR" if rec["api_error"] else "")
                )
    finally:
        _set_budget(builder, original_budget)

    after = guard.capture()
    before.assert_same_as(after)  # raises if generation wrote anything
    print("[budget-expt] Persistence guard: no store mutations. ✓")
    return mm


async def _judge_phase(args, outdir, queries, budgets, mm):
    from eval.no_store_generation import EvalGenerator
    from eval.judge import PairwiseJudge

    results = _load_jsonl(outdir / "results.jsonl")
    by_key = {(r["query_id"], r["budget"]): r for r in results}
    baseline_budget = budgets[0]

    generator = EvalGenerator(model_manager=mm)
    judge = PairwiseJudge(generator, judge_model=args.judge_model)

    verdicts_path = outdir / "verdicts.jsonl"
    done = {(v["snapshot_id"], v["variant_id"]) for v in _load_jsonl(verdicts_path)}

    for q in queries:
        qid = q["query_id"]
        base = by_key.get((qid, baseline_budget))
        if not base or base["api_error"]:
            print(f"[budget-expt] judge: skipping {qid} (no clean baseline)")
            continue
        for budget in budgets[1:]:
            variant_id = f"budget_{budget}"
            if (qid, variant_id) in done:
                continue
            cand = by_key.get((qid, budget))
            if not cand or cand["api_error"]:
                print(f"[budget-expt] judge: skipping {qid} @ {budget} (no clean candidate)")
                continue
            v = await judge.judge_pair(
                query=q["query_text"],
                baseline_response=base["response"],
                variant_response=cand["response"],
                snapshot_id=qid,
                variant_id=variant_id,
                strategy="token_budget",
                sections_removed=[f"budget:{baseline_budget}->{budget}"],
                sections_added=[],
            )
            _append_jsonl(verdicts_path, asdict(v))
            print(
                f"[budget-expt] judged {qid} vs {budget}: "
                f"winner_is_baseline={v.winner_is_baseline} conf={v.confidence:.2f}"
            )


def _report(outdir, budgets):
    results = _load_jsonl(outdir / "results.jsonl")
    verdicts = _load_jsonl(outdir / "verdicts.jsonl")
    baseline_budget = budgets[0]

    lines = ["# Token-Budget Experiment Report", ""]
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Baseline budget: {baseline_budget}")
    lines.append("")

    # --- Cost/latency per budget ---
    lines.append("## Prompt size & latency per budget")
    lines.append("")
    lines.append("| budget | n | mean prompt tokens | mean build s | mean gen ms | api errors |")
    lines.append("|---|---|---|---|---|---|")
    stats = {}
    for b in budgets:
        rs = [r for r in results if r["budget"] == b]
        clean = [r for r in rs if not r["api_error"]]
        if not rs:
            continue
        mean = lambda k, xs: (sum(x[k] for x in xs) / len(xs)) if xs else 0.0
        stats[b] = {
            "prompt_tokens": mean("prompt_tokens", clean),
            "gen_ms": mean("generation_ms", clean),
            "build_s": mean("build_s", clean),
        }
        lines.append(
            f"| {b} | {len(rs)} | {stats[b]['prompt_tokens']:.0f} "
            f"| {stats[b]['build_s']:.1f} | {stats[b]['gen_ms']:.0f} "
            f"| {len(rs) - len(clean)} |"
        )
    lines.append("")
    lines.append(
        "_Build times are warm-cache-biased (conditions run back-to-back per "
        "query); prompt tokens and generation ms are the primary speed metrics._"
    )
    lines.append("")

    # --- Quality per candidate ---
    lines.append("## Quality vs baseline (position-randomized pairwise judge)")
    criteria = ["accuracy", "helpfulness", "conciseness", "tone", "grounding"]
    decision = None
    for b in budgets[1:]:
        vs = [v for v in verdicts if v["variant_id"] == f"budget_{b}"]
        if not vs:
            continue
        base_wins = sum(1 for v in vs if v["winner_is_baseline"] is True)
        cand_wins = sum(1 for v in vs if v["winner_is_baseline"] is False)
        ties = sum(1 for v in vs if v["winner"] == "tie")
        n = len(vs)

        deltas = {c: [] for c in criteria}
        for v in vs:
            base_scores = v["scores_a"] if v["baseline_position"] == "A" else v["scores_b"]
            cand_scores = v["scores_b"] if v["baseline_position"] == "A" else v["scores_a"]
            for c in criteria:
                if c in base_scores and c in cand_scores:
                    deltas[c].append(cand_scores[c] - base_scores[c])
        mean_deltas = {
            c: (sum(d) / len(d) if d else 0.0) for c, d in deltas.items()
        }
        total_delta = sum(mean_deltas.values())
        base_win_rate = base_wins / n if n else 0.0

        lines.append("")
        lines.append(f"### Candidate budget {b} (n={n})")
        lines.append("")
        lines.append(f"- baseline wins: {base_wins} ({base_win_rate:.0%}) | "
                     f"candidate wins: {cand_wins} | ties: {ties}")
        lines.append(f"- mean score delta (candidate − baseline, out of 25): "
                     f"{total_delta:+.2f}")
        lines.append("- per criterion: " + ", ".join(
            f"{c} {mean_deltas[c]:+.2f}" for c in criteria))
        losers = [v for v in vs if v["winner_is_baseline"] is True]
        if losers:
            lines.append("- queries where baseline won: " + ", ".join(
                f"{v['snapshot_id']}({v['confidence']:.2f})" for v in losers))

        passes = base_win_rate <= 0.33 and total_delta >= -0.3
        lines.append(f"- **preregistered rule**: baseline_win_rate ≤ 0.33 → "
                     f"{'PASS' if base_win_rate <= 0.33 else 'FAIL'} "
                     f"({base_win_rate:.2f}); total delta ≥ −0.3 → "
                     f"{'PASS' if total_delta >= -0.3 else 'FAIL'} ({total_delta:+.2f})")
        if passes and (decision is None or b < decision):
            decision = b

    lines.append("")
    lines.append("## Decision (preregistered rule)")
    if decision is not None:
        saved = stats.get(baseline_budget, {}).get("prompt_tokens", 0) - \
            stats.get(decision, {}).get("prompt_tokens", 0)
        lines.append(
            f"**ADOPT budget {decision}** — lowest candidate passing both "
            f"criteria. Mean prompt-token saving vs baseline: {saved:.0f} "
            f"tokens/call (≈2 calls per agentic turn)."
        )
    else:
        lines.append("**KEEP baseline** — no candidate passed both criteria.")

    report = "\n".join(lines) + "\n"
    (outdir / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"[budget-expt] Report written to {outdir / 'report.md'}")


async def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--budgets", default="15000,12000,10000",
                    help="Comma-separated; FIRST is the baseline")
    ap.add_argument("--limit", type=int, default=0, help="Only first N corpus queries")
    ap.add_argument("--model", default=None, help="Generation model (default: active)")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--output-dir", default="eval/runs")
    ap.add_argument("--resume", default=None, help="Existing run dir to resume")
    ap.add_argument("--smoke", action="store_true",
                    help="2 queries, budgets 15000,10000")
    ap.add_argument("--force", action="store_true",
                    help="Run even if a live Daemon process is detected")
    ap.add_argument("--report-only", action="store_true",
                    help="Just regenerate the report from an existing run (needs --resume)")
    args = ap.parse_args()

    if args.smoke:
        args.limit = args.limit or 2
        args.budgets = "15000,10000"
    budgets = [int(b.strip()) for b in args.budgets.split(",")]

    if args.resume:
        outdir = Path(args.resume)
        if not outdir.exists():
            raise SystemExit(f"--resume dir not found: {outdir}")
    else:
        outdir = Path(args.output_dir) / (
            "budget_expt_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        outdir.mkdir(parents=True, exist_ok=True)
    print(f"[budget-expt] Run dir: {outdir}")

    from eval.corpus import SEED_CORPUS
    queries = SEED_CORPUS[: args.limit] if args.limit else SEED_CORPUS
    print(f"[budget-expt] {len(queries)} queries × {budgets} "
          f"(baseline={budgets[0]})")

    if args.report_only:
        _report(outdir, budgets)
        return

    if _daemon_running() and not args.force:
        raise SystemExit(
            "A live Daemon main.py process is running — it shares ChromaDB "
            "with this experiment. Close it first (or pass --force)."
        )

    (outdir / "config.json").write_text(json.dumps({
        "budgets": budgets, "limit": args.limit, "model": args.model,
        "judge_model": args.judge_model, "max_tokens": args.max_tokens,
        "web_search_disabled": True,
        "started": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))

    mm = await _generate_phase(args, outdir, queries, budgets)
    await _judge_phase(args, outdir, queries, budgets, mm)
    _report(outdir, budgets)


if __name__ == "__main__":
    asyncio.run(main())
