#!/usr/bin/env python3
"""Evaluate the tone arbiters on the labelled select cases (plan 2026-10-08, B6b; R1 tone only).

Compares THE deployed LLM arbiter (utils.tone_detector._llm_crisis_fallback, on the app's active chat
model, recorded in the report) with THE deployed decision-model arbiter
(_decision_model_crisis_arbiter). Jev is enabled for THIS PROCESS ONLY by patching the in-memory
config; no config file is written. With the policy unset the arbiter returns no verdict but its
receipt carries the probabilities, which are stored; every verdict is recomputed here with
utils.tone_detector.tone_verdict, so --replay and --select need no new inference.

    python scripts/probe_decision_model_eval.py                          # dry run: counts + price, NO network
    python scripts/probe_decision_model_eval.py --run --budget-usd 1.5   # spends credits (cap 2.00)
    python scripts/probe_decision_model_eval.py --replay metrics.json [--policy weighted --params '{"cuts":[.5,1.5,2.5]}']
    python scripts/probe_decision_model_eval.py --select --replay metrics.json   # freeze policy + YAML lines
    python scripts/probe_decision_model_eval.py --run --budget-usd 1 --private owner.jsonl | --audit   # aggregates / counts only

Outputs report.md + metrics.json (ids, labels, probabilities, verdicts, latency, cost; NEVER case text).
"""
import argparse
import asyncio
import contextlib
import copy
import itertools
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from config import app_config  # noqa: E402
from eval import decision_model_cases as cases_mod  # noqa: E402
from utils import tone_detector as tone  # noqa: E402

LEVELS = list(cases_mod.LABELS)  # CONVERSATIONAL, CONCERN, MEDIUM, HIGH (index = level)
BUDGET_CAP_USD = 2.00  # D4: owner cap
JEV_USD_PER_CALL = 1.9e-5  # measured live 2026-10-08 for a short message (~1,200 chars of payload)
JEV_PAYLOAD_BASE_CHARS = 1200
LLM_OVERHEAD_CHARS = 700  # arbiter prompt around the message
WEIGHTS = {"severe": 20.0, "missed": 3.0, "false_alarm": 1.0}  # section 9 E1, owner approves before the run
CALL_GUARD_S = 15.0
ADVERSARIAL = ("adversarial_severe", "adversarial_conv")
LONG = ("long_mid", "long_over")
DEFAULT_OUT = Path("~/daemon_exec/jev_decision_runs/E1/")


# ---------------------------------------------------------------- statistics (stdlib only)
def wilson(k, n, z=1.959964):
    """Two-sided 95% Wilson score interval for k successes in n; (None, None) when n == 0."""
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def _binom_cdf(k, n, p):
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    return sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                        + i * math.log(p) + (n - i) * math.log1p(-p)) for i in range(k + 1))


def clopper_pearson_upper(k, n, alpha=0.05):
    """One-sided (1 - alpha) Clopper-Pearson upper bound on a rate with k events in n; None when n == 0."""
    if n == 0:
        return None
    if k >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _binom_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def brier(prob_rows):
    """Mean multiclass Brier score over (probs, expected_index) pairs; None when empty."""
    if not prob_rows:
        return None
    return sum(sum((p - (1.0 if j == e else 0.0)) ** 2 for j, p in enumerate(ps)) for ps, e in prob_rows) / len(prob_rows)


def reliability(prob_rows, bins=5):
    """Reliability table over the top-class confidence (max probability) vs argmax accuracy."""
    out = [{"bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}", "n": 0, "conf": 0.0, "acc": 0.0} for b in range(bins)]
    for ps, e in prob_rows:
        top = max(range(4), key=lambda i: (ps[i], i))
        row = out[min(int(ps[top] * bins), bins - 1)]
        row["n"] += 1
        row["conf"] += ps[top]
        row["acc"] += 1.0 if top == e else 0.0
    return [{**r, "conf": r["conf"] / r["n"] if r["n"] else None, "acc": r["acc"] / r["n"] if r["n"] else None} for r in out]


def percentile(values, q):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, math.ceil(q * len(s)) - 1)]


# ---------------------------------------------------------------- verdicts and metrics
def jev_verdict(probs, policy, params):
    """Index verdict of a stored distribution under (policy, params) via THE deployed tone_verdict."""
    level = tone.tone_verdict(probs, policy, params) if probs else None
    return LEVELS.index(level.name) if level else None


def case_cost(expected, verdict):
    """Section 9 E1 cost: severe miss, other missed distress, false alarm (any over-call); 0 if right."""
    if verdict is None or verdict == expected:
        return 0.0
    if verdict < expected:
        return WEIGHTS["severe"] if (expected >= 2 and verdict == 0) else WEIGHTS["missed"]
    return WEIGHTS["false_alarm"]


def _rate(k, n):
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "wilson95": [lo, hi]}


def backend_metrics(items):
    """items: dicts {id, group, expected, verdict|None, latency_ms|None}. Unavailable is counted apart (BC-47)."""
    ans = [i for i in items if i["verdict"] is not None]
    conf = [[0] * 4 for _ in range(4)]
    for i in ans:
        conf[i["expected"]][i["verdict"]] += 1
    dist = [i for i in ans if i["expected"] >= 1]
    sev = [i for i in ans if i["expected"] >= 2]
    calm = [i for i in ans if i["expected"] == 0]
    misses = [i for i in sev if i["verdict"] == 0]

    def tranche(groups):
        t = [i for i in ans if i["group"] in groups]
        return {"n": len([i for i in items if i["group"] in groups]), "answered": len(t),
                "correct": sum(i["verdict"] == i["expected"] for i in t),
                "severe_misses": sum(i["expected"] >= 2 and i["verdict"] == 0 for i in t)}

    lat = [i["latency_ms"] for i in ans if i.get("latency_ms") is not None]
    return {"n": len(items), "answered": len(ans), "unavailable": len(items) - len(ans),
            "accuracy": _rate(sum(i["verdict"] == i["expected"] for i in ans), len(ans)),
            "confusion_expected_by_verdict": conf,
            "distress_recall": _rate(sum(i["verdict"] >= 1 for i in dist), len(dist)),
            "severe": {"n": len(sev), "misses": len(misses), "miss_ids": [i["id"] for i in misses],
                       "upper95_one_sided": clopper_pearson_upper(len(misses), len(sev))},
            "false_alarm": _rate(sum(i["verdict"] >= 1 for i in calm), len(calm)),
            "adversarial": tranche(ADVERSARIAL), "long": tranche(LONG),
            "latency_ms": {"p50": percentile(lat, 0.5), "p95": percentile(lat, 0.95)}}


def paired(llm, jev):
    """Discordant counts on cases both answered: b = LLM right and Jev wrong, c = the reverse."""
    both = [(a, b) for a, b in zip(llm, jev) if a["verdict"] is not None and b["verdict"] is not None]
    tests = {"exact": lambda e, v: v == e, "distress_caught": lambda e, v: v >= 1, "calm_kept": lambda e, v: v == 0}
    subsets = {"exact": lambda e: True, "distress_caught": lambda e: e >= 1, "calm_kept": lambda e: e == 0}
    return {name: {"b_llm_right_jev_wrong": sum(f(a["expected"], a["verdict"]) and not f(a["expected"], b["verdict"])
                                                 for a, b in both if subsets[name](a["expected"])),
                   "c_jev_right_llm_wrong": sum(f(a["expected"], b["verdict"]) and not f(a["expected"], a["verdict"])
                                                 for a, b in both if subsets[name](a["expected"]))}
            for name, f in tests.items()}


def summarize(rows, policy, params):
    """Recompute every Jev verdict from the stored probabilities, then all metrics (no inference)."""
    llm_items, jev_items, prob_rows, reasons = [], [], [], Counter()
    for r in rows:
        e = LEVELS.index(r["expected"])
        lv = LEVELS.index(r["llm"]["level"]) if (r.get("llm") or {}).get("level") else None
        llm_items.append({"id": r["id"], "group": r["group"], "expected": e, "verdict": lv,
                          "latency_ms": (r.get("llm") or {}).get("latency_ms")})
        j = r.get("jev") or {}
        j["verdict"] = jev_verdict(j.get("probs"), policy, params) if j.get("status") == "ok" else None
        if j.get("status") != "ok":
            reasons[str(j.get("reason") or j.get("status"))] += 1
        elif j.get("probs"):
            prob_rows.append((j["probs"], e))
        jev_items.append({"id": r["id"], "group": r["group"], "expected": e, "verdict": j["verdict"],
                          "latency_ms": j.get("latency_ms")})
    jev_cost = sum(r["jev"].get("cost_usd") or 0.0 for r in rows if r.get("jev"))
    return {"policy": policy, "params": params, "cases": len(rows),
            "llm": backend_metrics(llm_items), "jev": backend_metrics(jev_items), "paired": paired(llm_items, jev_items),
            "jev_unavailable_reasons": dict(reasons), "jev_brier": brier(prob_rows),
            "jev_reliability": reliability(prob_rows),
            "cost": {"jev_usd": round(jev_cost, 6), "jev_estimated_calls": sum(bool((r.get("jev") or {}).get("cost_estimated")) for r in rows),
                     "llm_usd_estimate": round(sum((r.get("llm") or {}).get("cost_usd_estimate") or 0.0 for r in rows), 6)}}


# ---------------------------------------------------------------- selection (E1)
def select_policy(rows):
    """Grid argmax/weighted/cumulative; minimise expected cost subject to ZERO severe misses. Returns the winner or None."""
    pool = [(LEVELS.index(r["expected"]), r["jev"]["probs"]) for r in rows
            if (r.get("jev") or {}).get("status") == "ok" and r["jev"].get("probs")]
    steps = [round(x, 2) for x in (0.05 * i for i in range(1, 20))]
    grids = [("argmax", {})]
    grids += [("weighted", {"cuts": [a, b, c]}) for a, b, c in itertools.product(
        [0.3, 0.4, 0.5, 0.6, 0.7], [1.3, 1.4, 1.5, 1.6, 1.7], [2.3, 2.4, 2.5, 2.6, 2.7])]
    grids += [("cumulative", {"taus": [a, b, c]}) for a, b, c in itertools.product(steps, steps, steps)]
    best = None
    for order, (policy, params) in enumerate(grids):
        cost, severe = 0.0, 0
        for e, probs in pool:
            v = jev_verdict(probs, policy, params)
            if v is None:
                break
            cost += case_cost(e, v)
            severe += e >= 2 and v == 0
        else:
            if severe == 0 and (best is None or (cost, order) < (best[0], best[1])):
                best = (cost, order, policy, params, len(pool))
    return None if best is None else {"cost": best[0], "policy": best[2], "params": best[3], "cases": best[4]}


def yaml_lines(choice):
    key = {"weighted": "cuts", "cumulative": "taus"}.get(choice["policy"])
    params = json.dumps({key: choice["params"][key]}) if key else "{}"
    return [f"  tone_policy: {choice['policy']}", f"  tone_policy_params: {params}"]


# ---------------------------------------------------------------- running the deployed arbiters
@contextlib.contextmanager
def jev_enabled_in_process():
    """Enable the decision model for THIS PROCESS ONLY: patch the in-memory config reads; write nothing."""
    cfg = copy.deepcopy(app_config.DECISION_MODEL_CFG or {})
    cfg["enabled"] = True
    cfg["tone_policy"] = "unset"
    cfg.setdefault("roles", {})["tone_arbiter"] = "shadow"
    with mock.patch.object(app_config, "DECISION_MODEL_CFG", cfg), \
            mock.patch.object(app_config, "DECISION_MODEL_TONE_POLICY", "unset"), \
            mock.patch.object(app_config, "DECISION_MODEL_TONE_POLICY_PARAMS", {}):
        yield


def jev_estimate_usd(text):
    return JEV_USD_PER_CALL * max(1.0, (len(text) + JEV_PAYLOAD_BASE_CHARS) / JEV_PAYLOAD_BASE_CHARS)


def llm_estimate_usd(text, usd_per_mtok_in):
    """Conservative: ~3 chars/token in, 16 tokens out billed at 3x the input rate. An ESTIMATE (no cost in the reply)."""
    return ((len(text) + LLM_OVERHEAD_CHARS) / 3 * usd_per_mtok_in + 16 * 3 * usd_per_mtok_in) / 1e6


async def _timed(coro):
    t0 = time.monotonic()
    try:
        return await asyncio.wait_for(coro, CALL_GUARD_S), (time.monotonic() - t0) * 1000
    except asyncio.TimeoutError:
        return None, (time.monotonic() - t0) * 1000


async def run_cases(cases, mm, budget_usd, llm_usd_per_mtok=10.0):
    """Call both deployed arbiters per case. Stops at the first http_402 or when summed spend reaches the budget."""
    rows, spent, stopped = [], 0.0, None
    for c in cases:
        if spent >= budget_usd:
            stopped = "budget"
            break
        (llm_r, llm_ms), (jev_r, _) = await asyncio.gather(
            _timed(tone._llm_crisis_fallback(c["text"], mm)), _timed(tone._decision_model_crisis_arbiter(c["text"], mm)))
        receipt = (jev_r[1] if jev_r else None) or {"status": "unavailable", "reason": "timeout"}
        est = receipt.get("cost_usd") is None and receipt.get("status") == "ok"
        jev_cost = receipt.get("cost_usd") if receipt.get("cost_usd") is not None else (jev_estimate_usd(c["text"]) if est else 0.0)
        llm_cost = llm_estimate_usd(c["text"], llm_usd_per_mtok)
        spent += jev_cost + llm_cost
        rows.append({"id": c["id"], "group": c["group"], "expected": c["expected"],
                     "llm": {"level": llm_r[0].name if llm_r else None, "latency_ms": round(llm_ms, 1),
                             "cost_usd_estimate": llm_cost},
                     "jev": {"status": receipt.get("status"), "reason": receipt.get("reason"), "probs": receipt.get("probs"),
                             "confidence": receipt.get("decision_confidence"), "latency_ms": receipt.get("latency_ms"),
                             "cost_usd": receipt.get("cost_usd") if not est else jev_cost, "cost_estimated": est,
                             "served_model": receipt.get("served_model"), "provider": receipt.get("provider"),
                             "retried": receipt.get("retried")}})
        if receipt.get("reason") == "http_402":
            stopped = "http_402"
            break
        if spent >= budget_usd:
            stopped = "budget"
            break
    return rows, spent, stopped


def make_model_manager():
    """The app's construction: key from the repo .env (never printed), active model from config models.active."""
    from dotenv import dotenv_values  # lazy import: only a --run needs it
    from models.model_manager import ModelManager  # lazy import: heavy; dry run/replay never build one
    mm = ModelManager(api_key=dotenv_values(REPO / ".env").get("OPENROUTER_API_KEY") or None)
    mm.switch_model((app_config.config.get("models", {}) or {}).get("active") or "gpt-5")  # as main.py does
    return mm


# ---------------------------------------------------------------- reporting
def _fmt(x, pct=False):
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%" if pct else (f"{x:.4f}" if isinstance(x, float) else str(x))


def render_report(meta, s):
    out = [f"# Decision-model tone evaluation ({meta['mode']})", "",
           f"- split: {meta['split']}; cases: {s['cases']}; policy: {s['policy']} {json.dumps(s['params'])}",
           f"- LLM arbiter model (BC-82): {meta['llm_model']}; Jev slug: {meta['jev_slug']}; run stopped: {meta.get('stopped')}",
           f"- cost: Jev ${s['cost']['jev_usd']} ({s['cost']['jev_estimated_calls']} estimated calls); "
           f"LLM ${s['cost']['llm_usd_estimate']} (ESTIMATE, no cost reported)", ""]
    for name in ("llm", "jev"):
        m = s[name]
        sev, rec, fa = m["severe"], m["distress_recall"], m["false_alarm"]
        out += [f"## {name.upper()}", f"- answered {m['answered']}/{m['n']}; unavailable {m['unavailable']} (not wrong)",
                f"- accuracy {_fmt(m['accuracy']['rate'], True)}; distress recall {_fmt(rec['rate'], True)} "
                f"(Wilson95 {_fmt(rec['wilson95'][0], True)}-{_fmt(rec['wilson95'][1], True)})",
                f"- severe misses {sev['misses']}/{sev['n']}, one-sided 95% CP upper bound {_fmt(sev['upper95_one_sided'], True)}; ids {sev['miss_ids']}",
                f"- false alarm {_fmt(fa['rate'], True)} ({fa['k']}/{fa['n']}); latency p50 {_fmt(m['latency_ms']['p50'])} ms, p95 {_fmt(m['latency_ms']['p95'])} ms",
                f"- adversarial {m['adversarial']}; long {m['long']}",
                f"- confusion (rows expected, cols verdict; {', '.join(LEVELS)}): {m['confusion_expected_by_verdict']}", ""]
    out += ["## Paired discordant counts (b = LLM right/Jev wrong, c = reverse)"] + [
        f"- {k}: b={v['b_llm_right_jev_wrong']} c={v['c_jev_right_llm_wrong']}" for k, v in s["paired"].items()]
    out += ["", f"## Jev calibration: Brier {_fmt(s['jev_brier'])}; unavailable reasons {s['jev_unavailable_reasons']}", "",
            "| bin | n | mean conf | accuracy |", "|---|---|---|---|"]
    out += [f"| {r['bin']} | {r['n']} | {_fmt(r['conf'])} | {_fmt(r['acc'])} |" for r in s["jev_reliability"]]
    return "\n".join(out) + "\n"


def write_outputs(out_dir, meta, rows, s, private=False):
    out_dir = Path(out_dir).expanduser().resolve()
    if (REPO / "data") in (out_dir, *out_dir.parents):
        raise SystemExit("refusing to write under data/")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_private" if private else ""
    doc = {"meta": meta, "summary": s} if private else {"meta": meta, "summary": s, "cases": rows}
    (out_dir / f"metrics{suffix}.json").write_text(json.dumps(doc, indent=1, default=str))
    (out_dir / f"report{suffix}.md").write_text(render_report(meta, s))
    return out_dir


def dry_run(cases, usd_per_mtok):
    limit = (app_config.DECISION_MODEL_MAX_STATE_CHARS or {}).get("tone_arbiter", 80000)
    jev_calls = [c for c in cases if len(c["text"]) <= limit]
    jev = sum(jev_estimate_usd(c["text"]) for c in jev_calls)
    llm = sum(llm_estimate_usd(c["text"], usd_per_mtok) for c in cases)
    sys.stdout.write(
        "DRY RUN (no network). Use --run --budget-usd X to spend credits.\n"
        f"cases: {len(cases)}  by group: {dict(Counter(c['group'] for c in cases))}\n"
        f"labels: {dict(Counter(c['expected'] for c in cases))}\n"
        f"calls per backend: LLM arbiter {len(cases)}; Jev {len(jev_calls)} "
        f"({len(cases) - len(jev_calls)} over {limit} chars resolve locally as state_too_large)\n"
        f"price estimate: Jev ${jev:.4f} (${JEV_USD_PER_CALL}/call, scaled by size); LLM ${llm:.4f} "
        f"(conservative ESTIMATE at ${usd_per_mtok}/Mtok in); total ${jev + llm:.4f}; budget cap ${BUDGET_CAP_USD:.2f}\n")


# ---------------------------------------------------------------- owner-local blinded audit (E5)
def audit(records_path, n, seed, out_dir, ask=input):
    """Blinded sample of shadow agreements; the owner labels from the text alone. Writes COUNTS only."""
    recs = []
    for line in Path(records_path).read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("tone_dm_mode") == "shadow" and r.get("tone_dm_agrees") is True and r.get("query"):
            recs.append(r)
    level = lambda r: r.get("tone_dm_level") or r.get("tone_dm_deciding_level")  # noqa: E731
    rng = random.Random(seed)
    conv = [r for r in recs if level(r) == "CONVERSATIONAL"]
    rest = [r for r in recs if level(r) != "CONVERSATIONAL"]
    sample = rng.sample(conv, min(n // 2, len(conv))) + rng.sample(rest, min(n - n // 2, len(rest)))
    rng.shuffle(sample)
    letters = {"c": "CONVERSATIONAL", "n": "CONCERN", "m": "MEDIUM", "h": "HIGH"}
    counts = Counter()
    for i, r in enumerate(sample, 1):
        sys.stdout.write(f"\n--- {i}/{len(sample)} ---\n{str(r['query'])[:1500]}\n")
        label = letters.get(str(ask("label [c]onversational/co[n]cern/[m]edium/[h]igh, Enter to skip: ")).strip().lower()[:1])
        if label is None:
            counts["skipped"] += 1
            continue
        counts["labelled"] += 1
        counts["owner_matches_agreed_level"] += label == level(r)
        counts["both_conversational_but_owner_severe"] += level(r) == "CONVERSATIONAL" and label in ("MEDIUM", "HIGH")
    res = {"eligible_agreements": len(recs), "sampled": len(sample), **counts,
           "boundary_filter": "unavailable: turn records carry no semantic top-distress score"}
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit_counts.json").write_text(json.dumps(res, indent=1))
    return res


# ---------------------------------------------------------------- CLI
def load_private(path):
    cases = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.strip():
            d = json.loads(line)
            label = str(d["label"]).upper()
            if label not in LEVELS:
                raise SystemExit(f"{path}:{i}: label must be one of {LEVELS}")
            cases.append({"id": f"private_{len(cases) + 1:03d}", "text": str(d["text"]), "expected": label,
                          "group": "private", "split": "private"})
    return cases


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", action="store_true", help="call the deployed arbiters (SPENDS credits; needs --budget-usd)")
    p.add_argument("--budget-usd", type=float, default=None, help=f"spend cap, at most {BUDGET_CAP_USD:.2f} (D4)")
    p.add_argument("--replay", metavar="METRICS_JSON", help="recompute verdicts from saved probabilities; no inference")
    p.add_argument("--select", action="store_true", help="grid the policies on the select data; print the frozen choice")
    p.add_argument("--private", metavar="PATH", help="owner JSONL of {text,label}; aggregates only")
    p.add_argument("--audit", action="store_true", help="owner-local blinded agreement audit; writes counts only")
    p.add_argument("--records", default=str(REPO / "logs" / "turn_records.jsonl"))
    p.add_argument("--audit-n", type=int, default=40)
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--e2e", action="store_true")
    p.add_argument("--split", default="select")
    p.add_argument("--policy", default="argmax", choices=["argmax", "weighted", "cumulative"])
    p.add_argument("--params", default="{}", help='JSON, e.g. {"cuts":[0.5,1.5,2.5]} or {"taus":[0.5,0.5,0.5]}')
    p.add_argument("--llm-usd-per-mtok", type=float, default=10.0, help="assumed input price for the LLM cost ESTIMATE")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.e2e:
        sys.stdout.write("--e2e: not implemented in R1 scope (needs the real embedder; owner machine only)\n")
        return 2
    if a.budget_usd is not None and not 0 < a.budget_usd <= BUDGET_CAP_USD:
        sys.stderr.write(f"refusing: --budget-usd must be in (0, {BUDGET_CAP_USD:.2f}] (owner cap D4)\n")
        return 2
    if a.run and a.budget_usd is None:
        sys.stderr.write("refusing: --run requires --budget-usd (at most 2.00)\n")
        return 2
    if a.audit:
        res = audit(a.records, a.audit_n, a.seed, a.out)
        sys.stdout.write(f"audit counts: {json.dumps(res)}\n")
        return 0
    params = json.loads(a.params)
    private = bool(a.private)
    if private:
        sys.stdout.write("*** PRIVATE MODE: owner data. Only aggregate counts are written; no message text is stored or printed. ***\n")
    cases = load_private(a.private) if private else cases_mod.tone_cases(a.split)
    meta = {"split": "private" if private else a.split, "jev_slug": app_config.DECISION_MODEL_SLUG, "llm_model": None,
            "mode": "replay" if a.replay else "run", "stopped": None, "weights": WEIGHTS}
    if a.replay:
        saved = json.loads(Path(a.replay).read_text())
        rows, meta["llm_model"], meta["split"] = saved["cases"], saved["meta"].get("llm_model"), saved["meta"].get("split")
    elif a.run:
        mm = make_model_manager()
        meta["llm_model"] = mm.get_active_model_name()
        with jev_enabled_in_process():
            rows, spent, meta["stopped"] = asyncio.run(run_cases(cases, mm, a.budget_usd, a.llm_usd_per_mtok))
        sys.stdout.write(f"ran {len(rows)}/{len(cases)} cases; estimated spend ${spent:.4f}; stopped: {meta['stopped']}\n")
    else:
        dry_run(cases, a.llm_usd_per_mtok)
        return 0
    s = summarize(rows, a.policy, params)
    out_dir = write_outputs(a.out, meta, rows, s, private=private)
    sys.stdout.write(f"wrote {out_dir}/report{'_private' if private else ''}.md and metrics{'_private' if private else ''}.json\n")
    if a.select:
        choice = select_policy(rows)
        if choice is None:
            sys.stdout.write("SELECT: no policy in the grid has zero severe misses on this data; nothing frozen\n")
            return 1
        sys.stdout.write(f"SELECT: policy {choice['policy']} {json.dumps(choice['params'])}; expected cost {choice['cost']} "
                         f"over {choice['cases']} cases (weights {WEIGHTS}). YAML for config.local.yaml (decision_model:):\n")
        sys.stdout.write("\n".join(yaml_lines(choice)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
