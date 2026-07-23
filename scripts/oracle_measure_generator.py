#!/usr/bin/env python3
"""
Pre-registered measurement of the pooled discovery generator's GENUINE-NOVEL yield,
using the calibrated literature oracle (knowledge/literature_oracle.py).

Corpus (fixes the spec's wrong "27/28-graded" corpus — those are old-pipeline
personal->wiki candidates, out of scope):
  pooled_csv             discovery_grading_MERGED_20260630_140521.csv (34 ACCEPT, 11 QUEUE)
  anchored_json_anchored anchored_gen_validation_20260630_121819.json, 5 accepts
  anchored_json_random   same file, 8 accepts; rejects feed the reject strata

Strata:
  primary   — accepts, NOT pre-seen: the measurement denominator N
  pre_seen  — the 11 blindspot-run candidates (MERGED top-6 + bottom-5 by composite) and
              anything colliding with calibration_set.jsonl: scored, reported separately,
              EXCLUDED from N (prompt-iteration contamination)
  queued    — MERGED QUEUE rows (not accepts; contrast only)
  reject_coherence / reject_composite — 10 + 10 seeded draws from the validation-run
              rejects (contrast: incoherent, expected to skew NOT_FOUND/UNCERTAIN)

Decision rule (COUNTS, not percentages — pre-registered in docs/LITERATURE_ORACLE.md §6,
sign-off enforced below): X = NOT_FOUND among primary; U = UNCERTAIN among primary.
  validity gate: U > 0.25*N or >2 PARSE_FAILUREs  => instrument insufficient, no conclusion
  X <= 2   => ~0 genuine-novel yield confirmed -> fork (b), generator concept-source rethink
  3..7     => owner reviews every NOT_FOUND individually (candidate genuine novels)
  X >= 8   => generator retains yield; re-examine before generator changes

Blinding: manifest rows carry NO gate-derived fields; status/composite/coherence live in
corpus_hidden.jsonl, joined at REPORT time only. The oracle never sees them.

Usage:
  python scripts/oracle_measure_generator.py --build-corpus       # manifest + SHA256
  # ... owner adds "PREREGISTERED: <sha> <date>" to docs/LITERATURE_ORACLE.md ...
  python scripts/oracle_measure_generator.py --dry-run            # cost estimate
  python scripts/oracle_measure_generator.py                      # the measurement
  python scripts/oracle_measure_generator.py --resume measure_X   # continue after budget split
"""
import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

try:  # same env bootstrap as main.py — must precede any config.app_config import
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_REPO, ".env"), override=True)
except ImportError:
    pass

MERGED_CSV = os.path.join(_REPO, "data", "synthesis_discovery",
                          "discovery_grading_MERGED_20260630_140521.csv")
ANCHORED_JSON = os.path.join(_REPO, "data", "synthesis_discovery",
                             "anchored_gen_validation_20260630_121819.json")
CALIBRATION_SET = os.path.join(_REPO, "data", "oracle", "calibration_set.jsonl")
DEFAULT_MANIFEST = os.path.join(_REPO, "data", "oracle", "measurement_corpus.jsonl")
HIDDEN_SIDECAR = os.path.join(_REPO, "data", "oracle", "corpus_hidden.jsonl")
DOCS_PATH = os.path.join(_REPO, "docs", "LITERATURE_ORACLE.md")
RUNS_DIR = os.path.join(_REPO, "data", "oracle", "runs")
ORACLE_CACHE_DIR = os.path.join(_REPO, "data", "oracle", "cache_chroma")

REJECTS_PER_STAGE = 10
DEFAULT_REJECTS_SEED = 20260701


def _claim_hash(claim: str) -> str:
    norm = re.sub(r"\s+", " ", (claim or "").strip().casefold())
    return hashlib.sha1(norm.encode()).hexdigest()[:16]


def _pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a.strip().casefold(), b.strip().casefold())))


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_corpus(manifest_path: str, rejects_seed: int) -> None:
    manifest, hidden = [], []
    seen_claims, seen_pairs = set(), {}

    def add(cid, a, b, claim, source, stratum, pre_seen, hidden_fields):
        ch = _claim_hash(claim)
        if ch in seen_claims:
            print(f"  dedup: dropping {cid} (identical claim already in corpus)")
            return
        pk = _pair_key(a, b)
        if pk in seen_pairs:
            print(f"  note: pair-level duplicate {a} <-> {b} "
                  f"({cid} vs {seen_pairs[pk]}), different claims — keeping both")
        seen_pairs.setdefault(pk, cid)
        seen_claims.add(ch)
        manifest.append({"candidate_id": cid, "concept_a": a, "concept_b": b,
                         "claim_text": claim, "source": source, "stratum": stratum,
                         "pre_seen": pre_seen})
        hidden.append({"candidate_id": cid, **hidden_fields})

    # Calibration collisions (flag 3): anything sharing a pair OR a claim with the
    # calibration set is contaminated by prompt iteration.
    calib_pairs, calib_claims = set(), set()
    if os.path.exists(CALIBRATION_SET):
        with open(CALIBRATION_SET) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    calib_pairs.add(_pair_key(r["concept_a"], r["concept_b"]))
                    calib_claims.add(_claim_hash(r["claim_text"]))

    # --- pooled CSV (MERGED): accepts + queued; top-6/bottom-5 = the blindspot 11 ---
    with open(MERGED_CSV) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: -float(r["composite"]))
    blindspot_ids = {r["pair_id"] for r in rows[:6] + rows[-5:]}
    for r in rows:
        pre_seen = (r["pair_id"] in blindspot_ids
                    or _pair_key(r["concept_a"], r["concept_b"]) in calib_pairs
                    or _claim_hash(r["connection_claim"]) in calib_claims)
        stratum = "queued" if r["status"] == "QUEUE" else ("pre_seen" if pre_seen else "primary")
        add(f"csv_pair{r['pair_id']}", r["concept_a"], r["concept_b"], r["connection_claim"],
            "pooled_csv", stratum, pre_seen,
            {"status": r["status"], "composite": r["composite"],
             "coherence_level": r["coherence_level"], "novelty": r["novelty"]})

    # --- anchored validation JSON: accepts into primary; rejects into contrast strata ---
    with open(ANCHORED_JSON) as f:
        aj = json.load(f)
    reject_pool = {"coherence_judge": [], "composite_scoring": []}
    for arm in ("anchored", "random"):
        for i, c in enumerate(aj[f"{arm}_claims"]):
            if not c.get("claim"):
                continue
            cid = f"{arm}_{i}"
            hidden_fields = {"status": c["stage"], "coherence_level": c.get("level", "")}
            if c["stage"] == "accepted":
                pre_seen = (_pair_key(c["a"], c["b"]) in calib_pairs
                            or _claim_hash(c["claim"]) in calib_claims)
                add(cid, c["a"], c["b"], c["claim"], f"anchored_json_{arm}",
                    "pre_seen" if pre_seen else "primary", pre_seen, hidden_fields)
            elif c["stage"] in reject_pool:
                reject_pool[c["stage"]].append((cid, c, hidden_fields))

    rng = random.Random(rejects_seed)
    for stage, pool in reject_pool.items():
        picks = rng.sample(pool, min(REJECTS_PER_STAGE, len(pool)))
        for cid, c, hidden_fields in picks:
            add(cid, c["a"], c["b"], c["claim"], "anchored_json_rejects",
                f"reject_{stage.split('_')[0]}", False, hidden_fields)

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    with open(HIDDEN_SIDECAR, "w") as f:
        for row in hidden:
            f.write(json.dumps(row) + "\n")

    strata = Counter(r["stratum"] for r in manifest)
    sha = _sha256_file(manifest_path)
    print(f"\nWrote {len(manifest)} rows to {manifest_path}")
    print(f"Strata: {dict(strata)}  (primary N = {strata['primary']})")
    print(f"Hidden sidecar (gate fields, report-time join only): {HIDDEN_SIDECAR}")
    print(f"\nManifest SHA256: {sha}")
    print(f"\nTO PRE-REGISTER, add to {DOCS_PATH} (with the prediction + decision rule):")
    print(f"  PREREGISTERED: {sha} {datetime.now().strftime('%Y-%m-%d')}")


def _check_preregistration(manifest_path: str) -> bool:
    if not os.path.exists(DOCS_PATH):
        return False
    sha = _sha256_file(manifest_path)
    with open(DOCS_PATH) as f:
        for line in f:
            m = re.match(r"\s*PREREGISTERED:\s*([0-9a-f]{64})\s+(\S+)", line)
            if m and m.group(1) == sha:
                print(f"Pre-registration found: {line.strip()}")
                return True
    return False


def write_report(run_dir: str, manifest_path: str, knobs: dict) -> None:
    results_path = os.path.join(run_dir, "results.jsonl")
    records = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    with open(manifest_path) as f:
        stratum_of = {r["candidate_id"]: r["stratum"]
                      for r in (json.loads(l) for l in f if l.strip())}
    hidden = {}
    if os.path.exists(HIDDEN_SIDECAR):
        with open(HIDDEN_SIDECAR) as f:
            hidden = {r["candidate_id"]: r for r in (json.loads(l) for l in f if l.strip())}

    primary = [r for r in records if stratum_of.get(r["candidate_id"]) == "primary"]
    parse_failures = sum(1 for r in records if r["verdict"] == "PARSE_FAILURE")
    usable = [r for r in primary if r["verdict"] != "PARSE_FAILURE"]
    n = len(usable)
    x = sum(1 for r in usable if r["verdict"] == "NOT_FOUND")
    u = sum(1 for r in usable if r["verdict"] == "UNCERTAIN")

    # Completeness guard: X counts NOT_FOUND over the rows that were SCORED, so a
    # budget-stopped / interrupted run silently understates X. The pre-registered
    # X-thresholds only apply to a full manifest run.
    scored_ids = {r["candidate_id"] for r in records}
    missing = [cid for cid in stratum_of if cid not in scored_ids]
    complete = not missing

    lines = [
        f"# Oracle measurement report — {os.path.basename(run_dir)}",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Knobs: {json.dumps(knobs)}",
        "",
        f"## PARSE FAILURES: {parse_failures}",
        "",
        f"## COMPLETENESS: {len(scored_ids)}/{len(stratum_of)} manifest rows scored"
        + ("" if complete else
           "  <-- INCOMPLETE RUN: the decision rule is SUPPRESSED — X over a partial "
           "manifest is not the pre-registered X. Resume the run before reading verdicts."),
        "",
        f"## Primary result: N={n} usable primary accepts, X={x} NOT_FOUND, U={u} UNCERTAIN",
        "",
    ]
    if complete:
        lines += [
            "Decision rule (must match the pre-registered block in docs/LITERATURE_ORACLE.md §6):",
            f"- validity gate: U({u}) > 0.25*N({n}) = {0.25 * n:.1f} or parse_failures({parse_failures}) > 2 "
            f"=> {'INSTRUMENT INSUFFICIENT — no generator conclusion' if (u > 0.25 * n or parse_failures > 2) else 'instrument OK'}",
            f"- X <= 2: ~0 genuine-novel yield | 3..7: owner reviews each NOT_FOUND | >= 8: yield retained",
        ]
    else:
        lines.append(f"(decision rule withheld — {len(missing)} manifest rows unscored: "
                     + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "") + ")")
    lines += [
        "",
        "## Verdicts by stratum",
    ]
    for stratum in ("primary", "pre_seen", "queued", "reject_coherence", "reject_composite"):
        sub = [r for r in records if stratum_of.get(r["candidate_id"]) == stratum]
        if sub:
            lines.append(f"- {stratum} (n={len(sub)}): {dict(Counter(r['verdict'] for r in sub))}")

    survivors = [r for r in usable if r["verdict"] == "NOT_FOUND"]
    if survivors:
        lines.append("\n## NOT_FOUND survivors (candidate genuine novels — owner review)")
        for r in survivors:
            h = hidden.get(r["candidate_id"], {})
            lines.append(f"\n### {r['candidate_id']}: {r['concept_a']} <-> {r['concept_b']}")
            lines.append(f"claim: {r['claim_text']}")
            lines.append(f"(joined at report time: composite={h.get('composite', '?')}, "
                         f"coherence={h.get('coherence_level', '?')})")
            lines.append(f"prior: {r['llm_prior_verdict'] or '(none)'} | queries: {r['queries']}")

    lines.append("\n## Per-item table")
    lines.append("| id | stratum | verdict | name | escalated | credits |")
    lines.append("|---|---|---|---|---|---|")
    for r in sorted(records, key=lambda r: (stratum_of.get(r["candidate_id"], ""), r["candidate_id"])):
        lines.append(f"| {r['candidate_id']} | {stratum_of.get(r['candidate_id'], '?')} "
                     f"| {r['verdict']} | {r.get('verdict_name', '')[:36]} "
                     f"| {r['escalated']} | {r['credits_used']} |")

    report_path = os.path.join(run_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    print(f"\nFull report: {report_path}")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-corpus", action="store_true")
    ap.add_argument("--corpus-manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--rejects-seed", type=int, default=DEFAULT_REJECTS_SEED)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-credits", type=float, default=0.0,
                    help="additional per-run credit cap under the daily limit")
    ap.add_argument("--aux", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.build_corpus:
        build_corpus(args.corpus_manifest, args.rejects_seed)
        return

    with open(args.corpus_manifest) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    from knowledge.literature_oracle import (ORACLE_WORST_CASE_CREDITS_PER_CANDIDATE,
                                             append_record, check_budget, load_done_ids)

    if args.dry_run:
        strata = Counter(r["stratum"] for r in rows)
        worst = len(rows) * ORACLE_WORST_CASE_CREDITS_PER_CANDIDATE
        print(f"Corpus: {len(rows)} rows, strata {dict(strata)}")
        print(f"Manifest SHA256: {_sha256_file(args.corpus_manifest)}")
        print(f"Pre-registered: {_check_preregistration(args.corpus_manifest)}")
        print(f"Cost: worst-case {worst:.0f} Tavily credits "
              f"(~{worst / float(os.getenv('ORACLE_TAVILY_DAILY', '100')):.1f} daily budgets), "
              f"typical ~{len(rows) * 3:.0f}; LLM calls ~{len(rows) * 2}-{len(rows) * 3}.")
        return

    if not _check_preregistration(args.corpus_manifest):
        print(f"REFUSING TO RUN: no 'PREREGISTERED: <sha256> <date>' line in {DOCS_PATH} "
              f"matching this manifest.")
        print("Write the pre-registration block (corpus SHA, prediction, decision rule) FIRST, "
              "then re-run. This is mechanical enforcement of 'registered before running'.")
        sys.exit(2)

    run_id = args.resume or args.run_id or f"measure_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.jsonl")

    from knowledge.literature_oracle import LiteratureOracle
    from knowledge.web_search_manager import (WebSearchCache, WebSearchManager,
                                              WebSearchRateLimiter)
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
    from models.model_manager import ModelManager

    done = load_done_ids(results_path)
    todo = [r for r in rows if r["candidate_id"] not in done]
    print(f"[{run_id}] {len(todo)} to score ({len(done)} already done)")

    from config.app_config import WEB_SEARCH_API_KEY
    limiter = WebSearchRateLimiter(
        daily_limit=int(os.getenv("ORACLE_TAVILY_DAILY", "100")))
    cache = WebSearchCache(chroma_store=MultiCollectionChromaStore(
        persist_directory=ORACLE_CACHE_DIR))  # isolated — no production chroma writes
    wsm = WebSearchManager(api_key=WEB_SEARCH_API_KEY, rate_limiter=limiter, cache=cache)
    if not WEB_SEARCH_API_KEY:
        print("WARNING: no Tavily API key (config web_search.api_key / TAVILY_API_KEY) — "
              "running arXiv-only. ABORT unless this is intentional.")
    oracle = LiteratureOracle(ModelManager(), web_search_manager=wsm, enable_aux=args.aux)
    knobs = {"model": oracle.model_name, "aux": args.aux,
             "rejects_seed": args.rejects_seed,
             "manifest_sha": _sha256_file(args.corpus_manifest)}

    spent_this_run = 0.0
    for i, row in enumerate(todo, 1):
        remaining = check_budget(limiter)
        if remaining < ORACLE_WORST_CASE_CREDITS_PER_CANDIDATE or (
                args.max_credits and
                spent_this_run + ORACLE_WORST_CASE_CREDITS_PER_CANDIDATE > args.max_credits):
            print(f"\nBUDGET GUARD: stopping cleanly ({remaining:.1f} daily credits left, "
                  f"{spent_this_run:.1f} spent this run).")
            print(f"Resume with: python scripts/oracle_measure_generator.py --resume {run_id}")
            break
        # Blinding: the oracle receives ONLY manifest fields — never status/composite/coherence.
        rec = await oracle.score_candidate(
            row["candidate_id"], row["concept_a"], row["concept_b"], row["claim_text"],
            run_id=run_id, source=row["source"], pre_seen=row["pre_seen"],
        )
        spent_this_run += rec.credits_used
        append_record(results_path, rec)
        print(f"  [{i}/{len(todo)}] {row['candidate_id']:<22} [{row['stratum']:<17}] "
              f"-> {rec.verdict.value:<20} (credits {rec.credits_used})", flush=True)

    write_report(run_dir, args.corpus_manifest, knobs)


if __name__ == "__main__":
    asyncio.run(main())
