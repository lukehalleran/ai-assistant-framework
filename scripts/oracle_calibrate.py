#!/usr/bin/env python3
"""
Calibrate the literature oracle (knowledge/literature_oracle.py) against a labeled set
BEFORE any measurement use. See docs/LITERATURE_ORACLE.md for the protocol.

Set composition (data/oracle/calibration_set.jsonl, regenerable via --init-set):
  positive_arxiv    — the 11 candidates the 2026-06-30 blindspot run confirmed DOCUMENTED
                      (reconstructed: MERGED grading CSV, top-6 + bottom-5 by composite).
  positive_textbook — ~15 hand-listed textbook identities, claim_text written in
                      pooled-generator paraphrase style (tests paraphrase-evasion).
  negative_mutated  — mutated identities (one side swapped for a plausible-but-wrong
                      neighbor). Every row ships owner_verified=false and this script
                      REFUSES to score negatives until a human verifies each label
                      (a mutation we invented can accidentally be real — e.g. Kelly's
                      detailed-balance<->queueing, which is documented and therefore
                      lives in the POSITIVES here).

Pre-registered acceptance thresholds (docs/LITERATURE_ORACLE.md section 4):
  positives:  >= 90% verdict in {DOCUMENTED_EXACT, DOCUMENTED_ADJACENT}
              >= 75% DOCUMENTED_EXACT specifically
  negatives:  <= 10% DOCUMENTED_EXACT
  parse failures: 0 tolerated in the final calibration run (counted LOUDLY regardless)

Usage:
  python scripts/oracle_calibrate.py --init-set          # (re)generate the labeled set
  python scripts/oracle_calibrate.py                     # run calibration
  python scripts/oracle_calibrate.py --resume calib_X    # continue an interrupted run
  python scripts/oracle_calibrate.py --only-label positive_textbook --limit 3   # spot run

Costs Tavily credits (shared production budget; ORACLE_TAVILY_DAILY caps this script's
view of the daily limit) + 2-3 LLM calls per item. Budget guard splits runs cleanly.
"""
import argparse
import asyncio
import csv
import json
import os
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
DEFAULT_SET = os.path.join(_REPO, "data", "oracle", "calibration_set.jsonl")
RUNS_DIR = os.path.join(_REPO, "data", "oracle", "runs")
ORACLE_CACHE_DIR = os.path.join(_REPO, "data", "oracle", "cache_chroma")

# --------------------------------------------------------------------------
# (b) Textbook positives — claim_text deliberately PARAPHRASED (pooled-generator
# style), never the canonical wording: calibration must test the paraphrase-evasion
# countermeasure, not string matching. All are established results; owner confirms
# known_name per row and flips owner_verified.
# --------------------------------------------------------------------------
TEXTBOOK_POSITIVES = [
    ("legendre_thermo_mech", "thermodynamics", "classical mechanics",
     "Swapping which variable is 'natural' in a thermodynamic potential and trading velocities "
     "for momenta in mechanics are the same mathematical move: both replace a function by one of "
     "a conjugate variable while preserving all information, so free-energy relations and the "
     "Hamiltonian-Lagrangian passage are instances of one transform.",
     "Legendre transform"),
    ("ising_lattice_gas", "Ising model", "lattice gas",
     "A magnet of up/down spins on a lattice and a fluid of particles that either occupy or "
     "vacate lattice cells are the same statistical system under a relabeling: spin orientation "
     "maps to site occupancy and the field maps to chemical potential, so their phase diagrams "
     "coincide.",
     "Ising-lattice gas equivalence (Lee-Yang)"),
    ("feynman_kac", "diffusion", "probability theory",
     "Solving a heat-type partial differential equation can be replaced by averaging a functional "
     "over random particle paths: the PDE's solution at a point equals the expected value of the "
     "terminal condition along Brownian trajectories started there, with a path-weight for the "
     "potential term.",
     "Feynman-Kac formula"),
    ("black_scholes_heat", "option pricing", "heat conduction",
     "The equation governing a financial option's fair value is, after a change of variables, "
     "the same equation that governs how temperature spreads through a metal bar; volatility "
     "plays the role of thermal diffusivity.",
     "Black-Scholes to heat equation transformation"),
    ("matrix_tree", "graph theory", "linear algebra",
     "Counting how many spanning trees a network has reduces to a determinant: delete any row "
     "and column of the network's Laplacian matrix and the determinant of what remains equals "
     "the number of spanning trees exactly.",
     "Kirchhoff's matrix-tree theorem"),
    ("codes_sphere_packing", "error-correcting codes", "sphere packing",
     "Designing a code that survives noise and packing non-overlapping spheres in a high-"
     "dimensional space are the same optimization: codewords are sphere centers in Hamming "
     "space, minimum distance is sphere radius, and the impossibility bounds on code rates "
     "are packing-density bounds.",
     "Hamming bound / sphere-packing bound"),
    ("annealing_metallurgy", "simulated annealing", "metallurgy",
     "The optimization heuristic that occasionally accepts worse solutions with a temperature-"
     "controlled probability is a direct computational transplant of how slowly cooled metals "
     "reach low-energy crystal configurations, including the Boltzmann acceptance rule and the "
     "cooling schedule.",
     "Kirkpatrick et al. simulated annealing"),
    ("walks_electrical", "random walk", "electrical circuit",
     "Questions about where a randomly wandering particle ends up translate exactly into "
     "circuit quantities: the probability of reaching one boundary before another is a voltage, "
     "and the expected number of visits is governed by effective resistance between nodes.",
     "random walks and electric networks (Doyle-Snell)"),
    ("pagerank_markov", "PageRank", "Markov chain",
     "Ranking web pages by importance is computing where a random surfer spends its time in the "
     "long run: the ranking vector is the stationary distribution of a stochastic matrix built "
     "from the link graph plus teleportation.",
     "PageRank as Markov stationary distribution"),
    ("fourier_heisenberg", "signal processing", "quantum mechanics",
     "The impossibility of localizing a signal simultaneously in time and frequency and the "
     "impossibility of pinning both position and momentum of a particle are the same theorem "
     "about a function and its Fourier transform; the quantum version just attaches physical "
     "units to the bandwidth product.",
     "Fourier uncertainty principle / Heisenberg"),
    ("kelly_queueing", "statistical mechanics", "queueing theory",
     "The no-net-flux condition that characterizes equilibrium ensembles also characterizes a "
     "large class of queueing networks: when each station's flow balances in detail, the joint "
     "queue-length distribution factorizes into a product form, exactly as reversible physical "
     "systems factorize.",
     "Kelly networks / reversibility and product form"),
    ("percolation_potts", "percolation theory", "Potts model",
     "Random connectivity of edges and an interacting spin model are the same object in a "
     "limit: the cluster expansion of the q-state spin system reproduces bond percolation as "
     "q approaches one, unifying their critical behavior.",
     "Fortuin-Kasteleyn representation"),
    ("wick_rotation", "quantum field theory", "statistical mechanics",
     "Evolving a quantum system in imaginary time turns its dynamical amplitudes into a "
     "partition function: quantum expectation values in d dimensions become classical "
     "statistical averages in d+1, with temperature as the inverse extent of the imaginary-"
     "time direction.",
     "Wick rotation / Euclidean QFT-statistical mechanics correspondence"),
    ("nash_fixed_point", "game theory", "topology",
     "The guarantee that every finite game has a stable strategy profile is a fixed-point "
     "statement: best-response correspondences satisfy the hypotheses of a topological "
     "fixed-point theorem, and the equilibrium is exactly its fixed point.",
     "Nash equilibrium via Kakutani fixed-point theorem"),
    ("hopfield_spin_glass", "neural network", "spin glass",
     "Associative memory networks that store patterns in symmetric weights are magnetic "
     "systems with quenched disorder: retrieval states are free-energy minima, storage "
     "capacity is a phase boundary, and the analysis imports mean-field spin-glass methods "
     "wholesale.",
     "Hopfield network as spin glass (Amit-Gutfreund-Sompolinsky)"),
]

# --------------------------------------------------------------------------
# (c) Mutated-identity negatives — one side of a real identity swapped for a
# plausible-but-wrong neighbor; fluent vocabulary, wrong specific claim.
# EVERY row starts owner_verified=false and negatives are NOT scored until a
# human checks each is genuinely undocumented (label error != oracle error).
# Correct oracle behavior: anything EXCEPT DOCUMENTED_EXACT.
# --------------------------------------------------------------------------
MUTATED_NEGATIVES = [
    ("mut_backprop_kalman", "neural network", "Kalman smoother",
     "Training a neural network by backpropagation is mathematically identical to running a "
     "discrete-time Kalman smoother: the backward error pass is the smoothing recursion, and "
     "the learning rate plays the role of the Kalman gain.",
     "mutation of: backprop = Pontryagin adjoint method"),
    ("mut_blackscholes_wave", "option pricing", "wave propagation",
     "The Black-Scholes equation transforms into the classical wave equation under a change of "
     "variables, so option prices propagate along characteristics at a speed set by volatility, "
     "exactly like acoustic wavefronts.",
     "mutation of: Black-Scholes = heat equation (parabolic, not hyperbolic)"),
    ("mut_matrixtree_permanent", "graph theory", "linear algebra",
     "Counting spanning trees of a network reduces to the permanent of its adjacency matrix: "
     "the permanent's monomial expansion enumerates the trees directly.",
     "mutation of: matrix-tree theorem (determinant of Laplacian, not permanent of adjacency)"),
    ("mut_huffman_spheres", "Huffman coding", "sphere packing",
     "The optimality bound of Huffman source codes is the sphere-packing bound: prefix-free "
     "codewords are non-overlapping spheres in Hamming space and entropy is the packing density "
     "limit.",
     "mutation of: channel-coding sphere-packing bound (wrong side: source vs channel coding)"),
    ("mut_feynman_kac_wave", "wave propagation", "probability theory",
     "Solutions of the classical wave equation equal expectations of boundary data along "
     "standard Brownian motion paths, the same Feynman-Kac construction used for the heat "
     "equation.",
     "mutation of: Feynman-Kac (parabolic PDEs; fails for hyperbolic)"),
    ("mut_ising_ideal_gas", "Ising model", "ideal gas",
     "There is an exact mapping between the Ising magnet and the classical ideal gas: spin "
     "orientation corresponds to particle presence and the ideal-gas equation of state "
     "reproduces the magnetization curve.",
     "mutation of: Ising = lattice gas (interacting; ideal gas has no interactions/transition)"),
    ("mut_pagerank_laplacian", "PageRank", "graph Laplacian",
     "The PageRank vector is the Perron eigenvector of the graph Laplacian: importance flows "
     "along the Laplacian's principal mode, with teleportation as a diagonal shift.",
     "mutation of: PageRank = stationary distribution of a stochastic matrix (not Laplacian)"),
    ("mut_annealing_ostwald", "simulated annealing", "Ostwald ripening",
     "Simulated annealing is a computational implementation of Ostwald ripening: accepted "
     "worse moves correspond to small crystallites dissolving, and the cooling schedule "
     "reproduces coarsening's power-law growth of grain size.",
     "mutation of: annealing metaphor (Kirkpatrick); ripening is a different process"),
    ("mut_legendre_action", "thermodynamics", "action principle",
     "The Legendre transform maps entropy directly to the classical action functional, so "
     "extremizing action and maximizing entropy are the same variational computation.",
     "category-mash mutation of: Legendre transform between potentials/Lagrangian-Hamiltonian"),
    ("mut_saw_electrical", "self-avoiding walk", "electrical circuit",
     "Self-avoiding walk statistics translate exactly into circuit quantities: the number of "
     "self-avoiding paths between two nodes equals the effective conductance between them.",
     "mutation of: SIMPLE random walks = electrical networks (SAW has no such correspondence)"),
    ("mut_tcp_detailed_balance", "statistical mechanics", "congestion control",
     "TCP's additive-increase multiplicative-decrease dynamics satisfy detailed balance: the "
     "steady-state window distribution is the equilibrium of a reversible Markov chain, making "
     "internet congestion an equilibrium ensemble.",
     "mutation of: detailed balance = reversible Markov chains (AIMD is not reversible)"),
    ("mut_hinf_bargaining", "control theory", "bargaining theory",
     "Robust H-infinity control is mathematically identical to a Nash bargaining problem: the "
     "controller and disturbance jointly maximize the product of their utility gains over the "
     "disagreement point.",
     "mutation of: H-infinity = two-player ZERO-SUM differential game (not cooperative)"),
    ("mut_rg_nash", "renormalization group", "game theory",
     "Fixed points of renormalization-group coarse-graining are Nash equilibria of an "
     "underlying potential game: block-spin transformations are best-response updates, so "
     "criticality is equilibrium selection.",
     "novel-sounding mash of two real frameworks; no documented identity"),
    ("mut_pontryagin_kmeans", "optimal control", "k-means clustering",
     "The k-means assignment step is the Pontryagin adjoint recursion: cluster "
     "responsibilities are costates and the centroid update integrates the adjoint equation "
     "backward.",
     "mutation of: backprop = Pontryagin adjoint (k-means is coordinate descent, no costate)"),
    ("mut_capacity_carnot", "information theory", "thermodynamics",
     "Shannon channel capacity IS the Carnot efficiency of an information engine: the capacity "
     "formula log(1+SNR) equals one minus the ratio of noise temperature to signal temperature, "
     "making coding theory a heat-engine bound.",
     "mutation in the dense info-thermo literature; specific identity not established"),
]


def build_init_set(path: str) -> None:
    """(Re)generate the calibration set: 11 reconstructed arXiv positives + drafts."""
    rows = []
    with open(MERGED_CSV) as f:
        merged = sorted(csv.DictReader(f), key=lambda r: -float(r["composite"]))
    for r in merged[:6] + merged[-5:]:
        rows.append({
            "id": f"arxiv_pos_pair{r['pair_id']}",
            "kind": "positive_arxiv",
            "concept_a": r["concept_a"], "concept_b": r["concept_b"],
            "claim_text": r["connection_claim"],
            "expected": "DOCUMENTED",
            "known_name": "",
            "owner_verified": False,  # owner confirms the reconstruction (see docs)
            "provenance": "blindspot_20260630_merged_top6_bottom5",
        })
    for cid, a, b, claim, name in TEXTBOOK_POSITIVES:
        rows.append({
            "id": f"tb_pos_{cid}", "kind": "positive_textbook",
            "concept_a": a, "concept_b": b, "claim_text": claim,
            "expected": "DOCUMENTED", "known_name": name,
            "owner_verified": False, "provenance": "hand_listed_20260701",
        })
    for cid, a, b, claim, note in MUTATED_NEGATIVES:
        rows.append({
            "id": cid, "kind": "negative_mutated",
            "concept_a": a, "concept_b": b, "claim_text": claim,
            "expected": "UNDOCUMENTED", "known_name": "",
            "owner_verified": False, "provenance": note,
        })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    n = Counter(r["kind"] for r in rows)
    print(f"Wrote {len(rows)} rows to {path}: {dict(n)}")
    print("NEXT: owner verifies EVERY negative_mutated row (and the arXiv-11 reconstruction),")
    print("      sets owner_verified=true per row. Calibration refuses unverified negatives.")


def write_report(run_dir: str, knobs: dict, set_path: str = DEFAULT_SET) -> bool:
    """Compute thresholds over the run's results.jsonl; returns overall PASS.

    A threshold-valid PASS additionally requires COMPLETENESS: every row of the
    calibration set scored and all three strata present. Without this, an empty or
    partial stratum passes its rate check vacuously (e.g. no negatives -> neg_exact
    = 0.0 -> "PASS"), so --only-label / --limit / budget-stopped runs could sign
    the gate by accident."""
    results_path = os.path.join(run_dir, "results.jsonl")
    records = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    pos = [r for r in records if r["calibration_label"] in ("positive_arxiv", "positive_textbook")]
    neg = [r for r in records if r["calibration_label"] == "negative_mutated"]
    parse_failures = sum(1 for r in records if r["verdict"] == "PARSE_FAILURE")

    try:
        with open(set_path) as f:
            expected = [json.loads(l) for l in f if l.strip()]
    except OSError:
        expected = []
    scored_ids = {r["candidate_id"] for r in records}
    missing = [r["id"] for r in expected if r["id"] not in scored_ids]
    strata = Counter(r["calibration_label"] for r in records)
    complete = (bool(expected) and not missing
                and all(strata.get(k) for k in ("positive_arxiv", "positive_textbook",
                                                "negative_mutated")))

    def rate(items, pred):
        return (sum(1 for r in items if pred(r)) / len(items)) if items else 0.0

    pos_doc = rate(pos, lambda r: r["verdict"] in ("DOCUMENTED_EXACT", "DOCUMENTED_ADJACENT"))
    pos_exact = rate(pos, lambda r: r["verdict"] == "DOCUMENTED_EXACT")
    neg_exact = rate(neg, lambda r: r["verdict"] == "DOCUMENTED_EXACT")

    checks = [
        ("positives DOCUMENTED_* recall >= 90%", pos_doc >= 0.90, f"{pos_doc:.0%} (n={len(pos)})"),
        ("positives DOCUMENTED_EXACT >= 75%", pos_exact >= 0.75, f"{pos_exact:.0%}"),
        ("mutated negatives DOCUMENTED_EXACT <= 10%", neg_exact <= 0.10,
         f"{neg_exact:.0%} (n={len(neg)})"),
        ("parse failures == 0", parse_failures == 0, str(parse_failures)),
    ]
    overall = all(ok for _, ok, _ in checks) and complete

    lines = [
        f"# Oracle calibration report — {os.path.basename(run_dir)}",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Knobs: {json.dumps(knobs)}",
        "",
        f"## PARSE FAILURES: {parse_failures}"
        + ("  <-- MUST BE 0 FOR A VALID CALIBRATION" if parse_failures else ""),
        "",
        f"## COMPLETENESS: {len(scored_ids)}/{len(expected)} calibration rows scored"
        + ("" if complete else
           "  <-- INCOMPLETE: partial/--only-label/--limit/budget-stopped run — "
           "rate checks below are NOT threshold-valid and OVERALL cannot PASS"),
    ]
    if missing:
        lines.append("Missing ids: " + ", ".join(missing[:20])
                      + (" ..." if len(missing) > 20 else ""))
    lines += [
        "",
        "## Thresholds (pre-registered, docs/LITERATURE_ORACLE.md)",
    ]
    for name, ok, val in checks:
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] {name}: {val}")
    lines.append(f"\n**OVERALL: {'PASS' if overall else 'FAIL (INCOMPLETE)' if all(ok for _, ok, _ in checks) else 'FAIL'}**\n")

    lines.append("## Verdict distribution")
    for label in ("positive_arxiv", "positive_textbook", "negative_mutated"):
        sub = [r for r in records if r["calibration_label"] == label]
        lines.append(f"- {label} (n={len(sub)}): "
                     f"{dict(Counter(r['verdict'] for r in sub))}")

    lines.append("\n## Per-item results")
    lines.append("| id | label | verdict | name | downgraded | escalated | credits |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sorted(records, key=lambda x: x["candidate_id"]):
        lines.append(
            f"| {r['candidate_id']} | {r['calibration_label']} | {r['verdict']} "
            f"| {r.get('verdict_name', '')[:36]} | {r['downgraded_from_prior']} "
            f"| {r['escalated']} | {r['credits_used']} |")

    # Every miss with its evidence, for owner review (spec requirement).
    misses = [r for r in pos if r["verdict"] not in ("DOCUMENTED_EXACT", "DOCUMENTED_ADJACENT")]
    misses += [r for r in neg if r["verdict"] == "DOCUMENTED_EXACT"]
    if misses:
        lines.append("\n## Misses (FN positives / FP negatives) with evidence")
        for r in misses:
            lines.append(f"\n### {r['candidate_id']} — {r['verdict']}")
            lines.append(f"claim: {r['claim_text'][:220]}")
            lines.append(f"prior: {r['llm_prior_verdict']} | downgraded: "
                         f"{r['downgraded_from_prior']} | escalated: {r['escalated']}")
            for c in r.get("evidence_citations", []):
                lines.append(f"  - cite: {c['url']} :: \"{c['supporting_passage'][:120]}\"")
            for e in r.get("evidence", [])[:4]:
                lines.append(f"  - ev[{e['source']}]: {e['title'][:80]} ({e['url']})")

    report_path = os.path.join(run_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:20]))
    print(f"\nFull report: {report_path}")
    return overall


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=DEFAULT_SET)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--resume", default=None, help="run_id of an interrupted run to continue")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-label", choices=["positive_arxiv", "positive_textbook",
                                             "negative_mutated"], default=None)
    ap.add_argument("--verdict-first", type=int, choices=[0, 1], default=1)
    ap.add_argument("--no-escalation", action="store_true")
    ap.add_argument("--aux", action="store_true",
                    help="also log doc_cooccurrence (wiki FAISS — consider running under "
                         "systemd-run --user --scope -p MemoryMax=9G)")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="score negatives with owner_verified=false (NON-THRESHOLD runs only)")
    ap.add_argument("--init-set", action="store_true")
    args = ap.parse_args()

    if args.init_set:
        build_init_set(args.set)
        return

    with open(args.set) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if args.only_label:
        rows = [r for r in rows if r["kind"] == args.only_label]

    unverified_neg = [r["id"] for r in rows
                      if r["kind"] == "negative_mutated" and not r.get("owner_verified")]
    if unverified_neg and not args.allow_unverified:
        print("REFUSING TO RUN: these negative_mutated rows are not owner-verified "
              "(a mutation we invented can accidentally be a real documented identity):")
        for i in unverified_neg:
            print(f"  - {i}")
        print("Verify each label, set owner_verified=true in the set file, and re-run. "
              "(--allow-unverified exists for non-threshold spot runs only.)")
        sys.exit(2)

    if args.limit:
        rows = rows[: args.limit]

    run_id = args.resume or args.run_id or f"calib_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.jsonl")

    from knowledge.literature_oracle import (LiteratureOracle, append_record, check_budget,
                                             load_done_ids,
                                             ORACLE_WORST_CASE_CREDITS_PER_CANDIDATE)
    from knowledge.web_search_manager import (WebSearchCache, WebSearchManager,
                                              WebSearchRateLimiter)
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
    from models.model_manager import ModelManager

    done = load_done_ids(results_path)
    todo = [r for r in rows if r["id"] not in done]
    print(f"[{run_id}] {len(todo)} to score ({len(done)} already done)")
    if not todo:
        write_report(run_dir, knobs={"resumed": True}, set_path=args.set)
        return

    from config.app_config import WEB_SEARCH_API_KEY
    limiter = WebSearchRateLimiter(
        daily_limit=int(os.getenv("ORACLE_TAVILY_DAILY", "100")))  # shared production budget
    cache = WebSearchCache(chroma_store=MultiCollectionChromaStore(
        persist_directory=ORACLE_CACHE_DIR))  # ISOLATED cache — no production chroma writes
    wsm = WebSearchManager(api_key=WEB_SEARCH_API_KEY, rate_limiter=limiter, cache=cache)
    if not WEB_SEARCH_API_KEY:
        print("WARNING: no Tavily API key (config web_search.api_key / TAVILY_API_KEY) — "
              "running arXiv-only, textbook recall will suffer.")
    oracle = LiteratureOracle(
        ModelManager(), web_search_manager=wsm,
        verdict_first=bool(args.verdict_first),
        escalate=not args.no_escalation,
        enable_aux=args.aux,
    )
    knobs = {"verdict_first": bool(args.verdict_first), "escalate": not args.no_escalation,
             "model": oracle.model_name, "aux": args.aux,
             "allow_unverified": args.allow_unverified, "only_label": args.only_label}

    for i, row in enumerate(todo, 1):
        remaining = check_budget(limiter)
        if remaining < ORACLE_WORST_CASE_CREDITS_PER_CANDIDATE:
            print(f"\nBUDGET GUARD: {remaining:.1f} Tavily credits left "
                  f"(< {ORACLE_WORST_CASE_CREDITS_PER_CANDIDATE} worst case). Stopping cleanly.")
            print(f"Resume tomorrow with: python scripts/oracle_calibrate.py --resume {run_id}")
            break
        rec = await oracle.score_candidate(
            row["id"], row["concept_a"], row["concept_b"], row["claim_text"],
            run_id=run_id, source="calibration", calibration_label=row["kind"],
        )
        append_record(results_path, rec)
        print(f"  [{i}/{len(todo)}] {row['id']:<28} -> {rec.verdict.value:<20} "
              f"(credits {rec.credits_used}, escalated={rec.escalated})", flush=True)

    write_report(run_dir, knobs, set_path=args.set)


if __name__ == "__main__":
    asyncio.run(main())
