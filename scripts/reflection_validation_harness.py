#!/usr/bin/env python3
"""
Phase 2 of the reflection personal-domain-distance signal: the VALIDATION HARNESS.

Architecture under test:  GROUNDED-TRUTH  ×  DOMAIN-DISTANCE  ×  SELF-NOVELTY
This harness validates the first two axes against HAND LABELS and runs the
orthogonality test — the single most important number — that decides whether the
unbundling (truth and value as SEPARATE axes) is real or a fiction.

Substrate (LOCKED 2026-06-29, approved by the owner):
  - entity-pooled, UMAP+fact-text, personal-only clusters from
    reflection_domain_clustering.py (loaded from its latest mcs=5 JSON artifact);
  - reflection SOURCES = LIFE domains only (project/coding domains excluded);
  - DISTANCE metric = same-cluster gate + entity-to-entity BGE cosine for ranking
    (centroid distance washes out at <0.09 — entity-cosine gives the +0.10 spread);
  - distance computed in ORIGINAL BGE cosine space (UMAP coords are non-metric).

What it does (generate mode, default):
  1. sample ~40 entity pairs stratified across the distance spectrum
     (same-domain / cross-near / cross-far) from clean LIFE entities, + the 6 gold
     anchors as known control points;
  2. for each pair run a PAIR-CONDITIONED generator (one best reflection, or NONE),
     then the unbundled TRUTH + DUAL-GROUNDING gate (mechanism-valid? each half
     traceable to that entity's REAL memory? — the anti-confabulation/anti-makyo gate);
  3. emit a CSV + JSON for HAND-LABELING on two SEPARATE columns:
        label_truth  = mechanism true & dual-grounded?  (0/1)
        label_value  = valuable / non-thin?              (0/1)
     (optional label_want_surfaced = would you want this surfaced? 0/1)

What it does (analyze mode, `--analyze <labeled.csv>`): runs the 4 CHECKS:
  1. grounded-truth (gate) predicts label_truth         (gate calibration)
  2. domain-distance (ent_dist) predicts label_value    (the distance signal)
  3. product (truth-pass × distance) predicts want_surfaced
  4. ORTHOGONALITY — are label_truth and label_value statistically independent?
     phi / chi-square / mutual information.  This is the number that matters.

Read-only on production memory (graph + facts collection). No production writes —
only re-embeds entity text and makes generator/gate LLM calls. Artifacts -> CSV+JSON.
Usage:
  python scripts/reflection_validation_harness.py [n_per_stratum]      # generate
  python scripts/reflection_validation_harness.py --analyze <file.csv> # score labels
"""
import asyncio
import csv
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

# Reuse the LOCKED substrate's helpers (import-safe after the argv guard).
from scripts.reflection_domain_clustering import (
    load_facts_map, collect_entities, load_embedder, _clean,
)

SEED = 42
N_PER_STRATUM = int(sys.argv[1]) if (len(sys.argv) > 1 and sys.argv[1].isdigit()) else 13
GATE_CONCURRENCY = 4
MIN_RICHNESS = 2   # entity text must have >= 2 context/fact sentences (real material)

# Pairability filters (the substrate is locked; these only decide which entities are
# eligible to DRAW pairs from — gold anchors bypass them as control points).
# mention_count is NOT usable (graph metadata barely populated: median 1, max 4).
#  (1) cluster-label denylist: whole LIFE-labelled clusters that are really mood/chat/
#      temporal/status scrap, not life domains.
_JUNK_CLUSTER_RE = re.compile(
    r"\bmood|\bstate|casual|\bchat\b|schedule|time of day|task status|\bstatus\b|\bdates?\b"
    r"|daemon|demo video",  # mixed project-leak clusters (Daemon/FAISS contaminate the "& friends")
    re.IGNORECASE)
#  (2) entity POS/NER filter: a real pairable entity is a concrete NOUN/PROPN, not an
#      adjective/adverb/verb state ("easy","warm","submitted","recently") or a bare
#      temporal/quantity token ("December","noon","Wed 2026-04-19").
_BAD_ENT_TYPES = {"DATE", "TIME", "ORDINAL", "CARDINAL", "QUANTITY", "PERCENT", "MONEY"}


def _pairable(name: str, nlp) -> bool:
    if nlp is None:
        return True
    doc = nlp(name.replace("_", " "))
    toks = [t for t in doc if not t.is_punct and not t.is_space]
    if not toks:
        return False
    if any(t.ent_type_ in _BAD_ENT_TYPES for t in toks):
        return False
    root = next((t for t in doc if t.head == t), toks[-1])
    return root.pos_ in ("NOUN", "PROPN")

# Gold anchors (same as Phase 1) carried through as control points.
GOLD_ANCHORS = [
    ("REAL", "dad", "kavarin"),
    ("REAL", "dad", "gym"),
    ("REAL", "brother", "isye 6501"),
    ("THIN", "brother", "sister"),
    ("THIN", "kavarin", "dexivar"),
    ("THIN", "gym", "lift"),
]

# ─────────────────────────── generator (pair-conditioned) ───────────────────────────
GEN_SYSTEM = (
    "You generate ONE candidate REFLECTION for a specific pair of areas in one person's "
    "life, using ONLY observations taken from their own memory. A reflection names a single "
    "underlying mechanism — a tunable factor whose change would alter behavior — that "
    "operates in BOTH areas. You are a generator, not a judge; a separate gate will reject "
    "weak ones. Use only the given observations as biographical fact; invent nothing. If the "
    "two areas share no real mechanism, say so honestly rather than forcing a connection."
)
GEN_PROMPT = """Two specific areas of one person's life, each with real observations from their own memory:

AREA A — {name_a}:
{obs_a}

AREA B — {name_b}:
{obs_b}

Propose the SINGLE best reflection connecting these two areas: one underlying mechanism
(a tunable factor whose change predicts a change in behavior) that genuinely runs in BOTH.
Write the claim in the first person, naming both areas and the differential prediction the
mechanism makes (which case is worse, or how they reinforce over time).

Do NOT invent biographical facts beyond the observations above. If the two areas honestly
share no real transferable mechanism (only a shared mood, category, or surface theme),
return claim "NONE".

Return ONLY JSON: {{"mechanism": "<the tunable factor>", "claim": "<first-person reflection, or NONE>"}}
"""

# ─────────────────────── truth + dual-grounding gate (combined) ───────────────────────
GATE_SYSTEM = (
    "You assess a candidate REFLECTION on TWO independent axes and report both. "
    "(1) MECHANISM VALIDITY: does the claim name a real GENERATIVE mechanism — a tunable "
    "factor whose change would make the person behave differently in circumstances BEYOND the "
    "two observed instances (a real counterfactual) — as opposed to a mere RE-DESCRIPTION of "
    "two behaviors linked by a shared feeling, category, or theme? The two areas are EXPECTED "
    "to be independent parts of one life and the mechanism is EXPECTED to be a known, general "
    "human pattern; do NOT require the two areas to be coupled or to predict each other. "
    "(2) DUAL GROUNDING: is each half of the claim actually supported by that area's provided "
    "real observations, or did the claim invent biographical detail not present there? This is "
    "an anti-confabulation check — be strict: a half is grounded only if the observations "
    "actually evidence the behavior the claim attributes to that area."
)
GATE_PROMPT = """Candidate reflection connecting two areas of one person's life.

AREA A — {name_a} — real observations:
{obs_a}

AREA B — {name_b} — real observations:
{obs_b}

CLAIM: {claim}

Step 1 — DE-JARGON: strip figurative language; state the underlying mechanism plainly.
Step 2 — COUNTERFACTUAL: name the tunable factor; state "if <factor> were <different>, the
  person would <act differently>" in a circumstance NOT already described. If you cannot, the
  mechanism is a theme, not generative.
Step 3 — GROUNDING: for EACH area, is the behavior the claim attributes to it actually
  evidenced by that area's observations above? (yes = grounded; no = invented/unsupported).

Respond EXACTLY in this form:
RATING: VALID    (mechanism is generative)  or  INVALID  (re-description / theme only)
GROUNDED_A: yes  or  no
GROUNDED_B: yes  or  no
DE-JARGON: <one line>
COUNTERFACTUAL: <the if-then, or "none">
REASON: <one line>"""


def _richness(text: str) -> int:
    return text.count(" . ")


def load_substrate():
    """Re-embed personal entities (BGE) and attach LOCKED cluster labels from the artifact."""
    from config.app_config import CHROMA_PATH
    from memory.graph_memory import GraphMemory
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

    art = sorted(glob.glob(os.path.join(
        _REPO, "data", "synthesis_discovery", "reflection_domains_v3_mcs5_*.json")))[-1]
    print(f"Substrate artifact: {os.path.basename(art)}")
    A = json.load(open(art))
    id2cluster, cluster_meta = {}, {}
    for cid, dom in A["domains"].items():
        cid = int(cid)
        cluster_meta[cid] = {"label": dom["label"], "kind": dom["kind"]}
        for m in dom["members"]:
            id2cluster[m["entity_id"]] = cid

    g = GraphMemory()
    store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    fmap = load_facts_map(store)
    ents = collect_entities(g, fmap)
    model = load_embedder()
    X = np.asarray(model.encode([e["text"] for e in ents], batch_size=64,
                                show_progress_bar=False, normalize_embeddings=True), dtype=np.float32)
    ent_vec, by_id = {}, {}
    name2id = {}
    for i, e in enumerate(ents):
        v = X[i] / (np.linalg.norm(X[i]) + 1e-9)
        ent_vec[e["entity_id"]] = v
        e["cluster"] = id2cluster.get(e["entity_id"], -1)
        e["richness"] = _richness(e["text"])
        by_id[e["entity_id"]] = e
        name2id.setdefault(e["name"].lower(), e["entity_id"])
        name2id.setdefault(e["entity_id"], e["entity_id"])

    # cluster centroids in BGE space (normalized) for the secondary domain-distance signal
    members = defaultdict(list)
    for e in ents:
        if e["cluster"] != -1:
            members[e["cluster"]].append(e["entity_id"])
    cnorm = {}
    for cid, ids in members.items():
        c = np.mean([ent_vec[i] for i in ids], axis=0)
        cnorm[cid] = c / (np.linalg.norm(c) + 1e-9)

    return {"ents": ents, "ent_vec": ent_vec, "by_id": by_id, "cluster_meta": cluster_meta,
            "cnorm": cnorm, "name2id": name2id, "graph": g, "artifact": os.path.basename(art)}


def ent_dist(sub, a_id, b_id):
    return 1.0 - float(np.dot(sub["ent_vec"][a_id], sub["ent_vec"][b_id]))


def domain_dist(sub, ca, cb):
    if ca in sub["cnorm"] and cb in sub["cnorm"] and ca != cb:
        return 1.0 - float(np.dot(sub["cnorm"][ca], sub["cnorm"][cb]))
    return 0.0  # same cluster or noise


def resolve(sub, nm):
    k = nm.lower().strip()
    for cand in (k, k.replace(" ", "_")):
        if cand in sub["name2id"]:
            return sub["name2id"][cand]
    rid = sub["graph"].resolve_entity(nm)
    return rid if rid in sub["ent_vec"] else None


def build_pair(sub, a, b, stratum, is_gold=False, gold_kind=""):
    """A pair dict consumed by run_pair() — shared by the random harness and anchored gen."""
    ca, cb = a["cluster"], b["cluster"]
    return {
        "stratum": stratum, "is_gold": is_gold, "gold_kind": gold_kind,
        "a_id": a["entity_id"], "b_id": b["entity_id"],
        "name_a": a["name"], "name_b": b["name"],
        "domain_a": sub["cluster_meta"].get(ca, {}).get("label", f"c{ca}"),
        "domain_b": sub["cluster_meta"].get(cb, {}).get("label", f"c{cb}"),
        "same_domain": (ca == cb),
        "ent_dist": round(ent_dist(sub, a["entity_id"], b["entity_id"]), 4),
        "domain_dist": round(domain_dist(sub, ca, cb), 4),
        "obs_a": a["text"], "obs_b": b["text"],
    }


def eligible_life_entities(sub, verbose=True):
    """The clean pairing pool: rich, LIFE-domain, non-junk-cluster, noun-like entities.
    Shared by the random harness and the anchored generator so they draw from the same
    substrate. Gold anchors bypass this (they're resolved by name as control points)."""
    from memory.fact_extractor import _get_nlp
    nlp = _get_nlp()

    def junk_cluster(cid):
        return bool(_JUNK_CLUSTER_RE.search(sub["cluster_meta"].get(cid, {}).get("label", "")))

    base = [e for e in sub["ents"]
            if e["cluster"] != -1
            and sub["cluster_meta"].get(e["cluster"], {}).get("kind") == "life"
            and e["richness"] >= MIN_RICHNESS]
    after_cluster = [e for e in base if not junk_cluster(e["cluster"])]
    life = [e for e in after_cluster if _pairable(e["name"], nlp)]
    if verbose:
        dropped_clusters = sorted({sub["cluster_meta"].get(e["cluster"], {}).get("label", "?")
                                   for e in base if junk_cluster(e["cluster"])})
        print(f"Eligibility: {len(base)} rich life-domain entities "
              f"-> drop {len(base) - len(after_cluster)} in junk clusters {dropped_clusters} "
              f"-> drop {len(after_cluster) - len(life)} non-noun scraps -> {len(life)} pairable")
    return life


def sample_pairs(sub, n_per):
    """Stratified pairs from clean LIFE entities + the gold anchors."""
    rng = random.Random(SEED)
    life = eligible_life_entities(sub)

    by_cluster = defaultdict(list)
    for e in life:
        by_cluster[e["cluster"]].append(e)

    def mk(a, b, stratum, is_gold=False, gold_kind=""):
        return build_pair(sub, a, b, stratum, is_gold, gold_kind)

    seen = set()

    def key(a, b):
        return tuple(sorted((a["entity_id"], b["entity_id"])))

    # SAME-DOMAIN (thin): pairs within one cluster
    same = []
    rich_clusters = [c for c, es in by_cluster.items() if len(es) >= 2]
    rng.shuffle(rich_clusters)
    for c in rich_clusters:
        es = by_cluster[c][:]
        rng.shuffle(es)
        for i in range(0, len(es) - 1, 2):
            a, b = es[i], es[i + 1]
            if key(a, b) not in seen:
                seen.add(key(a, b)); same.append(mk(a, b, "same_domain"))
            if len(same) >= n_per:
                break
        if len(same) >= n_per:
            break

    # CROSS pairs: enumerate cross-cluster candidates, rank by centroid distance, tercile-split
    cross = []
    pool = life[:]
    rng.shuffle(pool)
    cand = []
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            a, b = pool[i], pool[j]
            if a["cluster"] == b["cluster"]:
                continue
            cand.append((domain_dist(sub, a["cluster"], b["cluster"]), a, b))
    cand.sort(key=lambda t: t[0])
    if cand:
        t = len(cand) // 3
        bands = [("cross_near", cand[:t]), ("cross_mid", cand[t:2 * t]), ("cross_far", cand[2 * t:])]
        for stratum, band in bands:
            band = band[:]
            rng.shuffle(band)
            for dd, a, b in band:
                if key(a, b) in seen:
                    continue
                seen.add(key(a, b)); cross.append(mk(a, b, stratum))
                if sum(1 for p in cross if p["stratum"] == stratum) >= n_per:
                    break

    # GOLD anchors
    gold = []
    for gk, na, nb in GOLD_ANCHORS:
        ra, rb = resolve(sub, na), resolve(sub, nb)
        if not ra or not rb:
            print(f"  gold {na}↔{nb}: unresolved — skipped")
            continue
        a, b = sub["by_id"].get(ra), sub["by_id"].get(rb)
        if not a or not b:
            continue
        if key(a, b) in seen:
            continue
        seen.add(key(a, b)); gold.append(mk(a, b, "gold", is_gold=True, gold_kind=gk))

    pairs = same + cross + gold
    for i, p in enumerate(pairs):
        p["pair_id"] = i
    cc = Counter(p["stratum"] for p in cross)
    print(f"Sampled pairs: same={len(same)} cross_near={cc['cross_near']} cross_mid={cc['cross_mid']} "
          f"cross_far={cc['cross_far']} gold={len(gold)}  total={len(pairs)}")
    return pairs


def _parse_gen(raw):
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    claim = str(d.get("claim", "")).strip()
    return {"mechanism": str(d.get("mechanism", "")).strip(), "claim": claim}


def _parse_gate(raw):
    def f(pat, default=""):
        m = re.search(pat, raw or "", re.IGNORECASE)
        return m.group(1).strip() if m else default
    rating = f(r"RATING:\s*(VALID|INVALID)").upper()
    ga = f(r"GROUNDED_A:\s*(yes|no)").lower()
    gb = f(r"GROUNDED_B:\s*(yes|no)").lower()
    return {
        "mechanism_valid": rating == "VALID",
        "grounded_a": ga == "yes",
        "grounded_b": gb == "yes",
        "gate_dejargon": f(r"DE-JARGON:\s*(.+)")[:200],
        "gate_counterfactual": f(r"COUNTERFACTUAL:\s*(.+)")[:200],
        "gate_reason": f(r"REASON:\s*(.+)")[:200],
        "rating": rating or "PARSE_ERR",
    }


async def run_pair(mm, sem, sub, p):
    from config.app_config import SYNTHESIS_COHERENCE_MODEL
    async with sem:
        # GENERATE
        graw = await mm.generate_once(
            GEN_PROMPT.format(name_a=p["name_a"], obs_a=p["obs_a"][:550],
                              name_b=p["name_b"], obs_b=p["obs_b"][:550]),
            model_name=SYNTHESIS_COHERENCE_MODEL, system_prompt=GEN_SYSTEM,
            max_tokens=500, temperature=0.5, disable_reasoning=True)
        gen = _parse_gen(graw) or {"mechanism": "", "claim": ""}
        claim = gen["claim"]
        p["mechanism"] = gen["mechanism"]
        p["claim"] = claim
        declined = (not claim) or claim.strip().upper().startswith("NONE")
        p["declined"] = declined
        if declined:
            p.update({"mechanism_valid": False, "grounded_a": False, "grounded_b": False,
                      "truth_pass": False, "gate_dejargon": "", "gate_counterfactual": "",
                      "gate_reason": "generator declined", "rating": "DECLINED"})
            return p
        # GATE (truth + dual-grounding)
        graw2 = await mm.generate_once(
            GATE_PROMPT.format(name_a=p["name_a"], obs_a=p["obs_a"][:550],
                               name_b=p["name_b"], obs_b=p["obs_b"][:550], claim=claim),
            model_name=SYNTHESIS_COHERENCE_MODEL, system_prompt=GATE_SYSTEM,
            max_tokens=420, temperature=0.1, disable_reasoning=True)
        g = _parse_gate(graw2)
        p.update(g)
        p["truth_pass"] = bool(g["mechanism_valid"] and g["grounded_a"] and g["grounded_b"])
        return p


# CSV columns: machine-emitted + the two hand-label columns (left blank).
_LABEL_COLS = ["label_truth", "label_value", "label_want_surfaced", "label_notes"]
_CSV_COLS = [
    "pair_id", "stratum", "is_gold", "gold_kind", "same_domain", "ent_dist", "domain_dist",
    "name_a", "domain_a", "name_b", "domain_b",
    "declined", "rating", "mechanism_valid", "grounded_a", "grounded_b", "truth_pass",
    "mechanism", "claim", "gate_dejargon", "gate_counterfactual", "gate_reason",
] + _LABEL_COLS


async def generate_mode(n_per):
    from models.model_manager import ModelManager
    print(f"Reflection VALIDATION harness (generate) | n_per_stratum={n_per}\n")
    sub = load_substrate()
    pairs = sample_pairs(sub, n_per)
    mm = ModelManager()
    sem = asyncio.Semaphore(GATE_CONCURRENCY)
    print(f"\nGenerating + gating {len(pairs)} pairs (concurrency={GATE_CONCURRENCY})...", flush=True)
    done = await asyncio.gather(*(run_pair(mm, sem, sub, p) for p in pairs))

    # console summary
    print("\n" + "=" * 100)
    for p in sorted(done, key=lambda x: (x["stratum"], -x["ent_dist"])):
        tag = "GOLD-" + p["gold_kind"] if p["is_gold"] else p["stratum"]
        verdict = "DECLINED" if p["declined"] else ("TRUTH-PASS" if p["truth_pass"] else f"truth-fail({p['rating']})")
        print(f"[{p['pair_id']:2}] {tag:11} d={p['ent_dist']:.2f} same={int(p['same_domain'])} "
              f"{p['name_a'][:16]:16}↔{p['name_b'][:16]:16} {verdict}")
        if not p["declined"]:
            print(f"      {p['claim'][:150]}")
    print("=" * 100)
    by_strat = defaultdict(lambda: [0, 0, 0])  # [n, declined, truth_pass]
    for p in done:
        s = "gold" if p["is_gold"] else p["stratum"]
        by_strat[s][0] += 1
        by_strat[s][1] += int(p["declined"])
        by_strat[s][2] += int(p["truth_pass"])
    print("stratum         n  declined  truth_pass  mean_ent_dist")
    for s in ("same_domain", "cross_near", "cross_mid", "cross_far", "gold"):
        if s in by_strat:
            n, dec, tp = by_strat[s]
            md = np.mean([p["ent_dist"] for p in done if (("gold" if p["is_gold"] else p["stratum"]) == s)])
            print(f"  {s:12} {n:3} {dec:8} {tp:10}      {md:.3f}")

    ts = datetime.now()
    outdir = os.path.join(_REPO, "data", "synthesis_discovery")
    os.makedirs(outdir, exist_ok=True)
    stem = f"reflection_validation_{ts:%Y%m%d_%H%M%S}"
    csv_path = os.path.join(outdir, stem + ".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for p in done:
            row = {k: p.get(k, "") for k in _CSV_COLS}
            for lc in _LABEL_COLS:
                row[lc] = ""  # blank for hand-labeling
            w.writerow(row)
    json_path = os.path.join(outdir, stem + ".json")
    json.dump({"timestamp": ts.isoformat(timespec="seconds"), "artifact": sub["artifact"],
               "n_per_stratum": n_per, "pairs": done}, open(json_path, "w"), indent=2, default=str)
    print(f"\nHand-label this CSV  ->  {csv_path}")
    print("  fill label_truth (mechanism true & dual-grounded? 0/1), label_value (valuable/non-thin? 0/1),")
    print("  label_want_surfaced (optional 0/1). Then: "
          f"python scripts/reflection_validation_harness.py --analyze {os.path.basename(csv_path)}")
    print(f"Raw JSON -> {json_path}")


# ─────────────────────────────── analyze mode (4 checks) ───────────────────────────────
def _bool01(v):
    s = str(v).strip().lower()
    return 1 if s in ("1", "true", "yes", "y", "t") else (0 if s in ("0", "false", "no", "n", "f") else None)


def _phi_chi2(a):
    """2x2 contingency a=[[n00,n01],[n10,n11]] -> (phi, chi2, p, mutual_info_bits)."""
    a = np.asarray(a, dtype=float)
    n = a.sum()
    if n == 0:
        return 0.0, 0.0, 1.0, 0.0
    try:
        from scipy.stats import chi2_contingency
        chi2, p, _, _ = chi2_contingency(a, correction=False)
    except Exception:
        row = a.sum(1, keepdims=True); col = a.sum(0, keepdims=True)
        exp = row @ col / n
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = float(np.nansum((a - exp) ** 2 / exp))
        p = float("nan")
    phi = float(np.sqrt(chi2 / n)) * (1 if (a[0, 0] * a[1, 1] >= a[0, 1] * a[1, 0]) else -1)
    # mutual information (bits)
    px = a.sum(1) / n; py = a.sum(0) / n
    mi = 0.0
    for i in range(2):
        for j in range(2):
            pij = a[i, j] / n
            if pij > 0 and px[i] > 0 and py[j] > 0:
                mi += pij * np.log2(pij / (px[i] * py[j]))
    return phi, float(chi2), float(p), float(mi)


def _auc(scores, labels):
    """ROC AUC via rank statistic (Mann-Whitney)."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rpos = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
    return (rpos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def analyze_mode(path):
    if not os.path.isabs(path):
        path = os.path.join(_REPO, "data", "synthesis_discovery", path)
    rows = list(csv.DictReader(open(path)))
    labeled = [r for r in rows if _bool01(r.get("label_truth")) is not None
               and _bool01(r.get("label_value")) is not None]
    print(f"Reflection VALIDATION harness (analyze) | {os.path.basename(path)}")
    print(f"rows={len(rows)}  hand-labeled={len(labeled)}\n")
    if len(labeled) < 8:
        print("Too few labeled rows — fill label_truth + label_value (0/1) for at least ~8 pairs.")
        return

    lt = [_bool01(r["label_truth"]) for r in labeled]
    lv = [_bool01(r["label_value"]) for r in labeled]
    tp = [1 if str(r.get("truth_pass")).strip().lower() in ("1", "true", "yes") else 0 for r in labeled]
    dist = [float(r["ent_dist"]) for r in labeled]

    # CHECK 1 — gate truth_pass predicts label_truth
    print("CHECK 1 — grounded-truth GATE predicts hand label_truth")
    c = [[0, 0], [0, 0]]
    for g, h in zip(tp, lt):
        c[g][h] += 1
    acc = (c[0][0] + c[1][1]) / len(labeled)
    print(f"  confusion [gate_pass x label_truth]: {c}   accuracy={acc:.2f}")
    tpr = c[1][1] / (c[1][1] + c[0][1]) if (c[1][1] + c[0][1]) else float("nan")
    fpr = c[1][0] / (c[1][0] + c[0][0]) if (c[1][0] + c[0][0]) else float("nan")
    print(f"  recall(true)={tpr:.2f}  false-alarm(pass on untrue)={fpr:.2f}\n")

    # CHECK 2 — domain distance predicts label_value
    # The DEPLOYED signal is the inverted-U transform distance_value(d) from
    # reflection_pipeline.py, NOT raw distance. A monotone AUC on raw distance
    # structurally cannot validate (or falsify) an inverted-U — a correct peak scorer
    # can score 0.5 and a wrong one 0.9. So the check scores the deployed function
    # itself; raw-distance AUC is printed as a reference number only.
    from scripts.reflection_pipeline import distance_value  # lazy: pipeline imports this module
    dv = [distance_value(d) for d in dist]
    print("CHECK 2 — DOMAIN-DISTANCE predicts hand label_value")
    mv1 = np.mean([d for d, v in zip(dist, lv) if v == 1]) if any(lv) else float("nan")
    mv0 = np.mean([d for d, v in zip(dist, lv) if v == 0]) if any(x == 0 for x in lv) else float("nan")
    print(f"  mean ent_dist: value=1 -> {mv1:.3f}   value=0 -> {mv0:.3f}   sep={mv1-mv0:+.3f}")
    print(f"  AUC(distance_value(d) -> value)={_auc(dv, lv):.3f}   <-- DEPLOYED inverted-U signal "
          f"(0.5=no signal, 1.0=perfect)")
    print(f"  AUC(raw dist -> value)={_auc(dist, lv):.3f}   (reference only — monotone; "
          f"cannot judge the inverted-U)\n")

    # CHECK 3 — deployed product (truth_pass × distance_value) predicts want_surfaced
    ws = [_bool01(r.get("label_want_surfaced")) for r in labeled]
    if all(w is not None for w in ws) and any(ws) and any(w == 0 for w in ws):
        prod = [t * v for t, v in zip(tp, dv)]
        print("CHECK 3 — PRODUCT (truth_pass × distance_value) predicts want_surfaced")
        print(f"  AUC(product -> want_surfaced)={_auc(prod, ws):.3f}   <-- the deployed "
              f"reflection_score shape")
        print(f"  AUC(distance_value alone -> want_surfaced)={_auc(dv, ws):.3f}  "
              f"AUC(truth alone -> want_surfaced)={_auc([float(t) for t in tp], ws):.3f}\n")
    else:
        print("CHECK 3 — skipped (fill label_want_surfaced 0/1 to run)\n")

    # CHECK 4 — ORTHOGONALITY of label_truth and label_value (the key number)
    print("CHECK 4 — ORTHOGONALITY: are label_truth and label_value independent?")
    o = [[0, 0], [0, 0]]
    for t, v in zip(lt, lv):
        o[t][v] += 1
    phi, chi2, p, mi = _phi_chi2(o)
    print(f"  contingency [truth x value]:")
    print(f"      value=0  value=1")
    print(f"    truth=0   {o[0][0]:3}     {o[0][1]:3}")
    print(f"    truth=1   {o[1][0]:3}     {o[1][1]:3}")
    print(f"  phi={phi:+.3f}  chi2={chi2:.2f}  p={p:.3f}  mutual_info={mi:.3f} bits")
    if abs(phi) < 0.2:
        print("  => ORTHOGONAL: truth and value are ~independent — the two-axis design is real;")
        print("     a lenient truth floor + a SEPARATE distance/value signal is the right architecture.")
    elif phi > 0:
        print("  => CORRELATED (+): truthful claims tend to be valuable too — the axes partly collapse;")
        print("     value may be partly recoverable from the gate alone (less need for distance).")
    else:
        print("  => ANTI-CORRELATED (-): truthful claims tend to be LESS valuable (or vice-versa) —")
        print("     investigate; the gate may be rewarding thin-but-tidy claims.")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        if len(sys.argv) < 3:
            print("usage: --analyze <labeled.csv>")
            return
        analyze_mode(sys.argv[2])
    else:
        asyncio.run(generate_mode(N_PER_STRATUM))


if __name__ == "__main__":
    main()
