#!/usr/bin/env python3
"""
Phase 1 (v3) of the reflection personal-domain-distance signal: emergent domains by
ENTITY-POOLED embedding clustering. Substrate improvements approved 2026-06-29:
  - UMAP (n_neighbors=15 -> 10d, cosine) before HDBSCAN — PCA blobbed; UMAP preserves
    the local structure the within-life distance signal lives in.
  - FACT-ENRICHED entity text: display_name + graph context + the entity's real
    `facts` rows (subject OR object), so domains form on real content not terse triples.
  - LIFE vs PROJECT flag per domain (LLM): ~10 domains are Daemon coding minutiae, not
    life domains — flagged so they can be excluded as reflection sources.
  - ENTITY-LEVEL gold anchors: the distance signal is an ENTITY-PAIR signal, so anchors
    are resolved to real graph entities (dad, kavarin, brother…), NOT claim sentences.

Entity pooling (one vector per entity) collapses logging-volume bias. Personal-only
(no wiki_retrieved / user hub). Read-only on production memory; writes a JSON artifact.
Usage: python scripts/reflection_domain_clustering.py [min_cluster_size]
"""
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from sklearn.cluster import HDBSCAN
from umap import UMAP

from memory.graph_memory import GraphMemory
from utils.logging_utils import get_logger

logger = get_logger("reflection_domains")

# Guarded so importing this module's helpers from another script (which has its own
# argv) doesn't trip on a non-numeric arg. Direct CLI use (`... clustering.py 5`) is intact.
MIN_CLUSTER_SIZE = int(sys.argv[1]) if (len(sys.argv) > 1 and sys.argv[1].isdigit()) else 5

# ENTITY-level gold anchors (resolved to real graph entities at runtime).
# REAL = cross life-domain (should be FAR / different domain). THIN = same domain (NEAR).
GOLD_ENTITY_ANCHORS = [
    ("REAL", "dad", "kavarin"),         # family <-> health
    ("REAL", "dad", "gym"),            # family <-> fitness
    ("REAL", "brother", "isye 6501"),  # family <-> grad school
    ("THIN", "brother", "sister"),     # both family
    ("THIN", "kavarin", "dexivar"),     # both health/meds
    ("THIN", "gym", "lift"),           # both fitness
]

_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "it", "is", "i",
         "me", "my", "user", "poor", "today", "thing", "stuff", "lot", "bit"}

_JUNK_RE = re.compile(
    r"^(few|next|last|this|that|some|every|each)\b"
    r"|^\d"
    r"|\b(day|days|week|weeks|month|months|year|years|hour|hours|min|mins|am|pm)\b"
    r"|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday|tomorrow|yesterday|tonight|morning|evening)\b"
    r"|\b(do|did|get|got|go|went|make|made|take|took|peer reviews?)\b",
    re.IGNORECASE)


def _is_junk_entity(name: str) -> bool:
    n = name.strip().lower()
    if len(n) <= 2 or n in _STOP:
        return True
    if _JUNK_RE.search(n):
        return True
    if len(n.split()) >= 5:
        return True
    return False


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("_", " ")).strip()


def load_facts_map(store):
    """name(lower) -> list of real fact strings where the name is subject OR object."""
    col = store._get_collection("facts")
    got = col.get(include=["documents"])
    fmap = defaultdict(list)
    for doc in got["documents"]:
        parts = [p.strip() for p in (doc or "").split("|")]
        if len(parts) != 3:
            continue
        subj, pred, obj = parts
        readable = f"{subj} {pred.replace('_', ' ')} {obj}"
        for key in (subj.lower(), obj.lower()):
            if key and key != "user":
                fmap[key].append(readable)
    return fmap


def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


def collect_entities(g, fmap):
    ents = []
    for nid, d in g.graph.nodes(data=True):
        if nid == "user" or (d.get("metadata") or {}).get("source", "personal") != "personal":
            continue
        name = _clean(d.get("display_name", nid))
        if not name or _is_junk_entity(name):
            continue
        ctx = g.get_context_sentences(nid, depth=1, max_sentences=4)
        # real facts referencing this entity (by id, display_name, aliases)
        keys = {nid.lower(), name.lower()} | {a.lower() for a in (d.get("aliases") or [])}
        facts = []
        for k in keys:
            facts.extend(fmap.get(k, []))
        facts = list(dict.fromkeys(facts))[:8]   # dedup, cap
        text = " . ".join([name] + [_clean(c) for c in ctx] + [_clean(f) for f in facts])
        ents.append({"entity_id": nid, "name": name,
                     "mention_count": int(d.get("mention_count", 0)), "text": text[:600]})
    return ents


async def label_domains(clusters, mm):
    """One LLM call: per cluster -> {label, kind in life|project}."""
    from config.app_config import SYNTHESIS_COHERENCE_MODEL
    blocks = []
    for cid, members in sorted(clusters.items()):
        names = ", ".join(m["name"] for m in sorted(members, key=lambda x: -x["mention_count"])[:14])
        blocks.append(f"Cluster {cid}: {names}")
    prompt = (
        "Each cluster is a group of entities from one person's life. For each, give:\n"
        "  - label: SHORT (1-4 word) domain name (e.g. 'Family','Bar friends','Health/meds',"
        "'Brewing job','Grad school').\n"
        "  - kind: 'life' for a real life domain (family, friends, health, job/career, hobbies, "
        "travel, pets, activism…) OR 'project' for SOFTWARE/CODING implementation detail of the "
        "person's AI project 'Daemon' (e.g. prompt caching, FAISS, chromadb, build/deploy, "
        "pipelines, memory modules). A job/career like brewing is 'life'; Daemon code internals "
        "are 'project'.\n\n"
        + "\n".join(blocks)
        + "\n\nReturn ONLY JSON: {\"<cluster_id>\": {\"label\": \"...\", \"kind\": \"life|project\"}, ...}"
    )
    raw = ""
    try:
        raw = await mm.generate_once(prompt, model_name=SYNTHESIS_COHERENCE_MODEL,
                                     system_prompt="You label clusters of personal entities. Terse, concrete.",
                                     max_tokens=1800, temperature=0.2, disable_reasoning=True)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        obj = json.loads(m.group(0)) if m else {}
        return {int(k): (v if isinstance(v, dict) else {"label": str(v), "kind": "life"})
                for k, v in obj.items()}
    except Exception as e:
        logger.warning(f"labeling failed: {e}; raw head: {(raw or '')[:200]!r}")
        return {}


async def main():
    from models.model_manager import ModelManager
    from config.app_config import CHROMA_PATH
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

    print(f"Reflection domains v3 | UMAP+facts, personal-only, entity-pooled | mcs={MIN_CLUSTER_SIZE}\n")
    g = GraphMemory()
    store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    fmap = load_facts_map(store)
    print(f"Loaded fact-map for {len(fmap)} names")
    ents = collect_entities(g, fmap)
    n_with_facts = sum(1 for e in ents if " . " in e["text"])
    print(f"Personal entities after filter: {len(ents)}")

    model = load_embedder()
    Xfull = np.asarray(model.encode([e["text"] for e in ents], batch_size=64,
                                    show_progress_bar=False, normalize_embeddings=True), dtype=np.float32)

    reducer = UMAP(n_neighbors=15, n_components=10, min_dist=0.0, metric="cosine", random_state=42)
    Xr = reducer.fit_transform(Xfull)
    print(f"UMAP: {Xfull.shape[1]}d -> {Xr.shape[1]}d")

    print("\nmin_cluster_size sweep (domains / noise%):")
    for mcs in sorted({3, 4, 5, 8, MIN_CLUSTER_SIZE}):
        lbl = HDBSCAN(min_cluster_size=mcs, metric="euclidean").fit_predict(Xr)
        ncl = len(set(lbl) - {-1})
        print(f"  mcs={mcs:2}: {ncl:2} domains, {np.mean(lbl == -1)*100:4.0f}% noise"
              + ("  <-- chosen" if mcs == MIN_CLUSTER_SIZE else ""))

    labels = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric="euclidean").fit_predict(Xr)
    ent_cluster = {ents[i]["entity_id"]: int(l) for i, l in enumerate(labels)}
    clusters = defaultdict(list)
    for e, l in zip(ents, labels):
        if l != -1:
            clusters[int(l)].append(e)

    # Distance signal lives in the ORIGINAL BGE space (cosine). UMAP/HDBSCAN are used
    # ONLY for cluster ASSIGNMENT — UMAP coords distort global distance and aren't
    # cosine-meaningful (they sit in a positive orthant -> all cosines ~1 -> dist ~0).
    centroids = {cid: Xfull[[i for i, l in enumerate(labels) if l == cid]].mean(axis=0) for cid in clusters}
    cnorm = {cid: c / (np.linalg.norm(c) + 1e-9) for cid, c in centroids.items()}
    ent_vec = {ents[i]["entity_id"]: (Xfull[i] / (np.linalg.norm(Xfull[i]) + 1e-9)) for i in range(len(ents))}
    name2id = {}
    for e in ents:
        name2id.setdefault(e["name"].lower(), e["entity_id"])
        name2id.setdefault(e["entity_id"], e["entity_id"])

    names = await label_domains(clusters, ModelManager())
    def lbl_of(cid): return names.get(cid, {}).get("label", f"cluster_{cid}")
    def kind_of(cid): return names.get(cid, {}).get("kind", "life")

    print("\n" + "=" * 90)
    print(f"DOMAINS (mcs={MIN_CLUSTER_SIZE}) — [life]/[proj] flag, top-entity share (over-logged check)")
    print("=" * 90)
    for cid in sorted(clusters, key=lambda c: (kind_of(c), -len(clusters[c]))):
        members = sorted(clusters[cid], key=lambda x: -x["mention_count"])
        total = sum(m["mention_count"] for m in members) or 1
        share = members[0]["mention_count"] / total
        flag = "  ⚠over-logged" if share > 0.6 and len(members) > 2 else ""
        shown = ", ".join(f"{m['name']}({m['mention_count']})" for m in members[:14])
        print(f"\n[{cid}] {lbl_of(cid)}  [{kind_of(cid)}] — {len(members)} ents, top-share {share:.0%}{flag}")
        print(f"     {shown}{' …' if len(members) > 14 else ''}")
    life_n = sum(len(clusters[c]) for c in clusters if kind_of(c) == "life")
    proj_n = sum(len(clusters[c]) for c in clusters if kind_of(c) == "project")
    print(f"\nNOISE: {int(np.sum(labels == -1))}   LIFE entities: {life_n}   PROJECT entities: {proj_n}")

    # Entity-level gold anchors
    print("\n" + "=" * 90)
    print("ENTITY GOLD ANCHORS — REAL should be FAR/different-domain; THIN NEAR/same-domain")
    print("=" * 90)
    def resolve(nm):
        k = nm.lower().strip()
        for cand in (k, k.replace(" ", "_")):
            if cand in name2id:
                return name2id[cand]
        rid = g.resolve_entity(nm)
        return rid if rid in ent_vec else None
    rows = []
    for kind, a, b in GOLD_ENTITY_ANCHORS:
        ra, rb = resolve(a), resolve(b)
        if not ra or not rb or ra not in ent_vec or rb not in ent_vec:
            print(f"  [{kind}] {a} ↔ {b}: unresolved (a={ra}, b={rb}) — skipped")
            rows.append({"kind": kind, "a": a, "b": b, "status": "unresolved"})
            continue
        ca, cb = ent_cluster.get(ra, -1), ent_cluster.get(rb, -1)
        # direct entity-vector cosine distance (works even if noise)
        va, vb = ent_vec[ra] / (np.linalg.norm(ent_vec[ra]) + 1e-9), ent_vec[rb] / (np.linalg.norm(ent_vec[rb]) + 1e-9)
        ent_dist = 1.0 - float(np.dot(va, vb))
        # domain-centroid distance (the architecture's signal)
        dom_dist = (1.0 - float(np.dot(cnorm[ca], cnorm[cb]))) if (ca in cnorm and cb in cnorm) else None
        same = (ca == cb and ca != -1)
        la = lbl_of(ca) if ca in cnorm else ("NOISE" if ca == -1 else f"c{ca}")
        lb = lbl_of(cb) if cb in cnorm else ("NOISE" if cb == -1 else f"c{cb}")
        dd = f"{dom_dist:.2f}" if dom_dist is not None else "n/a"
        print(f"  [{kind}] {a:12}({la}) ↔ {b:12}({lb})  domain_dist={dd}  ent_dist={ent_dist:.2f}  same={same}")
        rows.append({"kind": kind, "a": a, "b": b, "domain_a": la, "domain_b": lb,
                     "domain_dist": dom_dist, "ent_dist": ent_dist, "same_domain": same})

    real = [r for r in rows if r.get("kind") == "REAL" and "ent_dist" in r]
    thin = [r for r in rows if r.get("kind") == "THIN" and "ent_dist" in r]
    if real and thin:
        mr = np.mean([r["ent_dist"] for r in real]); mt = np.mean([r["ent_dist"] for r in thin])
        print(f"\n  mean ent_dist  REAL={mr:.2f}  THIN={mt:.2f}  separation={mr - mt:+.2f} (want REAL > THIN)")

    ts = datetime.now()
    out = os.path.join(_REPO, "data", "synthesis_discovery", f"reflection_domains_v3_mcs{MIN_CLUSTER_SIZE}_{ts:%Y%m%d_%H%M%S}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"timestamp": ts.isoformat(timespec="seconds"), "min_cluster_size": MIN_CLUSTER_SIZE,
               "n_entities": len(ents), "n_noise": int(np.sum(labels == -1)),
               "domains": {str(cid): {"label": lbl_of(cid), "kind": kind_of(cid),
                                      "members": [{"name": m["name"], "entity_id": m["entity_id"],
                                                   "mention_count": m["mention_count"]} for m in clusters[cid]],
                                      "centroid": centroids[cid].tolist()} for cid in clusters},
               "gold_anchors": rows}, open(out, "w"), indent=2, default=str)
    print(f"\nSaved -> {out}")
    print("\nSTOPPED before validation harness — review domains + entity anchors above.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
