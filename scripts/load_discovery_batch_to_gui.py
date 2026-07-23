#!/usr/bin/env python3
"""
Load a generated discovery grading batch (CSV) into the PRODUCTION synthesis_results
collection so the candidates appear in the GUI audit queue for hand-grading.

ACCEPT rows -> stored as ACCEPTED (status=accepted) -> GUI "Accepted (ungraded)" view.
QUEUE  rows -> stored as REJECTED at composite_scoring -> GUI "Rejected (ungraded)" view.
Both stored with human_grade="" (ungraded). ACCEPTs go through SynthesisMemory.store_result
(which dedups vs existing via find_similar, so re-running won't duplicate). This is the same
audit-queue path dreaming uses — an additive write, no deletes.

Usage:
  python scripts/load_discovery_batch_to_gui.py <batch.csv> [--dry-run]
"""
import argparse
import csv
import os
import sys
from datetime import datetime

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="grading batch CSV (in data/synthesis_discovery/ or absolute)")
    ap.add_argument("--dry-run", action="store_true", help="show what would be written, write nothing")
    args = ap.parse_args()

    path = args.csv if os.path.isabs(args.csv) else os.path.join(_REPO, "data", "synthesis_discovery", args.csv)
    rows = list(csv.DictReader(open(path)))
    print(f"Batch: {os.path.basename(path)} | {len(rows)} candidates")

    from config.app_config import CHROMA_PATH
    from knowledge.synthesis_models import SynthesisCandidate, SynthesisResult, CoherenceLevel, CandidateStatus

    # Accurate endpoint_distance via the retrieval embedder (no FAISS index load needed).
    from sentence_transformers import SentenceTransformer
    import numpy as np
    emb = SentenceTransformer("BAAI/bge-small-en-v1.5")
    concepts = sorted({r["concept_a"] for r in rows} | {r["concept_b"] for r in rows})
    vecs = dict(zip(concepts, emb.encode(concepts, normalize_embeddings=True)))

    def build_result(r):
        a, b = r["concept_a"], r["concept_b"]
        cos = float(np.dot(vecs[a], vecs[b]))
        cand = SynthesisCandidate(
            concept_a=a, concept_b=b, connection_claim=r["connection_claim"],
            walk_path=["pooled", a.lower(), b.lower()], source_domains={a, b},
            endpoint_distance=round(1.0 - cos, 4), timestamp=datetime.now())
        res = SynthesisResult(candidate=cand)
        lvl = r.get("coherence_level", "")
        res.coherence_level = CoherenceLevel[lvl] if lvl in CoherenceLevel.__members__ else None
        res.composite_score = float(r["composite"])
        res.novelty_score_external = float(r.get("novelty", 0.0) or 0.0)  # aggregate proxy
        if r["status"] == "ACCEPT":
            res.status = CandidateStatus.ACCEPTED
        else:
            res.status = CandidateStatus.REJECTED
            res.rejection_stage = "composite_scoring"
            res.rejection_reason = f"Composite {res.composite_score:.3f} < 0.70"
        return res

    results = [build_result(r) for r in rows]
    n_acc = sum(1 for x in results if x.status == CandidateStatus.ACCEPTED)
    n_rej = len(results) - n_acc
    print(f"  -> {n_acc} ACCEPTED (GUI 'accepted (ungraded)'), {n_rej} REJECTED@composite (GUI 'rejected (ungraded)')")

    if args.dry_run:
        print("\n[DRY RUN] nothing written. Sample:")
        for x in results[:3]:
            print(f"  [{x.status.value}] comp={x.composite_score} {x.candidate.concept_a} <-> {x.candidate.concept_b}")
        return

    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
    from memory.synthesis_memory import SynthesisMemory
    store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    mem = SynthesisMemory(store)
    print(f"\nWriting to production synthesis_results ({CHROMA_PATH}) ...")
    wrote = 0
    for x in results:
        try:
            if x.status == CandidateStatus.ACCEPTED:
                mem.store_result(x)        # dedups vs existing
            else:
                mem.store_rejected_for_audit(x)
            wrote += 1
        except Exception as e:
            print(f"  ! failed {x.candidate.concept_a}<->{x.candidate.concept_b}: {e}")
    print(f"Wrote {wrote}/{len(results)} candidates into the audit queue.")
    print("Open the GUI Status tab -> Synthesis audit -> grade the 'accepted (ungraded)' + "
          "'rejected (ungraded)' views (grade 1-5; 4-5 = real discovery).")


if __name__ == "__main__":
    main()
