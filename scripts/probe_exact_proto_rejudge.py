#!/usr/bin/env python3
"""
Re-judge the EXACT stored prototype candidates N times each — corrects a slip:
the stability probe hand-transcribed a CLEANER paraphrase of the Oliver<->dad
claim, so it didn't test what the prototype actually judged.

This loads the EXACT (context_a, context_b, claim) from a saved
reflection_generator_proto_*.json and re-runs the existing judge REPS times on each,
byte-identical to what the prototype fed it.

READ:
  A WEAK row that re-judges reliably WEAK   -> judge stably rejects that EXACT text;
                                               the prototype WEAK was real (craft/quality).
  A WEAK row that re-judges mostly MODERATE -> the single prototype WEAK was a stochastic
                                               dip; the judge actually passes that claim.
Compare the same-system candidate (kavarin<->midterm) vs the cross-system one
(Oliver<->dad): if the first stays WEAK and the second flips, the judge is
DISCRIMINATING correctly and the bottleneck is generator YIELD, not the judge.

Usage: python scripts/probe_exact_proto_rejudge.py [REPS] [proto_json_path]
"""
import asyncio
import glob
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime as _dt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from config.app_config import SYNTHESIS_COHERENCE_MODEL
from models.model_manager import ModelManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from memory.synthesis_memory import SynthesisMemory
from knowledge.synthesis_filter import SynthesisFilter
from knowledge.synthesis_models import SynthesisCandidate, SynthesisResult

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
path = sys.argv[2] if len(sys.argv) > 2 else sorted(
    glob.glob(os.path.join(_REPO, "data/synthesis_discovery/reflection_generator_proto_*.json")))[-1]


async def main():
    d = json.load(open(path))
    rows = d["rows"]
    print(f"Re-judge EXACT prototype text | {os.path.basename(path)} | judge={SYNTHESIS_COHERENCE_MODEL} | reps={REPS}\n")

    tmp = tempfile.mkdtemp(prefix="reexact_chroma_")
    store = MultiCollectionChromaStore(persist_directory=tmp)
    mm = ModelManager()
    sm = SynthesisMemory(store)
    filt = SynthesisFilter(chroma_store=store, model_manager=mm, synthesis_memory=sm)
    filt.graph_memory = None
    filt.entity_resolver = None

    _PASS = {"MODERATE", "STRONG"}
    for r in rows:
        a = str(r.get("context_a", ""))[:80]
        b = str(r.get("context_b", ""))[:80]
        claim = str(r.get("claim", ""))
        levels = []
        for _ in range(REPS):
            sc = SynthesisCandidate(
                concept_a=a, concept_b=b, connection_claim=claim,
                walk_path=["self", a.lower(), b.lower(), "self"],
                source_domains={"da", "db"}, endpoint_distance=0.5, timestamp=_dt.now())
            result = SynthesisResult(candidate=sc)
            try:
                await filt._stage_5_coherence_judge(result)
                lvl = result.coherence_level.name if result.coherence_level else "None"
            except Exception as e:
                lvl = f"ERR:{type(e).__name__}"
            levels.append(lvl)
        npass = sum(1 for l in levels if l in _PASS)
        tag = f"orig={r.get('level')}"
        print(f"  {claim[:46]:46} {tag:12} -> PASS {npass}/{REPS}  {dict(Counter(levels))}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
