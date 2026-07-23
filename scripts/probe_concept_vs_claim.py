#!/usr/bin/env python3
"""
2x2 isolation: what flips the Oliver<->dad reflection WEAK<->MODERATE — the CLAIM
wording, or the concept_a/concept_b STRINGS the judge sees?

My 4/4 MODERATE cell changed BOTH (clean concepts + my paraphrase) vs the 0/4 WEAK
cells (messy journal-quote concepts + generator claim). The generator's own
two-system REWRITE is excellent yet scored 0/3 WEAK with messy concepts. So isolate:

  concepts: CLEAN short labels   vs  MESSY raw journal-quotes
  claim:    P = my paraphrase (was 4/4 MOD)   G = generator's 2-system rewrite

  cell C+P  expect MODERATE (re-confirm)
  cell M+P  if WEAK -> CONCEPT strings are the lever
  cell C+G  if MODERATE -> generator claim is fine; messy concepts alone caused WEAK
  cell M+G  already 0/3 WEAK (generator concepts + generator claim)

Usage: python scripts/probe_concept_vs_claim.py [REPS]
"""
import asyncio, os, sys, tempfile
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

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3

CLEAN_A = "swallowing my boss Oliver's arbitrary rules and covering extra shifts"
CLEAN_B = "approaching my dad about a balance transfer"
MESSY_A = ("Workplace drama with boss Oliver 'making arbitrary rules and talking down,' and me "
           "covering shifts (closing for a sick coworker, a double weekend)")
MESSY_B = ("Needing to 'talk to dad about a balance transfer' and the ongoing money pressure "
           "('pay yourself at least 100 Friday, near zero added balance')")

P = ("Swallowing Oliver's arbitrary rules and covering extra shifts, and approaching my dad about a "
     "balance transfer, are the same trade of standing-ground for security with a resource-holder. It "
     "predicts my concession rises as my independent buffer of the thing they control falls — I push "
     "back least exactly when I can least afford to lose what they hold.")
G = ("Two separate systems run the same relation. In the work system, Oliver controls my shift income "
     "and standing; as my buffer against losing his favor thins, my concession to him rises — I swallow "
     "his arbitrary rules and absorb extra shifts (closing for a sick coworker, a double weekend) rather "
     "than push back. In the family system, my dad controls financial relief (the balance transfer); as "
     "my buffer against my own money pressure thins ('pay yourself 100 Friday, near zero added balance'), "
     "my deference to him rises — I approach him on his terms rather than negotiate as an equal. The "
     "shared relation: in EACH system, as my slack against that system's specific dependency falls, my "
     "submission to that system's specific resource-holder climbs. Differential prediction: each tracks "
     "its OWN scarcity independently — a flush-income/tight-cash month would have me pushing back on "
     "Oliver while deferring to dad, and the reverse in a secure-savings/precarious-job month; whichever "
     "resource I'm currently shortest on is the one whose holder I'll concede to most.")

CELLS = [
    ("C+P  clean concepts + my paraphrase", CLEAN_A, CLEAN_B, P),
    ("M+P  messy concepts + my paraphrase", MESSY_A, MESSY_B, P),
    ("C+G  clean concepts + gen rewrite",   CLEAN_A, CLEAN_B, G),
    ("M+G  messy concepts + gen rewrite",   MESSY_A, MESSY_B, G),  # control, expect WEAK
]


async def main():
    print(f"Concept-vs-claim 2x2 | judge={SYNTHESIS_COHERENCE_MODEL} | reps={REPS}\n")
    tmp = tempfile.mkdtemp(prefix="cvc_chroma_")
    store = MultiCollectionChromaStore(persist_directory=tmp)
    mm = ModelManager()
    filt = SynthesisFilter(chroma_store=store, model_manager=mm, synthesis_memory=SynthesisMemory(store))
    filt.graph_memory = None; filt.entity_resolver = None
    _PASS = {"MODERATE", "STRONG"}

    for label, a, b, claim in CELLS:
        levels = []
        for _ in range(REPS):
            sc = SynthesisCandidate(concept_a=a[:80], concept_b=b[:80], connection_claim=claim,
                                    walk_path=["self", a.lower()[:40], b.lower()[:40], "self"],
                                    source_domains={"da", "db"}, endpoint_distance=0.5, timestamp=_dt.now())
            result = SynthesisResult(candidate=sc)
            try:
                await filt._stage_5_coherence_judge(result)
                lvl = result.coherence_level.name if result.coherence_level else "None"
            except Exception as e:
                lvl = f"ERR:{type(e).__name__}"
            levels.append(lvl)
        npass = sum(1 for l in levels if l in _PASS)
        print(f"  {label:42} -> PASS {npass}/{REPS}  {dict(Counter(levels))}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
