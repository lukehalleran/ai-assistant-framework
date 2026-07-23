#!/usr/bin/env python3
"""
Is the existing judge's MODERATE pass on genuine reflection a STABLE signal or
WEAK/MODERATE boundary NOISE? (firms up the prototype's conclusion before it
overturns the 2026-06-29 #2 "judge is a correct reflection gate" claim)

Runs the EXISTING _stage_5_coherence_judge REPS times on 3 fixed claims:
  A: advisor<->dad hand-authored exemplar      (passed MODERATE in the gen-judge test)
  B: Oliver<->dad GENERATED, mechanism-identical to A (floored WEAK in the prototype)
  C: anchor procrastination<->doctor            (passed MODERATE; calibration insight)

READ:
  A reliably MODERATE, B reliably WEAK  -> real craft/quality difference; judge tracks
                                           something; the prototype generator is just weak.
  A and C coin-flip MODERATE/WEAK       -> reflection lives ON the judge's boundary; a
                                           given genuine reflection passes partly by luck.
  A,B,C all reliably WEAK now           -> the earlier MODERATEs were luck; the judge
                                           floors genuine reflection -> it is a correct
                                           FLOOR but NOT a reliable reflection PASS-gate.

Isolated temp chroma; no production writes. ~REPS*3 opus calls.
Usage: python scripts/probe_judge_reflection_stability.py [REPS]
"""
import asyncio
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

CLAIMS = [
    ("A advisor<->dad (hand, was MODERATE)",
     "deferring to my advisor even when I think he's wrong", "agreeing with my dad as a teenager to keep peace",
     "The same submission-to-an-authority-I-depend-on operates in both: when someone controls something I need "
     "— funding and approval now, shelter and approval then — I suppress my disagreement to protect the "
     "support, trading my own judgment for security. The mechanism predicts the deference weakens as my "
     "dependence on that person falls, and reappears with any future figure who controls a resource I rely on."),

    ("B Oliver<->dad (generated, was WEAK)",
     "swallowing my boss Oliver's arbitrary rules and covering extra shifts", "approaching my dad about a balance transfer",
     "Swallowing Oliver's arbitrary rules and covering extra shifts, and approaching my dad about a balance "
     "transfer, are the same trade of standing-ground for security with a resource-holder. It predicts my "
     "concession rises as my independent buffer of the thing they control falls — I push back least exactly "
     "when I can least afford to lose what they hold."),

    ("C anchor procrastination<->doctor (hand, was MODERATE)",
     "procrastinating on my thesis", "avoiding doctor's appointments",
     "Procrastinating on my thesis and avoiding medical appointments run the same mechanism: each defers a "
     "moment of external evaluation I expect to go badly, buying short-term relief from anticipated judgment "
     "at the cost of a larger future penalty. Swap the domain — academic for medical — and the dynamics "
     "transfer: the nearer the evaluation, the stronger the avoidance, and the avoidance scales with how "
     "negative I expect the verdict to be."),
]


async def main():
    print(f"Judge reflection-stability probe | judge={SYNTHESIS_COHERENCE_MODEL} | reps={REPS}\n")
    tmp = tempfile.mkdtemp(prefix="reflstab_chroma_")
    store = MultiCollectionChromaStore(persist_directory=tmp)
    mm = ModelManager()
    sm = SynthesisMemory(store)
    filt = SynthesisFilter(chroma_store=store, model_manager=mm, synthesis_memory=sm)
    filt.graph_memory = None
    filt.entity_resolver = None

    _PASS = {"MODERATE", "STRONG"}
    for label, a, b, claim in CLAIMS:
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
            print(f"  {label[:40]:40} -> {lvl}", flush=True)
        npass = sum(1 for l in levels if l in _PASS)
        print(f"  == {label[:40]:40} PASS {npass}/{REPS}  {dict(Counter(levels))}\n")


if __name__ == "__main__":
    asyncio.run(main())
