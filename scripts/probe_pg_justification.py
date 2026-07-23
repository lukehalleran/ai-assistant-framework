#!/usr/bin/env python3
"""Why does P pass and G fail? Print the judge's full justification for each (clean
concepts, same insight). Distinguishes 'judge rewards terse style' (concerning) from
'G's decoupling prediction undermines its own shared-structure claim' (reassuring)."""
import asyncio, os, sys, tempfile
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from datetime import datetime as _dt
from models.model_manager import ModelManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from memory.synthesis_memory import SynthesisMemory
from knowledge.synthesis_filter import SynthesisFilter
from knowledge.synthesis_models import SynthesisCandidate, SynthesisResult
from scripts.probe_concept_vs_claim import CLEAN_A, CLEAN_B, P, G

async def main():
    tmp = tempfile.mkdtemp(prefix="pgjust_")
    store = MultiCollectionChromaStore(persist_directory=tmp)
    mm = ModelManager()
    filt = SynthesisFilter(chroma_store=store, model_manager=mm, synthesis_memory=SynthesisMemory(store))
    filt.graph_memory = None; filt.entity_resolver = None
    for tag, claim in (("P (passes)", P), ("G (fails)", G)):
        sc = SynthesisCandidate(concept_a=CLEAN_A, concept_b=CLEAN_B, connection_claim=claim,
                                walk_path=["self", "a", "b", "self"], source_domains={"da", "db"},
                                endpoint_distance=0.5, timestamp=_dt.now())
        r = SynthesisResult(candidate=sc)
        await filt._stage_5_coherence_judge(r)
        lvl = r.coherence_level.name if r.coherence_level else "None"
        print(f"\n===== {tag}  ->  {lvl} =====")
        print((getattr(r, "coherence_justification", "") or "").strip())

if __name__ == "__main__":
    asyncio.run(main())
