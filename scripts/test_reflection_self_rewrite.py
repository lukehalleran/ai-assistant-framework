#!/usr/bin/env python3
"""
Does a generator SELF-REWRITE stage close the reflection loop unaided? (the §7c next step)

The prototype found a genuinely-good cross-system pair (Oliver<->dad) but worded it so
the judge floored it (0/4 WEAK) — it collapsed two systems into one global driver
("how tight my money is"). A human rewrite stating the mechanism as a PER-SYSTEM
relation passed 4/4 MODERATE. This tests whether the GENERATOR can do that rewrite
itself, with a built-in anti-laundering control.

Fixed inputs (byte-exact from the latest reflection_generator_proto_*.json):
  Oliver<->dad     good pair, bad craft   -> predict: rewrite -> MODERATE (loop closes)
  kavarin<->midterm genuinely same-system  -> CONTROL: rewrite must NOT pass. Either the
                                             rewriter refuses (verdict=same_system) or the
                                             judge still floors it WEAK. If it flips to
                                             MODERATE, the rewrite is laundering -> BAD.

READ:
  Oliver MODERATE + kavarin NOT-MODERATE -> self-rewrite fixes craft, judge still
                                           discriminates -> loop closes unaided.
  both MODERATE                         -> rewrite games the judge; inspect the kavarin
                                           rewrite for a fabricated second system.
  Oliver still WEAK                     -> craft isn't the (whole) lever; rewrite insufficient.

Isolated temp chroma; no production writes. ~2 rewrite + ~REPS*N judge opus calls.
Usage: python scripts/test_reflection_self_rewrite.py [REPS]
"""
import asyncio
import glob
import json
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime
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

REWRITE_SYSTEM = (
    "You refine candidate reflection insights for STRUCTURAL PRECISION. You do NOT "
    "judge or approve them — a separate reviewer does that. You only fix HOW the shared "
    "mechanism is worded. You never invent facts not implied by the two contexts."
)

REWRITE_PROMPT = """A reflection insight pairs two contexts from one person's life and names a mechanism
that runs in both. A common flaw collapses the two contexts into ONE global driver,
which destroys the structure — it then reads as a single rule, not a mapping between
two systems, and a reviewer rejects it.

Rewrite the claim so the mechanism is a PER-SYSTEM RELATION:
  - Treat each context as its OWN system with its OWN actor, OWN resource, OWN stakes.
  - The shared thing must be a RELATION between a per-system quantity and a per-system
    behavior, instantiated SEPARATELY in each context — e.g. "as my buffer of the
    resource THAT context's authority controls falls, my concession to THAT authority
    rises," said once for each context with its own actor and resource named.
  - FORBIDDEN: a single global variable (e.g. "how tight my money is") that spans both
    contexts. Name the distinct resource/actor/stakes of EACH context.
  - State the differential prediction it makes (which case is worse, or how each moves).

Do not invent facts. If the two contexts genuinely share the SAME actor, resource, and
stakes — i.e. they are ONE system, not two — they CANNOT honestly be made into a
two-system relation: return exactly {{"verdict": "same_system", "claim": null}}.
Otherwise return {{"verdict": "rewritten", "claim": "<rewritten first-person insight>"}}.

CONTEXT A: {a}
CONTEXT B: {b}
ORIGINAL MECHANISM: {mech}
ORIGINAL CLAIM: {claim}

Return ONLY the JSON object."""


def _parse_obj(raw):
    if not raw:
        return {}
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        return json.loads(m.group(0) if m else raw)
    except Exception:
        return {}


async def rewrite(mm, cand):
    raw = await mm.generate_once(
        REWRITE_PROMPT.format(a=cand.get("context_a", ""), b=cand.get("context_b", ""),
                              mech=cand.get("parameter", ""), claim=cand.get("claim", "")),
        model_name=SYNTHESIS_COHERENCE_MODEL, system_prompt=REWRITE_SYSTEM,
        max_tokens=700, temperature=0.3, disable_reasoning=True)
    return _parse_obj(raw), raw


async def judge_n(filt, a, b, claim, reps):
    _PASS = {"MODERATE", "STRONG"}
    levels = []
    for _ in range(reps):
        sc = SynthesisCandidate(
            concept_a=a[:80], concept_b=b[:80], connection_claim=claim,
            walk_path=["self", a.lower()[:40], b.lower()[:40], "self"],
            source_domains={"da", "db"}, endpoint_distance=0.5, timestamp=_dt.now())
        result = SynthesisResult(candidate=sc)
        try:
            await filt._stage_5_coherence_judge(result)
            lvl = result.coherence_level.name if result.coherence_level else "None"
        except Exception as e:
            lvl = f"ERR:{type(e).__name__}"
        levels.append(lvl)
    return levels, sum(1 for l in levels if l in _PASS)


async def main():
    path = sorted(glob.glob(os.path.join(_REPO, "data/synthesis_discovery/reflection_generator_proto_*.json")))[-1]
    rows = json.load(open(path))["rows"]
    print(f"Self-rewrite test | {os.path.basename(path)} | model={SYNTHESIS_COHERENCE_MODEL} | reps={REPS}\n")

    tmp = tempfile.mkdtemp(prefix="selfrw_chroma_")
    store = MultiCollectionChromaStore(persist_directory=tmp)
    mm = ModelManager()
    sm = SynthesisMemory(store)
    filt = SynthesisFilter(chroma_store=store, model_manager=mm, synthesis_memory=sm)
    filt.graph_memory = None
    filt.entity_resolver = None

    out_rows = []
    for cand in rows:
        a, b = cand.get("context_a", ""), cand.get("context_b", "")
        tag = f"{a[:24]} <-> {b[:24]}"
        print(f"==== {tag}  (orig judge={cand.get('level')})")
        rw, raw = await rewrite(mm, cand)
        verdict = rw.get("verdict")
        new_claim = rw.get("claim")
        print(f"  rewriter verdict: {verdict}")
        rec = {"pair": tag, "orig_level": cand.get("level"), "rw_verdict": verdict,
               "rw_claim": new_claim}
        if verdict == "rewritten" and new_claim:
            print(f"  rewritten claim: {new_claim[:240]}")
            levels, npass = await judge_n(filt, a, b, new_claim, REPS)
            rec["rw_judge"] = dict(Counter(levels))
            rec["rw_pass"] = f"{npass}/{REPS}"
            flag = "PASS" if npass >= 1 else "fail"
            print(f"  -> judge on REWRITE: PASS {npass}/{REPS}  {dict(Counter(levels))}  [{flag}]")
        else:
            print(f"  (no rewrite to judge — rewriter declined)")
            rec["rw_judge"] = None
            rec["rw_pass"] = "n/a"
        out_rows.append(rec)
        print("-" * 80, flush=True)

    print("=" * 80)
    print("READ:")
    good = next((r for r in out_rows if "Oliver" in r["pair"] or "boss" in r["pair"].lower() or "Swall" in r["pair"]), None)
    ctrl = next((r for r in out_rows if "kavarin" in r["pair"].lower() or "stimul" in r["pair"].lower() or "Dosing" in r["pair"]), None)
    for label, r in (("good-pair (Oliver<->dad)", good), ("control same-system (kavarin<->midterm)", ctrl)):
        if r:
            print(f"  {label:38} orig={r['orig_level']:6} rw_verdict={str(r['rw_verdict']):10} rw_judge={r['rw_pass']}")
    if good and ctrl:
        gp = good.get("rw_pass", "0/0").split("/")[0]
        cp = ctrl.get("rw_pass", "0/0").split("/")[0] if ctrl.get("rw_pass") not in (None, "n/a") else "0"
        if gp.isdigit() and int(gp) >= 1 and (ctrl["rw_verdict"] == "same_system" or (cp.isdigit() and int(cp) == 0)):
            print("\n  => LOOP CLOSES UNAIDED: self-rewrite lifts the good pair, control held. Wire self-R novelty next.")
        elif gp.isdigit() and int(gp) >= 1 and cp.isdigit() and int(cp) >= 1:
            print("\n  => REWRITE LAUNDERS: control also passed — inspect the control rewrite for a fabricated 2nd system.")
        else:
            print("\n  => REWRITE INSUFFICIENT: good pair still floored — craft isn't the (whole) lever.")

    ts = datetime.now()
    outp = os.path.join(_REPO, "data", "synthesis_discovery", f"reflection_self_rewrite_{ts:%Y%m%d_%H%M%S}.json")
    json.dump({"timestamp": ts.isoformat(timespec="seconds"), "model": SYNTHESIS_COHERENCE_MODEL,
               "reps": REPS, "rows": out_rows}, open(outp, "w"), indent=2, default=str)
    print(f"\nSaved -> {outp}")


if __name__ == "__main__":
    asyncio.run(main())
