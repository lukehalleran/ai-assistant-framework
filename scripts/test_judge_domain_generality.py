#!/usr/bin/env python3
"""
Is the structural coherence judge DOMAIN-GENERAL? (tests the owner's unification hypothesis)

Hypothesis: personal insight and scientific insight share ONE underlying truth-structure — a
non-obvious relational mapping (Gentner structure-mapping). If so, the judge's rubric
(de-jargon test + variable-swap test) should rate genuinely-STRUCTURED personal insights as
highly as structured scientific ones, while still rejecting surface-only personal pairings.

We run each claim through ONLY Stage 5 (the judge) — isolating the truth-structure detector
from the wiki-tuned distance/novelty stages that mechanically reject personal concepts.

Test set (labeled):
  structured-personal   — real relational mappings (incl. the dad/financial-dependence one that
                          resonated): should PASS if the hypothesis holds.
  surface-personal      — thematic pairings, no structure (like the failed cross-store junk):
                          should be WEAK regardless.
  structured-scientific — known isomorphisms (controls): should be MODERATE/STRONG.

PREDICTION if unification holds: structured-personal ≈ structured-scientific (MODERATE/STRONG),
surface-personal = WEAK → one judge, two ground-truths (literature / psychologist). If
structured-personal gets WEAK, the justification tells us WHY (structural failure = real gap;
factual-skeptic failure = personal facts aren't externally verifiable → the psychologist is the
verifier, which still supports unification at the structural level).

~8 opus-judge calls. Isolated temp store. Usage: python scripts/test_judge_domain_generality.py
"""
import asyncio
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from config.app_config import SYNTHESIS_COHERENCE_MODEL
from models.model_manager import ModelManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from memory.synthesis_memory import SynthesisMemory
from knowledge.synthesis_filter import SynthesisFilter
from knowledge.synthesis_models import SynthesisCandidate, SynthesisResult
from datetime import datetime as _dt

# (category, concept_a, concept_b, claim)
SET = [
    ("structured-personal", "financial dependence on my father", "childhood approval-seeking",
     "Financial dependence on my father reproduces in adulthood the same submission-for-security "
     "loop that governed childhood: because he controls a resource I still need, I defer to his "
     "judgment and suppress disagreement to protect the support — exactly as a child trades "
     "autonomy for care. The medium changed (money, not protection) but the relational structure "
     "is invariant: control of a needed resource produces deference and self-suppression. This "
     "predicts the same pattern should appear with anyone who controls a resource I depend on — an "
     "advisor over funding, an employer over salary — and should weaken precisely as the dependence "
     "is removed."),
    ("structured-personal", "procrastinating on my thesis", "avoiding doctor's appointments",
     "Procrastinating on my thesis and avoiding medical appointments run the same mechanism: each "
     "defers a moment of external evaluation I expect to go badly, buying short-term relief from "
     "anticipated judgment at the cost of a larger future penalty. Swap the domain — academic for "
     "medical — and the dynamics transfer: the nearer the evaluation, the stronger the avoidance, "
     "and the avoidance scales with how negative I expect the verdict to be."),
    ("structured-personal", "over-preparing for social events", "over-testing my code before shipping",
     "Over-preparing before social situations and over-testing code before release are the same "
     "preemptive-perfectionism loop: both try to eliminate every possible flaw in advance to "
     "forestall rejection, and both paradoxically delay the very exposure being feared, which keeps "
     "the underlying fear untested. Raise the stakes of judgment in either domain and both behaviors "
     "intensify identically."),
    ("surface-personal", "eating ramen", "comfort",
     "Eating ramen connects to comfort because both involve warmth and a sense of familiarity, and "
     "a warm bowl of noodles feels comforting on a hard day."),
    ("surface-personal", "going to the gym", "pattern theory",
     "Going to the gym relates to pattern theory because workout routines repeat in patterns, and "
     "recognizing patterns helps build a consistent exercise habit."),
    ("surface-personal", "thinking about tomorrow", "tardiness",
     "Thinking about tomorrow connects to tardiness because both are about time, and planning for "
     "tomorrow can help you avoid being late."),
    ("structured-scientific", "simulated annealing", "metallurgical annealing",
     "Simulated annealing imports the structure of physical annealing directly: a temperature "
     "parameter sets the probability of accepting worse states, and lowering it gradually lets the "
     "system settle into a low-energy configuration while still escaping local minima — the same "
     "mechanism by which slow cooling lets a metal's atoms find a low-energy crystalline arrangement. "
     "Raise the cooling rate in either domain and both get trapped in suboptimal states."),
    ("structured-scientific", "predator-prey dynamics", "economic boom-bust cycles",
     "Predator-prey population dynamics and credit boom-bust cycles share one delayed-feedback "
     "oscillator: a resource (prey / cheap credit) lets the consumer (predators / borrowers) grow, "
     "the growth depletes or destabilizes the resource, the consumer crashes, and the resource "
     "recovers — a lagged negative feedback that produces self-sustaining oscillations in both. The "
     "feedback lag predicts the cycle period in each."),
]


async def main():
    print(f"Judge domain-generality test | judge={SYNTHESIS_COHERENCE_MODEL}\n")
    tmp = tempfile.mkdtemp(prefix="judgegen_chroma_")
    store = MultiCollectionChromaStore(persist_directory=tmp)
    mm = ModelManager()
    sm = SynthesisMemory(store)
    filt = SynthesisFilter(chroma_store=store, model_manager=mm, synthesis_memory=sm)

    rows = []
    for n, (cat, a, b, claim) in enumerate(SET, 1):
        cand = SynthesisCandidate(
            concept_a=a, concept_b=b, connection_claim=claim,
            walk_path=["test", a.lower(), b.lower(), "test"],
            source_domains={"domain_a", "domain_b"}, endpoint_distance=0.5, timestamp=_dt.now())
        result = SynthesisResult(candidate=cand)
        reason = meta = ""
        try:
            sr = await filt._stage_5_coherence_judge(result)
            lvl = result.coherence_level.name if result.coherence_level else "None"
            reason = getattr(sr, "reason", "") or ""
            meta = {k: str(v)[:300] for k, v in (getattr(sr, "metadata", None) or {}).items()}
        except Exception as e:
            lvl = f"ERROR:{type(e).__name__}"
        just = (getattr(result, "coherence_justification", "") or "")  # FULL, no truncation
        rows.append({"category": cat, "a": a, "b": b, "level": lvl,
                     "justification": just, "stage_reason": reason, "stage_meta": meta})
        print(f"  [{n}/{len(SET)}] {cat:22} -> {lvl:9} | {a} <-> {b}", flush=True)
        if just:
            print(f"        ↳ {just[:600]}", flush=True)

    # ---- report ----
    _PASS = {"MODERATE", "STRONG"}
    print("\n" + "=" * 80)
    print("JUDGE DOMAIN-GENERALITY — does one structural judge separate real-from-surface in BOTH?")
    print("=" * 80)
    for cat in ("structured-personal", "surface-personal", "structured-scientific"):
        rs = [r for r in rows if r["category"] == cat]
        levels = Counter(r["level"] for r in rs)
        passes = sum(1 for r in rs if r["level"] in _PASS)
        print(f"  {cat:24} pass(MOD/STRONG) {passes}/{len(rs)}   levels={dict(levels)}")
    print("=" * 80)
    sp = [r for r in rows if r["category"] == "structured-personal"]
    xp = [r for r in rows if r["category"] == "surface-personal"]
    ss = [r for r in rows if r["category"] == "structured-scientific"]
    sp_pass = sum(1 for r in sp if r["level"] in _PASS)
    xp_pass = sum(1 for r in xp if r["level"] in _PASS)
    ss_pass = sum(1 for r in ss if r["level"] in _PASS)
    print("READ:")
    if sp_pass >= 2 and xp_pass == 0 and ss_pass >= 1:
        print("  HYPOTHESIS SUPPORTED — the judge rates structured personal ≈ structured scientific")
        print("  and rejects surface personal. One truth-structure detector, domain-general.")
        print("  → the personal track needs the psychologist as ground-truth + personal plumbing, not a new judge.")
    elif sp_pass == 0:
        print("  DOMAIN GAP — even well-structured personal insights are rejected. Check the")
        print("  justifications: structural failure = real gap; factual-skeptic (pass 2) failure =")
        print("  personal facts aren't externally verifiable → the psychologist IS that verifier.")
    else:
        print("  MIXED / underpowered at this n — inspect the per-item justifications; don't force a call.")

    ts = datetime.now()
    out = os.path.join(_REPO_ROOT, "data", "synthesis_discovery",
                       f"judge_domain_generality_{ts:%Y%m%d_%H%M%S}.json")
    json.dump({"timestamp": ts.isoformat(timespec="seconds"), "judge": SYNTHESIS_COHERENCE_MODEL,
               "rows": rows}, open(out, "w"), indent=2, default=str)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    asyncio.run(main())
