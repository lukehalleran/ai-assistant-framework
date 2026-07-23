#!/usr/bin/env python3
"""
Does the EXISTING judge gate well-formed REFLECTION candidates? (the §7-next step in
docs/REFLECTION_SYNTHESIS_UNIFIED_SCORER.md)

The 28 graded candidates were the WRONG corpus (machine personal->wiki bridges). The
anchor `procrastination<->doctor` showed genuine reflection structure passes the
existing judge. This test feeds the existing judge a PROPER reflection corpus:
anchor-shaped candidates (two contexts in the person's OWN life sharing one mechanism),
PRE-REGISTERED and labeled by THERAPEUTIC-REALITY (would a therapist call this a real
pattern in the person's life?), NOT by structural-rubric-fit — so the test measures
whether the structural judge tracks the reflection axis, not my read of its rubric.

Categories + pre-registered predictions (production judge, Pass 1 + factual skeptic):
  cross_domain_real    two DISTINCT life-domains, one real mechanism  -> predict PASS
  same_relationship    one relationship/dynamic, recognized across     -> predict ???  (THE residual:
                       time/context (the dad-style case)                    `dad-now<->dad-as-child` died
                                                                            WEAK as "one relationship,
                                                                            not a mapping")
  thin_control         genuinely thematic, a therapist would NOT call   -> predict FAIL
                       it an insight
  anchor               the known-good calibration insight              -> predict PASS

READ:
  cross_domain_real PASS + thin_control FAIL  => judge correctly gates well-formed cross-domain reflection.
  same_relationship PASS                      => full unification; no rubric work needed.
  same_relationship FAIL                      => the residual is real; a within-relationship-recurrence
                                                 rubric tweak is the ONLY remaining truth-term work.
  cross_domain_real FAIL                      => judge genuinely can't do reflection (reopens everything).
  thin_control PASS                           => production judge leakier than the flip test implied.

Isolated temp chroma; runs the REAL _stage_5_coherence_judge (incl. skeptic pass). No
production writes. ~10-14 opus calls. Usage: python scripts/test_reflection_generator_judge.py
"""
import asyncio, json, os, sys, tempfile
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

# (category, predict, concept_a, concept_b, claim)
SET = [
    ("anchor", "PASS",
     "procrastinating on my thesis", "avoiding doctor's appointments",
     "Procrastinating on my thesis and avoiding medical appointments run the same mechanism: each defers a "
     "moment of external evaluation I expect to go badly, buying short-term relief from anticipated judgment "
     "at the cost of a larger future penalty. Swap the domain — academic for medical — and the dynamics "
     "transfer: the nearer the evaluation, the stronger the avoidance, and the avoidance scales with how "
     "negative I expect the verdict to be."),

    ("cross_domain_real", "PASS",
     "putting off replying to my advisor's emails", "letting unopened mail pile up on my desk",
     "Both run one avoidance loop: an item carrying a possible obligation sits unaddressed because opening it "
     "forces me to confront a demand I'm not ready to meet, and the longer it sits the more weight it accrues, "
     "which makes opening it harder still. The mechanism predicts the items I avoid longest are the ones I "
     "expect to contain the largest demand, and a trivially easy item gets handled at once regardless of "
     "whether it's an email or an envelope."),

    ("cross_domain_real", "PASS",
     "rehearsing arguments in the shower before a hard conversation", "rewriting a text ten times before sending",
     "Both are pre-emptive control attempts: I try to script every branch of an interaction in advance to "
     "eliminate any chance of being caught off-guard and judged, but the rehearsal itself raises the stakes so "
     "the real exchange feels more dangerous, which drives more rehearsal. Lower how much I fear the other "
     "person's judgment and both subside together; raise it and both intensify — the spoken and the written "
     "case move as one."),

    ("same_relationship", "?",
     "deferring to my advisor even when I think he's wrong", "agreeing with my dad as a teenager to keep peace",
     "The same submission-to-an-authority-I-depend-on operates in both: when someone controls something I need "
     "— funding and approval now, shelter and approval then — I suppress my disagreement to protect the "
     "support, trading my own judgment for security. The mechanism predicts the deference weakens as my "
     "dependence on that person falls, and reappears with any future figure who controls a resource I rely on."),

    ("same_relationship", "?",
     "feeling relief when my advisor cancels a meeting", "feeling relief when a close friend cancels plans",
     "The relief at cancellation reveals that each relationship has quietly become a site of anticipated "
     "evaluation rather than support — I'd been bracing to be assessed, not looking forward to connection. The "
     "mechanism predicts the relief scales with how much I expect to be judged in that specific relationship "
     "and vanishes in relationships where I expect no judgment, so it should grow precisely where I most fear "
     "falling short."),

    ("thin_control", "FAIL",
     "always ordering the same dish at restaurants", "rewatching shows I have already seen",
     "Both are about comfort and familiarity — I gravitate toward things I already know I'll enjoy because the "
     "familiarity feels good and safe."),

    ("thin_control", "FAIL",
     "liking background noise when I work", "sleeping better with a fan on",
     "Both connect to sound: a constant gentle noise in the background helps me settle, whether I'm trying to "
     "focus or trying to fall asleep."),
]


async def main():
    print(f"Reflection-generator judge test | judge={SYNTHESIS_COHERENCE_MODEL}\n")
    tmp = tempfile.mkdtemp(prefix="reflgen_chroma_")
    store = MultiCollectionChromaStore(persist_directory=tmp)
    mm = ModelManager()
    sm = SynthesisMemory(store)
    filt = SynthesisFilter(chroma_store=store, model_manager=mm, synthesis_memory=sm)
    filt.graph_memory = None; filt.entity_resolver = None

    rows = []
    for n, (cat, predict, a, b, claim) in enumerate(SET, 1):
        cand = SynthesisCandidate(
            concept_a=a, concept_b=b, connection_claim=claim,
            walk_path=["self", a.lower(), b.lower(), "self"],
            source_domains={"domain_a", "domain_b"}, endpoint_distance=0.5, timestamp=_dt.now())
        result = SynthesisResult(candidate=cand)
        try:
            sr = await filt._stage_5_coherence_judge(result)
            lvl = result.coherence_level.name if result.coherence_level else "None"
            passed = bool(getattr(sr, "passed", False))
        except Exception as e:
            lvl, passed = f"ERROR:{type(e).__name__}", False
        just = (getattr(result, "coherence_justification", "") or "")
        hit = "ok" if ((predict == "PASS" and lvl in ("MODERATE", "STRONG")) or
                       (predict == "FAIL" and lvl in ("WEAK", "INVALID"))) else (
              "??" if predict == "?" else "MISS")
        rows.append({"category": cat, "predict": predict, "a": a, "b": b,
                     "level": lvl, "passed": passed, "hit": hit, "justification": just})
        print(f"  [{n}/{len(SET)}] {cat:18} predict={predict:4} -> {lvl:9} [{hit}] | {a[:30]} <-> {b[:26]}", flush=True)
        if just:
            print(f"        ↳ {' / '.join(l.strip() for l in just.splitlines() if l.strip())[:400]}", flush=True)

    _PASS = {"MODERATE", "STRONG"}
    print("\n" + "=" * 84)
    print("REFLECTION-GENERATOR JUDGE TEST — does the EXISTING judge gate well-formed reflection?")
    print("=" * 84)
    for cat in ("anchor", "cross_domain_real", "same_relationship", "thin_control"):
        rs = [r for r in rows if r["category"] == cat]
        p = sum(1 for r in rs if r["level"] in _PASS)
        print(f"  {cat:20} pass(MOD/STRONG) {p}/{len(rs)}   levels={dict(Counter(r['level'] for r in rs))}")
    print("=" * 84)
    cdr = [r for r in rows if r["category"] == "cross_domain_real"]
    sr_ = [r for r in rows if r["category"] == "same_relationship"]
    thin = [r for r in rows if r["category"] == "thin_control"]
    cdr_p = sum(1 for r in cdr if r["level"] in _PASS)
    sr_p = sum(1 for r in sr_ if r["level"] in _PASS)
    thin_p = sum(1 for r in thin if r["level"] in _PASS)
    print("READ:")
    if cdr_p >= 2 and thin_p == 0:
        print("  Judge correctly gates well-formed CROSS-DOMAIN reflection (real pass, thin reject).")
        if sr_p == len(sr_):
            print("  Same-relationship reflections ALSO pass -> FULL unification; no rubric work needed.")
        elif sr_p == 0:
            print("  Same-relationship reflections FAIL -> the residual is real. A within-relationship-")
            print("  recurrence rubric tweak is the ONLY remaining truth-term work; cross-domain is done.")
        else:
            print(f"  Same-relationship MIXED ({sr_p}/{len(sr_)}) -> inspect justifications for the boundary.")
    elif cdr_p == 0:
        print("  Even well-formed cross-domain reflection FAILS -> judge genuinely can't gate reflection;")
        print("  reopen the truth-term question (NOT just the same-relationship residual).")
    if thin_p > 0:
        print(f"  WARNING: {thin_p}/{len(thin)} thin controls PASSED -> production judge leakier than expected.")

    ts = datetime.now()
    out = os.path.join(_REPO, "data", "synthesis_discovery", f"reflection_generator_judge_{ts:%Y%m%d_%H%M%S}.json")
    json.dump({"timestamp": ts.isoformat(timespec="seconds"), "judge": SYNTHESIS_COHERENCE_MODEL, "rows": rows},
              open(out, "w"), indent=2, default=str)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
