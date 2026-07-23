#!/usr/bin/env python3
"""
Reflection-mode rubric flip test (settles the (a)-vs-(b) fork in
docs/REFLECTION_SYNTHESIS_UNIFIED_SCORER.md §7).

QUESTION: the structural coherence judge floors 20/22 therapist-graded positives at
WEAK. Is that because (a) reflection truth genuinely isn't structural, or (b) the
judge's CURRENT rubric demands isomorphism between two DISTINCT academic fields and
penalizes "one mechanism recurring across my own life-contexts" (the reflection
archetype) by construction?

TEST: re-run the most-personal therapist-4/5 claims through BOTH rubrics, Pass-1 only.
  - CURRENT rubric: copied verbatim from synthesis_filter._stage_5_coherence_judge.
  - REFLECTION rubric: ONE variable changed — the de-jargon test is IDENTICAL; the
    cross-field "variable swap / peer-review" bar becomes a "recurrence across the
    person's own contexts" bar; the reject-by-default system prompt is neutralized.

CONTROLS (so a flip means "recovered real mechanism", not "rubric got lenient"):
  - surface-personal (ramen↔comfort, tomorrow↔tardiness): MUST stay WEAK in both.
  - positive anchor (procrastination↔doctor): MUST stay MODERATE in both.

READ:
  targets WEAK→MODERATE while surface controls stay WEAK  => (b) fixable rubric bias.
  targets stay WEAK                                       => (a) different truth term.
  surface controls flip too                              => rubric merely lenient (confound).

No production writes. ~ (len(set)) * 2 opus calls. Usage: python scripts/test_reflection_rubric_flip.py
"""
import asyncio, json, os, sys
from collections import Counter
from datetime import datetime

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from config.app_config import CHROMA_PATH, SYNTHESIS_COHERENCE_MODEL
from models.model_manager import ModelManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from knowledge.synthesis_filter import SynthesisFilter

# (category, concept_a_match, concept_b_match)  -- matched case-insensitively against the 28
TARGETS = [
    ("target", "parents", "Family reunion"),
    ("target", "fear of not graduating", "Economics"),
    ("target", "in bed", "Ecosystem"),
    ("target", "at the gym", "Pattern theory"),
    ("target", "finishing grad school", "Earth system"),
    ("target", "eating", "Dishabituation"),
    ("target", "to do more", "Human performance"),
    ("surface", "ramen", "Comfort"),
    ("surface", "tomorrow", "Tardiness"),
]
# positive anchor — NOT in the 28; a clean cross-domain structure that should pass both.
ANCHOR = ("anchor", "procrastinating on my thesis", "avoiding doctor's appointments",
    "Procrastinating on my thesis and avoiding medical appointments run the same mechanism: "
    "each defers a moment of external evaluation I expect to go badly, buying short-term relief "
    "from anticipated judgment at the cost of a larger future penalty. Swap the domain — academic "
    "for medical — and the dynamics transfer: the nearer the evaluation, the stronger the "
    "avoidance, and the avoidance scales with how negative I expect the verdict to be.")


def _current_prompt(a, b, claim):
    system = (
        "You are a structural methods reviewer. Most cross-domain claims are "
        "surface metaphor dressed in academic vocabulary. Your job is to reject "
        "these. Only rate MODERATE when the structural mapping survives both the "
        "de-jargon test and variable swap test. When in doubt, rate WEAK.")
    prompt = (
        f"Evaluate whether this claimed cross-domain connection identifies a real "
        f"structural isomorphism or is just surface-level pattern matching.\n\n"
        f"Concept A (personal): {a}\nConcept B (Wikipedia): {b}\nClaimed connection: {claim}\n\n"
        f"Apply these tests:\n\n"
        f"1. DE-JARGON TEST: Strip all field-specific nouns from the claim. Read only "
        f"the verbs, constraints, and logic. Does any structural content remain, or "
        f"does it collapse into 'both involve X' / 'can be seen as Y'?\n\n"
        f"2. VARIABLE SWAP TEST: If the two systems are truly isomorphic, changing a "
        f"variable in Concept B's ruleset should predict what happens in Concept A's "
        f"ruleset. Can you state one such prediction? If not, the mapping is cosmetic.\n\n"
        f"Rating criteria:\n"
        f"- WEAK: Fails de-jargon test (no structural content without the vocabulary), "
        f"OR fails variable swap (no transferable predictions). Most connections are WEAK. "
        f"Broad thematic overlap ('both involve feedback', 'iterative process', "
        f"'can be viewed through the lens of') is WEAK.\n"
        f"- MODERATE: Passes both tests. The mechanism maps structurally — changing a "
        f"parameter in one domain predicts behavior in the other. The connection would "
        f"survive peer review in a comparative methods paper.\n"
        f"- STRONG: MODERATE plus the mapping generates non-obvious testable predictions.\n"
        f"- INVALID: Concepts are misunderstood or the claim is incoherent.\n\n"
        f"Respond in EXACTLY this format. Your FIRST line MUST be the rating:\n"
        f"RATING: <INVALID|WEAK|MODERATE|STRONG>\n"
        f"De-jargon: <the claim stripped of field-specific nouns>\n"
        f"Variable swap: <one concrete prediction, or 'NONE'>\n"
        f"Against: <strongest criticism>")
    return system, prompt


def _reflection_prompt(a, b, claim):
    # SINGLE-VARIABLE CHANGE: de-jargon test identical; cross-field isomorphism bar
    # -> recurrence-across-own-contexts bar; reject-by-default system prompt neutralized.
    system = (
        "You are evaluating a person's reflective insight about their own life. They "
        "may be recognizing one real mechanism that recurs across different areas of "
        "their life. Many such claims are still just thematic overlap dressed up — reject "
        "those. But a real recurring mechanism does NOT need to bridge two distinct "
        "academic fields: one genuine mechanism recurring across the person's own "
        "contexts is exactly what counts here. Rate MODERATE only when a real, nameable "
        "mechanism survives the de-jargon test.")
    prompt = (
        f"Evaluate whether this claim identifies a REAL recurring mechanism in the "
        f"person's life, or is just surface-level thematic overlap.\n\n"
        f"Concept A (personal): {a}\nConcept B: {b}\nClaimed connection: {claim}\n\n"
        f"Apply these tests:\n\n"
        f"1. DE-JARGON TEST: Strip all field-specific nouns from the claim. Read only "
        f"the verbs, constraints, and logic. Does any structural content remain, or "
        f"does it collapse into 'both involve X' / 'can be seen as Y'?\n\n"
        f"2. RECURRENCE TEST: A real mechanism makes a prediction — under the same "
        f"conditions it should appear in OTHER contexts of the person's life. Can you "
        f"state one such prediction (e.g. 'this should also appear with anyone who "
        f"controls a resource they depend on')? If not, the mapping is cosmetic.\n\n"
        f"NOTE: the two concepts do NOT need to be distinct academic fields. A single "
        f"mechanism recurring across the person's own life-contexts is exactly what "
        f"counts — that is reflection, not a category error.\n\n"
        f"Rating criteria:\n"
        f"- WEAK: Fails de-jargon test (no structural content without the vocabulary), "
        f"OR fails recurrence (no transferable prediction across contexts). Pure thematic "
        f"overlap ('both involve comfort', 'both repeat', 'both about time') is WEAK.\n"
        f"- MODERATE: Passes both tests. A real, nameable mechanism operates here and "
        f"predicts its own recurrence across the person's contexts.\n"
        f"- STRONG: MODERATE plus the mechanism was non-obvious — the person plausibly "
        f"had not named it before.\n"
        f"- INVALID: Concepts are misunderstood or the claim is incoherent.\n\n"
        f"Respond in EXACTLY this format. Your FIRST line MUST be the rating:\n"
        f"RATING: <INVALID|WEAK|MODERATE|STRONG>\n"
        f"De-jargon: <the claim stripped of field-specific nouns>\n"
        f"Recurrence: <one concrete prediction, or 'NONE'>\n"
        f"Against: <strongest criticism>")
    return system, prompt


def _load_28():
    store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    col = store._get_collection("synthesis_results")
    data = col.get(include=["metadatas", "documents"])
    out = []
    for m, d in zip(data["metadatas"], data["documents"]):
        if not m or str(m.get("human_grade", "")).strip() in ("", "None"):
            continue
        out.append({"a": str(m.get("concept_a") or ""), "b": str(m.get("concept_b") or ""),
                    "claim": str(m.get("connection_claim") or d or "").strip(),
                    "grade": int(float(m.get("human_grade")))})
    return out


async def _judge(mm, system, prompt):
    try:
        resp = await mm.generate_once(prompt=prompt, model_name=SYNTHESIS_COHERENCE_MODEL,
                                      system_prompt=system, max_tokens=500, temperature=0.1,
                                      disable_reasoning=True)
        txt = (resp or "").strip()
    except Exception as e:
        return f"ERROR:{type(e).__name__}", ""
    if not txt:
        return "EMPTY", ""
    return SynthesisFilter._parse_coherence_level(txt).name, txt


async def main():
    print(f"Reflection-mode rubric flip test | judge={SYNTHESIS_COHERENCE_MODEL}\n")
    graded = _load_28()
    mm = ModelManager()

    items = []
    for cat, am, bm in TARGETS:
        hit = next((g for g in graded if am.lower() in g["a"].lower() and bm.lower() in g["b"].lower()), None)
        if not hit:
            print(f"  [warn] no match for {am!r} <-> {bm!r}")
            continue
        items.append({"cat": cat, "a": hit["a"], "b": hit["b"], "claim": hit["claim"], "grade": hit["grade"]})
    items.append({"cat": ANCHOR[0], "a": ANCHOR[1], "b": ANCHOR[2], "claim": ANCHOR[3], "grade": "-"})

    rows = []
    for n, it in enumerate(items, 1):
        cs, ctxt = await _judge(mm, *_current_prompt(it["a"], it["b"], it["claim"]))
        rs, rtxt = await _judge(mm, *_reflection_prompt(it["a"], it["b"], it["claim"]))
        flip = "FLIP→MOD" if (cs == "WEAK" and rs in ("MODERATE", "STRONG")) else (
               "stay" if cs == rs else f"{cs}->{rs}")
        rows.append({**it, "current": cs, "reflection": rs, "flip": flip,
                     "current_just": ctxt, "reflection_just": rtxt})
        print(f"  [{n:2}/{len(items)}] {it['cat']:7} g{str(it['grade']):2} | "
              f"CUR {cs:9} -> REF {rs:9} | {flip:10} | {it['a'][:22]} <-> {it['b'][:22]}", flush=True)

    # ---- report ----
    _PASS = {"MODERATE", "STRONG"}
    tgt = [r for r in rows if r["cat"] == "target"]
    srf = [r for r in rows if r["cat"] == "surface"]
    anc = [r for r in rows if r["cat"] == "anchor"]
    tgt_flip = [r for r in tgt if r["current"] == "WEAK" and r["reflection"] in _PASS]
    srf_flip = [r for r in srf if r["current"] not in _PASS and r["reflection"] in _PASS]

    print("\n" + "=" * 80)
    print("REFLECTION-MODE RUBRIC FLIP TEST")
    print("=" * 80)
    print(f"  targets:  {len(tgt)}   WEAK->MODERATE flips: {len(tgt_flip)}/{len(tgt)}")
    for r in tgt:
        print(f"     {r['flip']:10} g{r['grade']} {r['a'][:26]} <-> {r['b'][:24]}")
    print(f"  surface controls (MUST stay WEAK): flips: {len(srf_flip)}/{len(srf)}")
    for r in srf:
        print(f"     {r['flip']:10} g{r['grade']} {r['a'][:26]} <-> {r['b'][:24]}")
    print(f"  positive anchor (procrastination<->doctor): CUR {anc[0]['current'] if anc else '?'} "
          f"-> REF {anc[0]['reflection'] if anc else '?'}  (both should be MODERATE/STRONG)")
    print("=" * 80)
    print("READ:")
    if len(tgt_flip) >= max(2, len(tgt)//2) and len(srf_flip) == 0:
        print("  (b) FIXABLE RUBRIC BIAS — reflection rubric recovers real mechanisms the")
        print("  distinct-fields bar was rejecting, WITHOUT flipping surface junk. The SAME")
        print("  detector + a reflection-mode rubric can serve reflection. -> proceed to self-R.")
    elif len(tgt_flip) == 0:
        print("  (a) DIFFERENT TRUTH TERM — even with the bias removed, targets stay WEAK.")
        print("  Reflection truth isn't structural; the therapist is the truth oracle.")
    elif len(srf_flip) > 0:
        print("  CONFOUND — surface controls also flipped; the reflection rubric is merely more")
        print("  lenient. Discount the target flips; tighten the rubric and re-run.")
    else:
        print("  MIXED — inspect per-item justifications; don't force a call.")

    ts = datetime.now()
    out = os.path.join(_REPO, "data", "synthesis_discovery", f"reflection_rubric_flip_{ts:%Y%m%d_%H%M%S}.json")
    json.dump({"timestamp": ts.isoformat(timespec="seconds"), "judge": SYNTHESIS_COHERENCE_MODEL, "rows": rows},
              open(out, "w"), indent=2, default=str)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
