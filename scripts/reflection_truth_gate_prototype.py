#!/usr/bin/env python3
"""
The UNBUNDLED reflection truth gate (the §7d next build).

§7d showed the existing structural judge is NOT a correct reflection gate: its verdict
demands cross-system COUPLING (Gentner isomorphism = the discovery criterion), which
reflection — parallel-independent by nature — structurally fails. But the judge's
de-jargon EXTRACTION is excellent (it pulled the real mechanism out of both P and G).

This gate keeps the extraction's rigor and replaces the verdict:
  VALID   = a real, nameable, PREDICTIVE mechanism that operates in EACH of the two
            contexts (known/general is fine; the two contexts may be independent).
  INVALID = the two are linked only by a shared feeling, category, or surface theme,
            with no mechanism that predicts behavior.
It explicitly does NOT require the two systems to be coupled.

VALIDATION (binary, decisive). Run the probe set; the gate is only trustworthy if:
  - the two SURFACE controls FAIL (same-dish↔rewatch="comfort"; noise↔fan="sound"), AND
  - genuine reflection PASSES — especially G, the HONEST explicitly-independent claim
    the old judge reliably rejected, plus advisor↔dad/anchor/rehearsing.
If it cannot separate those, the unbundling is wrong and §7d reopens.
(kavarin↔midterm is an observe-only probe: real energy-debt mechanism but same-system —
under the unbundled design it should PASS the TRUTH gate; its triviality is self-R's job.)

Isolated; no production writes. ~len(SET)*REPS model calls (single-pass, no skeptic).
Usage: python scripts/reflection_truth_gate_prototype.py [REPS]
"""
import asyncio, os, re, sys
from collections import Counter
from datetime import datetime as _dt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from config.app_config import SYNTHESIS_COHERENCE_MODEL
from models.model_manager import ModelManager
from scripts.probe_concept_vs_claim import CLEAN_A, CLEAN_B, P, G

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3

GATE_SYSTEM = (
    "You assess whether a candidate REFLECTION names a real GENERATIVE MECHANISM that "
    "operates in two areas of one person's life. This is NOT a test of scientific novelty "
    "or cross-domain isomorphism: the two areas are EXPECTED to be independent parts of one "
    "life, and the mechanism is EXPECTED to be a known, general human pattern. Two "
    "independent instances of one real mechanism is exactly what a reflection IS — do not "
    "require the two contexts to be coupled or to predict each other. Your job is to separate "
    "a GENERATIVE mechanism (a tunable factor whose change would make the person behave "
    "differently, in circumstances BEYOND the two observed instances) from a mere "
    "RE-DESCRIPTION of the two behaviors. Beware the charitable trap: you can invent a "
    "plausible-sounding mechanism for almost any pair. A real reflective mechanism predicts "
    "a COUNTERFACTUAL ('if the factor were different, I'd act differently'); a thin theme only "
    "restates why the person does the two things you already see ('I do these because I like "
    "the familiar') and supports no counterfactual."
)

GATE_PROMPT = """Candidate reflection:
  CONTEXT A: {a}
  CONTEXT B: {b}
  CLAIM: {claim}

Step 1 — DE-JARGON. Strip the academic/figurative language. State, in plain words, the
underlying mechanism the claim asserts.

Step 2 — TUNABLE FACTOR + COUNTERFACTUAL. Name the factor the mechanism says drives the
behavior. Does a CHANGE in that factor predict a CHANGE in the person's behavior, in a
circumstance NOT already described by the two observations? State the counterfactual
explicitly ("if <factor> were <different>, the person would <do Y differently>").
  - If you cannot state such a counterfactual — if the "mechanism" only re-describes or
    explains the two behaviors already given, with nothing that would change under a
    different condition — then it is a THEME, not a generative mechanism.
  - Do NOT require the two contexts to predict each other; each may run the factor
    independently. Do NOT penalize a known/general factor.

Step 3 — VERDICT:
  VALID   = a generative mechanism: a tunable factor with a real counterfactual prediction
            that holds in EACH context and reaches beyond the two observed instances.
  INVALID = only re-describes the two behaviors / linked by a shared feeling, category, or
            theme / no counterfactual that goes beyond what was already observed.

Respond exactly:
RATING: VALID  (or INVALID)
DE-JARGON: <one line>
COUNTERFACTUAL: <the if-then, or "none">
REASON: <one line>"""

# (label, expect, a, b, claim)
SET = [
    ("surface: dish↔rewatch", "INVALID",
     "always ordering the same dish at restaurants", "rewatching shows I have already seen",
     "Both are about comfort and familiarity — I gravitate toward things I already know I'll enjoy because the "
     "familiarity feels good and safe."),
    ("surface: noise↔fan", "INVALID",
     "liking background noise when I work", "sleeping better with a fan on",
     "Both connect to sound: a constant gentle noise in the background helps me settle, whether I'm trying to "
     "focus or trying to fall asleep."),
    ("anchor: procrast↔doctor", "VALID",
     "procrastinating on my thesis", "avoiding doctor's appointments",
     "Procrastinating on my thesis and avoiding medical appointments run the same mechanism: each defers a "
     "moment of external evaluation I expect to go badly, buying short-term relief at the cost of a larger "
     "future penalty; the nearer the evaluation and the worse I expect the verdict, the stronger the avoidance."),
    ("real: rehearse↔rewrite", "VALID",
     "rehearsing arguments in the shower before a hard conversation", "rewriting a text ten times before sending",
     "Both are pre-emptive control attempts: I try to script every branch of an interaction to eliminate any "
     "chance of being caught off-guard and judged, but the rehearsal raises the stakes so the real exchange "
     "feels more dangerous, which drives more rehearsal."),
    ("advisor↔dad (P, terse)", "VALID", CLEAN_A, CLEAN_B, P),
    ("advisor↔dad (G, honest/independent)", "VALID", CLEAN_A, CLEAN_B, G),
    ("observe same-system: kavarin↔midterm", "?",
     "taking stimulants to manufacture a productive day", "pushing through a midterm on no sleep",
     "Dosing stimulants to manufacture a good day and white-knuckling an exam on no sleep are the same overdraw: "
     "I spend tomorrow's recovery reserve for today's output, and the size of the eventual crash scales with how "
     "hard I pushed."),
]


def _rating(raw):
    m = re.search(r"RATING:\s*(VALID|INVALID)", raw or "", re.IGNORECASE)
    return m.group(1).upper() if m else "PARSE_ERR"


async def main():
    print(f"Reflection TRUTH gate (unbundled) | model={SYNTHESIS_COHERENCE_MODEL} | reps={REPS}\n")
    mm = ModelManager()
    rows = []
    for label, expect, a, b, claim in SET:
        ratings = []
        last_raw = ""
        for _ in range(REPS):
            raw = await mm.generate_once(
                GATE_PROMPT.format(a=a, b=b, claim=claim),
                model_name=SYNTHESIS_COHERENCE_MODEL, system_prompt=GATE_SYSTEM,
                max_tokens=400, temperature=0.1, disable_reasoning=True)
            ratings.append(_rating(raw))
            last_raw = raw
        nvalid = sum(1 for r in ratings if r == "VALID")
        hit = ("ok" if ((expect == "VALID" and nvalid == REPS) or (expect == "INVALID" and nvalid == 0))
               else ("??" if expect == "?" else "MISS"))
        dj = ""
        m = re.search(r"DE-JARGON:\s*(.+)", last_raw or "")
        if m:
            dj = m.group(1).strip()[:120]
        rows.append((label, expect, nvalid, dict(Counter(ratings)), hit))
        print(f"  {label:38} expect={expect:8} VALID {nvalid}/{REPS} [{hit}]  {dict(Counter(ratings))}")
        if dj:
            print(f"      de-jargon: {dj}", flush=True)

    print("\n" + "=" * 80)
    surf = [r for r in rows if r[0].startswith("surface")]
    genuine = [r for r in rows if r[1] == "VALID"]
    surf_ok = all(r[2] == 0 for r in surf)
    genuine_ok = all(r[2] == REPS for r in genuine)
    g_row = next((r for r in rows if "(G," in r[0]), None)
    print(f"surface controls all INVALID: {surf_ok}   genuine all VALID: {genuine_ok}")
    if g_row:
        print(f"G (honest/independent — old judge REJECTED this): VALID {g_row[2]}/{REPS}")
    if surf_ok and genuine_ok:
        print("=> UNBUNDLING WORKS: the gate rejects surface themes AND accepts honest reflection")
        print("   (incl. G). This is a reflection TRUTH gate. Next: self-R novelty (§8).")
    elif not surf_ok:
        print("=> LEAK: a surface control passed — gate too lenient; tighten the 'no mechanism' rigor.")
    else:
        print("=> STILL FLOORS genuine reflection — unbundling insufficient; §7d reopens.")


if __name__ == "__main__":
    asyncio.run(main())
