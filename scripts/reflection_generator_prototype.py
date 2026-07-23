#!/usr/bin/env python3
"""
Reflection generator prototype — the FIRST end-to-end run on REAL data.

Pipeline (the "start small" step in docs/REFLECTION_SYNTHESIS_UNIFIED_SCORER.md):

    harvest real life-contexts  ->  GENERATOR (new)  ->  EXISTING judge
    (production chroma, read-only)   (1 LLM call)        (_stage_5_coherence_judge)

The investigation pinned the bottleneck to the GENERATOR and left the judge alone:
the existing structural judge is already a correct reflection truth-gate. So this
prototype changes nothing about the judge. It tests ONE new artifact — a generator
prompt that enforces the two requirements the judge's boundary implies:

  (1) pair GENUINELY-DISTINCT contexts/figures — never one figure with an
      abstraction of itself; never the same act in a different medium
      (kills the old dad/dad and advisor-email/mail failure modes).
  (2) articulate a TUNABLE mechanism that transfers — a single parameter whose
      change predicts behavior in BOTH contexts — not a co-observed feeling.

Division of labor: the generator proposes (liberally), the existing adversarial
judge disposes. The experiment's question is: of the generator's proposals built
from the owner's REAL memory, how many survive the real judge, and are the
survivors genuinely good?

Read-only on production chroma (harvest queries only). The judge runs against an
isolated temp chroma; no production writes. ~1 generation call + ~1-2 opus judge
calls per candidate. Usage: python scripts/reflection_generator_prototype.py
"""
import asyncio
import json
import os
import re
import sys
import tempfile
from collections import Counter, OrderedDict
from datetime import datetime
from datetime import datetime as _dt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from config.app_config import CHROMA_PATH, SYNTHESIS_COHERENCE_MODEL
from models.model_manager import ModelManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from memory.synthesis_memory import SynthesisMemory
from knowledge.synthesis_filter import SynthesisFilter
from knowledge.synthesis_models import SynthesisCandidate, SynthesisResult

# Thematic seeds spread across life-domains so the harvested pool isn't all one
# topic — the generator needs genuinely-distinct contexts to pair across.
SEEDS = [
    "my studies, thesis, research, and academic work",
    "my family, my parents, my siblings",
    "my friends and my social life",
    "my habits, routines, and the things I do every day",
    "my fears, my anxieties, the things I avoid",
    "my health, my body, my sleep",
    "money, work, and my responsibilities",
    "how I handle conflict, criticism, and authority",
]

GEN_SYSTEM = (
    "You generate candidate REFLECTION insights for one person, drawn ONLY from "
    "observations taken from their own memory. A reflection insight names ONE "
    "underlying mechanism that operates in TWO genuinely-distinct areas of their "
    "life. You are a generator, not a judge — a separate adversarial reviewer will "
    "reject weak candidates. Propose only your strongest; do not pad the list."
)

GEN_PROMPT = """Here are real observations from one person's journal (their own words, lightly trimmed):

{contexts}

EXEMPLARS — the shape that works (from a DIFFERENT person; do NOT reuse their
content, only their FORM). Each names two SEPARATE, self-standing episodes in
different life-systems, then a mechanism with a multi-parameter differential
prediction:
  - "Procrastinating on my thesis and avoiding medical appointments each defer a
     feared external evaluation for short-term relief at a larger later cost. The
     avoidance gets stronger the CLOSER the evaluation AND the WORSE I expect the
     verdict — so I stall longest on exactly the things I expect to go worst, in
     either domain." (academic x medical; two complete avoidance episodes, not one
     causing the other; prediction ranks WHICH cases are worst.)
  - "Deferring to my advisor when I think he's wrong and agreeing with my dad as a
     teenager to keep peace are the same submission to an authority I depend on:
     when someone controls a resource I need, I trade my own judgment for security.
     It predicts the deference WEAKENS as my dependence on that person falls, and
     reappears with any future figure who controls something I rely on." (career x
     family-of-origin; the mechanism makes a prediction that could be FALSE.)
What makes these pass, and your earlier drafts fail: (a) two INDEPENDENT episodes
that each stand alone — NOT "A causes B" (push -> crash is one process, not a
mapping); (b) a prediction that could be wrong and that RANKS or ORDERS cases, not a
single "more X -> more bad feeling" arrow.

Produce candidate REFLECTION insights. Each pairs TWO of the observations above and
names a single mechanism that runs in both. Obey BOTH hard requirements:

REQUIREMENT 1 — the two contexts must come from genuinely DIFFERENT life-systems.
  A "life-system" is a whole domain with its own actors, stakes, and rules — e.g.:
  academic/studies, medical/health, substance use, romantic, friendship, family-of-
  origin, money/work, creative work, exercise/body. The whole point of a reflection
  is that ONE mechanism unexpectedly TRANSFERS ACROSS a system boundary.
  FORBIDDEN — pairing two items from the SAME system. These all fail:
    - kavarin use <-> heavy drinking  (both = substance use: ONE system)
    - falling behind in a course <-> deferring an accommodations request  (both =
      academic tasks: ONE system)
    - hesitating to text a friend <-> wishing people cared  (both = social
      reassurance: ONE system)
  If both contexts would be filed under the same heading above, REJECT the pair —
  "I do the same thing in two corners of one domain" is consistency, not insight.
  Also forbidden: one figure/activity with an abstraction of itself; the same act in
  a different medium ("ignoring emails" vs "ignoring physical mail").
  The test: do the two contexts have DIFFERENT actors and DIFFERENT stakes? If they
  share the actor and stakes, they're the same system — reject.

REQUIREMENT 2 — a STRUCTURED mechanism, not a monotone feeling-law.
  The mechanism must have INTERNAL STRUCTURE that makes a DIFFERENTIAL prediction —
  one of: a feedback loop (A drives B which drives more A), a trade-off (buying X now
  costs Y later), or two-or-more interacting parameters that predict WHICH instances
  are worse than others.
  FORBIDDEN — a single monotone arrow like "the more unresolved things pile up, the
  more anxious I get, so I avoid them." That law fits almost any situation, so it
  discriminates nothing and is NOT an insight. If your mechanism is "more X -> more
  bad feeling -> avoid/withdraw," it is too generic; discard it.
  Test your own candidate: could you swap in two totally different activities and the
  claim still reads true? If yes, it's a monotone truism — discard it. A real
  mechanism predicts something specific: which case is worse, or how the two
  reinforce each other over time.

  GOOD shape (structured): "Avoiding my thesis and avoiding the doctor both defer a
  feared evaluation for short-term relief at a larger later cost; the avoidance gets
  STRONGER the closer the evaluation is AND the worse I expect the verdict — so it
  predicts I'll stall longest on exactly the things I expect to go worst."
  BAD shape (monotone): "Studying and exercising both depend on how much structure my
  day has — with structure I follow through, without it I stall." (one arrow, fits
  everything — REJECT).

Write each claim in the first person. It must (a) name both contexts, (b) state the
shared mechanism, and (c) state the DIFFERENTIAL prediction it makes (which instance
is worse, or how the two reinforce over time).

Use ONLY the observations above as biographical material — do not invent new facts
about the person. Quality over quantity: if you cannot find genuinely-distinct pairs
sharing a STRUCTURED mechanism, return fewer candidates (or an empty list) rather
than forcing weak ones.

Return ONLY a JSON array, 0 to 5 items, each:
  {{"context_a": "...", "context_b": "...", "parameter": "<the structured mechanism>", "claim": "<the first-person insight>"}}
"""


# Daemon's own session post-mortems live in `reflections` and pollute the pool with
# meta-text about the ASSISTANT, not the user's life. Drop anything that looks like one.
_META_NOISE = re.compile(
    r"the assistant|what went well|what could (be|have been)|areas for improvement|"
    r"session topic|the user'?s? (emotion|feeling|need)|empathetic response",
    re.IGNORECASE)


def harvest(store, per_seed=4, cap=24):
    """Pull a deduped pool of real life-contexts from the journal + facts (read-only).

    Source priority: obsidian_notes (the real journal — concrete life-narrative) is
    primary; facts supplements with atomic specifics. `reflections` is deliberately
    excluded — it's Daemon's session post-mortems, not the user's life.
    """
    pool = OrderedDict()  # content -> (collection, id)
    for seed in SEEDS:
        for coll in ("obsidian_notes", "facts"):
            try:
                rows = store.query_collection(coll, seed, n_results=per_seed)
            except Exception as e:
                print(f"  [harvest] {coll} <- {seed[:30]!r} ERR {type(e).__name__}: {e}")
                continue
            for r in rows:
                txt = (r.get("content") or "").strip()
                if not txt or len(txt) < 25 or _META_NOISE.search(txt):
                    continue
                txt = re.sub(r"\s+", " ", txt)
                if len(txt) > 320:
                    txt = txt[:317] + "..."
                if txt not in pool:
                    pool[txt] = (coll, r.get("id"))
    items = list(pool.items())
    return [(txt, meta) for txt, meta in items][:cap]


def _parse_candidates(raw):
    """Lenient JSON-array extraction from the generator output."""
    if not raw:
        return []
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    blob = m.group(0) if m else raw
    try:
        data = json.loads(blob)
    except Exception:
        return []
    out = []
    if isinstance(data, list):
        for d in data:
            if isinstance(d, dict) and d.get("claim") and d.get("context_a") and d.get("context_b"):
                out.append(d)
    return out


async def generate(mm, contexts):
    numbered = "\n".join(f"  {i+1}. {txt}" for i, (txt, _meta) in enumerate(contexts))
    prompt = GEN_PROMPT.format(contexts=numbered)
    raw = await mm.generate_once(
        prompt,
        model_name=SYNTHESIS_COHERENCE_MODEL,
        system_prompt=GEN_SYSTEM,
        max_tokens=1600,
        temperature=0.5,
        disable_reasoning=True,
    )
    cands = _parse_candidates(raw)
    return cands, raw


async def judge_candidate(filt, cand):
    a = str(cand.get("context_a", ""))[:80]
    b = str(cand.get("context_b", ""))[:80]
    claim = str(cand.get("claim", ""))
    sc = SynthesisCandidate(
        concept_a=a, concept_b=b, connection_claim=claim,
        walk_path=["self", a.lower(), b.lower(), "self"],
        source_domains={"domain_a", "domain_b"}, endpoint_distance=0.5, timestamp=_dt.now())
    result = SynthesisResult(candidate=sc)
    try:
        sr = await filt._stage_5_coherence_judge(result)
        lvl = result.coherence_level.name if result.coherence_level else "None"
        passed = bool(getattr(sr, "passed", False))
    except Exception as e:
        lvl, passed = f"ERROR:{type(e).__name__}", False
    just = (getattr(result, "coherence_justification", "") or "")
    return lvl, passed, just


async def main():
    print(f"Reflection generator prototype | gen+judge={SYNTHESIS_COHERENCE_MODEL}\n")

    # Harvest from PRODUCTION (read-only).
    prod = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    contexts = harvest(prod)
    print(f"Harvested {len(contexts)} real life-contexts:")
    for i, (txt, (coll, _id)) in enumerate(contexts, 1):
        print(f"  {i:2}. [{coll[:4]}] {txt[:110]}")
    if len(contexts) < 4:
        print("\nToo few contexts harvested — aborting.")
        return

    # GENERATE.
    print("\nGenerating candidates...", flush=True)
    mm = ModelManager()
    cands, raw = await generate(mm, contexts)
    print(f"Generator returned {len(cands)} candidate(s).")
    if not cands:
        print("\nRaw generator output (first 800 chars):\n" + (raw or "")[:800])
        return

    # JUDGE with the EXISTING production judge (isolated temp chroma).
    tmp = tempfile.mkdtemp(prefix="reflgen_proto_chroma_")
    store = MultiCollectionChromaStore(persist_directory=tmp)
    sm = SynthesisMemory(store)
    filt = SynthesisFilter(chroma_store=store, model_manager=mm, synthesis_memory=sm)
    filt.graph_memory = None
    filt.entity_resolver = None

    print("\n" + "=" * 88)
    rows = []
    for n, cand in enumerate(cands, 1):
        lvl, passed, just = await judge_candidate(filt, cand)
        rows.append({**cand, "level": lvl, "passed": passed, "justification": just})
        verdict = "PASS" if lvl in ("MODERATE", "STRONG") else "fail"
        print(f"[{n}/{len(cands)}] {lvl:9} [{verdict}]  param={str(cand.get('parameter',''))[:60]}")
        print(f"     A: {str(cand.get('context_a',''))[:80]}")
        print(f"     B: {str(cand.get('context_b',''))[:80]}")
        print(f"     claim: {str(cand.get('claim',''))[:200]}")
        if just:
            print(f"     ↳ {' / '.join(l.strip() for l in just.splitlines() if l.strip())[:320]}")
        print("-" * 88, flush=True)

    _PASS = {"MODERATE", "STRONG"}
    passes = [r for r in rows if r["level"] in _PASS]
    print("=" * 88)
    print(f"RESULT: {len(passes)}/{len(rows)} generated candidates PASSED the existing judge "
          f"(levels={dict(Counter(r['level'] for r in rows))})")
    print("=" * 88)
    print("READ:")
    print("  >=1 PASS that is genuinely good -> the generator+existing-judge loop produces real")
    print("       reflection for the first time; next is self-R novelty gating.")
    print("  0 PASS -> inspect: did the generator violate a requirement, or did the judge reject")
    print("       on a principle? (the judge is trusted; suspect the generator first.)")

    ts = datetime.now()
    outdir = os.path.join(_REPO, "data", "synthesis_discovery")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"reflection_generator_proto_{ts:%Y%m%d_%H%M%S}.json")
    json.dump({"timestamp": ts.isoformat(timespec="seconds"),
               "model": SYNTHESIS_COHERENCE_MODEL,
               "n_contexts": len(contexts),
               "raw_generator_output": raw,
               "rows": rows}, open(out, "w"), indent=2, default=str)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
