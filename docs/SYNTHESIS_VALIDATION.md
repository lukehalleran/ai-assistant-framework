# Synthesis Validation — Proving the Filter Without Being a Domain Expert

**Created**: 2026-06-26

The hard question this project keeps hitting: *a machine is proposing novel cross-domain
connections — in fields I'm not an expert in. How do I know any of it is real?* One
generous grader giving 4s and 5s is not an answer. This doc is the answer.

---

## The core insight

**Literature is the expert you don't have to be.** A connection that already appears in
published work is *expert-validated by definition*. So instead of judging validity yourself
(impossible outside your field), you use the corpus as ground truth. The whole validation
loop is built on this one move.

This decomposes the scary claim ("machine generates knowledge") into measurable parts. Only
the last mile — *is this specific novel connection real?* — needs a human expert, and that's
rare and high-value, exactly where you'd want to spend expert time.

---

## The protocol

### A. Judge discrimination test (does the filter separate real from spurious?)

The coherence judge has one job: tell real connections from surface metaphor. Test it against
ground truth, no grading required.

- **Positives** = *known* connections (literature confirms they're real). Run through the
  coherence judge → should mostly **PASS**. A judge that rejects literature-confirmed
  connections is broken. This is the judge's **sensitivity**.
- **Negatives** = connections that are *known-bad* or *forced*. Two kinds:
  - **Shuffle-negatives** (guaranteed-bad control): take a real claim written for pair (A,B)
    and relabel it as being about (A,C). The claim can't fit C, so a working judge *must*
    reject it. A pure floor.
  - **Far-pair / forced-negatives** (realistic floor): force the generator to articulate a
    bridge between two distant, unrelated concepts. Most are genuinely spurious → should
    **FAIL**. (Caveat: a few distant pairs *do* connect — that's the discovery you're
    hunting — so this set is "mostly negative," not pure. The pass-throughs are themselves
    interesting: real-or-error.)

**The metric is the GAP, not the absolute rates.** Positives passing 85% / negatives passing
10% = strong discrimination, the judge works. Positives 85% / negatives 80% = the judge
accepts everything, it's useless. You're measuring *separation* (signal-detection d').

### B. Generator rediscovery rate (does the generator find real structure?)

Generate connections, check each against literature (FAISS novelty/co-occurrence). The rate
at which the generator independently surfaces *already-known* connections is evidence it
finds real structure — random noise wouldn't match literature.

**Guard against the base-rate trap.** "Most are known → good" is gameable: sample obvious
co-occurring pairs and everything is trivially known while proving nothing. The real metric
is the generator's known-rate **vs a random-pair baseline at the same semantic distance**, on
*non-obvious* pairs. Above-baseline known-rate on distant pairs = real, non-obvious structure.

(The `rejected_known` capture — `synthesis_memory.store_known_connection`, GUI "Known
connections (rediscovered)" view, `[SYNTH KNOWN]` log line — feeds this. See
`SYNTHESIS_FILTER.md` Stage 3.)

### C. arXiv cross-corpus reindex delta (the masterstroke)

Your "novel candidate" set (passed coherence + passed novelty, unknown in Wikipedia) is a mix
of three things:
1. genuinely novel,
2. judge errors,
3. **real, but just not in Wikipedia.**

Re-index with arXiv, re-check known status. The candidate count **drops** — because bucket (3)
flips from unknown → known. That drop is *not a loss, it's a second proof*:

- The connections that flipped were **real connections your system found that Wikipedia didn't
  even contain**, confirmed by an *independent* corpus. Cross-corpus rediscovery is a stronger
  signal than wiki-only.
- The ones that **stay unknown** after arXiv are a **cleaner novel pile** — less (3)
  contamination, so a higher fraction is genuinely-novel-or-error.

The shrinkage rate is itself a metric: *what fraction of wiki-novel candidates turned out to be
arXiv-known* measures how much real-but-unindexed structure the system catches. A meaningful
drop = the system finds real connections that span corpora. **That is it working.** (This is
the strongest single argument for ingesting arXiv — it sharpens the one signal you care about,
not just "more candidates.")

---

## How the steps compose (the expertise-free loop)

1. Rediscovery rate (B) → **the generator finds real structure.**
2. Knowns-pass vs shuffled-fail gap (A) → **the judge discriminates real from spurious.**
3. Given a *validated* judge, the surviving unknowns are **credible candidates** — not just
   "stuff that slipped through."
4. arXiv reindex (C) → a chunk flip to known → **cross-corpus confirmation** + cleaner remainder.
5. Post-arXiv unknowns, having passed a validated judge and survived *two* corpora → **genuine
   novel leads** worth an expert's time.

Every step uses literature as ground truth. You never grade a domain you're not an expert in.

---

## The honest ceiling

A perfect rediscovery rate does **not** prove any *specific* novel output is real — it proves
the *system is capable* of producing real connections, which raises your prior (Bayesian) on
the novel ones. Post-arXiv unknowns stay **candidates, not discoveries.** The last mile (is
*this* novel connection real?) is the only place a human expert is irreducible.

But the claim changed shape: no longer "is this machine magic?" but "the machine demonstrably
rediscovers real connections across two corpora and discriminates them from nonsense — are
*these specific* survivors worth a look?" That's a normal, defensible scientific claim.

---

## Relationship to human grading

Human grading (you + therapist, two-layer panel — see `grading_plan.md`) becomes a
**supplement, not the gate**: use it to *calibrate the judge against your own labels* on the
personal-data candidates you CAN judge, and to spot-check the final novel leads. The gate is
**literature**, which scales to any corpus and needs no expertise. This is what unblocks the
two-month grading bottleneck: the 300-grade classifier was never the path to "proof" — this is.

---

## Implementation

`scripts/synthesis_validation.py` — runs test A (known-positives + shuffle-negatives +
curated known-bad negatives through the coherence judge, reports the discrimination gap), with
hooks for live positives from `get_known_connections()` and the arXiv reindex-delta (C) once
arXiv is ingested. Run it to turn this whole methodology into a single number you can watch
move.

Related: `knowledge/synthesis_filter.py` (Stage 3 novelty/known check, Stage 5 coherence
judge), `docs/SYNTHESIS_FILTER.md`, `docs/grading_plan.md`, `docs/GOALS.md` (Goal 5: arXiv
ingestion).

---

## Results log

### 2026-06-26 — first run found the judge was dead; fixed it

The very first Test A run returned a **0% discrimination gap** — but not because the judge
rated everything WEAK. It rated *nothing*: nearly every call logged "Could not parse coherence
level," so every candidate **defaulted** to WEAK. The coherence judge — the core of the filter
— had been **silently broken since the coherence model became `opus-4.8`**.

Root cause: the judge prompt put `Rating: <LEVEL>` as the **last** line. opus-4.8 is a verbose
reasoning model, so it either (a) swallowed the whole answer into its (hidden) reasoning channel
and returned **empty** visible content, or (b) wrote a long essay that **ran out of tokens
before reaching the Rating line**. A raw call at `max_tokens=400` returned `''`; at 600 with
reasoning off it wrote 1708 chars of analysis and *still* never emitted a clean `Rating:` line.
Net effect: since the model switch, the filter rejected ~everything at the coherence stage — a
large part of why the audit queue felt frozen and "I can't tell if it's working."

Fix (`_stage_5_coherence_judge`, both passes): put **`RATING:` on the FIRST line** (verdict
committed before the essay, truncation-proof and parseable), **disable reasoning** on the judge
calls (the structured analysis IS the reasoning — keep it visible, no swallow), small headroom
(400→500 / 150→200). Empirically validated before applying: rating-first → real isomorphism =
STRONG, surface metaphor = INVALID.

Re-run: **+50% gap (STRONG)** —
- POSITIVES (curated known-real): **5/8 pass** (entropy↔entropy STRONG, phase↔percolation
  STRONG, Bayesian↔Kalman STRONG, the annealings, SIR↔viral; the 3 WEAK are the judge's
  designed "when in doubt, WEAK" strictness — high precision, lower recall).
- Curated surface-metaphor NEGATIVES: **0/6 pass** (all correctly rejected).
- SHUFFLE-NEGATIVES: **1/8 pass** — and that one (SIR↔network percolation) is the judge being
  *right*: percolation theory genuinely is used in epidemic modeling, so the shuffle accidentally
  produced a real pair. True gap is cleaner than 50%.

Takeaway: the harness caught a critical, otherwise-invisible bug in **one run, zero human
grades** — and the fixed judge demonstrably discriminates. It is now the standing instrument to
confirm the judge still discriminates after any prompt/model/threshold change. (8+14 items is a
smoke test, not a definitive d′; the judge is conservative and may want recall tuning later.)

### 2026-06-26 — first live dreaming pass through the fixed judge

`scripts/test_end_to_end_synthesis.py --candidates 15` (all three generators → real filter,
live ChromaDB + FAISS wiki). Result: **13 generated, 0 accepted.** And that's *correct*:
- The judge did real work — `coherence_judge` rejected 5, averaging **9.2s each** (genuine
  analysis, not the WEAK-default it did pre-fix). novelty_external rejected 4 (saved as
  `rejected_known` — the rediscovery capture working).
- 0 accepted because the **candidates** were weak: WALK forced vague personal-graph nodes
  (`support↔fast`, `support↔normal`); XSTORE paired random personal facts with random wiki
  concepts (`jogging↔Price Theory`, `"so excited i can feel it" ↔ Adjunct (grammar)`).

**The bottleneck moved from the filter to the input.** This morning the (broken) judge rejected
everything; now the *generators* produce rejectable junk and the filter correctly kills it. That
is the "data over code" thesis made visible: the sparse personal graph (6 bridges < 40) and
random personal-fact↔wiki pairings can't produce strong candidates — only a denser, richer
concept space (wiki↔wiki, then arXiv) can. The personal anchor was always for *gradeability*,
not candidate quality.

### 2026-06-26 — wiki↔wiki experiment: the real bottleneck is PAIR SELECTION, not corpus

`scripts/wiki_synthesis_experiment.py --pairs 12` — sample wiki concepts, articulate bridges,
run novelty + the fixed judge. Result: **0% known, 0% coherence-pass** — but look at the *pairs*:
`Palio di Castellanza ↔ Where Mathematics Comes From`, `List of RTÉ One programmes ↔ A Secret
Atlas`, `Zubulake I (legal case) ↔ Pattern`. **Random, obscure, unrelated articles paired at
random.** Of 12 pairs, 6 couldn't even be articulated ("no connection"); the 6 that were all
scored WEAK and 0 were in literature.

Two findings, both important:
1. **This IS the null baseline.** Random concept pairs are not co-present in literature → ~0%
   known. Test B's whole point is that a real generator must beat this baseline. Now we have it.
2. **The hard problem is PAIR SELECTION, not the corpus and not articulation.** Any two things
   can be forced into a WEAK "connection" (that's why everything is WEAK). The generator's value
   was never "articulate a bridge between A and B" — it's "*find* pairs A,B that have a real,
   non-obvious connection worth articulating." Random distance (and obscure sampling) = noise.

So switching corpora (wiki, arXiv) alone won't help until sampling is fixed. The next lever is
**controlled-distance, core-concept sampling**: draw a prominent concept A, then find B at a
*moderate* cosine distance (related but not identical, cross-topic) — not random-far. That, not
"jack up the corpus," is what should move the known-rate above the random baseline.

### 2026-06-27 — controlled-distance run + the keystone discovery: the known-oracle was INVERTED

`scripts/synthesis_controlled_distance.py` (anchor on a prominent concept, take a partner at a
controlled true-cosine band; three bands per anchor so the result self-interprets). Clean run
(true-cosine bands, concept-only partner filter), n≈8/band:

| band | true-cosine | coherence-pass |
|------|-------------|----------------|
| near | 0.55–0.79 | **4/8 (50%)** |
| moderate | 0.30–0.55 | 1/7 (14%) |
| random | ~0 | 0/3 (0%) |

Two findings. (1) Coherence-pass falls *monotonically* with distance — "moderate is the sweet
spot" is **rejected**; closer is better *for judge-pass*. (2) But the near-band passes are
*trivial* same-field restatements (`entropy ↔ Chemical thermodynamics`, `diffusion ↔ Transport
phenomena`, `equilibrium ↔ Steady state`) — real, not discoveries. Judge-pass alone rewards
proximity. The lone *discovery-like* hit was the one moderate pass: `equilibrium ↔ Contestable
market` (physics ↔ economics).

**The keystone**: across all 15 anchored pairs the novelty_external "known" gate flagged **0**
as known — including textbook pairs. A free FAISS-only probe (`/tmp/probe_known_oracle*.py`)
showed the co-occurrence signal is **inverted**:

| signal | KNOWN pairs (want high) | UNRELATED (want low) |
|--------|------------------------|----------------------|
| bigram-FAISS `"A B"` query (OLD) | mean **0.564** | mean **0.746** |
| direct `cos(A,B)` (NEW) | mean **0.591** | mean **0.047** |

The bigram query was really measuring *string distinctiveness* — `"homeostasis Fair Oaks Bridge"`
matches the *Fair Oaks Bridge* article at 0.875 because the rare proper noun dominates, while
`"entropy thermodynamics"` (two abstract words) matches nothing well (0.43). Direct `cos(A,B)`
separates cleanly: a `≥0.45` gate flags **6/7 known, 0/5 false-positive**. This one bug explains
*both* symptoms — Test B's rediscovery rate couldn't work, **and** trivially-known pairs sailed
past stage 3 as "novel" into a judge-pass, which is exactly why the near band looked like 50%
"discoveries" that were trivia.

**Known limitation of the fix**: cosine measures topical proximity, not citation co-occurrence,
so genuinely *cross-domain* known connections slip under the gate (`percolation theory ↔ phase
transition`, cos 0.34 — a real connection cosine underweights). Catching those needs a real
co-occurrence/citation corpus; cosine is strictly better than an inverted signal, not the final
answer. Also still un-addressed: sub-check 1 (claim-similarity) uses the same un-normalized FAISS
inner product vs a `0.88` threshold documented as 0–1 cosine — same scale bug, separate follow-up.

### 2026-06-27 — the real Test-B instrument: document co-occurrence (cosine can't be the oracle)

The cos(A,B) gate fixed the *pipeline* (un-inverted, stops trivia leaking as novel) but is **not a
Test-B oracle**: it equates "known" with "topically close," so it (a) is *circular* with the
controlled-distance band selection (both are cos(A,B)) and (b) the base-rate trap this doc warned of
— it marks non-obvious cross-domain known pairs *novel* (they sit at moderate/low cosine) and trivial
same-field pairs *known*. It literally can't see the discovery target.

`scripts/synthesis_doc_cooccurrence.py` is the instrument that can — independent of embedding distance:
**do A and B get discussed in the same wiki articles?** Retrieve each concept's top-`depth` chunks;
KNOWN if they share article titles OR one concept's distinctive term appears in the *text* of the
other's articles (the crossover analogy lives in the body, not the title). Validated on a 3-group
labeled set (n=15), v2 (text scan, depth 40):

| group | result |
|-------|--------|
| KNOWN_OBVIOUS | 4/4 caught |
| **KNOWN_CROSSDOMAIN** | **5/6 caught — all `cos-MISS` (cosine catches 0/6)** |
| UNRELATED | 5/5 kept novel (0 false-positive) |

**93% accuracy.** `simulated annealing ↔ metallurgy` (cos 0.09), `network science ↔ epidemiology`
(cos 0.07), `Kalman ↔ Bayesian` (0.30), `percolation ↔ epidemic` (0.31), `game theory ↔ evolution`
(0.37) — all caught despite cosine calling them novel. v1 (title overlap only) caught 0/6: the
crossover is in article *text*, not titles. One miss (`information theory ↔ thermodynamics`) is a
depth/term-match tune, not structural.

**Why this matters:** Test B's rediscovery rate is now measurable **for free** (no LLM — the known
signal doesn't need the coherence judge) and **non-circularly** (the oracle is a different signal from
cos-based band selection). The next experiment: anchored sampling vs a random baseline, scored by
doc-co-occurrence known-rate — the actual proof-of-concept the whole methodology was building toward.

**Hardening (n=99, `scripts/synthesis_oracle_validation.py`, depth 40):** the oracle was promoted to
`knowledge/doc_cooccurrence.py` and validated at scale on 34 known / 25 confidently-unrelated / 40
random pairs:
- **Recall 33/34 (97%)** — **24/25 (96%) on the hard cos<0.45 cross-domain subset** (the recall cosine
  cannot provide); 9/9 on easy same-field.
- **Clean FP 1/25 (4%)** — `supply and demand + mitochondria` (mitochondria articles discuss energy
  *supply*/*demand* → common-word stem collision; the inherent risk, surfaced once).
- **Scale FP 4/40 (10% upper bound)** — but 3 of 4 are *genuinely real* (`enzyme+photosynthesis`,
  `homeostasis+thermodynamics`, `jazz+violin`); true FP ≈ 3–5%.
- **1 FN**: `information theory ↔ thermodynamics` (cos 0.31) still slips depth-40 text scan.

Verdict: trustworthy at ~**97% recall / ~4% FP**. The clean FP is inherent common-word ambiguity; the
obvious fix (blocklist `supply`/`demand`) is unsafe (those are the concept's distinctive terms), and
tightening the matcher would cost the 96% hard recall — so the operating point stays here. Stem-substring
matching remains crude in principle, but at scale it holds.

### 2026-06-27 — discovery miner demo: the engine works, and at scale it exposes the matcher's limit

`scripts/synthesis_discovery_miner.py` turns the oracle into a generator: from a 40-concept diverse pool,
retrieve each concept once (cache), then surface every **low-cosine (<0.40) + doc-co-occurring** pair —
the discovery quadrant. Sorted most-non-obvious-first, it mined real connections **organically** (not
hand-fed):
- `simulated annealing ↔ metallurgy` (cos **0.09**) — the archetype, rediscovered
- `phase transition ↔ metallurgy` (0.08), `morphogenesis ↔ information theory` (0.05),
  `diffusion ↔ immune system` (0.05), `phase transition ↔ natural selection` (0.04),
  `statistical mechanics ↔ metabolism` (0.09), `renormalization group ↔ error correction` (0.03)

**But the demo also exposed the matcher's scaling limit.** It flagged **209/746 (28%)** of non-obvious
pairs — inflated. Every top hit was `text-mention` (no `shared-title`), and the false positives cluster
hard on **common-word concept names**: *error **correction**, **control** theory, **natural selection**,
**supply** and **demand**, **social network*** collide via stem-matching with everyone. The curated
hardening's 4% FP **underestimated this** — pairwise mining stresses common words far harder than 25
hand-picked negatives. So unsupervised mining needs a common-word-robust mention signal first:
rank `shared-title` above `text-mention`; require the mention to be *distinctive* (rarer/longer term, in
a top-ranked chunk, or bidirectional); down-weight common-word concepts. Until then the oracle is solid
as a **gate on a given pair** (97%/4%) but not yet as an **unsupervised miner** (28% with FP tail).

Net of the day: a vague *"is any of this real?"* became a validated instrument that mines real
non-obvious cross-domain connections — plus a precise, evidence-backed spec for the one fix between it
and unsupervised discovery.

---

## ⟲ REVERT NOTE — stage-3 co-occurrence gate (2026-06-27)

`knowledge/synthesis_filter.py:_stage_3_novelty_external` sub-check 2 was changed from a
bigram-FAISS query to direct `cos(A,B)`. **To revert**, restore the OLD block and config below.

**OLD config** (`config.yaml` / `app_config.py` — still present, just unused by the gate):
`cooccurrence_known_threshold: 0.85` → `SYNTHESIS_COOCCURRENCE_KNOWN_THRESHOLD`.
**NEW config** added: `concept_cosine_known_threshold: 0.45` → `SYNTHESIS_CONCEPT_COSINE_KNOWN_THRESHOLD`.

**OLD code** (sub-check 2), to paste back over the new cosine block to revert:
```python
        # --- Sub-check 2: Co-occurrence via FAISS ---
        bare_query = f"{concept_a} {concept_b}"
        try:
            cooccurrence_results = semantic_search_with_neighbors(bare_query, k=3)
        except Exception as e:
            logger.debug(f"FAISS co-occurrence check failed: {e}. Skipping.")
            cooccurrence_results = []

        cooccurrence_sim = _extract_faiss_similarity(cooccurrence_results)
        result.cooccurrence_similarity = cooccurrence_sim

        # Hard gate: concepts already co-occur heavily in literature
        if cooccurrence_sim > SYNTHESIS_COOCCURRENCE_KNOWN_THRESHOLD:
            return StageResult(
                stage_name="novelty_external",
                passed=False,
                reason=(
                    f"Concepts already co-occur in literature "
                    f"(cooccurrence={cooccurrence_sim:.3f} > {SYNTHESIS_COOCCURRENCE_KNOWN_THRESHOLD})"
                ),
                score=result.novelty_score_external,
                metadata={"claim_similarity": claim_sim, "cooccurrence_similarity": cooccurrence_sim},
            )
```
Evidence the OLD signal was inverted (so we don't "revert to broken" without remembering why):
KNOWN-pair mean 0.564 < UNRELATED-pair mean 0.746. Keep this note before reverting.
