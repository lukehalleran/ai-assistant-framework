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

## The knownness triage funnel (design direction — NOT yet built)

> Captured 2026-06-27 from a design discussion. This is the intended shape of the
> post-discovery-miner pipeline, written down so it survives until we decide to build it.
> Nothing here is implemented; the discovery miner currently stops at "co-occurs? yes/no".

The keystone reframe: **a high known-rate is a calibration signal, not a yield signal.** When
today's miner reports "28% of low-cosine pairs co-occur in the literature," that number proves
the *instrument* works — the oracle is finding real co-occurrence, not hallucinating links. It
says the needle isn't broken *before* you go looking at what it points at. What it does **not**
say is that those pairs are valuable. So:

**The known pairs are the control group, not the product.** RG↔error correction, the metallurgy
pairs, morphogenesis↔info theory — they exist to prove the machine works. Once proven, you
strip them — along **two distinct axes**:

1. **Strip trivial** (e.g. the metallurgy pairs): real, co-occurring, but *graph-adjacent* —
   obvious. Low value. Kill.
2. **Strip known-but-nontrivial** (e.g. RG↔error correction): real, co-occurring, *graph-distant*,
   genuinely deep — but already in the literature. **Do not throw these away — label them.**
   They are the *richest validation evidence*: proof that when the instrument finds a
   distant-but-connected pair, it's finding something worth finding. This is your **validation
   gold / control set.**

What's left after both strips is the residue you're actually hunting:
**low cosine + co-occurring + graph-distant + NOT in the reference corpus** = coherent,
non-trivial, and not-yet-known = the *candidate-worth-a-human's-time* bucket.

The funnel shape:

```
generated
  → co-occurs? (doc-cooccurrence oracle)      ── no  → kill
  → non-trivial? (graph distance)             ── no  → kill (trivial/adjacent)
  → known? (cross-corpus knownness)
        ├─ known + nontrivial  → keep as VALIDATION GOLD (control set)
        ├─ unknown + coherent  → keep as CANDIDATE (human last mile)
        └─ everything else     → kill
```

**"Known" is not binary — it's cross-corpus.** Known-in-Wikipedia-but-unknown-in-PubMed is a
different and more interesting object than known-everywhere. A pair documented in physics but
*absent from the biology corpus*, sitting at high graph distance, is the texture of a real
cross-domain-transfer candidate. So the "strip the known" step is really **"sort by *where* it's
known"** — and the genuine unknowns fall out the bottom of that sort for free. (This is the
concrete mechanism behind the README's "cross-corpus knownness" roadmap bullet.)

**The honest constraint:** the unknown-and-coherent bucket will be small and noisy. Most of what
survives every gate is still a near-miss; the genuine ones are rare enough that the blind audit
queue is doing the real work. That's fine — that *is* the "raise the prior that something's worth
a human's time" thesis. This is not a discovery machine; it's a **triage machine** that hands a
human a short list instead of an infinite one. Today's miner run proved the triage's *first
stage* (co-occurrence detection) is sound — the rest of the funnel is the build-out.

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
answer. Sub-check 1 (claim-similarity) compared a raw FAISS distance against a `0.88` "cosine" threshold
— same scale-bug class. **RESOLVED 2026-06-28** at the retrieval layer (parallel work): the wiki index is
`metric_type=L2` (confirmed by mmap on the live 41M-vector index), and `semantic_search._to_similarity()` now
maps L2 d→1−d/2 (exact cosine for the normalized stored vectors), so the 0.88 gate is a real cosine. NB good
wiki matches top out ~0.79, so the gate now rarely trips — worth checking it isn't too *permissive* (letting
near-rehashes through). See memory `project_wiki_faiss_l2_metric_mismatch`.

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

### 2026-06-28 — anchored-vs-random PoC: pair-selection beats a distance-matched baseline

The methodology's actual proof-of-concept, finally run. ONE pre-registered, falsifiable claim: *anchored
pair-selection surfaces literature-known cross-domain connections at a higher rate than random pairing AT
THE SAME semantic distance.* Scorer = the doc-co-occurrence oracle (NOT cosine — non-circular with the
cos-based band selection). Arm A = prominent anchor + FAISS-neighbourhood partner at true-cos<0.45; Arm B =
random pairs from POOL∪ANCHORS **resampled to match Arm A's per-0.05-bin cosine distribution** (the
distance-match is the whole experiment). `scripts/synthesis_poc_anchored_vs_random.py`.

**Run 1 (seed 13, either-direction oracle, n=72/arm): apparent win, pre-registered FAIL.** A 63.9% vs B
37.5%, gap +26.4%, p=0.0015 — but the gap was carried almost entirely by text-mention hits (A 41/46, B
25/27), and the robustness gate (exclude text-mention-only → shared-title only) was **underpowered** (5 vs
2, p=0.245). Verdict FAIL: advantage not *demonstrated*.

**Directionality probe (`scripts/probe_poc_directionality.py`, on the seed-13 pairs).** Hypothesis: the
text-mention gap is asymmetric stem collision (anchor's term bleeds into the neighbour's articles without the
reverse; real co-occurrence is bidirectional). Honestly mixed: the *literal* hypothesis (anchored MORE
unidirectional) was **not** supported — anchored was if anything more bidirectional (34% vs 24%, n.s.). But
the collision *mechanism* was visible: within anchored, anchor→partner leakage ran 18:9 (2:1) vs random's
balanced 9:10; and requiring bidirectionality stripped ~60% of hits as one-way collision in **both** arms
while the anchored edge **survived** (bidirectional-only known-rate A 26.4% vs B 11.1%, gap +15.3%, p=0.019).
Clean eyeball split (`tragedy of the commons ↔ A Kindness Cup` via "common" vs plausible-real `equilibrium ↔
Le Chatelier's principle`). → bidirectionality is worth requiring; the edge is not pure artifact.

**Run 2 (seed 29 — fresh sample, bidirectional oracle, n=144/arm): pre-registered PASS on every cut.** A
29.9% vs B 10.4%, gap **+19.4%, p<0.0001**, 95% CI [+10.5%, +28.4%], perfect distance-match (mean cos 0.233
vs 0.234). Crucially the cleanest, least-gameable signal — **shared-title-only** — now had the power the
seed-13 run lacked (16 vs 3) and is **significant on its own: 11.1% vs 2.1%, p=0.002.** So the advantage is
NOT a text-mention artifact: even gold-standard co-listed-in-the-same-article co-occurrence shows anchored
beating random ~5:1, on an independent seed.

**Conclusion.** Anchored pair-selection **demonstrably** finds literature-co-occurring cross-domain structure
above a distance-matched random baseline — confirmed across two seeds, robust to the collision artifact, and
significant on the cleanest signal. The original FAIL was a *power* problem in the clean signal, not a null
effect. First hard evidence the generator's pair-selection finds real non-obvious structure (not that any
specific pair is novel-to-*you* — that remains the human last mile; absolute yield is modest: anchored
~30% bidirectional / ~11% shared-title vs random ~10% / ~2%). The oracle gained an **opt-in** `bidirectional`
param (`knowledge/doc_cooccurrence.py`, default unchanged — flipping it would lower recall and needs the n=99
oracle re-validation first); 7 unit tests in `tests/unit/test_doc_cooccurrence.py`. Caveat: this is a
confirmatory run built on an exploratory probe — the strongest single fact is the *fresh-seed* shared-title
significance (16 vs 3), which is the least gameable. **Unaffected by the concurrent 2026-06-28 retrieval-metric
fix** in `semantic_search.py`: the oracle consumes the k-nearest *set* (title union + text concat), never the
FAISS score or intra-set order, and that set is byte-identical pre/post-fix (same `index.search`/nprobe; the
fix only post-processes the score) — verified empirically (12/12 sampled seed-29 pairs reproduce their saved
verdict). So the metric fix's timing relative to the runs is moot.

### 2026-06-28 — bidirectional re-validation @ n=99: a precision knob, NOT a default

`scripts/validate_oracle_bidirectional.py` re-runs the n=99 hardening set (34 known / 25 unrelated / 40
random) under BOTH oracle modes in one pass (memoized retrieval). Default reproduced the documented baseline
exactly (recall 33/34=97%, hard cos<0.45 24/25=96%, clean FP 1/25, scale FP 4/40). Bidirectional:

| metric | default (either-dir) | bidirectional |
|--------|---------------------|---------------|
| KNOWN recall (all) | 33/34 (97%) | 17/34 (50%) |
| hard (cos<0.45) | **24/25 (96%)** | **9/25 (36%)** |
| easy (cos≥0.45) | 9/9 (100%) | 8/9 (89%) |
| clean FP | 1/25 (4%) | **0/25 (0%)** |
| scale FP | 4/40 (10%) | **0/40 (0%)** |

Bidirectionality eliminates **every** false positive (perfect precision) but **craters hard cross-domain
recall 96%→36%** — it loses 15 of the 25 cross-domain known pairs, i.e. the oracle's whole reason for
existing. The lost pairs are textbook-real: Kalman filter + Bayesian inference, percolation + epidemic, power
law + phase transition, chaos theory + weather forecasting, neural network + brain. **Why:** real cross-domain
knowledge is *asymmetrically documented* — the specific article mentions the general concept (Kalman→Bayesian),
but the general article rarely names the specific one. Requiring both directions kills the specific↔general
links, which are most of the value. So **unidirectional ≠ collision**: it's EITHER collision OR
real-but-asymmetric, and bidirectionality can't tell them apart — it discards both.

Decisions: (1) **Keep either-direction the oracle default** (97%/4%) — do NOT flip; the param stays opt-in.
(2) **The anchored-vs-random PoC PASS is unaffected** — it measured *relative* A-vs-B discrimination (both arms
took the same recall hit) and its load-bearing evidence (shared-title 16 vs 3, p=0.002) is flag-independent;
bidirectional text-mention is now understood as a precision-filtered, low-recall signal. (3) The
discovery-miner overfire fix should use the *surgical* signal (rank shared-title > text-mention, require
*distinctive*/rare matched terms, down-weight common-word concept names), with bidirectionality as one positive
signal — NOT a hard gate, which would discard most real discoveries.

### 2026-06-28 — surgical miner fix: position beats rarity, but text-mention can't be fully cleaned cheaply

Goal: kill the discovery miner's 28% common-word overfire WITHOUT cratering recall the way bidirectionality
does. Two mention gates tested on the n=99 set (`scripts/validate_distinctive_mention.py`), both keeping
shared-title, vs default (recall ceiling) and bidirectional (FP floor):

| variant | recall(all) | hard | easy | cleanFP | scaleFP |
|---------|-------------|------|------|---------|---------|
| default | 97% | 96% | 100% | 4% | 10% |
| bidirectional | 50% | 36% | 89% | 0% | 0% |
| distinctive@0.25 (stem rarity / set-DF) | 76% | 72% | 89% | 4% | 5% |
| toptext@10 (top-K-chunk position) | 71% | 60% | 100% | 0% | 2% |

(1) **Rarity FAILS.** "distinctive = stem rare in the comparison set" is biased by topical clustering — a
physics-heavy set inflates `thermo`/`networ`/`entrop` DF, so real same-domain links die (it killed `entropy +
thermodynamics`). Strictly worse than default. (2) **Position is the better axis.** Requiring the cross-mention
to land in the OTHER concept's top-K chunks (its core articles, not a tangential one) is immune to topical-DF
bias and to common-word concept names. `toptext@10` zeroes clean FP at 60% hard recall — the best
precision-first frontier found.

**Meta-finding (the honest ceiling): no cheap stem-substring gate hits default-recall + zero-FP.** Every
precision gain costs real recall — genuine cross-domain connections (`percolation+epidemic`, `neural
network+brain`, `Markov chain+statistical mechanics`) are documented just like collisions are: asymmetrically,
in secondary articles, via common terms. Closing that gap needs LLM semantics or a real citation corpus, not
stem tricks.

**Deployed `--top-chunks 10` as the miner default** (`scripts/synthesis_discovery_miner.py`; `0` = legacy
full-text). Real-pool A/B (40-concept pool, 746 non-obvious pairs):
- overfire **28% (209) → 10% (76)** — a 2.8× cut, archetypes intact (`simulated annealing ↔ metallurgy`,
  `renormalization group`, `morphogenesis`, `percolation ↔ fractal` all survive).
- worst common-word concepts cut ~60–74% (supply-and-demand 13→5, control-theory 18→7, error-correction 11→4,
  social-network 19→5).
- **PARTIAL:** ~21 of the 76 survivors still involve those 4 common-word concepts — `supply`/`contro`/`social`/
  `correc` are common enough to appear in many concepts' *core* text, so position-gating reduces but can't
  eliminate them. The residual is the downstream coherence judge's job; the miner is a cheap pre-filter that
  now hands the judge a 2.8×-cleaner candidate set, not a final arbiter.

### 2026-06-28 — Stage-6 composite re-sweep on the CORRECTED pipeline (live judge)

The composite (`0.30·coh + 0.40·nov + 0.15·dist + 0.15·struct ≥ 0.65`) was tuned while the judge was dead
(coherence≈floor → ~nothing cleared 0.65) AND Stage-3 novelty was inverted (wrong population reaching it), so
any pre-fix "accepted N" was meaningless. `scripts/resweep_composite.py` runs a realistic anchored candidate
batch through the FULL corrected filter (live opus-4.8 judge), ISOLATED to a temp ChromaDB (no production
audit-queue / graph writes). n=20 anchored pairs (reused from the seed-29 PoC, so no risky `idx.search(k=1500)`):

Funnel: 3 no-connection (generator declined) → 17 articulated → 1 reject@distance, **12 reject@coherence_judge**
(WEAK/INVALID — the *fixed* judge doing real discrimination, not the dead-WEAK default that rejected everything),
4 reached composite, **2 ACCEPTED** (12% of articulated). Composite @ stage 6: [0.609, 0.638, 0.669, 0.710];
the 0.65 gate cleanly takes the top two (STRONG 0.710, MODERATE 0.669 → accept: `diffusion ↔ nonlinear
dimensionality reduction`, `oscillation ↔ envelope (waves)` — both reasonable) and drops the low MODERATEs.

Read: the corrected pipeline **produces real, sensible acceptances** where the dead-judge era produced ~none —
the stale "accepted N" is replaced by ~12% on realistic anchored volume, with the **judge (not the composite) as
the primary filter** (12/17 die there). The 0.65 gate straddles the MODERATE band sensibly. **Caveats:** n=4 at
stage 6 is small, and the composite is COMPRESSED (even a STRONG candidate tops out at 0.71), so the threshold is
consequential — a deliberate larger-n calibration sweep would confirm 0.65 is the right cut before quoting
precision/yield in production.

### 2026-06-28 — composite calibration (n=60, live judge): the blend is mostly dead weight

`scripts/calibrate_composite.py` swept 60 anchored candidates and captured the full per-component breakdown at
Stage 6 (isolated temp chroma). Funnel: 11 no-connection, 3 text-sanity, 2 distance, **32 coherence-judge**
(the judge is the primary filter), 12 reached Stage 6 → **4 accepted (~8%)**. Component ranges at Stage 6 (n=12):

| component | weight | range | verdict |
|-----------|:------:|-------|---------|
| coherence | 0.30 | **0.660 only** | flat at the decision point |
| novelty | 0.40 | 0.718–0.802 | compressed |
| distance | 0.15 | 0.024–0.724 | the only well-spread live signal |
| structural | 0.15 | **0.500 only** | hard constant |

Confirmed in code: (1) `_stage_1_domain_crossing` scores `min(len(source_domains)/4, 1.0)`, and every
anchored/cross-store candidate has exactly 2 domains → **structural = 0.5 forever** (only 3–4-domain graph-walk
candidates would move it). (2) The judge emits LEVELS mapped to fixed enum values (`MODERATE.value=0.66`, STRONG
higher); only MODERATE/STRONG reach Stage 6 and STRONG is rare (0 here, 1 in the re-sweep), so **coherence is a
constant 0.66 for the MODERATE bulk**. (3) **Net: the accept decision is distance-DOMINATED** — coherence+
structural (45% of weight) are constants and novelty (40%) spans only 0.08, so the 0.15-weight distance term
(spanning 0.70) contributes the most decision-spread (0.15·0.70=0.105 vs novelty 0.40·0.08=0.034). The intended
4-signal blend collapses to ≈ "MODERATE-coherence pair that's sufficiently far apart" — by accident, not design.

Threshold sensitivity (Stage-6 reachers): ≥0.60→83%, **≥0.65→33%**, ≥0.70→0% (no MODERATE reaches 0.70; that's
STRONG-only territory). So 0.65 = "STRONG + top-third-by-distance MODERATE"; 0.70 = "STRONG only."

**Caveat:** `internal_novelty` showed constant 1.0 — an ARTIFACT of the isolated empty temp store (no priors to
dedup against), NOT a production property; the structural + coherence findings hold regardless.

**Implications (filter-design changes, not auto-applied — need a human-graded signal to set):** the
0.30/0.40/0.15/0.15 weighting is mostly fiction for 2-domain candidates. Either give structural a real signal
(graph distance / isomorphism depth) or drop it + redistribute; get a finer-grained coherence score (not 4
levels) so coherence discriminates among MODERATE; or consciously treat coherence as a GATE and reweight the
composite to refine on novelty+distance. The 0.65 cut is a real STRONG-only-vs-STRONG+MODERATE choice that needs
graded examples — which loops back to `grading_plan.md`.

### 2026-06-28 — re-scoring the 28 human grades: the system has TWO jobs, and they diverge

`scripts/rescore_graded.py` re-ran the 28 graded queue candidates (slider-only, no two-layer binaries;
distribution {5:12, 4:15, 3:1} — a positive-only set, all PERSONAL cross-domain, e.g. `parents ↔ Family
reunion`, `ramen ↔ Comfort`, `at the gym ↔ Pattern theory`) through the CORRECTED pipeline (fixed opus judge),
isolated to a temp store. Result is stark:

- **Recall on the owner's grade-4-5 insights: 1/27 (4%).** Mean composite OLD **0.725 → NEW 0.050**. Coherence
  shift: **19 STRONG→WEAK**, 6 STRONG→None (died pre-judge), 2 STRONG→MODERATE. The lone accept
  (`health care analytics ↔ process design`, the most *structural/professional* of the lot) is the exception.
- Where the 4-5s die: **19 @ coherence_judge** (the fixed opus judge rates them WEAK), 5 @ semantic_distance,
  1 @ novelty_external, 1 @ composite.

**Interpretation (the real finding): the system conflates two different jobs.** (1) *Scientific structural
discovery* — what all the validation work + the judge's de-jargon/variable-swap rubric measure. (2) *Personal
reflective insight* — what these candidates ARE and what the owner graded ("does this make me think about my
life differently"). On personal-life fragments the two criteria diverge hard: the science-calibrated judge
correctly says "`in bed ↔ Ecosystem` is not a structural isomorphism" (WEAK — defensible; grading_plan.md
itself warned the old pipeline's STRONG ratings were apophenia), while the owner valued personal resonance, an
orthogonal axis. Neither is "wrong" — they measure different things. (Caveat: ~6 of the rejects are mechanical
personal-vs-wiki artifacts — `Bartlett, IL ↔ Suburb` is trivially close, `project ↔ Project` is identical —
not quality judgments; the wiki-tuned distance/novelty stages don't fit personal concepts.)

**Consequence:** the 28 grades are personal-resonance labels; the pipeline (and its judge) are science-
calibrated. So even the old classifier plan was mismatched — it would have trained on one axis to gate the
other. **Fork to decide before more synthesis work:** is this a scientific-discovery machine (new judge right,
wiki↔wiki track applies, personal candidates out of scope) or a personal-reflective companion (needs a
different judge rubric + personal-aware distance/novelty stages)? [RESOLVED below — it's NOT two judges.]

### 2026-06-28 — judge domain-generality test: one detector, TWO axes of "non-obvious"

`scripts/test_judge_domain_generality.py` ran 3 structured-personal + 3 surface-personal + 2 structured-
scientific claims through ONLY the judge (isolating the truth-structure detector from wiki plumbing).
Result: scientific 2/2 pass, surface-personal 0/3 (correctly WEAK), structured-personal **1/3**. The
headline looks mixed, but the judge's *verbatim reasoning* resolves the whole fork:

- The judge is **domain-general** — same standard applied to personal and scientific; it de-jargoned clean
  structure from ALL 3 personal insights and passed one (`procrastination ↔ doctor-avoidance`) at the SAME
  level (MODERATE) as the scientific archetypes. It did NOT penalize "personal." So: **one detector.**
- The dad/financial insight got WEAK for a reason that is neither "no structure" nor "unverifiable": *"not a
  cross-domain mapping; it's a SINGLE domain (operant conditioning) applied to two examples... 'controlling a
  needed resource buys compliance' is true of every dependence relationship — landlord/tenant, addict/dealer.
  A structure that fits everything discriminates nothing."* The passing one passed because one domain's
  specific structure (non-monotonic fear-drive curves) transferred to a NON-OBVIOUS prediction on the other.

**The resolution:** "non-obvious" has TWO axes. (1) **Structural novelty** — no one has bridged these domains
(what the discovery judge rewards). (2) **Personal non-obviousness** — *I* never recognized this *known*
mechanism in *my own* life (why the dad insight landed: operant conditioning is structurally obvious but
personally revelatory). The dad insight scores HIGH on (2), LOW on (1); the judge correctly calls it WEAK *for
discovery*. So it's **not two judges** — it's **one structural detector feeding two objectives**: discovery =
maximize structural novelty (reject known general mechanisms); reflection = surface KNOWN mechanisms newly
applied to the owner's specific life (exactly what the discovery judge rejects, and exactly what a psychologist
is expert at validating). The 27 grades collapsed because the two objectives score *opposite* — not because
truth has two shapes, but because "non-obvious" does. Reflection objective ≈ personal-specificity ×
mechanism-realness × not-yet-recognized — the mirror of the discovery composite; the owner's grades + the
psychologist are the right calibration data FOR IT.

---

### 2026-06-29 — provenance of the 28 + reflection-mode rubric flip test: the bottleneck is the GENERATOR

Two checks this pass. Design captured in `docs/REFLECTION_SYNTHESIS_UNIFIED_SCORER.md`.

**(1) Provenance (`/tmp/inspect_graded.py`, read-only).** The `synthesis_results` queue has 98 rows: **28
graded — every one `status=accepted`** (old composite 0.63–0.79); **69 rejected** (≤0.63, mostly 0.0). The raw
garbage (`fast↔low`, `poor↔google`, `tests↔Excel`) is in the rejected pile and *never reached grading*. So the
28 are **survivors of the old accept gate, then therapist-graded 3–5** — a curated POSITIVE set, no graded
negatives. Cross-tab of grade × structural-coherence: the judge floors **20/22** at WEAK regardless of grade,
and its only 2 passes are the 2 LEAST-personal items (`Lakewood↔suburbanization`, `healthcare↔process-design`).
Looked like hard proof that structural-truth ⊥ reflection-value.

**(2) Flip test (`scripts/test_reflection_rubric_flip.py`).** Re-ran the most-personal therapist-4/5 claims
through the production rubric vs a single-variable **reflection-mode** rubric (de-jargon test IDENTICAL; cross-
field variable-swap → recurrence-across-own-contexts; adversarial system prompt neutralized), with surface
controls + a positive anchor. **Result = CONFOUND:** both surface controls (`ramen↔comfort`, `tomorrow↔tardiness`)
flipped WEAK→MODERATE too, so the 2 target flips are untrustworthy. Diagnosis from justifications: dropping the
adversarial "reject these" stance turned the judge into a *charitable mechanism-generator* — it invented a real
"fixed template + swappable components" mechanism for ramen that the claim never contained. **The adversarial
stance is load-bearing rigor, not removable.**

**The two findings that matter more than a clean flip:**
- **The 28 are the WRONG CORPUS.** The judge's own reflection reasoning flagged `in bed↔ecosystem` as *"a literal
  biological description of a bed, not a recurring mechanism in the person's life."* They are machine-confabulated
  **discovery-style personal→wiki bridges**, NOT genuine reflection insights (own-context↔own-context).
- **Genuine reflection passes the EXISTING judge.** The anchor `procrastination↔doctor` (well-formed two-life-
  domain reflection) scored MODERATE under BOTH rubrics.

**Walk-back:** the cross-tab "judge floors 20/22" is **confounded** — it measures machine-bridge quality vs
theme-resonance, not reflection-truth vs structural-truth. The earlier "the structural judge can't gate
reflection" claim is **withdrawn**. The truth detector and self-R are NOT the bottleneck. **The GENERATOR is** —
reflection needs anchor-shaped candidates (two contexts in the owner's OWN life sharing one mechanism), produced
deliberately. Next: a tiny reflection generator → ~6 anchor-shaped candidates → existing judge. The only residual
rubric question is *same-relationship* reflections (`dad-now↔dad-as-child` died WEAK as "one relationship, not a
mapping"). self-R novelty is downstream of having well-formed candidates.

**(3) Generator test (`scripts/test_reflection_generator_judge.py`, same day).** Fed a pre-registered,
therapeutic-reality-labeled probe set through the EXISTING judge (anchor + 2 cross-domain-real + 2 same-
relationship + 2 thin controls). Result: thin controls both correctly WEAK (no leakage); the dad-style same-
relationship case (`advisor-deference↔dad-as-teen`) **PASSED MODERATE** (real tunable mechanism across two
*different* authorities) — **the residual is NOT a wall.** The judge's boundary is exact: it passes a *tunable
mechanism transferring across genuinely-distinct contexts* and rejects (i) the same act in cosmetically-
different media (`advisor-email↔mail` — "not genuinely cross-domain") and (ii) the same observation without a
transferable parameter (`relief↔relief` — "two instances of one phenomenon," the old dad/dad failure mode).
**RESOLUTION:** the truth term needs NO change and neither does the rubric — the existing structural judge IS a
correct reflection gate. This **overturns the cross-tab's "swap the truth term"** reading (over-correction from
the confounded data); the handoff's original **"one detector, swap only the novelty corpus R" is vindicated.**
The whole remaining problem is GENERATION: (1) pair across genuinely-distinct contexts/figures (never one figure
with itself); (2) articulate a tunable mechanism that transfers. Then existing-judge × self-R-novelty over a
purpose-built reflection generator. Note: two of my own pre-registered labels were *looser than the judge* — it
was more precise about "distinct context" than I was, a strong endorsement of the detector.

### 2026-06-29 #3 — end-to-end prototype on REAL data: judge confirmed; bottleneck = generator yield + craft

> **[CONCLUSION SUPERSEDED by #4 below.]** "the judge is a correct reflection gate" is
> overturned by reading the judge's own justifications (run #4). The observations here
> (0/11; stability; exact-vs-paraphrase) stand; the conclusion does not.

First end-to-end reflection run on the owner's real memory (`scripts/reflection_generator_prototype.py`: harvest
production chroma read-only → NEW generator LLM call → EXISTING judge), with two verification probes
(`probe_judge_reflection_stability.py`, `probe_exact_proto_rejudge.py`). Doc: REFLECTION_SYNTHESIS_UNIFIED_SCORER.md §7c.
Four generator iterations (monotone → structured → cross-system → few-shot the passing exemplars) → **11
candidates from the real journal, ALL WEAK.** Naively 0/11 reads as "the judge floors reflection" (§7b wrong,
two jobs anti-correlated) — **FALSE**, killed by two checks: (a) **stability probe** — the judge is
deterministic, the two hand-authored MODERATEs *and a cleaned rewrite of a generated claim* each scored **4/4
MODERATE**; (b) **exact re-judge** — the generator's **byte-exact** output scored **0/4 WEAK** while my **cleaned
paraphrase** of the same Oliver↔dad insight scored **4/4 MODERATE** (same pair, same mechanism; only the wording
differed). **Lesson: a raw WEAK-rate conflates bad-pair with bad-wording, and one WEAK draw is not a verdict —
check stability + exact text first.** Confirms §7b on real data: the judge is a stable, discriminating, correct
reflection gate. The bottleneck is the GENERATOR, split in two — **yield** (most pairs are genuinely same-system,
e.g. kavarin↔midterm = one energy-overdraw system, correctly floored; good cross-system pairs rare but nonzero)
and **craft** (a good pair still floors if the claim **collapses both systems into one shared driver** —
WEAK "tolerance tracks how tight my money is" vs MODERATE "concession rises as my buffer of the thing **they**
control falls"; the fix is to state the mechanism as a **per-system relation**, never a global driver). Honesty
caveat: the passing rewrite was human-authored — generator found the pair+mechanism, a human supplied the precise
wording. Next: a **self-rewrite stage** (generator proposes → rewrites for two-system precision → judge); if its
own rewrite clears MODERATE the loop closes unaided, then wire self-R novelty.

### 2026-06-29 #4 — the judge's OWN justifications: it is NOT a correct reflection gate (overturns #2/#3)

Scripts: `test_reflection_self_rewrite.py`, `probe_concept_vs_claim.py`, `probe_pg_justification.py`. Doc §7d.
Chased #3's "craft is the lever" to ground and overturned the top-line. (1) **Self-rewrite didn't close the loop:**
the generator's *excellent* explicitly-two-system rewrite of Oliver↔dad scored **0/3 WEAK** (control kavarin↔midterm
also WEAK). (2) **Concepts aren't the lever:** 2×2 — my paraphrase **P** passed with clean AND messy concept_a/b
(3/3, 3/3); generator rewrite **G** failed with both (0/3, 0/3). The claim wording is the whole lever, but its
DIRECTION is opposite to #3's theory: terse-UNIFIED P passes, elaborated-explicitly-per-system G fails. (3)
**The judge's justifications (`probe_pg_justification.py`) are decisive:** on BOTH P and G the de-jargon correctly
*extracts the real mechanism*, then the verdict applies a **cross-system COUPLING** standard verbatim — *"a true
isomorphism would require some structural feature of the work system to predict a non-obvious feature of the
family system… here the systems track their OWN scarcity independently; there is no coupling… two parallel runs
of a truism, not a mapping."* That is the DISCOVERY criterion. **Reflection is parallel-independent by nature**
(advisor↔dad: same disposition in two *uncoupled* contexts) so it structurally fails. It passes only when terse
phrasing **hides** the independence (P ~90%, and P is NOT deterministic — it dips to WEAK); stated honestly (G
spells out the independence) it is reliably rejected. **The judge rewards obfuscation, penalizes honest
reflection — not a correct gate.** Why #2/#3 missed it: terse P squeaks past the judge's blind spot, so
hand-authored tests read as passes. **Fix = unbundle the judge:** KEEP de-jargon *extraction* as the reflection
truth gate (real mechanism operating in both? — thin controls fail, advisor↔dad passes); DROP the
coupling/isomorphism verdict (the discovery standard, anti-correlated with reflection value); gate novelty with
self-R. Re-grounds §3 + the 2026-06-28 "two axes" with the judge's verbatim reasoning. **Next build:** a
reflection-specific truth gate (reuse extraction, replace the verdict, no coupling requirement); validate it
rejects the 2 thin controls AND accepts genuine reflection (advisor↔dad, P and G). Confidence: this call has
flipped twice today; anchored now in the judge's verbatim coupling demand + the P/G obfuscation asymmetry, but
the validation above is the gate before trusting it. [#4's "validate it rejects the thin controls" prediction was
itself wrong — see #5.]

### 2026-06-29 #5 — unbundled gate built: truth = lenient FLOOR; discrimination is personal-domain-distance × self-R

Script: `scripts/reflection_truth_gate_prototype.py`. Doc §7e. Built the §7d gate (de-jargon extraction + a
verdict that drops coupling) and ran the §7d validation. **Two prompt designs, both leak the same way:** v1
("real mechanism in both?") and v2 (+ counterfactual/tunable-factor rigor) BOTH pass genuine reflection 3/3
(incl. the honest **G** the coupling judge rejected) AND pass **both surface controls 3/3** — the LLM charitably
manufactures a mechanism *and* a counterfactual for dish↔rewatch / noise↔fan. **Two motivated designs leaking
identically = the result:** there is **no claim-intrinsic property** separating genuine reflection from a thin
theme (both are coherent mechanism-claims w/ counterfactuals). The controls are thin because of **personal-domain
distance** — dish&rewatch both *leisure-comfort*, noise&fan both *ambient-sound* = SAME life-region; advisor&dad =
career vs family = DIFFERENT. **Architecture (= truth × distance-from-R, R = the person's own life-graph):** truth
gate = a **lenient coherence/mechanism FLOOR** (passes any coherent claim incl. thin; its *positive* half is solid
— G passes across both designs, the coupling judge never could; making it intrinsically reject thin is
INFEASIBLE, wrong job). Discrimination = **personal-domain-distance (career↔family far/valuable; leisure↔leisure
near/thin) × self-novelty**; controls die on same-domain distance, not truth. **Flip-test confound vindicated:**
can't validate the lenient gate alone — only `truth × self-R` vs human grades; self-R/personal-distance is
LOAD-BEARING, not polish. **NEXT BUILD (real core):** personal-domain-distance / self-R = pair-selection across
*distant regions of the personal graph* (both the novelty gate AND the generator's pair-selector) — the validated
anchored-vs-random machinery run over the personal graph instead of wiki ([[project_synthesis_anchored_vs_random_validated]]).

### 2026-06-30 #6 — Phase 2 built: random pairing ~0-yield; ANCHORED selection works (27% vs 0%); distance = same-domain FILTER + SWEET-SPOT, not a monotone ranker

Scripts: `scripts/reflection_validation_harness.py` (random/stratified baseline + 4-check analyzer incl. the
orthogonality test) + `scripts/reflection_anchored_generator.py` (the #5 "real core": anchored pair-selection over
the personal graph, R=self). Substrate = the LOCKED entity-pooled UMAP+fact-text mcs=5 clustering; distance =
same-cluster gate + entity-to-entity BGE cosine. Pairing pool = LIFE-domain, non-junk-cluster, noun-like entities
(spaCy POS+NER filter drops scrap members; label denylist drops whole mood/chat/schedule/status clusters —
`mention_count` is DEAD, median 1). Doc §7f. **(1) The random harness (46 pairs) overturns the generative
assumption "distance → value":** decline-rate climbs monotonically with distance (same-domain 20% / cross-near 90%
/ cross-mid 80% / cross-far 100%); the only claims are same-domain (true-but-THIN, 4 truth-pass) + the hand-picked
gold. Declined cross pairs (`linear algebra↔Benadryl`, `kitchen↔diet coke`) genuinely share no mechanism — the
generator declines CORRECTLY (it bridges from the same input whenever a real bridge exists). **Distance is a
NECESSARY FILTER (kills same-domain thinness), NOT a SUFFICIENT generator/ranker; random distant pairs are 90-100%
non-reflections, so the orthogonality test is unanswerable on random pairs — that emptiness IS the result.** **(2)
The anchored generator (bridge-first):** mine recurring SELF-mechanisms spanning ≥2 domains → pick one anchor
entity per domain → reuse the harness's grounded generator + truth/dual-grounding gate → cross-domain filter.
**27% truth-pass (3/11) vs random 0/30 cross.** Anchoring lifts cross-domain yield ~0→27% — the reflection
analogue of [[project_synthesis_anchored_vs_random_validated]]. **(3) Distance pinned to an INVERTED-U:** near-
anchored (d 0.09-0.11) thin/INVALID even when anchored; moderate (d **0.18-0.25**) = the passes (`kavarin↔beer`,
`chicago↔iceland`, `conceptual↔linear-algebra` — the gold band); extreme (d 0.44-0.48) declines/ungrounded. Value
= `distance ∈ sweet-spot band`, NOT max distance. Open: dual-grounding slightly strict on terse facts (rejected
genuine gold `dad↔kavarin`); project-entity leak via mixed "Daemon & friends" cluster; `kavarin↔beer` exposes
personal-domain vs semantic-system tension (a hand-label call). 16 real harness claims await the owner's two-column
hand-label; next = self-novelty (§8, `SurfacingHistory`) on the anchored sweet-spot + firm the 27% (n=11, 1 seed).

### 2026-06-30 (DISCOVERY arm) — discovery GENERATION fixed + shipped: the lever is concept PROMINENCE, not anchoring/graph-walks

The discovery (wiki↔wiki "scientific insight") arm's generation was effectively dead: live generators produced
~10-15 candidates/session but accept≈0 (recent runs 0/10, 1/12, 0/12, 0/15). **Audit diagnosis** (querying
`synthesis_results`): of 139 rows, 90 rejected / 78 rated **Coherence WEAK**, de-jargon snippets = thin pairs
("two things weakly linked", "both terms can serve as one endpoint"); accepted+graded = **27/28 grade 4-5**. So
the filter is HIGH-PRECISION and starved — a **generation-quality** problem, the discovery twin of the reflection
random≈0-yield result. **Validation** (`scripts/validate_anchored_generator.py`, the missing link: the PoC
validated against the doc-co-occurrence oracle, this runs candidates through the REAL coherence judge). At n=48/arm
the hypothesis that FAISS-neighbour ANCHORING wins was **overturned**: anchored 31% MOD+STRONG / 5 accepted vs
distance-matched **random-from-curated-prominent-pool 46% / 8 accepted (17%)**. Why: a prominent anchor's FAISS
neighbourhood in the band is full of OBSCURE titles (`resonance↔"Don't Touch My Hair"` (a song), `diffusion↔Laurent
Saloff-Coste` (a person)) → the judge can't map them → WEAK. **The lever is concept PROMINENCE** — pairing two
RICH, well-developed concepts in the non-obvious band (cos 0.2-0.45) is what the judge recognizes as real
isomorphism (`information theory↔statistical mechanics`, `Markov chain↔queueing theory`, `epidemic↔equilibrium`,
`diffusion↔thermodynamics`). The retired personal→wiki tiers failed for the inverse reason: they paired
LOW-prominence personal entities. **Shipped** (owner approved "replace"): `knowledge/synthesis_pooled_generator.py`
(PooledConceptSynthesisGenerator) + `knowledge/synthesis_concept_pool.py` (48 curated concepts, growable) wired as
the SOLE discovery generator in `shutdown_processor._run_synthesis_dreaming`; Tiers 0/1/2 (retrieval / graph-walk /
cross-store) retired behind `enabled:false` (one flag-flip back). Config: `synthesis_pooled` section
(`SYNTHESIS_POOLED_*`). Production smoke 6→4 MOD+STRONG / 2 accepted. `mention_count` is dead (median 1) so any future personal-anchoring
can't use it.

> **STATISTICAL CAVEAT (2026-07-07 audit):** "overturned" overstates the evidence. The artifact's own stats block
> (`anchored_gen_validation_20260630_121819.json`) gives anchored 31% vs random 46% at **z=−1.47, p=0.142, 95% CI
> [−0.34, +0.05]** — a null result whose point estimate favors random, not a demonstrated reversal; the verdict line
> in `validate_anchored_generator.py` gates on a bare `>` with no significance test. The correct reading: *anchoring
> showed no detectable advantage at n=48/arm, and the random-from-curated-prominent-pool arm was at least as good* —
> the pooled generator's justification is its own accept rates (~17% accept / ~46% MOD+STRONG vs ~0 for the retired
> tiers), not this comparison. The qualitative mechanism (obscure FAISS-neighbourhood titles → judge can't map →
> WEAK) still stands as the observed failure mode of anchoring.

**2nd lever FIXED same day — Stage-6 composite recalibrated** (`scripts/calibrate_composite.py` rebuilt for the
pooled generator; n=22 at stage 6). Confirmed on pooled candidates: **structural = constant 0.5 (range 0.000, dead
weight)**; **distance has a huge noisy range (0.31-0.98)** that corrupted ranking (low-novelty far pair `oscillation↔
chaos` outscored novel near pairs); **novelty is the ONLY discriminating term** (range 0.22). Under the old weights
MODERATE clustered 0.627-0.696 on the 0.65 knife-edge (accept distance-driven, the wrong 7/16). **Fix:** weights
`coherence 0.35 / novelty 0.60 / distance 0.05 / structural 0.0` (sum 1), threshold `0.70`. structural→0 (dead);
distance→0.05 (already GATED in-band at stage 2 — a tiebreaker, not a driver); novelty is now the MODERATE
discriminator; coherence is the STRONG bonus (gated MODERATE+ at stage 5). **Result (recomputed on the exact
candidates):** 16/22 accept = STRONG 6/6 + MODERATE 10/16, and accept is **NOVELTY-ranked** (accepted-MOD novelty
0.768 vs rejected 0.708, +0.06). The MODERATE now rejected → audit queue are the **already-documented** connections
(`evolution↔information theory`, `epidemic↔ecology`, `chaos↔equilibrium`) — correctly deprioritized for *discovery*;
the accepted MODERATE are the non-obvious ones (`Markov chain↔control theory`, `diffusion↔evolution`,
`simulated annealing↔neural network`). Composite is no longer "MODERATE that's far enough apart" but "STRONG, or
MODERATE that's genuinely novel." 207 synthesis/config tests pass. [[project_synthesis_composite_dead_weight]] now
resolved. Provisional threshold pending a human-graded batch on the pooled generator's output.

**3rd finding — WORLD-NOVELTY ORACLE IS BLIND (validated 2026-06-30; halted here to finish reflection).** A graded
batch was loaded into the GUI audit queue (`scripts/load_discovery_batch_to_gui.py`), but the top accepts were all
famous textbook identities. `scripts/probe_top_discovery_novelty.py` (per-term) + `scripts/validate_novelty_blindspot.py`
(real arXiv check) confirmed: stage-3's only "known" signals are EMBEDDING-PROXIMITY (`cos(A,B)>0.45` — and
`cooccurrence_similarity` is literally cos(A,B), `synthesis_filter.py` `_stage_3_novelty_external` sub-check 2, NOT doc co-occurrence — plus
verbatim-claim FAISS `>0.88`). A documented cross-domain identity is *paraphrased* + *embedding-distant* (e.g.
`neural network↔control theory` cos=0.229) → invisible to both gates. **arXiv documented-ness check: 11/11 sampled
candidates across the full composite range = DOCUMENTED → oracle false-novelty 100%** (the named results:
data-rate theorem, backprop=Pontryagin adjoint, detailed balance, H∞=zero-sum game, percolation=RG fixed point,
Gibbs/Shannon, GAN=minimax, Ising opinion dynamics, …). **Deeper:** pairing PROMINENT concepts that share real
structure surfaces THE CANON (prominent+real ⟹ already discovered) — "prominence is the lever" was a coherence win,
not a novelty win; the pooled generator's genuine-novel yield ≈ 0 even with a perfect oracle. **Open fork:** (a)
promote the arXiv→LLM check into a claim-level stage-3 oracle (DOCUMENTED → rejected_known) — it near-zeros current
output and thereby measures the generator's true yield; vs (b) rethink the generator's concept SOURCE toward the
novel frontier. Validation done; build paused to complete the reflection arm. See [[project_synthesis_discovery_generation]].

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
