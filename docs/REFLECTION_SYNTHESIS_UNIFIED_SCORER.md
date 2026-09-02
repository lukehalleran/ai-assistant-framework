# Reflection Synthesis — Unified Scorer (design note)

*Status: design + one open experiment. 2026-06-29.*
*Companion to `docs/SYNTHESIS_VALIDATION.md` (the results log). This note is the
**design**; that note is the **evidence**.*

---

## 1. The problem the 28 grades exposed

Re-scoring the 28 human-graded synthesis candidates through the *corrected*
pipeline (`scripts/rescore_graded.py`) accepted **1 of 27** grade-4/5 insights.
That looked like a regression. It is not a bug — it is a **category mismatch**,
and unpacking it produced the design below.

Provenance of the 28 (confirmed read-only against the production
`synthesis_results` queue, `/tmp/inspect_graded.py`):

- 98 total rows: **28 graded — every one `status=accepted`** (old composite
  0.63–0.79); **69 rejected** (old composite ≤0.63, mostly 0.0); 1 ungraded accept.
- The raw garbage (`fast↔low`, `poor↔google`, `tests↔Excel`) is in the **rejected**
  pile and **never reached grading**. The 28 are *survivors of the old accept gate,
  then therapist-graded 3–5.* They are a curated **positive** set — not raw output,
  and not a balanced label set.

Two consequences carried through everything below:

- **"Survived the old pipeline" is a weak certificate** — that pipeline is the one
  we showed is distance-driven and miscalibrated (`SYNTHESIS_VALIDATION.md`,
  composite-dead-weight). The **therapist grade is the signal**, not the survival.
- **No graded negatives.** The grade range is compressed to 3–5 *because the grader
  only ever saw pre-filtered accepts.* So the 28 give a **recall** signal only.

---

## 2. Core abstraction: `truth × distance-from-R`

Discovery and reflection are the **same scoring shape** with a **swapped reference
corpus R**:

```
score      = truth(candidate) × novelty_distance(candidate, R)
keep   iff   truth ≥ threshold  AND  novel_relative_to_R
```

- **Discovery:** R = world (wiki / co-occurrence oracle). Reject if the connection
  is *already known to the world*.
- **Reflection:** R = self-model (mechanisms the user has *explicitly recognized*).
  Reject if the user has *already named this mechanism in their own life*.

`dad/finances` is the canonical case: **low** novelty vs **world**-R (operant
conditioning is textbook), **high** novelty vs **self**-R (the user never named it
in their relationship with their father). Same operation, opposite verdict, only
because R changed.

---

## 3. The correction: reflection swaps **two** things, not one

The original handoff said "one truth detector, just swap the novelty corpus R."
**That overstates it.** Cross-tabulating the 28 therapist grades against the
*structural* coherence verdict (`SYNTHESIS_VALIDATION.md`, 2026-06-29 entry) shows
the structural judge and the therapist are measuring **different things**:

| therapist grade | judge MODERATE | judge WEAK |
|---|---|---|
| 5 | 2 | 7 |
| 4 | 0 | 12 |
| 3 | 0 | 1 |

The judge floors **20 of 22** (that reached it) at WEAK regardless of grade, and
its **only two passes are the two least-personal, most discovery-flavored** items
(`lakewood↔suburbanization`, `healthcare-analytics↔process-design`). The
viscerally personal ones the therapist valued (`in bed↔ecosystem`,
`fear of not graduating↔economics`, `parents↔family reunion`) are **all WEAK**.

Reconciliation with the earlier domain-generality test (which showed the judge is
**not** anti-personal — it passed `procrastination↔doctor-avoidance` at MODERATE):
the judge is anti-**thin**, not anti-personal. Most of the 28 are **structurally
thin but therapeutically real** — the therapist graded *personal resonance /
mechanism-in-my-life*, not *relational isomorphism*. On real graded data the two
axes are **~orthogonal**.

So the unification is **half right**:

- **What unifies:** the *abstraction* `truth × distance-from-R`.
- **What does NOT unify:** the **truth detector itself**.
  - Discovery truth = **structural isomorphism** (the current Gentner judge).
  - Reflection truth = **mechanism-is-real + personally-applicable** — the
    **psychologist is the truth oracle**, not only the novelty oracle.

For the reflection arm you swap **both** the novelty corpus (world→self) **and** the
truth term (structural-judge → therapist-grounded).

> **[SUPERSEDED by §7b]** The generator test overturned this. The existing structural
> judge **is** a correct reflection truth gate (it passes well-formed reflection,
> including the dad-style same-relationship case, and rejects thin controls). Only the
> **novelty corpus** swaps; the truth term does **not**. "Swap two things" was an
> over-correction from the confounded cross-tab. See §7b.

### 3a. The one open question (the experiment in §7)

There is a fork inside "reflection truth ≠ structural truth" we cannot settle from
the cross-tab alone:

- **(a) genuinely different axis** — the judge is *right* that these are thin; the
  therapist grades a different thing (usefulness). → reflection needs its own truth
  term, full stop.
- **(b) fixable rubric bias** — the judge's *current* rubric demands isomorphism
  between two **distinct academic fields** and "would survive peer review in a
  comparative methods paper." A mechanism recurring across the user's **own
  life-contexts** fails that bar *by construction*, even when the mechanism is real.
  → the *same* detector with a **reflection-mode rubric** might recover them.
- **(c) noise** — n=22, mixed set (some of the 28 are factual associations like
  `1pm↔daylight-saving`, not reflection at all).

Distinguishing (a) from (b) is the cheap experiment in §7. **It has now run — see §7
Results — and resolved as (c): wrong corpus.** Genuine reflection structure passes the
existing judge (the anchor); the 28 are machine-confabulated personal→wiki bridges, not
reflection insights. So §3's "the structural judge can't gate reflection" is **walked
back below**, and the bottleneck moves upstream to the *generator*. Read §7 before
acting on §3/§5/§8.

---

## 4. Terminology: keep the two verdicts separate

Stop using **WEAK** to mean "low novelty." WEAK is a **coherence** label only.
Encode the axes independently:

```
coherence_verdict : INVALID | WEAK | MODERATE | STRONG     (is the mechanism real?)
novelty_verdict   : already_in_world_R | already_in_self_R | novel_relative_to_R
```

A candidate can be `coherence=MODERATE/STRONG` while `already_in_world_R` AND
`novel_relative_to_self_R`. Never collapse novelty into the coherence label again.

---

## 5. Self-R: a recognized-mechanism registry (NOT raw-text similarity)

The sharp constraint. Self-R is a **narrow registry of mechanisms the user has
explicitly recognized**, *not* semantic similarity against journal/memory text.
The user has discussed dad/money/avoidance/control many times **without naming the
mechanism**; raw-text similarity would mark the best reflection insights as
`already_in_self_R` and reject exactly what motivates the branch.

```
correct:  self_novelty = 1 − max_sim(candidate.mechanism_label, known_mechanisms)
WRONG:    self_novelty = 1 − sim(candidate.full_claim, all_journal_text)
```

The question is **not** "has the user discussed related content?" It is **"has the
user explicitly recognized this mechanism as a pattern in their life?"**

Self-R v0 — intentionally dumb and explicit (YAML/JSON):

```yaml
known_self_mechanisms:
  - mechanism_label: "resource control creates deference in family money conversations"
    aliases: ["financial dependence", "money control", "dad finances deference"]
    evidence_source: "therapist_confirmed | user_confirmed | repeated_journal_pattern | assistant_named"
    notes: "where this was already recognized"
```

**Process discipline — author registry BEFORE peeking.** Write
`known_self_mechanisms` from journal/therapy history *before* looking at how it
ranks the 28. Authoring it with the 28 in mind tunes it to produce the desired
ranking and invalidates the test (circularity).

---

## 6. Delivery layer does NOT unify (keep separate)

Same candidate engine, **different delivery policy**:

- **Reflection** = therapeutic-witnessing register: crisis-aware tone +
  human-in-the-loop. Daemon already has the tone machinery for this.
- **Discovery** = analytical / research register.

Do not collapse delivery into the scorer.

---

## 7. Open experiment — reflection-mode rubric flip test

`scripts/test_reflection_rubric_flip.py`. Settles the §3a (a)-vs-(b) fork.

Single-variable change from the production judge (`synthesis_filter._stage_5`):
**keep the de-jargon test identical**; replace "isomorphism between two distinct
academic fields + variable-swap across two systems + peer-review bar" with
"**a real, nameable mechanism that recurs across the person's own life-contexts**"
(recurrence test instead of cross-field variable-swap). Neutralize the
reject-by-default system prompt.

Controlled set (stored claims, read-only from the 28, + anchors):

- **targets** — personal-reflective g4/g5 that the current judge floored at WEAK.
- **surface controls** — `ramen↔comfort`, `tomorrow↔tardiness` (must STAY WEAK; if
  the reflection rubric flips these too, it is merely *more lenient* → confound).
- **positive anchor** — `procrastination↔doctor-avoidance` (must stay MODERATE under
  both; sanity that the reflection rubric didn't break the obvious-good case).

Read:

- targets WEAK→MODERATE **while** surface controls stay WEAK ⇒ **(b)**: the bias was
  the distinct-fields requirement; the same detector + reflection rubric recovers
  real mechanisms → proceed to self-R.
- targets stay WEAK ⇒ **(a)**: reflection truth genuinely isn't structural; the
  therapist is the truth oracle → reflection needs a different truth term.
- surface controls flip too ⇒ rubric is just lenient → discount target flips.

**Results (run 2026-06-29, `reflection_rubric_flip_20260629_112917.json`):**

| item | grade | current | reflection |
|---|---|---|---|
| parents↔family reunion | 5 | WEAK | **MODERATE** |
| to-do-more↔human-performance | 5 | WEAK | **MODERATE** |
| fear↔economics | 5 | WEAK | WEAK |
| in bed↔ecosystem | 4 | WEAK | WEAK |
| at the gym↔pattern theory | 4 | WEAK | WEAK |
| finishing grad school↔earth-system | 4 | WEAK | WEAK |
| eating↔dishabituation | 5 | WEAK | WEAK |
| **ramen↔comfort** *(surface control)* | 3 | WEAK | **MODERATE ✗** |
| **tomorrow↔tardiness** *(surface control)* | 5 | WEAK | **MODERATE ✗** |
| procrastination↔doctor *(anchor)* | — | MODERATE | MODERATE ✓ |

**CONFOUND — the naive reflection rubric is merely more lenient.** Both surface
controls flipped to MODERATE, so the 2 target flips are not trustworthy. Diagnosis
from the justifications: removing the adversarial "your job is to reject these" stance
turned the judge from an *evaluator of the claim* into a *charitable generator of the
best mechanism the bare pair could support* — it invented a real "fixed template +
swappable components → variation without open-ended choice" mechanism for
`ramen↔comfort` that the user's actual claim ("warm and familiar") never contained.
The discovery rubric's rigor was the **adversarial stance**, not just the variable-swap
test; it is load-bearing and not removable. So the §3a fork resolves as **neither (a)
nor (b) cleanly — it is (c): wrong corpus.**

**The run produced two findings more valuable than a clean flip would have:**

1. **The 28 are the wrong corpus.** The judge's own reflection-mode reasoning flagged
   `in bed↔ecosystem` as *"not a recurring mechanism in the person's life — a literal
   biological description of a bed."* These are machine-confabulated **discovery-style
   bridges** (personal fragment → random wiki article), **not genuine reflection
   insights** (own-context ↔ own-context). Re-grading / re-rubricking them tests the
   wrong thing.
2. **Genuine reflection structure passes the EXISTING judge.** The anchor
   (`procrastination↔doctor`, a well-formed two-life-domain reflection) scored MODERATE
   under **both** rubrics. The structural truth detector is **not** the bottleneck for
   genuine reflection.

**This REVISES §3's orthogonality reading (walk-back).** The cross-tab "judge floors
20/22" is **confounded** — those 22 are machine bridges, so it measures *machine-bridge
quality vs theme-resonance*, not *reflection-truth vs structural-truth*. The earlier
strong claim "the structural judge can't gate reflection" is **withdrawn**: it can, on
genuine reflection candidates.

**The real bottleneck is the GENERATOR, not the truth term or self-R.** Reflection
needs candidates shaped like the anchor — two contexts in the user's *own life* sharing
one mechanism — generated deliberately, not the discovery generator (personal↔wiki)
pointed at personal data.

**Next step (replaces "build self-R first"):** build a tiny reflection candidate
generator that pairs two contexts from the user's own mechanisms, produce ~6
anchor-shaped candidates, run them through the **existing** judge.
- pass like the anchor ⇒ truth term is fine; the open question narrows to *same-
  relationship* reflections (`dad-now↔dad-as-child` died WEAK earlier as "one
  relationship, not a mapping") — the only place a rubric tweak might be needed.
- fail ⇒ then, and only then, reopen the truth-term question.

self-R (novelty) remains future work but is **downstream** of having well-formed
candidates to be novel about.

### 7b. Generator-test results (run 2026-06-29 #2, `reflection_generator_judge_20260629_122024.json`)

Ran a **pre-registered, therapeutic-reality-labeled** probe set through the EXISTING
production judge (`_stage_5`, incl. skeptic pass): 1 anchor, 2 cross-domain-real, 2
same-relationship (the residual), 2 thin controls. Labeled by *would a therapist call
this a real pattern in the person's life* — deliberately NOT by structural-rubric-fit.

| candidate | category | predict | judge |
|---|---|---|---|
| procrastination↔doctor | anchor | PASS | MODERATE ✓ |
| rehearsing speech↔rewriting a text | cross-domain | PASS | MODERATE ✓ |
| advisor-emails↔unopened mail | cross-domain | PASS | **WEAK** |
| advisor-deference↔dad-as-teen | same-relationship | ? | **MODERATE** |
| relief-advisor-cancels↔relief-friend-cancels | same-relationship | ? | WEAK |
| same-dish↔rewatch shows | thin | FAIL | WEAK ✓ |
| background-noise↔fan | thin | FAIL | WEAK ✓ |

**Every verdict is principled on inspection — including the two that missed my
prediction, where the JUDGE was right and my label was loose:**

- **Thin controls both correctly rejected** — no leakage in the production (adversarial)
  judge. The §7 confound was the rubric change, not the judge.
- **The dad-style same-relationship case PASSED.** The judge accepted a real tunable
  mechanism transferring across two *different* authority figures (advisor + dad). **The
  residual is NOT a wall.**
- The **boundary** the judge draws is exact: it passes a claim naming a **tunable
  mechanism that transfers across genuinely-distinct contexts**, and rejects (i) the
  **same act in cosmetically-different media** (`advisor-email↔physical-mail` — "not
  genuinely cross-domain"; my "distinct domains" label was wrong) and (ii) **the same
  observation without a transferable parameter** (`relief↔relief` — "two instances of the
  same single phenomenon," the *old* dad/dad failure mode).

**Resolution — the truth term needs NO change, and neither does the rubric.** The
existing structural judge is already a correct reflection gate. This **overturns §3's
"swap the truth term to therapist-grounded"** — that was an over-correction driven by the
confounded cross-tab. The handoff's original **"one detector, swap only the novelty
corpus R"** is **vindicated.**

**The entire remaining problem is GENERATION + novelty.** Two hard generator
requirements, both derived from the judge's boundary:
1. **Pair across genuinely-distinct contexts/figures** — never one figure with an
   abstraction of itself (kills old dad/dad), and not the same act in a different medium
   (kills advisor-email/mail).
2. **Articulate a tunable mechanism that transfers** — a parameter whose change predicts
   behavior in the other context — not a co-observed feeling.

Get those right and the existing judge passes the candidate (truth) for free; **self-R
then gates novelty.** Reflection arm = `existing structural judge × self-R novelty` over a
**purpose-built reflection generator**.

### 7c. End-to-end prototype on REAL data (run 2026-06-29 #3)

> **[TOP-LINE SUPERSEDED by §7d.]** This section concluded "the judge is a correct
> reflection gate; bottleneck = generator craft." §7d (same day, run #4) reads the
> judge's *own justifications* and overturns that: the judge's pass-verdict demands
> cross-system **coupling** (Gentner isomorphism), which reflection structurally
> lacks. The empirical observations below stand; the conclusion drawn from them does
> not. Keep reading to §7d.

First end-to-end reflection run on the owner's real memory:
`scripts/reflection_generator_prototype.py` (harvest production chroma read-only →
NEW generator LLM call → EXISTING judge), with two verification probes:
`scripts/probe_judge_reflection_stability.py` and
`scripts/probe_exact_proto_rejudge.py`.

**A methodological near-miss, recorded as a warning.** Four generator iterations
(monotone → structured → cross-system → few-shot the passing exemplars) produced
**11 candidates from the real journal, all WEAK**. Read naively, 0/11 looks like
"the judge floors genuine reflection" — i.e. §7b is wrong and the two jobs are
anti-correlated. **That reading is false.** Two checks killed it:
- **Stability probe** — the judge is deterministic, not boundary-noise: the two
  hand-authored MODERATEs *and a cleaned rewrite of a generated claim* each scored
  **4/4 MODERATE**. The judge reliably PASSES well-formed reflection, generated
  content included.
- **Exact re-judge** — the generator's **byte-exact** output scored **0/4 WEAK**
  (both run-4 candidates), while my **cleaned paraphrase** of the same Oliver↔dad
  insight scored **4/4 MODERATE**. Same pair, same mechanism; the *only* difference
  was the wording of the prediction clause.

**Lesson:** a raw WEAK-rate conflates *bad pair* with *bad wording* and a single WEAK
draw is not a verdict — always check stability + exact text before reading a rate.

**What it confirms (now on real data, deterministically):** §7b stands — the judge is
a **stable, discriminating, correct reflection gate**. The bottleneck is the GENERATOR,
and it splits in two:
1. **Yield** — most generated pairs are genuinely *same-system* (kavarin↔midterm = one
   energy-overdraw system) or monotone; the judge correctly floors them. Good
   cross-system pairs are rare but nonzero (Oliver↔dad *was* one).
2. **Craft** — even a genuinely-good pair gets floored if the claim is worded to
   **collapse the two systems into one shared driver**. The exact lever, isolated by
   the 0/4-vs-4/4 contrast:
   - WEAK: *"my tolerance tracks how tight my money is that month"* — one global driver
     (money) spanning both → reads as a single monotone arrow.
   - MODERATE: *"concession rises as my independent buffer of the thing **they** control
     falls"* — each authority is an independent system controlling a *different*
     resource, sharing one **relation** (buffer→concession) → variable-swap survives.

So generator requirement 2 sharpens once more: **state the mechanism as a per-system
relation instantiated separately in each context, never a single global driver.**

**Honesty caveat:** the passing rewrite was authored by a human (Claude), not the
generator. The generator found the right pair+mechanism; the structurally-precise
wording was supplied. **Next experiment (cheap, decisive):** add a **self-rewrite
stage** — generator proposes pair+mechanism, a second pass rewrites for two-system
precision, *then* judge. If the generator's own rewrite of Oliver↔dad clears MODERATE,
the reflection loop closes with no human in it. Only then is self-R novelty worth
wiring (§8).

### 7d. The judge's own reasoning: it is NOT a correct reflection gate (run 2026-06-29 #4)

Scripts: `scripts/test_reflection_self_rewrite.py`, `scripts/probe_concept_vs_claim.py`,
`scripts/probe_pg_justification.py`. This run chased §7c's "craft is the lever" claim
to ground and overturned the top-line. The chain:

1. **Self-rewrite stage didn't close the loop.** A generator self-rewrite of Oliver↔dad
   into an *explicitly* two-system form scored **0/3 WEAK** (control kavarin↔midstem also
   WEAK — good). The generated rewrite was *excellent* — named two distinct systems,
   per-system relations, and a decoupled differential prediction. It still floored.
2. **Concept strings are NOT the lever.** 2×2 (`probe_concept_vs_claim.py`): my
   paraphrase **P** passed with clean AND messy concept_a/b (3/3, 3/3); the generator
   rewrite **G** failed with clean AND messy (0/3, 0/3). So §7c's "maybe it's the messy
   concepts" is dead. **The claim wording is the entire lever** — but its *direction*
   is the opposite of §7c's theory: the terse, UNIFIED P passes; the elaborated,
   explicitly-per-system G fails.
3. **The judge's justifications explain why — and it's fatal.** On *both* P and G the
   de-jargon correctly extracts the real mechanism ("scarcity → deference to a
   resource-holder; leverage scales with outside options"). Then the verdict applies a
   **cross-system COUPLING** standard, verbatim: *"A true isomorphism would require some
   structural feature of the work system to map onto and predict a non-obvious feature
   of the family system — shared coupling, feedback, or a constraint linking the two.
   Here the systems track their OWN scarcity independently; there is no coupling… two
   parallel runs of a truism, not a mapping."*

**Conclusion (overturns §7b/§7c, re-grounds §3 and the 2026-06-28 "two axes"):** the
existing structural judge's pass-verdict is the **discovery** criterion — Gentner
isomorphism requiring the two systems to be **coupled** (a parameter in B predicts a
non-obvious feature of A). **Reflection is parallel-independent by nature** (the same
disposition surfacing in two *uncoupled* life-contexts — advisor↔dad), so it
structurally fails this test. It only "passes" when phrased TERSELY enough (P, ~90%) to
**hide** the independence; phrased HONESTLY and completely (G), which *states* the
independence, it is reliably rejected. **The judge rewards obfuscation and penalizes
honest, complete reflection** — it is not a correct reflection gate. (Why #2/#3 missed
this: P-style terse claims squeak past the judge's blind spot, so hand-authored tests
read as passes; only G, by spelling out the independence, exposed the true standard.)

**The fix — unbundle the judge (this is the real "swap only R", landed at the seam):**
- **KEEP** the de-jargon **extraction** as the reflection truth gate: *is there a real,
  nameable, predictive mechanism operating in both contexts?* Thin controls
  (`same dish↔rewatch shows` = "comfort", no mechanism) fail this; advisor↔dad passes.
- **DROP** the **coupling/isomorphism verdict** ("known truism applied twice → reject").
  That is the discovery standard; it is anti-correlated with reflection value (a *known*
  mechanism newly seen across two parts of one's life is exactly the point).
- **Gate novelty with self-R**, not world-coupling-novelty.

**Next build (replaces the self-rewrite step):** a reflection-specific truth gate that
reuses the de-jargon extraction but replaces the final verdict — passes "a real mechanism
operates in both," rejects "no mechanism / pure surface theme," and does NOT require
coupling. Validate it on the same probe set: must still reject the two thin controls AND
accept genuine reflection (advisor↔dad, P *and* G). If it cannot cleanly separate those,
the picture is muddier than this and we reopen. (Confidence note: the "judge as reflection
gate" call has flipped twice today; this version is anchored in the judge's *verbatim*
coupling requirement + the P/G obfuscation asymmetry, the strongest evidence so far — but
the falsifier above is the gate to clear before trusting it.)

### 7e. The unbundled gate, built and validated: truth gate is a LENIENT FLOOR; discrimination is personal-domain-distance × self-R (run 2026-06-29 #5)

Script: `scripts/reflection_truth_gate_prototype.py`. Built the §7d gate (de-jargon
extraction + a verdict that drops coupling) and ran the §7d validation. Two prompt
designs, 3 reps each over a 7-probe set:
- **v1** (verdict = "real mechanism operates in both?"): genuine reflection all VALID
  3/3 **including G** — but **both surface controls also VALID 3/3** (the LLM charitably
  manufactured a mechanism: dish↔rewatch → "choose known options to avoid disappointment";
  noise↔fan → "ambient sound settles the nervous system").
- **v2** (added counterfactual / tunable-factor rigor): **identical** — genuine all VALID
  incl. G; **both controls still VALID 3/3** (the LLM manufactured counterfactuals too).

**Two motivated designs leaking the same way is the result, not a tuning failure.**
There is **no claim-intrinsic property** separating a genuine reflection from a thin
theme: `dish↔rewatch` and `advisor↔dad` are *both* coherent mechanism-claims with
counterfactuals. What makes the controls thin is **personal-domain distance** — dish &
rewatch are both *leisure-comfort*; noise & fan are both *ambient-sound* — the two
contexts are the **same region of the person's life**. advisor & dad are *different*
regions (career vs family). The thinness is distance, not truth.

**Architecture (this is `truth × distance-from-R` with R = the person's own life-graph):**
- **Reflection truth gate = a lenient coherence/mechanism FLOOR.** Passes any coherent
  mechanism-claim (incl. thin ones), rejects only gibberish / pure-feeling-no-mechanism.
  Its *positive half is robustly validated*: genuine reflection incl. the honest **G
  passes 3/3 across two independent designs** — the §7d coupling judge structurally never
  could. Trying to make it *also* reject thin themes intrinsically is **infeasible** and
  was the wrong job to give it.
- **Reflection value = personal-domain distance (career↔family far/valuable;
  leisure↔leisure near/thin) × self-novelty (hadn't recognized it).** The surface controls
  die *here* — same-domain — not at the truth gate.
- **Flip-test confound, vindicated:** you cannot validate the lenient truth gate in
  isolation (any lenient gate passes the controls); only `truth × self-R` together,
  against human grades. **Self-R / personal-domain-distance is load-bearing, not deferred
  polish.**

**Next build (the real core of the reflection arm):** the personal-domain-distance / self-R
signal — pair-selection across *distant regions of the person's own life-graph*, which is
simultaneously (a) the novelty/value gate and (b) the generator's pair-selection strategy.
This is the **validated anchored-vs-random machinery** (`project_synthesis_anchored_vs_random_validated`)
run over the **personal graph** instead of wiki. Same-domain pairs (the thin controls) are
*near* and filtered; cross-domain pairs (advisor↔dad) are *far* and surfaced. Only then
can the reflection arm be validated end-to-end (`truth_floor × personal_distance × self_novelty`
vs human grades).

### 7f. Phase 2 built + run: random pairing is ~0-yield; ANCHORED selection works; distance is a same-domain FILTER + SWEET-SPOT, not a monotone ranker (run 2026-06-30 #6)

Scripts: `scripts/reflection_validation_harness.py` (random/stratified baseline + the 4-check
analyzer incl. the orthogonality test) and `scripts/reflection_anchored_generator.py` (the
§7e "real core" — anchored pair-selection over the personal graph, R=self). Substrate =
the LOCKED entity-pooled UMAP+fact-text mcs=5 clustering (`reflection_domain_clustering.py`);
distance metric = same-cluster gate + entity-to-entity BGE cosine (centroid distance washed
out at <0.09). Pairing pool = LIFE-domain, non-junk-cluster, noun-like entities (spaCy POS +
NER filter drops scrap members; a label denylist drops whole mood/chat/schedule/status
clusters — `mention_count` is dead, median 1).

**The validation harness (random/stratified, 46 pairs) overturns the *generative* assumption.**
The architecture read "distance → value: rank cross-domain pairs by distance, far = valuable."
The data says distance over RANDOM pairs predicts **DECLINE**, not value:

| stratum | mean d | decline% | claims | truth-pass |
|---|---|---|---|---|
| same_domain | 0.14 | 20% | 8 | 4 (true-but-**thin**) |
| cross_near | 0.23 | 90% | 1 | 0 |
| cross_mid | 0.22 | 80% | 2 | 0 |
| cross_far | 0.32 | 100% | 0 | 0 |
| gold (hand-picked) | 0.17 | 17% | 5 | 1 |

The declined cross pairs (`linear algebra↔Benadryl`, `kitchen↔diet coke`, `john↔veterinarian`)
genuinely share no mechanism — the generator declines **correctly** (it bridges from the same
terse input whenever a real bridge exists: every gold REAL anchor produced a claim). So:
**distance is a NECESSARY FILTER (kills same-domain thinness), not a SUFFICIENT GENERATOR.**
You cannot rank random distant pairs into reflections — 90-100% aren't reflections at all.
The orthogonality test is therefore **unanswerable on random pairs** (cross cells ~empty) —
and that emptiness *is* the result.

**The anchored generator (`reflection_anchored_generator.py`) — bridge-first, not pair-first.**
Stage 1 mines recurring SELF-mechanisms that span ≥2 distinct life-domains and picks one
anchor entity from each (the mechanism is the anchor); Stage 2 reuses the harness's grounded
generator + truth/dual-grounding gate verbatim; a cross-domain filter keeps only different-
cluster pairs. Result over 11 mined anchored pairs: **27% truth-pass (3/11), 36% decline —
vs the random baseline's 0/30 cross truth-pass (~93% decline).** Anchoring lifts cross-domain
reflection yield from ~0 to ~27%. This is the reflection analogue of the discovery arm's
validated anchored≫random result (`project_synthesis_anchored_vs_random_validated`).

**Distance role, now empirically pinned to an INVERTED-U (sweet spot), not monotone:**
- near-anchored (d 0.09-0.11): `dad's house↔Mom` INVALID re-description; `overwork↔hungover`
  ungrounded — **thin even when anchored** (distance filter earns its keep);
- moderate-anchored (d **0.18-0.25**): the passes — `kavarin↔beer`, `chicago↔iceland`,
  `conceptual-understanding↔linear-algebra` — the same band the gold REAL anchors sit in;
- extreme-anchored (d 0.44-0.48): `knowledge-graph↔study-guide` (ungrounded), `tear-gas↔
  rock-climbing` (declined) — too far, the "mechanism" is a stretch.

So the value axis is **`distance ∈ sweet-spot band`**, not `max distance`. Open items: (a) the
truth gate's dual-grounding looks slightly strict on terse per-entity facts (rejected genuine
gold `dad↔kavarin` on grounding); (b) a project-entity leak persists via the mixed "Daemon &
friends" cluster (`knowledge graph`); (c) `kavarin↔beer` exposes a **personal-domain vs
semantic-system** tension (semantically "both substances"/same-system, but the user's *life*
files them in different regions — health/psych vs brewing/career) — exactly a hand-label
judgment call. The 16 real harness claims await the owner's two-column hand-label
(truth-grounded? / valuable?) to lock gate calibration + the thin-vs-gold distance gap.

**Next:** self-novelty (§8, reuse `SurfacingHistory`) layered onto the anchored sweet-spot
candidates; firm up the 27% yield (small n=11, single mining seed).

### 7g. BUILD COMPLETE — all three axes integrated end-to-end (run 2026-06-30 #7)

Scripts: `scripts/reflection_self_novelty.py` (SelfNoveltyOracle) + `scripts/reflection_pipeline.py`
(the integrated scorer). The full `GROUNDED-TRUTH × DOMAIN-DISTANCE × SELF-NOVELTY` pipeline now
runs end-to-end on real data:
  [1] anchored generation (mine recurring self-mechanism → cross-domain pairs);
  [2] grounded-truth + dual-grounding gate (lenient FLOOR);
  [3] domain-distance inverted-U value (`distance_value()`, peak ~0.215, zero <0.085/>0.345);
  [4] self-novelty (SelfNoveltyOracle) — **growable mechanism registry + SurfacingHistory
      cooldown, CLAIM-LEVEL "same mechanism?" LLM check (NOT embedding-proximity).**
  SURFACED ⇔ truth_pass ∧ distance_value>0 ∧ self_novel; `reflection_score = truth · dv · self_novel`;
  surfaced mechanisms are recorded so they don't re-surface.

**Self-novelty discriminates (the §9 demonstration, done):** with a seeded registry, a PARAPHRASE
of a recognized mechanism *across different life-areas* (boss/dad vs advisor/dad) is correctly
caught as self-known, while a genuinely different mechanism passes — exactly the case embedding
distance would miss. This is why the check is claim-level (the discovery-arm lesson:
embedding-proximity novelty is blind — see [[project_synthesis_discovery_generation]]).

**The substrate eligibility filter grew** a project-leak exclusion: the mixed "Daemon & friends"
+ "Demo videos" clusters (Daemon/FAISS contaminating the life domains) are now denylisted in
`eligible_life_entities`. **Intrinsic limit confirmed:** yield is low and noisy per mining draw
(1/13 here, 3/11 in §7f) — genuine novel+truthful cross-domain reflections are RARE (the
reflection twin of discovery's "the canon is sparse"); and the lenient truth-floor + cluster-based
distance occasionally pass a borderline-thin pair (`McDonald's↔diet coke`, both consumption
defaults, landed in different clusters but near d=0.17 → distance correctly down-weighted it to
0.63, not rejected). **Build is complete; what remains is not building but VALIDATION** — the §8
hand-label / orthogonality test (owner) and a production-wiring decision (deferred, like discovery).

---

## 8. Cheap self-R validation (after §7 passes)

`scripts/score_graded_reflection_candidates.py` *(planned — not yet written;
the built self-novelty pieces live in `scripts/reflection_self_novelty.py` +
`scripts/reflection_pipeline.py`, see §7g)*. Two terms only:

```
reflection_score = truth_pass × self_novelty
```

Do **not** use a 4-term product (× mechanism_realness × personal_specificity) —
those have no scorers yet; including them tests *hand-rating*, not the architecture.

CSV columns (keep axes separate for diagnosis — do not read the product alone):
`candidate_id, candidate_text, human_grade, coherence_verdict, self_novelty_score,
nearest_self_R_match, reflection_score, keep_or_reject_reason`.

**Known limitation of running this on the 28:** they are coherence-gated (19/27 die
at coherence before `self_novelty` acts) and contain ~no already-recognized
mechanisms (they are the *interesting* ones), so `self_novelty ≈ 1` across the board.
On the 28 this test can only check that self-R **doesn't falsely reject** good
candidates — it cannot demonstrate self-R's discriminative value. To test that, add
a few constructed already-recognized negatives (§9).

---

## 9. Balanced negatives (later — where bigger n returns)

- surface-personal junk (ramen/gym) → tests coherence rejection.
- already-recognized mechanisms from journal/therapy → tests self-novelty rejection.
- known world mechanisms with high personal relevance → tests separation of world
  novelty from reflection novelty.

---

## 10. Implementation seam (minimal — no cathedral)

```
NoveltyOracle.score(candidate) -> {novelty_score, nearest_known, evidence, reason, reference_type}
  ├── WorldNoveltyOracle   (R = wiki / co-occurrence oracle; wraps existing logic)
  └── SelfNoveltyOracleV0   (R = small YAML mechanism registry; label/alias distance, NOT raw text)
```

---

## 11. Limits (don't over-trust)

- The 28 are **selection-biased positives** (recall signal, not precision).
- No graded negatives → cannot calibrate a threshold from them.
- Cross-tab orthogonality rests on n=22, mixed set — strongly suggestive, not proof.
  The §7 flip test is the disambiguator.
- A passing §7/§8 means "architecture doesn't contradict the labels," **not**
  "calibrated."
```
