# H02 — categorized-generic stressor coverage across life domains (design, parent draft)

## Status

- **Source:** the owner's hardcoding review, item 2 (2026-09-13); see `docs/execution/generalization/owner_hardcoding_review_20260913.md` "H02". It inherits `docs/GENERALIZATION_AUDIT_20260901.md` item 9, whose family and caregiving coverage is still open.
- **Doctrine:**
  - remedy patterns #5 (CATEGORIZED-GENERIC + ANCHORS) and #2 (SEEDS+LEARNED);
  - the vocabulary-miss rule in `docs/DEVELOPMENT_WORKFLOW.md`, and BC-76 (no phrase appends);
  - BC-28 (over-escalation / carry-over), BC-29 (self-teaching) and BC-58 (inherit at the producer).
- **Queue:** after R06. This document is parent design work, done in parallel with A05b-4 and A05b-5.
- **Owner decisions D-H02-1..3: decided 2026-09-13** (see "Owner decisions (decided)" below). The design and the round-3 probe reflect them.
- **Implementation status (2026-09-14):** H02a and H02b are integrated.
  - **H02a** categorized the tables. Its no-change proof is `probes/h02a_noop_round3_output.txt`, identical to the recorded round 3.
  - **H02b** added the domain-strain rule. The parent's sandboxed acceptance run on the deployed detector gave 22/23 stress rows raised and 0/41 controls changed: `probes/h02b_acceptance_probe.py` and `probes/h02b_acceptance_output.txt`.
  - Evidence and parent review: `batches/H02a.md` and `batches/H02b.md`.
- **Probe discipline:** every probe below ran through `docs/execution/generalization/probes/run_sandboxed_probe.py`. That means an empty temp adaptive store, `TONE_EXEMPLAR_LEARNING=0`, `data/` sizes and mtimes unchanged, and the real `detect_crisis_level(model_manager=None)` with the locally cached embedder offline. Candidate-rule logic is simulated on top of the deployed matchers and T02's deployed `_qualifying_first_person_hit`.

## Verified gap

Parent count on the candidate after T03:

| Domain | CONCERN (130 entries) | MEDIUM (106 entries) |
|---|---|---|
| family | 0 | 1 |
| caregiving | 0 | 0 |
| housing | 1 | 0 |
| health | 5 | 2 |
| money | 9 | 0 |
| work | 7 | 0 |
| school | 2 | 0 |
| grief/loss | 6 | 0 |

`CRISIS_EXEMPLARS["concern"]` holds 26 seeds, all generic worry or sadness with no domain stress shapes.

**Fresh-user baseline** (`probes/h02_candidate_probe_round2_output.txt`, "baseline"):
- **7/21** domain-stress messages reach CONCERN or higher.
  - family 3/3 (two only through unrelated MEDIUM phrases);
  - caregiving 0/3; health 1/3; work 2/3; school 1/3;
  - **money 0/3** ("I don't know how we'll make rent this month");
  - **housing 0/3** ("our landlord is trying to evict us").
- All **34** controls stay CONVERSATIONAL.

Correction: earlier parent messages said 10/21 (unsandboxed) and 8/21 (sandboxed). Those were miscounts; see the owner-review record. The label "(today 10/21)" printed by `probes/h02_candidate_probe_a2_output.txt` is a stale hard-coded string in that round-1 script. Its per-row results are unaffected.

## Candidates probed

| Variant | Mechanism | Stress ≥ CONCERN | Controls changed (of 34) |
|---|---|---|---|
| baseline | today's code | 7/21 | — |
| A′ (round 1) | domain anchor × strain cue in the same sentence; the strain's experiencer must be the user (T02 qualifier: first-person token inside the cue, or a first-person SUBJECT within 3 tokens before it; negation scoped after the pronoun); floor at CONCERN | 19/21 | 0/28 on the round-1 control set |
| A2 (round 2) | A′ plus object-bearing threat shapes ("evict us", "kick me out") | **19/21** | **1/34**: "I paid the bills this morning" |
| B | 8 domain-spanning seeds added to `CRISIS_EXEMPLARS["concern"]` (phrasing deliberately unlike the probe messages) | 11/21 | 1/34: "I'm not stressed about rent anymore" |
| A2+B | both | 19/21 | 2/34 |
| **A3 (round 3, after owner decisions)** | A2 with three changes: strain evidence from affect/strain categories only (stressor-topic nouns excluded); "us" only as the object of a composed threat_loss shape; the anchor and strain must share a clause (split at "but"). No new seeds. | **22/23**: the 21 originals + 2 threat "us" positives. Only miss: "cash flow is tight…" (D-H02-3) | **0/41**: the 34 earlier controls + 7 threat "us" controls |

Scripts and outputs: `probes/h02_baseline_probe.py`, `probes/h02_candidate_probe_a2.py` (round 1), `probes/h02_candidate_probe_round2.py` (round 2) and `probes/h02_candidate_probe_round3.py` (round 3), each with its `_output.txt`.

Round-3 threat "us" rows:
- **Positives:**
  - "our landlord is trying to evict us";
  - "the landlord says he's going to kick us out of the apartment next month";
  - "the bank is threatening to foreclose on us and the house is all we have".
- **Controls, all CONVERSATIONAL:**
  - "our landlord promised he won't evict us";
  - "they tried to evict us years ago but we're doing fine now";
  - "we were worried they'd kick us out but the lease got renewed";
  - "the bouncer is going to kick us out of the bar lol";
  - "the game keeps trying to kick us out of the lobby haha";
  - "the landlord is trying to evict the tenants downstairs, not us";
  - "my landlord would never evict us, he's like family".
- Round 2's two false positives ("I paid the bills this morning", "I'm not stressed about rent anymore") no longer change.

### What the probes show

1. **The structural rule does most of the work.** Coverage rises from 7 to 19 of 21. Seeds add nothing on top of it (A2+B equals A2).
2. **A2's one false positive has a clear cause.** Its strain evidence reused the existing CONCERN/MEDIUM entries. `CONCERN_KEYWORDS` mixes affect and strain words ("hopeless", "scared", "exhausted") with stressor-TOPIC nouns ("bills", "debt", "deadline", "pressure"). "I paid the bills this morning" paired the anchor "bills" with the topic word "bills" and a first-person subject.
3. **Seeds (B) help less and fail on negation.** The embedding ignores negation, so "I'm not stressed about rent anymore" became CONCERN. Seeds add +4 on their own and nothing beside A2.
4. **Remaining misses:**
   - "our landlord is trying to evict us": the first-person closed set deliberately excludes "us" ("UK or US politicians").
   - "cash flow is tight and payroll is due friday": no first-person evidence at all; small-business stress in an impersonal voice.
5. **Controls that already hold:**
   - positive outings ("my kids wore me out at the zoo but it was so fun");
   - someone else's strain ("my sister is exhausted from her new job", "the kids can't keep up with me on hikes");
   - negated or relieved statements ("we paid off the loan and I feel so relieved");
   - bare domain nouns ("my kids have soccer on saturday", "the landlord fixed the sink").

## Proposed design (parent draft; owner decisions marked)

1. **Categorized-generic restructure, no behaviour change (H02a).**
   - Split `CONCERN_KEYWORDS` and `MEDIUM_CRISIS_KEYWORDS` into named categories in one table per level: `affect` (feelings), `strain` (coping and load shapes) and `stressor_topic` (per domain: money, work, school, health, housing, family, caregiving, grief).
   - The flattened sets `_CONCERN_MATCHER` / `_MEDIUM_MATCHER` consume stay byte-identical, so Stage 1 scores and T02's mild tier are unchanged.
   - Proof: an equality test on the flattened sets against their pre-restructure contents.
2. **Domain anchors (H02b).**
   - A `DOMAIN_ANCHORS` categorized table of life-domain nouns and role words, in the `_PRIVATE_SPHERE_GENERIC_TOKENS` style: one set at runtime, extended by category, not per incident.
   - `stressor_topic` entries join the anchors.
   - Anchors alone never score.
3. **Domain-strain rule (H02b).**
   - Fires when, in one sentence, a domain anchor co-occurs with a strain cue whose experiencer is the user under T02's `_qualifying_first_person_hit` (crisis subject set, which includes "we") and which is not negated.
   - The strain cue comes from the `affect` or `strain` categories plus a small `strain_shapes` table (exhaustion, overload, threat/loss, fear), NEVER from `stressor_topic`. That is the fix for A2's false positive.
   - **Threat_loss "us" (D-H02-2).** "us" is NOT added to the first-person set. A threat_loss shape qualifies with "us" as its object only as a composition of two small categorized tables: `THREAT_LEADS` (trying to, going to, gonna, about to, threatening to, might, may, could, will, want(s) to, planning to) followed by `DISPLACEMENT_SHAPES` (evict us, kick us out, throw us out, foreclose on us, cut us off, force us out).
     - Past, resolved and negated forms do not compose ("tried to evict us", "they'd kick us out", "won't / would never evict us").
     - The composed shape still needs a domain anchor in the same clause, so bar and game contexts do not fire.
   - **Clause scoping.** The anchor and the strain cue must share a clause: sentences are split at "but". This keeps resolved contrasts ("…kick us out but the lease got renewed") from firing.
   - Effect: at least CONCERN, applied only when the deterministic stages would otherwise produce a lower level.
   - Trigger: `domain_strain`, a constant. Its explanation carries no message text.
4. **Carry-over and learning.**
   - A `domain_strain` CONCERN is organic evidence of the user's own strain, so it keeps normal carry-over (BC-28). The owner's T03 exclusion applies only to `observational_negated_crisis`.
   - No exemplar learning from the rule in H02 (BC-29). Revisit after an owner canary, because a false positive ("I paid the bills…") must never teach.
5. **T02 inheritance (BC-58).** Under news framing, the domain-strain rule's qualified hits count as T02 Tier B mild evidence (`observational_first_person_distress`), wired at the producer so there is no second copy.
6. **Owner decisions (decided 2026-09-13):**
   - **D-H02-1 Seeds:** no new CONCERN seeds. The learned channel continues per user.
   - **D-H02-2 Object "us":** allowed only as the object inside a threat_loss strain shape ("evict us", "kick us out"), and kept out of the general first-person set. Threat_loss "us" controls that must stay conversational are included. The implementation is the composed `THREAT_LEADS × DISPLACEMENT_SHAPES` form in item 3.
   - **D-H02-3:** impersonal business stress ("cash flow is tight") is left to the semantic and learned channels. The rule stays first-person.

## Acceptance (for the H02 batches)

- **Sandboxed probe re-run** on the deployed code (`probes/h02_candidate_probe_round2.py`, adapted to call the real rule):
  - stress ≥ CONCERN: at least **22/23**. The one accepted miss is "cash flow is tight and payroll is due friday" (D-H02-3).
  - **0/41** controls changed, including the 7 threat_loss "us" controls.
  - The worker may extend the control set (more positive, relieved, negated and third-party shapes) but may not remove rows.
- **Unit tests** on the deployed `detect_crisis_level`, clean and wrapped:
  - every stress row;
  - every control;
  - an equality proof for the restructured flattened sets;
  - no-learning spies;
  - T02 news-framing inheritance rows;
  - a carry-over row: a `domain_strain` CONCERN floors a neutral follow-up, unlike `observational_negated_crisis`.
- **BC-28:** the per-trigger share stays visible to DM-10.

## Proposed batches

| Batch | Scope | ~Lines |
|---|---|---|
| H02a | Categorize CONCERN/MEDIUM into affect / strain / stressor_topic tables; flattened sets byte-identical (proof test) | ~200, mostly moved lines |
| H02b | `DOMAIN_ANCHORS`, `strain_shapes`, the domain-strain rule, its trigger, T02 inheritance, tests | ~300 |

The owner decisions D-H02-1..3 are recorded. H02b follows H02a; its inputs are the categorized tables that H02a creates.
