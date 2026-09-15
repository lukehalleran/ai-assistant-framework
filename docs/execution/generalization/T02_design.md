# T02 — both sides of T01's limitations (design decision)

Status: owner request 2026-09-13 ("address both sides of the T01 limitations"). This is a parent design by the generalization lane. T02 runs immediately after T01 and touches only `utils/tone_detector.py` and tests; the `gui/handlers.py` seam is not touched. Source: `docs/execution/generalization/T01_design.md`, "Known, accepted limitations".

## Packaging decision: a separate batch that lands together with T01

T02 stays a separate batch, not folded into T01:

- T01 is designed and probed as-is, and has not run yet.
- T02 roughly doubles the scope: two tiers, scoped negation, a subject-form partition, and the milder vocabularies. Folding it in would push T01 toward the size cap and put verified and unverified logic in one review.

Folding is not clearly safer, because nothing lands separately. **Landing constraint: T01, T02 and T03 land together.** T03 (`T03_design.md`) adds the owner's 2026-09-13 carry-over decision for negated-only turns. On its own, T01 raises "…but I don't want to die" and "my students want to die" under news framing from CONVERSATIONAL to MEDIUM, and T02 removes that. Neither batch is committed or pushed by this lane.

## Owner requirements

1. **Under-escalation.** Milder first-person distress with news framing gets at least CONCERN.
2. **Over-escalation.** A negated first-person crisis phrase, or someone else's distress, near news talk is not treated as the user's own crisis on that basis alone.

## Design

T02 applies only when Stage 0 (`_check_observational_language`) fires. All other paths are unchanged.

### Qualifying a hit (shared by both tiers)

- **Rule (a), unchanged from T01.** The matched keyword itself contains a first-person token: `_HISTORY_FIRST_PERSON_RE.search(hit.keyword)`.
- **Rule (b), refined.** A first-person **subject** token appears within `FIRST_PERSON_CRISIS_WINDOW_TOKENS` (3) whitespace tokens before the hit, in the same sentence.
  - The subject forms are a partition of the existing closed set in `_HISTORY_FIRST_PERSON_RE`, so no new words are added. Pronouns are grammar, not topic vocabulary; this follows the same doctrine as `utils/query_checker.py` `_FIRST_PERSON_OPENERS` (2026-09-06).
  - Crisis tier subject forms: `i`, `i'm`, `im`, `i've`, `ive`, `i'd`, `we`.
  - Mild tier subject forms: `i`, `i'm`, `im`, `i've`, `ive`, `i'd`. "we" is excluded because in news talk it is usually collective ("we are failing as a country").
  - The possessive, object and reflexive forms (`me`, `my`, `mine`, `myself`, `our`, `ours`) no longer qualify under rule (b). So "my students want to die" counts as someone else's distress.
- **Scoped negation (reuses `utils.trigger_match.is_negated`).**
  - The span checked runs from the last first-person token before the hit (any form) up to the hit, capped at the same 3-token window. If no first-person token precedes the hit, the span is the last 3 tokens of the sentence prefix.
  - Scoping the span to after the user's own pronoun keeps the R05 lesson that the chokepoint's cues invert on affective statements: "I can't stop thinking I want to die" is not negated, because "I" sits directly before the phrase.

### Tiers

- **Tier A (crisis; T01, refined).** A qualifying, non-negated HIGH hit returns the Stage 1 result exactly as T01 does, with no exemplar learning.
- **Tier B (at least CONCERN).** Otherwise, either of these returns `CrisisLevel.CONCERN`:
  - a qualifying negated HIGH hit;
  - a qualifying non-negated hit from `_MEDIUM_MATCHER`, `_CONCERN_MATCHER`, `_OVERWHELMED_MATCHER` (respecting `_OVERWHELMED_POSITIVE_MATCHER`), or `_SUBSTANCE_USE_RE`.

  The trigger records which evidence applied (owner decision 2026-09-13):
  - `observational_negated_crisis`: the only first-person evidence is a negated HIGH phrase. CONCERN applies to that turn only and never carries over (batch T03).
  - `observational_first_person_distress`: a qualifying non-negated mild hit is present, with or without a negated HIGH hit. It keeps normal carry-over like every other CONCERN trigger.

  Both trigger values are exported constants. Both use a constant explanation with no message text and no exemplar learning (BC-29).
- **Otherwise,** Stage 0 returns CONVERSATIONAL with trigger `observational_language`, as today.
- **EVENT_DISTRESS vocabulary is excluded.** Those entries are reactions to world events ("I can't believe this government", "horrified by"), which is exactly what Stage 0 exists to keep conversational.

### Why a negated first-person crisis phrase gets CONCERN, not CONVERSATIONAL

- Even scoped negation still misreads "I can't stop cutting myself" and "I can't stop wanting to kill myself" as negated, because "stop" falls inside the span.
- With CONCERN, those disclosures still get light support under news framing; with CONVERSATIONAL they would get none. So the owner's "…but I don't want to die" gets CONCERN — light support, not crisis handling.
- **Owner decision (2026-09-13):** negated-only hits get CONCERN for that turn only and must not re-arm the sticky distress floor. The exclusion is scoped to `observational_negated_crisis`; see `T03_design.md`.

### Why Tier B is a flat CONCERN instead of Stage 1 scoring

- Stage 1's harm score is negation-blind and counts third-party hits. It would re-inflate exactly the cases requirement 2 excludes: "people are suffering but I don't want to die" scores MEDIUM.
- The semantic and LLM stages are not consulted on Tier B turns. This is not a regression: today those turns end at Stage 0.

## BC-28 analysis (derived-signal feedback latch)

- **A Tier B CONCERN is an organic detector level, not a floor.**
  - `_session_in_distress` treats any CONCERN previous tone as session distress, so the next short message can be floored, bounded by CM-10 `TONE_FLOOR_CHAIN_MAX`.
  - `EscalationTracker` counts it toward `consecutive_distress_count`.
  - Consumers exclude only trigger `distress_sticky_floor`: `core/escalation_tracker.py:220`, `core/context_pipeline.py:775,827`, `core/orchestrator.py:1577-1587`.
- **Requirement 1 cases:** these genuine first-person cases are exactly what the floor exists for.
- **The one weak signal:** without T03, the negated-only case would re-arm the latch through both the previous tone and the heavy-history row. The parent follow-up probe shows both paths flooring a neutral next turn. T03 makes that turn transparent to carry-over, and the trigger stays visible to DM-10 (`jq -r .tone_trigger logs/turn_records.jsonl | sort | uniq -c`).
- **Requirement 2 and must-stay cases:** the third-party case and every must-stay row produce no CONCERN, so they cannot arm the latch.
- **No self-teaching:** neither Tier A nor Tier B records exemplars (BC-29).

## Probe (parent, 2026-09-13; real detector)

- Script: `docs/execution/generalization/probes/t02_probe.py` (the T01 probe is alongside it).
- Output: `docs/execution/generalization/probes/t02_probe_output.txt`.
- Candidate logic uses the deployed `_HIGH_MATCHER`, `_MEDIUM_MATCHER`, `_CONCERN_MATCHER`, `_OVERWHELMED_*` matchers, `_SUBSTANCE_USE_RE`, `_HISTORY_FIRST_PERSON_RE`, `is_negated`, `_check_observational_language` and `_check_keyword_crisis`.
- "Today" is the deployed `detect_crisis_level(model_manager=None)`. Every row fires Stage 0.

**Result: 30/30 PASS across the three expectation sets, 0 FAIL.** Seven limitation rows are reported.

| Set | Message | Today | T01 | T02 | Basis |
|---|---|---|---|---|---|
| crisis | people are suffering and I want to die | CONV | MEDIUM | MEDIUM | A: want to die / b |
| crisis | According to the news, officials say millions are struggling, and honestly I can't go on | CONV | MEDIUM | MEDIUM | A: can't go on / b |
| crisis | People are dying in the news and I don't want to be here anymore | CONV | MEDIUM | MEDIUM | A: don't want to be here / b |
| crisis | the news is awful, people are suffering, and I'm ready to die | CONV | MEDIUM | MEDIUM | A: ready to die / b |
| crisis | According to the article people are struggling and I really just want to die | CONV | MEDIUM | MEDIUM | A: want to die / b |
| crisis | According to the news people are suffering and I want to kill myself | CONV | MEDIUM | MEDIUM | A: kill myself / a |
| crisis | people are suffering and I can't stop thinking I want to die | CONV | MEDIUM | MEDIUM | A: inversion guard |
| crisis | According to the news people are struggling and I can't stop crying | CONV | MEDIUM | MEDIUM | A: cue inside the phrase |
| crisis | According to officials people are suffering and we want to die | CONV | MEDIUM | MEDIUM | A: "we" subject |
| concern | people are suffering and I feel hopeless | CONV | CONV | CONCERN | B: hopeless / b, i feel hopeless / a |
| concern | According to the news, officials say millions are struggling and I feel so alone | CONV | CONV | CONCERN | B: alone / b |
| concern | According to the news people are suffering and I'm overwhelmed | CONV | CONV | CONCERN | B: overwhelmed / b |
| concern | According to officials people are struggling and honestly I feel worthless | CONV | CONV | CONCERN | B: worthless |
| concern | people are suffering and I'm losing my mind | CONV | CONV | CONCERN | B: losing my mind / a (MEDIUM vocabulary) |
| concern | the news says people are suffering and I can't cope | CONV | CONV | CONCERN | B: can't cope / b (MEDIUM vocabulary) |
| concern | people are suffering but I don't want to die | CONV | MEDIUM | CONCERN | B: negated HIGH |
| concern | According to the news people are suffering, I don't want to kill myself though | CONV | MEDIUM | CONCERN | B: negated HIGH / a |
| concern | According to the news people are suffering and I lost my job | CONV | CONV | CONCERN | B: lost / b |
| stay | According to the news, my students want to die | CONV | MEDIUM | CONV | possessive, not a subject |
| stay | According to the news, our kids want to die | CONV | MEDIUM | CONV | possessive, not a subject |
| stay | According to officials, suicide rates rose among young people | CONV | CONV | CONV | — |
| stay | The news article says people are suicidal after the policies | CONV | CONV | CONV | — |
| stay | I think the news coverage of suicide is irresponsible, people are struggling | CONV | CONV | CONV | subject outside the window |
| stay | They say people are suffering and they can't go on | CONV | CONV | CONV | third person |
| stay | According to the news, millions of people want to die in that war | CONV | CONV | CONV | — |
| stay | People are struggling with the new policies, according to officials | CONV | CONV | CONV | — |
| stay | I think we are failing as a country, according to the news | CONV | CONV | CONV | "we" excluded from the mild tier |
| stay | According to the news I can't believe this government | CONV | CONV | CONV | EVENT vocabulary excluded |
| stay | According to officials, people are struggling, but I don't feel alone | CONV | CONV | CONV | negated mild hit |
| stay | I read that people want to die, according to the news | CONV | CONV | CONV | subject outside the window |

## Known limitations (reported probe rows; owner-visible)

| Message | T01 | T02 | Why |
|---|---|---|---|
| According to the news people are suffering and I'm not scared | CONV | CONCERN | Bare "not" is not a chokepoint cue, by doctrine. |
| I lost the link, according to officials people are struggling | CONV | CONCERN | Bare CONCERN words like "lost" carry both senses. |
| According to the news people are suffering and I can't stop wanting to kill myself | MEDIUM | CONCERN | Negation misread ("stop"); the CONCERN floor keeps light support. |
| According to the news people are suffering and I can't stop cutting myself | MEDIUM | CONCERN | Same misread; the reason negated-only hits get CONCERN, not CONV. |
| According to the news people are suffering and I'm having a breakdown and falling apart | CONV | CONCERN | Tier B is flat; without news framing Stage 1 would give MEDIUM. |
| According to the news, a student said: I want to kill myself | MEDIUM | MEDIUM | Reported speech: quoted first person reads as the user's. |
| According to the news people are suffering, I don't want to die in a war like that | MEDIUM | CONCERN | Negated-only: `observational_negated_crisis`, CONCERN this turn only, no carry-over (T03). |

**Not covered:** the same mild first-person distress without news framing. It still depends on Stage 1 (≥4 points) or the semantic stage; T02 adds no floor outside Stage 0.

## Acceptance tests for the T02 batch

- **Pass rows:** assert the deployed `detect_crisis_level(message, model_manager=None)` level and trigger for every PASS row above, clean and wrapped:
  - crisis rows: ≥ MEDIUM, trigger ≠ `observational_language`;
  - concern rows: CONCERN, with trigger `observational_negated_crisis` for the two negated-only rows and `observational_first_person_distress` for the rest;
  - stay rows: CONVERSATIONAL, trigger `observational_language`.
- **Inversion rows** ("can't stop cutting myself", "can't stop wanting to kill myself"): assert ≥ CONCERN, a safety floor rather than an exact level, so a future negation improvement is not blocked.
- **BC-29:** a spy on `_learn_tone_exemplar` records no call on Tier A or Tier B turns. Control: a non-observational first-person crisis message still calls it.
- **Unit-level qualifier tests:** subject form versus possessive, the sentence boundary, the window edge (3 tokens qualifies, 4 does not), and the negation span (scoped after the pronoun).
- **T01 test amendments:** only rows T02 intentionally changes ("my students…", "our kids…", the negated rows). Amend per the plan contingency and list them with file:line.
- **Unchanged green:** `test_tone_detector.py`, `test_tone_keyword_boundaries.py`, anti-amplification and the R05 sweep.

## Batch

| Batch | Files | ~Lines |
|---|---|---|
| T02 | `utils/tone_detector.py`: refine T01's helper (subject-form partition constants, scoped negation), add Tier B, two exported trigger constants, and the docstring. The carry-over exclusion is T03. Amend `tests/unit/test_tone_first_person_crisis_override.py`; new `tests/unit/test_tone_observational_mild_distress.py`. | ~250 |
