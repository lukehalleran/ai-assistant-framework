# G04: Adaptive Personalization

## Objective

Start from population-neutral behavior, learn an individual user's stable needs,
and remain reversible, explainable, and resistant to poisoning. Personalization
must not mutate global policy or turn one owner's incidents into rules for all
users.

## Layer model

| Layer | Contents | Mutability |
|---|---|---|
| Safety invariants | Authorization, privacy, data-loss, acute-risk constraints | Not user-learned |
| Population baseline | Neutral prompts, classifiers, thresholds, relation seeds | Release-versioned |
| User-declared preferences | Name, pronouns, style, storage, tool permissions | User controlled |
| User-learned behavior | Confirmed exemplars, cadence, terminology, prominence | Evidence gated |
| Session state | Current topic, tone, temporary intent, recent corrections | Expires |

The system must always be able to identify which layer caused a decision.

## Adaptation record

Every learned item should store:

```text
adaptation_id
kind
value or bounded parameter
source_event_ids
evidence_count
distinct_day_count
confidence
created_at
last_confirmed_at
expires_at
status
scope
explanation
model_and_policy_version
```

Raw private text should not be duplicated when event IDs and a safe derived
representation are sufficient.

## Step-by-step plan

### G04-T01: Inventory every adaptive mechanism

1. Catalog adaptive exemplars, learned relations, profile promotion, retrieval
   thresholds, tone carryover, suggestion tracking, topic history, and proposal
   feedback.
2. Mark whether each mechanism changes a value, seed, threshold, rank, or prompt.
3. Record its evidence source, persistence, decay, poisoning protection, user
   visibility, and reset behavior.
4. Identify hardcoded owner calibration that has no learning path.
5. Disable undocumented adaptation in stable mode until registered.

Acceptance:

- `G04-A01`: Every persistent adaptive file has a registry entry and owner.
- `G04-A02`: CI rejects a new adaptive store or parameter without a contract.

### G04-T02: Establish neutral cold-start behavior

1. Separate owner data from shipped seeds.
2. Construct neutral seeds from external evaluation data and clearly authored
   universal examples.
3. Ensure the empty-profile path does not infer demographics, schedule, location,
   relationship structure, or emotional baseline.
4. Calibrate initial thresholds on a diverse development set rather than the
   owner corpus.
5. Keep an abstain/clarify path for uncertain classification.

Acceptance:

- `G04-A03`: Empty-profile scenarios meet the population baseline gates in G05.
- `G04-A04`: Owner corpus removal does not alter packaged baseline behavior.

### G04-T03: Standardize evidence requirements

1. Define evidence classes: explicit setting, explicit correction, repeated
   behavior, accepted suggestion, inferred pattern, and imported data.
2. Give each adaptation kind a minimum evidence class and count.
3. Require distinct sessions or days where recurrence matters.
4. Prevent the model's own output from confirming its inference.
5. Prevent retrieved webpages and documents from teaching autobiographical facts
   or behavior.
6. Treat user correction as stronger than passive recurrence.

Acceptance:

- `G04-A05`: Self-confirmation and external-content poisoning suites cannot
  promote an adaptation.
- `G04-A06`: One unusual session cannot globally recalibrate stable behavior.

### G04-T04: Bound every learned value

1. Define safe min/max ranges and maximum change per update.
2. Separate user-adjustable comfort/style values from safety routing thresholds.
3. Require shadow evaluation before a learned threshold becomes active.
4. Add decay or revalidation for behavior that can change over time.
5. Store prior values for rollback.
6. Fall back to baseline after corruption or unsupported schema version.

Acceptance:

- `G04-A07`: Fuzzed adaptive inputs cannot move values outside declared bounds.
- `G04-A08`: Rollback restores deterministic prior decisions on frozen cases.

### G04-T05: Make adaptation observable

1. Add a "Learned about how I communicate" view separate from factual memory.
2. Show plain-language reason, evidence class, confidence, and last use.
3. Permit disable, edit where meaningful, reset by kind, and reset all.
4. Explain active personalization on disputed responses without exposing hidden
   reasoning.
5. Record user reversals as evidence that the mechanism needs recalibration.

Acceptance:

- `G04-A09`: Users in a usability test can find and reverse a learned preference.
- `G04-A10`: Reset removes the item from prompts and decisions immediately.

### G04-T06: Generalize identity prominence

1. Score prominence from explicit pinning, durability, currentness, confirmation,
   recurrence, and task relevance.
2. Avoid a fixed owner-derived list of important relations.
3. Require stronger evidence for inferred sensitive attributes.
4. Cap the always-rendered profile and explain displacement.
5. Allow users to pin or hide facts independently of confidence.

Acceptance:

- `G04-A11`: Central facts from the G02 profile matrix become available without
  code changes.
- `G04-A12`: Sensitive inferences cannot become prominent without confirmation.

### G04-T07: Calibrate retrieval per corpus state

1. Measure score distributions separately by collection, query intent, corpus
   size, and model version.
2. Replace owner-specific absolute thresholds where calibrated probabilities or
   rank-relative rules perform better.
3. Maintain floors that prevent empty or noisy contexts.
4. Evaluate fresh, sparse, mature, and very large profiles.
5. Recalibrate only after sufficient data and rerun the frozen benchmark.
6. Keep all previous parameters available for rollback.

Acceptance:

- `G04-A13`: Corpus growth does not cause systematic empty retrieval or context
  flooding in the scale suite.
- `G04-A14`: Personal calibration cannot reduce safety-critical retrieval below
  its release floor.

### G04-T08: Protect against feedback poisoning

1. Separate intentional user feedback from ordinary conversation.
2. Rate-limit learned changes and require distinct evidence.
3. Detect copied prompt-injection text, external quotations, role-play, and tests.
4. Never learn permissions, privacy-mode changes, or action authorization from
   natural-language implication alone.
5. Quarantine suspicious adaptive items for user review.
6. Add restore points before batch recalibration.

Acceptance:

- `G04-A15`: Adversarial documents and conversations cannot grant capability or
  disable safety behavior.
- `G04-A16`: Quarantined items never influence runtime decisions.

### G04-T09: Prevent dogfood overfitting

For every owner-reported wrong behavior, classify the correction before code:

1. Personal preference: update user configuration or adaptation.
2. Missing general mechanism: implement invariant plus diverse tests.
3. Population-default uncertainty: add evidence to the external evaluation queue.
4. Safety/security invariant: fix globally with red-team cases.
5. Exact data correction: repair memory and propagation, not language policy.

Global code changes require paraphrases and counterexamples that do not contain
the owner identity. See G11 for the incident workflow.

Acceptance:

- `G04-A17`: Every global behavior fix links to a non-owner invariant case.
- `G04-A18`: Owner-only preference fixes do not modify baseline fixtures.

### G04-T10: Version and migrate adaptive state

1. Add schema and policy versions to every adaptive store.
2. Distinguish invalid state from empty state.
3. Quarantine newer or corrupt state instead of overwriting it.
4. Migrate with a preview, backup, deterministic transformation, and rollback.
5. Revalidate learned items when their embedding or classification model changes.

Acceptance:

- `G04-A19`: Model replacement marks incompatible learned embeddings for safe
  rebuild rather than silently using them.
- `G04-A20`: Corrupt adaptation never destroys the last known-good copy.

## Evaluation dimensions

- Cold-start quality before any learning.
- Time to useful adaptation.
- False-learning rate.
- Poisoning resistance.
- Reversal and reset success.
- Stability across restarts and model upgrades.
- Worst-group impact of baseline and learned behavior.
- User understanding and perceived control.

## Exit gate

G04 is validated when all adaptation is registered, evidence-bounded,
user-visible, resettable, versioned, poisoning-tested, and evaluated separately
from the population baseline. No personal learning may mutate global policy.

