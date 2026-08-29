# G05: Population Evaluation

## Objective

Replace owner-only confidence with reproducible evidence across external users,
communication patterns, risk classes, corpus maturity, and supported local model
packs.

The evaluation program determines what Daemon may claim. It does not attempt to
prove universal correctness.

## Evidence levels

| Level | Evidence | Permitted claim |
|---|---|---|
| E0 | Owner anecdote | Incident discovered; no general claim |
| E1 | Owner frozen replay | Owner regression fixed |
| E2 | Synthetic diverse matrix | Mechanism handles represented cases |
| E3 | Consented external labeled set | Development-set population evidence |
| E4 | Held-out external users | Generalization evidence for sampled population |
| E5 | Independent review or replication | Strongest release evidence |

Stable population claims require at least E4. High-risk claims should seek E5.

## Evaluation matrix

### Communication variation

- Regional, ethnic, generational, occupational, and social dialects.
- Slang, profanity, abbreviations, speech-like fragments, typos, and punctuation
  variation.
- Code-switching while the primary interaction remains American English.
- Terse and highly verbose styles.
- Literal, indirect, ironic, humorous, and emotionally expressive language.
- Cognitive, learning, and assistive-technology-influenced communication.

### Lifecycle variation

- Empty profile and first five turns.
- Sparse profile.
- Mature profile with months of history.
- Very large corpus.
- Restart during an active topic.
- Changed identity and corrected facts.
- Disabled memory, session-only memory, and selective storage.
- Offline and connected modes.

### Task variation

- Casual conversation and direct questions.
- Personal factual and temporal recall.
- Correction, contradiction, and uncertainty.
- Technical and project work.
- Emotional support and high-risk language.
- Web and academic research.
- Files, documents, images, and notes.
- Tool proposals, approval, rejection, failure, and execution receipt.
- Synthesis and proactive behavior where enabled.

## Step-by-step plan

### G05-T01: Create the evaluation registry

1. Give every evaluation case a stable ID, version, task, risk class, evidence
   level, population tags, privacy status, and expected behavior.
2. Separate owner cases, synthetic cases, external development data, and held-out
   data physically and logically.
3. Record model, runtime, prompt, configuration, corpus, and code versions.
4. Prevent development tools from reading held-out expected answers.
5. Generate coverage reports by matrix cell.

Acceptance:

- `G05-A01`: Every reported metric can be reproduced from a manifest.
- `G05-A02`: Training/development and held-out identities never overlap.

### G05-T02: Define task-specific metrics

Do not compress the system into one score.

1. Retrieval: recall, reciprocal rank, nDCG where multiple answers are valid,
   freshness, and abstention.
2. Facts: extraction precision/recall, polarity, entity scope, currentness,
   correction propagation, and unsupported inference.
3. Intent/tone: macro metrics, calibration, confusion matrix, and false escalation
   by subgroup.
4. Tools: correct tool, correct arguments, unnecessary-tool rate, permission
   compliance, and execution-claim accuracy.
5. Responses: factual support, instruction satisfaction, uncertainty, relevance,
   style fit, and harmful behavior.
6. Privacy: unexpected egress, sensitive-field disclosure, and deletion residue.
7. Performance: first-token latency, completion latency, memory use, energy proxy,
   cancellation, and crash rate by hardware tier.

Acceptance:

- `G05-A03`: Every release requirement maps to at least one metric or explicit
  manual evaluation.
- `G05-A04`: Aggregate metrics are always accompanied by sample count and
  uncertainty.

### G05-T03: Establish risk-based release thresholds

1. Classify failures as privacy/data loss, unauthorized action, high-risk safety,
   factual/retrieval, usability/accessibility, or style.
2. Set zero-tolerance release-suite conditions for cross-profile leakage,
   unauthorized action, silent hosted fallback, and unrecoverable data loss.
3. Set predeclared statistical thresholds for probabilistic quality tasks.
4. Include worst-group floors and acceptable aggregate-to-group gaps.
5. Require human review for high-risk ambiguous cases.
6. Version thresholds and forbid changing them after seeing held-out results
   without declaring a new evaluation round.

Acceptance:

- `G05-A05`: Release automation fails on any hard-invariant violation.
- `G05-A06`: Threshold changes create a new signed evaluation manifest.

### G05-T04: Build synthetic breadth without mistaking it for users

1. Expand every owner incident into neutral paraphrases and counterexamples.
2. Generate systematic perturbations of spelling, punctuation, verbosity,
   dialect features, names, pronouns, and context ordering.
3. Have humans review generated cases for plausibility and stereotype leakage.
4. Keep synthetic labels separate from human-observed labels.
5. Use mutation tests to prove a case fails when the intended invariant is broken.

Acceptance:

- `G05-A07`: Synthetic generation cannot directly edit the held-out set.
- `G05-A08`: Every P0/P1 incident gains both a reproduction and a non-triggering
  counterexample.

### G05-T05: Recruit external alpha cohorts

1. Begin only after owner-neutrality, data isolation, installer, privacy controls,
   and delete/export paths are functional.
2. Recruit 5 trusted clean-install users for installation and severe-defect
   discovery.
3. Expand to roughly 20 deliberately varied alpha users.
4. Use explicit consent and collect the minimum evidence needed.
5. Keep telemetry off by default; use user-triggered, previewed diagnostic export.
6. Compensate testers when asking for structured or sensitive evaluation work.

Acceptance:

- `G05-A09`: No tester requires repository access or manual source edits.
- `G05-A10`: A tester can withdraw and remove all locally held study material.

### G05-T06: Build a consented evaluation corpus

1. Define the exact collection purpose and retention before recruiting.
2. Prefer on-device labeling and export of derived labels over raw transcripts.
3. When text is essential, let the contributor review and redact it.
4. Store identity/contact consent records separately from evaluation content.
5. Remove direct identifiers and scan for residual private entities.
6. Track license/consent and permitted uses per item.
7. Never commit private external data to the public repository.

Acceptance:

- `G05-A11`: Every external item has valid consent metadata and retention policy.
- `G05-A12`: Revocation removes the item from active, cached, and backup-derived
  evaluation stores according to the declared process.

### G05-T07: Run stratified human evaluation

1. Define rubrics before generating candidate responses.
2. Blind raters to model and variant where practical.
3. Randomize presentation order.
4. Measure agreement and adjudicate unclear labels.
5. Report outcomes by relevant communication/user subgroup with sufficient
   sample sizes.
6. Avoid treating protected attributes as causal explanations for differences.
7. Use community-informed reviewers for dialect or disability-specific claims.

Acceptance:

- `G05-A13`: Human-evaluation reports include rubric, sample, agreement,
  exclusions, and limitations.
- `G05-A14`: No group-specific claim is based only on synthetic model judgments.

### G05-T08: Use counterfactual fairness probes

1. Construct paired prompts that preserve task and meaning while varying name,
   pronouns, dialect markers, location, or demographic cues.
2. Compare tool routing, safety escalation, refusal, helpfulness, and memory
   prominence.
3. Review pairs for semantic equivalence with knowledgeable humans.
4. Investigate systematic disparities, not isolated score differences.
5. Add confirmed failures to the frozen release suite.

Acceptance:

- `G05-A15`: The release report includes counterfactual results for every
  high-impact routing subsystem.
- `G05-A16`: Confirmed disparity fixes do not erase legitimate semantic
  differences.

### G05-T09: Evaluate longitudinal behavior

1. Define 30-, 60-, and 90-day checkpoints.
2. Measure correction retention, stale-fact behavior, adaptive drift, retrieval
   quality, storage growth, latency, and unwanted recurrence.
3. Test upgrades and model replacements against existing user state.
4. Sample ordinary successful turns, not only reported failures.
5. Track repeated annoyance and loss of trust as outcomes.

Acceptance:

- `G05-A17`: Stable release has at least one completed 60-90 day external cohort.
- `G05-A18`: No model upgrade ships without mature-profile replay.

### G05-T10: Qualify each local model pack

1. Run the same frozen task suite for every model, quantization, context setting,
   template, and runtime build.
2. Prevent a model qualified for one role from being assumed qualified for all.
3. Compare against the previous local release and a declared reference, which may
   be Fable during development using approved data.
4. Report quality, latency, memory, and failure-mode tradeoffs.
5. Requalify after runtime, tokenizer, template, or quantization changes.

Acceptance:

- `G05-A19`: The model manifest links to a passing role-specific report.
- `G05-A20`: Hardware-tier selection never chooses an unqualified combination.

### G05-T11: Advance to public beta evidence

1. Expand to 50-100 diverse beta users after alpha gates pass.
2. For broader subgroup claims, recruit adequately powered samples, likely
   hundreds overall rather than relying on a token participant per group.
3. Maintain a held-out release set not used for incident tuning.
4. Publish aggregate and worst-group outcomes, known gaps, and excluded scopes.
5. Commission independent review for high-risk and privacy claims.

Acceptance:

- `G05-A21`: Public claims use E4 or E5 evidence and identify the sampled scope.
- `G05-A22`: Owner-only results remain labeled and are never merged invisibly
  into population metrics.

## Required report sections

- Product contract and supported scope.
- Application, model, runtime, and corpus versions.
- Sampling and consent procedure.
- Task and subgroup coverage.
- Metric definitions and preregistered thresholds.
- Aggregate and worst-group outcomes with uncertainty.
- Hard-invariant violations.
- Human-rater procedure and agreement.
- Known limitations and missing cells.
- Regression comparison and release decision.

## Exit gate

G05 is validated for 1.0 when the local runtime passes the frozen owner and
synthetic suites, external held-out users meet predeclared task and subgroup
gates, a longitudinal cohort completes, high-risk behavior receives independent
review, and the public report states limitations without universal claims.

