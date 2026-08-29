# G11: Dogfooding, Incidents, and Claude Fable Development

## Objective

Turn daily owner use into a disciplined defect-discovery and regression process
without confusing one user's preferences with population requirements or sending
private data to a hosted development model.

Claude Fable is a development instrument, not an authority, production runtime,
external evaluator, or substitute for users.

## Core loop

```text
daily use
  -> owner marks behavior wrong
  -> local evidence snapshot
  -> severity and generalization classification
  -> minimal reproduction plus counterexample
  -> sanitized development bundle
  -> Fable-assisted analysis/patch
  -> human review and focused tests
  -> broader invariant and regression suites
  -> owner-canary deployment
  -> observed closure or rollback
  -> population-evaluation queue when applicable
```

## Incident classes

### Severity

| Severity | Meaning | Response |
|---|---|---|
| P0 | Privacy breach, destructive loss, unauthorized action | Stop affected use; preserve evidence; immediate fix |
| P1 | Dangerous safety error, repeatable security failure, major truth corruption | Disable/rollback path; fix before continued use |
| P2 | Material wrong answer/routing/memory behavior | Prioritized incident and regression |
| P3 | Style, friction, preference mismatch | Personal setting/adaptation or batched UX work |

### Generalization disposition

| Code | Meaning | Correct fix location |
|---|---|---|
| `PERSONAL_PREF` | Correct behavior for this user differs from baseline | User setting or adaptive state |
| `DATA_ERROR` | Stored fact/source/derivative is wrong | Correction and propagation path |
| `GENERAL_MECHANISM` | Invariant is missing or broken | Global code plus diverse tests |
| `POPULATION_UNKNOWN` | Correct default is not known from owner evidence | Evaluation queue; cautious fallback |
| `SAFETY_SECURITY` | Authorization, privacy, data-loss, or high-risk invariant | Global hard control and red team |
| `MODEL_LIMIT` | Qualified model cannot meet role reliably | Role/model gate, prompt, or capability limit |
| `PRODUCT_SCOPE` | Request lies outside contract | Clear limitation or contract change process |

No patch begins until both classifications are recorded, except immediate P0
containment.

## Step-by-step plan

### G11-T01: Add a local "Mark response wrong" workflow

1. Give every turn a stable local response ID.
2. Add a feedback action that never uploads data automatically.
3. Ask for optional category, expected behavior, and free-text note.
4. Snapshot model/runtime, config, prompt-section hashes, retrieved IDs, tool trace,
   action receipts, and relevant state versions.
5. Store full sensitive evidence only in the encrypted local incident area.
6. Permit the owner to exclude raw content while retaining metadata.

Acceptance:

- `G11-A01`: Feedback works offline and creates no network request.
- `G11-A02`: Snapshot identifies the exact model/config/code context needed for
  reproduction.

### G11-T02: Create the incident ledger

Each incident records:

```text
incident_id
opened_at
severity
generalization_disposition
response_id
short_sanitized_summary
affected_requirement_ids
privacy_class
reproduction_status
root_cause
containment
fix_commit
tests
evaluation_impact
canary_result
closed_at
revalidation_triggers
```

1. Store the private ledger outside the public source tree.
2. Commit only sanitized postmortems or case definitions.
3. Link P0/P1 incidents to threat-model updates.
4. Link population unknowns to G05 coverage cells.
5. Track recurrence separately from reopening.

Acceptance:

- `G11-A03`: Every code change labeled as dogfood repair links to an incident.
- `G11-A04`: Public artifacts contain no raw private incident data.

### G11-T03: Contain before diagnosing

1. Stop or disable the affected capability for P0/P1.
2. Preserve logs, stores, versions, and artifact hashes without repeatedly
   triggering the defect.
3. Back up before repairing data.
4. Define what remains safe to use.
5. Do not let Fable modify live data or run destructive commands.

Acceptance:

- `G11-A05`: P0/P1 checklist includes explicit containment and data-preservation
  evidence.
- `G11-A06`: Repair work occurs against a snapshot or synthetic reproduction.

### G11-T04: Produce a minimal reproduction

1. Identify the smallest input, state, and sequence that reproduces the behavior.
2. Separate required historical context from irrelevant personal context.
3. Replace private entities with fictional equivalents while preserving the
   causal structure.
4. Record the pre-fix failing assertion.
5. Add at least one non-triggering counterexample.
6. For sequence failures, preserve ordering, restart, and timing conditions.

Acceptance:

- `G11-A07`: The reproduction fails before the fix and passes after it.
- `G11-A08`: The counterexample prevents an overbroad fix.

### G11-T05: Decide personal versus global response

Ask in order:

1. Is the system factually or procedurally wrong for any user in this state?
2. Is a hard privacy, action, data-loss, or safety invariant violated?
3. Is this only the owner's desired style or cadence?
4. Is the assumed correct default supported by anyone other than the owner?
5. Could a global fix harm users with different dialect, schedule, culture,
   disability, household, or communication style?

Disposition rules:

- Personal preference changes user state, not source defaults.
- General mechanism changes require neutral paraphrases and counterexamples.
- Population unknowns use clarify/abstain/configure behavior and enter G05.
- Safety/security defects receive global deterministic controls where possible.

Acceptance:

- `G11-A09`: Every global incident fix includes a written invariant independent
  of the owner identity.
- `G11-A10`: Personal-preference incidents do not alter population baselines.

### G11-T06: Build a sanitized Fable bundle

1. Include only the minimum source files, synthetic fixture, failing command,
   expected invariant, and relevant sanitized trace.
2. Exclude live data roots, raw conversations, backups, exports, tokens, exact
   personal paths, and unrelated logs.
3. Run private-canary, secret, path, and known-entity scans.
4. Review the exact bundle manually before hosted transmission.
5. Store bundle hash and scan result locally.
6. Never permit Fable to browse the live private data root.

Acceptance:

- `G11-A11`: Seeded private values block bundle approval.
- `G11-A12`: Fable can reproduce the defect using only synthetic material.

### G11-T07: Use Fable under a constrained repair contract

The development request must state:

1. The invariant and failure, not only "make this case pass."
2. Files and subsystem boundary in scope.
3. User changes that must be preserved.
4. Forbidden destructive and private-data operations.
5. Required focused and counterexample tests.
6. Required broader regression suites.
7. No live deployment, merge, or data repair authority.

Review Fable's proposed root cause independently. Reject patches that merely add
the exact phrase, identity, date, medication, institution, or path from the
incident unless the fix is explicitly personal data/configuration.

Acceptance:

- `G11-A13`: Patch review checks scope, invariant, privacy, and overfit explicitly.
- `G11-A14`: No agent-authored patch reaches owner-canary without human diff
  review.

### G11-T08: Verify the repair

1. Run the minimal failing test.
2. Run counterexamples and nearby subsystem tests.
3. Run owner frozen replays affected by the change.
4. Run G02 neutral-profile and G09 language variants where routing is involved.
5. Run retrieval/evaluation ledgers before changing scoring or prompts.
6. Run privacy/security suites for data, tool, or network changes.
7. Compile/type/lint and package smoke test according to risk.
8. Record exact commands and artifact versions.

Acceptance:

- `G11-A15`: Required suites are derived from affected requirement IDs.
- `G11-A16`: A failed broader gate blocks canary even when the exact case passes.

### G11-T09: Deploy through owner-canary

1. Build a versioned canary artifact rather than editing the live installation.
2. Back up current data and preserve the prior executable/model pack.
3. Enable the change behind a narrow flag when risk warrants.
4. Re-run the original workflow naturally rather than forcing only the fixture.
5. Observe nearby behavior for a declared window.
6. Roll back on recurrence, new regression, or unexplained state mutation.

Acceptance:

- `G11-A17`: Canary and rollback versions are recorded in the incident.
- `G11-A18`: Closure requires observed owner use, not tests alone.

### G11-T10: Close with generalized evidence

1. Record confirmed root cause and why the fix belongs at its chosen layer.
2. Record the reproduction, counterexample, tests, and canary window.
3. Update architecture/threat/generalization docs where the mental model changed.
4. Add a G05 item if population evidence is still missing.
5. Define revalidation triggers such as model, prompt, threshold, or store change.
6. Sanitize any public postmortem again before commit.

Acceptance:

- `G11-A19`: Closed incidents have no missing containment, test, or canary field.
- `G11-A20`: Population-unknown closure never becomes a public universal claim.

### G11-T11: Review incident patterns monthly

1. Group incidents by root cause rather than symptom.
2. Identify repeated dead wiring, duplicated policy, owner phrase patches, model
   limitations, and missing observability.
3. Replace clusters with architectural fixes where evidence supports them.
4. Remove obsolete exact-case rules after generalized mechanisms prove coverage.
5. Sample ordinary successful turns to detect silent degradation and reporting
   bias.
6. Publish a private monthly quality summary with no raw personal content.

Acceptance:

- `G11-A21`: Repeated root-cause class creates a consolidation work item rather
  than another isolated patch.
- `G11-A22`: Exact owner rules have an owner, rationale, and retirement review.

### G11-T12: Transition from owner-only to external evidence

1. Keep owner dogfood cases as a permanent longitudinal suite.
2. Do not merge them silently into population metrics.
3. When G02/G03/G07/G08 gates pass, begin trusted external alpha under G05.
4. Compare whether owner-discovered defect classes occur externally.
5. Adjust population baselines only from the appropriate evidence level.
6. Retain per-user adaptation for genuine individual differences.

Acceptance:

- `G11-A23`: Reports label owner, synthetic, external-development, and held-out
  evidence separately.
- `G11-A24`: External feedback follows the same privacy and incident controls.

## Daily workflow

1. Use the current owner-canary normally.
2. Mark wrong behavior immediately when practical.
3. Continue use only if severity permits.
4. Triage P0/P1 immediately; queue P2/P3.
5. Process a bounded incident batch rather than changing code during every chat.
6. Release fixes through a versioned canary.
7. Observe and close only after natural reuse.

This preserves daily discovery without turning the production instance into an
unreviewed live-edit environment.

## Metrics

- Incidents by severity and disposition.
- Time to containment and verified repair.
- Recurrence rate.
- Percentage with minimal reproduction and counterexample.
- Percentage fixed as personal configuration versus global code.
- Broader regression rate caused by fixes.
- Private-bundle scan failures.
- Owner-canary rollback rate.
- Exact owner-rule count and retirement age.
- Population-unknown backlog.

## Exit gate

G11 is validated when every dogfood failure enters a private local ledger,
severity and generalization are decided before repair, Fable sees only sanitized
synthetic bundles, patches receive human and regression review, deployment uses a
versioned canary with rollback, and owner evidence remains distinct from
population validation.

