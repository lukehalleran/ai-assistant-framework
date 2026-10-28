# Personal claim support implementation — 2026-09-15

User authorized delegation of the audited repair to two inexpensive agents.
Scope: the personal-event audit from this session, not the separate September
13 generalization Plan 2. Parent integrates/reviews; no agent commits, pushes,
restarts Daemon, mutates stores, or uses paid APIs for tests.

## Contracts and ownership

- Agent A: `core/personal_claim_check.py` and
  `tests/unit/test_personal_claim_check.py` only. Build bounded role-preserved
  evidence, a semantic claim auditor with exact source-reference validation,
  and deterministic omission of unsupported claim sentences for opt-in mode.
- Agent B: `gui/handlers.py`, `config/app_config.py`, `config/config.yaml`,
  and `tests/unit/test_personal_claim_delivery.py` only. Capture independent
  personal-claim mode per turn; check enhanced and agentic replies regardless
  of tone/planning; default log-only; buffer opt-in correction mode; transport
  receipts through debug, telemetry and storage provenance. Repair factual
  grounding's skipped-versus-checked receipt while preserving existing modes.
- Parent: integration review, memory/provenance consumers, docs, new adversarial
  tests if needed, and sequential test runs. No worker runs pytest concurrently.

Shared API (Agent A implements, Agent B imports lazily):

```python
build_personal_evidence(query, context, history=(), *, max_chars=12000) -> list[dict]
async audit_personal_claims(response, evidence, model_manager, *, model_name,
                           timeout_s=5.0, max_tokens=900) -> PersonalClaimResult
omit_unsupported_claims(response, result) -> str
```

`PersonalClaimResult` exposes `status` (checked/unavailable/failed/skipped),
`reason` (constant, no private text), `claims` (list of claim dictionaries),
`elapsed_s`, and `receipt()` (JSON-safe counts, source IDs, status/reason;
never raw reply, query, excerpts, or chain of thought). Each claim has exact
`text`, `status` (supported/contradicted/insufficient), `kind`, and `evidence`
(list of `{source_id, quote}`). A supported personal completion must reference
user-origin evidence; assistant discussion may support only a discussion
claim. No tool text becomes evidence that the user performed an external task.
Model JSON schema and parsing details belong to Agent A, not duplicated in B.

Evidence entries: `{source_id, role, timestamp, text}`. Preserve roles and
original source identity; include current query plus structured recent and
relevant conversation. Derived summaries/profile/narrative cannot establish
completion. Deduplicate with roles intact. Bounded selection must retain the
current query and latest correction. Unknown timestamp remains unknown.

Default setting: `personal_claim_check.enabled: true`, `mode: log_only`,
model falls back to the existing review model. Opt-in `correct` omits exact
unsupported/contradicted sentence spans; it never invents a negation or an
alternate event. All-omitted fallback is a neutral statement of insufficient
context. Timeout/malformed output fail open with an explicit receipt. This
default is an evaluation rollout, not a claim the live confabulation is fixed.

## Acceptance and contingencies

1. Exact audit query/reply and meaningful prior user AND assistant context
   become clean and wrapped fixtures calling deployed functions. Scripted
   provider output tests transport/validation, not model accuracy.
2. Include plans, suggestions, real completion, negation, quotes, partial work,
   cancellation, different object/day, corrections, novel action vocabulary,
   external upload versus local attachment, and no-claims support language.
3. Supported text survives; unsupported claims are absent from BOTH delivery
   and storage in correction mode. Log-only text is unchanged and receipt
   reaches telemetry/provenance. Failure is not recorded as verified.
4. Enhanced/agentic dispatch buffer when either checker needs correction;
   preserve cancellation/error behavior and no concurrent config-mode drift.
5. Parent checks memory consumers so stored assistant output cannot become
   user-authored evidence; preserve support status on retrieval where present.
6. No broad model-accuracy claims from mocks. Before production correction
   rollout, measure false positives on a labeled set and run a live canary.

Escalate contract ambiguity, source-schema gaps, unrelated test failures or
needed edits outside ownership. No ad hoc phrase regexes. Parent resolves
shared-file changes and missing evidence conservatively, then records limits.

## Results (2026-09-15, Fable pickup)

The Codex session hit its usage limit after Agent A, the handler wiring and the
provenance transport were in the tree but before Agent B's
`tests/unit/test_personal_claim_delivery.py` existed and before the parent's
integration review finished. Picked up from the working tree:

- Integration defects fixed: the turn row froze at `personal_claim_status:
  pending` (telemetry writer never waited on the claim task; the hook copied
  only `grounding_`/`storage_` keys); handler-level receipt reasons bypassed the
  shared whitelist; `grounding_status="checked"` duplicated the existing
  `complete` contract (17 red tests); the new check added a second model call
  to the grounding-isolation fixtures (12 red); the YAML section had no schema
  model; six leaf-module imports hoisted; one checker test raised StopIteration.
- `tests/unit/test_personal_claim_delivery.py` written (16 tests, acceptance
  items 1, 3, 4 and the failure receipts); acceptance 5 is covered by
  `test_personal_claim_provenance.py` and the checker's role tests.
- 786 tests green across the dependent suites; class-guard scan OK after
  re-reviewing seven `accepted_debt` rows whose files changed outside the
  anchored symbols.
- Open: acceptance 6 (labeled-set precision + live canary before `correct`),
  post-restart live probe, owner commit. Nothing committed, applied or restarted.

### Live probe (committed d0232a5, relaunched 18:27)

The turn row carried a final status (no `pending` freeze). The checker itself
returned `failed/invalid_json`; an offline replay validated and caught the
repeated "resume uploaded" claim, so the all-or-nothing validator was the
defect (BC-84). Second commit: per-claim validation, `invalid_json` vs
`invalid_verdict`, dropped/demoted counts on the receipt. Replay through the
deployed audit afterwards: checked, 6 candidates, 1 supported, 5 insufficient,
1 demotion. Still open: acceptance 6 and a second live probe after the relaunch.
