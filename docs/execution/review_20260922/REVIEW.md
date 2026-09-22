# 2026-09-22 combined review: turn-audit guard fixes + isolated-runtime smoke

Clone `~/daemon_exec/reviewfix_0922`, branch `fix/sep22-reviewed-guards-smoke`,
base `2d563f0`. This record states what landed and what was verified; it is
not a claim that anything beyond the listed checks ran.

## Scope

Two lanes from 2026-09-22 were merged and adversarially reviewed (Codex
review + cheap subagents, then Claude Fable integration and referee):

1. **Lane GF — turn-audit guard fixes** (five deterministic defects from three
   live turns): BC-91 machinery text, the calendar state-claim backstop's false
   correction (BC-50), the memory top-up re-adding a deduplicated paste (BC-20),
   an STM cue routing a status update to factual recall (BC-51/52), the missing
   weekday in `[GOOGLE CALENDAR]`, and graph function-word nodes (BC-55).
2. **Lane 1 Subgoal C — isolated-runtime two-boot smoke**
   (`tests/smoke/`, Plan A `docs/PLAN_20260921_lane1_career_fair_readiness.md`).

## Review findings and what changed (final state)

- **Calendar corroboration (BC-50/58).** The GF anchor design let a request or
  a negated mention ("can you add an appointment", "I don't have an
  appointment") corroborate a fabricated calendar-state claim. Codex's first
  replacement fixed that but silently REGRESSED the live shape the lane was
  written for (a multi-event status message whose clock times belong to other
  clauses returned "not corroborated"; the private live fixture had been
  deleted, so no test caught it). Final `gui/handlers._calendar_claim_user_corroborated`:
  anchor on the claim's own calendar noun; the span mention must be an
  assertion (not negated, not in a question/request/command sentence, all via
  deployed `query_checker` shapes); a persisted-state claim needs the user's
  sentence to name the calendar; explicit weekday/date/clock time must agree
  where both sides state them, with clock times scoped to the anchor's own
  clause. Synthetic live-shaped case pinned in
  `tests/unit/test_sep22_calendar_backstop_user_words.py`.
- **Sequential revisions (BC-45/91).** `_apply_delivery_revisions` composes
  grounding → personal-claim on the revised body; the action-guard suffix is
  reattached once. `tests/unit/test_sep22_delivery_revision_sequence.py`.
- **Notice stripping (BC-91).** `strip_delivery_notices` removes only
  REGISTERED notice families at the tail; an authored blockquote before them
  survives; a registered notice quoted mid-answer is content.
- **Emitter registry (new, BC-58 closure).** Codex's registry was a second copy
  of the emitters' opening clauses. Final: the `NOTICE_*` constants in
  `utils/read_time_markers.py` ARE the registry, and every emitter composes its
  notice through `delivery_notice(NOTICE_*, detail)` (seven sites:
  `core/action_claim_guard.py` ×2, `utils/web_evidence_receipt.py` ×2,
  `gui/handlers.py` ×3). Notice bytes are unchanged. An unregistered opening
  raises; DM-38 rule 4 rejects a `> ⚠️` literal outside the leaf.
- **DM-38 (gate, contract v2).** Adapted to the pipeline helper (the handoff's
  expected integration conflict) and extended with the emitter-registry rule;
  13 mutation controls; policy 2026-09-22.1; harness 326. Response record:
  `docs/execution/generalization/class_guard_responses/CGR-20260922-001.md`.
  Codex's replacement of `tests/bug_class_guards/test_scanners.py` was in fact
  an append-only block; applying it as a file would have deleted the 52
  existing scanner controls — it was appended instead.
- **DM-39** stays an ordinary unit-lane guard (`test_sep22_graph_function_words.py`),
  not a stdlib scanner. Graph filtering keeps named-entity type and casing
  (a stopword can be a name); untyped lowercase homographs remain ambiguous.
- **Privacy (BC-61).** Graph and calendar fixtures are synthetic. Every changed
  file was scanned against the local term list: no hits in added content.
- **Pre-push contract (BC-37/83).** Fixtures no longer inherit
  `DAEMON_LIVE_REPO_ROOT` from the owner shell (27 passed with the live Daemon up).
- **Smoke (BC-63/62/83).** Runs the production ASGI lifespan (unrelated
  startup workers stubbed), real backup required after both boots; CI provisions
  the four model artifacts from the deployed constructors, runs the smoke
  explicitly with `DAEMON_SMOKE_REQUIRED=1`, and a JUnit verifier rejects an
  empty/skipped/wrong-case receipt. It proves isolated runtime persistence
  from a checkout, not installation, Docker or Windows packaging.
- **Formatter accepted debt.** `_format_session_header` AST verified unchanged
  at base vs final (independent check in this review); the two ledger records
  are re-pinned to the final file hash with that rationale.

## Verification (this clone, final tree)

See the commit body / owner handoff for the exact counts of: focused unit
suites, repo-wide guards, bug-class scan + harness + `verify-receipts`, ruff,
full unit lane, and the two-boot smoke (Daemon down). Anything not listed
there did not run.
