# Fable handoff — final pre-push review, 2026-09-10

Owner request: review the completed working tree before the owner commits and
pushes. Keep this focused; the owner reported under 25 hourly credits remaining.
Codex has finished implementation and live verification. No commit or push was
made. Return PASS or concrete blockers, then a concise owner commit runner;
`git push` remains the owner's separate command.

## Current state and scope

- HEAD: `52a7fbb` (B4 forced-action/SPA fixes). The working tree contains the
  T01 test cleanup, B6 speed phase 1, and today's B6 follow-up together.
- Daemon was gracefully restarted for the live probes and left running with
  the updated working tree. It binds to the configured private-network address,
  not localhost. Check actual process/listeners before launching another copy;
  the existing instance's lock prevented an initial duplicate launch today.
- Older handoffs' clean-tree/restart-pending statements are historical. The
  latest implementation and measurements are in
  `docs/PLAN_20260909_audit_repairs.md`, sections **T01 backlog repair**, **B6
  phase 1**, and **B6 follow-up**.
- Review the entire dirty tree, including untracked files. There are deliberate
  deletions of `tests/test_meta_query.py` and `tests/test_real_mutations.py` from
  T01; their disposition/replacement coverage is recorded in the plan.
- New files that must not be omitted from the eventual commit:
  `scripts/latency_rollup.py`, `tests/unit/test_no_vacuous_assertions.py`,
  `tests/unit/test_sep09_speed_batch.py`, `test_sep09_speed_images.py`,
  `test_sep09_latency_metrics.py` (all three under `tests/unit`), both Sep-09
  audit-followup/speed handoffs, and this handoff.
- `CLAUDE.md`, `CLAUDE_CHANGELOG.md`, and `commit_message_20/21/22.txt` are
  intentionally gitignored local material. Do not claim they are committed
  documentation or force-add them. The tracked repair plan contains the record.

## What changed since your B6 referee pass

1. **Real grounding failures were mislabeled complete.** Two independent probes
   reproduced provider timeout/error → `grounding_status="complete"`.
   `verify_grounding` caught failures internally; the existing failure test
   replaced `_apply_grounding_check` and missed that boundary.
2. **Fix:** `core/grounding_check.verify_grounding(..., telemetry=None)` records
   timeout, provider failure, invalid verdict, or completion in the turn's own
   dictionary. Cancellation propagates. Valid verdicts that are deliberately
   demoted remain complete. `gui/handlers.py` preserves the lower-level status
   instead of overwriting it; unexpected wrapper errors also record failure.
   Answer correction and display/storage behavior are unchanged.
3. **Measurements:** `pre_prepare_elapsed_s` captures ingress → preparation;
   `grounding_verifier_elapsed_s` captures provider-call duration, including
   timeout/cancellation. Receipts/debug and the read-only latency roll-up expose
   these separately from answer wall time. Old rows show n/a.
4. **Tests:** 20 additional request cases use the real verifier and fake only
   provider I/O: enhanced/agentic × log-only/correct × timeout/provider error/
   malformed/valid/demoted. They check the single receipt, final debug status,
   timing fields, and unchanged displayed/stored answer. Added roll-up coverage.

Primary review files: `core/grounding_check.py`, `gui/handlers.py`,
`utils/turn_telemetry.py`, `scripts/latency_rollup.py`, and the two expanded B6
request/metrics test files. Also retain your review of the existing B6 changes
in `core/orchestrator.py`, `core/prompt/gatherer_knowledge.py`, `gui/launch.py`,
`knowledge/clip_manager.py`, `knowledge/visual_memory_pipeline.py`, and `main.py`.

## Verification evidence

- Before today's fix: 108 existing focused tests passed; both independent real
  provider-failure probes failed. The 20 new request/measurement cases then
  failed before implementation (incorrect/missing status or missing timing).
- After today's fix: **354 passed in 41.49 seconds**, no failures/skips/xfails in
  this selection; `ruff check .` and `git diff --check` clean.
- Earlier T01 result is **Fable's prior evidence**, not rerun by Codex today:
  307 passed, one skip, one strict xfail. The known xfail is
  `MemoryCoordinator.debug_memory_state()` on lazily unopened collections;
  it is outside this repair. Do not drop the marker to manufacture green.
- Sandbox caveat: a minimal `asyncio.to_thread` example and image tests hung
  under Codex's sandbox. The same tests passed outside it. Do not mistake that
  environment failure for a product regression or silently skip the tests.

Exact successful selection (Python 3.11; offline fakes):

```bash
DAEMON_TEST_MODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q \
  tests/unit/test_sep09_speed_batch.py \
  tests/unit/test_sep09_speed_images.py \
  tests/unit/test_sep09_latency_metrics.py \
  tests/unit/test_warmup_guards.py \
  tests/unit/test_grounding_check.py \
  tests/unit/test_grounding_wiring.py \
  tests/unit/test_grounding_log_only.py \
  tests/unit/test_sep01_grounding_misfires.py \
  tests/unit/test_handle_submit.py \
  tests/unit/test_sep04_attachment_turn.py \
  tests/unit/test_request_path_parity.py \
  tests/unit/test_sep08_agentic_answer_integrity.py \
  tests/unit/test_no_git_state_in_tests.py \
  tests/unit/test_ordered_slice_guard.py \
  tests/unit/test_budget_meters_rendered_sections.py \
  tests/unit/test_tool_wiring_parity.py \
  tests/unit/test_model_capability_wiring.py \
  tests/unit/test_no_vacuous_assertions.py
ruff check .
git diff --check
```

## Live results — already done, no need to repeat without a reason

| Probe | Evidence |
|---|---|
| First image after restart and CLIP warmup | Correct red square/blue circle description; first progress 0.047 s; ingress → prepare **0.161 s**; server wall 27.721 s. Previous observed cold-image stall before prepare: 36 s. |
| Read-only fetch of the official Python downloads page | Actual `agentic-search` route; server wall 34.145 s; SSE complete arrived with grounding **pending**; real verifier subsequently completed in **4.595 s**. |

These prove the specific stall removal and background ordering on real turns;
they are not a matched before/after study or evidence of a median speedup.
Context preparation remains slow (25.389 s image / 22.595 s agentic). Retrieval
optimization is phase 2, outside this final review. The image's longest task was
memories at 19.294 s; concurrent task times must not be summed.

Durable receipts: `logs/turn_records.jsonl`, local timestamps
`2026-09-10T12:52:12-05:00` and `2026-09-10T12:59:18-05:00`.
Local inspection artifacts: `/tmp/daemon_b6_review/live_timings.json`,
`image_events.json`, `agentic_events.json`, and `live_probe.py`. Event files can
contain full debug context: leave them local. The two synthetic turns and shape
upload went through ordinary application persistence; no calendar mutation was
performed and no cleanup/deletion is requested.

## Final review / owner packaging

1. Check outcome propagation, per-turn isolation, callback ordering/one receipt,
   cancellation/draining, and the inline corrective path. In particular, retain
   the real-verifier tests; wrapper-only failure fakes were the original gap.
2. Review all untracked files and the T01 removals against their recorded scope.
   Do not expand into phase 2 or unrelated audit follow-ups.
3. Inspect the final commit contents for privacy and missing files. Exclude
   live artifacts and stores. Rewrite the commit description around the actual
   staged scope: messages 20/21/22 are inputs, not three independent ready-made
   commits. The shared plan and source hunks are already combined; follow the
   repository's no-hunk-surgery rule. Message 21's subject is too long, and its
   docs/test narrative predates today's follow-up.
4. **Before running the changed non-unit tests or the eventual pre-push hook,
   gracefully stop Daemon and wait for shutdown completion.** The dirty tree
   includes many non-unit T01 tests; `hooks/pre-push` selects them even without
   `PREPUSH_FULL=1`. Its 6 GiB memory cap does not replace the repo's requirement
   to run that batch with Daemon down. Use the documented memory-capped workflow
   and durable output for any independent non-unit rerun.
5. The pre-push hook requires a clean committed tracked tree and no untracked
   Python files. Do not invoke it on today's dirty tree and call that a code
   failure. After your PASS, leave an owner-run commit command with the final
   reviewed scope/message; the owner runs `git push` separately. No bypasses,
   commits, or pushes have been performed by Codex.

Requested reply: PASS with any non-blocking notes, or blockers with exact source
locations/reproductions; identify the final commit scope and which checks remain
for the owner before push.
