# Codex audit and implementation — 2026-09-13

Base: `c25d3ba`. Isolated checkout: `/tmp/daemon_codex_20260913`;
origin push URL `DISABLED`. No commits, pushes, live restart, or live-store access.
The user's checkpoint `~/daemon_checkpoints/CODEX_NO_TOUCH_20260913.md`
overrides overlapping ownership in older plans. Refresh Plan 2 status before
each application batch. Reserved files are read-only in every checkout;
proposed changes to them are delivered as unapplied patches.

## Worker policy

Use GPT-5.6 Luna for bounded investigation and implementation; the parent
reviews scope, evidence, edge cases and integration. Each worker receives
explicit inputs, output ownership, steps, failure modes and acceptance checks.
Ask the parent when stuck, when provider evidence conflicts, or when a required
change reaches a reserved file. Never invent provider support or update a debt
baseline to make a check pass. Only one test process at a time, capped below
the checkpoint's 4 GB ceiling. Use the installed Python 3.11.8 interpreter;
no dependency installation or paid model calls.

## A — Evidence audit (Luna)

1. Read the class catalog and relevant detection methods.
2. Read Sept 13 debug records, transcripts and JSONL telemetry outside `data/`.
   Stream inputs, keep file/line evidence and local-date bounds; exclude tests.
3. Separate reproduced defects, observed symptoms and improvement hypotheses.
   Map every defect to an existing mechanism or propose a new class.
4. Cross-check Plan 2 ownership and avoid duplicate implementation work.
5. Deliver `log_audit.md`, with prioritized behavior batches and checks.

## B — Model integration candidate (Luna)

1. Verify exact OpenRouter model IDs and supported request parameters using
   public provider metadata. User confirmed **DeepSeek-V4.1-Flash**; add a
   separate alias and retain the existing V4 Flash alias and caps row.
2. Trace aliases through both GUI selectors and all sync/async/tool call paths.
3. Add Fable 5.1 and Astra in an **unapplied patch**, preserve defaults, add
   verified context/capability data, and handle model-specific parameters.
   Astra's native Chat Completions and tool compatibility must not be inferred
   from a model-name pattern; verify the app's OpenRouter route separately.
4. Preserve DeepSeek reasoning separation and empty-visible-response recovery.
5. Test candidate code in memory with fake transports; no provider charges.
   Report uncertainty explicitly. Class mapping: BC-10/13/15/16/58.

## C — Repeatable offline diagnostics (Luna)

Observed inputs: 89 Sept 13 production rows in `logs/turn_records.jsonl`.
Rows carry wall/prepare/grounding timings, nested phase/task timings, model,
mode, routing decisions, and variable optional fields. They lack a consistent
turn ID, build version, token/cost accounting and complete outcome receipts.
Plan 2 A07 owns `utils/turn_telemetry.py`; producer edits require its owner.

1. Add `scripts/diagnostic_rollup.py`, standard library only, and focused
   `tests/unit/test_diagnostic_rollup.py`. Never import application modules.
2. Require an explicit JSONL path, date and IANA timezone. Resolve the input
   path and reject any `data` path component or symlink into `data` before
   opening. Reject directories; do not discover inputs recursively. No writes
   to input, stores, or a default output path: emit JSON only to stdout.
3. Stream JSONL with a per-record size bound. Count malformed, oversized,
   test, invalid-time and out-of-window records separately. Convert aware
   timestamps to the requested zone; interpret legacy naive timestamps in that
   zone and disclose their count. Treat absent values as unknown, never zero.
4. Emit fixed-schema aggregate counts, valid-value coverage, wall median/p90,
   phase/task timing summaries, model/mode groups, and top slow row numbers.
   Concurrent/nested timings must not be summed into wall latency. Use a
   bounded sample with an explicit truncation notice if a hard row cap is met.
5. Output no queries, plans, responses, retrieved text, session IDs, arbitrary
   row keys, exception bodies or free-text reasons. Use allowlisted labels for
   modes/models/phases/tasks and bucket unknown labels. Validate numeric fields
   against booleans, negative values, NaN/infinity and wrong shapes.
6. Track coverage for correlation ID, build version, finish/outcome reason,
   TTFT, tokens, cache/cost and provider retries. Missing diagnostics are
   measurement gaps, not evidence that a failure occurred.
7. Test malformed and adversarial rows, timezone boundaries, missing values,
   percentiles, test exclusion, bounds, privacy canaries, and forbidden paths.
   Use `systemd-run --user --scope -p MemoryMax=512M` with stdlib unittest
   discovery of this exact test file to avoid heavyweight root conftest imports.
8. Run on today's explicit telemetry file and retain only aggregate output in
   `diagnostic_summary.json`. Write a producer-field handoff for A07 with
   stable correlation IDs, monotonic per-stage times, typed outcomes, model
   usage/retries, sanitized exception categories, build/schema versions and
   positive/negative privacy tests. Map BC-20/47/69/70/72; do not claim closure
   until producers and CI guards are integrated.

## D — Audit-driven repairs and guard proposals

Parent selects safe fixes after reviewing A; no speculative core refactors.
Each batch specifies current behavior, intended behavior, ownership, class,
focused regression and limits. New classes need catalog + CI guard changes;
those reserved-file changes remain unapplied, named proposals for the guard
owner. Existing classes need incident evidence and any coverage gap recorded.

Selected safe repair: `utils/conversation_logger.py` rotation holds a regular
`threading.Lock` in `log_interaction`, then `_check_rotation` calls
`_write_session_header`, which acquires the same lock. Reproduce under a bounded
child-process timeout, distinguish this source defect from today's observed
latency, and remove recursive acquisition with explicit locked/unlocked helper
ownership. JSON rotation must derive its existence check from the actual format
instead of hardcoding `.txt`. Add focused tests for text/JSON rotation, multiple
parts, concurrency, and no-hang behavior in a new unit module. Propose new class
BC-79 (same-thread reacquisition of a nonreentrant lock) because BC-39 concerns
initialization coverage rather than reentry. Provide an unapplied catalog and
CI/hook/policy patch admitting a named guard that exercises the deployed logger
rotation contract. Keep closure scoped to this logger; do not claim a universal
deadlock detector. No guard harness test-count changes. Test before/after in
temporary directories; no production logging configuration or store imports.

Acceptance: review all diffs; targeted checks under memory caps; run the static
class scan when safe and report its real scope/result. Deliver an uncommitted
candidate and concrete owner handoff, clearly distinguishing implemented code,
unapplied patches and pending owner integration.
