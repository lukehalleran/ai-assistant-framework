# Codex handoff — 2026-09-13

Work is uncommitted in `/tmp/daemon_codex_20260913`, based on `c25d3ba`, with
origin pushing disabled. The running checkout, live stores, Claude's clone,
tmux session and reserved files were left unchanged. GPT-5.6 Luna workers
performed the audit and implementation; the parent reviewed and tested them.

## Today's evidence

The [audit](log_audit.md) covers three debug logs, three conversation logs,
turn telemetry and action/curation audit streams, all outside `data/`.
There are 89 production turns; 87 measured wall times have median 21.062 s,
p90 42.4536 s and maximum 138.667 s. There are 54 semantic-search timeout
warnings, inconsistent calendar-action status reports, and one provider
credit-exhaustion error. These observations are separated from proposed causes.
Personal conversation text is not copied into the reports.

Action-history receipts and retrieval outcome instrumentation need the Plan 2
owner's integration. Existing classes BC-45/48/72 describe the action symptoms;
BC-41/47 are investigation leads for retrieval, not established causes.

## Implemented in the isolated clone

- `scripts/diagnostic_rollup.py`: explicit-input, bounded, standard-library
  JSONL analysis with timezone selection, timing summaries, measurement
  coverage, test exclusion and sanitized output. [Today's aggregate](diagnostic_summary.json).
- `utils/conversation_logger.py`: fixes same-thread lock reacquisition during
  rotation and JSONL part-number reuse. This source defect was reproduced;
  it has not been established as a cause of today's observed slow turns.
- Focused regressions for both changes. Ten diagnostics tests and one bounded
  rotation test passed under 512 MB scopes. Lint and the full static class scan
  passed; [scan receipt](guard_receipt.json). The full application test suite
  and the 311-test guard harness were not run.

The scanner passing is scoped evidence: its baseline has 152 reviewed
occurrences and it reports 67 catalog classes without scanner coverage.
Separate behavioral guards provide additional coverage. No baseline or
disposition was edited to obtain a passing scan.

## Concrete patches awaiting ownership release

[Model integration](model_integration.md) and [model patch](model_integration.patch)
add `deepseek-v4.1-flash`, `claude-fable-5.1` / `fable-5.1`, and `gpt-6-astra`
through the existing OpenRouter transport. Existing V4 Flash and active defaults
are preserved. GUI/API selectors read the manager registry automatically.
The patch handles per-model parameters, vision/context metadata and mandatory
reasoning. Fable 5.1 cannot honor forced tool selection; the candidate reports
that restriction rather than silently accepting an unenforced request.
No paid provider request was made. See the model note for exact sources,
candidate validation and remaining live-provider limitations.

[Logger guard patch](logger_guard_integration.patch) proposes BC-79 for
nonreentrant lock reacquisition, adds the suffix incident to BC-16, and wires the
new rotation test into the policy, expected fixture, hook and CI guard lists.
It includes current workflow-document updates and preserves the harness count.
It is unapplied and passes `git apply --check`; the worker also validated
candidate catalog/enforcement consistency in memory. [Repair evidence](logger_rotation.md).

These patches are **not active in the GUI or CI**. The user's
[no-touch handoff](/home/lukeh/daemon_checkpoints/CODEX_NO_TOUCH_20260913.md)
says “Files Plan 2 is changing (do not edit these in any checkout)” and
reserves `models/model_manager.py` and the class-guard files. Their owners must
release those files before application; after Claude lands, refresh/rebase
the candidates against the new base and re-run the model parity and guard
checks. The user retains all commits, pushes and PR submission.

## Next bounded work

The [execution plan](PLAN.md) includes worker inputs, steps, ownership,
failure handling and checks. The [diagnostics handoff](diagnostics.md) specifies
A07 producer fields: correlation/build/schema identifiers, typed outcomes,
monotonic stage times, first-token latency, tokens/cache/cost and retries.
Producer wiring and trusted action-history receipts remain queued for their
owners. No additional investigation or worker batch is running after handoff.
