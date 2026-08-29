# Fable Midflight Handoff

Prepared for the next session after the 2026-08-28 credit/session limit. The
worktree is shared. Do not reset, clean, or overwrite existing changes.

## Current Freeze

Fable owns the active implementation lane involving tone, escalation, graph
ingestion, and adjacent orchestration. Until the checkpoint below is complete,
do not edit these files:

- `core/context_pipeline.py`
- `core/escalation_tracker.py`
- `utils/tone_detector.py`
- `utils/emotional_context.py`
- `core/orchestrator.py`
- `memory/graph_utils.py`
- `memory/memory_storage.py`
- `config/app_config.py`
- `config/config.yaml`
- `config/schema.py`
- Fable's existing touched tests

Do not cherry-pick or reset anything. All agents share the same filesystem.

## First Action on Resume

Before making another edit, report:

1. `git status --short` and the changed-file manifest.
2. The current diff summary and any uncommitted partial edit.
3. Which files are complete, in progress, or logically abandoned.
4. Tests run, including exact commands, pass counts, failures, and hangs.
5. Config or feature-flag changes that require restart.
6. Any state, cache, graph, corpus, or migration effects.

Then create a checkpoint commit or a patch snapshot. A commit is preferred; a
diff snapshot is acceptable if committing is not yet appropriate.

## Known Fable Work

The de-escalation batch added a bounded CONCERN grounding path, distinguishes
organic distress from `distress_sticky_floor`, and attempts to make
`GENTLE_REENGAGEMENT` reachable. Fable reported 22 dedicated tests and 297
touched-suite tests passing.

Independent verification found that the dedicated test run stalled after 15
tests in the current environment. Reproduce this with `pytest -vv`; do not mark
the batch complete based only on the earlier report.

The graph repair added `relation_species_conflict()` in
`memory/graph_utils.py`, but there is currently no call site. This is unfinished
and must either be wired into ingestion with focused tests or removed as an
unused partial change.

The grounding correction remains suffix-based. That is a separate unresolved
problem; do not claim the accuracy issue is fixed merely because an accuracy
clause was added to prompts.

## Required Review Order

1. Checkpoint and review Fable's existing batch.
2. Finish or revert only the incomplete graph helper, with a synthetic species
   conflict test.
3. Resolve the de-escalation test hang or document its deterministic fixture
   requirement.
4. Run the existing touched suites and a small owner-canary replay.
5. Review the independent G12 contract in
   `docs/generalization/12-response-integrity.md`.

Do not combine the G12 response-integrity implementation with the tone,
escalation, or graph patch.

## G12 Next Batch

After the checkpoint, add new tests and modules first. The required contract is:

```text
draft -> structured review -> PASS | REWRITE | FALLBACK -> final
```

The correction reviewer must never append user-facing prose. The only displayed,
stored, and indexed text must be the same final response. Drafts and reviewer
JSON may remain in a private debug ledger but must not enter retrieval.

Use only the sanitized incident in G12. Do not send the raw conversation,
identifiers, private logs, or personal-store contents to Fable.

## Completion Criteria

The current Fable batch is complete only when:

- The checkpoint manifest exists.
- Every claimed fix has a call site and focused regression test.
- The de-escalation suite completes reproducibly.
- No raw private material was used in the development bundle.
- Grounding suffix behavior is explicitly listed as unresolved until G12 lands.
- Human review has examined the final diff before canary or restart.
