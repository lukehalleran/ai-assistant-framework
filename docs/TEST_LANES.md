# Test lanes — selection ledger (2026-09-09, B5/T14)

Written as part of the 2026-09-09 independent-bug-audit repair plan
(`docs/PLAN_20260909_audit_repairs.md`, batch B5), which asked for "a
selection ledger: every `--ignore`/`-m` exclusion in
`.github/workflows/tests.yml` and `pytest.ini`, with reason, owning
subsystem, fast replacement coverage, and the exact bounded command that
runs it." This is documentation only — **`.github/workflows/tests.yml`
itself was NOT edited here** (batch B4 owns that file concurrently); a
proposed hunk is recorded at the end of this doc's companion entry in
`PLAN_20260909_audit_repairs.md` for Fable to merge.

All commands below assume the repo root (so `python` resolves to the
project's pyenv 3.11.8) and use the env this batch's verification runs
used (CI itself sets only `OPENAI_API_KEY` and `CHROMA_DEVICE`; the two
`*_OFFLINE` vars are a local convenience that stops HF network lookups):

```bash
DAEMON_TEST_MODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

## 1. `.github/workflows/tests.yml` — the former 17 `--ignore` entries (REMOVED in 52a7fbb, 2026-09-09)

**Verification done today (2026-09-09):** all 17 files were run together,
unignored, under the exact marker filter CI uses
(`-m "not slow and not benchmark and not semantic"`):

```bash
python -m pytest -q -m "not slow and not benchmark and not semantic" \
  tests/test_memory_coordinator_advanced.py tests/test_memory_deep_integration.py \
  tests/test_memory_internal_methods.py tests/test_edge_cases_comprehensive.py \
  tests/test_prompt_deep_paths.py tests/test_thread_tracking.py \
  tests/test_multi_collection_chroma_store.py tests/unit/test_memory_coordinator_methods.py \
  tests/unit/test_visual_memory_pipeline.py tests/unit/test_context_pipeline.py \
  tests/test_active_day_decay.py tests/test_fix_verification.py \
  tests/test_temporal_retrieval.py tests/test_thread_surfacing.py \
  tests/test_user_profile_schema_preferences.py tests/test_wizard.py \
  tests/unit/test_cross_deduplicator.py
```

Result: **439 passed, 3 skipped (each with an explicit, non-vacuous skip
reason — see below), 0 failed, 200.09s.** The exclusion commit
(`47122c0`, 2026-05-11, "ci: exclude stale test files referencing removed
methods") is itself now stale: whatever removed methods these referenced in
May have since been fixed forward, and the CI ignore list was never pruned.
None of the 17 needs product code changed by this finding — the fix is
removing the ignore lines (see "CI hunk for Fable to merge" in the plan
doc), which B4 applied in commit 52a7fbb (2026-09-09); CI has run green with
zero `--ignore`s since.

| # | File | Owning subsystem | Current status (2026-09-09) | Fast replacement if still excluded |
|---|---|---|---|---|
| 1 | `tests/test_memory_coordinator_advanced.py` | `memory/memory_coordinator.py` (summaries/dreams/debug state/consolidation) | PASS (1 skip: `debug_memory_state not fully implemented` — real, named gap) | `tests/unit/test_memory_coordinator_methods.py` covers adjacent pure methods but not this file's async coordinator flows — no full substitute |
| 2 | `tests/test_memory_deep_integration.py` | memory retrieval edge cases (empty/long/unicode queries) | PASS (1 skip: `Method signature different` — named gap, not a silent xfail) | none full; `tests/unit/test_request_path_parity.py` covers unrelated request-shape edges |
| 3 | `tests/test_memory_internal_methods.py` | thread detection, fact extraction, meta-conversational memories | PASS | none full |
| 4 | `tests/test_edge_cases_comprehensive.py` | cross-module edge cases | PASS | none full |
| 5 | `tests/test_prompt_deep_paths.py` | prompt builder deep paths | PASS | `tests/unit/test_independent_prompt_audit.py` overlaps partially (deployed-builder driving) but targets different scenarios |
| 6 | `tests/test_thread_tracking.py` | `memory/thread_manager.py` | PASS (1 test) | none |
| 7 | `tests/test_multi_collection_chroma_store.py` | real `MultiCollectionChromaStore` (the T03 "real database contracts" surface named in the handoff) | PASS | B2/B3 added narrower real-Chroma contract tests (`test_sep09_storage_repairs.py`, `test_sep09_curation_integrity.py`) but this file is the broad real-store suite T03 pointed at — no full substitute |
| 8 | `tests/unit/test_memory_coordinator_methods.py` | pure `memory_coordinator` helper methods (rank/sort/normalize) | PASS (96 tests) | is itself the fast/pure counterpart to #1 |
| 9 | `tests/unit/test_visual_memory_pipeline.py` | CLIP/visual memory pipeline | PASS (22 tests) | none |
| 10 | `tests/unit/test_context_pipeline.py` | `core/context_pipeline.py` (tone/topic/intent/STM orchestration) | PASS (39 tests) | none full — this is itself the fast unit suite for the subsystem named in the handoff |
| 11 | `tests/test_active_day_decay.py` | temporal decay scoring | PASS | none |
| 12 | `tests/test_fix_verification.py` | ad hoc historical fix check (1 test) | PASS | none needed — low value, candidate for deletion (owner call, out of scope) |
| 13 | `tests/test_temporal_retrieval.py` | temporal retrieval (1 test) | PASS | none |
| 14 | `tests/test_thread_surfacing.py` | thread surfacing / duplicate-thread regression | PASS (9 tests) | none full |
| 15 | `tests/test_user_profile_schema_preferences.py` | `memory/user_profile_schema.py` | PASS (13 tests) | `tests/unit/test_user_profile_schema.py` (confirmed present) covers other slices of the same module; not identical (preferences-specific behavior has no other test) |
| 16 | `tests/test_wizard.py` | `gui/wizard.py` first-run flow | PASS (37 tests) | none |
| 17 | `tests/unit/test_cross_deduplicator.py` | `memory/cross_deduplicator.py` contradiction-arm safety | PASS (55 tests) | none full |

**Recommendation (DONE in 52a7fbb):** all 17 lines removed. None
are a model-download, live-store, or genuinely-slow lane — they are unit-
shaped tests over real-but-cheap fixtures (ephemeral Chroma, tmp paths,
mocked providers) that happen to have been swept into a blanket exclusion
four months ago and never re-evaluated. The 3 documented skips are honest
(name a specific unimplemented method / signature drift), not silent
`assert True`-style vacuity — they are fine to ship as-is.

## 2. `pytest.ini` — 4 `--ignore` entries

| # | File | Reason | Owning subsystem | Fast replacement | Bounded command |
|---|---|---|---|---|---|
| 1 | `tests/memory_test.py` | **Collection error, current and real** (not stale): a legacy top-level smoke script — module-level `asyncio.run(smoke_test())` at import time, not `test_*` functions — that drives the real orchestrator's `_SimplePromptBuilder` fallback (`core/orchestrator.py`), which no longer has an `_assemble_prompt` method. Verified today: `AttributeError: '_SimplePromptBuilder' object has no attribute '_assemble_prompt'` during collection. | `core/orchestrator.py::_SimplePromptBuilder` (fallback prompt builder used when the modular `core/prompt` package can't be constructed) | **NONE** — `tests/unit/test_request_path_parity.py` and `tests/unit/test_independent_prompt_audit.py` drive the real `UnifiedPromptBuilder`, not the `_SimplePromptBuilder` fallback path. This fallback appears to have no current test coverage at all. Scoped gap, flagged for a follow-up ticket (fixing/deleting `_SimplePromptBuilder` or this script is a `core/orchestrator.py` change, out of B5's scope per its STOP conditions). | not runnable as a real test file; would need a rewrite, not a command |
| 2 | `tests/test_double_filtering_performance.py` | **Collection error, current and real**: `ModuleNotFoundError: No module named 'tests.integration.gate_system_helpers'`. That helper module does not exist anywhere in the tree (`tests/integration/` only has `test_context_pipeline_integration.py` and `test_golden_distress_replay.py`). | Cross-encoder / gate-system reuse performance (the double-filtering fix this repo is named for) | Partial: `tests/test_actual_caching.py::test_model_manager_cross_encoder_caching` (repaired in this batch — real process-global cache dict, model-loading boundary only) covers the caching half; `tests/test_performance_fix.py` covers a simulation of gate-system reuse but not through the deployed prompt-build path. Neither is a full substitute for "measure total prompt-build time end to end." | N/A (fails collection) |
| 3 | `tests/test_double_filtering_regression.py` | **Dead/stale reference**: this file does not exist in the tree (confirmed via `git ls-files` / `find`); the ignore line matches nothing and has no effect. | N/A | N/A | N/A — safe to delete the ignore line as pure cleanup, zero behavior change |
| 4 | `tests/test_gating_consistency.py` | **Collection error, current and real**: same missing `tests.integration.gate_system_helpers` module as #2. | Gate-system output consistency across instances/caching (`processing/gate_system.py`) | Partial: `tests/unit/test_gate_min_results_cap.py`, `tests/unit/test_gate_retrieval_space.py`, `tests/unit/test_gate_forced_quality_floor.py`, `tests/unit/test_lane_regressions.py` cover various gate-behavior slices, but none specifically re-asserts "the same memories produce identical filtering results regardless of gate instance/caching approach" — this is a scoped gap. | N/A (fails collection) |

**Do not silently remove #1/#2/#4's ignore lines** — unlike the 17 CI
entries above, these three still genuinely fail to collect today. Removing
their ignores without also either restoring `tests/integration/gate_system_helpers.py`
(for #2/#4) or repairing/retiring `tests/memory_test.py` (#1) would break
every local/CI run that stops ignoring them. #3 is the only pytest.ini
entry safe to prune as pure documentation hygiene.

## 3. Markers: `slow`, `benchmark`, `semantic`

CI and `hooks/pre-push` both run with `-m "not slow and not benchmark and
not semantic"`. Grep-counted marked test *files* (not exhaustive of every
individual `@pytest.mark.*`-decorated function, since some files mark only
a subset of their tests):

| Marker | Files carrying it | Example | Reason | Bounded command to run them |
|---|---|---|---|---|
| `slow` | `tests/agent_branch/test_eval_gate.py`, `tests/agent_branch/test_isolation_redteam.py`, `tests/agent_branch/test_portfolio.py` | agent-branch supervision suite | Long-running (spawns sandboxed evaluation) | `python -m pytest -q -m slow tests/agent_branch/` |
| `benchmark` | `tests/benchmarks/test_retrieval_quality.py` (+ the `tests/benchmarks/` package: `retrieval_benchmark.py`, `report_generator.py`) | the 283/296-case retrieval benchmark cited in `docs/METRICS_SNAPSHOT.md` | Loads real embedding models, runs the full 296-case corpus; minutes not seconds | `python -m pytest -q -m benchmark tests/benchmarks/` (daemon down; see §4) |
| `semantic` | `tests/test_need_detection.py` | semantic-similarity assertions needing a real embedder | Real model load, embedder-gated (skips cleanly if unavailable) | `python -m pytest -q -m semantic tests/test_need_detection.py` |

Golden-path semantic regression (`tests/integration/test_golden_distress_replay.py`)
is explicitly embedder-gated (per the handoff's T16) rather than marker-gated
— it self-skips when no real embedder is available, so it stays in the
default collection without a marker exclusion.

## 4. Lanes

**Fast CI lane (what runs on every push):**
`.github/workflows/tests.yml` → `python -m pytest -q -m "not slow and not
benchmark and not semantic" --tb=short` over `testpaths = tests` (from
`pytest.ini`) minus the 4 `pytest.ini` ignores (the 17 CI-only ignores are gone since
52a7fbb). This lane also runs the privacy guard, `ruff check .`, and a
separate `frontend` job: `npm run typecheck` then `npm test` (the Vitest
behaviour lane B4 added for F07/T09); there is no `npm run build` step
in CI.

**Bug-class scan lane (added 2026-09-11)** — stdlib-only, no application
import, ~2 s; runs in `hooks/pre-push` step 2a and in the CI backend job
before the suite:

| Lane | Command | What it gates | Baseline |
|---|---|---|---|
| bug-class scan ratchet | `python scripts/check_bug_classes.py scan --root .` then `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider --confcutdir=tests/bug_class_guards tests/bug_class_guards` | the DM scanners in `scripts/bug_class_guards/scanners/` (DM-01/17/18/31 gated, DM-16/29 report-only) + `docs/BUG_CLASSES.md` consistency | `config/bug_class_baseline.json` — content-anchored `(scanner, path, enclosing qualname, source line)`; exit 1 on a NEW finding, on a STALE entry, or when a gated scanner processed zero files; exit 2 when a scanner could not run |

**2026-09-13 verification:** the five gates caught synthetic new findings and
stale entries; the isolated harness passed 63 tests. The latest pushed tree
accepted 133 baselined findings. DM-29 processed zero files in GitHub because
its local changelog input is untracked; its zero is not a clean audit result.
The [consolidated review](GENERALIZATION_CI_REVIEW_20260913.md) records the
exact CI run, guard limitations, branch-protection state, and proposed probe
lane. No assembled-turn probe lane is wired yet.

**The five repo-wide guards** (must be green before every push per
`docs/DEVELOPMENT_WORKFLOW.md` §7.1, and are always included in
`hooks/pre-push`'s selection regardless of what changed):
- `tests/unit/test_no_git_state_in_tests.py` — bans tests reading git-tracked
  blobs/refs (the class that made 05fd300 pass dirty and fail committed).
- `tests/unit/test_ordered_slice_guard.py` — content-anchored guard against
  reintroducing newest-first-then-truncate bugs.
- `tests/unit/test_budget_meters_rendered_sections.py` — every rendered
  prompt section has a `PRIORITY_ORDER` row (the 2026-08-14 dead-key class).
- `tests/unit/test_tool_wiring_parity.py` — every tool/action has a
  dispatch/executor row.
- `tests/unit/test_model_capability_wiring.py` — every registered model has
  a capability row and the four classifiers agree.

Run together (also used as this batch's own regression sweep, §5 below):
```bash
python -m pytest -q tests/unit/test_no_git_state_in_tests.py \
  tests/unit/test_ordered_slice_guard.py \
  tests/unit/test_budget_meters_rendered_sections.py \
  tests/unit/test_tool_wiring_parity.py \
  tests/unit/test_model_capability_wiring.py
```

**New real-driver Chroma/JSON tests from this repair plan** — collection
verified today (2026-09-09) with:
```bash
python -m pytest -q --collect-only -m "not slow and not benchmark and not semantic" \
  tests/unit/test_sep09_storage_repairs.py tests/unit/test_sep09_curation_integrity.py \
  tests/unit/test_sep09_atomic_writers.py tests/unit/test_atomic_writer_guard.py \
  tests/unit/test_sep09_backup_recovery.py
# => 59 tests collected, 0 errors, 0 deselected
```
None of the five is in either ignore list and none carries a `slow`/
`benchmark`/`semantic` marker, so all 59 are already part of the fast CI
lane described above — no additional CI wiring is needed for them. (B4's
frontend Vitest lane, if adopted per B0's decision, is out of scope for this
verification — `web/` is B4's file per this batch's STOP conditions.)

**Non-unit remainder** (everything outside `tests/unit`, minus the
benchmark/slow/semantic-marked and the ignored files above): per
`docs/DEVELOPMENT_WORKFLOW.md` §3, this is **never** run beside the live
Daemon (a 2026-09-07 incident swap-thrashed the box to a hard reset when it
ran during a shutdown). Run it with the Daemon down, memory-capped:
```bash
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=512M \
  python -m pytest -q -m "not slow and not benchmark and not semantic" tests/ \
  --ignore=tests/unit
```
**`hooks/pre-push` and the cap (corrected 2026-09-10):** the hook runs
BOTH its selections — the changed-files-in-the-push-range selection and the
optional `PREPUSH_FULL=1` unit re-run — under the same `MemoryMax=6G`
wrapper, in ONE pytest process. The changed-files selection is not unit-only:
it is whatever test files the push range touched, plus the five guards, plus
the same-named test of every changed source file (T14 mapping). When the
push range touches non-unit test files, that selection IS a non-unit batch
under a cap 2 GiB below the one this section documents, and it gets
SIGKILLed with no test failure. Observed 2026-09-10 pushing the T01 repair
(10 non-unit files + 5 guards + 9 unit files, 24 in all): killed at ~40%
("Killed … pre-push: BLOCKED — tests"); the same 10 non-unit files + guards
alone peak at 5.97 GB RSS under 6G (killed near test 215/307, inside
`test_memory_coordinator_methods.py`/`test_memory_deep_integration.py`) and
pass under the documented 8G cap (305 passed / 1 skipped / 1 xfailed, peak
7.1 GB, 3 min — `~/daemon_checkpoints/prepush_nonunit_8G_20260910.txt`).
The unit remainder of that selection (incl. the CLIP-loading
`test_visual_memory_pipeline.py`/`test_clip_manager.py`, which the T14 mapping
pulls in even though the hook's own `PREPUSH_FULL` path ignores the former)
peaked at 1.3 GB — the non-unit files are the cost, not CLIP. Until the hook
is changed (follow-up: raise its cap to 8G to match §3, and skip
ignore-listed files in the mapping), a push range containing non-unit test
files must be verified by hand — Daemon down, the hook's exact selection
under the 8G wrapper above, output under `~/daemon_checkpoints/` — and then
pushed with `SKIP_PREPUSH=1 git push`, citing that output file in the
handoff/commit. Never bypass on a green-by-assertion.

**Eval suite** (`tests/test_eval/`, 246 tests by collect-only on 2026-09-10; not broken out in `docs/METRICS_SNAPSHOT.md`)
is unmarked and unignored — already part of the fast lane above.
