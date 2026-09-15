# F9b — memory expansion reports a failed store read as an explicit, uncached error; the timestamp-range producer stops returning a silently partial id list

Batch ID: F9b. Request: CGR-20260913-009 (follow-up response `-2`, anchor #122).
Base commit: 328a8ecea1dae91de16f5974ea71727c4e2e1aa0 (uncommitted tree, no commits made).
Worker started: 2026-09-14. Launched only after F9a was integrated (parent review confirmed in F9a.md).

## 1. Manifest check (before any edit)

```
git status --short   # 64 tracked M, ~50 untracked paths incl. docs/execution/generalization/ (new tree)
git rev-parse HEAD    # 328a8ecea1dae91de16f5974ea71727c4e2e1aa0
git diff --stat       # 68 files changed, 3590 insertions(+), 685 deletions(-)
git remote -v         # origin (fetch) /home/lukeh/Daemon_v1 ; origin (push) DISABLED
```

Assigned-file digests at start (match the brief's "Post-F9a re-verification" facts exactly):
```
a2152df35f33796ba2dd145141adce7255f4163288d70bc99abc23a57adc6495  memory/memory_expander.py
be9b48e56ac4edfb8c3ebb397d51788d5f9b8afa600a39c2ab18f145465fd5f5  memory/storage/multi_collection_chroma_store.py
```

Manifest check against `$S/manifest_post_F9a.txt` and `$S/manifest_paths_post_F9a.txt` (215 paths):
- `sha256sum -c $S/manifest_post_F9a.txt`: 215 lines, all `OK`. **0 mismatches.**
- Path listing (`git status --short -uall`, since `docs/execution/generalization/` is a single untracked-dir line under the default `git status --short`) vs `manifest_paths_post_F9a.txt`, both sorted: `diff` → 0 lines, 215 == 215. **Empty path diff.**

MemAvailable before first chunk: 6639 MB (≥4000, OK). `logs/` absent (confirmed, `ls -ld --time-style=full-iso logs` → No such file or directory).

data/ listing (top level only) matches `$S/data_baseline_f7a_rerun.txt` exactly (`diff` → 0 lines). No new/changed entry.

Sanity check: `PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin python -c "import utils; print(utils.__file__)"` → `/home/lukeh/daemon_exec/generalization/utils/__init__.py` (inside this clone, as required).

Manifest check: **PASS**. Proceeding.

## 2. Pre-edit scan (read-only)

`PYTHONPATH=.../scripts/bin python scripts/check_bug_classes.py scan --root .` — exit 1 (gate; pre-existing unrelated findings).
dm18 summary row: `dm18_except_returns_empty  gate  123  48  79  0  31` — matches the parent's post-F9a rerun exactly (found 48, base 79, new 0, stale 31).
Other gate scanners unchanged from the F9a parent rerun: dm01 (7/12/0/5), dm17 (36/39/0/3), dm31 (3/3/0/0). Scan stderr empty.
The plain-text scan output enumerates only STALE + report-only detail, not individual "found" anchors, so #122 does not appear by name pre-edit; `memory/memory_expander.py` is absent from the STALE list (expected — nothing there is fixed yet). This is the baseline to diff against post-edit.

## 3. New test file and failing-first run

New file: `tests/unit/test_expander_outcomes.py` (373 lines; fakes only — `MultiCollectionChromaStore.__new__` + a fake Chroma collection with independently-raising `.get()`/`.count()` for the producer side, and a `FakeChromaStore` extended from test_memory_expander.py's with on-demand raises + call counters for the expander side; never a real Chroma client). Covers: both `get_ids_by_timestamp_range` except-block reasons (`numeric:<class>`, `legacy_page:<class>`), the `count()`/malformed-bounds/healthy-union controls, `expand()`'s RetrievalError-not-cached contract (range-producer RetrievalError, range-producer non-RetrievalError wrap, anchor `get_by_id` RetrievalError, per-id `get_by_id` RetrievalError abort), the non-RetrievalError-still-propagates control, and the genuine-empty-range / healthy-expansion cached controls. Privacy markers asserted absent from both `RetrievalError.reason`/`str()` and the expander's `error` string.

FAILING-FIRST command (digest line + new tests, in one command, UNEDITED source):
```
sha256sum memory/memory_expander.py memory/storage/multi_collection_chroma_store.py
systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=512M /usr/bin/time -v env \
  DAEMON_TEST_MODE=1 CHROMA_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin \
  python -m pytest -q -p no:cacheprovider -p asyncio tests/unit/test_expander_outcomes.py
```
Digests (unchanged from section 1):
```
a2152df35f33796ba2dd145141adce7255f4163288d70bc99abc23a57adc6495  memory/memory_expander.py
be9b48e56ac4edfb8c3ebb397d51788d5f9b8afa600a39c2ab18f145465fd5f5  memory/storage/multi_collection_chroma_store.py
```
Result: **6 failed, 6 passed** in 3.95s (Maximum resident set size 1,109,112 KB ≈ 1.08 GB). Exit 1.

Failures (exactly the new-behavior assertions; every control already passes on unedited source):
- `TestTimestampRangeProducerOutcomes::test_numeric_query_failure_raises_retrieval_error` — `Failed: DID NOT RAISE RetrievalError` (old code swallows into `numeric_ids` staying empty, execution continues).
- `TestTimestampRangeProducerOutcomes::test_legacy_page_failure_raises_retrieval_error` — `Failed: DID NOT RAISE RetrievalError` (old code `break`s instead).
- `TestExpandRetrievalErrorOutcomes::test_range_producer_retrieval_error_not_cached` — old `_fetch_conversations_in_range` catches and returns `[]`, so `expand()` returns today's "no source_doc_ids or temporal anchors" fallback text instead of `"expansion_failed: timestamp_range: numeric:RuntimeError"`.
- `TestExpandRetrievalErrorOutcomes::test_range_producer_non_retrieval_error_wrapped` — same swallow, same wrong fallback text.
- `TestExpandRetrievalErrorOutcomes::test_anchor_get_by_id_retrieval_error_not_cached` — `expand()` has no try/except yet, so the raw `RetrievalError` propagates out of the test call instead of being turned into an error dict.
- `TestExpandRetrievalErrorOutcomes::test_per_id_get_by_id_retrieval_error_aborts_range_fetch` — same: `expand()` has no try/except yet, raw `RetrievalError` propagates.

Passing controls (6, unaffected by this batch's edit): `test_count_failure_still_returns_union`, `test_malformed_bounds_returns_empty`, `test_healthy_numeric_plus_legacy_mix_sorted_union`, `test_non_retrieval_error_from_store_propagates_out_of_expand`, `test_control_genuine_empty_range_uses_fallback_and_is_cached`, `test_control_healthy_expansion_is_cached`.

**FIXTURE RULE:** no existing test file was edited to repair a pinned old-swallow assertion. `tests/unit/test_memory_expander.py`'s `FakeChromaStore` never raises, so none of its assertions pin the old swallow — it needed no change. No edits made to any pre-existing test file this batch (confirmed after the source edit below).

## 4. Source edit

`memory/memory_expander.py`:
- import: `from utils.retrieval_outcome import RetrievalError` (new line, after the existing `memory.utils` import).
- `expand()`: wrapped only the `result = self._do_expand(...)` call in `try/except RetrievalError as e:`, returning `{anchor_id, collection, expansion_method="timestamp_window", turns=[], total_in_collection=0, error=f"expansion_failed: {e.source}: {e.reason}"}` and returning immediately (no `self._cache[...] =` write reached) — any other exception still propagates unchanged.
- `_fetch_conversations_in_range()`: the `except Exception as e: logger.warning(...); return []` around the `get_ids_by_timestamp_range` call is now `except RetrievalError: raise` (propagate a typed producer failure unchanged) followed by `except Exception as e: logger.warning(...); raise RetrievalError(source="timestamp_range", reason=type(e).__name__) from e` (wrap any other exception). The per-id `get_by_id` call inside the loop is unchanged (already had no try — still propagates). Added a short docstring paragraph noting the F9b raise behavior; added one comment line above the per-id loop noting the no-per-id-guard decision.

`memory/storage/multi_collection_chroma_store.py` (`get_ids_by_timestamp_range` body only):
- Pass 1 (numeric query) except: kept the existing warning line, added `raise RetrievalError(source="timestamp_range", reason=f"numeric:{type(e).__name__}") from e`.
- Pass 2 (legacy page) except: kept the existing warning line, replaced `break` with `raise RetrievalError(source="timestamp_range", reason=f"legacy_page:{type(e).__name__}") from e`.
- `count()` try/except, unknown-collection `[]`, malformed-bounds `[]`, the malformed-row skip and the healthy union are all byte-for-byte unchanged. No new import needed — `RetrievalError` was already imported by F9a (line 70).
- Added a short docstring paragraph noting the F9b raise behavior (mirrors `get_by_id`'s F9a docstring note).

No other method in either file was touched. `git diff --stat` for the two files: `memory/memory_expander.py: 30 +++++++++++++++++++++++--` (28 insertions, 2 deletions before context lines), `memory/storage/multi_collection_chroma_store.py: 24 ++++++++++++++++----` (20 insertions, 4 deletions before context lines) — `48 insertions(+), 6 deletions(-)` total across both files.

Digests after edit:
```
358007d2544cba94aa4c1f4447767bbe6d566b69ddf7c3493727c0cad9b2f6fe  memory/memory_expander.py (before: a2152df3...)
66833584f2dfe725c27e3c4fb39af2e71e28b822d341e3da12667bca6e0992fe  memory/storage/multi_collection_chroma_store.py (before: be9b48e5...)
```

No other tracked or new file was touched by this edit step. `git status --short -uall` re-checked immediately after: only these two files newly show `M`, plus the already-recorded new `tests/unit/test_expander_outcomes.py` and this packet file — nothing else changed.

## 5. Focused chunk (post-edit)

MemAvailable before: 6624 MB (≥4000, OK). Guard: no `-m pytest` process running.

```
systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=512M /usr/bin/time -v env \
  DAEMON_TEST_MODE=1 CHROMA_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin \
  python -m pytest -q -p no:cacheprovider -p asyncio \
  tests/unit/test_expander_outcomes.py tests/unit/test_memory_expander.py \
  tests/unit/test_sep09_storage_repairs.py tests/unit/test_store_get_by_id_outcomes.py \
  tests/unit/test_claim_tracker.py tests/unit/test_chroma_store_no_autodelete.py \
  tests/unit/test_recent_summaries_fix.py tests/unit/test_sep09_curation_integrity.py
```
Result: **163 passed, 0 failed** in 5.37s (elapsed 8.24s incl. systemd-run/time overhead; Maximum resident set size 1,177,664 KB ≈ 1.15 GB). Exit 0.

data/ listing (top level) after this chunk: unchanged from `$S/data_baseline_f7a_rerun.txt` (`diff` → 0 lines). `logs/` still absent. (test_sep09_storage_repairs.py and test_sep09_curation_integrity.py use a real in-memory `chromadb.EphemeralClient`, per the brief's allowance, checked here.)

## 6. Sweep

Grep for every remaining `tests/unit` importer of `memory.memory_expander`/`MemoryExpander`/`memory.storage.multi_collection_chroma_store`/`MultiCollectionChromaStore`/`memory.shutdown_processor`/`ShutdownProcessor`:
```
grep -rlE "memory\.memory_expander|MemoryExpander|memory\.storage\.multi_collection_chroma_store|MultiCollectionChromaStore|memory\.shutdown_processor|ShutdownProcessor" tests/unit/
```
27 total importers; 8 already covered by the focused chunk (§5); 19 remain. Each was read/vetted before running:
- Fake/mock store or monkeypatched class (no real Chroma, no real embedder): `test_backfill_stance.py` (monkeypatched `FakeStore`), `test_gate_retrieval_space.py` (`get_st_model` called unbound on a `SimpleNamespace`), `test_health_transient_retrieval.py` (`_FakeStore` stand-in), `test_hybrid_semantic_score.py` (comment-only shape reference, no construction), `test_proposal_store.py` (mock), `test_script_apply_guards.py` (monkeypatched `FakeStore`, same pattern as F9a's sweep), `test_synthesis_generator.py` (mock), `test_thread_store.py` (mock), `test_upload_retrieval_pool.py` (`MultiCollectionChromaStore.__new__`, the F9a/F9b bypass pattern).
- Pure-function / no-construction / bypassed-`__init__`: `test_retrieval_pool_caps.py` (imports only `_resolve_embed_device`), `test_retrospective_small_guards.py` (imports only the module-level `_log_shutdown_llm_failure`), `test_source_excerpt_pipeline.py` (static source-string count on the module, no object built), `test_stance_write_path.py` (calls `ShutdownProcessor._capture_tone_for_triple` unbound, no instance), `test_stream_artifacts.py` (`object.__new__(ShutdownProcessor)`, bypasses `__init__`), `test_ordered_slice_guard.py` (a static string-table guard; the `memory/shutdown_processor.py` lines are data rows, not an import/construction), `test_sep10_probe_dump_actions.py` (comment-string match only, confirmed by F9a's parent review too).
- `ShutdownProcessor(...)` constructed with `chroma_store`/`storage` as `MagicMock()` (never real): `test_entity_facts.py` (both call sites, read in full — `mock_chroma = MagicMock()`, `storage=MagicMock()`), `test_shutdown_pipeline_proposals.py` (`mock_chroma_store` fixture is `MagicMock()`, `storage=MagicMock()`), `test_knowledge_graph.py` (`mock_chroma = MagicMock()`; `storage` is a real `MemoryStorage` but wired to the same `mock_chroma` plus `graph`/`resolver` fixtures built from `tmp_graph_path`/`tmp_aliases_path` — tmp_path-backed, not default paths; read the fixture definitions to confirm).

No file matched any exclusion (`test_graph_integration.py`, the ContextGatherer/UserProfile hazard, `tests/test_web_search_manager.py`, `tests/test_prompt_internal_methods.py`) or needed exclusion on vetting grounds.

3 chunks (9, 9, 1):

**Sweep chunk 1** (MemAvailable 6696 MB before): `test_backfill_stance.py test_entity_facts.py test_gate_retrieval_space.py test_health_transient_retrieval.py test_hybrid_semantic_score.py test_knowledge_graph.py test_ordered_slice_guard.py test_proposal_store.py test_retrieval_pool_caps.py` → **192 passed, 0 failed** in 47.98s (Max RSS 3,248,948 KB ≈ 3.10 GB). Exit 0. data/ unchanged after; logs/ absent.

**Sweep chunk 2** (MemAvailable 6707 MB before): `test_retrospective_small_guards.py test_script_apply_guards.py test_sep10_probe_dump_actions.py test_shutdown_pipeline_proposals.py test_source_excerpt_pipeline.py test_stance_write_path.py test_stream_artifacts.py test_synthesis_generator.py test_thread_store.py` → **426 passed, 0 failed** in 12.97s (Max RSS 1,733,644 KB ≈ 1.65 GB). Exit 0. data/ unchanged after; logs/ absent.

**Sweep chunk 3**: `test_upload_retrieval_pool.py` → **28 passed, 0 failed** in 4.29s (Max RSS 1,114,268 KB ≈ 1.06 GB). Exit 0. data/ unchanged after; logs/ absent.

**Sweep total: 646 passed, 0 failed.** Combined with the focused chunk: **809 passed, 0 failed** across this batch's testing.

## 7. Ruff

```
PYTHONPATH=.../scripts/bin python -m ruff check memory/memory_expander.py memory/storage/multi_collection_chroma_store.py tests/unit/test_expander_outcomes.py
```
Result: **All checks passed!** Exit 0. ruff 0.14.9.

## 8. Post-edit scan

`PYTHONPATH=.../scripts/bin python scripts/check_bug_classes.py scan --root .` — exit 1 (gate; pre-existing unrelated findings, same as pre-edit). Scan stderr empty.

`diff` of pre-edit vs. post-edit scan text output — exactly three changes:
- `dm17_apply_without_guard` files 630→631 (the new test file joins the tree scanned by this report-adjacent count; found/base/new/stale unchanged at 36/39/0/3).
- `dm18_except_returns_empty` row: `found 48→47, base 79 (unchanged), new 0 (unchanged), stale 31→32`.
- One new STALE line: `dm18_except_returns_empty: memory/memory_expander.py [MemoryExpander._fetch_conversations_in_range] 'return []'` — **anchor #122 confirmed STALE.**

`memory/storage/multi_collection_chroma_store.py [MultiCollectionChromaStore.get_by_id] 'return None'` (F9a's anchor #139, already STALE) is unchanged and still present in the STALE list — `get_ids_by_timestamp_range`'s two edited except-blocks were never a dm18 finding pre-edit (except → continue/break, not `return []`, as the brief predicted), so there is no third STALE line to expect from this batch's store edit, and none appeared.

dm01 (7/12/0/5), dm31 (3/3/0/0) unchanged. **new = 0 everywhere — no new finding introduced by this batch's edit.**

## 9. BC-58 sibling search (DM-15: every read AND write of the primitive, every sibling by naming pattern)

```
grep -rn "get_ids_by_timestamp_range" --include="*.py" . | grep -v '^\./tests/' | grep -v '^\./docs/' | grep -v '^\./\.agent_snapshots/'
grep -rn "\.expand(" --include="*.py" . | grep -v '^\./tests/' | grep -v '^\./docs/'
grep -rEn "def get_[a-z_]*_by_[a-z_]*range" --include="*.py" . | grep -v '^\./tests/'
```
- **`get_ids_by_timestamp_range` (the read primitive):** exactly one definition (`memory/storage/multi_collection_chroma_store.py:301`, fixed this batch) and exactly two production call sites: `memory/memory_expander.py:382` (`_fetch_conversations_in_range`, fixed this batch) and `memory/shutdown_processor.py:489` (`_store_summary`, inside `try: ... except Exception as se: logger.debug("[Shutdown] Could not capture source doc IDs: %s", se)` at 484-495, read/re-confirmed here). The `RetrievalError` this batch adds now propagates out of the producer and out of the expander unchanged — `shutdown_processor.py`'s broad `except Exception` still catches it, so the summary is stored without `source_doc_ids`, identically to today's "empty result" outcome. This consumer is read-only for F9b (out of ownership) and is left unfixed — recorded here and in the response file, mapped to a later batch per the class-guard request's remaining-anchors note (the request packet does not list a `memory/shutdown_processor.py` anchor for this primitive, so no anchor number is owed here; the parent's F10-F12 split governs the rest of CGR-009).
- **No other sibling by naming pattern:** `def get_[a-z_]*_by_[a-z_]*range` matches only this one method anywhere outside `tests/`. No sibling "get_X_by_Y_range" primitive exists to check.
- **`MemoryExpander.expand()` (the other primitive this batch changes the failure contract of):** exactly one production caller, `core/agentic/tools.py:1972` (`_execute_memory_expand`, `try: ... return self.memory_expander.expand(...) except Exception as e: return {"anchor_id":..., "turns": [], "error": str(e)}`, ≈1963-1975) and one more via `core/insight/sweep.py:258-264` (`try: exp = await asyncio.to_thread(memory_expander.expand, ...) except Exception as e: logger.debug(...); continue`). Both re-confirmed by direct read this batch (not just brief-quoted). Neither call site is affected adversely: a `RetrievalError` from `_do_expand` is now caught INSIDE `expand()` itself and turned into an error dict (never raised past `expand()`), so these two outer `except Exception` blocks see it even less often than before; any non-`RetrievalError` exception still propagates to them exactly as before. No unit test drives `_execute_memory_expand` directly (confirmed by grep of `tests/`: `_execute_memory_expand` appears in `core/agentic/tools.py` only) — an existing gap, not introduced by this batch, left unfixed and recorded here (already noted in the brief's parent-verified facts).

## 10. Privacy / no-network note

No real institution names or real IPs anywhere in the new file or the edits. Synthetic markers only (`SYNTH_MARKER_numeric_9f3c`, `SYNTH_MARKER_legacy_2b7e`, `SYNTH_MARKER_range_5d1a`), each asserted absent from `RetrievalError.reason`, `str(RetrievalError)` and `expand()`'s returned `error` string — three separate marker-leak assertions across the producer and expander test classes. `RetrievalError.reason` in both edited call sites is always `type(e).__name__` or a `"label:ClassName"` composite — never the exception's message text. No network call, no LLM/provider call, no `pip install` — all test doubles are in-process fakes/mocks or the real in-memory `chromadb.EphemeralClient` (sep09 files only, already covered by the data/ check).

## 11. Final manifest re-check and git state (pre-handoff)

```
git status --short   # unchanged tracked-file set plus: M memory/memory_expander.py, M memory/storage/multi_collection_chroma_store.py,
                      # ?? tests/unit/test_expander_outcomes.py, ?? docs/execution/generalization/batches/F9b.md (this file, already untracked-tree),
                      # ?? docs/execution/generalization/class_guard_responses/CGR-20260913-009-2.md (written next)
git rev-parse HEAD    # 328a8ecea1dae91de16f5974ea71727c4e2e1aa0 (unchanged; no commits made)
git diff --cached --name-only   # empty — nothing staged, no git add used
```
Digests after edit:
```
memory/memory_expander.py: 358007d2544cba94aa4c1f4447767bbe6d566b69ddf7c3493727c0cad9b2f6fe (before: a2152df35f33796ba2dd145141adce7255f4163288d70bc99abc23a57adc6495)
memory/storage/multi_collection_chroma_store.py: 66833584f2dfe725c27e3c4fb39af2e71e28b822d341e3da12667bca6e0992fe (before: be9b48e56ac4edfb8c3ebb397d51788d5f9b8afa600a39c2ab18f145465fd5f5)
tests/unit/test_expander_outcomes.py (new): 373 lines, sha256 3838aa85226d208cef70b6db2a63b0c0fb16bea0cbbe970ec92fc2972915d6d3
```
Full `git diff --stat` at handoff (tracked files only): `68 files changed` (unchanged file count — no new tracked files), with `memory/memory_expander.py` and `memory/storage/multi_collection_chroma_store.py` now included in the delta. No stray files anywhere outside owned locations. No `git add`/stash/reset/checkout/restore/clean used. No commit made; HEAD unchanged at `328a8ecea1dae91de16f5974ea71727c4e2e1aa0`. `origin` push URL re-confirmed `DISABLED` (read-only `git remote -v`, no config change).

**Manifest re-check (after):** `sha256sum -c $S/manifest_post_F9a.txt` reports exactly **2 FAILED** out of 215: `memory/storage/multi_collection_chroma_store.py` (this batch's own edit, expected — digest now `66833584...`) and `docs/execution/generalization/briefs/PARENT_STATE.md` (a pre-existing, parent-caused divergence already flagged by F9a's own parent-review section — the parent edited it again while drafting/re-verifying this brief; not touched by this worker). `memory/memory_expander.py` is **not present** in the 215-path manifest at all — it was not yet part of the modified-file set when the F9a snapshot was taken (no batch had touched it before F9b), so this is its first appearance as `M`, not a manifest mismatch. Every other one of the 215 baseline paths is unchanged. `git status --short -uall`, diffed against the pre-edit snapshot (§1), shows exactly three new lines beyond the two owned-file `M`s: the new `tests/unit/test_expander_outcomes.py`, the new `memory/memory_expander.py` `M` line, and this packet file (`docs/execution/generalization/batches/F9b.md`, a new path under the already-untracked `docs/execution/generalization/` tree). No other worker's files were touched.

## 12. Size

```
git diff --stat memory/memory_expander.py memory/storage/multi_collection_chroma_store.py
# memory/memory_expander.py: 30 (28 insertions, 2 deletions minus context marks -> 48 total ins/del across both files, see below)
wc -l tests/unit/test_expander_outcomes.py
```
Source: 48 insertions(+), 6 deletions(-) = **54 changed lines** across the two owned files (per `git diff --stat`). New test file: **373 lines** (per `wc -l`, new-file rule). **Total: 427 changed lines** — over the 380 target, under the 450 hard cap. No split proposal needed (only required when the hard cap would be exceeded). The overage past target is almost entirely the new test file's fake infrastructure (`_FakeRangeCollection` plus the extended `FakeChromaStore`, ~110 lines of fixture code) needed to independently control three failure points (numeric query, legacy page, per-id read) across both the producer and the expander, plus paired controls for every new behavior per the doctrine (BC-64) and the FIXTURE RULE's paired-control requirement.

Batch F9b: **COMPLETE** (focused contract green: 163 passed; sweep green: 646 passed; 809 passed total, 0 failed). Response file written next.

## Parent review and integration (2026-09-14)

Accepted as delivered. No COMPLIANCE breach: failing-first was recorded before the source edit, and no non-pytest code ran beyond the standing exceptions. There were no FIXTURE RULE edits. The size is 427, over the 380 target but under the 450 hard cap; the overage is fake infrastructure in the new test file.

### Manifest

- **Checksums:** `sha256sum -c manifest_post_F9a.txt` gives exactly two mismatches:
  - `memory/storage/multi_collection_chroma_store.py`, this batch's own edit (`be9b48e5…` → `66833584…`);
  - `briefs/PARENT_STATE.md`, which the parent edited while F9b ran.
- **New paths (four):**
  - `memory/memory_expander.py` (`a2152df3…` → `358007d2…`, its first modification);
  - `tests/unit/test_expander_outcomes.py` (`3838aa85…`, 373 lines);
  - `class_guard_responses/CGR-20260913-009-2.md` (`a5604bf6…`, 274 lines);
  - this file.
- **Git state:** `git diff --cached --name-only` is empty, HEAD `328a8ec` is unchanged, the reflog shows only the clone entry, and the stash list is empty.
- **No new stray files:** the only untracked file at the repository root is `scan_pre.stderr`.
- **`data/`:** the 7-entry baseline is unchanged, and `logs/` is absent.
- **Parent edits while F9b ran or after it returned:**
  - `briefs/PARENT_STATE.md` gained the F11a and F11b map decisions and F9b's integration;
  - `briefs/F10a.md` was written (F10a is next);
  - `failure_outcome_design.md` gained the "F10 split and the write error type" amendment;
  - the F10b, F11a-1, F11a-2, F11b-1 and F11b-2 drafts live in the parent scratchpad.
- **Recorded tree:** `manifest_post_F9b.txt` records the tree after this section.

### Code review (the parent read both full diffs)

- **`get_ids_by_timestamp_range`:**
  - Pass 1 keeps its warning, then raises `RetrievalError(source="timestamp_range", reason="numeric:<class>")`.
  - Pass 2 keeps its warning, then raises `reason="legacy_page:<class>"`; the `break` is replaced.
  - Unchanged: unknown collection → [], malformed bounds → [], the advisory `count()` fallback, the malformed-row skip and the healthy union.
  - It reuses F9a's import, and the docstring is updated.
- **`_fetch_conversations_in_range`:**
  - `except RetrievalError: raise` comes first, so a typed producer failure propagates unchanged and is not wrapped twice.
  - `except Exception as e` keeps its warning, then raises `RetrievalError(source="timestamp_range", reason=<class>)`.
  - A genuine empty id list → []. The per-id `get_by_id` has no guard, by decision.
- **`expand()`:**
  - Only the `_do_expand(...)` call is wrapped.
  - `except RetrievalError as e` returns the error-template dict (`anchor_id`, `collection`, `expansion_method="timestamp_window"`, `turns=[]`, `total_in_collection=0`, `error="expansion_failed: <source>: <reason>"`) before the cache line, so a failure is never cached.
  - Any other exception still propagates.
- **Callers (parent grep):**
  - `MemoryExpander.expand()` is called by `core/agentic/tools.py:1972`, which catches it into an error dict; it now receives the explicit error dict.
  - It is also called by `core/insight/sweep.py` through `asyncio.to_thread(memory_expander.expand, …)` (the worker read it, and the F9a map agrees), which catches, debug-logs and continues.
  - `get_ids_by_timestamp_range` has one other caller, `memory/shutdown_processor.py:489`, inside a broad except. The summary is stored without `source_doc_ids`, as with an empty result.

### Tests (the parent read the whole new file)

- **Producer:**
  - numeric and legacy-page failures raise with exact reasons, and the markers are absent from `reason` and `str`;
  - controls: a failing `count()` still returns the union, malformed bounds → [], and a healthy numeric-plus-legacy mix gives the sorted union.
- **Expander:**
  - a range `RetrievalError` gives an explicit error and a second call hits the store again (not cached);
  - a non-`RetrievalError` range exception is wrapped with the marker absent;
  - an anchor `get_by_id` `RetrievalError` gives an explicit error, not cached;
  - a per-id `RetrievalError` aborts the fetch (`turns == []`);
  - a `ValueError` from the store still propagates out of `expand()`;
  - controls: a genuinely empty range gives the fallback text and IS cached (`r1 is r2`), and a healthy expansion is cached.
- **Failing-first:** 6 failed and 6 passed on the unedited sources.

### Parent rerun (guarded, capped, `-p asyncio`)

- **Focused** (MemAvailable 6,711 MB): the 8 files in §5 gave **163 passed** (8.91 s, 1,183,496 KB).
- **Sweep chunk 1** (6,617 MB, 9 files): **192 passed** (24.59 s, 3,198,260 KB).
- **Sweep chunk 2** (6,728 MB, 9 files): **426 passed** (13.63 s, 1,731,988 KB).
- **Sweep chunk 3** (6,707 MB, `test_upload_retrieval_pool.py`): **28 passed** (6.45 s, 1,115,184 KB).
- **Total:** **809 passed, 0 failed**, matching the worker.
- **Importer set:** the parent's independent grep found 27 `tests/unit` importers of the four modules (8 focused plus 19 sweep), identical to §6.
- **`data/`:** unchanged after every chunk, and `logs/` is absent.
- **ruff 0.14.9:** clean on all three files.
- **Scan:**
  - dm18: found 47, base 79, new 0, stale 32.
  - `MemoryExpander._fetch_conversations_in_range 'return []'` is now in the STALE list (40 STALE rows in total), next to F9a's `get_by_id`.
  - The only other change against the post-F9a scan is dm17's file count (630 → 631).
  - dm01 (7/12/0/5) and dm31 (3/3/0/0) are unchanged. Scan stderr is empty.

### Size

427 changed lines (54 source + 373 test), within the 450 cap.

### Limitations carried forward

- **`_store_summary`:** it still treats a range failure like an empty range. It is read-only here and has no anchor.
- **`_execute_memory_expand`:** no unit test drives it.
- **Store siblings still swallowing:** `search_all`, `query_multiple_collections`, `update_metadata` and `get_collection_stats`.

### Class-guard status

- **CGR-009:** #139 (base) and #122 (`-2`) answered.
- **Still open:**
  - #130 → F10b (`-3`);
  - #121, #147, #148 → F11a-1 (`-4`);
  - #132–#134, #136 → F11b-1 (`-5`);
  - #117–#120, #123, #125, #126, #141–#143 → F12.
- **CGR-010:** open; F10a is next.
