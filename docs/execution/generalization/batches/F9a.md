# F9a — MultiCollectionChromaStore.get_by_id distinguishes a failed read from "not found"

Batch ID: F9a. Request: CGR-20260913-009 (BASE response, anchor #139).
Base commit: 328a8ecea1dae91de16f5974ea71727c4e2e1aa0 (uncommitted tree, no commits made).
Worker started: 2026-09-14.

## 1. Manifest check (before any edit)

```
git status --short   # 67 tracked M, ~30 untracked paths incl. docs/execution/generalization/ (new tree)
git rev-parse HEAD    # 328a8ecea1dae91de16f5974ea71727c4e2e1aa0
git diff --stat       # 67 files changed, 3579 insertions(+), 682 deletions(-)
git remote -v         # origin (fetch) /home/lukeh/Daemon_v1 ; origin (push) DISABLED
```

Assigned-file digests at start (match parent-verified facts in F9a.md brief exactly):
```
e0884db3f2e0bf6ad8af503604cceacb742b3cf6554d2f78b7104f5050542605  memory/storage/multi_collection_chroma_store.py
fbae71f1a5661c1289ccc3db5532ebd52bd49c71367b8eca2ce2aa5a3a323013  memory/claim_tracker.py
a2152df35f33796ba2dd145141adce7255f4163288d70bc99abc23a57adc6495  memory/memory_expander.py
```

Manifest check against `$S/manifest_post_F8b.txt` and `$S/manifest_paths_post_F8b.txt` (211 paths):
- `sha256sum $(cat manifest_paths_post_F8b.txt)` vs `manifest_post_F8b.txt`: `diff` → 0 lines. **0 mismatches.**
- Path listing (git status expanded: untracked directories walked with `find -type f` since `docs/execution/generalization/` is a single untracked-dir line in `git status --short`) vs `manifest_paths_post_F8b.txt`, both sorted: `diff` → 0 lines, 211 == 211. **Empty path diff.**
- The untracked `scan_pre.stderr` at repo root is already present in `manifest_paths_post_F8b.txt` (line 136) — a pre-existing artifact from an earlier batch's manifest state, not created by this worker. Left untouched (no rm/mv available or attempted).

MemAvailable before first chunk: 6669 MB (≥4000, OK). `logs/` absent (confirmed, `ls -ld logs` → No such file or directory).

data/ listing (top level only) matches `$S/data_baseline_f7a_rerun.txt` exactly (7 entries: benchmark_per_case.csv, chroma_db_v4, chroma_multi, embedding_migration_manifest.json, pipeline, user_profile.json, web_search_credits.json). No new/changed entry. `user_profile.json` mtime/size unchanged from baseline.

Manifest check: **PASS**. Proceeding.

## 2. Pre-edit scan (read-only)

`PYTHONPATH=.../scripts/bin python scripts/check_bug_classes.py scan --root .` — exit 1 (gate, as expected; unrelated pre-existing gate findings elsewhere in the tree).
dm18 summary row (matches brief exactly): `dm18_except_returns_empty  gate  123  49  79  0  30`.
The plain-text scan output does not enumerate individual "found"/accepted-debt anchors (only STALE + report-only detail), so #139 does not appear by name pre-edit; the summary row above is the baseline to diff against post-edit.

## 3. New test file and failing-first run

New file: `tests/unit/test_store_get_by_id_outcomes.py` (fakes only — `MultiCollectionChromaStore.__new__` + a fake Chroma collection with a configurable `.get()`; never a real Chroma client). Covers: store contract (raise/None/None/dict for unknown collection), `ClaimIndex.cascade_staleness` through a mocked `get_by_id` (isolating the cascade's own exception handling, with two paired controls), and through the REAL edited store (retention + no message-leak via caplog).

FAILING-FIRST command (digest line + new tests, in one command, UNEDITED source):
```
sha256sum memory/storage/multi_collection_chroma_store.py memory/claim_tracker.py
systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=512M /usr/bin/time -v env \
  DAEMON_TEST_MODE=1 CHROMA_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin \
  python -m pytest -q -p no:cacheprovider -p asyncio tests/unit/test_store_get_by_id_outcomes.py
```
Digests (unchanged from section 1):
```
e0884db3f2e0bf6ad8af503604cceacb742b3cf6554d2f78b7104f5050542605  memory/storage/multi_collection_chroma_store.py
fbae71f1a5661c1289ccc3db5532ebd52bd49c71367b8eca2ce2aa5a3a323013  memory/claim_tracker.py
```
Result: **2 failed, 7 passed** in 1.61s (elapsed 3.56s incl. systemd-run/time overhead; Maximum resident set size 809316 KB ≈ 790 MB). Exit 1.

Failures (exactly as predicted by the brief):
- `TestGetByIdOutcomes::test_failed_read_raises_retrieval_error` — `Failed: DID NOT RAISE <class 'utils.retrieval_outcome.RetrievalError'>` (today `get_by_id` swallows the exception to `None`).
- `TestCascadeStalenessRealStore::test_failed_read_retains_index_entry` — `AssertionError: assert 'doc_1' in {}` (today the swallowed `None` makes `cascade_staleness` treat the doc as deleted and call `remove_document`, dropping the index entry).

The other 7 new tests (unknown-collection/not-found/found store contract, the 3 mocked-get_by_id cascade tests, and the real-store no-message-leak test) already pass today — they pin behavior that is unchanged by the fix or, for the mocked tests, the cascade's own pre-existing exception handling (already correct once `get_by_id` raises at all).

data/ listing after this chunk: unchanged from baseline (`diff` against `$S/data_baseline_f7a_rerun.txt` → 0 lines); `logs/` still absent.

## 4. Source edit

`memory/storage/multi_collection_chroma_store.py` (the two owned regions only):
- Added one import: `from utils.retrieval_outcome import RetrievalError` (placed after the existing import block, before `logger = logging.getLogger(__name__)`).
- `get_by_id`: expanded the docstring (documents the `None` vs `RetrievalError` split) and changed only the `except Exception:` clause from `return None` to `raise RetrievalError(source="chroma_get_by_id", reason=type(e).__name__) from e`. The try block itself, the not-found `return None`, and the found-dict construction are byte-for-byte unchanged — the diff is the docstring plus one line.

memory/claim_tracker.py: **no edit** (read-only in this batch, per the brief — cascade_staleness's own `try/except Exception` already keeps a raised `RetrievalError` from reaching the `remove_document` branch; only the store's swallow needed to stop).

`import utils` sanity check: `PYTHONPATH=.../scripts/bin python -c "import utils; print(utils.__file__)"` → `/home/lukeh/daemon_exec/generalization/utils/__init__.py` (this clone, correct).

## 5. Focused chunk (8 files, post-edit)

Command: the R_common_rules focused template with `tests/unit/test_store_get_by_id_outcomes.py tests/unit/test_claim_tracker.py tests/unit/test_memory_expander.py tests/unit/test_chroma_store_no_autodelete.py tests/unit/test_recent_summaries_fix.py tests/unit/test_upload_retrieval_pool.py tests/unit/test_sep09_storage_repairs.py tests/unit/test_sep09_curation_integrity.py`.
Result: **179 passed**, 120 warnings (pre-existing chromadb/SWIG deprecation noise, unrelated), exit 0, elapsed 9.15s, Maximum resident set size 1210584 KB ≈ 1.15 GB.
data/ after chunk: unchanged (`diff` → 0 lines); `logs/` still absent.

## 6. Sweep — vetting and chunk plan

Remaining tests/unit importers of `memory.storage.multi_collection_chroma_store` / `MultiCollectionChromaStore` / `memory.claim_tracker` / `memory.memory_expander` (grep `-rl`), excluding the 8 files already run in the focused chunk:
`test_backfill_stance.py, test_gate_retrieval_space.py, test_health_transient_retrieval.py, test_hybrid_semantic_score.py, test_proposal_store.py, test_retrieval_pool_caps.py, test_script_apply_guards.py, test_sep10_probe_dump_actions.py, test_synthesis_generator.py, test_thread_store.py, test_safe_json.py` (11 files).

Vetting (real store / default data path / real embedder check) before running:
- `test_backfill_stance.py` — imports the module only to `monkeypatch.setattr(mcs, "MultiCollectionChromaStore", lambda **kw: FakeStore())`; `GraphMemory` pinned to `tmp_path`. Safe.
- `test_gate_retrieval_space.py` — `get_st_model` driven with `SimpleNamespace` fakes only; no store construction. Safe.
- `test_health_transient_retrieval.py` — `UserProfile(temp_profile)` with an explicit `tempfile.mkstemp()` path (never the default dev path); facts-collection retriever tests use a local `_FakeColl`. Safe.
- `test_hybrid_semantic_score.py` — `HybridRetriever.__new__` + `MagicMock()` chroma_store. Safe.
- `test_proposal_store.py` — hand-rolled `MockChromaStore`/`MockChromaCollection` fixtures; no real Chroma. Safe.
- `test_retrieval_pool_caps.py` — `_resolve_embed_device()` (env/torch only, no store) plus `HybridRetriever.__new__`/`MemoryRetriever.__new__` with `MagicMock` stores. Safe.
- `test_script_apply_guards.py` — `monkeypatch.setattr(reclass, "MultiCollectionChromaStore", FakeStore)`; the real class is never instantiated. Safe.
- `test_synthesis_generator.py` — `mock_store = MagicMock()` fixture only. Safe.
- `test_thread_store.py` — hand-rolled `MockChromaStore`/`MockCollection`. Safe.
- `test_safe_json.py` — `ClaimIndex(persist_path=<tmp_path>)`, `UserProfile(profile_path=<tmp_path>)`, `EntityResolver(... aliases_path=<tmp_path>)` — all explicit tmp_path, corrupt-load contract tests. Safe.
- `test_sep10_probe_dump_actions.py` — **excluded from the sweep**: grep match is a single comment string ("mirroring the deployed MultiCollectionChromaStore's relevance_score...") at line 1378; the file has no `import` of the store module and never constructs one. Not a real importer, and it is a ~2000-line file that would waste chunk budget for zero coverage of the changed code. Recorded here rather than run.

None touch `data/`, a real embedder, or the ContextGatherer/UserProfile-default hazard (BR_common_rules NON-UNIT TESTS note); none are on the forbidden list (test_web_search_manager.py, test_prompt_internal_methods.py, test_graph_integration.py — none of these appear).

Sweep chunk plan: 2 chunks of ≤9 (5 + 5), split evenly, well under the memory-chunk limit.

## 7. Sweep runs

Chunk 1 (`test_backfill_stance.py test_gate_retrieval_space.py test_health_transient_retrieval.py test_hybrid_semantic_score.py test_proposal_store.py`): **56 passed**, 1 warning (pre-existing spacy/click deprecation, unrelated), exit 0, elapsed 12.29s, Maximum resident set size 1352276 KB ≈ 1.29 GB.
data/ after chunk: unchanged (`diff` → 0 lines); `logs/` still absent. Guard clear, MemAvailable 6654 MB before the chunk.

Chunk 2 (`test_retrieval_pool_caps.py test_script_apply_guards.py test_synthesis_generator.py test_thread_store.py test_safe_json.py`): **117 passed**, 4 warnings (pre-existing spacy/SWIG deprecation, unrelated), exit 0, elapsed 8.89s, Maximum resident set size 1727540 KB ≈ 1.65 GB.
data/ after chunk: unchanged (`diff` → 0 lines); `logs/` still absent.

Sweep total: **173 passed, 0 failed** across both chunks (56 + 117). Combined with the focused chunk (179 passed) and the new file's own 9 tests already counted there: **grand total across all runs in this batch: 179 + 56 + 117 = 352 passed, 0 failed** (post-edit; the failing-first run is reported separately in section 3).

## 8. ruff

`PYTHONPATH=.../scripts/bin python -m ruff --version` → ruff 0.14.9.
`python -m ruff check memory/storage/multi_collection_chroma_store.py tests/unit/test_store_get_by_id_outcomes.py` → **All checks passed!**

## 9. Post-edit scan and BC-58 sibling search

`python scripts/check_bug_classes.py scan --root .` (read-only) — exit 1 (gate; unrelated pre-existing findings elsewhere in the tree, unchanged in count).

dm18 summary row, before → after:
```
before: dm18_except_returns_empty  gate  123  49  79  0  30
after:  dm18_except_returns_empty  gate  123  48  79  0  31
```
found 49→48 (−1, the fixed anchor leaves the accepted-debt count), base 79 (unchanged), **new 0** (no new dm18 finding introduced anywhere), stale 30→31 (+1). `diff` of the full pre/post scan text shows exactly three changes: the dm18/dm17-files-scanned count line (dm17's "files" column moved 629→630 solely because the new test file now exists in the tree — its own found/base/new/stale counts are byte-identical), the STALE-entries header count 38→39, and one new STALE line:
```
dm18_except_returns_empty: memory/storage/multi_collection_chroma_store.py [MultiCollectionChromaStore.get_by_id] 'return None'
```
This is anchor #139 going STALE as the brief predicted — the class-guard owner removes this row and marks it `confirmed_fixed` with this request ID; not attempted here.

BC-58 sibling-site search (guard covers only the first-observed path — confirm no identical-shape sibling was left silently unfixed): grepped every `except` in `memory/storage/multi_collection_chroma_store.py` and cross-checked against the brief's parent-verified sibling map. Recorded, unedited (owned by later batches or already correct):
- `list_all` (~402) and `get_recent` (~471): no `try/except` at all — already propagate a raw exception (not `RetrievalError`, but never swallowed). Unchanged, out of scope.
- `query_collection` (~783): raises `ValueError` for an unknown collection; its Chroma-read try/except (separate region) is untouched, out of scope.
- `search_all` (~845), `query_multiple_collections` (~859), `update_metadata` (~920, returns `False`), `get_collection_stats` (~942): still swallow to an empty/false/typed-ish result — same BC-20/BC-47 shape as the old `get_by_id`, but each is a DIFFERENT symbol/region owned by F9b or F10 per the manifest split, not this batch's `get_by_id`-only ownership. Left unedited; not silently fixed-and-hidden, not silently ignored — explicitly named here as the sibling-search result.
- `get_ids_by_timestamp_range` (~299): explicitly F9b's anchor, unedited.
- Only ONE `get_by_id` definition exists in the whole tree (`grep -rn "def get_by_id"` → this one site); no duplicate/shadow implementation elsewhere to miss.
No sibling of the SAME symbol (`get_by_id`) was left unfixed; the siblings of the SAME defect SHAPE in this file belong to other in-flight/queued batches per the CGR-009 split and are recorded, not fixed here (per doctrine: "If a site is NOT an instance of the class [assigned to this batch], do not change it for the scanner's sake").

## 10. Privacy / no-network note

No real institution names, IPs, credentials, or network calls anywhere in this batch. The new test file uses only synthetic markers (`SYNTH_MARKER_7f2a9c1b`) and synthetic subjects (O-lower [label substituted by the parent 2026-09-15; the fixture is now `alex` after H03c-2], `biscuit` — already used by the pre-existing `test_claim_tracker.py` fixtures this batch reused the pattern from). No live Chroma client, no SentenceTransformer, no model download, no provider call anywhere in the new tests or the sweep (all vetted in section 6). data/ and logs/ never changed across any chunk (verified after every chunk above).

## 11. Size

Changed lines: source `memory/storage/multi_collection_chroma_store.py` = 11 insertions + 3 deletions = **14**. New file `tests/unit/test_store_get_by_id_outcomes.py` = **185** (all new). **Total = 199 changed lines**, under the ≤300 target and the ≤450 hard stop.

## 12. FIXTURE RULE

No edits to any existing test file. `tests/unit/test_claim_tracker.py` was checked (grep for `side_effect`/`get_by_id`) and contains no test pinning the old "exception → None → delete" swallow — every existing `get_by_id` mock there uses `.return_value` (a genuine found/not-found shape), never `.side_effect` to simulate a raise. The FIXTURE RULE (repair only with a new assertion plus paired control) therefore does not apply; nothing needed repair. All new assertions and their paired controls live entirely in the new file (section 3 above).

## 13. Deviations / escalations

None. No breach of any hard limit occurred (no commit/push/stash/reset, no non-pytest Python beyond the three standing exceptions, no `git add`, no mv/rm, no edits to class-guard-owned or read-only-owned paths, single pytest at a time throughout, all chunks ≤9 files, MemAvailable ≥4000 MB checked before every chunk, data/ and logs/ checked and unchanged after every chunk).

## 14. Milestone

Focused contract green (179 passed post-edit) plus the full sweep green (173 passed, 0 failed); ruff clean; scan confirms exactly the intended #139 anchor moved to STALE with 0 new findings. Response file `docs/execution/generalization/class_guard_responses/CGR-20260913-009.md` written next (this is the BASE response for CGR-20260913-009; see that file for the anchor mapping of the request's other rows to their owning batches).

## 15. Final state check (pre-handoff)

```
git status --short   # unchanged tracked-file set plus: M memory/storage/multi_collection_chroma_store.py,
                      # ?? tests/unit/test_store_get_by_id_outcomes.py,
                      # ?? docs/execution/generalization/batches/F9a.md (this file),
                      # ?? docs/execution/generalization/class_guard_responses/CGR-20260913-009.md (written next)
git rev-parse HEAD    # 328a8ecea1dae91de16f5974ea71727c4e2e1aa0 (unchanged; no commits made)
```
Digests after edit:
```
memory/storage/multi_collection_chroma_store.py: be9b48e56ac4edfb8c3ebb397d51788d5f9b8afa600a39c2ab18f145465fd5f5 (before: e0884db3f2e0bf6ad8af503604cceacb742b3cf6554d2f78b7104f5050542605)
memory/claim_tracker.py: fbae71f1a5661c1289ccc3db5532ebd52bd49c71367b8eca2ce2aa5a3a323013 (unchanged — read-only this batch)
tests/unit/test_store_get_by_id_outcomes.py (new): 9e693dbc5801acf8d67ab2db9d8a2321d0b3c070c5b3902f9c4214783e5e12fa
```
Full `git diff --stat` at handoff: 68 files changed, 3590 insertions(+), 685 deletions(-) — up from the pre-batch 67 files / 3579(+) / 682(-) by exactly this batch's one file / 11(+) / 3(-). `git status --short` at handoff otherwise identical in shape to the pre-batch snapshot in section 1, plus the new `memory/storage/multi_collection_chroma_store.py` M line and the new files (`tests/unit/test_store_get_by_id_outcomes.py`, and `batches/F9a.md` / `class_guard_responses/CGR-20260913-009.md` inside the already-untracked `docs/execution/generalization/` tree). No stray files anywhere outside these owned locations. No commit made; HEAD unchanged at `328a8ecea1dae91de16f5974ea71727c4e2e1aa0`.

**Manifest re-check (after):** re-ran both manifest commands against `$S/manifest_post_F8b.txt`/`manifest_paths_post_F8b.txt` conceptually superseded by this batch's own single tracked-file change (`memory/storage/multi_collection_chroma_store.py`, digest now `be9b48e5...`, expected and intentional); every other one of the 211 baseline paths is unchanged (confirmed by `git status --short` above showing no other new M lines). No other worker's files were touched.

Batch F9a: **COMPLETE**.

## Parent review and integration (2026-09-14)

Accepted as delivered. No COMPLIANCE breach: the failing-first run was recorded before the source edit, and no non-pytest code ran beyond the standing exceptions. There were no FIXTURE RULE edits.

### Manifest

- **Checksums:** `sha256sum -c manifest_post_F8b.txt` has exactly one mismatch, `briefs/PARENT_STATE.md`. That file was edited by the parent after the manifest was taken, while F9a ran.
- **New paths (four):**
  - `memory/storage/multi_collection_chroma_store.py` (`be9b48e5…`, its first modification);
  - `tests/unit/test_store_get_by_id_outcomes.py` (`9e693dbc…`, 185 lines);
  - `class_guard_responses/CGR-20260913-009.md` (`1cd6dd27…`, 203 lines);
  - this file.
- **Unchanged:** `memory/claim_tracker.py` is still `fbae71f1…` (read-only this batch).
- **Git state:** `git diff --cached --name-only` is empty, HEAD `328a8ec` is unchanged, the reflog shows only the clone entry, and the stash list is empty.
- **No new stray files:** the only untracked file at the repository root is `scan_pre.stderr`.
- **`data/`:** the 7-entry baseline (`$S/data_baseline_f7a_rerun.txt`) is unchanged, and `logs/` is absent.
- **Parent edits while F9a ran or after it returned:**
  - `briefs/PARENT_STATE.md` gained the F10 map decisions, the H03 source pointer and F9a's integration;
  - `briefs/F9b.md` gained the re-verified post-F9a line numbers;
  - the F10a/F10b drafts live in the parent scratchpad only.
- **Recorded tree:** `manifest_post_F9a.txt` records the tree after this section.

### Code review (the parent read the full diff)

- **Import:** `RetrievalError`.
- **Docstring:** None means an unknown collection or a genuine not-found, and `RetrievalError` is raised on a failed read.
- **`get_by_id` except:** `except Exception as e: raise RetrievalError(source="chroma_get_by_id", reason=type(e).__name__) from e`, with no log of the exception text.
- **Unchanged:** the unknown-collection and not-found returns, and the found-dict shape. Nothing else in the file changed.
- **Production callers (parent grep of `.get_by_id(` outside `tests/`, `docs/`, `data/` and `.agent_snapshots/`):** only `memory/memory_expander.py` (171, 240, 337, 371, 436) and `memory/claim_tracker.py:294`.
  - `_anchor_fingerprint` (171) already catches.
  - The other four expander sites propagate out of `expand()`, whose only callers (`core/agentic/tools.py` `_execute_memory_expand` and `core/insight/sweep.py`) catch broadly. That interim state is closed by F9b.
  - No API or GUI route calls this method directly.

### Tests (the parent read the whole new file)

- **Store:** a raising `.get` gives `RetrievalError` with source and reason labels, and the message marker is absent from both `reason` and `str`; not-found gives None; found gives the dict; an unknown collection gives None.
- **Cascade with a mocked `get_by_id`:**
  - a raising read does not call `remove_document`, retains the entry, returns no result and does not call `update_metadata`;
  - paired controls: not-found still removes the entry, and a healthy doc still updates and returns one result.
- **Cascade through the real edited store:** the entry is retained, and a caplog marker never reaches the claim-tracker warning.
- **Failing-first:** 2 failed and 7 passed on the unedited store (`e0884db3…`). The failures are the raise test and the real-store retention test.
- **Fixture subject O-lower** (label substituted by the parent 2026-09-15; now `"alex"` after H03c-2)**:** a first name already used 19 times in `tests/unit/test_claim_tracker.py` (2 files in total). This is existing practice, not a new exposure. It is not an institution name or IP.

### Parent rerun (guarded, capped, `-p asyncio`)

- **Focused** (MemAvailable 6,653 MB): the 8 files in §5 gave **179 passed** (8.73 s, 1,201,896 KB).
- **Sweep chunk 1** (6,567 MB): `test_backfill_stance`, `test_gate_retrieval_space`, `test_health_transient_retrieval`, `test_hybrid_semantic_score` and `test_proposal_store` gave **56 passed** (9.60 s, 1,362,596 KB).
- **Sweep chunk 2** (6,579 MB): `test_retrieval_pool_caps`, `test_script_apply_guards`, `test_synthesis_generator`, `test_thread_store` and `test_safe_json` gave **117 passed** (8.94 s, 1,728,148 KB).
- **Total:** **352 passed, 0 failed**, matching the worker.
- **Importer set:** the parent's independent grep found 19 `tests/unit` importers of the four modules. They are the 8 focused files, the 10 sweep files and the excluded `test_sep10_probe_dump_actions.py` (a comment-string match only), so the set is complete.
- **`data/`:** unchanged after every chunk, and `logs/` is absent.
- **ruff 0.14.9:** clean on both files.
- **Scan:**
  - dm18: found 48, base 79, new 0, stale 31.
  - `MultiCollectionChromaStore.get_by_id 'return None'` is now in the STALE list (39 STALE rows in total).
  - dm01 (7/12/0/5), dm17 (36/39/0/3; files 630, one more test file) and dm31 (3/3/0/0) are unchanged, and every gate scanner shows 0 new. Scan stderr is empty.

### Size

199 changed lines (14 source + 185 test), under the 300 target.

### Limitations carried forward

- **Interim expander propagation:** read-only evidence here; F9b owns it.
- **Still swallowing:** the store siblings `search_all`, `query_multiple_collections`, `update_metadata` and `get_collection_stats`. `get_ids_by_timestamp_range` belongs to F9b.

### Class-guard status

- **CGR-009:** #139 answered (base response).
- **Still open:**
  - #122 → F9b (`-2`);
  - #130 → F10b (`-3`; the parent's F10 split supersedes the base response's "F10" mapping);
  - #121, #147, #148 → F11a;
  - #132–#134, #136 → F11b;
  - #117–#120, #123, #125, #126, #141–#143 → F12.
- **Open:** CGR-010.
