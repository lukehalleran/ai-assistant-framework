# F3b — Obsidian note search reports failure distinctly from "no notes"
# (CGR-20260913-008 anchors #101, #102)

Worker evidence packet. Design source:
`docs/execution/generalization/failure_outcome_design.md`, "Decisions per
request" → CGR-008, batch table row F3, and the 2026-09-14 parent amendment
splitting F3 into F3a (reference docs, done) / F3b (Obsidian, this batch).
Request packet (read-only, the only file read in that checkout):
`/home/lukeh/daemon_exec/class_guards/docs/execution/class_guards/requests/CGR-20260913-008.md`.
Response file: `docs/execution/generalization/class_guard_responses/CGR-20260913-008-3.md`
(immutable once written; covers #101-#102 — the base response
(`CGR-20260913-008.md`, F2) covered #110/#111, `-2` (F3a) covered
#103-#106 plus the upload-refusal defect; #107-#109 are F4).

## Scope

`knowledge/obsidian_manager.py`, at exactly the regions the F3b brief named:
the return paths of `get_notes` and `_keyword_search`. New
`tests/unit/test_obsidian_failure_outcomes.py`. Adopts F1's leaf
`utils/retrieval_outcome.py` (`OutcomeList`, `outcome_status`) exactly as F3a
did. Read-only: `utils/retrieval_outcome.py`, the chroma store,
`core/prompt/gatherer_knowledge.py`, `config/**`, every class-guard file.

## 1. Manifest check (before any edit)

```
$ sha256sum -c --quiet $S/manifest_post_F3a.txt          # exit 0, no output
$ { git diff --name-only; git ls-files --others --exclude-standard; } | \
    sort -u | diff - $S/manifest_paths_post_F3a.txt       # exit 0, no output
```

Both passed. The pre-edit tree matched the parent's post-F3a record exactly.

## 2. Pre-edit git state

- `git rev-parse HEAD`: `328a8ecea1dae91de16f5974ea71727c4e2e1aa0` (unchanged
  throughout this batch — no commit made).
- `git remote -v`: `origin /home/lukeh/Daemon_v1 (fetch)`, `origin DISABLED
  (push)`. Not touched.
- `git status --short` (pre-edit): identical to the manifest's recorded
  path set — the carried-over modified/untracked paths from prior
  integrated batches (through F3a), plus F3a's own new paths
  (`knowledge/reference_docs_manager.py` modified,
  `tests/unit/test_refdocs_failure_outcomes.py`,
  `docs/execution/generalization/batches/F3a.md`,
  `docs/execution/generalization/class_guard_responses/CGR-20260913-008-2.md`).
  Neither `tests/unit/test_obsidian_failure_outcomes.py` (new) nor a
  modified `knowledge/obsidian_manager.py` present yet.
- `sha256sum knowledge/obsidian_manager.py`:
  `a2e9baea9b0525ddc9e3d2dd96e8161673b703e3672de67ba72a58244cc01b92` —
  matches both the request packet's recorded source SHA-256 for anchors
  #101/#102 and the brief's "parent-verified facts" exactly.

## 3. Parent-verified facts re-verified

All facts in the F3b brief's "PARENT-VERIFIED FACTS" section were read
directly from the deployed source before editing and matched exactly:
`async def get_notes(self, query, limit=10, include_images=False,
max_images_per_note=3)` (line 662), with the whole body inside one try;
keyword leg `keyword_results = self._keyword_search(query, keyword_limit *
3)` (687); semantic leg `self.chroma_store.query_collection('obsidian_notes',
...)` (690); combine `for note in keyword_results[:keyword_limit]` (736,
which drops any list-subclass status via the slice) then `final_results =
combined[:limit]` (754), then optional image loading; outer
`except Exception as e:` → warning → `return []` at line 781 = anchor #101.
`def _keyword_search(self, query, limit=10)` (783): `collection =
self.chroma_store._get_collection('obsidian_notes')`; `if not collection:
return []` (798); scoring/slicing; `except Exception as e:` → warning →
`return []` at line 927 = anchor #102. Production consumer, read-only here
(F7 migrates it): `core/prompt/gatherer_knowledge.py:642`
`notes = await manager.get_notes(...)`, followed by a list-comprehension
substance filter (line 658-662) which drops status, and `return notes or
[]` (717) which collapses a falsy (empty) `OutcomeList` to a bare `[]`. F1's
exact implemented API (`OutcomeList(items, status=, reason=)`,
`.failed(reason, items=())`, `.unavailable(reason, items=())`,
`outcome_status(value)`) read directly from `utils/retrieval_outcome.py` —
no drift, no `RetrievalError` needed here (same shape as F3a: the manager
converts failures into `OutcomeList.failed(...)` internally). The four
existing tests named in the brief were all found and grepped for both
`obsidian_manager` and `ObsidianManager` — confirmed as the complete
importer set (§8 below).

One fact determined during implementation, matching F3a's finding for the
identical method: `chroma_store.query_collection`
(`memory/storage/multi_collection_chroma_store.py:783-839`) has **no
internal try/except** — a raise from `.query()` (or the `ValueError` for an
unknown collection at line 792) propagates straight to the caller. The
obsidian call path is the same method, called identically
(`self.chroma_store.query_collection('obsidian_notes', query,
n_results=semantic_limit * 2)` at `get_notes` line 690) — so `get_notes`'
semantic leg needs no local try of its own; a semantic-leg raise is already
caught by `get_notes`' own outer `except Exception`, exactly as F3a found
for `get_documents`. Recorded as the sibling determination for F9 in the
response file (no new finding — this confirms F3a's determination extends
to the obsidian call path, since it is the same shared method).

No drift found; no escalation needed.

## 4. Order followed

manifest → pre-edit scan → write test file → failing-first run (digest in
the same command) → edit source → focused run (1 chunk, sweep = focused
set) → ruff → post-edit scan → data listing → response file → this packet.

## 5. Pre-edit scan (read-only, `scripts/check_bug_classes.py scan --root .`)

```
scanner                       mode    files  found  base  new  stale
----------------------------  ------  -----  -----  ----  ---  -----
dm01_raw_substring            gate    396    7      12    0    5
dm16_config_key_reachability  report  395    89     -     -    -
dm17_apply_without_guard      gate    618    36     39    0    3
dm18_except_returns_empty     gate    123    73     79    0    6
dm29_phrase_append_signature  report  0      0      -     -    -
dm31_live_state_default       gate    396    3      3     0    0
catalog                       gate    1      0      0     0    0
```

`dm18` findings for `knowledge/obsidian_manager.py` (digest `a2e9baea…`,
via `--json`): `ObsidianManager.get_notes` (line 781, #101),
`ObsidianManager._keyword_search` (line 927, #102) — exactly the two
candidate anchors, at exactly the lines the request packet and brief named.
The 6 pre-existing STALE dm18 rows are F2's `_tavily_search`/`_tavily_extract`
and F3a's four `ReferenceDocsManager` rows — unrelated to this batch.

## 6. Failing-first proof

Guard checked first (`/proc/comm` pytest guard: clear); MemAvailable 5411 MB
≥ 4000. One command, before any source edit:

```
$ sha256sum knowledge/obsidian_manager.py && \
  PYTHONPATH=.../scripts/bin systemd-run --user --scope -p MemoryMax=6G \
  -p MemorySwapMax=512M /usr/bin/time -v env DAEMON_TEST_MODE=1 \
  CHROMA_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider \
  -p asyncio tests/unit/test_obsidian_failure_outcomes.py
```

Digest: `a2e9baea9b0525ddc9e3d2dd96e8161673b703e3672de67ba72a58244cc01b92`
(matches). Result: **13 failed, 1 passed**, exit 1. Elapsed 0:04.71; Maximum
resident set size 1,069,560 KB.

The 1 passing test on the unedited source is
`TestGathererConsumerUnaffected::test_get_personal_notes_with_failing_manager_returns_empty`
— the "existing behaviour is unchanged" control (an empty return is empty
either way, whether a bare `[]` or a falsy `OutcomeList`), which passes
identically before and after the patch. All 13 other tests fail on the
unedited source exactly as expected: every plain-read status/reason
assertion (`isinstance(..., OutcomeList)` fails on a bare `list`/`[]`), all
`get_notes` hybrid status assertions (keyword-failed-items-kept,
semantic-raises, outer-except, both healthy controls, include_images
survival), and all three privacy tests (`AttributeError: 'list' object has
no attribute 'reason'` on the two that reach a `.reason` read; the
`_keyword_search` one is a plain `assert False` on the `isinstance` check
first).

## 7. Source edit, then focused run

Guard re-checked (clear) before the pytest run; MemAvailable checked before
the chunk (5413 MB ≥ 4000, see §8).

`knowledge/obsidian_manager.py` — 17 changed lines (12 insertions, 5
deletions; see the response file's "What changed" for the per-region
rationale, not duplicated here): one new import
(`utils.retrieval_outcome.OutcomeList, outcome_status`), `get_notes` reading
`kw_status, kw_reason = outcome_status(keyword_results)` immediately after
the keyword call (before the combine slice), its return wrapped in
`OutcomeList`/`OutcomeList(..., status=kw_status,
reason=f"keyword:{kw_reason}")`, its outer `except` now returning
`OutcomeList.failed(type(e).__name__)`; `_keyword_search`'s three return
paths converted to `OutcomeList(scored[:limit])` /
`OutcomeList.unavailable("collection_unavailable")` /
`OutcomeList.failed(type(e).__name__)`, both existing `logger.warning` lines
kept byte-for-byte. No other line changed: ranking, dedup, `limit`
handling, image loading and every consumer are untouched.

## 8. Focused run (1 chunk; sweep = focused set, see below)

`tests/unit/test_obsidian_failure_outcomes.py` plus the 4 existing files the
brief named, confirmed by grep (both `obsidian_manager` and
`ObsidianManager`) to be the complete importer set (MemAvailable 5413 MB ≥
4000):

```
$ PYTHONPATH=.../scripts/bin systemd-run --user --scope -p MemoryMax=6G \
  -p MemorySwapMax=512M /usr/bin/time -v env DAEMON_TEST_MODE=1 \
  CHROMA_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider \
  -p asyncio tests/unit/test_obsidian_failure_outcomes.py \
  tests/unit/test_api_misc.py \
  tests/unit/test_insight_completion_fixes.py \
  tests/unit/test_keyword_anchor_retrieval.py \
  tests/unit/test_sep03_live_probe_fixes.py
```

Result: **155 passed**, 0 failed, 0 skipped, exit 0. Elapsed 0:07.58;
Maximum resident set size 1,652,488 KB (new file: 14 tests; 4 existing
files: 141 tests).

**No existing test required a FIXTURE RULE edit** — verified two ways before
running anything: (1) by reading all four files' obsidian-touching code in
full: `test_api_misc.py:347-348` mocks `ObsidianManager` itself
(`MagicMock(return_value=manager)`) to test `/api/sync-notes`'s
`embed_vault` call, never touching `get_notes`/`_keyword_search`;
`test_insight_completion_fixes.py`'s `TestNoteDateInference` (610-638) only
drives `_infer_note_date` via `ObsidianManager.__new__`, and its
`_window_scan_collection` tests (769-783) exercise a *different* function in
`gui/handlers.py` that calls `store._get_collection` directly, never through
`ObsidianManager`; `test_keyword_anchor_retrieval.py`'s
`TestObsidianProperNounFloor` (228-267) drives the real
`ObsidianManager._keyword_search` end-to-end with `results[0]`/`for r in
results` iteration and `results, "..."` truthiness — all compatible with an
`OutcomeList` (a `list` subclass) unchanged, no raising `side_effect`
anywhere in the class; `test_sep03_live_probe_fixes.py`'s two
`get_personal_notes` tests use a hand-written `_Mgr` class with its own
`get_notes` stub returning a plain `list(notes)`, never touching the real
`ObsidianManager`. (2) confirmed by the green run itself — all 4 files
passed unmodified against the edited source.

## 9. Sweep

Every `tests/` file mentioning `obsidian_manager` or `ObsidianManager`
(grep, not just `import` statements, to catch every possible importer):

```
$ grep -rl "obsidian_manager" tests --include=*.py
$ grep -rl "ObsidianManager" tests --include=*.py
```

Both return exactly the same 4 files the brief already named (the
`obsidian_manager`-only match, `test_sep03_live_probe_fixes.py`, matches via
the `g.obsidian_manager = _Mgr()` attribute name, not an import — read in
§8 and confirmed unaffected). The sweep set and the focused set are
identical, so §8's single chunk satisfies both requirements at once; no
additional pytest run was needed or performed.

## 10. Ruff

`ruff 0.14.9`.

```
$ PYTHONPATH=.../scripts/bin python -m ruff check \
  knowledge/obsidian_manager.py tests/unit/test_obsidian_failure_outcomes.py
All checks passed!
```

## 11. Bug-class scan (read-only, `scripts/check_bug_classes.py scan --root .`)

Post-edit:

```
scanner                       mode    files  found  base  new  stale
----------------------------  ------  -----  -----  ----  ---  -----
dm01_raw_substring            gate    396    7      12    0    5
dm16_config_key_reachability  report  395    89     -     -    -
dm17_apply_without_guard      gate    619    36     39    0    3
dm18_except_returns_empty     gate    123    71     79    0    8
dm29_phrase_append_signature  report  0      0      -     -    -
dm31_live_state_default       gate    396    3      3     0    0
catalog                       gate    1      0      0     0    0
```

`dm18` `found` dropped 73→71 (exactly the 2 anchors fixed), `new` stayed 0
(no new finding introduced anywhere in the tree), `stale` rose 6→8: the 2
new STALE rows are exactly `ObsidianManager._keyword_search` and
`ObsidianManager.get_notes` — the designed class-guard handoff.
`knowledge/obsidian_manager.py` has **zero** live dm18 findings after this
edit (confirmed via `--json`). Full before/after tables and the STALE
listing are in the response file.

## 12. Data directory (top-level, sizes and mtimes)

Identical before the first pytest and after the last (failing-first run,
focused run, and both scans) — no change, nothing written, nothing deleted:

```
total 68
drwxr-xr-x. 1 lukeh lukeh   220 2026-09-14 04:35:38.766908925 -0500 .
drwxr-xr-x. 1 lukeh lukeh   996 2026-09-13 18:07:54.727650398 -0500 ..
-rw-r--r--. 1 lukeh lukeh 53235 2026-09-13 12:28:07.844792269 -0500 benchmark_per_case.csv
drwxr-xr-x. 1 lukeh lukeh     0 2026-09-13 13:09:16.898881228 -0500 chroma_db_v4
drwxr-xr-x. 1 lukeh lukeh   100 2026-09-14 04:35:38.400904050 -0500 chroma_multi
-rw-r--r--. 1 lukeh lukeh   753 2026-09-13 12:28:07.846270384 -0500 embedding_migration_manifest.json
drwxr-xr-x. 1 lukeh lukeh   304 2026-09-13 12:28:07.846364181 -0500 pipeline
-rw-------. 1 lukeh lukeh    50 2026-09-14 04:35:38.763134691 -0500 web_search_credits.json
```

(`chroma_multi/` and `web_search_credits.json` are the pre-existing F2
side-effect entries, carried over unchanged since F3a's snapshot — this
batch's own tests use only `MagicMock` stores, confirmed to touch neither.)

## 13. BC-58 siblings found

- **`chroma_store.query_collection` does not swallow failures** (§3 above)
  — confirms F3a's determination for the identical shared method, this time
  on the obsidian call path (`get_notes` line 690): no internal try/except,
  a raise propagates to `get_notes`' own outer `except`. Not a new finding;
  recorded here (and in the response file) to close out the "confirm for
  the obsidian call path" instruction from the F3b brief. Still relevant for
  F9/consumer-audit awareness, per F3a's original note.
- No other in-file sibling named by the brief (unlike F3a's
  `get_full_document`/`delete_document`/`get_stats`, this batch's brief
  contract does not name any unchanged sibling method — `get_vault_stats`
  and `clear_index` were read and are out of scope: neither reads
  `get_notes`/`_keyword_search`, and both were unaffected by the edit,
  confirmed unchanged in the diff).
- Production consumer, read-only per ownership:
  `core/prompt/gatherer_knowledge.py:642` (`get_notes`) already treats its
  result as a plain list/iterable (`if notes:`, list-comprehension filters,
  `return notes or []`) — behaviour unaffected by the `OutcomeList`
  wrapping, since it stays equal to `[]` and falsy when empty. F7 migrates
  it to read `.status`/`.reason`.

## 14. Privacy / no-network note

No network access, no LLM/paid API call, no daemon restart, no `pip
install`. Every fake exception message and query used to prove the privacy
contract carries a distinctive marker substring (`EXCZQX9`/`TITLEZQX9`), and
`TestPrivacyNoLeakedText` (3 tests) asserts that marker is absent from every
`.reason` string the manager produces on failure, at the plain
`_keyword_search` boundary and both `get_notes` failure paths (keyword-leg
failed, outer-except). `pytest` itself is the only subprocess spawned, via
the required `systemd-run`/`/usr/bin/time` wrapper. The one standing
`python -c "import utils; print(utils.__file__)"` sanity check
`R_common_rules.md` itself instructs (confirming `utils` resolves inside
this clone, not `/home/lukeh/Daemon_v1`) was run once at the start of this
batch, per that instruction — no other non-pytest code was run. Every file
edit used the Edit or Write tool; `git add` was never invoked; every pytest
invocation (including this batch's single failing-first and single focused
run) used the guarded, capped, `-p asyncio` command. No process deviation
to disclose this batch.

## 15. Size

`knowledge/obsidian_manager.py` 17 changed lines (12 insertions, 5
deletions) + `tests/unit/test_obsidian_failure_outcomes.py` 255 lines (new)
= **272 total changed lines** — over the 220-line target but comfortably
under the 450-line hard cap; no trimming needed. The test file is larger
than the two-method source edit because it drives both methods'
plain-read AND hybrid-combine paths (6 `get_notes` scenarios: keyword-failed
items-kept, semantic-raises, outer-except, two healthy controls,
include_images survival) plus the privacy suite and the read-only gatherer
consumer drive — narrower in source-method count than F3a (2 methods vs.
4 + upload refusal) but with proportionally similar per-method test depth.

## 16. Milestone

Focused contract green: 155/155 passed (new file's 14 tests + 4 existing
files' 141 tests), 0 failed, 0 skipped, exit 0. Failing-first: 13/14 failed
on the unedited source (digest confirmed), the 1 pass being the
unchanged-behaviour gatherer-consumer control. Ruff clean on both changed
files. Scan: dm18 `new` stayed 0 before and after; both anchors (#101,
#102) now STALE, 0 live dm18 findings remain in
`knowledge/obsidian_manager.py`. No existing test required a FIXTURE RULE
edit. No `data/` write at any checkpoint. No process deviation.

## 17. Open items / limitations

- The production consumer (`core/prompt/gatherer_knowledge.py:642`
  `get_notes`) still flattens `OutcomeList` to a plain list/falsy check
  (`return notes or []`) — F7 migrates it to read `.status`/`.reason`. This
  is demonstrated directly by
  `TestGathererConsumerUnaffected::test_get_personal_notes_with_failing_manager_returns_empty`,
  which passed both before and after this batch's edit.
- `chroma_store.query_collection` not swallowing its own failures (§3, §13)
  is recorded for F9's awareness (confirming F3a's finding extends to the
  obsidian call path), not this batch's file to change.
- No escalation: the manifest check, pre/post scans, and every test run
  came back exactly as the brief predicted; no drift from the
  parent-verified facts was found; no non-pytest code beyond the one
  brief-mandated sanity check; no `git add`; no unguarded pytest.

## Parent review and integration (2026-09-14)

Accepted as delivered, with no parent edit and no process deviation. This is the first batch since the tightened GIT INDEX AND PYTEST HYGIENE rules, and it ran clean.

### Manifest

- **Checksums:** `sha256sum -c manifest_post_F3a.txt` reports no failures.
- **New paths:** exactly four:
  - `knowledge/obsidian_manager.py` (`b9e6e8e8…`, tracked, newly modified);
  - `tests/unit/test_obsidian_failure_outcomes.py` (`a1f40428…`, 255 lines);
  - `class_guard_responses/CGR-20260913-008-3.md` (`e2ed6465…`);
  - this file.
- **Git state:** `git diff --cached --name-only` is empty, HEAD `328a8ec` is unchanged, push is DISABLED, and `git stash list` is empty.
- **`data/`:** identical to the parent's post-F2 snapshot.
- **Recorded tree:** `manifest_post_F3b.txt` records the tree after this section.

### Code review (the parent read the full 17-line diff)

- **`_keyword_search`:**
  - no collection → `OutcomeList.unavailable("collection_unavailable")`;
  - success → `OutcomeList(scored[:limit])`;
  - except → the warning is kept, then `OutcomeList.failed(type(e).__name__)`.
- **`get_notes`:**
  - `kw_status, kw_reason = outcome_status(keyword_results)` runs immediately after the keyword call, before the slice.
  - Image loading still runs on `final_results`.
  - A failed or unavailable keyword leg returns `OutcomeList(final_results, status=…, reason="keyword:<reason>")`; otherwise `OutcomeList(final_results)`.
  - The outer except returns `OutcomeList.failed(type(e).__name__)`.
- **Cosmetic nit (accepted):** `_keyword_search`'s return annotation still reads `List[Dict[str, Any]]`. `OutcomeList` is a list subclass, so this has no behavioural effect.
- **Semantic leg:** `query_collection` raises and does not swallow (confirmed again; recorded for F9).

### Tests (the parent read the class list and the image test)

- **Keyword search:** failed, unavailable, a succeeded control and a no_results control.
- **Hybrid search:** keyword failed with items kept, semantic raising, keyword raising reaching the outer except, and healthy and healthy-but-empty controls.
- **`test_status_survives_include_images`:** the status stays "failed", the reason carries the "keyword:" prefix, and `image_data` is present on the wrapped item.
- **Privacy:** three marker tests.
- **Deployed consumer:** the gatherer still returns an empty section today (F7 hand-off).

### Parent rerun (guarded, capped, `-p asyncio`, MemAvailable 5,341 MB)

- **Tests:** the new file plus test_api_misc, test_insight_completion_fixes, test_keyword_anchor_retrieval and test_sep03_live_probe_fixes gave **155 passed**, 0 failed (0:07.38, 1,652,600 KB). These are all the tests/ files that mention the module.
- **`data/`:** unchanged.
- **ruff** on both files: all checks passed.
- **Scan:**
  - dm18: found 71, base 79, new 0, stale 8. The 2 new STALE rows are exactly #101/#102 in `knowledge/obsidian_manager.py`; the other six are F2's #110/#111 and F3a's #103-#106.
  - Every other gate scanner shows 0 new.

### Size

272 changed lines (17 source + 255 test), within the 450 cap.

### CGR-008 status

Answered: #110/#111 (F2), #103-#106 plus upload refusal (F3a), #101/#102 (F3b). Open: #107-#109 (F4).
