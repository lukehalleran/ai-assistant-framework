# F8a: memory gatherer sections report a typed failure instead of an empty
# section (recent conversations, semantic memories, user profile, upcoming
# schedule; off-path evidence for facts/reflections)

Worker evidence packet. Design source:
`docs/execution/generalization/failure_outcome_design.md`, "F7 split and
gatherer outcome shape (parent decision, 2026-09-14)", the F8 split and
"Limits recorded now". Request packet (read-only, the only file read in
that checkout): `/home/lukeh/daemon_exec/class_guards/docs/execution/class_guards/requests/CGR-20260913-007.md`.
Brief: `docs/execution/generalization/briefs/F8a.md`. Rules:
`docs/execution/generalization/briefs/R_common_rules.md` (applies in FULL).
Anchors answered: **#87, #88, #90, #91** with code
(`_get_recent_conversations`, `_get_semantic_memories`,
`get_user_profile_context`, `get_upcoming_schedule`); **#85, #86, #89**
with off-path evidence only, no code (`get_recent_facts`, `get_facts`,
`_get_reflections`). BUG_CLASSES: BC-20, BC-47, CM-05. Response file
(immutable once written, written LAST):
`docs/execution/generalization/class_guard_responses/CGR-20260913-007-4.md`.

This file is written INCREMENTALLY per the INTERRUPTION rule: created
right after the manifest check, then a dated entry appended after each
ORDER step.

## 1. 2026-09-14 — Manifest check (before any edit)

Run from the checkout root with
S=/tmp/claude-1000/-home-lukeh-daemon-exec-generalization/1f0f3407-5796-4278-85e4-0c7ba4f50aa9/scratchpad:

- (a) `sha256sum -c --quiet $S/manifest_post_F7c.txt` → prints nothing, exit 0. PASS.
- (b) `{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u | diff - $S/manifest_paths_post_F7c.txt` → prints nothing, exit 0. PASS.

Both checks pass — proceeding.

### Read-only git state (pre-edit)

- `HEAD`: `328a8ecea1dae91de16f5974ea71727c4e2e1aa0` (matches the packet's
  "Base tree" and the brief's parent-verified base).
- `git remote -v`: `origin  /home/lukeh/Daemon_v1 (fetch)` / `origin
  DISABLED (push)`. Not touched.
- `git stash list`: empty. Not touched.
- `git diff --cached --name-only`: empty (0 lines).
- `git status --short`: 105 lines (carried-over modified/untracked paths
  through F7a/F7b/F7c), matching the manifest path-set exactly (check (b)
  above passed) and matching F7c.md's own final recorded count exactly.

### Sanity check (one-time `import utils`, standing exception)

`PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin python -c "import utils; print(utils.__file__)"`
→ `/home/lukeh/daemon_exec/generalization/utils/__init__.py` — resolves
inside this clone, as required.

### Assigned-file digest (ORDER REMINDER: re-verify here, print again with
### the failing-first command)

`sha256sum core/prompt/gatherer_memory.py` →
`2cbe777580a97aa06a35b9d224ca5e56f86ff57b6ff73b59595504d8e34efd68`

Matches the packet's/brief's recorded source SHA-256 for #85-#91 exactly
(`2cbe7775…`). `wc -l core/prompt/gatherer_memory.py` → 1070 lines,
matching the brief's "1070 lines" note.

### Re-verification of the PARENT-VERIFIED FACTS (read `core/prompt/gatherer_memory.py`
### directly in full, offset 1-1071, before any edit)

Every line the brief names matches exactly, no drift:

- **#85** `get_recent_facts` (def 144): try 146-154; producer
  `await self.memory_coordinator.get_recent_facts(limit)` (148); capability
  fallback `return await self.get_facts(limit)` (152); `except Exception`
  153 → `return []` 154-155.
- **#86** `get_facts` (def 157): try 159-166; producer
  `memory_coordinator.get_facts` (161); capability gate 163-164; except 165
  → `return []` 167.
- **#89** `_get_reflections` (def 755): except 797-799 → `return []`.
- **OFF-PATH** (re-confirmed by grep of core, gui, api, scripts, memory,
  utils, knowledge, excluding tests/ and this batch's own new file): no
  production caller invokes the MIXIN's `get_recent_facts`, `get_facts` or
  `_get_reflections`.
  - `core/context_pipeline.py:973` calls `self.memory_system.get_facts` —
    the memory COORDINATOR's own method (a different class).
  - `memory/shutdown_processor.py:1313` calls `mc.get_facts` — same
    coordinator method.
  - `scripts/sample_real_benchmark.py:552` / `sample_real_benchmark_v2.py:555`
    call a retriever's `get_facts`.
  - `memory/memory_coordinator.py:328/335` (`get_recent_facts`/`get_facts`)
    delegate to `self._retriever` (`MemoryRetriever`, a different class).
  - No `core/prompt/builder.py` task is named "facts" or "reflections"
    (bare) — the gather loop uses `_get_summaries_separate` (builder.py:1289)
    and `_get_reflections_separate` (builder.py:1307) instead, confirmed by
    grep: `_get_reflections(` with no trailing `_separat` appears nowhere
    in `core/prompt/builder.py`.
- **#87** `_get_recent_conversations` (def 169; builder task "recent" at
  builder.py:1268):
  - `corpus_manager.get_recent_memories(count=limit)` (177) is unguarded;
  - inner try 183-194 (`except` 193-194) swallows a FALLBACK-only failure —
    confirmed a sibling, left untouched;
  - annotation comprehension 199; `self.memory_id_map[...]` 217;
  - ANCHOR except 227-229 → `return []`.
  - Direct callers OUTSIDE the gather loop, re-verified via
    `grep -n "_get_recent_conversations(" core/prompt/builder.py`:
    1268 (gather loop), 1838 (Step 6.1 top-up), 1969 (recency floor), 2314
    (`_build_lightweight_context`, def 2309).
  - `_build_lightweight_context`'s try spans 2312-2373; its except (2373+)
    returns a DIFFERENT fallback dict carrying `upcoming_schedule`,
    `google_calendar`, `relevant_emails` keys the SUCCESS dict never sets
    (re-read in full, lines 2309-2400) — confirms a raise from #87 would
    divert this caller to the wrong fallback shape, so #87 MUST stay
    typed, never raise.
- **#88** `_get_semantic_memories` (def 489; builder task "memories" at
  builder.py:1274):
  - `if not query: return []` (502-504) is a legit gate;
  - `semantic_memories = []` (510); inner try 511-594 around
    `await self.memory_coordinator.get_memories(...)` (522/528); inner
    `except Exception` 593-594 SWALLOWS a coordinator failure;
  - transforms: `_apply_valence_cap` 590, `_deduplicate_memories` 597,
    slice `result[:limit]` 598, annotation 606; `memory_id_map` 617;
  - ANCHOR outer except 630-634 (with traceback debug) → `return []`.
- **#90** `get_user_profile_context` (def 927; builder task "user_profile"
  at builder.py:1281; its ONLY caller):
  - returns `str` on every path; `if not self.user_profile: return ""`
    (939-941) is a legit gate;
  - producer `self.user_profile.get_context_injection(...)` (944, sync)
    sits in try 943-964;
  - `memory_id_map["PROFILE_CONTEXT"]` 952-959;
  - ANCHOR except 962-964 → `return ""`.
  - Consumers already handle a non-str value: the gather loop stores
    `raw or []`; formatter.py:1708-1709 renders only
    `if user_profile and isinstance(user_profile, str)`.
- **#91** `get_upcoming_schedule` (def 966; builder task "upcoming_schedule"
  at builder.py:1450; its ONLY caller):
  - legit early returns: `SCHEDULE_EXTRACTION_ENABLED` off (983-984), no
    chroma store (988-989), empty collection (993-994), no schedule facts
    (1012-1013);
  - producer `store.query_collection("facts", ...)` (998) raises;
  - success returns the slice `upcoming[:min(limit, SCHEDULE_PROMPT_MAX_EVENTS)]`
    (1066);
  - ANCHOR except 1068-1070 → `return []`.
- `config.app_config` defaults re-verified:
  `SCHEDULE_EXTRACTION_ENABLED=True`, `SCHEDULE_PROMPT_MAX_EVENTS=10`,
  `SCHEDULE_PROMPT_LOOKAHEAD_DAYS=7`.
- `_get_summaries_separate` (312, except 386-388) and
  `_get_reflections_separate` (819, except 893-895) re-confirmed
  always-truthy dict shapes on their own except — NOT in scope, per the
  parent's F8 limitation decision.

No drift found from the brief. Proceeding.

### data/ and logs/ baseline (before the first pytest)

```
$ ls -la --time-style=full-iso data
total 68
drwxr-xr-x. 1 lukeh lukeh   254 2026-09-14 13:26:21.677170907 -0500 .
drwxr-xr-x. 1 lukeh lukeh  1026 2026-09-14 11:21:59.889459216 -0500 ..
-rw-r--r--. 1 lukeh lukeh 53235 2026-09-13 12:28:07.844792269 -0500 benchmark_per_case.csv
drwxr-xr-x. 1 lukeh lukeh     0 2026-09-13 13:09:16.898881228 -0500 chroma_db_v4
drwxr-xr-x. 1 lukeh lukeh   100 2026-09-14 04:35:38.400904050 -0500 chroma_multi
-rw-r--r--. 1 lukeh lukeh   753 2026-09-13 12:28:07.846270384 -0500 embedding_migration_manifest.json
drwxr-xr-x. 1 lukeh lukeh   304 2026-09-13 12:28:07.846364181 -0500 pipeline
-rw-------. 1 lukeh lukeh   598 2026-09-14 13:26:21.672354716 -0500 user_profile.json
-rw-------. 1 lukeh lukeh    50 2026-09-14 04:35:38.763134691 -0500 web_search_credits.json

$ ls -ld --time-style=full-iso logs
ls: cannot access 'logs': No such file or directory
```

Identical to F7c's recorded post-edit baseline and the brief's stated
7-entry baseline (`user_profile.json` unchanged, 598 bytes, mtime
`13:26:21.672354716`; `logs/` absent). No STOP condition triggered.

MemAvailable at this point: 6543 MB (≥4000 required). Pytest guard: clear
(no RUNNING `python -m pytest` process).

## 2. 2026-09-14 — PRE-EDIT SCAN

```
$ PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin python scripts/check_bug_classes.py scan --root . > /tmp/scan_pre_f8a.txt 2> /tmp/scan_pre_f8a.stderr
exit: 1
```
(stderr file has 0 bytes; captured to /tmp only, never redirected into the
repo.)

Full scanner table:

```
scanner                       mode    files  found  base  new  stale
dm01_raw_substring            gate    396    7      12    0    5
dm16_config_key_reachability  report  395    89     -     -    -
dm17_apply_without_guard      gate    627    36     39    0    3
dm18_except_returns_empty     gate    123    54     79    0    25
dm29_phrase_append_signature  report  0      0      -     -    -
dm31_live_state_default       gate    396    3      3     0    0
catalog                       gate    1      0      0     0    0
```

Identical to F7c's recorded post-edit baseline (dm18: found 54, base 79,
new 0, stale 25) — no drift since F7c integration, as expected (no source
file has changed in between).

JSON run (approved exception; stderr 57 lines of stdlib
DeprecationWarning noise from the scanner's own dependency scan, to /tmp
only): `python scripts/check_bug_classes.py scan --root . --json >
/tmp/scan_pre_f8a.json 2> /tmp/scan_pre_f8a_json.stderr`, exit 1. Inspected
via `jq` only (no `python -c`/heredoc used).

dm18 live findings for `core/prompt/gatherer_memory.py` (via `jq`, matching
the packet's anchors exactly):

```
MemoryRetrievalMixin.get_recent_facts line=155           -- #85 (off-path, no code)
MemoryRetrievalMixin.get_facts line=167                  -- #86 (off-path, no code)
MemoryRetrievalMixin._get_recent_conversations line=229  -- #87
MemoryRetrievalMixin._get_semantic_memories line=634     -- #88
MemoryRetrievalMixin._get_reflections line=799           -- #89 (off-path, no code)
MemoryRetrievalMixin.get_user_profile_context line=964   -- #90
MemoryRetrievalMixin.get_upcoming_schedule line=1070     -- #91
```

Exactly the 7 sites this batch's request packet lists (#85-#91) — no
`gatherer_web.py` (#92, F8b) findings inspected here, out of scope.

data/ and logs/ after this step: unchanged from §1's baseline (same 7
entries, same mtimes; `logs/` still absent). No STOP.

## 3. 2026-09-14 — New test file written (Write tool only)

`tests/unit/test_gatherer_outcomes_memory.py` (new, 342 lines via `wc -l`
at this point; grows to 350 after §4's test-fixture fix, see §12/§13 for
the corrected final count).
Drives the deployed `MemoryRetrievalMixin._get_recent_conversations` (#87),
`_get_semantic_memories` (#88), `get_user_profile_context` (#90) and
`get_upcoming_schedule` (#91) directly through a bare-host `_G(coordinator,
user_profile=...)` helper (`MemoryRetrievalMixin.__new__` + attributes —
the `test_sep10_probe_dump_interpretation.py` `_make_r5_gatherer`
precedent; never constructs a real `UserProfile()` or `ContextGatherer()`),
plus the deployed `UnifiedPromptBuilder.build_prompt` /
`_build_lightweight_context` through the `full_builder`/`retrieval_limits`
fake-builder pattern (`test_independent_prompt_audit.py`, also used by
F5/F7a/F7b/F7c). `SimpleNamespace`/`AsyncMock` fakes only — no real Chroma,
corpus store, UserProfile file, embedder or network anywhere.

- `TestRecentConversationsOutcomes` (#87, 4 cases): raising corpus_manager
  → failed/class and `== []`; healthy non-empty → today's items; healthy
  empty → no_results; the fallback-only inner-swallow sibling (contract
  point 1) stays unchanged (a raising fallback with a healthy partial
  corpus_manager result still succeeds).
- `TestLightPathUnchangedByTypedReturn` (#87 direct caller, 1 case): through
  the deployed `_build_lightweight_context` (builder.py:2314), a raising
  recent-memories store still returns the NORMAL light-context dict shape
  (`"web_search_decision"` present, `"upcoming_schedule"` absent — the
  except's DIFFERENT fallback dict is never reached) — proves the typed,
  never-raising return keeps this caller's contract unchanged.
- `TestSemanticMemoriesOutcomes` (#88, 4 cases): raising coordinator
  `get_memories` (inner swallow) → failed/`"retrieval:<class>"` and `== []`;
  an exception outside the inner try (`_deduplicate_memories` overridden to
  raise) → failed/class; empty query → `[]`/no_results; healthy → today's
  items.
- `TestUserProfileOutcomes` (#90, 3 cases): a raising `get_context_injection`
  → the method itself raises (`pytest.raises`); no profile → `""`; through
  the deployed `build_prompt`, `_section_outcomes["user_profile"] ==
  {"status": "failed", "reason": "RuntimeError"}`, `result["user_profile"]`
  is falsy, and the prompt still builds (`"_build_time" in result`).
- `TestUpcomingScheduleOutcomes` (#91, 3 cases): a raising `query_collection`
  → failed/class; the disabled flag → `[]`/no_results; healthy (all-7-weekday
  recurring fixture, deterministic regardless of the run date) → today's
  slice, 1-7 items, each carrying `display_date`.
- `TestOffPathAnchorsEvidence` (#85/#86/#89, 1 case, contract point 5):
  monkeypatches the three MIXIN methods (`get_recent_facts`, `get_facts`,
  `_get_reflections`) on the class to record a call and raise
  `AssertionError`, wires a bare `MemoryRetrievalMixin` instance as
  `context_gatherer` (with a coordinator that answers the REAL,
  unedited-by-this-batch `_get_summaries_separate`/`_get_reflections_separate`
  the builder actually calls), enables `max_reflections`/`max_summaries` in
  the retrieval overrides, and asserts the deployed `build_prompt` succeeds
  with the three spy list staying empty.

Privacy: every raising fixture's exception message and every #88/#91
query carries the marker `F8AMARKQ17_sensitive_detail_must_not_leak`;
every raising test asserts the marker is absent from the resulting
`.reason` or from `str(result)` (folded into the existing tests rather
than a separate privacy test class, per the F7b/F7c precedent).

## 4. 2026-09-14 — FAILING-FIRST (digest printed in the same command, before any source edit)

Guard checked first (clear, no RUNNING `python -m pytest`); MemAvailable
6608 MB (≥4000).

**Process note (disclosed in full, a test-file-only fix, not a FIXTURE
RULE edit — matches the F4/F5/F7b precedent for a bug found in the
batch's OWN new file before the authoritative run):** the first attempt
(before this run) failed 7/16, with one unexpected failure —
`TestOffPathAnchorsEvidence::test_facts_and_reflections_mixin_methods_never_called_through_builder`
hit `build_prompt`'s OUTER error-fallback path (`Prompt building failed:
'MemoryRetrievalMixin' object has no attribute '_get_web_search_results'`),
not the intended assertion. Root cause: `builder.py:1513` calls
`self.context_gatherer._get_web_search_results(...)` directly (synchronously,
to build the coroutine for `asyncio.create_task`) OUTSIDE any try —
unlike a `task.result()` exception (caught per-task by the gather loop),
a missing attribute at COROUTINE-CREATION time crashes the whole
`build_prompt` call into its outer except. Fixed by giving the test's bare
`_G(coordinator)` gatherer instance a working `_get_web_search_results =
AsyncMock(return_value=[])` stub (matching `full_builder`'s own scaffold)
before wiring it as `context_gatherer` — a test-fixture-only change to
this batch's OWN new file, source untouched throughout. Re-verified:
source digest re-confirmed unchanged (`2cbe7775…`) both before and after
this fix.

This is the final, authoritative run:

```
$ sha256sum core/prompt/gatherer_memory.py
2cbe777580a97aa06a35b9d224ca5e56f86ff57b6ff73b59595504d8e34efd68  core/prompt/gatherer_memory.py
$ systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=512M /usr/bin/time -v env \
    DAEMON_TEST_MODE=1 CHROMA_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin python -m pytest -q \
    -p no:cacheprovider -p asyncio tests/unit/test_gatherer_outcomes_memory.py
```

Result: **6 failed, 10 passed**, exit 1, wall 0:06.07, peak RSS
1,070,272 KB.

Failing (6, exactly the sites this batch is meant to fix):
- `TestRecentConversationsOutcomes::test_raising_corpus_manager_is_failed_and_empty` — ANCHOR #87
- `TestSemanticMemoriesOutcomes::test_raising_coordinator_get_memories_is_failed_retrieval_prefixed` — ANCHOR #88 (inner swallow)
- `TestSemanticMemoriesOutcomes::test_exception_outside_inner_try_is_failed_class` — ANCHOR #88 (outer except)
- `TestUserProfileOutcomes::test_raising_get_context_injection_reraises` — ANCHOR #90 (direct: `Failed: DID NOT RAISE`)
- `TestUserProfileOutcomes::test_through_builder_raise_recorded_failed_and_prompt_still_builds` — ANCHOR #90 (builder-level)
- `TestUpcomingScheduleOutcomes::test_raising_query_collection_is_failed_class` — ANCHOR #91

Every failure shape confirms the defect directly: `outcome_status(result)
== ('no_results', '')` instead of `('failed', ...)`; `Failed: DID NOT
RAISE <class 'RuntimeError'>` for #90's direct raise test (the swallow
currently eats the exception); the builder-level #90 test shows
`_section_outcomes["user_profile"] == {'status': 'no_results', 'reason':
''}` instead of `{'status': 'failed', 'reason': 'RuntimeError'}`.

Passing (10, the "existing behaviour is unchanged" controls, correctly
green before any edit): `test_healthy_nonempty_returns_todays_items`,
`test_healthy_empty_is_no_results`, `test_fallback_only_swallow_is_unchanged_sibling`
(#87); `test_build_lightweight_context_stays_on_success_path_when_store_raises`
(#87 light-path control — passes both before and after by design, since
#87 never raises in either state); `test_empty_query_is_no_results`,
`test_healthy_returns_todays_items` (#88); `test_no_profile_returns_empty_string`
(#90); `test_disabled_flag_is_no_results`, `test_healthy_returns_todays_slice`
(#91); `test_facts_and_reflections_mixin_methods_never_called_through_builder`
(off-path evidence — passes both before and after, since none of the
sections it exercises touch the bare facts/reflections methods either
way).

data/ and logs/ after this run:
```
$ ls -la --time-style=full-iso data   # unchanged from §1's baseline (same 7 entries, same mtimes)
$ ls -ld --time-style=full-iso logs   # still absent
```
No STOP condition triggered.

## 5. 2026-09-14 — Source edit (5 small complete Edit calls, ownership scope only)

`core/prompt/gatherer_memory.py`, ownership scope only (one import line
plus the bodies of `_get_recent_conversations`, `_get_semantic_memories`,
`get_user_profile_context` and `get_upcoming_schedule`):

- **Import** (after the `utils.ordered_slice` import): `from
  utils.retrieval_outcome import OutcomeList`.
- **`_get_recent_conversations`** ANCHOR except (227-229): `return []` →
  `return OutcomeList.failed(type(e).__name__)` — ANCHOR #87.
- **`_get_semantic_memories`** (3 sub-edits):
  - `retrieval_err = None` initialized right after `semantic_memories = []`
    (510), before the inner `try:`;
  - inner `except Exception as e:` (593-594) gains one line,
    `retrieval_err = type(e).__name__`, after the existing `logger.warning`
    line (unchanged);
  - the success return (628, `return result`) becomes conditional: `if
    retrieval_err: return OutcomeList(result, status="failed",
    reason=f"retrieval:{retrieval_err}")` then the unchanged `return
    result`;
  - outer ANCHOR except (630-634): `return []` → `return
    OutcomeList.failed(type(e).__name__)` — ANCHOR #88.
- **`get_user_profile_context`** ANCHOR except (962-964): `return ""` → a
  6-line comment plus `raise` (mirrors F7c's `get_narrative_context`
  precedent) — ANCHOR #90. The `if not self.user_profile: return ""` gate
  above the try (939-941) is untouched.
- **`get_upcoming_schedule`** ANCHOR except (1068-1070): `return []` →
  `return OutcomeList.failed(type(e).__name__)` — ANCHOR #91. All four
  legit early returns above the try (983-984, 988-989, 993-994, 1012-1013)
  are untouched.

No other line touched. `get_recent_facts`, `get_facts`, `_get_reflections`,
`_get_summaries_separate`, `_get_reflections_separate` and every other
method in the file are byte-for-byte unchanged (verified by re-reading the
full diff below).

```
$ sha256sum core/prompt/gatherer_memory.py
b994315ce47f5102d7b05d43a8bc0e3a1d7641c6e2418d8384e3098f4677e640  core/prompt/gatherer_memory.py
$ git diff --stat core/prompt/gatherer_memory.py
 core/prompt/gatherer_memory.py | 25 +++++++++++++++++++++----
 1 file changed, 21 insertions(+), 4 deletions(-)
```

This is the ONLY batch touching `core/prompt/gatherer_memory.py` in this
checkout (unlike `gatherer_knowledge.py`, which F7a/F7b/F7c edited
sequentially) — `git diff --stat` here is authoritative and isolated, no
cross-check needed. **25 changed lines** (21 insertions, 4 deletions).

data/ and logs/: unchanged from §1's baseline. No STOP.

## 6. 2026-09-14 — Focused chunk (1 chunk of 6, per the brief)

Guard clear, MemAvailable 6587 MB (≥4000):

```
$ systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=512M /usr/bin/time -v env \
    DAEMON_TEST_MODE=1 CHROMA_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin python -m pytest -q \
    -p no:cacheprovider -p asyncio \
    tests/unit/test_gatherer_outcomes_memory.py \
    tests/unit/test_section_outcomes.py \
    tests/unit/test_prompt_timeout.py \
    tests/unit/test_sep10_probe_dump_interpretation.py \
    tests/unit/test_independent_prompt_audit.py \
    tests/unit/test_light_prompt_path.py
```

Result: **228 passed**, 0 failed, 0 skipped, exit 0, wall 0:15.20, peak RSS
1,713,848 KB. 5 warnings, all pre-existing (3 SWIG DeprecationWarnings + 1
pre-existing `RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call'
was never awaited` from `test_section_outcomes.py`'s own
`TestGatherExceptionMarksAllFailedGatherError` case (a file this batch
does not touch) + 1 pre-existing `websockets.legacy` DeprecationWarning
from `test_sep10_probe_dump_interpretation.py`), unrelated to this batch.

**No existing test required a FIXTURE RULE edit.** All 5 brief-named
existing files passed unmodified against the edited source:
`test_section_outcomes.py` and `test_prompt_timeout.py` assert on
`_section_outcomes`/gathered values for `recent`/`user_profile`/`web_search`
under healthy or forced-exception fakes — never on #87/#88/#90/#91's own
except-branch shape, so an empty `OutcomeList` compares `== []` exactly
like the old bare `[]`; `test_sep10_probe_dump_interpretation.py`'s
`TestGathererMemoryProducerAnnotation` drives `_get_recent_conversations`/
`_get_semantic_memories` on their HEALTHY paths only (never raises a
producer), unaffected by the anchor-except changes;
`test_independent_prompt_audit.py` never triggers a producer exception in
these four methods; `test_light_prompt_path.py`'s
`test_build_prompt_returns_lightweight_context` mocks
`_build_lightweight_context` itself, never reaching the real
`_get_recent_conversations`.

data/ and logs/ after this chunk: unchanged (7 entries, same mtimes;
`logs/` absent). No STOP.

## 7. 2026-09-14 — SWEEP set identified

`grep -rl "gatherer_memory\|MemoryRetrievalMixin\|core\.prompt\.context_gatherer" tests/unit/`
→ 13 files. Widened with the F7a/F7b/F7c-precedent check for the bare
class name `ContextGatherer`: `grep -rl "ContextGatherer" tests/unit/` →
2 more candidates beyond the 13.
- `tests/unit/test_prompt_compat.py` is a genuine importer
  (`from core.prompt import ContextGatherer` at line 21 — added).
- `tests/unit/test_upload_retrieval_pool.py` is NOT an importer (confirmed
  by reading it: only imports `KnowledgeRetrievalMixin` from
  `gatherer_knowledge.py`; the one `ContextGatherer` mention (line 4) is a
  docstring root-cause note, not an import or construction — excluded).
- `tests/unit/test_ordered_slice_guard.py` is NOT an importer (confirmed
  by reading it: imports only `ast`/`re`/`collections`/`pathlib`, and the
  one `context_gatherer.py`/`ContextGatherer._bounded` mention is a
  STRING inside an allowlist tuple — excluded, matching F7a's/F7b's/F7c's
  identical finding for the same file).

Union (14) minus the 2 already run in the focused chunk
(`test_gatherer_outcomes_memory.py`, `test_sep10_probe_dump_interpretation.py`)
minus the permanently excluded `tests/unit/test_graph_integration.py`
(TEST EXCLUSION) = **10 files** for the sweep, chunked ≤9.

**Safety check (NON-UNIT TESTS / ContextGatherer-UserProfile hazard,
before running anything):** `grep -n "ContextGatherer("`, `grep -n
"UserProfile("` and `grep -ln "spec=\[\]"` over all 10 sweep files:
- `tests/unit/test_session_diff.py:28` (`ContextGatherer(` — plain
  `MagicMock()` coordinator, `hasattr(mc, 'user_profile')` is `True`,
  confirmed safe by the parent in F7a's review, re-confirmed here by
  reading the exact call site again).
- `tests/unit/test_proposal_filter.py:405,421` (`ContextGatherer(` — both
  `mock_coordinator = MagicMock()`, plain and unrestricted, confirmed by
  reading lines 400/416; `hasattr(mc, 'user_profile')` is `True`).
- `tests/unit/test_codex_followups.py:42,51` (`UserProfile(profile_path=
  str(p))`, `p = tmp_path / "user_profile.json"` — explicit, tmp_path-
  scoped path, not the default dev path. Safe).
- No `MagicMock(spec=[])` (the specific hazard shape) anywhere in the 10
  files.
All 10 files clear to run.

Chunked ≤9: Chunk A (9): test_codex_followups.py,
test_gate_min_results_cap.py, test_narration_turn_audit_fixes.py,
test_prompt_builder_self_report_trim.py, test_proposal_filter.py,
test_sep09_live_controls.py, test_sep12_repository_status_context.py,
test_session_diff.py, test_wiki_disambiguation_filter.py. Chunk B (1):
test_prompt_compat.py.

## 8. 2026-09-14 — Sweep chunks A-B

Guard clear before each; MemAvailable checked before each (6642 MB, 6625 MB
— both ≥4000).

- **Chunk A** (9 files) →
  `PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=512M /usr/bin/time -v env DAEMON_TEST_MODE=1 CHROMA_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider -p asyncio tests/unit/test_codex_followups.py tests/unit/test_gate_min_results_cap.py tests/unit/test_narration_turn_audit_fixes.py tests/unit/test_prompt_builder_self_report_trim.py tests/unit/test_proposal_filter.py tests/unit/test_sep09_live_controls.py tests/unit/test_sep12_repository_status_context.py tests/unit/test_session_diff.py tests/unit/test_wiki_disambiguation_filter.py`
  → **178 passed**, 0 failed, exit 0, wall 0:12.08, peak RSS 1,701,984 KB.
  8 warnings: 3 pre-existing SWIG DeprecationWarnings + 5 pre-existing
  `RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never
  awaited` from `test_prompt_builder_self_report_trim.py`'s own
  `session_reflections.sort` line in `core/prompt/builder.py:1679` — a
  file this batch does not touch (same warning every prior F7 sub-batch's
  chunks recorded). `data/`/`logs/` checked immediately after: unchanged
  (7 entries, same mtimes; `logs/` absent). No STOP.
  - `test_gate_min_results_cap.py::TestChainWiring::test_gatherer_passes_intent_budget`
    does `inspect.getsource(gm.MemoryRetrievalMixin._get_semantic_memories)`
    and asserts `"min_gated=limit" in src` — that line (inside the inner
    try, unchanged by this batch's edit) is still present; passed
    unmodified.
- **Chunk B** (1 file) → **7 passed**, 0 failed, exit 0, wall 0:05.26, peak
  RSS 1,066,340 KB. 3 warnings, all pre-existing SWIG DeprecationWarnings.
  `data/`/`logs/` checked immediately after: unchanged. No STOP.

**Sweep total: 185 passed, 0 failed, 0 skipped** across 10 files.
**Grand total, this batch (focused + sweep): 228 + 185 = 413 passed, 0
failed, 0 skipped.** No existing test required a FIXTURE RULE edit
anywhere in the sweep either — every one of the 10 files passed
unmodified.

## 9. 2026-09-14 — ruff

```
$ PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin python -m ruff check \
    core/prompt/gatherer_memory.py tests/unit/test_gatherer_outcomes_memory.py
All checks passed!
```
ruff 0.14.9.

## 10. 2026-09-14 — Post-edit scan (stderr to /tmp only)

```
$ PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin python scripts/check_bug_classes.py scan --root . > /tmp/scan_post_f8a.txt 2> /tmp/scan_post_f8a.stderr
exit: 1   (pre-existing STALE rows across dm01/dm17/dm18, same as every prior batch)
stderr: 0 bytes
```

Full scanner table:

```
scanner                       mode    files  found  base  new  stale
dm01_raw_substring            gate    396    7      12    0    5
dm16_config_key_reachability  report  395    89     -     -    -
dm17_apply_without_guard      gate    628    36     39    0    3
dm18_except_returns_empty     gate    123    50     79    0    29
dm29_phrase_append_signature  report  0      0      -     -    -
dm31_live_state_default       gate    396    3      3     0    0
catalog                       gate    1      0      0     0    0
```

Compared to pre-edit (§2): dm18 `found` dropped **54 → 50** (exactly the 4
anchors fixed: #87, #88, #90, #91), `new` is **0** across every scanner
(no new finding introduced anywhere in the tree by this edit), `stale`
rose **25 → 29** (the 4 newly-STALE rows below, plus the 25 pre-existing
ones unrelated to this batch). `dm17_apply_without_guard`'s `files` count
ticked 627→628 (this scanner's file selection includes `tests/`, and this
batch adds one new test file; its own `found`/`new`/`stale` are
unchanged, a benign files-processed count effect, not a finding).

JSON run (approved exception; stderr 61 lines of stdlib
DeprecationWarning noise, to /tmp only): `python scripts/check_bug_classes.py
scan --root . --json > /tmp/scan_post_f8a.json 2> /tmp/scan_post_f8a_json.stderr`,
exit 1.

dm18 findings for `core/prompt/gatherer_memory.py`, live vs STALE (via
`jq`/grep — the expected split, exactly as designed):

**Now STALE (removed from the live report, present only in the STALE
section)** — #87, #88, #90, #91:
```
dm18_except_returns_empty: core/prompt/gatherer_memory.py [MemoryRetrievalMixin._get_recent_conversations] 'return []'   -- #87
dm18_except_returns_empty: core/prompt/gatherer_memory.py [MemoryRetrievalMixin._get_semantic_memories] 'return []'      -- #88
dm18_except_returns_empty: core/prompt/gatherer_memory.py [MemoryRetrievalMixin.get_upcoming_schedule] 'return []'       -- #91
dm18_except_returns_empty: core/prompt/gatherer_memory.py [MemoryRetrievalMixin.get_user_profile_context] 'return ""'   -- #90
```

**Still LIVE (unchanged, by design — off-path, no code)** — #85, #86, #89:
```
MemoryRetrievalMixin.get_recent_facts line=156   -- #85
MemoryRetrievalMixin.get_facts line=168          -- #86
MemoryRetrievalMixin._get_reflections line=810   -- #89
```
(Line numbers shifted by +1 from the pre-edit 155/167/799 — the 21
inserted lines above them in the file, minus the 4-line net reduction
inside `get_user_profile_context`'s except that sits BETWEEN #86 and #89
but after #85/#86, account for the small shifts; content and symbol names
are otherwise byte-for-byte identical, confirmed by reading the file at
the new line numbers.)

This is the designed handoff (F7a/F7b/F7c precedent): the class-guard
owner removes the 4 newly-STALE rows and marks them `confirmed_fixed`
with this request ID after integration; #85/#86/#89 stay `accepted_debt`
(evidence-only, by design) until the class-guard owner decides between a
detector change and continued accepted debt, per the request packet's own
"Class-guard acceptance" section.

data/ and logs/ after the scan: identical to §1's baseline (7 entries,
`user_profile.json` unchanged); `logs/` still absent. No STOP.

## 11. BC-58 sibling search (off-path evidence, contract point 5)

`grep -rn "\.get_recent_facts(\|\.get_facts(\|\._get_reflections(" --include=*.py .`,
excluding `tests/` and `gatherer_memory.py` itself, re-confirms the
packet's off-path claim with no new finding:

```
core/context_pipeline.py:973       self.memory_system.get_facts(...)        -- COORDINATOR method, different class
memory/memory_coordinator.py:328   self._retriever.get_recent_facts(limit)  -- delegates to MemoryRetriever
memory/memory_coordinator.py:335   self._retriever.get_facts(query, limit)  -- delegates to MemoryRetriever
memory/shutdown_processor.py:1313  mc.get_facts(synthetic_query, limit=5)   -- COORDINATOR method, different class
scripts/sample_real_benchmark.py:552      retriever.get_facts(...)          -- MemoryRetriever, different class
scripts/sample_real_benchmark_v2.py:555   retriever.get_facts(...)          -- MemoryRetriever, different class
tests/benchmarks/retrieval_benchmark.py:140  self.retriever.get_facts(...)  -- MemoryRetriever, different class (test-only)
```

No production caller anywhere in the tree invokes the MIXIN's
`get_recent_facts`, `get_facts` or `_get_reflections`. `core/prompt/builder.py`
has no task named "facts" or bare "reflections" — the gather loop uses
`_get_summaries_separate` (builder.py:1289) and `_get_reflections_separate`
(builder.py:1307) instead (re-confirmed by grep: no `_get_reflections(`
call without a trailing `_separat` prefix appears in `builder.py`).
`TestOffPathAnchorsEvidence` (§3/§4) proves this at the deployed-builder
level: the three MIXIN methods are monkeypatched to record a call and
raise, `_get_summaries_separate`/`_get_reflections_separate` are enabled
and run for real (unedited by this batch), and the build succeeds with
zero calls recorded.

**Within `gatherer_memory.py` itself:** every other `except: return
[]`/`return ""` site in the file (#85, #86, #89) is exactly this off-path
case; nothing remains unaccounted for — the post-edit scan (§10) confirms
these are the only 3 live dm18 findings left in the file.

## 12. Privacy / no-network note

No network access, no LLM/paid API call, no daemon restart, no `pip
install`, no real ChromaDB/embedder/UserProfile file (every coordinator/
profile/chroma-store in the new test file is a `SimpleNamespace`/
`AsyncMock`/hand-built fake; `_G()` never constructs a real `UserProfile()`
or `ContextGatherer()`). Every raising fixture's exception message and
every #88/#91 query carries the distinctive marker
`F8AMARKQ17_sensitive_detail_must_not_leak`; every raising test asserts
the marker is absent from the resulting `.reason` (#87, #88 ×2, #91) or
from `str(result)`/`str(str(result))` (#87's light-path control, #90's
builder-level test) — folded into the existing tests rather than a
separate privacy test class, per the F7b/F7c precedent. Every reason this
batch's source produces is either `type(e).__name__` (an exception class
name) or the fixed `f"retrieval:{retrieval_err}"` label (itself built
from `type(e).__name__`, never `str(e)`) — enforced structurally: no site
in this batch's diff ever calls `str(e)` when building a `reason`. The one
standing `python -c "import utils; print(utils.__file__)"` sanity check
was run once at the top of this session (§1); no other non-pytest code
ran anywhere in this batch. Every file edit used the Edit or Write tool;
`git add` was never invoked; every pytest invocation (failing-first, the
focused chunk, and both sweep chunks) used the guarded, capped,
`-p asyncio` command, preceded by a `/proc/comm` guard check and a
MemAvailable check.

**Process deviations (disclosed in full):** one test-fixture-only
correction to this batch's OWN new file before the authoritative
failing-first run shipped (§4 — `TestOffPathAnchorsEvidence` needed a
working `_get_web_search_results` stub on its bare gatherer instance to
avoid `build_prompt`'s outer error-fallback path, a pre-existing builder
behaviour this batch's edit does not touch or cause). No FIXTURE RULE
edit to any EXISTING test file. One reporting-only slip: §3's initial
`wc -l` (342) was recorded immediately after the Write call but BEFORE
the §4 test-fixture fix above grew the file to 350 lines; every digest
and pytest run in this packet and the response file was taken from the
FINAL 350-line file (the fix landed before the authoritative failing-first
run), so no evidence is stale — only §3's descriptive line count was
briefly wrong, caught and corrected (along with §13's size arithmetic and
§17's final-state note) before this packet or the response file were
treated as final. No non-pytest Python of any kind ran beyond the three
standing exceptions (ruff, the read-only scan, and the one `import utils`
sanity check). No source edit occurred before the failing-first proof was
recorded. No file was created outside this batch's ownership
(`tests/unit/test_gatherer_outcomes_memory.py`,
`docs/execution/generalization/batches/F8a.md`, and the response file
below). No INTERRUPTION occurred in this session.

## 13. Size

`core/prompt/gatherer_memory.py`: **25 changed lines** (21 insertions, 4
deletions — `git diff --stat`, authoritative and isolated since this is
the only batch touching this file, §5).
`tests/unit/test_gatherer_outcomes_memory.py`: **350 lines** (new, `wc -l`
— the §3 write recorded 342 before §4's test-fixture fix added the
`_get_web_search_results` stub + comment to `TestOffPathAnchorsEvidence`;
350 is the final, post-fix count, matching the SHA-256 in this packet and
the response file, both computed after that fix — disclosed as a process
note in §12).

**Total: 25 + 350 = 375 changed lines** — under the 380-line soft target
and comfortably under the 450-line hard cap. `docs/execution/generalization/batches/F8a.md`
and the response file are evidence/response artifacts, not counted toward
the cap, per precedent (F4.md, F7a.md, F7b.md, F7c.md).

## 14. Contract — how each of the 7 points is implemented

1. **#87 (`_get_recent_conversations`): ANCHOR except → `OutcomeList.failed(type(e).__name__)`, log line unchanged; the fallback-only inner swallow (183-194) unchanged, recorded as a sibling.**
   §5's single-line swap. Proven by `TestRecentConversationsOutcomes::test_raising_corpus_manager_is_failed_and_empty`
   (the anchor) and `::test_fallback_only_swallow_is_unchanged_sibling`
   (the sibling — a raising fallback with a healthy partial corpus_manager
   result still succeeds, unaffected). The light-path proof (contract's
   "#87 must return typed, never raise" requirement) is
   `TestLightPathUnchangedByTypedReturn::test_build_lightweight_context_stays_on_success_path_when_store_raises`:
   through the DEPLOYED `_build_lightweight_context` (builder.py:2309),
   a raising recent-memories store still produces the NORMAL success dict
   (`"web_search_decision"` present) rather than the except's DIFFERENT
   fallback dict (`"upcoming_schedule"` present) — proving the typed,
   never-raising contract keeps builder.py:2314's caller unchanged. (The
   two other direct callers, builder.py:1838 and :1969, are already
   locally try/excepted around their own call sites — re-verified in §1 —
   and unaffected by a typed-vs-bare-empty return either way, since both
   only check truthiness/length.)
2. **#88 (`_get_semantic_memories`): inner except records `retrieval_err`; success return wraps with `retrieval:<class>` when set; outer ANCHOR except → `OutcomeList.failed(type(e).__name__)`; `if not query` stays `[]`.**
   §5's three sub-edits. Proven by
   `TestSemanticMemoriesOutcomes::test_raising_coordinator_get_memories_is_failed_retrieval_prefixed`
   (inner swallow → `failed`/`"retrieval:ValueError"`),
   `::test_exception_outside_inner_try_is_failed_class` (a raising
   `_deduplicate_memories`, OUTSIDE the inner try → `failed`/`RuntimeError`
   via the OUTER anchor), `::test_empty_query_is_no_results` (the legit
   gate, untouched), `::test_healthy_returns_todays_items` (unaffected
   healthy path).
3. **#90 (`get_user_profile_context`): ANCHOR except keeps its warning log, then re-raises; the `not self.user_profile` gate still returns `""`.**
   §5's comment + `raise`. Proven directly:
   `TestUserProfileOutcomes::test_raising_get_context_injection_reraises`
   (`pytest.raises(RuntimeError)`),
   `::test_no_profile_returns_empty_string`. Proven at the builder level
   (the brief's explicit requirement):
   `::test_through_builder_raise_recorded_failed_and_prompt_still_builds`
   drives the DEPLOYED `UnifiedPromptBuilder.build_prompt` with the real
   bound mixin method wired as `context_gatherer.get_user_profile_context`
   — the prompt still builds (`"_build_time" in result`),
   `result["_section_outcomes"]["user_profile"] == {"status": "failed",
   "reason": "RuntimeError"}` (F5's existing try/except at
   builder.py's gather loop, unedited by this batch, now actually
   exercises its except branch for this method for the first time), and
   `result["user_profile"]` is falsy.
4. **#91 (`get_upcoming_schedule`): ANCHOR except → `OutcomeList.failed(type(e).__name__)`; every legit early return stays `[]`.**
   §5's single-line swap. Proven by
   `TestUpcomingScheduleOutcomes::test_raising_query_collection_is_failed_class`,
   `::test_disabled_flag_is_no_results` (legit gate, untouched),
   `::test_healthy_returns_todays_slice` (unaffected healthy path, an
   all-7-weekday recurring fixture deterministic regardless of the run
   date).
5. **#85, #86, #89: NO code change; off-path, evidence only.**
   §11's grep re-confirmation (byte-for-byte matching the packet's own
   evidence) plus `TestOffPathAnchorsEvidence` (§3/§4): the deployed
   `build_prompt`, with `_get_summaries_separate`/`_get_reflections_separate`
   genuinely enabled and run (unedited by this batch), never calls
   `get_recent_facts`, `get_facts` or `_get_reflections` — confirmed by a
   monkeypatched spy on the class that would both record a call AND raise
   `AssertionError` if reached, and the recorded call list stays empty
   while the build still succeeds. §10 confirms all three anchors stay
   LIVE in the post-edit scan (0 change), matching the "by design" scan
   expectation exactly.
6. **No change to transforms, dedup, caps, annotation, memory_id_map, log text, `_get_summaries_separate`/`_get_reflections_separate`, or any other method.**
   Verified by re-reading the full diff (§5 — only the 5 named regions
   changed) and by every existing test in the focused (§6) and sweep (§8)
   chunks passing unmodified (0 of 15 files touched under the FIXTURE
   RULE), including `test_gate_min_results_cap.py`'s source-text pin on
   `_get_semantic_memories`'s unchanged `min_gated=limit` line and
   `test_sep10_probe_dump_interpretation.py`'s `_annotate_memory_item_claim`
   wiring tests (all pass unmodified — the annotation call sites, at 199
   and 606, are untouched).
7. **Privacy: reasons are constant labels or exception class names only.**
   Every `reason` this batch produces is `type(e).__name__` or
   `f"retrieval:{type(e).__name__}"` — never `str(e)` or query text.
   Proven by §12's marker-based tests (folded into the raising tests
   rather than a separate class, per the F7b/F7c precedent).

## 15. Milestone

Focused contract green: 228/228 passed (new file's 16 tests + 5 existing
files' 212 tests), 0 failed. Sweep green: 178+7 = 185/185 across 10 files,
0 failed (1 file, `test_graph_integration.py`, permanently excluded per
TEST EXCLUSION; 2 candidates confirmed not genuine importers and
excluded: `test_upload_retrieval_pool.py`, `test_ordered_slice_guard.py`).
Combined: **413 passed, 0 failed, 0 skipped** across 16 files this batch
ran (1 new file, 5 focused existing files, 10 sweep existing files).
Failing-first (authoritative run): 6/16 failed on the unedited source
(digest `2cbe7775…` confirmed), the 10 passes being exactly the
unaffected-by-design controls (including the off-path evidence test and
the #87 light-path control, both pass identically before and after by
design). One test-fixture-only correction to this batch's own new file
before that authoritative run (§4/§12), no FIXTURE RULE edit to any
existing test. Ruff clean on both changed/new files. Scan: dm18 `found`
dropped 54→50 (exactly the 4 code-fixed anchors), `new` stayed 0 for
every gate scanner both before and after, and `core/prompt/gatherer_memory.py`
now has exactly 3 live dm18 findings left (#85, #86, #89 — by design, no
code). No `data/` write at any of the 11 explicit checkpoints (before the
first pytest, after the pre-edit scan, after failing-first, after the
focused chunk, after each of the 2 sweep chunks, after the post-edit
scan) — `logs/` absent throughout.

## 16. Open items / limitations

- **#85, #86, #89** (`get_recent_facts`, `get_facts`, `_get_reflections`)
  are answered with evidence only, no code — the class-guard owner
  decides between a detector change (excluding these from dm18, since
  they are structurally off-path) and continued `accepted_debt`, per the
  request packet's own "Class-guard acceptance" section.
- **The fallback-only inner swallow** in `_get_recent_conversations`
  (183-194) is a sibling, left unchanged (§11's `TestRecentConversationsOutcomes::test_fallback_only_swallow_is_unchanged_sibling`
  proves it is unaffected) — a future batch touching that specific
  fallback path would need its own evidence.
- **`_get_summaries_separate`** (312, except 386-388) and
  **`_get_reflections_separate`** (819, except 893-895) remain
  always-truthy dict shapes on their own except (`{"recent": [...],
  "semantic": [...]}`), so the builder's `_section_outcomes` reads a
  failure inside either as `succeeded`/`no_results` today — this is the
  parent's F8-Ready decision, recorded as a limitation, not fixed here
  (needs a builder change to read a dict section's status; per
  `failure_outcome_design.md`, "Limits recorded now").
- **#92** (`core/prompt/gatherer_web.py` `_get_web_search_results`) is out
  of this batch's ownership — F8b, response `CGR-20260913-007-5`.
- No escalation: the manifest check, the pre-edit scan, the post-edit
  scan, and every test run came back exactly as the brief predicted; the
  only deviation from a strictly literal reading of the ORDER section is
  the one test-fixture-only correction disclosed in §4/§12, which never
  touched an existing test or the source.

## 17. 2026-09-14 — Final state (pre-handoff)

- `git status --short`: 107 lines (was 105 at F7c's final recorded state
  — +1 for `core/prompt/gatherer_memory.py` transitioning from unmodified
  to `M`, +1 for the new `tests/unit/test_gatherer_outcomes_memory.py`;
  `docs/execution/generalization/batches/F8a.md` lands inside the
  already-untracked `docs/execution/generalization/` directory, which
  `git status --short` reports as one line regardless of file count,
  matching F7a's/F7b's/F7c's own identical observation).
  `git diff --cached --name-only`: empty. HEAD:
  `328a8ecea1dae91de16f5974ea71727c4e2e1aa0` (unchanged throughout — no
  commit, no `git add` in any form). `git stash list`: empty, not
  touched. `git remote -v`: `origin /home/lukeh/Daemon_v1 (fetch)`,
  `origin DISABLED (push)` — not touched, unchecked again here only for
  the record.
- **Manifest delta** (against `manifest_paths_post_F7c.txt`): exactly 3
  new/changed paths — `core/prompt/gatherer_memory.py` (modified,
  `2cbe7775…` → `b994315c…`), `tests/unit/test_gatherer_outcomes_memory.py`
  (new, 350 lines — §13's corrected final count), `docs/execution/generalization/batches/F8a.md`
  (new, this file).
- **Size:** 25 (source) + 350 (new test file) = **375 changed lines**,
  within the 380 soft target and the 450 hard cap (§13).
- **Final data/ and logs/:** identical to the §1 baseline (7 entries,
  `user_profile.json` unchanged at 598 bytes / mtime
  `13:26:21.672354716`); `logs/` absent. No STOP at any point in this
  batch.
- **Class-guard status:** F8a answers #87, #88, #90, #91 with code and
  #85, #86, #89 with off-path evidence in `CGR-20260913-007-4.md`
  (written next, LAST per ORDER). #92 remains open for F8b (response
  `-5`). CGR-009 and CGR-010 are untouched by this batch.

## Parent review and integration (2026-09-14)

Accepted as delivered. The worker followed the COMPLIANCE rules: no source edit before failing-first and no non-pytest code. It disclosed two self-caught items (a fixture fix to its own new test file, and a line count corrected before handoff).

### Manifest

- **Checksums:** `sha256sum -c manifest_post_F7c.txt` shows no mismatch.
- **New or changed paths:** exactly four:
  - `core/prompt/gatherer_memory.py` (`b994315c…`, its first modification);
  - `tests/unit/test_gatherer_outcomes_memory.py` (`c21be821…`, 350 lines);
  - `class_guard_responses/CGR-20260913-007-4.md` (`092b6ef3…`, 350 lines);
  - this file.
  - §17 above lists three and omits the response file. That is a reporting slip; the tree is correct.
- **Git state:** `git diff --cached --name-only` is empty, HEAD `328a8ec` is unchanged, and the reflog still shows only the clone entry.
- **No new stray files:** the only untracked file at the repository root is `scan_pre.stderr`.
- **`data/`:** the 7-entry baseline is unchanged, and `logs/` is absent.
- **Parent edits after the worker returned:**
  - `briefs/F8b.md` was replaced with the corrected draft (a budget refusal is checked before a provider error: `web_search_manager.py` sets both `error` and `blocked="budget"` on a budget refusal);
  - `briefs/PARENT_STATE.md` was updated.
- **Recorded tree:** `manifest_post_F8a.txt` records the tree after this section.

### Code review (the parent read the full 25-line diff)

- **Import:** `OutcomeList`.
- **#87:** the `_get_recent_conversations` anchor except returns `OutcomeList.failed(type(e).__name__)`. The fallback-only inner swallow is unchanged, as a sibling.
- **#88 (`_get_semantic_memories`):**
  - `retrieval_err = None` is set before the inner try, and the inner except records the class name;
  - at the final return, `OutcomeList(result, status="failed", reason="retrieval:<class>")` when that is set;
  - the outer except returns `failed(<class>)`;
  - there is no early return between the inner except and the final return (parent-checked), so the status cannot be dropped.
- **#90 (`get_user_profile_context`):** logs, then re-raises. The gather loop records failed and stores `[]`, which the formatter (`isinstance(..., str)` check at 1709) and the builder's oversize scan already treat as "no profile".
- **#91 (`get_upcoming_schedule`):** the anchor except returns `failed(<class>)`, and every legit early return is unchanged.
- **Unchanged by design:** `get_recent_facts`, `get_facts` and `_get_reflections` (off-path), and the `_get_summaries_separate` / `_get_reflections_separate` dict sections (a recorded limitation).

### Tests (the parent read the whole file)

- **#87:**
  - a raising corpus manager gives failed / RuntimeError;
  - healthy non-empty and healthy empty controls;
  - the fallback-only swallow stays succeeded (sibling proof);
  - through `_build_lightweight_context`, a raising store keeps the NORMAL light dict (it has `web_search_decision` and not `upcoming_schedule`), which proves the typed return keeps builder.py:2314 unchanged.
- **#88:**
  - a raising coordinator gives failed / `retrieval:ValueError`;
  - a raising `_deduplicate_memories` gives failed / RuntimeError;
  - an empty query gives no_results, and healthy returns today's items.
- **#90:**
  - the method re-raises;
  - with no profile it returns "";
  - through `full_builder`, `_section_outcomes["user_profile"]` is failed / RuntimeError and the prompt still builds.
- **#91:** a raising `query_collection` gives failed; the disabled flag gives no_results; healthy returns the slice.
- **#85, #86, #89:** the three mixin methods are monkeypatched to spy-and-raise, and a builder run with summaries and reflections enabled shows zero calls, with the build succeeding.
- **Privacy:** markers are absent from every reason.
- **Failing-first:** 6 failed and 10 passed at digest `2cbe7775…`.

### Parent rerun (guarded, capped, `-p asyncio`)

- **Focused** (MemAvailable 6,602 MB): the new file plus test_section_outcomes, test_prompt_timeout, test_sep10_probe_dump_interpretation, test_independent_prompt_audit and test_light_prompt_path gave **228 passed** (0:11.37, 1,714,244 KB).
- **Sweep A** (6,611 MB): the 9 files in §7 gave **178 passed** (0:09.87, 1,703,088 KB).
- **Sweep B** (6,613 MB): test_prompt_compat gave **7 passed** (0:04.68, 1,066,364 KB).
- **Total:** **413 passed, 0 failed**, matching the worker. `tests/unit/test_graph_integration.py` stays excluded.
- **`data/`:** unchanged after every chunk, and `logs/` is absent.
- **ruff 0.14.9:** clean on both files.
- **Scan:**
  - dm18: found 50, base 79, new 0, stale 29.
  - The 4 new stale rows are exactly `MemoryRetrievalMixin._get_recent_conversations`, `_get_semantic_memories`, `get_user_profile_context` and `get_upcoming_schedule` (#87, #88, #90, #91).
  - #85, #86 and #89 stay live by design (evidence-only answers).
  - dm01 (7/12/0/5), dm17 (36/39/0/3) and dm31 (3/3/0/0) are unchanged, and every gate scanner shows 0 new.

### Size

375 changed lines (25 source + 350 test), within the 450 cap.

### Class-guard status

- **CGR-007:** #71–#91 answered. #85, #86 and #89 are answered as not on the prompt path; the class-guard owner decides between a detector change and accepted debt.
- **Still open:** #92 (F8b, response `-5`), CGR-009 and CGR-010.
