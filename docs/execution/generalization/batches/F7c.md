# F7c: the remaining knowledge gatherers return a typed failure instead of
# an empty section (git, proposals, skills, graph, threads, narrative,
# self-notes, emails)

Worker evidence packet. Design source:
`docs/execution/generalization/failure_outcome_design.md`, "F7 split and
gatherer outcome shape (parent decision, 2026-09-14)", F7c row and "Limits
recorded now". Request packet (read-only, the only file read in that
checkout): `/home/lukeh/daemon_exec/class_guards/docs/execution/class_guards/requests/CGR-20260913-007.md`.
Brief: `docs/execution/generalization/briefs/F7c.md`. Rules:
`docs/execution/generalization/briefs/R_common_rules.md` (applies in FULL).
Anchors answered: **#75, #76, #77, #78, #79, #81, #82, #83, #84**
(`get_git_commits`, `get_proposed_features`, `get_procedural_skills`,
`get_graph_context`, `get_unresolved_threads`, `get_narrative_context`,
`get_daemon_self_notes`, `get_relevant_emails` ×2). BUG_CLASSES: BC-20,
BC-47, CM-05. Response file (immutable once written, written LAST):
`docs/execution/generalization/class_guard_responses/CGR-20260913-007-3.md`.

This file is written INCREMENTALLY per the INTERRUPTION rule: created right
after the manifest check, then a dated entry appended after each ORDER step.

## 1. 2026-09-14 — Manifest check (before any edit)

Run from the checkout root with
S=/tmp/claude-1000/-home-lukeh-daemon-exec-generalization/1f0f3407-5796-4278-85e4-0c7ba4f50aa9/scratchpad:

- (a) `sha256sum -c --quiet $S/manifest_post_F7b.txt` → prints nothing, exit 0. PASS.
- (b) `{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u | diff - $S/manifest_paths_post_F7b.txt` → prints nothing, exit 0. PASS.

Both checks pass — proceeding.

### Read-only git state (pre-edit)

- `HEAD`: `328a8ecea1dae91de16f5974ea71727c4e2e1aa0` (matches the packet's
  "Base tree" and the brief's parent-verified base).
- `git remote -v`: `origin  /home/lukeh/Daemon_v1 (fetch)` / `origin
  DISABLED (push)`. Not touched.
- `git stash list`: empty. Not touched.
- `git diff --cached --name-only`: empty.
- `git status --short`: 104 lines (modified/untracked paths carried over
  through F7a/F7b), matching the manifest path-set exactly (check (b) above
  passed).

### Sanity check (one-time `import utils`, standing exception)

`PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin python -c "import utils; print(utils.__file__)"`
→ `/home/lukeh/daemon_exec/generalization/utils/__init__.py` — resolves
inside this clone, as required.

### Assigned-file digest (ORDER REMINDER: re-verify here, print again with
### the failing-first command)

`sha256sum core/prompt/gatherer_knowledge.py` →
`1839020273dcf804b2be94ef67a65159fd71c89f0e3addd2a1657c82cbab98b3`

Matches the brief's "POST-F7b RE-LOCATION" digest (`18390202…`) exactly.

### Re-verification of the POST-F7b RE-LOCATION line numbers

Read `core/prompt/gatherer_knowledge.py` directly (offsets 1000-1310,
1965-2298) before any edit. Every line the brief names matches exactly:

- import `from utils.retrieval_outcome import OutcomeList, outcome_status`
  at line 65 — present (F7a added it; nothing to add).
- `get_git_commits` 1023 (except warning "Failed to get git commits" 1115,
  `return []` 1116) — **ANCHOR #75**.
- `get_proposed_features` 1118 (warning 1170, `return []` 1171) —
  **ANCHOR #76**.
- `get_procedural_skills` 1173 (warning 1213, `return []` 1214) —
  **ANCHOR #77**.
- `get_graph_context` 1216 (warning "Graph context retrieval failed" 1275,
  `return []` 1276) — **ANCHOR #78**.
- `get_unresolved_threads` 1278 (warning 1302, `return []` 1303) —
  **ANCHOR #79**.
- `get_narrative_context` 2006, SYNC, returns `str`
  (`corpus.get_narrative_context()` at 2026; warning "Failed to retrieve
  narrative context" 2035, `return ""` 2036) — **ANCHOR #81**. Its only
  caller, `core/prompt/builder.py:1072`
  (`narrative_state = self.context_gatherer.get_narrative_context()`),
  confirmed inside a try (1071-1079) whose except (1077-1079) already
  records `section_outcomes["narrative"] = {"status": "failed", "reason":
  type(e).__name__}` (F5-integrated) — re-raising from the gatherer feeds
  this existing except directly.
- `get_daemon_self_notes` 2038 (annotation import 2088, call 2093; outer
  debug "daemon_self_notes retrieval failed" 2100, `return []` 2101) —
  **ANCHOR #82**. Inner per-item try 2083 / except 2094 (annotation
  failures) confirmed a read-only sibling to leave untouched.
- `get_relevant_emails` 2133 ("Email relevance filtering failed" 2290 =
  **ANCHOR #83**, `return []` 2293; "Email retrieval failed" 2296 =
  **ANCHOR #84**, `return []` 2297).

No drift found. Proceeding.

### data/ and logs/ baseline (before the first pytest)

```
$ ls -la --time-style=full-iso data
total 68
drwxr-xr-x. 1 lukeh lukeh   254 2026-09-14 13:26:21.677170907 -0500 .
drwxr-xr-x. 1 lukeh lukeh  1026 2026-09-14 11:21:59.889459216 -0500 ..
-rw-r--r--. 1 lukeh lukeh 53235 2026-09-13 12:28:07.844792269 -0500 benchmark_per_case.csv
drwxr-xr-x. 1 lukeh lukeh     0 2026-09-13 13:09:16.898881228 -0500 chroma_db_v4
drwxr-xr-x. 1 lukeh lukeh   100 2026-09-14 04:35:38.400904050 -0500 chroma_multi
-rw-r--r--. 1 lukeh lukeh   753 2026-09-13 12:28:07.846364181 -0500 embedding_migration_manifest.json
drwxr-xr-x. 1 lukeh lukeh   304 2026-09-13 12:28:07.846364181 -0500 pipeline
-rw-------. 1 lukeh lukeh   598 2026-09-14 13:26:21.672354716 -0500 user_profile.json
-rw-------. 1 lukeh lukeh    50 2026-09-14 04:35:38.763134691 -0500 web_search_credits.json

$ ls -ld --time-style=full-iso logs
ls: cannot access 'logs': No such file or directory
```

Identical to the parent's post-F7b baseline (7 entries including
`user_profile.json`, 598 bytes, mtime `13:26:21.672354716` — the
`test_graph_integration.py` writer, parent-accepted and never touched
here; `logs/` absent). No STOP condition triggered.

MemAvailable at this point: 6568 MB (≥4000 required). Pytest guard: clear
(no RUNNING `python -m pytest` process).

## 2. 2026-09-14 — PRE-EDIT SCAN

```
$ PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin python scripts/check_bug_classes.py scan --root . > /tmp/scan_pre_f7c.txt 2> /tmp/scan_pre_f7c.stderr
exit: 1
```
(stderr file has 0 bytes; captured to /tmp only, never redirected into the
repo.)

Full scanner table:

```
scanner                       mode    files  found  base  new  stale
dm01_raw_substring            gate    396    7      12    0    5
dm16_config_key_reachability  report  395    89     -     -    -
dm17_apply_without_guard      gate    626    36     39    0    3
dm18_except_returns_empty     gate    123    63     79    0    16
dm29_phrase_append_signature  report  0      0      -     -    -
dm31_live_state_default       gate    396    3      3     0    0
catalog                       gate    1      0      0     0    0
```

Identical to F7b's post-edit baseline (dm18: found 63, base 79, new 0,
stale 16) — no drift since F7b integration, as expected (no source file has
changed in between).

JSON run (approved exception; stderr 48 lines of stdlib
DeprecationWarning noise from the scanner's own dependency scan, to /tmp
only): `python scripts/check_bug_classes.py scan --root . --json >
/tmp/scan_pre_f7c.json 2> /tmp/scan_pre_f7c_json.stderr`, exit 1. Inspected
via `jq` only (the approved tool for JSON inspection — no `python -c`/
heredoc used).

dm18 live findings for `core/prompt/gatherer_knowledge.py` (via `jq`,
matching the packet's anchors exactly):

```
KnowledgeRetrievalMixin.get_git_commits line=1116          -- #75
KnowledgeRetrievalMixin.get_proposed_features line=1171    -- #76
KnowledgeRetrievalMixin.get_procedural_skills line=1214    -- #77
KnowledgeRetrievalMixin.get_graph_context line=1276        -- #78
KnowledgeRetrievalMixin.get_unresolved_threads line=1303   -- #79
KnowledgeRetrievalMixin.get_narrative_context line=2036    -- #81
KnowledgeRetrievalMixin.get_daemon_self_notes line=2101    -- #82
KnowledgeRetrievalMixin.get_relevant_emails line=2293      -- #83
KnowledgeRetrievalMixin.get_relevant_emails line=2297      -- #84
```

Exactly the 9 sites this batch owns — no more, no fewer. `_get_wiki_content_timed`
(#80, F7b) and #71-#74 (F7a) are confirmed ABSENT from the live report
(already STALE). No F8-owned (`gatherer_memory.py`/`gatherer_web.py`)
findings inspected here — out of scope.

data/ and logs/ after this step: unchanged from §1's baseline (same 7
entries, same mtimes; `logs/` still absent). No STOP.

## 3. 2026-09-14 — Interruption (API session limit) and resumption

An API session limit cut the worker session off at approximately 14:32,
immediately after §2 (the pre-edit scan) was recorded — no test file, no
source edit and no response file existed at that point. The parent
independently verified the tree at 15:01 before instructing resumption:
`core/prompt/gatherer_knowledge.py` UNEDITED at digest `18390202…`; the
only new path is this file (175 lines at that point); everything else
matches `manifest_post_F7b`; nothing staged; `data/` at the 7-entry
baseline (`user_profile.json` unchanged, 598 bytes, mtime
`13:26:21.672354716`); `logs/` absent; no pytest running.

Re-verified independently on resumption (all match the parent's report
exactly):

```
$ git rev-parse HEAD
328a8ecea1dae91de16f5974ea71727c4e2e1aa0
$ sha256sum core/prompt/gatherer_knowledge.py
1839020273dcf804b2be94ef67a65159fd71c89f0e3addd2a1657c82cbab98b3
$ git status --short | wc -l
104
$ git ls-files --others --exclude-standard | grep -E "F7c|test_gatherer_outcomes_remaining_knowledge|CGR-20260913-007-3"
docs/execution/generalization/batches/F7c.md
docs/execution/generalization/briefs/F7c.md   # parent-owned, pre-existing
```

- (a) `sha256sum -c --quiet $S/manifest_post_F7b.txt` → exit 0.
- (b) `{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u | diff - <(manifest_paths_post_F7b.txt plus this file)` → exit 0 (the only addition since the recorded manifest is this evidence packet itself, as expected per the INTERRUPTION rule).
- `data/`: identical 7-entry baseline (`user_profile.json` 598 bytes, mtime `13:26:21.672354716`); `logs/`: absent.
- Pytest guard: clear (no RUNNING `python -m pytest`). MemAvailable: 6520 MB (≥4000).

No drift from the parent's report. Per INTERRUPTION rule: this relaunched
continuation does NOT read any `batches/F7c_attempt*/` artifact (none
exists — this is a same-session resumption of the ORIGINAL F7c batch, not
a parent-relaunched fresh worker), and the failing-first proof below is
still produced independently, in one command, printing the source digest
immediately before any test runs against the unedited source, per ORDER
REMINDER. Resuming at ORDER step 4 (write the new test file).

## 4. 2026-09-14 — New test file written (Write tool only)

`tests/unit/test_gatherer_outcomes_remaining_knowledge.py` (new, 409 lines
via `wc -l`). Drives the deployed `KnowledgeRetrievalMixin.get_git_commits`,
`get_proposed_features`, `get_procedural_skills`, `get_graph_context`,
`get_unresolved_threads`, `get_daemon_self_notes` and `get_relevant_emails`
directly through a minimal `_G(KnowledgeRetrievalMixin)` host (the
`test_sep10_probe_dump_actions.py`-precedent pattern), plus
`get_narrative_context` both directly (`_NarrativeSource`) and through the
deployed `UnifiedPromptBuilder.build_prompt` via a local copy of the
`full_builder`/`retrieval_limits` fake-builder pattern established by
`tests/unit/test_independent_prompt_audit.py` (also used by F5/F7a/F7b).
`Mock`/`AsyncMock`/`SimpleNamespace` fakes only — no real ChromaDB,
embedder, graph store, email service or network anywhere;
`memory.graph_utils.extract_graph_entities` and
`core.action_claim_guard.annotate_unverified_action_claim` are exercised as
real (pure, dependency-free) functions, matching the precedent in
`test_sep10_probe_dump_actions.py`. The six direct-swap sections (git,
proposals, skills, graph, threads, self-notes) share an identical
(raise/empty/nonempty[/disabled]) contract shape, so their cases are
parametrized across sections (`SECTION_BUILDERS`) instead of repeated per
method — 6×3 = 18 cases in 3 shared test functions, plus a 4th shared
"disabled flag unchanged" control (5 of the 6 sections; `daemon_self_notes`
has no feature flag). Emails (4 cases: ANCHOR #83 raising embedder, ANCHOR
#84 raising `service.search`, a no-cue control, a healthy-ranked control)
reuse the `_run_gatherer`-style fake-service/fake-embedder monkeypatch
technique from `tests/unit/test_email_passive_context.py` (a local copy,
not an import, to avoid cross-test-module coupling). Narrative (ANCHOR #81)
has 6 cases: a direct raise-reraises test, a direct disabled-flag control, a
direct missing-corpus-manager control, a through-builder raise test
(contract point 3's builder-level proof), and a through-builder
`max_narrative=0` control (F5 precedent — the producer is never called).
Privacy: every raising fixture's exception message and every section's
query carries the marker `F7CQX9`; every raise-producing test asserts the
marker is absent from the resulting `.reason` (folded into the existing
tests rather than a separate privacy test class, per the F7b precedent).

## 5. 2026-09-14 — FAILING-FIRST (digest printed in the same command, before any source edit)

Guard checked first (clear, no RUNNING `python -m pytest`); MemAvailable
6585 MB (≥4000).

```
$ sha256sum core/prompt/gatherer_knowledge.py
1839020273dcf804b2be94ef67a65159fd71c89f0e3addd2a1657c82cbab98b3  core/prompt/gatherer_knowledge.py
$ systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=512M /usr/bin/time -v env \
    DAEMON_TEST_MODE=1 CHROMA_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin python -m pytest -q \
    -p no:cacheprovider -p asyncio tests/unit/test_gatherer_outcomes_remaining_knowledge.py
```

Digest confirmed matching §1 (`18390202…`). Result: **10 failed, 22 passed**,
exit 1, wall 0:05.32, peak RSS 1,091,372 KB. No test-file bug was found
this time (unlike F5/F7a/F7b, each of which found 1-2) — every failure and
every pass landed exactly where designed on the first attempt.

Failing (10, exactly the sites this batch is meant to fix):
- `test_raising_producer_is_failed[daemon_self_notes]` — ANCHOR #82
- `test_raising_producer_is_failed[git_commits]` — ANCHOR #75
- `test_raising_producer_is_failed[graph_context]` — ANCHOR #78
- `test_raising_producer_is_failed[procedural_skills]` — ANCHOR #77
- `test_raising_producer_is_failed[proposed_features]` — ANCHOR #76
- `test_raising_producer_is_failed[unresolved_threads]` — ANCHOR #79
- `test_raising_embedder_is_failed_relevance_unavailable` — ANCHOR #83
- `test_raising_service_search_is_failed` — ANCHOR #84
- `test_narrative_producer_raise_reraises` — ANCHOR #81 (direct)
- `test_through_builder_narrative_raise_recorded_failed` — ANCHOR #81 (builder-level)

Every failure shape confirms the defect directly: `outcome_status(result)
== ('no_results', '')` instead of `('failed', ...)`; `Failed: DID NOT
RAISE <class 'RuntimeError'>` for the narrative direct-raise test (the
swallow currently eats the exception); the through-builder test shows
`_section_outcomes["narrative"] == {"status": "no_results", "reason": ""}`
instead of `{"status": "failed", "reason": "RuntimeError"}` (F5's own
gather/narrative wiring already reads whatever the gatherer returns — the
gatherer is the only thing that must change).

Passing (22, the "existing behaviour is unchanged" controls, correctly
green before any edit): all 6×3 = 18 `test_control_healthy_empty_is_no_results`/
`test_control_healthy_nonempty_is_succeeded` (6 each) and
`test_control_disabled_flag_returns_empty_unchanged` (5, `daemon_self_notes`
excluded) cases; `test_control_no_cue_is_no_results`,
`test_control_healthy_ranked_messages_succeeds`;
`test_narrative_disabled_flag_returns_empty_string`,
`test_narrative_missing_corpus_manager_returns_empty_string`,
`test_through_builder_max_narrative_zero_skips_entry_control`.

data/ and logs/ after this chunk:
```
$ ls -la --time-style=full-iso data   # unchanged from §1's baseline (same 7 entries, same mtimes)
$ ls -ld --time-style=full-iso logs   # still absent
```
No STOP condition triggered.

## 6. 2026-09-14 — Source edit (9 small complete Edit calls, ownership scope only)

`core/prompt/gatherer_knowledge.py`, exactly the 9 except-sites named in
the brief's CONTRACT — "direct swaps" per the design doc's F7 split (no
leg-status-reading restructuring needed, unlike F7a's producer-consuming
methods; these 8 methods have no already-typed producer to read from):

- `get_git_commits` except (1114-1116): `return []` → `return
  OutcomeList.failed(type(e).__name__)` — ANCHOR #75.
- `get_proposed_features` except (1169-1171): same shape — ANCHOR #76.
- `get_procedural_skills` except (1212-1214): same shape — ANCHOR #77.
- `get_graph_context` except (1274-1276): same shape — ANCHOR #78.
- `get_unresolved_threads` except (1301-1303): same shape — ANCHOR #79.
- `get_narrative_context` except (2034-2036): `return ""` → a 5-line
  comment plus `raise` (contract point 3 — the warning log line is
  unchanged; the disabled-flag and missing-corpus-manager early returns
  above the try are untouched, still `return ""`) — ANCHOR #81.
- `get_daemon_self_notes` except (2099-2101): same shape as the first
  five (`logger.debug` line unchanged) — ANCHOR #82.
- `get_relevant_emails` inner except (2289-2293): `return []` → `return
  OutcomeList.failed("relevance_unavailable")` (comment kept byte-for-byte
  per contract point 2) — ANCHOR #83.
- `get_relevant_emails` outer except (2295-2297): same shape as the first
  five — ANCHOR #84.

No other line touched. `git diff --stat` against HEAD shows **58
insertions(+), 17 deletions(-)** for the whole file, but this conflates
F7a's (27 ins/4 del, already-integrated-pattern uncommitted state) and
F7b's (≈17 ins/4 del) prior edits to the SAME file with this batch's own —
confirmed by reading the full diff and isolating only the hunks touching
the 9 sites above (F7a's `get_personal_notes`/`get_reference_docs`/
`_fetch_upload_roster`/`get_user_uploads` hunks and F7b's
`_get_wiki_content_timed`/`_get_semantic_chunks_timed` hunks are present
in the diff but are NOT this batch's edits — they were already in the
working tree before this batch started, per the manifest check in §1/§3).
**This batch's own isolated contribution: 23 changed lines** (8 sites ×
1-line swap = 16 lines [8 deletions, 8 insertions], plus the narrative
site's 1 deletion + 6 insertions [1-line `raise` replacing `return ""`,
plus 5 new comment lines] = 7 lines; 16 + 7 = 23). Cross-check: cumulative
58 ins/17 del minus F7a's 27 ins/4 del minus F7b's isolated 17 ins/4 del ≈
14 ins/9 del for this batch — the small discrepancy from the exact 8/15
split is git's diff algorithm re-aligning hunks differently across three
edits sharing nearby context (same class of arithmetic imprecision F7b's
own packet recorded, §14 there), not a measurement error; the per-Edit-call
isolated count (23) is authoritative for this batch's own contribution,
consistent with every `old_string`/`new_string` pair passed to the Edit
tool above.

```
$ sha256sum core/prompt/gatherer_knowledge.py
00870a5bd0447c7b02cdfd9d993c68698d16f9f20b29dddbed2661319581f4a9  core/prompt/gatherer_knowledge.py
```

data/ and logs/: unchanged from §1's baseline. No STOP.

## 7. 2026-09-14 — Focused chunk (1 chunk of 9, per the brief)

Guard clear, MemAvailable 6570 MB (≥4000):

```
$ systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=512M /usr/bin/time -v env \
    DAEMON_TEST_MODE=1 CHROMA_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin python -m pytest -q \
    -p no:cacheprovider -p asyncio \
    tests/unit/test_gatherer_outcomes_remaining_knowledge.py \
    tests/unit/test_sep12_repository_status_context.py \
    tests/unit/test_proposal_filter.py \
    tests/unit/test_independent_prompt_audit.py \
    tests/unit/test_narrative_staleness.py \
    tests/unit/test_sep10_probe_dump_actions.py \
    tests/unit/test_sep10_probe_dump_interpretation.py \
    tests/unit/test_email_passive_context.py \
    tests/unit/test_sep03_followups_gating.py
```

Result: **534 passed**, 0 failed, 0 skipped, exit 0, wall 0:15.64, peak RSS
1,754,272 KB. 4 warnings, all pre-existing (3 SWIG DeprecationWarnings + 1
`websockets.legacy` DeprecationWarning from
`test_sep10_probe_dump_actions.py`), unrelated to this batch.

**No existing test required a FIXTURE RULE edit.** All 8 brief-named
existing files passed unmodified against the edited source:
`test_sep12_repository_status_context.py`'s
`test_git_timeout_does_not_substitute_stale_index` (`== []`) is unaffected
— a `TimeoutExpired` is swallowed inside `GitMemoryExtractor.extract_commits`
(a read-only sibling), never reaching our except; `test_proposal_filter.py`'s
`get_proposed_features` tests never raise the producer; `test_independent_
prompt_audit.py` never touches these 8 methods' exception paths;
`test_narrative_staleness.py` tests `CorpusManager.get_narrative_context`
directly (a different class/method entirely — the producer, not our
gatherer); `test_sep10_probe_dump_actions.py`'s/`test_sep10_probe_dump_
interpretation.py`'s self-notes tests exercise only healthy paths and a
source-text `inspect.getsource` check (still finds
`annotate_unverified_action_claim` in the body, untouched by our edit);
`test_email_passive_context.py`'s
`test_embedding_failure_does_not_inject_unranked_email` (`out == []`) stays
true for an empty `OutcomeList.failed("relevance_unavailable")` (list-
subclass equality compares elements only); `test_sep03_followups_gating.py`
never raises the email producer.

data/ and logs/ after this chunk: unchanged (7 entries, same mtimes;
`logs/` absent). No STOP.

## 8. 2026-09-14 — SWEEP set identified

`grep -rl "gatherer_knowledge\|KnowledgeRetrievalMixin\|core\.prompt\.context_gatherer" tests/unit/`
→ 37 files. Widened with the F7a/F7b-precedent check for the bare class
name `ContextGatherer`: `tests/unit/test_prompt_compat.py` is a genuine
importer (`from core.prompt import ContextGatherer` at line 21 — added);
`tests/unit/test_ordered_slice_guard.py` is NOT an importer (confirmed by
reading it: imports only `ast`/`re`/`collections`/`pathlib`, and the one
`ContextGatherer` mention is a STRING inside an allowlist tuple —
excluded, matching F7a's and F7b's identical finding for the same file).
Union: 38 files.

Minus the 9 already run in the focused chunk (§7) and the permanently
excluded `tests/unit/test_graph_integration.py` (TEST EXCLUSION) = **30
files** for the sweep, chunked ≤9.

**Safety check (NON-UNIT TESTS / ContextGatherer-UserProfile hazard,
before running anything):** `grep -n "ContextGatherer("` and `grep -n
"UserProfile("` and `grep -ln "spec=\[\]"` over all 30 sweep files:
- `tests/unit/test_session_diff.py:28` (`ContextGatherer(` — plain
  `MagicMock()` coordinator, `hasattr(mc, 'user_profile')` is `True`,
  confirmed safe by the parent in F7a's review, re-confirmed here by
  reading the exact call site again).
- `tests/unit/test_codex_followups.py:42,51` (`UserProfile(profile_path=
  str(p))`, `p = tmp_path / "user_profile.json"` — explicit, tmp_path-
  scoped path, not the default dev path. Safe).
- No `MagicMock(spec=[])` (the specific hazard shape) anywhere in the 30
  files.
All 30 files clear to run.

Chunked ≤9: Chunk A (8), Chunk B (7), Chunk C (8), Chunk D (7).

## 9. 2026-09-14 — Sweep chunks A-D

Guard clear before each; MemAvailable checked before each (6580, 6489,
6595, 6573 MB — all ≥4000).

- **Chunk A** (test_audit0831_fixes, test_calendar_prompt,
  test_calendar_update_delete, test_codex_followups,
  test_gatherer_latency_guards, test_gatherer_outcomes_background_knowledge
  [F7b's own new file], test_gatherer_outcomes_notes_docs_uploads [F7a's
  own new file], test_ingest_turn_misfires) → **178 passed**, exit 0, wall
  0:21.50, peak RSS 1,110,640 KB.
- **Chunk B** (test_narration_turn_audit_fixes, test_obsidian_failure_outcomes,
  test_paste_turn_misfires, test_prompt_builder_self_report_trim,
  test_prompt_compat, test_retrieval_context_quality,
  test_sep03_live_probe_fixes) → **166 passed**, exit 0, wall 0:08.07, peak
  RSS 1,636,568 KB. 8 warnings: 3 pre-existing SWIG DeprecationWarnings + 5
  pre-existing `RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call'
  was never awaited` from `test_prompt_builder_self_report_trim.py`'s own
  `session_reflections.sort` line in `core/prompt/builder.py:1679` — a file
  this batch does not touch (same warning F7a's/F7b's chunks recorded).
- **Chunk C** (test_sep04_attachment_turn, test_sep06_conversation_response,
  test_sep07_followups, test_sep08_homework_tone_misfires,
  test_sep09_latency_metrics, test_sep09_live_controls, test_sep09_speed_images,
  test_session_diff) → **172 passed**, exit 0, wall 0:12.33, peak RSS
  1,685,908 KB. 4 warnings: 3 SWIG + 1 pre-existing `websockets.legacy`
  DeprecationWarning from `test_sep09_speed_images.py`.
- **Chunk D** (test_trigger_match, test_upload_keyword_score_leak,
  test_upload_retrieval_pool, test_visual_memory_gating, test_warmup_guards,
  test_wiki_disambiguation_filter, test_wiki_fallback_stub_filter) →
  **117 passed**, exit 0, wall 0:11.27, peak RSS 1,175,844 KB. 4 warnings:
  3 SWIG + 1 pre-existing `websockets.legacy` DeprecationWarning from
  `test_warmup_guards.py`.

**Sweep total: 633 passed, 0 failed, 0 skipped** across 30 files.
`data/`/`logs/` checked and unchanged after every chunk (identical 7-entry
baseline throughout, `logs/` absent). No STOP at any point.

**Grand total, this batch (focused + sweep): 534 + 633 = 1,167 passed,
0 failed, 0 skipped.** No existing test required a FIXTURE RULE edit
anywhere in the sweep either — every one of the 30 files passed unmodified.

## 10. 2026-09-14 — ruff

```
$ PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin python -m ruff check \
    core/prompt/gatherer_knowledge.py tests/unit/test_gatherer_outcomes_remaining_knowledge.py
All checks passed!
```
ruff 0.14.9.

## 11. 2026-09-14 — Post-edit scan (stderr to /tmp only)

```
$ PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin python scripts/check_bug_classes.py scan --root . > /tmp/scan_post_f7c.txt 2> /tmp/scan_post_f7c.stderr
exit: 1   (pre-existing STALE rows across dm01/dm17/dm18, same as every prior batch)
stderr: 0 bytes
```

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

Compared to pre-edit (§2): dm18 `found` dropped **63 → 54** (exactly the 9
anchors fixed: #75-#79, #81-#84), `new` is **0** across every scanner (no
new finding introduced anywhere in the tree by this edit), `stale` rose
**16 → 25** (the 9 newly-STALE rows below, plus the 16 pre-existing ones
unrelated to this batch). `dm17_apply_without_guard`'s `files` count
ticked 626→627 (this scanner's file selection includes `tests/`, and this
batch adds one new test file; its own `found`/`new`/`stale` are unchanged,
a benign files-processed count effect, not a finding).

JSON run (stderr 57 lines of stdlib DeprecationWarning noise, to /tmp
only): `python scripts/check_bug_classes.py scan --root . --json >
/tmp/scan_post_f7c.json 2> /tmp/scan_post_f7c_json.stderr`, exit 1.

dm18 STALE rows for #75-#79, #81-#84 (live findings confirmed ABSENT via
`jq` — `select(.path=="core/prompt/gatherer_knowledge.py")` over the live
findings returns **zero rows total**, meaning every method in this file,
across F7a+F7b+F7c combined, is now gone from the live dm18 report;
present only in the plain-text STALE section, as designed):

```
dm18_except_returns_empty: core/prompt/gatherer_knowledge.py [KnowledgeRetrievalMixin.get_daemon_self_notes] 'return []'      -- #82
dm18_except_returns_empty: core/prompt/gatherer_knowledge.py [KnowledgeRetrievalMixin.get_git_commits] 'return []'            -- #75
dm18_except_returns_empty: core/prompt/gatherer_knowledge.py [KnowledgeRetrievalMixin.get_graph_context] 'return []'          -- #78
dm18_except_returns_empty: core/prompt/gatherer_knowledge.py [KnowledgeRetrievalMixin.get_narrative_context] 'return ""'      -- #81
dm18_except_returns_empty: core/prompt/gatherer_knowledge.py [KnowledgeRetrievalMixin.get_procedural_skills] 'return []'      -- #77
dm18_except_returns_empty: core/prompt/gatherer_knowledge.py [KnowledgeRetrievalMixin.get_proposed_features] 'return []'      -- #76
dm18_except_returns_empty: core/prompt/gatherer_knowledge.py [KnowledgeRetrievalMixin.get_relevant_emails] 'return []'   ×2    -- #83, #84
dm18_except_returns_empty: core/prompt/gatherer_knowledge.py [KnowledgeRetrievalMixin.get_unresolved_threads] 'return []'     -- #79
```

Exactly the 9 sites this batch owns — all moved to STALE. The re-raising
`get_narrative_context` except (now `raise` instead of `return ""`) is
**confirmed NOT a dm18 finding** either before or after this batch (0
matches in the live report both times, via the same `jq` query pattern) —
dm18 matches an `except: return X` shape; a bare `raise` returns nothing,
so it structurally cannot match. This is the designed handoff (F7a/F7b
precedent): the class-guard owner removes these rows and marks them
`confirmed_fixed` with this request ID after integration.

data/ and logs/ after the scan: identical to §1's baseline (7 entries,
`user_profile.json` unchanged); `logs/` still absent. No STOP.

## 12. BC-58 sibling search (contract point 5)

The six producer-internal swallows named in the brief's CONTRACT point 5,
each confirmed by direct read, with file:line and owning batch (READ-ONLY,
none edited by this batch):

- **`knowledge/git_memory.py` `GitMemoryExtractor.extract_commits`**
  (def 67, swallow 103-109: `except (FileNotFoundError, OSError,
  subprocess.TimeoutExpired)` and a non-zero `returncode` both →
  `return []`). Other production callers besides our gatherer (1052):
  `knowledge/git_memory_loader.py:67,130` (a different consumer,
  unaffected, out of this batch's scope). No batch is named for this
  sibling in the design doc's batch table — a durable limitation, not a
  future batch's known assignment.
- **`memory/proposal_store.py` `ProposalStore.query_proposals`** (def
  105, swallow: `except Exception as e: logger.error(...); return []`
  at ≈161-163). Only production caller: `core/prompt/proposal_filter.py:661`
  (itself only called by our gatherer's `get_proposed_features`, 1150).
  Owning batch: **F11b** (per the design doc's batch table).
- **`memory/memory_retriever.py` `MemoryRetriever.get_skills`** (def
  1107, swallow: `except Exception as e: logger.warning("Failed to
  retrieve skills"); return []` at ≈1183-1185), reached via
  `memory/memory_coordinator.py:578-583`'s pass-through. Other production
  caller besides our gatherer (1194): `memory/shutdown_processor.py:1312`
  (`mc.get_skills(synthetic_query, limit=3)` — a separate consumer,
  unaffected, out of this batch's scope). Owning batch: **F12** (per the
  design doc's "More sequential seams" — "F12: `get_skills` proposal
  context at ≈1312, if touched").
- **`memory/memory_coordinator.py` `MemoryCoordinator.get_unresolved_threads`**
  (def 585, swallow: `except Exception as e: logger.debug(...); return
  []` at 598-600), delegating to `ThreadStore.get_top_threads`. Only
  production caller: our gatherer (1297; builder.py:1414 is the task
  wrapper, not a second caller of the coordinator method). Owning batch:
  **F11a** (per the design doc's batch table, thread writes/reads).
- **`memory/corpus_manager.py` `CorpusManager.get_narrative_context`**
  (def 605, swallow: `except Exception as e: return ""` at 660-662; two
  more internal `return ""` at 619/632, both legit "file not found"/
  "empty" branches, not swallows of a raised error). Only production
  caller: our gatherer (2026). Owning batch: **F12** (consolidator status
  guard + readers, per the design doc's batch table).
- **`core/email/service.py` `EmailService._fan_out`** (def 53, per-provider
  swallow: `if isinstance(res, Exception): logger.warning(...); continue`
  at 57-60 — a total per-provider outage reduces `merged` to `[]`/partial
  without raising, so `service.search()` never raises for this reason;
  our gatherer's `if not messages: return []` at 2201-2202 is therefore a
  legit early return, not ANCHOR #83/#84, when EVERY provider is down).
  Other production callers of `get_email_service()` besides our gatherer
  (2193): `core/agentic/tools.py`, `core/agentic/controller.py`,
  `gui/handlers.py` (three separate consumers, unaffected, out of this
  batch's scope; no owning batch named for this sibling in the design
  doc's batch table).
- **The self-notes annotation loop** (inner try 2083 / except 2094 in
  `get_daemon_self_notes` itself, this batch's own file but explicitly
  named as a leave-alone sibling by the brief): swallows a single note's
  `annotate_unverified_action_claim` failure and continues with the
  unmarked note rather than failing the whole section. Confirmed
  unchanged by this batch's diff (only the OUTER except, 2099-2101, was
  touched). No owning batch named — a structural, low-severity read-only
  sibling.

**Consequence (per contract point 5), stated as a limitation per
section:** until F9-F12 land, a failure INSIDE these six producers still
reads as `no_results` at the gatherer boundary, not `failed` — this
batch's fix only converts the GATHERER's own except (an exception that
escapes the producer, e.g. the producer object itself being absent/
misconfigured, or the producer's own code raising past its swallow) into
`OutcomeList.failed(...)`. Each producer's own internal swallow is
unaffected and unowned by this batch.

**Off-path caller noted, unaffected:** `debug_graph_context.py:43` (repo
root, a standalone manual debug script, not imported by any production or
test module) calls `get_graph_context` directly and only inspects
truthiness/length/iteration of the result — all identical for a plain
`list` and an `OutcomeList` subclass. Confirmed unaffected; not part of
the class-guard's scope.

**Within `gatherer_knowledge.py` itself:** every other `except: return
[]`/`return ""` site in the file (#71-#74 F7a, #80 F7b) is already fixed
by earlier batches; nothing remains in this file outside this batch's 9
sites (confirmed by the post-edit scan in §11 — zero live dm18 findings
anywhere in `core/prompt/gatherer_knowledge.py`).

## 13. Privacy / no-network note

No network access, no LLM/paid API call, no daemon restart, no `pip
install`, no real ChromaDB/embedder/graph store/email service/model
(every store/service/embedder in the new test file is a
`MagicMock`/`AsyncMock`/`SimpleNamespace` fake; `memory.graph_utils.
extract_graph_entities` and `core.action_claim_guard.
annotate_unverified_action_claim` are exercised as real but pure,
dependency-free functions, matching `test_sep10_probe_dump_actions.py`'s
precedent). Every raising fixture's exception message AND every section's
query carries the distinctive marker `F7CQX9`; `test_raising_producer_is_
failed` (all 6 parametrized sections), `test_raising_embedder_is_failed_
relevance_unavailable`, `test_raising_service_search_is_failed`, and
`test_through_builder_narrative_raise_recorded_failed` each assert the
marker is absent from the resulting `.reason` — every reason this batch
produces is either a fixed constant label (`"relevance_unavailable"`,
`f"chroma:{chroma_err}"`-shaped strings are F7b's, not this batch's) or
`type(e).__name__` (an exception class name), never exception message
text or query text, enforced structurally (no site in this batch's diff
ever calls `str(e)` when building a `reason` — only `type(e).__name__` or
a fixed label). The one standing `python -c "import utils; print(utils.
__file__)"` sanity check was run once at the top of this session (§1); no
other non-pytest code ran. Every file edit used the Edit or Write tool;
`git add` was never invoked; every pytest invocation (failing-first, the
focused chunk, and all 4 sweep chunks) used the guarded, capped,
`-p asyncio` command, preceded by a `/proc/comm` guard check and a
MemAvailable check.

**Process deviations (disclosed in full):** none in this session's own
execution. The one INTERRUPTION (an API session limit at ≈14:32, resumed
at 15:01 after parent verification, §3) is not a process-rule breach —
it is exactly the scenario the INTERRUPTION rule anticipates, and the
resumption re-verified the manifest, the digest, and `data/`/`logs/`
independently before proceeding, per the rule's own text. No non-pytest
Python of any kind ran beyond the three standing exceptions (ruff, the
read-only scan, and the one `import utils` sanity check). No source edit
occurred before the failing-first proof was recorded. No file was created
outside this batch's ownership (`tests/unit/test_gatherer_outcomes_
remaining_knowledge.py`, `docs/execution/generalization/batches/F7c.md`,
and the response file below).

## 14. Size

`core/prompt/gatherer_knowledge.py`: **23 changed lines** (this batch's
own isolated contribution, per-Edit-call — §6's cross-check against the
cumulative `git diff --stat`, which conflates F7a's and F7b's prior
uncommitted edits to the same file).
`tests/unit/test_gatherer_outcomes_remaining_knowledge.py`: **409 lines**
(new, `wc -l`).

**Total: 23 + 409 = 432 changed lines** — over the 400-line soft target
but within the 450-line hard cap. Over target because this batch's
contract spans 8 methods across 6 distinct producer shapes (index-branch
chroma, an injected `_proposal_filter`, a coordinator method, a graph
store + resolver, a coordinator method, a chroma collection) plus emails
(2 anchors, its own fake-service/fake-embedder harness) plus narrative
(1 anchor, both a direct test and a full deployed-builder integration
test) — a wider method surface than F7a's four same-shaped return-site
conversions (343 total) or F7b's two methods (434 total), despite heavy
parametrization (6 sections × 3-4 cases each folded into 4 shared test
functions) and folding privacy assertions into existing tests rather than
a separate test class (both applied specifically to stay under the cap).
No split was needed; `docs/execution/generalization/batches/F7c.md` and
the response file are evidence/response artifacts, not counted toward
the cap, per precedent (F4.md, F7a.md, F7b.md).

## 15. Contract — how each of the 6 points is implemented

1. **Anchors #75, #76, #77, #78, #79, #82, #84 → `OutcomeList.failed(type(e).__name__)`, log line byte-identical.**
   Each of the 7 sites' `except Exception as e:` block keeps its existing
   `logger.warning`/`logger.debug` call untouched and returns
   `OutcomeList.failed(type(e).__name__)` instead of `[]` (§6). Proven per
   site by `test_raising_producer_is_failed[<section>]` (6 sections) and
   `test_raising_service_search_is_failed` (#84).
2. **Anchor #83 (email ranking) → `OutcomeList.failed("relevance_unavailable")`, no items, comment stays.**
   The inner scoring `except Exception as e:` keeps its two-line comment
   ("Relevance could not be established. Fail closed...") byte-for-byte
   and its `logger.debug` line, and returns
   `OutcomeList.failed("relevance_unavailable")` (§6). Proven by
   `test_raising_embedder_is_failed_relevance_unavailable` (asserts both
   the exact reason string and `result == []`).
3. **Anchor #81 (narrative, str) → except re-raises; disabled/missing-corpus stay ""; builder-level proof.**
   The except keeps its warning log, adds a 5-line comment, then `raise`
   (§6) — the two early returns above the try (`NARRATIVE_CONTEXT_ENABLED`
   false; no corpus manager / no `get_narrative_context` attribute) are
   untouched, still `return ""`. Proven directly:
   `test_narrative_producer_raise_reraises` (the gatherer re-raises,
   `pytest.raises(RuntimeError)`), `test_narrative_disabled_flag_returns_
   empty_string`, `test_narrative_missing_corpus_manager_returns_empty_
   string`. Proven at the builder level (the brief's explicit requirement):
   `test_through_builder_narrative_raise_recorded_failed` drives the
   DEPLOYED `UnifiedPromptBuilder.build_prompt` — the prompt still builds
   (`"_build_time" in result`), `result["narrative_state"] == ""`, and
   `result["_section_outcomes"]["narrative"] == {"status": "failed",
   "reason": "RuntimeError"}` (F5's existing try/except at
   `builder.py:1071-1079`, unedited by this batch, now actually exercises
   its except branch for the first time — previously dead code since the
   gatherer never raised). `test_through_builder_max_narrative_zero_skips_
   entry_control` proves the sibling F5 control (`max_narrative=0` → no
   entry, producer never called) is unaffected.
4. **Every legit early return stays unchanged (no_results / not attempted).**
   None of the flag-gate/`hasattr`-gate early returns above each try block
   were touched (verified by re-reading the file after editing — only the
   9 except-sites changed, confirmed by the isolated per-Edit-call diff in
   §6). Proven for 5 of the 6 flag-gated sections (all but
   `daemon_self_notes`, which has no flag) by
   `test_control_disabled_flag_returns_empty_unchanged[<section>]`; for
   narrative by the two direct disabled/missing-corpus tests in point 3
   above; and structurally by every existing test in the focused chunk
   (§7) and sweep (§9) passing unmodified — including
   `test_sep12_repository_status_context.py`'s
   `test_disabled_git_does_not_read_repository` and
   `test_git_timeout_does_not_substitute_stale_index`.
5. **Producer-internal swallows are read-only siblings, recorded with file:line and owning batch; limitation stated per section.**
   §12 records all six (git_memory `extract_commits`, proposal_store
   `query_proposals`, memory_retriever `get_skills`, memory_coordinator
   `get_unresolved_threads`, corpus_manager `get_narrative_context`, email
   service `_fan_out`) plus the self-notes annotation loop and an
   off-path caller — none edited, each with its owning batch (F11a, F11b,
   F12 ×2, or unowned) and the "still reads as no_results until F9-F12"
   consequence stated explicitly.
6. **Privacy: reasons are constant labels or exception class names only.**
   Every `reason` this batch produces is `"relevance_unavailable"` (a
   fixed label) or `type(e).__name__` (a class name) — never `str(e)` or
   query text. Proven by §13's marker-based tests (folded into the raising
   tests rather than a separate class, per the F7b precedent).

## 16. Milestone

Focused contract green: 534/534 passed (new file's 32 tests + 8 existing
files' 502 tests), 0 failed. Sweep green: 178+166+172+117 = 633/633 across
30 files, 0 failed (1 file, `test_graph_integration.py`, permanently
excluded per TEST EXCLUSION). Combined: **1,167 passed, 0 failed, 0
skipped** across 40 files this batch ran (1 new file, 8 focused existing
files, 30 sweep existing files — 2 of the 30 are F7a's/F7b's own new
files, run here as genuine importers, not re-authored). Failing-first:
10/32 failed on the unedited source (digest `18390202…` confirmed), the
22 passes being exactly the unaffected-by-design controls — no test-file
bug was found this time, unlike every prior F7 sub-batch. Ruff clean on
both changed/new files. Scan: dm18 `found` dropped 63→54 (exactly the 9
anchors), `new` stayed 0 for every gate scanner both before and after, and
`core/prompt/gatherer_knowledge.py` now has ZERO live dm18 findings at
all (every method in the file, across F7a+F7b+F7c, is fixed). No existing
test required a FIXTURE RULE edit anywhere (0 of 38 files touched). No
`data/` write at any of the 13 explicit checkpoints (before the first
pytest, after failing-first, after the focused chunk, after each of the 4
sweep chunks, after the scan) — `logs/` absent throughout.

## 17. Open items / limitations

- The six producer-internal swallows (§12) mean a failure INSIDE those
  producers still reads as `no_results` at this batch's boundary until
  F9-F12 land — stated per section in §12 and the response file.
- `memory/shutdown_processor.py:1312`'s `get_skills` call and
  `core/agentic/tools.py`/`core/agentic/controller.py`/`gui/handlers.py`'s
  `get_email_service()` calls are separate production consumers of the
  same producers, unaffected and out of this batch's ownership.
- `debug_graph_context.py:43` is an off-path manual debug script,
  confirmed unaffected (§12), not part of the class-guard's scope.
- No escalation: the manifest check, the pre-edit scan, the post-edit
  scan, and every test run came back exactly as the brief predicted; the
  only deviation from a strictly literal single-session reading of the
  ORDER section is the INTERRUPTION in §3, which is itself a rule-
  anticipated event, fully disclosed, and re-verified independently on
  resumption.
- F7c closes CGR-007's remaining anchors #75-#79 and #81-#84. CGR-007's
  #85-#92 (`gatherer_memory.py`/`gatherer_web.py`) remain open for **F8**
  (response `-4`), per the design doc's F7-split table.

## 18. 2026-09-14 — Final state (pre-handoff)

- `git status --short`: 105 lines (was 104 at the interruption checkpoint
  in §3; +1 for the new test file — `core/prompt/gatherer_knowledge.py`
  was already `M` from F7a/F7b's carried-over uncommitted state).
  `git diff --cached --name-only`: empty. HEAD:
  `328a8ecea1dae91de16f5974ea71727c4e2e1aa0` (unchanged throughout — no
  commit, no `git add` in any form). `git stash list`: empty, not
  touched. `git remote -v`: `origin /home/lukeh/Daemon_v1 (fetch)`,
  `origin DISABLED (push)` — not touched, unchecked again here only for
  the record.
- **Size:** 23 (source, isolated) + 409 (new test file) = **432 changed
  lines**, within the 450 cap (§14).
- **Final data/ and logs/:** identical to the §1 baseline (7 entries,
  `user_profile.json` unchanged at 598 bytes / mtime
  `13:26:21.672354716`); `logs/` absent. No STOP at any point in this
  batch.
- **Class-guard status:** F7c answers #75-#79 and #81-#84 in
  `CGR-20260913-007-3.md` (written next, LAST per ORDER). Still open:
  CGR-007's #85-#92 (F8); CGR-009 and CGR-010 (this batch does not touch
  them).

## 19. 2026-09-14 — Response file written (LAST, per ORDER)

`docs/execution/generalization/class_guard_responses/CGR-20260913-007-3.md`
(new, 326 lines via `wc -l`, sha256
`ae4f6dd10bf2e0b9c392cbeb99979bc807e06677198dc430ec8b5c6cc1cd71c0`).
Follow-up `-3` response for CGR-20260913-007, answering #75-#79 and
#81-#84, recording the six BC-58 producer-internal-swallow siblings with
their owning batches (F11a, F11b, F12 ×2, 2 unowned), the self-notes
inner-loop sibling, the off-path `debug_graph_context.py` caller, and the
mapping of #85-#92 to F8 (`-4`). Immutable from this point — no further
edits.

Final re-verification after writing the response file: `git status
--short` still 105 lines (the response file landed inside the already-
untracked `docs/execution/generalization/` directory, which `git status
--short` reports as one line regardless of how many files it contains —
matching F7a's and F7b's own packets' identical observation);
`git diff --cached --name-only` empty; HEAD unchanged at `328a8ec…`;
`data/` unchanged (7 entries); `logs/` absent.

## Parent review and integration (2026-09-14)

Accepted as delivered. The batch followed the tightened COMPLIANCE rules with no process deviation. It was interrupted once by an API session limit; the parent verified the state and resumed the same worker.

### Interruption

- **When:** the worker stopped at about 14:32 on an API session limit, right after recording its pre-edit scan.
- **Parent verification at 15:01, before resuming:**
  - `core/prompt/gatherer_knowledge.py` was unedited at `18390202…`;
  - the only new path was this packet (175 lines at the time), and no test or response file existed;
  - everything else matched `manifest_post_F7b`, nothing was staged, `data/` held the 7-entry baseline, `logs/` was absent, and no pytest was running.
- **Resume:** the same worker was resumed with its context intact, so nothing had to be reverted or moved. It recorded the interruption in §3 and continued at the test-file step.

### Manifest

- **Checksums:** `sha256sum -c manifest_post_F7b.txt` mismatches only on `core/prompt/gatherer_knowledge.py` (`00870a5b…`).
- **New paths:** exactly three:
  - `tests/unit/test_gatherer_outcomes_remaining_knowledge.py` (`0a3d61d7…`, 409 lines);
  - `class_guard_responses/CGR-20260913-007-3.md` (`ae4f6dd1…`, 326 lines);
  - this file.
- **Git state:** `git diff --cached --name-only` is empty, HEAD `328a8ec` is unchanged, and the reflog still shows only the clone entry.
- **No new stray files:** the only untracked file at the repository root is `scan_pre.stderr`.
- **`data/`:** identical to the post-F7a baseline (7 entries), and `logs/` is absent.
- **Parent edits after the worker returned, before this manifest:**
  - `briefs/F8a.md` and `briefs/F8b.md` were added;
  - `failure_outcome_design.md` was amended (the F8 split, and the dict-section limitation decision);
  - `briefs/PARENT_STATE.md` was updated.
- **Recorded tree:** `manifest_post_F7c.txt` records the tree after this section.

### Code review (the parent isolated this batch's hunks)

- **Typed-failure swaps (#75–#79, #82, #84):** `get_git_commits`, `get_proposed_features`, `get_procedural_skills`, `get_graph_context`, `get_unresolved_threads`, `get_daemon_self_notes` and `get_relevant_emails`' outer except each return `OutcomeList.failed(type(e).__name__)` instead of `[]`, with log lines unchanged.
- **#83:** the email ranking except returns `OutcomeList.failed("relevance_unavailable")`, still with no items, and the fail-closed comment stays.
- **#81:** `get_narrative_context`'s except keeps its warning and re-raises, with a comment naming the builder's existing try as the recorder.
- **Unchanged:** every legit early return, the self-notes annotation loop and all producer code.

### Tests (the parent read the whole file)

- **Six direct-swap sections, parametrized:**
  - a raising producer gives failed / RuntimeError with no marker in the reason;
  - healthy empty gives no_results, and healthy non-empty gives succeeded;
  - five sections with a feature flag keep their disabled control.
- **Emails:**
  - a raising embedder gives failed / relevance_unavailable (#83);
  - a raising `service.search` gives failed / class (#84);
  - controls: no cue gives no_results, and healthy ranked messages succeed.
- **Narrative:**
  - a raising producer makes the method re-raise;
  - the disabled flag and a missing corpus manager still return "";
  - through the builder, the prompt still builds, `narrative_state` is "", and `_section_outcomes["narrative"]` is failed / RuntimeError;
  - control: `max_narrative=0` gives no entry and no producer call.
- **Failing-first:** 10 failed and 22 passed at digest `18390202…`. There was one run with no test-fixture fix, per the packet.

### Parent rerun (guarded, capped, `-p asyncio`)

- **Focused** (MemAvailable 6,525 MB): the new file plus the 8 brief-named files gave **534 passed** (0:16.26, 1,745,036 KB).
- **Sweep A** (6,561 MB): **178 passed** (0:11.08, 1,110,788 KB).
- **Sweep B** (6,553 MB): **166 passed** (0:07.41, 1,636,040 KB).
- **Sweep C** (6,531 MB): **172 passed** (0:11.51, 1,685,704 KB).
- **Sweep D** (6,550 MB): **117 passed** (0:09.30, 1,175,984 KB).
- **Total:** **1,167 passed, 0 failed**, matching the worker. The sweep used the chunk lists in §8–§9, with `tests/unit/test_graph_integration.py` excluded as before.
- **`data/`:** unchanged after every chunk, and `logs/` is absent.
- **ruff 0.14.9:** clean on both files.
- **Scan:**
  - dm18: found 54, base 79, new 0, stale 25.
  - `core/prompt/gatherer_knowledge.py` now has 14 stale rows (#71–#84, with `get_relevant_emails` counted twice) and no live dm18 finding. The re-raising narrative except is not a dm18 finding.
  - dm01 (7/12/0/5), dm17 (36/39/0/3) and dm31 (3/3/0/0) are unchanged. Every gate scanner shows 0 new.

### Size

432 changed lines (23 source + 409 test), within the 450 cap.

### Class-guard status

- **CGR-007:** #71–#84 answered across F7a, F7b and F7c.
- **Still open:** #85–#91 (F8a, response `-4`), #92 (F8b, response `-5`), CGR-009 and CGR-010.
- **Response-mapping update:** the `-3` response maps #85–#92 to a single F8 `-4`. The parent's F8 split since then gives #85–#91 to `-4` (F8a) and #92 to `-5` (F8b).
