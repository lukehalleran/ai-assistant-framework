# F6a — [ACTIVE FEATURES] renders "could not check" for failed sections;
# obsidian= driven by config (ATTEMPT 2)

Worker evidence packet. Design source:
`docs/execution/generalization/failure_outcome_design.md`, "Real defects"
row (`core/prompt/formatter.py:793`), "Decisions per request" → CGR-007,
batch row F6a (parent split of F6 on 2026-09-14: F6a is the prompt-facing
formatter part; F6b is receipts — orchestrator / handlers / telemetry), and
the 2026-09-14 amendment ("Omitting a section also drops its guidance.
Default: no content instructions, only the `[ACTIVE FEATURES]` label.").
F6a closes **no** scanner anchor and writes **no** class-guard response
file, per the brief. BUG_CLASSES named by the brief: BC-47, CM-05.

**This is ATTEMPT 2.** Attempt 1 was cut off by an API usage limit
mid-edit (weekly limit, HTTP 429) while editing `core/prompt/formatter.py`.
The parent recorded the interruption, reversed the partial formatter edit
in the working tree only (`git apply -R` of a diff, no index write), and
removed attempt 1's untracked test file from `tests/unit/` (this needed an
owner-authorized `/usr/bin/rm` after the shell guard's own unlock path
proved broken — see
`docs/execution/generalization/batches/F6a_attempt1/ABORT.md`, which this
packet was permitted to read and did, but no other file in that directory
was read or reused). This attempt starts from the unedited formatter,
digest `2e83856b7f14957c95ae5d330d3e59aeb91f478d7d900ba840b034904bd9e3db`,
and its failing-first proof (§6 below) is independently produced, not
copied from attempt 1's reported numbers.

**Flag-semantics change to record explicitly:** contract point 2 changes
`obsidian=`'s source of truth. Today it is `_on_off(bool(notes))` — the
flag literally reads whether any notes were returned, so a healthy vault
with zero matching notes prints `obsidian=OFF` ("feature off") even though
Obsidian integration is fully enabled and working; a *failed* vault read
also prints `OFF`, indistinguishable from "disabled". After this edit,
`obsidian=` reads `_on_off(getattr(cfg, 'OBSIDIAN_ENABLED', False))`, like
every other flag in the inventory, so an enabled vault with no matching
notes this turn now reads `obsidian=ON` (no suffix) instead of `OFF`, and a
failed read reads `obsidian=ON(could not check)` instead of silently `OFF`.
This is a **behavior change for the no-notes case**, not only the failure
case — recorded per contract point 5's explicit instruction.

## 1. Manifest check (before any edit)

```
$ S=/tmp/claude-1000/-home-lukeh-daemon-exec-generalization/1f0f3407-5796-4278-85e4-0c7ba4f50aa9/scratchpad
$ sha256sum -c --quiet $S/manifest_post_A01b.txt          # exit 0, no output
$ { git diff --name-only; git ls-files --others --exclude-standard; } | \
    sort -u | diff - $S/manifest_paths_post_A01b.txt       # exit 0, no output
```

Both passed (2026-09-14T11:00 -0500). `git rev-parse HEAD`:
`328a8ecea1dae91de16f5974ea71727c4e2e1aa0`. The pre-edit tree matches the
parent's post-A01b record exactly.

Pre-edit state also recorded:
- `sha256sum core/prompt/formatter.py`:
  `2e83856b7f14957c95ae5d330d3e59aeb91f478d7d900ba840b034904bd9e3db` —
  matches the brief's digest and ABORT.md's recorded pre-edit digest exactly.
- `git diff --stat core/prompt/formatter.py`: empty (unmodified).
- `git diff --cached --name-only`: empty.
- `git remote -v`: `origin /home/lukeh/Daemon_v1 (fetch)`, `origin DISABLED
  (push)`.
- `git status --short`: matches the manifest's recorded path set exactly
  (55 modified tracked files + `api/launch_auth.py` and 24 other untracked
  paths carried over from earlier batches; none of this batch's two new
  files exist yet).
- `data/` (top-level, `ls -la --time-style=full-iso`): `benchmark_per_case.csv`,
  `chroma_db_v4/`, `chroma_multi/`, `embedding_migration_manifest.json`,
  `pipeline/`, `web_search_credits.json` — identical to the F2/F5-recorded
  snapshot.
- `logs/`: absent (`ls -ld` → "No such file or directory"), as expected.
- MemAvailable: 4758 MB ≥ 4000.

No escalation needed; proceeding.

## 2. Pre-edit scan (read-only, `scripts/check_bug_classes.py scan --root .`)

2026-09-14T11:03 -0500. Not pytest; this is the standing read-only
exception. `PYTHONPATH=.../scripts/bin` prefixed. Exit code 1 (non-zero
because of STALE baseline rows elsewhere in the tree; not a gate failure —
`new` is 0 for every gate scanner below, which is what this batch's edit
must preserve):

```
scanner                       mode    files  found  base  new  stale
----------------------------  ------  -----  -----  ----  ---  -----
dm01_raw_substring            gate    396    7      12    0    5
dm16_config_key_reachability  report  395    89     -     -    -
dm17_apply_without_guard      gate    622    36     39    0    3
dm18_except_returns_empty     gate    123    68     79    0    11
dm29_phrase_append_signature  report  0      0      -     -    -
dm31_live_state_default       gate    396    3      3     0    0
catalog                       gate    1      0      0     0    0
```

Identical found/base/new/stale numbers to F5's recorded pre-edit baseline
(`docs/execution/generalization/batches/F5.md` §5), apart from dm17's
`files` count (622 vs 620/621), which reflects other in-flight batches'
untouched files elsewhere in the tree, not this one. Confirmed via
`--json`:
- `core/prompt/formatter.py` DOES appear twice in `dm01_raw_substring`'s
  baselined findings (`_format_session_header`, lines 287 and 289: `if
  "(today)" in rel.lower():` / `elif "(yesterday)" in rel.lower():`) — a
  different function, untouched by this batch's edit, already part of the
  base=12 count, present in neither `.new` nor `.stale`.
- Neither `core/prompt/formatter.py` nor the not-yet-created
  `tests/unit/test_feature_inventory_outcomes.py` appears anywhere in
  `.new` or `.stale` for any scanner, and the strings
  `_build_feature_inventory`/`feature_inventory` appear nowhere in the
  scan JSON at all — `_build_feature_inventory` (747-837) is not a
  dm01/dm17/dm18/dm31 candidate today. This is the baseline the post-edit
  scan (§8 below) must reproduce with `new` still 0 everywhere.

## 3. New test file written

2026-09-14T11:10 -0500. `tests/unit/test_feature_inventory_outcomes.py`
written with the Write tool (new file; the FIXTURE RULE does not apply —
nothing existing was touched). `wc -l`: 304 lines. sha256:
`7702462a86930069c5d01bca4be89780aab85ce372c99d67f62607fce88d2705`.

Coverage, against the contract in the brief:
- `TestObsidianFlagSemantics` (4 tests): failed+enabled → `ON(could not
  check)`; disabled+no-outcomes → bare `OFF`; enabled+3 notes → `ON(3
  notes)`; enabled+no-notes+no-failure → bare `ON` (the changed case).
- `TestPerItemCouldNotCheckSuffix`: `graph_context` unavailable replaces a
  non-empty count (proves REPLACE, not append, even with stale leftover
  items in context); a parametrized case for the other 4 replace-suffix
  items (git_commits, reference_docs, threads, insights, skills); a
  parametrized narrative case for both flag states (append, not replace,
  since narrative carries no count).
- `TestCouldNotCheckLine`: the brief's exact two-section example
  (`upcoming_schedule`/`relevant_emails`) sorted; itemized sections
  excluded from the line even when failed; `web_search` failure produces
  no extra line and an unchanged label; the line is omitted (and line
  count stays 4) when nothing is NOT CHECKED.
- `test_reason_labels_never_appear_in_output`: a distinctive marker string
  used as every `reason` across 4 different failed/unavailable sections,
  asserted absent from the rendered text.
- `TestControls`: an unaffected-by-design control (Memory/Proactive/
  Analysis lines identical with no outcomes key, will pass before AND
  after the edit); the one case that legitimately changes (Knowledge line,
  no outcomes key, obsidian enabled with no notes — will FAIL pre-edit,
  pass post-edit, pre-edit value pinned in §4 below); all-succeeded
  outcomes produce no "could not check" text anywhere.
- `test_could_not_check_line_appears_inside_active_features_section`: the
  deployed `_assemble_prompt` path, isolating the `[ACTIVE FEATURES]`
  section text specifically (not just "appears somewhere in the prompt").

## 4. Failing-first proof (UNEDITED source)

Guard checked first (`/proc/comm` pytest guard: clear, no RUNNING python
under `-m pytest`); MemAvailable 4988 MB ≥ 4000. One command, before any
source edit:

```
$ sha256sum core/prompt/formatter.py && \
  PYTHONPATH=.../scripts/bin systemd-run --user --scope -p MemoryMax=6G \
  -p MemorySwapMax=512M /usr/bin/time -v env DAEMON_TEST_MODE=1 \
  CHROMA_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider \
  -p asyncio tests/unit/test_feature_inventory_outcomes.py
```

Digest: `2e83856b7f14957c95ae5d330d3e59aeb91f478d7d900ba840b034904bd9e3db`
(matches §1/the brief/ABORT.md exactly). Result: **14 failed, 7 passed**,
exit 1. Elapsed 0:04.77; Maximum resident set size 1,071,152 KB.

**14 failures** (every test that depends on the NOT-YET-IMPLEMENTED
contract — reading `_section_outcomes` and the new obsidian semantics):
`TestObsidianFlagSemantics::test_failed_personal_notes_with_flag_enabled`,
`TestObsidianFlagSemantics::test_enabled_no_notes_no_failure_shows_bare_on`,
`TestPerItemCouldNotCheckSuffix::test_graph_context_unavailable_replaces_count_not_appends`,
`TestPerItemCouldNotCheckSuffix::test_replace_suffix_for_each_itemized_section`
(all 5 parametrizations: git_commits, reference_docs, unresolved_threads,
proactive_insights, procedural_skills),
`TestPerItemCouldNotCheckSuffix::test_narrative_failed_appends_could_not_check`
(both parametrizations, flag True and False),
`TestCouldNotCheckLine::test_two_non_itemized_sections_produce_exact_sorted_line`,
`TestCouldNotCheckLine::test_itemized_sections_excluded_from_the_line_even_when_failed`,
`TestControls::test_no_outcomes_key_obsidian_reflects_new_semantics`,
`test_could_not_check_line_appears_inside_active_features_section`.

**7 passes** — the unaffected-by-design controls (the unedited function
ignores `_section_outcomes` entirely, so these hold both before and after):
`TestObsidianFlagSemantics::test_flag_disabled_and_no_outcomes_shows_bare_off`,
`TestObsidianFlagSemantics::test_enabled_with_three_notes_shows_count`,
`TestCouldNotCheckLine::test_web_search_failed_produces_no_extra_line_label_unchanged`,
`TestCouldNotCheckLine::test_line_omitted_when_no_not_checked_sections`,
`test_reason_labels_never_appear_in_output`,
`TestControls::test_memory_proactive_analysis_unaffected_by_missing_outcomes_key`,
`TestControls::test_all_succeeded_outcomes_no_could_not_check_anywhere`.

**Pinned pre-edit output** (for
`TestControls::test_no_outcomes_key_obsidian_reflects_new_semantics`, the
one control that legitimately changes per contract point 5): the
assertion diff shows the UNEDITED function's actual Knowledge line for
that test's context (`GIT_MEMORY_ENABLED=True`, `OBSIDIAN_ENABLED=True`,
`REFERENCE_DOCS_AUTO_SEED=True`, `git_commits=[{"content":"c1"}]`,
`personal_notes=[]`, `reference_docs=[{"content":"d1"}]`, no
`_section_outcomes` key) is exactly:

```
Knowledge: git_commits=ON(1) | obsidian=OFF | reference_docs=ON(1) | web_search=OFF
```

— confirming the manual trace used to write the test: `obsidian=OFF`
today even though `OBSIDIAN_ENABLED=True`, because the unedited code reads
`_on_off(bool(notes))`, not the config flag. Only the `obsidian=` token
differs from the post-edit expectation
(`obsidian=ON`); every other token on that line is unchanged.

Data listing after this chunk (`ls -la --time-style=full-iso data`,
`ls -ld --time-style=full-iso logs`): identical to §1 — no change, `logs/`
still absent.

## 5. Source edit

2026-09-14T11:16 -0500. One atomic Edit call, `_build_feature_inventory`
only (747-837 pre-edit); no other function or file touched. `git diff
--stat core/prompt/formatter.py`: **59 insertions, 8 deletions** (67
changed lines). sha256 after:
`a5036ffd4c9f99be507cf4d86574720bf8b735779c33ccc920215395e826bec6`.
`git diff --cached --name-only`: empty (no `git add` ever run).

What changed, mapped to the contract:
- `outcomes = context.get("_section_outcomes") or {}` read once, at the
  top of the `try:` block (point 1).
- `_not_checked(name)`: `isinstance(info, dict) and info.get("status") in
  ("failed", "unavailable")` — a missing key (`outcomes.get(name)` →
  `None`) is `not_checked=False`, matching "missing key means not
  attempted" / point 5's "byte-identical" default.
- `_suffix(name, items, unit="")`: the shared REPLACE helper for the six
  count-bearing itemized fields. It always records `name` into
  `shown_names` (so the field counts as "already shown by an inventory
  item" regardless of its outcome), then returns `"(could not check)"` when
  NOT CHECKED, else the original count expression (`f"({len(items)}{"
  {unit}" if unit else ""})"`), else `""` — textually the same output the
  unedited inline f-strings produced for every non-NOT-CHECKED case.
  Applied to: `graph_context` (unit `"edges"`, feeds `knowledge_graph=`),
  `git_commits`, `reference_docs`, `unresolved_threads` (unit `"open"`,
  feeds `threads=`), `proactive_insights` (feeds `insights=`),
  `procedural_skills` (feeds `skills=`).
- `obsidian=`: `_on_off(getattr(cfg, 'OBSIDIAN_ENABLED', False))` (was
  `_on_off(bool(notes))`) with its own priority chain — could-not-check,
  else `(N notes)`, else nothing — mirroring `_suffix` but inlined because
  its flag source changed too (point 2). `personal_notes` is added to
  `shown_names` unconditionally, same as every `_suffix` call.
- `narrative=`: unchanged flag computation
  (`NARRATIVE_CONTEXT_ENABLED`/`hasattr`/`narrative_state` fallback,
  byte-identical); `"(could not check)"` is now **appended** (not
  replacing anything, since narrative never had a count) when NOT CHECKED.
  `"narrative"` added to `shown_names` unconditionally.
- `web_search=`: **byte-identical** — no line inside the `if web_enabled:`
  block changed. `"web_search"` is added to `shown_names` unconditionally
  so it can never appear on the catch-all line, matching point 4's
  explicit carve-out, even though its own label is untouched.
- Trailing catch-all line: `sorted(name for name, info in
  outcomes.items() if name not in shown_names and isinstance(info, dict)
  and info.get("status") in ("failed", "unavailable"))`, joined and
  appended as `"Could not check this turn: " + ", ".join(...)` only when
  non-empty (point 4). No new header (point 4); token budgeting, section
  order, `section_instructions.py` and every other formatter section are
  untouched (point 6) — no other file was opened for writing.
- `intent=`, `escalation=`, `fact_verification=`, `truth_scorer=`,
  `dedup=`, and the dead (already-unused-in-original) `_count` helper are
  byte-for-byte unchanged.

## 6. Focused run (one chunk of 4, per the brief)

Guard re-checked (clear); MemAvailable 4952 MB ≥ 4000.

Non-unit read-first decision, recorded per R_common_rules "NON-UNIT
TESTS" (both brief-pre-cleared, both independently confirmed here before
running):
- `tests/test_eval/test_checks.py` (344 lines) — imports only
  `eval.checks` (pure `CheckResult`/`ResponseCheckResults` dataclasses and
  string-based checker functions over literal `prompt_text=` fixtures).
  Grepped for `ChromaStore|ModelManager|CorpusManager|open(...'w'|
  requests\.|urlopen|socket\.`: no matches. The two `[ACTIVE FEATURES]`
  occurrences (lines 159, 253) are literal fixture text used as a
  section-boundary marker for a checker function, not an assertion on
  `_build_feature_inventory`'s output — unaffected by this batch. **RUN.**
- `tests/test_eval/test_section_registry.py` (250 lines) — imports only
  `eval.section_registry` (a static `SECTION_REGISTRY` dict/dataclass, no
  I/O). Same grep: no matches. Its one `"[ACTIVE FEATURES]"` occurrence
  (line 226) is a literal header string in a `SectionDef`, unrelated to
  this batch's suffix/catch-all-line changes. **RUN.**

```
$ PYTHONPATH=.../scripts/bin systemd-run --user --scope -p MemoryMax=6G \
  -p MemorySwapMax=512M /usr/bin/time -v env DAEMON_TEST_MODE=1 \
  CHROMA_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider \
  -p asyncio tests/unit/test_feature_inventory_outcomes.py \
  tests/unit/test_feature_inventory.py \
  tests/test_eval/test_checks.py \
  tests/test_eval/test_section_registry.py
```

Result: **88 passed**, 0 failed, 0 skipped, exit 0. Elapsed 0:05.07;
Maximum resident set size 1,072,836 KB. (21 new tests — all
parametrizations counted — + 5 in `test_feature_inventory.py` + the
remaining 62 across the two `eval/` files.)

**No FIXTURE RULE edit to any existing file.** `test_feature_inventory.py`
sets no `_section_outcomes` key anywhere and asserts nothing about
`obsidian=`, so its five tests render byte-identically before and after
(confirmed green here); `test_checks.py`/`test_section_registry.py` never
call `_build_feature_inventory` or `_assemble_prompt` at all.

Data listing after this chunk: identical to §1/§4 — no change; `logs/`
still absent.

## 7. Sweep (3 chunks: 9 + 9 + 1)

Every `tests/unit/` importer of `core.prompt.formatter`/`PromptFormatter`,
established by `grep -rl "core\.prompt\.formatter\|PromptFormatter"
tests/unit/` before running anything: **19 files** (excludes
`test_feature_inventory.py` and this batch's own new file, both already
covered in the focused chunk).

Chunk A (MemAvailable 4975 MB, guard clear):

```
$ PYTHONPATH=.../scripts/bin systemd-run --user --scope -p MemoryMax=6G \
  ... python -m pytest -q -p no:cacheprovider -p asyncio \
  tests/unit/test_calendar_prompt.py \
  tests/unit/test_email_passive_context.py \
  tests/unit/test_evidence_transport.py \
  tests/unit/test_header_anchor_and_embed_cache.py \
  tests/unit/test_prompt_builder_methods.py \
  tests/unit/test_prompt_compat.py \
  tests/unit/test_proposal_filter.py \
  tests/unit/test_retrieval_context_quality.py \
  tests/unit/test_schedule_extraction.py
```

Result: **241 passed, 1 skipped**, exit 0. Elapsed 0:45.22; Maximum
resident set size 3,126,228 KB. The one skip
(`test_prompt_builder_methods.py:396`, "test only meaningful when 1h ago
crosses midnight") is the same pre-existing time-of-day skip F5 recorded
— unrelated to this batch.

Chunk B (MemAvailable 4986 MB, guard clear):

```
$ ... python -m pytest -q -p no:cacheprovider -p asyncio \
  tests/unit/test_section_outcomes.py \
  tests/unit/test_sep03_followups_continuity.py \
  tests/unit/test_sep05_evening_turn_audit.py \
  tests/unit/test_sep10_probe_dump_interpretation.py \
  tests/unit/test_sep10_web_search_gap.py \
  tests/unit/test_session_boundaries.py \
  tests/unit/test_source_excerpt_pipeline.py \
  tests/unit/test_stm_new_data_override.py \
  tests/unit/test_upload_keyword_score_leak.py
```

Result: **404 passed**, 0 failed, exit 0. Elapsed 0:23.42; Maximum
resident set size 1,727,340 KB. (The `RuntimeWarning: coroutine ...
_execute_mock_call was never awaited` in
`test_section_outcomes.py::TestGatherExceptionMarksAllFailedGatherError`'s
warnings summary is F5's own pre-existing `AsyncMock` fixture artifact,
unrelated to and unaffected by this batch — that file's tests all pass.
`test_sep10_web_search_gap.py::test_feature_inventory_reports_search_decision_honestly`
(7 parametrizations, all asserting `web_search=` labels only) passed
unmodified, confirming `web_search=` is untouched.)

Chunk C (MemAvailable 4963 MB, guard clear):

```
$ ... python -m pytest -q -p no:cacheprovider -p asyncio \
  tests/unit/test_upload_retrieval_pool.py
```

Result: **28 passed**, 0 failed, exit 0. Elapsed 0:06.73; Maximum resident
set size 1,116,268 KB.

**No FIXTURE RULE edit to any of the 19 swept files.** None sets
`_section_outcomes` or asserts on `obsidian=`/`[ACTIVE FEATURES]` text
(confirmed by the earlier grep for those literals, §"parent-verified
facts" cross-check — only `test_feature_inventory.py` and the two
`eval/` files matched, both already handled in §6).

Combined focused + sweep: **761 passed, 1 skipped, 0 failed** across 23
files (4 focused, 19 swept). Data listing after every one of the 3 sweep
chunks: identical to §1 — no change; `logs/` still absent.

## 8. Ruff

`ruff 0.14.9`.

```
$ PYTHONPATH=.../scripts/bin python -m ruff check \
  core/prompt/formatter.py tests/unit/test_feature_inventory_outcomes.py
All checks passed!
```

## 9. Post-edit bug-class scan (read-only)

```
scanner                       mode    files  found  base  new  stale
----------------------------  ------  -----  -----  ----  ---  -----
dm01_raw_substring            gate    396    7      12    0    5
dm16_config_key_reachability  report  395    89     -     -    -
dm17_apply_without_guard      gate    623    36     39    0    3
dm18_except_returns_empty     gate    123    68     79    0    11
dm29_phrase_append_signature  report  0      0      -     -    -
dm31_live_state_default       gate    396    3      3     0    0
catalog                       gate    1      0      0     0    0
```

Identical to §2's pre-edit baseline in every found/base/new/stale column
(dm17's `files` count ticks 622→623, one more file scanned by another
in-flight batch or this batch's own new test file entering the corpus —
not a finding change). `new` is 0 for every gate scanner, before and
after. Confirmed via `--json`:
- Neither `core/prompt/formatter.py` nor
  `tests/unit/test_feature_inventory_outcomes.py` appears in `.new` or
  `.stale` for any scanner.
- `dm01_raw_substring`'s two pre-existing `core/prompt/formatter.py`
  findings are byte-identical to §2 — still `_format_session_header`
  lines 287/289, untouched by this batch's edit at 747+.
- The strings `_build_feature_inventory`/`test_feature_inventory_outcomes`
  appear in no finding anywhere in the post-edit scan JSON.

**F6a closes no scanner anchor** (per the brief) — this scan is purely a
"did this edit introduce a new defect" check, and it did not.

## 10. Final data/logs listing and git state (pre-handoff)

`ls -la --time-style=full-iso data`: identical to every prior checkpoint
in this packet (§1, §4, §6, §7 ×3) — `benchmark_per_case.csv`,
`chroma_db_v4/`, `chroma_multi/`, `embedding_migration_manifest.json`,
`pipeline/`, `web_search_credits.json`, no new or changed entry across 6
pytest invocations in this batch. `ls -ld logs`: still "No such file or
directory".

- `git status --short`: 98 lines — the same modified/untracked path set
  as §1's pre-edit snapshot, plus exactly the two files this batch owns
  (`core/prompt/formatter.py`, already `M` pre-edit, content now changed;
  `tests/unit/test_feature_inventory_outcomes.py`, new `??`) and this
  packet (`docs/execution/generalization/batches/F6a.md`, new `??`, under
  the already-untracked `docs/execution/generalization/` tree). No other
  path changed.
- `git rev-parse HEAD`: `328a8ecea1dae91de16f5974ea71727c4e2e1aa0` —
  unchanged throughout (no commit).
- `git diff --stat core/prompt/formatter.py`: 59 insertions, 8 deletions
  (67 changed lines).
- `wc -l`: `tests/unit/test_feature_inventory_outcomes.py` 304 lines;
  `docs/execution/generalization/batches/F6a.md` 388 lines (this packet
  itself — not counted toward the size cap, matching F5's precedent).
- `git diff --cached --name-only`: empty — no `git add` in any form was
  ever run.
- `git remote -v`: unchanged, `origin DISABLED (push)`.

## 11. Contract — how each of the 6 points is implemented

1. **Read outcomes once; NOT CHECKED = failed or unavailable.**
   `outcomes = context.get("_section_outcomes") or {}` at the top of the
   `try:` block; `_not_checked(name)` is the single predicate every other
   piece of logic calls. Proven by every test in the new file — none
   reads any other status value as NOT CHECKED (`test_line_omitted_when_no_not_checked_sections`
   uses `"succeeded"`/`"no_results"` and asserts no could-not-check text).
2. **`obsidian=` from `OBSIDIAN_ENABLED`, priority-ordered suffix.**
   `_on_off(getattr(cfg, 'OBSIDIAN_ENABLED', False))` replaces
   `_on_off(bool(notes))`; suffix is could-not-check, else `(N notes)`,
   else nothing. Proven by all 4 `TestObsidianFlagSemantics` tests,
   including the changed bare-`ON` case
   (`test_enabled_no_notes_no_failure_shows_bare_on`).
3. **Per-item suffix REPLACES the count (APPENDS for narrative).**
   The shared `_suffix()` helper implements REPLACE for
   `graph_context`→`knowledge_graph`, `git_commits`, `reference_docs`,
   `unresolved_threads`→`threads`, `proactive_insights`→`insights`,
   `procedural_skills`→`skills`; narrative's could-not-check is appended
   inline since it never had a count. `web_search` is untouched — no line
   inside its `if web_enabled:` block changed. Proven by
   `test_graph_context_unavailable_replaces_count_not_appends` (stale
   leftover items do NOT leak through when NOT CHECKED — the suffix wins
   over content), the 5-way parametrized
   `test_replace_suffix_for_each_itemized_section`, the 2-way
   parametrized `test_narrative_failed_appends_could_not_check`, and
   `test_web_search_failed_produces_no_extra_line_label_unchanged`.
4. **Trailing "Could not check this turn:" line.** Every `_suffix()` call
   and the inline `personal_notes`/`narrative`/`web_search` handling
   records its name into `shown_names` unconditionally (regardless of
   check status), so the catch-all line only ever lists NOT CHECKED
   sections that were never itemized. `sorted(...)`, comma-joined, no new
   header, omitted when empty. Proven by
   `test_two_non_itemized_sections_produce_exact_sorted_line` (exact
   text, alphabetical order),
   `test_itemized_sections_excluded_from_the_line_even_when_failed`
   (`graph_context` shown via `knowledge_graph=`, not duplicated on the
   line), `test_web_search_failed_produces_no_extra_line_label_unchanged`,
   and `test_line_omitted_when_no_not_checked_sections`.
5. **No outcomes key / all succeeded-or-no_results → byte-identical
   apart from obsidian.** `test_memory_proactive_analysis_unaffected_by_missing_outcomes_key`
   pins the Memory/Proactive/Analysis lines exactly (passes on both the
   unedited and edited source — confirmed unaffected in §4's failing-first
   run). `test_no_outcomes_key_obsidian_reflects_new_semantics` pins the
   one line that legitimately changes, with the pre-edit actual value
   recorded in §4. `test_all_succeeded_outcomes_no_could_not_check_anywhere`
   proves `"succeeded"` outcomes never trigger could-not-check text.
6. **No change elsewhere.** `git diff --stat` touches only
   `core/prompt/formatter.py`, only inside `_build_feature_inventory`
   (747-837 pre-edit); no other function, `section_instructions.py`,
   `token_manager.py`, `builder.py` or any other file was opened for
   writing. All 5 pre-existing tests in the focused chunk
   (`test_feature_inventory.py`) and all 19 sweep files passed unmodified
   — no FIXTURE RULE edit anywhere in this batch.

**Exact rendered lines for one failing-sections example** (all config
flags at their patched-False default, everything context-empty,
`_section_outcomes={"upcoming_schedule": failed, "relevant_emails":
unavailable}` — `TestCouldNotCheckLine::test_two_non_itemized_sections_produce_exact_sorted_line`
and `test_could_not_check_line_appears_inside_active_features_section`;
first 4 lines cross-checked against §4's failing-first diff, which shows
them byte-identical on the unedited source too):

```
Memory: knowledge_graph=OFF | fact_verification=OFF | truth_scorer=OFF | dedup=OFF
Knowledge: git_commits=OFF | obsidian=OFF | reference_docs=OFF | web_search=OFF
Proactive: threads=OFF | insights=OFF | narrative=OFF
Analysis: intent=OFF | escalation=OFF | skills=OFF
Could not check this turn: relevant_emails, upcoming_schedule
```

## 12. Privacy / no-network note

No network access, no LLM/paid API call, no daemon restart, no `pip
install`. `_build_feature_inventory` never reads `info.get("reason")`
anywhere in the new code — only `info.get("status")` — so no reason label
can reach the rendered prompt structurally, not just by test luck; proven
by `test_reason_labels_never_appear_in_output`, which plants a
distinctive marker string as the `reason` on 4 different failed/
unavailable sections and asserts it absent from the output. No query text
or exception text is read or rendered anywhere in the function (unchanged
from before this batch). The standing `PYTHONPATH=.../scripts/bin python -c "import utils;
print(utils.__file__)"` sanity check was run once in this session (the
one explicit exception besides ruff/the scan to the pytest-only rule) and
printed `/home/lukeh/daemon_exec/generalization/utils/__init__.py` — this
clone, confirming the environment mandate holds. No other non-pytest code
(no `python -c`, `python3 -`, REPL or heredoc) ran in this batch. Every
file edit used the Edit or Write tool; `git add` was never invoked; every
pytest invocation (failing-first, the focused chunk, and all 3 sweep
chunks) used the guarded, capped, `-p asyncio` command, preceded by a
`/proc/comm` guard check and a MemAvailable check.

**Process deviations (disclosed in full; none affected the deployed
edit):** none. Every ORDER step ran in the sequence the brief specifies:
manifest → create this packet → pre-edit scan → write test file →
failing-first (digest in the same command) → edit → focused chunk →
sweep → ruff → post-edit scan → data/logs listing after every chunk →
this packet, appended incrementally after each step.

## 13. Size

`core/prompt/formatter.py`: 67 changed lines (`git diff --stat`: 59
insertions, 8 deletions) + `tests/unit/test_feature_inventory_outcomes.py`:
304 lines (new, `wc -l`) = **371 total changed lines** — under the
220-line soft target is exceeded (the six-item suffix helper plus the
catch-all line needed more surface than a single-flag fix), but
comfortably under the 450-line hard cap. No split needed.

## 14. Milestone

Focused contract green: 88/88 passed (21 new-file tests incl.
parametrizations + 5 `test_feature_inventory.py` + 62 across the two
`eval/` files), 0 failed. Sweep green: 241/241 (+1 pre-existing skip,
chunk A) + 404/404 (chunk B) + 28/28 (chunk C) = 673/673 (+1 pre-existing
skip) across 19 files. Combined: **761 passed, 1 skipped, 0 failed**
across 23 files this batch ran. Failing-first: 14/21 failed on the
unedited source (digest confirmed), the 7 passes being exactly the
unaffected-by-design controls. Ruff clean on both changed/new files. Scan:
`new` stayed 0 for every gate scanner, before and after; neither of this
batch's two files appears in any new finding or stale row. No existing
test required a FIXTURE RULE edit anywhere in this batch. No `data/`
write and no `logs/` appearance at any of the 7 checkpoints (before the
first pytest, and after each of the 6 pytest invocations: failing-first,
the focused chunk, and 3 sweep chunks).

## 15. Open items / limitations

- `web_search`'s own gatherer-level swallow (`_get_web_search_results`'s
  `except Exception: return None` → renders as `web_search=ON(no search
  this turn)` rather than reflecting a real failure) is untouched by this
  batch, exactly per the contract's point 3 ("`web_search` is unchanged")
  — it is assigned to F8 in the design doc (`gw #92 evidence`), not F6a.
- The debug record / turn record publication of `_section_outcomes`
  (`sections_not_checked`, reason labels for receipts) is F6b's scope,
  not this batch's — F6a renders only the prompt-facing label text, with
  no reason ever surfaced.
- No escalation: the manifest check, the pre-edit scan, the post-edit
  scan, and every test run came back exactly as the brief predicted; no
  process deviation occurred (§12).
- F6a closes no scanner anchor and writes no class-guard response file,
  per the brief; `docs/execution/generalization/failure_outcome_design.md`
  is unedited (out of ownership).
- **Attempt 2 note:** this packet's failing-first proof, test file and
  source edit are independently produced — no file under
  `batches/F6a_attempt1/` other than `ABORT.md` was read, and `ABORT.md`'s
  reported numbers (7 failed/7 passed against a 341-line, differently
  structured test file) were not reused or compared against; this
  attempt's own failing-first count (14 failed/7 passed against this
  attempt's own 304-line file) stands on its own evidence.

## Parent review and integration (2026-09-14)

Accepted as delivered (attempt 2). The parent made no edit to source or tests and found no process deviation.

### Manifest

- **Checksums:** `sha256sum -c manifest_post_A01b.txt` shows no mismatch.
- **New paths:** exactly three (177 paths in total):
  - `core/prompt/formatter.py` (`a5036ffd…`);
  - `tests/unit/test_feature_inventory_outcomes.py` (`7702462a…`, 304 lines);
  - this file.
- **Git state:** `git diff --cached --name-only` is empty, HEAD `328a8ec` is unchanged, and `git stash list` is empty.
- **`data/`:** identical to the post-F2 snapshot, and `logs/` is absent.
- **Attempt 1 is isolated:** its preserved artifacts under `batches/F6a_attempt1/` are unchanged and were not reused. The worker read only ABORT.md.
- **Report correction:** the worker's final report gave this file as 388 lines in one place; the actual count at handoff was 606.
- **Recorded tree:** `manifest_post_F6a.txt` records the tree after this section.

### Code review (the parent read the full 67-line diff)

- **Reading the outcomes:** `outcomes = context.get("_section_outcomes") or {}` is read once. `_not_checked(name)` means a dict entry whose status is failed or unavailable.
- **Count suffixes:** `_suffix(name, items, unit)` replaces the count with `(could not check)` for graph_context, git_commits, reference_docs, unresolved_threads, proactive_insights and procedural_skills. Healthy counts keep their units (`edges`, `open`). It also records the name in `shown_names`.
- **Obsidian:** `obsidian=` uses `OBSIDIAN_ENABLED`; the suffix is could-not-check, then the note count, then nothing.
- **Narrative:** `(could not check)` is appended to the narrative flag.
- **Web search:** `web_search` is marked as shown and its tri-state label is untouched.
- **Trailing line:** `Could not check this turn: <sorted names>` covers failed or unavailable sections not already shown. It carries no reason labels, no header, and is omitted when empty.
- **Robustness note (accepted):** a non-dict `_section_outcomes` would raise inside the existing try, which returns "" and drops the whole inventory. F5 always attaches a dict, so this cannot happen on the deployed path.

### Tests (the parent read the whole file)

- **Coverage:** obsidian semantics (4 tests), per-item replacement (graph plus 5 parametrized items), narrative (2 parametrized), the trailing line (exact sorted text, itemized exclusion, web_search exclusion, omitted when empty), privacy (marker absent), controls (lines unchanged without the key, the pinned obsidian change, no could-not-check when every section succeeded), and the deployed `_assemble_prompt` path (the line sits inside `[ACTIVE FEATURES]`).
- **Failing-first:** 14 failed and 7 passed at digest `2e83856b…`. The pinned pre-edit line reads `Knowledge: git_commits=ON(1) | obsidian=OFF | reference_docs=ON(1) | web_search=OFF`.

### Parent rerun (guarded, capped, `-p asyncio`)

- **Focused** (MemAvailable 5,145 MB): the new file, test_feature_inventory, tests/test_eval/test_checks and tests/test_eval/test_section_registry gave **88 passed** (0:04.90, 1,071,036 KB).
- **Sweep A** (5,155 MB): the 9 files in §7 gave **241 passed, 1 skipped** (0:25.94, 3,132,604 KB).
- **Sweep B** (5,142 MB): the 9 files in §7 gave **404 passed** (0:22.44, 1,724,872 KB).
- **Sweep C** (5,192 MB): test_upload_retrieval_pool gave **28 passed** (0:08.76, 1,116,768 KB).
- **Total:** **761 passed, 1 skipped, 0 failed**, matching the worker.
- **`data/`:** unchanged after every chunk, and `logs/` is absent.
- **ruff 0.14.9:** both files pass.
- **Scan:** identical gate counts to post-A01b, with 0 new:
  - dm18: found 68, base 79, new 0, stale 11;
  - dm01: found 7, base 12, new 0, stale 5;
  - dm17: found 36, base 39, new 0, stale 3;
  - dm31: found 3, base 3, new 0, stale 0.

### Behaviour change recorded for the owner

With the Obsidian vault enabled and no matching notes, the prompt now reads `obsidian=ON`, not `obsidian=OFF`. A failed vault search reads `obsidian=ON(could not check)`.

### Size

371 changed lines (67 source + 304 test), within the 450 cap.

### Class-guard status

- **F6a:** answers no anchor.
- **Still open:** CGR-007 (22 anchors: F7a/F7b/F7c, F8), CGR-009 and CGR-010.
