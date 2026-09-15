# F8b: web search section outcome agrees with the web receipt (typed
# failure for exceptions, provider errors and budget refusals)

Worker evidence packet. Design source:
`docs/execution/generalization/failure_outcome_design.md`, "F7 split and
gatherer outcome shape (parent decision, 2026-09-14)", the F8b bullet.
Request packet (read-only, the only file read in that checkout):
`/home/lukeh/daemon_exec/class_guards/docs/execution/class_guards/requests/CGR-20260913-007.md`.
Brief: `docs/execution/generalization/briefs/F8b.md`. Rules:
`docs/execution/generalization/briefs/R_common_rules.md` (applies in FULL).
Anchor answered: **#92** (`WebSearchMixin._get_web_search_results`).
BUG_CLASSES: BC-20, BC-47, CM-05. Response file (immutable once written,
written LAST): `docs/execution/generalization/class_guard_responses/CGR-20260913-007-5.md`.

This file is written INCREMENTALLY per the INTERRUPTION rule: created
right after the manifest check, then a dated entry appended after each
ORDER step.

## 1. 2026-09-14 — Manifest check (before any edit)

Run from the checkout root with
S=/tmp/claude-1000/-home-lukeh-daemon-exec-generalization/1f0f3407-5796-4278-85e4-0c7ba4f50aa9/scratchpad:

- (a) `sha256sum -c --quiet $S/manifest_post_F8a.txt` → prints nothing, exit 0. PASS.
- (b) `{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u | diff - $S/manifest_paths_post_F8a.txt` → prints nothing, exit 0. PASS.

Both checks pass — proceeding.

### Read-only git state (pre-edit)

- `HEAD`: `328a8ecea1dae91de16f5974ea71727c4e2e1aa0` (matches the packet's
  "Base tree" and the brief's parent-verified base).
- `git remote -v`: `origin  /home/lukeh/Daemon_v1 (fetch)` / `origin
  DISABLED (push)`. Not touched.
- `git stash list`: empty. Not touched.
- `git diff --cached --name-only`: empty (0 lines).
- `git status --short`: 107 lines (carried-over modified/untracked paths
  through F7a/F7b/F7c/F8a), matching the manifest path-set exactly (check
  (b) above passed) and matching F8a.md's own final recorded count exactly.

### Sanity check (one-time `import utils`, standing exception)

`PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin python -c "import utils; print(utils.__file__)"`
→ `/home/lukeh/daemon_exec/generalization/utils/__init__.py` — resolves
inside this clone, as required.

### Assigned-file digest (ORDER REMINDER: re-verify here, print again with
### the failing-first command)

`sha256sum core/prompt/gatherer_web.py` →
`270b6208e97519812e91d88c235dd5eea5845167e58c478b13c60a1eef85a005`

Matches the packet's/brief's recorded source SHA-256 for #92 exactly
(`270b6208…`). `wc -l core/prompt/gatherer_web.py` → 364 lines, matching
the brief's "364 lines" note.

### Re-verification of the PARENT-VERIFIED FACTS (read `core/prompt/gatherer_web.py`
### directly in full, offset 1-365, before any edit)

Every line the brief names matches exactly, no drift:

- `_get_web_search_results` (def 68; only caller: builder task "web_search"
  at builder.py 1513, confirmed by `grep -n "_get_web_search_results("
  core/prompt/builder.py` → single hit).
- `self.last_web_decision` initialised at 96-108, updated on every branch.
- Success path: `result = await manager.multi_search(...)` (242/253);
  `pages = list(getattr(result, "pages", None) or [])`, `last_web_decision["results"]
  = len(pages)` (262-263); a budget-blocked result sets
  `last_web_decision["requested"]`/`["blocked"] = "budget"` (268-273);
  `if result.has_results:` → `memory_id_map["WEB_SEARCH"]` → `return result`
  (275-294).
- else, if `result.error`: `last_web_decision["error"] = str(result.error)`,
  then `return None` (295-299) — the PROVIDER-ERROR path.
- with no error and no results (genuine empty, or a budget refusal with no
  error) → `return None`.
- Outer `except Exception as e` (301-304): `last_web_decision["error"] =
  type(e).__name__`, warning, `return None` — ANCHOR #92.
- `knowledge/web_search_manager.py` re-read (read-only): a budget refusal
  sets BOTH `error` and `blocked="budget"` (1317-1322 reservation refused,
  1401-1406 spend refused); a merged `MultiSearchResult` sets
  `error="; ".join(errors)` only when no pages remain, and
  `blocked="budget"` when any sub-query was budget-refused (2521-2532);
  other error results (`client not available`, `invalid key`, `provider
  failed (<reason>)`) carry `error` with no `blocked` — confirms the
  budget check must come FIRST inside the `not result.has_results` branch.
- `tests/unit/test_sep10_web_search_gap.py::test_gatherer_exposes_search_exception`
  (218-233, `_gatherer(decision, error=TimeoutError(...))`): `multi_search`
  raises directly → hits the ANCHOR #92 except. FIXTURE RULE candidate 1.
- `tests/unit/test_tavily_failure_outcomes.py::TestDeployedConsumers::test_gatherer_receipt_records_provider_failure`
  (312-339): `manager._tavily_client` raises inside `_tavily_search`, which
  F2 already converts to a `RetrievalError`; `_execute_search`'s Step-1
  `except RetrievalError` converts THAT to a `WebSearchResult(error=...)`
  with no `blocked` — so `multi_search` never raises, and this test hits
  the PROVIDER-ERROR result path (`result.error` set, not budget-blocked),
  not the outer except. FIXTURE RULE candidate 2.

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

Identical to F8a's recorded baseline (7 entries; `user_profile.json`
unchanged, 598 bytes, mtime `13:26:21.672354716`; `web_search_credits.json`
unchanged, 50 bytes, mtime `04:35:38.763134691`; `logs/` absent). No STOP
condition triggered.

MemAvailable at this point: 6635 MB (≥4000 required). Pytest guard: clear
(no RUNNING `python -m pytest` process).

## 2. 2026-09-14 — PRE-EDIT SCAN

```
$ PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin python scripts/check_bug_classes.py scan --root . > /tmp/scan_pre_f8b.txt 2> /tmp/scan_pre_f8b.stderr
exit: 1
```
(stderr file has 0 bytes; captured to /tmp only, never redirected into the
repo.)

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

Identical to F8a's recorded post-edit baseline (dm18: found 50, base 79,
new 0, stale 29) — no drift since F8a integration, as expected (no source
file has changed in between).

JSON run (approved exception; stderr 61 lines of stdlib
DeprecationWarning noise from the scanner's own dependency scan, to /tmp
only): `python scripts/check_bug_classes.py scan --root . --json >
/tmp/scan_pre_f8b.json 2> /tmp/scan_pre_f8b_json.stderr`, exit 1. Inspected
via `jq` only (no `python -c`/heredoc used).

dm18 live finding for `core/prompt/gatherer_web.py` (via `jq`, matching
the packet's anchor exactly):

```
WebSearchMixin._get_web_search_results line=304   -- #92
```

Exactly the 1 site this batch owns. data/ and logs/ after this step:
unchanged from §1's baseline (same 7 entries, same mtimes; `logs/` still
absent). No STOP.

## 3. 2026-09-14 — New test file written (Write tool only)

`tests/unit/test_gatherer_outcomes_web.py` (new, 292 lines via `wc -l`).
Drives the DEPLOYED `WebSearchMixin._get_web_search_results` directly
through a bare `WebSearchMixin.__new__` host (the
`test_sep10_web_search_gap.py` `_gatherer` precedent), plus one
builder-level case through the DEPLOYED `UnifiedPromptBuilder.build_prompt`
using a local copy of `test_independent_prompt_audit.py`'s
`full_builder`/`retrieval_limits` pattern (per the F5/F7a/F7b/F7c/F8a
precedent against cross-test-module coupling), with the REAL bound
`WebSearchMixin._get_web_search_results` wired onto the `SimpleNamespace`
context_gatherer via `types.MethodType` (the F8a `#90` builder-level-proof
precedent) so the builder's task genuinely drives this batch's own edited
body end to end. `MagicMock`/`AsyncMock`/`SimpleNamespace` fakes only — no
real WebSearchManager, rate limiter, cache, Tavily client or network
anywhere.

- `TestTypedFailures` (4 cases): a raising `multi_search` → failed/class,
  `== []`, receipt error = class, marker absent; a provider-error result
  → failed/provider_error, receipt error unchanged (still the raw text,
  contract point 5), marker absent from the new `.reason`; a
  budget-refused empty result → unavailable/budget, receipt blocked =
  "budget"; both `error` and `blocked="budget"` set together → still
  unavailable/budget (the budget-first-ordering proof, PARENT-VERIFIED
  FACT).
- `TestUnaffectedControls` (4 cases): a genuine empty result → `None`; a
  not-triggered decision → `None`, provider never awaited; the disabled
  toggle → `None`, provider never awaited; success → the same result
  object returned, citation tracking unaffected.
- `TestThroughBuilder` (1 case): a raising search through the deployed
  builder → `_section_outcomes["web_search"] == {"status": "failed",
  "reason": "RuntimeError"}`, `web_search_results == []` (the web section
  absent), marker absent from `_section_outcomes`, and the deployed
  `PromptFormatter._build_feature_inventory` still renders
  `web_search=ON(error)` (the receipt-only label is untouched).

Privacy: every raising fixture's exception message and the provider-error
text carries the marker `F8BMARKQ23_sensitive_detail_must_not_leak`; every
test on a new typed `.reason` asserts the marker is absent (folded into
the existing tests rather than a separate privacy test class, per the
F7b/F7c/F8a precedent).

## 4. 2026-09-14 — FAILING-FIRST (digest printed in the same command, before any source edit)

Guard checked first (clear, no RUNNING `python -m pytest`); MemAvailable
6644 MB (≥4000).

```
$ sha256sum core/prompt/gatherer_web.py
270b6208e97519812e91d88c235dd5eea5845167e58c478b13c60a1eef85a005  core/prompt/gatherer_web.py
$ systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=512M /usr/bin/time -v env \
    DAEMON_TEST_MODE=1 CHROMA_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin python -m pytest -q \
    -p no:cacheprovider -p asyncio tests/unit/test_gatherer_outcomes_web.py
```

Digest confirmed matching §1 (`270b6208…`). Result: **5 failed, 4 passed**,
exit 1, wall 0:04.75, peak RSS 1,069,924 KB. No test-file bug was found —
every failure and every pass landed exactly where designed on the first
attempt.

Failing (5, exactly the sites this batch is meant to fix):
- `TestTypedFailures::test_raising_multi_search_is_failed_with_exception_class` — ANCHOR #92 (outer except)
- `TestTypedFailures::test_provider_error_result_is_failed_provider_error` — provider-error sibling
- `TestTypedFailures::test_budget_refused_empty_result_is_unavailable_budget` — budget sibling
- `TestTypedFailures::test_budget_and_error_both_set_checks_budget_first` — budget-first-ordering proof
- `TestThroughBuilder::test_raising_search_is_failed_and_label_reads_error_web_section_absent` — builder-level proof

Every failure shape confirms the defect directly: `outcome_status(result)
== ('no_results', '')` instead of `('failed', ...)`/`('unavailable', ...)`;
the builder-level test shows `_section_outcomes["web_search"] ==
{'status': 'no_results', 'reason': ''}` instead of `{'status': 'failed',
'reason': 'RuntimeError'}`.

Passing (4, the "existing behaviour is unchanged" controls, correctly
green before any edit): `test_genuine_empty_result_stays_none`,
`test_not_triggered_decision_stays_none`,
`test_disabled_deliberate_non_search_stays_none`,
`test_success_returns_the_same_result_object`.

data/ and logs/ after this run:
```
$ ls -la --time-style=full-iso data   # unchanged from §1's baseline (same 7 entries, same mtimes)
$ ls -ld --time-style=full-iso logs   # still absent
```
No STOP condition triggered.

## 5. 2026-09-14 — Source edit (2 small complete Edit calls, ownership scope only)

`core/prompt/gatherer_web.py`, ownership scope only (one import line plus
the body of `_get_web_search_results`):

- **Import** (after `from datetime import datetime`): `from
  utils.retrieval_outcome import OutcomeList`.
- **`else:` branch** (not `result.has_results`, 295-299): after the
  unchanged receipt write and debug log, three new lines — budget checked
  FIRST (`getattr(result, "blocked", None) == "budget"` →
  `OutcomeList.unavailable("budget")`), then provider-error
  (`getattr(result, "error", None)` → `OutcomeList.failed("provider_error")`
  — a constant label, never `str(result.error)`), then the unchanged
  `return None` for a genuine empty result — ANCHOR #92 siblings 2 and 3.
- **Outer `except Exception as e` (301-304)**: the receipt write
  (`last_web_decision["error"] = type(e).__name__`) and the warning log are
  unchanged; `return None` → `return OutcomeList.failed(type(e).__name__)`
  — ANCHOR #92.
- The success path (`if result.has_results:` block, 275-294) is
  byte-for-byte unchanged. Every deliberate non-search early return
  (disabled, intent veto, crisis veto, no manager, manager unavailable,
  trigger unavailable, not-triggered) is byte-for-byte unchanged, still
  `return None`. `_cached_web_evidence` and `should_trigger_web_search`
  are untouched.

```
$ sha256sum core/prompt/gatherer_web.py
b6dae37f5096dc41e8e06ac8a1d13f6975221da18d9d084988a8f6bf7130c37f  core/prompt/gatherer_web.py
$ git diff --stat core/prompt/gatherer_web.py
 core/prompt/gatherer_web.py | 20 +++++++++++++++++++-
 1 file changed, 19 insertions(+), 1 deletion(-)
```

This is the ONLY batch touching `core/prompt/gatherer_web.py` — `git diff
--stat` here is authoritative and isolated. **20 changed lines** (19
insertions, 1 deletion). `git diff --cached --name-only`: empty (no `git
add` ever run).

data/ and logs/: unchanged from §1's baseline. No STOP.

## 6. 2026-09-14 — Focused run on the edited source (new file alone)

Guard clear, MemAvailable 6559 MB (≥4000):

```
$ systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=512M /usr/bin/time -v env \
    DAEMON_TEST_MODE=1 CHROMA_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin python -m pytest -q \
    -p no:cacheprovider -p asyncio tests/unit/test_gatherer_outcomes_web.py
```

Result: **9 passed**, 0 failed, 0 skipped, exit 0, wall 0:04.73, peak RSS
1,069,188 KB. Every test that failed against the unedited source in §4 now
passes; every control still passes. data/ and logs/: unchanged. No STOP.

## 7. 2026-09-14 — FIXTURE RULE edits (2 existing tests)

Both edits reconfirmed by exact-diff against the pre-edit content (the
second file, `test_tavily_failure_outcomes.py`, is untracked since F2 —
its pre-edit content was reconstructed byte-for-byte from this session's
own earlier full read and hashed BEFORE editing; the reconstruction's
SHA-256, `63b036a6c843f800d78f9646deec3219f5143628a179db02cfeadc070d722f12`,
matches F2's own parent-review-recorded digest for this file exactly,
confirming the reconstruction is authoritative).

- **`tests/unit/test_sep10_web_search_gap.py::test_gatherer_exposes_search_exception`**
  (the ANCHOR #92 except path — `_gatherer(decision, error=TimeoutError(...))`
  makes `multi_search` raise directly). Added one import
  (`from utils.retrieval_outcome import outcome_status`) and, in the test
  body, replaced the bare `assert ... is None` with `result = await
  gatherer._get_web_search_results(T1)` followed by `assert
  outcome_status(result) == ("failed", "TimeoutError")` and `assert result
  == []`. Every existing receipt assertion below it (`triggered`,
  `source`, `results`, `error`) is kept, unchanged.
  `sha256sum`: `12f2f3494da16770356d73233562d45e500cf4b352e8374dac2d55b6e88b2ae1`
  (238 lines, `wc -l`). `git diff --stat`: 6 insertions, 1 deletion (7
  changed lines).
- **`tests/unit/test_tavily_failure_outcomes.py::TestDeployedConsumers::test_gatherer_receipt_records_provider_failure`**
  (the PROVIDER-ERROR result path — `manager._tavily_client` raises, which
  F2's `_tavily_search` converts to a `RetrievalError`, which
  `_execute_search`'s Step-1 except converts to a `WebSearchResult(error=...)`
  with no `blocked`; `multi_search` never raises). `outcome_status` was
  already imported (F2, line 41). Replaced `assert result is None` with
  `assert outcome_status(result) == ("failed", "provider_error")` and
  `assert result == []`, immediately above the two unchanged receipt/marker
  assertions (`last_web_decision["error"]` truthy, marker absent from it).
  `sha256sum`: `a12919f6a53ee410ebb21123c23c13254014f7f4006efa2400deb9a7f3a68656`
  (344 lines, `wc -l`). Pre-edit (reconstructed, verified) →
  `63b036a6c843f800d78f9646deec3219f5143628a179db02cfeadc070d722f12`
  (339 lines). Isolated diff: 6 insertions (5 lines shown plus the
  replaced assertion line), 1 deletion.

`git diff --cached --name-only`: still empty. No other existing test file
touched.

## 8. 2026-09-14 — Focused chunks (2 chunks, per the brief's "9 web test
## files … (2 chunks)")

Guard clear before each; MemAvailable checked before each (6586, 6612 MB —
both ≥4000).

**Chunk 1** (9 files: the new file, the 2 FIXTURE RULE files, and the 6
"Other references" files named by the brief) →
`PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=512M /usr/bin/time -v env DAEMON_TEST_MODE=1 CHROMA_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider -p asyncio tests/unit/test_gatherer_outcomes_web.py tests/unit/test_sep10_web_search_gap.py tests/unit/test_tavily_failure_outcomes.py tests/unit/test_sep12_web_evidence_budget.py tests/unit/test_sep12_followup_budget_outcomes.py tests/unit/test_sep09_live_controls.py tests/unit/test_web_fallback_general_intent.py tests/unit/test_sep12_repository_status_context.py tests/unit/test_independent_prompt_audit.py`
→ **186 passed**, 0 failed, exit 0, wall 0:15.32, peak RSS 1,728,808 KB. 4
warnings, all pre-existing (3 SWIG DeprecationWarnings + 1 pre-existing
`websockets.legacy` DeprecationWarning from
`test_sep12_web_evidence_budget.py::TestDeliverySites::test_enhanced_turn_discloses_the_blocked_search_once`),
unrelated to this batch.

**Chunk 2** (1 file: `test_sep12_search_budget_reservation.py`, the source
of the `_make_manager`/`_page` fixtures both FIXTURE RULE-edited files and
the new file's builder-level test indirectly rely on) →
`... python -m pytest -q -p no:cacheprovider -p asyncio tests/unit/test_sep12_search_budget_reservation.py`
→ **13 passed**, 0 failed, exit 0, wall 0:01.90, peak RSS 761,884 KB.

**No existing test outside the two named FIXTURE RULE tests required an
edit.** `test_sep12_web_evidence_budget.py`'s and
`test_sep12_followup_budget_outcomes.py`'s own budget-refusal assertions
all hit the PRE-`multi_search` trigger-level veto (`not decision.should_search`,
a deliberate non-search this batch does not touch) or the SUCCESS path
(one sub-query funded, `result.has_results` True) — never the
`not result.has_results` branch this batch edited; confirmed by reading
every `result is None`/`is not None` assertion in both files before
running. `test_sep09_live_controls.py`'s `_recording_gatherer` reaches the
new provider-error branch (`NS(has_results=False, error="synthetic
provider; no network")`) but asserts only on `calls` (dispatch counts),
never on `_get_web_search_results`'s return value. `test_web_fallback_general_intent.py`
and `test_sep12_repository_status_context.py` only exercise the success
path or mock `_get_web_search_results` entirely. `test_independent_prompt_audit.py`
never triggers a producer exception in this method.

Combined chunk 1 + chunk 2: **199 passed, 0 failed, 0 skipped** across 10
files (1 new file, 2 FIXTURE RULE files, 7 other files).

data/ and logs/ after each chunk: unchanged (7 entries, same mtimes;
`logs/` absent). No STOP.

## 9. 2026-09-14 — SWEEP set identified

`grep -rl "gatherer_web\|WebSearchMixin\|core\.prompt\.context_gatherer" tests/unit/`
→ 15 files. Widened with the F7a/F7b/F7c/F8a-precedent check for the bare
class name `ContextGatherer`: `grep -rl "ContextGatherer" tests/unit/` →
14 files, union 18.

Minus the 10 already run in the focused chunks (§8) and the permanently
excluded `tests/unit/test_graph_integration.py` (TEST EXCLUSION) leaves 7
candidates, each checked directly:

- `tests/unit/test_narration_turn_audit_fixes.py` — genuine importer
  (`ContextGatherer.__new__(ContextGatherer)`, a bare host; `__init__`
  never runs). **RUN.**
- `tests/unit/test_prompt_builder_self_report_trim.py` — genuine importer
  (`MagicMock(spec=ContextGatherer)`, a spec double, never a real
  construction). **RUN.**
- `tests/unit/test_proposal_filter.py` — genuine importer
  (`ContextGatherer(mock_coordinator, mock_model, mock_token)` with a
  plain, unrestricted `MagicMock()` coordinator — `hasattr(mc,
  'user_profile')` is `True`, confirmed safe by the parent in F7a's
  review, re-confirmed here by reading the exact call sites at lines
  405/421). **RUN.**
- `tests/unit/test_session_diff.py` — genuine importer
  (`ContextGatherer(memory_coordinator=mc, ...)` with a plain
  `MagicMock()` coordinator, same safe shape). **RUN.**
- `tests/unit/test_wiki_disambiguation_filter.py` — genuine importer
  (`ContextGatherer.__new__(ContextGatherer)`, a bare host). **RUN.**
- `tests/unit/test_prompt_compat.py` — genuine importer (`from
  core.prompt import ContextGatherer` at line 21, an import-smoke test).
  **RUN.**
- `tests/unit/test_gatherer_outcomes_memory.py` — F8a's own new file;
  genuine importer via its own `_G()`/`full_builder` helpers, and it
  stubs `_get_web_search_results = AsyncMock(return_value=[])` on its bare
  gatherer host (confirmed tmp_path/fake-only throughout, matching F8a's
  own recorded precedent). **RUN.**

Excluded, confirmed NOT genuine importers of this batch's file by reading
each:

- `tests/unit/test_proposal_risk.py` — the only match is a literal string
  path `"core/prompt/gatherer_web.py"` inside a classification-input list
  (`classify_proposal([...])`); no import, no construction.
- `tests/unit/test_ordered_slice_guard.py` — the only match is a STRING
  inside an allowlist tuple (`("core/prompt/context_gatherer.py",
  "ContextGatherer._bounded", ...)`); no import (matches the identical
  F7a/F7b/F7c/F8a finding for this same file).
- `tests/unit/test_upload_retrieval_pool.py` — the only match is a
  docstring root-cause note (`ContextGatherer.get_user_uploads pooled`);
  no import, no construction (matches the identical F7a/F8a finding for
  this same file).
- `tests/unit/test_codex_followups.py` — the only match is a log-format
  regex literal (`r'\[ContextGatherer\] IMAGE DEBUG)'`); its actual
  imports (`json`, `logging`, `re`, `pathlib.Path`, `pytest`,
  `memory.user_profile.profile_shape_error`, `utils.safe_json.CorruptStoreError`)
  never touch `core.prompt.context_gatherer` or `core.prompt.gatherer_web`
  at all — not a genuine importer.

7 files for the sweep, one chunk (≤9).

## 10. 2026-09-14 — Sweep chunk

Guard clear, MemAvailable 6601 MB (≥4000):

```
$ systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=512M /usr/bin/time -v env \
    DAEMON_TEST_MODE=1 CHROMA_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin python -m pytest -q \
    -p no:cacheprovider -p asyncio \
    tests/unit/test_narration_turn_audit_fixes.py \
    tests/unit/test_prompt_builder_self_report_trim.py \
    tests/unit/test_proposal_filter.py \
    tests/unit/test_session_diff.py \
    tests/unit/test_wiki_disambiguation_filter.py \
    tests/unit/test_prompt_compat.py \
    tests/unit/test_gatherer_outcomes_memory.py
```

Result: **140 passed**, 0 failed, 0 skipped, exit 0, wall 0:07.27, peak RSS
1,665,220 KB. 8 warnings: 3 pre-existing SWIG DeprecationWarnings + 5
pre-existing `RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call'
was never awaited` from `test_prompt_builder_self_report_trim.py`'s own
`session_reflections.sort` line in `core/prompt/builder.py:1679` — a file
this batch does not touch (same warning every prior F5/F7/F8a chunk that
ran this file recorded).

**No existing test required a FIXTURE RULE edit in the sweep.** None of
the 7 files raises a producer exception through `_get_web_search_results`
or asserts on its return value on the failure path.

data/ and logs/ after this chunk: unchanged (7 entries, same mtimes;
`logs/` absent). No STOP.

**Sweep total: 140 passed, 0 failed, 0 skipped** across 7 files.
**Grand total, this batch (focused + sweep): 199 + 140 = 339 passed, 0
failed, 0 skipped** across 17 files. No existing test required a FIXTURE
RULE edit anywhere beyond the 2 the brief named.

## 11. 2026-09-14 — ruff

```
$ PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin python -m ruff check \
    core/prompt/gatherer_web.py tests/unit/test_gatherer_outcomes_web.py \
    tests/unit/test_sep10_web_search_gap.py tests/unit/test_tavily_failure_outcomes.py
All checks passed!
```
ruff 0.14.9.

## 12. 2026-09-14 — Post-edit scan (stderr to /tmp only)

```
$ PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin python scripts/check_bug_classes.py scan --root . > /tmp/scan_post_f8b.txt 2> /tmp/scan_post_f8b.stderr
exit: 1   (pre-existing STALE rows across dm01/dm17/dm18, same as every prior batch)
stderr: 0 bytes
```

Full scanner table:

```
scanner                       mode    files  found  base  new  stale
dm01_raw_substring            gate    396    7      12    0    5
dm16_config_key_reachability  report  395    89     -     -    -
dm17_apply_without_guard      gate    629    36     39    0    3
dm18_except_returns_empty     gate    123    49     79    0    30
dm29_phrase_append_signature  report  0      0      -     -    -
dm31_live_state_default       gate    396    3      3     0    0
catalog                       gate    1      0      0     0    0
```

Compared to pre-edit (§2): dm18 `found` dropped **50 → 49** (exactly the 1
anchor fixed: #92), `new` is **0** across every scanner (no new finding
introduced anywhere in the tree by this edit), `stale` rose **29 → 30**
(the 1 newly-STALE row below, plus the 29 pre-existing ones unrelated to
this batch). `dm17_apply_without_guard`'s `files` count ticked 628→629
(this scanner's file selection includes `tests/`, and this batch adds one
new test file; its own `found`/`new`/`stale` are unchanged, a benign
files-processed count effect, not a finding).

JSON run (approved exception; stderr 62 lines of stdlib
DeprecationWarning noise, to /tmp only): `python scripts/check_bug_classes.py
scan --root . --json > /tmp/scan_post_f8b.json 2> /tmp/scan_post_f8b_json.stderr`,
exit 1.

dm18 finding for #92, live vs STALE (via `jq` — the expected split,
exactly as designed):

```
Live findings for core/prompt/gatherer_web.py (jq select .path==...): 0 rows
STALE: dm18_except_returns_empty: core/prompt/gatherer_web.py [WebSearchMixin._get_web_search_results] 'return None'
```

This is the designed handoff (F7a/F7b/F7c/F8a precedent): the class-guard
owner removes this row and marks it `confirmed_fixed` with this request ID
after integration.

data/ and logs/ after the scan: identical to §1's baseline (7 entries,
`user_profile.json` unchanged); `logs/` still absent. No STOP.

## 13. BC-58 sibling search

`grep -rn "_get_web_search_results(" --include=*.py .` (excluding
`tests/`/`docs/`): the only production caller is
`core/prompt/builder.py:1513` (`self.context_gatherer._get_web_search_results(...)`,
the builder's "web_search" task), matching the PARENT-VERIFIED FACT
exactly — no sibling call site anywhere else in the tree.

Within the SAME method, the provider-error and budget-refusal branches
(the `not result.has_results` else-branch's two new `if` checks) are BC-58
siblings of ANCHOR #92 itself: all three sites shared the identical
"a real problem collapses into the same `None` a genuine empty search
returns" defect, and all three are fixed by this one edit — the request
packet's own note for #92 ("confirm that the web evidence receipt
distinguishes this outcome") already named the receipt as the source of
truth this batch's return value now agrees with.

`multi_search`'s own BC-47 aggregation sibling (recorded, not fixed, by
F2's response `CGR-20260913-008.md`: a joined `errors` string is dropped
the instant any sub-query returns pages) and
`WebSearchManager._select_links_for_following`'s dm18 finding (also
recorded, not fixed, by F2) are outside this batch's ownership
(`knowledge/web_search_manager.py` is read-only here) — carried forward
unchanged, not rediscovered.

## 14. Privacy / no-network note

No network access, no LLM/paid API call, no daemon restart, no `pip
install`, no real WebSearchManager/rate limiter/cache/Tavily client
anywhere (every manager in the new test file is a bare `MagicMock`/
`AsyncMock`/`SimpleNamespace`; the two FIXTURE RULE-edited tests and the
9 "other reference" focused-chunk files already used `_make_manager`
(tmp_path-scoped) or pure fakes, unedited by this batch). Every reason
this batch's source produces is either `type(e).__name__` (an exception
class name) or a fixed constant label (`"provider_error"`, `"budget"`) —
never `str(result.error)` or query/exception-message text, enforced
structurally (no site in this batch's diff ever calls `str(e)` or
`str(result.error)` when building the NEW typed `.reason`; the PRE-EXISTING
`last_web_decision["error"] = str(result.error)` receipt write at line 297
is untouched, per contract point 5's "byte-for-byte unchanged" requirement
— it was already writing that text before this batch, and stays that way).
Proven by the marker-based tests: `MARKER = "F8BMARKQ23_sensitive_detail_must_not_leak"`
is embedded in every raising fixture's exception message and in the
provider-error text, and every test on a new `.reason` asserts the marker
is absent from it (`test_raising_multi_search_is_failed_with_exception_class`,
`test_provider_error_result_is_failed_provider_error`,
`test_raising_search_is_failed_and_label_reads_error_web_section_absent`),
while separately confirming the OLD receipt field still carries the marker
unchanged (proving contract point 5, not a privacy regression — that
field's behavior predates this batch). The one standing `python -c "import
utils; print(utils.__file__)"` sanity check was run once at the top of
this session (§1); no other non-pytest code ran anywhere in this batch.
Every file edit used the Edit or Write tool; `git add` was never invoked;
every pytest invocation (failing-first, both focused chunks, and the
sweep chunk) used the guarded, capped, `-p asyncio` command, preceded by a
`/proc/comm` guard check and a MemAvailable check.

**Process deviations (disclosed in full):** one process note, not a rule
breach — `tests/unit/test_tavily_failure_outcomes.py` is untracked (added
by F2, never `git add`ed by any batch, per the GIT INDEX rule), so its
pre-edit SHA-256 could not be obtained by `git show`/`git diff`. It was
instead reconstructed byte-for-byte from this session's own earlier full
`Read` of the file (before any edit) and hashed with `sha256sum` BEFORE
the FIXTURE RULE edit was made; the reconstruction's digest,
`63b036a6c843f800d78f9646deec3219f5143628a179db02cfeadc070d722f12`,
matches F2's own parent-review-recorded digest for this file exactly (see
`class_guard_responses/CGR-20260913-008.md`'s parent review, "New paths"),
independently confirming the reconstruction is authoritative and that no
batch between F2 and F8b touched the file. No source edit occurred before
the failing-first proof was recorded. No file was created outside this
batch's ownership (`tests/unit/test_gatherer_outcomes_web.py`,
`docs/execution/generalization/batches/F8b.md`, and the response file
below, plus the two named FIXTURE RULE edits). No INTERRUPTION occurred in
this session.

## 15. Size

`core/prompt/gatherer_web.py`: **20 changed lines** (19 insertions, 1
deletion — `git diff --stat`, authoritative and isolated since this is the
only batch touching this file, §5).
`tests/unit/test_gatherer_outcomes_web.py`: **292 lines** (new, `wc -l`).
`tests/unit/test_sep10_web_search_gap.py` (FIXTURE RULE): **7 changed
lines** (6 insertions, 1 deletion, `git diff --stat`).
`tests/unit/test_tavily_failure_outcomes.py` (FIXTURE RULE, untracked —
isolated via `git diff --no-index` against the verified pre-edit
reconstruction): **7 changed lines** (6 insertions, 1 deletion).

**Total: 20 + 292 + 7 + 7 = 326 changed lines** — over the 300-line soft
target (driven by the builder-level proof test's own local
`full_builder`/`retrieval_limits` copy, ~60 lines, required by the brief's
explicit "through the builder" TESTS requirement) but comfortably under
the 450-line hard cap. No split needed.
`docs/execution/generalization/batches/F8b.md` and the response file are
evidence/response artifacts, not counted toward the cap, per precedent
(F4.md, F7a.md, F7b.md, F7c.md, F8a.md).

## 16. Contract — how each of the 6 points is implemented

1. **Anchor #92 (outer except): receipt write and log kept; `return
   OutcomeList.failed(type(e).__name__)` instead of `None`.**
   §5's single-line swap inside the unchanged `except Exception as e:`
   block. Proven by
   `TestTypedFailures::test_raising_multi_search_is_failed_with_exception_class`
   (`outcome_status(result) == ("failed", "RuntimeError")`, `result ==
   []`, receipt `error == "RuntimeError"`, marker absent) and by the
   FIXTURE RULE edit to `test_gatherer_exposes_search_exception` (the
   identical except site, `TimeoutError`).
2. **Budget refusal, checked FIRST inside the `not result.has_results`
   branch: `getattr(result, "blocked", None) == "budget"` → receipt
   writes kept (including `last_web_decision["error"]` exactly as
   today) → `return OutcomeList.unavailable("budget")`.**
   §5's new `if` block, positioned before the provider-error check.
   Proven by `TestTypedFailures::test_budget_refused_empty_result_is_unavailable_budget`
   and, critically, `::test_budget_and_error_both_set_checks_budget_first`
   — the PARENT-VERIFIED FACT that a budget refusal sets BOTH `error` and
   `blocked="budget"` together, and the budget check wins (this is the
   budget-first-ordering proof).
3. **Provider-error result (`not result.has_results`, not budget-blocked,
   `result.error` set): receipt write kept; `return
   OutcomeList.failed("provider_error")` — a constant label, never
   `str(result.error)`; no string-matching to split "client not
   available"/"invalid key" into other states (BC-76).**
   §5's second new `if` block. Proven by
   `TestTypedFailures::test_provider_error_result_is_failed_provider_error`
   (asserts the exact label `"provider_error"` regardless of the
   underlying error text, and that the marker embedded in that text never
   reaches `.reason`) and by the FIXTURE RULE edit to
   `test_gatherer_receipt_records_provider_failure` (the real F2-produced
   `WebSearchResult(error=...)` shape, end to end through `multi_search`).
   Recorded as a limitation (single label, not split by reason) per the
   brief's explicit instruction.
4. **A genuine empty search (no error, not blocked) and every deliberate
   non-search stay `None`.**
   The unchanged `return None` at the end of the `else:` branch, reached
   only when neither new `if` matches. Every earlier deliberate-non-search
   `return None` (disabled, intent veto, crisis veto, no manager, manager
   unavailable, trigger unavailable, not-triggered) is untouched.
   Proven by `TestUnaffectedControls::test_genuine_empty_result_stays_none`,
   `::test_not_triggered_decision_stays_none`,
   `::test_disabled_deliberate_non_search_stays_none`.
5. **Success path and every receipt write are byte-for-byte unchanged.**
   The `if result.has_results:` block (275-294) was not touched by either
   Edit call — verified by re-reading the file after editing, and by
   `TestUnaffectedControls::test_success_returns_the_same_result_object`
   (the SAME object identity is returned, citation tracking unaffected).
   Every `last_web_decision` write on every branch (268-273, 296-297,
   302) is unchanged text — verified by the two FIXTURE RULE tests'
   unchanged receipt assertions and by
   `test_budget_and_error_both_set_checks_budget_first`'s explicit
   `last_web_decision["error"]`/`["blocked"]` checks. The formatter's
   `web_search=` label (formatter.py 833-851) reads ONLY
   `web_search_decision`, so it is unaffected by construction — proven
   at the builder level by `TestThroughBuilder`'s
   `"web_search=ON(error)" in inventory` assertion, and by the sweep's
   `test_sep10_web_search_gap.py::test_feature_inventory_reports_search_decision_honestly`
   (unedited, still passing).
6. **Privacy: reasons are constant labels or exception class names only.**
   Every reason this batch's source produces is `type(e).__name__` or one
   of the two fixed labels `"provider_error"`/`"budget"` — proven by §14's
   marker-based tests.

**Builder-level proof (contract's explicit requirement, brief TESTS
section):** `TestThroughBuilder::test_raising_search_is_failed_and_label_reads_error_web_section_absent`
drives the DEPLOYED `UnifiedPromptBuilder.build_prompt` with the REAL
bound `WebSearchMixin._get_web_search_results` (via `types.MethodType`,
not a mock) wired onto the context_gatherer, and a manager whose
`multi_search` raises. Result: `"_build_time" in result` (the builder does
not silently fall back to its error path);
`result["_section_outcomes"]["web_search"] == {"status": "failed",
"reason": "RuntimeError"}` (F5's existing gather-loop wiring, unedited,
now genuinely exercises its except-typed-status-read path for this
section for the first time); `result["web_search_results"] == []` (the
web section is absent — formatter.py 1090-1094's `hasattr(web_search,
'has_results')` gate excludes a plain empty list exactly like today);
and the deployed `PromptFormatter._build_feature_inventory` on that same
result still renders `web_search=ON(error)` (the receipt-only label,
untouched).

## 17. Open items / limitations

- **The single "provider_error" label does not distinguish "client not
  available" from "invalid key" from "provider failed (<reason>)"** — per
  contract point 3's explicit instruction (BC-76: never string-match error
  text to split states). A future batch wanting finer granularity would
  need a typed field from `knowledge/web_search_manager.py` itself (read-
  only here), not a string match in the gatherer.
- **`multi_search`'s own BC-47 aggregation sibling** (a joined `errors`
  string dropped once any sub-query succeeds) and
  **`WebSearchManager._select_links_for_following`'s dm18 finding** are
  both recorded by F2's response, not rediscovered or fixed here — outside
  this batch's ownership (`knowledge/web_search_manager.py` read-only).
- **The trigger-level budget veto** (`not decision.should_search` with
  `_blocked == "budget"`, before `multi_search` is ever called) is a
  DELIBERATE non-search per contract point 4, unchanged — it already
  returns `None` today (or the cached result), and stays that way; only
  the POST-`multi_search` budget-refused-with-no-pages result is a new
  typed `unavailable("budget")`.
- No escalation: the manifest check, the pre-edit scan, the post-edit
  scan, and every test run came back exactly as the brief predicted. The
  one disclosed process note (§14: reconstructing an untracked file's
  pre-edit digest from this session's own earlier read, since `git
  add`/`git diff` cannot see untracked-file history) is not a rule breach
  and is independently corroborated by F2's own recorded digest for the
  same file.

## 18. 2026-09-14 — Final state (pre-handoff)

- `git status --short`: 110 lines (was 107 at F8a's final recorded state —
  +1 for `core/prompt/gatherer_web.py` transitioning from unmodified to
  `M`, +1 for `tests/unit/test_sep10_web_search_gap.py` transitioning from
  unmodified to `M`, +1 for the new `tests/unit/test_gatherer_outcomes_web.py`;
  `tests/unit/test_tavily_failure_outcomes.py` was already `??` from F2,
  unchanged classification; `docs/execution/generalization/batches/F8b.md`
  and the response file land inside the already-untracked
  `docs/execution/generalization/` directory, which `git status --short`
  reports as one line regardless of file count, matching every prior
  F-series packet's identical observation).
  `git diff --cached --name-only`: empty. HEAD:
  `328a8ecea1dae91de16f5974ea71727c4e2e1aa0` (unchanged throughout — no
  commit, no `git add` in any form). `git stash list`: empty, not
  touched. `git remote -v`: `origin /home/lukeh/Daemon_v1 (fetch)`,
  `origin DISABLED (push)` — not touched, unchecked again here only for
  the record.
- **Manifest delta** (against `manifest_paths_post_F8a.txt`): exactly 3
  new/changed paths beyond the two FIXTURE RULE files (already-tracked,
  content-changed) and the already-untracked
  `tests/unit/test_tavily_failure_outcomes.py` (content-changed, same
  path) — `core/prompt/gatherer_web.py` (modified,
  `270b6208…` → `b6dae37f…`), `tests/unit/test_gatherer_outcomes_web.py`
  (new, 292 lines), `docs/execution/generalization/batches/F8b.md` (new,
  this file). Content-changed-but-same-path:
  `tests/unit/test_sep10_web_search_gap.py`
  (`12f2f349…`, was already a manifest path, content now different) and
  `tests/unit/test_tavily_failure_outcomes.py` (`a12919f6…`, was already
  a manifest path since F2, content now different).
- **Size:** 20 (source) + 292 (new test file) + 7 + 7 (two FIXTURE RULE
  edits) = **326 changed lines**, over the 300 soft target, within the
  450 hard cap (§15).
- **Final data/ and logs/:** identical to the §1 baseline (7 entries,
  `user_profile.json` unchanged at 598 bytes / mtime
  `13:26:21.672354716`); `logs/` absent. No STOP at any point in this
  batch.
- **Class-guard status:** F8b answers #92, the LAST open CGR-007 anchor,
  in `CGR-20260913-007-5.md` (written next, LAST per ORDER). CGR-007 is
  now fully answered (-1 through -5). CGR-009 and CGR-010 are untouched by
  this batch.

## Parent review and integration (2026-09-14)

Accepted as delivered. No COMPLIANCE breach: no source edit before failing-first, and no non-pytest code beyond the one sanity check. The worker disclosed one process note: it reconstructed the pre-edit digest of the untracked `test_tavily_failure_outcomes.py` from its own earlier read.

### Manifest

- **Checksums:** `sha256sum -c manifest_post_F8a.txt` has exactly one mismatch, `tests/unit/test_tavily_failure_outcomes.py`.
  - That manifest records its pre-edit digest as `63b036a6…`, which independently confirms the worker's reconstructed "before" digest.
  - Its current digest is `a12919f6…`.
- **New paths (five):**
  - `core/prompt/gatherer_web.py` (`b6dae37f…`, its first modification);
  - `tests/unit/test_gatherer_outcomes_web.py` (`1e575483…`, 292 lines);
  - `tests/unit/test_sep10_web_search_gap.py` (`12f2f349…`, its first modification);
  - `class_guard_responses/CGR-20260913-007-5.md` (`3a666fac…`);
  - this file.
  - §18 above counts the paths differently and leaves out the response file. That is a reporting slip; the tree is correct.
- **Git state:** `git diff --cached --name-only` is empty, HEAD `328a8ec` is unchanged, and the reflog still shows only the clone entry.
- **No new stray files:** the only untracked file at the repository root is `scan_pre.stderr`.
- **`data/`:** the 7-entry baseline is unchanged (the listing in `$S/data_baseline_f7a_rerun.txt`; `user_profile.json` is 598 bytes, mtime 13:26:21), and `logs/` is absent. F9a uses this as its data/ baseline.
- **Parent edits after the worker returned:**
  - `briefs/F9a.md` and `briefs/F9b.md` were written; F9b is not yet run;
  - the design doc gained the F9 split amendment;
  - `briefs/PARENT_STATE.md` was updated.
- **Recorded tree:** `manifest_post_F8b.txt` records the tree after this section.

### Code review (the parent read the full diff)

- **Import:** `OutcomeList`.
- **#92 outer except:** the receipt write and the warning are unchanged, and the except returns `OutcomeList.failed(type(e).__name__)`.
- **`not result.has_results` branch:** after the unchanged receipt write and debug log:
  - `blocked == "budget"` gives `unavailable("budget")`, checked first;
  - `result.error` gives `failed("provider_error")`, a constant label;
  - otherwise it returns `None` as before.
- **Unchanged:** the success block and every deliberate non-search return.
- **Consumer check (parent, read-only):** every production reader of `web_search_results` treats an empty list exactly like `None`.
  - `formatter.py` 1090–1094 enters the block only when `has_results` is present.
  - `_format_web_search_results` 674–695 takes the `else: return ""` branch.
  - `controller._context_value_nonempty` 2333–2345 calls `bool([])`, which is False.
  - The controller's seed step at 920–921 uses `getattr(..., "has_results", False)`.
  - The `response_planner` digest loop skips `[]`.
  - `token_manager._meter_web_search_results` returns 0 when `has_results` is absent.
  - The only production caller is `builder.py:1513`.
- **No double disclosure:**
  - The formatter's catch-all "Could not check this turn:" line excludes `web_search` (`shown_names.add("web_search")`), so the prompt still shows only the receipt-driven `web_search=ON(error)` label.
  - The F6b `sections_not_checked` receipt (`orchestrator.py` ≈1710) now gains a label-only `web_search: failed:<class>`, `failed:provider_error` or `unavailable:budget` entry. That is intended.

### Tests (the parent read the whole new file and both FIXTURE RULE hunks)

- **Typed failures:**
  - a raising `multi_search` gives failed / RuntimeError, the receipt `error` is the class name, and the marker is absent;
  - a provider error gives failed / provider_error, the marker is absent from the reason, and the receipt keeps its raw text as before;
  - a budget refusal gives unavailable / budget;
  - with both `error` and `blocked` set, budget wins.
- **Controls:** a genuine empty search, a not-triggered decision and a disabled toggle stay `None`; success returns the same object.
- **Through the builder:** a real bound method gives `_section_outcomes["web_search"]` = failed / RuntimeError, `web_search_results == []`, and the inventory still reads `web_search=ON(error)`.
- **FIXTURE RULE:** each of the two edits replaces a bare `is None` with the typed assertion plus `== []` and keeps every existing receipt assertion. The paired controls (genuine empty stays `None`) are in the new file.
- **Failing-first:** 5 failed and 4 passed at digest `270b6208…`.

### Parent rerun (guarded, capped, `-p asyncio`)

- **Focused chunk 1** (MemAvailable 6,680 MB): the 9 files in §8 gave **186 passed** (14.61 s, 1,728,144 KB).
- **Focused chunk 2** (6,683 MB): `test_sep12_search_budget_reservation.py` gave **13 passed** (1.90 s, 761,292 KB).
- **Sweep** (6,616 MB): the 7 files in §10 gave **140 passed** (7.18 s, 1,664,676 KB).
- **Total:** **339 passed, 0 failed**, matching the worker. `tests/unit/test_graph_integration.py` stays excluded.
- **`data/`:** unchanged after every chunk, and `logs/` is absent.
- **ruff 0.14.9:** clean on all four files.
- **Scan:**
  - dm18: found 49, base 79, new 0, stale 30.
  - `WebSearchMixin._get_web_search_results 'return None'` is now in the STALE list (38 STALE rows in total: dm01 5, dm17 3, dm18 30).
  - dm01 (7/12/0/5), dm17 (36/39/0/3; files 629, one more test file) and dm31 (3/3/0/0) are unchanged, and every gate scanner shows 0 new. Scan stderr is empty.

### Size

326 changed lines (20 source + 292 test + 7 + 7 fixture), within the 450 cap.

### Limitations carried forward

- **One `provider_error` label:** it is not split by cause (BC-76); a finer split needs a typed field from `web_search_manager.py`.
- **Recorded by F2, not fixed:** `multi_search`'s joined-errors aggregation and the `_select_links_for_following` dm18 finding.
- **Trigger-level budget veto:** it stays a deliberate non-search (`None`).

### Class-guard status

- **CGR-007:** fully answered, #71–#92 across responses `-1` to `-5`. #85, #86 and #89 are evidence-only and stay live in dm18; the class-guard owner decides between a detector change and accepted debt.
- **Next:** CGR-009, starting with F9a (#139).
- **Open:** CGR-010 (F10–F13).
