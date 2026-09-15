=== COMMON RULES FOR GENERALIZATION WORKER BATCHES (parent session) ===
(Durable copy, re-created in the repo on 2026-09-14 after a machine crash wiped the /tmp scratchpad. The content is unchanged from the scratchpad version, including every rule added during the session.)

Working directory: /home/lukeh/daemon_exec/generalization. It is a plain clone and the ONLY checkout you may touch.
- Never read, edit, or run anything in /home/lukeh/Daemon_v1.
- A parallel class-guard session works in /home/lukeh/daemon_exec/class_guards. You may READ only your own request packet at /home/lukeh/daemon_exec/class_guards/docs/execution/class_guards/requests/<REQUEST-ID>.md, and nothing else in that checkout.

Read these sections of docs/PLAN_20260913_generalization_execution.md: "File ownership and parallel-work contract" (request/response contract), "Routine failure contingencies", "Evidence packet and stop conditions" and "Parent escalation packet". Also read docs/BUG_CLASSES.md for the named classes, and your request packet in full.

Never edit:
- the request packet;
- config/bug_class_baseline.json or any class-guard policy or disposition file;
- scripts/check_bug_classes.py, scripts/bug_class_guards/**, tests/bug_class_guards/** and the five repo-wide guard tests;
- hooks/**, .github/**, docs/BUG_CLASSES.md, docs/DEVELOPMENT_WORKFLOW.md, docs/TEST_LANES.md, docs/execution/class_guards/**;
- tests/conftest.py, conftest.py, config/** (unless the brief assigns specific config regions), data/.
Files from integrated earlier batches are not yours either (see the manifest).

Before editing:
- Report read-only git status, HEAD, diff stat, assigned-file digests and remote URLs, and verify the manifest. If anything differs, stop and report.
- Never git stash/reset/checkout/restore/clean. Never overwrite a tracked file with `git show` content. Do not commit or push.
- origin's push URL is already DISABLED; verify it with `git remote -v` only.

ENVIRONMENT (MANDATORY, verified by the parent 2026-09-13):
- The shell exports PYTHONPATH=/home/lukeh/Daemon_v1/scripts/bin, whose usercustomize.py imports `utils` from the MAIN checkout at interpreter startup.
- Prefix EVERY python/pytest/ruff/scanner command with `PYTHONPATH=/home/lukeh/daemon_exec/generalization/scripts/bin`. That is this clone's identical copy, so utils and the filesystem guard then load from this clone.
- Sanity-check once that `python -c "import utils; print(utils.__file__)"` prints a path inside this clone.

SHELL GUARD (added 2026-09-14):
- `mv` and `rm` in this shell route through the owner's scripts/safe_cmd.sh guard. It silently exits 1 on protected paths such as tests/ and config/.
- You should never need to move or delete anything.
- NEVER use the guard's unlock mechanisms (ALLOW_DESTRUCTIVE_OPS, the lockfile), and never call /usr/bin/rm or /usr/bin/mv. If a file must be removed, STOP and report.
- Never leave stray files anywhere in the repository (e.g. redirected stderr). Capture command output in the packet or under /tmp only.

Doctrine:
- Close a vocabulary/containment miss with the chokepoint (`utils/trigger_match.py`) or a structural check, never by appending a phrase (BC-76).
- Tests drive the deployed function and assert outcomes, with counterexamples and paired positive controls. Where the input is user text, text fixtures appear in both clean and wrapped/indented form (BC-64).
- Failing proof first: add the tests, run them on the unpatched source, record the failures, then patch.
- If a site is NOT an instance of the class, do not change it for the scanner's sake. Show it with deployed-function evidence in the response.

NON-PYTEST CODE (added 2026-09-14 after two breaches): run pytest only.
- This covers `python -c`, REPL sessions, heredoc snippets (`python - <<EOF`) and throwaway scripts, whether or not they import project modules. Each counts as a non-pytest script and needs parent approval BEFORE it runs.
- The standing exceptions are `python -m ruff`, `scripts/check_bug_classes.py scan` (read-only, as listed below) and the one `import utils` sanity check.
- If you need ground truth about deployed behaviour, write it as a pytest test and run it failing-first.
- Clarified 2026-09-14 after two more breaches (F10c, a `python3 -c` to slice a list; F12a, a no-op `python -c "print('noop')"`). `python` may appear in a Bash command ONLY in these exact forms:
  - `python -m pytest …` inside the capped systemd-run;
  - `python -m ruff check …`, and `python -m ruff --version` to report the version (clarified 2026-09-14 after F13a; line 42 already allowed it);
  - `python scripts/check_bug_classes.py scan --root .` (read-only; `--scanner`/`--json` are allowed, `--write-baseline` never is);
  - the one `python -c "import utils; print(utils.__file__)"` sanity check.
  - Anything else, including a harmless no-op, is a breach. Build chunk lists and sizes with grep, sed, wc or by hand.

EVIDENCE TIMESTAMPS (added 2026-09-14 after F12a): every dated packet entry takes its time from `date` output run in the same command as the step it records. Never estimate or back-fill a time. The parent checks packet times against file mtimes.

GIT INDEX AND PYTEST HYGIENE (added 2026-09-14 after F3a):
- No `git add` in any form, including `git add -N` / intent-to-add, and no other command that writes the git index.
- To size a new file, use `wc -l`; `git diff --stat` is for tracked files only.
- IMPORTER GREPS (added 2026-09-14 after F13b's sweep missed three handler consumers):
  - A sweep's importer set must cover every way a test reaches a module: `import pkg.mod`, `from pkg.mod import …`, `from pkg import mod`, and dotted strings such as `patch("pkg.mod.name")`.
  - Use at least `grep -rlE "pkg\.mod|from pkg import (.*[ ,])?mod\b"` over tests/unit.
  - Record the exact pattern in the packet.
- EVERY pytest invocation, including `--collect-only`, uses the /proc/comm guard, a MemAvailable check and the capped systemd-run command. No exceptions.
- File edits use the Edit or Write tools only. Never write a script to modify files.

NON-UNIT TESTS (added 2026-09-14 after a data/ write): tests outside tests/unit may build stores, rate limiters and caches with DEFAULT data/ paths.
- Known offender: tests/test_web_search_manager.py. It creates data/web_search_credits.json (the default WebSearchRateLimiter state file) and data/chroma_multi (WebSearchCache's lazy default MultiCollectionChromaStore). NEVER run it.
- NEVER run tests/test_prompt_internal_methods.py (it builds a real Chroma store and ModelManager).
- NEVER run tests/unit/test_graph_integration.py (added 2026-09-14 after F7a). It is inside tests/unit but writes data/user_profile.json.
  - Why: `TestQueryExpansion::test_expansion_no_graph_on_coordinator` passes `MagicMock(spec=[])` into the real `ContextGatherer`. Because that mock has no `user_profile`, `core/prompt/context_gatherer.py` (≈204–211) falls back to a real `UserProfile()`, which uses the default dev path.
  - The same hazard applies to ANY test that builds `ContextGatherer` with a memory coordinator lacking `user_profile` (`MagicMock(spec=[])`, `SimpleNamespace`, `object()`), or calls `UserProfile()` without patching the profile path. Check before running such a test; when unsure, exclude it.
- tests/unit/test_daily_notes_auto_update.py is RUNNABLE since F12b (integrated 2026-09-14; the parent reran it alone: 23 passed, data/ unchanged). It was excluded before that.
  - Why it was excluded: its success-path tests (`test_auto_update_proceeds_when_threshold_met`, `test_force_overrides_auto_update_check`) reach `DailyNotesGenerator._trigger_narrative_refresh`. That builds a real `MemoryConsolidator(model_manager)` with no profile, so it lazily constructs a real default-path `UserProfile()` (the data/user_profile.json hazard above) and reads the real `OBSIDIAN_VAULT_PATH`.
  - What makes it hermetic: its `generator` fixture sets `gen._trigger_narrative_refresh = AsyncMock(return_value=None)` right after the fake profile. Before running it, confirm that stub line is still present; if a batch removes or bypasses it, the file is excluded again.
  - The same hazard applies to ANY test that builds `MemoryConsolidator` or `DailyNotesGenerator` without an explicit fake profile, or without faking the Obsidian notes path and readers. Check before running such a test; when unsure, exclude it.
- Being "offline" is not enough. Run a non-unit test only after reading it and confirming that every store, limiter, cache, state file and telemetry path it constructs uses tmp_path, or is passed an explicit fake, and that no real Chroma/embedder/ModelManager is constructed.
- When in doubt, do not run it; record it as a limitation.
- If a data/ listing shows a new or changed entry after your runs, stop, report which test run produced it, and never delete it.

INTERRUPTION (added 2026-09-14 after F6a attempt 1 was cut off by an API limit mid-edit):
- Write the evidence packet docs/execution/generalization/batches/<BATCH-ID>.md INCREMENTALLY:
  - create it right after the manifest check;
  - append a dated entry after each ORDER step (pre-edit scan, failing-first, source edit, each test chunk, ruff, post-edit scan).
  - A cut-off batch then still leaves its evidence.
- Keep each source edit as small, complete Edit calls. Do not leave a half-applied contract between tool calls longer than necessary.
- Handling an interrupted batch is the PARENT's job:
  - the parent records the pre-state under batches/<BATCH-ID>_attemptN/;
  - it reverses owned edits in the working tree only;
  - it relaunches.
- A relaunched worker NEVER reads batches/<BATCH-ID>_attempt*/ artifacts (ABORT.md excepted); its failing-first proof must be independent.

MEMORY (host under memory pressure; a machine crash on 2026-09-14 came from an 8 GB non-unit pass running beside the live Daemon):
- Run tests in the FOREGROUND, in chunks of ≤9 files.
- Before each chunk, `awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo` must be ≥4000; otherwise wait and re-check.
- Never run pytest in the background.
- Never run a broad or full-suite pass.

Test commands (checkout root; one pytest process at a time across all sessions):
- Guard before each pytest: `for p in $(pgrep -f -- '-m pytest'); do case "$(cat /proc/$p/comm)" in python*) echo RUNNING;; esac; done`. If RUNNING, wait in the foreground and re-check every ~60s. Do not use plain `pgrep -af pytest`: it matches your own shell wrapper.
- Data listing: before the first pytest and AFTER EVERY CHUNK, record `ls -la --time-style=full-iso data` (top level only; never recurse) and `ls -ld --time-style=full-iso logs` (logs/ does not exist in this clone). Report any new or changed entry and never delete it.
- Focused command (`-p asyncio` is REQUIRED):
  systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=512M /usr/bin/time -v env DAEMON_TEST_MODE=1 CHROMA_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider -p asyncio <assigned test files>
  Record exit code, pass/skip/fail counts, elapsed time and "Maximum resident set size".
- `python -m ruff check <each changed .py file>`; report the ruff version.
- Bug-class scan, read-only, before editing and before handoff: `python scripts/check_bug_classes.py scan --root .`. Never pass `--write-baseline`. This checkout's scanner and baseline are the base (older contract) versions.
  - When your fix removes a baselined anchor, the base scan reports it STALE and exits non-zero. That is the designed handoff: the class-guard owner removes the row after integration. Record it; do not try to make it green.
  - Any NEW finding caused by your edit is a defect: fix it within your files.

Response file (class-guard batches only; immutable once written): docs/execution/generalization/class_guard_responses/<REQUEST-ID>.md. It must contain:
- request ID; owning batch ID;
- each changed file's SHA-256 before and after; changed files;
- the focused deployed-function test command with its exit code and counts;
- each anchor (review #) and whether it is fixed or left unfixed, with the reason and evidence;
- the before/after scan tables for the anchors' scanner;
- limitations.
Never edit the packet or a submitted response.

Evidence packet: docs/execution/generalization/batches/<BATCH-ID>.md, holding the plan's evidence items:
- manifest before and after;
- failing-before evidence;
- commands with counts, time and RSS;
- data/ and logs/ listings;
- scan results;
- the BC-58 sibling search;
- a privacy/no-network note;
- the milestone reached ("focused contract green" at most).

Size: ≤450 changed lines including tests is a HARD stop. Stop BEFORE exceeding it and return a split proposal. Do not compress code or comments to hit the number.
- Clarified 2026-09-14 after F10b: if a draft (source or test) measures over the hard cap at ANY point, stop and return a split proposal. Do not trim docstrings, comments, test docstrings or "redundant" assertions to squeeze under. Trimming counts as compressing, and it is a breach even if coverage survives.
- Clarified 2026-09-14 after F11b-1: estimate the size BEFORE editing each later part of the contract (e.g. after the tests and the first source region). If the whole contract will not fit, stop and return the split proposal before editing the rest. Do not implement everything and then revert parts to fit.
- Clarified 2026-09-14 after F13c-1:
  - Deleting a whole test, test class or helper to get under the cap counts as trimming, and it is a breach.
  - The pre-edit estimate must include the planned SOURCE lines, not just the test file.
  - If a post-edit measurement comes out over the cap, stop, record it, and return a split proposal. Do not change the tests to fit.

WEB TEST LANE (added 2026-09-14 for F13c-2a; applies ONLY to a batch whose brief names it):
- Permitted forms, run from the clone root, foreground, one at a time, each inside the same capped scope:
  - `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=512M env CI=1 npm --prefix web test -- <file> …`
  - `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=512M env CI=1 npm --prefix web test` (the full vitest suite)
  - `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=512M env CI=1 npm --prefix web run typecheck`
- NEVER run: `npm install`, `npm ci`, `npm update`, `npm run build`, `npm run dev`, `npm run preview`, `npx`, or any other npm or node command. `web/dist` must never change.
- Guard before each run:
  - no python process running pytest;
  - no `node` process whose cmdline contains `vitest` or `tsc` (check `/proc/*/comm` = `node` plus cmdline);
  - MemAvailable ≥4000 MB.
- Snapshot `git status --short web` before and after each run, and diff it against the baseline file the brief names (web/ already carries earlier batches' changes). The only allowed differences are the brief's owned files. vitest/vite caches under `web/node_modules/` are gitignored. Any other new or changed path under `web/` means STOP.
- List data/ and logs/ after each run, as for pytest chunks.

SCANNER SHAPE (added 2026-09-14 after F10b): never restructure code only so that a scanner stops matching it (e.g. moving a `return` out of an `except` behind a flag) without escalating first.
- If your deliberate, contract-required degrade trips a gate scanner as a NEW finding, stop and report it to the parent with the code and the scan row. The parent decides between an explicit, documented shape and a not-an-instance disposition for the class-guard owner.

Final report to the parent, concise, no diff paste:
- pre-edit and pre-handoff `git status --short`, HEAD, `git diff --stat`, untracked files;
- sha256 of every changed or new file;
- commands with results, failing-before, scan results;
- the per-anchor disposition;
- open decisions or escalation packet; limitations.
