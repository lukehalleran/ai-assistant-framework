# Class-guard completion — handoff report (2026-09-13)

This report covers execution of `PLAN_20260913_class_guard_completion.md` in
`/home/lukeh/daemon_exec/class_guards`, base `328a8ec`. The deliverable is the
uncommitted tree in that checkout. Evidence is in
[`WORKLOG_20260913.md`](WORKLOG_20260913.md),
[`BASELINE_REVIEW_20260913.md`](BASELINE_REVIEW_20260913.md) and
[`requests/`](requests/).

**Status: implementation, review and local verification are complete. The
plan is NOT complete.** Checklist items 9, 10 and 12 need a GitHub run on the
exact commit, and item 10 also needs branch protection. Only the owner can
supply either: the hard limits forbid commits, pushes, pull requests and
settings changes here.

## Checklist

| # | Item | State | Evidence |
|---:|---|---|---|
| 1 | Scanner IDs/modes/classes pinned; deletion, downgrade, unknown additions, partial selection, malformed registry fail | done locally | `test_policy.py` (8 registry mutations exit 2; 19 policy defects exit 2) |
| 2 | Output distinguishes coverage, unavailable evidence, unresolved, baseline mismatch, scanner error | done locally | `test_coverage.py`, `test_cli_contract.py`; JSON report schema 1 |
| 3 | Roots/legs counted independently; syntax preflight; shallow inventory; prefilter, DM-01 and DM-17 controls | done locally | `test_coverage.py`, `test_scanners.py` |
| 4 | Missing or wrongly typed baseline and malformed rows fail closed; no baseline write in enforcement | done locally | `test_cli_contract.py`, `test_enforcement.py` |
| 5 | 133 legacy occurrences (4 duplicate extras) reconcile one-for-one; accepted debt separate from fixed/false positive | done locally | ledger legacy history re-renders to `9bb56e6d…`; 152 `accepted_debt`, 0 resolved |
| 6 | Full-anchor migration reviewed, multiplicity preserving, hash bound | done; owner review pending | work log Phase 3 signoff |
| 7 | Five repo-wide guards strengthened with red/green controls | done locally | 225 passed; red evidence against the base guards in the work log |
| 8 | Immutable product requests; responses in Plan 2's directory; single writer for shared files | requests written; **all 10 open** | `requests/CGR-20260913-001…010.md`; no response exists |
| 9 | Hook and one-owner workflow run the full contract; final status verifies dependencies and receipts | implemented and tested locally; **not run on GitHub** | `test_enforcement.py`; YAML parses |
| 10 | GitHub run matches the reviewed SHA; protection requires PRs and the unique check with strict base, no bypass, documented solo review | **not done — owner action** | payload `branch_protection_master.json`; `master` currently unprotected |
| 11 | Documentation states scoped coverage, unavailable inputs and uncovered classes | done | `BUG_CLASSES.md`, `TEST_LANES.md` §4, `DEVELOPMENT_WORKFLOW.md` §3a.8/§7.1 |
| 12 | Changed paths exclusively owned; checks pass at the tested SHA; handoff names open requests and limitations | ownership and local checks done; tested SHA pending a commit and CI | 52 paths, 0 outside ownership; the privacy hook's gitleaks check passes on them |

## Local verification (uncommitted tree on `328a8ec`)

| Check | Result |
|---|---|
| `python scripts/check_bug_classes.py inspect --catalog docs/BUG_CLASSES.md` | 78 classes, 31 DM, 14 CM, 16 singletons, no diagnostics |
| `python scripts/check_bug_classes.py scan --root .` | exit 0. DM-01 31/31, DM-17 39/39, DM-18 79/79, DM-31 3/3, catalog 0 matched, no new or stale occurrences. DM-16 (report-only) has 91 candidates, 4 unresolved. DM-29 unavailable. Coverage 11/78 classes |
| `tests/bug_class_guards` (isolated, 6G scope) | 311 passed (the policy pins 311) |
| Five repo-wide guards (plan's bounded command, 6G scope) | 225 passed (85 at base) |
| `verify-receipts --scan … --harness-junit … --guards-junit … --expect-sha <HEAD>` | OK; a different SHA is rejected |
| `ruff` (plan's owned-path command; whole tree) | clean; clean |
| Privacy hook, gitleaks half: `gitleaks git --pre-commit --redact --staged --verbose` over a scratch index of the 52 files | exit 0, no leaks (3 false positives before the fix below) |

These results were re-run after the privacy-hook fix below. No CI run exists
for this tree, so no run URL can be given.

**Privacy hook finding (2026-09-13).** The owner's runner runs the main
checkout's `hooks/pre-commit-privacy` (gitleaks plus the local terms list).
It blocked the commit on three gitleaks `generic-api-key` false positives:
- `tests/bug_class_guards/test_policy.py:83`
- `tests/bug_class_guards/test_coverage.py:78`
- `tests/bug_class_guards/test_coverage.py:149`

On each line the DM-16 scanner ID, which contains the word "key", is followed
by another quoted name. No secret is involved.

Each of the three lines now ends with `# gitleaks:allow`. Neither file has
baseline or disposition rows, and comments do not change scanner anchors. The
gitleaks half was reproduced here on a scratch index: 3 findings before the
fix, none after. The terms-list half uses the owner's private list, so only
the runner runs it.

The commit line this report first gave would have bypassed the hook, because
the clone has no hooks installed. Step 1 below now uses the owner's runner.

## Owner steps

1. **Commit with the owner's runner, `~/daemon_checkpoints/plan1_commit.sh`.**
   Do not run a plain `git commit` in this clone. The clone has no hooks
   installed, so a plain commit would skip the privacy pre-commit hook.

   The runner commits only when all of these hold:
   - the clone is on `master` at `328a8ec` and push is disabled;
   - nothing else is working in the clone;
   - the changed files and the gitignored commit message match its pins byte
     for byte, with no file extra or missing;
   - the main checkout's `hooks/pre-commit-privacy` (gitleaks plus the local
     terms list) passes on exactly the tree being committed, staged in a
     scratch index.

   It commits on `class-guards/contract-v2-20260913` and pushes nothing.

   Start it only after the Claude session that produced this tree has exited.
   A session process whose working directory is inside the clone makes the
   runner stop with "something is working in the clone".

   ```bash
   ~/daemon_checkpoints/plan1_commit.sh --repin   # accept the tree as it is now (the files changed after the first pins)
   ~/daemon_checkpoints/plan1_commit.sh --check   # every check, including the privacy hook; commits nothing
   ~/daemon_checkpoints/plan1_commit.sh           # the checks again, then the commit (--log replays a run's output)
   ```

2. **Push the branch and open the pull request.** The clone's push URL is
   disabled, so this runs from the main checkout. Switch to the branch before
   pushing, so the installed pre-push hook tests the tree being pushed. That
   switch changes the working tree the Daemon runs from, so keep the Daemon
   down (§3a). The branch also carries the two local, unpushed commits
   `432c217` and `328a8ec`.

   ```bash
   cd /home/lukeh/Daemon_v1 && git fetch /home/lukeh/daemon_exec/class_guards class-guards/contract-v2-20260913:class-guards/contract-v2-20260913 && git switch class-guards/contract-v2-20260913
   git push -u origin class-guards/contract-v2-20260913
   gh pr create --base master --head class-guards/contract-v2-20260913 --title "Class guards: contract v2 and the bug-class-gate landing status" --body-file docs/execution/class_guards/COMPLETION_REPORT_20260913.md
   ```

3. **Verify the run** on the PR head before trusting it. The expected jobs are
   `bug-class-scan`, `test (3.11)`, `frontend` and `bug-class-gate`. The
   `bug-class-gate` check must come from app 15368 and conclude success, with
   `head_sha` equal to the PR head:

   ```bash
   gh pr checks class-guards/contract-v2-20260913
   gh api repos/lukehalleran/ai-assistant-framework/commits/$(git rev-parse HEAD)/check-runs --jq '.check_runs[] | {name, app_id: .app.id, conclusion, head_sha}'
   ```

   Recommended before relying on the gate: one throwaway PR with a deliberate
   failure in a dependency job, to confirm that `bug-class-gate` fails and is
   not skipped.

4. **Apply protection** only after step 3 shows the check name and app, then
   verify read-only (`DEVELOPMENT_WORKFLOW.md` §3a.8):

   ```bash
   gh api -X PUT repos/lukehalleran/ai-assistant-framework/branches/master/protection --input docs/execution/class_guards/branch_protection_master.json
   gh api repos/lukehalleran/ai-assistant-framework/branches/master/protection
   ```

   If GitHub refuses the payload, protection is not in force. Record the exact
   API error.

## Open requests (Plan 2 source owner)

| Request | Subject | Occurrences |
|---|---|---:|
| CGR-20260913-001 | Four `--apply` scripts with no daemon guard (DM-17) | 4 |
| CGR-20260913-002 | Agentic gate keyword tests outside `utils.trigger_match` (DM-01) | 7 |
| CGR-20260913-003 | Tone-detector keyword tests (DM-01) | 2 |
| CGR-20260913-004 | Safety canary enum-string membership (DM-01) | 1 |
| CGR-20260913-005 | Query-routing keyword lists (DM-01) | 4 |
| CGR-20260913-006 | Git commit tags by raw substring (DM-01, low) | 3 |
| CGR-20260913-007 | Prompt gatherers collapse failure into empty sections (DM-18) | 22 |
| CGR-20260913-008 | Knowledge retrieval managers return empty on errors (DM-18) | 11 |
| CGR-20260913-009 | Memory store reads return empty on failure (DM-18) | 20 |
| CGR-20260913-010 | Memory store writes return None on failure (DM-18) | 6 |

Each request stays open, with its rows kept as accepted debt, until a response
is integrated and a real scan shows the anchors gone.

## Limitations

- **Bounded review.** Candidates were read in their enclosing statement and
  function; consumers were not traced exhaustively. `product_risk` means a
  plausible class instance, not a reproduced defect. The gate is green only
  because all 152 occurrences are recorded as accepted debt.
- **Re-review coupling.** 61 files carry accepted debt, 44 of them outside
  `tests/` and `scripts/`, including Plan 2's shared seams `core/orchestrator.py`
  and `memory/user_profile.py`. Any edit to one of these files fails the scan
  with `disposition_source_changed` until the class-guard owner re-reviews it,
  so each Plan 2 fix will turn the scan red until its response is processed.
- **Report-only evidence.** DM-16 has 91 candidates, 4 of them unresolved;
  their triage was bounded, not a verdict. DM-29's input is untracked, so CI
  reports it unavailable and BC-76 is never scanned there.
- **Coverage gaps.**
  - 67 of the 78 catalog classes have no scanner.
  - Python in `agent_branch`, `eval`, `hooks`, `integrations`, root
    `conftest.py` and `debug_graph_context.py` is classified but unscanned.
  - The shallow inventory does not see Python nested inside a new top-level
    directory that has no direct `*.py` file.
- **Detector scope boundaries.**
  - DM-17 accepts the repository's ImportError-tolerant (fail-open) guard
    wrappers. It does not see dynamically built flag names, or `data` paths
    assembled with `os.path.join`/`Path`.
  - The git-state guard does not follow argument lists held in variables, nor
    git reads made by application code (observed:
    `tests/test_git_memory.py::TestHotFiles` reads project history through
    `GitMemoryExtractor`'s default path).
  - The ordered-slice guard covers only name-matched slices.
  - The budget guard reads only `core/prompt/formatter.py`.
- **Cost.** The full scan takes 7.8 s, up from 3.4 s. The harness count in the
  policy must change together with the harness tests.
- **No independent review.** A solo maintainer with 0 required approvals means
  none is enforced, and this change itself was reviewed only by the executing
  session. The hook's 6 GiB cap is unchanged; it needs a separate measured
  repair.
- **Unverified on GitHub.** The `pytest==9.0.2` install in `bug-class-scan`,
  the artifact upload/download steps, and the check name and app used in the
  payload have not been exercised on GitHub.
