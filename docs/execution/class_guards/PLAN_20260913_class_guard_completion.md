# Class-guard completion plan — 2026-09-13

This is an execution brief for Opus. It closes the concrete holes in the
repo-static scanner ratchet and landing checks without claiming that a small
scanner set covers every bug class or reviving the superseded signed-receipt
framework. The work is complete only when the guard contract, the 133 existing
candidate occurrences, their disposition record, the hook/workflow wiring, and
server-side landing policy all reconcile to tested evidence.

## Target outcome

The full scan must prove that every policy-required scanner ran in its pinned
mode over every declared input leg, including syntax and inventory checks;
that its baseline is present, valid, and reconciled in both directions; and
that a green result cannot be produced by deleting or downgrading a scanner,
omitting an input, adding a prefilter blind spot, or enabling baseline writes
in enforcement. Report-only scanners stay visibly report-only. Candidate
findings remain candidates until people review them.

The current reference set has seven scanners: `dm01_raw_substring` (gate),
`dm16_config_key_reachability` (report), `dm17_apply_without_guard` (gate),
`dm18_except_returns_empty` (gate), `dm29_phrase_append_signature` (report),
`dm31_live_state_default` (gate), and `catalog` (gate). The catalog gate
checks catalog structure; it does not detect product behavior across the
catalog. The other six detectors cover only their listed patterns and inputs.
The resulting lane is **not** an all-class behavioral guarantee for the 78
catalog classes, and a protected green check does not make that claim.

## Owner decisions (2026-09-13)

1. **Landing enforcement is adopted as written in Phase 5:** pull requests
   required into `master`, the unique bug-class gate status required from the
   expected GitHub App, strict up-to-date base, no direct-push or admin
   bypass, and 0 required approvals for a solo maintainer. Do not re-ask.
   Prepare the settings payload and the exact `gh api` command; the owner
   applies them.
2. The Phase 5–6 verification run comes from a pull request into `master`
   that the owner opens from a branch. The workflow runs on pushes to
   `master` and `refactor/prompt-modular` and on pull requests into `master`,
   so a push to any other branch runs nothing. Give the owner the branch name
   and commands; never push or open the pull request yourself.
3. Once protection is active, every landing, Plan 2's included, goes through
   a pull request. Record that flow in `docs/DEVELOPMENT_WORKFLOW.md` §3a
   together with the settings payload.
4. The cross-plan boundary is the one-way request/response contract under
   “Exclusive ownership and execution boundary”, mirrored in Plan 2's “File
   ownership and parallel-work contract”. Neither session changes it.

## Fresh-session start

A new Claude session with no memory of the planning conversation executes
this plan; everything it needs is in this file and the documents it names.

1. **Base.** Before launch the owner commits the planning documents on
   `master`: this plan, Plan 2
   (`docs/PLAN_20260913_generalization_execution.md`),
   `docs/GENERALIZATION_CI_REVIEW_20260913.md`, `docs/PROBE_HARNESS_DESIGN.md`,
   and the review's edits to `docs/DEVELOPMENT_WORKFLOW.md`,
   `docs/TEST_LANES.md`, `docs/WORKPLAN_2_hardening_executable_beta.md`,
   `docs/generalization/README.md`, and
   `docs/generalization/08-windows-executable.md`. Both sessions start from
   that commit. If either plan or the review is missing from HEAD, or the
   tree is dirty at start, stop and report.
2. **Checkout.** Work only in `/home/lukeh/daemon_exec/class_guards`, a plain
   clone of `/home/lukeh/Daemon_v1` at that commit with its push URL
   disabled; the owner creates it before launch. Never edit, reset, or run
   tests in `/home/lukeh/Daemon_v1`. Plan 2 runs at the same time in
   `/home/lukeh/daemon_exec/generalization`.
3. **Interpreter.** `python --version` must print 3.11.8. The version comes
   from `/home/lukeh/.python-version`, so it resolves only under
   `/home/lukeh`; elsewhere `python` is system 3.13. There is no project
   venv: use the shared pyenv interpreter read-only and never `pip install`.
   A missing dependency is a stop-and-report.
4. **What the clone lacks, by design.** Gitignored `data/`, `.env`, and
   `config/config.local.yaml` are absent: no API keys, owner vocabulary, or
   live stores. Relative `./data/...` paths resolve inside the clone; never
   point `CHROMA_PATH` or another path override at `/home/lukeh/Daemon_v1`.
   Git does not clone hooks, so `hooks/pre-push` is not installed; run its
   checks by command, never by pushing.
5. **Test resources.** Wrap every pytest in
   `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=512M`.
   Run one pytest process at a time across both sessions: if
   `pgrep -af pytest` shows the other session's run, wait. The scan lane and
   the five guard files are the only local test runs; CI owns the full
   backend suite. Never search or traverse `data/` or `.agent_snapshots/`.
6. **Plan 2 exchange.** Request packets are written here, in
   `docs/execution/class_guards/requests/`. Plan 2's files are read-only here
   and are read from its checkout under
   `/home/lukeh/daemon_exec/generalization/docs/execution/generalization/`:
   responses at `class_guard_responses/<request-id>.md`, integration signals
   at `class_guard_signals/<batch-id>.md`, and workflow proposals elsewhere in
   that directory. A response counts only after the owner integrates its fix
   into a tree this session can scan; until then the request stays open and
   its gate stays red.
7. **Hard limits and deliverable.** No commit, amend, push, pull request,
   remote-setting change, deploy, daemon restart, or call to Daemon's LLM
   providers or paid tool APIs. The deliverable is the uncommitted tree in
   this checkout plus evidence under `docs/execution/class_guards/`; the
   owner commits there and fetches it into the main checkout.

## Exclusive ownership and execution boundary

Opus owns exactly these paths for this plan:

- `scripts/check_bug_classes.py` and `scripts/bug_class_guards/**`
- `config/bug_class_baseline.json`, new `config/bug_class_policy.json`, and
  new `config/bug_class_dispositions.json`
- `tests/bug_class_guards/**`
- `tests/unit/test_no_git_state_in_tests.py`,
  `tests/unit/test_ordered_slice_guard.py`,
  `tests/unit/test_budget_meters_rendered_sections.py`,
  `tests/unit/test_tool_wiring_parity.py`, and
  `tests/unit/test_model_capability_wiring.py`
- `hooks/pre-push`, `.github/**`, `docs/BUG_CLASSES.md`,
  `docs/DEVELOPMENT_WORKFLOW.md`, `docs/TEST_LANES.md`, and
  `docs/execution/class_guards/**`

Plan 2 (`docs/PLAN_20260913_generalization_execution.md`) owns all
application source and all other tests, configuration, scripts, web files,
package/spec files, and generalization documents. A file belongs to only one
plan, even when proposed edits touch different hunks. Plan 2 must not edit a
Plan 1 path to make its source pass a guard; Opus must not edit Plan 2 source
or tests. A newly exposed product candidate becomes an immutable request
packet under `docs/execution/class_guards/requests/`, with a request ID,
scanner fingerprint, multiplicity, source SHA-256, and reproduction evidence.
The Plan 2 code owner resolves it in one immutable response file at
`docs/execution/generalization/class_guard_responses/<request-id>.md` in
Plan 2's checkout. Neither plan edits the other's packet. Do not maintain a
joint status JSON or shared mutable tracker.

The baseline, catalog, and `.github/**` each have one writer: Opus. Plan 2
may not edit them. When Plan 2's tree turns red under the new policy, Plan 2
records a pending integration signal at
`docs/execution/generalization/class_guard_signals/<batch-id>.md`, and Opus
decides whether it becomes a request. A signal is not permission for Plan 2
to admit or remove baseline
rows. Opus incorporates only the reviewed source-owner response, then alone
changes the baseline/catalog/workflow. No response means leave the issue
pending and the relevant gate red.

Work only in the checkout named under “Fresh-session start”; do not alter
the main checkout. During planning, do not `git add`, reset, commit, push,
deploy, run paid models, or change remote settings. During execution,
mutations stay in the authorized checkout and within the owners' paths
above. Never install into the shared interpreter, and run resource-intensive
lanes sequentially, with one pytest process across both sessions. The owner retains commit, push,
and remote-policy final review under `docs/DEVELOPMENT_WORKFLOW.md` §3/§3a.

## Contracts and file interfaces

### Scanner and coverage policy

Add `config/bug_class_policy.json`, schema version 1, as the human-reviewed
source of the enforcement contract. It must declare:

1. The exact required scanner IDs, expected mode, class IDs, and contract
   version for all seven current registry entries. Compare the whole registry
   to policy: a missing, extra, renamed, or mode/class-changed entry is an
   error until policy and its controls are deliberately updated together.
2. A per-scanner list of input legs and the root paths/globs each leg owns.
   Declare the common Python source roots, the separate DM-17 script and test
   legs, DM-18's four roots, DM-16's required YAML/app-config inputs and its
   consumer roots, the catalog document, and DM-29's changelog input.
3. Which inputs are required for enforcement and which are optional report
   evidence. DM-29's absent local-only changelog must produce `unavailable`,
   never “0 findings / clean”; it remains report-only until a tracked,
   policy-approved input exists. A report-only unavailable leg is visible in
   output and cannot be cited as evidence that BC-76 was scanned.
4. An explicit top-level Python source-root inventory, required roots, known
   non-Python roots, and excluded private/generated roots. New Python-bearing
   roots fail inventory validation until reviewed and classified. The
   inventory check may inspect top-level names and direct `*.py` presence,
   but must not traverse stores, snapshots, backups, `.git`, or other private
   data to discover candidates.
5. The limited claim: this is a scoped structural-detector lane, not complete
   coverage of every BC ID. Generate the list/count of catalog IDs absent
   from the current scanner `class_ids` as an explicit coverage gap in the
   scan report. Do not make unimplemented classes appear covered by assigning
   them to the catalog consistency scanner.

The deployed interface remains `python scripts/check_bug_classes.py scan
--root .`; add an explicit scope in its result (`full` or `partial`) and
coverage records per leg: `id`, `required`, `available`, `files_processed`,
and `unresolved`. JSON output must carry this evidence, expected and observed
scanner IDs/modes, class IDs, comparison counts, and exit. The table output
must distinguish `unavailable` from a valid zero. `--scanner` is an explicit
partial scan and can never be used in hook/workflow enforcement. Partial
scans still validate policy/registry parity, selected inputs, and baseline
syntax, but must label unselected legs and make no full-scan claim.

Do not duplicate one detector's implementation in its test. Test the deployed
CLI and each scanner's public `scan(root)` over synthetic trees. Assert
observable exit codes, machine output, findings, and coverage-leg receipts.

### Baseline and disposition records

Ordinary scanning with a missing baseline exits 2, even when no finding is
present. The baseline loader rejects malformed JSON, unsupported or wrongly
typed schema, missing/extra fields, non-string or empty fingerprint fields,
unknown scanner IDs, report-only scanner rows, invalid multiplicities, and
orphan rows. Boolean schema values must not pass as integer schema 1.
Full scans compare all and only required gate scanners; a baseline row for a
removed/unknown gate is an error, not something filtering can hide. Partial
scans compare only selected gate scanners, while preserving their explicit
partial label. `--write-baseline` remains an explicit developer bootstrap
operation; it is never used by tests as an enforcement command, the hook, or
CI, and its output still requires dispositions before it can pass the full
gate. No automatic refresh or auto-admission is allowed.

The present file contains 133 legacy fingerprint occurrences across four
gates: DM-01 12, DM-17 39, DM-18 79, and DM-31 3. They are 129 unique
four-field keys with four additional duplicate occurrences. Preserve and
reconcile every occurrence; do not set-convert them, collapse duplicates, or
bulk-rebaseline them away.

`config/bug_class_dispositions.json`, schema version 1, is a per-occurrence
review ledger. Each active record contains the legacy fingerprint, stable
occurrence ordinal among identical legacy keys, new anchor digest, source-file
SHA-256 captured at review, status, rationale, reviewer, review date, and
evidence/request ID. Every active baseline occurrence has exactly one
`accepted_debt` disposition and the file hash still matches. Each resolved
occurrence is retained as history with an explicit `confirmed_fixed` or
`false_positive` status, rationale, and evidence; it is absent from the
active baseline. Thus accepted debt is visibly distinct from a confirmed
fix, and a fixed row cannot remain as green-laundered debt. Missing,
duplicate, orphan, unreviewed, or changed-source dispositions fail. Any
source-file hash change requires re-review of retained active debt.

Replace the lossy 240-character source-line-only anchor with a reviewed,
versioned full candidate anchor: scanner ID, relative path, enclosing
qualified name, candidate kind, and SHA-256 of a deterministic normalized
AST/source expression. Retain a bounded human excerpt separately for logs.
Do not use line numbers as identity. Migrate the existing schema in a
separate, reviewed step: emit old and proposed keys to a temporary artifact,
prove a one-to-one occurrence mapping for all 133 rows including the four
extra duplicate occurrences, and require dispositions to reconcile by count
before replacing the committed baseline. Never silently rewrite historical
fingerprints. If a candidate cannot produce a reliable full anchor, fail
closed and keep its legacy key pending explicit review.

### Scanner input semantics

Validate Python syntax across every declared Python input before applying
scanner-specific lexical prefilters. A malformed file that does not happen
to match a prefilter is still a scan error (exit 2). Each required scanner
root and each declared leg must be checked independently; a nonzero combined
DM-17 total cannot disguise an empty scripts leg or an empty tests leg.
Nested new directories under a declared root are included by the existing
sorted walk. New source roots outside the manifest are detected by the
shallow root inventory and require explicit classification. Keep the
existing exclusions deliberate; never expand an inventory walk into data,
snapshots, backups, or private stores.

Close known detector escape hatches:

- **DM-01:** remove the whole-module exemption triggered by importing
  `utils.trigger_match`. A matcher import alone is not proof that every raw
  test in the file uses it. A raw membership expression beside a valid
  matcher call must still be found. Add positive controls for real matcher
  invocation and negative controls for raw expressions under the same
  module/import. Audit/remove or explicitly test the prefilter; if its
  candidate vocabulary cannot be proved complete for the declared pattern,
  parse all files and use AST matching without that prefilter.
- **DM-17:** `daemon_guard`/`daemon_running` text is not proof of execution or
  order. Treat comments, strings, unused imports, and unreachable calls as
  findings/unresolved evidence. Recognize only reviewed guard call/context
  shapes that execute before the protected write; keep unknown dynamic write
  or guard patterns as unresolved candidates for human review. Track
  `scripts/*.py` and `tests/**/*.py` independently, including the deliberate
  `tests/conftest.py` exception and why it is exempt.
- **DM-16:** require its YAML and app-config inputs and every consumer root;
  report missing schema as a declared optional input only if that remains the
  implementation contract. Unsupported YAML constructs or unresolved dynamic
  key/reader forms must be surfaced as unresolved report candidates, not
  interpreted as a successful reachability proof.
- **DM-29:** absent changelog means unavailable evidence, as above. Preserve
  report-only mode and do not convert 0 processed files/findings to a clean
  result.
- **DM-18/DM-31:** keep their existing declared scopes, but validate the full
  Python inventory before `except` prefiltering or AST candidate selection.
  Add tests around representative indirect/unresolved shapes so documented
  scope boundaries are explicit.

## Phased execution

### Phase 0 — isolate and inventory

1. Confirm this checkout is clean and at the same base SHA as
   `/home/lukeh/daemon_exec/generalization` (read its HEAD only); record
   `git status --short` and `git rev-parse HEAD` in the work log. No
   main-checkout mutation.
2. Verify the exact ownership lists above against changed files before every
   batch. A cross-owner change pauses that batch and becomes a request packet.
3. Capture scanner registry modes/class IDs, all current input roots, current
   baseline bytes/counts, installed hook target, workflow check/job names,
   `gh` read-only branch state, and the existing CI run/SHA. Do not change
   remote protection now.

### Phase 1 — red contract tests first

Add focused tests in `tests/bug_class_guards/test_policy.py`,
`test_coverage.py`, and `test_cli_contract.py` (or similarly named files in
that directory). Extend the existing `test_scanners.py`, `test_baseline.py`,
and `test_catalog.py` only where their current deployed-function coverage is
the natural home. The first run must demonstrate each new control red against
the old behavior; record that as execution evidence, not a permanent test
asserting a failure.

Pin the exact seven IDs, modes, classes, and uniqueness. Synthetic registry
mutation controls must show that deletion, renaming, downgrade from gate to
report, upgrade/report drift, class-ID drift, duplicate IDs, and an
unreviewed extra scanner fail before a scan can turn green. Test unknown
scanner selection and partial scan output separately.

The baseline/CLI red matrix must include: missing baseline on a finding-free
tree; wrong schema type/value; malformed JSON; wrong container and field
types; empty/extra/unknown scanner fields; report-only/orphan rows; invalid
fingerprint anchors; mismatched occurrence counts; new and stale findings;
duplicate count increase/decrease; explicit partial-scan scoping; and
`--write-baseline` never appearing in enforcement invocations. Preserve the
existing harmless line-shift and duplicate-multiplicity semantics.

The input-coverage red matrix must remove the entire common source root, each
individual required root, each DM-17 leg in turn, each DM-16 required input,
and the catalog. Also add a new Python-bearing top-level root and prove the
inventory rejects it without descending into private/generated roots. Put a
syntax error in a Python file which does not match any prefilter and require
exit 2. Add prefilter controls for each declared scanner shape and a fixture
where a matching expression is present outside the old DM-01 prefilter form.

DM-01 controls must show that an import or actual matcher use does not exempt
an unrelated raw membership expression. DM-17 controls must show that a
comment, string, import-only, late, or unreachable `daemon_guard` mention
does not satisfy guard evidence; a recognized executed-before-write guard
does. Keep the `tests/conftest.py` exception as a narrowly named fixture.
Report-only tests assert `unavailable` versus `available with zero findings`.

### Phase 2 — policy, scanner, and coverage implementation

Implement the policy reader and full/partial scan result contracts in
`scripts/bug_class_guards/` and `scripts/check_bug_classes.py`. Validate
policy before scanner execution, then registry parity, shallow root
inventory, Python syntax, each required input leg, scanner results, strict
baseline/disposition integrity, and finally comparisons. Exit 2 means the
gate could not evaluate its contract; exit 1 means new/stale/unreviewed
findings or a broken coverage contract; exit 0 means all required gate
contracts passed and report evidence is clearly labeled. Keep report-only
findings out of the baseline and exit comparison.

The JSON report is a stable interface for tests/CI, not a signed receipt
framework: it contains policy version, scan scope, expected/observed IDs and
modes, class-ID mapping and uncovered catalog IDs, input-leg availability
and counts, unresolved count, baseline/disposition occurrence counts,
new/stale keys, and final exit. Validate it in tests from the actual CLI.
Malformed or missing policy/dispositions fail closed. Do not claim the
policy/tests make themselves tamper-proof; a change to guard and its tests
still needs ordinary reviewed code ownership.

### Phase 3 — reconcile the 133 baseline occurrences

1. Export a sorted review file under `docs/execution/class_guards/` with all
   133 occurrences. Show the exact scanner/path/symbol/legacy-line anchor,
   occurrence ordinal, current full source expression, source SHA-256, and
   suggested disposition. Duplicate keys appear once per occurrence.
2. Review each candidate against its actual source and declared detector
   semantics. Classify it as `accepted_debt`, `confirmed_fixed`, or
   `false_positive`; use a separate immutable Plan 2 request for every
   product-source fix. Never resolve uncertainty as a false positive.
3. Keep product candidates in the baseline as accepted debt until Plan 2 has
   made and tested the source fix. For a detector false positive, Opus first
   improves the detector and red/green controls; the finding must disappear
   in the real scan before its active baseline occurrence is removed. For
   confirmed fixed, require the source-owner response/test evidence and a
   real scan showing it is gone. A stale row must still fail until Opus
   removes it in the same reviewed batch.
4. Bind each active disposition to the current candidate anchor and source
   hash; re-review any source-file change before retaining acceptance. Bind
   resolved history to the reviewed pre-fix source hash and fix/test/request
   evidence. Exact-count invariants: active baseline occurrences equal
   `accepted_debt` occurrences; `confirmed_fixed`/`false_positive` occurrences
   have no active row; active + resolved history reconciles to all 133 legacy
   occurrences unless a separately documented, reviewable new candidate is
   added.
5. Do schema/anchor migration only after review categories and occurrence
   mappings reconcile. Generate proposed new baseline into a temporary file,
   compare each multiplicity against the legacy set, and require exact
   reviewer signoff in the plan execution report. Do not use the write flag
   as a shortcut.

Review DM-16's 87 report-only candidates as a separate report sample/evidence
question, not as baseline rows or as confirmed defects. Record its input
availability and a bounded human triage outcome. The DM-29 zero-file CI
result is unavailable evidence and remains clearly labeled.

### Phase 4 — resolve cross-plan candidates and the five repo-wide guards

For each candidate requiring Plan 2 source work, Opus writes an immutable
request packet in its execution directory with request ID, exact candidate
anchor and multiplicity, current source file SHA-256, observed behavior or
semantic risk, expected deployed-function regression test, and exact owned
source path. Plan 2 responds in its own execution directory. Opus verifies
that response against the integrated source, then updates only its baseline
and disposition paths. A request remains open if it lacks source-owner
evidence.

Strengthen the five named repo-wide guards at their deployed source/audit
boundaries; do not mirror their production implementation or turn them into
full parsers without a demonstrated failure mode:

- `test_no_git_state_in_tests.py`: self-tests must include split/multiline
  subprocess forms and project-root aliases; tmp-path Git repositories
  remain allowed. Check every test file recursively and report path only.
- `test_ordered_slice_guard.py`: preserve content/multiplicity anchoring;
  exercise regex/AST boundary cases and ensure a new suspicious slice is not
  hidden by line shifts or a same-line duplicate. Existing explicit
  allowlist stays short and reasoned.
- `test_budget_meters_rendered_sections.py`: derive rendered keys from all
  formatter paths, including supported access spellings and helper indirection
  that the current `context.get("...")` regex misses. Test that new rendered
  keys fail until metered or entered as a reasoned exception; retain real
  TokenManager behavior tests.
- `test_tool_wiring_parity.py`: keep declared flags/action registry as source
  of truth, and test the dispatch/executor handoffs from real registries.
  Add a control for a registered but unconsumed new tool/action and an
  unresolvable executor reference; avoid restating router code in a test.
- `test_model_capability_wiring.py`: preserve registry parity and invoke the
  public classifiers/protocol chooser. Ensure new aliases, orphan rows, and
  capability combinations are checked; avoid copying classifier decision
  lists into test-owned logic.

All new guards must retain current valid behavior and produce a specific
failure message. Do not edit any other `tests/unit/**` file under this plan.

### Phase 5 — hook and single workflow integration

Opus is the sole writer for `hooks/pre-push` and `.github/**`. Use the exact
full-scan command without `--scanner` or `--write-baseline`, followed by the
stdlib-only guard harness. A small reusable command validator may inspect
the literal hook/workflow commands, but do not build an enormous shell
parser. Tests must fail if those commands omit the policy scan, pass a
partial-scan selector, enable a baseline write, use `continue-on-error`, or
put the lane behind a path filter. Keep the five repo-wide guards selected
regardless of changed files.

Provide one stable, uniquely named required status for the final bug-class
gate. It must explicitly require all dependency jobs to conclude success and
must check the expected scanner IDs/modes, all required leg receipts,
disposition/baseline occurrence counts, and expected test-case count. A
skipped or neutral dependency cannot count as success. No `paths` filter,
`continue-on-error`, `always()` success masking, or silent empty collection.
Likewise, generalization/probe/Windows jobs or checks arrive as Plan 2's
workflow proposal artifacts; Opus alone merges all workflow changes once.
Do not claim those jobs or assembled-turn probes ran until GitHub ran them on
the exact tested commit.

Before remote policy is considered ready, verify the actual run URL and
commit SHA, each required check conclusion, the app/source expected to
publish the check, and that every required job is present. Prepare a concrete
branch-protection settings payload for owner review: require PRs into
`master`, require the verified unique status checks from the expected GitHub
App, require strict latest-base checks, disallow direct-push bypasses and
admin bypass, and set required approvals to 0 in a solo-maintainer setup to
avoid self-review deadlock. Label that independent review is absent while
there is only one maintainer; add an independent reviewer/code-owner rule
when an eligible second reviewer exists. Required checks alone do not
prevent a direct push of a previously checked commit, so PR-required is
explicit. The owner applies settings only after reviewing the payload; if
permissions or repository policy block application, stop and give the owner
the precise remaining action. Verify afterward with read-only branch API
output; do not report protection based only on a prepared payload.

GitHub documents that successful, skipped, or neutral checks can satisfy a
required check, and that required checks can be scoped to an expected app;
that is why the final gate must verify dependencies and receipts explicitly.
Admin bypass and solo-review handling must be verified against the selected
repository policy. See [protected branches and required status checks](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
and [code-owner review behavior](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-owners/about-code-owners).

### Phase 6 — final evidence and handoff

Run the cheap scanner lane and five guard tests locally; use CI for the full
workflow. Check exact changed paths against ownership, baseline/disposition
counts against the triage artifact, clean tested SHA against the GitHub run,
and actual branch settings against the approved payload. Update only the
owned `BUG_CLASSES.md`, workflow/lane docs, and `docs/execution/class_guards`
with final scoped claims. Include what remains unavailable: report-only
evidence, all-class behavioral gaps, any unclosed Plan 2 request, and absent
independent review. Owner alone creates the commits and pushes through normal
review; no deployment is part of this plan.

## Required red controls and accepted green controls

| Control | Required result |
|---|---|
| Remove/rename a required scanner; gate→report; change its classes; add an unreviewed scanner | Full scan exits nonzero before baseline comparison |
| Missing baseline on zero-finding tree; malformed/wrongly typed baseline or policy/dispositions | Exit 2; no “clean” success |
| Add/remove one duplicate fingerprint occurrence | Exit 1 with exact NEW/STALE multiplicity |
| Missing declared source root, DM-17 leg, DM-16 input, or catalog | Required input fails closed; optional report input says unavailable |
| New top-level Python root absent from policy; private data/snapshot root contains Python | First must fail shallow inventory; latter is never traversed |
| Syntax error in prefilter nonmatching Python | Exit 2 from syntax preflight |
| Raw DM-01 expression with matcher import in same module | Finding still reported |
| DM-17 guard token only in comment/string/import/late/unreachable call | Candidate/unresolved finding; no false guard proof |
| DM-29 absent local changelog | Report is unavailable, never green “zero files” evidence |
| Missing `--root .` full scan or adding `--scanner`/`--write-baseline`/`continue-on-error`/path filter in enforcement | Hook/workflow contract test fails |
| GitHub dependency skipped/neutral, expected receipt absent, or result is for another SHA | Final required status fails or evidence is rejected |
| Exact baseline, accepted dispositions, line-only movement outside the anchor, all required legs present | Green with accurate scoped report |

## Resource-bounded commands

Run these from the repository root and serially. The scan/harness are
stdlib-only; avoid enabling pytest plugin autoload for the isolated lane.

```bash
python scripts/check_bug_classes.py inspect --catalog docs/BUG_CLASSES.md
python scripts/check_bug_classes.py scan --root .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider \
  --confcutdir=tests/bug_class_guards tests/bug_class_guards
python -m ruff check scripts/check_bug_classes.py scripts/bug_class_guards \
  tests/bug_class_guards tests/unit/test_no_git_state_in_tests.py \
  tests/unit/test_ordered_slice_guard.py \
  tests/unit/test_budget_meters_rendered_sections.py \
  tests/unit/test_tool_wiring_parity.py \
  tests/unit/test_model_capability_wiring.py
env DAEMON_TEST_MODE=1 CHROMA_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout 120s python -m pytest -q \
  -p no:cacheprovider tests/unit/test_no_git_state_in_tests.py \
  tests/unit/test_ordered_slice_guard.py \
  tests/unit/test_budget_meters_rendered_sections.py \
  tests/unit/test_tool_wiring_parity.py \
  tests/unit/test_model_capability_wiring.py
```

Do not run the full backend suite locally as part of this bounded plan. CI
owns it. If a broader local run becomes necessary, follow the daemon-down,
8 GiB cap, durable-output rule in `docs/DEVELOPMENT_WORKFLOW.md` §3 and run
only after pausing other test work. Never run two heavy test processes at
once. For branch settings use read-only verification after owner application:

```bash
gh api repos/lukehalleran/ai-assistant-framework/branches/master \
  --jq '{protected: .protected, sha: .commit.sha, protection: .protection}'
```

The command output alone is insufficient unless it shows the reviewed
required-check names/app, strict up-to-date policy, PR-required policy, and
no bypass/admin exception. If GitHub's endpoint omits a setting, inspect the
specific settings page/API response before claiming it.

## Stop, escalation, and completion rules

Stop the affected batch and preserve a red state when a required root is
ambiguous, a source owner has not resolved a product candidate, an anchor
migration loses occurrence identity, any unreviewed disposition is present,
or an expected workflow check did not run on the tested SHA. Do not remove a
candidate from the baseline to get green. Do not edit Plan 2 files; write a
request packet and wait for its response. Do not broaden scanner scope into
private/generated roots to manufacture counts. If the hosted repository
cannot accept the protection settings, stop before claiming required checks
and give the owner the exact setting/API failure and the prepared payload.

No helper/paid-model calls are needed. Use one Opus execution owner for the
shared policy, baseline/dispositions, catalog, hook, and workflow. Use Plan 2
only for owned application/test source requests. Avoid helper calls unless a
concrete independent code audit is needed; if one is justified, delegate a
single bounded review of a frozen diff and forbid edits to shared files.

The plan is complete only when all checklist items below pass:

- [ ] Exact scanner IDs/modes/classes are pinned; deletion, downgrade,
      unknown additions, partial-scan selection, and malformed registry fail.
- [ ] Full/partial output distinguishes input coverage, unavailable report
      evidence, unresolved cases, baseline mismatch, and scanner error.
- [ ] All required roots/legs are independently counted; syntax preflight,
      shallow new-root inventory, prefilter controls, DM-01, and DM-17 red
      controls pass.
- [ ] Missing/wrongly typed baseline and malformed rows fail closed; baseline
      writes are impossible in hook/workflow enforcement.
- [ ] Exactly 133 legacy baseline occurrences, including four duplicate
      extras, reconcile one-for-one to reviewed disposition history; active
      accepted debt is separate from confirmed fixed and false positives.
- [ ] Full-anchor schema migration is explicitly reviewed, multiplicity
      preserving, and bound to source hashes; no silent fingerprint rewrite.
- [ ] Five repo-wide guard files are strengthened within scope and pass their
      own red/green tests without mirroring implementation logic.
- [ ] Product requests have immutable request IDs/hashes, Plan 2 responses
      remain in its directory, and only Opus edits shared baseline/catalog/CI.
- [ ] Hook and one-owner workflow run the required full scan and guard lane;
      final status verifies dependency success and expected coverage/test
      receipts, with no path filter or status masking.
- [ ] GitHub run matches the reviewed commit SHA; owner-reviewed master
      protection requires PRs and unique expected checks with strict latest
      base, no bypass/admin exception, and a documented solo-review policy.
- [ ] Documentation says the ratchet is scoped structural coverage and
      lists unavailable inputs/uncovered catalog classes honestly.
- [ ] Changed paths are exclusively owned, checks pass at the tested SHA,
      and handoff names all unresolved requests/limitations.

## Ready-to-paste Opus brief

Execute `docs/execution/class_guards/PLAN_20260913_class_guard_completion.md`
as the sole writer for its listed paths. Read this plan, `docs/GENERALIZATION_CI_REVIEW_20260913.md`,
`docs/DEVELOPMENT_WORKFLOW.md` §3/§3a, `docs/TEST_LANES.md`,
`docs/BUG_CLASSES.md`, and the named scanners/hooks/workflow/tests before
editing. Follow “Fresh-session start” and work only in
`/home/lukeh/daemon_exec/class_guards`. Do not touch the main checkout, Plan 2 files, remote
settings, commits, pushes, or deployment while planning. Implement red
controls first; pin all seven scanner IDs/modes/classes; fail closed on
missing baseline, malformed schema/types, root/leg gaps, syntax errors, and
unresolved required evidence; correct DM-01 module exemption and DM-17 token
heuristic; preserve report-only DM-16/DM-29 as explicit candidate/unavailable
evidence. Review all 133 baseline occurrences by multiplicity and source
hash, including duplicates, then reconcile accepted debt separately from
confirmed fixes and false positives. Keep --write-baseline out of every
enforcement command. Strengthen only the five named repo-wide guard files.
Make Opus the sole workflow writer, integrate Plan 2 workflow requests once,
and do not claim CI or branch protection until a real run on the exact SHA
and owner-reviewed read-only proof. Prepare the concrete master settings
payload for owner review; do not apply it. Keep every product-source fix as
an immutable request for Plan 2's code owner. Stop red on unresolved review,
ownership collision, lost multiplicity, missing CI evidence, or protection
permission failure. Follow the scoped completion checklist and report exact
tests, run URL/SHA, settings proof, changed paths, open requests, and
limitations. No paid models or helper calls are planned.
