# Development Workflow — three prongs, one owner, credit-aware

_Written 2026-09-05 after an eight-day run of 19 commits (+49K net Python
lines, half of them tests). This document records how the work is actually
organised, the rules that keep it safe, and the changes recommended after
reviewing that run. It is the canonical description; CLAUDE.md carries the
per-session doctrine and points here._

## 0. Start here — the map (added 2026-09-16)

Point a model at this file and it should be able to find everything else without the owner
re-explaining. The rules of the road are in §3–§5; this section is the index.

### 0.1 What you may and may not do

- **Never** commit, amend, push, delete or overwrite `data/`, restart the Daemon, or apply a
  store script. Draft the commit message and a runner (§3a.7); the owner types the two lines.
- Work in a clone under `~/daemon_exec/` (never the live checkout `~/Daemon_v1` — the running
  Daemon deploys whatever is on disk there), with the push URL disabled until the owner enables it.
- Every python invocation from a clone is `env -u PYTHONPATH DISABLE_FS_GUARD=1 DAEMON_TEST_MODE=1
  PYTHONDONTWRITEBYTECODE=1 python -s …` and prints `module.__file__` once — the login shell's
  `usercustomize.py` otherwise imports the LIVE repo's `utils` into every interpreter.
- pytest only under `systemd-run --user --scope -p MemoryMax=4G` (unit) / `8G` (the non-unit
  remainder, Daemon DOWN); never `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` on a pytest line (it kills
  pytest-asyncio); never traverse `data/` (41M-row parquet); `rm` is a no-op wrapper — use
  `/usr/bin/rm` only for your own clone-local files.
- A live probe is wrapped `[test]…[/test]` or it becomes a profile fact.

### 0.2 Reading order by task

| You are asked to… | Read, in order | Then |
|---|---|---|
| Understand the system | `ARCHITECTURE_GUIDE.md` (concepts, request lifecycle, memory tiers) → `PROJECT_SKELETON.md` (every module, one line each) → `QUICK_REFERENCE.md` (API + constants lookup) → `FORMAL_MODEL.md` (the scoring/gating math) | the subsystem doc for the area (0.3) |
| Fix a bug / audit a turn dump | `BUG_CLASSES.md` — run the DM detectors for the families in scope BEFORE reading code → `BUG_RETROSPECTIVE_20260715_20260904.md` (the narrative behind the classes) → `TEST_LANES.md` (which tests to run, memory caps) | the loop in §3; every fix names `class: BC-nn` |
| Add or change a feature | `GOALS.md` → `WORKPLAN_2_hardening_executable_beta.md` (priorities: no-bugs over features) → `generalization/README.md` + `generalization/01-product-contract.md` (what must stay owner-neutral) → `FORMAL_MODEL.md` if it touches scoring | `PROMPT_BUILDING_PIPELINE.md` if it touches the prompt |
| Verify behaviour live | `PROBE_HARNESS_DESIGN.md` (probe set + expected response shapes, the planned third prong) → `test_live_system.md` (manual CLI plan) → `eval/corpus.py` (27 labelled seed queries with expected-behaviour tags) → `tests/benchmarks/` | relay a turn (§2) and audit its debug record |
| Ship (test, commit, push, PR) | §3a (commit discipline, runners, hooks, branch protection) → `TEST_LANES.md` | `hooks/pre-push` runs the guards + full unit pass; CI's `bug-class-gate` blocks the merge |
| Run/build/deploy | `BUILD_GUIDE.md`, `DOCKER_README.md`, `WEB_FRONTEND.md`, `AGENT_SAFETY.md` | — |
| Pick up open work | `FOLLOWUPS.md` (repo root, untracked — one line per open item, newest first) → the owner-local `CLAUDE.md` (doctrine + dated one-liners) and `CLAUDE_CHANGELOG.md` (full narratives) → `docs/HANDOFF_*.md` (owner-local, gitignored) | §5 handoff format when you stop |

### 0.3 Subsystem documents (contracts per area)

| Area | Document | Code |
|---|---|---|
| Memory tiers, collections, retrieval, TTL | `MEMORY_SYSTEM.md` | `memory/`, `memory/storage/` |
| Prompt assembly, gating, budget | `PROMPT_BUILDING_PIPELINE.md` | `core/prompt/`, `processing/gate_system.py` |
| Agentic loop, tools, actions | `AGENTIC_SEARCH.md`, `EMAIL_INTEGRATION_DESIGN.md` | `core/agentic/`, `core/actions/` |
| Tone / crisis detection, anti-amplification | `TONE_DETECTION_SUMMARY.md`, `postmortems/2026-07-tone-flatline.md`, `ADAPTIVE_LEARNING.md` | `utils/tone_detector.py`, `utils/adaptive_exemplars.py` |
| Pattern / insight mode | `PATTERN_ANALYSIS.md` | `memory/pattern_engine.py`, `core/insight/` |
| Curation (never-delete engine) | `AUTONOMOUS_CURATION_DESIGN.md` | `memory/curation/`, `api/routes/curation.py` |
| Self-improvement proposals | `PROPOSAL_SYSTEM.md` | `knowledge/proposal_generator.py`, `agent_branch/` |
| Synthesis (frozen line of work) | `SYNTHESIS_FILTER.md`, `SYNTHESIS_VALIDATION.md`, `SYNTHESIS_CALIBRATION_PLAN.md`, `LITERATURE_ORACLE.md`, `grading_plan.md` | `knowledge/synthesis_*` |
| Thinking blocks, tags, web UI | `THINKING_BLOCKS_IMPLEMENTATION.md`, `TAG_GENERATION.md`, `WEB_FRONTEND.md` | `core/response_parser.py`, `utils/tag_generator.py`, `api/`, `web/` |
| Evaluation & metrics | `BENCHMARK_METRICS.md`, `METRICS_SNAPSHOT.md` (generated — `scripts/generate_doc_metrics.py`), `eval/README.md` | `eval/`, `tests/benchmarks/` |
| Dated audits and plans | `AUDIT_*.md`, `INDEPENDENT_AUDIT_*.md`, `PLAN_*.md`, `GENERALIZATION_*.md`, `docs/execution/` (batch records, briefs, class-guard requests/responses) | historical evidence; read when a class or batch is cited |

Three of these are **runtime inputs**, not just prose: `PROJECT_SKELETON.md`, `QUICK_REFERENCE.md`
and the root `CLAUDE.md` are read in full by the proposal generator, `GOALS.md` is sliced for
ranking queries, and every top-level `docs/*.md` is chunked into the `reference_docs` collection at
startup. Reformatting them changes model behaviour (`docs/execution/compaction_20260915/PLAN.md`).

### 0.4 How work is organised (the credit pattern)

Frontier plans → cheap subagents execute in parallel on disjoint files → frontier referees (§4).
A plan is fool-proof or it is not a plan: verified `file:line` facts, exact commands, acceptance
checks, stop conditions, contingencies. Executors get a filled copy of
`docs/templates/EXECUTOR_BRIEF.md`; they escalate on a stop condition and never improvise. The
referee reads the diff, re-runs a check, greps the BC-58 siblings, and only then writes the combined
commit message. At the 20 % weekly-credit warning the frontier stops implementing; near the limit it
emits a §5 handoff so Codex can continue.

### 0.5 The bug-class discipline in one paragraph

Every audit starts by running the detectors (`DM-nn`) of `BUG_CLASSES.md` for the families in scope;
every fix ends by naming its class in the commit body and changelog line; a mechanism with no entry
gets one (checklist at the end of that file — two incidents or a strong reason, a runnable Find, an
honest Status). The structural lane is `python scripts/check_bug_classes.py scan --root .` +
`tests/bug_class_guards/` + the accepted-debt ledger `config/bug_class_dispositions.json` (whole-file
SHA — any edit to a debt file needs the class-guard owner's re-review before push). Repo-wide guard
tests (`tests/unit/test_*_guard.py`, the wiring/metering parity tests, `test_import_hygiene_guard.py`)
run in `hooks/pre-push` and CI; a guard is lowered by a batch, never raised.

### 0.6 Where things live

| What | Where |
|---|---|
| Live checkout (the Daemon runs from it) | `~/Daemon_v1` — read it, never edit it from an agent |
| Working clones and per-batch run artifacts | `~/daemon_exec/<name>/`, handoffs in `~/daemon_exec/<lane>_runs/` |
| Owner runners (commit / PR), read-only for agents | `~/daemon_checkpoints/` |
| Owner-local doctrine and history (gitignored) | `~/Daemon_v1/CLAUDE.md`, `CLAUDE_CHANGELOG.md`, `docs/HANDOFF_*.md`, `FOLLOWUPS.md` |
| Stores (never write from a script while the Daemon runs) | `data/` (`chroma_db_v4`, `corpus_v4.json`, `user_profile.json`, `knowledge_graph.json`, backups) |
| Logs and receipts | `daemon_debug.log` (live) + rotated archives, `logs/turn_records.jsonl`, `logs/actions_audit.jsonl`, `logs/curation_audit.jsonl` |
| Tests | `tests/unit/` (unit lane), `tests/` (non-unit remainder — Daemon down), `tests/bug_class_guards/` (stdlib-only), `tests/benchmarks/` |

## 1. The three prongs

| Prong | What it is | Strengths | Cost model | Typical role |
|---|---|---|---|---|
| **Frontier Claude** (Claude Code, Fable 5.1, xhigh reasoning) | Interactive agent with repo, shell, tests, and the local stores | Root-cause analysis from debug records, plans with contingencies, referee review of other agents' diffs, live probes of deployed functions | Metered weekly plan (5×); the expensive resource | Audit, plan, review, verify, write docs |
| **Codex** (GPT-6 Astra) | Independent agent on the same repo | Second frontier opinion, adversarial review, independent audits, executing a well-specified plan in parallel | Separate credit pool that resets on its own schedule | Second reviewer, parallel executor, continuation when Claude's budget is low |
| **Daemon** (the system itself) | The running product, with full memory (Chroma collections, corpus, profile, graph, notes) and any OpenRouter model per turn | Ground truth about what the stores contain; end-to-end reproduction of turn behaviour; per-turn debug records | Pay-per-token through OpenRouter; every wasted call is a real cost | Subject under test AND an oracle; live verification after a restart |

Under each frontier sits a **cheap tier**: Sonnet/Haiku subagents on the
Claude side, small GPT models on the Codex side. Daemon's own cheap tier is
its model routing (fast models for the trigger, verifier, STM, and narrative
work; the active chat model is user-selected).

Nothing is committed, pushed, deleted, or applied to `data/` by any prong.
The owner does those by hand, after a dry run, with Daemon down.

## 2. The relay protocol (the owner is the switchboard)

The prongs cannot talk to each other. Any frontier may **stop and ask the
owner to relay** a message to another prong, and the owner pastes back the
reply and, when asked, the debug record for that turn.

Rules for a relay request:

1. **State the target and the exact text** to send, in one copy-pasteable
   block. Say what should come back (reply only, or reply + debug record).
2. **Batch.** One relay with three questions beats three relays. Each
   round-trip costs the owner minutes of copy-paste, often from a phone.
3. **Clear the value bar.** Relay for: a design disagreement that needs a
   second frontier, live verification after a restart, or data that only
   Daemon's memory holds. Do not relay for anything a local grep or a
   read-only script can answer.
4. **Mark relayed text.** When another agent's output is pasted into
   Daemon, prefix it with `[relay: <agent>]` or paste it as a blockquote.
   Daemon's fact extractor otherwise mines the pasted sentences as the
   owner's own words (2026-09-05: a status line from a Claude session became
   a `works_on=` profile fact). The extractor's quoted-correspondence rule
   already skips blockquotes; the `[relay:` prefix is the convention to add.
5. **Debug records are the evidence.** A relayed "it worked" is an opinion;
   the per-turn debug record (prompt sections, timings, gate decision,
   response) is what gets audited.

Direct OpenRouter access from a Claude session (scripts, curl) needs no relay
and costs the owner nothing. Use it to probe a model's quirks or validate a
capability row; it has no Daemon memory and no Codex context.

## 3. The standard loop

```
turn dump / telemetry  →  root cause (read the code that ran, not a re-derivation)
  →  fix + regression test that calls THE deployed function
  →  name the bug CLASS (docs/BUG_CLASSES.md); a new mechanism gets a new
     entry with a Find method and a Closure, and its detector is run once
     across the repo before the batch closes (sibling sites — BC-58)
  →  targeted suites green + ruff clean
  →  probe the deployed function live (read-only script, or a relayed Daemon turn)
  →  CLAUDE.md one-liner + CLAUDE_CHANGELOG.md narrative + memory note
  →  owner: reviewed exact-file commit runner, then separate git push (§3a.7; BEFORE restart)
  →  restart Daemon
  →  live probe after the restart (the fix is not real until this passes)
```

Rules that hold at every step (from CLAUDE.md, repeated because they are the
ones agents break):

- **Validation calls the deployed function.** Never a proxy, never a
  hardcoded baseline. Five incidents came from this class.
- **Human-gated commits.** Agents draft `commit_message.txt`; the owner
  commits. The pre-commit hook runs gitleaks and a private-term scan; use
  synthetic names in comments and fixtures (the repo is public).
- **Dry-run first, pre-image always.** Every store-writing script defaults
  to dry run, writes a backup before `--apply`, and refuses to run while a
  live Daemon holds the store (`utils/daemon_guard.py`). Look at a target
  before overwriting it, even under `data/`.
- **Never load large datasets fully.** 16 GB machine; batch the test suite
  (a single full-suite process gets cgroup-killed). The non-unit batch
  (everything outside `tests/unit`) is NEVER run beside the live Daemon —
  on 2026-09-07 it ran during a daemon shutdown (LLM extraction + summaries
  in flight) and the box swap-thrashed to a hard reset with no OOM kill in
  the journal. Run it with the daemon down, under
  `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=512M` so the
  kernel kills pytest instead of the machine, and write the output somewhere
  that survives a reboot (`~/daemon_checkpoints/`, not `/tmp`).
- **Prefer neutralization over deletion.** Read-time suppression, then
  reversible metadata, then additive changes; deletion is a human click or
  a terminal step, never automatic (`docs/AUTONOMOUS_CURATION_DESIGN.md`).
- **Every probe/dump text becomes an outcome fixture (added 2026-09-10).**
  A live probe text and every dump text that exposed a defect goes into that
  batch's test file verbatim, driven through the deployed function — and the
  assertion checks the OUTCOME the owner would actually see (the card's
  params, the reply's claims, the notice text), never just which arm or
  route fired. 2026-09-10: the first probe pass judged the calendar/note
  turns on routing alone; the second probe passed routing and still failed
  at execution (a card minted with a bare clock time that rejected on
  approve) — a fixture asserting only "gate chose tools" would have stayed
  green through that failure. Round 2 (same day): a live fixture must also
  carry the live CONTEXT, not just the live query text — the actual plan
  point, the actual STM topic/user_question, and the actual window
  including the assistant's OWN prior replies — since a synthetic stand-in
  exchange can accidentally omit the exact word (e.g. "doctor") whose
  presence in Daemon's own reply was the thing masking the bug. Round 3
  (same day): a live text fixture must ALSO carry the client's actual
  surface form, not just its content — the client line-wraps long messages
  ("...a new doc I\n  think will be helpful"), and a shape predicate fixed
  and tested against the clean string alone stayed broken on the live input
  (BC-58/BC-64). Every live-text fixture is now asserted in BOTH its clean
  and its wrapped/indented form.
- **A vocabulary miss is closed with a categorized-generic table or a
  learned channel, never a new phrase appended to a regex** (added
  2026-09-10, round 3; `docs/GENERALIZATION_AUDIT_20260901.md` §"Remedy
  patterns"; catalogued as `docs/BUG_CLASSES.md` BC-76). Four dated batches
  (08-15, 08-27, 09-07, 09-10) closed the same class of vocabulary-drift
  incident (BC-15) by literally adding one more phrase to an existing
  regex/list; the fix pattern itself is what recurs, not any one incident.
  Reach for one of this project's own generalized remedies instead — a
  single chokepoint module (CM-01: `utils/trigger_match.py`), a categorized-
  generic vocabulary with per-user anchors (`terms_are_private_sphere_generic`),
  or a seeds+learned/auto-promoted channel (`adaptive_exemplars`,
  `learned_relations`) — and name which one a fix used.
- **Before a round is called done, run the BC-58 Find method on every
  function the round touched** (added 2026-09-11, round 5): grep every
  sibling site that renders, routes, or consumes the same data
  (`rg -n "<field or helper>" core/ gui/ utils/ knowledge/`) and either wire
  the fix at the PRODUCER so all consumers inherit it, or list each sibling
  as covered/exempt in the handoff. Rounds 2-5 of the 2026-09-10 probe batch
  each exposed a sibling path the previous fix skipped (a clean-string
  predicate vs the wrapped input; `key_points` vs `strategy`; the formatter
  render sites vs the agentic decision digest and planner digest) — a live
  retest is not the tool for finding siblings, the grep is.

## 3a. Commit discipline (added 2026-09-07)

The Daemon runs from the working tree: every restart deploys whatever is on
disk, committed or not. So a commit is not "saving work", it is the record of
what production ran. On 2026-09-06/07 two batches (~1,600 lines) sat
uncommitted across two days of restarts; by the time the tree was reviewed the
batches shared hunks in the controller, the types module, the formatter and
one test file, and the "one commit per root cause" the doc asks for was no
longer possible without hand-editing hunks. A bisect over that window would
have been blind, and a revert all-or-nothing.

Principles:

1. **Commit at every verified batch boundary, before the restart that deploys
   it.** "Verified" = referee pass done, touched suites and the four repo-wide
   guards green, ruff clean. Two or three commits a day at the current pace is
   normal; each one must be revertable on its own.
2. **Keep the message shaped, not long.** One subject line, one paragraph of
   root cause and fix, one line naming the tests. The narrative belongs in
   `CLAUDE_CHANGELOG.md` and the handoff doc, both of which are written at the
   same boundary. A 50-line body is a sign the commit is too big.
3. **A retest that proves a fix wrong gets a follow-up commit, never an
   amend.** The record of what the retest taught is worth more than a tidy
   history (2026-09-07 needed two retest rounds after the first "green" batch).
4. **No hunk surgery.** If two batches have already merged in the tree, one
   combined commit is the honest history; write it and move on. The fix is
   principle 1, not `git add -p`.
5. **Docs and code move together.** The changelog entry, the CLAUDE.md
   one-liner, the handoff doc's results section and the memory note are part of
   the batch, written before the commit. Public documentation is staged with
   code; owner-private handoffs, CLAUDE.md, CLAUDE_CHANGELOG.md and memory
   notes remain local when excluded by .gitignore. Do not force-add them.
   Any commit `Docs:` line must describe the files actually included.
6. **The pushed commit is the tested tree, and tests never read git state.**
   The first push after adopting these principles (05fd300) went red: four
   tests proved their "failed-before" evidence with `git show HEAD:<file>`, so
   they were green only while the change was uncommitted and failed the moment
   it was committed — and the full local suite had run on the dirty tree before
   the commit. Two structural closures: `tests/unit/test_no_git_state_in_tests.py`
   fails on any test that reads a blob from a project ref or runs git against
   the repo root (git against a `tmp_path` repo stays allowed), and
   `hooks/pre-push` (installed as `.git/hooks/pre-push`) refuses a push from a
   dirty tracked tree or with untracked `.py` files, then mirrors the CI privacy
   guard and ruff and runs the changed test files + the five repo-wide guards
   — in one pytest process under `MemoryMax=6G`, 2 GiB below the §3 cap the
   non-unit batch needs. A push range that touches non-unit test files
   therefore gets SIGKILLed, not failed (2026-09-10: the T01 repair push, 24
   files, killed at ~40%; the ten non-unit files alone peak at 5.97 GB under
   6G and pass under 8G at 7.1 GB). For such a push: Daemon down, run the
   hook's selection by hand under the 8G wrapper with durable output, then
   `SKIP_PREPUSH=1 git push` citing that output (`docs/TEST_LANES.md` §4).
   Failed-before evidence is a recorded result in the handoff doc, never an
   assertion. The only way to make a red landing literally impossible is
   GitHub branch protection. The owner adopted it on 2026-09-13; §3a.8 has the
   pull-request flow, the `bug-class-gate` required status, and the settings
   payload.

7. **Two owner lines, always — a runner and a push.** The owner commits from
   a phone. Every batch therefore ends with (a) a commit message file
   `commit_message_<N>.txt` that the agent has re-read against the ACTUAL
   staged scope (`git diff --stat` + the untracked list) right before handing
   it over — never a message written earlier for a scope that has since
   grown — and (b) ONE runner `~/daemon_checkpoints/commit_<batch>.sh` (or a
   single `git add <exact files> && git commit -F commit_message_<N>.txt`
   line) that: refuses to run unless HEAD is the expected short SHA; adds
   exactly the batch's files by name (never `git add -A`, never gitignored
   files — a runner that `git add`ed CLAUDE.md aborted mid-way on
   2026-09-10); commits with `-F`; and ends with a clean-tree check. Line 1
   is that runner. Line 2 is `git push` — separate, typed by the owner, never
   chained into the runner (`SKIP_PREPUSH=1 git push` only when the handoff
   cites the hand-run evidence, §3a.6). The agent's final message to the
   owner shows exactly those two lines and nothing else to type. Once `master`
   is protected (§3a.8), line 2 pushes the batch BRANCH, and the pull request
   and merge follow.

8. **Every landing goes through a pull request (owner decision, 2026-09-13).**
   `master` is to be protected with:
   - pull requests required;
   - one required status check, `bug-class-gate`, published by the GitHub
     Actions app (id 15368);
   - strict "branch up to date" before merge;
   - administrators included (no admin or direct-push bypass);
   - 0 required approvals.

   **How the gate works.** `bug-class-gate` needs every other job in
   `.github/workflows/tests.yml` (`bug-class-scan`, `test`, `frontend`). It
   runs even when a dependency failed or was skipped, and fails unless each
   dependency's result is `success`. It then re-verifies the scan receipt and
   the harness and guard JUnit receipts, against the checked-out policy,
   baseline and dispositions, for `$GITHUB_SHA`. GitHub lets skipped and
   neutral checks satisfy a required status, so the gate itself is never
   conditional and never trusts job colours alone.

   **Why "pull request required" is explicit.** A required check alone does not
   stop a direct push of an already-checked commit.

   **Solo maintainer.** 0 approvals avoids self-review deadlock, so **no
   independent review is enforced** while there is one maintainer. Add a
   required reviewer or code-owner rule when an eligible second reviewer
   exists.

   **Landing flow** (every batch, Plan 2's included):
   1. the §3a.7 runner commits on a branch;
   2. `git push -u origin <branch>`;
   3. `gh pr create --base master --head <branch>`;
   4. merge only after `bug-class-gate` has passed on the pull request's
      current head.

   A push to a branch that is neither `master` nor `refactor/prompt-modular`
   runs no workflow until its pull request exists.

   **Settings.** The reviewed payload is
   `docs/execution/class_guards/branch_protection_master.json`. Only the owner
   applies it, and only after a real pull-request run shows the `bug-class-gate`
   check name and app:

   ```bash
   gh api -X PUT repos/lukehalleran/ai-assistant-framework/branches/master/protection \
     --input docs/execution/class_guards/branch_protection_master.json
   ```

   **Verify read-only afterwards.** Protection is never reported from the
   payload alone. The output must show the `bug-class-gate` check with app
   15368, `strict: true`, `enforce_admins.enabled: true`, and a
   `required_pull_request_reviews` block:

   ```bash
   gh api repos/lukehalleran/ai-assistant-framework/branches/master/protection
   gh api repos/lukehalleran/ai-assistant-framework/branches/master \
     --jq '{protected: .protected, sha: .commit.sha, protection: .protection}'
   ```

   If the API refuses the payload (plan or permission limits), stop. The
   landing policy is then not enforced, and the refusal is recorded with the
   exact API error.

## 4. Credit discipline

The frontier tier is for judgment; the cheap tier is for execution. Default
team shape for any multi-step task:

1. **Frontier plans.** Problem statement, verified facts with `file:line`,
   numbered steps, exact commands, acceptance checks, and a contingency
   table ("if X fails, do Y").
2. **Two cheap subagents execute in parallel** whenever the plan splits
   cleanly (for example, one on the fix, one on the tests, or two
   independent fixes). The count is flexible; two is the default, and the
   split must not put both agents on the same file.
3. **Subagents escalate, never improvise.** The plan names the stop
   conditions: ambiguity in the spec, a failed contingency, unexpected
   store state, a test failing in an unrelated module, anything that
   touches a shared contract (cache keys, dispatch tables, store schemas),
   anything under `data/`, anything with privacy implications. On a stop
   condition the subagent reports state and waits for its frontier.
4. **Frontier referees.** Reads the diff, runs or re-runs the tests, and
   probes the deployed function before the change is called done. The
   afternoon of 2026-09-05 caught an over-firing rule this way.

Budget signals:

- At the **20 % weekly-credit warning** on the Claude plan: frontier keeps
  planning and reviewing only; all reading and implementation moves to
  cheap subagents with precise briefs.
- **Near the limit:** stop and emit a handoff (section 5) so Codex can
  finish. Work lost to a hard credit stop is worse than a clean handoff.
- **Daemon-side cost is engineering work, not a plan setting.** Duplicate
  calls hide in parallel paths: the web-search trigger ran twice per turn
  for a day because two callers built different cache keys. Per-turn
  timings and token counts in the debug record are the place to look.

## 5. Handoff format

Used whenever one prong hands work to another (frontier → cheap, Claude →
Codex, session → next session). One file under `docs/HANDOFF_<date>_<topic>.md`
or a block at the end of the session:

```
STATE        what is on disk vs. what is committed vs. what the running Daemon has
CLASS        the bug class(es) from docs/BUG_CLASSES.md (BC-nn), or "new: <mechanism>"
ACTIONS      changes made, file:line, tests run and their counts
PLANNED      numbered next steps with exact commands and acceptance checks
CONTINGENCY  if step N fails → do this; stop and ask if …
WHY          the reasoning, so the receiver does not re-derive it
OWNER        anything only the owner may do (commit, apply, restart, credentials)
```

## 6. Roles matrix

| Action | Frontier Claude | Codex | Cheap subagents | Daemon | Owner |
|---|---|---|---|---|---|
| Read debug records, find root cause | ✓ | ✓ | with a brief | — | relays them |
| Write plans with contingencies | ✓ | ✓ | — | — | — |
| Implement + write tests | when small | ✓ | ✓ (default) | — | — |
| Referee a diff | ✓ | ✓ | — | — | — |
| Run tests / lint | ✓ | ✓ | ✓ | — | CI after push |
| Probe deployed functions live | ✓ (read-only scripts) | ✓ | — | ✓ (relayed turns) | relays |
| Regenerate derived files (narrative, metrics) | ✓ with backup | ✓ | — | ✓ on its schedule | — |
| Write `data/` stores | — | — | — | ✓ (its own) | scripts, dry-run first, Daemon down |
| Commit / push | drafts message | — | — | — | ✓ |
| Restart Daemon | — | — | — | — | ✓ |

## 7. Recommended changes (prioritised)

The loop above works; the eight-day review found the gaps in cadence and
batch size, not in the loop.

1. **Nightly full test suite.** Full runs are rare (2026-08-03, then
   2026-09-07 — the `docs/METRICS_SNAPSHOT.md` date, 0 failures); the
   batches between ran targeted suites, and the 2026-09-05 audit found two
   regressions introduced by the previous day. Add a scheduled run (systemd
   timer or a scheduled GitHub Actions job) in the memory-capped batches;
   a red result blocks the next batch. CI on push covers only what was
   pushed. Same-day example: the a690a91 push went red on
   `tests/unit/test_ordered_slice_guard.py` alone — its allowlist pinned
   line numbers, and unrelated hunks higher in two files shifted them
   (the allowlist is now anchored on function + source line, so pure
   drift no longer goes red; a real edit to a slice still does).
   Repo-wide guard tests (`test_no_git_state_in_tests`, `test_ordered_slice_guard`,
   `test_budget_meters_rendered_sections`, `test_tool_wiring_parity`,
   `test_model_capability_wiring`) are never in a module-scoped local
   selection; `hooks/pre-push` runs them on every push (§3a.6).
   Beside them, `hooks/pre-push` and CI also run the bug-class contract
   (`python scripts/check_bug_classes.py scan --root .`, its stdlib-only lane
   `tests/bug_class_guards`, and `verify-receipts`; `docs/TEST_LANES.md` §4).
   The scan checks the pinned policy (seven scanners, every input leg and root,
   the top-level Python inventory, a syntax preflight). It then ratchets the
   gate scanners in both directions against `config/bug_class_baseline.json`.
   Each baseline occurrence needs a reviewed record in
   `config/bug_class_dispositions.json`, bound to its source file's SHA-256.

   Fixed debt is REMOVED from the baseline and kept as `confirmed_fixed`
   history. An edited file with accepted debt fails until the class-guard
   owner re-reviews it. Current gates are DM-01/17/18/31 plus catalog
   structure; DM-16/29 are report-only.

   This is a scoped structural ratchet, not coverage of every known class;
   each scan lists the uncovered classes. A failing CI check prevents landing
   only once branch protection requires `bug-class-gate` (§3a.8). The independent verification,
   generalization findings, and third-prong probe backlog are consolidated
   in [the 2026-09-13 review](GENERALIZATION_CI_REVIEW_20260913.md).
2. **One commit per root cause, restart immediately, then probe.** Three
   agents' work landed on one dirty tree on 2026-09-05 and the running
   Daemon predated every fix for hours. Commit as soon as a fix is green
   (five commits a day is fine), restart, run the live probe. Move the
   restart button up the curation-center roadmap so this stops needing a
   terminal.
3. **Cap batch size for review.** Today's commit was 1,900 lines. A
   referee misses things above roughly 500 lines; split by root cause.
4. **Add a sensor that is not the owner.** Every defect this week was found
   by the owner pasting a dump. Schedule an agent (nightly) that reads
   `logs/turn_records.jsonl` and the day's debug records, ranks anomalies
   (timeouts, duplicate calls, verifier flags, gate misroutes, junk facts),
   and files candidates as curation cards or a handoff. The curation engine
   already has the queue, journal, and undo; the trust ladder
   (shadow → queue → auto) graduates a curator only after dismissal-free
   runs.
5. **Keep CLAUDE.md as doctrine.** Dated entries are one line with a
   pointer; narrative lives in `CLAUDE_CHANGELOG.md`. The file is loaded
   into every session; 400 dense lines is context spent before work starts.
6. **Adopt the `[relay:` marker** (section 2) and teach `fact_source` to
   skip marked blocks. Until then, relayed agent text becomes profile facts.
7. **Runner scripts for any multi-command owner operation.** A long command
   pasted from a phone wrapped and executed a data file as a command on
   2026-09-05. Ship `data/apply_<date>.sh`-style runners with `set -euo
   pipefail` and one-line invocation.
8. **Cost roll-up.** Per-turn token and latency numbers exist in the debug
   record; add a daily roll-up (by call site: trigger, verifier, STM,
   narrative, chat) so duplicate-call regressions show up as a number, not
   in an audit weeks later.
9. **Parity tests for every shared contract.** The tool-wiring parity test
   pattern (every dispatch entry has a handler) should extend to cache
   keys, store adapters, and prompt-section metering: a second call site
   that drifts from the first must fail a test, not an audit.

10. **Keep the bug-class catalog current and run its detectors.**
   `docs/BUG_CLASSES.md` (started 2026-09-10 from the whole audit record: 72
   classes, 26 detection methods, 14 closure methods) is the index every
   audit starts from and every fix ends at: the commit body and changelog
   line carry `class: BC-nn`; a fix with no matching mechanism adds a class;
   an audit runs the detectors (DM-nn) for the families in scope BEFORE free
   reading, so the same class is not rediscovered by dump. Status is
   downgraded, never argued up. A periodic (weekly) detector sweep — the
   grep/AST/telemetry methods, not a code read — is the cheap sensor that
   finds a recurrence of a `partial`/`recurs` class before the owner does.

## 8. Anti-patterns observed (so they are recognisable)

- The fix that creates the next bug: a "policy-aware" cache key added in
  the morning defeated its own cache by the afternoon.
- The live instance re-saving over a script's writes (2026-08-05), and the
  guard that was supposed to prevent it having a hole (2026-08-21), and one
  script never having the guard at all (2026-09-05).
- Model errors becoming memory: a reply misread "up til 5" as "five hours
  of sleep"; the next turn inherited the number from the conversation
  history. The verifier is log-only, so only a human sees it.
- A count copied across days: the user's "day 6" on Sep 2 was restated as
  "six days" for three days by the reply, the daily note, and the narrative;
  nothing advanced it. Deterministic ledgers beat prompt rules for anything
  arithmetic.
- Deleting or overwriting a target without looking at it first, including
  candidate files under `data/`.
- Batches merging in the working tree before they merge in history
  (2026-09-06/07): two days of restarts deployed ~1,600 uncommitted lines,
  and shared hunks then made per-root-cause commits impossible. Commit at
  the batch boundary, before the restart (§3a).
