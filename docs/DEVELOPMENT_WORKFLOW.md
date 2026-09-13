# Development Workflow — three prongs, one owner, credit-aware

_Written 2026-09-05 after an eight-day run of 19 commits (+49K net Python
lines, half of them tests). This document records how the work is actually
organised, the rules that keep it safe, and the changes recommended after
reviewing that run. It is the canonical description; CLAUDE.md carries the
per-session doctrine and points here._

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
   assertion. The only way to make a red push literally impossible is GitHub
   branch protection with the Tests check required on `master` (direct pushes
   are then rejected; work lands via a branch and PR) — owner's call.

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
   owner shows exactly those two lines and nothing else to type.

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
   Beside them, `hooks/pre-push` and CI also run the bug-class scan ratchet
   (`python scripts/check_bug_classes.py scan --root .` plus its stdlib-only
   lane `tests/bug_class_guards`): a NEW finding from any DM scanner, or a
   STALE entry in `config/bug_class_baseline.json` — fixed debt must be
   REMOVED from the baseline, never left behind — fails the push.
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
