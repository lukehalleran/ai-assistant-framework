# Autonomous Curation — Design

**Status (2026-08-28): Phases 1–3 core BUILT** — `memory/curation/` engine
(types/journal/adapters/engine/service), Wave-1 curators (error-sentinel
quarantine, stream-artifact repair, junk-fact quarantine, temporal
staleness), quarantine respected at the three retrieval filter sites,
shutdown phase 5/6, `api/routes/curation.py`, SPA Curation Center page
(🧹 navbar view), config section (`curation`, max_mode=queue), conftest
sandbox. Tests: `tests/unit/test_curation_engine.py` (27) +
`test_api_curation.py` (5). **Remaining:** contradiction-supersession wiring
(cross_deduplicator → queued curator), ops buttons (re-auth / backups /
restart / export), Wave-3 judgment curators (dual-LLM, queue-forever).
Owner decisions recorded: quarantine-not-delete YES; ceiling=queue for now,
auto later per-curator after graduation; judgment curators never auto.

## Goal

A new user runs Daemon and never opens a terminal. Data stays healthy without
hand-curated candidate files, `--apply` flags, or owner-gated scripts. The
curator's prime directive is **never be wrong**: it must always prefer inaction
over a wrong action, and every action it does take must be reversible.

## Why this is designable now

The 2026-08 curation-script inventory (see below) shows the manual work
decomposes into exactly three things:

1. **A coordination lock.** Most scripts refuse `--apply` while Daemon runs,
   because an external process writing a store the live daemon holds in memory
   gets clobbered on the next save (the 08-05 profile-clobber incident). This
   is not a judgment problem — it disappears entirely if the same logic runs
   *inside* the daemon, which owns its stores.
2. **A deployed-predicate button-press.** `purge_error_memories`,
   `strip_special_token_artifacts`, `repair_thinking_leaks`,
   `backfill_stance`, `purge_junk_facts`, `reclassify_proposals` all apply THE
   deployed classifier (never a re-derivation — CLAUDE.md rule). The human
   adds nothing except pressing the button and being present for a disaster.
3. **A genuine judgment call**, in a small minority of cases: "is
   `nickname='hot bartender'` a joke artifact?", "is graph node `user_k` a
   real person?", synthesis candidate grading. Only here does a human (or a
   very carefully guarded model) actually decide something.

Classes 1–2 can be fully automated. Class 3 gets a **one-click in-app queue**
(no terminal), with a narrow, guarded path to autonomy for the reversible
subset.

## Prime directive, operationalized

Ordered preference of instruments — the curator must always use the
**highest** instrument that solves the problem:

1. **Read-time neutralization (no write at all).** Junk filters at retrieval,
   TTL aging, load-time skipping (e.g. the 08-28 `_alias_bindable` guard that
   neutralizes junk graph aliases at index rebuild without touching disk).
   Zero risk; wrong rules are fixed by a code change, no data was mutated.
2. **Reversible metadata writes.** Supersession (`is_current=False` +
   `superseded_by`), staleness flags, `[CORRECTION …]` annotations. Nothing
   is lost; a wrong call is one metadata flip away from undone.
3. **Additive writes.** New facts, new flags, new journal entries.
4. **Deletion.** NEVER automatic. Ever. (CLAUDE.md rule #1.) Deletion is
   always a human click in the UI, with preview, pre-image backup, and undo.

Confidence requirements per instrument:

| Instrument | Minimum evidence |
|---|---|
| Read-time neutralize | deterministic rule, regression-tested |
| Auto metadata write | deterministic deployed predicate, OR dual-independent-LLM agreement + sentinel pass |
| Queued proposal (human click) | single signal of any kind |
| Deletion | human click, always |

A single LLM opinion is **never** sufficient for any automatic write. This is
the `auto_label_intents --verify` doctrine (dual-model agreement or marked
uncertain) promoted to a system-wide rule.

## Architecture

### 1. CurationEngine (in-daemon)

A new module `memory/curation/` running as a **shutdown phase** (alongside
dedup/backup — the daemon owns its stores, so the daemon-running guard becomes
moot) and optionally as an idle-time pass.

Each curator implements one contract:

```
class Curator(Protocol):
    name: str
    def scan(self, stores) -> list[CurationProposal]   # NEVER writes
```

`CurationProposal` (Pydantic): target store + doc/fact/node id, instrument
(neutralize/metadata/additive/delete), the exact change, human-readable
evidence, confidence source (deterministic | dual-llm | single-llm),
pre-image. The engine — not the curator — decides disposition:

- instrument ≤ metadata AND confidence == deterministic → **auto-apply**
  (journal + undo record)
- instrument ≤ metadata AND confidence == dual-llm AND sentinels pass →
  **auto-apply**, but only after the curator has graduated (see Trust ladder)
- everything else → **queue** for one-click review in the UI

Engine-level guards (all curators inherit, none can opt out):

- **Sentinel abort**: each curator ships sentinel cases (the `backfill_stance`
  casey-evil pattern — known items that must classify a specific way). Any
  sentinel violation aborts the curator's ENTIRE batch for that run and
  raises a UI notice. Sentinels are also run in CI.
- **Rate cap**: max auto-actions per run (default 25); exceeding the cap
  converts the overflow to queued proposals — a runaway rule can't sweep a
  store.
- **Anomaly halt**: if a curator's proposals in one run exceed X% of the
  store (default 5%), nothing auto-applies — everything queues. Mass action
  is exactly when a rule is most likely wrong.
- **Journal**: append-only `logs/curation_audit.jsonl` (same pattern as
  actions_audit) — every scan, proposal, apply, skip, undo.
- **Undo**: every applied action stores its pre-image in the journal;
  `undo(action_id)` restores it. Surfaced as a button in the UI for the last
  N days of actions.
- **Pre-image backups**: batch-level, under `data/backups/curation/`,
  retention via the existing backup_manager policy.

### 2. Trust ladder (how a curator earns autonomy)

New curators NEVER start in auto mode, regardless of instrument:

1. **Shadow**: scan runs, proposals journaled only (not even queued). Used to
   measure over-fire before a human ever sees noise.
2. **Queue**: proposals appear in the UI; human clicks apply/dismiss. Every
   dismissal is recorded with the proposal's features.
3. **Auto**: eligible only for instrument ≤ metadata, only after ≥ N queued
   runs (default 5) with a dismissal rate of 0 for deterministic curators
   (any dismissal of a "deterministic" proposal means the rule is wrong —
   back to shadow) or < 2% for dual-LLM curators. Graduation is recorded in
   config, visible and revocable in Settings.

This is the measured-adoption doctrine (budget experiment, probe-first
exemplars) applied to the curator itself. The FP-halt pattern
(`SYNTHESIS_AUDIT_FP_HALT_THRESHOLD`) already exists for exactly this shape.

### 3. The curators (initial set)

**Wave 1 — deterministic ports of existing scripts (auto-eligible):**

| Curator | Replaces | Instrument |
|---|---|---|
| stream-artifact repair | `strip_special_token_artifacts`, trailing-`e`, `<\|sep\|>` | metadata (content repair w/ pre-image) |
| thinking-leak repair | `repair_thinking_leaks` | metadata |
| error-sentinel quarantine | `purge_error_memories` | **metadata, not delete**: add `quarantined=True`, retrieval already filters via `is_junk_*`; the stored pile stops mattering. Human may bulk-delete quarantined docs from the UI whenever. |
| junk-fact quarantine | `purge_junk_facts` | metadata (same quarantine flip) |
| stance backfill | `backfill_stance` | metadata (additive tags) |
| proposal risk reclassify | `reclassify_proposals` | metadata |
| relation re-canonicalization | `graph_relation_normalize` | metadata (rename, journaled) |
| profile junk-fact retirement (2026-09-05) | `purge_profile_facts` + curated id file | metadata (`supersede_profile_fact`, reason `junk_object`; the quick profile — what [USER PROFILE] renders — was uncovered by the chroma-side junk-fact curator) |
| when-word graph-node quarantine (2026-09-05) | `graph_junk_cleanup` T1 temporal class (`today`, `tomorrow`, `on_thursday`, weekdays) | metadata (node-level `curation_quarantined` through the graph adapter; `GraphMemory.edge_is_suppressed` hides every edge touching a flagged node; the T2/T3 judgment classes stay with `GraphNodeReviewer` below) |

Note the quarantine move: today's purge scripts delete; the curator instead
flips a flag the retrieval filters already respect. Deletion stops being
load-bearing — it becomes optional disk hygiene a human can trigger in bulk
from the UI ("Empty quarantine: 312 docs, 2.1 MB").

Owner-side by-id counterpart (2026-09-02): `scripts/quarantine_facts.py` flips the same
`curation_quarantined` metadata on facts a provenance audit names directly (dry-run first,
pre-image backup, `--undo` reverses; Daemon must be down). First use: the WHOOP and two
invented `has_dog` facts from the 2026-09-02 claim-support trace.

**Wave 2 — deterministic staleness (the drop-deadline class):**

`TemporalStalenessCurator`: any fact/thread/note-flag whose *own content*
carries a resolvable date that has passed (deadline=, appointment=, due=,
event dates) is auto-superseded with reason `date_passed` once
`now > date + grace`. This is provably safe: "drop deadline is Fri 2026-08-28
3 PM" is stale on Aug 29 by arithmetic, not judgment. Uses the deployed
`temporal_resolver`; unresolvable/ambiguous dates → no action (prime
directive). Superseded-by-date facts remain queryable ("what was the
deadline?") — supersession, never deletion. This kills the single biggest
recurring manual-curation driver observed in practice: profile facts that
were true when stored and silently rot.

Also in wave 2: **contradiction supersession** already exists in
cross_deduplicator (08-03 rework) — wire it from dry-run-on-shutdown into the
engine as a queued curator first, per the trust ladder.

**Wave 3 — judgment curators (dual-LLM, queue-first, maybe-never-auto):**

- `JunkIdentityFactReviewer`: the 'hot bartender' / 'empty house' / joke-
  artifact class. Two independent models judge "is this a durable identity
  fact or a conversational artifact?", shown WITH the source excerpt (already
  stored on profile facts). Agreement → queued proposal to supersede.
  Disagreement → nothing (not even a queue entry — anti-noise).
- `GraphNodeReviewer`: the T2/T3 graph-junk class, same shape. (The
  deterministic T1 temporal subset moved to Wave 1 on 2026-09-05 —
  `GraphTemporalNodeCurator`.)
- These may reasonably stay queue-forever. The queue IS the win: the human
  decision collapses from "run a scanner, read a candidate file in an editor,
  uncomment lines, run --apply with the daemon down" to one glance and one
  click in the app.

**Explicitly NOT automated:**

- Synthesis candidate grading — it trains the classifier and encodes taste;
  it already has a good UI (Gradio tab; needs SPA port only).
- Personality/config choices, proposal approval (already risk-gated UI).
- Any deletion, anywhere.

### 4. Curation Center (SPA)

New SPA page (+ thin `api/routes/curation.py`), replacing candidate files and
`--apply` runs:

- **Queue**: proposals grouped by curator, each showing evidence, source
  excerpt, exact before/after, confidence source. Apply / Dismiss per item;
  select-all within a group. Dismissals feed the trust ladder.
- **Activity**: journal view of auto-applied actions with per-action **Undo**.
  This is the transparency that makes auto mode acceptable: nothing is
  silent, everything is reversible, the user can watch it work.
- **Quarantine**: count + size, "review" list, optional bulk delete
  (the ONLY deletion surface, explicit and human).
- **Settings integration**: per-curator mode (shadow/queue/auto) with the
  graduation state visible; global kill switch.

### 5. Terminal-elimination checklist (ops gaps beyond curation)

From the UI-coverage sweep, a non-technical user still needs the terminal
for: Google re-auth on token revocation (`reauth_google`), backup restore
(`restore_backup`), data export (`export_user_data`), daemon restart, and
first-time index/backfill tasks. Each becomes a Settings-page action:

- **Re-connect Google** button → runs the OAuth consent flow, browser opens,
  token backed up first (existing script logic, in-process).
- **Backups panel** → list snapshots, one-click restore *staged for next
  restart* (restore itself must not run under a live daemon — the restart
  boundary is the natural, safe apply point), export archive download.
- **Restart Daemon** button (clean shutdown + relaunch via the existing
  single-instance/idle machinery) — also how curation fixes and restores
  take effect without a terminal.
- Backfills (visual memory captions, wiki index download) → wizard steps /
  Settings buttons with progress, reusing the existing job pattern from
  `/api/sync-notes`.

Historical one-time repairs (zelphex annotations, refdocs dedup, etc.) are
NOT ported — they're Luke-specific incidents; new users start behind today's
inflow guards.

### 6. What "never wrong" costs, explicitly

- Stale-but-unresolvable facts (no date in content) will sit until a
  judgment curator or human touches them. Accepted: wrongness > staleness.
- Dual-LLM disagreement produces silence, not a queue entry. Some real junk
  survives. Accepted: the queue must stay high-precision or the human stops
  reading it (the boy-who-cried-wolf failure kills the whole design).
- Auto mode arrives slowly (trust ladder). Accepted: graduation is cheap to
  wait for and catastrophic to rush.

## Build order

1. **Engine + journal + undo + quarantine flag + Wave 1 curators** (shutdown
   phase, shadow→queue). Largest de-risking step; no new judgment anywhere.
   *Status 2026-09-05:* shipped, plus the profile junk-fact and when-word
   graph-node curators and a graph store adapter (`StoreBundle.graph_memory`,
   `store="graph"` / `quarantine_node`, pre-image + undo). Stores the engine
   can write: chroma collections, the quick profile, graph node metadata.
2. **Curation Center SPA page** (queue + activity + undo). From this point
   the terminal is no longer needed for any recurring curation.
3. **TemporalStalenessCurator** (the drop-deadline class) + contradiction
   supersession wiring.
4. **Ops buttons** (re-auth, backups, restart, export).
5. **Wave 3 judgment curators** (dual-LLM, queue-first).

---

## Appendix: inventory snapshot (2026-08-28)

Ongoing human-gated needs found: synthesis grading (UI exists),
adaptive-exemplar poisoning recovery (guarded, script-only), profile/graph
junk review (script + candidate files — the deterministic subset became
in-app cards on 2026-09-05; node DELETION and the T2/T3 judgment classes
remain script/candidate-file work), intent labeling (optional, LLM path
exists). One-time repairs pending: profile 9-fact list + graph candidates
(20260805 files), error-sentinel purge remainder. ~15 automatic hygiene
mechanisms already deployed (TTL, supersession, junk filters, learned
relations, adaptive exemplars, canonicalization, read-time strips) — the
curator extends this doctrine; it does not replace it. UI today: settings,
synthesis grading, proposals, dedup button (Gradio-only), debug/provenance;
no UI for profile facts, memory cleanup, backfills, restore, or re-auth.
