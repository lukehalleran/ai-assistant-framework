# Plan — repairs for the 2026-09-09 independent bug audit

_Agreed 2026-09-09 between the owner, Fable 5.1 (plan + referee) and Codex
(second reviewer + executor of B2/B5). Source: `docs/HANDOFF_20260909_independent_bug_audit.md`
(F01–F11, T01–T16). Every finding was verified on source by Fable; F01's
driver semantics were re-probed independently by both frontiers._

## Facts the plan rests on

| Fact | Established by | Consequence |
|---|---|---|
| Chroma 1.0.7 `collection.update(metadatas=)` MERGES and raises `ValueError` on a `None` value | Fable probe + Codex probe (same day) | A key can never be removed via update. Undo restores captured values exactly; a captured *absence* becomes `False` ("semantic restoration"). Never delete/re-insert to clear metadata. |
| `curation_quarantine_reason` is a separate key (`adapters.py:25`) | source | A `False` flip leaves the reason behind; the restore policy must cover both keys. |
| `get_ids_by_timestamp_range` has passed ISO strings to `$gte`/`$lte` since it shipped | source + Codex repro | Summary `source_doc_ids` were NEVER populated. Fix is forward-looking; historical linkage repair is a separate card. |
| `/api/graph` has no SPA consumer | grep `web/src` | F09 is lowest priority; fix stays cheap. |
| 15 temp-file writers exist outside `utils/safe_json.py` (`+ ".tmp"`, f-strings, `with_suffix(".md.tmp")`) | grep | F03's generalization must inventory all three forms; `google_auth.py` must keep 0600 perms. |
| `run_scan → _dispose → self.apply` (engine.py:210/261) | source | The curation op lock must be reentrant or scoped below `apply`; a naïve lock deadlocks auto-apply. |
| `api/routes/curation.py` calls `engine.apply/dismiss/undo` synchronously inside async handlers; only `scan_now` uses `to_thread` (with a timeout that does not stop the worker) | source | Lock acquisition must never block the event loop; a timed-out scan still owns the lock until it finishes. |
| Daemon pid 139086 started 2026-09-09 11:56, after the Sep-08 B7 tone edits | `ps` + mtimes | B7 is DEPLOYED but UNCOMMITTED. It must be committed before any batch here (§3a). |

## Batches

One commit per batch, each ≤ ~500 lines, each with its own regression tests
that call the deployed function. Sequence is fixed; B1 and B4 are the only
two that may run concurrently (disjoint files).

### B0 — owner, before anything else
- `git add -A && git commit -F commit_message_11.txt` (Sep-08 B7 tone batch).
- Decide: add a minimal Vitest/React lane to `web/` (both frontiers recommend yes).
- Daemon may stay up for B1–B5: nothing here writes `data/`.

### B1 — Live controls (F04, F10, F06) — executor Fable, referee Codex
- **F04** `gatherer_web.py`: read `WEB_SEARCH_ENABLED` / credit limit from
  `config.app_config` at call time (import-doctrine case 3), in BOTH the
  heuristic and LLM-trigger paths before provider dispatch. `apply_web_search`
  additionally sets `daily_limit` on the already-instantiated
  `WebSearchRateLimiter` reachable via `orchestrator.prompt_builder.context_gatherer.web_search_manager`.
  Check the agentic web tool path reads the same live value.
- **F10** `apply_streaming`: refresh `orchestrator.context_pipeline._enable_query_rewrite`.
  Keep timeout=0 semantics.
- **F06** `google_calendar_modify.py` update/delete success → `clear_cache()`;
  `google_calendar_create.py` partial-success batch → `clear_cache()`.
  Failure-only batches leave the cache alone.
- Tests (T07/T08 pattern): initialise the consumer once, observe, call the
  deployed setter, use the SAME instance again. Enable→disable→enable; lazy
  manager creation after a toggle; calendar read→mutate→read on one cache.
- Lead to look at alongside (not blocking): calendar cache key lacks
  `max_events`/`lookahead_days`; 250-event un-paginated fetches.

### B2 — Curation integrity (F01, F11, F03) — executor Codex, referee Fable
- **F01** `adapters.revert_change`: restore every captured key to its exact
  prior value; captured `None` (absent) → `False` for `curation_quarantined`
  and `""` for `curation_quarantine_reason`; document the policy in the
  module docstring. For generic `set_metadata` items: only keys with a
  defined neutral value are reversible; an item whose `before` cannot be
  restored is REJECTED before mutation (never applied, never "undone").
- **F11** write-ahead recovery: capture pre-images and write the journal
  `apply_started` record (with items + pre-images) BEFORE the first
  `apply_change`; if the journal or queue write fails, abort with no
  mutation. After mutation, record `applied`; on rollback record
  `apply_failed` with per-item outcome. Define the crash-between states
  (`apply_started` without `applied` = "interrupted", surfaced in the queue
  as undoable-from-journal). Persistence failures must not be swallowed
  into a success return.
- **F03 helper** `safe_json.atomic_write_json`: `tempfile.mkstemp` in the
  destination directory, fsync retained, cleanup only this invocation's
  file, `os.replace`. Add `atomic_write_text` for the notes/narrative/token
  writers and route the 15 hand-written sites through it (`google_auth`
  keeps 0600 — set mode before replace). Guard test: fail on any
  `.tmp` literal / f-string / `with_suffix(...tmp)` outside `safe_json.py`
  with a justified allowlist keyed by (path, function) — content-anchored
  like `test_ordered_slice_guard`.
- **F03 curation serialization**: one engine-level operation lock held by
  `run_scan`, `apply`, `dismiss`, `undo`; `apply` acquires it re-entrantly
  (RLock, or `_apply_locked` inner called by `_dispose`). API layer: all
  four engine calls go through `asyncio.to_thread`; a conflicting request
  gets an immediate 409 "busy" (try-acquire), never a wait on the loop. A
  timed-out `scan_now` returns "still running" — the worker keeps the lock
  until it finishes. No cancellation in this batch.
- Tests (T03/T05): a real `chromadb.EphemeralClient` collection through
  the deployed `MultiCollectionChromaStore` wrapper (apply→undo for absent /
  False / True prior states, unrelated metadata survives, reopen where it
  matters); the fake collection's `update` corrected to MERGE semantics and
  validated against the real one; deterministic overlapping-writer test with
  Events (both published files valid); target-write / queue-write /
  journal-write / rollback failures each separately + the F11 combination;
  engine reconstructed from disk after an interrupted apply; scan-vs-apply
  overlap returns busy; auto-apply under the lock does not deadlock.

### B3 — Storage (F02, F05, F08) — executor cheap Claude sub (Fable brief), referee Fable
- **F02** staged replacement in `upload_document` and `upload_text`: insert
  new chunks under a fresh `upload_batch` id FIRST; on success delete old
  chunks by title where batch ≠ new; on any failure delete only the new
  partial chunks (they are not user data yet) and leave the old version
  untouched. Check `delete_document`'s return value. Same-title collisions
  across type/source must not cross-delete.
- **F05** `timestamp_epoch` (float, UTC) written beside `timestamp` on every
  conversation write path (`add_conversation_memory`, `add_to_collection`,
  batch path, memory_storage's direct adds). `get_ids_by_timestamp_range`
  = numeric `$gte/$lte` query UNION a legacy pass over rows lacking the
  field, paged via `get(include=["metadatas"], limit, offset)` — never full
  documents, never the whole collection in one call; results merged even
  when the numeric query is non-empty. Naive historical timestamps are
  interpreted as local time (the writer's convention) and converted once;
  malformed values are skipped and counted. Measure the legacy-scan cost on
  a copy-sized collection and record it in the handoff; backfill card only
  if it matters.
- **F08** `MemoryExpander`: hygiene (`is_quarantined`, supersession, junk)
  applied to the ANCHOR and to summary/unsupported-collection fallbacks;
  cache entries versioned by (doc id, document hash, metadata hash) or
  bounded by a TTL, and `clear_cache()` called from the curation adapter
  after any chroma mutation. Ordinary tool retrieval honours suppression;
  an explicit "show quarantined" path is out of scope.
- Tests (T04/T08/T12): both ingestion entry points × {chunking, embedding,
  insertion} failures, retry with identical / changed content; real-driver
  timestamp boundaries inclusive, adjacent rows excluded, mixed new+legacy
  rows, offsets, malformed values; summary writer through the helper
  asserting persisted `source_doc_ids`; expand after quarantine / undo /
  edit of the anchor without a manual cache clear.

### B4 — Frontend + graph boundary (F07, F09) — executor cheap Claude sub, referee Fable
- **F07** `ActionOutcome` type gains `next_action_id`/`next_summary`; the card
  forwards the full outcome; `App`/chat state sets `pendingActionId` to the
  next id (or clears on the final item); `busy` resets when the id changes so
  the same mounted card is not left disabled. Approval AND rejection chains.
- **F09** `api/routes/system.py` converts the writer schema (`nodes` dict,
  `source_id`/`target_id`) at the boundary; fixtures produced by
  `GraphMemory.save()`; limits below/equal/above size, empty graph, degree
  ties, multiple relations per pair; selected edges reference selected nodes.
- Tests: Vitest lane (if B0 approves) driving the real card callback and the
  real state transition with an API-shaped response; interim node-transpile
  test acceptable only if CI runs it and it fails-before/passes-after. The
  appendix assertions are NOT retained (they assert the bug). Typecheck +
  build stay separate CI steps.

### B5 — Test trust (T01, T02, T13, T14) — executor Codex, referee Fable
- **T01** `tests/test_response_generator_comprehensive.py` (11 bodies) and
  `tests/test_actual_caching.py`: assertions propagate; provider failures
  become explicit failure-path tests; empty stream fails a success test.
- **T02** `test_prompt_timeout.py`: drive `UnifiedPromptBuilder.build_prompt`
  with event-controlled gatherers (partial timeout, one-section exception,
  cancellation); golden distress replay gains one sequence through the
  deployed pipeline entry point.
- **T13** backup target enumerator against a temp data root with every
  post-07-14 store present; backup→restore→reopen on tiny fixtures through
  the deployed functions; dry-run writes nothing; Daemon-stopped guard.
- **T14** a selection ledger (`docs/TEST_LANES.md`): every CI `--ignore`
  and pytest.ini ignore with reason, owner subsystem, replacement coverage,
  and the bounded command that runs it; new B2/B3 real-driver tests and the
  B4 frontend test collected in CI; `hooks/pre-push` maps source-only
  changes to their subsystem tests.
- Remaining T-items stay OPEN and assigned: T10/T11/T15/T16 → leads after
  B5; T12 partly absorbed by B3's tests; upload-lifetime retention contract
  → owner policy; whole-request budget → tracked separately.

## Definition of done (per batch)
1. Regression tests call the deployed function, fail on the pre-fix tree
   (recorded in the handoff results section, never via git reads), pass after.
2. Touched suites + the five repo-wide guards + ruff (+ typecheck for B4) green.
3. Referee pass on the diff; changelog entry + CLAUDE.md one-liner + memory note.
4. Owner commits with the drafted message, then restarts, then the live probe.

## Live probes after the restart (Daemon prong, relayed)
- B1: Settings → web search OFF → ask a fresh-news question → debug record
  shows no provider call; set credit limit 3 → a deep search is refused.
- B1/B4: a two-event calendar request → one batched card; approve → second
  card appears without reload; approve → both events in the next
  `[UPCOMING SCHEDULE]` without waiting out the 5-minute cache.
- B2: apply then undo a real quarantine card → the doc is retrievable again;
  activity log shows `apply_started` before `applied`.
- B3: after a clean shutdown, the newest summary's metadata carries
  `source_doc_ids`; re-attach an upload with the same title while forcing an
  embedding failure is NOT probed live (unit only).

## Contingencies
| Condition | Action |
|---|---|
| A reproduction from the appendix no longer fails | stop; diff the cited lines vs. the handoff before implementing |
| Chroma semantics differ from the two probes above | re-probe the installed driver; the probe is the authority, not the fake |
| A B3 legacy scan exceeds ~2 s on the live-sized collection | record the number, ship the numeric path anyway, open a backfill card (dry-run, owner-applied) |
| B2 lock design forces an API contract change | stop and ask; the SPA Curation Center consumes it |
| A repair needs historical store changes | read-only report + dry-run plan; owner applies with Daemon down |
| Any test reaches network/model download | stop; replace the transport boundary |
