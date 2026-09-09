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

## Results

### B1 — live controls (2026-09-09, Fable) — Codex PASS (round 2), COMMITTED e651582, live-probed 14:57–15:02
- Files: `core/prompt/gatherer_web.py`, `core/prompt/context_gatherer.py`, `knowledge/web_search_manager.py`, `core/agentic/tools.py`, `gui/settings_core.py`, `core/actions/google_calendar_modify.py`, `core/actions/google_calendar_create.py`, `tests/unit/test_web_fallback_general_intent.py` (patch target), new `tests/unit/test_sep09_live_controls.py`. 121 insertions / 15 deletions in source.
- Referee round 1 (Codex): (1) P1 `search()`/`multi_search()` bypassed the flag — three provider-adapter calls with Settings off; FIXED: both entry points return `DISABLED_ERROR` before the cache check, test through the public entry points. (2) P2 the helper sat inside the `except ImportError` block, orphaning four fallback constants; FIXED: helper moved below the block, import-time check confirms the fallbacks bind. (3) the F10 test re-derived the predicate; REPLACED with a deployed `build()` enabled → disabled → enabled case on one pipeline (stages stubbed on the instance). Private-attr traversal and calendar changes accepted.
- Failed-before evidence (recorded result, no git reads in tests): the revised file run against a detached worktree at HEAD `df61f4f` with the pyenv 3.11.8 interpreter → **16 failed, 4 passed**; the 4 passes are the negative controls (failed delete keeps cache, failure-only create keeps cache, orchestrator without a pipeline, full create already cleared). On the fixed tree → 20 passed.
- Regression: 555 in one run — the new file + `test_web_fallback_general_intent`, `test_api_debug_settings`, `test_calendar_update_delete`, `test_calendar_create`, `test_google_calendar`, `test_web_search_manager`, `test_query_rewrite_wiring`, `test_sep07_calendar_offer_continuation`, `test_sep05_web_cache_same_query`, `test_fetch_url`, the tool-health consumers (`test_github_write`, `test_calendar_turn_round3`, `test_sep07_reuse_stm_intent`, `test_sep08_document_continuity`, `test_evidence_transport`) + the five guards. `ruff check .` clean.
- Design notes: the setter reaches managers through `_web_search_manager` / `_agentic_controller` (private) because both public properties lazily construct instances — a settings write must never build a controller. `is_available()` reports the disabled state and `search()`/`multi_search()` enforce it themselves (`DISABLED_ERROR`, before the cache); instrument callers (`search(localize=False)`) are blocked too, which is what "disabled" means.
- Lead carried (not fixed): the calendar read cache is keyed on nothing — `max_events`/`lookahead_days` are not part of the key; 250-event un-paginated fetches (handoff "remaining leads").
- Live probe after the restart: Settings → web search OFF → fresh-news question → debug record shows no provider call and the tool-health block reads DISABLED; set the credit limit to 3 → a deep search is refused; delete a calendar event via card → the next turn's [UPCOMING SCHEDULE] omits it without waiting out the cache.
- Live probes (daemon restarted 14:49 from e651582; read from `daemon_debug.log`, not relayed): **P1 PASS** — Settings OFF → "Web search disabled in config", no Tavily call, reply said search is disabled. **P2 PASS** — cap set to 10 (UI minimum) → the live limiter logged `total today: 8.0/10` on the very next search; the old 100 is gone. **P3 half-PASS** — create → card → executed → `[GoogleCalendar] Fetched 5 upcoming events` on the next turn (cache cleared); the DELETE turn never produced a card, see F12.

### F12 — NEW (found by P3, 2026-09-09 15:02): forced action round proposes the wrong action type, then narrates a card that does not exist
Gate: `Tier 1: explicit write action detected (calendar_delete_event)` → controller: `Explicit action intent (calendar_delete_event) — forcing propose_action on first decision round` → protocol: `propose_action rejected: action_type='calendar_create_event' is unknown or missing required fields` → `Forced action round produced no action marker — retrying once` → `Model ready to answer (implicit)` → final reply "Queued the deletion: … Confirm and it's off" with NO pending card and no `NO_CARD_NOTICE`. Three defects: (a) the force prompt does not pin the DETECTED action type, so the model may emit a sibling type; a proposal whose type ≠ the detected type should be rejected with a retry prompt naming the required type (or coerced when params fit the detected spec); (b) the no-marker retry is one generic round — it should carry the rejection reason; (c) `claims_pending_card` missed "Queued the deletion … Confirm" (queue/confirm shapes). Owner: B4 or a B1-followup commit; regression = replay this exact turn shape through `controller` with a scripted model that answers create-then-nothing.
- Side observation: `gui/settings_core.save_settings` rewrites `config/config.yaml` WITHOUT comments (53 comment lines dropped by the P1/P2 toggles; `git diff config/config.yaml`). Pre-existing; owner should not commit that diff — restore the file from HEAD and re-set the credit limit, or accept the loss. Lead for B5/T07: comment-preserving YAML writer (ruamel) or a separate settings overlay file.

### B3 — storage (2026-09-09, subagent)

Files touched: `knowledge/reference_docs_manager.py`,
`memory/storage/multi_collection_chroma_store.py`, `memory/memory_expander.py`,
`memory/curation/adapters.py`, `tests/unit/test_memory_expander.py` (fixture
updates — see below), new `tests/unit/test_sep09_storage_repairs.py` (22
tests).

**F02 — staged replacement.** `upload_document`/`upload_text` now snapshot
the existing same-title chunks BEFORE inserting anything, stamp every new
chunk with a fresh `upload_batch` uuid4, insert the new version first, and
only on a successful insert delete the old chunks whose batch id differs
AND whose `type` metadata matches the type being written (when that field
is present on the old chunk) — a same-title doc of a different type/source
is never cross-deleted. A failed `add_batch_to_collection` call leaves the
prior version completely untouched (`add_batch_to_collection` only returns
ids on success, so there is nothing to clean up on the exception path) and
`result.success=False`. `delete_document(title)` (the explicit whole-doc
delete path) is unchanged in behavior, refactored onto a new
`_delete_chunk_ids()` helper shared with the replacement path. A delete
failure after a successful insert is logged at WARNING and does NOT turn
the upload into a failure (the new version is already durably stored).

**F05 — timestamp_epoch + real range query.** New
`multi_collection_chroma_store.timestamp_to_epoch()` parses ISO strings
(naive → interpreted as LOCAL time, matching this module's
`datetime.now().isoformat()` writing convention; offset-aware → converted
respecting its own offset; malformed → `None`). A new `_derive_epoch()`
helper adds `timestamp_epoch` (float) beside `timestamp` in
`add_conversation_memory`, `add_to_collection`, and
`add_batch_to_collection` whenever a `timestamp` string is present and no
epoch yet exists. `get_ids_by_timestamp_range()` now runs a numeric
`$gte`/`$lte` query over `timestamp_epoch` UNIONed with a legacy scan
(`coll.get(include=["metadatas"], limit=500, offset=N)`, paged, never the
whole collection or documents/embeddings in one call) over rows that lack
the field — malformed legacy rows are skipped and counted in a DEBUG log
line, malformed bounds return `[]` with a WARNING. `memory_expander.
_fetch_conversations_in_range` now calls `get_ids_by_timestamp_range()` +
per-id `get_by_id()` instead of `list_all()` (which pulled the entire
conversations collection into memory on every summary→source-conversation
expansion via the temporal-anchor strategy).

**F08 — expander anchor hygiene + fingerprinted/TTL cache.** `_do_expand()`
now applies the same `is_quarantined`/junk/supersession hygiene check to
the ANCHOR (via a new `_hygiene_block_reason()` that `_passes_hygiene()`
now delegates to) that was previously applied only to neighbors — covering
the plain temporal-window path, the non-expandable-collection fallback, and
the summary-anchor path (checked once, centrally, before dispatch, so
`_expand_summary` never sees a bad anchor). A blocked anchor returns
`{"turns": [], "anchor_id": ..., "error": "anchor suppressed: <reason>",
...}` in the same dict shape every caller (`core/agentic/tools.py`,
`core/agentic/formatters.format_expanded_results`) already handles via its
`error`/`turns` keys — no caller changes were needed. The cache is now
keyed the same `(memory_id, window, collection)` tuple but stores
`(anchor_fingerprint, result, cached_at)`; every cache hit re-fetches the
anchor's current doc+metadata (one `get_by_id`, no window fetch) and
recomputes a sha1 fingerprint over content+sorted-metadata, invalidating on
a mismatch, and separately bounds entries by `EXPANSION_CACHE_TTL_S=300`.
Because the anchor-only fingerprint check cannot see a mutation to a
*neighbor* document, `memory/curation/adapters.py` now calls a new
`memory.memory_expander.notify_chroma_mutation(doc_id)` after every
successful chroma `apply_change`/`revert_change` (a weakref registry;
`MemoryExpander.__init__` self-registers) — this clears every live
expander's whole cache, which a dedicated test (
`test_curation_mutation_on_a_neighbor_invalidates_the_whole_cache`) exists
specifically to prove is load-bearing (it fails if the notify wiring is
removed even though the anchor-only tests still pass). No import cycle:
verified both directions import cleanly (`memory.curation.adapters` →
`memory.memory_expander` has no path back).

**Test-fixture deviation (noted per the brief's "pick the conservative
reading" clause):** F08's anchor-hygiene fix is a genuine, audited behavior
change — a junk/quarantined/superseded anchor used to be served anyway.
Six pre-existing assertions in `tests/unit/test_memory_expander.py` encoded
the OLD (buggy) behavior or used deliberately-short placeholder content
that only survived because summary/reflection anchors were hygiene-exempt:
`test_junk_doc_excluded_from_window` (renamed
`test_junk_anchor_is_suppressed`, assertion inverted to match the fix,
plus a new `test_junk_neighbor_excluded_from_window` added to keep the
neighbor-exclusion behavior covered), and seven summary-anchor fixtures
whose placeholder text (`"Summary text"`, `"Summary"`, `"Orphan summary"`,
`"found it"`) was under `SUMMARY_MIN_CHARS=40` and would now be suppressed
as junk — lengthened to realistic sentences with no change to what each
test actually verifies (source_doc_ids/temporal-anchor/priority logic).

**Failed-before / passed-after (recorded by reverting the 4 touched source
files to their HEAD content via `git show HEAD:<path>` into a scratch
location + a plain filesystem `cp` over the worktree copies — no
`checkout`/`reset`/`stash` used; restored byte-for-byte from an md5-verified
backup afterward; `tests/unit/test_sep09_storage_repairs.py` was not
touched by the revert):**

| Class | Tests | Failed before | Passed after |
|---|---|---|---|
| `TestReplacementPreservesOriginal` (F02) | 5 | 3 | 5 |
| `TestTimestampToEpochHelper` (F05) | 4 | 4 | 4 |
| `TestTimestampRange` (F05) | 6 | 5 | 6 |
| `TestShutdownSummarySourceDocIds` (F05) | 1 | 1 | 1 |
| `TestExpansionHygieneAndCache` (F08) | 6 | 6 | 6 |
| **Total** | **22** | **19** | **22** |

Of the 3 F02/F05 tests that passed even before the fix: two F02 tests
(`test_success_replaces_old_version`, `test_retry_after_failure_with_
changed_content_succeeds`) only assert END state, which the old
delete-then-insert code also reaches on a clean run (the bug only shows up
when an insert FAILS mid-replacement, which those two tests don't do); one
F05 test (`test_malformed_bound_returns_empty`) passes on both trees
because the old code's broad `except` already turned a driver
`ValueError` on garbage bounds into `[]` — same outward result as the new
explicit malformed-bound check, for a different reason. All three genuinely
discriminating scenarios (mid-replacement failure preserving the old
version, cross-type non-deletion, every real range/legacy/offset/anchor-
hygiene/cache case) fail before and pass after.

**Legacy-scan timing** (ad hoc measurement script, not part of the
committed suite — a permanent 5,000-row insert would slow every CI run for
no ongoing benefit): a synthetic ALL-legacy (no `timestamp_epoch` on any
row) 5,000-row `conversations`-shaped ephemeral collection, queried through
the deployed `get_ids_by_timestamp_range()`:
- Full-range query (returns all 5,000 ids): **0.066 s**
- Narrow 10-minute window (returns 11 ids, same paging cost): **0.062 s**

Both are far under the ~2 s contingency threshold — per the plan's
contingency table, the numeric path ships as-is with no backfill card. (Insert
of the 5,000 rows themselves took 0.501 s, irrelevant to the read-path cost
being measured.) Real production timing may differ somewhat from this
ephemeral/fixed-embedding harness, but the cost is dominated by
metadata-only paging, not embedding computation, so it should be
representative.

**Guards + full sweep:** `ruff check .` — all checks passed. Five repo-wide
guards (`test_no_git_state_in_tests`, `test_ordered_slice_guard`,
`test_budget_meters_rendered_sections`, `test_tool_wiring_parity`,
`test_model_capability_wiring`) — 85 passed. Touched-suite sweep
(`test_sep09_storage_repairs`, `test_memory_expander`,
`test_refdocs_lazy_collection`, `test_upload_retrieval_pool`,
`test_upload_keyword_score_leak`, `test_curation_engine`,
`test_api_curation`, `test_multi_collection_chroma_store`,
`test_recent_summaries_fix`) — 175 passed.

**Deviations from the brief:**
- The "pick the conservative reading" clause was exercised once: `upload_
  document` has no per-call `metadata_overrides` parameter (unlike
  `upload_text`), so its written `type` is always the hardcoded
  `'reference_doc'` literal — the cross-type guard is exercised by
  `upload_text`'s `metadata_overrides={'type': ...}` path in the tests.
- `scripts/dedup_reference_docs.py` was NOT modified — it already groups
  replacement batches by the shared per-call `timestamp` value (one value
  per upload call), which the new `upload_batch` field doesn't replace or
  conflict with; confirmed by reading the script, no code change needed.
- No historical backfill script was written for `timestamp_epoch` (not
  requested — new rows get it going forward; the legacy scan serves old
  rows, and its cost is negligible per the timing above).
- `memory/memory_expander.py`'s module docstring and
  `test_memory_expander.py` fixture text were updated to document/reflect
  the F08 behavior change (see the deviation note above) — outside the
  4 core files but necessary for the suite to stay both green and honest
  about the new (correct) contract.
- Everything else matches the brief as written; no stop condition was hit.

### B2 — curation integrity (2026-09-09, Codex) — Fable referee PASS 16:20; owner commit pending

Base: `d18dd15` (B1 and B3 already pushed). Scope: F01/F11/F03 and their
acceptance tests. Both B3 `notify_chroma_mutation` calls and its module-level
import survive; the B3 storage and expander suites pass with the rewritten
adapter. F12 remains with its assigned executor.

**Failed-before evidence.** Before any source edits, the two resumed test files
(`test_sep09_atomic_writers.py`, `test_sep09_curation_integrity.py`) ran on
unchanged `d18dd15`: **12 failed, 3 passed, 4 warnings, 2.14 s**. Eleven failures
exercise deployed defects; the twelfth is the expected missing new
`atomic_write_text` API, not an independent behavioral reproduction. The three
passing controls were prior-False/prior-True restoration and preserving the old
file when publication fails. Further acceptance cases were added after this
baseline; the count does not claim every later-added test was run failed-before.
Receipt: `/tmp/daemon_b2_before_tests.txt`, also copied into the B2 snapshot.

**F01 — reversible adapters.** `prepare_change` validates and captures every
item before the first target write. Chroma absent flag/reason restore to
`False` / `""`; existing values restore exactly. An absent arbitrary metadata
key without a neutral policy is rejected before mutation. JSON stores record
missing keys separately from explicit nulls. Undo only patches touched fields,
preserves unrelated metadata, refuses later conflicting edits, and supports
retry after partial restoration. No delete/reinsert operation was introduced.
Profile/graph curation saves request `raise_on_error=True`; ordinary callers
retain their prior error-handling behavior. The fake Chroma collection now
merges metadata and rejects None, checked against installed Chroma 1.0.7.

**F11 — durable preparation and recovery.** Full proposal snapshots include
pre-images and monotonically increasing proposal revisions. The queue and
`apply_started` record must both sync before target writes. The append-only
journal propagates write/fsync errors and separates a previous torn append
from later complete records. Startup streams journal snapshots to reconcile a
missing, corrupt or stale derived queue; startup does not write target stores.

| Boundary | Result / recovery |
|---|---|
| Preflight, initial queue or prepare-journal failure | Apply raises; no target mutation starts. |
| Crash after preparation, during apply or during undo | Proposal reopens as `interrupted`; it appears in the queue and accepts Undo. |
| Target raises, including after writing | Roll back every attempted item, including the failing item; record per-item outcomes. Complete rollback is `failed`; incomplete rollback stays `interrupted`. |
| Applied-journal record fails | Apply raises and attempts rollback; failure evidence is written to both recovery copies independently. |
| Final queue refresh fails after a committed journal record | Operation raises; the committed journal still supplies the applied/undone state and undo pre-images after restart. |
| Undo conflicts with a later edit | Raise and retain the interrupted proposal; do not overwrite that edit. |

Unresolved interrupted proposals block further applies. Auto-apply propagates
persistence failures through `run_scan` too. Legacy activity-only journal rows
cannot reconstruct a lost pre-B2 queue; a surviving legacy queue remains
readable. The operation lock is per engine, not a transaction across other
writers/processes or hardware failure of both recovery copies.

**F03 — writer isolation and operation serialization.** The shared JSON/text
writer uses same-directory `mkstemp`, file and directory fsync, `os.replace`,
and cleanup of only its own temporary path. Existing permissions are retained;
new files default to 0600, and both OAuth writers explicitly request 0600 before
replacement. A directory-fsync error after replace is reported even though
the new contents may already be visible. Unique temporary names prevent
publication collisions; they do not serialize arbitrary read/modify/write
transactions outside the curation engine.

All 15 handwritten writer sites were migrated: profile, graph, corpus and
narrative context, category cache, narrative-staleness marker, daily/weekly/
monthly notes, visual metadata, Google/Outlook tokens, visual-entity backfill,
FAISS checkpoint, and duration-repair script. Maintenance scripts were edited,
not executed. `test_atomic_writer_guard.py` scans production Python for `.tmp`
literal, concatenation, f-string and suffix forms; its justified allowlist is
currently empty and stale entries fail.

One nonblocking reentrant engine lock serializes scan/apply/dismiss/undo and
queue snapshots. API operations execute with real `asyncio.to_thread` workers;
overlap returns 409 busy, scan wait timeout returns 504 "still running", and
the worker retains the lock until completion. Auto-apply can reacquire it.
There is no cancellation. The SPA recognizes the already-planned `interrupted`
status, offers Undo, disables Dismiss for it, refreshes after failed mutations,
and exposes queue-load errors plus a Refresh button. Existing endpoint URLs
and successful response shapes are retained with additive recovery fields.

**Validation.** **487 passed, 121 warnings, 10.77 s** in one bounded unit run
using pyenv Python 3.11.8, two-thread CPU caps, offline flags and
`DAEMON_TEST_MODE=1`. The run contains the three new B2 files; existing
`test_curation_engine`, `test_api_curation`, `test_safe_json`,
`test_sep05_curation_wave2`, `test_sep09_storage_repairs`, `test_memory_expander`,
`test_user_profile`, `test_user_profile_schema`, `test_corpus_manager`,
`test_google_auth`, `test_email_providers`, `test_graph_integration`,
`test_narrative_staleness`, `test_daily_notes_auto_update`,
`test_visual_memory_store`; and all five guards (`test_no_git_state_in_tests`,
`test_ordered_slice_guard`, `test_budget_meters_rendered_sections`,
`test_tool_wiring_parity`, `test_model_capability_wiring`). Warnings are Chroma
legacy embedding configuration, SWIG, and an existing source escape sequence
encountered by the AST guard. Ruff and the SPA TypeScript/production build
also pass; Vite reports a circular `katex → vendor → katex` chunk warning.

Tests include real ephemeral Chroma with supplied tiny embeddings, profile and
graph reopening from temporary files, torn journal/corrupt queue recovery,
write/rollback failures, all-target preflight, conflicting edits, overlapping
publishers, reentrant auto-apply, busy operations, and actual ASGI requests
through scan timeout and interrupted Undo. Old test fixture changes are
limited to Chroma's real merge/None contract, semantic False undo assertions,
and the new strict-save keyword on profile doubles. No model download or live
store mutation was needed. The sandbox blocks local socketpair notifications;
the complete run required approved execution outside it so worker completion
could wake the event loop. An earlier sandbox regression run was interrupted;
its partial results are not counted as a passing run.

**Scope / workflow notes.** The original approximate 500-line guideline is
exceeded by the coupled write-ahead recovery, 15-site migration and failure
tests; this remains the agreed single B2 batch. Minimal curation UI changes
make the planned recovery state usable and handle the planned 409 response.
Snapshot receipt: `.agent_snapshots/b2_20260909_154343` contains pre-edit source
archive/hashes, original local CLAUDE notes, and test/build logs. A bounded
source snapshot was used instead of repeating the session-start script's
105+ GiB store hashing/rotation. No claim is made that concurrently live data
files remained byte-identical. The historical handoff points here; local
CLAUDE notes are updated; commit draft is `commit_message_b2.txt`.

**Resume / memory note for Fable.** B2 is implemented on `d18dd15`, uncommitted,
and ready for referee review. Review recovery revision ordering, rollback of
the item that raises after mutation, API worker lock lifetime, and the two
preserved B3 invalidation calls. B5 still needs to wire the new B2 guard and
real-driver tests into permanent CI lanes. No commit, push, Daemon restart or
live quarantine/undo probe has been performed by Codex. After referee PASS,
the owner commits/pushes and performs the already-planned B2 live probe.

**Fable referee (B2):** diff read end-to-end (adapters preflight/`prepare_change` + conflict-aware undo with `missing_before`, engine RLock + `CurationBusyError` + `apply_started` write-ahead + `interrupted` recovery from the journal, worker-owned API operations returning 409/504, `mkstemp` writer with directory fsync and mode preservation, 15-site migration, content-anchored guard). Independent run: 517 passed (three new B2 files + curation/storage/expander/profile/graph/corpus/notes/auth/visual suites + five guards), ruff clean, `tsc --noEmit` clean. Both B3 `notify_chroma_mutation` calls verified present. Non-blocking note: `pending()` is `@_serialized` under the non-blocking lock, so `GET /api/curation/queue` returns 409 while a scan runs (SPA shows an error + Refresh) — a later batch could serve a snapshot from a separate short lock. Scope exceeded the ~500-line guideline as Codex documented; accepted as one batch because F03/F11 share the engine.
