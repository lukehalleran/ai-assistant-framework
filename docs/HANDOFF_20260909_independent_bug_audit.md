# Independent broad bug audit — 2026-09-09

**Repair tracking:** this document preserves the original audit evidence.
The [agreed repair plan](PLAN_20260909_audit_repairs.md#results) records subsequent
implementation and verification, including B2 (F01/F11/F03), awaiting Fable review.

**Status: investigation complete; no fixes applied.** Eleven findings were
reproduced against the current working tree: five P1 and six P2. Existing
targeted tests pass despite these defects. This is a handoff for a later repair
session, not authorization to implement repairs or modify stores.

**Test-gap extension:** the [handoff to Fable 5.1 high](#test-gap-review--handoff-to-fable-51-high)
below adds 16 prioritized coverage work items, maps all eleven findings to
acceptance tests, and records confirmed assertion weaknesses. No tests or
product code were changed by this review.

Method: `docs/DEVELOPMENT_WORKFLOW.md` §§3, 5, 8; trace producers to consumers,
call deployed functions, challenge fake-store assumptions with the installed
database, inject bounded failures, check sibling paths, and distinguish
reproductions from production observations. Earlier audits were read for
deduplication and known limitations; their conclusions were not used as proof.

## STATE

- Branch: `master`; HEAD: `21a914e7f0ccb962e71b8bdc62b3deee7a537c94`.
- The audited code is the **working tree**, including pre-existing uncommitted
  September 8 changes. HEAD alone does not reproduce this exact tree.
- Pre-existing modified files, preserved byte for byte:
  `core/prompt/gatherer_knowledge.py`, `core/response_parser.py`,
  `docs/HANDOFF_20260908_homework_session_audit.md`, `utils/need_detector.py`,
  `utils/query_checker.py`, `utils/tone_detector.py`.
- Pre-existing untracked file, also preserved:
  `tests/unit/test_sep08_homework_tone_misfires.py`.
- This audit adds only this handoff to the reviewable repository changes.
  No source/test fixes, configuration changes, commits, staging, pushes,
  restarts, live-store writes, or external actions were performed.
- Pre-audit snapshot: `.agent_snapshots/20260909_020845/`, including starting
  status, diffs, untracked test archive, and filesystem hashes. The seven
  pre-existing changed/untracked files match its hashes.
- The running Daemon's loaded revision was **not established**. No claim below
  is a verified post-restart production incident. Sandboxed process visibility
  is insufficient to certify that the owner’s daemon is down.
- On the final check, production data/log files had changed since the starting
  snapshot, with timestamps during the September 9 continuation. The stores
  were therefore not an immutable snapshot throughout this session. Do not
  interpret preservation of the source tree as a claim that all runtime data
  hashes stayed fixed; no runtime changes were reverted or repaired.
- Python used: 3.11.8; installed ChromaDB used in probes: **1.0.7**.

## ACTIONS

### Coverage and limits

| Area | Reviewed / exercised | Boundary |
| --- | --- | --- |
| Chat/API/frontend | SSE admission/cleanup, history, approval routes, actual card and App callback | No browser or live chat replay |
| Curation | Apply/undo, pre-images, queue/journal durability, scan threading, real metadata updates | Synthetic stores; no live curation cards applied |
| Storage/retrieval | JSON writer concurrency, reference replacement, timestamp linking, expanded memory hygiene/cache | Tiny ephemeral Chroma collection; no full live-store scan |
| Runtime settings | Setter → cached consumer, search enable/credit limit, query rewriting | Providers replaced with recording stubs |
| Calendar/actions | Create/modify/delete, proposal status/chaining, read-cache invalidation | HTTP MockTransport; no Google calls or credentials used |
| Graph | Actual GraphMemory serialization → API trimming | Two synthetic nodes and one relation |
| Prompt/agentic | Retrieval ingress, expansion dispatch, controller lifetime, recent contracts | Whole-model behavior and all gate heuristics not exhaustively tested |
| Operations | Snapshot, baseline guards, lint/typecheck, bounded telemetry aggregates | No full suite, non-unit batch, model downloads, or paid LLM calls |

Read `logs/turn_records.jsonl` incrementally for September 7–8: 116 rows,
including eight explicitly flagged test rows; **108 remaining rows**. Those
108 include 51 agentic-search turns, 56 enhanced turns, one insight turn,
and 27 sticky-floor triggers. Zero had `response_len == 0`; none had recorded
preparation above 30 seconds. These are descriptive aggregates, not proof of
an error rate, model quality, or absence of unflagged test traffic. The recent
homework/tone findings already under repair were not refiled from these counts.
No private conversation excerpts are copied into this document.

### Verification results

- Unit batch A: **180 passed**, 3 warnings, 22.21 seconds.
- Unit batch B: **100 passed**, 4 warnings, 9.57 seconds.
- `ruff check . --output-format concise`: **passed**.
- `npm run typecheck` in `web/`: **passed**.
- Python reproduction appendix: **10 finding records**, all assertions passed.
- Frontend reproduction appendix: **1 finding record**, assertions passed.
- Preservation check: every tracked code/config/document file present in the
  starting manifest still matches its starting hash; so does the pre-existing
  untracked test. HEAD is unchanged. The new handoff is the only added
  reviewable file. Post-session checking used those hashes and a runtime-data
  stat comparison instead of rerunning the full session-audit hash sweep:
  the starting manifest covers **105.45 GiB**. The full
  `scripts/agent_session_audit.sh` was not rerun.

The appendices assert the **observed failing behavior** to make this audit
repeatable. They are evidence scripts, not proposed regression tests. After a
fix, replace those assertions with the acceptance conditions below; do not
preserve a bug merely to keep an audit script green.

Recorded output paths for this session:

```text
/tmp/daemon_audit_20260909_baseline.txt
/tmp/daemon_audit_20260909_baseline2.txt
/tmp/daemon_audit_20260909_ruff.txt
/tmp/daemon_audit_20260909_typecheck.txt
/tmp/daemon_audit_20260909_probes.txt
/tmp/daemon_audit_20260909_frontend.txt
```

Reproduction source is embedded below so a later session does not depend on
those temporary files surviving. The first harness draft had an incorrect
`GraphMemory.add_edge` call; it was corrected to the actual `add_relation`
method and rerun successfully. That harness error is not a product finding.

## FINDINGS

P1 = prioritize for data integrity or enforcement of an explicit setting;
P2 = meaningful behavior/reliability defect. Priority reflects consequence and
reachability, not a claim that the fault occurred in production.

| ID | Priority | Confirmed defect |
| --- | --- | --- |
| F01 | P1 | Curation reports undo while Chroma memory remains quarantined |
| F02 | P1 | Failed document replacement deletes the old stored copy |
| F03 | P1 | Overlapping atomic JSON writes can publish invalid JSON |
| F04 | P1 | Search disable and daily credit changes miss live consumers |
| F05 | P2 | Summary source-ID lookup always rejects ISO timestamp ranges |
| F06 | P2 | Successful calendar deletion leaves deleted event in prompt cache |
| F07 | P2 | Browser discards the next pending action after approval |
| F08 | P2 | Memory expansion returns quarantined anchors and stale cached text |
| F09 | P2 | Graph API trimming cannot consume the saved graph schema |
| F10 | P2 | Query-rewrite toggle does not update the running pipeline |
| F11 | P1 | Curation can mutate durable data without a durable undo record |

### F01 — Curation undo uses replacement semantics against a merge API

**Locations:** `memory/curation/adapters.py:138` (metadata restore),
`memory/curation/engine.py:377` (undo),
`tests/unit/test_curation_engine.py:60` (fake collection update).

`revert_change()` removes newly introduced keys from a Python dict, then calls
`coll.update(metadatas=[restored])`. Chroma merges the provided metadata;
omitting a key does not remove the stored key. The fake used by the engine
and API tests replaces the entire metadata dict, concealing the mismatch.

**Reproduction:** a real ephemeral collection starts with no quarantine key.
Call the actual engine's `apply()` then `undo()`. The proposal becomes
`undone`, but the stored `curation_quarantined` remains `True`. This affects
ordinary quarantine cards and metadata rollback that relies on the same adapter.
If the remaining metadata is empty, also test the installed driver's handling
of an empty metadata update; the successful probe retained a `source` field.

**Acceptance:** actual Chroma apply/undo and failed-batch rollback restore
read visibility and the prior values of every touched key; unrelated metadata
survives. Test absent, initially false, and initially true quarantine states.
Use supported explicit clearing/neutralization semantics. Correct the fake
contract as well as adding a tiny real-driver test.

### F02 — Replacement deletes before it can succeed

**Locations:** `knowledge/reference_docs_manager.py:234`, `:314`, `:346`,
`:737`; callers include `gui/handlers.py:389`, `:410` and startup sync in
`gui/launch.py:290`.

Both file and text upload paths delete existing chunks with the same title
before chunking/embedding/inserting the replacement. A failed insertion
returns an error but never restores the old chunks. The delete return value
is also ignored. This is replacement reached through normal attachment and
sync paths, not just the explicit delete UI.

**Reproduction:** insert an original `user_upload` into a real ephemeral
collection; call actual `upload_text()` with the same title while only the
batch-insert boundary raises a synthetic embedding error. Result:
`success=False`, and querying the original ID returns `[]`. The original
source file, if any, is outside this probe; the demonstrated loss is its
stored document/chunks. Pasted text need not have another durable source.

**Acceptance:** failures during chunking, embedding, insertion, or publication
leave the prior version retrievable. Stage a new version before switching
which version is current; retain a recoverable pre-image. Test both entry
points and title collisions across source/type. Do not turn a failed delete
into duplicate mixed versions, and do not introduce automatic data deletion
as the recovery mechanism.

### F03 — The shared `.tmp` name defeats atomicity under overlap

**Locations:** `utils/safe_json.py:107` / `:117`;
`memory/curation/engine.py:138`, `:157`;
`api/routes/curation.py:54` / `:63`.

Every writer to a given target opens the same `path + '.tmp'` with truncation.
Two open descriptors can therefore refer to the same temporary inode, write
at incompatible offsets, and rename a partial/mixed file onto the target.
Unique rename operations alone do not provide writer isolation.

**Reproduction:** two actual `atomic_write_json()` calls are scheduled with
thread Events at the JSON-write and rename boundaries. A writes half a large
payload; B opens/truncates the same temp file and writes a short payload; A
finishes and publishes. A returns successfully, B raises `FileNotFoundError`,
and the published target fails `json.loads()`. No production files are used.

**Reachability:** the curation scan route runs the same engine in worker
threads without an operation lock; scans and queue-changing API operations
can overlap. A timed-out `to_thread` await also does not stop its worker.
The helper race is experimentally confirmed; its production frequency was
not measured. Graph/profile/corpus also have hand-written temp-file writers;
do not claim they are repaired merely by changing this helper.

**Acceptance:** unique per-write temp files prevent inode sharing; appropriate
per-store synchronization protects read/modify/write state. Deterministically
overlap real calls and validate every published file. Serialize or reject
overlapping curation operations, including a timed-out scan still running.
Do not equate cancellation of an awaiter with termination of a worker.

### F04 — Search settings report success while search stays enabled

**Locations:** `gui/settings_core.py:173`,
`core/prompt/gatherer_web.py:27`, `:73`, `:114`, `:216`,
`core/prompt/context_gatherer.py:235`,
`knowledge/web_search_manager.py:604`, `:652`.

`apply_web_search()` updates the orchestrator config and `app_config` module.
The mixin reads its already-imported `WEB_SEARCH_ENABLED` binding, and the
instantiated rate limiter retains its original `daily_limit`. Neither live
consumer is updated by the settings operation.

**Reproduction:** start with search enabled and a real limiter set to 100.
Apply `enabled=False, daily_credit_limit=3`, then call the actual mixin search
method with a recording provider. The setter returns `ok=True`, global config
is false, yet one provider call occurs. The limiter remains 100 and admits a
four-credit request. Provider and trigger are synthetic; no query leaves the
machine. This proves a live setting-enforcement defect, not measured spending.

**Acceptance:** changing the setting on already-created objects suppresses
both heuristic and LLM-trigger paths before provider dispatch, and the next
credit check uses the new limit. Test enable→disable→enable, lazy creation
after a toggle, and existing manager reuse. Include agentic and enhanced
callers; updating only one imported constant is not a shared-contract fix.

### F05 — Chroma does not accept string range-comparison operands

**Locations:** `memory/storage/multi_collection_chroma_store.py:239`,
`memory/shutdown_processor.py:488`.

`get_ids_by_timestamp_range()` passes ISO strings to `$gte` and `$lte`.
Installed Chroma requires numeric operands for these comparisons. The broad
exception handler converts the driver validation error to an empty result,
so shutdown summary creation silently omits `source_doc_ids`.

**Reproduction:** store one conversation at `2026-09-09T12:00:00` and query
the containing day through the deployed store helper. It returns `[]` and
logs `Expected operand value to be an int or a float for operator $gte`.

**Consequence:** explicit summary→conversation linkage is broken. The
expander's timestamp-based fallback can still work, so this is **not** a
claim that every summary becomes impossible to expand.

**Acceptance:** a real-driver test returns the correct IDs at inclusive
boundaries and excludes adjacent rows. Choose a supported numeric timestamp
index or a bounded compatible legacy lookup; preserve legacy rows. Exercise
the summary writer through the helper and assert persisted source IDs.
Any migration/backfill remains dry-run-first and owner-applied.

### F06 — Calendar modifications do not invalidate prompt evidence

**Locations:** `core/actions/google_calendar_modify.py:185`, `:221`,
`core/actions/google_calendar.py:23`, `:47`;
contrast `core/actions/google_calendar_create.py:399`.

The calendar read cache survives successful update and delete operations.
Create invalidates it only in the all-success branch; a partially successful
batch also leaves it stale. A subsequent answer can therefore receive the
old schedule immediately after a truthful execution receipt.

**Reproduction:** seed a fresh cache with a synthetic meeting, execute the
actual delete function through `httpx.MockTransport` (GET then 204 DELETE),
then call actual `fetch_upcoming_events()`. Deletion succeeds but the deleted
meeting is returned without another GET. TTL is 300 seconds. Update and
partial-create are sibling source findings; the appendix executes deletion.

**Acceptance:** every successful mutation, including partial creation, makes
the next read reflect current state. Test failure-only batches separately.
Also inspect the existing cache's missing request dimensions: `max_events`
and `lookahead_days` are not part of a key. Do not conflate that sibling
issue with the experimentally verified post-delete stale result.

### F07 — Approval chaining ends at the browser boundary

**Locations:** `core/actions/types.py:109`, `api/routes/actions.py:33`,
`web/src/components/chat/ActionApprovalCard.tsx:21`, `web/src/App.tsx:239`,
`web/src/api/types.ts` (`ActionOutcome` omits the next-action fields).

The server returns `outcome.next_action_id` and updates its session to the
next proposal. The card forwards only `resp.message.content`; App appends
that line and unconditionally clears its pending action. Thus a multi-proposal
turn loses its second approval card in the browser, despite backend chaining.
A page reload can restore the server's next ID if the proposal has not expired.

**Reproduction:** transpile and execute the deployed card with a synthetic
approval response carrying `next_action_id='second'`. Execute the actual App
callback extracted from its TypeScript AST. Result: server next ID `second`,
browser pending ID `null`. This is a source-execution probe, not a browser
automation run. The backend-only chaining tests still pass.

**Acceptance:** consume the full outcome, carry the next ID/summary into
frontend state, and allow approval/rejection of every item in sequence.
Check the card's `busy` state when an ID changes; keeping the same component
mounted must not leave the next card disabled. Test final-item clearing and
both approval and rejection chains.

### F08 — Expansion bypasses quarantine for anchors and keeps stale results

**Locations:** `memory/memory_expander.py:76`, `:297`,
`core/agentic/tools.py:1891`, `core/agentic/controller.py:246`,
`core/orchestrator.py:876`.

Window expansion applies hygiene only to **non-anchor** documents. A requested
anchor is returned even when quarantined or superseded. Separately, the cache
returns previously rendered text before checking the store, with no version
or TTL. The controller/expander are retained across turns; no runtime caller
of `MemoryExpander.clear_cache()` was found in the controller/tool/handler
paths reviewed.

**Reproduction:** expand a quarantined fact using the actual expander and
real store wrapper. Its text is returned. Modify its text in the ephemeral
collection and expand again: old text is returned. Calling `clear_cache()`
manually reveals the changed text, still despite quarantine.

**Acceptance:** consistently enforce quarantine/currentness for anchors,
neighbors, summary fallback, unsupported-collection fallback, and cache hits.
Invalidate/version expansions after relevant mutations and bound cache
lifetime. Test an already-cached document that is corrected or quarantined.
An explicit user request to inspect quarantined data, if supported, needs a
separate clearly defined path; ordinary tool retrieval should honor suppression.

### F09 — `/api/graph` expects a different schema from GraphMemory

**Locations:** `api/routes/system.py:97`,
`memory/graph_memory.py:739` / `:753`, `memory/graph_models.py`.

The saved graph has `nodes` as an ID→attributes dict and edges with
`source_id`/`target_id`. The route treats nodes as dict items in a list with
an `id` field and expects edge `source`/`target` fields. Above its limit it
iterates node ID strings and calls `.get()` on them.

**Reproduction:** construct/save a real GraphMemory with two synthetic nodes
and one edge. Call the route with `limit=1`: `AttributeError: 'str' object has
no attribute 'get'` (an unhandled endpoint error). With a larger limit it
returns the original node dict, not the trimmed list schema it assumes.
Default limit is 300. No claim is made that the current frontend displays
this endpoint; it is documented as a stretch graph view.

**Acceptance:** convert the actual serialized node and relation schema at
the boundary. Test below/above the limit, empty graph, multiple relations
between a pair, and edge retention after trimming. Fixtures must be produced
by `GraphMemory.save()`, not hand-written in the route's assumed schema.

### F10 — Disabling query rewriting misses the initialized pipeline

**Locations:** `gui/settings_core.py:143`, `core/context_pipeline.py:315`,
`:471`, `core/orchestrator.py:454` / `:549`.

The settings setter changes `orchestrator.config['features']`. ContextPipeline
copied the value into `_enable_query_rewrite` during initialization and uses
that field to decide whether to run rewriting. The setter never refreshes it.

**Reproduction:** create an actual ContextPipeline with rewriting enabled and
a positive timeout. Apply the deployed streaming-settings function with
`disable_query_rewrite=True`. It returns `ok=True`; config becomes false;
the pipeline flag remains true. Tone persistence is stubbed to avoid reading
live session state. No model is called. The exact flag consumer is the
`run_rewrite` condition at line 471.

**Scope:** this is distinct from the already-fixed wrong keyword argument and
timeout fallback. A zero rewrite timeout can mask the toggle defect in the
default configuration. It affects an enabled pipeline or the reverse toggle
from a disabled one; the probe is not evidence of current unnecessary calls.

**Acceptance:** toggle an existing pipeline in both directions and drive
`build()` with recording dependencies. Confirm actual rewrite invocation
changes without reconstructing the orchestrator. Keep timeout=0 semantics
and existing rewrite-wiring tests intact.

### F11 — Curation success does not require durable recovery evidence

**Locations:** `memory/curation/engine.py:138`, `:348`,
`memory/curation/journal.py:41`, `memory/curation/adapters.py`.

Target mutation happens before queue persistence and the journal's pre-image
record. Both persistence functions swallow failures. The engine can return
`applied` after neither recovery record was saved, even though the target
store saved successfully. There is also no write-ahead recovery record for a
process crash between target mutation and those later writes.

**Reproduction:** use an actual UserProfile in a disposable writable path,
and make queue/journal paths unavailable by placing a regular file at their
parent path. Apply a profile supersession through the real engine/adapter.
Result: `applied`; the on-disk profile is changed; a newly constructed engine
has no proposal to undo. Fault conditions are deliberately injected; no
live disk-full/permission incident is claimed.

**Acceptance:** capture and durably prepare recovery evidence before changing
targets; fail before mutation when that cannot be done. Define recovery for
failures/crashes between prepare, target write, and completion. Exercise
separate target-write, queue-write, journal-write, and rollback failures with
actual disposable stores. A general best-effort activity log should not be
the only undo journal.

## PLANNED — for the repair session, not executed here

1. **Reconcile state first.** Read this file and the September 8 handoff, inspect
   current status, and preserve any additional owner edits. Reproduce relevant
   findings using the appendices before choosing an implementation. Do not
   assume this exact working tree is still deployed.
2. **Repair data integrity in bounded batches:** F01 metadata semantics;
   F02 staged replacement; F03 writer isolation and curation serialization;
   F11 durable recovery. Give each root cause its own acceptance tests and
   review boundary. F03/F11 share curation code and need one coordinated owner.
3. **Repair live controls:** F04 first, then F10. Exercise already-initialized
   objects and all consumers; config-dict assertions alone are insufficient.
4. **Repair state continuity:** F07 approval UI, F06 calendar invalidation,
   F08 expansion invalidation/hygiene. Recheck interaction between approval
   callbacks and an in-flight chat before accepting frontend changes.
5. **Repair schema boundaries:** F05 timestamp/source linking, F09 graph
   transport. Use installed-driver/writer-produced fixtures. Plan any store
   backfill separately; do not migrate as part of a unit-test run.
6. Run the pertinent suites plus the five repo-wide guards, Ruff and TypeScript.
   Suggested baseline commands, run as **separate batches** from repo root:

```bash
DAEMON_TEST_MODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q tests/unit/test_curation_engine.py tests/unit/test_api_curation.py tests/unit/test_calendar_update_delete.py tests/unit/test_safe_json.py tests/unit/test_no_git_state_in_tests.py tests/unit/test_ordered_slice_guard.py tests/unit/test_budget_meters_rendered_sections.py tests/unit/test_tool_wiring_parity.py tests/unit/test_model_capability_wiring.py
```

```bash
DAEMON_TEST_MODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q tests/unit/test_api_chat.py tests/unit/test_api_debug_settings.py tests/unit/test_memory_expander.py tests/unit/test_query_rewrite_wiring.py tests/unit/test_google_calendar.py tests/unit/test_calendar_create.py
```

```bash
ruff check . --output-format concise
npm --prefix web run typecheck
```

7. Prepare one reviewable repair batch at a time, with tests calling deployed
   functions and receipts recording what was actually tested. The owner commits
   before restart and performs live follow-up checks after deployment. A green
   unit batch alone does not establish the production fix.

## CONTINGENCY

| Condition | Next action |
| --- | --- |
| Reproduction no longer fails | Compare current source/dependency versions and inspect the responsible change; do not implement from stale line numbers |
| Installed Chroma semantics differ | Recheck the real driver; use neither an assumed API nor the existing replacing fake as authority |
| A repair needs historical store changes | Produce a bounded read-only report and dry-run plan with pre-images; owner applies with Daemon down |
| New test reaches network/model loading | Stop that probe, replace the transport boundary or use cached bounded dependencies; do not spend paid calls merely for coverage |
| Queue/journal/target failure leaves partial state | Preserve evidence and repair recovery semantics before exposing another apply/undo operation |
| Full suite is requested later | Confirm Daemon is down, use workflow memory caps and persistent output, and split batches; this audit did not run the non-unit suite |

### Remaining leads — not counted as confirmed findings

- **Calendar cache dimensions / pagination:** request parameters do not key
  the read cache. Both duplicate checking and modification resolution request
  at most 250 events without pagination, including omitting `nextPageToken`
  from requested fields. Test narrowed/widened windows and multiple pages
  before claiming uniqueness or completeness. No live calendar was inspected.
- **Chat/action overlap:** `/api/actions` appends history outside the chat
  stream lock; the card remains reachable while chat is streaming. Reproduce
  callback ordering through the real reducer before classifying user impact.
  Duplicate execution of the same pending proposal was not established:
  `store.approve()` changes its status synchronously before the executor await.
- **Upload lifetime:** successful uploads remain in the registry/temp directory
  without an evident bounded cleanup policy. Clarify intended lifetime and
  retained-source requirements before designing cleanup; this is not permission
  to delete uploads.
- **Temporal normalization:** `get_recent()` and expansion range comparisons
  can mix naive/aware datetimes; test real writer shapes and inspect bounded
  metadata samples before asserting that production contains the required mix.
- **Whole-request budget:** system text, attachment transport, schemas and
  rendered overhead remain a previously documented boundary. This is carried
  forward from the September 5 audit, not a newly reproduced defect here.

## WHY

The important recurring failure is disagreement at a boundary: fake vs real
database updates, JSON writer vs graph consumer, configuration setter vs cached
consumer, execution receipt vs cached evidence, and backend outcome vs browser
state. Existing green tests often validate each side separately. Add a small
number of contract tests that cross those boundaries, plus controlled failure
tests where losing stored content or recovery evidence is possible.

The confirmed findings do not depend on an LLM's wording or a speculative
reading of a private conversation. Each has a deployed-function reproduction
or, for F07, execution of actual frontend source with a synthetic response.
Broader model behavior, all tools, all stores, and historical corruption were
not exhaustively verified; this document is a broad deep pass, not a claim
that the repository is otherwise bug-free.

## OWNER

No owner action or relay is required to finish this audit. The deliverable is
this document. Repairs, commits, pushes, applying store changes, restarting
Daemon, and any live external-action probe remain for a later authorized
session and the owner-controlled workflow. No commit message is drafted for
unimplemented fixes.

## Test-gap review — handoff to Fable 5.1 high

**Added at the owner's request on September 9; investigation and documentation
only.** This extends the repair handoff with a prioritized test plan. It does
not add product findings to F01–F11, implement tests, or establish that every
uncovered scenario currently fails. Fable should close the confirmed blind
spots first, then investigate the explicitly marked coverage leads.

### Scope, evidence and confidence

The static inventory covered **440 Python test modules and 8,348 test-function
definitions** under `tests/`, using the configured filename patterns and
excluding `integration.bak`. These are source counts, including some ignored
files, not collected/parametrized cases or a coverage percentage. The review
traced the suites closest to all eleven findings, then sampled generation,
retrieval, prompt assembly, tools, settings, persistence, safety, frontend and
test selection. It did not manually inspect every assertion in all 440 files.

The inventory found 156 modules mentioning MagicMock/AsyncMock and 33 using
source inspection. Neither technique is itself a defect. Strict custom fakes
and recording transports already support useful tests. Real Chroma integration
tests also exist through `MultiCollectionChromaStore`; a direct-client-name
search alone would miss them. The problem is whether a test observes the
contract at risk and runs in a relevant validation lane.

A bounded **assertion-sensitivity probe** executed the actual bodies of eleven
tests in `tests/test_response_generator_comprehensive.py` with deliberately
broken synthetic dependencies. All eleven returned normally both on a wrong
output type and on a raised RuntimeError. The two streaming tests also
accepted an empty stream. Two additional cases showed that
`test_memory_relevance_scoring` accepted both no memories and a dictionary
without a score; the backup inventory test accepted an empty target list.
That is **27 undetected input cases across 13 existing test bodies**, not 27
new product bugs. The probe extracted functions using Python AST, removed
decorators, and supplied arguments; it did not run their original fixtures,
load models, import the application, or claim a complete pytest run.

A scan also identified 55 broad-exception-handler candidates for triage.
Some legitimately call `pytest.fail`, skip unavailable resources, or clean up
cancelled tasks. **Do not count all 55 as broken tests or mechanically remove
every handler.** The eleven generator tests above are confirmed cases where
the handler catches the test's own AssertionError.

The earlier 280 passing unit tests, lint and TypeScript results belong to the
original bug audit. This extension used source review and the bounded probe;
it did not rerun those batches or the full/non-unit suite.

Extension validation: the embedded assertion probe was extracted and executed
successfully, reproducing all 27 undetected cases. All 16 work-item headings
were checked. The original handoff text and reproduction appendices remain
unchanged apart from the added pointer and this section. Hash comparison over
the 1,020-file source/config/document manifest found only this handoff changed.

Evidence labels below:

- **Confirmed:** directly inspected test weakness or reproduced missed
  regression. This does not imply a new deployed product incident.
- **Scoped gap:** absent from the inspected nearby coverage; extend or locate
  an equivalent behavioral test before writing another one.
- **Lead:** a risk-based scenario to investigate, with no newly reproduced
  product failure.

### Finding-to-test map

Every original finding has a concrete coverage owner below. Preserve the
original reproduction and use its acceptance criteria as the primary oracle.

| Finding | Coverage work | Essential observation |
| --- | --- | --- |
| F01 | T03, T05 | Apply then undo against real Chroma; restored eligibility and metadata, including after reopening |
| F02 | T04 | Failed replacement preserves the original searchable chunks and content |
| F03 | T05 | Coordinated overlapping writers leave complete valid state; API overlap is exercised |
| F04 | T07 | Changing settings changes the next call from the already-running consumer |
| F05 | T03, T12 | Writer-produced timestamps yield the intended source IDs and usable summary linkage |
| F06 | T08 | Read, execute calendar mutation, then read again using the same cache |
| F07 | T09, T10 | Server-returned next action survives the real card callback and App state transition |
| F08 | T08 | Quarantined anchors and subsequently changed cached documents cannot leak stale evidence |
| F09 | T06 | GraphMemory output survives the graph API and the frontend's graph input boundary |
| F10 | T07 | The initialized pipeline stops rewriting after the setter reports success |
| F11 | T05 | Persistent mutation and durable undo evidence remain consistent through write failure/restart |

### T01 — Tests that cannot reject the broken result [first; confirmed]

**Evidence:** `tests/test_response_generator_comprehensive.py`, from
`test_generate_full_basic` through the judge/ensemble cases and both streaming
tests, encloses assertions in `except Exception: assert True`. The probe above
demonstrates the consequence. `tests/test_actual_caching.py` similarly catches
assertions and returns False; its separate integration test only imports
modules. A returned Boolean is not a pytest assertion.

**Work:** retain the real generator or cache under test; replace only provider
or model-loading boundaries with deterministic responses. Move expected
provider failures into explicit failure-path tests. Assert the promised answer,
selection/score, complete stream and terminal state. An environment prerequisite
may have a narrowly justified skip; an assertion failure must propagate.

**Acceptance:** wrong type/content, provider exception, empty stream where an
answer is required, and a second model load each fail the pertinent success
test. Expected-error tests assert the actual surfaced result and relevant
cleanup. Audit the other broad-exception candidates individually, recording
legitimate exceptions rather than creating a blanket ban.

### T02 — Reimplemented production logic in tests [first; confirmed]

**Evidence:** the four tests in `tests/unit/test_prompt_timeout.py` construct
tasks and reproduce the gather loop; `_make_builder` is defined but unused.
Changing the deployed builder to discard completed results would not change
those assertions. The golden distress replay in
`tests/integration/test_golden_distress_replay.py::_replay` calls real detector
and tracker functions, but recreates the intervening mapping/state wiring
itself. That replay alone cannot prove that the production caller uses the
same mapping.

**Work:** exercise `UnifiedPromptBuilder.build_prompt` with event-controlled
gatherers for partial timeout, one-section exception and cancellation. Add a
sequence through the deployed tone/context and post-response entry points.
Keep useful component tests while making their narrower scope explicit.

**Acceptance:** breaking the deployed gather retention, tone mapping or tracker
call makes the corresponding boundary test fail. Reuse
`test_independent_prompt_audit.py::test_cancelled_builder_drains_retrieval_before_resetting_shared_state`
and `test_request_path_parity.py` as examples of driving actual entry points.

### T03 — Real database contracts [high; confirmed, F01/F05]

**Evidence:** `tests/unit/test_curation_engine.py::FakeCollection.update`
replaces metadata; installed Chroma 1.0.7 merges it. Its apply/undo test therefore
misses F01. Memory expansion tests hand-author source links and fake store
results; they do not establish that the real timestamp query creates those
links. `tests/test_multi_collection_chroma_store.py` uses a real store but is
explicitly ignored in checked-in CI.

**Work:** add a tiny installed-driver contract batch, using disposable storage,
supplied numeric embeddings and the actual application wrappers. Cover metadata
merge/removal conventions, unchanged unrelated fields, document replacement,
include/result shapes, empty/missing IDs, and writer-produced timestamp filters.
Validate custom fakes against this contract before reusing them elsewhere.

**Acceptance:** F01 and F05 fail before repair and pass afterward; IDs/metadata
survive a reopen where persistence matters. No downloaded embedding model or
production Chroma directory is needed. Run this batch on driver upgrades as
well as changes to the storage wrappers.

### T04 — Replacement and ingestion failure points [high; confirmed, F02]

**Evidence:** `test_refdocs_lazy_collection.py` checks lazy access and unchanged
sync, while upload retrieval tests focus on retrieval/ranking. Searches and
the inspected suites did not locate direct failure-preservation tests for
`ReferenceDocsManager.upload_document` or `upload_text`. F02 provides the
actual missing regression.

**Work:** exercise both public ingestion methods with an original version and
a replacement. Inject chunking, embedding and insertion failures separately;
include partial insertion and retry. Cover success, identical-content retry,
changed-content retry, and different documents with similar names.

**Acceptance:** on failure the old document remains complete and retrievable;
on success the intended replacement is complete without duplicate active
versions. Verify content, IDs, hash and metadata through retrieval and reopening,
not only a success Boolean or a delete/insert call count. Establish the intended
atomic publication behavior before prescribing an implementation.

### T05 — Overlap, durable undo and interrupted writes [high; confirmed, F03/F11]

**Evidence:** `test_safe_json.py` covers sequential writes and serialization
failure; it does not reproduce competing writers. Curation's
`test_apply_is_all_or_nothing` fails on a missing later target and uses the
replacing fake. Queue round-trip tests do not cover mutation plus failed queue
and journal writes or recovery after an interrupted apply.

**Work:** use barriers/events to control two calls to the actual JSON writer
and two curation requests. Test target-write, queue-write, journal-write and
rollback failures independently and in the F11 combination. Reconstruct the
engine from disposable files after each relevant interruption; for a real
crash boundary use a tiny isolated subprocess, not a live Daemon.

**Acceptance:** published JSON is always a complete allowed version; unrelated
concurrent queue updates are not silently lost. No successful mutation lacks
the durable recovery state promised by the API. Interrupted batches have a
defined recoverable status, and undo restores the pre-image including the item
whose operation failed after partially changing it. Follow
`test_semantic_load_concurrency.py` for deterministic overlap; avoid sleeps
whose success depends on scheduler luck.

### T06 — Fixtures produced by the real writer [high; confirmed, F09]

**Evidence:** `test_api_misc.py::TestSystem.test_graph_trims_to_top_degree`
supplies a list of nodes and source/target edges. `GraphMemory.save` writes
a node dictionary and source_id/target_id edges. The fixture matches the
consumer's assumption while missing the deployed writer's schema.

**Work:** construct a small graph through GraphMemory, save it, then invoke
the actual API with limits below, equal to and above its size. Pass the
result through the frontend's graph consumption boundary. Apply this
writer-to-reader pattern to summary linkage and other persisted schema tests
when their fixtures are otherwise handwritten.

**Acceptance:** graph retrieval/trimming neither crashes nor drops/misidentifies
relations; selected edge endpoints refer to selected nodes. Cover empty graphs,
isolated nodes and degree ties. Malformed or older schemas are follow-up cases
whose support policy must be stated, not invented by the test.

### T07 — Runtime settings reach existing consumers [high; confirmed, F04/F10]

**Evidence:** `test_api_debug_settings.py` checks successful setters, config
dictionaries, YAML and module values. Those checks do not establish that the
already-created web consumer, credit limiter or ContextPipeline changed.

**Work:** initialize the actual consumer once, observe its behavior, call the
settings route, then use that same instance again. Build a compact inventory
of mutable settings and their consumers; extend to streaming, generation
parameters, synthesis and action controls where a cached copy exists.

**Acceptance:** disabling search prevents the next external transport call;
changed credit limits are enforced by the existing limiter; query rewriting
stops/resumes on the retained pipeline. Include invalid input, failed
persistence and restart semantics as specified by each setting. Restore test
state through fixtures; do not merely assert that a global flag changed.

### T08 — Cache behavior across mutations and turns [high; confirmed plus leads]

**Evidence:** calendar read tests clear the cache before/after each test and
exercise repeat reads; mutation tests separately check update/delete success.
This is valid isolation but misses F06's within-test read–mutate–read sequence.
`test_memory_expander.py` tests cache hits/manual clearing and quarantine of
neighbors, missing the quarantined anchor and retained stale cache in F08.

**Work:** retain the same calendar cache or controller/expander for the entire
sequence. Test calendar create/update/delete after a warm read; expansion after
quarantine, undo, edit and deletion of the anchor or a neighbor. Add explicit
cache-key tests for request dimensions and permission changes.

**Acceptance:** the next prompt/expansion reflects the mutation and eligibility
rules without a test manually clearing the cache. Correct read-cache isolation
between tests remains. Narrow/wide windows, small/large result limits,
pagination, failed mutations and permission revocation are **leads**, not newly
confirmed defects. `test_sep05_web_cache_same_query.py` has useful existing
key/TTL/mutation-isolation tests to emulate.

### T09 — Executed frontend behavior [high; confirmed, F07]

**Evidence:** `web/package.json` has dev/build/preview/typecheck scripts but no
test runner. No frontend behavior-test files were found, and the checked-in
GitHub workflow has no frontend steps. TypeScript passed while F07 remained.

**Work:** establish a small React component/state test lane. Drive the actual
ActionApprovalCard and App callback with a backend-shaped approval response
containing `next_action_id`; add one browser-level synthetic chat/approval
flow after the component test is stable. Use API-produced response fixtures
or a shared schema assertion to prevent independently invented contracts.

**Acceptance:** approving one action renders the next pending action and keeps
its ID available for the next approval. Dismiss/error/retry paths preserve the
intended state and never imply unexecuted success. Then add Stop/resubmit,
history restoration, clear failure, attachments and fenced-code rendering as
bounded cases. Network/provider behavior is simulated; no live action is
needed. Add behavior tests, typecheck and build to the appropriate CI lane.

### T10 — Request lifecycle and mode parity [medium; scoped gaps/leads]

**Evidence:** `test_api_chat.py`, `test_handle_submit.py` and
`test_request_path_parity.py` already exercise real handlers/routes with
mocked orchestration. The September 8 answer-integrity and recovery tests are
also valuable. They do not establish every combined browser, stream, action,
storage and cancellation lifecycle.

**Work:** extend these tests with a few representative complete request paths:
normal answer, agentic final answer, recovery, cancellation and an action-chain
continuation. Coordinate overlapping chat/action responses and late SSE events.
Assert at the actual storage, display and debug-output boundaries.

**Acceptance:** one intended turn produces the specified history/storage/debug
receipts, preserves code fences and final content, and runs post-response hooks
the intended number of times. Cancellation/exception releases admission state;
a late event cannot overwrite a newer turn or resurrect a dismissed action.
The concurrency scenarios remain **leads** until reproduced. Avoid a full
Cartesian product of every model, mode and failure.

### T11 — Registry entries through actual dispatch and transport [medium; scoped gap]

**Evidence:** `test_tool_wiring_parity.py` and
`test_model_capability_wiring.py` usefully check registry completeness and
protocol/classifier agreement, with some source-shape guards. These do not
alone prove argument forwarding or request payloads. Existing
`test_forced_top_p.py` and `test_disable_reasoning_wiring.py` already capture
real outgoing provider arguments; preserve and extend them.

**Work:** inventory tools/protocols against existing behavioral tests. For each
uncovered distinct route, pass a minimally valid schema example through the
actual XML/native parser and dispatcher into a strict recording executor.
At provider boundaries, cover representative capability combinations, aliases,
tool schemas/selection, streaming and response shapes.

**Acceptance:** dropping a required tool argument, using the wrong executor,
losing an explicit control, or sending the wrong resolved model fails a test
at the consumer boundary. Unknown tools and malformed/negated requests follow
the specified rejection behavior. Do not add one redundant test per registry
row when several rows use the same exercised path.

### T12 — Meaningful retrieval, provenance and total-budget assertions [medium]

**Evidence:** `test_integration_workflows.py::test_memory_relevance_scoring`
accepts empty results and scoreless dictionaries (confirmed by the probe).
Some other workflow checks assert only list/dict types. In contrast,
`test_hybrid_semantic_score.py`, `test_budget_meters_rendered_sections.py`
and `test_independent_prompt_audit.py` contain useful semantic-score and
actual rendered-budget checks.

**Work:** use tiny synthetic records with known IDs and distinct relevance.
Assert selected IDs, score propagation, eligibility, source links and the
actual rendered evidence. Add a summary-generation-to-expansion case for F05.
Extend existing budget tests to the final outgoing request where attachment,
system, schema and tool-result overhead can enter after section allocation.

**Acceptance:** returning no relevant evidence, erasing semantic scores,
including excluded evidence, dropping summary source IDs, or exceeding the
specified request ceiling makes the relevant test fail. Treat whole-request
budget and timestamp normalization cases as carried-forward/scoped work;
this extension did not reproduce another budget defect. Pin tokenizer and
normalization assumptions and test boundary values with Unicode/structured
content, without making model wording a brittle oracle.

### T13 — Backup completeness and recovery [high consequence; scoped gap]

**Evidence:** `test_backup_manager.py::test_backup_targets_includes_core_stores`
only asserts a list; an empty inventory passes. Other tests use real temporary
SQLite backups and injected backup targets, which is useful. A repository test
search found no references to `scripts/restore_backup.py` or
`scripts/export_user_data.py`; indirect coverage was not established.

**Work:** test the actual target enumerator with representative existing stores
under a temporary data root. Run backup → restore → reopen on tiny JSON/SQLite
fixtures through deployed commands/functions, with process-state detection
controlled in the test. Exercise export manifest/content separately.

**Acceptance:** all intended stores appear in the manifest and retain content
on recovery; dry-run writes nothing; restore enforces its Daemon-stopped guard;
interrupted restore follows documented recovery semantics. Export contains the
promised data and excludes secrets. Missing/corrupt members and destination
write failure are leads to test. Never point these tests at live stores or
invoke a live restore while trying to establish coverage.

### T14 — Make the right tests part of validation [first; confirmed]

**Evidence:** `.github/workflows/tests.yml` excludes slow, benchmark and
semantic markers and explicitly ignores 17 files, including the real Chroma
store, ContextPipeline, temporal retrieval, thread surfacing and profile-schema
suites. `pytest.ini` has four further ignores. The checked-in workflow has no
scheduled broader lane. This does not prove that the owner never runs these
tests manually.

The default `hooks/pre-push` selection is changed **test** files plus five
guards. A source-only change therefore does not automatically select its
subsystem tests. `PREPUSH_FULL=1` adds the unit selection with four ignores,
not the non-unit remainder. CI still provides a wider subsequent check.

**Work:** record every exclusion/marker lane with reason, responsible subsystem,
fast replacement coverage and a bounded command that actually runs the
remaining important cases. Add source/dependency-to-contract-test selection
where cheap; ensure new T03 and T09 tests are collected in CI. Produce actual
pass/skip/deselection receipts for each named lane.

**Acceptance:** a new boundary test cannot be present yet silently absent from
the advertised validation command. Source-only and dependency changes invoke
relevant checks. No unexplained safety/recovery skips. Do not simply remove
all ignores or force the full suite onto the 16 GB workstation alongside
Daemon; scheduled/offline resource-heavy validation is a separate lane.

### T15 — Test isolation across imports and retained state [setup; scoped gap]

**Evidence:** `tests/conftest.py` already redirects several mutable stores and
guards safety skips. `test_test_mode_store_guards.py` verifies selected
test-mode redirects, and `tests/test_eval/test_persistence_guard.py` tests
fingerprint detection. These are useful partial guarantees, not proof that
every late import, default path and external transport is isolated.

**Work:** before expanding integration tests, make their data roots explicit
and use strict transport stubs. Add small fresh-process checks that import
relevant store owners in different orders under test mode, then exercise the
actual default-path resolution. Reuse fixtures without shared mutable payloads;
run a bounded group in reversed order when globals/caches are involved.

**Acceptance:** new tests write only into their disposable roots, require no
credentials or downloaded models, and produce the same result alone and with
the selected neighboring tests. A newly initialized store cannot escape its
sandbox merely because it was imported after fixture setup. This review did
not prove an additional live-store write by the tests; runtime-file changes
noted earlier must not be attributed to this test review.

### T16 — Sequence behavior and semantic validation receipts [medium; scoped gap]

**Evidence:** safety canaries, the golden distress replay and many September
regressions already exist. The golden replay is explicitly embedder-gated;
the marker guard requires an ALLOW_SKIP reason but is not a record that every
critical semantic case actually executed in a validation run.

**Work:** maintain a small table separating deterministic wiring tests from
real-embedding calibration tests and connect it to T14's lane receipts.
Exercise retained state through actual pipeline entry points for ordinary
task continuation, escalating distress, recovery, correction and session reset.
Use synthetic messages; retain positive and negative cases when changing
detectors instead of adding only the latest incident phrase.

**Acceptance:** the wiring lane runs without external models and catches the
mapping/retained-state failures described in T02. The semantic lane reports
actual execution with a pinned available model, or an explicit unresolved
validation gap. Do not present a stubbed-embedding pass as evidence of semantic
calibration, or a justified skip as a passing calibration test. Follow existing
safety policy without copying private conversation fixtures into the repository.

### Execution order and definition of done

Fable 5.1 high should use the original repair priorities with the following
bounded sequence. These are implementation instructions for the later session,
not a request to repair source or tests in this documentation pass.

1. **Establish trustworthy checks:** T15 setup, T01 assertion repair, T02
   actual-entry-point tests, and a T14 selection ledger. Preserve existing
   useful mocks and guards; repair misleading tests in place where possible.
2. **Protect durable data:** T03–T05 against F01/F02/F03/F05/F11, plus the small
   T13 backup inventory/restore checks. Separate coupled curation changes from
   unrelated ingestion changes so each repair can be reviewed.
3. **Cover live boundaries:** T06–T09 and the relevant T10 lifecycle cases,
   closing F04/F06/F07/F08/F09/F10. Share deterministic fixtures only after
   their real-contract behavior is established.
4. **Fill the remaining risk-based matrix:** T11, T12, T16 and unresolved
   scoped leads. First identify equivalent existing tests; do not duplicate
   coverage because a different filename was expected.

For each completed work item, record: actual entry point; crossed boundary;
test node IDs; fixture/dependency version; a deliberately broken behavior
the test rejects; pass/fail/skip result; CI selection; and remaining limits.
For F01–F11, the original reproduction should fail under the new correct
assertion before repair and pass afterward. For other items, a temporary,
narrow perturbation in an isolated copy may establish sensitivity; an import
error or missing fixture does not count as detecting the intended regression.
Never weaken assertions or add unexplained xfails to obtain green results.

If a proposed test exposes an unrelated behavior dispute, record the expected
contract and evidence before calling it a product bug. Group root causes,
not one handoff item per assertion. After a repair batch, run only pertinent
tests plus required repo guards/checks, then expand testing when new failures
or unresolved concerns justify it. Keep non-unit/model-heavy batches under
the workflow's Daemon-down and memory limits.

**Completion for this extension:** the confirmed blind spots have owners,
acceptance criteria and a coverage map for all eleven findings. Implementation,
new test results and real-model validation remain future work. The static
inventory and targeted traces support a broad plan for the recurring failure
classes; they cannot establish that all or most individual test gaps in this
repository have been found. No weekly-credit balance or remaining token
allowance was visible to this reviewer.

### Reproduce the assertion-sensitivity result

Temporary evidence from this extension:

```text
/tmp/daemon_test_gap_inventory_20260909.json
/tmp/daemon_test_oracle_probe_20260909.py
/tmp/daemon_test_oracle_probe_20260909.json
/tmp/daemon_test_gap_before_20260909.json
/tmp/daemon_handoff_before_test_gap_20260909.md
```

The following standalone probe uses only Python's standard library and the
existing test source. It deliberately supplies invalid dependencies to inspect
the existing assertions; it is not a substitute for a regression test that
calls the deployed application. Run from the repository root.

<!-- test-oracle-probe -->
```python
"""Execute existing assertion bodies only; never import application/test fixtures."""
import ast
import asyncio
import json
from pathlib import Path

ROOT = Path.cwd()


def test_body(relative, name, **globals_):
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == name)
    node.decorator_list = []
    namespace = dict(globals_)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class BrokenGenerator:
    def __init__(self, mode):
        self.mode = mode

    def __getattr__(self, name):
        async def call(*args, **kwargs):
            if self.mode == 'raises':
                raise RuntimeError('synthetic provider failure')
            return None

        async def stream(*args, **kwargs):
            if self.mode == 'raises':
                raise RuntimeError('synthetic provider failure')
            if self.mode != 'empty':
                yield None

        return stream if name == 'generate_streaming_response' else call


class EmptyMemory:
    def __init__(self, rows):
        self.rows = rows

    async def store_interaction(self, *args, **kwargs):
        pass

    async def get_memories(self, *args, **kwargs):
        return self.rows


async def main():
    path = 'tests/test_response_generator_comprehensive.py'
    tree = ast.parse((ROOT / path).read_text())
    names = [n.name for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
             and n.name.startswith('test_')
             and any(isinstance(h, ast.ExceptHandler)
                     and isinstance(h.type, ast.Name) and h.type.id == 'Exception'
                     and any(isinstance(a, ast.Assert)
                             and isinstance(a.test, ast.Constant)
                             and a.test.value is True for a in h.body)
                     for h in ast.walk(n))]
    records = []
    for name in names:
        for mode in ('wrong_type', 'raises'):
            await test_body(path, name)(BrokenGenerator(mode))
            records.append({'test': name, 'broken_input': mode, 'body_returned_normally': True})
        if 'streaming' in name:
            await test_body(path, name)(BrokenGenerator('empty'))
            records.append({'test': name, 'broken_input': 'empty_stream', 'body_returned_normally': True})
    for rows in ([], [{}]):
        await test_body('tests/test_integration_workflows.py', 'test_memory_relevance_scoring')(EmptyMemory(rows))
        records.append({'test': 'test_memory_relevance_scoring', 'broken_input': rows, 'body_returned_normally': True})
    test_body('tests/unit/test_backup_manager.py', 'test_backup_targets_includes_core_stores',
              backup_targets=lambda: [])(None, None, None)
    records.append({'test': 'test_backup_targets_includes_core_stores', 'broken_input': [], 'body_returned_normally': True})
    assert len(names) == 11
    assert len(records) == 27
    print(json.dumps({'generator_test_bodies': len(names), 'undetected_cases': len(records), 'records': records}, indent=2))


asyncio.run(main())
```


## Reproduction appendices

Run from the repository root. The Python script imports repo code, then changes
to a disposable directory; its Chroma database is ephemeral, and its only HTTP
transport is mocked. The JavaScript script compiles existing local TypeScript
and mocks React hooks and transport. Neither script starts the application.

The audit assertions intentionally describe the current bugs. They should
fail when those behaviors have been repaired.

To extract and run the embedded scripts without copying long commands manually:

```bash
python - <<'PY'
from pathlib import Path
doc = Path('docs/HANDOFF_20260909_independent_bug_audit.md').read_text()
for marker, name in [('audit-python', '/tmp/daemon_audit_20260909_probes.py'),
                     ('audit-javascript', '/tmp/daemon_audit_20260909_frontend.cjs')]:
    start = doc.index('<!-- ' + marker + ' -->')
    block = doc.index('```', start)
    body = doc.index('\n', block) + 1
    end = doc.index('\n```', body)
    Path(name).write_text(doc[body:end] + '\n')
PY
```

```bash
DAEMON_TEST_MODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python /tmp/daemon_audit_20260909_probes.py
node /tmp/daemon_audit_20260909_frontend.cjs
```

<!-- audit-python -->

```python
"""Read-only repository audit; all mutations use a disposable directory/DB.

Run from repository root with DAEMON_TEST_MODE=1 and offline model flags.
Assertions record the observed bugs, not desired post-fix behavior.
"""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import patch

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
os.environ.setdefault("DAEMON_TEST_MODE", "1")
import config.app_config as cfg


def emit(finding, **details):
    print(json.dumps({"finding": finding, **details}, default=str), flush=True)


def proposal(pid, items):
    from memory.curation.types import CurationProposal
    return CurationProposal(proposal_id=pid, curator="synthetic",
                            instrument="metadata", confidence="deterministic",
                            title="Synthetic proposal", evidence="Synthetic input",
                            items=items)


def engine_at(root, store):
    from memory.curation.engine import CurationEngine, StoreBundle
    from memory.curation.journal import CurationJournal
    return CurationEngine(StoreBundle(chroma_store=store),
                          queue_path=str(root / "queue.json"),
                          journal=CurationJournal(str(root / "journal.jsonl")))


def db_probes(root):
    import chromadb
    from chromadb.config import Settings
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
    from memory.curation.types import ItemChange
    from memory.memory_expander import MemoryExpander
    from knowledge.reference_docs_manager import ReferenceDocsManager

    client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
    coll = client.create_collection("independent_audit", embedding_function=None)
    coll.add(ids=["fact"], documents=["Synthetic durable fact"],
             embeddings=[[1.0, 0.0]], metadatas=[{"source": "synthetic",
             "timestamp": "2026-09-09T12:00:00"}])
    store = MultiCollectionChromaStore.__new__(MultiCollectionChromaStore)
    store.collections = {"facts": coll, "conversations": coll, "reference_docs": coll}
    engine = engine_at(root, store)
    p = proposal("undo", [ItemChange(store="chroma:facts", doc_id="fact",
                                     change_type="quarantine")])
    engine._proposals[p.proposal_id] = p
    engine.apply(p.proposal_id)
    engine.undo(p.proposal_id)
    metadata = coll.get(ids=["fact"])["metadatas"][0]
    assert p.status.value == "undone" and metadata["curation_quarantined"] is True
    emit("F01", chromadb=chromadb.__version__, status=p.status.value,
         quarantine_after_undo=metadata["curation_quarantined"])

    ids = store.get_ids_by_timestamp_range("conversations",
                                          "2026-09-09T00:00:00", "2026-09-10T00:00:00")
    assert ids == [] and coll.count() == 1
    emit("F05", matching_rows=coll.count(), returned_ids=ids)

    expander = MemoryExpander(store)
    first = expander.expand("fact", collection="facts")
    assert first["turns"][0]["content"] == "Synthetic durable fact"
    coll.update(ids=["fact"], documents=["Synthetic corrected fact"], embeddings=[[1., 0.]])
    cached = expander.expand("fact", collection="facts")
    assert cached["turns"][0]["content"] == "Synthetic durable fact"
    expander.clear_cache()
    fresh = expander.expand("fact", collection="facts")
    assert fresh["turns"][0]["content"] == "Synthetic corrected fact"
    emit("F08", quarantined_anchor_returned=True,
         cached_content=cached["turns"][0]["content"],
         after_manual_cache_clear=fresh["turns"][0]["content"])

    coll.add(ids=["old_upload"], documents=["Original synthetic upload"],
             embeddings=[[0., 1.]], metadatas=[{"title": "Synthetic worksheet",
                                              "type": "user_upload", "chunk_index": 0}])
    manager = ReferenceDocsManager(store)
    with patch.object(store, "add_batch_to_collection", side_effect=RuntimeError("synthetic embedding failure")):
        result = manager.upload_text("Replacement synthetic text", "Synthetic worksheet")
    remaining = coll.get(ids=["old_upload"])["ids"]
    assert not result.success and remaining == []
    emit("F02", success=result.success, original_upload_ids_after_failure=remaining,
         error=result.errors)


def atomic_overlap(root):
    from utils import safe_json
    target = str(root / "atomic.json")
    Path(target).write_text('{"original": true}')
    first_half = threading.Event()
    second_written = threading.Event()
    published = threading.Event()
    dump = json.dump
    replace = os.replace
    outcomes = {}

    def scheduled_dump(data, stream, **kwargs):
        if data.get("writer") == "A":
            value = json.dumps(data)
            mid = len(value) // 2
            stream.write(value[:mid])
            stream.flush()
            first_half.set()
            assert second_written.wait(5)
            stream.write(value[mid:])
        else:
            dump(data, stream, **kwargs)
            stream.flush()
            second_written.set()
            assert published.wait(5)

    def scheduled_replace(src, dst):
        replace(src, dst)
        published.set()

    def writer(label, data):
        try:
            safe_json.atomic_write_json(target, data)
            outcomes[label] = "returned successfully"
        except Exception as exc:
            outcomes[label] = type(exc).__name__

    with patch.object(safe_json.json, "dump", scheduled_dump), patch.object(safe_json.os, "replace", scheduled_replace):
        a = threading.Thread(target=writer, args=("A", {"writer": "A", "text": "x" * 400}))
        b = threading.Thread(target=writer, args=("B", {"writer": "B"}))
        a.start()
        assert first_half.wait(5)
        b.start()
        a.join(10)
        b.join(10)
        assert not a.is_alive() and not b.is_alive()
    try:
        json.loads(Path(target).read_text())
        valid = True
    except ValueError:
        valid = False
    assert outcomes["A"] == "returned successfully" and not valid
    emit("F03", writer_outcomes=outcomes, published_json_valid=valid)


async def runtime_settings(root):
    from gui.settings_core import apply_streaming, apply_web_search
    from core.context_pipeline import ContextPipeline
    from core.prompt import gatherer_web
    from knowledge.web_search_manager import WebSearchRateLimiter
    calls = []
    limiter = WebSearchRateLimiter(daily_limit=100, state_file=str(root / "credits.json"))

    async def search(**kwargs):
        calls.append(kwargs["query"])
        return NS(has_results=False, error="synthetic provider; no network")

    decision = NS(should_search=True, depth=NS(value="quick"), confidence=1.0,
                  reason="synthetic", search_terms=[])
    manager = NS(is_available=lambda: True, multi_search=search, rate_limiter=limiter)
    gatherer = NS(web_search_manager=manager, web_search_trigger_llm=None,
                  web_search_trigger=lambda q: decision, model_manager=None, memory_id_map={})
    with patch.object(ContextPipeline, "_load_persisted_tone", return_value=None):
        pipeline = ContextPipeline(model_manager=None, topic_manager=None,
                                   config={"enable_query_rewrite": True, "REWRITE_TIMEOUT_S": 1})
    orch = NS(config={"features": {}, "web_search": {}}, context_pipeline=pipeline,
              prompt_builder=NS(context_gatherer=gatherer))
    with patch.object(gatherer_web, "WEB_SEARCH_ENABLED", True), \
         patch.object(cfg, "WEB_SEARCH_ENABLED", True), \
         patch.object(cfg, "WEB_SEARCH_DAILY_CREDIT_LIMIT", 100):
        response = apply_web_search(orch, enabled=False, daily_credit_limit=3,
                                    save=lambda updater: (True, None))
        await gatherer_web.WebSearchMixin._get_web_search_results(gatherer, "Synthetic current query")
        assert response["ok"] and not cfg.WEB_SEARCH_ENABLED and len(calls) == 1
        assert limiter.daily_limit == 100 and limiter.can_search(4)
        emit("F04", reported_ok=response["ok"], config_enabled=cfg.WEB_SEARCH_ENABLED,
             provider_calls_after_disable=len(calls), requested_limit=3,
             actual_limit=limiter.daily_limit)
    response = apply_streaming(orch, disable_best_of=True, disable_query_rewrite=True,
                               disable_llm_summaries=True, best_of_latency_budget_s=0,
                               save=lambda updater: (True, None))
    assert response["ok"] and pipeline._enable_query_rewrite is True
    emit("F10", reported_ok=response["ok"], config_rewrite=orch.config["features"]["enable_query_rewrite"],
         pipeline_rewrite=pipeline._enable_query_rewrite)


async def calendar_cache():
    import httpx
    from core.actions import google_calendar as cal, google_calendar_modify as modify
    from core.actions.types import ActionProposal
    old = {"summary": "Synthetic meeting", "start": "2026-09-10T10:00:00-05:00",
           "end": "2026-09-10T11:00:00-05:00", "all_day": False, "location": ""}
    real_client = httpx.AsyncClient
    requests = []
    def handler(request):
        requests.append(request.method)
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json={"id": "synthetic-event", "summary": "Synthetic meeting",
                                       "start": {"dateTime": old["start"]}})
    def client(*args, **kwargs):
        return real_client(transport=httpx.MockTransport(handler))
    p = ActionProposal(action_type="calendar_delete_event", params={"event_id": "synthetic-event"})
    with patch.object(cal, "_cache", [old]), patch.object(cal, "_cache_ts", time.time()), \
         patch.object(modify, "_prereq_error", return_value=(NS(token="synthetic-token"), None)), \
         patch.object(httpx, "AsyncClient", client):
        result = await modify.delete_calendar_event(p)
        subsequent = await cal.fetch_upcoming_events()
        assert result.success and subsequent == [old]
        emit("F06", delete_success=result.success, subsequent_prompt_events=subsequent,
             requests=requests)


async def graph_route(root):
    from memory.graph_memory import GraphMemory
    from memory.graph_models import GraphNode, GraphEdge
    from api.routes.system import graph
    path = root / "graph.json"
    gm = GraphMemory(str(path))
    gm.add_entity(GraphNode(entity_id="synthetic_a", display_name="Synthetic A"))
    gm.add_entity(GraphNode(entity_id="synthetic_b", display_name="Synthetic B"))
    gm.add_relation(GraphEdge(source_id="synthetic_a", relation="knows", target_id="synthetic_b"))
    gm.save()
    with patch.object(cfg, "KNOWLEDGE_GRAPH_PERSIST_PATH", str(path)):
        result = await graph(NS(), limit=10)
        try:
            await graph(NS(), limit=1)
            error = None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    assert isinstance(result["nodes"], dict) and error and "AttributeError" in error
    emit("F09", untrimmed_node_type=type(result["nodes"]).__name__, trim_error=error,
         edge_keys=list(result["edges"][0]))


def journal_loss(root):
    from memory.curation.types import ItemChange
    from memory.curation.engine import CurationEngine, StoreBundle
    from memory.curation.journal import CurationJournal
    from memory.user_profile import UserProfile
    # A regular file used as a directory deterministically simulates unavailable
    # queue/journal storage while a separate target store remains writable.
    blocker = root / "not-a-directory"
    blocker.write_text("synthetic")
    fact = {"fact_id": "f", "is_current": True}
    profile = UserProfile(str(root / "synthetic_profile.json"))
    profile.profile["categories"]["career"] = [fact]
    profile.save()
    engine = CurationEngine(StoreBundle(user_profile=profile),
                           queue_path=str(blocker / "queue.json"),
                           journal=CurationJournal(str(blocker / "journal.jsonl")))
    p = proposal("durability", [ItemChange(store="profile", doc_id="f",
                     change_type="supersede_profile_fact", after={"reason": "synthetic"})])
    engine._proposals[p.proposal_id] = p
    result = engine.apply(p.proposal_id)
    restarted = CurationEngine(engine.stores, queue_path=engine.queue_path, journal=engine.journal)
    assert result.status.value == "applied" and fact["is_current"] is False
    persisted = json.loads(Path(profile.profile_path).read_text())
    assert persisted["categories"]["career"][0]["is_current"] is False
    assert restarted.get(p.proposal_id) is None
    emit("F11", returned_status=result.status.value, target_persistently_changed=True,
         undo_available_after_restart=restarted.get(p.proposal_id) is not None)


async def main():
    with tempfile.TemporaryDirectory(prefix="daemon-audit-sep09-") as directory:
        root = Path(directory)
        os.chdir(root)  # Any incidental relative test files stay disposable.
        for name, function in [("db", lambda: db_probes(root)),
                               ("atomic", lambda: atomic_overlap(root)),
                               ("settings", lambda: runtime_settings(root)),
                               ("calendar", calendar_cache),
                               ("graph", lambda: graph_route(root)),
                               ("durability", lambda: journal_loss(root))]:
            try:
                result = function()
                if hasattr(result, "__await__"):
                    await result
            except Exception as exc:
                emit("PROBE_ERROR", probe=name, error=f"{type(exc).__name__}: {exc}")
                raise


if __name__ == "__main__":
    asyncio.run(main())
```

<!-- audit-javascript -->

```javascript
// Execute the deployed card and App callback with synthetic transport/hooks.
// No browser, network, application server, or source edits.
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const root = process.cwd();
const ts = require(path.join(root, 'web/node_modules/typescript'));
const jsx = (type, props) => ({ type, props });
const response = {
  message: { role: 'assistant', content: 'Synthetic action completed' },
  outcome: { status: 'executed', next_action_id: 'second', next_summary: 'Next synthetic action' },
};
const cardFile = path.join(root, 'web/src/components/chat/ActionApprovalCard.tsx');
const code = ts.transpileModule(fs.readFileSync(cardFile, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
}).outputText;
const exportsObject = {};
vm.runInNewContext(code, {
  exports: exportsObject,
  require(name) {
    if (name === 'react') return { useState: () => [null, () => {}] };
    if (name === 'react/jsx-runtime') return { jsx, jsxs: jsx };
    if (name === '@mantine/core') return { Button: 'Button', Card: 'Card', Group: 'Group', Text: 'Text' };
    if (name === '@mantine/notifications') return { notifications: { show() {} } };
    if (name === '../../api/client') return { api: { approveAction: async () => response } };
    throw Error(`Unexpected import ${name}`);
  },
});
const appFile = path.join(root, 'web/src/App.tsx');
const source = ts.createSourceFile(appFile, fs.readFileSync(appFile, 'utf8'),
                                   ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
let callbackSource;
function visit(node) {
  if (ts.isJsxAttribute(node) && node.name.getText(source) === 'onActionDecided') {
    callbackSource = node.initializer.expression.getText(source);
  }
  ts.forEachChild(node, visit);
}
visit(source);
if (!callbackSource) throw Error('App callback not found');
let pending = 'first';
const messages = [];
const callback = vm.runInNewContext(`(${callbackSource})`, {
  chat: { appendAssistant: line => messages.push(line), clearPendingAction: () => { pending = null; } },
});
const tree = exportsObject.default({ actionId: 'first', onDecided: callback });
function button(node) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'Button' && node.props.children === 'Approve') return node;
  return [].concat(node.props?.children || []).map(button).find(Boolean);
}
button(tree).props.onClick().then(() => {
  if (pending !== null || messages.length !== 1) throw Error('Unexpected result');
  console.log(JSON.stringify({ finding: 'F07', server_next_action_id: response.outcome.next_action_id,
                              browser_pending_action_id: pending, messages }));
});
```
