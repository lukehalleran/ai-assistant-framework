# Failure-versus-empty outcome design — CGR-20260913-007 to -010

Status: parent design decision (generalization lane). Inputs come from a read-only survey on 2026-09-13 at base `328a8ec`; the 17 source SHA-256s match the request packets. This is a design, not an implementation. Every batch below still follows the plan's Ready → failing proof → patch → review → integrate sequence. Batches touching a shared seam get explicit parent sign-off at Ready.

## Real defects behind the scanner findings (priority inputs)

| Defect | Evidence | Request |
|---|---|---|
| A failed snapshot read during an upload's staged replacement means old chunks are never replaced; stale and new versions are both retrievable | `knowledge/reference_docs_manager.py:237,299-300,340,393-394` → `_get_document_chunks` (484-509) | 008 |
| The narrative status guard silently turns off on a read failure; the narrative is generated and persisted without it | `memory/memory_consolidator.py:204-216,704-705,757`; duplicate at `utils/daily_notes_generator.py:440-452` (sibling, not in packet) | 009 |
| Consolidator input readers that fail render as "(No … available)"; the narrative is built from partial inputs | `memory/memory_consolidator.py:455-618,654-699` | 009 |
| Range expansion failure falls through to the wrong error reason and is cached | `memory/memory_expander.py:318-328,348-382,149-151` | 009 |
| Shutdown thread pass: an empty open-thread list (failed read) skips resolution and de-dup, so duplicate threads are stored | `memory/shutdown_processor.py:1543-1556` ← `ThreadStore.list_open_threads` (259-277) | 009 |
| Synthesis `find_similar` failure stores a duplicate instead of updating convergence | `memory/synthesis_memory.py:99,107-115` | 009 |
| A failed vault search prints `obsidian=OFF` ("feature off") | `core/prompt/formatter.py:793` | 007 |
| Tavily provider errors become `pages=[], error=None`, which renders as "ON(0 results)" | `knowledge/web_search_manager.py:1451-1458,1460-1538` | 008 |
| Write failures and deliberate skips both return None; "CRITICAL … data loss" logging never fires; five `except: pass` sites | `memory/memory_storage.py:790-1024`; `core/orchestrator.py:2169,2300,2477`; `gui/handlers.py:196,1698,2590,3388` | 010 |

## Precedents to reuse (CM-05)

- `FetchedPages(list)` with `blocked=` (`knowledge/web_search_manager.py:234-245`): a typed empty that still equals `[]`.
- `last_web_decision` (`core/prompt/gatherer_web.py:96-108,301-304`) → `web_search_decision` context key (unrendered, `core/prompt/token_manager.py:110-113`) → prompt label `ON(error | not triggered | N results)` (`formatter.py:799-812`) → turn record `web_error` (`core/orchestrator.py:1682-1735`).
- `grounding_status` / `grounding_failure_reason` (`core/grounding_check.py:760-780`), carried into the debug record by `_capture_delivery` (`gui/handlers.py:3068`).
- `_task_timings` underscore context key (`core/prompt/builder.py:2099-2102`) → popped by the orchestrator (1873) → debug record and telemetry (`gui/handlers.py:3065-3067,4484`). This is the channel for per-section outcomes.
- The expander result `error` field (`memory/memory_expander.py:295-301`).

## Decisions per request

- **CGR-008 (knowledge managers)**
  - Obsidian `get_notes`/keyword leg, reference docs `get_documents`/`_keyword_search`/`_get_document_chunks`/`list_documents`, `SemanticSearchIndex.search`, `VisualMemoryStore.search_by_text`: return a `FetchedPages`-style list subclass carrying `status` (`ok | empty | failed`) and `reason`. Callers read it immediately (slicing drops attributes).
  - `_tavily_search` / `_tavily_extract`: raise a typed error. All callers are in the same file, and `search()` already converts raises (wsm:1348-1353); keep the invalid-key flag and credit accounting.
  - Upload staged replacement must REFUSE (explicit failure, previous version intact) when the snapshot read failed.
  - Callers without a try that must be checked when migrating: `knowledge/doc_cooccurrence.py:87`, `knowledge/visual_retrieval.py:80`.
- **CGR-007 (gatherers)**
  - Gatherers stop swallowing; failures propagate into the builder's gather loop (`builder.py:1261-1550`). The loop records a per-section outcome `ok | empty | failed | timed_out` under an underscore context key following `_task_timings`, and registers it in `UNRENDERED_CONTEXT_KEYS`.
  - The formatter renders "could not check <section>" inside `[ACTIVE FEATURES]`, with no new header, and fixes `obsidian=OFF` on failure.
  - The orchestrator publishes the outcomes in the turn record.
  - Off-path sites #85, #86 and #89 are answered as not on the prompt path; they get evidence, not code.
- **CGR-009 (memory reads)**
  - Raise on failure (not-found stays None/`[]`): `MultiCollectionChromaStore.get_by_id`, `ProposalStore.get_proposal`, the consolidator readers and `_current_status_facts`, `_fetch_conversations_in_range`, `ThreadStore.list_open_threads`.
  - Each owner degrades explicitly:
    - consolidation marks `status_guard=unavailable` and does not persist a guard-less narrative as authoritative;
    - expansion returns an explicit error and never caches a failure;
    - the shutdown thread pass skips thread extraction when open threads cannot be read.
  - Typed lists for the synthesis, retriever and `query_*` reads.
- **CGR-010 (memory writes)**
  - A failure raises; None remains the documented deliberate skip.
  - Callers record a failed write (turn-record receipt) instead of `except: pass`.
  - A result object is rejected: it is always truthy, so `if doc_id:` counts would treat failures as stored.

## Batch order (≤450 lines each; disjoint files; producer before consumer)

| # | Scope | Shared seam (parent sign-off) |
|---|---|---|
| F1 | Outcome vocabulary: leaf `utils/retrieval_outcome.py` (typed error + list subclass) + tests | receipt vocabulary |
| F2 | Tavily wrappers (wsm) | — |
| F3a | Reference docs (#103-#106) + upload refusal on a failed snapshot read; response `CGR-20260913-008-2` (parent split 2026-09-14 for the size cap) | — |
| F3b | Obsidian notes (#101-#102); response `CGR-20260913-008-3` | — |
| F4 | FAISS semantic search + visual memory store, and their no-try callers | — |
| F5 | Builder gather loop records per-section outcomes; `UNRENDERED_CONTEXT_KEYS` | builder / budget registry |
| F6a | Formatter `[ACTIVE FEATURES]` "could not check" label + `obsidian=` driven by `OBSIDIAN_ENABLED` (parent split 2026-09-14) | formatter |
| F6b | Receipts: orchestrator pops `_section_outcomes` → debug_info / `_last_section_outcomes` / turn signal `sections_not_checked`; handlers debug record + `_capture_delivery` | orchestrator, handlers, telemetry |
| F7a/F7b | Knowledge gatherer sites (gk), two halves | — |
| F8 | Memory gatherer (gm) + web gatherer (gw #92 evidence) | — |
| F9 | `get_by_id` + expander + claim tracker | store reader |
| F10 | Turn and skill writes: raise on failure, skips stay None | store writer |
| F11a | Thread writes + shutdown thread pass | store writer |
| F11b | Proposal reads/writes + consumers (after F11a; shared shutdown_processor) | store writer |
| F12 | Consolidator status guard + readers; retriever; synthesis (+ daily_notes_generator sibling) | — |
| F13 | Write-failure receipt in the turn record; replace the five `except: pass` sites | orchestrator, handlers |

Sequencing against the plan: F1–F4 can run as soon as the small request batches (R02–R06) finish. F5/F6 share the orchestrator with A05/A07 and must not overlap them. F10–F13 touch store writers and the orchestrator; schedule them after A05 (atomic correction delivery), which owns gui/handlers.py.

## Parent review amendments (2026-09-14)

Source: a read-only flow map of all 59 anchors and their callers, on the post-A05c tree. Every source file still matches its request packet's SHA-256. Each item is marked **[verified]**, meaning the parent read the lines, or **[map]**, meaning agent-reported and to be re-verified at that batch's Ready. The decisions above stand; these add defects, fix the batch table, and flag one vocabulary decision.

### New real defects (priority inputs)

- **[verified] Delete-then-store loses the record.** These are not scanner anchors. Each batch fixes them and records them as BC-58 siblings in its response.
  - **Where:**
    - `ProposalStore.update_status` (`memory/proposal_store.py:206`) deletes at 264, calls `store_proposal` at 267 without checking the result, and returns True at 270.
    - `ThreadStore.resolve_thread` (`memory/thread_store.py:371`) deletes the document, calls `store_thread` without checking, and returns True (≈396-399).
    - `ThreadStore._update_thread` (461) does the same (≈482-484).
  - **Effect:** today a failed re-store silently loses the record while the caller sees success. Once CGR-010 makes write failures raise, the raise happens after the delete, and the record is still lost.
  - **Remedy (CM-07 staged replacement):** write the new version first, or upsert, and delete the old one only after a confirmed write. Report failure.
  - **Batches:** threads in F11a, proposals in F11b.
- **[verified] The builder's gather loop turns failures into `[]`.**
  - **Where:** `core/prompt/builder.py:1534-1556`, `gathered[name] = task.result() or []`.
  - **Effect:** an exception becomes `[]` with only a warning log. A pending task at the deadline is cancelled and becomes `[]`, and its name is only logged. The `or []` also turns a legitimate `""`/None return into `[]`.
  - **F5:** records its per-section outcome in this loop as designed. The pending-task branch is the `timed_out` source.
- **[verified] `get_narrative_context` (anchor #81) runs outside the gather loop.**
  - **Where:** builder 1064-1073, with its own try that leaves `narrative_state = ""` after a debug log.
  - **F5:** must record this section's outcome separately.
- **[verified, F4 parent review] A total row-read failure reads as `no_results`.**
  - **Where:** `SemanticSearchIndex._read_rows` (`knowledge/semantic_search.py`) catches each row-group read error, logs a warning and continues.
  - **Effect:**
    - When FAISS returns hits but no row can be read, `search` returns `OutcomeList()` (`no_results`), and `doc_cooccurrence` scores `known=False`.
    - A partial read failure returns fewer rows, labelled `succeeded`.
  - **Why the scanner misses it:** the site is `except → continue`, not dm18's `except → return []`. It predates F4 and was outside F4's brief.
  - **Batch:** F7a, where the `semantic` gatherer task starts reading this status. Return `failed("row_read_failed")` when there were hits but none could be read. In the same edit, fix `search`'s docstring, which wrongly says a not-loaded index is `no_results`.
- **[map] Omitting a section also drops its guidance.** Per-section instructions are added only when the section has content (`core/prompt/section_instructions.py:108`). F6 decides what a "could not check" section gets. Default: no content instructions, only the `[ACTIVE FEATURES]` label.
- **[map] The debug record has no section-status field.** `_build_debug_record` (`gui/handlers.py` ≈628-697) has none, and `_capture_delivery` copies only timings and `grounding_*` keys. F6 publishes section outcomes to the debug record as well as the turn record, following the `task_timings` path.
- **[map] The expander caches its wrong error.** When `_fetch_conversations_in_range` returns `[]`, `_expand_summary` falls through to "no source_doc_ids or temporal anchors". `expand()` then caches it (≈149-151). The design already names this; the map confirms the cache step.

### Batch table corrections

- **[verified] F9 and F10 share a file.** Both edit `memory/storage/multi_collection_chroma_store.py`: `get_by_id` at 489 (F9) and `add_conversation_memory` at 520 (F10). The table's "disjoint files" claim is wrong for this file. They run sequentially, F9 then F10, and F10's manifest includes F9's state.
- **[verified] Anchor #130 had no batch.** `MemoryStorage._get_recent_summaries_by_timespan` (≈1635) is not in the table. Assign it to **F10**, which owns `memory/memory_storage.py`. Its consumer `_maybe_regenerate_narrative` (≈1564-1569) is in the same file and must degrade explicitly: never regenerate a narrative from a failed read. F12 handles the consolidator side.
- **[map] More sequential seams.** Each batch's ownership must list the exact regions it touches in these files:
  - `memory/shutdown_processor.py`, in order:
    - F10: the `store_skill` count at ≈1161;
    - F11a: thread pass ≈1545 and ≈1629;
    - F11b: ≈1253 and ≈1466;
    - F12: `get_skills` proposal context at ≈1312, if touched.
  - `memory/memory_coordinator.py` pass-throughs: `store_interaction` ≈369, `list_open_threads` ≈386, `store_skill` ≈576, `get_skills` ≈583, `get_unresolved_threads` (#121) ≈585-600.
  - `core/orchestrator.py`: F6 (turn-record key), then F13 (write-failure receipt).
- **[map] Two consumers outside CGR-007's list.** Both consume F4's producers; F7 carries outcomes through them and records them as BC-58 siblings.
  - `_get_semantic_chunks_timed` (gatherer_knowledge ≈1859, builder task "semantic") consumes `SemanticSearchIndex.search` through `semantic_search_with_neighbors`.
  - `get_visual_memories` (gatherer_knowledge ≈1504) consumes `VisualMemoryStore.search_by_text` through `knowledge/visual_retrieval.py:80`.

### F7 split and gatherer outcome shape (parent decision, 2026-09-14)

- **[verified] No knowledge gatherer reads a producer's status today.** `core/prompt/gatherer_knowledge.py` does not import `utils.retrieval_outcome`, so each gatherer rebuilds or slices the list before returning and the F3a/F3b/F4 status is lost:
  - notes: call at 642, rebuilt at 658/674/689, returned at 717;
  - reference docs: call at 741, rebuilt at 744/759/765, returned at 785;
  - uploads: call at 897, rebuilt at 902/931/940/942, returned at 994;
  - semantic chunks: call at 1859, only a falsy check at 1881, then the filter at 1886.

  Visual memories is the only pass-through: a dict whose F4 keys survive.
- **Outcome shape: a typed return, not a raise.**
  - **Failures:** each anchor's `except` returns `OutcomeList.failed(type(e).__name__)` instead of `[]`. Timeouts and busy guards return `OutcomeList.unavailable(<label>)`. This matches F2–F4.
  - **Why:** the value still equals `[]`, so direct callers and tests that compare with `[]` are unchanged. F5's gather loop reads the status before `or []`.
  - **Typed producers:** a gatherer that consumes one reads its status immediately after the call and re-wraps the final list with that status (the F3b precedent: items are kept, and the reason is prefixed with the leg name).
  - **Exception:** `get_narrative_context` returns `str`, so its swallow is removed instead. F5's narrative branch then records the failure; its only caller, builder 1068, already catches.
- **Split:** three sequential batches on the same file instead of two halves, each ≤450 lines.
  - **F7a (typed-producer consumers):** `get_personal_notes` (#71), `get_reference_docs` (#72), `_fetch_upload_roster` (#73) and `get_user_uploads` (#74). Response `CGR-20260913-007.md`.
  - **F7b (background-knowledge legs), response `-2`:**
    - `_get_wiki_content_timed` (#80): the timeout and in-flight guards become `unavailable`.
    - `_get_semantic_chunks_timed` (not an anchor): read the status before the 1886 filter, and cover its timeout, in-flight guard and except.
    - The F4 follow-up: fix `SemanticSearchIndex._read_rows` total read failure and the `search` docstring.
  - **F7c (direct swaps), response `-3`:**
    - `get_git_commits` (#75), `get_proposed_features` (#76), `get_procedural_skills` (#77), `get_graph_context` (#78) and `get_unresolved_threads` (#79);
    - `get_narrative_context` (#81) and `get_daemon_self_notes` (#82);
    - `get_relevant_emails`: #84, plus #83, whose fail-closed ranking failure becomes `failed("relevance_unavailable")`.
  - **F8** was split by the parent on 2026-09-14, after a read-only map of both gatherer files. Briefs: `briefs/F8a.md`, `briefs/F8b.md`.
    - **F8a (memory gatherer), response `-4`:**
      - `_get_recent_conversations` (#87) and `get_upcoming_schedule` (#91) return typed failures.
      - `_get_semantic_memories` (#88) also records its inner coordinator-read swallow as a typed failure.
      - `get_user_profile_context` (#90) returns `str`, so it re-raises, mirroring F7c's narrative.
      - `get_recent_facts` (#85), `get_facts` (#86) and `_get_reflections` (#89) have no production caller of the mixin methods. They get evidence only (grep plus a deployed builder test), no code.
      - #87 must return typed, never raise: its direct caller `_build_lightweight_context` (builder ≈2314) switches to a different fallback dict on an exception.
    - **F8b (web gatherer), response `-5`:** `_get_web_search_results` (#92):
      - an exception and a provider-error result (`result.error` with no pages) return typed failures;
      - a budget refusal returns `unavailable("budget")`;
      - a genuine empty search and deliberate non-searches stay `None`;
      - receipts are unchanged, and the formatter's `web_search=` label reads only the receipt, so it does not change;
      - two existing tests that pin `None` are updated under the FIXTURE RULE.
- **Limits recorded now:**
  - **Swallows inside memory/:** four producers catch their own errors: `proposal_store.query_proposals`, `memory_retriever.get_skills`, `memory_coordinator.get_unresolved_threads` and `corpus_manager.get_narrative_context`. F7c closes the gatherer anchors, but end-to-end failure detection for those sections waits for F9–F12.
  - **Visual memories:** not converted in F7. It is a dict section with its own `except → return empty` (≈1521–1523) and an unprotected CLIP leg. F5 reads any truthy dict as succeeded, and fixing that needs a builder change. Carried as a follow-up for the batch that reopens the builder.
  - **Upload existence probe:** `_any_user_uploads_exist` fails OPEN (returns True). That is not an instance of the class.
  - **[verified, F5 packet §14; decided at F8 Ready] Two more always-truthy dict sections:** `summaries` (`_get_summaries_separate`) and `reflections` (`_get_reflections_separate`) in `core/prompt/gatherer_memory.py` return `{"recent": [...], "semantic": [...]}`.
    - They return that dict even from their own outer excepts (386–388 and 893–895), so F5 records them as succeeded even on failure.
    - Neither is a packet anchor, and the `[ACTIVE FEATURES]` inventory never itemizes them.
    - **Decision: not fixed in F8; recorded as a limitation.** Together with visual memories, all three need the builder to read a dict section's status, which belongs to whichever batch next reopens the builder.

### F9 split (parent decision, 2026-09-14)

The split followed a read-only map of the store, the expander and the claim tracker. Briefs: `briefs/F9a.md`, `briefs/F9b.md`.

- **F9a (store reader and claim tracker), base response `CGR-20260913-009.md`:**
  - **#139, `MultiCollectionChromaStore.get_by_id`:** it raises `RetrievalError(source="chroma_get_by_id", reason=<class>)` on a failed read. An unknown collection and a genuine not-found stay `None`, because Chroma's `get(ids=…)` returns empty lists for a missing id instead of raising.
  - **Claim tracker, no code change:** `ClaimIndex.cascade_staleness` currently treats the swallowed `None` as "document deleted" and removes the index entry after a transient failure. After F9a, the raise lands in the loop's existing per-doc `except` before that branch, so no entry is removed. A retention test proves it, and its warning's `{e}` renders labels only.
  - **Interim until F9b:**
    - `MemoryExpander.expand()` has no `try`, so the raise reaches its two callers, `core/agentic/tools.py` `_execute_memory_expand` and `core/insight/sweep.py`.
    - Both already catch broadly, so nothing crashes; this is recorded as evidence.
- **F9b (expander and timestamp-range producer), response `-2`:**
  - **#122, `_fetch_conversations_in_range`:** it propagates the failure instead of returning `[]`.
  - **`get_ids_by_timestamp_range`:** a numeric-pass or legacy-page failure raises. Today each is logged and the union comes back silently partial.
  - **Per-id reads:** one failed read fails the whole range fetch. A silently partial turn list is the defect class, so there is no per-id guard.
  - **`expand()`:** it catches only `RetrievalError` around `_do_expand` and returns an explicit `expansion_failed: <source>: <reason>` error with the usual error-template keys. That result is never cached, which implements "never caches a failure".
  - **Unchanged consumer:** `shutdown_processor._store_summary` already catches around its range call and stores the summary without `source_doc_ids`.
- **Sequencing:** `memory/storage/multi_collection_chroma_store.py` is edited in turn by F9a (`get_by_id`), F9b (`get_ids_by_timestamp_range`) and F10 (`add_conversation_memory`). Each batch owns only its own method body, and each manifest includes the previous batch's state.

### F10 split and the write error type (parent decision, 2026-09-14)

Source: a read-only map of the CGR-010 turn and skill writes plus CGR-009 #130, re-verified on the post-F9b tree. Briefs: `briefs/F10a.md`, with F10b drafted next.

- **Write error type.** A new `StoreWriteError(RuntimeError)` is added to the leaf `utils/retrieval_outcome.py` next to `RetrievalError`. Its constructor takes keyword-only `source` and `reason`, and `str` gives `"source: reason"` (labels only).
  - `RetrievalError` names a read, so writes get their own type.
  - None stays the deliberate skip only. A result object stays rejected, as decided above.
- **F10a (turn write), base response `CGR-20260913-010.md`:**
  - **#140 and #127:** `add_conversation_memory` and `store_interaction` raise `StoreWriteError`. The four deliberate skips in `store_interaction` still return None.
  - **Coordinator:** `MemoryCoordinator.store_interaction` syncs the conversation context back in a `finally`. Without that, a failed write would drop the turn from in-session context, because the next turn's sync-in would overwrite it.
  - **Transcript:** `gui/handlers.py` `_background_store_interaction` still writes the transcript with `db_id=None` on failure, as the None path does today.
  - **Interim until F13:** the orchestrator is unchanged.
    - Its "CRITICAL … data loss" log becomes live; `self.logger` is always set.
    - The five `except: pass` sites ignore the raise silently, exactly as they ignore the None today, so nothing regresses.
- **F10b (skill write and summaries read), responses `CGR-20260913-010-2.md` and `CGR-20260913-009-3.md`:**
  - **#128 and #129:** `store_skill` raises on a collection-unavailable or write failure. Disabled and dedup still return None.
  - **Shutdown skills:** `_extract_procedural_skills` counts a per-skill `StoreWriteError` and continues.
  - **#130:** `_get_recent_summaries_by_timespan` raises `RetrievalError`. `_maybe_regenerate_narrative` never generates or persists a narrative when a summaries read or `get_recent_memories` read failed.
- **[verified] Pre-existing defect, not fixed (owner decision):** `_get_recent_summaries_by_timespan` calls `corpus_manager.get_summaries(limit=50)`, but the production `CorpusManager.get_summaries(self, count=5)` has no `limit` parameter.
  - So the call raises `TypeError` every time and has always returned `[]`; narrative regeneration after consolidation has never run.
  - Fixing the call would turn on an LLM call plus a persisted narrative on the non-default `SUMMARIZE_AT_SHUTDOWN_ONLY=0` path. After F10b, the failure is logged instead of hidden.
- **Sequencing:** F10a runs, then F10b. Both edit `memory/memory_storage.py`, and F10a also edits the shared chroma store file after F9b.

### F11a split (parent decision, 2026-09-14)

Source: a read-only map of the thread store, the coordinator's unresolved-threads read and the shutdown thread pass, re-verified on the post-F10b tree. Briefs: `briefs/F11a-1.md`, with F11a-2 drafted next.

- **F11a-1: the four anchors plus the shutdown pass.** Responses `CGR-20260913-009-4.md` (#121, #147, #148) and `CGR-20260913-010-3.md` (#146).
  - **#146:** `store_thread` raises `StoreWriteError`.
  - **#147:** `list_open_threads` raises `RetrievalError`.
  - **#148:** `query_threads`, which has no production caller, returns `OutcomeList.failed` or `unavailable("collection_missing")`.
  - **#121:** `MemoryCoordinator.get_unresolved_threads` drops its swallow and raises. Its only caller, F7c's gatherer, flattens a typed list with `threads or []`. A raise, by contrast, already reaches its typed except, so the builder records `failed`.
  - **Shutdown thread pass:** `_process_open_threads` catches the typed `RetrievalError` from the open-threads read and returns before extraction. Today it continues, and every extracted thread is stored again as a duplicate. The write loop counts a per-thread `StoreWriteError` and keeps going, and `enforce_cap` still runs.
- **F11a-2: the "[verified] Delete-then-store loses the record" defect for threads. No anchor and no response file.**
  - `resolve_thread` and `_update_thread` write the new version first, then delete the old document only after a confirmed write.
  - `add_to_collection` assigns a fresh `uuid4`, so the two ids never collide.
  - If deleting the old document fails, the new one is rolled back.
  - An item without a document id returns False instead of storing a duplicate.
- **Scanner shape (lesson from F10b):** a contract-required degrade should catch the TYPED failure (`except RetrievalError` / `except StoreWriteError`). Neither a broad `except Exception` nor a flag that moves the `return` out of the except is acceptable. A new gate finding is escalated to the parent, not restructured around.

### Owner decisions and the F11b split (2026-09-14)

**Owner decisions, 2026-09-14 ("yes to all fixes"):** the parent's plan is recorded in `briefs/PARENT_STATE.md`. Effects on this design:
- **`get_summaries(limit=)`:** fixed by the new batch F10c (integrated).
- **Synthesis audit auto-halt:** becomes fail-closed in F12d-2. It was planned for F12d but split off on size; F12d shipped only the synthesis reads. The audit read catches the typed `RetrievalError` and skips dreaming; any other failure propagates to `_run_synthesis_dreaming`'s existing outer except, which ends dreaming.
- **`main.py check-proposals`:** reports a proposal read failure cleanly (F11b-1).
- **Failed-memory-save reporting:** extended.
  - F13b adds an inline "not saved" notice where the reply is still unsent.
  - A new F13c carries the background-save failure into the debug record and the turn record, and adds a one-time notice on the next reply.
  - **Revised by the owner the same day (while F12c ran):** a failed save is never reported inside the chat reply text, on the same turn or the next one.
    - It goes to the logs, and to a small transient GUI status message where "generating…" appears.
    - F13b therefore records receipts and a labelled warning only: `ctx.telemetry`, the debug record, transcript metadata and logs. Every yielded chunk is unchanged.
    - F13c splits in two:
      - **F13c-1:** the delivered debug record is updated in place (grounding precedent). The turn-record row is deferred through one combined waiter over the store and grounding tasks, gated on a real `asyncio.Task`; the post-response hooks still run immediately.
      - **F13c-2:** the status message. For the inline sites it is a `complete` payload field; for the background save, which the UI can only learn about on the next turn, a one-shot flag is consumed. It spans the SPA (`ProgressIndicator`) and legacy Gradio (`typing_md`).
    - Map facts: parent scratchpad `F13c_map.md`, summarized in `briefs/PARENT_STATE.md`.
- **Real school names:** the seeds in `scripts/build_wiki_subset.py` and the test fixtures are replaced in a new H03b.

**F11b split (parent decision):** the split is based on a read-only map of the proposal store and its callers, re-verified after F11a-1 and F10c. Briefs: `briefs/F11b-1.md`, with F11b-2 drafted next.

- **F11b-1, the five anchors:** responses `CGR-20260913-009-5.md` (#132–#134, #136) and `CGR-20260913-010-4.md` (#131).
  - **#131:** `store_proposal` raises `StoreWriteError`.
  - **#132, #133, #134, #136:** `query_proposals`, `get_proposal`, `get_pending` and `get_pending_and_approved` raise `RetrievalError`. Not-found stays None and a genuinely empty result stays `[]`.
  - `query_proposals` must RAISE rather than return a typed list, because `ProposalFilter.get_proposals` flattens a typed list to `[]`. A raise already reaches F7c's gatherer except.
  - **Batch loops:** the shutdown `_generate_proposals` loop and `agent_branch` `ingest_survivors` count a per-item `StoreWriteError` and continue.
  - **CLI:** `check-proposals` prints one labelled line and exits 2.
- **F11b-2 (no anchor):**
  - `ProposalStore.update_status` becomes a staged replacement, following the F11a-2 precedent: write first with a fresh `uuid4`, then delete the old copy, with rollback. A missing collection handle returns False instead of storing a duplicate.
  - `update_tracking_metadata` returns `update_metadata`'s bool instead of True.
- **F11b-1b (added 2026-09-14 after F11b-1 split for size):** F11b-1 delivered the ProposalStore anchors. The two loop callers (shutdown `_generate_proposals`, `ingest_survivors`, each with a per-item typed `StoreWriteError` count) and the `check-proposals` CLI wrap move to F11b-1b, with response `CGR-20260913-010-5.md`. It runs immediately after F11b-1 to close the interim window, during which a failed write aborts the rest of its loop (logged).

### F12 split (parent decision, 2026-09-14)

Source: a read-only map of the consolidator, the daily-notes generator, the retriever and synthesis memory, re-verified during F11b-1b. Briefs: `briefs/F12a.md`; F12b–F12d are drafted next.

- **F12a (#117–#120; response `CGR-20260913-009-6.md`):**
  - `_current_status_facts` raises `RetrievalError`: reason `profile_unavailable` when the profile is None, or the exception class when `get_current_view` fails.
  - The three Obsidian readers raise `RetrievalError`; a missing notes path stays the `[]` skip.
  - `generate_narrative_context` catches the typed `RetrievalError` and returns "" before any LLM call. Its callers persist only a truthy result, so no guard-less or partial narrative is saved — this is "status_guard=unavailable".
  - `main.py refresh_narrative_context` reports a reader failure in one labelled line and does not show a traceback.
- **F12b (no anchor):**
  - The duplicate status guard in `utils/daily_notes_generator.py` raises.
  - `generate_for_date` returns `success=False, error="status_guard_unavailable"` before any LLM call or note write.
  - `tests/unit/test_daily_notes_auto_update.py` is made hermetic. Until then it stays excluded, because it reaches a real default-path `UserProfile()` and the real `OBSIDIAN_VAULT_PATH`.
- **F12c (#123, #125, #126; response `-7`):**
  - `_metadata_fallback_search`, `get_recent_facts` and `get_skills` RAISE rather than return typed lists. The gatherer's `skills or []` and the hybrid merge would flatten a typed list, while existing catches already handle a raise.
- **F12d (#141–#143; response `-8`, which completes CGR-009):**
  - `find_similar`, `get_recurring` and `get_all_results` raise. `store_result` then never inserts a duplicate after a failed similarity read, and the synthesis filter rejects the candidate.
  - Owner decision 2026-09-14: the shutdown synthesis audit check becomes FAIL-CLOSED.
    - A typed `RetrievalError` from the audit read skips dreaming with a warning.
    - Any other failure propagates to `_run_synthesis_dreaming`'s existing outer except, which ends dreaming.
    - No new broad except-and-return is added.
- **Order:** F12a → F12b → F12c → F12d. F12a and F12b both touch the narrative path; F12c and F12d have disjoint core files.

### F13 split and receipt carriers (parent decision, 2026-09-14)

Based on the parent's read-only F13 map (post-F10a tree), the owner's decision 4 as revised the same day, and the F13c map. Briefs: `briefs/F13a.md`; F13b, F13c-1 and F13c-2 are drafted in the parent scratchpad and are re-verified on the tree before each launch.

- **Reachability:**
  - None of the four orchestrator store sites (`_handle_deictic`, `_maybe_document_generation`, `_maybe_agentic_search`, `_store_interaction`) is reached by the GUI or API. Those call `process_user_query(..., use_raw_mode=True)`; only the `main.py` CLI and tests reach the four sites.
  - The live failures are the three `gui/handlers.py` inline `pass` sites (doc generation, daemon note, action retry) and the background store task. That task runs after the reply, the debug record and the turn record.
- **Receipt label:** `"<source>: <reason>"` from a `StoreWriteError`, or `"store_interaction: <exception class name>"` for any other exception. Never exception text.
- **No chat text on any path** (owner revision). A failed save is reported through receipts and logs, and through a transient GUI status message (F13c-2).
- **F13a (orchestrator):**
  - `debug_info["storage_failed"]` at all four sites, plus the turn-record telemetry for `_store_interaction`.
  - The CRITICAL log must not raise when `self.logger` is None (test fixtures set it to None).
  - Receipt code never raises, because a raise would take the fallback path and give a second answer.
  - Nothing goes into `phase_timings`, and nothing is inserted between `debug_info["section_outcomes"]` and `task_timings` (the F6b proximity test).
- **F13b (handlers):**
  - The same receipt in `ctx.telemetry` (guarded when there is no `telemetry` attribute) and in the debug record's `extra`, at the three inline sites.
  - `storage_failed` goes into the background store's transcript metadata only on failure.
  - Every yielded chunk is unchanged.
- **F13c-1:**
  - The delivered debug record is updated in place, on the grounding precedent.
  - The turn-record row is deferred through ONE combined waiter over the store task and the grounding task. It applies only when `_dispatch_storage` returned a real `asyncio.Task`. The post-response hooks still run immediately.
  - `_hook_turn_telemetry` also copies `storage_*` keys.
- **F13c-2:** the transient status message where "generating…" appears. Revised at launch after the read-only F13c-2 map.
  - **Existing signals suffice, so there is no one-shot flag and no new server field.**
    - Inline failures: the `complete` event's debug record already carries `storage_failed`, and redaction keeps every key.
    - Background failures: the delivered record is updated in place (F13c-1), and GET /api/debug shows it.
  - **F13c-2a (TypeScript):** the SPA shows the fixed text "Memory save failed" in the ProgressIndicator position after the stream ends.
    - It shows immediately when `complete.debug.storage_failed` is set.
    - Otherwise it does one follow-up read of the last `/api/debug` record 2.5 s later, guarded by a per-send counter.
    - The notice clears after 4 s or on the next send.
    - The wording is "failed", not "not saved", because a fact-extraction or consolidation failure is also wrapped as `StoreWriteError` after the row was written.
    - Tests run in the new WEB TEST LANE (vitest, `tsc --noEmit`).
  - **F13c-2b (Python):**
    - duel and insight-assembly set `ctx.debug_record` before their yield, so a background failure reaches their record;
    - legacy Gradio stays logs-only for failed saves (parent decision at F13c-2a's integration): its `submit_chat` is nested in the UI builder and cannot be tested without building Gradio. The default SPA carries the status notice.
  - **Missed cases, logs only:** failures later than the follow-up, non-`StoreWriteError` failures.

### F1 Ready decision: one status vocabulary (CM-01, BC-58)

- **[verified] A status vocabulary already exists.** `core/insight/coordinator.py` defines `CHANNEL_STATES` = {succeeded, no_results, no_relevant_results, partial, unavailable, failed, skipped, insufficient} and the `ChannelStatus` dataclass (≈36-66). The `ok | empty | failed | timed_out` names above would be a second vocabulary for the same distinction.
- **Parent recommendation for F1:**
  - reuse the existing state names: ok → `succeeded`, empty → `no_results`, failed → `failed`, timed_out → `unavailable` with reason `timeout`;
  - define them once in the leaf `utils/retrieval_outcome.py`;
  - have `core/insight/coordinator.py` import them only if that change stays behaviour-neutral and inside the size cap. Otherwise, leave insight untouched and pin the shared names with a parity test.
- Final call at F1 Ready.
- **Decided at F1 Ready (parent, 2026-09-14).**
  - **Shared names:** reuse the insight state names. `RETRIEVAL_STATES = {"succeeded", "no_results", "unavailable", "failed"}` is defined once, in a new stdlib-only leaf, `utils/retrieval_outcome.py`. Its contents:
    - `RetrievalError(RuntimeError)`, with keyword-only `source` and `reason`;
    - `OutcomeList(list)`, carrying `status` and `reason`; it stays equal to `[]` and falsy when empty, so unmigrated callers are unchanged;
    - one reader, `outcome_status(value)`.
  - **Parity with insight:** `core/insight/coordinator.py` is NOT edited in F1. A parity test pins `RETRIEVAL_STATES <= CHANNEL_STATES`. Folding insight onto the leaf is a later, separate decision.
  - **Web search manager:** `knowledge/web_search_manager.py` (`FetchedPages`) is not touched either; F2 owns it.
  - **Scope:** F1 has no production caller and closes no anchor. CGR-007..010 stay open until F2 onward adopt the leaf.
  - **Brief:** the parent's F1 brief (scratchpad `F1_draft.md`).

### Line-reference drift (re-verify at every Ready)

- **[map] `gui/handlers.py`.** The `except: pass` sites 1698/2590/3388 cited above are now ≈1707/2599/3500. Line 196 is `logger.error`, not `pass`.
- **[map] `core/orchestrator.py`.** 2169/2300/2477 above map to the pass sites at ≈2164/2295 and the "CRITICAL … data loss" log at ≈2471.
- **[map] Off-path anchors #85, #86, #89.** `get_recent_facts`, `get_facts` and `_get_reflections` have no production callers; the builder uses `_get_reflections_separate`. This matches the evidence-only answer above.

### Sequencing

Queue position is unchanged: the F-series starts after S03. The condition that F5/F6 not overlap A05/A07, and that F10-F13 run after A05, is met once A05d (integrated 2026-09-14) and H01 are in.
