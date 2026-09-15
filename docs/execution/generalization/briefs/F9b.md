=== F9b: memory expansion reports a failed store read as an explicit, uncached error; the timestamp-range producer stops returning a silently partial id list ===
(Parent brief, drafted 2026-09-14. NOT YET RUN: launch only after F9a is integrated; the parent re-verifies these facts against the post-F9a tree first.)

Design source: docs/execution/generalization/failure_outcome_design.md
- "Decisions per request" → CGR-009: "Raise on failure (not-found stays None/[]): … `_fetch_conversations_in_range` …" and "expansion returns an explicit error and never caches a failure".
- Real-defects row: "Range expansion failure falls through to the wrong error reason and is cached".
- Amendment "[map] The expander caches its wrong error".
- Batch row F9, amendment "F9 split": F9b = expander plus timestamp-range producer.
- Amendment "[verified] F9 and F10 share a file": multi_collection_chroma_store.py is edited in sequence by F9a (`get_by_id`), F9b (`get_ids_by_timestamp_range`) and F10 (`add_conversation_memory`).
Request packet (the ONLY class-guard file you may read): /home/lukeh/daemon_exec/class_guards/docs/execution/class_guards/requests/CGR-20260913-009.md. This batch answers anchor #122, whose packet note reads "the original get_ids_by_timestamp_range incident site".
BUG_CLASSES: BC-20, BC-47, CM-05.
Response file (immutable once written): docs/execution/generalization/class_guard_responses/CGR-20260913-009-2.md.
Rules: docs/execution/generalization/briefs/R_common_rules.md applies in FULL.

COMPLIANCE (read twice):
- Your first source write comes only after the failing-first run is recorded in batches/F9b.md.
- No non-pytest Python of any kind; the exceptions are ruff, the read-only scan and the one sanity check.
- Check each fixture against the gates it must pass (hygiene filters, window clamps, cache TTL/fingerprint).
- Disclose any breach immediately.

MANIFEST CHECK: against `$S/manifest_post_F9a.txt` and `$S/manifest_paths_post_F9a.txt` (S = the parent scratchpad).
GUARD, MEMORY, DATA NOTE: as in R_common_rules. The data/ baseline is the 7-entry listing recorded in the latest parent section; user_profile.json stays untouched. Re-check after every chunk and STOP on any change.
ORDER REMINDER: print `sha256sum memory/memory_expander.py memory/storage/multi_collection_chroma_store.py` in the SAME command that first runs your new tests, BEFORE any source edit.
FIXTURE RULE (S01 precedent): an existing test that pins the old swallow may be repaired only with the new assertion plus a paired control. List every edit.

PARENT-VERIFIED FACTS (read-only, 2026-09-14; re-verify at the manifest check; take the store file's post-F9a sha from the manifest)
- After F9a, `MultiCollectionChromaStore.get_by_id` raises `RetrievalError(source="chroma_get_by_id", reason=<class>)` on a failed read; not-found stays None. Confirm this from batches/F9a.md's parent section.
- `RetrievalError(source=, reason=)` renders as `"source: reason"` (utils/retrieval_outcome.py:36–39).
- memory/memory_expander.py `a2152df3…` (= the packet's source sha):
  - `expand()` (def 109):
    - cache lookup and freshness check at ≈134–147;
    - `result = self._do_expand(memory_id, window, collection)` (149), with NO try;
    - `fingerprint = self._anchor_fingerprint(...)` (150);
    - `self._cache[cache_key] = (fingerprint, result, time.time())` (151), which caches whatever came back.
  - `_anchor_fingerprint` (≈162–175): try/except around `get_by_id` → None → forces a recompute. Unchanged; record it.
  - `_do_expand` (def 226):
    - `error_template` (229–236) has keys anchor_id, collection, expansion_method ("timestamp_window"), turns, total_in_collection, error;
    - anchor `get_by_id` (240) and `_find_doc_across_collections` (243) have no try;
    - not-found returns error dicts (241–246);
    - anchor hygiene suppression at ≈253–262.
  - `_expand_summary` (def 282):
    - Strategy 1: `_fetch_docs_by_ids` (≈310);
    - Strategy 2: `_fetch_conversations_in_range(ts_start, ts_end, memory_id)` (≈318);
    - fallback `base["error"] = "Summary has no source_doc_ids or temporal anchors; returning summary text only"` (≈325–328).
  - `_fetch_docs_by_ids` (def 331): `get_by_id` (337) with no try; `if doc:`, otherwise the id is dropped.
  - `_fetch_conversations_in_range` (def 348):
    - `try: doc_ids = self._store.get_ids_by_timestamp_range("conversations", ts_start, ts_end)` / `except Exception as e: logger.warning("[MemoryExpander] get_ids_by_timestamp_range failed: %s", e); return []` (359–365) → ANCHOR #122;
    - `if not doc_ids: return []` (367);
    - per-id `doc = self._store.get_by_id("conversations", doc_id)` (371) with no try; `if not doc: continue`; hygiene filter; sort; turns.
  - `_find_doc_across_collections` (def 431): `get_by_id` per collection (436) with no try.
- memory/storage/multi_collection_chroma_store.py `get_ids_by_timestamp_range` (def 299; docstring 302–318):
  - unknown collection → `return []` (320–321): keep;
  - malformed bound(s) → warning and `return []` (324–331): keep; covered by `test_malformed_bound_returns_empty`;
  - Pass 1 numeric query, `try` 334–344 → `except Exception as e: logger.warning(...numeric query failed...)` (345–346): `numeric_ids` stays empty and execution CONTINUES.
    - The docstring says both passes ALWAYS run and are unioned, so a pass-1 failure silently drops every epoch-carrying row, and a partial result is reported as complete.
  - `try: total = coll.count() / except Exception: total = None` (352–355): advisory only, since pagination also stops on a short page. KEEP.
  - Pass 2 legacy page read, `try` 358–363 → `except Exception as e: logger.warning(...legacy page failed...); break` (364–366): the legacy set is silently partial or empty.
  - The malformed-row skip is counted (377–381): deliberate; KEEP (`test_malformed_row_is_skipped_and_counted`).
  - Neither pass's except is a dm18 finding (except → continue/break, not return []). Confirm this from the pre-edit scan.
  - **Post-F9a re-verification (parent, 2026-09-14):** F9a added one import near the top of the file, so every store line number above moved down by 2.
    - Store sha is `be9b48e5…`.
    - `get_ids_by_timestamp_range` is at def 301. Unknown collection → [] at ≈322–323; malformed bounds → [] at ≈326–333.
    - Pass 1: the numeric `try` is at ≈336, and its except with the "numeric query failed" warning is at 347–348.
    - The `count()` try is at 354–357.
    - Pass 2: the legacy-page `try` is at ≈360, and its except with the "legacy page failed" warning and `break` is at 366–368.
    - `get_by_id` is now at def 491 and raises `RetrievalError` on a failed read; `add_conversation_memory` is at def 528.
    - memory/memory_expander.py (`a2152df3…`) and memory/shutdown_processor.py (`01afac00…`; the range call is at 489) are unchanged, and their line numbers above still hold.
    - tests/unit/test_sep09_storage_repairs.py: `TestTimestampRange` 279, `test_malformed_row_is_skipped_and_counted` 332, `test_malformed_bound_returns_empty` 351, `TestExpansionHygieneAndCache` 464.
- PRODUCTION callers:
  - `get_ids_by_timestamp_range`: memory_expander.py:360 (this batch) and memory/shutdown_processor.py:489 (`_store_summary`, inside `try/except Exception as se: logger.debug("Could not capture source doc IDs")` ≈484–495). A raise there stays caught and the summary is stored without `source_doc_ids`, exactly as with an empty result. Unchanged consumer; record it.
  - `expand()`: core/agentic/tools.py `_execute_memory_expand` (≈1971–1975, try → error dict) and core/insight/sweep.py (≈255–261, try → debug and continue). No unit test drives `_execute_memory_expand` (parent grep).
- Existing tests:
  - tests/unit/test_memory_expander.py: `FakeChromaStore` with fake `get_by_id` / `get_ids_by_timestamp_range` that never raise; `TestCache` 174–210; `TestSummaryExpansion` including `test_summary_temporal_anchors_no_matching_convos`; `TestHygieneFilters` including `test_junk_excluded_from_time_range_fetch`;
  - tests/unit/test_sep09_storage_repairs.py: `TestTimestampRange` 279–395 and `TestExpansionHygieneAndCache` 464+. These use a REAL in-memory `chromadb.EphemeralClient`; run them with the data/logs check after the chunk.
  - F9a's new tests/unit/test_store_get_by_id_outcomes.py.

OWNERSHIP
- memory/memory_expander.py: the bodies of `expand` and `_fetch_conversations_in_range` ONLY, plus one import line.
- memory/storage/multi_collection_chroma_store.py: the body of `get_ids_by_timestamp_range` ONLY. F9a owns `get_by_id`; F10 owns `add_conversation_memory`.
- New tests/unit/test_expander_outcomes.py.
- New docs/execution/generalization/batches/F9b.md.
- The response file named above.
- Read-only: every other expander method (`_do_expand`, `_expand_summary`, `_fetch_docs_by_ids`, `_find_doc_across_collections`, `_anchor_fingerprint`), every other store method, memory/shutdown_processor.py, core/agentic/tools.py, core/insight/sweep.py, memory/claim_tracker.py, utils/retrieval_outcome.py, config/**, briefs/**, LANDING_NOTES.md, failure_outcome_design.md, and every class-guard path.

CONTRACT
1. `get_ids_by_timestamp_range`:
   - Pass 1 numeric-query failure → keep the existing warning line, then `raise RetrievalError(source="timestamp_range", reason=f"numeric:{type(e).__name__}") from e`.
   - Pass 2 legacy-page failure → keep the existing warning line, then `raise RetrievalError(source="timestamp_range", reason=f"legacy_page:{type(e).__name__}") from e`.
   - The `count()` fallback, unknown collection, malformed bounds, malformed-row skip and healthy union are all unchanged.
2. `_fetch_conversations_in_range`:
   - The anchor except no longer returns []. A `RetrievalError` from the producer propagates unchanged. Any other exception from the range call keeps the warning, then `raise RetrievalError(source="timestamp_range", reason=type(e).__name__) from e`.
   - A genuine empty id list → [] (unchanged).
   - A per-id `get_by_id` that raises (after F9a) propagates, with no per-id guard. Parent decision: a silently partial turn list is the defect class, so one failed read fails the range fetch.
   - A genuine per-id not-found is still skipped.
3. `expand()`: wrap ONLY the `_do_expand(...)` call.
   - `except RetrievalError as e:` → return a dict with `_do_expand`'s error_template keys (`anchor_id`=memory_id, `collection`=collection, `expansion_method`="timestamp_window", `turns`=[], `total_in_collection`=0), with `error=f"expansion_failed: {e.source}: {e.reason}"`, and DO NOT write the cache.
   - Any other exception propagates as today.
   - A success, a genuine not-found, a suppressed anchor or the summary linkage fallback are cached exactly as today.
4. Consequence to prove: `_expand_summary`'s "no source_doc_ids or temporal anchors" fallback text is reached only for genuinely missing linkage or a genuinely empty range, never on a failed read.
5. No change to hygiene, window clamping, cache TTL or fingerprint logic, or any other method.
6. Privacy: the `error` text and `RetrievalError` fields carry labels and class names only, never exception message text or document content.

TESTS (tests/unit/test_expander_outcomes.py; fakes only; never a real Chroma client in the new file)
- FAILING FIRST in one command: the digest line above, then the new tests on the UNEDITED sources. Record the failures.
- Producer: build via `MultiCollectionChromaStore.__new__` with a fake collection whose `.get(where=…)` / `.get(limit=, offset=)` / `.count()` can raise independently.
  - numeric query raising → `RetrievalError` with reason "numeric:<class>";
  - legacy page raising → reason "legacy_page:<class>";
  - `count()` raising → still returns the union (control);
  - malformed bounds → [] (control);
  - a healthy numeric-plus-legacy mix → sorted union (control).
- Expander: a FakeChromaStore like test_memory_expander.py's, able to raise on demand, with call counters.
  - A summary with temporal anchors whose range producer raises `RetrievalError` → `expand()` returns `error` starting "expansion_failed: timestamp_range:", `turns == []`, and a SECOND `expand()` call hits the store again (not cached).
  - The anchor `get_by_id` raising `RetrievalError` → explicit error, not cached.
  - A per-id `get_by_id` raising inside the range loop → explicit error, not cached.
  - Controls:
    - a genuinely empty range → today's fallback text, and the second call is served from the cache;
    - a healthy expansion → cached;
    - a non-`RetrievalError` exception from the store still propagates out of `expand()` (unchanged).
- Privacy: a distinctive marker placed in the exception messages never appears in the returned `error` or in `RetrievalError.reason`.
- Focused (≤9 per chunk): the new file, test_memory_expander.py, test_sep09_storage_repairs.py, test_store_get_by_id_outcomes.py, test_claim_tracker.py, test_chroma_store_no_autodelete.py, test_recent_summaries_fix.py, test_sep09_curation_integrity.py.
- Sweep:
  - every remaining tests/unit importer of memory.memory_expander / MemoryExpander / memory.storage.multi_collection_chroma_store / memory.shutdown_processor (grep and list), in chunks of ≤9;
  - vet any shutdown_processor test for real stores and default paths before running it;
  - usual exclusions (test_graph_integration.py, the ContextGatherer/UserProfile hazard, tests/test_web_search_manager.py, tests/test_prompt_internal_methods.py).

SCAN: pre- and post-edit read-only scan. Expect the dm18 row for #122 to go STALE, with new = 0.
RESPONSE FILE (`-2`):
- #122 fixed, with deployed-function evidence;
- the timestamp-range producer siblings fixed (both passes);
- the `expand()` no-cache explicit error (the design-doc "never caches a failure" decision);
- the per-id abort decision;
- unchanged consumers (shutdown_processor `_store_summary`, tools.py, sweep.py);
- the remaining CGR-009 anchors mapped to F10–F12.
SIZE: target ≤380 changed lines; HARD stop at ≤450. Stop before exceeding it and return a split proposal (producer vs expander).
ORDER: manifest → create F9b.md → pre-edit scan → tests → failing-first (digests in the same command) → edit → focused → sweep → ruff → scan → data/logs listing → packet → response file.
