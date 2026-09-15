=== F9a: MultiCollectionChromaStore.get_by_id distinguishes a failed read from "not found"; the claim-tracker staleness cascade never deletes an index entry because of a failed read ===
(Parent brief, 2026-09-14. Launched after F8b was integrated. The design-doc F9 split amendment was written at the same time.)

Design source: docs/execution/generalization/failure_outcome_design.md
- "Decisions per request" → CGR-009: "Raise on failure (not-found stays None/[]): `MultiCollectionChromaStore.get_by_id`, …".
- Batch row F9 ("`get_by_id` + expander + claim tracker", shared seam "store reader").
- Amendment "[verified] F9 and F10 share a file": `memory/storage/multi_collection_chroma_store.py`, which F9a, then F9b, then F10 edit in sequence.
- Amendment "F9 split" (parent decision, 2026-09-14): F9a = `get_by_id` plus claim tracker (response `CGR-20260913-009.md`, the BASE response); F9b = expander plus timestamp-range producer (response `-2`).
Request packet (the ONLY class-guard file you may read): /home/lukeh/daemon_exec/class_guards/docs/execution/class_guards/requests/CGR-20260913-009.md. This batch answers anchor #139.
BUG_CLASSES: BC-20, BC-47, CM-05.
Response file (immutable once written): docs/execution/generalization/class_guard_responses/CGR-20260913-009.md. This is the BASE response; none exists yet.
Rules: docs/execution/generalization/briefs/R_common_rules.md applies in FULL.

COMPLIANCE (read twice):
- Your first source write comes only after the failing-first run is recorded in batches/F9a.md.
- No non-pytest Python of any kind; the exceptions are ruff, the read-only scan and the one sanity check.
- Check each fixture against the gates it must pass.
- Disclose any breach immediately.

MANIFEST CHECK: against `$S/manifest_post_F8b.txt` and `$S/manifest_paths_post_F8b.txt` (S = the parent scratchpad; the same two commands as other briefs).
GUARD, MEMORY, DATA NOTE: as in R_common_rules. The data/ baseline is the 7-entry listing recorded in batches/F8b.md's parent section; user_profile.json stays untouched. Re-check after every chunk and STOP on any change.
ORDER REMINDER: print `sha256sum memory/storage/multi_collection_chroma_store.py memory/claim_tracker.py` in the SAME command that first runs your new tests, BEFORE any source edit.
FIXTURE RULE (S01 precedent): an existing test that pins the old swallow may be repaired only with the new assertion plus a paired control. List every edit.

PARENT-VERIFIED FACTS (read-only map, 2026-09-14; re-verify at the manifest check)
- utils/retrieval_outcome.py: `RetrievalError(source=, reason=)`; `__str__` is `f"{source}: {reason}"` (36–39), made of labels only.
- memory/storage/multi_collection_chroma_store.py `e0884db3…` (= the packet's source sha). `get_by_id` (def 489):
  - `if collection_name not in self.collections: return None` (495–496);
  - try 498: `results = coll.get(ids=[doc_id], include=["documents", "metadatas"])`; `if not docs: return None` (502–503, the genuine not-found path); success returns `{"id", "content", "metadata"}`;
  - `except Exception: return None` (509–510) → ANCHOR #139. Failure and not-found share None.
  - Chroma's `Collection.get(ids=...)` returns empty lists for a missing id; it does not raise. The except only fires on a real failure.
- Sibling store reads (record them; do NOT change):
  - `list_all` (403), `get_recent` (471) and `query_collection` (783) already raise;
  - `search_all` (845), `query_multiple_collections` (859), `update_metadata` (920, returns False) and `get_collection_stats` (942, typed-ish) swallow;
  - `get_ids_by_timestamp_range` (299) belongs to F9b.
- PRODUCTION callers of `get_by_id` (read-only map):
  - memory/memory_expander.py:
    - `_anchor_fingerprint` (170–173) already catches → None → recompute; unchanged;
    - `_do_expand` (240), `_fetch_docs_by_ids` (337), `_find_doc_across_collections` (436) and `_fetch_conversations_in_range`'s per-id loop (371) have NO try, so after F9a a `RetrievalError` propagates out of `MemoryExpander.expand()` (which has no try);
    - F9b turns that into an explicit, uncached error result;
    - in the interim, the two external `expand()` callers already catch broadly: core/agentic/tools.py `_execute_memory_expand` (≈1971–1975 → `{"anchor_id", "turns": [], "error": str(e)}`) and core/insight/sweep.py (≈255–261 → debug log, continue). There is no crash. No unit test drives `_execute_memory_expand` (parent grep of tests/unit is empty), so record this interim state as READ-ONLY evidence (quote the lines) in the packet and the response; do NOT write a test for it.
  - memory/claim_tracker.py `ClaimIndex.cascade_staleness` (def 254; file sha `fbae71f1…`, no packet anchor):
    - inside the per-doc loop, `try:` (290) → `_doc = chroma_store.get_by_id(collection, doc_id)` (294) → `existing = _doc.get("metadata") if _doc else None`;
    - if existing: update stale_claims via `chroma_store.update_metadata(...)` (return value unchecked, a sibling) and append a result;
    - else: "Document doesn't exist (deleted?), clean up index" → `self.remove_document(doc_id)` (312–315);
    - `except Exception as e: logger.warning(f"[Staleness] Failed to update {doc_id} in {collection}: {e}")` (316–317).
    - TODAY: a failed read returns None, so the else branch DELETES the index entry on a transient failure.
    - AFTER F9a: the raise jumps to the loop's except before the else branch, so no deletion happens. The warning's `{e}` renders as "chroma_get_by_id: <class>" (labels only), so it stays privacy-safe.
    - memory/claim_tracker.py is therefore READ-ONLY in this batch: NO edit. The claim-tracker item is evidence plus a retention test.
  - Callers of `cascade_staleness`: core/orchestrator.py `run_post_response_detectors` (≈2564) and memory/cross_deduplicator.py `run` (≈167) both wrap it in a broad try; unchanged.
- Existing tests (read-only map):
  - tests/unit/test_claim_tracker.py `TestCascadeStaleness` (342–441; `MagicMock(spec=MultiCollectionChromaStore)`; `test_cascade_missing_doc_cleaned_up` sets `get_by_id.return_value = None`);
  - tests/unit/test_memory_expander.py (`FakeChromaStore` never raises);
  - tests/unit/test_chroma_store_no_autodelete.py, test_recent_summaries_fix.py (the `side_effect=RuntimeError` precedent) and test_upload_retrieval_pool.py use `__new__` with fake collections;
  - tests/unit/test_sep09_storage_repairs.py and test_sep09_curation_integrity.py use a REAL in-memory `chromadb.EphemeralClient` store. The A01 parent rerun ran curation_integrity without data/ writes; allowed, with the data/logs check after the chunk.

OWNERSHIP
- memory/storage/multi_collection_chroma_store.py: the body of `get_by_id` ONLY, plus one import line for `RetrievalError`.
- New tests/unit/test_store_get_by_id_outcomes.py.
- New docs/execution/generalization/batches/F9a.md.
- The response file named above.
- Read-only: memory/claim_tracker.py, every other store method (F9b and F10 own later regions of the same file), memory/memory_expander.py (F9b), core/agentic/tools.py, core/insight/sweep.py, core/orchestrator.py, memory/cross_deduplicator.py, utils/retrieval_outcome.py, config/**, briefs/**, LANDING_NOTES.md, failure_outcome_design.md, and every class-guard path.

CONTRACT
1. Unknown collection → None (unchanged; record why: it is a caller/config mismatch, not a store read).
2. A genuine not-found (empty `documents`) → None (unchanged).
3. `coll.get(...)` raises → `raise RetrievalError(source="chroma_get_by_id", reason=type(e).__name__) from e`. Do not log exception text.
4. Found → the same dict as today.
5. Claim tracker (no code change):
   - a `get_by_id` that raises must NOT reach `remove_document`: the index entry for that doc is retained, `cascade_staleness` does not raise, and that doc gets no result entry;
   - a genuine not-found still removes it (existing test);
   - a healthy doc still updates.
6. No change to any other store method or caller.
7. Privacy: `RetrievalError` carries only the source label and the exception class name.

TESTS (tests/unit/test_store_get_by_id_outcomes.py; fakes only; never a real Chroma client in the new file)
- FAILING FIRST in one command: the digest line above, then the new tests on the UNEDITED sources. Record the failures (the claim-tracker retention test should FAIL today, because the swallowed None deletes the entry).
- Store: build via `MultiCollectionChromaStore.__new__` with a fake collection, following test_chroma_store_no_autodelete.py / test_recent_summaries_fix.py. Cases:
  - `.get` raising → `RetrievalError` with source/reason labels and no message marker;
  - empty result → None;
  - found → dict;
  - unknown collection → None.
- Claim tracker, through the deployed `ClaimIndex.cascade_staleness` with a `MagicMock(spec=MultiCollectionChromaStore)`:
  - `get_by_id.side_effect = RetrievalError(...)` → no exception; `remove_document` is not invoked (or `_doc_claims` still holds the doc); results exclude it;
  - control: `get_by_id.return_value = None` → the entry is removed;
  - control: a healthy doc → `update_metadata` is called and a result is returned.
- Claim tracker through the REAL edited store: a `MultiCollectionChromaStore.__new__` store whose fake collection's `.get` raises, passed to `cascade_staleness` → the index entry is retained (this proves the store edit and the cascade together, not just the mock).
- Privacy: a distinctive marker placed in the exception message never appears in `RetrievalError.reason`, `str(RetrievalError)`, or the claim-tracker warning (caplog).
- Focused (≤9 per chunk): the new file, test_claim_tracker.py, test_memory_expander.py, test_chroma_store_no_autodelete.py, test_recent_summaries_fix.py, test_upload_retrieval_pool.py, test_sep09_storage_repairs.py, test_sep09_curation_integrity.py.
- Sweep:
  - every remaining tests/unit importer of memory.storage.multi_collection_chroma_store / MultiCollectionChromaStore / memory.claim_tracker / memory.memory_expander (grep and list), in chunks of ≤9;
  - vet each for real stores, default paths and real embedders BEFORE running it; exclude and record any that would touch data/;
  - usual exclusions (test_graph_integration.py, the ContextGatherer/UserProfile hazard);
  - never run tests/test_web_search_manager.py or tests/test_prompt_internal_methods.py.

SCAN: pre- and post-edit read-only scan. Expect the dm18 row for #139 to go STALE, with new = 0. Baseline after F8b: dm18 found 49 / base 79 / new 0 / stale 30.
RESPONSE FILE (BASE `CGR-20260913-009.md`):
- #139 fixed, with deployed-function evidence;
- the claim-tracker no-deletion evidence (as a sibling);
- the interim expander behaviour and the external callers' catch (read-only evidence);
- store siblings recorded;
- the remaining CGR-009 anchors mapped to owning batches: #122 → F9b (`-2`); #130 → F10; #121, #147, #148 → F11a; #132, #133, #134, #136 → F11b; #117–#120, #123, #125, #126, #141–#143 → F12.
SIZE: target ≤300 changed lines; HARD stop at ≤450.
ORDER: manifest → create F9a.md → pre-edit scan → tests → failing-first (digests in the same command) → edit → focused → sweep → ruff → scan → data/logs listing → packet → response file.
