=== F7a: knowledge gatherers keep the producer's failure status (personal notes, reference docs, user uploads, upload roster) ===
(Durable copy, re-created in the repo on 2026-09-14 after a machine crash wiped the /tmp scratchpad. Content unchanged; only the rules path is now explicit.)

Design source: docs/execution/generalization/failure_outcome_design.md
- "Decisions per request" → CGR-007.
- "Parent review amendments" → "F7 split and gatherer outcome shape (parent decision, 2026-09-14)". Read it in full; this batch implements its F7a row.
Request packet (the ONLY class-guard file you may read): /home/lukeh/daemon_exec/class_guards/docs/execution/class_guards/requests/CGR-20260913-007.md.
This batch answers anchors #71, #72, #73 and #74.
BUG_CLASSES: BC-20, BC-47, CM-05.
Response file (immutable once written): docs/execution/generalization/class_guard_responses/CGR-20260913-007.md. This is the BASE response; none exists yet.
Rules: docs/execution/generalization/briefs/R_common_rules.md applies in FULL, including "Response file", SHELL GUARD, NON-PYTEST CODE, GIT INDEX AND PYTEST HYGIENE, NON-UNIT TESTS, INTERRUPTION and MEMORY.

MANIFEST CHECK (before any edit; if either check fails, stop and report)
Run from the checkout root with S=/tmp/claude-1000/-home-lukeh-daemon-exec-generalization/1f0f3407-5796-4278-85e4-0c7ba4f50aa9/scratchpad:
  (a) `sha256sum -c --quiet $S/manifest_post_F6b.txt` must print nothing and exit 0.
  (b) `{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u | diff - $S/manifest_paths_post_F6b.txt` must print nothing.
GUARD: run the /proc/comm pytest guard before EVERY pytest command, `--collect-only` included. If another pytest is running, wait in the foreground and re-check about every 60s. Never start pytest in the background.
MEMORY: run tests in the FOREGROUND, in chunks of ≤9 files. Before each chunk, MemAvailable (`awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo`) must be ≥4000; otherwise wait and re-check.
DATA NOTE:
- Before the first pytest and AFTER EVERY CHUNK, record:
  - `ls -la --time-style=full-iso data` (top level only);
  - `ls -ld --time-style=full-iso logs`.
- Compare against the parent's post-F6b baseline recorded in batches/F6b.md.
- If any `data/` entry appears or changes, or `logs/` appears or changes, STOP and report which run did it. Never delete anything.
SCRIPTS: pytest only.
- `python -c`, `python3 -`, REPL, heredoc snippets (even empty ones) and throwaway scripts all need parent approval BEFORE they run.
- ruff, the read-only scan and the one `import utils` sanity check are the only exceptions.
- File edits use the Edit or Write tools only. Never leave stray files in the repository.
- No real Obsidian vault, Chroma store, embedder, model or network. Use fakes.
ORDER REMINDER: print `sha256sum core/prompt/gatherer_knowledge.py` in the SAME command that first runs your new tests, BEFORE any source edit.
FIXTURE RULE (S01 precedent): an existing test that pins the old flattening may be repaired only with the new assertion plus a paired control. List every existing-test edit. Anything else is a STOP with an escalation packet.

PARENT-VERIFIED FACTS (post-F4 tree: core/prompt/gatherer_knowledge.py sha `77d094d6…` = the packet's source SHA-256; re-verified unchanged after F6b and after the 2026-09-14 crash. Re-verify the digest at the manifest check.)
- The file does NOT import `utils.retrieval_outcome`. No gatherer reads `.status` today.
- `get_personal_notes` (def 601):
  - `if not manager: return []` (624): not configured; leave it.
  - try at 627: `notes = await manager.get_notes(` (642).
    - Producer knowledge/obsidian_manager.py `get_notes` (F3b) returns:
      - `OutcomeList(final_results)`;
      - `OutcomeList(final_results, status=<failed|unavailable>, reason="keyword:<reason>")` when its keyword leg failed (items kept);
      - `OutcomeList.failed(type(e).__name__)`.
  - Transforms that rebuild the list (and drop status): 658 (substance filter), 674 (rollup filter), 689 (mood filter).
  - `self.memory_id_map[...]` (702–712); `return notes or []` (717).
  - except 719–721 → `return []`: ANCHOR #71.
- `get_reference_docs` (def 723):
  - `if not manager: return []` (736).
  - try at 739: `docs = await manager.get_documents(query, limit=limit * 2)` (741). Producer knowledge/reference_docs_manager.py `get_documents` (F3a) has the same OutcomeList shape (keyword-leg status with items kept; failed on exception).
  - Transforms: 744 (drop user_upload), 759 (`_is_self_doc`), 765 (`docs[:limit]`).
  - memory_id_map 772–781; `return docs or []` (785).
  - except 787–789: ANCHOR #72.
- `_fetch_upload_roster` (def 806):
  - `if not manager: return []` (815).
  - try 817: `coll = manager.chroma_store._get_collection('reference_docs')` (818); `coll.get(where={"type": "user_upload"}, include=["metadatas"])` (819).
  - except 820–822 (debug log) → `return []`: ANCHOR #73.
  - Builds a plain roster of title/date dicts (844–849).
- `get_user_uploads` (def 871):
  - `if not manager: return []` (884).
  - `if not self._any_user_uploads_exist(): return []` (888–889). The probe fails OPEN (returns True) on error (858–862): not the class; leave it.
  - try 891: `docs = await manager.get_documents(query, limit=limit * 2, doc_type="user_upload")` (897).
  - Transforms: 902, 917–920 (same-turn dedupe), 931 (`_upload_is_live`), 940 (`_dedupe_upload_content`), 942 (slice).
  - memory_id_map 958–966.
  - Roster:
    - `wants_roster` (981–983); `roster = self._fetch_upload_roster() if wants_roster else []` (984);
    - `self._last_upload_roster = roster` (985; class attribute declared at 804; no reader anywhere outside this file, per parent grep);
    - the roster entry is inserted first (986–992).
  - `return uploads or []` (994).
  - except 996–998: ANCHOR #74.
- F1 leaf: `OutcomeList(items=(), *, status=None, reason="")`, where a None status means succeeded if items else no_results, and a no_results status with items is rejected. It also provides `.failed(reason, items=())`, `.unavailable(reason, items=())` and `outcome_status(value)`.
- Downstream (READ-ONLY; re-read batches/F5.md, F6a.md, F6b.md and their parent sections):
  - F5's gather loop reads `outcome_status(task.result())` BEFORE `or []` and records `_section_outcomes[name]`.
  - F6a renders personal_notes NOT CHECKED as `obsidian=ON(could not check)` and other NOT CHECKED sections in "Could not check this turn: …".
  - F6b publishes receipts (`sections_not_checked`, debug record, turn record).
  - Confirm each against the integrated code before relying on it.
- Existing tests that reference these methods (parent grep):
  - get_personal_notes:
    - tests/unit/test_obsidian_failure_outcomes.py. `TestGathererConsumerUnaffected::test_get_personal_notes_with_failing_manager_returns_empty` (≈242) asserts `== []`; that stays true for a failed OutcomeList.
    - tests/unit/test_retrieval_context_quality.py
    - tests/unit/test_sep03_live_probe_fixes.py
  - get_reference_docs: tests/unit/test_narration_turn_audit_fixes.py
  - get_user_uploads:
    - tests/unit/test_gatherer_latency_guards.py
    - tests/unit/test_prompt_builder_self_report_trim.py
    - tests/unit/test_sep04_attachment_turn.py
    - tests/unit/test_sep08_homework_tone_misfires.py
    - tests/unit/test_upload_retrieval_pool.py

OWNERSHIP
- core/prompt/gatherer_knowledge.py, ONLY:
  - one import line (`from utils.retrieval_outcome import OutcomeList, outcome_status`);
  - the bodies of `get_personal_notes`, `get_reference_docs`, `_fetch_upload_roster` and `get_user_uploads`.
- New tests/unit/test_gatherer_outcomes_notes_docs_uploads.py.
- New docs/execution/generalization/batches/F7a.md.
- The response file named above.
- Read-only: every other method in gatherer_knowledge.py (F7b and F7c own them), knowledge/obsidian_manager.py, knowledge/reference_docs_manager.py, core/prompt/builder.py, formatter.py, core/orchestrator.py, gui/handlers.py, utils/retrieval_outcome.py, config/**, docs/execution/generalization/briefs/**, and everything else. Every class-guard-owned path is also read-only (see R_common_rules.md "Never edit").

CONTRACT
1. The four anchor excepts return `OutcomeList.failed(type(e).__name__)` instead of `[]`. The existing log lines stay byte-identical.
2. `get_personal_notes`, `get_reference_docs`, `get_user_uploads`:
   - Immediately after the producer call and BEFORE any transform: `leg_status, leg_reason = outcome_status(<producer result>)`.
   - At the success return:
     - if `leg_status in ("failed", "unavailable")` → `return OutcomeList(<final list>, status=leg_status, reason=leg_reason)`, keeping the items exactly as today;
     - otherwise return the final list as today (plain list or `OutcomeList(<final list>)`; state which).
   - The producer's reason is already a privacy-safe label; pass it through verbatim.
3. Upload roster:
   - `r_status, r_reason = outcome_status(roster)` right after line 984's call.
   - When the roster was wanted and `r_status == "failed"` and the documents leg is NOT already failed/unavailable, the final status is `failed` with reason `"roster:" + r_reason`, and the upload items are kept.
   - A failed/unavailable documents leg wins over the roster.
   - `self._last_upload_roster` still receives the roster value.
   - Record in the packet that a roster-only failure marks the whole `user_uploads` section NOT CHECKED (F3b precedent). The parent reviews that choice.
4. Legit early returns stay as they are (`if not manager`, the `_any_user_uploads_exist` gate). They read as no_results.
5. No change to filters, dedupe, slices, citation ids, memory_id_map, log text, roster insertion order or `_any_user_uploads_exist`.
6. Privacy: every reason is an exception class name or a producer label. Never query, note, document, title or exception-message text.

TESTS (tests/unit/test_gatherer_outcomes_notes_docs_uploads.py; fakes only)
- Construct the gatherer the way the existing tests do. Start from test_obsidian_failure_outcomes.py `TestGathererConsumerUnaffected` and test_upload_retrieval_pool.py.
- FAILING FIRST in one command: `sha256sum core/prompt/gatherer_knowledge.py`, then the new tests on the UNEDITED source. List the failures.
- For each of notes, reference docs and uploads, through the deployed method:
  - the manager call raises → `outcome_status(result) == ("failed", "<ExcClass>")` and `result == []`;
  - the producer returns `OutcomeList.failed("X")` (no items) → failed / "X";
  - the producer returns `OutcomeList(items, status="failed", reason="keyword:TimeoutError")` whose items pass every filter → the returned items equal today's output for the same items (compare against the same items given as a plain `OutcomeList(items)`), AND the status is failed / "keyword:TimeoutError". This is the "status survives the transforms" proof;
  - the producer returns an unavailable status → unavailable;
  - controls: healthy non-empty → succeeded with today's items; healthy empty → no_results; no manager → `[]` / no_results.
- Roster:
  - a roster-wanting query with a raising `_get_collection` → section failed, reason starts with "roster:", upload items unchanged;
  - a roster-wanting query when the documents leg failed → the documents status wins;
  - a query that does not want a roster → roster not fetched (control).
- Through the builder (F5 integrated): the `full_builder` pattern from tests/unit/test_prompt_timeout.py / test_independent_prompt_audit.py, with a failing notes manager → `_section_outcomes["personal_notes"]["status"] == "failed"`. Also through the formatter if the existing fixture reaches it: the inventory shows `obsidian=…(could not check)`.
- Privacy: a distinctive marker placed in the query and in the exception message never appears in any reason.
- Focused: the new file plus the 9 existing files above (2 chunks, ≤9 each).
- Sweep:
  - every tests/unit importer of core.prompt.gatherer_knowledge, KnowledgeRetrievalMixin or core.prompt.context_gatherer not already run (grep and list), in chunks of ≤9.
  - A non-unit importer runs only if reading it proves every store, model, limiter, cache, state file and telemetry path it builds is a fake or tmp_path-scoped AND no real Chroma/embedder/ModelManager is constructed. Otherwise list it with the reason.
  - Never run tests/test_web_search_manager.py or tests/test_prompt_internal_methods.py.

SCAN: pre- and post-edit `python scripts/check_bug_classes.py scan --root .` (never `--write-baseline`). Expect the dm18 rows for #71–#74 to go STALE (4 rows) and new = 0. A `return OutcomeList.failed(...)` in an except is not a dm18 finding (F2–F4 precedent).
RESPONSE FILE:
- Include everything R_common_rules requires, per anchor #71–#74.
- List #75–#92 as "not in this batch": #80 → F7b (-2); #75–#79 and #81–#84 → F7c (-3); #85–#92 → F8 (-4).
- Record the roster decision and the read-only producer siblings.
SIZE: target ≤350 changed lines; HARD stop at ≤450. Stop before exceeding it and return a split proposal.
ORDER: manifest → create F7a.md → pre-edit scan → tests → failing-first (digest in the same command) → edit → focused → sweep → ruff → scan → data/logs listing → packet → response file.
