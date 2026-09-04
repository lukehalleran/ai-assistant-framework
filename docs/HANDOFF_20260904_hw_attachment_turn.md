# HANDOFF 2026-09-04 — homework-attachment turn audit (Fable at 97%, handing off)

Source: owner's Debug dump of TURN #1 (enhanced, kimi-3, 128.5s, prompt 265,651 tok).
Query = 1 message + attached UsedCars.csv (1,264 rows) + Homework1-1.pdf + ~10 lecture
transcripts. Owner's ask: "this is the sort of response we want to get EXACTLY correct."

## A. Verified so far (Fable, from the dump + code reads)

1. **Attachment bundle rendered TWICE in [CURRENT QUERY]** (CSV→PDF→transcripts, then the
   identical block again) — ~130K of the 265K prompt tokens are duplication.
   - `utils/file_processor.process_files_structured` (l.123-190) appends each document's
     `content_text` ONCE → not the culprit by itself. So either `files` held every file twice
     (api/chat_service.py:57 `state.resolve_uploads(req.file_ids)` — check the SPA for a
     double-append of file_ids, and resolve_uploads for id-dedupe), or a second append of
     `files_result.documents` happens downstream of `merged_input` (gui/handlers.py:4271; used
     at 1100/1179/2910/765 — check `_run_enhanced` + formatter for `documents`/`uploaded`).
   - Also `[USER UPLOADED ITEMS] n=3` = chunks of the SAME csv (tmp6uvd38su.csv) retrieved in
     the turn that uploaded it → third copy of rows. Dedupe: skip upload chunks whose source
     filename matches a file attached THIS turn.

2. **Intent misfired: TEMPORAL RECALL** (style block injected; its retrieval profile pulled
   RECENT CONVERSATION n=18 incl. three giant evidence-sweep replies). Query has no "?" and is an
   effort-estimate question. Cause (unverified but near-certain): intent runs over
   `merged_input`, and the keyword map at core/intent_classifier.py:437-443 ("history",
   "timeline", "past", "previous", "over time") matches words INSIDE the attached transcripts
   ("previous video" appears dozens of times). Regex arms at :323-360.
   → Classify intent/tone/topic/STM/web-trigger/gate/query-rewrite on the user's OWN text
   (`ctx.user_text`, or merged_input head up to the first attachment boundary), keep full
   merged_input for prompt rendering, fact extraction (paste guard) and storage.

3. **Obsidian notes relevance 1.00 for unrelated 2024 daily notes** ("11 22 24", "8 10 24",
   "7 27 24") — keyword scorer saturated: with a 130K-token query nearly every note word is
   "in the query". Check `knowledge/obsidian_manager._keyword_search`; same fix as (2): score
   against user text, not the attachment bundle.

4. **Latency**: prepare_prompt 97.6s (context_pipeline 49s, memories 38.6s, relevant_emails
   11.8s) — every analysis LLM call/embedding got the giant merged query. (2) fixes most of it.

5. **Debug-export redaction false positive**: `[REDACTED PHONE]` replaced "1085\n999  1000"
   inside the CSV (utils/privacy_redaction.py phone regex spans newline + spaces across numeric
   table cells). Tighten: no newline inside a match, require phone-shaped separators.
   Note only: "1726 Howard" (street address) is NOT redacted — addresses aren't covered.

6. **Response correctness** (the owner's real question):
   - WRONG: "~1,400 rows" — attached file has 1,264 rows (Id 1–1264). Model used prior
     knowledge of the classic Corolla set (1,436) instead of the provided data.
   - MISSED: deadline trap. User said "11 pm central"; instructions say 11:59 PM **Eastern**
     = 10:59 PM Central. Profile timezone=Central was in context; reply said nothing.
   - MISSED (biggest for effort allocation): PDF title is "Homework 1 – **Part 1**" and the
     pasted Canvas text carries an "onHousing.csv" fragment → almost certainly a Part 2
     (Housing.csv; syllabus lists dummy/polynomial regression next). 2–4h estimate covers
     Part 1 only; should have asked.
   - CORRECT: 12 tasks, mechanical mirror of the demo, 9 predictors, CC=16000 outlier (Id 65)
     is real, Age in months → ×12, `car` package, adj-R² comparison, 9 days, didn't do the HW.
   - Grounding verifier can't catch the row count (it sees head/tail 2500 chars of the query).
   - Noise: [KNOWLEDGE GRAPH] n=12 identity edges irrelevant to the turn (incl. junk
     "mom name lol", "name andrew"); 8,803 chars of profile middle-out snipped.

## B. Plan for the executing agent (Codex or Sonnet) — dry-run/tests first, no commits

1. Duplication root cause + regression test: fake 2 text files through handle_submit's
   prompt path (or prepare_prompt with merged_input) and assert each file's content occurs
   EXACTLY once in the rendered [CURRENT QUERY]; grep `web/src` for file_ids construction.
2. `SubmitContext.analysis_text` (user text / pre-attachment head, cap ~2K chars) wired into
   ContextPipeline (intent, tone, topic, STM), web trigger, agentic gate, query rewrite,
   obsidian keyword search. Tests: this query shape must NOT classify temporal_recall; an
   unrelated 2024 note must not score 1.0 against it.
3. Same-turn upload dedupe in the uploads gatherer (filename match against ctx.file_names).
4. Deterministic CSV manifest: prepend "[<name>: N rows × M cols; columns: …]" to attached
   tabular content in file_processor (gives the model the true row count for free). Test.
5. Redaction phone regex: newline-free matches; test with the CSV fragment above.
6. Prompt-level: TEMPORAL REASONING gains one line — if a pasted document states a deadline
   in a timezone different from the user's, state the converted local time.
7. Run the touched test files + the sep03/sep01 regression files; report pass/fail honestly.

Owner follow-ups (not code): confirm whether HW1 Part 2 (Housing.csv) is also due Sep 13;
real cutoff is 10:59 PM Central.

## B2. Addendum (owner-confirmed 2026-09-04 afternoon)

ADDENDUM — owner confirmed HW1 Part 2 exists (Housing.csv, same due datetime, was not attached). Items 8–9 added; items 4 and 6 are must-ship. All deterministic, no LLM calls; tests in tests/unit/test_sep04_attachment_turn.py.

Item 8 — referenced-but-missing attachment audit (new utility, e.g. utils/attachment_audit.py, wired where merged_input is built in gui/handlers.py ~l.4271): (a) extract filename-shaped tokens from the USER'S OWN text and from attached document text (`[\w\-. ]+\.(csv|pdf|xlsx|xls|docx|txt|json|ipynb|R|py)` word-bounded, dedupe case-insensitively, tolerate glued Canvas fragments like "onHousing.csvHomDownload" → "Housing.csv" by trimming a leading lowercase run before a TitleCase filename); (b) compare against attached basenames (orig_name via the api/state.py resolve_uploads shim, else .name); (c) detect multi-part titles in attached docs: /\bPart\s+(\d+|[IVX]+)\b/ and /Homework\s*\d+\s*[-–]\s*\d+/ within the first ~300 chars; (d) emit ONE short block appended to the query context, e.g. "[ATTACHMENT NOTE] Pasted material references files not attached: Housing.csv. Attached PDF is titled 'Homework 1 – Part 1'; other parts may exist. Ask before estimating scope." Silent when nothing is missing. Tests: the exact fragment "Instructions\nonHousing.csvHomDownload the following homework instruction file and data file:\n\nHomework1-1.pdf\n\nUsedCars.csvDownload UsedCars.csv" with attached names UsedCars.csv + Homework1-1.pdf → note names Housing.csv and Part 1; negative case (everything attached, no Part title) → no note.

Item 9 — deadline timezone conversion, deterministic: regex over user text + attached doc text for a time with an explicit zone token (/\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\s*\(?\s*(Eastern|Central|Mountain|Pacific|ET|EST|EDT|CT|CST|CDT|MT|PT|PST|PDT|UTC)\b/i); resolve the user zone via utils/timezone_resolver.py (see how core/actions executors use it; fall back to config/profile timezone; unknown → emit nothing); when zones differ append "[DEADLINE NOTE] 11:59 PM Eastern = 10:59 PM Central (your timezone)." Tests: "11:59pm Eastern Time" with America/Chicago → "10:59 PM Central"; same zone → no note; no zone token → no note. Keep the one-line TEMPORAL REASONING prompt rule from item 6 as belt-and-suspenders.

Item 4 reminder (manifest): compute from the parsed table, not raw text — "UsedCars.csv: 1,264 rows × 12 columns (Id, Model, Price, …)" — prepended once per document; add one line to the section instruction that ships with attachments: counts and figures about attached data come from the attachment, never from prior knowledge of similarly named datasets.

Ordering: item 1 (duplication) first, then 4, 8, 9, then 2, 3, 5, 7.

## C. Execution report (Sonnet, 2026-09-04, executing the plan above)

No commits made (human-gated per CLAUDE.md). All changes are working-tree edits. Nothing in `data/` or `config.local.yaml` touched.

**Item 1 — attachment bundle duplication.**
Root cause (confirmed, not guessed): `gui/handlers.py` (`_handle_submit_inner`, ~l.4271) builds `ctx.merged_input` via `file_processor.process_files_structured(user_text, files)` — user text + every file's content, ONE copy. The enhanced-mode prelude then called `orchestrator.prepare_prompt(user_input=ctx.merged_input, files=ctx.files, ...)` (`gui/handlers.py:1099`, old). Inside `ContextPipeline.build()` (`core/context_pipeline.py`), Stage 3 (`_process_files`, l.851) unconditionally re-runs `FileProcessor.process_files(user_input, files)` whenever `files` is truthy — i.e. it re-merged the SAME files onto the ALREADY-merged text, appending every attachment's content a second time. `build_full_prompt` then renders `context.file_context` (the doubled string) verbatim into `[CURRENT QUERY]`. This exactly reproduces "CSV→PDF→transcripts, then the identical block again."
Change: `gui/handlers.py:1099-1128` — enhanced-mode `prepare_prompt` now passes `user_input=ctx.analysis_text or ctx.user_text` (raw, pre-merge) and `files=ctx.files`, letting `ContextPipeline.build()` do the ONE canonical merge itself (mirrors how RAW mode already worked). This also fixes item 2 for free (below). Defensive dedupe shipped alongside since the exact double-fire trigger on the frontend side could not be reproduced with certainty: `api/state.py:resolve_uploads` now dedupes `file_ids` preserving order (`dict.fromkeys`); `utils/file_processor.py:process_files_structured` now tracks `(filename, md5(content_text))` signatures and skips appending a same-batch duplicate's text a second time (still counted/persisted once each — only the rendered text is deduped).
Tests: `TestFileProcessorDuplicateGuard` (3), `TestResolveUploadsDedupe` (1) in `tests/unit/test_sep04_attachment_turn.py` — pass.

**Item 4 — deterministic CSV/XLSX manifest.**
Change: `utils/file_processor.py` — new `FileProcessor._tabular_manifest(basename, n_rows, columns)` (staticmethod, ~l.560-585) computes `"[name: N rows × M columns (col1, col2, …). Use only this data for any counts or figures about it — do not rely on prior knowledge of similarly named datasets.]"` from the PARSED `pd.DataFrame`/openpyxl sheet, never raw text; wired into the CSV branch of `_process_single_file` (~l.538-547) and per-sheet into `_extract_xlsx` (~l.696-722, row count = parsed rows minus header). The "don't use prior knowledge" doctrine line lives inline in the manifest (ships with every attachment automatically, no prompt_ctx gating needed — there's no existing "attachments" section-instruction key to hang it on; `section_instructions.py` only gates personal_notes/reference_docs/narrative_state).
Tests: `TestTabularManifest` (4: CSV row count, CSV doesn't fabricate a `1,400`-shaped guess, column truncation, XLSX per-sheet) — pass.

**Item 8 — referenced-but-missing attachment audit.**
New module `utils/attachment_audit.py`: `audit_attachments(user_text, files, documents)`. Two-pass filename extraction (`_extract_filenames`): a "clean" pass requires a word boundary after the extension; a "glued" pass (Canvas-paste fragments) only fires when the extension is immediately followed by another word character AND the preceding stem matches `^([a-z]+)([A-Z]...)$` (lowercase run + TitleCase), trimming to the TitleCase remainder — this is what turns "onHousing.csvHomDownload" into "Housing.csv" while leaving the ambiguous "UsedCars.csvDownload" (no lowercase prefix) alone rather than guessing. Compares against attached basenames via `.orig_name` (api/state.py shim) falling back to `.name`. `_detect_multipart_title` scans each attached doc's first 300 chars for `\bPart\s+(\d+|[IVX]+)\b` / `Homework\s*\d+\s*[-–]\s*\d+`. Wired into `gui/handlers.py:_handle_submit_inner` (~l.4300) right after `files_result` is built; note appended to both `merged_input` (RAW mode + logging) and the new `analysis_text` (enhanced mode).
Tests: `TestAttachmentAudit` (4, incl. the exact fragment from the addendum) — pass.

**Item 9 — deadline timezone conversion.**
`utils/attachment_audit.py:deadline_timezone_note(text, user_tz=None)`. Regex exactly as specified; zone token → IANA mapping (Eastern/Central/Mountain/Pacific + abbreviations + UTC); resolves the user's zone via `utils.timezone_resolver.get_user_timezone()` (lazy import, matches doctrine) unless `user_tz` is passed explicitly (tests use this to avoid touching `data/user_profile.json`); silent on same-zone or no-zone-token. Wired alongside item 8 in `gui/handlers.py`. Belt-and-suspenders prompt line added to `core/tone_instructions.py:get_session_headers_instructions()` (item 6, below) — kept even though item 9 is deterministic, per addendum.
Tests: `TestDeadlineTimezoneNote` (4: Eastern→Central conversion, same-zone no-note, no-zone-token no-note, no-time no-note) — pass.

**Item 2 — classify on the user's own text, not the attachment blob.**
Root cause: `ContextPipeline.build()` was already correctly designed — every classification stage (topics/tone Stage 1-2, intent Stage 4a, heavy-topic Stage 4b, query-rewrite Stage 5, STM Stage 6, STM-refine Stage 6b) reads the `user_input` **parameter**, evaluated before Stage 3 merges file content in. The bug was entirely at the call site: item 1's root cause fed `ctx.merged_input` (already the full blob) into that very parameter. Fixing item 1 (raw text + files into `prepare_prompt`) therefore fixes item 2 for topic/tone/intent/heavy-topic/STM/query-rewrite/STM-refine for free — verified with a test that mocks `stm_analyzer.analyze` to record what `user_query` it was actually called with: with a giant-blob-returning `file_processor.process_files` mock and files attached, STM is still called with the short original text, and `result.original_query` stays short while `result.file_context` carries the full blob.
Residual scope (obsidian keyword search / web-search trigger / agentic gate): these read `context.processed_query` via `UnifiedPromptBuilder.build_prompt_from_context` → `build_prompt(user_input=...)`, which is a DIFFERENT value from `original_query` — it's Stage 3's merged text, or a rewritten query if query-rewrite fired. I could not prove the exact mechanism behind the reported "unrelated 2024 daily note scored 1.00" (`knowledge/obsidian_manager._keyword_search`'s scoring branches mathematically cannot reach 1.0 for a query with zero word-overlap against a short title — verified by hand — so that specific 1.00 most likely came from the embedding/semantic path, which the fix below also protects since it stops pathological text reaching the embedder in the same fallback case). Added `ANALYSIS_QUERY_MAX_CHARS = 2000` in `core/prompt/builder.py` and a bounded fallback in `build_prompt_from_context` (~l.2029-2060): when `processed_query` exceeds 2000 chars AND the untouched `original_query` doesn't, `build_prompt`'s `user_input`/`search_query` use the short original instead — the full content still reaches the rendered prompt via the separate `context.file_context` code path in `build_full_prompt`, untouched. This bounds obsidian/web-trigger/gate/memory-search input in the residual case where query-rewrite doesn't fire (short raw text + huge attachment); it does NOT trace every individual downstream consumer's call site to hard-confirm each one receives the capped value — they all read the same `user_input` local inside `build_prompt`, so this should be correct, but wasn't exhaustively call-site-audited given the size of that method.
Tests: `TestAnalysisTextScoping` (2), `TestAnalysisQueryCap` (3, incl. asserting the 2000-char constant and both the capped and pass-through paths via a stubbed `build_prompt`) — pass.

**Item 3 — same-turn upload dedupe.**
Change: `ContextResult` gained `uploaded_filenames: List[str]` (`core/context_pipeline.py`), populated in Stage 3 from the `files` list's basenames. Threaded through `build_prompt_from_context` → `build_prompt(_uploaded_filenames=...)` → set on `self.context_gatherer._current_turn_upload_filenames` before gather, cleared after (mirrors the existing `_distress_active` pattern) in `core/prompt/builder.py`. `core/prompt/gatherer_knowledge.py:get_user_uploads` (~l.733) now drops any retrieved `type=user_upload` chunk whose stored title (`"upload:<filename>"`, stripped via new `_upload_title_filename`) matches a file attached this same turn — the just-persisted chunk of a file already fully verbatim in `[CURRENT QUERY]` no longer surfaces a third time.
Tests: `TestSameTurnUploadDedupe` (4) — pass.

**Item 5 — redaction phone regex.**
Root cause: the phone regex's separator classes were `[\s.()-]*`/`[\s.-]*` — `\s` matches newlines, and `*` allows unbounded runs, so adjacent numeric CSV cells across a line break could accidentally line up into a phone-shaped 3-3-4 digit run.
Change: `utils/privacy_redaction.py` — separators tightened to a single non-newline character (`[ .()-]?`/`[ .-]?`), matching "phone-shaped" formats (space/dot/hyphen/parens, at most one between groups) without being able to span a newline.
Tests: `TestPhoneRedactionNewlineGuard` (4, incl. the exact "1085\n999  1000" fragment) — pass; existing `tests/unit/test_privacy_redaction.py` (parenthesized and hyphenated real numbers) still green.

**Item 6 — TEMPORAL REASONING prompt line.**
Change: one paragraph added to `core/tone_instructions.py:get_session_headers_instructions()` instructing the model to state the converted local time when a pasted document's deadline timezone differs from the user's own. Belt-and-suspenders alongside item 9's deterministic note.
Test: `TestTemporalReasoningDeadlineLine` (1) — pass.

**Item 7 — test run, honest accounting.**

New file: `tests/unit/test_sep04_attachment_turn.py` — **30 passed**, 0 failed.

Required regression files: `test_sep03_live_probe_fixes.py` + `test_light_prompt_path.py` + `test_intent_semantic_tier.py` + `test_retrieval_context_quality.py` (one combined run) — **119 passed**, 0 failed.

Additional regression sweep (every file touched, run in batches per CLAUDE.md — never the whole suite in one process):
- `test_file_processor.py` + `test_file_processor_security.py` + `test_privacy_redaction.py` + `test_context_pipeline.py` — 90 passed.
- `tests/integration/test_context_pipeline_integration.py` + `test_escalation_gui_wiring.py` + `test_process_user_query.py` — 56 passed.
- `test_handle_submit.py` + `test_turn_progress.py` — 53 passed.
- `tests/test_tone_fix.py` + `test_thinking_blocks.py` + `test_system_prompt_placeholders.py` + `test_orchestrator_profile_injection.py` — 26 passed.
- `tests/test_cross_dedup.py` + `test_full_meta_query.py` + `test_prompt_builder_wrapper.py` + `test_prompt_internal_methods.py` + `test_prompt_sections.py` + `test_thread_surfacing.py` + `test_tone_execution.py` — 39 passed, 6 skipped (pre-existing skip markers, unrelated to this batch: `UnifiedPromptBuilder` can't instantiate in test env / manual integration scripts).
- 14 `core.prompt.builder`-touching unit files (`test_calendar_turn_round2`, `test_codex_followups`, `test_continuation_answer`, `test_entity_resolution`, `test_feature_inventory`, `test_floor_topup_order`, `test_fs_snapshot`, `test_llm_compression`, `test_note_image_gate`, `test_prompt_timeout`, `test_proposal_filter`, `test_sep03_followups_gating`, `test_session_diff`, `test_token_budget_cap`) — 244 passed.
- `test_gatherer_latency_guards.py` + `test_paste_turn_misfires.py` — 36 passed.
- `test_escalation_deescalation_fix.py` + `test_narration_turn_audit_fixes.py` + `test_intent_style_instructions.py` + `test_orchestrator_helpers.py` + `test_grounding_check.py` + `test_style_block_precedence.py` + `test_request_path_parity.py` — 157 passed.
- `test_api_misc.py` — 12 passed.

**Totals: 862 passed, 6 skipped (pre-existing), 0 failed** across every file touched by this change plus the four caller-required regression files. `agent_session_start.sh`/`agent_session_audit.sh` run per CLAUDE.md Agent Session Safety.

**Undone / not fully verified, and why:**
- Item 2's obsidian/web-trigger/gate scoping is fixed via the general `ANALYSIS_QUERY_MAX_CHARS` bound rather than a literal reproduction of the reported "unrelated note scored 1.00" — that exact mechanism could not be reproduced against the deployed `_keyword_search` (mathematically it can't hit 1.0 with zero word overlap; most likely the semantic/embedding path, not unit-testable cheaply here).
- `ctx.merged_input`'s other call sites in `gui/handlers.py` (duel/insight `_dispatch_storage` calls at ~l.1346/2261, agentic-mode locals at ~l.2870/3417) were reviewed but left untouched — they're storage/local-variable uses, not a second rendering of `[CURRENT QUERY]`, and the shared `_prepare_submit_context` prelude (which IS fixed) already produces `ctx.full_prompt` for all of enhanced/duel/agentic/insight modes.
- No commits made (human-gated). Owner should restart the daemon and review this diff before committing.

## D. Fable desktop review (2026-09-04 ~17:00, after the mobile hand-back)

Diff reviewed line by line (8 modified + 3 new files). Kept everything in §C; changed three things:

1. **Deadline note over-fired.** A live probe of the DEPLOYED `deadline_timezone_note` with the real resolver (tests pass `user_tz`, so that path had never run) converted correctly but also emitted a `[DEADLINE NOTE]` for "Office hours are at 3 pm ET on Tuesdays." — the handlers feed the function every attached document, so lecture transcripts would trigger it. Now a zoned time needs a deadline cue in the SAME sentence (`_DEADLINE_CUE_RE` + bare "by" directly before the time; `_SENTENCE_BREAK_RE` stops the windows at a newline / terminator+space+Capital). First cued match wins. +3 tests (transcript stays silent; cued match beats an earlier uncued one; Canvas "Sep 13 by 11:59pm ET" still fires).
2. **Imports hoisted** per the import doctrine: `utils.attachment_audit` in `gui/handlers.py` (module level, beside the other `utils.` imports) and `utils.timezone_resolver.get_user_timezone` in `utils/attachment_audit.py` — cheap internal modules, no cycle, not patch points.
3. **Docs:** CLAUDE.md stats line + utils tree entry + Important Patterns bullet; CLAUDE_CHANGELOG.md entry (newest at bottom); `scripts/generate_doc_metrics.py --update-readme` (817 py files / 8,512 tests / 396 test files); `commit_message.txt` rewritten for this batch.

Verified, not changed: `pd.read_csv` has no row cap (manifest count is the true count); `_persist_uploads` titles are `upload:<filename>` (matches the dedupe); the agentic gate evaluates raw `user_text`, so the notes never reach it; ruff clean on all touched files; exactly one live daemon (pid started 16:28, after the edits — it is running this code). Post-review run: `test_sep04_attachment_turn.py` 33 passed; combined with handle_submit/file_processor/privacy_redaction/context_pipeline/sep03_live_probe/retrieval_context_quality: 222 passed, 0 failed.

Live observation for the owner (not chased): `logs/turn_records.jsonl` has the 16:39:30 and 16:40:41 turns carrying the identical Outlook-inbox query — same text submitted twice 71 s apart (resend, or the ingress dedupe window is shorter than that gap).
