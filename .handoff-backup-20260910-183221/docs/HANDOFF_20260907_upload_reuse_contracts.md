# Fable contracts — 2026-09-07 four-item bundle (upload retrieval, reuse gate, STM clock, intent regex)

Source: six-turn debug dump of 2026-09-07 01:29–10:06 (turns 1–6), traced against the
deployed functions and the live log `daemon_debug_20260907_013125.log`. Owner approved
the bundle ("lets do it"). Rules as in Phase A/B: agents never commit, restart, or write
stores; synthetic data only in tests; `DAEMON_TEST_MODE=1` on every pytest run; no full
suite; disjoint files per delegate.

## Verified root causes

R1. `ContextGatherer.get_user_uploads` pools `get_documents(query, limit=10)` over the
    WHOLE `reference_docs` collection (1,644 chunks; 242 are uploads) with no type
    filter, keeps the user_upload survivors, then gates them. Deployed-function replay
    on the turn-5 query: `upload:Homework1-2.pdf` (uploaded 2026-09-05 13:43, fresh) is
    absent from the top-60 semantic hits; the only upload in the 10-slot pool was the
    wrong course's syllabus. Turn 6: five image stubs at 0.66–0.67 filled the pool.
    There is no title/keyword leg, so "first assignment" never meets "Homework1-2.pdf".
R2. The `search_memory` tool definition (`core/agentic/types.py`) says
    `reference_docs: Your own architecture/documentation`. The loop therefore searched
    obsidian_notes/conversations five times (turn 5) and listed the REPO with
    file_list/file_grep (turn 6), then said "I checked the uploads directory".
    `get_full_document(title)` exists but the model is never shown upload titles.
R3. `_decision_saw_admitted_evidence` (Phase A, A4) returns True whenever
    `session.accumulated_context` is non-empty. Turn 6: repo file-listing output counted
    as "evidence seen" while the base retrieval's 3 memories, 5 notes and 5 uploads never
    reached the decision prompt → `answer_call=decision_reuse` shipped a narration.
    Turn 4 (no tool results) correctly fell back.
R4. The STM prompt carries no clock. Turn 3 (08:31) `temporal_facts` said "Current time
    is approximately 1:31 AM", copied from Daemon's 01:31 reply; the planner repeated it.
R5. `intent_classifier` CREATIVE_EXPLORATION regex matches bare `idea(s)?`; "give me an
    idea of what we're working with" → creative_exploration 0.75 → CREATIVE style block
    on a document-lookup request.

## Delegate A — upload retrieval (files: knowledge/reference_docs_manager.py,
## memory/storage/multi_collection_chroma_store.py, core/prompt/gatherer_knowledge.py,
## core/prompt/formatter.py [USER UPLOADED ITEMS render site only], core/agentic/types.py,
## core/agentic/tools.py, core/agentic/formatters.py; tests/unit/test_upload_retrieval_pool.py)

A1. `MultiCollectionChromaStore.query_collection(..., where=None)`: an explicit `where`
    kwarg is forwarded to `collection.query(where=...)` when not None. No behavior change
    when absent. (Currently `**kwargs` swallows it silently.)
A2. `ReferenceDocsManager.get_documents(query, limit=10, *, doc_type=None)`: when
    `doc_type` is given, BOTH legs are restricted to that type — semantic via A1's
    `where={"type": doc_type}`, keyword via a metadata filter in `_keyword_search`
    (add the same keyword-only `doc_type` parameter). Default `None` = byte-identical to
    today for every existing caller.
A3. `get_user_uploads` calls `get_documents(query, limit=limit * 2, doc_type="user_upload")`.
    The post-hoc `type == 'user_upload'` filter stays as a belt-and-suspenders no-op.
    Everything after it (same-turn dedupe, `_upload_is_live`, cap) is unchanged.
A4. Fresh-upload roster leg (generic, no vocabulary): when the query carries a document
    cue (reuse the SAME `document_context` regex `_upload_is_live` already uses — hoist it
    to a module-level compiled constant, do not duplicate it) OR the query contains a
    filename-shaped token (`\w+\.(?:pdf|docx?|csv|xlsx?|txt|md|json|png|jpe?g)\b`), the
    gatherer additionally fetches the distinct upload titles (metadata-only
    `collection.get(where={"type":"user_upload"}, include=["metadatas"])`, cached per
    turn) whose timestamp is fresh (`_upload_is_fresh`), sorted newest-first, capped at
    `USER_UPLOADS_ROSTER_MAX` = 8 (env override), and attaches them to the result as a
    roster: `self._last_upload_roster = [{"title": <display filename>, "date": "YYYY-MM-DD"}, ...]`
    exposed to the builder/formatter through whatever mechanism the section already uses
    for its list (delegate: find the [USER UPLOADED ITEMS] render site in formatter.py and
    render ONE trailing line):
      `Recently uploaded files (full text retrievable by title with get_full_document): Homework1-2.pdf (2026-09-05), UsedCars2.csv (2026-09-05), …`
    Image stubs are excluded from the roster. The roster renders even when the semantic
    leg admitted nothing (the section then contains only the roster line). Titles are the
    display filename (strip the `upload:` prefix; `tmp…` legacy names render as-is).
    Never load document content for the roster — metadata only.
A5. Tool doc truth (`core/agentic/types.py` MEMORY_SEARCH_TOOL_DEFINITION and the XML
    protocol text in `core/agentic/protocols.py` if it lists collections):
    `reference_docs: Your own architecture/documentation AND the user's uploaded files
    (homework, syllabi, datasets, PDFs; stored titles look like "upload:<filename>") —
    search here for anything the user attached or uploaded, then call get_full_document
    with the title to read it whole.` `_execute_memory_search` result rendering for
    reference_docs hits must show the title (verify in `core/agentic/formatters.py`; add
    it if absent, e.g. `[title: upload:Homework1-2.pdf]` on the hit header).
A6. Tests (`tests/unit/test_upload_retrieval_pool.py`, synthetic fixtures, no live store):
    - A1: `where` forwarded to `.query`; absent → not passed.
    - A2: `doc_type` restricts both legs; `doc_type=None` returns the same ordering as
      before (regression pin with a fake store whose `query_collection` records kwargs).
    - A3: with 1,000 fake reference_doc chunks outranking one fresh upload, the upload is
      admitted (pool restricted to uploads). Pin the pre-fix behaviour as the failing
      shape in the test docstring.
    - A4: roster appears on "pull up the first assignment" and "look in the user uploads"
      (document cue), absent on "how are the cats", capped at 8, newest-first, image stubs
      excluded, stale titles excluded, metadata-only (fake collection asserts
      `include=["metadatas"]`).
    - A5: tool definition string mentions uploads; formatter shows the title for a
      reference_docs hit with an `upload:` title.
    Run: `DAEMON_TEST_MODE=1 python -m pytest tests/unit/test_upload_retrieval_pool.py
    tests/unit/test_upload_keyword_score_leak.py tests/unit/test_sep04_attachment_turn.py
    tests/unit/test_refdocs_lazy_collection.py tests/unit/test_gatherer_latency_guards.py
    tests/unit/test_retrieval_context_quality.py tests/unit/test_tool_wiring_parity.py
    tests/unit/test_budget_meters_rendered_sections.py -q --color=no -p no:cacheprovider`

## Delegate B — reuse gate, STM clock, intent regex (files: core/agentic/controller.py,
## core/agentic/types.py is OFF-LIMITS (A owns it), core/stm_analyzer.py,
## core/context_pipeline.py [only to thread `now` if needed], core/intent_classifier.py;
## tests/unit/test_sep07_reuse_stm_intent.py)

B1. `_decision_saw_admitted_evidence`: tool results alone never prove the base evidence
    was seen. Split `_ADMITTED_EVIDENCE_KEYS` into two class-level tuples:
    `_BACKGROUND_EVIDENCE_KEYS = ("user_profile", "recent_summaries", "recent_reflections")`
    and `_RETRIEVAL_EVIDENCE_KEYS` = every other current key PLUS the user-uploads context
    key (delegate: confirm its exact name in the gatherer/builder — the section renders
    as [USER UPLOADED ITEMS]; add it if it is missing from the tuple). Keep
    `_ADMITTED_EVIDENCE_KEYS` = background + retrieval so receipts (`omitted_sections`)
    are unchanged. New rule: return True iff NO retrieval key is non-empty in
    initial_context (background-only context may be reused when tool results exist, as
    before). Any non-empty retrieval key → False → full synthesis, regardless of
    accumulated_context. `reuse_skipped_reason` names the first non-empty retrieval key:
    `"decision prompt lacked admitted evidence: memories"`.
B2. `_usable_decision_answer`: additionally reject a candidate whose text, after
    sanitization, is question-dominated — ≥2 lines that end with "?" AND fewer than
    2 lines that do not (a "Can you tell me: 1. … 2. … 3. …" clarification list is not a
    final answer when admitted evidence exists; log reason "question-dominated"). Do
    not touch `_PROMISSORY_RE`/`_LOOP_META_RE`.
B3. STM clock anchor (`core/stm_analyzer.py`): `analyze(..., now: Optional[datetime] = None)`
    (default `datetime.now()`); the prompt gains, directly above `Current user query:`,
    the line `Current time: <Weekday, YYYY-MM-DD HH:MM> (authoritative — the ONLY source
    for the present time)`. Add disambiguation rule 8: "temporal_facts must never
    restate a clock time, date, or elapsed-time figure taken from an earlier ASSISTANT
    reply; a time-of-day fact comes only from the user's CURRENT message or the Current
    time line above. Older exchanges carry [relative] prefixes — treat them as past."
    Thread the pipeline's clock if `ContextPipeline` already has one in scope
    (TimeManager / `mark_query_time`); otherwise leave the default. Every existing
    `analyze` caller keeps working unchanged.
B4. Intent regex (`core/intent_classifier.py` CREATIVE_EXPLORATION): replace bare
    `idea(s)?` with `(?<!no )(?<!any )\bideas?\b(?!\s+(?:of|what|how|why|whether|if|about|where|when)\b)`
    inside the same alternation (keep every other alternative). Keyword map entry
    `"idea"` (line ~480) is used by a different path — inspect it; if it can classify
    "give me an idea of X" alone, guard it the same way, otherwise leave it.
B5. Tests (`tests/unit/test_sep07_reuse_stm_intent.py`):
    - B1: initial_context with only user_profile + accumulated tool text → reuse allowed;
      with `memories` non-empty + accumulated tool text → False and reason names
      `memories`; with the uploads key non-empty → False; empty initial_context → True.
      Also extend/keep green `tests/unit/test_agentic_decision_answer_reuse.py` and
      `tests/unit/test_evidence_transport.py` (update any test that pinned the old
      tool-results-suffice rule; explain in the docstring).
    - B2: the exact turn-6 reply text ("I checked the uploads directory … Can you tell
      me: 1. … 2. … 3. …") → None; a long answer with one trailing question → unchanged.
    - B3: prompt contains the Current time line and rule 8; `now=` value is rendered;
      default path does not crash; the last-assistant-reply text "yours currently reads
      1:31 AM" appears in the prompt but the rule text follows it.
    - B4: "give me an idea of what we're working with" → not creative (general or the
      regex's other winner); "no idea what to do" / "any idea why" → not creative;
      "I have an idea. For later on." / "brainstorm some ideas" / "what if we …" → creative.
    Run: `DAEMON_TEST_MODE=1 python -m pytest tests/unit/test_sep07_reuse_stm_intent.py
    tests/unit/test_agentic_decision_answer_reuse.py tests/unit/test_evidence_transport.py
    tests/unit/test_narration_turn_audit_fixes.py tests/unit/test_agentic_digest_order.py
    tests/unit/test_stm_gate_recent_history.py tests/unit/test_stm_new_data_override.py
    tests/unit/test_intent_classifier.py tests/unit/test_intent_semantic_tier.py
    tests/unit/test_intent_style_instructions.py -q --color=no -p no:cacheprovider`

## Stop conditions (either delegate)
- A contract line cannot be met without touching the other delegate's files → stop, report.
- An existing test pins the opposite of a contract line and the docstring gives a reason
  you cannot refute → stop, report the test name.
- Any need to read the live stores or run the daemon → stop (tests are synthetic only).

## Results (Fable referee, 2026-09-07)

Both delegates delivered on contract. Referee deviations, all in the same commit:

- A4: the roster marker is inserted FIRST in the returned list (the token budget's list trim
  breaks at the first item that does not fit, so a trailing marker died behind any oversized
  chunk); the formatter numbers real items with its own counter.
- A3 side effect: once the pool is upload-only the keyword leg surfaces content-word-overlap
  hits (0.2–0.4); `_upload_is_live` refuses keyword hits below 0.6 on the freshness leg unless
  the query names the file (a live replay had admitted three lecture transcripts at 0.35).
- B1: the literal rule also blocked on `web_search_results` when A3 had SEEDED it into the
  decision prompt, undoing A3; `session.seeded_base_web` exempts that one key.
- B2: the live turn-6 reply had six non-question lines around its numbered list, so the
  line-ratio rule alone let it through; an enumerated-question-list arm was added.

Retest round 2 (11:14/11:15, after the owner's restart) exposed two more layers, fixed the same
hour: the agentic loop never SAW the roster (context inventory omitted `user_uploads`;
`_build_final_prompt` never rendered it; a reference_docs semantic search returns syllabus chunks,
not the homework) → inventory roster line, `[USER UPLOADED ITEMS]` in the final prompt, exact
upload-title listing appended to every reference_docs memory search; and the gate never fired on
"Can we look in the user uploads for the MGT 6203 homework…" → `uploads`/`attachments`
self-anchor `is_personal_doc_search`, `_REQUEST_SHAPED_RE` accepts "can we".

Retest round 3 (11:23/11:24) PASSED live: both turns `triggered: tools`,
`get_full_document: Homework1-2.pdf` in round 1, `answer_call=final_synthesis`, task 1 quoted.

Also fixed alongside: weak visual verbs need a non-document message; `daily_note_catchup.py`
logs to `logs/daily_note_catchup.log` (it had renamed the live daemon log nightly).

Tests: `tests/unit/test_upload_retrieval_pool.py` (28), `tests/unit/test_sep07_reuse_stm_intent.py`
(12), `tests/unit/test_sep07_followups.py` (27); `test_evidence_transport.py` seeded-web case
restored plus a seeded-web-plus-memories case. Still deferred: B4, C1, C5, D3, E, the agentic
final-prompt omissions (graph/emails/calendar), and the handlers payload DEBUG dump.
