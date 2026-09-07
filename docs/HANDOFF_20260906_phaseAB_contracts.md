# Phase A / Phase B frozen contracts — 2026-09-06 (Fable)

Companion to `docs/HANDOFF_20260906_context_integrity_plan.md` (Codex plan). This file is the
FROZEN contract two delegates implement on DISJOINT files. Fable referees. Nothing here is
committed, applied to live stores, or restarted by an agent.

Rules for both delegates
- Read `CLAUDE.md` "Critical Rules" + "Agent Session Safety" and `docs/DEVELOPMENT_WORKFLOW.md`.
- Only touch the files listed for your phase. If a fix needs a file outside your list, STOP and
  report the exact need (file, function, why) instead of editing it.
- Synthetic evidence only in tests (no real names/meds/places from the owner's stores).
- Run only the focused pytest commands given. `ruff check <changed files>`. `git diff --check`.
- Never add a regex that hardcodes this incident's wording. Structural/grammatical cues only.
- Report: changed files, each new test's FAILED-before / PASSED-after evidence (run the new test
  file against a `git stash`-free approach: write the test first, run it, show the failure, then
  fix), limits, unresolved questions. No scope expansion.

---------------------------------------------------------------------------------------------
## Phase A — evidence transport + truthful receipts

Files (ONLY): `core/prompt/token_manager.py`, `core/prompt/formatter.py`,
`knowledge/web_search_manager.py`, `core/agentic/controller.py`, `core/agentic/types.py`,
`gui/handlers.py` (receipt fields ONLY — `_attach_agentic_provenance`, the telemetry dict, and
the debug-record token-count label), new `tests/unit/test_evidence_transport.py`.

### Verified defects (do not re-investigate)
1. `token_manager._manage_token_budget` non-list branch: `item_text = str(val)` then, when the
   repr exceeds `SEMANTIC_ITEM_MAX_TOKENS` (800) and `val` is not a dict, writes the middle-out
   STRING back (`trimmed[name] = item_text`). A `WebSearchResult` dataclass repr with more than
   ~1 page always exceeds 800 tokens, so the base prompt's web section is dropped by the
   formatter (`formatter._assemble_prompt` ~line 1014 requires `hasattr(web_search,'has_results')`).
   Second-pass trim also treats only `(str, dict)` — structured objects are neither.
2. `controller.run_agentic_search`: when round 1 is NOT a web search (`skip_initial_search`,
   no seed terms, or URL fetch), `initial_context["web_search_results"]` is never rendered into
   `session.accumulated_context`, the decision prompt (`_build_iteration_prompt`), or
   `_build_final_prompt`. `_compute_context_inventory` doesn't even list it. Decision-answer
   reuse then answers from counts + a 4×220-char digest.
3. Receipt: `final_prompt_hash="decision-answer-reuse"` is a sentinel, not a hash; nothing
   records which call produced the answer or what evidence it saw.
4. `_build_final_prompt` labels ALL accumulated tool context `[WEB SEARCH RESULTS - N rounds]`
   and `has_web = bool(session.accumulated_context)` — memory-only loops get the
   "every claim MUST cite [WEB_N]" instruction.

### A1. One shared web renderer (knowledge/web_search_manager.py)
Add pure functions beside `assign_web_ids`:
- `render_numbered_web_sources(pages, *, max_sources=8, max_chars_per_source=2000,
  existing_url_to_id=None, start_index=0) -> tuple[list[str], dict]` — calls `assign_web_ids`
  with the pass-through args, renders each numbered source EXACTLY as the formatter does today
  (`f"[{sid}] **{title}** ({url})\n{content}"`, content clipped to `max_chars_per_source` with
  `"..."`), skips empty content, returns `(lines, source_map)` where `source_map` contains ONLY
  ids that were rendered (a clipped-away/empty page never gets an id in the map).
- `trim_web_search_result(result, *, max_sources, max_chars_per_source) -> WebSearchResult` —
  returns a NEW `WebSearchResult` (`dataclasses.replace`) whose `pages` are the top
  `max_sources` after `assign_web_ids` ranking (dedupe canonical URL, score desc), each page a
  new `WebPage` with `content` clipped to `max_chars_per_source`. Never mutates the input
  (cached originals are shared). Preserves `query`, `from_cache`, `timestamp`, `error`, `search_depth`,
  `total_credits_used`.
Formatter's inline block must call `render_numbered_web_sources` (byte-identical section text;
`context["_web_source_map"]` semantics unchanged). Use a function-body import in
token_manager marked `# lazy import: cycle` if a module-level import would form one; otherwise
module-level.

### A2. Budget manager never converts a structured section to a string (token_manager.py)
- Add `STRUCTURED_SECTION_ADAPTERS: dict[str, tuple[meter_fn, shrink_fn]]` with ONE entry
  `"web_search_results"`. `meter_fn(val, model_name) -> int` = token count of
  `"\n\n".join(render_numbered_web_sources(val.pages, max_sources=8, max_chars_per_source=2000)[0])`
  (i.e. what will actually be rendered). `shrink_fn(val, step) -> val|None` = `trim_web_search_result`
  down a fixed ladder `[(8,2000),(6,1500),(4,1000),(2,800),(1,600)]`; past the end → `None`.
- Non-list branch: `isinstance(val, str)` → existing middle-out; `isinstance(val, dict)` →
  existing (meter, never mutate); registered adapter → meter via `meter_fn`, if over
  `WEB_SEARCH_SECTION_MAX_TOKENS` (new env/const, default 3200) shrink down the ladder until it
  fits; ANY other object type → meter by `str()` but NEVER write back; `logger.debug` once
  with the type name and section. Empty results (`has_results` False) count 0 and pass through.
- Second pass: adapter sections shrink one ladder step per pass; when the ladder is exhausted set
  `None`. `_total_tokens` must meter adapter sections with `meter_fn` (not `str(v)`).
- Dict-shaped `web_search_results` (`{"pages": [...]}`) is NOT an adapter case — leave as today.
- Run `git log -S"trimmed[name] = item_text" --oneline -- core/prompt/token_manager.py` and put
  the introducing commit hash + date in your report (this dates how long the base-prompt web
  section has been silently dropped).

### A3. Agentic loop sees pre-gathered web evidence (controller.py)
After the round-1 branch selection in `run_agentic_search`, when the round-1 branch was NOT a web
search (the `skip_initial_search or not initial_search_terms` branch AND the URL-fetch branch)
and `initial_context.get("web_search_results")` has `has_results`:
- `numbered = self._tool_executor._merge_web_ids(pages)` (guard with `getattr`; a mock executor
  lacking it → skip silently, log debug). Render with `render_numbered_web_sources` using the
  ids from `_current_web_source_map` (pass `existing_url_to_id`/`start_index` so numbering is
  continuous — simplest: call `_merge_web_ids` first, then render the returned numbered sources
  yourself with the same line format; do NOT number twice).
- `self._append_accumulated(session, "[Pre-gathered web results — base retrieval for this turn]\n" + "\n\n".join(lines))`
- `session.seeded_base_web = True` (new field). Do NOT change `_web_mode_no_seed` semantics.
- When round 1 WAS a web search, do nothing (the base result is the same query; the cache reuses it).

### A4. Reuse is permitted only when the answering call saw admitted evidence (controller.py)
`_decision_saw_admitted_evidence(session, initial_context) -> bool`: True when
`session.accumulated_context.strip()` is non-empty, OR none of
`("memories","personal_notes","user_profile","recent_summaries","semantic_summaries",
"recent_reflections","reference_docs","web_search_results","graph_context","relevant_emails",
"google_calendar")` is non-empty in `initial_context`. At the reuse site: if False → skip reuse,
`session.reuse_skipped_reason = "decision prompt lacked admitted evidence"`, fall through to
`_generate_final_response`. Existing reuse tests (no initial_context) must still reuse.

### A5. Receipts (types.py, controller.py, handlers.py)
New `AgenticSearchSession` fields (defaults keep old constructors valid):
`answer_call: str = ""` ∈ {"decision_reuse","final_synthesis","error_fallback"};
`decision_prompt_hash: str = ""` (sha256[:16] of the LAST iteration prompt passed to
`_get_model_decision` — set where it is built); `visible_sources: dict` =
`{"web_ids": sorted ids in _current_web_source_map at answer time, "sections": [...]}` where
sections = the bracketed section names actually rendered into the answering prompt (for reuse:
"[RECENT CONVERSATION — EARLIER TURNS]" if digest present, "Search Results So Far" if
accumulated non-empty, "Context inventory" if present; for final: the `[...]` headers
`_build_final_prompt` appended); `omitted_sections: list[str]` = keys from the A4 list that
are non-empty in initial_context but not rendered to the answering call;
`reuse_skipped_reason: str = ""`; `seeded_base_web: bool = False`.
On reuse: `final_prompt_hash = decision_prompt_hash` (real hash; the sentinel string is removed).
`get_provenance_summary()` adds all new fields. `handlers._attach_agentic_provenance` forwards
`answer_call`, `decision_prompt_hash`, `visible_sources`, `omitted_sections`,
`reuse_skipped_reason`, `seeded_base_web`; telemetry adds `agentic_answer_call`.
Token label: where `_safe_count_tokens`'s result lands in the debug record, add
`"token_count_kind": "local_input_estimate"` beside it (no UI change).

### A6. Source kinds in the final prompt (controller.py)
Header `[WEB SEARCH RESULTS - N rounds]` → `[TOOL RESULTS - N rounds] (each block is labeled
by its source kind: web search, memory, file, computation, fetched page, pre-gathered web)`.
`has_web = bool(getattr(self._tool_executor, "_current_web_source_map", None))` instead of
`bool(session.accumulated_context)`.

### Acceptance tests — `tests/unit/test_evidence_transport.py` (CharacterTokenizer fixture
from `test_independent_prompt_audit.py`; no providers/stores/embedders)
T1 one 12,000-char page + 3 small pages → after `_manage_token_budget(budget=10000)` the value is
   still a `WebSearchResult`, `has_results`, ≤8 pages, content clipped; then
   `PromptFormatter._assemble_prompt` output contains `[WEB SEARCH RESULTS]` and `[WEB_1]`, and
   `context["_web_source_map"]` keys == the ids present in the rendered text (write this test
   FIRST; show it failing on HEAD with `after_type == str`).
T2 shrinking budget (e.g. 900, 300, 50): pages decrease down the ladder, then value is `None`;
   at no step is it a `str`; `_prompt_token_usage <= budget`.
T3 empty result (`pages=[]`), error result (`error="x"`), dict-shaped `{"pages":[...]}`:
   no exception, types preserved, formatter renders nothing for empty/error.
T4 CJK/emoji page content with a 1-token-per-char tokenizer: type preserved, fits budget.
T5 `SimpleNamespace(foo=1)` under `web_search_results` (unknown structured type): never a str
   after budgeting; formatter renders no web section; no exception.
T6 list-of-dict `web_search_results` (the shape `test_independent_prompt_audit` uses) still
   behaves exactly as today (that existing test keeps passing).
T7 seeding: controller fixture from `test_agentic_decision_answer_reuse.py`; bind the REAL
   `ToolExecutor._merge_web_ids` onto the mock executor (`types.MethodType`) with
   `_current_web_source_map = {}`; run with `skip_initial_search=True`,
   `initial_context={"web_search_results": WebSearchResult(query="q", pages=[WebPage(url="https://ex.org/a", title="Alpha Title", content="alpha body", score=0.9)])}`;
   fake `_get_model_decision` captures the prompt → it contains `[WEB_1]` and "Alpha Title";
   `_current_web_source_map` has `WEB_1`; `session.seeded_base_web` True;
   `controller._build_final_prompt(...)` contains it under `[TOOL RESULTS`.
T8 seeding does NOT happen when round 1 was a web search (mock `_execute_search` returning a
   result) — no duplicate block, map has ids only once.
T9 reuse gate: initial_context with `memories=[{"content":"m"}]`, no accumulated → reuse skipped,
   `_generate_final_response` called, `reuse_skipped_reason` set, `answer_call=="final_synthesis"`;
   same but with seeded web (T7 shape) → reuse fires, `answer_call=="decision_reuse"`,
   `final_prompt_hash == decision_prompt_hash` (16 hex chars, not the sentinel).
T10 `get_provenance_summary()` carries every new field; `_attach_agentic_provenance` forwards
   them (use a SimpleNamespace orchestrator).
T11 memory-only loop's final prompt lacks the "MUST include a [WEB_N] citation" line; a loop with
   a populated `_current_web_source_map` has it; header reads `[TOOL RESULTS`.
Focused commands:
```
python -m pytest tests/unit/test_evidence_transport.py tests/unit/test_independent_prompt_audit.py tests/unit/test_web_search_citations.py tests/unit/test_agentic_decision_answer_reuse.py tests/unit/test_agentic_digest_order.py tests/unit/test_agentic_timeout_fallback.py tests/unit/test_token_budget_cap.py tests/unit/test_middle_out_no_grow.py tests/unit/test_budget_meters_rendered_sections.py -q --color=no -p no:cacheprovider
```

---------------------------------------------------------------------------------------------
## Phase B — provenance, temporal claim kind, completion identity, contamination report

Files (ONLY): `memory/fact_source.py`, `memory/relation_classifier.py`, `memory/user_profile.py`,
`utils/completed_plan_claims.py`, `memory/memory_storage.py`, `memory/llm_fact_extractor.py`,
`memory/fact_extractor.py`, new `scripts/report_claim_contamination.py`, new
`tests/unit/test_claim_temporal_kind.py`, new `tests/unit/test_completion_identity.py`,
new `tests/unit/test_extraction_replay_provenance.py`. NOT `gui/handlers.py`, NOT
`memory/shutdown_processor.py` (report needs there instead of editing).

### Verified defects (do not re-investigate)
1. A live profile fact carries an affirmative dose whose `source_excerpt` is a same-day denial;
   the source turn also states an affirmative PRIOR-day dose. `fact_source` now has clause-level
   negation (`_clause_is_negated`, 2026-09-06) — whether the CURRENT code still admits that fact
   is unknown (the record may predate the fix).
2. `user_profile.get_category` treats `is_current=True` as "true now"; expiry is by relation
   NAME only (`relation_classifier.ephemeral_ttl_hours`). An EVENT ("had drinks last night")
   stored under a durable-looking relation projects to today forever.
3. `completed_plan_claims.completed_by_user`: completion cue anywhere in the statement + ≥2 shared
   tokens + weak digit guard. An unrelated statement ("attached the exercise about completed
   tasks") closed a real-world plan.

### B1. Replay first (tests/unit/test_extraction_replay_provenance.py) — REPORT before fixing
Synthetic, generalized messages (never the owner's wording), e.g. user turn:
"Took an extra 5mg of Zelphex yesterday. Didn't take any today." (Zelphex is the repo's
fictional med name — see existing tests) and a proposed triple `user | medication_dose | 5mg today`
(and variants: object "extra 5mg", "5mg Zelphex", reversed clause order, contraction "Didn't"/"did not",
implicit subject "Skipped it today. Had the extra 5 yesterday.").
Drive: (a) `fact_source.find_supporting_user_span` directly; (b) the per-turn path
`MemoryStorage.extract_and_store_facts` with a stub `fact_extractor` returning the bad triple and
a recording stub profile/chroma; (c) the shutdown LLM path's normalize/provenance join in
`llm_fact_extractor` with the same stub LLM output. Assert: the triple is never stored with the
today-clause as excerpt; an affirmative "yesterday" fact, if admitted, carries the yesterday
clause as `source_excerpt`.
STOP CONDITION: if (a)-(c) already reject the bad fact on HEAD, keep the tests as regression,
say so explicitly in your report, and do NOT add parser patches. Then B8's report is the
repair path for the legacy record.

### B2. Claim temporal kind + event date (fact_source.py; leaf helpers only)
Add `classify_claim_time(span: str, *, observed_at: datetime|None) -> ClaimTime` (dataclass:
`kind` ∈ {"event","state","habit","plan","unknown"}, `event_date: date|None`,
`event_date_source` ∈ {"explicit","relative","none"}). Grammar-level cues ONLY:
habit = habitual adverbs/quantifiers (every/each/daily/nightly/weekly/usually/always/
typically/most days) or "I <verb>" simple-present with a frequency phrase; event = past-tense
cue + a temporal anchor (yesterday, last night/week/…, this morning, N days ago, on <weekday>,
an explicit date) resolved via `utils.temporal_resolver.resolve_date_expression` relative to
`observed_at` (unknown → `event_date=None`, kind still "event" if past-tense + anchor);
plan = will/going to/plan to/tomorrow/next <unit>; state = present copula/progressive
("I am", "I've been", "I feel") with no habit cue; else unknown. Under-fire by design.
Attach to `EvidenceSpan` (new optional fields) and forward via the existing metadata plumbing
in BOTH extractors as `claim_kind`, `event_date` (ISO or ""), `observed_at` (ISO). Do not
change any existing key.
Read side (`user_profile.get_category`, default `include_historical=False`): additionally skip a
fact whose `claim_kind == "event"` and `event_date` is before today (a past event is not current
state); habits/states/unknown untouched; `include_historical=True` returns everything. Legacy
facts (no `claim_kind`) behave exactly as today.
`relation_classifier`: no new relation names. If B1 shows a consumption/event relation that is
durable by name, do NOT add it to the ephemeral lists — B2's event kind is the general fix; note
it in the report.

### B3. Test-block origin (fact_source.py + memory_storage.py)
Extend the existing `[relay: …]…[/relay]` convention with `[test]…[/test]`: `fact_extraction_source_text`
strips test blocks (like quoted correspondence), so genuine commentary outside the block still
yields facts; `MemoryStorage.store_interaction` stores `origin="test"` in the corpus/Chroma
metadata when the user text contains a test block (otherwise no new key). Nothing infers "test"
from wording, repetition, or a medication name.

### B6/B7. Completion identity (utils/completed_plan_claims.py)
Replace the token-overlap match in `completed_by_user` with identity + status:
- Plan side: `_plan_object_head(plan_sentence)` = head noun of the plan's object phrase (last
  content token before a trailing temporal/adverbial tail); plans with a habitual cue ("every",
  "each week", "-ly" frequency) are NOT plans — `plan_sentences` must skip them (a completed
  occurrence never closes a recurring plan).
- Statement side: split into clauses (reuse `memory.fact_source._split_clauses` — it is a leaf
  import, allowed); a qualifying clause has the completion cue AND the plan's object head (or a
  determiner-anchored reference "that/the/my <head>"; bare "it"/"got it done" only when the
  immediately previous user statement on the same day names the head); the cue must not be
  governed by negation/"not yet"/"haven't"/"still need to"/"if"/"would"/"planning to" in that
  clause; "cancelled/canceled/rescheduled/moved/tried to/attempted/couldn't" in the clause →
  NOT completed (distinct status, no match); document-sharing verbs ("attached", "uploaded",
  "here's the file", "sent you") whose object is a document noun never complete a non-document
  plan. Keep the digit guard; keep same-day ordering by timestamp when available.
- Caution line quotes the exact qualifying CLAUSE and its date (not the statement's first 160 chars).
Held-out tests (`tests/unit/test_completion_identity.py`), each with a counterpart control:
appointment scheduled ✓ vs "attached the exercise about completed tasks" ✗; "got the car fixed"
✓ vs "haven't gotten the car fixed yet" ✗ vs "cancelled the repair" ✗; "paid the electric bill"
✓ vs "paid the water bill" ✗ (object head mismatch); recurring "gym every Tuesday" never a plan;
"that appointment is done" ✓ via determiner reference; "got it done" ✓ only with the head in the
previous same-day statement; pronoun/contraction/reordered-clause variants of each.
All existing `tests/unit/test_completed_plan_claims.py` cases must keep passing (the six
Codex-audit scenarios are still valid: the positive there is a genuine social plan completed).

### B8. Dry-run contamination report (scripts/report_claim_contamination.py) — READ-ONLY
`--profile PATH` (default `data/user_profile.json`), `--out PATH` (JSON), no `--apply` at all.
For every `is_current` fact: run the DEPLOYED `fact_source.find_supporting_user_span` with the
fact's own `source_excerpt` as the only user message → `unsupported_by_own_excerpt` when None;
`negated_object_clause` when `_clause_is_negated` on the excerpt's object clause; `past_event_current`
when B2's classifier on the excerpt yields event with `event_date` < today; `origin_test` when the
excerpt contains a `[test]` block. Output rows: fact_id, category, relation, value (truncated
80 chars), reasons, proposed_action ∈ {"supersede","quarantine","review"}, dependent hints
(graph edge key `subject|relation|object`). Print a count table. Use `utils.daemon_guard` only to
WARN (read-only). Test with a synthetic profile written to `tmp_path`.
Focused commands:
```
python -m pytest tests/unit/test_extraction_replay_provenance.py tests/unit/test_claim_temporal_kind.py tests/unit/test_completion_identity.py tests/unit/test_completed_plan_claims.py tests/unit/test_fact_source_negation_scope.py tests/unit/test_sep06_evidence_ownership.py tests/unit/test_fact_source.py tests/unit/test_user_profile_schema.py tests/unit/test_profile_confirmation_recurrent.py tests/unit/test_chore_relation_hygiene.py -q --color=no -p no:cacheprovider
```

---------------------------------------------------------------------------------------------
## Results (Fable referee, 2026-09-06 evening) — ALL UNCOMMITTED, daemon not restarted

- Phase A: accepted as delivered (17 tests; 271 green incl. guards + handler tests). Live-reachable
  since e0f08e0 (2026-03-26). Receipts expose that `_build_final_prompt` never renders
  graph_context / relevant_emails / google_calendar (follow-up, not fixed here).
- Phase B: B1 = STOP CONDITION (HEAD already rejects the false dose on all three paths → the live
  record is legacy; run B8 against the real profile for owner review). Referee fixes applied on
  top: explicit-date year rollback for past-tense claims, `extract_facts(observed_at=)` threading
  (shutdown passes the turn timestamp), shutdown `src_dict` forwards claim_kind/event_date/
  observed_at. 371 green across B focused + touched-module suites + guards.
- Deferred: B4 correction propagation; phases C (intent/calibration), D (context value),
  E (telemetry); `[🔍 FINAL MESSAGE PAYLOAD]` DEBUG dump in handlers bypasses the 09-02 privacy gate.
- Owner next: `python scripts/report_claim_contamination.py --out data/claim_contamination_20260906.json`
  (read-only), review; restart daemon; commit (commit_message_3.txt).
