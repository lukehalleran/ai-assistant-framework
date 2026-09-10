# Handoff — web-search gap: news shares never reach the trigger, "search came back empty" confabulated (Codex executes, Fable referees)

Plan author: Fable, 2026-09-10 afternoon. Executor: Codex. Referee: Fable.
Nothing here is committed, applied, or restarted by an agent. Unit lane only
(`tests/unit`, daemon may be running). No store writes, no live provider calls
from tests, no daemon restart.

## STATE

- HEAD `8dafdf9` (docs drift audit), pushed 2026-09-10 ~13:55. Tree clean.
- Daemon: DOWN since 13:48:42 (owner restarts after this batch lands).
- Owner-reported dump (2026-09-10 13:36–13:46, 4 enhanced turns) + yesterday
  19:47. Credits 6/100, `TAVILY_API_KEY` present, `_web_search_enabled()` True,
  `WebSearchManager.is_available()` True in the live process ("Tavily client
  initialized" 13:39:24). The Tavily layer is NOT the defect.
- Follow-up row 11 in `docs/HANDOFF_20260909_audit_followups.md` is the
  one-paragraph version of this handoff; this file supersedes it.

## Verified root causes (each pinned to a log line + code)

1. **Statement-shaped news share never reaches any classifier.**
   `daemon_debug.log` 13:46:25: `[WebSearchTrigger] Query: 'The president says
   he will pay everyone 5000 if hi...' | confidence=0.00 | should_search=False
   | reason=No strong indicators` then `Skipping LLM: heuristic confident
   no-search (conf=0.0, no keywords)`; gate: `LLM trigger: should_search=False`.
   - `utils/web_search_trigger.py:448 requires_fresh_public_evidence` returns
     False on the exact text: it needs `temporal AND question AND public_event`
     (deliberately conservative — dated QUESTIONS, Astra 2026-09-05). A share
     is a statement: `presidents?` hits `_PUBLIC_EVENT_STRONG_RE` (:440) but
     there is no temporal cue and no question shape; `says` is in neither
     event regex.
   - `should_search_heuristic` (:650) then scores 0.0 with no keywords, and
     `analyze_query_with_llm` (:1720–1745) SKIPS the LLM on exactly that
     combination unless `query_depends_on_context` (:1129, pronoun regex
     :1080) fires with context. The message has no pronoun → skipped.
   - The gate's Tier-1 arm (`core/agentic/gate.py:848`) uses the same rule;
     its Tier-4 LLM path is the same `analyze_query_with_llm` → same skip.
   - Even if consulted, the trigger prompt (:1226–1240) lists "Statements of
     opinion, feeling, or reaction" under NEVER SEARCH and requires "an actual
     information need" — a relayed public claim with "Oh boy" attached is
     ambiguous under those rules. The prompt must distinguish the user's OWN
     feelings from a RELAYED public claim.
2. **"Please investigate" skipped the same way, then the model narrated a search
   that never ran.** `daemon_debug_20260909_221600.log` 19:47:10: heuristic
   0.0 / no keywords / `Skipping LLM`; 19:47:15 agentic pre-check enabled but
   no tool loop (first stream chunk 19:47:19 already says "back empty"). The
   reply: "my web search came back empty this turn". `investigate` is in no
   keyword list (`_INFO_SEEKING_CUES` gate.py:1484 has confirm/verify, not
   investigate), and "Uhm. Please investigate thank you" has no referential
   pronoun, so context (the pasted IRGC carrier claim in the prior turn, which
   `_build_recent_context` gate.py:637 DOES include) was never consulted.
3. **`[ACTIVE FEATURES] web_search=ON(0)` reads as "a search ran and found
   nothing".** `core/prompt/formatter.py:757 _count` renders `(0)` for an empty
   list/str whether a search ran or not; `:792` builds the label from the
   RESULTS key, not from any decision. Every turn in the dump shows `ON(0)`.
   Probe of the deployed `_build_feature_inventory`: `None` → `web_search=ON`,
   `''` → `web_search=ON(0)` — so something upstream coerces the untriggered
   `None` (builder.py:1697/2274/2333 set None) to `''` before the formatter;
   the site is NOT yet identified (see PLANNED 0).
4. **No sensor.** `logs/turn_records.jsonl` has no web-trigger fields at all
   (`utils/turn_telemetry.py` docstring lists none; `orchestrator.py:1680
   _last_turn_signals` carries intent/tone/plan only). The gatherer's
   `decision` object dies inside `_get_web_search_results`
   (`core/prompt/gatherer_web.py:59–150`). `tests/unit/
   test_public_news_verification.py` covers dated questions + request wrappers
   only; no statement-shaped share, no verification imperative, no label test.

## PLANNED (in this order; write the failing tests FIRST and record the counts)

Regression fixtures — use these EXACT strings, they are the live texts:
- `T1 = "The president says he will pay everyone 5000 if his party wins midterm. Oh boy"`
- `T2 = "Uhm. Please investigate thank you"` with `conversation_context` =
  a prior user turn containing a pasted public claim (synthetic: "BREAKING:
  IRGC says it struck a US carrier in the Gulf on Sept 5" + a Daemon reply).

0. **Find the `''` coercion** (≤20 min, then move on): drive the deployed path
   with `web_search_results=None` — `PromptFormatter._build_feature_inventory`
   gives `ON`, so search upward: `formatter.py` between :1010 and :1634
   (does anything assign `context["web_search_results"] = ""`?),
   `core/orchestrator.py` `build_full_prompt`/`_assemble_prompt` (any
   `v or ""` / `str(v)` pass over `prompt_ctx`), the light path consumers
   (builder.py:2274). Record the site in this file. Do NOT fix it by special-
   casing the label; step 3 replaces the label's data source entirely.

1. **Tests first** — new `tests/unit/test_sep10_web_search_gap.py`
   (isolate `_llm_trigger_cache`, `_semantic_search_boost`, `LLM_FIRST_ENABLED`
   exactly like `test_public_news_verification.py`'s fixture; stub
   `_classify_with_llm_unified_shared` with an AsyncMock; never a live call):
   a. `public_actor_statement(T1) is True`; also True for "The minister announced
      a new tax yesterday", "Congress passed the sanctions bill", "Police
      arrested the mayor". False for: "I feel like the president is terrible"
      (first person), "my company announced layoffs" (private-sphere token),
      "The president was charged this week?" (freshness rule owns it — assert
      `should_search_heuristic(...).source == "freshness_rule"`), "nice",
      "thanks", the 08-15 vent shapes from
      `tests/unit/test_tone_arbiter_hardening.py::TestVentShapeNarrowing`.
   b. `should_search_heuristic(T1)` → `should_search is False`,
      `consult_classifier is True`, `confidence == 0.0`, `matched_patterns == []`
      (NOT a suppression marker — see WHY), `reason` mentions
      "public-actor statement".
   c. `analyze_query_with_llm(T1, ...)`: the stub IS awaited (it was skipped
      before — this test must FAIL on HEAD); stub returns should_search=True
      → result `should_search is True`, `source == "llm"`; stub returns False
      → `should_search is False` and `adaptive_exemplars.get_store().record`
      was NEVER called (no teaching from a classifier verdict).
   d. `analyze_query_with_llm("I feel way more fucked up then unusual do when I
      go just one night poor sleep. Is the benedryl still affecting me?", ...)`
      → stub NOT awaited (personal state; unchanged skip). Same for "Hey".
   e. `is_verification_request(T2) is True`; also "verify this", "fact check
      that", "look into it please", "is this real?"; False for "I'll
      investigate later" (user's own plan), "investigate how memory gating
      works" (>12 words / knowledge shape — leave to existing routing), "ok".
      `analyze_query_with_llm(T2, conversation_context=<prior claim>)` → stub
      awaited; with `conversation_context=None` → NOT awaited (nothing to
      resolve against).
   f. Gate: `_is_info_seeking("Please investigate thank you") is True`
      (currently False — assert on HEAD first).
   g. Label: `_build_feature_inventory({...})` with
      `web_search_decision={"triggered": False, "reason": "No strong
      indicators", ...}` → contains `web_search=ON(not triggered)`; with
      `{"triggered": True, "results": 0}` → `ON(0 results)`; `results: 3` →
      `ON(3 results)`; `{"error": "timeout"}` → `ON(error)`; decision ABSENT
      and `web_search_results in (None, "", [])` → `ON(no search this turn)`;
      the literal `web_search=ON(0)` never appears in any case
      (`assert "ON(0)" not in out` on every branch).
   h. Telemetry: after `gatherer._get_web_search_results(...)` returns on the
      not-triggered path, `gatherer.last_web_decision == {"triggered": False,
      "source": ..., "reason": ..., "confidence": 0.0, "results": None,
      "error": None}`; on an exception path `error` is set; `record_turn`
      round-trips `web_trigger_should_search/source/reason/confidence`,
      `web_results_n`, `web_error` (extend `tests/unit/test_turn_telemetry.py`).
   Record: `pytest -q tests/unit/test_sep10_web_search_gap.py` → N failed on
   HEAD (expect a, b, c, e, f, g, h red; d green).

2. **Trigger** (`utils/web_search_trigger.py`):
   - `WebSearchDecision` gains `consult_classifier: bool = False` (dataclass
     default; `replace()` keeps working).
   - New `_REPORTED_SPEECH_RE` (word-bounded): `says|said|claims?|claimed|
     announce[sd]?|promise[sd]?|vow(?:s|ed)?|threaten(?:s|ed)?|declare[sd]?|
     will\s+pay|plans?\s+to|signed|ordered|tweeted|posted|passed|struck`.
   - New `public_actor_statement(query) -> bool`: reuse the freshness rule's
     wrapper strip + first/second-person exclusion + `is_personal_doc_search`
     + `is_personal_state_statement` vetoes; require `_PUBLIC_EVENT_STRONG_RE`
     hit AND `_REPORTED_SPEECH_RE` hit in the remainder AND ≥5 words; NO
     temporal/question requirement (that is the whole point). Keep it a
     leaf function; put it right after `requires_fresh_public_evidence`.
   - In `should_search_heuristic`, AFTER the freshness-rule return and the
     suppression check: if `public_actor_statement(query)`: return
     `WebSearchDecision(should_search=False, confidence=0.0, reason="Public-
     actor statement — consult classifier", matched_keywords=[],
     matched_patterns=[], consult_classifier=True, source="heuristic")`.
   - In `analyze_query_with_llm` (:1720): the conf=0.0 skip becomes
     `if heuristic_result.confidence <= 0.0 and not
     heuristic_result.matched_keywords and not
     heuristic_result.consult_classifier:` — the consult flag joins the
     referential-follow-up exemption; add a log line
     `"[WebSearchTrigger] conf=0.0 public-actor statement — consulting LLM"`.
   - New `is_verification_request(query) -> bool`: head-anchored (after an
     optional filler `uhm|um|ok|okay|hmm|so|please`) imperative
     `(?:please\s+)?(?:investigate|verify|fact[- ]?check|look\s+into\s+
     (?:it|this|that)|check\s+(?:this|that|it)(?:\s+out)?|is\s+(?:this|that|it)
     \s+(?:true|real|legit))`, ≤12 words, not `is_personal_state_statement`,
     no first-person plan cue (`i'?ll|i will|i'?m going to|later`). In the
     same skip block: `_referential_followup = bool(conversation_context …)
     and (query_depends_on_context(query) or is_verification_request(query))`.
   - Trigger prompt (:1226 SEARCH-if list): add one bullet — "a claim about a
     public figure, institution or event that the user RELAYS or reacts to
     ('The president says he will…', 'X announced Y', a pasted headline,
     'investigate'/'verify' after such a paste): search to VERIFY the claim
     even when it is phrased as a statement or carries a reaction ('oh boy',
     'wtf'). The NEVER-SEARCH rule for opinions/feelings is about the user's
     OWN state, not a relayed public claim." Keep "when in doubt, false".
   - Do NOT touch `requires_fresh_public_evidence`, the gate's Tier-1 forced
     arm, `apply_intent_veto`, or any adaptive-exemplar teacher.

3. **Gate** (`core/agentic/gate.py`): add `"investigate"`, `"fact check"`,
   `"fact-check"` to `_INFO_SEEKING_CUES` (:1484). Nothing else.

4. **Honest label + prompt rule**:
   - `core/prompt/gatherer_web.py`: at the top of `_get_web_search_results`
     set `self.last_web_decision = {"triggered": False, "source": None,
     "reason": None, "confidence": None, "results": None, "error": None}`;
     update it on EVERY return path (manager missing / not available / not
     triggered with `decision.reason|source|confidence` / triggered with
     `results = len(pages)` after the search / exception with `error =
     type(e).__name__`).
   - `core/prompt/builder.py`: wherever `"web_search_results":` is set
     (:1697, :2047, :2095, :2274, :2333) also set
     `"web_search_decision": getattr(self.context_gatherer,
     "last_web_decision", None)` (light/small-talk paths: a literal
     `{"triggered": False, "reason": "light path", ...}`).
   - `core/prompt/formatter.py:791–792`: build the label from
     `context.get("web_search_decision")` per test 1g; delete the
     `_count('web_search_results')` use for this label only (keep `_count`
     for the other keys). Add `web_search_decision` to
     `UNRENDERED_CONTEXT_KEYS` if the budget-parity guard
     (`tests/unit/test_budget_meters_rendered_sections.py`) complains — it
     is metadata, not a rendered section.
   - `config/prompts/operating_principles.txt` AI Limitations block (:6–7):
     add "- Never say a search ran, failed, or 'came back empty' unless a
     [WEB SEARCH RESULTS] section or a tool result is present in THIS turn.
     If you did not search, say so and offer to." (This is inside the cached
     prefix; one cache write on the next turn is expected.)

5. **Telemetry**: `core/orchestrator.py:1680` — add to `_last_turn_signals`
   from `getattr(getattr(self.prompt_builder, "context_gatherer", None),
   "last_web_decision", None)`: `web_trigger_should_search`,
   `web_trigger_source`, `web_trigger_reason` (≤120 chars),
   `web_trigger_confidence`, `web_results_n`, `web_error`. Update the
   `utils/turn_telemetry.py` docstring field list. Optional (only if under
   budget): read-only `scripts/web_trigger_rollup.py --days N` printing, per
   day × mode, counts of should_search True/False by source and how many
   triggered searches returned 0 results — same stdlib-only shape as
   `scripts/latency_rollup.py`, with a test like
   `test_rollup_ten_turns_read_only`.

6. **Run** (unit lane; daemon may be up):
   ```bash
   DAEMON_TEST_MODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q -p no:cacheprovider \
     tests/unit/test_sep10_web_search_gap.py tests/unit/test_public_news_verification.py \
     tests/unit/test_sep05_web_cache_same_query.py tests/unit/test_sep05_heavy_history_first_person.py \
     tests/unit/test_feature_inventory.py tests/unit/test_turn_telemetry.py \
     tests/unit/test_tone_arbiter_hardening.py tests/unit/test_adaptive_adopters.py \
     tests/unit/test_purge_adaptive_exemplars.py \
     $(ls tests/unit/test_web_search_trigger*.py tests/unit/test_agentic_gate*.py tests/unit/test_gate_*.py 2>/dev/null) \
     tests/unit/test_no_git_state_in_tests.py tests/unit/test_ordered_slice_guard.py \
     tests/unit/test_budget_meters_rendered_sections.py tests/unit/test_tool_wiring_parity.py \
     tests/unit/test_model_capability_wiring.py tests/unit/test_no_vacuous_assertions.py
   ruff check . && git diff --check
   ```
   Also `tests/test_web_search_trigger.py` (non-unit, in the privacy allowlist)
   ONLY if the daemon is down — it is heavier; otherwise list it for the owner.
   Record every count in the "Results" section you append to this file.

7. **Docs**: one CLAUDE.md-style bullet in `docs/PLAN_20260909_audit_repairs.md`
   is NOT needed; append a `## Results (Codex)` section to THIS file (counts,
   the `''` site from step 0, deviations) and update row 11 of
   `docs/HANDOFF_20260909_audit_followups.md` to "IN TREE, see
   HANDOFF_20260910_web_search_gap.md". Draft `commit_message_25.txt` (subject
   ≤72 chars; body = what/why/tests; no hunk surgery; docs move with code).

## CONTINGENCY

- Step 1c red because `has_active_suppression` (:1788) vetoes the LLM "yes":
  that is why the consult decision must carry EMPTY `matched_patterns`; if a
  suppression pattern also matched the text, the veto is correct — leave it.
- Step 2: if `public_actor_statement` fires on any string in
  `test_public_news_verification.py::test_personal_mixed_static_and_
  unresolved_queries_use_existing_routing` or on the vent shapes → tighten
  the verb list / add the private-sphere token check; never loosen the
  pronoun exclusion. Do not make it FORCE a search (no `should_search=True`
  from this arm) — the classifier decides.
- Step 4: if `test_budget_meters_rendered_sections` goes red on the new
  `web_search_decision` key → add it to `UNRENDERED_CONTEXT_KEYS` (it is
  metadata). If the `''` site (step 0) is inside `token_manager` write-back,
  STOP and report — that area has a fresh 09-06 contract
  (`STRUCTURED_SECTION_ADAPTERS`, "never a str") and a referee must look.
- Prompt-file edit changes the cached prefix; `tests/unit/test_section_
  instructions.py` / prompt snapshot tests may need their expected text
  updated — update the expectation, do not skip the test.
- If any existing web-trigger test asserts the OLD skip on a statement that
  is now consulted, read its docstring: if it guards the 08-05/08-15
  no_search-poisoning class, the new arm must not TEACH (it doesn't) but may
  consult — adjust the assertion to "not taught", not "not consulted". If
  unsure, STOP and ask Fable.
- Anything requiring a live provider call, a daemon restart, or a store write
  → OWNER section, not you.

## WHY

- The 09-05 rule was built for dated questions and is correct as far as it
  goes; the gap is the SHAPE the owner actually uses — pasting or paraphrasing
  a public claim as a statement. Forcing a search from a regex would reopen
  the Tavily-waste class (08-15, 09-01), so the arm only re-enables the
  classifier that the conf=0.0 short-circuit had switched off, and the prompt
  is told what a relayed claim is. Teaching stays outcome-based ([WEB_
  citations) so a classifier "no" cannot poison the anchors.
- `matched_patterns` doubles as the heuristic-veto signal (:1788), so a
  consult marker must live in its own field or the LLM's "yes" is silently
  overridden — the same dead-wiring class as the 07-03 `.intent_type` alias.
- The label lie is the direct cause of the confabulated "search came back
  empty": the model was told `ON(0)`. A label that carries the DECISION cannot
  be misread, and the prompt rule closes the residual.
- Telemetry is the "sensor that is not the owner" (WORKFLOW §7.4): with
  `web_trigger_*` in every row, tomorrow's roll-up shows how many public-claim
  turns were consulted, searched, and returned zero — without a dump.

## OWNER

- After Fable's PASS: commit (`commit_message_25.txt`), `git push` (the hook
  runs the unit selection fine; no non-unit files in this range unless
  `tests/test_web_search_trigger.py` is touched — then Daemon down or
  `SKIP_PREPUSH=1` with evidence), restart under tmux.
- Live probe (two relayed turns, exact texts T1 and, after pasting any public
  claim, T2): expect either a [WEB SEARCH RESULTS]-cited answer or an honest
  "I didn't search"; then `tail -2 logs/turn_records.jsonl` must show the
  `web_trigger_*` fields. One gpt-4o-mini trigger call + ≤1 Tavily credit each.
- Codex relay line: "Read docs/HANDOFF_20260910_web_search_gap.md, execute
  PLANNED 0–7 in order (tests first, record failing counts on HEAD), append
  Results, draft commit_message_25.txt, do not commit/push/restart."

## Results (Codex)

Executed PLANNED 0–7 in order on 2026-09-10. No provider calls, store writes,
commit, push, or daemon restart were performed.

- Step 0 coercion site: `core/prompt/builder.py`'s parallel gather loop stores
  each completed task as `task.result() or []`. An untriggered web task returns
  `None`, which therefore reaches the formatter as `[]` and renders `ON(0)`.
  The earlier `''` hypothesis was too narrow; both empty shapes render the same
  misleading label. No token-manager write-back is involved.
- Tests-first HEAD receipt (HEAD was `8dafdf9` when the baseline ran):
  `tests/unit/test_sep10_web_search_gap.py` = **34 failed, 3 passed**. Failures
  covered the absent public-statement and verification predicates/consult flag,
  skipped classifier, missing gate cue, dishonest labels, and missing gatherer
  receipt. The three preserved skip cases passed.
- Post-fix regression: `tests/unit/test_sep10_web_search_gap.py` = **37 passed**.
  Combined calendar + web 09-10 regressions = **70 passed**.
- Prescribed unit selection: the single-process command was run but stalled in
  the pre-existing `test_gate_forced_quality_floor.py::test_near_misses_rescued_junk_refused`
  after the first three tests in that file group. A faulthandler probe showed
  the event loop waiting on `_encode_texts` while its executor worker was stuck
  inside the fake embedder's NumPy call. Split receipts completed around it:
  the web/public-news/cache/history/feature/telemetry/tone/adaptive group =
  **149 passed, 1 skipped**; `test_agentic_gate.py` = **48 passed**; the six
  repo guard files = **87 passed**. The hanging forced-quality test was not
  changed because it is outside this handoff.
- Non-unit privacy-allowlisted lane (run based on the handoff's daemon-DOWN
  state; the sandbox could not inspect the tmux socket):
  `tests/test_web_search_trigger.py` = **113 passed**.
- Static checks: `ruff check .` passed; `git diff --check` passed.
- Deviations required by internally inconsistent fixtures: the public-actor
  word floor is four, because required fixture `Police arrested the mayor` has
  four words despite the prose specifying at least five. `congress` was added
  to the public-event actor vocabulary and `arrested` to the reportable-event
  verbs because two required positive fixtures otherwise cannot satisfy the
  stated two-regex rule. The exact negative `investigate how memory gating works
  in this assistant architecture today` has nine words despite the prose saying
  it is over twelve, so knowledge-interrogative continuations after
  `investigate` are explicitly rejected.
- Optional `scripts/web_trigger_rollup.py` was omitted under the handoff's
  credit constraint. The required per-turn telemetry receipt was implemented.
- During execution another process advanced HEAD to `a9432a1` and placed the
  calendar forced-action handoff changes in the same dirty tree. Codex did not
  commit. `commit_message_25.txt` describes the actual combined scope per the
  no-hunk-surgery rule.
