# Bug-class catalog

A permanent register of the MECHANISMS behind the defects found in this project,
not a list of bugs. Every class carries: what the mechanism is, where it has
struck, how to find further instances WITHOUT reading the whole project, what
structural closure exists, and an honest status. It is the reference an audit
starts from and the place a fix ends: naming the class is part of shipping.

Started 2026-09-10 from the whole audit record (2026-07-14 → 2026-09-10:
`docs/BUG_RETROSPECTIVE_20260715_20260904.md` §3's fifteen classes; the
independent audits `AUDIT_SWEEP_20260831.md`, `AUDIT_SWEEP_20260903_CODEX.md`,
`docs/INDEPENDENT_AUDIT_20260905.md`, `docs/AUDIT_20260906_CONVERSATION.md`,
`docs/HANDOFF_20260909_independent_bug_audit.md`; the generalization,
situation-coverage and claim-support audits; `CLAUDE_CHANGELOG.md`'s 46
batches; and the 2026-09-10 doc-drift audit). Four read-only extraction passes
produced ~90 candidate blocks; duplicates were merged here by mechanism.

## How to use and maintain this file

- **Every fix names its class.** The commit body and the changelog line carry
  `class: BC-nn` (several if it is several). A fix whose mechanism matches no
  entry ADDS one — that is the point where the catalog grows.
- **Every audit starts here.** An audit runs the detectors of the classes in
  scope (§ Detection methods) before free reading; new findings are filed
  under an existing class or a new one. Handoffs carry a `CLASS` line
  (`docs/DEVELOPMENT_WORKFLOW.md` §5).
- **A class entry is five lines**: Mechanism · Incidents (dated, file, one
  line, source) · Find (a runnable method or a named guard; `(proposed)` when
  untested) · Closure (named module/test/doctrine, or "per-incident") ·
  Status (`closed` = a structural guard fails on recurrence · `partial` =
  doctrine or chokepoint exists but adoption is manual · `recurs` = fixed
  per-incident only · `open` = no closure).
- **Status is downgraded, never argued up.** A class marked `closed` that
  strikes again becomes `partial` with the new incident and the reason the
  guard missed it.
- **IDs are stable.** Never renumber; retire by marking `merged into BC-nn`.
- The retrospective (`docs/BUG_RETROSPECTIVE_*`) is the narrative; this file is
  the index. `CLAUDE.md` links here; it does not repeat entries.

## Index

| ID | Class | Family | Status |
|---|---|---|---|
| BC-01 | Bare-substring keyword matching | A matching | partial |
| BC-02 | Negation-blind trigger matching | A | partial |
| BC-03 | Proximity regex crosses a sentence boundary | A | partial |
| BC-04 | Classifier missing an anchor or qualifier | A | partial |
| BC-05 | Short-circuit skips the only component that could judge the shape | A | partial |
| BC-06 | Agentic gate over-fire on non-requests | A | recurs |
| BC-07 | Probabilistic verdict overrides a deterministic route | A | closed |
| BC-08 | Continuity/topic misread on fragments and greetings | A | recurs |
| BC-09 | Classification on the merged attachment blob | A | partial |
| BC-10 | Dead wiring: consumer reads what no producer sets | B wiring | partial |
| BC-11 | Live setting never reaches an already-built consumer | B | partial |
| BC-12 | Config key never reaches its runtime reader | B | partial |
| BC-13 | Feature flag gates one path, not its siblings | B | partial |
| BC-14 | Call-signature kwarg mismatch (dead call) | B | recurs |
| BC-15 | Protocol/schema vocabulary drift (taught ≠ parsed; enum gap) | B | partial |
| BC-16 | Related constants drift / one constant, two purposes | B | partial |
| BC-17 | Sentinel/prefix registered in one list, not all | B | partial |
| BC-18 | Newest-first list consumed as oldest-first | C ordering/shape | closed |
| BC-19 | List position treated as priority; parallel arrays assumed aligned | C | partial |
| BC-20 | Writer/reader shape mismatch swallowed to an empty result | C | partial |
| BC-21 | Compare or coerce on the serialized form | C | partial |
| BC-22 | Budget write-back corrupts a typed or keyed section | C | partial |
| BC-23 | Metered key ≠ rendered key | C | closed |
| BC-24 | Floors and top-ups re-admit ungated or unbudgeted content | C | partial |
| BC-25 | Downstream consumer blind to gathered evidence | C | recurs |
| BC-26 | Duplicate "canonical" write or render paths | C | partial |
| BC-27 | Fallback path omits a check the primary enforces | C | partial |
| BC-28 | Derived-signal feedback latch | D state | partial |
| BC-29 | Self-teaching store poisoned by a non-independent signal | D | recurs |
| BC-30 | State scoped to the wrong lifetime | D | partial |
| BC-31 | Read-through cache not invalidated by the mutation path | D | closed |
| BC-32 | Cache-key inconsistency between call sites | D | partial |
| BC-33 | Non-transactional multi-step write | D | closed |
| BC-34 | Undo built on replace semantics against a merge-only store | D | closed |
| BC-35 | Non-atomic or colliding store writes | D | closed |
| BC-36 | Corrupt or wrong-shape store loaded as empty state | D | partial |
| BC-37 | Processes fighting over shared state (script/test vs daemon) | D | partial |
| BC-38 | Check-then-act race | D | partial |
| BC-39 | Lock guards construction, not initialization | D | closed |
| BC-40 | Cancellation does not reach thread-pool work | D | open |
| BC-41 | Eager expensive acquisition blocking the turn | D | partial |
| BC-42 | Shared mutable object returned by reference | D | closed |
| BC-43 | Liveness recorded only at operation end | D | closed |
| BC-44 | Model/stream artifacts leaking to display or storage | E model output | recurs |
| BC-45 | Shown ≠ stored ≠ recorded | E | partial |
| BC-46 | Prompt instruction loses to a structural input or model prior | E | recurs |
| BC-47 | Failure or not-run collapsed into a valid empty result | E | partial |
| BC-48 | Confabulated action-completion claim | E | recurs |
| BC-49 | Mid-loop narration shipped as the answer | E | partial |
| BC-50 | Verifier's own false corrections | E | partial |
| BC-51 | Attribute or count inferred from a label, not the user's words | E | recurs |
| BC-52 | Lossy multi-hop derivation; stale claim resurfaces unrevalidated | E | partial |
| BC-53 | Speaker attribution lost across merge, clip or extraction | E | closed |
| BC-54 | Fact provenance failure (unsupported span) | F extraction | recurs |
| BC-55 | Junk-object admission | F | recurs |
| BC-56 | Polarity inversion | F | partial |
| BC-57 | LLM sentinel string treated as literal data | F | closed |
| BC-58 | Guard covers only the first-observed path | G coverage | recurs |
| BC-59 | Owner-identity hardcoding | G | partial |
| BC-60 | Corpus-calibrated constants shipped as universal | G | open |
| BC-61 | Owner-domain-scoped vocabulary lists | G | partial |
| BC-62 | Tests that cannot fail | H tests/tooling | closed |
| BC-63 | Validation not against the deployed function | H | recurs |
| BC-64 | Fixture or fake contract drift | H | partial |
| BC-65 | Git-state-dependent tests | H | closed |
| BC-66 | Exclusion/ignore lists drift from their justification | H | partial |
| BC-67 | Resource cap miscalibrated against the selected workload | H | open |
| BC-68 | Undefined names without a lint floor | H | closed |
| BC-69 | Silent ops failures (exit-0 jobs, dead backups) | H | partial |
| BC-70 | Log, comment or severity misdescribes the control flow | H | partial |
| BC-71 | Documentation and tool self-description drift | H | partial |
| BC-72 | Unobservable decision (no receipt in telemetry or debug) | H | partial |
| BC-73 | Ambiguous short-form token expanded to an unstated referent via background context | E | partial |
| BC-74 | Broad continuation/affirmation arm outranks a narrower explicit-request arm | A | partial |
| BC-75 | Self-authored artifact laundered into evidence via a persistent store | E | open |
| BC-76 | Closure by phrase-append (narrow remedy pattern) | J process | open |
| BC-77 | Autonomy guardrail vetoes an explicit user request; dispatcher receipt still claims success | B | partial |

## A. Matching and routing (deterministic classifiers)

### BC-01 Bare-substring keyword matching
- Mechanism: a keyword list is matched by plain substring against lowered text, so the keyword fires inside an unrelated longer word or enum string.
- Incidents: 2026-07-21 `"crisis"` ⊂ `str(CrisisLevel.CONVERSATIONAL)` floored every session (found 08-27); 2026-08-28 `'solve'` ⊂ "resolution"; 2026-08-29 `'how'` ⊂ "shower" (13 s loop); 2026-09-01 `'actions'` ⊂ "not taking any actions" (152 s loop); 2026-09-08 `"ice"` ⊂ "Price" armed the distress floor on homework turns (RETRO §3.1; CHANGELOG 08-28, 08-29, 09-08). Nine incidents in the window.
- Find: DM-01 — `rg -n "\bin (query|text|q|msg)_?lower\b|\bin \w+\.lower\(\)" --type py` and confirm each hit routes through `utils/trigger_match.py`; enum checks must compare `.value` membership, never `str(enum)`.
- Closure: CM-01 chokepoint `utils/trigger_match.py` (word-boundary for bare words, substring for phrases, negation window) adopted at gate keyword lists, action intents, email arms, visual gate, web-search phrases; `query_checker.heavy_keyword_hits`; `tests/unit/test_trigger_match.py`.
- Status: partial — the sixth instance (ice⊂Price) shipped in a list the chokepoint had not reached; no repo-wide static guard rejects a new raw `in` test.

### BC-02 Negation-blind trigger matching
- Mechanism: a cue fires without checking a nearby negation ("don't search", "I did not take").
- Incidents: 2026-08-18 vent-shape test stripped stance markers but not negation, teaching 7 `no_search` exemplars; 2026-09-04 nine sites (web search, action detection, visual gate, file retrieval, email arms) ignored negation entirely; 2026-09-06 `fact_source` supported `took=5mg` from "I did not take <medication> today" (CHANGELOG 08-18, 09-04 item 8, 09-06).
- Find: DM-01 — any deterministic trigger site not calling `trigger_match.is_negated`/`has_non_negated_hit`; `fact_source._clause_is_negated` for extraction.
- Closure: `utils/trigger_match.NEGATION_CUE_RE` + 5-token window (09-04); clause-scoped negation in `memory/fact_source.py` (09-06). Deliberately NOT applied to `_ACTION_DISAVOWAL_RE`/probabilistic adders (documented).
- Status: partial — each new domain needed its own adoption.

### BC-03 Proximity regex crosses a sentence boundary
- Mechanism: a `.{0,N}` window meant to bound two cues spans `? . !`, so an unrelated clause satisfies the pattern.
- Incidents: 2026-09-08 `core/insight/detector._IMPLICIT_PERSONAL_COMPARISON_RE` routed a t-distribution question into 32 s of pattern analysis + PubMed (HANDOFF_20260908 F4).
- Find: `rg -n '\.\{0,[0-9]+\}' --type py` then check the window excludes `[^.?!\n]`.
- Closure: per-site rewrite to `[^.?!\n]{0,N}`.
- Status: partial — one site fixed; no lint.

### BC-04 Classifier missing an anchor or qualifier
- Mechanism: a keyword/regex router lacks a required anchor (possessive, personal-record cue, request shape, TitleCase), so it over- or under-fires.
- Incidents: 2026-08-27 unanchored "can you"/`document` rode file-continuation into a 106 s loop; 2026-08-31 `pattern_temporal` had no personal-record anchor (AUDIT 08-31 F3); 2026-09-05 `requires_fresh_public_evidence` under-fired on second-person wrappers and over-fired on pronoun-less private questions (FABLE_HANDOFF 09-05); 2026-09-08 `_REQUEST_SHAPED_RE` matched `read` the R function (HANDOFF_20260908 N1); 2026-09-10 calendar-create matched the user's narration "I only put professors hours in calendar" because the detector lacked a request/self-narration anchor (calendar forced-action handoff T9).
- Find: DM-09 — adversarial probe set against the deployed function (`scripts/probe_tone_backstop.py` pattern), with the live texts as fixtures.
- Closure: per-detector anchors (`_entity_mention_is_proper` TitleCase doctrine, head-anchored request shapes, `is_personal_doc_search`, private-sphere token sets, `_match_is_self_narration`).
- Status: partial — no standing adversarial-probe suite across detectors.

### BC-05 Short-circuit skips the only component that could judge the shape
- Mechanism: a conservative deterministic rule plus a "confident no" short-circuit means a whole message shape is never judged by anything — the cheap stage says "nothing here" and the expensive judge is skipped on exactly that verdict.
- Incidents: 2026-09-10 "The president says he will pay everyone 5000…" scored heuristic 0.0/no keywords → the LLM trigger was skipped, `requires_fresh_public_evidence` needs question+temporal cues a share lacks; 2026-09-09 "Uhm. Please investigate thank you" same skip, no context consult (HANDOFF_20260910_web_search_gap); 2026-09-10 (evening) "jot down a note for this session: TA sessions are Saturdays at 11 CT" scored gate "no trigger" though `create_daemon_note` exists — the deterministic tiers' conservative "no" left the note-save shape unjudged by any component, and the reply confabulated a queued calendar event instead (probe-dump handoff T5).
- Find: for every `if conf <= 0 and not hits: return` style short-circuit, enumerate which shapes reach 0.0 (a probe set of statement-shaped inputs) and whether any later stage sees them.
- Closure: `consult_classifier` flag + `public_actor_statement` arm + contextual `is_verification_request`, pinned by `test_sep10_web_search_gap.py` (09-10); `query_checker.is_note_save_request` + a gate Tier-1 tools arm for the note-save shape (probe-dump A3).
- Status: partial — the public-claim, verification and note-save shapes are closed; other zero-signal short-circuits still require shape enumeration.

### BC-06 Agentic gate over-fire on non-requests
- Mechanism: a gate tier classifies conversation as tool-worthy and launches a multi-second to multi-minute loop for no informational need.
- Incidents: 2026-07-15 60 s on a vibe remark (continuation override); 2026-08-18 49 s via `temporal_recall@0.85`; 2026-08-27 106 s paste, 369 s decision timeout with zero tools; 2026-08-29 151 s on lyrics; 2026-09-02 129 s on an emotional check-in, cause invisible until `gate_reason` was surfaced (RETRO §3.12); 2026-09-10 course-document narration forced a calendar-create loop that invented a 17:00 event (calendar forced-action handoff T9); 2026-09-10 (evening) a forced calendar-create round on an explicit recurring-event request still burned 7.8 s of web search + 3.6 s of wiki lookup inside the agentic loop with no informational need (probe-dump handoff T6/probe 2).
- Find: DM-10 — `jq -r 'select(.mode=="agentic-search") | .gate_reason' logs/turn_records.jsonl | sort | uniq -c` against `wall_elapsed_s` outliers; the 09-04 audit found `llm-fallback` launched 27 % of agentic turns.
- Closure: per-arm (word caps, head anchors, vent-shape veto, action arm, decision-timeout one-shot); `gate_reason` in every record.
- Status: recurs — no precision measurement of the Tier-4 LLM fallback; the web/wiki legs of a forced action round are still unguarded (see BC-58 sibling incident, same handoff).

### BC-07 Probabilistic verdict overrides a deterministic route
- Mechanism: an LLM flag early-returns and discards a routing decision already computed deterministically, or is accepted without a speech-act check.
- Incidents: 2026-08-31 `needs_pattern_analysis` early-returned `modes=["insight"]` (AUDIT 08-31 F7); 2026-09-06 pattern flag accepted on a self-report (AUDIT 09-06 item 1).
- Find: `if <llm_verdict>: return` after a deterministic result variable is assigned in the same function.
- Closure: preemption stands down when a deterministic route exists; `detector.allows_pattern_classification` speech-act gate.
- Status: closed for the two sites; no repo-wide check.

### BC-08 Continuity/topic misread on fragments and greetings
- Mechanism: topic/thread/STM scores a fragment, pronoun continuation or greeting in isolation and asserts a shift or a restatement.
- Incidents: 2026-07-28 "It was maybe 3 years…" → "Exercise Routine"; 2026-08-22 a two-word riff → "Tactical Gear" with an invented question; 2026-09-03 "Hey" after 13 h rendered as message #3 of the old topic; 2026-09-05 previous heavy flag carried onto a fresh session (RETRO §3.13; CHANGELOG 09-03, 09-05).
- Find: any topic-shift/thread site not using the shared `query_checker.topics_related` / `is_anaphoric_continuation` / `is_fragment_continuation` / `is_greeting_opener`.
- Closure: those shared predicates; `orchestrator._thread_context_is_stale`; `stm_skip_shape`.
- Status: recurs — every new terse shape needs its own detector; no "resolve against the previous exchange by default" architecture.

### BC-09 Classification on the merged attachment blob
- Mechanism: intent/tone/topic/STM/keyword scoring reads user text + full attachment content, so words inside a pasted transcript drive routing.
- Incidents: 2026-09-04 `temporal_recall@0.85` from "history/timeline" inside lecture transcripts (18 turns + 8 summaries pulled); a 2024 note scored 1.00 against a 130 K-token merged query; 2026-09-06 `_keyword_search` scored 0.9 on `'' in query` for single-chunk uploads (CHANGELOG 09-04, 09-06).
- Find: each classifier call site receives `ctx.user_text`/`analysis_text`, never `merged_input`/`processed_query`; `ANALYSIS_QUERY_MAX_CHARS` is the fix signature.
- Closure: enhanced mode passes raw text; Stage 3 does the one merge; `tests/unit/test_sep04_attachment_turn.py`.
- Status: partial.

## B. Wiring and configuration

### BC-10 Dead wiring: consumer reads what no producer sets
- Mechanism: a key, field or flag is read that nothing emits (or a value is computed and never forwarded), so the intended behaviour silently never runs.
- Incidents: `distance` key never emitted → `semantic_score` zero for every doc until 2026-07-14; `.intent_type` None until the 07-03 alias; `EscalationTracker` updated only on the unused `process_user_query` path (08-18); `.trigger` vs `tone_trigger` (08-27); query-rewrite `model=` kwarg raised on every call since 2026-01-23 (09-04); `daemon_self_notes` fetched and discarded (INDEPENDENT 09-05 F2). Sixteen incidents — the largest class (RETRO §3.2).
- Find: DM-02 registry parity tests; DM-16 config-key reachability; DM-19 kwarg grep; a WARNING log that repeats every turn is itself a signal (N1 was found that way).
- Closure: CM-02 parity tests for the four registries (`test_tool_wiring_parity`, `test_model_capability_wiring`, `test_budget_meters_rendered_sections`, API-error prefix parity); `POST_RESPONSE_HOOKS` single registry.
- Status: partial — recurs for any knob outside a registry.

### BC-11 Live setting never reaches an already-built consumer
- Mechanism: a setter updates config and reports success, but a frozen `from`-import, an `__init__`-copied attribute, a lazily-built singleton or a cache keeps the old value.
- Incidents: 2026-09-09 `gatherer_web` frozen `WEB_SEARCH_ENABLED`; `WebSearchRateLimiter.daily_limit`; `ContextPipeline._enable_query_rewrite`; calendar read cache after update/delete (HANDOFF_20260909 F04/F06/F10).
- Find: DM-04 — same-instance before/after probe (construct once, exercise, call the setter, reuse the SAME instance); grep module-level `from config.app_config import X` in modules a setter targets.
- Closure: CM-06 call-time reads (import doctrine case 3) + setters walk live instances; `tests/unit/test_sep09_live_controls.py`.
- Status: partial — each new setting needs the probe.

### BC-12 Config key never reaches its runtime reader
- Mechanism: a YAML key is validated by schema but read under another name, or never threaded into the constructor.
- Incidents: 2026-09-01 `google_calendar_lookahead_days` never passed; 2026-09-04 `features.rewrite_timeout_s: 0` never reached `ContextPipeline`; `PROMPT_TOKEN_BUDGET_DEFAULT` dead for API models until 07-25 (CHANGELOG 09-01, 09-04; CLAUDE.md Key Config).
- Find: DM-16 — for each YAML leaf, count readers outside `schema.py`/`app_config.py`; 0 or constant-only = candidate.
- Closure: per-incident (`_build_context_pipeline_config()` factored).
- Status: partial.

### BC-13 Feature flag gates one path, not its siblings
- Mechanism: a kill switch is read by one implementing surface only.
- Incidents: 2026-08-31 `PATTERN_ANALYSIS_ENABLED` gated the agentic tool but not the insight facet (AUDIT 08-31 F17, found by two auditors).
- Find: `rg -n "<FLAG>"` across the tree; every path whose docstring names the feature must read it.
- Closure: flag reaches both paths.
- Status: partial (one instance).

### BC-14 Call-signature kwarg mismatch (dead call)
- Mechanism: a caller passes `model=` where the signature is `model_name=` (or duplicates a kwarg inside `**kwargs`); the `TypeError` is swallowed by a fallback.
- Incidents: query rewrite dead 2026-01-23 → 09-04; `summarizer.generate_async` stray kwarg (09-04); `generate_async` local branches "multiple values for model_name" (09-06 referee).
- Find: DM-19 — `rg -n "generate_(once|async|with_openai)\(" --type py | rg "model="`; drive call sites through a kwargs-capturing fake client (`tests/unit/test_forced_top_p.py` pattern).
- Closure: per-incident.
- Status: recurs (three instances in eight months).

### BC-15 Protocol/schema vocabulary drift (taught ≠ parsed; enum gap)
- Mechanism: the vocabulary the model is taught and the vocabulary the parser or schema accepts diverge; a forced round has no valid expression and substitutes a sibling type.
- Incidents: 2026-08-29 fixed attribute groups couldn't express calendar fields; 2026-09-01 `ACTION_ATTR_RE` truncated at an apostrophe (F20); 2026-09-09 `PROPOSE_ACTION_TOOL_DEFINITION` enum lacked update/delete → a forced delete emitted a create (F12); 2026-09-10 action detectors missed the system's own "queue it up" / "approval card pop up" wording and recurrence amendments, leaving affirmative and retry turns unrouteable; 2026-09-10 (evening) a note-save request found no expressible route in the action-type vocabulary (only a bare tool, `create_daemon_note`, existed, with nothing bridging "jot down a note" phrasing to it) — the reply disclaimed the capability outright rather than routing to the tool (probe-dump handoff T5, sibling of the same turn's BC-05 incident); 2026-09-10 round 3 retest — `claims_pending_card`/`claims_calendar_state` still lacked the modal/participle vocabulary the assistant's OWN retry offers actually use ("the card should be up now", "it's already in place", "re-queued"), and `is_offer_affirmation`/`is_action_retry_request`/`is_clarification_answer` all returned False on "Yes it failed" — a five-way vocabulary gap between what the system says/asks and what its own parsers recognize, all on the SAME turn.
- Find: diff `ACTION_SPECS`/`ActionType` against every tool-schema enum and every taught tag in `core/agentic/types.py` vs what `protocols.py` matches; replay the assistant's own offer/completion phrasings through the detectors; DM-02 `test_tool_wiring_parity`.
- Closure: registry-driven attribute parsing; `build_forced_tool_schema` one-value enum; `resolve_forced_action` coerce-or-reject; forced-type and assistant-self-phrasing parity tests in the 09-09/09-10 regression modules; `query_checker.is_note_save_request` bridges the note-save phrasing to the `create_daemon_note` tool (probe-dump A3); round 3 widens `claims_pending_card`/`claims_calendar_state` with the modal/participle forms (A13) and adds `is_failure_report` ("yes it failed", "didn't go through"…) to the affirmation family so a bare failure report after a retry offer routes to a forced retry (A12).
- Status: partial — forced action types, current self-phrasing, and the note-save bridge are guarded, but vocabulary families can still drift outside those registries; round 3 closed the specific five phrasings found, not the class.

### BC-16 Related constants drift / one constant, two purposes
- Mechanism: two constants encode one relationship (collect 6000, truncate 3500) or one constant serves two features.
- Incidents: 2026-08-31 `_SOURCE_MATERIAL_TRUNC` vs the 6000-char collector (F23); `GOOGLE_CALENDAR_MAX_EVENTS` as fetch limit AND batch cap (F35).
- Find: grep both halves of a collect/consume pair; grep a constant's hit sites and check they belong to one feature.
- Closure: F23 aligned; F35 not split.
- Status: partial.

### BC-17 Sentinel/prefix registered in one list, not all
- Mechanism: an error prefix is added to `API_ERROR_PREFIXES` but not the display map or junk filter (or vice versa).
- Incidents: 2026-08-14 `[Streaming Error` missing everywhere; 2026-09-03/04 `[OpenAI unavailable` registered but no display entry → CI red (CHANGELOG 08-14, 09-04 item 3).
- Find: diff `API_ERROR_PREFIXES` against `_API_ERROR_DISPLAY`; `tests/unit/test_api_error_fail_fast.py`.
- Closure: parity test exists; no single registry ties prefix→display→filter.
- Status: partial.

### BC-77 Autonomy guardrail vetoes an explicit user request; dispatcher receipt still claims success
- Mechanism: a guardrail written to bound the model's OWN unprompted behavior (a per-session cap, a semantic near-duplicate skip, a rate limit) sits on the same execution path a user-forced request takes, so an explicit "save this" is silently vetoed by a rule meant for autonomous note-taking; the dispatcher above it emits a fixed success-shaped event/receipt regardless of the executor's result, so the loop's context and the progress stream both read "saved" while the store is unchanged, and the model narrates completion.
- Incidents: 2026-09-11 10:10 "jot down a note for this session: TA sessions are Saturdays at 11 CT" — the model called `create_daemon_note`; `DaemonNotesManager.create_autonomous_note` skipped it as a 0.879 near-duplicate of two poisoned self-notes; `_dispatch_create_daemon_note` still yielded `note_saved` and a "Self-Note Saved" round header; the reply said "Already covered".
- Find: for every executor reachable from a gate-forced or user-directed tool route, `grep -n "return None\|skipped" <executor>` and check whether the dispatcher keys its end event / round header off the executor's RESULT or off a constant; `grep -n "event_type="[a-z_]*_saved"\|_created"\|_sent"" core/agentic/tools.py` and confirm each sits under a result check.
- Closure: the request's provenance rides on the decision object (`SearchDecision.daemon_note_user_requested`, set wherever the gate-detected body override is applied) and the guardrail branches on it (cap and dedup skip bypassed, the near-duplicate still logged); the executor exposes the skip reason (`last_skip_reason`) and the dispatcher emits `note_skipped` / "Self-Note NOT Saved" unless the result string starts with the success prefix. Tests: `tests/unit/test_sep10_probe_dump_actions.py::TestA15*`.
- Status: partial — closed for `create_daemon_note` (2026-09-11); other autonomy-bounded executors (proposal capacity in `PendingActionsStore`, email/contact rate caps) not yet audited with the Find grep.

## C. Ordering, data shape, transport

### BC-18 Newest-first list consumed as oldest-first
- Mechanism: a store returns newest-first; a consumer slices `[-N:]`, `[:N]` or `[::-1]` assuming the opposite.
- Incidents: 2026-08-02 agentic digest served yesterday's turns; 2026-08-05 shutdown extractor dropped the newest turns; 2026-08-22 `_recent_distress_from_history`; 2026-08-27 four floor top-ups restored the OLDEST summaries; 2026-09-04 two more in `summarizer.py` found while migrating (RETRO §3.3; CHANGELOG 09-04 item 10).
- Find: DM-03 `tests/unit/test_ordered_slice_guard.py` (AST scan, content-anchored allowlist).
- Closure: `utils/ordered_slice.py` + the guard.
- Status: closed — a new un-allowlisted slice fails the guard.

### BC-19 List position treated as priority; parallel arrays assumed aligned
- Mechanism: positional iteration where order is not the priority, or `zip(documents, metadatas, ids)` with no length check.
- Incidents: 2026-09-05 admission iterated `PRIORITY_ORDER` positionally (INDEPENDENT F2); 2026-09-03 curation scans crashed on a missing parallel entry (CODEX 09-03).
- Find: positional loops feeding a running total without `sorted(key=priority)`; `zip(` over Chroma arrays without a length guard.
- Closure: sort by priority value; degrade missing entries to `None`/`{}`.
- Status: partial.

### BC-20 Writer/reader shape mismatch swallowed to an empty result
- Mechanism: a writer emits shape A, a reader assumes shape B, and a broad `except` turns the crash into "no rows".
- Incidents: 2026-08-28 corpus `{query,response}` vs retrieval `{content}` dedup keys never collided; 2026-09-09 `/api/graph` assumed a node list (F09); `get_ids_by_timestamp_range` passed ISO strings to numeric `$gte` — raised on every call since it shipped, `source_doc_ids` never written (F05); 2026-08-18 `add_summary` dict payload.
- Find: DM-18 `rg -Pzo "except Exception[^\n]*:\n\s*(logger\.\w+\([^\n]*\)\n\s*)?return \[\]" --type py`; DM-07 build fixtures through the real writer; a driver validation message in the log beside an empty result.
- Closure: `_canonical_turn_key`; boundary conversion in `/api/graph`; `timestamp_epoch` + numeric query; explicit WARNING on malformed bounds.
- Status: partial — no schema layer; each found live.

### BC-21 Compare or coerce on the serialized form
- Mechanism: ISO strings sorted as strings; an int epoch handed to a `str` Pydantic field.
- Incidents: 2026-08-31 StackExchange `creation_date` int crashed deliberation after the budget was spent (F8); 2026-09-03 email sort on ISO strings with mixed offsets; 2026-09-10 `str(ActionType.X)` was compared with `ActionType.X.value`, so real pending cards never matched while plain-string fixtures did.
- Find: `sorted(..., key=lambda x: x["timestamp"])` on raw strings; Pydantic construction from raw API dicts without coercion.
- Closure: coerce at parse; parse instants before sorting; compare enum `.value` through `_action_type_value`.
- Status: partial — the cited sites are fixed, but there is no cross-type static guard against serialized-form comparisons.

### BC-22 Budget write-back corrupts a typed or keyed section
- Mechanism: the compressor `str()`s a dataclass or writes a compressed value back under the wrong key.
- Incidents: `WebSearchResult` stringified → web section dropped 2026-03-26 → 09-06; compressed RESPONSE overwrote QUERY (2025-11-28 → 2026-09-01); recent-conversation metering read only the response field (09-04).
- Find: `git log -S"<write-back line>"` to date; `STRUCTURED_SECTION_ADAPTERS` membership for every non-string section; DM-02 `test_budget_meters_rendered_sections`.
- Closure: `STRUCTURED_SECTION_ADAPTERS` (never a bare `str()`), `_TEXT_KEYS` single source.
- Status: partial — new structured sections need registration.

### BC-23 Metered key ≠ rendered key
- Mechanism: the budget meters a key nothing renders while the rendered key is unmetered and untrimmable.
- Incidents: 2026-08-14 `summaries`/`reflections` metered, `recent_summaries`… rendered (~5 K tokens invisible); 2026-09-05 formatter ignored a preformatted `text` field the meter recognized (INDEPENDENT F2).
- Find: `tests/unit/test_budget_meters_rendered_sections.py` (fails when a rendered key lacks a `PRIORITY_ORDER` row).
- Closure: the guard + `UNRENDERED_CONTEXT_KEYS`.
- Status: closed.

### BC-24 Floors and top-ups re-admit ungated or unbudgeted content
- Mechanism: a "never fewer than N" floor or a post-budget top-up restores items with no quality or size check.
- Incidents: 2026-08-18 `GATE_MIN_MEMORIES` forced 8 below-threshold memories into "Hey"; 2026-08-29 top-up refilled 1 → 30 ungated; 2026-09-05 floors ran after the budget and restored uncapped records (INDEPENDENT F1); upload bar 0.5 ≈ any text (08-14/27).
- Find: `rg -n "min_results=|_FLOOR\s*=|TOPUP" --type py` and check for a margin and a post-floor re-measure.
- Closure: CM-08 survival-minimum doctrine (`GATE_FORCED_FLOOR_MARGIN`, `MEMORY_TOPUP_FLOOR`=3, `min_results=0` for notes); budget computed after floors.
- Status: partial — doctrine, not lint.

### BC-25 Downstream consumer blind to gathered evidence
- Mechanism: a planner digest, verifier, reuse gate or parallel task gets a narrower or different view than the answering model (hard-coded key lists; parallel context forks).
- Incidents: 2026-08-29 verifier never saw the retrieved syllabus; 2026-09-02 planner ran in parallel and reversed speaker/addressee (CLAIM_SUPPORT T02); 2026-09-06 `_CONTEXT_DIGEST_KEYS` omitted `user_uploads` (F3); 2026-09-07 reuse gate took any tool result as "evidence seen" (R3).
- Find: diff every hard-coded key list (`_CONTEXT_DIGEST_KEYS`, `_RETRIEVAL_EVIDENCE_KEYS`, direct-rendered keys) against the rendered `PRIORITY_ORDER` keys; `rg -n "asyncio.gather" core/orchestrator.py` and check siblings share one built context.
- Closure: gather-then-plan; `_RETRIEVAL_EVIDENCE_KEYS`/`_BACKGROUND_EVIDENCE_KEYS`; uploads reserved first in the digest; source_material fed to the verifier.
- Status: recurs — no invariant that every gathered section is visible to every consumer.

### BC-26 Duplicate "canonical" write or render paths
- Mechanism: two code paths each believe they are the single merge/render/persist step.
- Incidents: 2026-09-04 attachments rendered twice in [CURRENT QUERY] (+ a third partial copy) — a 265 K-token turn; 2026-08-22 a third summary write path bypassed `add_summary`'s sanitize (CHANGELOG 09-04, 08-18 addendum).
- Find: grep every call site of a content-producing function; assert each file's content occurs exactly once in the rendered prompt (`test_sep04_attachment_turn.py` template).
- Closure: one canonical merge in Stage 3; `_store_summary` sanitizes at entry.
- Status: partial.

### BC-27 Fallback path omits a check the primary enforces
- Mechanism: the secondary implementation of a step skips a validation, filter, budget or hygiene the primary applies (anchor vs neighbour; light path vs full; XML vs native).
- Incidents: 2026-09-05 light-context path bypassed the token budget (INDEPENDENT F1); 2026-08-31 `<propose_action>`/`<invoke>` skipped `forward_params` (F6); 2026-09-09 expander hygiene on neighbours only (F08); wiki fallback unfiltered (09-06).
- Find: grep the guard-function name used in the primary branch and confirm it appears in every sibling branch; black-box: send the same input through each path.
- Closure: per-site; `_hygiene_block_reason()` centralized in the expander.
- Status: partial.

## D. State, concurrency, persistence

### BC-28 Derived-signal feedback latch
- Mechanism: a derived value re-enters as an upstream input with no reset (the floor's own CONCERN seeds the next turn and the persisted state).
- Incidents: `session_distress` always true 07-21 → 08-27 (422 floored turns); floor output persisted to `tone_state.json`; floored CONCERN counted toward `distress_threshold` (08-28); `"ice"` re-armed the floor for 30 min per paste (09-08) (RETRO §3.4).
- Find: DM-10 — `jq -r .tone_trigger logs/turn_records.jsonl | sort | uniq -c` weekly; a `distress_sticky_floor` share climbing, or constant `conf=0.43` runs, is the signature.
- Closure: CM-10 `TONE_FLOOR_CHAIN_MAX`, `allow_sticky_floor`, tracker holds on floored turns, `distress_grounding_max`; share fell 71 % → 7 %.
- Status: partial — bounded, not eliminated; new upstream false positives re-arm it.

### BC-29 Self-teaching store poisoned by a non-independent signal
- Mechanism: an adaptive exemplar/relation store learns from a corrupted or non-independent channel and reinforces the wrong verdict.
- Incidents: 2026-08-15 news share + "look it up" taught `no_search`; 2026-08-18 seven political statements; 2026-09-01 `temporal_recall` store at its 40 cap with vents; 2026-09-08 need-detector learned 13 exemplars from homework code incl. an injected marker (RETRO §3.14; CHANGELOG 09-08).
- Find: any label at its cap (`python -c "import json;d=json.load(open('data/adaptive_exemplars.json'));print({k:len(v) for k,v in d.items()})"`); every `get_store().record(` call must sit behind a gate.
- Closure: CM-09 independent-channel teachers, tone/code/length gates, `purge_adaptive_exemplars.py`.
- Status: recurs — each adopter domain needed its own gate.

### BC-30 State scoped to the wrong lifetime
- Mechanism: a flag or cache that should reset at a round/turn/session boundary persists past it.
- Incidents: 2026-08-31 `_forced_action` never cleared across rounds (F13); expired token reported AVAILABLE (F15); `_conversation_depth` zeroed by restart so STM skipped the first 3 messages (08-05 round 3); 2026-09-10 an older calendar card blocked a fresh amendment/retry because proposal ownership was not scoped to the turn that minted it; 2026-09-10 (evening, BC-30/48 sibling) an agentic tool-loop turn resolved an unnamed "look at it" reference to a document uploaded five days earlier and asserted "You uploaded [it] today" — a retrieved file's actual upload date was not checked against the current turn/session, so a stale file read as freshly scoped to the live turn (probe-dump handoff T4).
- Find: attributes assigned once and read across boundaries with no reset at boundary start.
- Closure: per-site resets; `_has_recent_history`; pending-card ownership uses `_card_created_by_turn` and explicit supersession; `action_claim_guard.claims_fresh_upload` + a handlers post-check comparing the claim against `ActiveDocumentRegistry`/the upload roster's actual date (probe-dump B6).
- Status: partial.

### BC-31 Read-through cache not invalidated by the mutation path
- Mechanism: update/delete never clears the cache; a fingerprint/TTL is absent.
- Incidents: 2026-09-09 calendar cache after update/delete/partial create (F06); `MemoryExpander` served old text forever (F08).
- Find: DM-05 read-mutate-read on one live instance.
- Closure: `_invalidate_read_cache`; fingerprint + TTL + `notify_chroma_mutation`; `tests/unit/test_sep09_storage_repairs.py`.
- Status: closed (residual: calendar key lacks lookahead dims).

### BC-32 Cache-key inconsistency between call sites
- Mechanism: two callers of one cached operation derive different keys for logically identical calls.
- Incidents: 2026-09-05 web trigger ran twice per turn (gate `None`/100 vs gatherer live values); same query billed at two depths 11 s apart.
- Find: DM-11 — two identical LLM-trigger or search log lines seconds apart; diff the arguments each call site passes.
- Closure: bucketed policy/credits key + shared in-flight task; `WebSearchCache._recent_by_query`.
- Status: partial.

### BC-33 Non-transactional multi-step write
- Mechanism: the old state is destroyed before the new state or its recovery record is durable.
- Incidents: 2026-09-09 refdocs delete-before-insert (F02); curation mutated targets before the journal pre-image (F11).
- Find: DM-06 inject failure after the destructive step, before the recovery step; assert the pre-state reads back.
- Closure: CM-07 staged insert then delete; write-ahead `apply_started` + pre-images + `interrupted` recovery; `test_sep09_curation_integrity.py`.
- Status: closed.

### BC-34 Undo built on replace semantics against a merge-only store
- Mechanism: "remove the key" cannot clear a value when the driver's `update()` merges; a replacing fake hides it.
- Incidents: 2026-09-09 `curation_quarantined` stayed True after "undo" (F01).
- Find: DM-07 apply→undo on `chromadb.EphemeralClient`, assert the stored metadata including absence.
- Closure: explicit neutral values; reject items with no defined restoration; fake corrected to merge.
- Status: closed.

### BC-35 Non-atomic or colliding store writes
- Mechanism: direct `open(path,'w')` for a store, or every writer sharing `path + '.tmp'` so concurrent writers interleave.
- Incidents: 2026-07-14 `entity_aliases`/`claim_index` direct-write; 2026-09-09 `safe_json` shared temp name corrupted under concurrency (F03); 15 hand-rolled `.tmp` writers found by grep.
- Find: `rg -n "\.tmp['\"]|with_suffix\(.*tmp" --type py` outside `utils/safe_json.py`; `tests/unit/test_atomic_writer_guard.py`.
- Closure: `mkstemp` + fsync + `os.replace`; `atomic_write_text`; engine op lock; the guard test.
- Status: closed.

### BC-36 Corrupt or wrong-shape store loaded as empty state
- Mechanism: a load failure (or valid JSON of the wrong shape) is treated as "no data" and the next save wipes the user's data.
- Incidents: 2026-07-14 every JSON store silently reset on corruption; 2026-09-03 wrong-shape `user_profile.json` would have been accepted (CODEX 09-03 item 1).
- Find: each `safe_json` loader raises `CorruptStoreError`; each has a shape validator (`profile_shape_error` exists for the profile only).
- Closure: `utils/safe_json.py` quarantine-and-raise; `profile_shape_error()`.
- Status: partial — shape validation only on the profile.

### BC-37 Processes fighting over shared state (script/test vs daemon)
- Mechanism: a script, a test run, or a second daemon instance writes a store the daemon holds in memory; the daemon's next save (or a restore) clobbers or resurrects it.
- Incidents: 2026-08-05 curated profile writes wiped by the daemon's save; 2026-08-21 the guard itself defeated by a relative-path launch; 2026-08-22 pytest seeded the live floor via `tone_state.json`; 2026-09-01 a lazily-importing test wrote prod `pending_actions.json`; 2026-09-02 48 synthetic facts + 35 edges live from a calibration script; 2026-09-05 `graph_junk_cleanup.py --apply` had no guard.
- Find: DM-17 `grep -L daemon_guard scripts/*.py | xargs grep -l -- --apply`; `rg -n '"data/' tests/ | grep -v conftest`.
- Closure: CM-11 `utils/daemon_guard.py` (cwd-based), `DAEMON_TEST_MODE`, sandbox-only scripts, `sys.modules["main"]` alias.
- Status: partial — `logs/actions_audit.jsonl` still has no test sandbox; each new store needs its redirect.

### BC-38 Check-then-act race
- Mechanism: a guard is evaluated but the resource is acquired later, leaving a window.
- Incidents: 2026-09-05 chat turn-lock checked before the stream, acquired in the generator (INDEPENDENT F4); SSRF guard resolves DNS then httpx re-resolves (F34, deferred).
- Find: does reservation precede the return/yield point?
- Closure: reserve before returning; SSRF deferred.
- Status: partial.

### BC-39 Lock guards construction, not initialization
- Mechanism: the singleton lock wraps instance assignment, not the expensive `.load()`.
- Incidents: 2026-09-05 two threads cold-loaded the multi-GB FAISS index (F6); CLIPManager load vs warmup/upload overlap (B6 S1).
- Find: for each lazy singleton (`clip_manager`, `semantic_search`, `memory_retriever._cross_encoder`), does the lock wrap load?
- Closure: instance load locks; `test_semantic_load_concurrency.py`.
- Status: closed.

### BC-40 Cancellation does not reach thread-pool work
- Mechanism: timing out an await abandons the coroutine; the executor thread keeps running against torn-down state.
- Incidents: 2026-09-03 shielded next-chunk task after disconnect; 2026-09-05 a timed-out wiki search kept loading 54 s later (F6, flagged unresolved); zombie USB-stalled wiki threads (07-15).
- Find: `asyncio.to_thread`/`run_in_executor` under `wait_for` with no cooperative flag in the thread body.
- Closure: dedicated executors + in-flight semaphores (skip when saturated); disk work itself not cancellable.
- Status: open.

### BC-41 Eager expensive acquisition blocking the turn
- Mechanism: a sandbox, model cold-load or O(n²) scan runs synchronously or unconditionally before it is known to be needed.
- Incidents: 2026-09-08 26 E2B sandboxes created, 13 at zero executions (F5); 2026-09-10 CLIP cold load blocked the same turn 36 s (B6 S1); 2026-08-29 700 keywords × 800 candidates (20 s).
- Find: DM-11 "Created … sandbox" vs "0 executions"; blocking library calls inside `async def` not wrapped in `to_thread`.
- Closure: `LazySandboxSession`; `asyncio.to_thread` in the visual pipeline; `KEYWORD_MATCH_MAX_KEYWORDS`.
- Status: partial.

### BC-42 Shared mutable object returned by reference
- Mechanism: a cache accessor returns its internal dict; `from_dict` pops from the caller's dict.
- Incidents: 2026-09-03 email cache by reference; snapshot `from_dict` mutating input (CODEX 09-03).
- Find: `return self._cache` without copy; `.pop(` on a parameter.
- Closure: defensive copies.
- Status: closed.

### BC-43 Liveness recorded only at operation end
- Mechanism: the heartbeat the idle watchdog reads is poked at completion, so slow work looks idle (or, inverted, in-flight work blocks shutdown forever).
- Incidents: 2026-09-09 a 36 s in-flight image call looked idle and the owner killed the daemon (FOLLOWUPS #7).
- Find: grep heartbeat call sites; at least one precedes the first blocking step.
- Closure: poke at ingress + age-bounded `has_inflight_turns(max_age_s=)` (B6 S3).
- Status: closed.

## E. Model output and epistemics

### BC-44 Model/stream artifacts leaking to display or storage
- Mechanism: provider quirks (literal `<thinking>` tags, trailing `e`, `<|sep|>`, degenerate loops, raw 402 JSON) reach the bubble, the corpus or the log.
- Incidents: 2026-08-03 trailing `e`; 08-14 raw 402 streamed three times; 08-22 tag shell; 08-31 3.5-min degenerate loop; 09-03 `[OpenAI unavailable` (50 docs) (RETRO §3.8).
- Find: every stream consumer routes through `sanitize_for_storage()` and `strip_trailing_stream_artifact` (source-level guard asserts ≥4 handler sites); `jq -r .response logs/turn_records.jsonl | rg '^\['` for unregistered error shapes.
- Closure: `sanitize_for_storage` steps 7–8, `looks_degenerate_stream`, `API_ERROR_PREFIXES` registry; `tests/unit/test_stream_artifacts.py`.
- Status: recurs by nature — each model swap can add a shape.

### BC-45 Shown ≠ stored ≠ recorded
- Mechanism: a post-generation mutation updates one of display/storage/debug record and not the others.
- Incidents: 2026-08-28 review gate's only production "fix" was a silent swap (→ log-only); 2026-08-31 agentic debug record built before grounding ran, `grounding_corrected` set before integration (F9/F24); 2026-09-08 no-round line stripper mutated display only — fences vanished from the bubble, history intact (F1/P0).
- Find: two co-existing variables (`display_output` vs `final_output`) not updated from one function; `_answer_bodies_agree` parity guard absent at a new mutation site.
- Closure: whole-bubble replacement yields; `_capture_delivery` after mutations; log-only parity guard.
- Status: partial — each new mutation path has violated it once.

### BC-46 Prompt instruction loses to a structural input or model prior
- Mechanism: prose tells the model X; an attached image, an offset convention or a self-belief about tool access wins.
- Incidents: vault screenshot narrated (07-14, again 08-27); `13:00:00-04:00` emitted three turns despite "ET = America/New_York" (09-01); "I don't have calendar access" with healthy OAuth (08-29); planner invented a birthday (09-03); a forced calendar proposal invented 17:00 despite its own reasoning saying the time needed confirmation (09-10) (RETRO §3.5; calendar forced-action handoff); 2026-09-10 (evening) a calendar-create card was minted with `start_time="15:00:00"`/`end_time="16:00:00"` — a bare clock time with no date — and failed only at approve time ("invalid ISO 8601 start/end time"); nothing validated the full-ISO-datetime shape structurally before the card was created (probe-dump handoff T6/probe 2); 2026-09-10 round 3, T2 — a forced round instructed the model to call `propose_action` for a calendar request, but the model refused ("We actually already locked this one in… Approving that card puts it on your calendar"), its own PRIOR narration in `[RECENT CONVERSATION]` outweighing the current-turn forced-tool instruction; no card resulted despite the retry.
- Find: DM-20 — every prohibitive prompt instruction (`never|don't|do not`) must have a deterministic sibling (gate, executor check, post-check); for a forced-round refusal specifically, check whether the refusal's stated reason quotes the assistant's OWN earlier turn.
- Closure: CM-03 executor/backstop functions (`wall_clock_time`, `get_runtime_action_health`, `weekday_date_mismatches`, `unsupported_key_points`, `calendar_times_ungrounded`, visual-intent gate); calendar datetime validation at PROPOSAL time — reject-and-retry when start/end is not a full `YYYY-MM-DDTHH:MM:SS`, with deterministic weekday/clock-time backfill from `wall_clock_time()` (probe-dump A2, "validate at parse"); the forced round's own "[PENDING CARDS] none — any earlier 'queued'/'locked in'/'re-queued' wording was NOT backed by a card" prompt line (round 3 A11) plus a deterministic mint-anyway fallback when the model still declines and a weekday/time/title all resolve.
- Status: recurs — bespoke backstop per incident; the T2 refusal is the first instance of the model's OWN prior narration (not an image or offset convention) as the overriding "structural input", closed only for the calendar forced-round path.

### BC-47 Failure or not-run collapsed into a valid empty result
- Mechanism: timeout/unavailable/never-ran is encoded like a genuine negative — in code (same return shape) or in the prompt (`web_search=ON(0)` for both "0 results" and "never ran") — so downstream, including the model, cannot tell "nothing" from "couldn't check".
- Incidents: 2026-09-05 classifier timeout reused the no-search verdict; 2026-09-06 failed plan = zero events; 2026-09-09 verifier timeouts recorded `complete` (B6 follow-up); 2026-09-09 the model narrated "my web search came back empty" from `ON(0)` when no search ran (HANDOFF_20260910_web_search_gap).
- Find: any except/timeout branch returning the negative-verdict shape without a `source`/`reason` discriminator; feature labels built from result counts instead of decisions.
- Closure: CM-05 tri-state outcomes (`source=fallback`, `grounding_status`, "unavailable not zero" doctrine); web labels now derive from `web_search_decision` and distinguish not-triggered, zero-result and error outcomes (09-10).
- Status: partial.

### BC-48 Confabulated action-completion claim
- Mechanism: the reply asserts it sent/created/queued when nothing executed.
- Incidents: 2026-09-01 "Re-queuing… Approve that one" with nothing queued; 2026-09-07 "Confirmed — creating the recurring event now"; expired proposal re-served as "Queued"; 2026-09-10 "Queued up" / "Queuing it now … approval card pop up" shipped with no backing card.
- Find: `claims_pending_card()`/`_COMPLETION_PATTERNS` match with empty `proposed_kinds` and `executed_kinds` for the turn.
- Closure: `core/action_claim_guard.py`, `NO_CARD_NOTICE` backstop.
- Status: recurs — each new phrasing found individually.

### BC-49 Mid-loop narration shipped as the answer
- Mechanism: decision-answer reuse or the synthesis call ships "let me aim at…" as the final response.
- Incidents: 2026-08-28 promissory check read only 150 chars; 2026-08-29 synthesis call unguarded; 2026-09-07 question-list clarification passed as an answer.
- Find: telemetry `agentic_narration_recovered`; every final-answer call site runs `_PROMISSORY_RE`/`_LOOP_META_RE`/`narration_shaped_final`.
- Closure: whole-text checks at both call sites; `regenerate_final_answer`.
- Status: partial.

### BC-50 Verifier's own false corrections
- Mechanism: a post-answer LLM judge with a narrow view flags correct answers at high confidence and the integrator ships the correction.
- Incidents: 2026-08-29 correct semester flagged 0.9; Saturday→Sunday invented; 2026-09-01 correct Sep-13 rewritten to Sep-6 at 1.00 against its own source; PSA spliced over the user's own figure (09-02). 42 fires → 25 shipped corrections, ≥9 false, 0 documented true.
- Find: DM-10 `grounding_flagged` rows whose flagged span overlaps the source material; precision ledger.
- Closure: CM-04 `GROUNDING_MODE=log_only`; deterministic demotions (`claim_date_user_stated`, `_is_safety_psa`, advice-shaped verdicts).
- Status: partial — correct mode unproven.

### BC-51 Attribute or count inferred from a label, not the user's words
- Mechanism: a topic label, graph edge, planner point or narrative asserts species/role/event/duration/count the user never stated.
- Incidents: 2026-09-03 cat called a dog (edge + label + planner aligned); "birthday celebration" from "turned 2"; 2026-09-04 "withdrawn from the semester" vs a one-course drop; 2026-09-05 "six consecutive days" copied across days.
- Find: `ResponsePlanner.unsupported_key_points`-style overlap check; `utils/streak_claims.py`, `utils/status_claims.py`, `completed_plan_claims` post-checks; DM-27 corpus-replay planner canary (see BC-73) catches the same shape one stage earlier, at resolution time.
- Closure: those modules + `GraphMemory.edge_is_suppressed`.
- Status: recurs — bespoke checker per incident shape.

### BC-52 Lossy multi-hop derivation; stale claim resurfaces unrevalidated
- Mechanism: a stage builds from the previous derived artifact (note → narrative), keeping one value per predicate; older claims re-enter without checking newer statements.
- Incidents: 2026-09-02 "six days" vs "two weeks" flattened in the narrative (CLAIM_SUPPORT T04); 2026-09-06 completed plans still shown pending, streak warnings quoting a discarded claim (AUDIT 09-06).
- Find: generators whose declared input is another generator's output and do not cross-check the transcript; `remove_stale_streak_claims` post-check.
- Closure: `build_temporal_claim_audit`; streak ledger; narrative status-claim check.
- Status: partial.

### BC-53 Speaker attribution lost across merge, clip or extraction
- Mechanism: mixed User/Assistant text is labeled as one speaker after clipping or when the merged field is searched instead of `user_text`.
- Incidents: 2026-09-06 provenance labeled assistant text as user statements; `search_keyword` ran on the merged attachment text (AUDIT 09-06 items 3–4).
- Find: `.get("query")`/`.get("response")` used where a `user_text` field exists.
- Closure: speaker-split provenance, `search_keyword(authored_only=)`, corpus `user_text`.
- Status: closed for cited sites.

## F. Fact extraction

### BC-54 Fact provenance failure (unsupported span)
- Mechanism: a stored triple's excerpt does not establish the claim — quoted/pasted text, third party, prospective clause, negated clause, relayed agent text.
- Incidents: `lived_in=Atlanta` from song lyrics (09-02); `enrolled_in` from a quoted email superseded a curated fact (09-03); `took=melatonin` from "ready to take" (09-08); `works_on="streak-count ledger…"` from relayed status text (09-05).
- Find: DM-21 `scripts/report_claim_contamination.py` (read-only, runs the deployed `find_supporting_user_span`).
- Closure: `memory/fact_source.py` (user-authored span, quote/response exclusion, cue families, clause negation, `completion_claim_prospective`, `[relay]`/`[test]` blocks).
- Status: recurs — new cue families keep being added.

### BC-55 Junk-object admission
- Mechanism: `_is_junk_object` lacks the exemption/rejection family for a new object shape.
- Incidents: "for a bit" (08-02); `has_dad="dad is picking up at 5:30"` (09-03); `works_on="this assistant"` (09-05); `gym_schedule` from a calendar request (09-08).
- Find: `scripts/purge_junk_facts.py` dry-run against the deployed predicate; `ProfileJunkFactCurator`.
- Closure: regex families widened per incident; curation card.
- Status: recurs.

### BC-56 Polarity inversion
- Mechanism: a positive-preference triple stored from negated or cursed text.
- Incidents: `likes | my fucking life` and three more (08-02); `_canonicalize_preferences` rewrote co-occurring triples.
- Find: `purge_junk_facts.py` inversion scan; `tests/unit/test_fact_junk_polarity_guard.py`.
- Closure: `_polarity_conflict` on both extractor paths.
- Status: partial.

### BC-57 LLM sentinel string treated as literal data
- Mechanism: a planner emits `"null"`/zero phases; structural validation passes it through.
- Incidents: 2026-09-06 `"null"` end date killed the pattern channel; zero-phase spec.
- Find: date/window parse sites matching literal `"null"|"none"|"tbd"|""` before arithmetic.
- Closure: sentinel normalization, per-phase drop, deterministic two-phase fallback.
- Status: closed.

## G. Coverage and generalization

### BC-58 Guard covers only the first-observed path
- Mechanism: the fix lands where the incident was seen; structurally identical siblings (read sites, sibling generators, other collections, other display paths) stay unguarded. The project's meta-class.
- Incidents: artifact strip covered storage + enhanced only (08-14); species guard ingestion-only until read sites (09-03); `backup_targets` missed every store since 07-14 (09-01); `weekly_notes_generator` missed when daily/monthly were fixed (09-01); privacy scrub covered git only, not the share surfaces (09-02); chroma junk curator never touched `user_profile.json` (09-05); the pending-action filesystem guard allowed the canonical state file but blocked its atomic temp siblings (09-10); action-offer and completion guards omitted phrases emitted by the assistant itself (09-10); 2026-09-10 (evening) the same-day 0aeffb7 batch closed web-trigger over-fire for PUBLIC-NEWS-statement shapes (`public_actor_statement`) but left the sibling PERSONAL/ACTION-shaped path unguarded — a 14 s Tavily search still ran for a personal medication-dosing question and a forced calendar-create round still burned 7.8 s web + 3.6 s wiki (probe-dump handoff T2, T6); 2026-09-10 round 3 — the round-2 fix for `is_status_report` (and its `is_self_report`/`is_request_shaped`/`is_note_save_request`/`is_personal_doc_search`/`is_casual_acknowledgment` siblings) closed the CLEAN probe-text string only; the live client sends the same message LINE-WRAPPED ("...a new doc I\n  think will be helpful"), and every one of those shape predicates went blind on the wrapped form even though the regression test for the clean form passed.
- Find: DM-15 — after fixing site A, enumerate every read AND write of the same primitive/store (`rg` the store method or the sibling naming pattern) before closing the batch; for a text-shape predicate specifically, re-run it against `" ".join(text.split())` and flag any predicate whose verdict changes.
- Closure: none structural; situation-coverage sweeps (`docs/SITUATION_COVERAGE_AUDIT_20260901.md`); `is_personal_routine_question` + action-request web-trigger stand-down (probe-dump A4) closes this sibling for the two cited shapes only; round 3 closes the wrapped-text sibling with `utils/trigger_match.normalize_ws` applied ONCE at ingress (`gui/handlers.py`, CM-01 chokepoint) rather than patching each predicate — the generalized closure this class has otherwise lacked.
- Status: recurs — the wrapped-text instance is now closed by a chokepoint rather than a per-predicate patch, but the class's other cited siblings (privacy scrub, backup_targets, action-offer vocabulary) remain per-incident.

### BC-59 Owner-identity hardcoding
- Mechanism: name, vault path, timezone or owner few-shot embedded in generators/prompts.
- Incidents: the owner's first name + vault path literal in the notes generators (08-21); `America/Chicago` in calendar executors and tool schema (09-01); GT few-shot in committed prompts (09-01).
- Find: `rg -n '<owner first name>|<vault dir name>|America/Chicago|<owner institution>' --type py --type txt -g '!tests/**' -g '!docs/**'` (take the literals from `config/config.local.yaml`); any literal in `core/prompt/`/`config/prompts/` that also appears in `config.local.yaml`.
- Closure: `utils/user_identity.py`, `utils/timezone_resolver.py`, neutral few-shots.
- Status: partial.

### BC-60 Corpus-calibrated constants shipped as universal
- Mechanism: a threshold derived from a probe on the owner's corpus is hardcoded with no re-derivation path.
- Incidents: gate 0.60/0.61, tone 0.37 backstop, escalation 3/5, budget 10000, upload bar 0.62 (GENERALIZATION audits P2).
- Find: `rg -n "calibrat|quantile|probe-|preregist" config/app_config.py utils/tone_detector.py core/escalation_tracker.py` and check a re-runnable, scheduled probe exists.
- Closure: none (deferred by both audits).
- Status: open.

### BC-61 Owner-domain-scoped vocabulary lists
- Mechanism: a list meant to generalize is built from one person's incidents (HOI4 exemplar, care-team relations, therapist-only audience markers).
- Incidents: `_SELF_MODEL_NOUNS`, `_PERSONAL_MARKER_RE`, `HEAVY_KEYWORDS` conflating personal and professional vocabulary (GENERALIZATION 09-01 P1).
- Find: `rg -n "_NOUNS\s*=|_KEYWORDS\s*=|_EXEMPLARS\s*=|_MARKER_RE\s*=" core/ memory/ utils/` and check for an adaptive-store sibling.
- Closure: categorized-generic rewrites; adaptive exemplar stores.
- Status: partial.

## H. Tests, tooling, docs, ops

### BC-62 Tests that cannot fail
- Mechanism: `except Exception: assert True`, bare `return True/False`, `try/except: pytest.skip`.
- Incidents: 2026-08-28 vacuous double-filtering file; 2026-09-09 11 generator bodies (27 undetected corrupt cases), 31 more across 12 files, two fossils calling nonexistent methods, `debug_memory_state` crash hidden as a skip.
- Find: DM-08 assertion-sensitivity probe (corrupt the provider, the test must fail); `tests/unit/test_no_vacuous_assertions.py` (AST, empty allowlist); `PytestReturnNotNoneWarning`.
- Closure: the guard test; sensitivity receipt 33/33.
- Status: closed.

### BC-63 Validation not against the deployed function
- Mechanism: a test or harness checks a re-derivation, a mock, a getsource string, a hardcoded literal, a proxy metric or only the shape of the result.
- Incidents: the five Critical-Rule incidents (inverted oracle, L2-as-cosine, mocked `.intent_type`, monotone AUC, `distance` key); `test_prompt_timeout` rebuilt the gather loop (T02); getsource-only coverage of a 230-line branch (F12); hardcoded date vs live clock (F4); shape-only assertions (T12/T13).
- Find: `rg -n "inspect.getsource" tests/`; helpers defined but unused in a test module; DM-08 perturbation proof.
- Closure: CLAUDE.md Critical Rule; per-test rewrites; `tests/unit/test_hybrid_semantic_score.py`.
- Status: recurs — doctrine, no static check.

### BC-64 Fixture or fake contract drift
- Mechanism: a fake replaces where the driver merges; a fixture is hand-typed to the consumer's assumption, not the producer's output.
- Incidents: `FakeCollection.update` replaced metadata (T03, hid F01); `/api/graph` fixture matched the wrong schema (T06/F09); the 09-07 pending-card fixture used a plain `.value` string, hiding the real `ActionType` enum/string mismatch found 09-10; 2026-09-10 round 3 — every regression fixture for the round-1/2 probe texts was typed as a CLEAN string (`"Cool. Managed to push today..."`), never the line-wrapped form the client actually transmits (`"...a new doc I\n  think will be helpful"`); the fixtures matched the developer's mental model of the input, not the producer's (the browser/API client's) actual output shape, so a genuinely fixed predicate still failed live.
- Find: DM-07 build fixtures through the real writer (`GraphMemory.save()`), validate fakes against `chromadb.EphemeralClient`; for text-input fixtures specifically, add the wrapped/indented form alongside the clean one (`docs/DEVELOPMENT_WORKFLOW.md` §3 fixture rule).
- Closure: corrected fakes + real-driver contract tests; 09-10 action sequence tests use real `ActionProposal` values through `PendingActionsStore`; round 3 requires every live-text fixture in `test_sep10_probe_dump_*.py` to be asserted in BOTH clean and wrapped form, and the WORKFLOW fixture rule now names line-wrapping explicitly.
- Status: partial.

### BC-65 Git-state-dependent tests
- Mechanism: a test reads `git show HEAD:` and is green only while the fix is uncommitted.
- Incidents: 2026-09-07 four tests went red on push (05fd300).
- Find: `tests/unit/test_no_git_state_in_tests.py`.
- Closure: the guard + `hooks/pre-push` dirty-tree refusal.
- Status: closed.

### BC-66 Exclusion/ignore lists drift from their justification
- Mechanism: a `--ignore` added for a dated reason outlives the reason, or is copied into a second place that never follows the first.
- Incidents: 17 CI ignores stale since 2026-05 (removed 52a7fbb); the hook's `PREPUSH_FULL` carried 4 of them until 2026-09-10; `pytest.ini` has 3 still-genuine + 1 dead entry.
- Find: DM-13 rerun every ignored file under the CI marker filter; diff ignore lists across CI, hook and pytest.ini.
- Closure: `docs/TEST_LANES.md` ledger; hook aligned 09-10.
- Status: partial.

### BC-67 Resource cap miscalibrated against the selected workload
- Mechanism: a fixed memory cap wraps a selection whose composition changed; the run is SIGKILLed with no test failure.
- Incidents: 2026-09-10 `hooks/pre-push` 6 GiB vs a 10-file non-unit push (5.97 GB killed; 7.1 GB passes under 8 GiB).
- Find: DM-14 measure peak RSS of the exact selection under `systemd-run -p MemoryMax=` whenever selection logic changes.
- Closure: none (FOLLOWUPS row 9).
- Status: open.

### BC-68 Undefined names without a lint floor
- Mechanism: `NameError`s and shadowed imports ship because nothing checks.
- Incidents: 2026-08-28 `launch.py` logger ×14, `core/prompt.py` shadowed imports.
- Find: DM-22 ruff bug-rule floor in `pyproject.toml`, run in CI and pre-push.
- Closure: the floor.
- Status: closed for the rules in the floor.

### BC-69 Silent ops failures
- Mechanism: a background job exits 0 or logs at an unwatched level while dead.
- Incidents: Backblaze dead 46 days; daily-notes timer failing 7.5 months (452 tracebacks, exit 0); wiki FAISS index unmounted; People API 403; OpenRouter credits out mid-session (RETRO §3.15).
- Find: `grep -c Traceback logs/*.log` trended; remote object counts, not exit codes; `get_runtime_action_health()`.
- Closure: per-item (OnFailure alert, prune-on-success, explicit disabled state).
- Status: partial.

### BC-70 Log, comment or severity misdescribes the control flow
- Mechanism: a log says "skipping" but the code falls through; a degradation logs at DEBUG; an idempotent outcome logs ERROR.
- Incidents: "skipping wiki" fell through to the live API (F26); pattern-preemption crash at DEBUG "how the arbiter stayed dead" (F7/F22); absent FAISS index at ERROR, E2B 404 at ERROR (CODEX 09-03).
- Find: `rg -n "skip|checked first|never|always"` near a branch and read the next lines; `rg -n "logger.debug"` in gate/trigger except blocks.
- Closure: per-site.
- Status: partial.

### BC-71 Documentation and tool self-description drift
- Mechanism: prose (README, ledgers, workflow docs, hook headers, docstrings, CLAUDE.md) asserts a state the code no longer has; runners assume tree state they never verified.
- Incidents: 2026-09-10 audit — 27 verified mismatches: TEST_LANES described retired CI ignores as "mid-edit"; README launch port 7860 (FastAPI 8000), `pytest.ini` marker claim, stale counts, escalation "budget" claim, `gitleaks protect` command; hook header vs body; AGENT_SAFETY manifest omitted `data/backups/`; a runner `git add`ed gitignored files and would have aborted mid-way; Docker compose targets a port nothing listens on; CLAUDE.md claimed all `--apply` scripts had the daemon guard when one did not (09-05).
- Find: DM-12 — three read-only auditors, one per doc slice, every claim checked against the file it describes, output only verified mismatches with file:line.
- Closure: CM-13 docs move with code (§3a); doc audit at batch boundaries; `scripts/generate_doc_metrics.py` for counts.
- Status: partial — no automated claim checker; counts regenerate, prose does not.

### BC-72 Unobservable decision (no receipt in telemetry or debug)
- Mechanism: a routing/verification decision is made inside a function and dies there; nothing in `turn_records.jsonl` or the debug record says what was decided or why, so the defect is found only when the owner pastes a dump.
- Incidents: gate reason invisible until 09-02; `tone_trigger` absent until 07-25 (the latch went unmeasured for weeks); grounding failures labeled `complete` (09-10); web-trigger decision has no field at all (09-10).
- Find: for each decision function, does its verdict + source + reason reach `_last_turn_signals`/the debug record? DM-26 rollups exist only for fields that exist.
- Closure: CM-12 receipts (`gate_reason`, `tone_trigger`, `grounding_status`, `answer_call`, timings); `web_trigger_*`, result count and error fields added 09-10.
- Status: partial.

## I. New mechanisms (2026-09-10 probe-dump)

Two mechanisms from the 2026-09-10 evening probe-dump handoff did not match any
of the 72 classes above closely enough to file as an incident; both are judged
genuinely new — distinct failure points from their nearest relatives (BC-51's
downstream embellishment, and BC-04/06/07/15's anchor/over-fire/override/
vocabulary mechanisms) — and are added here rather than stretched into an
existing entry. A third, from the same handoff's round-3 retest, is added
below for the same reason (BC-75).

### BC-73 Ambiguous short-form token expanded to an unstated referent via background context
- Mechanism: an LLM-driven interpretation stage (STM/coreference/short-term-memory analysis) resolves an ambiguous abbreviation or elliptical noun to a SPECIFIC referent found elsewhere in background context — an older mention, an unrelated topic — rather than leaving it unresolved; the guess then rides unchecked into a planner or response as if the user had said it. Distinct from BC-51 (a downstream stage embellishes a stated fact with unstated detail): here the referent itself is substituted at the resolution step, before any planning happens, and can poison every consumer that reads the resolved value, not just one planner's key points.
- Incidents: 2026-09-10 "Cool. Managed to push today and there is a new doc I think will be helpful" — the STM analyzer expanded "doc" to "doctor" by drawing on an unrelated earlier TEMPORAL GROUNDING mention (a psychiatrist reference), and the planner's three key points (all about a new doctor) survived `unsupported_key_points` because "doctor" appeared in the digest, not because the query or exchange supported it; the reply answered about a psychiatrist instead of addressing the actual document (probe-dump handoff T3); 2026-09-10 round 3 retest — the SAME "doc"→"doctor" expansion survived `unsupported_key_points` a second time because the round-1/2 closure checked the plan's `key_points` field only: the LLM's `key_points` came back empty ("Cover: (none)"), but its `strategy` field ("Acknowledge the user's progress and express support for their new doctor.") carried the identical unsupported referent, unchecked because no sibling field of the SAME planner output was ever covered.
- Find: DM-27 corpus-replay planner canary — run the deployed `ResponsePlanner`/STM stage over recorded turns from `logs/turn_records.jsonl` and flag any resolved topic/key-point head noun absent from BOTH the current query and the immediately preceding exchange; `(proposed)`; for a structured multi-field model output specifically, check that an embellishment guard fixed for one field is applied to every sibling field of the same schema, not just the one where the incident was first observed (a BC-58 sibling check).
- Closure: `stm_analyzer.abbreviation_expansion_conflicts` (a query token of ≤4 letters that is a strict prefix of a longer STM-resolved token absent from the query + recent window is flagged and the resolved-state line dropped) + `ResponsePlanner.unsupported_key_points` requiring the query/last-exchange, not the digest alone, to support a key point's head noun (probe-dump B4/B5); round 3 factors the same head-noun/prefix-expansion check into `ResponsePlanner._statement_unsupported` and applies it to `strategy` and every `avoid` line too, discarding the whole plan (rather than injecting an empty "Cover: (none)" plus a leaked strategy sentence) when nothing survives (probe-dump B12).
- Status: partial — the doc→doctor shape is now closed across ALL of a plan's text fields (key_points, strategy, avoid), not just key_points; the general "confident wrong-referent expansion" pattern has no standing detector until DM-27 runs across recorded turns, and other structured-output schemas in the codebase have not been checked for the same per-field coverage gap.

### BC-74 Broad continuation/affirmation arm outranks a narrower, self-sufficient explicit-request arm (arm precedence)
- Mechanism: when a continuation/affirmation arm (matches short go-ahead phrasing after a prior offer) is evaluated before, or matches more permissively than, an explicit-new-request arm on the same text, a long, fully-specified imperative gets classified as "affirming the prior offer" instead of "a new request in its own right" — the forced type then comes from the PRIOR turn's offer, not the current message. Distinct from BC-15 (a vocabulary/schema/enum the model is taught has no expressible parser match) and BC-07 (a probabilistic LLM verdict overrides an already-computed deterministic route): here both competing arms are deterministic and each individually fires correctly on its own trigger text, but no precedence rule prefers the narrower, more specific arm over the broader one when both match.
- Incidents: 2026-09-10 a 17-word explicit calendar-creation imperative ("put a recurring calendar event on my google calendar for the MGT study group, Tuesdays at 3, through Dec 4") was classified by `registry.is_offer_affirmation` as an affirmation of the prior turn's calendar offer rather than parsed as its own explicit request — the resulting action type was right only by coincidence, since a differently-typed prior offer (e.g. an unbacked email-send narration) would have forced the wrong action on the same explicit text (probe-dump handoff T6/probe 2).
- Find: for every offer/continuation-affirmation arm, check whether it runs before an explicit-action/explicit-request detector on the SAME text, and whether it is capped to a terse go-ahead shape (short, no object noun of its own) rather than any text that merely doesn't contradict the offer; `(proposed)` as a standing detector.
- Closure: `core/agentic/gate.py` runs `detect_action_intent(user_text)` BEFORE `_prior_turn_offer_action`; `registry.is_offer_affirmation` returns False when `detect_action_intent` matches the current text or the head clause exceeds a terse-go-ahead word count with an object noun of its own (probe-dump A1).
- Status: partial — the calendar-offer vs. explicit-action pair is closed; other continuation arms (note-save, retry-after-failure, thread-resolution) have not been checked for the same precedence gap.

### BC-75 Self-authored artifact laundered into evidence via a persistent store
- Mechanism: an LLM's own unverified narration or claim, made in the course of answering the current turn, is persisted through a tool call into a store that a LATER turn's context-assembly step reads back and renders as trusted background — with no attribution or verification tag distinguishing "the model said this" from "the user said this" or "this was checked." The model then treats its own earlier, uncorroborated assertion as independently-established fact, compounding a single hallucination into a recurring, self-reinforcing false context across turns. Distinct from BC-48 (a single-turn confabulated completion claim, contained to that turn's reply text) because the false claim here escapes the turn boundary through a WRITE path with no provenance concept, and distinct from BC-28 (a derived signal like tone level feeding back into its own future scoring) because what re-enters is free-text natural-language content read by the model, not a scored/derived value.
- Incidents: 2026-09-10 round 3, T1 — a note-save request ("jot down a note for this session: TA sessions are Saturdays at 11 CT,") routed correctly to `create_daemon_note`, but the SAME reply also asserted an unbacked calendar-state claim ("the recurring Saturday 11:00 AM CT calendar event … is already in place from earlier today"); the persisted note in `daemon_notes/ta-sessions-schedule-2026-09-10.md` was not limited to the user's stated content, and the following turn's [DAEMON SELF-NOTES] section fed the false claim back as apparent established fact ("A recurring calendar event was already created earlier today") — a contamination loop that will keep re-asserting itself on every future turn that surfaces this note until it is deleted (owner action pending; see this handoff's Owner section).
- Find: DM-28 — for every tool/generator that persists model-authored text into a store later re-rendered into a prompt (daemon self-notes, synthesis results, narrative/daily notes, proposal descriptions), check whether the WRITE path can distinguish and strip content the model itself asserted (as opposed to content it is merely recording on the user's behalf) from what it persists, and whether the READ/render path labels provenance the way `core/insight/provenance.py`'s "assistant-inferred" marker convention already does for insight-mode evidence — a convention this write path did not use.
- Closure: round-3 A14 — the note-save write path persists only the user's stated content (the text after the note cue/colon), stripping model-authored elaboration before the `create_daemon_note` call; a note whose body itself trips `claims_calendar_state`/`detect_completion_claims` is rendered in [DAEMON SELF-NOTES] with an appended "[unverified action claim]" marker, borrowing the insight-mode provenance-labeling convention rather than inventing a bespoke one.
- Status: open — the note-save path is guarded; the other model-output-persisting paths named in Find (synthesis results, narrative/daily-note generation, proposal descriptions) have not been swept for the same write-time attribution gap.

## J. Process (how the catalog's own fixes are made)

Not a failure mechanism in the running system — a pattern in HOW this project's
fixes are made, observed across enough dated batches to name. Filed separately
from A–I because it audits the remedy, not the defect.

### BC-76 Closure by phrase-append (narrow remedy pattern)
- Mechanism: a vocabulary/coverage miss (a phrase the model uses that a parser doesn't recognize, a keyword a matcher lacks) is closed by literally appending one more phrase/keyword to the existing regex or list, rather than adopting one of this project's own generalized remedies (a chokepoint module, a categorized-generic vocabulary with per-user anchors, or a seeds+learned/auto-promoted channel). The fix pattern itself — not any single incident — is what recurs: the next unseen phrasing of the SAME underlying vocabulary family reopens the class a batch or two later.
- Incidents: 2026-08-15 the tone-veto no_search teacher gained one exemplar phrase after "look it up" was missed, then needed a second narrowing pass 2026-08-15 for a poisoned pair; 2026-08-27 institution/action-claim regexes grew phrase by phrase per incident rather than through a resolver; 2026-09-07 calendar offer/retry vocabulary (`is_offer_affirmation`, `is_action_retry_request`, `claims_pending_card`) grew phrase by phrase across two same-day rounds; 2026-09-10 round 3 the SAME three functions (`claims_pending_card`, `claims_calendar_state`, plus a new `is_failure_report`) grew again by literal phrase addition ("card should be up", "already in place", "yes it failed") rather than a generalized mechanism — the fourth dated batch to widen this exact vocabulary family by appending phrases.
- Find: `grep -n 'gained\|added.*exemplar\|widen\|extended.*regex\|new phrase' CLAUDE_CHANGELOG.md` grouped by the function/list touched; a function or list edited in three or more dated batches for "one more phrase" each time is the signature — cross-reference against `docs/GENERALIZATION_AUDIT_20260901.md`'s "Remedy patterns" to see whether a generalized alternative already exists elsewhere in the codebase for the same shape of problem.
- Closure: none structural — `docs/GENERALIZATION_AUDIT_20260901.md` §"Remedy patterns" names four alternatives this project already uses successfully for OTHER vocabulary families (SEEDS+LEARNED: `adaptive_exemplars`; AUTO-PROMOTE/DERIVE: `learned_relations`; CALIBRATE-ON-DATA: probe scripts; CATEGORIZED-GENERIC+ANCHORS: `terms_are_private_sphere_generic`); adopting one of these for a repeatedly-patched vocabulary family, instead of the next phrase-append, is the closure — judged per family by a human/frontier reviewer, not automatable from the Find grep alone.
- Status: open — a process observation tracked so an audit can flag "this is the Nth phrase-append to the same function" as a signal to escalate to a generalized mechanism, rather than filing a tenth incident line under whatever failure-mechanism class the symptom happens to match.

## Detection methods (DM) — find instances without a full read

| ID | Method | Runs as | Classes |
|---|---|---|---|
| DM-01 | Raw-substring / negation audit: `rg` for `in <x>.lower()` and cue regexes not routed through `utils/trigger_match.py` | `check_bug_classes.py scan` — dm01_raw_substring (gated) | BC-01, BC-02 |
| DM-02 | Registry parity tests that walk a table and assert every entry is wired | `test_tool_wiring_parity`, `test_model_capability_wiring`, `test_budget_meters_rendered_sections`, `test_api_error_fail_fast` | BC-10, BC-15, BC-17, BC-22, BC-23 |
| DM-03 | Ordered-slice AST guard (content-anchored allowlist) | `tests/unit/test_ordered_slice_guard.py` | BC-18 |
| DM-04 | Same-instance before/after settings probe | test pattern, `test_sep09_live_controls.py` | BC-11 |
| DM-05 | Read → mutate → read on one live instance | test pattern | BC-31 |
| DM-06 | Failure injection between the destructive and the recovery step | test pattern (`threading.Event` barriers, never sleeps) | BC-33, BC-35 |
| DM-07 | Real-driver contract: fixtures via the real writer, fakes validated on `chromadb.EphemeralClient` | test pattern | BC-20, BC-34, BC-64 |
| DM-08 | Assertion-sensitivity / perturbation proof: corrupt the provider or fixture, the test must go red | script in HANDOFF_20260909 §T01; `test_no_vacuous_assertions.py` | BC-62, BC-63 |
| DM-09 | Deployed-function adversarial probe with live texts as fixtures | `scripts/probe_*.py` pattern; regression fixtures | BC-04, BC-05, BC-06 |
| DM-10 | Telemetry queries over `logs/turn_records.jsonl` (`jq`): `tone_trigger` shares, `gate_reason` by mode, `grounding_flagged`, `wall_elapsed_s` outliers | shell | BC-06, BC-28, BC-50, BC-72 |
| DM-11 | Log-signature greps: duplicate trigger calls seconds apart; "Created sandbox" vs 0 executions; repeated per-turn WARNINGs | shell over `daemon_debug*.log` | BC-32, BC-41, BC-10 |
| DM-12 | Doc-claim verification sweep: read-only auditors, one per doc slice, claim → file:line | agents | BC-71 |
| DM-13 | Unignore rerun under the CI marker filter | shell | BC-66 |
| DM-14 | Peak-RSS measurement of the exact hook/CI selection under `systemd-run -p MemoryMax=` + `/usr/bin/time -v` | shell | BC-67 |
| DM-15 | Sibling-site enumeration after a fix: every read AND write of the primitive, every sibling by naming pattern | grep, manual | BC-58, BC-27, BC-13 |
| DM-16 | Config-key reachability: each YAML leaf → readers outside schema/app_config | `check_bug_classes.py scan` — dm16_config_key_reachability (report-only, 87 live candidates) | BC-12, BC-10 |
| DM-17 | `grep -L daemon_guard scripts/*.py \| xargs grep -l -- --apply`; `rg '"data/' tests/` | `check_bug_classes.py scan` — dm17_apply_without_guard (gated) | BC-37 |
| DM-18 | Broad `except` returning an empty result beside a store call | `check_bug_classes.py scan` — dm18_except_returns_empty (gated) | BC-20, BC-47 |
| DM-19 | Dead-call kwarg grep + kwargs-capturing fake client | grep + test pattern | BC-14 |
| DM-20 | Prohibitive prompt instruction → deterministic sibling check | grep `config/prompts/*.txt` | BC-46 |
| DM-21 | Contamination and junk reports on the deployed predicates | `scripts/report_claim_contamination.py`, `purge_junk_facts.py` dry-run, exemplar cap check | BC-54, BC-55, BC-56, BC-29 |
| DM-22 | Static floor: ruff bug rules, `compileall`, `vulture --min-confidence 90`, `git diff --check`, `PytestReturnNotNoneWarning` | CI + pre-push | BC-68, BC-62 |
| DM-23 | Live-turn probe + debug-dump review (designed turns, then `turn_records` and debug record line by line) | owner relay | all E-family |
| DM-24 | Disposable-store runtime smoke and read-only budget replay | `scripts/audit_runtime_smoke.py`, `scripts/audit_context_budget.py` | BC-22–BC-26 |
| DM-25 | Controlled race reproduction (two threads into one loader; count invocations) | test pattern | BC-38, BC-39 |
| DM-26 | Receipt rollups (`scripts/latency_rollup.py`; web-trigger rollup planned) | shell, read-only | BC-72, BC-41 |
| DM-27 | Corpus-replay planner canary: run the deployed planner/STM stage over recorded turns, flag key points or resolved referents whose head noun is absent from the query + last exchange | script `(proposed)` | BC-51, BC-73 |
| DM-28 | Persisted-model-output sweep: for every tool/generator writing model text into a store later re-rendered into a prompt, check write-path attribution stripping and read-path provenance marking | grep + manual | BC-75 |
| DM-29 | Changelog phrase-append signature: group `gained`/`added exemplar`/`extended regex` hits by touched function/list; ≥3 dated batches on the same one is the signature | `check_bug_classes.py scan` — dm29_phrase_append_signature (report-only; the judgment stays human) | BC-76 |

## Closure methods (CM) — what has actually stopped a class

| ID | Method | Examples |
|---|---|---|
| CM-01 | One chokepoint module, adopted at every site | `utils/trigger_match.py`, `utils/ordered_slice.py`, `utils/safe_json.py`, `utils/daemon_guard.py`, `memory/fact_source.py`, `utils/privacy_redaction.py`, `sanitize_for_storage`, `ACTION_SPECS`, `MODEL_CAPABILITIES`, `API_ERROR_PREFIXES` |
| CM-02 | A guard/parity test that fails on drift or on a new instance | the DM-02 tests, `test_ordered_slice_guard`, `test_no_git_state_in_tests`, `test_no_vacuous_assertions`, `test_atomic_writer_guard` |
| CM-03 | Deterministic executor or post-check instead of a prompt instruction | `wall_clock_time`, `weekday_date_mismatches`, `unsupported_key_points`, `streak_claims`, `status_claims`, `claims_pending_card`, visual-intent gate |
| CM-04 | Log-only / shadow mode until precision is measured | review gate, grounding verifier, curation trust ladder |
| CM-05 | Tri-state outcomes: unavailable ≠ zero, with `source`/`reason` | `source=fallback`, `grounding_status`, "not triggered" labels |
| CM-06 | Live-config reads at call time; setters push into live instances | import doctrine case 3, `_live_web_search_managers()` |
| CM-07 | Write-ahead journal, pre-images, staged replacement, unique temp names, undo with explicit neutral values | curation engine, refdocs upload, `atomic_write_json` |
| CM-08 | Survival-minimum floors with a quality margin, budget after floors | `GATE_FORCED_FLOOR_MARGIN`, `MEMORY_TOPUP_FLOOR` |
| CM-09 | Teachers only from independent channels, gated recording, purge scripts | adaptive exemplars, learned relations |
| CM-10 | Chain budgets and exits for derived signals | `TONE_FLOOR_CHAIN_MAX`, `distress_grounding_max` |
| CM-11 | Isolation: test mode, sandbox-only scripts, daemon guard, `DAEMON_TEST_MODE` | conftest fixtures, `generate_test_facts.py --sandbox-dir` |
| CM-12 | Receipts in telemetry and debug for every decision, plus read-only rollups | `gate_reason`, `tone_trigger`, `grounding_*`, `answer_call`, `latency_rollup.py` |
| CM-13 | Docs move with code; doc-claim audit at batch boundaries; generated counts | §3a, `generate_doc_metrics.py`, the 09-10 audit |
| CM-14 | Regression fixtures are the exact live texts against the deployed function | every `test_sepNN_*` file |

## Audit lenses (how past audits were run)

- Six-lens frontier sweep (08-31): new-module wiring · agentic/action stack · handlers+grounding · config consistency · silent-failure sweep · test honesty; every claim verified file:line.
- Runtime-vs-getsource discrimination (08-31 F12): tests calling the deployed function vs tests pinning source text or gated by a disabling fixture.
- Static suite (09-03): ruff, compileall, per-area pytest, vulture, `git diff --check`, `npm run build`.
- Read-only aggregate replay + disposable-store smoke (09-05): `audit_context_budget.py`, `audit_runtime_smoke.py`.
- Matched production log-trace replay (09-05): one `daemon_debug*.log` trace annotated phase by phase.
- Negative-reproduction-before-fix (09-05, 09-09): a regression that fails on the original code for the claimed reason.
- Producer→consumer tracing over real drivers, bounded failure injection, sibling-path checks, reproduction vs observation kept distinct (09-09 audit method).
- AST extraction of test bodies executed against corrupted doubles (09-09 T01).
- Frontend executed outside a browser: `ts.transpileModule` + Node `vm` (09-09 F07/T09).
- Failed-before/passed-after receipts from scratch-copied `git show HEAD:` content, never a checkout (B1–B6).
- Live-turn probes after every deployed fix; retest rounds until the exact failing shape passes (every batch).
- Doc-claim verification by parallel read-only auditors (09-10).

## Coverage review (Codex, 2026-09-10)

Cross-checked this catalog against the incident families and open findings in
`BUG_RETROSPECTIVE_20260715_20260904.md`, both generalization audits, the
09-05 independent/runtime audits, the 09-06 conversation audit, the 09-08 and
09-09 independent audit handoffs, the 09-09 follow-up ledger, and both 09-10
handoffs. The 76 classes cover every repeated mechanism named in those
sources; the remaining one-off mechanisms are retained below rather than
promoted without recurrence evidence.

The calendar forced-action handoff added incidents, not a new mechanism. Its
failures map to BC-04/06/15/21/30/46/48/58/64: missing request anchors, gate
over-fire, producer/consumer vocabulary drift, enum serialization comparison,
proposal lifetime, prompt-only grounding, unsupported completion claims,
sibling-path omissions, and a fake that did not match the real enum contract.
The web-search handoff maps to BC-05/47/72: the only competent classifier was
short-circuited, not-run collapsed into zero results, and the decision had no
receipt. This review updated those incident and closure lines.

The 2026-09-10 evening probe-dump handoff (T1-T6, six turns after the same
day's restart) added incidents to BC-05, BC-06, BC-15, BC-30, BC-46 and BC-58,
and introduced two new mechanisms. BC-73 (an ambiguous short-form token
expanded to an unstated referent via background context — "doc" read as
"doctor") is a resolution-time ambiguity failure distinct from BC-51's
downstream embellishment of a stated fact. BC-74 (a broad continuation/
affirmation arm outranking a narrower explicit-request arm) is an arm-
precedence failure between two individually-correct deterministic matchers,
distinct from BC-15's vocabulary/schema gap and BC-07's probabilistic-override
of a deterministic route. DM-27 (corpus-replay planner canary) was added to
detect the BC-73/BC-51 shape across recorded turns without a live probe.

The same handoff's round-3 retest (T1-T5, 20:56-20:59) added incidents to
BC-15, BC-46, BC-58, BC-64 and BC-73, and introduced two more mechanisms.
BC-75 (a self-authored artifact laundered into evidence via a persistent
store — the model's own unbacked calendar-state claim got written into a
daemon self-note and fed back as trusted context the following turn) is a
write-path provenance failure distinct from BC-48 (a same-turn confabulated
claim, contained to that turn) and BC-28 (a scored/derived signal, not
free text, feeding back). BC-76 (closure by phrase-append) is filed as a
PROCESS class in a new §J, not a runtime failure mechanism: it names the
recurring pattern, across four dated batches (08-15, 08-27, 09-07, 09-10),
of closing a vocabulary-drift incident (BC-15's territory) by appending one
literal phrase to a regex/list instead of adopting one of this project's own
generalized remedies (`docs/GENERALIZATION_AUDIT_20260901.md` §"Remedy
patterns") — this round's own BC-15 fix (A12/A13) is itself an instance,
noted honestly rather than exempted.

Completeness here means all mechanisms encountered in the cited repository
audits through 2026-09-10 are either classified or listed as a singleton. It
does not claim that unknown production defects have already been enumerated;
new audits still apply the lenses and checklist below.

## Unclassified singletons (one incident, no mechanism family yet)

- Multi-item backend result collapsed to one at the SPA boundary (F07, fixed).
- Server temp filename shown instead of the display name (09-04, fixed via `attachment_display_name`).
- One oversized tool block with no delimiter escapes the cap (INDEPENDENT F5, fixed).
- Parallel-round telemetry join breaks under concurrency (F32).
- Time-blind calendar duplicate pre-flight (F16, policy).
- Asymmetric persisted canonical form for integrated turns (F25, deferred).
- Non-idempotent `label_evidence` at two call sites (09-06, guarded).
- Upload registration before the aggregate size check, no rollback (09-03, fixed).
- Scoped Chroma id assumed globally unique in cross-collection dedup (09-03, fixed).
- Unescaped interpolation into `gr.HTML` (09-03, fixed).
- Debug export re-imported into a session compounds context (09-05, open).
- NetworkX DiGraph keeps one relation per pair (09-03, index-authoritative reads).
- SIGHUP from a dead SSH session killed the idle shutdown (09-07, fixed).
- `config.yaml` comments dropped on every settings save (FOLLOWUPS #4, open).
- Docker path stale (FOLLOWUPS #10, open; also BC-71).
- Repeated final-answer display duplication: investigated, not reproduced (09-05).

## Adding a class — checklist

1. Is it really a new MECHANISM, or a new incident of an existing one? Read the family first.
2. Name it by mechanism, not symptom ("consumer reads what no producer sets", not "web search broken").
3. Two incidents or a strong reason to expect recurrence; otherwise it goes under singletons.
4. Write the Find line as something a cheap agent can run today. Mark `(proposed)` if untested.
5. Name the closure precisely, or write "per-incident". Do not describe a plan as a closure.
6. Set Status honestly. `closed` requires a guard that FAILS on recurrence.
7. Add the ID to the index, to the commit body, and to the changelog line.
