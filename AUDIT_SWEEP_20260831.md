# Audit sweep — uncommitted tree, 2026-08-31

Six-lens frontier audit (new-module wiring, agentic/action stack, handlers+grounding, config
consistency, silent-failure sweep, test honesty) over the ~3,400-line uncommitted diff + 19 new
files. Findings deduped and severity-ranked. **This file doubles as the execution plan for the
cheap agent** — follow the Fix Plan section verbatim; escalation triggers at the bottom.

Every claim below was verified against working-tree code by a dedicated auditor (file:line cited).
One finding (F4) was additionally confirmed by an actual test run.

---

## P0 — confirmed broken

**F1. `_retry_fetch_urls_from_context` is dead on arrival (never fires on either production path).**
`gui/handlers.py:2416-2440`, consumed at `:2493`. It takes the *last* user / *last* assistant
message from `chat_history`, but both callers pass history that already contains the current turn:
- SPA: `api/chat_service.py:59` appends the current user message BEFORE `handle_submit`, so
  `previous_user` is the retry message itself ("try again, I fixed it") — no URL → `[]`.
- Legacy Gradio: `gui/launch.py:971-972` appends user + an assistant `"…"` typing placeholder, so
  the "blank page"/"couldn't fetch" check tests `"…"` → `[]`.
The `_fastpath_ok` retry arm (`handlers.py:2497-2504`) is unreachable; zero test coverage.
This is the digest-order class, 5th occurrence.

**F2. `PendingActionsStore` persistence silently no-ops during agentic dispatch (FSGuard).**
`core/actions/types.py:158-172` `propose()` → `_save()` → `os.replace` onto
`data/pending_actions.json`, called from `core/agentic/tools.py:1188` inside
`_fs_agent_mode()` (`core/agentic/controller.py:1250`). `data/` is protected; the exemption list
`_DAEMON_STATE_EXEMPT_PREFIXES` (`utils/python_fs_guard.py:43-45`) still contains ONLY
`web_search_credits.json` (exact 07-16 incident class). `_save()` swallows the PermissionError at
warning level (`types.py:133-134`) — so the store's core scenario (proposal survives a restart
before approval) is dead, and the tmp-file cleanup `os.remove` is also blocked → `.tmp` leak.

**F3. `pattern_temporal` deterministic detector hijacks mundane/general questions, veto-exempt.**
`core/insight/detector.py:224-241, 175-183, 331-372`; gate return at `core/agentic/gate.py:932-944`.
Three holes: pattern #2 (`how many times …`) has NO personal anchor ("How many times has Trump
been impeached?" → insight mode); pattern #4 (`trend in …`) has optional `my`; and
`_IMPLICIT_PERSONAL_COMPARISON_RE` matches aux + first-person + {changed|before|after|more|less…}
within 100 chars ("Is my appointment before noon tomorrow?", "Have I told you about Morgan
before?" → should be memory recall). `_trigger_is_incidental` only guards >25-word messages.
Consequence: a logistics question runs the full 35s deliberation coordinator (LLM planner +
Tavily + PubMed adapters, `gui/handlers.py:1519-1627`) and bypasses every tone veto. Contradicts
the "UNDER-fires by design" comment. Negative tests cover only vents + 3 easy negatives.

**F4. Red test on any run after 08-29 (CONFIRMED by run: 1 failed / 25 passed).**
`tests/unit/test_calendar_turn_round3.py:586-587` asserts the hardcoded string
`'Saturday, 2026-08-29'` against a real `datetime.now()` clock block (`gui/handlers.py:2319-2323`).
It only ever passed on the day it was written.

## P1 — likely production impact

**F5. Round-2 "XML can't express actions" bug survives on UNFORCED turns.**
`core/agentic/protocols.py:1130-1137` keeps the old fixed-group `propose_action` regex
(type/recipient/subject only), but the always-injected XML vocabulary
(`core/agentic/types.py:1285-1289`) teaches `<propose_action type=...>` and lists
`calendar_create_event`. A model following its standing instructions emits attrs the regex can't
match → marker silently falls through to `wants_answer` with raw XML as the "answer". The
`<invoke name="propose_action">` fallback (`protocols.py:1540-1555`) forwards only
message/recipient/subject → calendar proposal with no calendar fields that fails AFTER approval.
Only `<action>` (forced-round syntax) got the generic attr parsing.

**F6. XML attr parsing forwards model-supplied `repo` — trust-doctrine bypass.**
`protocols.py:1460-1481` puts every `key="value"` into `action_params` unfiltered; native path
filters via `spec.forward_params` precisely to drop this; `core/actions/github_write.py:64`
honors `params["repo"]` FIRST. `<action type="github_create_issue" repo="someone/else">` targets
an arbitrary repo and the approval card never shows it.

**F7. Gate pattern-preemption: LLM verdict overrides deterministic routing + cost regression.**
`core/agentic/gate.py:949-990` + `utils/web_search_trigger.py:843-865, 1378, 1465-1471`.
(a) `_looks_like_pattern_candidate` is extremely broad (first-person + before/after/notes/data/
changed/…), so the gate now awaits an LLM call for a wide slice of ordinary turns; (b) a
`needs_pattern_analysis` verdict early-returns `modes=["insight"]`, DISCARDING a deterministic
`_explicit_action`/`needs_files`/web route computed above it (LLM overriding deterministic —
inverted doctrine; "Add my notes review to my calendar" can lose its write action); (c) on a
"not pattern" verdict Tier 4 calls `analyze_for_web_search_llm` a second time; (d) the whole
block's `except` logs at DEBUG (`gate.py:981-984`) — a crash kills the routing invisibly;
(e) pattern candidacy bypasses `quick_prefilter` and both heuristic short-circuits, re-opening
LLM exposure for personal-state statements the 08-05 guard excluded.

**F8. StackExchange epoch-int date crashes the whole insight/deliberation turn.**
`knowledge/research_search.py:85` passes the API's int `creation_date` → handlers
`EvidenceItem(date=...)` (`gui/handlers.py:1689-1705`) where `date` is Pydantic-v2 `Optional[str]`
(`core/insight/types.py:84`) — int is NOT coerced → ValidationError → outer except at
`handlers.py:1946` abandons the turn AFTER the deliberation spent its LLM/API budget.

**F9. Agentic debug record built BEFORE grounding mutates the response.**
`gui/handlers.py:2806` (record) vs 2859-2893 (action guard + grounding integration). On exactly
the turns grounding changed facts, the debug/audit surface shows the uncorrected text. Enhanced
path does it right (record at 3451, after grounding at 3435-3441).

**F10. Deliberation-shape check shadows the adversarial assessment path.**
`core/insight/detector.py:331-341` runs before `_ASSESS_PATTERNS` (349); non-temporal assessment
shapes ("assess my theory against my history") route `pattern_temporal` and lose the fail-honest
worst-of `assess()` machinery (handlers only runs it for `kind == "insight_assessment"`,
`handlers.py:1838`). Comment at 343-344 still claims assessment is "checked first".

**F11. `_direct_fetch` reads unbounded bodies when Content-Length is absent.**
`knowledge/web_search_manager.py:1216-1223` checks the 2MB cap against the HEADER only, then
`client.get()` buffers the whole body before slicing. Chunked responses can pull hundreds of MB
on the 16GB box. Fix: stream with a byte cap.

**F12. The insight `pattern_temporal` handlers branch (~230 lines) has ZERO runtime coverage.**
`gui/handlers.py:1495-1699` — coordinator construction, 9 adapter closures, keepalive loops,
freeze-status branches, EvidenceItem merge. Only an `inspect.getsource` string test exists
(`tests/unit/test_insight_pattern_facet.py:216-227`). A kwarg typo there passes every test in
the tree. Same-class: SSRF validators are only ever referenced by the autouse fixture that
DISABLES them (`tests/unit/test_url_fetch_layers.py:211-216`) — no test that a private IP is
rejected or that the validator is invoked.

## P2 — worth fixing before/with commit (deduped)

- **F13.** Forced-action retry can re-arm after the action already dispatched → duplicate
  proposal (`controller.py:1046-1059`; `_forced_action` never cleared; multi-marker round path).
- **F14.** Pending store: capacity counts terminal proposals forever (5 outcomes → all future
  proposals rejected until restart, `types.py:158-172, 234-249`); restored-expired proposals
  consume slots; multi-proposal turns orphan all but the newest via `get_pending()`.
- **F15.** `get_runtime_action_health` reports AVAILABLE for an expired token with no refresh
  token (`registry.py:246-263`, `google_auth.py:96-99,184-189`) — disk-only expiry check available.
- **F16.** Calendar duplicate pre-flight is time-blind (same title+day skips a different TIMED
  event, reports success, `google_calendar_create.py:19-21, 219-229`); all round-3 tests
  unknowingly run the "dup check unavailable" fallback (bare AsyncMock `client.get`).
- **F17.** `pattern_analysis.enabled` / `PATTERN_ANALYSIS_ENABLED` gates ONLY the agentic
  pattern_scan tool — the v1 primary surface (insight pattern_temporal) ignores it. Kill switch
  that doesn't kill. (2 auditors independently.)
- **F18.** conftest pending-actions sandbox is conditional on prior import
  (`tests/conftest.py:185-203` `sys.modules.get(...)`) — lazy-importing test writes PROD
  `data/pending_actions.json`, which the persistent store would RESTORE into the live daemon.
  (3 auditors independently.)
- **F19.** Insight debug record rebuilds the synthesis prompt WITHOUT `patterns`/
  `deliberation_manifest` (`handlers.py:1917-1922` vs 1855-1863) — understates by up to 14K chars.
  (2 auditors.)
- **F20.** `ACTION_ATTR_RE` (`protocols.py:1114/1123`) truncates values at the other quote char —
  `summary="Miller's exam"` → `Miller`; and `<pattern_scan spec='{json}'/>` attr form can never
  carry JSON (lazy match stops at first interior quote → garbage spec runs as "computed evidence").
- **F21.** `regenerate_final_answer` uses a cross-turn stash (`controller.py:1678-1720, 1787-1790`)
  — null `_last_final_prompt/_model/_system_prompt` at the top of `run_agentic_search`.
- **F22.** `ctx.agentic_fallback_reason` write-only (`handlers.py:2481`); `[APPLICATION ACTION
  STATUS]` block build wrapped in `except: pass`; gate/grounding degradation logs at DEBUG —
  promote to warning (the "how the borderline arbiter stayed dead" pattern).
- **F23.** Grounding source-material: handlers collects 6000 chars (`handlers.py:2885`) but
  verifier truncates at `_SOURCE_MATERIAL_TRUNC=3500` (`grounding_check.py:219`) — silent discard.
  Also enhanced path never passes source_material (accepted v1 scope).
- **F24.** `grounding_corrected=True` telemetry set before the integrate attempt (`handlers.py:2346`)
  — can claim a correction that never shipped; set immediately before each return.
- **F25.** Integrated turns persist a different canonical form (linkified citations + Sources
  footer + proposal card) into the corpus than non-integrated turns; verifier/integrator also
  see + may rewrite the proposal card.
- **F26.** Wiki-chroma timeout: log says "skipping wiki this turn" but control falls through to
  the LIVE Wikipedia API fallback; timed-out queries keep running in the 2-worker
  `_WIKI_CHROMA_EXECUTOR` with no inflight guard → pool exhaustion (the zombie-thread class,
  `gatherer_knowledge.py:1315-1365`).
- **F27.** `is_personal_doc_search` false-positives on third-party docs ("Look at the FastAPI
  docs … my SPA" → deterministic no-search + file-tool reroute, `query_checker.py:539-570`).
- **F28.** `_SONG_FRAME_RE` marks 1200+-char "listening to this podcast…" messages as lyrics 0.70
  (`content_type_detector.py:66-70,139-143`) — also poisons the pattern engine's content_type dim.
- **F29.** `is_continuation_answer` fires on "?" ANYWHERE in the prior reply (`query_checker.py:225`)
  — nearly every Daemon reply contains one; terse new statements get misframed as answers.
- **F30.** XML injection never teaches `<pubmed>`/`<pattern_scan>`/arxiv/SE tags — parse-only
  wiring on the deployed (kimi-3 = XML) protocol; agentic surface of those tools unreachable.
- **F31.** Gate LLM-route pattern intents hardcode `window_days: 0` (`gate.py:979,1046`) —
  `parse_window_days` is available; "last 3 months" lost on that route.
- **F32.** Telemetry round join: parallel tool rounds `base_round+i` never match
  `session.current_round` (`types.py:608-617`, `controller.py:827,1110`) — decision_ms lost after
  round 1; `_decision_action` logs pattern_scan as "done".
- **F33.** `_PATTERN_TAIL` rule 9 trailing backslash glues rules 9+10 into one line
  (`core/insight/synthesizer.py:100-102`).
- **F34.** SSRF TOCTOU: DNS validated, then httpx re-resolves (rebinding window,
  `web_search_manager.py:1188-1197`) — accepted risk unless transport pins the validated IP.
- **F35.** Misc: `MEMORY_TOPUP_FLOOR` YAML key exists nowhere (code default only; inherited
  pattern); `GOOGLE_CALENDAR_MAX_EVENTS` now overloaded (prompt fetch limit AND batch cap);
  `_ENQUEUE_RE` lazy match drops enqueue blocks containing `")`; `_REQUEST_FRAMING_STOPWORDS`
  is me-shaped (built from one live query); deliberation progress bubble renders the entire
  message as theme; `InsightIntent.dimension` has no producer (relation/session_rhythm dims are
  test-only); `freeze_query`/`freeze_spec` dead shims; `validate_and_freeze` all-or-nothing on
  minor planner omissions; pubmed worst-case latency > coordinator 25s adapter timeout;
  duplicated `if results:` in gatherer_knowledge:1326.

## Verified correct (for confidence calibration)

Grounding tuple contract + whole-bubble replacement + fail-open paths; runtime clock built at
call time and truncation-safe; weekday demotion at parse AND integrator-vet; calendar batch
executor (validate-all-before-any, all-day fields, partial-failure honesty); gate action arm +
coalescing + XML force path + session-scoped retry flag; `get_runtime_action_health` no-network
claim; pattern_scan corpus injection wired end-to-end (orchestrator→controller→ToolExecutor);
pattern engine config/telemetry/store contracts; deliberation stack is genuinely fail-honest
(channel statuses, claim demotion, insufficient-not-guessed); fetch_url layering + Tavily-only-
on-shortfall; STM chronological fix; MEMORY_TOPUP_FLOOR runtime behavior; safety-tracker and
post-response-detector call sites intact; no schema rejection risk (validate-only). Test suite
quality is well above repo baseline (live-turn reproductions through deployed entry points,
boundary-only mocking) — weaknesses are the getsource string-pins at handler/controller glue.

---

# FIX PLAN (for the executing agent)

Ground rules: work in small commits-worth of change-sets but DO NOT COMMIT (human-gated).
After each step run the named tests memory-capped:
`systemd-run --user --scope -p MemoryMax=9G python -m pytest -q -m "not slow and not semantic and not benchmark" <files>`.
Never run the full suite in one process. Match surrounding code style. Every fix gets a runtime
(not getsource) regression test unless noted.

**Step 1 — F2 (one line + test).** Add `"data/pending_actions.json"` to
`_DAEMON_STATE_EXEMPT_PREFIXES` in `utils/python_fs_guard.py`. Test: under
`agent_mode`, `PendingActionsStore.propose()` persists to a tmp-sandboxed path with the guard
active. Contingency: if the guard matches by prefix and the store path is configurable
(`PENDING_ACTIONS_STORE_PATH`), exempt the RESOLVED configured path, not just the literal.

**Step 2 — F4 (test-only).** In `tests/unit/test_calendar_turn_round3.py:586`, replace the
hardcoded date assert with `datetime.now().astimezone().strftime('%A, %Y-%m-%d')` computed in
the test, or a regex `r"[A-Z][a-z]+day, \d{4}-\d{2}-\d{2}"` + assert the header string. Run the
file; expect 26/26.

**Step 3 — F1.** Fix `_retry_fetch_urls_from_context` to (a) skip messages whose content equals
the current `user_text`, (b) skip assistant entries that are `"…"`/empty placeholders, then take
last-assistant + the user message PRECEDING it. Add runtime tests with BOTH real history shapes
(SPA: current-turn-appended; Gradio: +placeholder). Contingency: if history shapes are unclear,
read `api/chat_service.py:50-70` and `gui/launch.py:960-980` first; if still ambiguous → escalate.

**Step 4 — F3 (+F10, F31).** In `core/insight/detector.py`: add a mandatory personal-record
anchor (`my|I've|I have|me`) to `_PATTERN_TEMPORAL_PATTERNS` #2 and make `my` non-optional in #4;
shrink `_IMPLICIT_PERSONAL_COMPARISON_RE` — drop bare `before|after|more|less` (require a
temporal-span cue like `over the (last|past)|since (January|20\d\d)|these (days|weeks)`), and
apply `_trigger_is_incidental`-style guarding to short queries too. Move `_ASSESS_PATTERNS`
check back ahead of the deliberation-shape check (or restrict deliberation-shape to inputs with
a temporal cue) and fix the stale comment. In `gate.py:979,1046` use `parse_window_days(user_text)`
instead of hardcoded 0. Add negative tests: "How many times has Trump been impeached?",
"What's the trend in AI regulation?", "Is my appointment before noon tomorrow?", "Have I told
you about Morgan before?" (must NOT be pattern_temporal), plus positives from the existing suite
must stay green (`test_insight_pattern_facet.py`, `test_pattern_routing_generic.py`,
`test_deliberation_coordinator.py`). Doctrine: UNDER-fire; a missed pattern query costs a retry,
a hijacked logistics turn costs 35s + credits + a wrong answer.

**Step 5 — F7.** In `gate.py:949-990`: (a) stand the preemption down when `_explicit_action is
not None` or Tier-1 already produced needs_tools/needs_files/web routing; (b) cache/reuse the
LLM result so Tier 4 doesn't call again (the trigger already caches by (query, context) — verify
the gate call passes the same key; if so this is free); (c) raise the except log to
`logger.warning`; (d) in `web_search_trigger.py`, when `is_personal_state_statement` fires,
do NOT let pattern candidacy fall through to the LLM (personal-state guard wins). Tighten
`_looks_like_pattern_candidate`: require pronoun + record-word (notes/history/data/record/
pattern/trend) rather than pronoun + any comparator. Tests: explicit calendar request +
pattern-y phrasing routes to tools, not insight; vent with "before" does not reach the LLM.

**Step 6 — F5 + F6 + F20.** In `protocols.py`: (a) generalize `PROPOSE_ACTION_PATTERN` to the
same generic-attr parse as `<action>` (or rewrite the XML injection tool-14 text to teach
`<action>` syntax instead — pick ONE canonical taught form and make the parser accept both);
(b) filter XML `action_params` through `spec.forward_params` exactly like the native path
(preserving `message` merge) — this kills the `repo` bypass; (c) fix `ACTION_ATTR_RE` to match
`"([^"]*)"|'([^']*)'` (paired quotes) so apostrophes survive; same for the `spec` attr — and for
`<pattern_scan>` prefer body-form JSON in the taught syntax; (d) `<invoke name="propose_action">`
fallback: forward via `spec.forward_params` too (calendar fields currently dropped). Extend
`tests/unit/test_tool_wiring_parity.py` or the round2/3 files with runtime parses:
`<propose_action type="calendar_create_event" summary="Miller's exam" start_time="..."/>` must
yield a complete proposal; `repo="someone/else"` must NOT survive into params.

**Step 7 — F8 (one line + test).** In `knowledge/research_search.py:85` (and any sibling site):
`"date": str(item["creation_date"]) if item.get("creation_date") is not None else None` — or
normalize epoch→ISO. Test: `parse_stackexchange_items` output feeds
`EvidenceItem(date=row["date"])` without ValidationError.

**Step 8 — F9 + F19 + F24.** Handlers: (a) agentic path — move `debug_record` build after the
grounding block (mirror enhanced), or update `debug_record["response"]` post-mutation;
(b) insight debug — pass `patterns=_patterns, deliberation_manifest=...` to the debug-side
`build_synthesis_prompts` call so record == sent; (c) move `grounding_corrected=True` to
immediately before the successful returns. Runtime tests where feasible; at minimum assert
`debug_record["response"] == displayed_final` in a driven fake-run.

**Step 9 — F11.** `_direct_fetch`: stream the response (`client.stream("GET", ...)`) accumulating
up to `_DIRECT_FETCH_MAX_BYTES`, abort beyond. Keep header fast-reject. Test with a fake
transport emitting >cap chunked bytes → truncated, no memory blow.

**Step 10 — F13 + F21.** Controller: guard the forced-action retry on "no action decision
dispatched this session" (scan `session.rounds` for an action round or set a session flag at
dispatch); null `_last_final_prompt/_model/_system_prompt` at the top of `run_agentic_search`.
Runtime test: round emits action+tool marker, later round is tool-less → NO second force.

**Step 11 — F18 (conftest).** Make the pending-actions sandbox unconditional: import
`core.actions.types` at fixture top and patch always; keep the ToolExecutor reset. Also set the
path via `PENDING_ACTIONS_STORE_PATH`-equivalent monkeypatch so lazily-constructed stores hit tmp.

**Step 12 — F22 + F26 (logging/hygiene).** Promote the named DEBUG degradation logs to warning;
record `agentic_fallback_reason` into `ctx.telemetry`; log (warning) when the enhanced action-
status block build fails. Wiki-chroma: on timeout, actually SKIP wiki (no live-API fallthrough)
and add an inflight-semaphore skip like `_WIKI_SEM_INFLIGHT` (copy that pattern).

**Step 13 — smaller P2s as time allows**, each trivial: F15 (expiry+refresh-token disk check →
UNAVAILABLE), F23 (raise `_SOURCE_MATERIAL_TRUNC` to 6000 or lower the collect cap to 3500 —
pick one, keep clock-first), F28 (require song/track/lyrics noun near "listening to this"),
F29 (require the "?" in the last ~2 sentences of the prior reply), F33 (delete the stray
backslash), F32 (thread `base_round+i` into the telemetry key), F17 (check
`PATTERN_ANALYSIS_ENABLED` in the detector or gate arm), F27 (require possessive-anchored doc
noun), F14 (evict terminal proposals from the slot count).

**Deferred to owner/frontier (do NOT attempt):** F16 policy (time-blind dup check — needs a
product decision on same-title-different-time), F25 canonical-form asymmetry (touches storage
semantics), F34 SSRF pinning (transport surgery), F30 (teaching XML tags to the model = prompt
budget decision), F12 full runtime harness for the insight branch (worth doing, but design the
fake-coordinator fixture with the frontier first).

## Escalation triggers — STOP and ask the frontier agent when:
1. Any named test file fails for a reason unrelated to your change (pre-existing red ≠ yours to fix silently).
2. A fix requires changing a function signature used by >2 call sites, or touching storage/
   persistence semantics (safe_json stores, corpus, chroma).
3. The two history shapes in Step 3 don't match what you observe in the code.
4. Anything in Step 6 forces a choice between changing the taught prompt vs the parser — that's
   a design decision, present both options.
5. You would need to write to `data/`, `logs/`, or any live-daemon store to test something.
6. A regression test you write passes even when you revert the fix (your test is dishonest — redo it).
