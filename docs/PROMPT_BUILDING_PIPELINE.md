# Prompt Building Pipeline

Operational guide for the modular prompt assembly system
(`core/prompt/`: builder, context gatherer + 3 retrieval mixins,
formatter, hygiene, summarizer, token manager, proposal filter) —
parallel retrieval, token budgets, section ordering, intent-driven
overrides, and middle-out compression.

For formal notation see `FORMAL_MODEL.md`. For config constants see
`QUICK_REFERENCE.md`. For memory retrieval details see `MEMORY_SYSTEM.md`.

---

## What the Pipeline Does

Every user query triggers a full prompt build: up to 22 parallel async
retrievals across memory, knowledge graph, web search, Google Calendar,
files, and profile data. Results are filtered, deduplicated, scored, compressed
to fit a token budget, and assembled into a final prompt string with
carefully ordered sections that exploit LLM attention patterns.

The pipeline lives in `core/prompt/` and is orchestrated by
`UnifiedPromptBuilder.build_prompt()`.

---

## File Map

| File | Purpose |
|------|---------|
| `core/prompt/builder.py` | Thin orchestrator (~1,821 lines): parallel task dispatch, intent overrides, budget, eval hooks. Delegates assembly to formatter and hygiene to ContentHygiene |
| `core/prompt/context_gatherer.py` | Mixin compositor (~380 lines): init, properties, utilities. Composes WebSearchMixin + MemoryRetrievalMixin + KnowledgeRetrievalMixin |
| `core/prompt/gatherer_web.py` | WebSearchMixin (~226 lines): `_get_web_search_results()`, `should_trigger_web_search()` |
| `core/prompt/gatherer_memory.py` | MemoryRetrievalMixin (~946 lines): 18 memory/summary/reflection/facts/profile/schedule retrieval methods |
| `core/prompt/gatherer_knowledge.py` | KnowledgeRetrievalMixin (~1,285 lines): 22 knowledge retrieval methods (notes, docs, git, graph, threads, insights, wiki, semantic chunks, dreams, codebase, visual memories, daemon self-notes, Google Calendar) |
| `core/prompt/formatter.py` | Section formatting + prompt assembly (~1,650 lines): `_assemble_prompt()`, `_build_feature_inventory()`, `_staleness_prefix`, `_is_multimodal_model`, `_load_upload_image` |
| `core/prompt/hygiene.py` | ContentHygiene (~345 lines): `_hygiene_and_caps()`, `_backfill_recent_conversations()` |
| `core/prompt/summarizer.py` | LLMSummarizer (~403 lines): LLM dynamic compression of oversized items |
| `core/prompt/proposal_filter.py` | Code proposal retrieval + filtering (~765 lines) |
| `core/prompt/token_manager.py` | Budget computation, priority trimming, middle-out compression |

---

## Build Flow

Entry point: `UnifiedPromptBuilder.build_prompt(user_input, ...)` returns
a context dict ready for formatting.

### Step 1 — Query Analysis

- `analyze_query(user_input)` detects small-talk (→ lightweight path)
- Intent classification triggers retrieval/weight overrides
- First message: detect codebase changes since last session

### Step 2 — Narrative Context (sync)

- `get_narrative_context()` — cheap file read for temporal grounding
- Hard cap at `NARRATIVE_STATE_MAX_TOKENS` (500)

### Step 3 — Apply Intent Overrides

- Set `scorer._intent_weight_overrides` for scoring adjustments
- Set `scorer._graph_memory` / `_entity_resolver` for graph-boosted scoring
- Cleared after gather completes
- Intent-type extraction reads `IntentResult.intent_type` (property alias
  for `.intent`, added 2026-07-03 — before the alias this silently
  resolved to None and wiki/web/visual intent suppression never fired)
- Intent also shapes the response: `get_intent_style_instructions()`
  injects a short per-intent style block into the system-prompt tail
  (after `PROMPT_CACHE_BREAKPOINT`; confidence ≥ 0.60, CONVERSATIONAL
  tone only — crisis suppresses; `INTENT_STYLE_INSTRUCTIONS_ENABLED`)

### Step 3.5 — Response Planning (parallel with Step 4)

If `RESPONSE_PLANNING_ENABLED` and `ResponsePlanner.should_plan()` passes
(skips small-talk, crisis/elevated tone, disabled config), the planner
runs in parallel with the retrieval tasks below via `asyncio.wait()` in
the orchestrator's `build_full_prompt()`. The planner makes a lightweight
LLM call (~200 tokens, 5s timeout) producing a `ResponsePlan` with
key_points, tone, avoid list, and strategy. The plan is injected into
the system prompt before `_assemble_prompt()`. If the planner times out
or fails, the prompt proceeds without a plan.

### Step 4 — Parallel Retrieval (up to 22 tasks, 30s timeout)

All tasks execute simultaneously via `asyncio.wait()` (not `asyncio.gather`).
Completed tasks survive a timeout — only the still-pending sections fall back
to `[]`. A single `warning` names the pending sections; per-task exceptions
default to `[]` without affecting other sections.

| Task | Method | Default Limit |
|------|--------|---------------|
| recent | `_get_recent_conversations()` | 15 |
| memories | `_get_semantic_memories()` | 15 |
| user_profile | `get_user_profile_context()` | 3000 tokens |
| summaries | `_get_summaries_separate()` | 5 recent + 5 semantic |
| dreams | `_get_dreams()` | 3 |
| semantic | `_get_semantic_chunks()` | 8 (dedicated 2-worker executor + non-blocking in-flight semaphore [2026-07-15] — a USB-stalled search that outlives SEM_TIMEOUT_S can't starve the shared default executor, and a saturated pool makes the turn SKIP wiki chunks instead of queuing; warmup touch at startup via `_run_model_warmup`) |
| reflections | `_get_reflections_separate()` | 5 recent + 5 semantic |
| wiki | `_get_wiki_content()` | 3 |
| personal_notes | `get_personal_notes()` | 5 |
| reference_docs | `get_reference_docs()` | 5 |
| user_uploads | `get_user_uploads()` | 5 (skipped via ~ms metadata existence probe when no uploads exist — was ~0.9s/turn; negative cached 60s, positive for session) |
| git_commits | `get_git_commits()` | varies |
| procedural_skills | `get_procedural_skills()` | 5 (over-fetched 3x, filtered by SkillActivationPolicy) |
| proposed_features | `get_proposed_features()` | 3 |
| graph_context | `get_graph_context()` | 12 sentences |
| unresolved_threads | `get_unresolved_threads()` | 3 |
| proactive_insights | `get_proactive_insights()` | 2 |
| visual_memories | `get_visual_memories()` | varies |
| web_search | `_get_web_search_results()` | 5 results |
| upcoming_schedule | `get_upcoming_schedule()` | 5 events (gated by TEMPORAL_RECALL/PROJECT_WORK intent) |
| google_calendar | `get_google_calendar_events()` | 10 events (gated by `GOOGLE_CALENDAR_ENABLED`) |

Per-task timing is tracked and logged for bottleneck detection. Since 2026-07-14
the gather also emits live progress lines to the streaming UI via the
`utils/turn_progress.py` bus: a "Retrieving context from N sources" line at
launch, a per-task completion line for any task taking ≥0.2s (label + duration +
hit count, from the `_TASK_LABELS` map in `builder.py`), and a "Context
retrieved — gating, dedup, token budget…" line when the gather finishes.
`gui/handlers.py` relays these events to the chat UI (0.3s poll, heartbeat
after ~6s of silence); emission is a no-op outside a live turn.

**Note-image gate** [2026-07-14]: Obsidian note images are attached to the
multimodal call only when `builder._should_include_note_images()` passes —
config (`OBSIDIAN_INCLUDE_IMAGES`) + multimodal model + the QUERY signaling
visual intent (`_query_wants_visual`, the same gate visual memories use). An
attached image reads to the model as "the user just showed me this": a course
note surfacing on an unrelated turn shipped its embedded screenshot and the
model narrated it as a topic pivot. The `[N image(s) attached]` text indicator
still renders either way; user uploads are unaffected. Regression:
`tests/unit/test_note_image_gate.py`.

### Step 5 — Post-Fetch Processing

- Merge separated summaries/reflections (recent + semantic)
- Apply reflections session filter if enabled
- Sort reflections by timestamp (newest first)
- Top-up reflections via on-demand LLM synthesis if count < target
- **Skill activation** — `SkillActivationPolicy.filter()` applied to
  procedural_skills results: intent suppression (EMOTIONAL_SUPPORT,
  CASUAL_SOCIAL return []), min score threshold (0.25), STM topic bonus
  (+0.10), cooldown filter (48h via `SkillCooldownStore`), cap to 3.
  Skills are over-fetched at `SKILL_ACTIVATION_FETCH_MULTIPLIER` (3x)
  during Step 4 to give the policy a wider candidate pool.

### Step 6 — Hygiene & Caps

`ContentHygiene._hygiene_and_caps(context, stm_summary)` (in `hygiene.py`):

- **Per-section dedup** (by content field or query+response)
- **Cross-section dedup** (prevent duplicates across recent/memories/notes)
- **Semantic chunk stitching** (combine chunks by title, up to 4000 chars)
- **Backfill recent** if dedup drops count below target
- **Memory top-up** (Step 6.1) — fetch extra if memories < `PROMPT_MAX_MEMS`
- **Summaries/reflections top-up** (Step 6.2) — pull additional without gating

### Step 6.9 — LLM-Compress Oversized Items

- Items >= 3x token limit → LLM-generated summary
- Items 1-3x limit → defer to middle-out
- Fallback: middle-out if LLM compression fails
- **Content-hash result cache** [2026-07-15]: identical content compresses
  once per session (`builder._llm_compress_cache`, bounded FIFO 200). A
  recorded TIMEOUT is also cached (skip straight to middle-out) — the same
  `git_commits` blob was re-paying the flat 3s timeout on every turn of a
  session. Transient API errors are NOT cached (retried next turn).

### Step 7 — Token Budget Management

See "Token Budget" section below.

### Step 7.1 — Post-Budget Floors

Restore critical sections that budget trimming may have removed:

- Recent conversations: minimum 5 items (`PROMPT_MIN_RECENT_FLOOR`)
- Summaries: restore to `PROMPT_MAX_SUMMARIES` if dropped
- Reflections: restore to `PROMPT_MAX_REFLECTIONS` if dropped

### Step 8 — Final Assembly

`PromptFormatter._assemble_prompt()` (in `formatter.py`) arranges sections in attention-optimized order.

---

## Token Budget

### Budget Computation

```
Priority: env override > model-aware > default

Model-aware = 0.12 of context window
  Local models:  max(8000, min(computed, 12000))
  API models:    max(8000, min(computed, 16000))
  Default:       10000
  Floor:         8000
```

### Priority Ordering (10 levels)

| Priority | Sections |
|----------|----------|
| 10 | stm_summary (metadata, never trimmed) |
| 9 | user_profile |
| 8 | narrative_state (capped at 500 tokens), web_search_results |
| 7 | recent_conversations, graph_context, unresolved_threads |
| 6 | semantic_chunks, personal_notes, user_uploads |
| 5 | reference_docs, memories |
| 4 | procedural_skills, facts |
| 3 | summaries, proposed_features, git_commits, proactive_insights |
| 2 | reflections, dreams, codebase_changes |
| 1 | wiki |

### Compression Algorithm

**Phase 1 — Per-item middle-out.** Items exceeding `max_item_tokens` get
60% head + 40% tail preserved, middle snipped with marker.

**Phase 2 — Iterative section trimming** (max 3 passes). Starting from
lowest priority, drop 25% of items per pass. String sections blanked
entirely.

**Phase 3 — Emergency whole-prompt compression.** If still over budget
after assembly, split at `[CURRENT USER QUERY]` marker and apply
middle-out to everything before it. Query is always protected.

---

## Section Ordering in Final Prompt

`PromptFormatter._assemble_prompt()` (in `formatter.py`) arranges sections to exploit LLM recency bias —
high-attention items (user profile, time, query) placed last:

1. `[RECENT CONVERSATION]` — session continuity (1-15 items)
2. `[RELEVANT MEMORIES]` — semantic hits (1-15 items)
3. `[RECENT SUMMARIES]` — compressed recent history (1-5)
4. `[SEMANTIC SUMMARIES]` — query-relevant compressed history (1-5)
5. `[RECENT REFLECTIONS]` — meta insights, recent (1-5)
6. `[SEMANTIC REFLECTIONS]` — meta insights, query-relevant (1-5)
7. `[BACKGROUND KNOWLEDGE]` — wiki snippets (1-3)
8. `[WEB SEARCH RESULTS]` — real-time web with `[WEB_N]` source IDs + citation instruction (if triggered, up to 8)
9. `[RELEVANT INFORMATION]` — semantic chunks (1-8)
10. `[DREAMS]` — synthesis insights, if enabled (up to 3; all generators currently disabled in config.yaml pending grading validation)
11. `[USER'S PERSONAL NOTES]` — Obsidian vault (1-5)
12. `[USER UPLOADED ITEMS]` — uploaded documents (1-5)
13. `[VISUAL MEMORIES]` — CLIP-matched image metadata (if available; **two-gated**: (1) a visual-intent gate `_query_wants_visual()` requires an explicit visual word (`show`/`see`/`pic`/`photo`/`image`/`look`) OR a recall intent (`factual_recall`/`temporal_recall`) before any photo surfaces, and (2) word-boundary entity matching (`re.search(r"\bname\b", ...)`, not raw substring) so a name can't match inside a path/identifier — e.g. `luke` no longer matches `/home/lukeh/...`; entity-gated via `extract_graph_entities()` with junk filtering and intent-proximity disambiguation; model-gated: images dropped for non-vision models). `builder.py` passes the classified `intent_type` into `get_visual_memories()`.
14. `[DAEMON DOCUMENTATION]` — reference docs (1-5)
15. `[PROJECT COMMIT HISTORY]` — git commits (varies)
16. `[ADAPTIVE WORKFLOWS]` — procedural skills (1-5)
17. `[PROPOSED FEATURES]` — code proposals (1-3)
18. `[KNOWLEDGE GRAPH]` — entity relationships, natural language (up to 12 sentences)
19. `[UNRESOLVED THREADS]` — open commitments/deadlines (1-3)
20. `[UPCOMING SCHEDULE]` — schedule facts for next N days (intent-gated: TEMPORAL_RECALL/PROJECT_WORK or keyword-gated, up to 5 events)
21. `[GOOGLE CALENDAR]` — real-time Google Calendar events via OAuth2 (up to 10 events; gated by `GOOGLE_CALENDAR_ENABLED`). Numbered list: `date time_start – time_end: Summary (location)`. All-day events show `[all day]` instead of times.
22. `[DAEMON SELF-NOTES]` — Daemon's own working notes from prior sessions (non-ground-truth, trust-weighted below user content). Includes disclaimer.
23. `[PROACTIVE INSIGHTS]` — cross-domain connections (1-2)
24. `[USER PROFILE]` — categorized facts with inline anti-confabulation instruction and source excerpts when available (high-attention zone)
25. `[ACTIVE FEATURES]` — feature inventory (always)
26. `[CODEBASE CHANGES SINCE LAST SESSION]` — git diff (first message only)
27. `[TIME CONTEXT]` — current time + time deltas (high-attention zone)
28. `[TEMPORAL GROUNDING]` — narrative context (if available)
29. `[SHORT-TERM CONTEXT SUMMARY]` — STM analysis (if available). Includes `Reference Type:` line (`new_event` / `recall` / `clarification` / `correction` / `unclear`) with explicit WARNING directive when type ≠ `new_event`. Also renders `Resolved State:` from `temporal_facts`, `Open Threads:` from `open_threads` (ongoing commitments/topics), and `Constraints:` from `constraints` (implicit/explicit response limits). STM internally injects last 2 daily notes from the Obsidian vault for cross-day recall disambiguation.
30. `[CURRENT USER QUERY]` — always last, protected from compression

Items with `staleness_ratio >= 0.6` get `[HISTORICAL — PARTIALLY OUTDATED]` prefix.

---

## Intent-Driven Overrides

The intent classifier can override both retrieval counts and scoring
weights. Applied in `build_prompt()` before parallel tasks launch.

### Retrieval Count Overrides

```python
retrieval_overrides = {
    "max_recent": 20,           # Override PROMPT_MAX_RECENT
    "max_mems": 25,             # Override PROMPT_MAX_MEMS
    "max_summaries": 15,        # Total; split evenly if not specified
    "max_reflections": 15,
    "max_dreams": 5,
    "max_semantic": 12,
    "max_wiki": 5,
    "max_skills": 8,
    "max_proposals": 5,
    "max_git_commits": 15,
    "max_surfaced_threads": 5,
    # Phase 8 additions (eval-driven gating):
    "max_reference_docs": 15,   # Override PROMPT_MAX_REFERENCE_DOCS
    "max_user_uploads": 5,      # Override PROMPT_MAX_USER_UPLOADS
    "max_proactive": 2,         # Override PROMPT_MAX_PROACTIVE_INSIGHTS
    "max_personal_notes": 5,    # Override PROMPT_MAX_PERSONAL_NOTES
    "max_narrative": 1,         # 0 = skip narrative_state entirely
}
```

Setting any key to 0 skips that section's retrieval task entirely (no async
call launched). Config flag `PROMPT_SECTION_GATING_ENABLED` (default True)
controls whether Phase 8 gating keys are applied; when disabled, only the
original pre-Phase-8 retrieval overrides take effect.

### Weight Overrides

Set on scorer before gather, cleared after:

```python
scorer._intent_weight_overrides = weight_overrides
# ... parallel retrieval with adjusted scores ...
scorer._intent_weight_overrides = None
```

---

## Graph Query Expansion

Before semantic retrieval, `_expand_query_with_graph()` appends knowledge
graph neighbor names to the search query:

1. Extract entities from query via entity resolver
2. Rank by lateral connectivity (skip hubs like "user")
3. Append top N display names

Example: `"Tesla"` → `"Tesla Rivian Ford EV market"`

Config: `GRAPH_QUERY_EXPANSION_ENABLED`, `GRAPH_QUERY_EXPANSION_MAX_TERMS`

### Hub-aware BFS (2026-06)

`rank_expansion_candidates()` walks the graph with a hub guard: a hub node —
one in `skip_ids` (e.g. "user") or with degree >= `GRAPH_EXPANSION_HUB_DEGREE`
(default 30) — may be *reached* but is never expanded *through*. Only the seed
entities fan out freely. This fixes a bug where an incidental query token that
linked to the "user" star hub (~797 of 889 graph edges) dumped the whole
personal neighbourhood (pets, project terms) into an unrelated query.
`extract_graph_entities()` also drops common-noun/participle stopwords
(video/videos/done/homework/stuff) that had been wrongly resolving as entities.

### Read-time TTL (2026-06)

Expansion honours the same read-side staleness the profile, `facts` collection
and graph-context (`get_context_sentences`) already apply: stale transient edges
(mood/activity/illness past their per-relation horizon, via
`GraphMemory._edge_is_stale_transient` → `relation_classifier`) are dropped from
both traversal and scoring. Nothing is deleted — a fresh mention refreshes
`last_seen` and the edge returns.

---

## Web Search Integration

`_get_web_search_results()` decides whether to search, then executes:

1. Check `WEB_SEARCH_ENABLED` and crisis suppression (HIGH/MEDIUM → skip)
2. LLM-first trigger classifies query (fallback: sync heuristics)
3. If triggered, execute via `WebSearchManager`:
   - LLM-optimized search terms bypass auto-decompose
   - Depth levels: QUICK (1-2 sources), STANDARD (3-5), DEEP (5+)
4. Results cached by query hash
5. Citations tracked in `memory_id_map`

Failures (no API key, rate limit, timeout) return `None` silently.

**Query localization (2026-07-02):** location-dependent queries (weather, local
news, "near me") carry the user's location. The trigger prompt and
`decompose_query()` prompt both receive a `User location:` line
(`utils/location_resolver.py`: config override → IP geolocation, cached +
background-refreshed → profile `lives_in` fact filtered to place-shaped values).
A deterministic backstop in `WebSearchManager.search()` (`_localize_query`)
substitutes literal "my area"/"near me" and appends the location to placeless
weather-type queries — applied before the cache check so cache keys are
location-aware. Without this, "temperature in my area" went to Tavily verbatim
and returned arbitrary big-market results (the DC-weather-in-Illinois bug).

**Localization scope guard (2026-07-08):** location is for physical-surroundings
queries ONLY. Both LLM prompts (trigger + decompose) forbid attaching the
location to institution/account/login queries, and parsed trigger search terms +
decompose sub-queries pass through
`location_resolver.strip_unjustified_location()` — a deterministic backstop that
removes the injected place whenever the original query gave no local cue
(weather/current-conditions shape, "near me"/local phrasing, or the user naming
the place themselves). Downstream, the `[WEB SEARCH RESULTS]` block (formatter)
and the agentic final-response instructions carry an institution-identity guard:
never present a geo-matched institution from results as the user's own.
Regression guards for the wrong-college incident — a school-login query
localized to "Springfield IL" retrieved Springfield Community College and its
IT-desk number was presented as the user's school's.

In agentic mode, the `fetch_url` tool provides direct URL content retrieval
via `WebSearchManager._tavily_extract()`. URLs detected in user messages are
auto-fetched in Round 1. If a web_search query contains a URL, it is
auto-rerouted to fetch_url. Fetched content is registered in `web_source_map`
for `[WEB_N]` citation tracking.

---

## Google Calendar Integration

`get_google_calendar_events()` (in `gatherer_knowledge.py`) fetches upcoming
events from the user's Google Calendar via OAuth2:

1. Check `GOOGLE_CALENDAR_ENABLED` config flag
2. Call `core.actions.google_calendar.fetch_upcoming_events(max_events)`
3. Returns list of event dicts: `{summary, start, end, all_day, location}`

The task runs in parallel with all other retrieval tasks in Step 4.
Rendering in `formatter.py` formats events as a numbered list:
`1) 2026-05-27 9:00 AM – 10:00 AM: Meeting (Room 3)`. All-day events
display as `2026-05-28 [all day]: Holiday`. All failures return `[]`
silently — calendar is best-effort context.

Config: `GOOGLE_CALENDAR_ENABLED`, `GOOGLE_CALENDAR_MAX_EVENTS`

---

## Upcoming Schedule (Extracted)

Schedule facts extracted from past conversations (shift patterns, exams,
appointments) are rendered in the `[UPCOMING SCHEDULE]` section. Retrieval
is keyword-gated: activated when the query contains schedule trigger words
(`schedule`, `shift`, `free`, `busy`, `exam`, `when do i`, etc.) combined
with temporal signals (`today`, `tomorrow`, `this week`, `next`, etc.),
or when the intent classifier returns TEMPORAL_RECALL or PROJECT_WORK.

The `get_upcoming_schedule()` method (in `gatherer_memory.py`) queries
schedule-tagged facts from ChromaDB. Events are rendered with human-readable
times, confidence qualifiers for heuristic resolutions, and scope markers
(`(one-time)` for non-recurring events).

---

## Small-Talk Fast Path

If `query_analysis.is_small_talk = True`, returns minimal context:
3 recent conversations, no memories/summaries/reflections/web search.

**This path was dead wiring until 2026-07-15** — the builder read the flag
but nothing ever set it, so a 7-word acknowledgment ("Hmm not working yet")
pulled a full ~23K-token prompt. It's now set by
`utils/query_checker.is_casual_acknowledgment()` (conservative: ack opener +
≤ `light_prompt.max_words` (8); any question/command/request/meta shape or
heavy topic disqualifies) and routed through `builder._should_use_light_path`:
config-gated (`light_prompt.enabled` / `LIGHT_PROMPT_ENABLED`), and elevated
tone (both `"CrisisLevel.HIGH"` and `"crisis_support"`-style encodings)
always gets the full context. The ContextPipeline computes a *separate*
`is_small_talk` (CASUAL_SOCIAL intent, conf ≥ 0.70) used for heavy-topic
skip and telemetry — the two signals coexist deliberately.
Tests: `tests/unit/test_light_prompt_path.py`.

---

## Key Configuration

```python
# Token budget
# Token budget (active values from config/app_config.py)
PROMPT_TOKEN_BUDGET_DEFAULT = 10000   # lowered from 15000 on 2026-07-15 (preregistered experiment)
PROMPT_TOKEN_BUDGET_LOCAL = 12000
PROMPT_TOKEN_BUDGET_FLOOR = 8000
PROMPT_TOKEN_BUDGET_CEILING = 16000
# Note: builder.py import-failure fallbacks use 40000/60000

# Retrieval limits
PROMPT_MAX_RECENT = 15
PROMPT_MAX_MEMS = 15
PROMPT_MAX_SUMMARIES = 10
PROMPT_MAX_REFLECTIONS = 10
PROMPT_MAX_DREAMS = 3
PROMPT_MAX_SEMANTIC = 10
PROMPT_MAX_WIKI = 3
PROMPT_MIN_RECENT_FLOOR = 5

# Compression
NARRATIVE_STATE_MAX_TOKENS = 500
MEMORY_ITEM_MAX_TOKENS = 512
SEMANTIC_ITEM_MAX_TOKENS = 800
LLM_COMPRESSION_RATIO_THRESHOLD = 3.0

# Skill activation (post-retrieval filter)
SKILL_ACTIVATION_ENABLED = True
SKILL_ACTIVATION_MAX_SKILLS = 3
SKILL_ACTIVATION_MIN_SCORE = 0.25
SKILL_ACTIVATION_COOLDOWN_HOURS = 48.0
SKILL_ACTIVATION_FETCH_MULTIPLIER = 3
SKILL_ACTIVATION_STM_BONUS = 0.10

# Google Calendar
GOOGLE_CALENDAR_ENABLED = False       # Requires OAuth2 setup
GOOGLE_CALENDAR_MAX_EVENTS = 10
```

---

## Typical Performance

- **Parallel gather**: 3-8s (30s timeout, partial context preserved on timeout)
- **Hygiene & dedup**: 0.5-2s
- **Token budget**: 1-3s
- **Final assembly**: 0.5-1s
- **Total**: 5-15s (excluding web search latency)

Bottlenecks: web search (2-5s), semantic search over 40+ memories (1-2s),
personal notes with images (1-2s), LLM compression (1-3s).
