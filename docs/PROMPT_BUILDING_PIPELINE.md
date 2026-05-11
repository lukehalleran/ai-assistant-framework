# Prompt Building Pipeline

Operational guide for the 4-module prompt assembly system — parallel
retrieval, token budgets, section ordering, intent-driven overrides,
and middle-out compression.

For formal notation see `FORMAL_MODEL.md`. For config constants see
`QUICK_REFERENCE.md`. For memory retrieval details see `MEMORY_SYSTEM.md`.

---

## What the Pipeline Does

Every user query triggers a full prompt build: 19 parallel async
retrievals across memory, knowledge graph, web search, files, and
profile data. Results are filtered, deduplicated, scored, compressed
to fit a token budget, and assembled into a final prompt string with
carefully ordered sections that exploit LLM attention patterns.

The pipeline lives in `core/prompt/` and is orchestrated by
`UnifiedPromptBuilder.build_prompt()`.

---

## File Map

| File | Purpose |
|------|---------|
| `core/prompt/builder.py` | Thin orchestrator (~1,583 lines): parallel task dispatch, intent overrides, budget, eval hooks. Delegates assembly to formatter and hygiene to ContentHygiene |
| `core/prompt/context_gatherer.py` | Mixin compositor (~379 lines): init, properties, utilities. Composes WebSearchMixin + MemoryRetrievalMixin + KnowledgeRetrievalMixin |
| `core/prompt/gatherer_web.py` | WebSearchMixin (~216 lines): `_get_web_search_results()`, `should_trigger_web_search()` |
| `core/prompt/gatherer_memory.py` | MemoryRetrievalMixin (~834 lines): 17 memory/summary/reflection/facts/profile retrieval methods |
| `core/prompt/gatherer_knowledge.py` | KnowledgeRetrievalMixin (~963 lines): 16 knowledge retrieval methods (notes, docs, git, graph, threads, insights, wiki, semantic chunks, dreams, codebase) |
| `core/prompt/formatter.py` | Section formatting + prompt assembly (~1,597 lines): `_assemble_prompt()` (737 lines), `_build_feature_inventory()`, `_staleness_prefix`, `_is_multimodal_model`, `_load_upload_image` |
| `core/prompt/hygiene.py` | ContentHygiene (~345 lines): `_hygiene_and_caps()`, `_backfill_recent_conversations()` |
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

### Step 3.5 — Response Planning (parallel with Step 4)

If `RESPONSE_PLANNING_ENABLED` and `ResponsePlanner.should_plan()` passes
(skips small-talk, crisis/elevated tone, disabled config), the planner
runs in parallel with the retrieval tasks below via `asyncio.wait()` in
the orchestrator's `build_full_prompt()`. The planner makes a lightweight
LLM call (~200 tokens, 5s timeout) producing a `ResponsePlan` with
key_points, tone, avoid list, and strategy. The plan is injected into
the system prompt before `_assemble_prompt()`. If the planner times out
or fails, the prompt proceeds without a plan.

### Step 4 — Parallel Retrieval (19 tasks, 30s timeout)

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
| semantic | `_get_semantic_chunks()` | 8 |
| reflections | `_get_reflections_separate()` | 5 recent + 5 semantic |
| wiki | `_get_wiki_content()` | 3 |
| personal_notes | `get_personal_notes()` | 5 |
| reference_docs | `get_reference_docs()` | 5 |
| user_uploads | `get_user_uploads()` | 5 |
| git_commits | `get_git_commits()` | varies |
| procedural_skills | `get_procedural_skills()` | 5 |
| proposed_features | `get_proposed_features()` | 3 |
| graph_context | `get_graph_context()` | 12 sentences |
| unresolved_threads | `get_unresolved_threads()` | 3 |
| proactive_insights | `get_proactive_insights()` | 2 |
| visual_memories | `get_visual_memories()` | varies |
| web_search | `_get_web_search_results()` | 5 results |

Per-task timing is tracked and logged for bottleneck detection.

### Step 5 — Post-Fetch Processing

- Merge separated summaries/reflections (recent + semantic)
- Apply reflections session filter if enabled
- Sort reflections by timestamp (newest first)
- Top-up reflections via on-demand LLM synthesis if count < target

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

Model-aware = 25% of context window
  Local models:  max(8000, min(computed, 12000))
  API models:    max(8000, min(computed, 16000))
  Default:       15000
  Floor:         8000
```

### Priority Ordering (10 levels)

| Priority | Sections |
|----------|----------|
| 10 | stm_summary (metadata, never trimmed) |
| 9 | user_profile |
| 8 | narrative_state (capped at 500 tokens) |
| 7 | recent_conversations, graph_context, unresolved_threads |
| 6 | semantic_chunks, personal_notes, user_uploads |
| 5 | reference_docs, memories, web_search_results |
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
13. `[VISUAL MEMORIES]` — CLIP-matched image metadata (if available; entity-gated via `extract_graph_entities()` with junk filtering and intent-proximity disambiguation; short messages (<=5 words) without intent override suppress retrieval; model-gated: images dropped for non-vision models)
14. `[DAEMON DOCUMENTATION]` — reference docs (1-5)
15. `[PROJECT COMMIT HISTORY]` — git commits (varies)
16. `[ADAPTIVE WORKFLOWS]` — procedural skills (1-5)
17. `[PROPOSED FEATURES]` — code proposals (1-3)
18. `[KNOWLEDGE GRAPH]` — entity relationships, natural language (up to 12 sentences)
19. `[UNRESOLVED THREADS]` — open commitments/deadlines (1-3)
20. `[PROACTIVE INSIGHTS]` — cross-domain connections (1-2)
21. `[USER PROFILE]` — categorized facts with inline anti-confabulation instruction and source excerpts when available (high-attention zone)
22. `[ACTIVE FEATURES]` — feature inventory (always)
23. `[CODEBASE CHANGES SINCE LAST SESSION]` — git diff (first message only)
24. `[TIME CONTEXT]` — current time + time deltas (high-attention zone)
25. `[TEMPORAL GROUNDING]` — narrative context (if available)
26. `[SHORT-TERM CONTEXT SUMMARY]` — STM analysis (if available). Includes `Reference Type:` line (`new_event` / `recall` / `clarification` / `correction` / `unclear`) with explicit WARNING directive when type ≠ `new_event`. Also renders `Resolved State:` from `temporal_facts`, `Open Threads:` from `open_threads` (ongoing commitments/topics), and `Constraints:` from `constraints` (implicit/explicit response limits). STM internally injects last 2 daily notes from the Obsidian vault for cross-day recall disambiguation.
27. `[CURRENT USER QUERY]` — always last, protected from compression

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

---

## Small-Talk Fast Path

If `query_analysis.is_small_talk = True`, returns minimal context:
3 recent conversations, no memories/summaries/reflections/web search.

---

## Key Configuration

```python
# Token budget
PROMPT_TOKEN_BUDGET_DEFAULT = 40000
PROMPT_TOKEN_BUDGET_LOCAL = 12000
PROMPT_TOKEN_BUDGET_FLOOR = 8000
PROMPT_TOKEN_BUDGET_CEILING = 60000

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
