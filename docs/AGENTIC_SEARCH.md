# Agentic Search System

Operational guide for Daemon's ReAct-style agentic search loop — tool
execution, context budgeting, protocol handling, query relaxation,
and provenance tracking.

For config constants see `QUICK_REFERENCE.md`. For prompt assembly
details see `PROMPT_BUILDING_PIPELINE.md`.

---

## What Agentic Search Does

When a user query needs external information, Daemon can enter a
multi-round ReAct (Reasoning + Acting) loop where the LLM iteratively
decides which tools to call — web search, memory search, Wolfram Alpha,
Python sandbox, file access, memory expansion, git stats, or full-document
retrieval — until it has enough
context to answer. The loop is budget-enforced and streams progress
events to the UI in real time.

---

## File Map

| File | Purpose |
|------|---------|
| `core/agentic/controller.py` | Main loop: session management, tool dispatch, budget, final generation |
| `core/agentic/types.py` | Data models: SearchDecision, ProgressEvent, SearchRound, tool schemas |
| `core/agentic/protocols.py` | Protocol detection, native tool parsing, XML marker parsing |
| `core/git_stats_manager.py` | Git stats tool: intent parsing, safe subprocess, output formatting |
| `core/orchestrator.py` | Trigger logic and lazy initialization of controller |

---

## When It Triggers

Agentic search activates when ALL conditions are met:

1. `use_agentic_search=True` passed to `process_user_query()`
2. Config: `agentic_search.enabled = true`
3. At least one trigger fires in `gui/handlers.py`:
   - Keyword heuristic: computation, memory, or knowledge keywords detected
   - Entity match: query mentions known knowledge graph entity
   - LLM trigger: `should_search=True`, `needs_memory_search=True`, or `needs_knowledge_search=True`

The controller is lazy-initialized on first use via the orchestrator's
`agentic_controller` property.

### Uncertainty Fallback (Post-Generation Trigger)

In addition to the pre-generation gate above, the agentic loop can also
be triggered **after** standard generation. If `UNCERTAINTY_FALLBACK_ENABLED`
is true and the response indicates uncertainty ("I don't have information",
"I can't recall"), `UncertaintyDetector` (`core/uncertainty_detector.py`)
fires and retries via agentic search with a memory-search hint. Detection
layers: keyword regex (~18 patterns) + semantic embedding similarity
against 8 anchor sentences. Long responses (>max_length after hedge-stripping)
skip detection. Config: `UNCERTAINTY_FALLBACK_ENABLED`,
`UNCERTAINTY_SEMANTIC_THRESHOLD` (default 0.70),
`UNCERTAINTY_MAX_LENGTH` (default 400).

---

## ReAct Loop Lifecycle

Entry point: `run_agentic_search()` — async generator yielding
`ProgressEvent` objects and response string chunks.

### Round 1 — Automatic Initial Search

Unless `skip_initial_search=True`, uses `initial_search_terms` from the
LLM trigger for the first web search. Results are compressed and
accumulated. Low-quality detection may suggest query relaxation.

### Rounds 2-N — Model-Driven Iteration

Loop continues while `session.can_continue`:
- `not model_signaled_done`
- `current_round <= max_rounds` (default 5)
- `state not in (DONE, ERROR)`

Each round:

1. **THINKING** — Build iteration prompt with `[TIME CONTEXT]` (current date/time) +
   accumulated context + inventory of already-gathered RAG context + relaxation/diversity hints
2. **DECIDE** — Call `_get_model_decision()` (native tools or XML markers)
3. **EXECUTE** — Dispatch to the appropriate tool handler

### Final Generation

After the loop exits:
- Assemble final prompt: `[TIME CONTEXT]` + RAG context + accumulated search results + query
- Budget-enforce: trim low-priority sections if over `context_budget * 5`
- Compute `final_prompt_hash` (SHA-256[:16]) for provenance
- Stream response chunks to caller

---

## Available Tools

### web_search

Search the web for current information.

```
Parameters: query (required), reason (optional)
Execution: WebSearchManager.search() with STANDARD depth
Fallback: None — empty results trigger relaxation hints
```

### wolfram_alpha

Compute mathematical expressions, solve equations.

```
Parameters: query (required), reason (optional)
Execution: WolframManager.query()
Fallback: Falls back to web search if computation fails
```

### execute_python

Run Python code in a secure sandbox with numpy, pandas, matplotlib,
scipy, sympy, scikit-learn pre-installed.

```
Parameters: code (required), purpose (optional)
Execution: Persistent SandboxSession (variables survive across turns)
Cleanup: Session closed in finally block
```

### search_memory

Search Daemon's own memory and knowledge base.

```
Parameters: query (required), collection (required), reason (optional)
Valid collections: reference_docs, facts, conversations, summaries,
                   reflections, obsidian_notes, wiki_knowledge,
                   procedural, procedural_skills
Diversity: Per-collection search counts tracked; hints injected after 2+ searches
wiki_knowledge: Always prefers FAISS semantic search (40M Wikipedia vectors,
                ~2 GB IVFPQ index) over ChromaDB which has sparse legacy data.
                Falls back to ChromaDB only if FAISS index is unavailable.
```

### expand_memory

Expand a memory hit to see surrounding temporal context.

```
Parameters: memory_id (required), collection (optional), window (1-5, default 3)
Gated by: EXPAND_MEMORY_ENABLED, EXPAND_MAX_PER_SESSION
Summaries: retrieves original source conversations
Others: shows N neighbors on each side by timestamp
```

### file_read / file_grep / file_list

File system access (restricted to approved directories).

```
file_read:  filepath (required), start_line/end_line (optional)
file_grep:  pattern (required), folder/file_glob/case_sensitive (optional)
file_list:  dirpath (required), recursive (optional)
```

### git_stats

Query the local git repository for activity statistics.

```
Parameters: query (required), reason (optional)
Intent parsing: Keyword-based — no LLM call needed
Time windows: "today", "this week", "last N days", "this month", etc.
Safety: Read-only git subcommands only (log, shortlog, diff, status,
        branch, rev-list, rev-parse, show, describe, tag, stash)
Output: Formatted summary + raw git output, capped at 50 lines
Config: GIT_STATS_ENABLED, GIT_STATS_TIMEOUT, GIT_STATS_MAX_OUTPUT_LINES
```

### get_full_document

Retrieve the complete text of an uploaded document by title.

```
Parameters: title (required), reason (optional)
Fuzzy matching: Exact match first, then case-insensitive word overlap
Execution: ReferenceDocsManager.get_full_document(title) — fetches all
           chunks, sorts by chunk_index, reassembles into single text
Truncation: Hard cap at 60k chars (budget enforcement handles the rest)
On miss: Returns list of available document titles for self-correction
Use case: User asks to "pull up" or "check" an uploaded PDF/DOCX/syllabus
          and search_memory only returned fragments
```

### done_searching

Signal that enough information has been gathered.

```
Parameters: reason (optional)
Effect: Sets model_signaled_done=True, exits loop
```

---

## Protocol Handling

### Detection

`detect_protocol(model_name)` classifies by model family:

- **Native tools**: OpenAI (gpt-4/4o/5), Anthropic (claude-*), DeepSeek
- **XML markers**: All others (local models, unknown models)

### Native Tools

Uses OpenAI-style function calling. LLM response includes
`tool_calls[0].function.name` and `.arguments` (JSON). Parsed by
`NativeToolsHandler` → `SearchDecision`.

### XML Markers

For models without native tool support. Markers embedded in text:

```
<search>query here</search>
<wolfram>2+2</wolfram>
<python purpose="calculate">code here</python>
<memory collection="facts">query here</memory>
<expand_memory id="doc-123" window="3" collection="summaries"/>
<file_read path="/path/to/file" start="1" end="50"/>
<file_grep pattern="regex" folder="src/" glob="*.py"/>
<file_list path="/path/to/dir" recursive="true"/>
<git_stats>commits this week</git_stats>
<done/>
```

Parsed by `XMLMarkerHandler` using regex. Checked in order:
python → wolfram → memory → expand → file_read → file_grep →
file_list → git_stats → web_search → done → implicit answer.

### System Prompt Augmentation

- **XML models**: Full `AGENTIC_SYSTEM_PROMPT_INJECTION` with tool usage
  guide, query reformulation strategies, and tool selection guidelines
- **Native models**: Minimal augmentation — tool list, memory guidance,
  done signal instruction

---

## Context Budget Enforcement

### Level 1 — Accumulated Context

`_append_accumulated(session, new_context)`:

- Limit: `context_budget_tokens` (default 8000)
- When over budget: split into round blocks, drop oldest until under limit
- Effect: keeps only the most recent rounds

### Level 2 — Final Prompt

`_build_final_prompt()`:

- Limit: `context_budget_tokens * 5` (~40K for default 8K)
- Trim order: dreams → reflections → docs → semantic summaries →
  recent summaries → personal notes
- Always preserved: recent conversations, agentic search results,
  user profile

### Token Estimation

Uses `TokenManager.get_token_count()` if available, otherwise
`len(text) // 4` (~4 chars per token).

---

## Search Quality & Query Relaxation

### Low-Quality Detection

`_is_low_quality_result(search_result, query)` checks:

1. No results → low quality
2. Only 1 result → low quality
3. Top result lacks >= 30% of query terms → low quality

### Relaxation

Up to 2 relaxation attempts before forcing synthesis:

- **Attempt 1**: Suggest removing version numbers, year specifics,
  exact phrases, or simplifying to core subject
- **Attempt 2**: Final attempt with same hint style
- **Attempt 3+**: `MAX_RELAXATION_HINT` — answer with what you have,
  acknowledge gaps, no more searches

### Memory Diversity

Per-collection search counts tracked in `session.memory_search_counts`.
After 2+ searches on the same collection, hints suggest trying a
different collection for broader coverage.

---

## Context Inventory

`_compute_context_inventory()` analyzes the RAG-gathered `initial_context`
dict and generates a summary of what's already available:

```
Context already gathered by retrieval pipeline:
- [USER PROFILE]: N categorized facts
- [RECENT SUMMARIES]: N session summaries
- [RELEVANT MEMORIES]: N conversation memories
...
Do NOT re-search for information already covered above.
```

Injected into iteration prompts to prevent redundant searches.

---

## Progress Events

Real-time UI updates via `ProgressEvent(event_type, message, round_number, metadata)`:

| Event | When |
|-------|------|
| `thinking` | Skipping initial search |
| `searching` | Starting web/memory search |
| `found_results` | Search completed |
| `computing` | Starting Wolfram computation |
| `computed` | Computation done |
| `executing_code` | Starting sandbox |
| `code_executed` / `code_error` | Sandbox result |
| `searching_memory` | Starting memory search |
| `expanding_memory` / `memory_expanded` | Memory expansion |
| `reading_file` / `file_read` | File read |
| `searching_files` / `files_searched` | File grep |
| `listing_files` / `files_listed` | File list |
| `querying_git` / `git_queried` | Git stats |
| `synthesizing` | Starting final generation |
| `done` | Session complete |
| `error` | Error occurred |

---

## Provenance Tracking

`session.get_provenance_summary()` returns:

```python
{
    "total_rounds": int,
    "protocol": "native_tools" | "xml_markers",
    "agentic_rounds": [
        {"round": 1, "action": "web_search", "query": "...", "duration_ms": ...},
        {"round": 2, "action": "memory_search", "query": "...", ...},
    ],
    "model_signaled_done": bool,
    "done_reason": str | None,
    "context_inventory": str,
    "memory_search_counts": {"summaries": 2, ...},
    "expand_count": int,
    "final_prompt_hash": str,       # SHA-256[:16]
    "total_duration_ms": float,
}
```

Round actions classified by prefix: `[Memory:` → memory_search,
`[Python:` → sandbox, `[File Read]` → file_read, `[Git:` → git_stats, etc.

---

## Termination Conditions

The loop exits when ANY condition is met:

1. **Model signals done** — calls `done_searching` tool or `<done/>`
2. **Model wants to answer** — no tool markers found (implicit)
3. **Max rounds exceeded** — `current_round > max_rounds` (default 5)
4. **Session error** — exception caught, attempts fallback generation
   with accumulated context

---

## Error Handling

- **Per-tool failures**: Logged, loop continues with available context
- **Session-level errors**: State set to ERROR, fallback generation
  attempted if any accumulated context exists
- **Sandbox cleanup**: Session closed in `finally` block regardless
  of success/failure

---

## Key Configuration

```python
# Agentic search (YAML: agentic_search:)
agentic_search.enabled = true       # Master switch
agentic_search.max_rounds = 5       # Default loop limit
agentic_search.context_budget_tokens = 8000
agentic_search.compression_model = "gpt-4o-mini"

# Memory tools
AGENTIC_MEMORY_SEARCH_LIMIT         # Results per memory search
EXPAND_MEMORY_ENABLED               # Feature gate
EXPAND_MAX_PER_SESSION              # Max expansions per session
EXPAND_MAX_WINDOW                   # Max neighbors to retrieve
EXPAND_ANCHOR_CHAR_LIMIT_LONG       # Long-form anchor limit (3000, for obsidian/reference_docs)
EXPAND_CONTEXT_CHAR_LIMIT_LONG      # Long-form context limit (2000)

# Git stats tool
GIT_STATS_ENABLED                   # Feature gate (default True)
GIT_STATS_TIMEOUT                   # Subprocess timeout in seconds
GIT_STATS_MAX_OUTPUT_LINES          # Cap raw output (default 50)

# Uncertainty fallback (YAML: uncertainty_fallback:)
UNCERTAINTY_FALLBACK_ENABLED        # Post-generation retry gate (default True)
UNCERTAINTY_SEMANTIC_THRESHOLD      # Cosine sim threshold for semantic layer (default 0.70)
UNCERTAINTY_MAX_LENGTH              # Max response length to check (default 400)
```
