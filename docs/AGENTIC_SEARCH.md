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
decides which tools to call — web search, URL fetch, memory search, Wolfram Alpha,
Python sandbox, file access, memory expansion, git stats, GitHub API, StackExchange,
arXiv, PubMed, Hacker News, full-document retrieval, image recall, document
generation, or daemon self-notes — until it has enough
context to answer. The loop is budget-enforced and streams progress
events to the UI in real time.

---

## File Map

| File | Purpose |
|------|---------|
| `core/agentic/controller.py` | Main loop: session management, prompt building, model interaction, quality heuristics, nudge retry, no-reasoning decision phase, tool hints |
| `core/agentic/tools.py` | ToolExecutor: 15 dispatch methods + 13 execute helpers (sandbox and memory_expand execute inline in their dispatch methods) + `get_tool_health()` status summary |
| `core/agentic/formatters.py` | AgenticFormatter: 20 pure formatting methods (context, results, prompts) |
| `core/agentic/types.py` | Data models: SearchDecision, ProgressEvent, SearchRound, tool schemas |
| `core/agentic/protocols.py` | Protocol detection, native tool parsing, XML marker parsing, nested XML support, github_available gating |
| `core/git_stats_manager.py` | Git stats tool: intent parsing, safe subprocess, output formatting |
| `core/github_manager.py` | GitHub API tool: read-only `gh` CLI access (issues, PRs, actions, releases) |
| `core/orchestrator.py` | Trigger logic and lazy initialization of controller |

---

## When It Triggers

Agentic search activates when ALL conditions are met:

1. `use_agentic_search=True` passed to `process_user_query()`
2. Config: `agentic_search.enabled = true`
3. At least one trigger fires in `gui/handlers.py`:
   - Keyword heuristic: computation, memory, knowledge, or web search keywords detected ("web search", "search for", "search online", "look up")
   - Tool name keywords: explicit tool mentions ("github", "git stats", "wolfram", "sandbox", "pull request", "open issues", etc.) trigger agentic mode via `needs_tools`
   - URL detection: `http://` or `https://` in query triggers agentic mode with auto-fetch
   - Entity match: query mentions known knowledge graph entity
   - LLM trigger: `should_search=True`, `needs_memory_search=True`, or `needs_knowledge_search=True`. Memory search takes priority: if LLM returns both `should_search` and `needs_memory_search`, the query routes to memory (not web).
   - Explicit search keywords bypass intent veto; skip initial search when Tier 1 fires without LLM-generated terms
   - `needs_tools` added to `skip_initial_search` condition (skips Round 1 web search for tool-name queries)

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

If `initial_urls` are provided (URLs detected in the user message),
Round 1 auto-fetches each URL via `fetch_url` before any web search.
If a web search query contains a URL, it is auto-rerouted to `fetch_url`.

### Rounds 2-N — Model-Driven Iteration

Loop continues while `session.can_continue AND session.current_round <= self.max_rounds`:

`session.can_continue` is True when all of:
- `not model_signaled_done`
- `state not in (DONE, ERROR)`

The explicit `current_round <= max_rounds` guard (default 5) is checked
separately in the `while` condition alongside `can_continue`.

Each round:

1. **THINKING** — Build iteration prompt with `[TIME CONTEXT]` (current date/time) +
   accumulated context + inventory of already-gathered RAG context + relaxation/diversity hints +
   tool hints (injected via `_detect_tool_hints()` when query mentions tool names like "github", "git stats", etc.)
2. **DECIDE** — Call `_get_model_decision()` (native tools or XML markers).
   For XML protocol, uses `_generate_decision_no_reasoning()` to bypass
   native reasoning (e.g. DeepSeek chain-of-thought) that would burn the
   token budget, leaving no room for XML tool markers.
3. **EXECUTE** — Dispatch to the appropriate tool handler via `_dispatch_single()`,
   which routes github, stackexchange, arxiv, pubmed, and hackernews tools
   in addition to the core tools.  Tool dispatch runs under the
   `python_fs_guard.agent_mode()` context manager (see
   `utils/python_fs_guard.py`), which intercepts destructive Python
   filesystem calls (`os.remove`, `shutil.rmtree`, `os.rename`,
   `shutil.copyfile`, `shutil.copy`, `shutil.copy2`, etc.) and
   blocks them when they target protected repo paths.  This guard applies to
   in-process tool execution; child Python interpreters also inherit it
   when `scripts/bin/` is on PYTHONPATH (via `usercustomize.py`).

**Nudge Retry**: If round 1 produces no tool calls but the response text
mentions tools ("github", "let me check", "commits", etc.), the controller
retries once with an explicit nudge instructing the model to emit XML
markers instead of narrating what it would do.

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

### fetch_url

Fetch and read web page content by URL.

```
Parameters: url (required), reason (optional)
Execution: WebSearchManager._tavily_extract([url])
Citations: Result registered in web_source_map for [WEB_N] citation tracking
Availability: Gated on web_search_manager.is_available() (requires Tavily API key)
Auto-trigger: URLs detected in user messages are auto-fetched in Round 1
URL reroute: If a web_search query contains a URL, it is auto-rerouted here
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
                   procedural, procedural_skills, daemon_self_notes
Diversity: Per-collection search counts tracked; hints injected after 2+ searches
wiki_knowledge: ChromaDB is queried first (like all collections). Then FAISS
                semantic search (41M Wikipedia vectors, ~2 GB IVFPQ index) is
                additionally attempted. If FAISS returns results, they are
                preferred over the ChromaDB results. If FAISS is unavailable
                or returns nothing, the ChromaDB results are used as fallback.
                When FAISS is unavailable (checked via is_faiss_available() in
                knowledge/semantic_search.py — file-existence check, no full load),
                a prominent warning is prepended to the result telling the LLM
                that the 41M-vector index could not be loaded and instructing it
                NOT to claim Wikipedia/FAISS search is working.
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

### github

Query GitHub repository data via `gh` CLI (read-only).

```
Parameters: query (required), reason (optional)
Execution: GitHubManager.execute_query() — parses natural-language query into
           gh CLI subcommand (issues, prs, actions, releases, workflows, labels,
           milestones, contributors, code_search)
Safety: Read-only — only allowlisted gh subcommands (issue list, pr list,
        run list, release list, workflow list, label list, api, search code)
Output: Formatted summary + raw gh output, capped at max_output_lines
Config: GITHUB_API_ENABLED, GITHUB_API_TIMEOUT, GITHUB_API_MAX_OUTPUT_LINES, GITHUB_API_REPO
```

### recall_image

Search visual memory for CLIP-matched images by text query.

```
Parameters: query (required), reason (optional)
Dispatch: _dispatch_recall_image → _execute_recall_image → VisualRetriever
Execution: Queries visual_memories ChromaDB collection using CLIP embeddings
           matched against the text query. Returns image metadata and descriptions.
```

### generate_document

Generate a structured markdown report or summary from web search and memory sources.

```
Parameters: topic (required), doc_type ("report" or "summary", default "report"), reason (optional)
Execution: DocumentGenerator.generate() — web search + ChromaDB retrieval, LLM synthesis
Output: Markdown file in documents/reports/ or documents/summaries/ with YAML frontmatter
Citations: Inline [WEB_N] references + Sources section
Index: documents/index.json tracks all generated docs
Direct trigger: "write a report about X" bypasses agentic loop for direct invocation
Config: DOCUMENT_* constants; YAML section document_generation:
```

### create_daemon_note

Save a structured note for Daemon's future sessions (architecture decisions, risks, next steps).

```
Parameters: topic (required), content (required), reason (optional)
Execution: DaemonNotesManager.save_note() — writes markdown + stores in ChromaDB
Output: Markdown file in daemon_notes/{slug}-{date}.md with YAML frontmatter
Collection: daemon_self_notes (ground_truth: False in metadata)
Retrieval: get_daemon_self_notes() in context gatherer, max 2 per prompt
Direct trigger: "save a note for yourself about X" bypasses agentic loop
Config: DAEMON_NOTES_* constants; YAML section daemon_notes:
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
`NativeToolsHandler` -> `SearchDecision`. GitHub tool definitions are
conditionally included based on `github_available` parameter. Empty
arguments for git_stats, github, and search_memory default to the
original query rather than failing.

### XML Markers

For models without native tool support. Markers embedded in text:

```
<search>query here</search>
<fetch_url url="https://example.com">reason</fetch_url>
<wolfram>2+2</wolfram>
<python purpose="calculate">code here</python>
<memory collection="facts">query here</memory>
<expand_memory id="doc-123" window="3" collection="summaries"/>
<file_read path="/path/to/file" start="1" end="50"/>
<file_grep pattern="regex" folder="src/" glob="*.py"/>
<file_list path="/path/to/dir" recursive="true"/>
<git_stats>commits this week</git_stats>
<github>open issues labeled bug</github>
<recall_image>query</recall_image>
<done/>
```

**XML Alias Patterns**: `XMLMarkerHandler` also accepts these aliases:
- `<web_search>query</web_search>` and `<web_search query="...">` as aliases for `<search>`
- `<search_memory query="...">` as an alias for `<memory>`
- `<search_memory><query>X</query></search_memory>` nested-tag pattern (DeepSeek-style) via `MEMORY_NESTED_PATTERN`

**Nested XML Support**: `_strip_xml_tags()` removes inner XML tags from
extracted content, and `_extract_nested_tag()` extracts specific child
elements from nested XML structures (e.g. `<query>` and `<collection>`
inside `<search_memory>`).

Parsed by `XMLMarkerHandler` using regex. `<done/>` is checked first
and returns immediately if present. Remaining markers are collected
in order: python -> wolfram -> memory -> expand_memory ->
get_full_document -> file_read -> file_grep -> git_stats -> github ->
file_list -> fetch_url -> recall_image -> search -> implicit answer
(if no markers found).

### System Prompt Augmentation

- **XML models**: Full `AGENTIC_SYSTEM_PROMPT_INJECTION` with tool usage
  guide, query reformulation strategies, and tool selection guidelines
- **Native models**: Minimal augmentation — tool list, memory guidance,
  done signal instruction

### Tool Health Injection

`ToolExecutor.get_tool_health()` probes each tool backend and returns a
multi-line status summary (AVAILABLE / UNAVAILABLE / DISABLED per tool).
Checked backends: web_search, wiki_knowledge (FAISS), memory_search
(ChromaDB), wolfram, file_access, git_stats, github, expand_memory, recall_image.

The status block is injected at three points under the header
`[TOOL STATUS — DO NOT LIE ABOUT THESE]`:

1. **System prompt** (`run_agentic_search`) — appended after protocol
   augmentation, with an instruction that the LLM must report tool status
   accurately and never claim a tool works if its status says UNAVAILABLE.
2. **Iteration prompt** (`_build_iteration_prompt`) — appended before the
   "What would you like to do?" decision instruction.
3. **Final prompt** (`_build_final_prompt`) — appended after the query,
   with an instruction to only report what `[TOOL STATUS]` says when asked.

The FAISS availability check (`is_faiss_available()` in
`knowledge/semantic_search.py`) tests file existence of the index and
metadata parquet without triggering a full load. If the singleton is
already loaded it returns immediately.

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
- Trim order: dreams -> reflections -> docs -> semantic summaries ->
  recent summaries -> personal notes
- Always preserved: recent conversations, agentic search results,
  user profile

### Token Estimation

Uses `TokenManager.get_token_count()` if available, otherwise
`len(text) // 4` (~4 chars per token).

---

## Search Quality & Query Relaxation

### Low-Quality Detection

`_is_low_quality_result(search_result, query)` checks:

1. No results -> low quality
2. Only 1 result -> low quality
3. Top result lacks >= 30% of query terms -> low quality

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
- [VISUAL MEMORIES]: N images already retrieved
- [PROJECT COMMIT HISTORY]: N commits
- [KNOWLEDGE GRAPH]: N relationship sentences
- [UNRESOLVED THREADS]: N open threads
- [PROACTIVE INSIGHTS]: N insights
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
| `retrieving_document` / `document_retrieved` | Full document retrieval |
| `querying_git` / `git_stats_done` | Git stats |
| `querying_github` / `github_done` | GitHub API |
| `fetching_url` / `url_fetched` | URL content fetch |
| `recalling_image` / `recall_image_done` | Visual memory recall |
| `generating_document` / `document_generated` | Document generation |
| `creating_note` / `note_created` | Daemon self-note creation |
| `synthesizing` | Starting final generation |
| `done` | Session complete (suppressed in GUI after response starts) |
| `error` | Error occurred |

**Note:** `handlers.py` skips any `ProgressEvent` that arrives after response
chunks have started accumulating, preventing late events (e.g. `done`) from
overwriting the streamed response in the chatbot.

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

Round actions classified by prefix: `[Memory:` -> memory_search,
`[Python:` -> sandbox, `[File Read]` -> file_read, `[Git:` -> git_stats, etc.

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

# GitHub API tool (YAML: github_api:)
GITHUB_API_ENABLED                  # Feature gate (default True)
GITHUB_API_TIMEOUT                  # Subprocess timeout in seconds
GITHUB_API_MAX_OUTPUT_LINES         # Cap raw output
GITHUB_API_REPO                     # Target repo (owner/name)

# Uncertainty fallback (YAML: uncertainty_fallback:)
UNCERTAINTY_FALLBACK_ENABLED        # Post-generation retry gate (default True)
UNCERTAINTY_SEMANTIC_THRESHOLD      # Cosine sim threshold for semantic layer (default 0.70)
UNCERTAINTY_MAX_LENGTH              # Max response length to check (default 400)

# Response review gate (YAML: response_planning:)
RESPONSE_REVIEW_ENABLED             # Post-answer review against plan (default True)
RESPONSE_REVIEW_CONFIDENCE_THRESHOLD  # Min confidence to trigger agentic retry (default 0.80)
RESPONSE_REVIEW_TIMEOUT             # Seconds before review skipped (default 5.0)
```
