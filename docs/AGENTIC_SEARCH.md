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
| `core/agentic/gate.py` | 4-tier agentic gate: `evaluate_agentic_gate()` → `AgenticDecision` (keyword → entity → doc/note → LLM fallback) |
| `core/agentic/controller.py` | Main loop: session management, prompt building, model interaction, quality heuristics, nudge retry, no-reasoning decision phase, tool hints |
| `core/agentic/tools.py` | ToolExecutor: 17 dispatch methods + 15 execute helpers (sandbox and memory_expand execute inline in their dispatch methods) + `get_tool_health()` status summary + `_resolve_email_recipient()` |
| `core/agentic/formatters.py` | AgenticFormatter: 20 pure formatting methods (context, results, prompts) |
| `core/agentic/types.py` | Data models: SearchDecision, ProgressEvent, SearchRound, tool schemas, LOOKUP_CONTACT_TOOL_DEFINITION |
| `core/agentic/protocols.py` | Protocol detection, native tool parsing, XML marker parsing, nested XML support, github_available gating, contact lookup aliases |
| `core/git_stats_manager.py` | Git stats tool: intent parsing, safe subprocess, output formatting |
| `core/github_manager.py` | GitHub API tool: read-only `gh` CLI access (issues, PRs, actions, releases) |
| `core/actions/` | Internet action types, executors (telegram/discord/email/calendar), audit log, pending store, Google Calendar create, Google OAuth, Google Contacts (`google_contacts.py`), Gmail search (`gmail_search.py`) |
| `core/orchestrator.py` | Trigger logic and lazy initialization of controller |

---

## When It Triggers

Agentic search activates when ALL conditions are met:

1. `use_agentic_search=True` passed to `process_user_query()`
2. Config: `agentic_search.enabled = true`
3. `evaluate_agentic_gate()` in `core/agentic/gate.py` returns `AgenticDecision.should_trigger=True`:
   - Tier 1: Keyword heuristic — computation, memory, knowledge, web search, and tool name keywords; also **file/saved-document retrieval** intent (so `file_read` / `file_list` / `get_full_document` are offered) and email-by-name patterns
   - Tier 2: Entity match — query mentions known knowledge graph entity + recall signal
   - Tier 3: Document *generation* or self-note intent detection
   - Tier 4: LLM fallback — piggybacks on web search trigger call (`needs_memory_search`, `needs_knowledge_search`, `needs_document_generation`)
   - Casual skip filter, continuation override, and intent-based veto all handled inside gate
   - `AgenticDecision.skip_initial_search` computed by gate (True for computation, memory, knowledge, tools modes)

The controller is lazy-initialized on first use via the orchestrator's
`agentic_controller` property.

### File / Saved-Document Retrieval Routing [2026-06-08]

A request to read or print a saved file/document must reach the agentic loop so
the `file_read` / `file_list` / `get_full_document` tools are actually offered —
otherwise it falls through to enhanced (tool-less) mode and the model
confabulates "I don't have file access". The gate detects this in three layers
(`core/agentic/gate.py`):

- `FILE_ACCESS_KEYWORDS` — literal fast-path for noun phrases ("the full
  document", "file you saved", "document you wrote").
- `FILE_ACCESS_PATTERNS` — robust regexes tolerant of verb inflection and
  intervening words ("pulling up and printing the document"), capability
  assertions ("you have the tool", "use the file_read tool"), and bare tool
  names.
- **Continuation** — terse follow-ups route to tools when the *previous* turn
  was file/document themed: a pronoun retrieval ("pull it up") gated on prior
  file/doc context (`FILE_DOC_CONTEXT_WORDS`), or a bare affirmation ("yes
  please") right after the model OFFERED to pull a file (`FILE_OFFER_MARKERS`).
  This is what lets the enhanced-mode honesty offer ("Want me to pull that up?")
  get carried out on the next turn.

File access counts as an **explicit request**, so the intent-based veto cannot
suppress it. Distinct from Tier 3 document *generation*. When the gate still
misses, the enhanced (tool-less) path carries an `[ACTION HONESTY]` note
(`gui/handlers.py`) so the degraded turn is an honest "I can't this turn" + offer
— never a confabulated reason (e.g. "I'm on mobile").

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

**Trigger guard** [2026-06-08]: `detect_document_intent()` requires the doc-noun
to be the (near) OBJECT of the save-verb (`DOCUMENT_TRIGGER_PATTERN` bounded gap,
~4 words). Incidental save-verb + doc-noun co-occurrence across a long
multi-sentence message ("Create a new dataframe … Print the model summary") no
longer fires — that false-fire previously routed to the direct
`_run_doc_generation` bypass (`gui/handlers.py`), which hijacks the whole turn
with a "Document saved" receipt and emits no conversational reply.

**LLM-failure safety** [2026-06-08]: `generate_once()` returns API-error
sentinel strings ("[API Error] … 402", "[CREDITS EXHAUSTED] …") instead of
raising. `DocumentGenerator` detects these on the topic-refine, outline, and
draft calls and aborts with a `RuntimeError` so a corrupt frontmatter-only file
is never written or indexed.

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

### propose_action

Propose an internet write action requiring user confirmation before execution.

```
Parameters:
  Common:    action_type (required), reason (required)
  Messaging: recipient, message, subject (for email)
  Calendar:  summary (event title), start_time (ISO 8601), end_time (ISO 8601),
             description, time_zone (IANA, default America/Chicago),
             calendar_id (default "primary"), location
Note: "message" is required for messaging types but NOT for calendar events,
      which use "summary" instead. Required fields are ["action_type", "reason"].
Email auto-resolution: For send_email, if recipient is a name (no '@'),
  _resolve_email_recipient() resolves it via google_contacts.resolve_contact()
  before creating the ActionProposal. Single match auto-resolves; multiple
  matches return a descriptive error listing candidates.

Action types: send_telegram, send_discord, send_email,
              github_create_issue, github_comment_pr, calendar_create_event
Execution: Creates an ActionProposal stored in PendingActionsStore.
           GUI displays approve/reject buttons. On approval, the action is
           dispatched to the type-specific executor. On rejection, the proposal
           is discarded and an audit log entry is written.
Audit: All proposals and outcomes logged to logs/actions_audit.jsonl
```

### lookup_contact

Look up a contact's email address from Google Contacts (read-only, no confirmation needed).

```
Parameters: name (required), reason (optional)
Execution: resolve_contact() from google_contacts.py — searches saved contacts,
           then other contacts, then Gmail headers as fallback.
Aliases: search_contacts, find_contact, search_gmail, search_email,
         gmail_search, search_inbox (all parsed by NativeToolsHandler + XMLMarkerHandler)
Config: GOOGLE_CONTACTS_ENABLED, GOOGLE_OTHER_CONTACTS_ENABLED,
        GOOGLE_GMAIL_SEARCH_ENABLED (all default true)
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
original query rather than failing. `propose_action` is parsed as a
native tool call — the handler detects `calendar_create_event` action
type and accepts `summary` as an alternative to `message`, forwarding
all 7 calendar-specific params (`summary`, `description`, `start_time`,
`end_time`, `time_zone`, `calendar_id`, `location`) into `action_params`.
`NativeToolsHandler` also has a `_parse_text_tool_calls()` fallback
that detects text-embedded action proposals when the LLM narrates a
proposal instead of emitting a proper tool call (Pattern 1:
`[propose_action: <type>]` followed by JSON). The text-based parser
also handles calendar events with the same calendar-param forwarding.
The `send_` prefix normalization in the text parser only applies to
messaging types (telegram, discord, email), not to calendar or github
action types. `lookup_contact` is parsed with aliases: `search_contacts`,
`find_contact`, `search_gmail`, `search_email`, `gmail_search`,
`search_inbox` — all map to `wants_lookup_contact=True` in SearchDecision.
`<invoke name="...">` XML patterns are also parsed as a fallback for
both contact lookup and propose_action.

**Text-leak recovery (Pattern 4)** [2026-06-08]: when the API returns NO
structured `tool_calls` — common with OpenRouter-proxied native models like
DeepSeek, which emit the call as plain content — `NativeToolsHandler` delegates
the text to `XMLMarkerHandler` and recovers any actionable tool markers
(`<file_read>`, `<search>`, `<memory>`, `<fetch_url>`, …). Without this the raw
XML leaks into the answer and the tool never executes.

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
<action type="send_email" recipient="..." reason="...">message</action>
<propose_action type="send_email" recipient="..." subject="..." reason="...">body</propose_action>
<lookup_contact name="Meaghan">reason</lookup_contact>
<done/>
```

**XML Alias Patterns**: `XMLMarkerHandler` also accepts these aliases:
- `<web_search>query</web_search>` and `<web_search query="...">` as aliases for `<search>`
- `<search_memory query="...">` as an alias for `<memory>`
- `<search_memory><query>X</query></search_memory>` nested-tag pattern (DeepSeek-style) via `MEMORY_NESTED_PATTERN`
- `<invoke name="lookup_contact">` / `<invoke name="search_contacts">` etc. as Anthropic-style fallback for contact lookup

**Nested XML Support**: `_strip_xml_tags()` removes inner XML tags from
extracted content, and `_extract_nested_tag()` extracts specific child
elements from nested XML structures (e.g. `<query>` and `<collection>`
inside `<search_memory>`).

**Nested file/doc/url tool forms** [2026-06-08]: models (notably
OpenRouter-proxied DeepSeek) frequently emit the params as nested child tags
instead of attributes — e.g. `<file_read><path>foo.md</path></file_read>` rather
than `<file_read path="foo.md">`. Dedicated nested-form fallbacks
(`FILE_READ_NESTED_PATTERN`, `FILE_GREP_NESTED_PATTERN`,
`FILE_LIST_NESTED_PATTERN`, `FETCH_URL_NESTED_PATTERN`,
`GET_FULL_DOCUMENT_NESTED_PATTERN`) parse these, plus a bare-content path form
(`<file_read>foo/bar.md</file_read>`). The nested patterns match only the
attribute-less opening tag (`\s*>`), so they never double-parse the attribute
form. Without them the marker failed to parse and the raw XML leaked into the
answer (tool never executed).

Parsed by `XMLMarkerHandler` using regex. `<done/>` is checked first
and returns immediately if present. Remaining markers are collected
in order: python -> wolfram -> memory -> expand_memory ->
get_full_document -> file_read -> file_grep -> git_stats -> github ->
file_list -> fetch_url -> recall_image -> action -> propose_action ->
lookup_contact -> search -> implicit answer (if no markers found).
`propose_action` is parsed from both `<action type="...">` and
`<propose_action type="...">` XML markers. `lookup_contact` is parsed
from `<lookup_contact name="...">` markers.

### System Prompt Augmentation

- **XML models**: Full `AGENTIC_SYSTEM_PROMPT_INJECTION` with tool usage
  guide, query reformulation strategies, and tool selection guidelines
- **Native models**: Minimal augmentation — tool list, memory guidance,
  done signal instruction

### Tool Health Injection

`ToolExecutor.get_tool_health()` probes each tool backend and returns a
multi-line status summary (AVAILABLE / UNAVAILABLE / DISABLED per tool).
Checked backends: web_search, wiki_knowledge (FAISS), memory_search
(ChromaDB), wolfram, file_access, git_stats, github, expand_memory,
recall_image, propose_action. The propose_action AVAILABLE line
dynamically lists `calendar_create_event` when `GOOGLE_CALENDAR_ENABLED`
is true (e.g. "send_email, send_telegram, send_discord, calendar_create_event").

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
| `proposing_action` | "Proposing action: {summary}" |
| `action_proposed` | "Action proposed: {summary}" (metadata: action_id, action_type, summary) |
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

# Internet actions (YAML: internet_actions:)
INTERNET_ACTIONS_ENABLED            # Master switch
INTERNET_ACTIONS_TELEGRAM_BOT_TOKEN # Telegram Bot API token
INTERNET_ACTIONS_TELEGRAM_CHAT_ID   # Target Telegram chat ID
INTERNET_ACTIONS_DISCORD_WEBHOOK_URL # Discord webhook URL
INTERNET_ACTIONS_SMTP_HOST          # SMTP server host
INTERNET_ACTIONS_SMTP_PORT          # SMTP server port
INTERNET_ACTIONS_SMTP_USER          # SMTP username
INTERNET_ACTIONS_SMTP_PASSWORD      # SMTP password
INTERNET_ACTIONS_SMTP_FROM          # Sender email address
INTERNET_ACTIONS_GITHUB_WRITE_ENABLED # GitHub write actions gate
INTERNET_ACTIONS_PLAYWRIGHT_ENABLED # Playwright browser actions gate
INTERNET_ACTIONS_TTL                # Pending action time-to-live
INTERNET_ACTIONS_MAX_PENDING        # Max pending actions before rejection
INTERNET_ACTIONS_AUDIT_LOG          # Audit log path (default logs/actions_audit.jsonl)

# Google Calendar (YAML: google_calendar:)
GOOGLE_CALENDAR_ENABLED             # Feature gate for calendar_create_event

# Google Contacts (YAML: internet_actions:)
GOOGLE_CONTACTS_ENABLED             # Search saved contacts (default True)
GOOGLE_OTHER_CONTACTS_ENABLED       # Search other/auto contacts (default True)
GOOGLE_GMAIL_SEARCH_ENABLED         # Gmail header search fallback (default True)
```

---

## Internet Actions (Human-in-the-Loop)

Daemon can propose internet write actions (sending messages, creating
issues, etc.) but never executes them autonomously. All actions follow
a **propose → confirm → execute** flow with mandatory human approval.

### Flow

1. **Propose** — The `propose_action` agentic tool creates an
   `ActionProposal` (action type, recipient, message, reason) and
   stores it in `PendingActionsStore`. For `send_email` actions, if the
   recipient is a name (no `@`), `_resolve_email_recipient()` resolves
   it via `google_contacts.resolve_contact()` before creating the
   proposal. The proposal is returned to the GUI as a pending action.
2. **Confirm** — The GUI displays approve/reject buttons when a
   pending action exists. The user reviews the proposed action and
   decides.
3. **Execute** — On approval, `ActionExecutorRegistry` routes the
   proposal to the type-specific executor (e.g. `telegram.py`,
   `discord.py`, `email.py`, `google_calendar_create.py`). On rejection,
   the proposal is discarded and an audit log entry is written.

### Audit

All actions — proposals, approvals, rejections, execution results,
and errors — are logged to `logs/actions_audit.jsonl` (one JSON object
per line).

### Implemented Executors

| Action Type | Backend | Notes |
|-------------|---------|-------|
| `send_telegram` | Telegram Bot API via httpx | Requires bot token + chat ID |
| `send_discord` | Discord webhook via httpx | Requires webhook URL |
| `send_email` | Gmail API (preferred) + SMTP fallback | Auto-resolves recipient names via Google Contacts + Gmail header search. Defense-in-depth `_resolve_recipient()` in `email.py`. |
| `calendar_create_event` | Google Calendar API via httpx | Requires Google OAuth + `calendar.events` scope. API error responses include the response body for debugging. Config: `GOOGLE_CALENDAR_ENABLED`. Implementation in `core/actions/google_calendar_create.py`. |
| `github_create_issue` | `gh` CLI (`gh issue create`) | Requires `gh` installed + `gh auth login`. Config: `INTERNET_ACTIONS_GITHUB_WRITE_ENABLED`. Two-entry write-allowlist + explicit arg lists + timeout. Implementation in `core/actions/github_write.py`. |
| `github_comment_pr` | `gh` CLI (`gh pr comment`) | Same gate/safety. PR number parsed from `pr_number`/recipient/subject. |

> All executors are now declared once in `core/actions/registry.py` (`ACTION_SPECS`),
> the single source of truth — see "Registry-Driven Write Actions + `github_write`" below.

---

## Registry-Driven Write Actions + `github_write` [2026-05-31]

<!-- registry-driven write actions + github_write [2026-05-31] -->

The agentic loop now exposes **21 tools** (was 20): `github_write` is a dedicated
GitHub write proposal (`github_create_issue`, `github_comment_pr`) alongside the
generic `propose_action`.

**`ACTION_SPECS` — single source of truth.** Every write action is declared once
in `core/actions/registry.py` as an `ActionSpec` (`executor_ref`, `required`/
`optional` params, `intent_patterns`, `enabled_flag`, tool-health line,
`field_hint`, deterministic `backfill`, `summary`). All consumers derive behavior
from the registry instead of hand-wiring per action:

- `core/actions/executors.py` — `ActionExecutorRegistry.execute()` resolves the
  executor via `spec.resolve_executor()` (lazy `"module:function"` import).
- `core/agentic/protocols.py` — native-tool `propose_action` parse accepts when
  all `spec.required` present OR `spec.backfill` can fill them; forwards only
  `spec.forward_params` (a model-supplied `repo` is dropped).
- `core/agentic/tools.py` — `get_tool_health()` lists `enabled_action_types()`;
  the github query line now points writers at `propose_action`.
- `core/agentic/controller.py` — `detect_action_intent(query)` forces a
  `propose_action` call on the first round (restricting offered tools to just
  `propose_action` so research-eager models can't wander), then
  `backfill_params()` fills blank content (e.g. a GitHub issue title/body) from
  the user's request.

**Shared `DISPATCH_TABLE`.** Both routers — `ToolExecutor.dispatch_single` and
the controller's `_dispatch_single_inner` — now iterate the single
`DISPATCH_TABLE` in `core/agentic/tools.py` (with `reroute_url_search()` applied
first), so they can no longer drift. The controller previously lacked branches
for `generate_document` / `create_daemon_note` / `lookup_contact` / `action`,
silently dropping those calls; the table fixes that class of bug.

**`github_write` executor** (`core/actions/github_write.py`): `create_github_issue`
and `comment_github_pr` shell out to the `gh` CLI behind a two-entry
`WRITE_ALLOWLIST` (`("issue","create")`, `("pr","comment")`), explicit arg lists
(never `shell=True`), a subprocess timeout, and a config gate
(`INTERNET_ACTIONS_GITHUB_WRITE_ENABLED`). It is a separate path from the
read-only `core/github_manager.py`. All failures degrade to a failed
`ActionResult`; execution is human-gated (GUI Approve).

**Parity guard:** `tests/unit/test_tool_wiring_parity.py` fails loudly if a
SearchDecision tool flag has no `DISPATCH_TABLE` row, if the two routers stop
using the table, or if any `ActionType` / advertised `propose_action` enum value
lacks an executor.
