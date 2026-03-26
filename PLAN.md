# `expand_memory` Tool — Implementation Plan (Adjusted)

## Problem
When the ReAct agent calls `search_memory`, it gets isolated snippets (500-char truncated turns). For conversations, this loses critical surrounding context — the user's setup, the assistant's follow-up, and the flow of the exchange. The agent has no way to "zoom in" on a promising hit.

## Reality Check vs. Original Plan

The original plan assumed:
1. **`session_id` exists on conversation metadata** — **it doesn't**. ChromaDB conversations have `timestamp`, `query`, `response`, `topic`, `truth_score`, `importance_score`, `access_count`, `last_accessed`. No session_id. Corpus (JSON) has `thread_id`/`thread_depth` but these are NOT propagated to ChromaDB metadata.
2. **`source_session_id` links summaries→conversations** — **it doesn't**. Summaries have no backreference to source conversation IDs. Facts have no `src_id` either.
3. **Document IDs are visible to the agent** — **they're not**. `_format_memory_results()` shows `[1] timestamp (score: 0.95) content` but never the ChromaDB doc ID. The agent can't reference a hit by ID.

These gaps shape the adjusted plan below.

---

## Architecture: 5 Phases

### Phase 0: Forward-Compatible Metadata — Start Writing `thread_id` to ChromaDB NOW

**Rationale**: Thread tracking already exists in corpus (JSON) but isn't propagated to ChromaDB conversation metadata. Starting now means future expansions can use thread boundaries for precise grouping. Timestamp-window remains the fallback for all existing data.

#### 0a. `memory/memory_storage.py` — Propagate thread metadata to ChromaDB
In `store_interaction()`, after building `raw_metadata` dict (~line 248), add thread fields from `thread_info`:
```python
if thread_info.get("thread_id"):
    raw_metadata["thread_id"] = thread_info["thread_id"]
if thread_info.get("depth") is not None:
    raw_metadata["thread_depth"] = int(thread_info["depth"])
```
This is a 4-line change. All new conversations get thread metadata going forward. Old data stays timestamp-only.

---

### Phase 1: Prerequisites — Expose Doc IDs + Add `get_by_id()` [MANDATORY]

Without this, the agent cannot call `expand_memory` at all. This is not optional.

#### 1a. `memory/storage/multi_collection_chroma_store.py` — Add `get_by_id()` method
```python
def get_by_id(self, collection_name: str, doc_id: str) -> Optional[Dict]:
    """Fetch a single document by ID. Returns {id, content, metadata} or None."""
    coll = self.collections.get(collection_name)
    if not coll:
        return None
    result = coll.get(ids=[doc_id], include=["documents", "metadatas"])
    docs = result.get("documents", []) or []
    metas = result.get("metadatas", []) or []
    if not docs:
        return None
    return {
        "id": doc_id,
        "content": docs[0] or "",
        "metadata": metas[0] if metas else {},
    }
```

#### 1b. `core/agentic/controller.py` — Add doc IDs to `_format_memory_results()`
Include the ChromaDB document ID in each search result header so the agent can reference it:
```
[1] (id: abc12345) 2025-10-08T14:30:00 (score: 0.95) [conversations]
User: What's my dog's name?
Assistant: Your dog is Flapjack...
```
Change: Add `doc_id = r.get("id", "")` and include `(id: {doc_id[:8]})` in the header. Also include `[{collection}]` so the agent knows which collection to pass to expand_memory.

---

### Phase 2: Core — `MemoryExpander` Class

**New file: `memory/memory_expander.py`**

Timestamp-based window expansion. Since conversations don't have `session_id` (yet — Phase 0 starts writing `thread_id` for future use), we use timestamp proximity as the grouping heuristic — neighboring turns in time are from the same conversation session.

```python
class MemoryExpander:
    """Expands a memory hit into surrounding context using timestamp proximity."""

    def __init__(self, chroma_store):
        self.chroma_store = chroma_store
        self._cache = {}  # (memory_id, window, collection) -> expansion result

    def expand(self, memory_id: str, window: int = 3,
               collection: str = None) -> dict:
        """
        Fetch turns surrounding a memory hit.

        Strategy:
        1. Fetch anchor doc by ID → get its timestamp
        2. list_all() on the same collection
        3. Sort by (timestamp, doc_id) for determinism
        4. Find anchor index, slice [max(0, idx-window) : min(len, idx+window+1)]
        5. Return structured result with anchor marked

        Returns: {
            "anchor_id": str,
            "collection": str,
            "expansion_method": "timestamp_window" | "thread_id",
            "turns": [{"id", "content", "timestamp", "is_anchor"}, ...],
            "total_in_collection": int,
            "error": str or None,
        }
        """
```

**Key design decisions:**
- **Timestamp proximity, not session_id** — No session_id exists on old data. Temporal window is the pragmatic approach. A 3-turn window around a conversation from 3pm will naturally include the surrounding exchange from the same session.
- **Deterministic sort: `(timestamp, doc_id)`** — Multiple documents (especially facts, shutdown outputs) can share the exact same timestamp. Secondary sort on doc_id (UUID string) ensures the same anchor always expands to the same neighborhood. No wobble.
- **Boundary guards** — `max(0, idx-window)` and `min(len(all_docs), idx+window+1)` for first/last-turn edge cases. If the anchor is turn 1 of 20 with window=3, return turns 1-4 (not crash).
- **`list_all()` is acceptable** — Conversations collection caps at `CORPUS_MAX_ENTRIES` (2000). Sorting 2000 dicts by timestamp is ~1ms. We're not doing this on every query, only when the agent explicitly requests expansion.
- **Cache within ReAct session** — Cache keyed on `(memory_id, window, collection)`. Same combo = cached response. Prevents the agent from burning rounds re-expanding.
- **Default window=3** (not 5) — 3 turns each direction = 7 total. At ~300 chars each = ~2,100 chars ~ 600 tokens. Reasonable budget.
- **Content truncation** — Anchor turn shown in full (up to 600 chars). Surrounding turns truncated to 300 chars.
- **`expansion_method` field in output** — Tells the agent (and debugging) whether the expansion used timestamp proximity or thread_id grouping. Helps downstream reasoning and bad-expansion debugging.

**Collection-specific behavior:**
| Collection | Expansion Strategy |
|---|---|
| `conversations` | Prefer `thread_id` if present on anchor (Phase 0 data). Fallback: timestamp window. |
| `summaries` | Timestamp window — shows neighboring session summaries for temporal context |
| `reflections` | Timestamp window — shows neighboring reflections |
| `facts` | Timestamp window — shows facts extracted around the same time |
| `obsidian_notes` | Timestamp window — shows notes from similar time period |
| Others | Return anchor only with message "expansion not supported for this collection" |

---

### Phase 3: Agent Integration — Tool Registration + Wiring

#### 3a. `core/agentic/types.py` — Tool definition
```python
EXPAND_MEMORY_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "expand_memory",
        "description": (
            "Fetch surrounding conversation turns around a specific memory hit. "
            "Use AFTER search_memory when a result looks relevant but needs more context — "
            "like seeing only one side of a conversation, or a summary that hints at "
            "something discussed nearby. Requires the memory ID from a previous search result. "
            "IMPORTANT: If the expanded context still does not answer your question, "
            "try a DIFFERENT search query rather than re-expanding the same memory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The document ID from a previous search_memory result (shown as 'id: ...' in results)."
                },
                "window": {
                    "type": "integer",
                    "description": "Number of turns to fetch in each direction. Default: 3, max: 5."
                },
                "collection": {
                    "type": "string",
                    "description": "Which collection the memory came from (shown as [collection] in results).",
                    "enum": [
                        "conversations", "summaries", "reflections",
                        "facts", "obsidian_notes"
                    ]
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why expansion is needed."
                }
            },
            "required": ["memory_id"]
        }
    }
}
```

Note: The tool description includes anti-loop guidance: "If the expanded context still does not answer your question, try a DIFFERENT search query rather than re-expanding the same memory."

#### 3b. `core/agentic/types.py` — SearchDecision fields
Add to the `SearchDecision` dataclass:
```python
# Memory expansion (context around a search hit)
wants_memory_expand: bool = False
expand_memory_id: Optional[str] = None
expand_window: int = 3
expand_collection: Optional[str] = None
expand_reason: Optional[str] = None
```

#### 3c. `core/agentic/protocols.py` — Parse expand_memory tool calls
Add `elif func_name == "expand_memory":` block in `NativeToolsHandler.parse_response()` (after the `search_memory` block):
```python
elif func_name == "expand_memory":
    memory_id = args.get("memory_id", "")
    if memory_id:
        return SearchDecision(
            wants_memory_expand=True,
            expand_memory_id=memory_id,
            expand_window=args.get("window", 3),
            expand_collection=args.get("collection"),
            expand_reason=args.get("reason"),
        )
```

Also update:
- `NativeToolsHandler.__init__()` — import + store `EXPAND_MEMORY_TOOL_DEFINITION`
- `NativeToolsHandler.get_tools()` — include expand tool when memory is available
- `NativeToolsHandler.augment_system_prompt()` — add to tool list

For XML protocol (`AGENTIC_SYSTEM_PROMPT_INJECTION`), add:
```
9. **Memory Expand**: <expand_memory id="doc_id" collection="conversations" window="3">reason</expand_memory>
   Use AFTER search_memory to see surrounding turns around a relevant hit.
   If the expanded context does not help, try a different search query instead of re-expanding.
```
Also add XML parsing in `XMLMarkerHandler`.

#### 3d. `core/agentic/controller.py` — Execution handler
Add `elif decision.wants_memory_expand and decision.expand_memory_id:` block in the main loop (after `wants_memory_search`):

```python
elif decision.wants_memory_expand and decision.expand_memory_id:
    yield ProgressEvent(
        event_type="expanding_memory",
        message=f"Expanding context around {decision.expand_memory_id[:8]}...",
        round_number=session.current_round,
        metadata={
            "memory_id": decision.expand_memory_id,
            "collection": decision.expand_collection,
            "window": decision.expand_window,
            "reason": decision.expand_reason,
        }
    )
    session.state = AgentState.SEARCHING
    start_time = time.time()
    expand_result = await self._execute_memory_expand(
        decision.expand_memory_id,
        decision.expand_window,
        decision.expand_collection,
    )
    # ... format, accumulate, yield progress (same pattern as memory search)
```

Add `_execute_memory_expand()` method:
```python
async def _execute_memory_expand(self, memory_id: str, window: int = 3,
                                  collection: str = None) -> str:
    if not self.memory_expander:
        return "[Memory expansion unavailable]"
    result = self.memory_expander.expand(memory_id, window, collection)
    if result.get("error"):
        return f"[Expansion error: {result['error']}]"
    return self._format_expanded_results(result)
```

Add `_format_expanded_results()`:
```python
def _format_expanded_results(self, result: dict) -> str:
    header = (
        f"[EXPANDED — {result['collection']} — "
        f"method: {result.get('expansion_method', 'timestamp')} — "
        f"{len(result['turns'])} turns shown]"
    )
    lines = [header]
    for turn in result["turns"]:
        marker = " <<<< TARGET" if turn["is_anchor"] else ""
        ts = turn.get("timestamp", "")[:19]
        content = turn["content"]
        max_len = EXPAND_ANCHOR_CHAR_LIMIT if turn["is_anchor"] else EXPAND_CONTEXT_CHAR_LIMIT
        if len(content) > max_len:
            content = content[:max_len] + "..."
        lines.append(f"[{ts}]{marker}\n{content}")
    return "\n---\n".join(lines)
```

**Controller.__init__()** — Create `MemoryExpander` instance when `chroma_store` is available.

---

### Phase 4: Gating + Cost Control

#### 4a. Config constants (`config/app_config.py`)
```python
# Memory Expansion (expand_memory tool)
EXPAND_CFG = config.get("memory_expansion", {})
EXPAND_MEMORY_ENABLED = bool(EXPAND_CFG.get("enabled", True))
EXPAND_MAX_PER_SESSION = int(EXPAND_CFG.get("max_per_session", 3))
EXPAND_MAX_WINDOW = int(EXPAND_CFG.get("max_window", 5))
EXPAND_DEFAULT_WINDOW = int(EXPAND_CFG.get("default_window", 3))
EXPAND_MAX_TOTAL_TOKENS = int(EXPAND_CFG.get("max_total_tokens", 2000))
EXPAND_ANCHOR_CHAR_LIMIT = int(EXPAND_CFG.get("anchor_char_limit", 600))
EXPAND_CONTEXT_CHAR_LIMIT = int(EXPAND_CFG.get("context_char_limit", 300))
```

#### 4b. `config/config.yaml` — YAML section
```yaml
memory_expansion:
  enabled: true
  max_per_session: 3
  max_window: 5
  default_window: 3
  max_total_tokens: 2000
  anchor_char_limit: 600
  context_char_limit: 300
```

#### 4c. Gating rules in controller
- **Max expansions per ReAct session**: Track `session.expand_count`. After `EXPAND_MAX_PER_SESSION` (3), return "expansion limit reached — try a different search query instead."
- **Dedup**: Cache expansions by `(memory_id, window, collection)`. Same combo = cached response immediately.
- **Window clamp**: `window = min(window, EXPAND_MAX_WINDOW)`.
- **Token tracking**: Estimate tokens from expanded result length (~4 chars/token). If cumulative expansion tokens exceed `EXPAND_MAX_TOTAL_TOKENS`, truncate surrounding turns more aggressively.
- **Anti-loop in system prompt**: Both the tool description and the XML prompt injection explicitly tell the agent: "If expanded context doesn't answer, search differently, don't re-expand."

---

## Rollout Strategy

**Ship Phase 0 + Phase 1 + Phase 2 together. Evaluate before wiring the full tool path (Phase 3).**

This means:
1. `thread_id` starts flowing into ChromaDB immediately (Phase 0)
2. `get_by_id()` and doc IDs in search results are available (Phase 1)
3. `MemoryExpander` class exists and is unit-tested (Phase 2)
4. **Pause here.** Run the evaluation plan below on a replay set before committing to the full agent integration.

Only after evaluation confirms that timestamp-window expansion actually recovers useful context (not just more text), proceed to Phase 3 + Phase 4.

---

## Evaluation Plan

Run these before wiring expand_memory into the live agent loop:

### 1. Anchor Usefulness Test
Given a top search_memory hit, does expansion recover the setup/follow-up/closure that was missing from the isolated snippet? Manually inspect 10 real conversation hits.

### 2. Answer Lift Test
Compare answer quality with and without expansion on a fixed replay set of real queries. Score: did the expanded context change the agent's answer? Was the change an improvement?

### 3. Cost Test
Measure extra tokens and latency per expansion. Verify the 2000-token budget cap actually holds. Measure `list_all()` sort time on a full-size conversations collection.

### 4. Boundary Test
- First item in collection (window extends only forward)
- Last item in collection (window extends only backward)
- Same-timestamp clusters (3+ facts with identical timestamps — verify deterministic ordering via doc_id secondary sort)

### 5. Noisy-Neighbor Test
Ensure irrelevant neighbors (from a different conversation that happened to be temporally close) do not drown out the anchor. If this is a frequent problem, it validates the need for Phase 0's `thread_id` propagation.

---

## Files Summary

### New Files
| File | Purpose |
|---|---|
| `memory/memory_expander.py` | MemoryExpander class — timestamp-based window expansion |
| `tests/unit/test_memory_expander.py` | Unit tests: window slicing, boundary guards, tie-breaking, caching |

### Modified Files
| File | Change |
|---|---|
| `memory/memory_storage.py` | Propagate `thread_id`/`thread_depth` to ChromaDB metadata (~4 lines) |
| `memory/storage/multi_collection_chroma_store.py` | Add `get_by_id()` method |
| `core/agentic/types.py` | Add `EXPAND_MEMORY_TOOL_DEFINITION` + `SearchDecision` fields + XML prompt |
| `core/agentic/protocols.py` | Parse `expand_memory` in both native + XML handlers |
| `core/agentic/controller.py` | Add `_execute_memory_expand()`, `_format_expanded_results()`, gating, init |
| `config/app_config.py` | Add `EXPAND_*` config constants |
| `config/config.yaml` | Add `memory_expansion:` section |

---

## What This Does NOT Do (Intentionally)

1. **No cross-collection drilldown (summary→conversations)** — Summaries don't store `source_turn_ids`. Adding that metadata linkage is a separate, larger project. When it exists, expand_memory can be upgraded to follow those links.
2. **No embedding-based expansion** — Timestamp proximity is simpler and more predictable. Semantic neighbors often span different topics.
3. **No full-text expansion** — Turns are still truncated (600/300 chars). The agent can call `expand_memory` with `window=0` on a specific ID to get just that turn with more content if needed.

---

## Implementation Order

1. **Phase 0**: `memory_storage.py` — Propagate `thread_id`/`thread_depth` to ChromaDB metadata (4 lines)
2. **Phase 1a**: `multi_collection_chroma_store.py` — Add `get_by_id()`
3. **Phase 1b**: `controller.py` — Add doc IDs + collection tag to `_format_memory_results()`
4. **Phase 2**: `memory_expander.py` — Core expansion logic with deterministic sort + boundary guards
5. **Phase 2b**: `test_memory_expander.py` — Unit tests including same-timestamp tie-breaking
6. **EVALUATE** — Run evaluation plan on replay set. Go/no-go for Phase 3.
7. **Phase 4**: Config (app_config.py + config.yaml) — Constants
8. **Phase 3a**: `types.py` — Tool definition + SearchDecision fields + XML prompt
9. **Phase 3b**: `protocols.py` — Parse expand_memory calls (native + XML)
10. **Phase 3c**: `controller.py` — Execution handler, formatting, gating, init
11. **Run full test suite** — Verify no regressions

---

## Future Enhancements

- **Thread-aware expansion**: Once enough conversations have `thread_id` in ChromaDB (Phase 0 data accumulates), prefer thread grouping over timestamp window. Timestamp becomes the fallback for pre-Phase-0 data only.
- **Source linking**: Store `source_turn_ids: [id1, id2, ...]` on summaries and `source_turn_id` on facts. Enables cross-collection drilldown (the "real microscope").
- **Smart truncation**: Use LLM summarization on surrounding turns instead of hard character truncation.
- **Session ID propagation**: Add `session_id` for even cleaner grouping than thread_id (one session = one GUI conversation).
