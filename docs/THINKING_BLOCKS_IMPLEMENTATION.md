# Two-Step Generation with Thinking Blocks - Implementation Summary

*Last verified: 2026-06-10*

## Overview
Two-step generation where the LLM provides internal reasoning before delivering the final answer. The thinking block is logged for debugging but only the final answer is shown to users and stored in memory.

Three core layers of thinking separation exist, plus six operational layers (defense in depth, 9 total — see table below):
1. **Native API reasoning** — for Claude/DeepSeek-R1, thinking is separated at the OpenRouter API level via `extra_body={"reasoning": {"effort": "medium"}}`. Thinking arrives in `delta.reasoning_content`, not in the text response.
2. **Tag-based parsing** — `<thinking>`/`<think>`/`<output>` tags parsed by `ResponseParser.parse_thinking_block()`
3. **Heuristic fallback** — `_detect_untagged_thinking()` catches chain-of-thought dumped without tags (meta-reasoning patterns, instruction echoes)
4. **Storage boundary guard** — `ResponseParser.sanitize_for_storage()` runs inside `memory_storage.store_interaction()`; no storage path can persist thinking artifacts into memory (see "Storage Boundary Guard" section)

## Changes Made

### 1. Response Parser (`core/response_parser.py`)

#### `parse_thinking_block()` Static Method

```python
@staticmethod
def parse_thinking_block(response: str) -> Tuple[str, str]:
    """
    Parse response to extract thinking block and final answer.

    Handles (in order):
    1. <thinking>...</thinking> (Anthropic/OpenAI style)
    2. <think>...</think> (DeepSeek/Qwen/GLM style)
    3. <output>...</output> wrapper (some OpenRouter providers)
    4. Heuristic detection of untagged thinking (fallback)

    Returns:
        Tuple of (thinking_part, final_answer_part)
        - If no thinking block found, thinking_part is empty
          and final_answer_part is the full response
    """
```

**Functionality:**
- Tries both `</thinking>` and `</think>` delimiters
- Extracts content between open and close tags
- Unwraps `<output>...</output>` wrapper if present in final answer
- Falls back to `_detect_untagged_thinking()` when no tags found
- Returns clean final answer (everything after closing tag)
- Handles edge cases: no thinking block, malformed tags, empty responses

#### `_detect_untagged_thinking()` Static Method (NEW 2026-04-05)

Heuristic fallback for when models dump chain-of-thought without wrapping in tags.

```python
@staticmethod
def _detect_untagged_thinking(response: str) -> Tuple[str, str]:
    """Heuristic fallback: detect untagged thinking dumped before the real answer."""
```

**Pattern categories** (`_THINKING_HEURISTIC_PATTERNS`):
- Meta-reasoning: "I should...", "I need to...", "Let me think..."
- Third-person user references: "He's saying...", "The user is asking..."
- Planning language: "What would actually be useful...", "I could mention..."
- System prompt instruction echo: "Walk through the context step-by-step"
- Conversational meta-analysis: "This is a casual...", "not asking me to..."
- Bullet-style reasoning: "- Explicitly...", "- Temporal..."

**Sentence-level patterns** (`_THINKING_SENTENCE_PATTERNS`, NEW 2026-05):
- Catches single-paragraph thinking where multi-line heuristic fails
- `_count_sentence_pattern_hits()` scores sentence-level indicators within
  the first paragraph
- Enables detection when the model dumps chain-of-thought as a single
  dense paragraph without blank-line breaks

**Guards:**
- Requires `_HEURISTIC_MIN_HITS = 2` distinct patterns to trigger (prevents false positives)
- Response must be ≥80 chars and ≥3 lines (multi-line path) or sentence-level patterns hit threshold (single-paragraph path)
- Remaining answer must be ≥20 chars (won't strip entire response)
- Splits at first blank line after last pattern hit

### 2. Native Reasoning via API (`models/model_manager.py`) (NEW 2026-04-05)

Both `generate_async()` (streaming) and `generate_once()` (non-streaming) now request native reasoning separation for models where `supports_reasoning()` returns True:

```python
if self.supports_reasoning(target_model):
    create_kwargs["extra_body"] = {
        "reasoning": {"effort": "medium"},
    }
```

When enabled, OpenRouter returns thinking in `delta.reasoning_content` streaming chunks (not in the text body). The `ResponseGenerator` already handles these via synthetic `<thinking>` tag emission (lines 177-188 of `response_generator.py`).

**Supported models:** All `anthropic/claude-*` models, `deepseek-r1`.

### 3. Conditional Thinking Instruction (`core/orchestrator.py`) (CHANGED 2026-04-05)

The system prompt thinking instruction is now **skipped** for models with native reasoning:

```python
if not use_raw_mode:
    _active = getattr(self.model_manager, "get_active_model_name", lambda: None)()
    _has_native = (
        _active
        and hasattr(self.model_manager, "supports_reasoning")
        and self.model_manager.supports_reasoning(_active)
    )
    if not _has_native:
        thinking_instruction = (
            "\n\n[IMPORTANT] Before your final response, include your reasoning "
            "in <thinking>...</thinking> tags. Walk through the context step-by-step, "
            "then provide your answer outside the tags."
        )
        system_prompt = system_prompt.rstrip() + thinking_instruction
```

**Rationale:** The prompt instruction was redundant for Claude/DeepSeek-R1 (thinking is already separated at the API level) and could cause the model to echo the instruction text, leading to thinking leaks.

### 4. Thinking Block Handling in `process_user_query()`

**Location:** `core/orchestrator.py` at lines ~1835-1851

1. **Full response accumulation** — Response is accumulated first, not streamed immediately
2. **Thinking block parsing** (delegates to ResponseParser)
   ```python
   thinking_part, final_answer = ResponseParser.parse_thinking_block(full_response)
   final_answer = ResponseParser.strip_xml_wrappers(final_answer)
   ```
3. **Debug logging**
   ```python
   if thinking_part:
       logger.debug(f"[THINKING BLOCK]\n{thinking_part}")
       debug_info["thinking_length"] = len(thinking_part)
   ```
4. **Memory storage** — Only final answer is stored (not thinking block)
5. **Return value** — Returns only the final answer to the user

### 5. Streaming Thinking Handling (`gui/handlers.py`)

During streaming, `handlers.py`:
1. Checks `ResponseParser.has_incomplete_thinking_block(final_output)` after each chunk
2. Also checks `ResponseParser.likely_untagged_thinking(final_output)` to suppress untagged chain-of-thought before `parse_thinking_block()` can find a clean split point
3. Shows "Thinking..." indicator while thinking block is incomplete
4. Once `</thinking>` arrives, switches to displaying final answer
5. After streaming completes, re-parses to ensure clean storage

**Image limitation:** When images are present in the request, native reasoning is skipped in `generate_async()` because OpenRouter may not support both extended thinking and image input simultaneously. The system falls back to tag-based and heuristic parsing layers.

### 6. Interleaved Reasoning-Stream Filter (`core/reasoning_stream_filter.py`) (NEW 2026-06)

A **streaming-layer** defense for reasoning models (via OpenRouter) that interleave reasoning and content *within a single streamed response*. The naive consumer — "yield every content delta, suppress reasoning-only chunks" — fused a discarded pre-answer reasoning draft directly onto the real answer with **no separator**, producing a user-visible leak like `synthesis system.Let me check…` (a 17-char discarded fragment `synthesis system.` glued to the answer `Let me check…`). Because the fragment is untagged and separator-less, none of the tag / heuristic / storage strippers caught it — it survived `sanitize_for_storage()` and reached both display and storage.

`InterleavedReasoningFilter` sits between the raw delta stream and the consumer (`feed()` per delta, `finish()` once at end):
- Suppresses reasoning-only chunks, emitting synthetic `<thinking>` / `</thinking>` markers around them (preserving the existing thinking-block contract the GUI relies on).
- Holds the **leading content run** until it is confirmed genuine: a run that grows past `draft_max_chars` (default 50) is committed and then streams live, so long real answers still stream token-by-token; a short run cut off because reasoning *resumes* is dropped as a provisional draft — but restored at `finish()` if nothing ever replaces it, so a genuinely short final answer is never lost.
- Stays completely **inert until the first reasoning chunk**, so non-reasoning models stream exactly as before (no added latency, no behaviour change).

Shared by `core/response_generator.py` (`generate_streaming_response`) and `core/agentic/controller.py` (`_generate_final_response`) so the two streaming paths cannot drift.

**Reasoning-only recovery (`ResponseGenerator._recover_reasoning_only()`, NEW 2026-06):** the complementary failure — a reasoning model swallows the *entire* answer into the hidden reasoning channel and streams **empty** visible content. When the filter reports `reasoning_seen` but not `content_emitted`, the generator closes the dangling `<thinking>` marker and retries **once** non-streaming via `generate_once(disable_reasoning=True)`, forcing the answer into normal content instead of resolving to an empty/dead response.

### 7. Test Suite (`test_thinking_blocks.py`)

Covers (tag-based parsing):
- Normal thinking block extraction (both `<thinking>` and `<think>` variants)
- No thinking block (passthrough)
- Thinking blocks with newlines
- Empty responses, malformed tags

Heuristic detection has dedicated test coverage in `tests/unit/test_thinking_heuristic.py` (23 test functions) covering `likely_untagged_thinking()` true positives, true negatives, length bail-outs, edge cases, and consistency with `_detect_untagged_thinking()`.

## How It Works

### Flow Diagram

```
User Query
    ↓
System Prompt
    ├─ (non-reasoning model) + thinking instruction appended
    └─ (reasoning model) no instruction — API handles separation
    ↓
API Call
    ├─ (reasoning model) extra_body={"reasoning": {"effort": "medium"}}
    └─ (other models) standard call
    ↓
LLM Generation
    ↓
Streaming Chunks
    ├─ delta.reasoning_content → synthetic <thinking> tags → suppressed
    └─ delta.content → yielded to user
    ↓
Full Response
    ↓
ResponseParser.parse_thinking_block()
    ├─ Layer 1: Tag-based (<thinking>/<think>/<output>)
    ├─ Layer 2: Heuristic (_detect_untagged_thinking)
    └─ Layer 3: Strip leaked tag fragments
    ↓
    ├─→ thinking_part → Logged for debugging
    └─→ final_answer → Returned to user & stored in memory
```

### Example

**Input:** "What is 2+2?"

**LLM Response:**
```
<thinking>
The user is asking a simple arithmetic question.
1. Operation: addition
2. 2 + 2 = 4
3. Provide clear answer
</thinking>

The answer to 2 + 2 is 4.
```

**What happens:**
- **Logged (debug):** The reasoning steps from `<thinking>` block
- **Shown to user:** "The answer to 2 + 2 is 4."
- **Stored in memory:** "The answer to 2 + 2 is 4."

## Defense in Depth

| Layer | Mechanism | Handles |
|-------|-----------|---------|
| API | `extra_body={"reasoning": {"effort": "medium"}}` | Claude, DeepSeek-R1 — thinking never reaches text body |
| Tags | `parse_thinking_block()` | `<thinking>`, `<think>`, `<output>` wrappers |
| Heuristic | `_detect_untagged_thinking()` | Models that ignore tag instruction and dump reasoning as plain text |
| Sentence | `_count_sentence_pattern_hits()` | Single-paragraph chain-of-thought without blank-line breaks |
| Agentic | `_detect_untagged_thinking()` on final output | Thinking leak in agentic path final generation |
| Streaming | `strip_thinking_tag_leaks()` + stuck recovery | XML marker fragments and stuck thinking state after streaming ends |
| Interleaved | `InterleavedReasoningFilter.feed()` (`core/reasoning_stream_filter.py`) | Reasoning model fuses a discarded pre-answer draft onto the real answer with NO separator, e.g. `…system.Let me check…` — untagged, so tag/heuristic/storage strippers miss it [NEW 2026-06] |
| Cleanup | `strip_thinking_tag_leaks()` | Partial/malformed tags (e.g., `/think>`, `<|think|>`) |
| **Storage boundary** | `ResponseParser.sanitize_for_storage()` in `memory_storage.store_interaction()` | ANY leak that survives display-layer defenses — never persisted. All-thinking responses skip storage entirely (returns None) [NEW 2026-06-10] |

## Storage Boundary Guard (2026-06-10)

**Why it exists:** The agentic storage path persisted the *raw* accumulated stream
(`final_output`), which contains the synthetic `<thinking></thinking>` markers
`response_generator.py` emits around API-separated reasoning. Display stripped them;
storage did not. Result: 752/5920 conversation docs, 429/2000 corpus entries,
16 reflections, and 1 summary contained literal thinking tags. Replayed into
`[RECENT CONVERSATION]` context, this taught the model to emit literal
`<thinking></thinking>` itself — a self-reinforcing pollution loop.

**The fix (three parts):**
1. `ResponseParser.sanitize_for_storage(text)` — canonical pre-persistence strip:
   empty `<thinking></thinking>` pairs anywhere, leading tagged blocks, unclosed
   leading blocks (→ `""`), stray fragments, reflection blocks. Deliberately does
   NOT apply untagged-thinking heuristics (false positives must not eat real
   conversation content at the storage boundary).
2. `memory_storage.store_interaction()` calls it before ANY persistence (corpus,
   conversation context, ChromaDB) — covers every storage path including future ones.
   The agentic handler path additionally routes through `_sanitize_response_text()`.
3. `scripts/repair_thinking_leaks.py` — one-time repair of historical pollution
   (dry-run by default, `--apply` to write, refuses to run while Daemon is up,
   content-update only — never deletes, re-embeds via the collection's embedder).
   Hardened 2026-07-04: `--apply` writes a pre-image JSONL backup
   (`logs/repair_thinking_leaks_preimage.<ts>.jsonl`) before any update, and the
   scrubber is quoted-tag-safe — only leading / `Assistant:`-anchored
   `<tag>…</tag>` blocks are deleted whole; a mid-text pair loses just the tag
   literals, so conversations QUOTING the tags (e.g. discussing the leak bug)
   keep their content.

Tests: `tests/unit/test_thinking_leak_storage.py` (30 tests).

## Backward Compatibility

- **No breaking changes:** If LLM doesn't include `<thinking>` tags AND heuristic finds nothing, full response is returned as-is
- **Raw mode:** Thinking instruction NOT added in raw mode; API reasoning NOT requested
- **Graceful degradation:** Parser handles malformed tags safely
- **Multi-provider:** Supports `<thinking>` (Anthropic/OpenAI), `<think>` (DeepSeek/Qwen/GLM), and native API reasoning

## Configuration

No configuration needed — feature is automatic for all non-raw mode queries.

- To disable thinking blocks entirely, set `use_raw_mode=True` when calling `process_user_query()`
- Native reasoning is auto-enabled for models where `supports_reasoning()` returns True
- Heuristic threshold (`_HEURISTIC_MIN_HITS = 2`) can be adjusted in `response_parser.py`
