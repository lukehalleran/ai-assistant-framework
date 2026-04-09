# Two-Step Generation with Thinking Blocks - Implementation Summary

## Overview
Two-step generation where the LLM provides internal reasoning before delivering the final answer. The thinking block is logged for debugging but only the final answer is shown to users and stored in memory.

Three layers of thinking separation exist (defense in depth):
1. **Native API reasoning** — for Claude/DeepSeek-R1, thinking is separated at the OpenRouter API level via `extra_body={"reasoning": {"effort": "medium"}}`. Thinking arrives in `delta.reasoning_content`, not in the text response.
2. **Tag-based parsing** — `<thinking>`/`<think>`/`<output>` tags parsed by `ResponseParser.parse_thinking_block()`
3. **Heuristic fallback** — `_detect_untagged_thinking()` catches chain-of-thought dumped without tags (meta-reasoning patterns, instruction echoes)

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

**Guards:**
- Requires `_HEURISTIC_MIN_HITS = 2` distinct patterns to trigger (prevents false positives)
- Response must be ≥80 chars and ≥3 lines
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

**Location:** `core/orchestrator.py` at line ~1683

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
2. Shows "Thinking..." indicator while thinking block is incomplete
3. Once `</thinking>` arrives, switches to displaying final answer
4. After streaming completes, re-parses to ensure clean storage

### 6. Test Suite (`test_thinking_blocks.py`)

Covers (tag-based parsing):
- Normal thinking block extraction (both `<thinking>` and `<think>` variants)
- No thinking block (passthrough)
- Thinking blocks with newlines
- Empty responses, malformed tags

Note: Heuristic detection (`_detect_untagged_thinking`) and `<output>` wrapper were verified via inline smoke tests during development but do not yet have dedicated test file coverage.

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
| Cleanup | `strip_thinking_tag_leaks()` | Partial/malformed tags (e.g., `/think>`, `<|think|>`) |

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
