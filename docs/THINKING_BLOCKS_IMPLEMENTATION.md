# Two-Step Generation with Thinking Blocks - Implementation Summary

## Overview
Successfully implemented a two-step generation feature where the LLM provides internal reasoning in a `<thinking>` block before delivering the final answer. The thinking block is logged for debugging but only the final answer is shown to users and stored in memory.

## Changes Made

### 1. Response Parser (`core/response_parser.py`)

#### `parse_thinking_block()` Static Method
**Location:** `ResponseParser.parse_thinking_block()` at line ~90

```python
@staticmethod
def parse_thinking_block(response: str) -> Tuple[str, str]:
    """
    Parse response to extract thinking block and final answer.

    Handles:
    - <thinking>...</thinking> (Anthropic/OpenAI style)
    - <think>...</think> (DeepSeek/Qwen/GLM style)
    - <output>...</output> wrapper (some OpenRouter providers)

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
- Returns clean final answer (everything after closing tag)
- Handles edge cases: no thinking block, malformed tags, empty responses

### 2. Thinking Instruction Injection (`core/orchestrator.py`)

#### In `prepare_prompt()` Method
**Location:** `core/orchestrator.py` at lines ~1210-1216

Added thinking block instruction to system prompt:

```python
if not use_raw_mode:
    thinking_instruction = (
        "\n\n[IMPORTANT] Before your final response, include your reasoning "
        "in <thinking>...</thinking> tags. Walk through the context step-by-step, "
        "then provide your answer outside the tags."
    )
    system_prompt = system_prompt.rstrip() + thinking_instruction
```

**Key Points:**
- Only added in enhanced mode (not raw mode)
- Appended to system prompt after all other composition
- Instructs LLM to use `<thinking>` tags for reasoning

### 3. Thinking Block Handling in `process_user_query()`

**Location:** `core/orchestrator.py` at line ~1669

**Changes to response handling:**

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

### 4. Test Suite (`test_thinking_blocks.py`)

Created comprehensive test suite covering:

1. **Parsing Tests**
   - Normal thinking block extraction
   - No thinking block (passthrough)
   - Thinking blocks with newlines
   - Empty responses
   - Malformed tags
   - `<think>` variant (DeepSeek/Qwen style)
   - `<output>` wrapper unwrapping

2. **Integration Example**
   - Demonstrates full flow from LLM response to user output
   - Shows what gets logged vs. stored vs. displayed

## How It Works

### Flow Diagram

```
User Query
    ↓
System Prompt (with thinking instruction appended in orchestrator.prepare_prompt())
    ↓
LLM Generation
    ↓
Full Response: "<thinking>reasoning...</thinking>Final answer"
    ↓
ResponseParser.parse_thinking_block()
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

## Benefits

1. **Improved Reasoning Quality**
   - LLM explicitly works through problem before answering
   - Reduces rushed or incorrect responses

2. **Debugging Capability**
   - Thinking blocks logged for analysis
   - Can diagnose why LLM made certain decisions

3. **Clean User Experience**
   - Users see only polished final answers
   - No verbose reasoning cluttering the UI

4. **Memory Efficiency**
   - Only relevant answers stored in conversation history
   - Thinking blocks don't pollute retrieval

## Backward Compatibility

- **No breaking changes:** If LLM doesn't include `<thinking>` tags, full response is returned as-is
- **Raw mode:** Thinking instruction NOT added in raw mode
- **Graceful degradation:** Parser handles malformed tags safely
- **Multi-provider:** Supports both `<thinking>` (Anthropic/OpenAI) and `<think>` (DeepSeek/Qwen/GLM) variants

## Testing Results

All core parsing tests passed:
- Normal thinking block extraction
- No thinking block (passthrough)
- Newlines and formatting preserved
- Empty response handling
- Malformed tag safety
- Multi-provider tag variants

## Configuration

No configuration needed - feature is automatic for all non-raw mode queries.

To disable thinking blocks, set `use_raw_mode=True` when calling `process_user_query()`.
