# Two-Step Generation with Thinking Blocks - Implementation Summary

## Overview
Successfully implemented a two-step generation feature where the LLM provides internal reasoning in a `<thinking>` block before delivering the final answer. The thinking block is logged for debugging but only the final answer is shown to users and stored in memory.

## Changes Made

### 1. Core Orchestrator (`core/orchestrator.py`)

#### Added `_parse_thinking_block()` Static Method
**Location:** Lines 118-149

```python
@staticmethod
def _parse_thinking_block(response: str) -> Tuple[str, str]:
    """
    Parse response to extract thinking block and final answer.

    Args:
        response: Full LLM response potentially containing <thinking>...</thinking>

    Returns:
        Tuple of (thinking_part, final_answer_part)
        - If no thinking block found, thinking_part is empty
          and final_answer_part is the full response
    """
```

**Functionality:**
- Looks for `</thinking>` delimiter
- Extracts content between `<thinking>` and `</thinking>`
- Returns clean final answer (everything after `</thinking>`)
- Handles edge cases: no thinking block, malformed tags, empty responses

#### Modified `prepare_prompt()` Method
**Location:** Lines 378-388

Added thinking block instruction to system prompt:

```python
if not use_raw_mode and isinstance(system_prompt, str) and system_prompt.strip():
    thinking_instruction = (
        "\n\n"
        "IMPORTANT: Before you provide your final answer, you must include a <thinking> block. "
        "Inside this block, detail your step-by-step reasoning and analysis of the user's request. "
        "After the </thinking> block, provide your final, concise, and helpful response to the user."
    )
    system_prompt = system_prompt.rstrip() + thinking_instruction
```

**Key Points:**
- Only added in enhanced mode (not raw mode)
- Appended after topic hint
- Instructs LLM to use `<thinking>` tags for reasoning

#### Modified `process_user_query()` Method
**Location:** Lines 669-720

**Changes to streaming response handling:**

1. **Accumulation without yielding** (Lines 677-682)
   - Full response is accumulated first
   - No immediate streaming to allow parsing

2. **Thinking block parsing** (Lines 684-691)
   ```python
   thinking_part, final_answer = self._parse_thinking_block(full_response)

   if thinking_part:
       if self.logger:
           self.logger.debug(f"[THINKING BLOCK]\n{thinking_part}")
       debug_info["thinking_length"] = len(thinking_part)
   ```

3. **Memory storage** (Lines 693-705)
   - Only final answer is stored (not thinking block)
   ```python
   answer_for_storage = final_answer if final_answer else full_response

   await self.memory_system.store_interaction(
       query=user_input,
       response=answer_for_storage,
       tags=["conversation"]
   )
   ```

4. **Return value** (Line 720)
   - Returns only the final answer to the user
   ```python
   return answer_for_storage, debug_info
   ```

5. **Debug info updates** (Lines 711-717)
   - Added `thinking_length` field
   - Added `full_response_length` field

### 2. Test Suite (`test_thinking_blocks.py`)

Created comprehensive test suite covering:

1. **Parsing Tests**
   - Normal thinking block extraction
   - No thinking block (passthrough)
   - Thinking blocks with newlines
   - Empty responses
   - Malformed tags

2. **Integration Example**
   - Demonstrates full flow from LLM response to user output
   - Shows what gets logged vs. stored vs. displayed

## How It Works

### Flow Diagram

```
User Query
    ↓
System Prompt (with thinking instruction)
    ↓
LLM Generation
    ↓
Full Response: "<thinking>reasoning...</thinking>Final answer"
    ↓
Parse Thinking Block
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

## Testing Results

All core parsing tests passed ✅:
- Normal thinking block extraction
- No thinking block (passthrough)
- Newlines and formatting preserved
- Empty response handling
- Malformed tag safety

Integration example demonstrated:
- Correct separation of thinking vs. final answer
- Proper logging of thinking content
- Clean final answer delivery

## Configuration

No configuration needed - feature is automatic for all non-raw mode queries.

To disable thinking blocks, set `use_raw_mode=True` when calling `process_user_query()`.

## Future Enhancements

Potential improvements:
1. Add `ENABLE_THINKING_BLOCKS` config flag
2. Support multiple thinking blocks in one response
3. Add thinking block analysis/metrics
4. Optional user-visible thinking toggle
5. Structured thinking formats (e.g., chain-of-thought templates)
