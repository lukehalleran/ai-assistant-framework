# STM System & GUI Rendering - Debugging Session Handoff

**Date**: 2025-11-24
**Status**: STM system FIXED and working, GUI rendering issue DIAGNOSED but not fixed

---

## 1. STM (Short-Term Memory) System - COMPLETED ✅

### Problem Summary
The STM system was implemented but the `[SHORT-TERM CONTEXT SUMMARY]` section was not appearing in prompts even though:
- The STM analyzer was running successfully (confirmed in logs)
- The conversation depth threshold (3+ messages) was met
- The `stm_summary` dict was being generated with correct data

### Root Causes Identified

#### Issue 1: Token Manager Filtering (FIXED)
**Location**: `core/prompt/token_manager.py`

**Problem**: The `PRIORITY_ORDER` list (line 33) didn't include `stm_summary`, so the token manager was silently dropping it during budget processing.

**Fix Applied** (lines 34, 158-160, 223-224):
```python
PRIORITY_ORDER = [
    ("stm_summary",          10),  # Highest priority - STM context should never be trimmed
    ("recent_conversations", 7),
    # ... rest of priorities
]

# In _manage_token_budget():
if name == "stm_summary":
    logger.warning(f"[TOKEN BUDGET] Preserving stm_summary (metadata, no token cost)")
    continue

# In _total_tokens() helper:
if name == "stm_summary":
    continue  # Skip counting - it's metadata
```

#### Issue 2: Final Context Assembly Filtering (FIXED)
**Location**: `core/prompt/builder.py:635-649`

**Problem**: The "Step 8: Final context assembly" creates a new `prompt_ctx` dict with an explicit whitelist of keys. The `stm_summary` was not in this whitelist, so it got filtered out right before returning from `build_prompt()`.

**Fix Applied** (line 649):
```python
prompt_ctx = {
    "recent_conversations": context.get("recent_conversations", []),
    "memories": context.get("memories", []),
    # ... other keys ...
    "wiki": context.get("wiki", []),
    "stm_summary": context.get("stm_summary")  # STM context summary (dict or None)
}
```

#### Issue 3: STM Placement Optimization (FIXED)
**Location**: `core/prompt/builder.py`

**Problem**: STM was rendering at the top of the prompt (after TIME CONTEXT), but research shows it should be in the highest attention area - right before the user's query.

**Fix Applied** (lines 1330-1352):
- Removed STM rendering from line ~1165 (near TIME CONTEXT)
- Added STM rendering at line 1330, immediately before `[CURRENT USER QUERY]`
- This places STM in the maximum attention window for the LLM

**Reasoning**: STM is short-term working memory for the immediate conversation. Placing it right before the query means:
- It's the last context the model sees before processing the user's input
- It acts as an immediate "lens" for interpreting the current message
- Better for multi-turn coherence vs. being thousands of tokens earlier

### Verification
STM system is now fully functional as confirmed by user:
```
[SHORT-TERM CONTEXT SUMMARY]
Topic: STM testing
User Question: Is the STM system online?
Intent: Confirm the operational status of the STM system
Tone: neutral
Open Threads: Weird outputs from STM, Effectiveness of the STM system
```

### Files Modified
1. `core/prompt/token_manager.py` - Lines 34, 158-160, 223-224
2. `core/prompt/builder.py` - Lines 649, 1330-1352, removed lines ~1165-1188

---

## 2. GUI Response Rendering Issue - DIAGNOSED BUT NOT FIXED ⚠️

### Problem Summary
Responses generate successfully on the backend (visible in terminal logs and debug tab) but do NOT render in the Gradio chat interface. User sees empty responses.

### Symptoms
- Terminal shows: `HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"`
- Terminal shows: `[HANDLE_SUBMIT] Storing interaction in memory...` followed by `Interaction successfully stored.`
- Debug tab shows the full response text
- BUT: Chat interface shows no response (blank/empty)

### Investigation Findings

#### ✅ Confirmed Working Components
1. **Backend Response Generation**: HTTP requests succeed, responses are generated
2. **Memory Storage**: Interactions are stored successfully in corpus
3. **GUI Handler Code**: `gui/handlers.py` lines 500-537 look correct for streaming
4. **GUI Receiver Code**: `gui/launch.py` lines 353-379 look correct for receiving chunks
5. **Placeholder Message**: Assistant message placeholder is added (launch.py:267)

#### ❌ Suspected Issue: Streaming Chunks Not Flowing

**Key Observation**: No `[STREAMING]` debug logs appear in terminal output, which suggests:
- Either `generate_streaming_response()` is not yielding chunks
- Or the async generator is not being iterated properly
- Or chunks are arriving but are empty/falsy and being filtered

**Evidence**:
```python
# gui/launch.py:356 - Only updates if content is truthy
if assistant_reply:  # Only update if there's actual content
    if chat_history and isinstance(chat_history[-1], dict):
        chat_history[-1]["content"] = assistant_reply
```

If chunks arrive as empty strings or None, they won't update the chat history.

### Code Locations for Investigation

#### 1. Response Generator (Streaming Source)
**File**: `core/response_generator.py`
**Key Method**: `generate_streaming_response()` (lines 37-196)

**Streaming loop**: Lines 113-175
- Wraps OpenAI/OpenRouter streaming with timeout protection
- Extracts delta content from different chunk shapes
- Should yield each delta as it arrives

**Expected behavior**: Should see `[STREAMING]` logs in terminal if chunks are flowing

#### 2. GUI Handler (Streaming Consumer)
**File**: `gui/handlers.py`
**Key Method**: `handle_submit()` (lines 41-676)

**Main streaming path**: Lines 496-537 (the `else` block when `use_bestof=False`)
```python
async for chunk in orchestrator.response_generator.generate_streaming_response(
    prompt=full_prompt,
    model_name=model_name,
    system_prompt=system_prompt
):
    final_output = smart_join(final_output, chunk)
    # ... yields to GUI
```

**Should yield**: Lines 514, 519, 528, 537

#### 3. GUI Launch (Async Iterator)
**File**: `gui/launch.py`
**Key Method**: `submit_chat()` (lines 260-443)

**Async iteration loop**: Lines 295-379
```python
agen = handle_submit(...)
next_task = _a.create_task(agen.__anext__())
while True:
    # ... wait for chunk or tick
    if next_task in done:
        chunk = next_task.result()
        # ... process chunk
```

**Should update**: Lines 356-379 when `isinstance(chunk, dict) and "content" in chunk`

---

## 3. GAMEPLAN FOR NEW INSTANCE

### PRIORITY 1: Add Debug Logging to Trace Chunk Flow

#### Step 1: Add logging to response_generator.py
**File**: `core/response_generator.py`
**Location**: Around line 160 (inside the streaming while loop)

Add BEFORE the yield:
```python
# After extracting delta_content and before yielding
if delta_content:
    self.logger.warning(f"[STREAMING DEBUG] Yielding chunk: '{delta_content[:50]}...' (len={len(delta_content)})")
    yield delta_content
else:
    self.logger.warning(f"[STREAMING DEBUG] Skipping empty chunk")
```

#### Step 2: Add logging to handlers.py
**File**: `gui/handlers.py`
**Location**: Line 505 (inside the streaming async for loop)

Add right after receiving chunk:
```python
async for chunk in orchestrator.response_generator.generate_streaming_response(...):
    logger.warning(f"[GUI HANDLER DEBUG] Received chunk: {chunk!r}")  # Log raw chunk
    final_output = smart_join(final_output, chunk)
    # ... rest of code
```

#### Step 3: Add logging to launch.py
**File**: `gui/launch.py`
**Location**: Line 354 (when processing chunks)

Add:
```python
elif isinstance(chunk, dict) and "content" in chunk:
    assistant_reply = chunk["content"]
    logging.warning(f"[GUI LAUNCH DEBUG] Processing chunk content: '{assistant_reply[:100] if assistant_reply else 'EMPTY'}'")
    if assistant_reply:  # Only update if there's actual content
        # ... existing code
```

### PRIORITY 2: Test and Observe

1. **Restart the GUI** to load the debug logging changes
2. **Send a test message** in the GUI
3. **Watch the terminal output** for the new `[STREAMING DEBUG]`, `[GUI HANDLER DEBUG]`, and `[GUI LAUNCH DEBUG]` logs
4. **Analyze the output**:
   - If NO streaming logs appear → problem is in `response_generator.py` not yielding
   - If streaming logs appear but no handler logs → problem is async iteration
   - If handler logs appear but no launch logs → problem is chunk format/structure
   - If all logs appear → problem is in the GUI update logic (line 356+ in launch.py)

### PRIORITY 3: Based on Observations

#### If NO streaming logs (chunks not yielding):
- Check if `stream=True` is being passed to the OpenAI/OpenRouter client
- Check if the streaming iterator is being created correctly
- Look for exceptions being swallowed in the try/except blocks

#### If streaming logs but no handler logs (async iteration issue):
- Check if the async for loop is being entered
- Add logging BEFORE line 500 to confirm the else block is reached
- Check if `use_bestof` is accidentally True (should be False per logs)

#### If handler logs but no launch logs (chunk format issue):
- Check the actual structure of the chunk dict
- Verify it has `{"role": "assistant", "content": "..."}` format
- Check if `chunk["content"]` might be None or empty string

#### If all logs appear (GUI update issue):
- Check if `chat_history[-1]` exists and is the right format
- Check if Gradio is receiving the yield properly
- Look for Gradio version issues or compatibility problems

### PRIORITY 4: Quick Fixes to Try

#### Quick Fix 1: Force non-empty yields
In `gui/handlers.py` around line 528, change:
```python
# OLD:
if assistant_reply:  # Only update if there's actual content
    # ... update

# NEW:
if assistant_reply is not None:  # Update even if empty string (remove truthy check)
    if not assistant_reply:  # If empty, use placeholder
        assistant_reply = "…"
    # ... update
```

#### Quick Fix 2: Ensure streaming is enabled
In `gui/handlers.py` around line 423, add logging:
```python
logger.warning(f"[GUI DEBUG] About to start streaming with model={model_name}")
async for chunk in orchestrator.response_generator.generate_streaming_response(...):
```

#### Quick Fix 3: Check ModelManager streaming flag
In `models/model_manager.py`, verify that `stream=True` is being passed:
```python
# In async_completion() or similar method
logger.warning(f"[MODEL MANAGER DEBUG] Calling API with stream=True")
response = await client.chat.completions.create(
    model=model_name,
    messages=messages,
    stream=True,  # MUST be True
    # ...
)
```

---

## 4. KNOWN GOOD STATE

### What's Working
- ✅ STM analyzer runs and generates summaries
- ✅ STM summary appears in prompts at the right location
- ✅ Conversation depth tracking works correctly
- ✅ Token manager preserves STM metadata
- ✅ Prompt assembly includes all sections including STM
- ✅ Backend generates responses successfully
- ✅ Memory storage works
- ✅ Debug tab shows full responses

### What's NOT Working
- ❌ GUI chat interface does not display responses
- ❌ No streaming chunk logs appear in terminal

### Configuration at Time of Issue
From terminal logs:
```
[GUI] Duel check: duel_mode=False, gen=2, sel=2, use_bestof(duel-only)=False
[ModelManager] Active model set to: claude-opus
```

So the system is in:
- **Streaming mode** (not duel mode, not best-of mode)
- **Single model** (claude-opus)
- **Should be taking the main streaming path** (gui/handlers.py:496-537)

---

## 5. ADDITIONAL CONTEXT

### Orchestrator Flow (for reference)
```
User submits message
    ↓
gui/launch.py:submit_chat() - adds placeholder message
    ↓
gui/handlers.py:handle_submit() - prepares prompt
    ↓
orchestrator.prepare_prompt() - builds full prompt with STM
    ↓
core/prompt/builder.py:build_prompt() - assembles context
    ↓
core/prompt/builder.py:_assemble_prompt() - renders to string
    ↓
core/response_generator.py:generate_streaming_response() - yields chunks
    ↓
gui/handlers.py - receives chunks, yields to GUI
    ↓
gui/launch.py - updates chat_history with chunk content
    ↓
Gradio renders updated chat_history
```

### Git Status (for new instance awareness)
Modified files in this session:
- `core/prompt/token_manager.py` - STM fixes
- `core/prompt/builder.py` - STM fixes and placement
- `core/prompt/formatter.py` - Last exchange indexing fix (recent[0] not recent[-1])

These changes are NOT yet committed. To preserve them:
```bash
git add core/prompt/token_manager.py core/prompt/builder.py core/prompt/formatter.py
git commit -m "fix: STM system now working - added to token manager priority and prompt assembly"
```

---

## 6. QUICK START FOR NEW INSTANCE

```bash
# 1. Verify STM is still working
python main.py  # Start GUI
# Send 3+ messages and check prompt for [SHORT-TERM CONTEXT SUMMARY]

# 2. Add debug logging (see PRIORITY 1 above)
# Edit: core/response_generator.py, gui/handlers.py, gui/launch.py

# 3. Restart and test
# Watch terminal for [STREAMING DEBUG], [GUI HANDLER DEBUG], [GUI LAUNCH DEBUG] logs

# 4. Analyze and fix based on which logs appear (see PRIORITY 2 & 3 above)
```

---

## 7. REFERENCES

### Key Files
- `core/prompt/token_manager.py` - Token budgeting and STM preservation
- `core/prompt/builder.py` - Prompt assembly and STM rendering
- `core/prompt/formatter.py` - Alternative prompt formatting (legacy)
- `core/orchestrator.py` - Main request coordination
- `core/response_generator.py` - Streaming response generation
- `gui/handlers.py` - GUI event handlers and streaming consumer
- `gui/launch.py` - Gradio app setup and async iteration
- `core/stm_analyzer.py` - STM analysis logic

### Log Patterns to Watch For
- `[STM] Analysis complete:` - STM analyzer finished
- `[TOKEN BUDGET] Preserving stm_summary` - Token manager sees STM
- `STM RENDERING: Added STM section before query` - STM rendered in prompt
- `[STREAMING]` - Chunks being processed in response_generator
- `[GUI HANDLER DEBUG]` - Chunks arriving at handler
- `[GUI LAUNCH DEBUG]` - Chunks being processed by GUI

### Environment
- Python async/await with asyncio
- Gradio for web UI (async generators for streaming)
- OpenRouter API for LLM inference
- Streaming mode (not duel/best-of)

---

**END OF HANDOFF DOCUMENT**
