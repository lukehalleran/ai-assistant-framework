# Live System Test Plan - Meta-Conversational Queries

## Quick Start: CLI Mode Test

### Step 1: Start CLI mode
```bash
python main.py cli
```

### Step 2: Have a short conversation to establish context
```
You: Hey, I'm planning to take Monday off as a rest day
[Agent responds]

You: I've been working really hard lately and need the break
[Agent responds]

You: I had my last day off on Tuesday I think
[Agent responds]
```

### Step 3: Test meta-conversational recall (THE KEY TEST)
```
You: Do you recall the last day off I mentioned?
```

**Expected behavior:**
- ✅ Agent should mention "Tuesday" (from step 2)
- ✅ Agent should reference the actual conversation you just had
- ✅ Agent should NOT mention random unrelated topics
- ✅ You should see debug log: `[MemoryCoordinator] Detected meta-conversational query`

**Bad behavior (what we fixed):**
- ❌ "I'm coming up short on specifics"
- ❌ Mentions topics you never discussed (hallucination)
- ❌ Generic response without referencing the actual conversation

### Step 4: Test more meta-conversational patterns
```
You: We talked about this earlier, remember?
You: Didn't you say something about Monday?
You: What did I tell you about my work schedule?
```

Each should retrieve the relevant part of your current conversation.

### Step 5: Check the logs

Look for these log messages to confirm the fix is working:

```bash
grep "Meta-conversational" conversation_logs/*.txt
grep "Detected meta-conversational query" conversation_logs/*.txt
```

You should see entries like:
```
[MemoryCoordinator] Detected meta-conversational query: Do you recall...
[MemoryCoordinator] Using meta-conversational retrieval strategy
[MemoryCoordinator] Meta-conversational retrieval returned N memories
```

---

## Option 2: GUI Test (More Realistic)

### Step 1: Start the GUI
```bash
python main.py
# or
make -f Makefile.fast run
```

### Step 2: Open browser
Navigate to the Gradio URL (usually http://localhost:7860)

### Step 3: Run the same conversation script
Follow the same conversation flow as the CLI test above.

### Step 4: Monitor the terminal
Watch for debug logs in the terminal where you started the GUI:
```
[MemoryCoordinator] Detected meta-conversational query...
```

---

## Option 3: Reproduce the Original Failure (Regression Test)

This replicates the exact scenario from the conversation log:

### Step 1: Start CLI
```bash
python main.py cli
```

### Step 2: Reproduce the original conversation

**Message 1:**
```
You: Hey man. Im oddly fried for having woken up at noon. I slept pretty and didn't drink anything yesterday but just kinda feel like..I just wanna lay with eyes half open lol. Do you recall last totally off day? I think I may have had one on Tuesday but can't recall. I had one intended recently but worked out for sure and did some project stuff
```

**Message 2 (THE CRITICAL TEST):**
```
You: Yeah I gotta work 8 hours today but on Monday I can. But do you recall the last fully off day I told you about? Just trying to get a sense
```

**Expected NEW behavior:**
- ✅ Agent should reference "Tuesday" from Message 1
- ✅ Agent should say something like "Yes, you mentioned Tuesday in your first message"
- ✅ NO hallucinations about anarchism, ICE, or other topics

**OLD buggy behavior (should NOT happen):**
- ❌ "I've been wracking my brain... coming up short"
- ❌ Mentions anarchism, ICE raids, Broadview (hallucination)

---

## Verification Checklist

After testing, verify these behaviors:

### ✅ Detection Working
- [ ] Debug logs show "Detected meta-conversational query"
- [ ] Debug logs show "Using meta-conversational retrieval strategy"
- [ ] Queries with "do you recall", "we discussed", etc. are detected

### ✅ Retrieval Working
- [ ] Agent references actual recent conversation content
- [ ] Agent recalls specific details you mentioned (e.g., "Tuesday")
- [ ] Responses feel contextually aware of the current conversation thread

### ✅ No Hallucinations
- [ ] Agent does NOT mention topics you never discussed
- [ ] Agent does NOT pull in memories from old/unrelated conversations
- [ ] Agent stays focused on current thread

### ✅ Normal Queries Still Work
- [ ] Non-meta queries like "What is Python?" still work normally
- [ ] Semantic search still triggers for knowledge questions
- [ ] System hasn't broken other functionality

---

## Debug Commands

If something seems wrong, check these:

### 1. Check if detection is working
```bash
python test_meta_conversational.py
```
Should show all tests passing.

### 2. Check recent conversation logs
```bash
ls -lt conversation_logs/ | head -5
cat conversation_logs/$(ls -t conversation_logs/ | head -1)
```

### 3. Grep for meta-conversational debug logs
```bash
grep -r "meta-conversational\|Meta-conversational" conversation_logs/ | tail -20
```

### 4. Check if the query checker is imported correctly
```python
python -c "from utils.query_checker import is_meta_conversational; print(is_meta_conversational('do you recall that?'))"
```
Should print: `True`

---

## Troubleshooting

### Issue: No debug logs appearing

**Fix**: Set logging level to DEBUG
```bash
export LOG_LEVEL=DEBUG
python main.py cli
```

### Issue: Agent still doesn't recall correctly

**Check**:
1. Verify the query is actually meta-conversational:
   ```python
   from utils.query_checker import is_meta_conversational
   print(is_meta_conversational("your query here"))
   ```

2. Check if memory_coordinator is using the new code:
   ```bash
   grep "_get_meta_conversational_memories" memory/memory_coordinator.py
   ```
   Should show the new method exists.

### Issue: Tests pass but live system still fails

**Possible causes**:
1. Code changes not reflected (restart the process)
2. Cached .pyc files - clear them:
   ```bash
   find . -name "*.pyc" -delete
   find . -name "__pycache__" -type d -exec rm -rf {} +
   ```
3. Different code path being used - check orchestrator imports

---

## Success Criteria

The fix is working correctly if:

1. ✅ When you ask "Do you recall X?", the agent references the actual conversation
2. ✅ Debug logs confirm meta-conversational detection
3. ✅ No hallucinations about unrelated topics
4. ✅ Agent stays on topic and maintains thread continuity
5. ✅ Normal queries (non-meta) still work as expected

---

## Recommended Test Flow (5 minutes)

**Fastest way to verify everything works:**

```bash
# 1. Run unit tests
python test_meta_conversational.py

# 2. Start CLI
python main.py cli

# 3. Quick conversation
You: I had Tuesday off from work
Agent: [responds]

You: Do you recall my day off?
Agent: [should mention Tuesday] ✓

# 4. Check logs
grep "Meta-conversational" conversation_logs/*.txt

# Done! If all 3 steps work, the fix is live.
```
