# Daemon RAG Agent - Project Skeleton

**Purpose**: Compressed architectural overview for LLM context windows. This skeleton captures the essential structure, data flow, and patterns without full implementation details.

**Last Updated**: 2026-03-10

---

## 1. Architecture Overview

```
USER QUERY
    ↓
[Orchestrator] ← Main entry point, coordinates all systems
    ↓
[Tone Detection] → Crisis level detection (harm scoring + semantic + LLM fallback)
    ↓
[Topic Extraction] → spaCy NER + optional LLM fallback
    ↓
[Heavy Topic Check] → Inline fact extraction for crisis/sensitive content
    ↓
[Intent Classification] → Regex-first query intent (9 types, no LLM) [NEW 2026-02-15]
    ↓
[STM Analysis] → Short-term memory pass (topic, intent, tone, threads)
    ↓ (refines low-confidence intent via STM free-text)
    ↓
[Hybrid Memory Retrieval] → Query rewriting + Semantic + Keyword matching
    ↓               ↓ (via MemoryCoordinator → modular components)
    ↓               ├─> MemoryRetriever (parallel ChromaDB queries)
    ↓               ├─> MemoryScorer (ranking with composite scores)
    ↓               ├─> ThreadManager (conversation continuity)
    ↓               └─> ShutdownProcessor (session-end summaries + facts)
    ↓
[Multi-Stage Gating] → FAISS → Cosine → Cross-Encoder
    ↓
[Prompt Building] → Token-budgeted context with separated sections + STM summary
    ↓
[Agentic Search Check] → LLM-first trigger decides if web search needed **[NEW 2026-01]**
    ↓ (if triggered)
    ├─> [AgenticSearchController] → ReAct loop: Think → Search → Observe → Repeat
    ↓ (else normal flow)
[Thinking Block] → LLM generates <thinking>...</thinking> + final answer
    ↓
[LLM Generation] → Multi-provider streaming with Best-of-N/Duel modes
    ↓
[Response Tag Stripping] → Remove <reply>, <response>, <answer>, EOS markers
    ↓
[Citation Extraction] → Extract memory citations (if enabled), remove tags [NEW 2025-12-04]
    ↓
[Memory Storage] → Corpus JSON + Vector embeddings
    ↓               ↓ (via MemoryStorage) [REFACTORED]
    ↓               ├─> CorpusManager (JSON persistence)
    ↓               ├─> ChromaStore (vector embeddings)
    ↓               └─> FactExtractor (optional fact extraction)
    ↓
RESPONSE (thinking stripped) + MEMORY PERSISTENCE
```

**Note**: Memory system fully refactored (Jan 2026). `memory/memory_coordinator.py` is now a thin orchestrator (~498 lines, down from 1,694) delegating to:
- `memory/memory_retriever.py` - Retrieval operations (incl. `get_semantic_top_memories`)
- `memory/memory_storage.py` - Storage operations
- `memory/memory_scorer.py` - Scoring and ranking
- `memory/shutdown_processor.py` - Session-end summaries, facts, procedural skills, code proposals, cross-dedup preview, reflections
- `memory/procedural_skill.py` - ProceduralSkill dataclass + SkillCategory enum
- `memory/cross_deduplicator.py` - Cross-collection dedup (duplicates + fact contradictions, dry-run only on shutdown)
- `memory/dedup_models.py` - Pydantic models: DedupPlan, DuplicatePair, ContradictionCluster
- `memory/thread_manager.py` - Thread tracking
- `memory/memory_interface.py` - Protocol contracts

The incomplete V2 `memory/coordinator.py` has been deleted.

---

## 2. Core Components

### 2.1 core/orchestrator.py (Main Controller)
**Purpose**: Central request handler coordinating all subsystems

**Key Methods**:
- `process_user_query(user_input, files, use_raw_mode, personality)` → Main entry point
  - Detects tone/crisis level via ToneDetector
  - Extracts topics via TopicManager
  - Checks for heavy topics (triggers inline fact extraction)
  - Builds prompt via PromptBuilder
  - Generates response with thinking blocks
  - Parses thinking block and strips from final response
  - Stores interaction back to memory

- `build_context(user_input, files, use_raw_mode, personality)` → Context building **[NEW 2026-01-23]**
  - Uses ContextPipeline for clean separation of concerns
  - Returns ContextResult with all processed context
  - Preferred method for new code (see section 2.1.2)

- `prepare_prompt(user_input, files, use_raw_mode)` → Prompt preparation (legacy)
  - Tone detection (HIGH/MEDIUM/CONCERN/CONVERSATIONAL)
  - File processing (PDF/DOCX/CSV)
  - Heavy topic inline fact extraction
  - Optional query rewrite for retrieval
  - System prompt resolution with tone instructions
  - Thread context injection
  - **Deprecated**: Use `build_context()` + `prompt_builder.build_prompt_from_context()` instead

- `_get_tone_instructions(tone_level)` → Mode-specific response guidelines
- `_get_session_headers_instructions()` → Temporal reasoning guidance
- `_extract_citations(response, memory_map)` → Extract memory citations, remove tags **[NEW 2025-12-04]**

**Data Flow Pattern**:
```python
query → detect_crisis_level() → extract_topics()
→ check_heavy_topic() → [inline_fact_extraction if heavy]
→ prepare_prompt(query, tone_instructions, thread_context)
→ generate_response(prompt) → parse_thinking_block()
→ strip_reflections() → store_interaction(query, final_answer)
→ return final_answer
```

**Tone Modes** (injected into system prompt):
- `CRISIS_SUPPORT`: Full therapeutic mode, multi-paragraph, resources
- `ELEVATED_SUPPORT`: 2-3 paragraphs, empathetic validation
- `LIGHT_SUPPORT`: 2-4 sentences, brief acknowledgment
- `CONVERSATIONAL`: Max 3 sentences, direct, no fluff

**Dependencies**: MemoryCoordinator, PromptBuilder, ResponseGenerator, TopicManager, ToneDetector, QueryChecker, STMAnalyzer, ResponseParser

---

### 2.1.1 core/response_parser.py (Response Parsing Utilities) **[NEW 2026-01-23]**
**Purpose**: Pure utility functions for parsing and cleaning LLM responses

**Module Contract**:
```
Purpose: Parse and clean LLM responses (thinking blocks, reflections, XML wrappers, artifacts)
Inputs: Raw response strings from LLM
Outputs: Cleaned response strings
Side effects: None (pure functions)
```

**Key Methods** (all static):
- `parse_thinking_block(response)` → Tuple[str, str] - Extract thinking content and final answer
- `strip_reflection_blocks(response)` → str - Remove `<reflect>`, `[SYSTEM QUALITY REFLECTION]` blocks
- `strip_xml_wrappers(text)` → str - Remove `<result>`, `<answer>`, `<final>` wrappers
- `strip_prompt_artifacts(text)` → str - Remove echoed prompt headers like `[TIME CONTEXT]`, `[FACTS]`

**Usage**:
```python
from core.response_parser import ResponseParser

# Parse thinking block from LLM response
thinking, answer = ResponseParser.parse_thinking_block(llm_response)

# Clean response before storage
cleaned = ResponseParser.strip_reflection_blocks(answer)
cleaned = ResponseParser.strip_xml_wrappers(cleaned)
cleaned = ResponseParser.strip_prompt_artifacts(cleaned)
```

**Consumers**: orchestrator.py, response_generator.py, gui/handlers.py, prompt/formatter.py

---

### 2.1.2 core/context_pipeline.py (Context Builder) **[NEW 2026-01-23]**
**Purpose**: Builder pattern for transforming raw user input into processed context (pre-retrieval query analysis)

**Module Contract**:
```
Purpose: Build context from user input through a multi-stage pipeline
Inputs: User query, optional files, configuration flags
Outputs: ContextResult with all context components
Side effects: May call LLM for tone detection, query rewriting, STM analysis
```

**SCOPE**: Query Analysis ONLY (pre-retrieval). Does NOT handle:
- Memory retrieval → That's MemoryCoordinator's job
- Prompt assembly → That's PromptBuilder's job
- LLM generation → That's ResponseGenerator's job

**Pipeline Stages**:
1. **Topic Extraction** - Extract topics via TopicManager
2. **Tone Detection** - Detect emotional state via analyze_emotional_context
3. **File Processing** - Extract text from PDF/DOCX/CSV via FileProcessor
4. **Heavy Topic Check** - Check for sensitive content via QueryChecker
4.5. **Intent Classification** - Regex-first query intent via IntentClassifier (no LLM) **[NEW 2026-02-15]**
5. **Query Rewriting** - Rewrite for better retrieval (LLM)
6. **STM Analysis** - Analyze recent conversation via STMAnalyzer
   - 6b. **STM Intent Refinement** - Low-confidence intents refined by STM free-text intent **[NEW 2026-02-15]**
7. **Identity Injection** - Add user identity context via UserProfile
8. **Thread Context** - Get active thread via memory_system

**Key Classes**:
- `ToneLevel` - Enum mapping crisis levels (CRISIS/ELEVATED/CONCERN/CONVERSATIONAL)
- `ContextResult` - Immutable dataclass with all processed context
- `ContextPipeline` - Main builder class
- `IntentResult` - Intent classification output (carried on ContextResult) **[NEW 2026-02-15]**

**ContextResult Fields**:
```python
@dataclass
class ContextResult:
    processed_query: str          # Original or rewritten query
    original_query: str           # Always preserved
    tone_level: ToneLevel         # Detected tone
    tone_instructions: str        # Mode-specific guidelines
    topics: List[str]             # Extracted topics
    primary_topic: Optional[str]  # Main topic
    file_context: Optional[str]   # Processed file content
    thread_context: Optional[Dict]# Active thread metadata
    stm_summary: Optional[Dict]   # STM analysis result
    identity_block: str           # User identity context
    is_heavy_topic: bool          # Heavy topic flag
    extracted_facts: List[Dict]   # Inline facts if heavy
    intent: Optional[IntentResult]# Intent classification result [NEW 2026-02-15]
    metadata: Dict[str, Any]      # Additional context
```

**Usage**:
```python
# In orchestrator - preferred new flow
context = await self.build_context(user_input, files, use_raw_mode)

# Use context for downstream processing
# context.processed_query → for memory retrieval
# context.tone_instructions → for system prompt
# context.topics → for relevance filtering

# Or use with PromptBuilder directly
prompt_ctx = await prompt_builder.build_prompt_from_context(context)
```

**Clean Data Flow**:
```
ContextPipeline.build()     →  ContextResult
                                    ↓
MemoryCoordinator.get_memories(context.processed_query, context.topics)  →  memories
                                    ↓
PromptBuilder.build_prompt(context, memories)  →  final prompt
```

**Dependencies**: TopicManager, ToneDetector (analyze_emotional_context), FileProcessor, QueryChecker, STMAnalyzer, UserProfile, memory_system

**Consumers**: orchestrator.py (build_context method), prompt/builder.py (build_prompt_from_context)

---

### 2.2 core/stm_analyzer.py (Short-Term Memory Analyzer) **[NEW]**
**Purpose**: Lightweight LLM pass to analyze recent conversation context and generate structured summaries

**Key Methods**:
- `analyze(recent_memories, user_query, last_assistant_response)` → Dict[str, Any]
  - Formats recent conversation history (last 10 lines, truncated to 200 chars each)
  - Calls LLM (gpt-4o-mini) with low temperature (0.3) for structured JSON output
  - Returns parsed JSON with: topic, user_question, intent, tone, open_threads, constraints
  - Fallback to empty summary on failures

- `_format_memories(memories)` → Formatted conversation string
- `_parse_json(raw)` → Robust JSON parser (handles markdown wrapping, validates fields)
- `_empty_summary()` → Fallback for failures

**Activation Logic** (in orchestrator):
- Requires `STM_MIN_CONVERSATION_DEPTH` (default: 3) conversation turns
- Skips queries < 10 characters
- Uses last `STM_MAX_RECENT_MESSAGES` (default: 8) for analysis

**Output Structure**:
```python
{
  "topic": "Python debugging",                    # 2-5 words
  "user_question": "How to fix timeout error",    # One sentence
  "intent": "Get practical solution",             # One sentence
  "tone": "frustrated",                           # Single word
  "open_threads": ["Performance", "Error handling"], # List
  "constraints": ["Standard library only"]        # List
}
```

**Integration**:
- Injected into prompt RIGHT BEFORE `[CURRENT USER QUERY]` (maximum attention window)
- Token manager preserves STM (priority=10, no token cost - metadata only)
- Passed through: orchestrator → prompt_builder.build_prompt() → formatter._assemble_prompt()

**Configuration**:
- `USE_STM_PASS`: Enable/disable STM system (default: True)
- `STM_MODEL_NAME`: Model for analysis (default: "gpt-4o-mini")
- `STM_MAX_RECENT_MESSAGES`: Messages to analyze (default: 8)
- `STM_MIN_CONVERSATION_DEPTH`: Minimum depth to trigger (default: 3)

**Dependencies**: ModelManager

---

### 2.3 memory/memory_coordinator.py (Thin Orchestrator)
**Purpose**: Unified interface for all memory operations — thin delegation layer (~498 lines)

**Status**: Fully refactored (Jan 2026). All inline logic extracted to modular components. Acts as state-syncing orchestrator with ~24 delegation methods.

**Architecture**:
```python
MemoryCoordinator (memory_coordinator.py, ~498 lines)
    ↓
├── MemoryRetriever (memory_retriever.py) - Retrieval + semantic top memories
├── MemoryScorer (memory_scorer.py) - Scoring and ranking
├── MemoryStorage (memory_storage.py) - Persistence operations
├── ShutdownProcessor (shutdown_processor.py) - Session-end processing
├── ThreadManager (thread_manager.py) - Conversation thread tracking
├── HybridRetriever (hybrid_retriever.py) - Query rewrite + keyword
└── UserProfile (user_profile.py) - User fact profile
```

**5 Memory Types** (MemoryType enum):
1. `EPISODIC` - Raw conversation turns (query/response pairs)
2. `SEMANTIC` - Extracted facts and entities
3. `PROCEDURAL` - Learned patterns and behaviors
4. `SUMMARY` - Compressed conversation blocks
5. `META` - Reflections and meta-patterns

**Key Methods** (all delegate to components):
- `get_memories(query, limit, topics)` → delegates to `self._retriever.get_memories()`
- `get_semantic_top_memories(query, limit)` → delegates to `self._retriever.get_semantic_top_memories()`
- `store_interaction(query, response, tags)` → delegates to `self._storage.store_interaction()`
- `process_shutdown_memory(session_conversations)` → delegates to `self._shutdown.process_shutdown_memory()`
- `run_shutdown_reflection(session_conversations, session_summaries)` → delegates to `self._shutdown.run_shutdown_reflection()`
- `store_skill(skill)` → delegates to `self._storage.store_skill()`
- `get_skills(query, limit)` → delegates to `self._retriever.get_skills()`
- `detect_or_create_thread(query, is_heavy)` → delegates to `self.thread_manager`
- `get_thread_context()` → delegates to `self.thread_manager`

**Hierarchical Memory**:
- ⚠️ Parent-child relationships: **DISABLED** (caused retrieval bugs, intentionally deactivated)
- Temporal decay: `1.0 / (1.0 + decay_rate * age_hours)`
- Access count boosts truth score

**Init**: Passes `memory_coordinator=self` to ShutdownProcessor, enabling pipeline-enriched proposal generation **[ENHANCED 2026-02-09]**

**Dependencies**: MemoryRetriever, MemoryStorage, MemoryScorer, ShutdownProcessor, ThreadManager, HybridRetriever, CorpusManager, MultiCollectionChromaStore, MultiStageGateSystem, MemoryConsolidator, UserProfile

---

### 2.3.0 memory/memory_scorer.py **[NEW - REFACTORED]**
**Purpose**: Memory scoring and ranking operations with multi-factor composite scoring

**Key Methods** (implements MemoryScorerProtocol):
- `calculate_truth_score(query, response)` → Calculate truth/reliability score (0.0-1.0)
- `calculate_importance_score(content)` → Calculate importance based on content analysis (0.0-1.0)
- `rank_memories(memories, query, current_topic=None, is_meta_conversational=False, weight_overrides=None)` → Rank memories by composite score for given query. `weight_overrides` dict overrides global SCORE_WEIGHTS per call (used by intent classifier). **[UPDATED 2026-02-17]**
- `update_truth_scores_on_access(memories)` → No-op (legacy, truth now evidence-based via TruthScorer)

**Instance Attributes**:
- `_intent_weight_overrides: Optional[Dict[str, float]]` — Set/cleared by PromptBuilder around gather. Fallback when `rank_memories()` called without explicit `weight_overrides` parameter (deep in retrieval chain). **[NEW 2026-02-15]**

**Additional Methods**:
- `apply_temporal_decay(memories)` → Apply time-based decay to memory scores
- `_calculate_topic_match(memory, current_topic)` → **[NEW]** Score topic alignment (1.0 exact, 0.5 neutral, 0.2 different)
- `_calculate_size_penalty(memory)` → **[NEW]** Penalize large documents without keyword relevance (scaled: -0.25 × size_multiplier, capped at -1.0)

**Scoring Algorithm** (in rank_memories):
```python
final_score = (
    SCORE_WEIGHTS['relevance'] * (relevance + collection_boost) +
    SCORE_WEIGHTS['recency'] * recency_decay +
    SCORE_WEIGHTS['truth'] * truth_score +
    SCORE_WEIGHTS['importance'] * importance_score +
    SCORE_WEIGHTS['continuity'] * continuity_score +
    SCORE_WEIGHTS['topic_match'] * topic_match +  # NEW
    structural_alignment +
    anchor_bonus +
    meta_bonus +  # NEW - for meta-conversational queries
    penalties +
    size_penalty  # NEW - scaled document size penalty
)
```

**Temporal-Aware Recency Decay** **[NEW 2026-02-17]**:
When `_temporal_anchor_hours` is present in weight_overrides (set by IntentClassifier for TEMPORAL_RECALL), the standard exponential decay curve is replaced with a temporal-aware curve:
```python
# Standard decay (all other intents):
recency = 1.0 / (1.0 + RECENCY_DECAY_RATE * age_hours)

# Temporal-aware decay (TEMPORAL_RECALL with anchor window):
if age_hours <= temporal_anchor:
    # Gentle decay within referenced window (1.0 → 0.7)
    recency = 1.0 - (age_hours / temporal_anchor) * 0.3
else:
    # Standard decay from 0.7 baseline for memories outside window
    hours_past = age_hours - temporal_anchor
    recency = 0.7 / (1.0 + RECENCY_DECAY_RATE * hours_past)
```
This ensures memories from the referenced time period (e.g., "last week" → 168h window) score 0.70–1.00 instead of being buried at 0.10 by standard decay. The `_temporal_anchor_hours` key is popped from weights before scoring (not a real scoring weight). Temporal anchor takes priority over both `time_manager.calculate_active_day_decay()` and the standard fallback.

**Configuration Constants**:
```python
# Size Penalty (for large documents)
LARGE_DOC_SIZE_THRESHOLD = 10000      # 10KB threshold
LARGE_DOC_KEYWORD_THRESHOLD = 0.3     # Min keyword_score to avoid penalty
LARGE_DOC_BASE_PENALTY = -0.25        # Base penalty, scaled by size multiplier
```

**Meta-Conversational Bonuses** (when is_meta_conversational=True):
- EPISODIC memories: +0.15
- SUMMARY memories: +0.10
- META memories: +0.12
- SEMANTIC memories: +0.00 (no bonus)

**Heuristics**:
- Deictic follow-up detection (keywords: "that", "it", "this", "explain", "again")
- Salient token extraction (removes stopwords, frequency-based)
- Math density calculation (numbers and operators)
- Analogy marker detection ("it's like", "imagine", etc.)
- Anchor token building from recent conversation context
- Topic match scoring (checks topic tags and metadata)

**Dependencies**: TimeManager (optional)

---

### 2.3.1 memory/shutdown_processor.py **[NEW - EXTRACTED 2026-01-25, ENHANCED 2026-02-09]**
**Purpose**: Session-end processing — block summaries, fact extraction, procedural skill extraction, code proposal generation (pipeline-enriched), user profile updates, reflections

**Key Methods**:
- `async process_shutdown_memory(session_conversations=None)` → Main shutdown pipeline:
  ```python
  1. Gather unsummarized conversations (from session or corpus)
  2. Create block summaries (consolidator) for every N turns
  3. Store summaries in ChromaDB (summaries collection)
  4. Extract facts via LLM (LLMFactExtractor) + regex (FactExtractor)
  5. Extract procedural skills via LLM (0-3 per session)
  6. Generate code proposals via GoalDirectedGenerator (0-5 per session) [ENHANCED 2026-02-09]
  7. Cross-collection dedup preview (dry_run=True only, NEVER auto-deletes) [NEW 2026-02-13]
  8. Update UserProfile with extracted facts
  9. Log statistics
  ```

- `async run_shutdown_reflection(session_conversations=None, session_summaries=None)` → End-of-session LLM reflection:
  ```python
  1. Check REFLECTIONS_ENABLED env var
  2. Gather recent conversations + summaries
  3. Build reflection prompt (what patterns, what learned, what to remember)
  4. Call LLM to generate reflection text
  5. Store reflection in ChromaDB (reflections collection)
  6. Return success/failure
  ```

**Configuration** (env vars):
- `REFLECTIONS_ENABLED` — Enable/disable reflections (default: True)
- `REFLECTION_MAX_TOKENS` — Max tokens for reflection (default: 500)
- `REFLECTION_MODEL` — Model alias (default: gpt-4o-mini)
- `SHUTDOWN_SUMMARY_BLOCK_SIZE` — Conversations per summary block (default: 10)
- `SHUTDOWN_FACT_EXTRACTION` — Enable fact extraction (default: True)

**Additional Methods**:
- `async _extract_procedural_skills(session_conversations)` → LLM-based pattern extraction:
  - Identifies generalizable WHEN/THEN problem-solving patterns
  - Categories: debugging, workflow, prompt_engineering, interpersonal, architectural, optimization, testing
  - Stores via MemoryStorage.store_skill() with semantic deduplication

- `async _generate_proposals(session_conversations)` → Goal-directed code proposal generation **[ENHANCED 2026-02-09]**:
  - Guards: CODE_PROPOSALS_ENABLED, model_manager with generate_once, ≥3 session items
  - Creates GoalDirectedGenerator and ProposalStore instances
  - **Pipeline-enriched path** (when memory_coordinator available):
    - Calls `_gather_proposal_context()` for rich semantic context
    - Routes through `generate_proposals_with_context()` for better awareness
    - Falls back to cold generation on failure
  - **Cold fallback** (`_generate_proposals_cold()`):
    - Builds extra_context from last 8 session turns (truncated to 500 chars each)
    - Includes existing proposals via get_for_dedup() for dedup context
  - Checks similarity before storing each proposal (CODE_PROPOSALS_DEDUP_THRESHOLD)
  - Logs count of stored proposals

- `async _generate_proposals_cold(sess_items, generator, dedup_context)` → Original cold generation **[NEW 2026-02-09]**:
  - Truncated conversation excerpts + file reads only (no retrieval pipeline)

- `async _gather_proposal_context(sess_items)` → Rich context assembly **[NEW 2026-02-09]**:
  - Builds synthetic query from session topics
  - Parallel retrieval via MemoryCoordinator: memories (5), skills (3), facts (5), reflections (2)
  - Sync retrieval: summaries (3), git commits (5), user profile (300 tokens)
  - Returns structured context sections: Recent Conversation, Relevant Memories, Session Summaries, Past Reflections, Problem-Solving Patterns, Known User Facts, User Profile, Recent Git Activity

**Constructor**: `__init__(..., memory_coordinator=None)` — accepts MemoryCoordinator for pipeline-enriched generation **[ENHANCED 2026-02-09]**

**Dependencies**: CorpusManager, MultiCollectionChromaStore, MemoryConsolidator, LLMFactExtractor, FactExtractor, ModelManager, UserProfile, MemoryScorer, TimeManager, ProceduralSkill, GoalDirectedGenerator, ProposalStore, MemoryCoordinator (optional)

---

### 2.3.1a memory/cross_deduplicator.py + dedup_models.py (Cross-Collection Dedup) **[NEW 2026-02-13]**
**Purpose**: Unified deduplication across ChromaDB collections — detects near-duplicate documents and resolves fact contradictions (same subject+predicate, different object).

**Data Models** (`memory/dedup_models.py`):
```python
class DedupAction(str, Enum):   # KEEP, DELETE, MERGE, SUPERSEDE
class DedupReason(str, Enum):   # CROSS_DUPLICATE, WITHIN_DUPLICATE, FACT_CONTRADICTION, SUBSUMED
class DuplicatePair(BaseModel):  # doc_id_a/b, collection_a/b, similarity, keep_id, delete_id
class ContradictionCluster(BaseModel):  # subject, predicate, entries[], keep_id, delete_ids[]
class DedupPlan(BaseModel):      # duplicate_pairs[], contradiction_clusters[], stats, to_markdown()
```

**Key Methods** (`memory/cross_deduplicator.py`):
- `run(dry_run=True) -> DedupPlan` → Full dedup pass:
  ```python
  1. Load documents from target collections (via chroma_store.list_all)
  2. Compute embeddings (chroma_store.embedding_fn), L2-normalize
  3. Pairwise cosine similarity → find duplicates >= 0.92 threshold
  4. Group facts by (subject, predicate) → find contradictions (differing objects)
  5. Skip ephemeral predicates (from PROFILE_EPHEMERAL_RELATIONS config)
  6. If not dry_run: execute deletions via collection.delete(ids=[...])
  ```

- `_find_cross_duplicates(docs, embeddings) -> List[DuplicatePair]` → Pairwise similarity scan
  - Cascade prevention: `marked_for_delete` set prevents chain deletions
  - Ephemeral fact pairs skipped via `_is_ephemeral_fact_pair()`
  - Float clamping: `score = min(float(sim_matrix[i, j]), 1.0)` for fp32 overshoot

- `_find_fact_contradictions(fact_docs) -> List[ContradictionCluster]` → Subject+predicate grouping
  - Skips predicates in `PROFILE_EPHEMERAL_RELATIONS` (current_feeling, is, has, thinks, etc.)
  - Keeps most recent entry per group (by timestamp)

- `_pick_keep_delete(doc_a, doc_b) -> (keep, delete)` → Priority-based selection
  - Collection priority: summaries(5) > reflections(4) > skills(3) > proposals(2) > facts(1)
  - Same priority: keep more recent by timestamp

- `_extract_triple(doc) -> (subject, predicate, object)` → Triple extraction
  - Tries metadata fields first (subject/predicate/object or entity/relation/value)
  - Falls back to content parsing: pipe-separated, dash-separated, natural language patterns

- `_is_ephemeral_fact_pair(doc_a, doc_b, ephemeral) -> bool` → Ephemeral guard
  - Both docs must be from "facts" collection
  - Checks metadata then falls back to `_extract_triple()` content parsing

**Protected Collections** (never scanned):
- `conversations`, `obsidian_notes`, `reference_docs`, `wiki_knowledge`

**Target Collections** (configurable):
- `facts`, `summaries`, `procedural_skills`, `proposals`, `reflections`

**Integration Points**:
- Shutdown: `ShutdownProcessor._run_cross_collection_dedup()` — dry_run=True only, class-level `_dedup_ran` guard prevents double-run
- GUI: Status tab → "Preview Dedup" (dry_run=True) / "Run Dedup" (dry_run=False) buttons

**Configuration** (`config/app_config.py`):
```python
CROSS_DEDUP_ENABLED = True
CROSS_DEDUP_DUPLICATE_THRESHOLD = 0.92      # Cosine similarity for duplicates
CROSS_DEDUP_CONTRADICTION_THRESHOLD = 0.85  # Currently unused (reserved)
CROSS_DEDUP_MAX_DOCS_PER_COLLECTION = 1000  # Per-collection cap (most recent kept)
CROSS_DEDUP_ON_SHUTDOWN = True              # Enable shutdown preview (dry_run only)
CROSS_DEDUP_COLLECTIONS = ["facts", "summaries", "procedural_skills", "proposals", "reflections"]
```

**Tests**: `tests/unit/test_cross_deduplicator.py` — 47 tests across 11 classes

---

### 2.1.3 core/escalation_tracker.py (Crisis Cooldown) **[NEW 2026-02-13]**
**Purpose**: Session-level emotional momentum tracker for adaptive tone de-escalation. Prevents the "therapeutic echo chamber" problem where identical validating responses repeat during user spiraling.

**Key Types**:
```python
class ResponseStrategy(Enum):
    VALIDATE_AND_SUGGEST   # Default: validate feelings, offer suggestions
    GROUNDING_PRESENCE     # Sustained escalation: drop advice, pure acknowledgment (2-3 sentences)
    QUIET_COMPANIONSHIP    # Sustained + ignored suggestions: minimal presence (1-2 sentences)
    GENTLE_REENGAGEMENT    # After de-escalation: carefully re-introduce engagement
```

**Key Methods**:
- `update(tone_level, user_message, need_type=None) -> ResponseStrategy` → Main update loop:
  - Tracks consecutive elevated/calm message counts
  - Detects user engagement with previous suggestions (keyword overlap + engagement phrases)
  - Computes strategy based on accumulated signals
  - Stores need_type for de-escalation nuance

- `record_response(response) -> None` → Extract suggestions from assistant response for engagement tracking on next turn

- `get_strategy_instructions() -> str` → Supplemental tone instructions appended to system prompt:
  - GROUNDING_PRESENCE: "Max 2-3 sentences. No advice. Pure acknowledgment."
  - QUIET_COMPANIONSHIP: "Max 1-2 sentences. Just be present."
  - GENTLE_REENGAGEMENT: "Acknowledge shift warmly. May offer ONE small suggestion."
  - VALIDATE_AND_SUGGEST: empty string (use standard tone instructions)

- `get_token_budget_override() -> Optional[int]` → Token budget for brevity enforcement:
  - QUIET_COMPANIONSHIP: 300 tokens
  - GROUNDING_PRESENCE: 500 tokens
  - GENTLE_REENGAGEMENT: 800 tokens
  - VALIDATE_AND_SUGGEST: None (default budget)

- `get_escalation_velocity() -> float` → 0.0 (stable) to 1.0 (rapid escalation) from last 5 tone values

- `get_debug_info() -> Dict` → Strategy, counts, velocity for debug logging

**Strategy Transitions**:
```
VALIDATE_AND_SUGGEST  (default, < threshold consecutive elevated)
    ↓ (threshold+ consecutive elevated)
GROUNDING_PRESENCE    (drop suggestions, pure acknowledgment)
    ↓ (2+ ignored suggestions while elevated)
QUIET_COMPANIONSHIP   (minimal presence, 1-2 sentences max)
    ↓ (tone drops to CONCERN/CONVERSATIONAL)
GENTLE_REENGAGEMENT   (carefully re-introduce engagement)
    ↓ (sustained calm past deescalation_window)
VALIDATE_AND_SUGGEST  (back to normal)
```

**De-escalation Nuance**: When `need_type == "PERSPECTIVE"`, user is shifting to analytical mode (intensity shift, not genuine calming) → goes to VALIDATE_AND_SUGGEST instead of GENTLE_REENGAGEMENT.

**Integration** (in `core/orchestrator.py`):
```python
# __init__: self.escalation_tracker = EscalationTracker()
# After build_context(): tracker.update(tone_level, user_message)
# Build prompt: append tracker.get_strategy_instructions() to system prompt
# Token budget: tracker.get_token_budget_override() overrides tone-based budget
# After generation: tracker.record_response(response_text)
```

**Configuration** (`config/app_config.py`):
```python
ESCALATION_ENABLED = True
ESCALATION_THRESHOLD = 3          # Consecutive elevated before strategy shift
ESCALATION_DEESCALATION_WINDOW = 2  # Consecutive calm before gentle re-engagement ends
```

**Tests**: `tests/unit/test_escalation_tracker.py` — 51 tests

---

### 2.1.4 core/intent_classifier.py (Query Intent Classifier) **[NEW 2026-02-15]**
**Purpose**: Fast, regex-first classification of user query intent. No LLM calls. Produces per-intent weight overrides, retrieval count overrides, and gate threshold overrides that tune downstream memory retrieval and scoring.

**Module Contract**:
```
Purpose: Classify user query into 1 of 9 intent types via regex patterns
Inputs: query (str), optional tone_level (str)
Outputs: IntentResult with intent type, confidence, weight/retrieval/gate overrides
Side effects: None (pure computation, no LLM calls)
```

**Key Types**:
```python
class IntentType(str, Enum):
    FACTUAL_RECALL       # "What's my dog's name?"
    TEMPORAL_RECALL      # "What happened last week?"
    EMOTIONAL_SUPPORT    # "I feel so sad"
    CASUAL_SOCIAL        # "hey", "thanks", "bye"
    TECHNICAL_HELP       # "How do I fix this bug?"
    CREATIVE_EXPLORATION # "Let's brainstorm ideas"
    META_CONVERSATIONAL  # "What do you know about me?"
    PROJECT_WORK         # "Add a feature for notifications"
    GENERAL              # Fallback (no strong pattern match)

@dataclass
class IntentResult:
    intent: IntentType
    confidence: float              # 0.0 – 1.0
    source: str = "regex"          # "regex" | "stm_refined"
    weight_overrides: Dict[str, float]   # Override SCORE_WEIGHTS per intent
    retrieval_overrides: Dict[str, int]  # Override PROMPT_MAX_* per intent
    gate_threshold_override: Optional[float]  # Override gate threshold
```

**Key Methods**:
- `classify(query, tone_level=None) -> IntentResult` → Regex patterns checked in order, highest confidence wins. Tone bias: HIGH/MEDIUM tone biases toward EMOTIONAL_SUPPORT for ambiguous queries. For TEMPORAL_RECALL, extracts temporal anchor via `extract_temporal_window()` and adds `_temporal_anchor_hours` to weight_overrides. **[UPDATED 2026-02-17]**
- `refine_with_stm(result, stm_intent) -> IntentResult` → Upgrades low-confidence results (< 0.50) using STM's free-text intent via keyword matching. No extra LLM call.

**Temporal Anchor Extraction** **[NEW 2026-02-17]**:
When TEMPORAL_RECALL is classified, `classify()` calls `extract_temporal_window()` from `utils/query_checker.py` to convert temporal phrases ("last week" → 7 days, "yesterday" → 1 day, "last month" → 30 days) into hours. The value is added as `_temporal_anchor_hours` in `weight_overrides`, flowing automatically through the existing builder → scorer pipeline. `MemoryScorer.rank_memories()` pops this key and uses it to reshape the recency decay curve (see Section 2.3.0).

**Per-Intent Profiles** (`_PROFILES` dict):
Each intent type defines:
- `weights`: Override dict for SCORE_WEIGHTS (e.g., FACTUAL_RECALL boosts truth to 0.30)
- `retrieval`: Override dict for PROMPT_MAX_* counts (e.g., CASUAL_SOCIAL reduces max_mems to 3)
- `gate`: Override for gate threshold (e.g., CASUAL_SOCIAL raises to 0.65)

**Integration Points**:
1. **ContextPipeline** Stage 4.5: `intent_classifier.classify()` runs after heavy topic check
2. **ContextPipeline** Stage 6b: `intent_classifier.refine_with_stm()` runs after STM analysis
3. **ContextResult.intent**: Carries IntentResult downstream
4. **UnifiedPromptBuilder.build_prompt_from_context()**: Extracts retrieval_overrides and weight_overrides
5. **UnifiedPromptBuilder.build_prompt()**: Applies retrieval overrides to all max_* counts, sets weight overrides on scorer
6. **MemoryScorer.rank_memories()**: Uses weight_overrides to tune scoring formula per intent; pops `_temporal_anchor_hours` for temporal-aware decay

**Configuration** (`config/app_config.py`):
```python
INTENT_ENABLED = True                    # Feature toggle
INTENT_STM_REFINEMENT_THRESHOLD = 0.50   # Below this confidence, STM can refine
```

**Tests**: `tests/unit/test_intent_classifier.py` — 74 tests

---

### 2.3.2 memory/memory_retriever.py **[REFACTORED]**
**Purpose**: Memory retrieval operations (~966 lines)

**Key Methods** (implements MemoryRetrieverProtocol):
- `async get_memories(query, limit, topic_filter)` → Main retrieval pipeline
- `async get_semantic_top_memories(query, limit)` → **[MIGRATED 2026-01-25]** Top semantic memories with meta-conversational routing and gating
- `async get_facts(query, limit)` → Retrieve semantic facts
- `async get_recent_facts(limit)` → Retrieve most recent facts
- `async get_reflections(limit)` → Retrieve recent reflections
- `async get_reflections_hybrid(query, limit)` → Hybrid recent+semantic reflections
- `get_summaries(limit)` → Retrieve recent summaries (sync)
- `get_summaries_hybrid(query, limit)` → Hybrid recent+semantic summaries (sync)
- `get_dreams(limit)` → Retrieve dream memories
- `async search_by_type(type_name, query, limit)` → Search by memory type

**Helper Methods**:
- `_get_recent_conversations(k)` → Fetch recent conversations from corpus
- `async _get_semantic_memories(query, limit)` → Semantic search via ChromaDB
- `async _fallback_semantic_search(query, limit)` → Fallback when primary fails
- `async _combine_memories(recent, semantic)` → Merge and deduplicate results
- `async get_skills(query, limit)` → Hybrid retrieval (1/3 recent + 2/3 semantic) from procedural_skills collection
- `async _gate_memories(query, memories, threshold)` → Apply gating filter
- `_get_memory_key(memory)` → Generate unique key for deduplication
- `_parse_result(item, source)` → Parse ChromaDB result to standard format

**Retrieval Pipeline**:
```python
1. Get recent conversations from corpus
2. Query ChromaDB collections in parallel
3. Combine recent + semantic results
4. Apply multi-stage gating
5. Rank with MemoryScorer
6. Deduplicate by memory key
7. Return top K
```

**Dependencies**: CorpusManager, MultiCollectionChromaStore, MultiStageGateSystem, MemoryScorer, HybridRetriever

---

### 2.3.3 memory/memory_storage.py **[NEW - REFACTORED]**
**Purpose**: Memory persistence operations extracted from coordinator

**Key Methods** (implements MemoryStorageProtocol):
- `async store_interaction(query, response, tags)` → Persist to corpus + Chroma
- `async add_reflection(text, tags, source, timestamp)` → Store reflection memory
- `async extract_and_store_facts(query, response, truth_score)` → Fact extraction and storage
- `async consolidate_and_store_summary()` → Generate and store conversation summary
- `async store_skill(skill)` → Store ProceduralSkill with semantic deduplication (SKILL_DEDUP_THRESHOLD=0.85)

**Helper Methods**:
- `_calculate_truth_score(query, response)` → Calculate truth score via scorer or fallback
- `_calculate_importance_score(content)` → Calculate importance via scorer or fallback
- `_now()`, `_now_iso()` → Time utilities

**Storage Flow**:
```python
1. Create memory dict with metadata
2. Calculate truth/importance scores (via MemoryScorer)
3. Add to corpus JSON
4. Add to ChromaDB with embeddings
5. Optional: Extract facts if configured
```

**Configuration**:
- `FACTS_EXTRACT_EACH_TURN`: Extract facts every turn (default: False)
- `SUMMARIZE_AT_SHUTDOWN_ONLY`: Defer summarization (default: True)

**Dependencies**: CorpusManager, MultiCollectionChromaStore, FactExtractor, MemoryConsolidator, MemoryScorer, MemoryCoordinator (optional)

---

### 2.3.4 memory/thread_manager.py **[NEW - REFACTORED]**
**Purpose**: Conversation thread tracking extracted from coordinator

**Key Methods** (implements ThreadManagerProtocol):
- `get_thread_context()` → Retrieve current thread metadata if active
- `detect_or_create_thread(query, is_heavy)` → Detect existing thread or create new one
- `detect_topic_for_query(query)` → Extract topic from specific query

**Helper Methods**:
- `_now()`, `_now_iso()` → Time utilities

**Thread Detection Factors**:
- Keyword overlap between queries
- Time proximity (<10 minutes)
- Both heavy topics (crisis continuity)
- Same topic match
- Threshold: `THREAD_CONTINUITY_THRESHOLD = 0.5`

**Thread Context Returns**:
```python
{
  "thread_id": str,
  "thread_depth": int,
  "thread_started": datetime,
  "thread_topic": str,
  "is_heavy_topic": bool
}
```

**Dependencies**: CorpusManager, TopicManager, TimeManager

---

### 2.3.5 memory/memory_interface.py **[NEW - REFACTORED]**
**Purpose**: Protocol contracts for memory components

**Protocols Defined**:
- `MemoryScorerProtocol` - Interface for scoring and ranking operations
  - `calculate_truth_score(query, response) -> float`
  - `calculate_importance_score(content) -> float`
  - `rank_memories(memories, query) -> List[Dict]`
  - `update_truth_scores_on_access(memories) -> None`

- `MemoryStorageProtocol` - Interface for storage/persistence operations
  - `async store_interaction(query, response, tags) -> None`
  - `async add_reflection(text, tags, source, timestamp) -> bool`
  - `async extract_and_store_facts(query, response, truth_score) -> None`

- `MemoryRetrieverProtocol` - Interface for retrieval operations
  - `async get_memories(query, limit, topic_filter) -> List[Dict]`
  - `async get_facts(query, limit) -> List[Dict]`
  - `async get_recent_facts(limit) -> List[Dict]`
  - `async get_reflections(limit) -> List[Dict]`
  - `async get_reflections_hybrid(query, limit) -> List[Dict]`
  - `get_summaries(limit) -> List[Dict]` (sync)
  - `get_summaries_hybrid(query, limit) -> List[Dict]` (sync)
  - `get_dreams(limit) -> List[Dict]`
  - `async search_by_type(type_name, query, limit) -> List[Dict]`

- `ThreadManagerProtocol` - Interface for thread tracking
  - `get_thread_context() -> Optional[Dict]`
  - `detect_or_create_thread(query, is_heavy) -> Dict`
  - `detect_topic_for_query(query) -> str`

- `MemoryConsolidatorProtocol` - Interface for memory consolidation
  - `async process_shutdown_memory(session_conversations) -> None`
  - `async run_shutdown_reflection(session_conversations, model_manager) -> Optional[str]`
  - `async consolidate_and_store_summary() -> None`

**Benefits**:
- Type safety with Protocol typing
- Clear contracts for component interactions
- Enables mock implementations for testing
- All implementations verified compliant as of 2025-11-30

---

### 2.3.6 memory/user_profile.py & user_profile_schema.py (User Profile System) **[NEW 2025-12-01, ENHANCED 2026-02-13]**
**Purpose**: ChatGPT-style categorized user profile with append-only fact storage, historical tracking, temporal queries, and hybrid retrieval

**Schema Version**: 2.0 (auto-migrates from 1.0)

**Architecture Overview**:
```
Query → LLMFactExtractor (categorizes) → UserProfile.add_facts_batch() → JSON persistence
                                              ↓
                                   Append-only: old values marked is_current=False
                                              ↓
Query → get_user_profile_context(query) → hybrid retrieval → [USER PROFILE] in prompt
                                              ↓ (if temporal query)
                                   Includes historical timeline
```

**Replaces**: Previous [SEMANTIC FACTS] and [RECENT FACTS] sections in prompts

**Key Components**:

**user_profile_schema.py**:
- `SCHEMA_VERSION = "2.0"` — auto-migration from 1.0
- `ProfileCategory` enum: 12 life domains (identity, education, career, projects, health, fitness, preferences, hobbies, study, finance, relationships, goals)
- `ProfileFact` dataclass: relation, value, category, confidence, timestamp, source_excerpt, **fact_id** (UUID), **is_current** (bool), **supersedes** (fact_id of replaced fact)
- `RELATION_CATEGORY_MAP`: 40+ predefined relation→category mappings
- `categorize_relation(relation)`: Direct lookup + heuristic fallbacks (pattern matching)

**user_profile.py**:
- `UserProfile` class: Persistent manager for categorized user facts
- Storage: `data/user_profile.json` with atomic writes (temp file + os.replace())
- Thread-safe: threading.Lock() for concurrent access
- **Append-only**: Facts never deleted, only marked `is_current=False` when superseded
- Conflict resolution:
  - Same (relation, value) → confidence boost (+0.05, capped at 1.0)
  - Same relation, different value → old fact marked `is_current=False`, new fact added with `supersedes=old_fact_id`

**Key Methods**:
- `add_fact(relation, value, confidence, source_excerpt, category, timestamp)` → Append-only fact addition
- `add_facts_batch(triples)` → Batch add from LLM extraction (used at shutdown)
- `get_category(category, include_historical=False)` → Get facts for category
  - Default: only `is_current=True` facts (backward compatible)
  - `include_historical=True`: all facts including superseded ones
- `get_relevant_facts(query, category, limit=3)` → **Hybrid retrieval (2/3 semantic + 1/3 recent)**
  - Temporal queries automatically include historical facts
- `get_context_injection(max_tokens, query)` → Format profile for prompt injection
  - Temporal queries add `[TIMELINE]` section with historical values
- `get_current_view(category=None)` → Dict of {category: [current_facts]} **[NEW v2]**
- `get_profile_at(timestamp)` → Point-in-time snapshot: returns most recent fact per relation as of given datetime **[NEW v2]**
- `get_fact_history(relation, category=None)` → Chronological list of all values a relation has had **[NEW v2]**
- `export_markdown()` → Markdown export (current facts only per category)
- `get_quick_profile()` → Extract identity fields (name, location, age)
- `_is_temporal_query(query)` → Detect temporal keywords (history, progress, over time, used to, etc.) **[NEW v2]**
- `_prune_category(cat_key)` → Ephemeral relation pruning when category exceeds soft cap **[NEW v2]**
- `migrate_schema()` → Auto-migrate v1.0 → v2.0 (adds fact_id, is_current, supersedes chain) **[NEW v2]**

**Ephemeral Pruning** (prevents unbounded growth of temporal facts):
- When category size exceeds `PROFILE_CATEGORY_SOFT_CAP` (default: 50)
- Only prunes relations listed in `PROFILE_EPHEMERAL_RELATIONS` (current_feeling, is, has, thinks, etc.)
- Keeps at most `PROFILE_EPHEMERAL_MAX_HISTORY` (default: 20) historical entries per ephemeral relation
- Non-ephemeral relations (name, squat_max, etc.) are never pruned

**Integration Points**:

1. **memory/shutdown_processor.py** (process_shutdown_memory):
   ```python
   # After LLM fact extraction
   if triples and hasattr(self, 'user_profile'):
       added = self.user_profile.add_facts_batch(triples)
   ```

2. **memory/llm_fact_extractor.py** (Enhanced):
   - Auto-categorization via `categorize_relation()` in `_normalize_triple()`
   - Enhanced prompt with 12 category descriptions
   - Confidence scoring (0.0-1.0)

3. **core/prompt/context_gatherer.py**:
   - `get_user_profile_context(query, max_tokens=500)` → Calls UserProfile.get_context_injection()

4. **core/prompt/formatter.py**:
   - `[USER PROFILE]` section → Pre-formatted string from context["user_profile"]

**Configuration** (`config/app_config.py`):
```python
PROFILE_CATEGORY_SOFT_CAP = 50           # Trigger ephemeral pruning above this
PROFILE_EPHEMERAL_MAX_HISTORY = 20       # Max historical entries per ephemeral relation
PROFILE_EPHEMERAL_RELATIONS = [          # Relations whose history gets pruned
    "current_feeling", "current_mood", "current_activity", "current_time",
    "is", "has", "was", "thinks", "needs", "plans", "wants", "likes",
    "greeting", "expressed_feeling", "testing", ...
]
```

**CLI Commands** (main.py):
- `python main.py export-profile` → Write markdown to `data/user_profile_export.md`
- `python main.py show-profile` → Print profile to console with fact count

**Testing**:
- tests/unit/test_user_profile_schema.py (5 tests): Schema validation, categorization, serialization
- tests/unit/test_user_profile.py (~35 tests): Persistence, append-only storage, conflict resolution, computed views, migration, ephemeral pruning, temporal retrieval, backward compatibility
- tests/test_user_profile_schema_preferences.py: Schema version assertion (2.0)

**Dependencies**: threading, datetime, pathlib, uuid, config.app_config

---

### 2.4 memory/corpus_manager.py (JSON Persistence)
**Purpose**: Short-term memory storage in JSON

**Structure**:
```python
{
  "conversations": [
    {
      "id": "uuid",
      "query": "user input",
      "response": "assistant output",
      "timestamp": "ISO-8601",
      "truth_score": 0.0-1.0,
      "importance_score": 0.0-1.0,
      "tags": ["topic1", "topic2"],
      "memory_type": "episodic|semantic|procedural|summary|meta",
      "metadata": {...}
    }
  ]
}
```

**Key Methods**:
- `add_memory(memory, memory_type)` → Append to JSON
- `get_recent_memories(n, memory_type)` → Last N entries
- `save()` → Write to disk (atomic with temp file)

---

### 2.5 memory/storage/multi_collection_chroma_store.py (Vector Store)
**Purpose**: Semantic search across 10 ChromaDB collections

**Collections**: conversations, summaries, wiki_knowledge, facts, reflections, obsidian_notes, reference_docs, procedural, procedural_skills, proposals

**Key Methods**:
- `add_memory(text, metadata, collection)` → Embed and store
- `query(text, collection, n_results)` → Semantic search
  - Uses sentence-transformers for embeddings
  - Returns: {ids, documents, metadatas, distances}

**Embedding Model**: `all-MiniLM-L6-v2` (default, configurable)

---

### 2.6 processing/gate_system.py (Multi-Stage Filter)
**Purpose**: 3-stage relevance filtering

**Pipeline**:
```python
Stage 1: FAISS semantic search
  → Returns top 50 candidates

Stage 2: Cosine similarity threshold
  → Filter by threshold (default ~0.45-0.65)
  → Reduces to ~20-30 results

Stage 3: Cross-encoder reranking
  → MS-MARCO cross-encoder
  → Rerank and return top K
```

**Key Methods**:
- `filter_memories(query, memories, k)` → Run all 3 stages
- `_cosine_filter(query, memories, threshold)` → Stage 2
- `_cross_encoder_rerank(query, memories, k)` → Stage 3

**Configuration**: `GATE_COSINE_THRESHOLD` controls Stage 2 aggressiveness

---

### 2.7 core/prompt/ (Modular Prompt System)
**Purpose**: Refactored prompt building split into specialized components

**Architecture**:
```python
UnifiedPromptBuilder (core/prompt/builder.py)
    ↓
├── ContextGatherer (context_gatherer.py) - Data collection
├── PromptFormatter (formatter.py) - Text assembly
├── TokenManager (token_manager.py) - Budget management
├── LLMSummarizer (summarizer.py) - LLM operations
└── Base utilities (base.py) - Common functions
```

**Token Budget Strategy**:
```python
Total budget: 15000 tokens (default, configurable)

Allocation with separated sections:
1. System prompt (fixed ~500-800)
2. Current query (variable)
3. Recent conversation (last 15 turns)
4. Relevant memories (15 semantic search results)
5. Semantic facts (30) + Recent facts (30)
6. Recent summaries (5) + Semantic summaries (5)
7. Recent reflections (5) + Semantic reflections (5)
8. Wikipedia context (3 max)
9. Dreams (3 max)

Priority order (with weights):
- stm_summary: 10               # HIGHEST - metadata only, no token cost
- narrative_state: 8            # Temporal grounding - synthesized life context [NEW 2026-01-17]
- recent_conversations: 7
- semantic_chunks: 6
- personal_notes: 6             # Obsidian notes
- user_uploads: 6               # Persisted user file/image uploads [NEW 2026-02-10]
- reference_docs: 5             # Daemon self-knowledge docs
- git_commits: 5               # Git commit history (procedural)
- procedural_skills: 5         # Reusable problem-solving patterns
- memories: 5
- semantic_facts/fresh_facts: 4
- summaries: 3
- proposed_features: 3         # Code proposals (project-related queries only) [NEW 2026-02-09]
- reflections/dreams: 2
- wiki: 1
```

**Context Dict Structure**:
```python
{
  "stm_summary": {...},           # STM context summary dict (topic, intent, tone, etc.)
  "narrative_state": "...",       # Temporal grounding - synthesized life context [NEW 2026-01-17]
  "recent_conversations": [...],
  "memories": [...],
  "semantic_facts": [...],
  "fresh_facts": [...],
  "summaries": [...],
  "recent_summaries": [...],      # Separated for distinct headers
  "semantic_summaries": [...],    # Separated for distinct headers
  "reflections": [...],
  "recent_reflections": [...],    # Separated for distinct headers
  "semantic_reflections": [...],  # Separated for distinct headers
  "dreams": [...],
  "semantic_chunks": [...],
  "wiki": [...],
  "procedural_skills": [...],   # Adaptive workflows (WHEN/THEN patterns)
  "proposed_features": [...],   # Code proposals for project-related queries [NEW 2026-02-09]
}
```

**Key Methods**:
- `builder.py::build_prompt(user_input, config, ..., retrieval_overrides=None, weight_overrides=None)` → Complete context dict. Intent-driven overrides applied to all max_* retrieval counts and scorer weights. **[UPDATED 2026-02-15]**
- `builder.py::_assemble_prompt(context, user_input)` → Format to string with headers
- `builder.py::_hygiene_and_caps(context)` → Dedupe with semantic similarity (0.90 threshold)
- `builder.py::_backfill_recent_conversations()` → Top-up after deduplication
- `builder.py::_build_lightweight_context()` → Minimal context for small-talk
- `token_manager.py::_manage_token_budget(context)` → Budget enforcement
- `context_gatherer.py::_get_summaries_separate()` → Hybrid recent+semantic retrieval
- `summarizer.py::_reflect_on_demand()` → Generate reflections if below threshold
- `formatter.py::_sanitize_embedded_headers(text)` → Escape section headers in memory content **[NEW 2026-01-08]**
  - Prevents prompt pollution when conversations discuss prompt structure
  - Converts `[RECENT CONVERSATION]` → `(RECENT CONVERSATION)` in stored memories
  - Applied to: memories, summaries, reflections, facts in builder.py
  - Includes `[PROPOSED FEATURES]` pattern **[ENHANCED 2026-02-09]**

**Deduplication Features**:
- Cross-section semantic deduplication using embedder
- SIMILARITY_THRESHOLD = 0.90 for duplicate detection
- Chunk stitching by title (max 4000 chars)
- Automatic backfill when dedup removes too many items

**Prompt Assembly Format**:
```
[TIME CONTEXT]
Current time: 2025-11-18 10:30:00
Time since last message: 5 m
Time since last session: 2 d 3 h

[RECENT CONVERSATION] n=15
1) 2025-11-18 10:00:00: User: ... Daemon: ...
2) ...

[RELEVANT MEMORIES] n=15
1) 2025-11-17 14:00:00: User: ... Daemon: ...
...

[SEMANTIC FACTS] n=30
1) subject | relation | object
...

[RECENT SUMMARIES] n=5
1) 2025-11-15: Summary content...
...

[SEMANTIC SUMMARIES] n=5
...

[RECENT REFLECTIONS] n=5
...

[TEMPORAL GROUNDING] [NEW 2026-01-17 - synthesized life context]
**Current Life State**
The user is navigating a phase of academic and personal growth...

**Active Threads**
- Memory system improvements
- Academic preparation
- Emotional processing

**Emotional Trajectory**
Mood trending positive after resolving recent stressors...

**Recurring Themes**
Integration of technology into studies, self-care practices...

[SHORT-TERM CONTEXT SUMMARY]
Topic: Python debugging
User Question: How to fix the timeout error
Intent: Get practical solution to immediate technical problem
Tone: frustrated
Open Threads: Performance optimization, Error handling strategy
Constraints: Limited to standard library, Production environment

[CURRENT USER QUERY]
User input here
```

**Dependencies**: All prompt submodules, memory system, gate system, HybridRetriever

---

### 2.8 core/response_generator.py (LLM Interface)
**Purpose**: Streaming response generation with multi-provider support

**Providers**: OpenAI, Anthropic (Claude), OpenRouter, Local models

**Key Methods**:
- `generate_streaming_response(prompt, model_name, system_prompt, max_tokens)` → async generator
- `generate_full(prompt, model_name, system_prompt, temperature)` → complete response
- `generate_best_of(prompt, model_name, system_prompt, n, temps, max_tokens)` → best of N
- `generate_duel_and_judge(prompt, model_a, model_b, judge_model, ...)` → A vs B
- `generate_best_of_ensemble(prompt, generator_models, selector_models, ...)` → multi-model

**Generation Modes**:
- **Standard streaming**: Single model, async chunks
- **Best-of-N**: Generate N responses at different temps, heuristic scoring picks best
- **Duel mode**: Two models compete, judge model picks winner
- **Ensemble**: Multiple generator models, multiple selector models vote

**Best-of-N Scoring** (heuristic):
- Relevance to question (keyword overlap)
- Response length appropriateness
- Coherence markers
- No repetition penalties

**Streaming Implementation** (FIXED 2025-11-24):
- Word-by-word yielding with buffer accumulation
- STOP_MARKERS for special token filtering: `<|user|>`, `<|assistant|>`, `<|system|>`, `<|end|>`, `<|eot_id|>`, `<｜end▁of▁sentence｜>` (DeepSeek)
- Timeout protection (60s per chunk)
- Critical fix: `buffer += delta_content` must execute for ALL non-empty chunks (not inside else block)

**Dependencies**: ModelManager, CompetitiveScorer

---

### 2.8a core/best_of_handler.py (Best-of Orchestration) **[NEW 2026-01]**
**Purpose**: Extracted handler for best-of-N, duel, and ensemble response generation

**Key Classes**:
- `BestOfResult` (dataclass): Contains response, mode, metadata
- `BestOfHandler`: Orchestrates mode selection and execution

**Key Methods**:
- `should_use_best_of(user_input, use_raw_mode)` → bool
  - Checks ENABLE_BEST_OF config
  - Uses analyze_query() to check if question with sufficient tokens
- `generate(prompt, user_input, system_prompt, model_name)` → BestOfResult
  - Mode selection: duel → ensemble → single
  - Timeout handling with fallback to streaming
  - Returns structured result with metadata

**Mode Selection Logic**:
```python
if duel_mode and len(generator_models) == 2:
    mode = "duel"      # 2 generators + 1 judge
elif generator_models:
    mode = "ensemble"  # N generators + M judges
else:
    mode = "single"    # 1 model, N temperatures
```

**Config** (app_config.py):
- `ENABLE_BEST_OF`: Master toggle
- `BEST_OF_DUEL_MODE`: Enable duel (requires 2 generators)
- `BEST_OF_GENERATOR_MODELS`: List of generator model names
- `BEST_OF_SELECTOR_MODELS`: List of judge model names
- `BEST_OF_LATENCY_BUDGET_S`: Timeout before fallback to streaming

**Dependencies**: ResponseGenerator, analyze_query

---

### 2.8b core/agentic/ (Agentic Search System) **[NEW 2026-01]**
**Purpose**: Multi-round ReAct-style web search with LLM-driven iteration

**Key Components**:

**core/agentic/types.py** - Data structures
- `AgentState` (Enum): `THINKING`, `SEARCHING`, `OBSERVING`, `SYNTHESIZING`, `DONE`, `ERROR`
- `SearchProtocol` (Enum): `NATIVE_TOOLS` (API models), `XML_MARKERS` (local models)
- `SearchRequest`: Query + depth + max results
- `SearchRound`: Results from a single search iteration
- `AgenticSearchSession`: Full session state (rounds, tokens used, etc.)
- `ProgressEvent`: Status updates for UI (event_type, message)
- `SEARCH_TOOL_DEFINITION`, `DONE_TOOL_DEFINITION`: OpenAI-style tool schemas
- `AGENTIC_SYSTEM_PROMPT_INJECTION`: Instructions for local models to use XML markers

**core/agentic/protocols.py** - Protocol detection and parsing
- `detect_protocol(model_name)` → `SearchProtocol` (native for gpt/claude, XML for local)
- `NativeToolsHandler`: Parses OpenAI/Anthropic tool_calls from response
- `XMLMarkerHandler`: Parses `<search>query</search>` and `<done/>` from local model output
- `BaseProtocolHandler`: Common interface for both

**core/agentic/controller.py** - Main controller
- `AgenticSearchController`: Orchestrates the ReAct loop
  - `run_agentic_search(query, system_prompt, model_name, initial_terms)` → AsyncGenerator
  - Yields `ProgressEvent` for status updates, `str` for response chunks
  - Max 5 rounds of search before forcing synthesis
  - Compresses accumulated context to fit token budget
  - Falls back gracefully on errors

**ReAct Loop Flow**:
```
1. Initial search with LLM-provided terms
2. LOOP (max 5 rounds):
   a. LLM sees: query + search results so far
   b. LLM decides: search_more(query) OR done()
   c. If search_more: execute search, add to context
   d. If done: generate final response
3. Stream final response to UI
```

**Config** (config.yaml):
```yaml
agentic_search:
  enabled: true
  max_rounds: 5
  context_budget_tokens: 8000
  compression_model: gpt-4o-mini
  prefer_native_tools: true
```

**Wolfram Alpha Integration** [NEW 2026-01-22]:
- `wolfram_manager` parameter added to `AgenticSearchController.__init__()`
- `WOLFRAM_TOOL_DEFINITION` in types.py - tool schema for native protocol
- `<wolfram>query</wolfram>` XML marker for local models
- `SearchDecision` extended with `wants_wolfram`, `wolfram_query`, `wolfram_reason`
- `_execute_wolfram()` method with fallback to web search on failure
- Progress events: `computing` → `computed` (parallel to `searching` → `found_results`)

**E2B Code Sandbox Integration** [NEW 2026-01-22]:
- `sandbox_manager` parameter added to `AgenticSearchController.__init__()`
- `SANDBOX_TOOL_DEFINITION` in types.py - tool schema for native protocol
- `<python purpose="...">code</python>` XML marker for local models
- `SearchDecision` extended with `wants_sandbox`, `sandbox_code`, `sandbox_purpose`
- Persistent session created at loop start (variables survive across turns)
- Session cleanup in finally block
- Progress events: `executing_code` → `code_executed` / `code_error`
- `skip_initial_search` parameter skips Round 1 web search for computation-only queries

**Dependencies**: WebSearchManager, WolframManager (optional), SandboxManager (optional), ModelManager, TokenizerManager

---

### 2.9 gui/ (Gradio Web Interface) **[MAJOR FIXES 2025-11-24, AGENTIC 2026-01]**
**Purpose**: Async streaming GUI with thinking block support, response tag handling, and agentic search routing

**Key Components**:

**gui/handlers.py** - Event handlers, streaming relay, agentic routing, and Fast Mode **[ENHANCED 2026-03-10]**
- `handle_submit()` → Main async generator yielding response chunks to GUI
  - **Fast Mode** [NEW 2026-03-10]: When `fast_mode=True`, temporarily reduces retrieval limits (PROMPT_MAX_MEMS→10, PROMPT_MAX_RECENT→5, PROMPT_MAX_SEMANTIC→8), sets `context_gatherer._fast_mode` and `hybrid_retriever._fast_mode`. Yields progress keepalive messages ("Thinking...", "Analyzing context...", etc.) every 2s during `prepare_prompt()` to prevent mobile timeouts. All overrides restored in `finally` block.
  - **Agentic Search Path** (NEW 2026-01):
    - Quick filter skips casual acknowledgments (< 5 words, "nice", "thanks", etc.)
    - Calls `analyze_for_web_search_llm()` to decide if search needed
    - If triggered, routes through `AgenticSearchController.run_agentic_search()`
    - Yields progress events (🔍 searching, 📄 found results, ✨ synthesizing)
    - Strips thinking blocks from final response
  - **Standard Path**:
    - Receives streaming chunks from response_generator
    - Parses thinking blocks in real-time
    - Applies response tag stripping
  - Yields dict chunks: `{"role": "assistant", "content": text, "is_progress"?: bool}`

**Tag Stripping Logic** (FIXED):
- Strips outer wrapper tags: `<reply>`, `<response>`, `<answer>`, `<result>`
- Uses backreference regex `\1` to match opening/closing tags properly
- Regex: `r"^\s*<\s*(result|reply|response|answer)\s*>\s*([\s\S]*?)\s*<\s*/\s*\1\s*>\s*$"`
- ONLY strips tags at start/end of string (not tags mentioned in content)
- Prevents truncation when LLM discusses tags in response

**gui/launch.py** - Gradio app setup, theme config, and async iteration
- `get_dark_theme()` → Creates dark mode theme with JetBrains Mono font
  - Uses `gradio.themes.Soft` base with slate hues
  - Dark backgrounds: `rgb(17, 24, 39)` body, `rgb(31, 41, 55)` blocks
  - Light text colors for readability on dark backgrounds
  - Font stack: JetBrains Mono (Google Font) → ui-monospace → Consolas → monospace
- `submit_chat()` → Async iteration over handler chunks
  - Creates placeholder assistant message
  - Updates chat_history with streamed content
  - Handles thinking blocks with HTML collapsible sections
  - Yields to Gradio for real-time display
- **Sync Notes Button** → Embeds new Obsidian notes to ChromaDB
  - Calls `ObsidianManager.embed_vault(force_reindex=False)`
  - Shows status: success count, skipped count, or error message
  - Non-blocking incremental sync (skips already-indexed notes)
- **Proposals Tab** → Browse, filter, manage, generate, and implement code proposals **[ENHANCED 2026-02-09]**
  - Collapsible HTML cards with `<details>/<summary>` for each proposal
  - Status badges with color coding (pending=gray, approved=amber, completed=green, rejected/failed=red)
  - Filter by status (pending/approved/rejected/completed/failed) and type (feature/refactor/etc.)
  - Auto-loads proposals on page open via `demo.load()`
  - "Generate Proposals Now" button for on-demand LLM generation
  - **Manage Proposals** section: dropdown selector + Mark Built / Reject / Approve buttons **[NEW 2026-02-09]**
    - Rejection reason text input (optional)
    - Auto-refreshes proposal list after status change
    - Uses ProposalStore.update_status() for status transitions
  - Code Generation: dropdown selector → "Generate Code" → staging directory output
  - Rendered as HTML code blocks with syntax highlighting

**Streaming Flow**:
```
response_generator.generate_streaming_response()
    → yields word chunks
handlers.py async for loop
    → accumulates in final_output
    → parses thinking/answer
    → strips tags
    → yields {"role": "assistant", "content": display_output}
launch.py async iteration
    → updates chat_history[-1]["content"]
    → yields to Gradio
Gradio renders updated chat_history
```

**Critical Fixes**:
1. Added `<reply>` and `<response>` to tag stripping (was only `<result>`)
2. Fixed regex to use backreference preventing content truncation
3. Ensured chunks flow through all three stages (generator → handler → launch)

**Dependencies**: Orchestrator, Gradio

---

### 2.9.1 gui/wizard.py (Onboarding Wizard) **[NEW 2025-12-11]**
**Purpose**: Conversational first-run setup wizard for new users

**Wizard Flow**:
```
WELCOME → API_KEY → STYLE → NAME → PRONOUNS → BACKGROUND → COMPLETE
```

**Key Components**:
- **WizardStep** (Enum) - Tracks current step in wizard flow
- **WizardState** (Dataclass) - Maintains wizard state (step, collected_data, error_count)
- **process_wizard_message()** - Main async handler routing to step-specific handlers

**Step Handlers**:
1. `_handle_welcome()` - Acknowledges first interaction, advances to API_KEY
2. `_handle_api_key()` - Validates format + tests API key with live call, writes to .env
3. `_handle_style()` - Parses style preference (warm/balanced/direct)
4. `_handle_name()` - Collects user name (optional, supports skip)
5. `_handle_pronouns()` - Collects pronouns (optional, supports skip)
6. `_handle_background()` - Extracts facts from background text via LLMFactExtractor
7. `_finalize_wizard()` - Saves profile, returns completion message

**API Key Validation**:
- Format check: Must start with 'sk-or-' and length > 20
- Live test: Calls `model_manager.generate_once()` with gpt-4o-mini
- Success: Writes to `.env` as `OPENAI_API_KEY=<key>` AND sets `os.environ`
- Failure: Returns error message, stays on API_KEY step

**Style Parsing** (parse_style_preference):
- Numeric: "1" → warm, "2" → balanced, "3" → direct
- Text: "warm & supportive" → warm, "direct & concise" → direct
- Keywords: "empathy/caring" → warm, "short/brief" → direct
- Default: balanced

**Skip Detection** (is_skip):
- Recognized: "skip", "none", "n/a", "pass", "no", "-", ""
- Applied to: NAME, PRONOUNS, BACKGROUND steps

**Error Recovery**:
- Max retries: 3 (configurable via WizardState.max_retries)
- Error count tracked across failures
- After max retries: Helpful error message with manual fallback instructions

**Integration with UserProfile**:
- Calls `profile.update_identity(name, pronouns)`
- Calls `profile.update_preferences(style, ...)`
- Adds background facts via `profile.add_fact()`
- Profile automatically saved to `data/user_profile.json`

**Wizard Routing** (gui/launch.py):
- First-run check: `user_profile.is_first_run(corpus_manager)`
  - Condition: `corpus_count < 5 AND no identity.name`
- If first-run: Launch `_launch_wizard_ui()` instead of normal chat
- Force mode: `python main.py wizard` bypasses first-run check

**Wizard UI** (_launch_wizard_ui):
- Simple Gradio interface: Chatbot + Text Input + Submit Button
- Initial message: Wizard welcome automatically displayed
- State storage: WizardState serialized to dict for Gradio State
- Completion: Shows "✅ Setup complete! Please refresh the page to start chatting."

**Files Modified for Wizard**:
- `memory/user_profile_schema.py` - Added ProfilePreferences, ProfileIdentity, SCHEMA_VERSION
- `memory/user_profile.py` - Added is_first_run(), update_preferences(), update_identity(), get_style_modifier()
- `core/orchestrator.py` - Runtime placeholder substitution ({USER_NAME}, {USER_PRONOUNS})
- `gui/launch.py` - First-run detection and wizard routing
- `core/system_prompt.txt` - Replaced hardcoded "Luke" with {USER_NAME} placeholders
- `main.py` - Added `python main.py wizard` command for testing

**Test Coverage**: 100 new tests (6 test files)
- test_user_profile_schema_preferences.py (13 tests)
- test_user_profile_manager.py (22 tests)
- test_wizard.py (34 tests)
- test_orchestrator_profile_injection.py (14 tests)
- test_wizard_routing.py (10 tests)
- test_system_prompt_placeholders.py (7 tests)

---

### 2.9.2 Desktop Executable System **[NEW 2025-12-12]**
**Purpose**: PyInstaller-based desktop executable with splash screen, platform-specific paths, and staged imports

**Architecture**:
```
[PyInstaller Bootloader] → runtime_hook.py (set DAEMON_FROZEN=1)
         ↓
[main.py bootstrap block] → utils/bootstrap.py (paths, .env, migration)
         ↓
[Splash Screen] ← utils/startup.py (staged imports with progress)
         ↓
[Gradio GUI or Wizard] → normal operation
```

**Key Components**:

**daemon.spec** (PyInstaller spec file):
- **Mode**: One-dir (prevents 30s startup delay of one-file)
- **Splash**: Native PyInstaller splash (assets/splash.png)
- **Icon**: Platform-specific (assets/daemon_icon.ico)
- **Excludes**: pytest, matplotlib, tkinter, tensorflow
- **Hidden imports**: chromadb, gradio, sentence_transformers, spacy, tiktoken
- **Data files**: Bundled configs, spaCy model, tiktoken encodings, gradio templates
- **Runtime hook**: Sets DAEMON_FROZEN env var before main.py

**utils/bootstrap.py**:
- `IS_FROZEN`: Boolean detecting PyInstaller frozen mode
- `get_app_dir()`: Executable directory (frozen) or project root (dev)
- `get_user_data_dir()`: Platform-specific user data directory
  - Windows: `%APPDATA%/Daemon/`
  - macOS: `~/Library/Application Support/Daemon/`
  - Linux: `~/.daemon/`
- `get_resource_path(relative_path)`: Bundled resource lookup
- `setup_environment()`: Sets env vars, loads .env, migrates data
- `migrate_user_data()`: One-way migration from ./data/ to user dir
- `update_splash(text)`, `close_splash()`: PyInstaller splash control

**utils/startup.py**:
- `StartupProgress`: Progress tracker with splash/console updates
- `staged_import(progress)`: Ordered imports with timing
  - Stage 1-2: Core Python + config (~0.1s)
  - Stage 3: PyTorch (~1.8s)
  - Stage 4: Sentence transformers (~4.1s)
  - Stage 5: spaCy (~0.5s)
  - Stage 6: ChromaDB (~0.6s)
  - Stage 7: Gradio (~1.4s)
  - Stage 8-10: Application modules
- `run_startup(preload=False)`: Complete startup sequence
- `check_requirements()`: Validate required packages

**hooks/** (PyInstaller custom hooks):
- `hook-chromadb.py`: Data files + hnswlib + posthog
- `hook-gradio.py`: Static files + fastapi/starlette/uvicorn
- `hook-sentence_transformers.py`: Data files + transformers + cross-encoder
- `hook-spacy.py`: Language data + en_core_web_sm model
- `hook-tiktoken.py`: Encoding files + registry
- `runtime_hook.py`: DAEMON_FROZEN env var + multiprocessing spawn (Windows)

**Startup Timing** (from audit 2025-12-12):
- Total: ~12-17s (splash essential)
- Heaviest: sentence_transformers (4.1s), torch (1.8s)
- First token latency: <1s after startup

**Build Commands**:
```bash
# Clean and build
rm -rf build/ dist/
pyinstaller daemon.spec --clean --noconfirm

# Output: dist/Daemon/Daemon (or Daemon.exe on Windows)
# Size: ~650MB (with CPU-only PyTorch)
```

**Frozen Mode Path Resolution**:
```python
# main.py bootstrap (runs before config import)
if getattr(sys, 'frozen', False):
    from utils.bootstrap import setup_environment
    setup_environment()  # Sets CORPUS_FILE, CHROMA_PATH, etc.

# Then config imports read correct paths
from config.app_config import config
```

**Data Migration**:
- On first run: migrates ./data/ contents to user data directory
- Migrated: corpus_v4.json, user_profile.json, chroma_db_v4/, time files
- NOT migrated: .env (created fresh by wizard)

**Browser Auto-Launch**:
- Frozen mode only: Opens browser after Gradio starts
- Platform-specific: xdg-open (Linux), open (macOS), os.startfile (Windows)

**Dependencies**: PyInstaller 6.x, pyi_splash module

---

### 2.10 models/model_manager.py (Multi-Provider LLM)
**Purpose**: Unified interface for multiple LLM providers

**Supported APIs**:
- OpenAI (GPT-3.5, GPT-4, GPT-4o)
- Anthropic (Claude Sonnet, Opus, Haiku)
- OpenRouter (aggregator)
- Local models via OpenAI-compatible endpoints

**Key Methods**:
- `generate(prompt, model_alias, stream)` → async response
- `_get_client(provider)` → Provider-specific client
- `_map_alias_to_model(alias)` → e.g., "sonnet-4.6" → "anthropic/claude-sonnet-4.6" **[NEW 2026-03-10]**

**Model Aliases** (includes `sonnet-4.6` added 2026-03-10):
- All routing goes through OpenRouter base URL

**Environment Variables**:
- `OPENAI_API_KEY` (used for OpenRouter — single key for all providers)

---

### 2.11 utils/topic_manager.py (Topic Extraction)
**Purpose**: Hybrid topic extraction (3-stage: heuristics → spaCy NER → LLM fallback)

**3-Stage Detection Pipeline**:
1. **Heuristic extraction** (fast ~1ms):
   - Strip leading prompts ("can you", "what is", etc.)
   - Extract capitalized spans (proper nouns, entities)
   - Prefer "role of the entity" patterns
   - Title-case for wiki compatibility
   - Guardrails: detect whole-utterance topics (>70% word overlap or >6 words) → return "general"

2. **spaCy noun chunks + NER extraction** (when ambiguous ~10-20ms) **[ENHANCED 2026-01-08]**:
   - Triggered when: heuristic is ambiguous (returns "general", >6 words, or vague terms)
   - Lazy-loaded spaCy model (en_core_web_sm)
   - **Stage 2a: Noun chunks first** (captures compound terms like "first degree murder")
     - Prefers multi-word chunks with substantive content (MIN_CHUNK_WORDS=2)
     - Filters: pronouns, determiners, single-word stopwords
     - Domain term bonus: murder, case, trial, shooting, crisis, policy
   - **Stage 2b: NER fallback** if no good noun chunks
     - Entity priority: PERSON > ORG > GPE > PRODUCT > EVENT > WORK_OF_ART > LAW > NORP
     - Filters: skip low-quality types when alone (ORDINAL, CARDINAL, DATE, TIME, PERCENT)
   - Returns highest quality match or None

3. **LLM fallback** (when no entities found ~50ms):
   - Triggered when: spaCy returns None (no entities)
   - Enhanced prompt: handles emotional/conversational messages
   - Examples: "I am lonely" → "Loneliness", "School starts soon" → "School starting"
   - Returns 2-5 word noun phrase

**Ambiguity Detection**:
- Always triggers stages 2-3 if heuristic returns "general"
- Catches conversational messages without clear entities
- Detects overly long candidates (>6 words)

**Key Methods**:
- `update_from_user_input(text)` → None (updates internal state)
- `get_primary_topic(text=None)` → str | None
- `_extract_primary_from_text(text)` → str (heuristic)
- `_is_ambiguous(candidate, source)` → bool
- `_spacy_ner_extraction(text)` → str | None (noun chunks + NER)
- `_extract_best_noun_chunk(doc)` → str | None (spaCy noun chunk extraction) **[NEW 2026-01-08]**
- `_llm_fallback(text)` → str | None (LLM extraction)

**Configuration**:
- `enable_llm_fallback`: Default True
- `llm_model`: Default "gpt-4o-mini"

**Cost Optimization**: spaCy NER as stage 2 reduces LLM costs for entity-heavy messages (free, fast)

---

### 2.12 knowledge/WikiManager.py (Wikipedia Search)
**Purpose**: FAISS-based Wikipedia semantic search

**Data Sources**:
- Preprocessed Wikipedia dump
- FAISS index (~350MB)
- Memory-mapped embeddings (~800MB)

**Key Methods**:
- `search(query, k)` → Top K Wikipedia passages
- `_embed_query(query)` → Query embedding
- `_faiss_search(embedding, k)` → FAISS retrieval

**Integration**: Called by PromptBuilder if query matches Wikipedia patterns

---

### 2.12.1 knowledge/web_search_manager.py (Real-Time Web Search) **[ENHANCED 2026-01]**
**Purpose**: Tavily API integration for real-time web search with caching, rate limiting, and query decomposition

**Data Classes**:
- `WebSearchDepth` (Enum): `QUICK` (1 credit), `STANDARD` (2 credits), `DEEP` (3-5 credits)
- `WebPage`: Individual search result with title, url, content, snippet, score
- `WebSearchResult`: Container for pages + metadata (credits, cache status, timestamp)
- `WebSearchSession`: Maintains search state across prompt building
- `QueryDecomposition` **[NEW 2026-01]**: Result of query analysis (should_decompose, sub_queries, confidence)
- `MultiSearchResult` **[NEW 2026-01]**: Extended result with decomposition metadata

**Key Components**:

1. **WebSearchRateLimiter**:
   - Daily credit tracking (default 100 credits/day)
   - Persists to `data/web_search_credits.json`
   - Auto-resets at midnight
   - Methods: `can_search(credits)`, `record_usage(credits)`, `get_remaining_credits()`

2. **WebSearchCache**:
   - ChromaDB-backed with 72-hour TTL
   - Query normalization for cache hits
   - Methods: `get(query)`, `store(query, result)`

3. **WebSearchManager**:
   - Tavily Search API (basic results) + Extract API (page content)
   - Automatic depth selection based on query complexity
   - **[NEW 2026-01]** Query decomposition for complex/multi-entity queries
   - Methods:
     - `search(query, depth)` → `WebSearchResult`
     - `multi_search(query, depth, auto_decompose=True)` → `MultiSearchResult` **[NEW]**
     - `decompose_query(query)` → `QueryDecomposition` **[NEW]**
     - `is_available()` → Check API key + rate limits

**Query Decomposition** **[NEW 2026-01]**:
```
"Compare Tesla and Rivian stock prices"
    ↓ decompose_query() [LLM analysis]
["Tesla stock 2026", "Rivian stock 2026"]
    ↓ parallel Tavily searches
Merged + deduplicated results
```
- LLM-based detection of multi-entity/comparison queries
- Max 4 sub-queries (credit budget aware)
- 0.6 confidence threshold to trigger decomposition
- Results merged by URL, keeping highest-scoring version

**Configuration** (`config/app_config.py`):
```python
WEB_SEARCH_ENABLED = True
WEB_SEARCH_API_KEY = os.getenv("TAVILY_API_KEY", "")
WEB_SEARCH_TIMEOUT = 30.0
WEB_SEARCH_MAX_CONTENT_CHARS = 10000
WEB_SEARCH_DAILY_CREDIT_LIMIT = 100
WEB_SEARCH_CACHE_TTL_HOURS = 72
```

**Integration**: Called by ContextGatherer during parallel context retrieval

---

### 2.12.1a knowledge/wolfram_manager.py (Wolfram Alpha Computation) **[NEW 2026-01-22]**
**Purpose**: Wolfram Alpha LLM API integration for computational queries - math, science, unit conversions

**Module Contract**:
```
Purpose: Handle mathematical computations, scientific data, unit conversions via Wolfram Alpha LLM API
Inputs:
  - query(input_text: str) -> WolframResult: Execute computational query
  - is_available() -> bool: Check if Wolfram Alpha is configured
  - format_for_prompt(result: WolframResult) -> str: Format result for LLM context
Outputs:
  - WolframResult with success status, result text, assumptions, execution time
Side effects:
  - HTTP requests to Wolfram Alpha API
  - In-memory caching of results
  - Rate limiting tracking
Dependencies: config.app_config (WOLFRAM_* constants), httpx
```

**Data Classes**:
- `WolframResult`: Query result container (query, success, result, assumptions, error, execution_time, cached)
- `WolframRateLimiter`: Token bucket rate limiter for API calls

**Key Methods**:
- `query(input_text)` → `WolframResult`: Execute Wolfram Alpha query
- `is_available()` → `bool`: Check if API key configured
- `format_for_prompt(result)` → `str`: Format for LLM context injection
- `get_rate_limit_status()` → `Dict`: Current rate limit state
- `clear_cache()` → `int`: Clear cache, return count

**Caching**:
- In-memory cache with configurable TTL (default 1 hour)
- Cache key: MD5 hash of normalized query (lowercase, stripped)
- Automatic expiry and cleanup

**Rate Limiting**:
- Token bucket algorithm, configurable per-minute limit (default 60)
- Async-safe with lock
- Returns `False` from `acquire()` when exhausted

**Configuration** (`config/app_config.py`):
```python
WOLFRAM_ENABLED = True
WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID", "")
WOLFRAM_API_URL = "https://www.wolframalpha.com/api/v1/llm-api"
WOLFRAM_TIMEOUT = 30.0
WOLFRAM_MAX_OUTPUT_CHARS = 10000
WOLFRAM_CACHE_TTL_SECONDS = 3600
WOLFRAM_RATE_LIMIT_PER_MINUTE = 60
```

**Error Handling**:
- 403 → "Invalid Wolfram Alpha API key"
- 501 → "Wolfram Alpha could not process this query"
- Timeout → "Request timed out after Xs"
- Connection error → Graceful degradation with error in result

**Integration**: Called by AgenticSearchController._execute_wolfram() during ReAct loop

---

### 2.12.1b knowledge/sandbox_manager.py (E2B Code Sandbox) **[NEW 2026-01-22]**
**Purpose**: Secure Python code execution in ephemeral Firecracker microVMs via E2B API

**Module Contract**:
```
Purpose: Handle multi-step computations, data analysis, visualizations via E2B secure code interpreter
Inputs:
  - execute_code(code: str) -> SandboxResult: Execute in ephemeral sandbox
  - create_session() -> PersistentSession: Create stateful session (variables persist across calls)
  - is_available() -> bool: Check if E2B API key is configured
  - format_for_prompt(result: SandboxResult, purpose: str) -> str: Format result for LLM context
Outputs:
  - SandboxResult with success status, stdout, stderr, error, results, execution_time
Side effects:
  - HTTP requests to E2B API
  - In-memory caching of results (ephemeral mode only)
  - Rate limiting tracking
Dependencies: config.app_config (SANDBOX_* constants), e2b_code_interpreter
```

**Data Classes**:
- `SandboxResult`: Execution result container (code, success, stdout, stderr, error, results, execution_time)
- `SandboxRateLimiter`: Token bucket rate limiter for API calls
- `ExecutionCache`: TTL cache for identical code execution results
- `PersistentSession`: Stateful session wrapper with variable persistence

**Key Methods**:
- `execute_code(code)` → `SandboxResult`: Execute code in ephemeral sandbox
- `create_session()` → `PersistentSession`: Create persistent session for multi-turn execution
- `is_available()` → `bool`: Check if E2B API key configured
- `format_for_prompt(result, purpose)` → `str`: Format for LLM context injection
- `get_rate_limit_status()` → `Dict`: Current rate limit state
- `clear_cache()` → `int`: Clear cache, return count

**PersistentSession** (for ReAct loop):
- Variables persist across multiple `run()` calls (Jupyter-like experience)
- Auto-cleanup via context manager or explicit `close()`
- Rate limiting shared with manager
- Tracks execution count and lifetime

**Caching**:
- In-memory cache with configurable TTL (default 1 hour)
- Cache key: MD5 hash of normalized code (whitespace-stripped)
- Only caches successful executions

**Rate Limiting**:
- Token bucket algorithm, configurable per-minute limit (default 30)
- Async-safe with lock
- Returns rate-limited SandboxResult when exhausted

**Configuration** (`config/app_config.py`):
```python
SANDBOX_ENABLED = True
SANDBOX_API_KEY = os.getenv("E2B_API_KEY", "")
SANDBOX_TIMEOUT_SECONDS = 60
SANDBOX_SESSION_TIMEOUT_MINUTES = 30
SANDBOX_MAX_OUTPUT_CHARS = 4000
SANDBOX_CACHE_TTL_SECONDS = 3600
SANDBOX_RATE_LIMIT_PER_MINUTE = 30
```

**Pre-installed Packages**: NumPy, Pandas, SciPy, SymPy, Matplotlib (via E2B base image)

**Error Handling**:
- Timeout → "Execution timed out after Xs"
- Rate limited → "Rate limit exceeded"
- API errors → Graceful degradation with error in result
- Closed session → "Session closed" error

**Integration**: Called by AgenticSearchController during ReAct loop via `decision.wants_sandbox`

---

### 2.12.2 utils/web_search_trigger.py (Web Search Detection) **[ENHANCED 2026-01]**
**Purpose**: LLM-first detection with heuristic fallback for when to trigger web search

**Output**:
```python
WebSearchDecision(
    should_search=True,
    confidence=0.80,
    reason="LLM classified as current events query",
    depth=WebSearchDepth.STANDARD,
    search_terms=["optimized query 1", "optimized query 2"],  # LLM-generated
    num_searches=2,
    source="llm"  # or "heuristic"
)
```

**LLM-First Detection** (Primary - used by agentic search):
- `analyze_for_web_search_llm(query, model_manager)` → Primary entry point
- LLM prompt asks: "Does this query need real-time web search?"
- Returns optimized search terms (not just yes/no)
- 70/30 confidence blend with heuristics
- Falls back to pure heuristics on LLM timeout/error

**LLM Decision Criteria** (in prompt) **[UPDATED 2026-01-08]**:
- SEARCH if: current events, recent news, live data, time-sensitive info
- DON'T SEARCH if: historical facts, how-to guides, general knowledge
- NEVER SEARCH for: casual acknowledgments (nice, thanks), meta-comments, greetings, short responses
- NEVER SEARCH for: follow-up references ("watched it", "saw that", "just read it") - these refer to conversation context **[NEW]**

**Heuristic Detection** (Fallback - Multi-Signal Scoring):
1. **Strong recency keywords** (+0.4): "latest", "current", "today", "right now", "breaking"
2. **Moderate recency keywords** (+0.2): "recent", "new", "update", "now"
3. **Explicit search requests** (+0.5): "search for", "look up", "find out about"
4. **News/events keywords** (+0.3): "news", "happening", "announced", "released"
5. **Fast-changing topics** (+0.25): sports scores, stock prices, weather, elections
6. **Year patterns** (+0.15): "2024", "2025", "2026", "this year"

**Suppression Rules**:
- Static/timeless queries: definitions, how-to, history → No search
- Suppression keywords: "always", "generally", "in theory"
- Crisis levels HIGH/MEDIUM → Search suppressed

**Key Functions**:
- `analyze_for_web_search_llm(query, model_manager)` → LLM-first with heuristic fallback **[PRIMARY]**
  - Heuristic veto: if heuristic conf ≤ 0.1, don't let LLM override **[NEW 2026-01-08]**
  - LLM override only if heuristic conf ≥ 0.2 (prevents false positives on follow-ups)
- `analyze_for_web_search(query, crisis_level)` → Heuristic-only entry point
- `should_search_heuristic(query)` → Pure heuristic (fast, ~1ms)
- `quick_prefilter_should_skip(query)` → Deictic follow-up detection **[ENHANCED 2026-01-08]**
  - Patterns: "watched it", "saw it", "read it", "just watched", etc.

**Depth Selection**:
- `QUICK` (1 credit): Simple factual queries
- `STANDARD` (2 credits): News, analysis, multiple sources needed
- `DEEP` (3-5 credits): Research, comprehensive coverage

**Integration**:
- Called by gui/handlers.py for agentic search routing
- Called by ContextGatherer for standard web search embedding

---

### 2.12.3 knowledge/obsidian_manager.py (Personal Notes Integration) **[ENHANCED 2026-03-10]**
**Purpose**: Parse, embed, and retrieve user's personal notes from Obsidian vault with multimodal image support and mtime-based change detection

**Data Classes**:
- `EmbedResult`: Statistics from vault embedding (total_files, embedded_files, chunks, errors, duration, updated_files)

**Key Methods**:
- `embed_vault(force_reindex)` → `EmbedResult`: Index vault to ChromaDB (mtime-based change detection for re-embedding)
- `_delete_file_chunks(file_path)`: Delete old chunks by file_path metadata filter before re-embedding **[NEW 2026-03-10]**
- `get_notes(query, limit, include_images, max_images_per_note)` → `List[Dict]`: Hybrid retrieval (1/3 keyword + 2/3 semantic)
- `_keyword_search(query, limit)` → `List[Dict]`: Title/tag/content/file_path keyword matching **[ENHANCED 2026-01-30]**
- `get_vault_stats()` → `Dict`: Index statistics
- `clear_index()` → `bool`: Clear ChromaDB collection

**Image Support (Multimodal Models)** **[NEW 2026-01-30]**:
- `_extract_images(content)` → `List[str]`: Extract `![[image.png]]` references from content
- `_resolve_image_path(image_name, note_path)` → `Optional[Path]`: Find image in vault (checks: same folder, parent, attachments folders, vault root, global search)
- `_load_image_as_base64(image_path)` → `Optional[Dict]`: Load image as base64 with media_type (PNG, JPG, GIF, WebP, max 10MB)
- `load_images_for_chunk(image_names, note_path, max_images, max_total_mb)` → `List[Dict]`: Load multiple images with size limits

**Image Loading Flow**:
```
get_notes(include_images=True)
    → metadata['images'] contains comma-separated image names
    → load_images_for_chunk() resolves and loads each image
    → note['image_data'] = [{data: base64, media_type, filename, size_bytes}, ...]
    → GUI handler extracts and passes to generate_streaming_response(images=...)
```

**Smart Chunking**:
```
Note < 1500 chars → Embed whole note
Note >= 1500 chars → Split by ## headers
Each chunk stores: images specific to that chunk (not whole-note)
```

**Metadata Extraction**:
- `#tags` - Preserved as comma-separated metadata
- `[[wiki links]]` - Extracted as related_notes references
- `![[images]]` - Extracted per-chunk as comma-separated metadata **[NEW 2026-01-30]**
- YAML frontmatter - Stripped from content
- Inline tags - Stripped from content for cleaner embedding

**Hybrid Retrieval** **[ENHANCED 2026-01-30]**:
```
Query → _keyword_search() → 1/3 results
            ↓ Scoring includes: title + file_path components
            ↓ e.g., "ISYE 6501 week 2" matches path "Vault/OMSA/Courses/ISYE 6501/Week 2/"
      → ChromaDB semantic → 2/3 results (vector similarity)
      → Deduplicate by title → Combined results
```

**Keyword Search Scoring** (in order of priority):
1. Exact phrase in title: 1.0
2. Title starts with query: 0.95
3. All query words in title: 0.9
4. All query words in title + file_path: 0.88 **[NEW 2026-01-30]**
5. Exact phrase in section: 0.85
6. All query words in section: 0.8
7. Partial title+path match: 0.6-0.85 **[NEW 2026-01-30]**
8. Tag match: 0.5
9. Content keyword match: 0.2-0.45

**Configuration** (`config/app_config.py`):
```python
OBSIDIAN_ENABLED = True
OBSIDIAN_VAULT_PATH = "~/Documents/Luke Notes"
OBSIDIAN_CHUNK_THRESHOLD = 1500
OBSIDIAN_MAX_NOTES_PROMPT = 5
# Image support [NEW 2026-01-30]
OBSIDIAN_INCLUDE_IMAGES = True        # Enable image loading for multimodal models
OBSIDIAN_MAX_IMAGES_PER_NOTE = 3      # Max images per note chunk
MULTIMODAL_MODELS = ["opus-4", "claude-3", "sonnet-4", "gpt-4o", "gpt-4-vision", "gemini-pro", ...]
```

**CLI Commands** (`main.py`):
- `python main.py embed-vault` - Index vault to ChromaDB
- `python main.py embed-vault --force` - Force full re-index
- `python main.py vault-stats` - Show indexed chunk count
- `python main.py clear-vault` - Clear collection

**GUI Integration** (`gui/launch.py`):
- "📝 Sync Notes" button in main Chat tab
- Calls `embed_vault(force_reindex=False)` for incremental sync
- Shows status message: new notes synced, skipped count, or error

**Integration**:
- ContextGatherer retrieves via `get_personal_notes(include_images=True)` for multimodal models
- Builder adds `[USER'S PERSONAL NOTES]` section after dreams, before time context
- Builder collects `note_images` list and adds to context for multimodal API calls **[NEW 2026-01-30]**
- TokenManager includes `personal_notes` at priority 6 (high)
- Notes filtered through 3-stage gate system (Cosine → Blended → CrossEncoder)
- GUI handler (`handlers.py`) extracts `note_images` from context and passes to `generate_streaming_response(images=...)` **[NEW 2026-01-30]**

---

### 2.12.4 knowledge/reference_docs_manager.py (Daemon Documentation / Self-Knowledge) **[NEW 2026-01]**
**Purpose**: Upload and retrieve reference documents that provide Daemon with self-knowledge (architecture docs, PROJECT_SKELETON, etc.)

**Data Classes**:
- `UploadResult`: Statistics from document upload (success, title, total_chunks, file_type, errors, duration)

**Key Methods**:
- `upload_document(file_path, title)` → `UploadResult`: Index file to ChromaDB
- `upload_text(content, title, metadata_overrides=None)` → `UploadResult`: Upload text directly (GUI paste). metadata_overrides merged into per-chunk metadata (used by file upload to set type='user_upload') **[ENHANCED 2026-02-10]**
- `get_documents(query, limit)` → `List[Dict]`: Hybrid retrieval (1/3 keyword + 2/3 semantic)
- `list_documents()` → `List[Dict]`: List all uploaded documents
- `delete_document(title)` → `bool`: Remove document from index
- `get_stats()` → `Dict`: Collection statistics
- `clear_all()` → `bool`: Clear entire collection

**Smart Chunking**:
```
Document < 2000 chars → Embed whole document
Document >= 2000 chars → Split by ## headers
```

**Supported File Types**:
- `.md` (markdown) - YAML frontmatter stripped
- `.txt` (text)
- `.py`, `.js`, `.json`, `.yaml`, `.rst` (code/config files)

**Hybrid Retrieval**:
```
Query → _keyword_search() → 1/3 results (title/section match priority)
      → ChromaDB semantic → 2/3 results (vector similarity)
      → Deduplicate by title|section → Combined results
```

**Configuration** (`config/app_config.py`):
```python
REFERENCE_DOCS_ENABLED = True
REFERENCE_DOCS_CHUNK_THRESHOLD = 2000
REFERENCE_DOCS_MAX_PROMPT = 5
```

**CLI Commands** (`main.py`):
- `python main.py upload-doc <file_path> [title]` - Upload document
- `python main.py list-docs` - List all uploaded documents
- `python main.py delete-doc <title>` - Delete specific document
- `python main.py clear-docs` - Clear all documents

**Integration**:
- ContextGatherer retrieves via `get_reference_docs()` method
- Builder adds `[DAEMON DOCUMENTATION]` section in prompt (self-knowledge about architecture)
- TokenManager includes `reference_docs` at priority 5
- Documents filtered through 3-stage gate system (Cosine → Blended → CrossEncoder)

---

### 2.12.5 knowledge/git_memory.py + git_memory_loader.py (Git Commit History) **[NEW 2026-01-27]**
**Purpose**: Populate PROCEDURAL ChromaDB collection with git commit history, giving Daemon visibility into project evolution and decision rationale.

**Key Classes**:
- `GitMemoryExtractor` (`git_memory.py`): Extracts commits via `git log` with metadata
- `GitMemoryLoader` (`git_memory_loader.py`): Loads extracted commits into ChromaDB

**GitMemoryExtractor Methods**:
- `extract_commits(limit, since, include_diffs, diff_max_lines)` → `List[Dict]`: Extract commit history
- `get_recent_since_hash(last_hash)` → `List[Dict]`: Incremental updates since last sync
- `_get_diff_summary(commit_hash, max_lines)` → `str`: Abbreviated `--stat` diff
- `_extract_tags(subject)` → `List[str]`: Conventional commit tag extraction (feat→feature, fix→bugfix, etc.)

**GitMemoryLoader Methods**:
- `backfill(limit, include_diffs)` → `int`: Initial population of PROCEDURAL collection
- `incremental_update(include_diffs)` → `int`: Add new commits since last sync
- `clear()` → `int`: Wipe PROCEDURAL collection and reset sync state

**Commit Memory Format**:
```python
{
    "id": "git-abc12345",
    "content": "Commit: feat: add new feature\n\nDetailed body...\n\nChanges:\n...",
    "metadata": {
        "commit_hash": "abc12345",
        "author": "Luke",
        "age_relative": "3 days ago",
        "timestamp": "2026-01-24T10:30:00+00:00",
        "source": "git",
        "memory_type": "procedural",
        "tags": "git-commit,feature"
    }
}
```

**Configuration** (`config/app_config.py`):
```python
GIT_MEMORY_ENABLED = True
GIT_MEMORY_INCLUDE_DIFFS = False
GIT_MEMORY_DEFAULT_LIMIT = 200
```

**CLI Commands** (`main.py`):
- `python main.py git-backfill [LIMIT]` - Initial load (default: 200 commits)
- `python main.py git-update` - Incremental sync since last backfill
- `python main.py git-status` - Show PROCEDURAL collection stats
- `python main.py git-clear` - Wipe collection and reset sync state

**Integration**:
- ContextGatherer retrieves via `get_git_commits()` method (hybrid: 1/3 recent + 2/3 semantic)
- Builder adds `[PROJECT COMMIT HISTORY]` section in prompt
- TokenManager includes `git_commits` at priority 5
- HybridRetriever and MemoryRetriever include `procedural` collection in queries

---

### 2.12.5a Code Proposals System (Goal-Directed Code Generation) **[NEW 2026-02-05]**
**Purpose**: LLM-generated code improvement proposals with full code generation, stored in ChromaDB for browsing and implementation

**Key Components**:

**memory/code_proposal.py** - Data models (Pydantic BaseModel):
- `ProposalType` (Enum): feature, refactor, bugfix, test, docs, infra
- `ProposalStatus` (Enum): PENDING → APPROVED → COMPLETED (or REJECTED/FAILED)
- `ProposalSource` (Enum): GOAL_DIRECTED, CONVERSATION, MANUAL
- `ImplementationStep`: order, description, file_path, action (create/modify/delete/test)
- `CodeProposal`: Full proposal with title, type, priority, reasoning, description, steps, affected_files, tags, complexity
  - `to_metadata()` / `from_metadata()` for ChromaDB (flat primitives only)
  - `to_dict()` / `from_dict()` for JSON serialization
  - Status lifecycle: `mark_approved()`, `mark_rejected()`, `mark_completed()`, `mark_failed()`

**knowledge/proposal_generator.py** - GoalDirectedGenerator:
- `gather_context()` → Reads CLAUDE.md, QUICK_REFERENCE.md (full), PROJECT_SKELETON.md (filtered), GOALS.md, git log. All live from disk. **[ENHANCED 2026-02-10]**
- `_filter_skeleton_sections(text)` → Strips "Core Components" section (~112K chars) from skeleton, keeps architecture/config/patterns/etc. **[NEW 2026-02-10]**
- `_build_prompt(context, extra_context)` → Creative LLM prompt with Project Conventions, API Quick Reference, architecture, goals, commits
- `_parse_response(response)` → Robust JSON parsing (fences, arrays, line-delimited)
- `_parse_proposal(data)` → Validates raw dict into CodeProposal with type coercion
- `generate_proposals(extra_context, max_proposals)` → Main entry: context → LLM → parsed proposals
- `generate_proposals_with_context(pipeline_context, extra_context, max_proposals)` → Pipeline-enriched generation with semantic memories, summaries, reflections, skills, user profile **[NEW 2026-02-09]**
- `generate_code_for_proposal(proposal, output_dir)` → Full code generation:
  - Iterates implementation steps
  - Reads actual source files for 'modify' actions (capped at 15KB)
  - Action-specific prompts: create, modify, test, delete
  - Writes to staging directory: `data/proposal_code/<proposal_id>/`
  - Creates `_manifest.json` with file list and errors
  - Nothing touches the working tree
- `_generate_step_code(proposal, step)` → Per-step code generation via LLM

**memory/proposal_store.py** - ProposalStore:
- `store_proposal(proposal)` → Embed title+reasoning, store metadata in ChromaDB proposals collection
- `query_proposals(query_text, n_results, status, proposal_type)` → Semantic search with filters
- `get_proposal(proposal_id)` → Retrieve by ID
- `get_pending()` → All pending proposals
- `update_status(proposal_id, new_status)` → Status transition via delete-and-re-add
- `check_similarity(proposal, threshold=0.70)` → Dedup check: cosine similarity OR title word overlap (Jaccard >= 0.60). n_results=5 **[ENHANCED 2026-02-09]**
- `get_for_dedup(limit=25)` → Retrieve recent proposals for dedup context in generator prompt **[ENHANCED 2026-02-09]**

**Configuration** (`config/app_config.py`):
```python
CODE_PROPOSALS_ENABLED = True
CODE_PROPOSALS_MAX_PER_SESSION = 5
CODE_PROPOSALS_DEDUP_THRESHOLD = 0.70
# Context: CLAUDE.md + QUICK_REFERENCE.md (full), PROJECT_SKELETON.md (filtered), GOALS.md, git log

# Prompt integration [NEW 2026-02-09]
CODE_PROPOSALS_PROMPT_ENABLED = True
CODE_PROPOSALS_PROMPT_MAX = 3
CODE_PROPOSALS_KEYWORD_DEDUP_TAG_THRESHOLD = 0.60
CODE_PROPOSALS_SEMANTIC_DEDUP_THRESHOLD = 0.75
CODE_PROPOSALS_LLM_RANKING = False             # Tournament-bracket pairwise ranking
CODE_PROPOSALS_LLM_RANKING_MODEL = "gpt-4o-mini"
CODE_PROPOSALS_WEIGHT_PRIORITY = 0.30
CODE_PROPOSALS_WEIGHT_BREADTH = 0.20
CODE_PROPOSALS_WEIGHT_RECENCY = 0.10
CODE_PROPOSALS_WEIGHT_GOAL_ALIGNMENT = 0.40
```

**Integration Points**:
1. **Shutdown**: `ShutdownProcessor._generate_proposals()` auto-generates after skills extraction (pipeline-enriched or cold fallback)
2. **GUI**: Proposals tab with browse, filter, manage (Mark Built/Reject/Approve), generate, and code generation
3. **Prompt**: `[PROPOSED FEATURES]` section via ProposalFilter pipeline (project-related queries only) **[NEW 2026-02-09]**
4. **ChromaDB**: `proposals` collection (10th collection)

**Data Flow** (generation):
```
Session end / GUI button click
    ↓
GoalDirectedGenerator.generate_proposals()
    ↓ reads CLAUDE.md, QUICK_REFERENCE.md, PROJECT_SKELETON.md (filtered), GOALS.md, git log, session context
    ↓ (if memory_coordinator: also retrieves memories, skills, facts, reflections, summaries, user profile)
LLM generates structured JSON proposals
    ↓ parse + validate
ProposalStore.store_proposal() → ChromaDB (with dedup check: cosine + title word overlap)
    ↓
GUI Proposals tab displays collapsible cards
    ↓ user clicks "Generate Code"
GoalDirectedGenerator.generate_code_for_proposal()
    ↓ reads source files, generates per-step code
data/proposal_code/<proposal_id>/ staging directory
    ↓ user reviews and manually copies/commits
```

**Data Flow** (prompt injection) **[NEW 2026-02-09]**:
```
Query → ProposalFilter.get_proposals(query, limit=3)
    ↓ project-relevance keyword check (fast skip for non-project queries)
    ↓ ProposalStore query with utility query (GOALS.md-based)
    ↓ keyword dedup → semantic dedup
    ↓ gate system scoring → composite scoring (priority+breadth+recency+goal)
    ↓ novelty penalty (git commit overlap) → diversity selection
    ↓ optional LLM pairwise ranking (tournament bracket)
→ [PROPOSED FEATURES] section in prompt
```

**Tests**: 71+ unit tests across 5 files (test_code_proposal.py, test_proposal_generator.py, test_proposal_store.py, test_proposal_filter.py, test_shutdown_pipeline_proposals.py)

**Dependencies**: ModelManager, MultiCollectionChromaStore, config.app_config, MultiStageGateSystem (optional), MemoryCoordinator (optional)

---

### 2.12.5b core/prompt/proposal_filter.py (Proposal Prompt Injection Pipeline) **[NEW 2026-02-09]**
**Purpose**: Retrieval, dedup, gating, and ranking of code proposals for injection into the `[PROPOSED FEATURES]` prompt section

**Module Contract**:
```
Purpose: Surface the most valuable pending/approved proposals in prompts for project-related queries
Inputs: User query, optional limit
Outputs: List of scored proposal dicts [{content, metadata, relevance_score}]
Side effects: Reads GOALS.md, calls git log, optionally calls LLM for pairwise ranking
Dependencies: ProposalStore, MultiStageGateSystem (optional), ModelManager (optional)
```

**Pipeline** (in `get_proposals()`):
1. **Project-relevance check**: Fast keyword intersection with `_PROJECT_KEYWORDS` (50+ project terms). Non-project queries (casual/personal) return empty immediately
2. **Utility query**: Reads GOALS.md active goals, builds composite query for goal-aligned retrieval. Cached with MD5 file-hash invalidation
3. **ProposalStore query**: Semantic search for top 20 pending/approved proposals
4. **Keyword dedup**: Pairwise Jaccard on tags (threshold 0.60) + title word overlap (50%) — keeps higher priority
5. **Semantic dedup**: sentence-transformers cosine similarity (threshold 0.75) — keeps higher priority
6. **Gate scoring**: MultiStageGateSystem for goal_alignment signal (one of four scoring factors)
7. **Composite scoring**: `w_priority(0.30) × priority + w_breadth(0.20) × breadth + w_recency(0.10) × recency + w_goal(0.40) × goal_alignment`
8. **Novelty penalty**: Compares proposal title words against recent git commit messages (Jaccard). High overlap → multiplicative penalty (0.5-1.0 factor)
9. **Diversity selection**: Greedy selection using overlap coefficient (|intersection|/|smaller set|, threshold 0.34) — prevents topic clustering
10. **Optional LLM pairwise ranking**: Tournament bracket comparison of top 6 candidates

**Key Methods**:
- `get_proposals(query, limit=3)` → Main entry point, runs full pipeline
- `_is_project_related(query)` → Fast keyword check (static)
- `_build_utility_query()` → GOALS.md-derived query (cached)
- `_keyword_dedup(proposals, threshold=0.60)` → Tag + title dedup (static)
- `_semantic_dedup(proposals, threshold=0.85)` → Embedding-based dedup (static)
- `_compute_composite_score(proposal, goal_alignment, w_*)` → Weighted composite (static)
- `_score_priority(proposal)` → Normalize 1-10 to 0.0-1.0
- `_score_breadth(proposal)` → Files + dirs + tags diversity
- `_score_recency(proposal)` → Exponential decay (14-day half-life)
- `_score_novelty(proposal)` → Git commit overlap check
- `_diverse_select(scored_dicts, limit, overlap_threshold)` → Anti-clustering selection (static)
- `_llm_pairwise_rank(proposals, limit)` → Tournament bracket LLM ranking

**Integration**:
- `ContextGatherer.get_proposed_features(query, limit)` → Calls `ProposalFilter.get_proposals()`
- Builder adds `[PROPOSED FEATURES]` section after `[ADAPTIVE WORKFLOWS]`
- TokenManager includes `proposed_features` at priority 3 (trimmed before core context)
- Formatter `_sanitize_embedded_headers()` includes `[PROPOSED FEATURES]` pattern

**Dependencies**: ProposalStore, MultiStageGateSystem (lazy-init), ModelManager (optional, for LLM ranking)

---

### 2.12.6 utils/daily_notes_generator.py (Auto-Generated Daily Summaries) **[ENHANCED 2026-01-22]**
**Purpose**: Automatically generate daily summary notes from Daemon conversations with LLM-based tag generation, written from Daemon's perspective

**Data Classes**:
- `GenerationResult`: Statistics from note generation (success, date, output_path, conversation_count, intensity, skipped_reason)

**Key Methods**:
- `generate_for_date(date, force)` → `GenerationResult`: Generate note for specific date
- `generate_yesterday_if_missing()` → `Optional[GenerationResult]`: Startup catch-up hook
- `note_exists(date)` → `bool`: Check if note already exists
- `_get_conversations_for_date(date)` → `List[Dict]`: Filter corpus by timestamp
- `_format_conversations(convos)` → `str`: Format for LLM prompt
- `_calculate_active_duration(convos)` → `float`: Estimate actual usage time in hours **[NEW 2026-01-18]**
- `_calculate_intensity(convos, active_hours)` → `int`: 1-10 score based on count/active duration/complexity

**Active Duration Calculation** [NEW 2026-01-18]:
Estimates actual user engagement time (vs wall-clock span):
```python
# For each conversation:
# - Reading time: ~200 words/min for response (~1000 chars/min)
# - Typing time: ~40 words/min for query (~200 chars/min)
# - Gap time: Capped at 30 seconds between exchanges (idle time excluded)
# - Minimum: 10 seconds per exchange

# Example: 8-hour span (10am-6pm) might = 1.5 hours active
```

**Note Structure** (Obsidian-compatible markdown):
```markdown
---
date: 2026-01-16
usage_intensity: 7
conversations: 18
span_hours: 8.5
active_hours: 1.8
main_quest: "Daily Notes Feature"
tags: [daily, daemon-generated, coding, productivity, learning, focused] **[ENHANCED 2026-01-22]**
---
# Daily Note - January 16, 2026
## Summary (2-3 sentences from Daemon's perspective)
## Main Quest: [Primary Focus] (3-5 bullets)
## Side Quests (other topics discussed)
## Life Events [NEW 2026-01-18]
- **Work**: Duration, what was done, how it went. "Not discussed today" if not mentioned.
- **Study**: Subject/material, duration, progress. "Not discussed today" if not mentioned.
- **Sleep**: Quality, duration, issues. "Not discussed today" if not mentioned.
- **Exercise/Health**: Activities, outcomes. "Not discussed today" if not mentioned.
- **Other Events**: Social activities, appointments, errands.
## Emotional State (mood tracking)
## Key Decisions (explicit choices made)
## Knowledge Gained (new concepts)
## Open Threads (unresolved items)
## Intensity: X/10
```

**Frontmatter Schema Update** [2026-01-18]:
- `intensity` → `usage_intensity` (clarifies what it measures)
- `duration_hours` → split into `span_hours` + `active_hours`

**LLM Prompt Design**:
- Written from Daemon's perspective ("Today we...", "Luke seemed...")
- Extracts Main Quest vs Side Quests (RPG-style framing)
- Tracks emotional state throughout the day
- Uses `sonnet-4.5` with 800 max tokens

**Folder Structure** **[UPDATED 2026-03-10]**:
Notes now use a 3-level hierarchy: `<Month YYYY>/<Week N Mon YYYY>/<filename>` instead of flat `<Week N Mon YYYY>/`.
`note_exists()` checks all 3 locations (flat, weekly-at-root, monthly/weekly) for backward compatibility.

**Configuration** (`config/app_config.py`):
```python
DAILY_NOTES_ENABLED = True
DAILY_NOTES_FOLDER = "Vault/Daily Notes and To Do's"
DAILY_NOTES_MODEL = "sonnet-4.5"   # was gpt-4o-mini [UPDATED 2026-03-10]
DAILY_NOTES_MAX_TOKENS = 800
TAG_GENERATION_ENABLED = True  **[NEW 2026-01-22]**
TAG_GENERATION_MODEL = "sonnet-4.5"  # was gpt-4o-mini [UPDATED 2026-03-10]
TAG_GENERATION_MAX_TAGS = 10  **[NEW 2026-01-22]**
TAG_GENERATION_MIN_TAGS = 3  **[NEW 2026-01-22]**
```

**CLI Commands** (`main.py`):
- `python main.py daily-note` - Generate for today
- `python main.py daily-note yesterday` - Generate for yesterday
- `python main.py daily-note 2026-01-15` - Generate for specific date
- `python main.py daily-note --force` - Overwrite existing
- `python main.py daily-note-catchup` - Startup hook (yesterday if missing)

**Filename Convention**: Matches user's existing format: `M D YY Daily Note.md` (e.g., `1 16 26 Daily Note.md`)

**Scheduling**:
- Cron (recommended): `0 2 * * * cd /path/to/daemon && python main.py daily-note yesterday`
- GUI startup catch-up [NEW 2026-01-18]: `_run_daily_notes_catchup()` in gui/launch.py
  - Runs in background thread on GUI launch (non-blocking)
  - Calls `generate_yesterday_if_missing()`
  - Respects `DAILY_NOTES_ENABLED` config flag
  - Errors logged but don't affect GUI startup

**Integration**:
- Writes to Obsidian vault at `OBSIDIAN_VAULT_PATH / DAILY_NOTES_FOLDER`
- Notes can be retrieved via ObsidianManager once embedded
- Atomic file writes (temp → replace pattern)
- **Tag generation** [NEW 2026-01-22]: Analyzes note content using TagGenerator to extract 3-10 contextual tags (life domains, activities, emotions, topics, productivity states, meta)

---

### 2.12.6 utils/weekly_notes_generator.py (Auto-Generated Weekly Summaries) **[ENHANCED 2026-01-22]**
**Purpose**: Organize daily notes into weekly folders and generate weekly summaries with LLM-based tag generation by aggregating daily notes

**Data Classes**:
- `WeeklyGenerationResult`: Statistics from note generation (success, week_num, year, week_folder, output_path, daily_notes_found, daily_notes_moved, total_conversations, avg_intensity, skipped_reason)

**Key Methods**:
- `generate_for_week(date, force)` → `WeeklyGenerationResult`: Generate summary for week containing date
- `generate_last_week_if_complete()` → `Optional[WeeklyGenerationResult]`: Startup catch-up hook
- `week_summary_exists(date)` → `bool`: Check if summary already exists
- `_get_daily_notes_for_week(date)` → `List[Dict]`: Find daily notes in week range
- `_move_daily_notes_to_week_folder(notes, week_folder)`: Organize notes into folders
- `_parse_daily_note(path)` → `Dict`: Read and parse daily note (frontmatter + content)

**Folder Structure** **[UPDATED 2026-03-10]**:
```
Obsidian vault/
└── Daily Notes and To Do's/
    └── January 2026/                    # Monthly parent folder [NEW 2026-03-10]
        └── Week 3 Jan 2026/
            ├── 1 13 26 Daily Note.md
            ├── 1 14 26 Daily Note.md
            └── Week 3 Jan 2026 Summary.md
        └── January 2026 Summary.md     # Monthly summary [NEW 2026-03-10]
```

**Note Structure** (Obsidian-compatible markdown):
```markdown
---
week_num: 3
year: 2026
avg_usage_intensity: 6.5
total_conversations: 42
total_active_hours: 8.5
daily_notes: 5
date_range: "2026-01-13 to 2026-01-19"
tags: [weekly, daemon-generated, work, study, productivity, learning, health, exercise] **[ENHANCED 2026-01-22]**
---
# Weekly Summary - Week 3, January 2026
## Week at a Glance (3-4 sentences)
## Main Quests This Week (aggregated from daily notes)
## Life Events Summary [NEW 2026-01-18]
- **Work**: Aggregated across week
- **Study**: Aggregated across week
- **Sleep**: Patterns noted
- **Exercise/Health**: Activities summarized
## Recurring Themes
## Emotional Arc
## Lessons Learned
## Open Threads → Next Week
```

**Backward Compatibility** [UPDATED 2026-01-18]:
- Reads both old field names (intensity, duration_hours) and new (usage_intensity, span_hours, active_hours)
- Writes new field names (avg_usage_intensity, total_active_hours)

**CLI Commands** (`main.py`):
- `python main.py weekly-note` - Generate for current week
- `python main.py weekly-note 2026-01-12` - Generate for week containing date
- `python main.py weekly-note --force` - Overwrite existing
- `python main.py weekly-note-catchup` - Generate last week if missing

**Folder Structure** **[UPDATED 2026-03-10]**:
Notes now use monthly parent folders: `<Month YYYY>/<Week N Mon YYYY>/`. Weekly generator searches 3 paths for daily notes: flat (legacy), weekly-at-root (legacy), monthly/weekly (new).

**Configuration** (`config/app_config.py`):
```python
WEEKLY_NOTES_ENABLED = True
WEEKLY_NOTES_MODEL = "sonnet-4.5"   # was gpt-4o-mini [UPDATED 2026-03-10]
WEEKLY_NOTES_MAX_TOKENS = 1200
```

**Integration**:
- Writes to Obsidian vault at `OBSIDIAN_VAULT_PATH / DAILY_NOTES_FOLDER / <Month YYYY> / Week N Month Year/`
- Week summaries can be read by narrative context generator
- Atomic file writes (temp → replace pattern)
- **Tag generation** [NEW 2026-01-22]: Analyzes aggregated weekly content using TagGenerator to extract 5-10 contextual tags capturing recurring themes, patterns, and activities

---

### 2.12.6b utils/monthly_notes_generator.py (Auto-Generated Monthly Summaries) **[NEW 2026-03-10]**
**Purpose**: Generate monthly summary notes from daily notes, organize weekly folders into monthly parent folders

**Key Methods**:
- `generate_for_month(year, month, force)` → Generate monthly summary from daily notes in that month
- `migrate_weekly_folders()` → Move legacy root-level `Week N` folders into `<Month YYYY>/` parents
- `_find_daily_notes_for_month(year, month)` → Find all daily notes within month range

**Monthly Summary Structure**:
- LLM-synthesized from all daily notes in the month
- Covers: month overview, major themes, emotional arc, accomplishments, open threads

**CLI Commands** (`main.py`):
- `python main.py monthly-note` - Generate for last month
- `python main.py monthly-note 2026-02` - Generate for specific month (YYYY-MM)
- `python main.py monthly-note --force` - Overwrite existing
- `python main.py monthly-note-catchup` - Migrate weekly folders + generate last month
- `python main.py migrate-monthly` - Just run weekly-to-monthly folder migration

**Configuration** (`config/app_config.py`):
```python
MONTHLY_NOTES_ENABLED = True
MONTHLY_NOTES_MODEL = "sonnet-4.5"
MONTHLY_NOTES_MAX_TOKENS = 2000
```

**GUI Startup**: `_run_monthly_notes_catchup()` runs in background thread on GUI launch — migrates legacy weekly folders, then generates last month's summary if complete.

---

### 2.12.7 utils/tag_generator.py (LLM-Based Tag Generation) **[NEW 2026-01-22]**
**Purpose**: Generate contextual Obsidian tags for notes using LLM analysis with consistent vocabulary. Future-ready for .md-based memory migration with tags as 4th filtering stage.

**Data Classes**:
- `TagGenerationResult`: Result of tag generation (tags, tag_count, known_tags, custom_tags, skipped_tags, error)

**Key Methods**:
- `generate_tags(content, note_type, metadata)` → `TagGenerationResult`: Analyze content and generate 3-10 tags
- `get_tag_categories()` → `Dict[str, Set[str]]`: Return tag vocabulary by category
- `get_all_known_tags()` → `Set[str]`: Return all known tags
- `_normalize_tag(tag)` → `str`: Convert to Obsidian format (lowercase, hyphenated)
- `_validate_tag(tag)` → `bool`: Check tag quality (length, stopwords)
- `_parse_llm_tags(response)` → `List[str]`: Parse LLM output (comma/newline/numbered)
- `_generate_fallback_tags(note_type, metadata)` → `List[str]`: Heuristic fallback if LLM fails

**Tag Vocabulary** (100+ tags across 6 categories):
1. **Life domains** (17 tags): work, study, health, exercise, sleep, social, family, relationships, dating, friends, hobbies, finances, career, education, wellness, mental-health, physical-health
2. **Activities** (18 tags): coding, programming, learning, reading, writing, research, debugging, building, designing, planning, gaming, travel, cooking, art, music, sports, meditation, journaling
3. **Emotions** (23 tags): stress, anxiety, happy, sad, frustrated, excited, calm, motivated, tired, energized, confused, confident, worried, optimistic, pessimistic, grateful, angry, peaceful, lonely, content, overwhelmed, focused, distracted
4. **Productivity** (15 tags): productive, unproductive, deep-work, flow-state, procrastinating, efficient, struggling, breakthrough, blocked, progress, achievement, setback, milestone, deadline, time-management
5. **Topics** (23 tags): ai, machine-learning, programming, python, javascript, web-dev, data-science, algorithms, math, science, physics, biology, history, philosophy, psychology, economics, politics, technology, linguistics, literature, art-history, music-theory, engineering
6. **Meta** (16 tags): crisis, decision, reflection, planning, goal-setting, review, brainstorming, problem-solving, question, insight, realization, important, follow-up, unresolved, completed, archived

**LLM Prompt Strategy**:
- Analyzes note content + metadata (main quest, intensity, conversations, duration)
- Shows known vocabulary to encourage consistency
- Allows custom tags for new concepts not in vocabulary
- Requests 5-10 relevant tags in comma-separated format
- Uses `gpt-4o-mini` with temperature=0.3 for consistent but slightly creative extraction

**Tag Format** (Obsidian-compatible):
- Lowercase with hyphens instead of spaces (e.g., `deep-work`, `mental-health`)
- No special characters
- Quoted in YAML frontmatter to handle hyphens
- Example: `tags: ["daily", "daemon-generated", "coding", "productive", "focused"]`

**Validation Pipeline**:
1. Parse LLM response (handles comma/newline/numbered formats)
2. Normalize tags (lowercase, hyphenate, remove special chars)
3. Validate quality (min 2 chars, filter stopwords)
4. Categorize as known vs custom
5. Enforce min/max limits (3-10 tags)

**Fallback Behavior** (if LLM fails):
- Heuristic tags based on note type (daily → reflection, weekly → review)
- Intensity-based tags (high → productive, low → quiet)
- Keyword extraction from main quest/topic
- Ensures minimum tag count with default tags (conversation, general, reflection)

**Configuration** (`config/app_config.py`):
```python
TAG_GENERATION_ENABLED = True
TAG_GENERATION_MODEL = "gpt-4o-mini"
TAG_GENERATION_MAX_TAGS = 10
TAG_GENERATION_MIN_TAGS = 3
```

**Integration**:
- Used by `DailyNotesGenerator` to tag daily summaries
- Used by `WeeklyNotesGenerator` to tag weekly summaries
- Future: .md-based memory files with tags as 4th filtering stage (semantic/keyword/temporal/tag)
- Graceful degradation: failures don't block note generation, falls back to system tags only

**Use Cases**:
- Obsidian tag search: Find all notes tagged `#stress` or `#coding`
- Pattern tracking: "How often do I discuss `#mental-health`?"
- Multi-tag filtering: `#productivity AND #deep-work`
- Tag trend analysis: Identify behavioral patterns over time
- Future memory filtering: Fast tag-based prefiltering before expensive semantic search

---

### 2.13 utils/time_manager.py (Temporal Context & Session Tracking)
**Purpose**: Time-aware operations, decay calculation, and conversation pacing metrics **[ENHANCED 2025-12-05]**

**Key Functions**:
- `current()` → datetime.now()
- `current_iso()` → ISO-8601 string
- `calculate_decay(age_hours, decay_rate)` → float
  ```python
  decay = 1.0 / (1.0 + decay_rate * age_hours)
  ```
- `mark_query_time()` → Record message timestamp, detect session boundaries **[NEW]**
- `mark_session_end()` → Record session end time for gap tracking **[NEW]**
- `time_since_previous_message()` → Calculate gap between consecutive messages **[NEW]**
- `elapsed_since_last_session()` → Calculate gap since last session ended **[NEW]**

**Persistence**:
- `data/last_query_time.json` - Stores last message time + previous message time
- `data/last_session_time.json` - Stores session end timestamp **[NEW]**
- Survives TimeManager instance recreation (shared state across prompt builder, formatter, response generator)

**Session Detection**:
- Compares `last_query_time` vs `last_session_end_time` to detect new sessions
- Resets `previous_query_time` to None when starting a new session
- Enables "N/A (first message in session)" vs actual time deltas

**Used By**:
- MemoryCoordinator for recency scoring
- PromptFormatter for TIME CONTEXT display **[NEW]**
- Shutdown handlers for session boundary marking **[NEW]**

---

### 2.14 memory/memory_consolidator.py (Summarization + Narrative Synthesis)
**Purpose**: LLM-based conversation summarization and 3-tier narrative context synthesis (monthly/weekly/daily)

**Trigger**: Every N conversations (default 20) at shutdown

**Process**:
```python
1. Collect last N unsummarized conversations
2. Build summarization prompt
3. Call LLM (default: sonnet-4.5)
4. Store summary as SUMMARY memory type
5. Link to parent conversations
```

**Key Methods**:
- `consolidate_memories(conversations, max_tokens)` → async summary
- `generate_narrative_context(weeklies, monthlies)` → async narrative synthesis **[ENHANCED 2026-03-10]**
- `_read_obsidian_monthly_summaries(limit)` → List[Dict] from `<Month YYYY> Summary.md` files **[NEW 2026-03-10]**
- `_read_obsidian_weekly_summaries(limit)` → List[Dict] from Obsidian vault (searches both root-level and monthly-parent folders)
- `_read_obsidian_daily_notes(limit)` → List[Dict] from weekly folders (searches both root and monthly parent paths)

**Narrative Context Synthesis** [ENHANCED 2026-03-10]:
Generates a "Current Life State" narrative from 3-tier notes (monthly/weekly/daily) for temporal grounding.
- Primary source: Obsidian monthly summaries + weekly summaries + daily notes
- Monthly for arc, weekly for active threads, daily for recent shifts
- Fallback: Corpus summaries
- Output: ~300 word synthesis covering life chapter, active threads, emotional trajectory, recurring themes
- Cached to `./data/narrative_context.txt` (0ms retrieval latency)
- CLI: `python main.py refresh-narrative`

---

### 2.14.1 Narrative Context System (Temporal Grounding) **[ENHANCED 2026-03-10]**

**Purpose**: Provide trajectory-aware context by synthesizing monthly/weekly/daily notes into a cached "Life State" narrative.

**Architecture Decision**: Background cached updates (not per-query synthesis)
```
Per-Query Synthesis: +3-5s latency, ~5k tokens/query  ❌
Background Cached:   0ms latency, ~500 tokens/query   ✅
```

**Data Sources (3-Tier Hierarchy)** **[UPDATED 2026-03-10]**:
| Source | Priority | Count | Location |
|--------|----------|-------|----------|
| Obsidian Monthly Summaries | Primary | 1 | `<Month YYYY>/<Month YYYY> Summary.md` |
| Obsidian Weekly Summaries | Primary | 3 | `<Month YYYY>/Week */Week * Summary.md` (+ root-level legacy) |
| Obsidian Daily Notes | Primary | 6 | `<Month YYYY>/Week */*Daily Note.md` (+ root-level legacy) |
| Corpus Summaries | Fallback | 3-5 | `corpus_v4.json` summaries |

**Note**: Reads daily/weekly notes from both monthly-parent and root-level (legacy) paths for backward compatibility.

**Output Structure**:
```markdown
**Current Life State**
The user is navigating [life phase description]...

**Active Threads**
- Project/goal 1
- Project/goal 2
- Concern/focus 3

**Emotional Trajectory**
Mood trending [direction] because [reasons]...

**Recurring Themes**
Patterns across multiple time periods...
```

**Update Triggers**:
1. **Primary**: Daily note creation → `_trigger_narrative_refresh()` in DailyNotesGenerator
2. **Secondary**: After summary consolidation (memory_storage.py)
3. **Fallback**: Startup check logs warning if >24 hours stale
4. **Manual**: `python main.py refresh-narrative`

**Files Modified**:
- `memory/corpus_manager.py` - `save_narrative_context()`, `get_narrative_context()`
- `memory/memory_consolidator.py` - `generate_narrative_context()`, Obsidian readers
- `utils/daily_notes_generator.py` - `_trigger_narrative_refresh()` primary trigger
- `memory/memory_storage.py` - `_maybe_regenerate_narrative()` secondary trigger
- `core/prompt/context_gatherer.py` - `get_narrative_context()` retrieval
- `core/prompt/builder.py` - Context gathering and assembly
- `core/prompt/token_manager.py` - Priority 8, 500 token cap
- `core/orchestrator.py` - `_check_narrative_freshness()` startup check
- `config/app_config.py` - `NARRATIVE_CONTEXT_*` settings

**Configuration**:
```python
NARRATIVE_CONTEXT_ENABLED = True      # Feature toggle
NARRATIVE_CONTEXT_PATH = "./data/narrative_context.txt"
NARRATIVE_MAX_TOKENS = 500            # Hard cap in prompt
NARRATIVE_SYNTHESIS_MODEL = "sonnet-4.5"  # was gpt-4o-mini [UPDATED 2026-03-10]
NARRATIVE_WEEKLIES_COUNT = 3          # was 4 [UPDATED 2026-03-10]
NARRATIVE_MONTHLIES_COUNT = 1         # [NEW 2026-03-10]
NARRATIVE_DAILIES_COUNT = 6           # [NEW 2026-03-10]
```

---

### 2.15 memory/fact_extractor.py (Entity/Fact Extraction)
**Purpose**: Extract facts from conversations for semantic memory

**Methods**:
1. **Regex patterns** (fast):
   - "X is Y" patterns
   - "X does Y" patterns
   - Entity definitions

2. **LLM extraction** (optional):
   - For complex facts
   - Calls LLMFactExtractor

**Key Methods**:
- `extract_facts(conversation)` → List[Dict]
  ```python
  {
    "entity": "Python",
    "fact": "is a programming language",
    "confidence": 0.8
  }
  ```

---

### 2.16 utils/tone_detector.py (Crisis Detection)
**Purpose**: Detect distress/crisis language to adjust response tone

**Crisis Levels** (CrisisLevel enum):
- `CONVERSATIONAL` - Default friend mode (85% of conversations)
- `CONCERN` - Light support, brief validation (10%)
- `MEDIUM` - Elevated support, empathetic (4%)
- `HIGH` - Crisis support, full therapeutic mode (1%)

**3-Stage Detection Pipeline** [ENHANCED 2025-12-09]:
1. **Observational check** - Is user discussing world events vs personal crisis?
2. **Harm scoring** - Composite keyword system (replaces "first keyword wins")
   - Scans entire message for ALL crisis indicators
   - Accumulates weighted points: HIGH (10pts), MEDIUM (5pts), CONCERN (2pts)
   - Applies pattern multipliers for dangerous combinations (1.2x-1.4x)
   - Routes based on thresholds: ≥20 HIGH, ≥10 MEDIUM, ≥4 CONCERN
3. **Semantic similarity** - Compare embeddings to crisis exemplars (fallback for nuanced language)
4. **LLM fallback** - For borderline cases near thresholds

**Harm Scoring System** (NEW):
- **250+ keywords** across HIGH/MEDIUM/CONCERN categories
- **Keyword points**:
  - HIGH (10 pts): Suicidal ideation, self-harm, severe crying ("want to die", "sobbing", "peel off my skin")
  - MEDIUM (5 pts): Panic attacks, dissociation, abuse/trauma, substance relapse ("breaking down", "flashback", "abusive")
  - CONCERN (2 pts): Anxiety, stress, emotional distress ("anxious", "lonely", "worried")
- **Pattern multipliers** (for dangerous combinations):
  - Hopelessness + suicidal ideation: 1.4x
  - Self-harm + crying: 1.3x
  - Dissociation + trauma: 1.3x
  - Substance relapse + crisis: 1.3x
  - Multiple HIGH indicators: 1.2x
  - Abuse + distress: 1.2x
  - Sleep deprivation + mental distress: 1.2x

**Example Scoring**:
```python
Message: "I'm sobbing. She was abusive. I want to peel off all my skin."
Keywords matched:
  - "sobbing" (HIGH): 10 pts
  - "peel off all my skin" (HIGH): 10 pts
  - "abusive" (MEDIUM): 5 pts
Base score: 25 pts

Pattern multipliers:
  - Self-harm + crying: 1.3x
  - Abuse + distress: 1.2x
Total multiplier: 1.56x

Final score: 25 × 1.56 = 39 pts → HIGH crisis (≥20)
```

**Configuration** (env vars):
```python
# Semantic thresholds (for fallback when harm score < 4)
TONE_THRESHOLD_HIGH = 0.58
TONE_THRESHOLD_MEDIUM = 0.50
TONE_THRESHOLD_CONCERN = 0.43
TONE_CONTEXT_WINDOW = 3  # Prior turns to check
TONE_ESCALATION_BOOST = 1.2  # Boost if prior distress detected
```

**Key Methods**:
- `detect_crisis_level(message, conversation_history, model_manager)` → async ToneAnalysis
- `_calculate_harm_score(message)` → Composite scoring with pattern multipliers
- `_check_keyword_crisis(message)` → Harm score-based routing
- `_semantic_crisis_detection(message, history)` → Embedding similarity
- `_llm_crisis_fallback(message, model_manager)` → LLM classification

**Returns**:
```python
ToneAnalysis(
  level=CrisisLevel.HIGH,
  confidence=1.0,
  trigger="harm_score: 39.0 (2H, 1M, 0C)",
  raw_scores={},
  explanation="Explicit crisis language detected: harm_score: 39.0 (2H, 1M, 0C)"
)
```

---

### 2.17 utils/need_detector.py (Need-Type Detection)
**Purpose**: Detect what type of response the user needs (presence vs perspective)

**Need Types** (NeedType enum):
- `PRESENCE` - User needs warmth, acknowledgment, "I'm here with you"
- `PERSPECTIVE` - User wants engagement, questions, reframes, problem-solving
- `NEUTRAL` - Mixed or unclear signals

**Detection Strategy**:
- **PRESENCE markers**: Short emotional statements ("I am lonely", "ugh"), no questions, high emotional word density
- **PERSPECTIVE markers**: Problem-framing ("I think", "should I"), questions, causal/contrastive connectors ("because", "but")

**2-Stage Detection Pipeline**:
1. **Keyword detection** (fast path ~1ms) - Pattern matching + structural signals (length, punctuation)
2. **Semantic detection** (~50ms) - Embedding similarity to exemplar prototypes (20 examples per type)

**Hybrid Combination**:
```python
# Weighted scoring
keyword_weight = 0.4
semantic_weight = 0.6

# Agreement boost: if both detect same type, boost confidence by +0.1
# Disagreement: use higher confidence result, reduce by 0.8x
```

**Configuration** (env vars):
```python
NEED_THRESHOLD_PRESENCE = 0.60
NEED_THRESHOLD_PERSPECTIVE = 0.60
NEED_KEYWORD_WEIGHT = 0.4
NEED_SEMANTIC_WEIGHT = 0.6
NEED_HIGH_CONF = 0.8  # Fast-path threshold
```

**Key Methods**:
- `detect_need_type(message, model_manager)` → NeedAnalysis
- `_keyword_need_detection(message)` → Fast keyword check
- `_semantic_need_detection(message, model_manager)` → Embedding similarity
- `_combine_scores(keyword_result, semantic_result)` → Weighted hybrid

**Returns**:
```python
NeedAnalysis(
  need_type=NeedType.PRESENCE,
  confidence=0.83,
  trigger="keyword",
  raw_scores={"presence": 1.5, "perspective": 0.0},
  explanation="Keyword presence signals: 1.5"
)
```

**Embedder Sharing**: Uses same sentence-transformers model as tone_detector to avoid duplicate memory usage.

---

### 2.18 utils/emotional_context.py (Combined Emotional Analysis)
**Purpose**: Unified emotional analysis combining severity (tone) and need-type

**Combines**:
- Tone detector (CrisisLevel): CONVERSATIONAL → CONCERN → MEDIUM → HIGH
- Need detector (NeedType): PRESENCE vs PERSPECTIVE vs NEUTRAL

**Data Structure**:
```python
@dataclass
class EmotionalContext:
    crisis_level: CrisisLevel      # From tone_detector
    need_type: NeedType            # From need_detector
    tone_confidence: float
    need_confidence: float
    tone_trigger: str
    need_trigger: str
    explanation: str
```

**Key Method**:
- `analyze_emotional_context(message, conversation_history, model_manager)` → async EmotionalContext

**Integration with Orchestrator**:
1. Orchestrator calls `analyze_emotional_context()` early in prepare_prompt
2. Stores result in `self.current_emotional_context`
3. Uses `_get_response_instructions(ctx)` to generate mode-specific instructions
4. Injects combined instructions into system prompt

**Response Mode Matrix**:
| Crisis Level | Need Type | Instructions |
|--------------|-----------|--------------|
| HIGH | any | Crisis support (safety first, ignores need type) |
| MEDIUM | PRESENCE | Base elevated support + "Lead with warmth first" |
| MEDIUM | PERSPECTIVE | Base elevated support + "Engage with their framing" |
| CONCERN | PRESENCE | Base light support + "Brief acknowledgment, stay present" |
| CONCERN | PERSPECTIVE | Base light support + "Light engagement, offer reframes" |
| CONVERSATIONAL | PRESENCE | Base casual + "Short, warm responses preferred" |
| CONVERSATIONAL | PERSPECTIVE | Base casual + "Questions and reframes welcome" |
| any | NEUTRAL | Base tone instructions only |

---

### 2.19 utils/query_checker.py (Query Analysis)
**Purpose**: Fast query analysis for routing and heavy topic detection

**Key Functions**:
- `analyze_query(q)` → QueryAnalysis dataclass
- `is_deictic(query)` → True if refers to earlier context **[ENHANCED 2026-01-08]**
  - Threshold raised from 6 to 10 words for short follow-up detection
  - Added verb+pronoun patterns: "watched it", "saw it", "read it", "heard it", etc.
- `is_meta_conversational(query)` → True if asking about conversation history
- `extract_temporal_window(query)` → Days to look back (e.g., "yesterday" → 1)
- `_is_heavy_topic_heuristic(q)` → Fast keyword check for crisis/sensitive content
- `analyze_query_async(q, model_manager)` → Async with LLM heavy topic classification

**Heavy Topic Keywords** (triggers inline fact extraction):
- Political violence: raid, deportation, protest, military
- Crisis/trauma: suicide, panic attack, breakdown
- Human rights: persecution, discrimination, refugee
- Mental health: depression, anxiety, PTSD
- Emotional distress: breakup, grief, job loss

**Thread Detection**:
- `belongs_to_thread(current_query, last_conversation, current_topic)` → bool
- `calculate_thread_continuity_score()` → 0.0-1.0 score
- Factors: keyword overlap, time proximity, both heavy, same topic, deictic reference **[UPDATED 2026-01-08]**
- `THREAD_WEIGHT_DEICTIC = 0.25` → Bonus for deictic follow-ups **[NEW]**
- `THREAD_CONTINUITY_THRESHOLD = 0.5`

**QueryAnalysis Result**:
```python
QueryAnalysis(
  text="...",
  tokens=["..."],
  is_question=True,
  is_command=False,
  is_deictic=False,
  is_followup=False,
  token_count=8,
  char_count=45,
  intents={"question"},
  is_heavy_topic=True,
  is_meta_conversational=False
)
```

---

### 2.20 memory/hybrid_retriever.py (Hybrid Search)
**Purpose**: Combine query rewriting, semantic search, and keyword matching

**3-Tier Approach**:
1. **Query rewriting** - Expand casual queries with synonyms
2. **Semantic search** - ChromaDB embedding similarity
3. **Keyword matching** - Exact term boosting

**Scoring Weights**:
```python
semantic_weight = 0.7
keyword_weight = 0.3
hybrid_score = semantic_weight * semantic_score + keyword_weight * keyword_score
```

**Key Methods**:
- `retrieve(query, limit)` → List of memories with hybrid scores
- `_semantic_search(query, n_results)` → Query multiple ChromaDB collections
- `_keyword_match(query, candidates)` → Score with keyword overlap
- `_hybrid_score(semantic_results, keyword_results, query)` → Combine scores

**Collections Queried**: conversations, summaries, reflections

**Dependencies**: MultiCollectionChromaStore, query_rewriter, keyword_matcher

---

### 2.21 utils/query_rewriter.py (Query Expansion)
**Purpose**: Expand casual/slang queries for better retrieval

**Key Functions**:
- `rewrite_query(query)` → Expanded query with synonyms
- `extract_keywords(query)` → List of meaningful keywords

---

### 2.22 utils/keyword_matcher.py (Term Matching)
**Purpose**: Calculate keyword overlap scores for retrieval boosting

**Key Functions**:
- `calculate_keyword_score(keywords, candidate)` → 0.0-1.0 score

---

### 2.23 Memory Citation System **[NEW 2025-12-04]**
**Purpose**: Track and display which memories inform LLM responses for transparency and professional deployment

**Architecture**:
```
Query → Context Retrieval → Memory ID Tracking
                              ↓
                         memory_id_map built
                              ↓
                         System prompt injection (citation instructions)
                              ↓
                         LLM generates response with [MEM_X] tags
                              ↓
                         Citation extraction (handlers.py)
                              ↓
                         Clean response + Citations JSON
```

**Components**:

**1. core/prompt/context_gatherer.py**:
- `memory_id_map` dict: Maps citation IDs to memory metadata (hybrid relative/absolute ID system)
- Tracks during retrieval:
  - `MEM_RECENT_{idx}` → Recent conversation memories
  - `MEM_SEMANTIC_{idx}` → Semantic memory hits
  - `SUM_RECENT_{idx}` → Recent summaries
  - `SUM_SEMANTIC_{idx}` → Semantic summary hits
  - `REFL_RECENT_{idx}` → Recent reflections
  - `REFL_SEMANTIC_{idx}` → Semantic reflection hits
  - `PROFILE_CONTEXT` → User profile facts
- Metadata: `{type, timestamp, content[:500], relevance_score, db_id}`
  - `db_id`: Absolute database ID (UUID or generated ID) for traceability to ChromaDB/corpus records

**2. core/prompt/builder.py**:
- Passes `memory_id_map` through context dict
- Available in `context["memory_id_map"]`

**3. core/orchestrator.py**:
- `enable_citations` flag (set from GUI checkbox)
- `citation_pattern` regex: `r'\[(MEM_\w+_\d+|FACT_\d+|SUM_\d+|PROFILE_\w+)\]'`
- `_extract_citations(response, memory_map)` → (clean_response, citations_list)
  - Finds citation tags via regex
  - Builds citation metadata list
  - Removes tags from response
  - Returns clean text + citations
- System prompt injection when `enable_citations=True`:
  ```
  MEMORY CITATION PROTOCOL:
  When referencing specific memories, include inline citations:
  - [MEM_RECENT_{index}] for recent conversations
  - [MEM_SEMANTIC_{index}] for semantic memories
  - [PROFILE_CONTEXT] for user profile facts
  ```

**4. gui/handlers.py**:
- Citation extraction in all response paths:
  - Default streaming
  - Best-of/Duel mode
  - Fallback streaming
- Adds to debug_info: `{citations: [...], citations_enabled: bool}`

**5. gui/launch.py**:
- "Enable Memory Citations" checkbox
- Citations tab (displays citation JSON)
- Wires checkbox → `orchestrator.enable_citations`

**6. config/app_config.py**:
- `ENABLE_MEMORY_CITATIONS` (default: False)
- `MAX_CITATIONS_DISPLAY` (default: 10)
- `CITATION_CONTENT_LENGTH` (default: 200)

**Data Flow**:
```python
# Retrieval phase
memories = context_gatherer._get_recent_conversations(10)
for idx, mem in enumerate(memories):
    memory_id_map[f"MEM_RECENT_{idx}"] = {
        'type': 'episodic_recent',
        'timestamp': mem['timestamp'],
        'content': mem['content'][:500],
        'relevance_score': 1.0,
        'db_id': mem.get('id', None)  # UUID or generated ID from ChromaDB/corpus
    }

# Generation phase (if citations enabled)
raw_response = "You mentioned [MEM_RECENT_2] that you're starting OMSA..."

# Extraction phase
clean, citations = _extract_citations(raw_response, memory_id_map)
# clean = "You mentioned that you're starting OMSA..."
# citations = [{'memory_id': 'MEM_RECENT_2', 'type': 'episodic_recent', 'db_id': 'b9e22f59-f5cb-...', ...}]

# Lookup phase (if needed)
citation_id = 'MEM_RECENT_2'  # Relative ID from citation
db_id = memory_id_map[citation_id]['db_id']  # Get absolute DB ID
full_memory = chroma_store.get(ids=[db_id])  # Query database for full record
```

**Testing**:
- tests/test_citation_system.py (9 tests):
  - test_extract_citations_basic
  - test_extract_citations_no_citations
  - test_extract_citations_empty_memory_map
  - test_extract_citations_multiple_same_citation
  - test_citation_mode_toggle
  - test_citation_content_truncation
  - test_citation_pattern_matches

**Use Cases**:
- Professional deployment with audit trails
- Debugging memory retrieval quality
- User transparency (which memories informed response)
- Citation-based feedback loops

**Conversation Log Integration** [NEW 2025-12-10]:
- Database IDs (`db_id`) now tracked in conversation logs (utils/conversation_logger.py)
- `store_interaction()` returns database ID (UUID or generated ID) from ChromaDB
- TEXT FORMAT logs: Shows "Memory ID: {db_id}" in metadata header and "[Memory ID: {db_id}]" after response
- JSON FORMAT logs: Includes `db_id` at top level for easy querying + in metadata for completeness
- Flow: store_interaction() → capture db_id → pass to conversation_logger → write to log file
- Enables full audit trail: conversation_logs/{timestamp}.txt → db_id → ChromaDB/corpus query
- Gracefully handles missing db_ids (older memories without UUIDs, generated profiles)

**Example Conversation Log**:
```
--- Conversation #1 ---
Time: 2025-12-10 12:07:50
Topic: pets
Memory ID: 4c78d78e-f093-4818-a2eb-144576db6788

USER:
Tell me about Flapjack

A:
Flapjack is your black cat [MEM_SEMANTIC_4]...

[Memory ID: 4c78d78e-f093-4818-a2eb-144576db6788]
```

---

## 3. Configuration (config/app_config.py)

**Key Settings**:
```python
# Paths
CORPUS_FILE = "./data/corpus_v4.json"
CHROMA_PATH = "./data/chroma_db_v4_v2"
SYSTEM_PROMPT_PATH = "./core/system_prompt.txt"

# Memory Limits (in prompt/builder.py)
PROMPT_TOKEN_BUDGET = 15000  # Higher for middle-out compression
PROMPT_MAX_RECENT = 15
PROMPT_MAX_MEMS = 15  # Semantic search results only
PROMPT_MAX_FACTS = 30
PROMPT_MAX_RECENT_FACTS = 30
PROMPT_MAX_SUMMARIES = 10  # Hybrid: 5 recent + 5 semantic
PROMPT_MAX_REFLECTIONS = 10  # Hybrid: 5 recent + 5 semantic
PROMPT_MAX_DREAMS = 3
PROMPT_MAX_SEMANTIC = 8
PROMPT_MAX_WIKI = 3

# Decay & Scoring
RECENCY_DECAY_RATE = 0.05
TRUTH_SCORE_UPDATE_RATE = 0.02
TRUTH_SCORE_MAX = 0.95

# Gating
GATE_REL_THRESHOLD = 0.18
DEICTIC_THRESHOLD = 0.60
NORMAL_THRESHOLD = 0.35
SUMMARY_COSINE_THRESHOLD = 0.30
REFLECTION_COSINE_THRESHOLD = 0.25

# Graceful Threshold Fallback (NEW)
GATING_MIN_RESULTS = 5              # Minimum results before relaxing threshold
GATING_RELAXED_MULTIPLIER = 0.7     # Relaxation multiplier (70% of original)

# Size Penalty Configuration (NEW)
LARGE_DOC_SIZE_THRESHOLD = 10000      # 10KB threshold
LARGE_DOC_KEYWORD_THRESHOLD = 0.3     # Min keyword_score to avoid penalty
LARGE_DOC_BASE_PENALTY = -0.25        # Base penalty, scaled by size multiplier

# Collection Boosts
COLLECTION_BOOSTS = {
    "facts": 0.15,
    "summaries": 0.10,
    "conversations": 0.0,
    "semantic": 0.05,
    "wiki": 0.05
}

# Score Weights
SCORE_WEIGHTS = {
    "relevance": 0.30,      # Reduced from 0.60 (old config value)
    "recency": 0.22,        # Increased from 0.15 (old config value)
    "truth": 0.18,          # Reduced from 0.20 (old config value)
    "importance": 0.05,     # Increased from 0.02 (old config value)
    "continuity": 0.10,     # Increased from 0.02 (old config value)
    "topic_match": 0.10,    # NEW - topic alignment scoring
    "structure": 0.05
}

# Hybrid Retrieval
HYBRID_SUMMARIES_ENABLED = True
HYBRID_REFLECTIONS_ENABLED = True

# Short-Term Memory (STM) [NEW]
USE_STM_PASS = True                    # Enable STM analysis
STM_MODEL_NAME = "gpt-4o-mini"         # Model for STM (fast/cheap)
STM_MAX_RECENT_MESSAGES = 8            # Messages to analyze
STM_MIN_CONVERSATION_DEPTH = 3         # Minimum depth to trigger

# Best-of-N Generation
ENABLE_BEST_OF = True
BEST_OF_N = 2
BEST_OF_TEMPS = [0.2, 0.7]
BEST_OF_MAX_TOKENS = 256
BEST_OF_LATENCY_BUDGET_S = 2.0
BEST_OF_DUEL_MODE = False  # Optional 2-model duel

# Query Rewrite
ENABLE_QUERY_REWRITE = True
REWRITE_TIMEOUT_S = 1.2

# Intent Classifier [NEW 2026-02-15]
INTENT_ENABLED = True                    # Feature toggle
INTENT_STM_REFINEMENT_THRESHOLD = 0.50   # Below this, STM can refine classification

# Memory Citation System [NEW 2025-12-04]
ENABLE_MEMORY_CITATIONS = False      # Feature flag for citation mode
MAX_CITATIONS_DISPLAY = 10           # Max citations to show in UI
CITATION_CONTENT_LENGTH = 200        # Content snippet length in citations

# Web Search (Tavily API) [NEW 2025-12-22]
WEB_SEARCH_ENABLED = True            # Feature flag for web search
WEB_SEARCH_API_KEY = ""              # Set via TAVILY_API_KEY env or config.yaml
WEB_SEARCH_TIMEOUT = 30.0            # API timeout in seconds
WEB_SEARCH_MAX_CONTENT_CHARS = 10000 # Max content per page
WEB_SEARCH_DAILY_CREDIT_LIMIT = 100  # Daily API credit limit
WEB_SEARCH_CACHE_TTL_HOURS = 72      # Cache expiry (3 days)
WEB_SEARCH_CONFIDENCE_THRESHOLD = 0.5 # Min confidence to trigger search

# Models
DEFAULT_MAX_TOKENS = 2048
HEAVY_TOPIC_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.7

# Wikipedia
WIKI_FETCH_FULL_DEFAULT = True
WIKI_MAX_CHARS_DEFAULT = 15000
WIKI_TIMEOUT_DEFAULT = 1.2

# Corpus Management
CORPUS_MAX_ENTRIES = 2000

# Code Proposals [ENHANCED 2026-02-10]
CODE_PROPOSALS_ENABLED = True
CODE_PROPOSALS_MAX_PER_SESSION = 5
CODE_PROPOSALS_DEDUP_THRESHOLD = 0.70

# File Upload [ENHANCED 2026-02-10]
FILE_UPLOAD_ALLOWED_EXTENSIONS = ['.txt', '.docx', '.csv', '.py', '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.webp']  # .pdf added [2026-03-10]
FILE_UPLOAD_IMAGE_DIR = "data/uploads"
PROMPT_MAX_USER_UPLOADS = 5
```

**Environment Overrides**: Any setting can be overridden via environment variables

---

## 4. Data Flow Examples

### 4.1 Basic Query Flow
```python
# User asks: "What is Python?"

1. orchestrator.process_user_query("What is Python?")
2. tone_detector.detect_crisis_level() → CONVERSATIONAL
3. topic_manager.extract_topics() → ["Python"]
4. query_checker.analyze_query() → is_heavy_topic=False
5. orchestrator.prepare_prompt():
   a. Inject tone instructions (CONVERSATIONAL: max 3 sentences)
   b. prompt_builder.build_prompt():
      - context_gatherer._get_recent_conversations(15)
      - context_gatherer._get_semantic_memories(query, 15)
      - context_gatherer.get_facts(query, 30)
      - context_gatherer._get_summaries_separate() → {recent: 5, semantic: 5}
      - context_gatherer._get_reflections_separate() → {recent: 5, semantic: 5}
      - _hygiene_and_caps() → Dedupe with semantic similarity
      - token_manager._manage_token_budget()
   c. _assemble_prompt(context, user_input) → Formatted string
6. response_generator.generate_streaming_response(prompt)
   → "<thinking>Python is a general-purpose language...</thinking>\nPython is a high-level programming language."
7. orchestrator._parse_thinking_block() → Extract final answer
8. memory_coordinator.store_interaction(query, final_answer)
9. Return final_answer to user
```

### 4.2 Heavy Topic Flow
```python
# User shares news article about ICE raids

1. orchestrator.process_user_query(article_text)
2. tone_detector.detect_crisis_level() → CONCERN (keyword: "raid")
3. query_checker.analyze_query() → is_heavy_topic=True
4. orchestrator.prepare_prompt():
   a. Inline fact extraction (with 5s timeout):
      - memory_system._extract_and_store_facts(query, "", 0.7)
      - memory_system.get_facts(query, 10) → fresh_facts
   b. Inject tone instructions (LIGHT_SUPPORT: 2-4 sentences)
   c. Set response_max_tokens = HEAVY_TOPIC_MAX_TOKENS (8192)
   d. prompt_builder.build_prompt() with fresh_facts
5. response_generator.generate_streaming_response()
6. Store interaction and return
```

### 4.3 Shutdown Consolidation
```python
# At application shutdown:

1. orchestrator.shutdown()
2. memory_coordinator.process_shutdown_memory()
   a. Count total conversations vs summaries
   b. If (convos / SUMMARY_EVERY_N) > summaries:
      i. Get last N unsummarized conversations
      ii. memory_consolidator.consolidate_memories(convos)
      iii. Store summary as SUMMARY memory type
   c. fact_extractor.extract_facts(recent_conversations)
   d. Store facts as SEMANTIC memory type
   e. Extract procedural skills via LLM (0-3 per session)
   f. Generate code proposals via GoalDirectedGenerator (0-5 per session) [NEW 2026-02-05]
   g. Update UserProfile with extracted facts
3. corpus_manager.save()
```

---

## 5. Key Algorithms

### 5.1 Memory Ranking (memory_scorer.rank_memories)
```python
def rank_memories(memories, query, weight_overrides=None):
    # Merge intent-driven weight overrides on top of global SCORE_WEIGHTS
    weights = dict(SCORE_WEIGHTS)
    if weight_overrides:
        weights.update(weight_overrides)
    for memory in memories:
        # 1. Relevance + collection boost
        rel = memory.get('relevance_score', 0.5)
        rel += COLLECTION_BOOSTS.get(memory.get('collection'), 0)

        # 2. Recency decay
        age_hours = (now - memory['timestamp']).total_seconds() / 3600
        recency = 1.0 / (1.0 + RECENCY_DECAY_RATE * age_hours)

        # 3. Truth (boosted by access count)
        truth = memory.get('truth_score', 0.6)
        access_count = memory.get('metadata', {}).get('access_count', 0)
        truth = min(1.0, truth + TRUTH_SCORE_UPDATE_RATE * access_count)

        # 4. Importance
        importance = memory.get('importance_score', 0.5)

        # 5. Continuity (recent + token overlap)
        continuity = 0.0
        if (now - memory['timestamp']) < timedelta(minutes=10):
            continuity += 0.1
        token_overlap = len(query_tokens & memory_tokens) / len(query_tokens)
        continuity += 0.3 * token_overlap

        # 6. Structural alignment (math density)
        query_density = _num_op_density(query)
        memory_density = _num_op_density(memory_text)
        alignment = 1.0 - min(1.0, abs(query_density - memory_density) * 3.0)
        structure = 0.15 * alignment

        # 7. Penalties/bonuses
        penalty = 0.0
        if is_math_query and has_analogy_markers(memory):
            penalty -= 0.1
        if has_negative_tone(memory):
            truth -= 0.2

        # 8. Anchor bonus (deictic queries)
        anchor_bonus = 0.0
        if is_deictic(query):
            anchor_tokens = extract_from_conversation_context()
            anchor_overlap = len(anchor_tokens & memory_tokens) / len(anchor_tokens)
            anchor_bonus = 0.2 * anchor_overlap if anchor_overlap > 0.05 else -PENALTY

        # Final score
        memory['final_score'] = (
            SCORE_WEIGHTS['relevance'] * rel +
            SCORE_WEIGHTS['recency'] * recency +
            SCORE_WEIGHTS['truth'] * truth +
            SCORE_WEIGHTS['importance'] * importance +
            SCORE_WEIGHTS['continuity'] * continuity +
            SCORE_WEIGHTS['topic_match'] * topic_match +  # NEW
            structure +
            anchor_bonus +
            meta_bonus +      # NEW - for meta-conversational queries
            penalty +
            size_penalty      # NEW - scaled document size penalty
        )

    return sorted(memories, key=lambda m: m['final_score'], reverse=True)
```

### 5.2 Multi-Stage Gating (gate_system.filter_memories)
```python
def filter_memories(query, memories, k=20):
    # Stage 1: FAISS semantic search (top 50)
    query_embedding = embed(query)
    faiss_results = faiss_index.search(query_embedding, k=50)

    # Stage 2: Cosine similarity filter
    filtered = []
    for memory in faiss_results:
        memory_embedding = embed(memory['text'])
        similarity = cosine_similarity(query_embedding, memory_embedding)
        if similarity >= GATE_COSINE_THRESHOLD:
            memory['relevance_score'] = similarity
            filtered.append(memory)

    # Stage 3: Cross-encoder reranking
    pairs = [(query, mem['text']) for mem in filtered]
    scores = cross_encoder.predict(pairs)

    for i, memory in enumerate(filtered):
        memory['relevance_score'] = scores[i]

    reranked = sorted(filtered, key=lambda m: m['relevance_score'], reverse=True)
    return reranked[:k]
```

### 5.3 Token Budget Allocation (prompt_builder._allocate_tokens)
```python
def _allocate_tokens(query, memories, facts, summaries, wiki, budget):
    # Fixed allocations
    system_prompt_tokens = count_tokens(SYSTEM_PROMPT)
    query_tokens = count_tokens(query)
    reserved = system_prompt_tokens + query_tokens + 100  # buffer

    remaining = budget - reserved

    # Priority allocation
    allocations = {
        'recent_context': min(500, remaining * 0.3),
        'episodic': min(800, remaining * 0.35),
        'semantic': min(400, remaining * 0.15),
        'summaries': min(300, remaining * 0.10),
        'wiki': min(200, remaining * 0.10)
    }

    # Fill sections in priority order
    sections = []
    for key, token_limit in allocations.items():
        content = get_content(key)
        truncated = truncate_to_tokens(content, token_limit)
        sections.append(truncated)

    return "\n\n".join(sections)
```

---

## 6. File Organization

```
daemon/
├── config/
│   ├── app_config.py          # Central configuration loader
│   └── config.yaml            # YAML config (optional)
│
├── core/
│   ├── orchestrator.py        # Main controller (tone, STM, coordinates subsystems)
│   ├── response_parser.py     # Response parsing utilities (thinking blocks, XML stripping) [NEW 2026-01-23]
│   ├── stm_analyzer.py        # Short-term memory analyzer [NEW]
│   ├── intent_classifier.py   # Regex-first query intent classifier (9 types) [NEW 2026-02-15]
│   ├── response_generator.py  # LLM streaming + Best-of-N/Duel (FIXED)
│   ├── best_of_handler.py     # Best-of orchestration (duel/ensemble/single) [NEW]
│   ├── competitive_scorer.py  # Judge-based response selection
│   ├── dependencies.py        # Dependency injection setup
│   ├── wiki_util.py          # Wikipedia utility functions
│   ├── prompt_builder.py      # Legacy prompt builder (deprecated)
│   ├── prompt.py             # Legacy unified prompt (deprecated)
│   └── prompt/               # Modular prompt system (STM integrated)
│       ├── __init__.py       # Public API and imports
│       ├── builder.py        # Main UnifiedPromptBuilder
│       ├── context_gatherer.py # Data collection + hybrid retrieval
│       ├── formatter.py      # Text formatting
│       ├── token_manager.py  # Budget management
│       ├── summarizer.py     # LLM summarization + on-demand reflections
│       └── base.py          # Utilities and fallbacks
│
├── memory/
│   ├── memory_coordinator.py      # Thin orchestrator (~498 lines, delegates to components)
│   ├── shutdown_processor.py      # Session-end summaries, facts, reflections [NEW - EXTRACTED]
│   ├── memory_scorer.py           # Scoring and ranking operations [NEW - REFACTORED]
│   ├── memory_retriever.py        # Retrieval operations [NEW - REFACTORED]
│   ├── memory_storage.py          # Storage/persistence operations [NEW - REFACTORED]
│   ├── thread_manager.py          # Thread tracking [NEW - REFACTORED]
│   ├── memory_interface.py        # Protocol contracts [NEW - REFACTORED]
│   ├── procedural_skill.py        # ProceduralSkill dataclass + SkillCategory enum [NEW 2026-01-27]
│   ├── corpus_manager.py          # JSON persistence
│   ├── memory_consolidator.py     # Summarization
│   ├── fact_extractor.py          # Pattern-based extraction
│   ├── llm_fact_extractor.py      # LLM-based extraction
│   ├── hybrid_retriever.py        # Query rewrite + semantic + keyword
│   └── storage/
│       └── multi_collection_chroma_store.py  # Vector DB
│
├── core/agentic/              # Agentic search system [NEW 2026-01]
│   ├── __init__.py            # Package exports
│   ├── types.py               # Data structures (AgentState, SearchProtocol, etc.)
│   ├── protocols.py           # Protocol detection and parsing
│   └── controller.py          # AgenticSearchController (ReAct loop)
│
├── processing/
│   └── gate_system.py         # Multi-stage filtering
│
├── models/
│   ├── model_manager.py       # Multi-provider LLM client
│   └── tokenizer_manager.py   # Token counting
│
├── utils/
│   ├── bootstrap.py           # Frozen executable environment setup [NEW 2025-12-12]
│   ├── startup.py             # Staged imports with splash progress [NEW 2025-12-12]
│   ├── topic_manager.py       # Topic extraction
│   ├── time_manager.py        # Temporal utilities
│   ├── tone_detector.py       # Crisis detection (harm scoring + semantic + LLM)
│   ├── need_detector.py       # Need-type detection (PRESENCE vs PERSPECTIVE) [NEW]
│   ├── emotional_context.py   # Combined emotional analysis (tone + need) [NEW]
│   ├── query_checker.py       # Query analysis + heavy topic + thread detection
│   ├── query_rewriter.py      # Query expansion for retrieval
│   ├── keyword_matcher.py     # Keyword overlap scoring
│   ├── logging_utils.py       # Centralized logging
│   ├── file_processor.py      # File/image/PDF upload processing with security validation [ENHANCED 2026-03-10]
│   ├── health_check.py        # Docker/K8s health endpoint
│   ├── conversation_logger.py # Conversation persistence
│   ├── web_search_trigger.py  # Web search detection (LLM-first + heuristics) [ENHANCED 2026-01]
│   ├── daily_notes_generator.py # Auto-generated daily summaries with monthly folder hierarchy [ENHANCED 2026-03-10]
│   ├── weekly_notes_generator.py # Auto-generated weekly summaries with monthly folder hierarchy [ENHANCED 2026-03-10]
│   ├── monthly_notes_generator.py # Auto-generated monthly summaries from daily notes [NEW 2026-03-10]
│   └── tag_generator.py       # LLM-based tag generation for notes [NEW 2026-01-22]
│
├── knowledge/
│   ├── WikiManager.py         # Wikipedia FAISS search
│   ├── semantic_search.py     # General semantic utilities
│   ├── topic_manager.py       # Topic-specific utilities
│   ├── web_search_manager.py  # Tavily API + caching [NEW 2025-12-22]
│   ├── wolfram_manager.py     # Wolfram Alpha LLM API [NEW 2026-01-22]
│   ├── sandbox_manager.py     # E2B code sandbox execution [NEW 2026-01-22]
│   ├── obsidian_manager.py    # Obsidian vault integration [NEW 2026-01]
│   ├── reference_docs_manager.py  # Daemon self-knowledge docs + user uploads [ENHANCED 2026-02-10]
│   ├── proposal_generator.py  # LLM-based code proposal generation [ENHANCED 2026-02-10]
│   ├── git_memory.py          # Git commit extractor [NEW 2026-01-27]
│   └── git_memory_loader.py   # Git → PROCEDURAL ChromaDB loader [NEW 2026-01-27]
│
├── gui/
│   ├── launch.py              # Gradio web interface (async chunk processing, tag stripping)
│   ├── handlers.py            # UI event handlers (streaming, agentic routing) [ENHANCED 2026-01]
│   └── wizard.py              # First-run onboarding wizard [NEW 2025-12-11]
│
├── integrations/
│   └── wikipedia_api.py       # Wikipedia API client
│
├── data/
│   ├── corpus_v4.json         # Short-term memory store (current)
│   ├── vector_index_ivf.faiss # FAISS index (781MB)
│   ├── chroma_db_v4_v2/       # ChromaDB vector store (current)
│   ├── wiki/                  # Wikipedia source data (102GB)
│   └── pipeline/              # Wikipedia processing scripts (43GB)
│
├── tests/                     # All test files (114 files)
│   ├── unit/                  # Unit tests (20+ files)
│   │   ├── test_tone_detector.py
│   │   ├── test_need_detection.py  # [NEW]
│   │   ├── test_query_checker.py
│   │   ├── test_corpus_manager.py
│   │   ├── test_web_search_manager.py   # [NEW 2025-12-22]
│   │   ├── test_web_search_trigger.py   # [NEW 2025-12-22]
│   │   ├── test_agentic_search.py       # [NEW 2026-01] 42 tests for agentic system
│   │   ├── test_sandbox_manager.py      # [NEW 2026-01-22] 42 tests for E2B sandbox
│   │   └── ...
│   ├── integration.bak/       # Backup integration tests
│   ├── test_*.py             # Integration tests (50+ files)
│   │   ├── test_integration_prompt_builder.py
│   │   ├── test_integration_gate_system.py
│   │   ├── test_memory_coordinator_advanced.py
│   │   ├── test_tone_detection.py
│   │   ├── test_need_detection.py  # [NEW]
│   │   ├── test_llm_fallback.py
│   │   └── ...
│   └── fixtures/             # Test fixtures
│
├── scripts/                   # Utility scripts and demos
│   ├── demo_*.py             # Demo scripts
│   ├── debug_*.py            # Debug utilities
│   ├── mutation_*.py         # Mutation testing
│   ├── *.sh                  # Shell scripts
│   └── ...
│
├── hooks/                     # PyInstaller custom hooks [NEW 2025-12-12]
│   ├── runtime_hook.py       # Pre-main environment setup (DAEMON_FROZEN, multiprocessing)
│   ├── hook-chromadb.py      # ChromaDB data files + hidden imports
│   ├── hook-gradio.py        # Gradio static files + fastapi/uvicorn
│   ├── hook-sentence_transformers.py  # Transformers + cross-encoder
│   ├── hook-spacy.py         # spaCy model + language data
│   └── hook-tiktoken.py      # Tiktoken encoding files
│
├── assets/                    # Application assets [NEW 2025-12-12]
│   ├── daemon_icon.ico       # Windows icon
│   ├── daemon_icon.png       # PNG icon
│   └── splash.png            # PyInstaller splash screen
│
├── docs/                      # Documentation
│   ├── PROJECT_SKELETON.md   # This file
│   ├── QUICK_REFERENCE.md    # Quick reference guide
│   ├── BUILD_GUIDE.md        # Desktop executable build guide [NEW 2025-12-12]
│   ├── DOCKER_README.md      # Docker setup
│   └── ...
│
├── daemon.spec                # PyInstaller spec file [NEW 2025-12-12]
│
└── build/                     # Build configuration
    ├── Makefile.fast         # Fast profile
    ├── Makefile.balanced     # Balanced profile
    └── Makefile.max          # Maximum quality
```

---

## 7. Common Patterns

### 7.1 Async/Await Pattern
```python
# Most public methods are async for I/O operations
async def handle_request(query: str) -> str:
    memories = await memory_coordinator.get_memories(query)
    response = await response_generator.generate(prompt)
    await memory_coordinator.store_interaction(query, response)
    return response

# Call with: asyncio.run(handle_request(query))
```

### 7.2 Dependency Injection
```python
# Dependencies initialized once, passed to components
deps = Dependencies(
    config=app_config,
    corpus_manager=CorpusManager(config),
    chroma_store=MultiCollectionChromaStore(config),
    model_manager=ModelManager(config)
)

orchestrator = Orchestrator(deps)
```

### 7.3 Scoring Heuristics
```python
# All scoring functions follow pattern:
def calculate_score(content: str, context: dict) -> float:
    score = BASE_SCORE  # e.g., 0.5

    # Apply boosts
    if condition1:
        score += BOOST1
    if condition2:
        score += BOOST2

    # Apply penalties
    if bad_condition:
        score -= PENALTY

    # Cap at limits
    return max(0.0, min(1.0, score))
```

### 7.4 Memory Storage Pattern
```python
# All memory additions follow:
1. Create memory dict with required fields
2. Calculate scores (truth, importance)
3. Add to corpus JSON
4. Add to ChromaDB with embeddings
5. Update metadata (access count, timestamps)
```

### 7.5 Error Handling
```python
# Graceful degradation pattern
try:
    result = primary_method()
except Exception as e:
    logger.warning(f"Primary failed: {e}")
    result = fallback_method()

# Common fallbacks:
# - spaCy fails → use LLM extraction
# - ChromaDB fails → use corpus only
# - LLM fails → return cached response
```

---

## 8. Testing Strategy

### 8.1 Unit Tests
- **Location**: `tests/unit/`
- **Count**: 20+ test files
- **Pattern**: Mock dependencies, test methods in isolation

**Key Unit Test Files**:
- `test_tone_detector.py` - Crisis detection levels and thresholds
- `test_query_checker.py` - Heavy topic and thread detection
- `test_corpus_manager.py` - JSON persistence operations
- `test_memory_coordinator_helpers.py` - Ranking and scoring
- `test_prompt_builder_methods.py` - Context assembly
- `test_orchestrator_helpers.py` - Thinking block parsing
- `test_intent_classifier.py` - Intent classification, tone bias, STM refinement, profiles (74 tests) **[NEW 2026-02-15]**

### 8.1b Retrieval Quality Benchmarks **[NEW 2026-02-17]**
- **Location**: `tests/benchmarks/`
- **Purpose**: End-to-end retrieval quality testing with real embeddings (all-MiniLM-L6-v2)
- **Run**: `pytest tests/benchmarks/ -m benchmark -v` (~5s runtime)
- **Exclude**: `pytest -m "not benchmark"`

**Components**:
- `tests/benchmarks/conftest.py` — Session-scoped fixtures: seeded ChromaDB + CorpusManager, MockTimeManager for deterministic recency scoring
- `tests/benchmarks/retrieval_benchmark.py` — BenchmarkResult dataclass + RetrievalBenchmark harness (recall@K, MRR, marker substring matching)
- `tests/benchmarks/test_retrieval_quality.py` — 19 parametrized test cases across all 9 intent types + structural validation
- `tests/benchmarks/report_generator.py` — Markdown report grouped by intent type with pass/fail/recall/MRR summary
- `tests/fixtures/retrieval_benchmarks.yaml` — 30 seed memories + 19 test cases (queries, expected intents, must_retrieve/must_not_retrieve markers)

**Design**:
- Seeds both ChromaDB AND CorpusManager for dual-retrieval testing
- Uses marker substring matching to identify returned memories by `benchmark_id`
- MockTimeManager with fixed reference time (2026-02-15 12:00) for deterministic scores
- Covers all 9 intent types: FACTUAL_RECALL, TEMPORAL_RECALL, EMOTIONAL_SUPPORT, CASUAL_SOCIAL, TECHNICAL_HELP, CREATIVE_EXPLORATION, META_CONVERSATIONAL, PROJECT_WORK, GENERAL

### 8.2 Integration Tests
- **Location**: `tests/`
- **Count**: 40+ test files
- **Pattern**: Test full pipelines with real components

**Key Integration Test Files**:
- `test_tone_detection.py` - End-to-end crisis detection
- `test_need_detection.py` - Need-type detection (PRESENCE vs PERSPECTIVE) [NEW]
- `test_llm_fallback.py` - LLM fallback behavior
- `test_memory_coordinator.py` - Full memory retrieval pipeline
- `test_gated_prompt.py` - Gating + prompt building
- `test_summaries.py` - Summary generation and retrieval
- `test_competitive_scorer.py` - Best-of-N scoring

### 8.3 Mutation Testing
- **Tool**: Custom mutation testing framework
- **Location**: `demo_mutation_testing.py`
- **Purpose**: Verify tests catch code mutations

### 8.4 Test Suite Status (Current)
**Last Full Run**: December 2024

**Test Collection**:
- **Total tests**: 1554 tests across all files
- **Collection errors**: 0 (previously 2, now fixed)

**Test Results**:
- ✅ **Passed**: 1480 (95.2%)
- ❌ **Failed**: 45 (2.9%) - Pre-existing issues
- ⚠️ **Errors**: 22 (1.4%) - API signature mismatches
- ⏭️ **Skipped**: 7 (0.5%) - Method signature differences

**Run Time**: ~2:22 (142 seconds)

**Fixed Issues**:
- `test_cross_dedup.py`: Fixed `Dependencies` → `DependencyContainer` import
- `test_full_meta_query.py`: Fixed `Orchestrator` → `DaemonOrchestrator` import

**Test Files**:
- **Unit tests**: ~20 files in `tests/unit/`
- **Integration tests**: ~45 files in `tests/`
- **Total test files**: 114

**Known Failure Categories** (not caused by recent changes):
- UnifiedPromptBuilder API changes (missing `get_facts`, `get_recent_facts`)
- MultiCollectionChromaStore __init__ kwargs mismatches
- DaemonOrchestrator __init__ signature changes
- Need detection edge cases (8 tests, acceptable tolerance)

---

## 9. Performance Considerations

### 9.1 Bottlenecks
1. **ChromaDB queries** - Embedding generation is slow
   - Mitigation: Batch operations, cache embeddings
2. **Cross-encoder reranking** - CPU-intensive
   - Mitigation: Limit to top 20-30 candidates
3. **LLM calls** - Network latency + generation time
   - Mitigation: Streaming, async operations

### 9.2 Optimization Flags
```bash
# CPU-only mode (no GPU)
export CHROMA_DEVICE=cpu

# Reduce memory footprint
export PROMPT_TOKEN_BUDGET=2048
export PROMPT_MAX_MEMS=20

# Fast profile
make -f build/Makefile.fast run
```

---

## 10. Extension Points

### 10.1 Adding New Memory Type
```python
1. Add to MemoryType enum in corpus_manager.py
2. Create collection in multi_collection_chroma_store.py
3. Add handling in memory_coordinator.py
4. Update core/prompt/context_gatherer.py to retrieve new type
5. Update core/prompt/formatter.py to format new type
```

### 10.2 Adding New LLM Provider
```python
1. Add credentials to environment
2. Update model_manager.py with new provider client
3. Add alias mappings in config
4. Test with: python main.py --model new-provider-model
```

### 10.3 Custom Scoring Functions
```python
1. Add scoring function to memory_coordinator.py
2. Integrate into _rank_memories()
3. Add weight to SCORE_WEIGHTS in config
4. Test with unit tests
```

---

## 11. Key Invariants

1. **All memories have timestamps** - Required for decay calculation
2. **Scores are 0.0-1.0** - All scoring functions capped
3. **Token budgets are respected** - Prompts never exceed limits
4. **Async operations use await** - No blocking I/O in main thread
5. **ChromaDB and corpus stay in sync** - Same memories in both
6. **Memory types are immutable** - Once set, never changed
7. **Summaries link to parents** - ⚠️ DISABLED (parent-child relations intentionally deactivated)

---

## 12. Debug/Introspection

### 12.1 Logging Levels
```python
# Set in environment
export LOG_LEVEL=DEBUG  # Shows all scoring details
export LOG_LEVEL=INFO   # Shows major operations
export LOG_LEVEL=WARNING  # Errors only
```

### 12.2 Inspection Commands
```bash
# View conversation logs
ls -ltr conversation_logs/

# Inspect corpus
python -c "import json; print(json.load(open('data/corpus.json'))['conversations'][-5:])"

# Check ChromaDB collections
python tests/inspect_chroma.py

# Test summaries
python main.py inspect-summaries
```

---

## 13. Compressed Cheat Sheet

**One-line descriptions** for quick reference:

| Component | One-Liner |
|-----------|-----------|
| orchestrator.py | Main loop: tone → topic → heavy check → STM → prompt → LLM → store |
| response_parser.py | Parse: extract thinking blocks, strip reflections/XML/prompt artifacts [NEW 2026-01-23] |
| stm_analyzer.py | Analyze: lightweight LLM pass for recent context summary (topic/intent/tone/threads) [NEW] |
| intent_classifier.py | Classify: regex-first query intent (9 types), produces weight/retrieval/gate overrides [NEW 2026-02-15] |
| memory_coordinator.py | Hub: thin orchestrator (~498 lines) delegating to modular components [REFACTORED] |
| shutdown_processor.py | Shutdown: block summaries + fact extraction + procedural skills + reflections + user profile [NEW - EXTRACTED] |
| memory_scorer.py | Scoring: calculate truth/importance, rank by composite score with temporal decay [NEW - REFACTORED] |
| memory_retriever.py | Retrieval: get_memories pipeline with gating and ranking [NEW - REFACTORED] |
| memory_storage.py | Storage: store_interaction and fact extraction [NEW - REFACTORED] |
| thread_manager.py | Threads: conversation continuity tracking [NEW - REFACTORED] |
| memory_interface.py | Protocols: type contracts for memory components [NEW - REFACTORED] |
| corpus_manager.py | JSON CRUD: load/save/query short-term memories |
| multi_collection_chroma_store.py | Vector DB: embed, store, semantic search across 9 collections |
| gate_system.py | Filter: FAISS → cosine → cross-encoder → top K |
| prompt/builder.py | Assemble: system + separated context sections + STM within 15K tokens |
| response_generator.py | Stream: async LLM + Best-of-N + Duel modes (buffer fix + DeepSeek EOS) [FIXED] |
| best_of_handler.py | Orchestrate: duel/ensemble/single mode selection + timeout fallback [NEW] |
| gui/handlers.py | Relay: streaming chunks + thinking blocks + tag stripping (reply/response/answer) [FIXED] |
| gui/wizard.py | Onboard: first-run wizard (API key, style, name, pronouns, background facts) [NEW] |
| utils/bootstrap.py | Freeze: platform-specific paths, .env loading, data migration [NEW] |
| utils/startup.py | Progress: staged imports with splash screen updates [NEW] |
| daemon.spec | Build: PyInstaller one-dir spec (splash, hooks, hidden imports) [NEW] |
| model_manager.py | Unified: OpenAI/Claude/OpenRouter/Local via single interface |
| topic_manager.py | Extract: Heuristics + LLM fallback (handles conversational/emotional) |
| memory_consolidator.py | Summarize: LLM compresses N conversations at shutdown |
| fact_extractor.py | Parse: regex + optional LLM for entity/fact extraction |
| WikiManager.py | Search: FAISS over Wikipedia dump for context injection |
| wolfram_manager.py | Compute: Wolfram Alpha LLM API for math/science queries [NEW 2026-01-22] |
| tone_detector.py | Detect: 4-level crisis (harm scoring + semantic + LLM fallback) |
| need_detector.py | Detect: need-type (PRESENCE vs PERSPECTIVE, keyword + semantic hybrid) |
| emotional_context.py | Combine: tone + need type for unified emotional analysis |
| query_checker.py | Analyze: heavy topics, thread detection, temporal windows |
| hybrid_retriever.py | Retrieve: query rewrite + semantic + keyword scoring |
| time_manager.py | Utils: timestamps and temporal decay calculations |
| daily_notes_generator.py | Generate: daily summaries with Life Events + tags, monthly folder hierarchy [ENHANCED 2026-03-10] |
| weekly_notes_generator.py | Organize: monthly/weekly folders + summaries + tags [ENHANCED 2026-03-10] |
| monthly_notes_generator.py | Generate: monthly summaries from daily notes [NEW 2026-03-10] |
| tag_generator.py | Tags: LLM-based tag extraction for Obsidian notes (100+ vocabulary, 6 categories) [NEW 2026-01-22] |
| git_memory.py | Extract: Git commit history with metadata + conventional commit tagging [NEW 2026-01-27] |
| git_memory_loader.py | Load: Backfill/incremental sync of git commits to PROCEDURAL ChromaDB [NEW 2026-01-27] |
| procedural_skill.py | Data: ProceduralSkill dataclass + SkillCategory enum for adaptive workflows [NEW 2026-01-27] |

---

## 14. Critical Constants (Quick Reference)

```python
# Scoring
RECENCY_DECAY_RATE = 0.05
TRUTH_SCORE_UPDATE_RATE = 0.02
DEICTIC_ANCHOR_PENALTY = 0.1
DEICTIC_CONTINUITY_MIN = 0.12

# Weights
SCORE_WEIGHTS = {
    'relevance': 0.30,      # Reduced from 0.60 (old config value)
    'recency': 0.22,        # Increased from 0.15 (old config value)
    'truth': 0.18,          # Reduced from 0.20 (old config value)
    'importance': 0.05,     # Increased from 0.02 (old config value)
    'continuity': 0.10,     # Increased from 0.02 (old config value)
    'topic_match': 0.10,    # NEW - topic alignment scoring
    'structure': 0.05
}

# Collection Boosts
COLLECTION_BOOSTS = {
    'facts': 0.15,
    'summaries': 0.10,
    'conversations': 0.0,
    'semantic': 0.05,
    'wiki': 0.05
}

# Thresholds
GATE_REL_THRESHOLD = 0.18
DEICTIC_THRESHOLD = 0.60
NORMAL_THRESHOLD = 0.35
SUMMARY_COSINE_THRESHOLD = 0.30
REFLECTION_COSINE_THRESHOLD = 0.25
THREAD_CONTINUITY_THRESHOLD = 0.5

# Tone Detection Thresholds
TONE_THRESHOLD_HIGH = 0.58
TONE_THRESHOLD_MEDIUM = 0.50
TONE_THRESHOLD_CONCERN = 0.43

# Hybrid Retrieval Weights
HYBRID_SEMANTIC_WEIGHT = 0.7
HYBRID_KEYWORD_WEIGHT = 0.3

# Prompt Limits
PROMPT_TOKEN_BUDGET = 15000
PROMPT_MAX_RECENT = 15
PROMPT_MAX_MEMS = 15
PROMPT_MAX_FACTS = 30
PROMPT_MAX_SUMMARIES = 10  # 5 recent + 5 semantic
PROMPT_MAX_REFLECTIONS = 10
PROMPT_MAX_WIKI = 3

# Heavy Topic Detection
HEAVY_TOPIC_CHAR_THRESHOLD = 2500
HEAVY_TOPIC_TIMEOUT = 2.0  # seconds

# Short-Term Memory (STM) [NEW]
USE_STM_PASS = True
STM_MODEL_NAME = "gpt-4o-mini"
STM_MAX_RECENT_MESSAGES = 8
STM_MIN_CONVERSATION_DEPTH = 3

# Feature Toggles
REFLECTIONS_ON_DEMAND = True  # Generate reflections if below threshold
REFLECTIONS_SESSION_FILTER = False  # Filter to session-level only
REFLECTIONS_TOPUP = True  # Top up reflections from storage

# Deduplication
SIMILARITY_THRESHOLD = 0.90  # Semantic similarity for duplicates
```

---

## 15. Docker Containerization **[NEW]**

### 15.1 Architecture
**Purpose**: Production-ready Docker deployment with offline model baking and persistent storage

**Components**:
- `Dockerfile` - Multi-stage build (builder + runtime)
- `docker-compose.yml` - Service orchestration with volumes
- `docker-entrypoint.sh` - Container initialization script
- `.dockerignore` - Excludes large data dirs (149GB) from build context
- `.env.example` - Minimal Docker-specific environment template
- `utils/health_check.py` - Health endpoint for monitoring

### 15.2 Multi-Stage Build

**Stage 1: Builder**
```dockerfile
FROM python:3.11-slim AS builder
- Install build tools (gcc, g++, git)
- Create venv at /opt/venv
- Install Python dependencies
- Pre-download AI models:
  - spaCy: en_core_web_sm
  - sentence-transformers: all-MiniLM-L6-v2
  - Result: Models baked into image (no download on startup)
```

**Stage 2: Runtime**
```dockerfile
FROM python:3.11-slim
- Copy venv from builder
- Copy pre-downloaded models to /app/data/cache/huggingface
- Install only runtime dependencies (libgomp1, curl)
- Create non-root daemon user
- Copy application code
- Set HF_HUB_OFFLINE=1 (use baked models)
- EXPOSE 7860 (Gradio)
- HEALTHCHECK /health endpoint
```

**Result**: ~60% smaller final image, 2-3 min faster startup

### 15.3 Configuration Strategy

**Minimal .env Approach** (selected):
```bash
# Only Docker-specific essentials
OPENAI_API_KEY=sk-...
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_PORT=7860
CHROMA_DEVICE=cpu
HF_HUB_OFFLINE=1
HF_HOME=/app/data/cache/huggingface
```

**Rationale**:
- Let config.yaml remain single source of truth
- Prevents accidental feature disabling (e.g., MEM_TOPUP_ENABLE=0)
- Environment variables override config.yaml (precedence issue)

**Alternative** (not selected): Comprehensive .env with 105+ variables
- Pro: All configuration in one place
- Con: Env vars override config.yaml, causing unexpected behavior
- Con: Previous testing showed disabled reflections, summaries, memory top-up

### 15.4 Data Persistence

**Volumes** (docker-compose.yml):
```yaml
volumes:
  daemon-data:  # Persists corpus, ChromaDB, logs, cache
  daemon-logs:  # Separate logging persistence
```

**Mounted Paths**:
- `/app/data/corpus_v4.json` - Conversation memory
- `/app/data/chroma_db_v4` - Vector embeddings
- `/app/conversation_logs` - Session transcripts
- `/app/data/cache/huggingface` - Pre-downloaded models

**Container Initialization** (docker-entrypoint.sh):
```bash
1. Create directories: /app/data, /app/logs, /app/conversation_logs
2. Initialize fresh corpus if missing: echo '[]' > corpus_v4.json
3. Validate environment variables
4. Support multiple modes: gui (default), cli, test-summaries
```

### 15.5 Memory Consolidation on Shutdown

**Graceful Shutdown Flow**:
```
SIGTERM (podman stop daemon-rag-agent)
    ↓
main.py catches signal
    ↓
orchestrator.shutdown()
    ↓
memory_coordinator.process_shutdown_memory()
    ├─> Fact extraction from recent conversations
    ├─> Summary generation (every N conversations)
    ├─> Reflection generation (session-end)
    ├─> Corpus persistence to JSON
    └─> ChromaDB embeddings written to disk
    ↓
Container exits gracefully
```

**Critical**: Use `podman stop` (sends SIGTERM) NOT `podman kill` (sends SIGKILL immediately)

### 15.6 Health Check

**Endpoint**: `GET /health`

**Checks**:
- Corpus file exists and is readable
- ChromaDB connectivity (can query collections)
- API key present in environment

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-29T21:00:00Z"
}
```

**Integration**:
- Docker HEALTHCHECK directive (30s interval, 10s timeout)
- Kubernetes liveness/readiness probes
- Load balancer health monitoring

### 15.7 Build & Run

**Build** (auto-detects podman/docker):
```bash
./build-docker.sh
# Produces: daemon-rag-agent:v4, daemon-rag-agent:latest
```

**Run with docker-compose**:
```bash
docker-compose up -d
# Access at http://localhost:7860
```

**Run directly**:
```bash
docker run -p 7860:7860 --env-file .env daemon-rag-agent:latest
```

**Monitor logs**:
```bash
docker logs -f daemon-rag-agent
```

### 15.8 Production Deployment

**Kubernetes Example**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: daemon-rag-agent
spec:
  replicas: 1
  containers:
  - name: daemon-rag
    image: daemon-rag-agent:latest
    ports:
    - containerPort: 7860
    volumeMounts:
    - name: data
      mountPath: /app/data
    livenessProbe:
      httpGet:
        path: /health
        port: 7860
      initialDelaySeconds: 60
      periodSeconds: 30
```

**File References**:
- `Dockerfile` - Multi-stage build definition
- `docker-compose.yml` - Service orchestration
- `docker-entrypoint.sh` - Initialization script (chmod +x)
- `build-docker.sh` - Build helper (auto-detects podman/docker)
- `docker-compose-helper.sh` - Compose wrapper for podman compatibility
- `.dockerignore` - Excludes data/, wiki/, pipeline/ (149GB)
- `utils/health_check.py` - Health endpoint implementation
- `DOCKER_README.md` - Comprehensive Docker documentation

---

## 16. Desktop Executable Build System [NEW 2025-12-12]

**Purpose**: Build standalone desktop executables using PyInstaller for distribution without requiring Python installation.

### 16.1 Architecture

```
SOURCE CODE
    ↓
[PyInstaller] ← daemon.spec configuration
    ↓
[Analysis] → Dependency detection, hidden imports
    ↓
[Custom Hooks] → hooks/hook-*.py for complex packages
    ↓
[Bundle] → One-dir mode (instant startup vs 30s one-file)
    ↓
dist/Daemon/
├── Daemon           # Main executable
├── _internal/       # Python runtime + packages
└── assets/          # Icons, splash screen
```

### 16.2 Key Files

| File | Purpose |
|------|---------|
| `daemon.spec` | PyInstaller configuration (one-dir mode, excludes, hooks) |
| `utils/bootstrap.py` | Frozen executable path resolution and data migration |
| `hooks/hook-*.py` | Custom hooks for chromadb, gradio, spacy, tiktoken, sentence_transformers |
| `hooks/runtime_hook.py` | Pre-import environment setup |
| `assets/` | Icons (ico, png) and splash screen |
| `docs/BUILD_GUIDE.md` | Comprehensive build documentation |

### 16.3 Bootstrap Flow (Frozen Mode)

```python
# main.py - CRITICAL: Bootstrap runs BEFORE any imports
if getattr(sys, 'frozen', False):
    multiprocessing.freeze_support()  # Windows fork-bomb prevention
    from utils.bootstrap import setup_environment
    setup_environment()  # Sets CORPUS_FILE, CHROMA_PATH to ~/.daemon/

# Then regular imports (config reads env vars correctly)
from config.app_config import config, CORPUS_FILE  # Now points to ~/.daemon/
```

### 16.4 Data Directory Strategy

| Mode | Location | Purpose |
|------|----------|---------|
| Development | `./data/` | Project-local data |
| Frozen (Linux) | `~/.daemon/` | User data directory |
| Frozen (Windows) | `%APPDATA%/Daemon/` | User data directory |
| Frozen (macOS) | `~/Library/Application Support/Daemon/` | User data directory |

**Migration**: On first frozen run, bootstrap attempts to migrate data from `dist/data/` or `dist/Daemon/data/` to user data directory.

### 16.5 Build Commands

```bash
# Prerequisites
pip install pyinstaller
python -m spacy download en_core_web_sm

# Build (10-20 minutes)
pyinstaller daemon.spec --clean --noconfirm

# Output
dist/Daemon/Daemon  # Linux/macOS executable
dist/Daemon/Daemon.exe  # Windows executable
```

### 16.6 Bundle Size

- **Estimated**: ~650MB (with torch CPU)
- **Excludes**: pytest, matplotlib, tensorflow, CUDA (optional)
- **Includes**: torch, transformers, sentence_transformers, chromadb, gradio, spacy

### 16.7 Browser Auto-Open

```python
# gui/launch.py - Robust browser opening for desktop mode
IS_FROZEN = getattr(sys, 'frozen', False)

if IS_FROZEN:
    # Use platform-specific commands (more reliable than webbrowser module)
    if sys.platform.startswith('linux'):
        subprocess.Popen(['xdg-open', local_url])
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', local_url])
    elif sys.platform == 'win32':
        os.startfile(local_url)
```

**File References**:
- `daemon.spec` - PyInstaller configuration
- `utils/bootstrap.py` - Path resolution, migration, environment setup
- `hooks/` - Custom PyInstaller hooks
- `assets/` - Application icons and splash screen
- `docs/BUILD_GUIDE.md` - Build documentation

---

**End of Skeleton**

This document compresses a ~50K line codebase by focusing on architecture, data flow, and patterns rather than implementation details.

**Last Updated**: 2026-02-17

**Recent Changes** (2026-02-17):
- **Temporal-Aware Recency Decay** (Section 2.3.0) — `rank_memories()` now reshapes decay curve for TEMPORAL_RECALL queries using `_temporal_anchor_hours` from IntentClassifier
- **Temporal Anchor Extraction** (Section 2.1.4) — `classify()` extracts temporal window and threads `_temporal_anchor_hours` through weight_overrides pipeline
- **Retrieval Quality Benchmarks** (Section 8.1b) — New benchmark suite with 30 seed memories, 19 test cases, real embeddings, and session-scoped fixtures
- **Wizard Test Env Fix** — Fixed `tests/test_wizard.py` env var poisoning (save/restore `OPENAI_API_KEY` in try/finally)

**Previous Changes** (2025-12-12):
- **Added Desktop Executable Build System (Section 16)** - PyInstaller-based build for standalone distribution
- **Created bootstrap module** - Handles frozen executable path resolution, data migration, and environment setup
- **Custom PyInstaller hooks** - For chromadb, gradio, spacy, tiktoken, sentence_transformers
- **Browser auto-open** - Platform-specific (xdg-open/open/startfile) for reliable desktop behavior

**Previous Changes** (2025-11-30):
- **Verified and updated all Protocol contracts** - Added MemoryScorerProtocol and MemoryConsolidatorProtocol with complete method signatures
- **Added Section 2.3.0 (MemoryScorer)** - Documented scoring and ranking operations with algorithm details
- **Updated all refactored module sections** - Corrected method signatures for MemoryRetriever, MemoryStorage, ThreadManager to match actual implementations
- **Contract compliance verified** - All 4 core modules (Scorer, Storage, Retriever, ThreadManager) confirmed fully compliant with Protocol contracts
- All async/sync specifications, parameter types, and return types verified accurate

**Previous Changes** (2025-11-29):
- Added Docker containerization section (Section 15) - Multi-stage builds, persistent volumes, graceful shutdown, health checks
- Added memory system refactoring documentation (coordinator.py V2 and modular components)
- Documented new files: memory_retriever.py, memory_storage.py, thread_manager.py, memory_interface.py
- Updated architecture diagrams to show modular memory flow
- Added refactoring notes and protocol contract information
