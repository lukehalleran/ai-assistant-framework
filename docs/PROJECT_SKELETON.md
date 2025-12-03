# Daemon RAG Agent - Project Skeleton

**Purpose**: Compressed architectural overview for LLM context windows. This skeleton captures the essential structure, data flow, and patterns without full implementation details.

**Last Updated**: 2025-12-02

---

## 1. Architecture Overview

```
USER QUERY
    ↓
[Orchestrator] ← Main entry point, coordinates all systems
    ↓
[Tone Detection] → Crisis level detection (keyword + semantic + LLM fallback)
    ↓
[Topic Extraction] → spaCy NER + optional LLM fallback
    ↓
[Heavy Topic Check] → Inline fact extraction for crisis/sensitive content
    ↓
[STM Analysis] → Short-term memory pass (topic, intent, tone, threads) [NEW]
    ↓
[Hybrid Memory Retrieval] → Query rewriting + Semantic + Keyword matching
    ↓               ↓ (via MemoryCoordinatorV2) [REFACTORED]
    ↓               ├─> MemoryRetriever (parallel ChromaDB queries)
    ↓               ├─> MemoryScorer (ranking with composite scores)
    ↓               └─> ThreadManager (conversation continuity)
    ↓
[Multi-Stage Gating] → FAISS → Cosine → Cross-Encoder
    ↓
[Prompt Building] → Token-budgeted context with separated sections + STM summary
    ↓
[Thinking Block] → LLM generates <thinking>...</thinking> + final answer
    ↓
[LLM Generation] → Multi-provider streaming with Best-of-N/Duel modes
    ↓
[Response Tag Stripping] → Remove <reply>, <response>, <answer>, EOS markers
    ↓
[Memory Storage] → Corpus JSON + Vector embeddings
    ↓               ↓ (via MemoryStorage) [REFACTORED]
    ↓               ├─> CorpusManager (JSON persistence)
    ↓               ├─> ChromaStore (vector embeddings)
    ↓               └─> FactExtractor (optional fact extraction)
    ↓
RESPONSE (thinking stripped) + MEMORY PERSISTENCE
```

**Note**: Memory system refactored (Nov 2025) into modular components:
- `memory/coordinator.py` (V2) - Thin orchestration layer
- `memory/memory_retriever.py` - Retrieval operations
- `memory/memory_storage.py` - Storage operations
- `memory/thread_manager.py` - Thread tracking
- `memory/memory_interface.py` - Protocol contracts

Legacy `memory/memory_coordinator.py` still in use but being migrated.

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

- `prepare_prompt(user_input, files, use_raw_mode)` → Prompt preparation
  - Tone detection (HIGH/MEDIUM/CONCERN/CONVERSATIONAL)
  - File processing (PDF/DOCX/CSV)
  - Heavy topic inline fact extraction
  - Optional query rewrite for retrieval
  - System prompt resolution with tone instructions
  - Thread context injection

- `_parse_thinking_block(response)` → Extract thinking from response
- `_strip_reflection_blocks(response)` → Remove reflections before storage
- `_strip_xml_wrappers(text)` → Remove `<result>`, `<answer>` wrappers
- `_strip_prompt_artifacts(text)` → Remove echoed prompt headers
- `_get_tone_instructions(tone_level)` → Mode-specific response guidelines
- `_get_session_headers_instructions()` → Temporal reasoning guidance

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

**Dependencies**: MemoryCoordinator, PromptBuilder, ResponseGenerator, TopicManager, ToneDetector, QueryChecker, STMAnalyzer

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

### 2.3 memory/memory_coordinator.py (Memory Hub - Legacy)
**Purpose**: Unified interface for all memory operations (original monolithic version)

**Status**: Still in use, but refactored into modular components in `memory/coordinator.py` (MemoryCoordinatorV2)

**5 Memory Types** (MemoryType enum):
1. `EPISODIC` - Raw conversation turns (query/response pairs)
2. `SEMANTIC` - Extracted facts and entities
3. `PROCEDURAL` - Learned patterns and behaviors
4. `SUMMARY` - Compressed conversation blocks
5. `META` - Reflections and meta-patterns

**Key Methods**:
- `get_memories(query, limit, topics)` → async retrieval pipeline
  ```python
  1. Get recent from corpus (last N conversations)
  2. Query 5 ChromaDB collections in parallel
  3. Gate results (MultiStageGateSystem)
  4. Rank by composite score (_rank_memories)
  5. Return top K
  ```

- `store_interaction(query, response, tags)` → persist to corpus + Chroma
- `_rank_memories(memories, query)` → Scoring algorithm:
  ```
  final_score =
    0.35 * relevance +
    0.25 * recency_decay +
    0.20 * truth_score +
    0.05 * importance +
    0.10 * continuity +
    0.15 * structural_alignment +
    anchor_bonus - penalties
  ```

**Hierarchical Memory**:
- Parent-child relationships between memories
- Temporal decay: `1.0 / (1.0 + decay_rate * age_hours)`
- Access count boosts truth score

**Dependencies**: CorpusManager, MultiCollectionChromaStore, MultiStageGateSystem, MemoryConsolidator

---

### 2.3.0 memory/memory_scorer.py **[NEW - REFACTORED]**
**Purpose**: Memory scoring and ranking operations with multi-factor composite scoring

**Key Methods** (implements MemoryScorerProtocol):
- `calculate_truth_score(query, response)` → Calculate truth/reliability score (0.0-1.0)
- `calculate_importance_score(content)` → Calculate importance based on content analysis (0.0-1.0)
- `rank_memories(memories, query, current_topic=None, is_meta_conversational=False)` → Rank memories by composite score for given query
- `update_truth_scores_on_access(memories)` → Boost truth scores when memories are accessed

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

### 2.3.1 memory/coordinator.py (Modular Memory Coordinator V2) **[NEW - REFACTORED]**
**Purpose**: Thin orchestration layer delegating to specialized components

**Architecture**:
```python
MemoryCoordinatorV2 (coordinator.py)
    ↓
├── MemoryScorer (memory_scorer.py) - Scoring and ranking
├── MemoryStorage (memory_storage.py) - Persistence operations
├── MemoryRetriever (memory_retriever.py) - Retrieval operations
└── ThreadManager (thread_manager.py) - Conversation thread tracking
```

**Key Methods** (same public API as legacy coordinator):
- `get_memories(query, limit, topics)` → Delegates to MemoryRetriever
- `store_interaction(query, response, tags)` → Delegates to MemoryStorage
- `process_shutdown_memory()` → Delegates to MemoryStorage

**Refactoring Benefits**:
- Single Responsibility: Each module handles one concern
- Easier testing: Components can be tested in isolation
- Better maintainability: Changes isolated to specific modules
- Protocol-based: Uses memory_interface.py contracts

**Dependencies**: MemoryScorer, MemoryStorage, MemoryRetriever, ThreadManager, CorpusManager, MultiCollectionChromaStore

---

### 2.3.2 memory/memory_retriever.py **[NEW - REFACTORED]**
**Purpose**: Memory retrieval operations extracted from coordinator

**Key Methods** (implements MemoryRetrieverProtocol):
- `async get_memories(query, limit, topic_filter)` → Main retrieval pipeline
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

**Dependencies**: CorpusManager, MultiCollectionChromaStore, FactExtractor, MemoryConsolidator, MemoryScorer

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

### 2.3.2 memory/user_profile.py & user_profile_schema.py (User Profile System) **[NEW - 2025-12-01]**
**Purpose**: ChatGPT-style categorized user profile with persistent fact storage and hybrid retrieval

**Architecture Overview**:
```
Query → LLMFactExtractor (categorizes) → UserProfile.add_facts_batch() → JSON persistence
                                              ↓
Query → get_user_profile_context(query) → hybrid retrieval → [USER PROFILE] in prompt
```

**Replaces**: Previous [SEMANTIC FACTS] and [RECENT FACTS] sections in prompts

**Key Components**:

**user_profile_schema.py**:
- `ProfileCategory` enum: 12 life domains (identity, education, career, projects, health, fitness, preferences, hobbies, study, finance, relationships, goals)
- `ProfileFact` dataclass: Structured facts with relation, value, category, confidence, timestamp, source_excerpt, supersedes
- `RELATION_CATEGORY_MAP`: 40+ predefined relation→category mappings
- `categorize_relation(relation)`: Direct lookup + heuristic fallbacks (pattern matching)

**user_profile.py**:
- `UserProfile` class: Persistent manager for categorized user facts
- Storage: `data/user_profile.json` with atomic writes (temp file + os.replace())
- Thread-safe: threading.Lock() for concurrent access
- Conflict resolution: Newer + higher confidence wins, tracks supersedes

**Key Methods**:
- `add_fact(relation, value, category, confidence, source_excerpt)` → Add single fact with conflict resolution
- `add_facts_batch(triples)` → Batch add from LLM extraction (used at shutdown)
- `get_relevant_facts(query, category, limit=3)` → **Hybrid retrieval (2/3 semantic + 1/3 recent)**:
  ```python
  # Semantic scoring: keyword overlap between query and fact values/relations
  query_words = set(query.lower().split())
  fact_words = set(value.lower().split()) | set(relation.lower().split())
  semantic_score = len(query_words & fact_words) / len(query_words)

  # Get top 2/3 by semantic score
  semantic_facts = top_scored[:semantic_count]

  # Get remaining 1/3 most recent (not in semantic set)
  recent_facts = sorted_by_timestamp[:recent_count]

  return semantic_facts + recent_facts
  ```
- `get_context_injection(max_tokens, query)` → Format profile for prompt injection
- `export_markdown()` → Generate markdown export of all facts by category
- `get_quick_profile()` → Extract identity fields (name, location, age) for compact summary

**Integration Points**:

1. **memory/memory_coordinator.py** (process_shutdown_memory):
   ```python
   # After LLM fact extraction
   if triples and hasattr(self, 'user_profile'):
       added = self.user_profile.add_facts_batch(triples)
   ```

2. **memory/llm_fact_extractor.py** (Enhanced):
   - Increased max_triples: 10 → 15
   - Increased max_tokens: 400 → 600
   - Auto-categorization via `categorize_relation()` in `_normalize_triple()`
   - Enhanced prompt with 12 category descriptions
   - Confidence scoring (0.0-1.0)
   - Case preservation for proper nouns

3. **memory/fact_extractor.py** (Enhanced regex patterns):
   - Temporal patterns: age, graduation year, experience duration
   - Location patterns: lives_in with various phrasings
   - Relationship patterns: pet names, family, friends
   - Goal patterns: plans, intentions

4. **core/prompt/context_gatherer.py** (NEW):
   - `get_user_profile_context(query, max_tokens=500)` → Calls UserProfile.get_context_injection()
   - Replaces old `get_facts()` and `get_recent_facts()` methods

5. **core/prompt/builder.py** (UPDATED):
   - Removed tasks: `semantic_facts`, `recent_facts`
   - Added task: `user_profile` → async call to get_user_profile_context()
   - Context dict changed: `{"user_profile": str}` replaces `{"semantic_facts": list, "fresh_facts": list}`

6. **core/prompt/formatter.py** (UPDATED):
   - Removed sections: `[SEMANTIC FACTS]`, `[RECENT FACTS]`
   - Added section: `[USER PROFILE]` → Pre-formatted string from context["user_profile"]

**CLI Commands** (main.py):
- `python main.py export-profile` → Write markdown to `data/user_profile_export.md`
- `python main.py show-profile` → Print profile to console with fact count

**Data Flow**:
```
User conversation
    ↓
process_shutdown_memory() (at session end)
    ↓
LLMFactExtractor.extract_batch() → categorized triples with confidence
    ↓
UserProfile.add_facts_batch() → conflict resolution + JSON persistence
    ↓
[Next conversation]
    ↓
get_user_profile_context(query) → hybrid retrieval per category
    ↓
[USER PROFILE] section in formatted prompt
```

**Example Profile Output**:
```
## Quick Profile
- name: Luke
- location: San Francisco
- age: 28

## fitness (3 facts)
- squat_max: 365 lb (confidence: 0.9)
- bench_max: 275 lb (confidence: 0.85)
- training_frequency: 5x per week (confidence: 0.8)

## career (2 facts)
- current_role: Senior Engineer (confidence: 0.9)
- programming_languages: Python, Go, Rust (confidence: 0.85)
```

**Testing**:
- tests/unit/test_user_profile_schema.py (5 tests): Schema validation, categorization, serialization
- tests/unit/test_user_profile.py (10 tests): Persistence, conflict resolution, hybrid retrieval, markdown export

**Dependencies**: threading, datetime, pathlib, logging

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
**Purpose**: Semantic search across 5 ChromaDB collections

**Collections**: episodic, semantic, procedural, summary, meta

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
- stm_summary: 10               # HIGHEST - metadata only, no token cost [NEW]
- recent_conversations: 7
- semantic_chunks: 6
- memories: 5
- semantic_facts/fresh_facts: 4
- summaries: 3
- reflections/dreams: 2
- wiki: 1
```

**Context Dict Structure**:
```python
{
  "stm_summary": {...},           # STM context summary dict (topic, intent, tone, etc.) [NEW]
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
  "wiki": [...]
}
```

**Key Methods**:
- `builder.py::build_prompt(user_input, config)` → Complete context dict
- `builder.py::_assemble_prompt(context, user_input)` → Format to string with headers
- `builder.py::_hygiene_and_caps(context)` → Dedupe with semantic similarity (0.90 threshold)
- `builder.py::_backfill_recent_conversations()` → Top-up after deduplication
- `builder.py::_build_lightweight_context()` → Minimal context for small-talk
- `token_manager.py::_manage_token_budget(context)` → Budget enforcement
- `context_gatherer.py::_get_summaries_separate()` → Hybrid recent+semantic retrieval
- `summarizer.py::_reflect_on_demand()` → Generate reflections if below threshold

**Deduplication Features**:
- Cross-section semantic deduplication using embedder
- SIMILARITY_THRESHOLD = 0.90 for duplicate detection
- Chunk stitching by title (max 4000 chars)
- Automatic backfill when dedup removes too many items

**Prompt Assembly Format**:
```
[TIME CONTEXT]
Current time: 2025-11-18 10:30:00

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

[SHORT-TERM CONTEXT SUMMARY] [NEW - placed immediately before query]
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

### 2.9 gui/ (Gradio Web Interface) **[MAJOR FIXES 2025-11-24]**
**Purpose**: Async streaming GUI with thinking block support and response tag handling

**Key Components**:

**gui/handlers.py** - Event handlers and streaming relay
- `handle_submit()` → Main async generator yielding response chunks to GUI
  - Receives streaming chunks from response_generator
  - Parses thinking blocks in real-time
  - Applies response tag stripping
  - Yields dict chunks: `{"role": "assistant", "content": text}`

**Tag Stripping Logic** (FIXED):
- Strips outer wrapper tags: `<reply>`, `<response>`, `<answer>`, `<result>`
- Uses backreference regex `\1` to match opening/closing tags properly
- Regex: `r"^\s*<\s*(result|reply|response|answer)\s*>\s*([\s\S]*?)\s*<\s*/\s*\1\s*>\s*$"`
- ONLY strips tags at start/end of string (not tags mentioned in content)
- Prevents truncation when LLM discusses tags in response

**gui/launch.py** - Gradio app setup and async iteration
- `submit_chat()` → Async iteration over handler chunks
  - Creates placeholder assistant message
  - Updates chat_history with streamed content
  - Handles thinking blocks with HTML collapsible sections
  - Yields to Gradio for real-time display

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
- `_map_alias_to_model(alias)` → "gpt-4o-mini" → "gpt-4o-mini"

**Environment Variables**:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY`

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

2. **spaCy NER extraction** (when ambiguous ~10-20ms):
   - Triggered when: heuristic is ambiguous (returns "general", >6 words, or vague terms)
   - Lazy-loaded spaCy model (en_core_web_sm)
   - Entity priority: PERSON > ORG > GPE > PRODUCT > EVENT > WORK_OF_ART > LAW > NORP
   - Filters: skip single-letter entities, skip stopwords/vague terms
   - Returns highest priority entity or None

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
- `_spacy_ner_extraction(text)` → str | None (spaCy NER)
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

### 2.13 utils/time_manager.py (Temporal Context)
**Purpose**: Time-aware operations and decay

**Key Functions**:
- `current()` → datetime.now()
- `current_iso()` → ISO-8601 string
- `calculate_decay(age_hours, decay_rate)` → float
  ```python
  decay = 1.0 / (1.0 + decay_rate * age_hours)
  ```

**Used By**: MemoryCoordinator for recency scoring

---

### 2.14 memory/memory_consolidator.py (Summarization)
**Purpose**: LLM-based conversation summarization

**Trigger**: Every N conversations (default 20) at shutdown

**Process**:
```python
1. Collect last N unsummarized conversations
2. Build summarization prompt
3. Call LLM (default: gpt-4o-mini)
4. Store summary as SUMMARY memory type
5. Link to parent conversations
```

**Key Methods**:
- `consolidate_memories(conversations, max_tokens)` → async summary

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

**3-Stage Detection Pipeline**:
1. **Observational check** - Is user discussing world events vs personal crisis?
2. **Keyword match** - Fast pattern matching for explicit crisis terms
3. **Semantic similarity** - Compare embeddings to crisis exemplars
4. **LLM fallback** - For borderline cases near thresholds

**Configuration** (env vars):
```python
TONE_THRESHOLD_HIGH = 0.58
TONE_THRESHOLD_MEDIUM = 0.50
TONE_THRESHOLD_CONCERN = 0.43
TONE_CONTEXT_WINDOW = 3  # Prior turns to check
TONE_ESCALATION_BOOST = 1.2  # Boost if prior distress detected
```

**Key Methods**:
- `detect_crisis_level(message, conversation_history, model_manager)` → async ToneAnalysis
- `_check_keyword_crisis(message)` → Fast keyword check
- `_semantic_crisis_detection(message, history)` → Embedding similarity
- `_llm_crisis_fallback(message, model_manager)` → LLM classification

**Returns**:
```python
ToneAnalysis(
  level=CrisisLevel.HIGH,
  confidence=0.72,
  trigger="semantic",
  raw_scores={"high": 0.72, "medium": 0.45, "concern": 0.38, "conversational": 0.35},
  explanation="Semantic similarity to crisis_support: 0.72"
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
- `is_deictic(query)` → True if refers to earlier context
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
- Factors: keyword overlap, time proximity, both heavy, same topic
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
3. corpus_manager.save()
```

---

## 5. Key Algorithms

### 5.1 Memory Ranking (memory_coordinator._rank_memories)
```python
def _rank_memories(memories, query):
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
│   ├── orchestrator.py        # Main controller (tone, STM, thinking blocks)
│   ├── stm_analyzer.py        # Short-term memory analyzer [NEW]
│   ├── response_generator.py  # LLM streaming + Best-of-N/Duel (FIXED)
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
│   ├── memory_coordinator.py      # Memory hub (legacy monolithic)
│   ├── coordinator.py             # Memory hub V2 (modular refactor) [NEW]
│   ├── memory_scorer.py           # Scoring and ranking operations [NEW - REFACTORED]
│   ├── memory_retriever.py        # Retrieval operations [NEW - REFACTORED]
│   ├── memory_storage.py          # Storage/persistence operations [NEW - REFACTORED]
│   ├── thread_manager.py          # Thread tracking [NEW - REFACTORED]
│   ├── memory_interface.py        # Protocol contracts [NEW - REFACTORED]
│   ├── corpus_manager.py          # JSON persistence
│   ├── memory_consolidator.py     # Summarization
│   ├── fact_extractor.py          # Pattern-based extraction
│   ├── llm_fact_extractor.py      # LLM-based extraction
│   ├── hybrid_retriever.py        # Query rewrite + semantic + keyword
│   └── storage/
│       └── multi_collection_chroma_store.py  # Vector DB
│
├── processing/
│   └── gate_system.py         # Multi-stage filtering
│
├── models/
│   ├── model_manager.py       # Multi-provider LLM client
│   └── tokenizer_manager.py   # Token counting
│
├── utils/
│   ├── topic_manager.py       # Topic extraction
│   ├── time_manager.py        # Temporal utilities
│   ├── tone_detector.py       # Crisis detection (keyword + semantic + LLM)
│   ├── need_detector.py       # Need-type detection (PRESENCE vs PERSPECTIVE) [NEW]
│   ├── emotional_context.py   # Combined emotional analysis (tone + need) [NEW]
│   ├── query_checker.py       # Query analysis + heavy topic + thread detection
│   ├── query_rewriter.py      # Query expansion for retrieval
│   ├── keyword_matcher.py     # Keyword overlap scoring
│   ├── logging_utils.py       # Centralized logging
│   ├── file_processor.py      # PDF/DOCX ingestion
│   └── conversation_logger.py # Conversation persistence
│
├── knowledge/
│   ├── WikiManager.py         # Wikipedia FAISS search
│   ├── semantic_search.py     # General semantic utilities
│   └── topic_manager.py       # Topic-specific utilities
│
├── gui/
│   ├── launch.py              # Gradio web interface (async chunk processing, tag stripping)
│   └── handlers.py            # UI event handlers (streaming response relay, thinking block support)
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
├── tests/                     # All test files (70+ files)
│   ├── unit/                  # Unit tests (20+ files)
│   │   ├── test_tone_detector.py
│   │   ├── test_need_detection.py  # [NEW]
│   │   ├── test_query_checker.py
│   │   ├── test_corpus_manager.py
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
├── docs/                      # Documentation
│   ├── PROJECT_SKELETON.md   # This file
│   ├── QUICK_REFERENCE.md    # Quick reference guide
│   ├── DOCKER_README.md      # Docker setup
│   └── ...
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

### 8.4 Test Count
- **Total test files**: 65
- **Unit tests**: ~20 files in `tests/unit/`
- **Integration tests**: ~45 files in `tests/`

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
7. **Summaries link to parents** - Hierarchical relationships preserved

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
| orchestrator.py | Main loop: tone → topic → heavy check → STM → prompt → thinking → LLM → store |
| stm_analyzer.py | Analyze: lightweight LLM pass for recent context summary (topic/intent/tone/threads) [NEW] |
| memory_coordinator.py | Hub (legacy): retrieve from 5 collections, gate, rank, return top K |
| coordinator.py | Hub V2: thin orchestration delegating to modular components [NEW - REFACTORED] |
| memory_scorer.py | Scoring: calculate truth/importance, rank by composite score with temporal decay [NEW - REFACTORED] |
| memory_retriever.py | Retrieval: get_memories pipeline with gating and ranking [NEW - REFACTORED] |
| memory_storage.py | Storage: store_interaction and fact extraction [NEW - REFACTORED] |
| thread_manager.py | Threads: conversation continuity tracking [NEW - REFACTORED] |
| memory_interface.py | Protocols: type contracts for memory components [NEW - REFACTORED] |
| corpus_manager.py | JSON CRUD: load/save/query short-term memories |
| multi_collection_chroma_store.py | Vector DB: embed, store, semantic search across 5 types |
| gate_system.py | Filter: FAISS → cosine → cross-encoder → top K |
| prompt/builder.py | Assemble: system + separated context sections + STM within 15K tokens |
| response_generator.py | Stream: async LLM + Best-of-N + Duel modes (buffer fix + DeepSeek EOS) [FIXED] |
| gui/handlers.py | Relay: streaming chunks + thinking blocks + tag stripping (reply/response/answer) [FIXED] |
| model_manager.py | Unified: OpenAI/Claude/OpenRouter/Local via single interface |
| topic_manager.py | Extract: Heuristics + LLM fallback (handles conversational/emotional) |
| memory_consolidator.py | Summarize: LLM compresses N conversations at shutdown |
| fact_extractor.py | Parse: regex + optional LLM for entity/fact extraction |
| WikiManager.py | Search: FAISS over Wikipedia dump for context injection |
| tone_detector.py | Detect: 4-level crisis (keyword + semantic + LLM fallback) |
| need_detector.py | Detect: need-type (PRESENCE vs PERSPECTIVE, keyword + semantic hybrid) |
| emotional_context.py | Combine: tone + need type for unified emotional analysis |
| query_checker.py | Analyze: heavy topics, thread detection, temporal windows |
| hybrid_retriever.py | Retrieve: query rewrite + semantic + keyword scoring |
| time_manager.py | Utils: timestamps and temporal decay calculations |

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

**End of Skeleton**

This document compresses a ~50K line codebase by focusing on architecture, data flow, and patterns rather than implementation details.

**Last Updated**: 2025-12-02

**Recent Changes** (2025-11-30):
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
