# Daemon RAG Agent - Project Skeleton

**Purpose**: Compressed architectural overview for LLM context windows. This skeleton captures the essential structure, data flow, and patterns without full implementation details.

**Last Updated**: 2026-01-23

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
- ⚠️ Parent-child relationships: **DISABLED** (caused retrieval bugs, intentionally deactivated)
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
- `get_context_injection(max_tokens, query)` → Format profile for prompt injection **[UPDATED 2025-12-04: now includes timestamps]**
  - Format: `relation=value [ISO_timestamp]`
  - Enables temporal reasoning about when facts were learned
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

**Example Profile Output (Prompt Format with Timestamps)**:
```
[USER PROFILE]
User: name=Luke, location=San Francisco [profile_updated: 2025-12-04T15:30:00]
fitness: squat_max=365 lb [2025-09-15T10:20:00]; bench_max=275 lb [2025-09-20T14:30:00]; training_frequency=5x per week [2025-10-01T08:15:00]
career: current_role=Senior Engineer [2025-08-01T09:00:00]; programming_languages=Python, Go, Rust [2025-08-05T11:30:00]
```

**Example Markdown Export**:
```
## Quick Profile
- name: Luke
- location: San Francisco
- age: 28

## fitness (3 facts)
- squat_max: 365 lb (confidence: 0.9)
- bench_max: 275 lb (confidence: 0.85)
- training_frequency: 5x per week (confidence: 0.8)
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
**Purpose**: Semantic search across 7 ChromaDB collections

**Collections**: conversations, summaries, wiki_knowledge, facts, reflections, obsidian_notes, reference_docs

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
- reference_docs: 5             # User uploaded docs
- memories: 5
- semantic_facts/fresh_facts: 4
- summaries: 3
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
- `formatter.py::_sanitize_embedded_headers(text)` → Escape section headers in memory content **[NEW 2026-01-08]**
  - Prevents prompt pollution when conversations discuss prompt structure
  - Converts `[RECENT CONVERSATION]` → `(RECENT CONVERSATION)` in stored memories
  - Applied to: memories, summaries, reflections, facts in builder.py

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

### 2.8a core/agentic/ (Agentic Search System) **[NEW 2026-01]**
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

**gui/handlers.py** - Event handlers, streaming relay, and agentic routing
- `handle_submit()` → Main async generator yielding response chunks to GUI
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

### 2.12.3 knowledge/obsidian_manager.py (Personal Notes Integration) **[NEW 2026-01]**
**Purpose**: Parse, embed, and retrieve user's personal notes from Obsidian vault

**Data Classes**:
- `EmbedResult`: Statistics from vault embedding (total_files, embedded_files, chunks, errors, duration)

**Key Methods**:
- `embed_vault(force_reindex)` → `EmbedResult`: Index vault to ChromaDB
- `get_notes(query, limit)` → `List[Dict]`: Hybrid retrieval (1/3 keyword + 2/3 semantic)
- `_keyword_search(query, limit)` → `List[Dict]`: Title/tag/content keyword matching
- `get_vault_stats()` → `Dict`: Index statistics
- `clear_index()` → `bool`: Clear ChromaDB collection

**Smart Chunking**:
```
Note < 1500 chars → Embed whole note
Note >= 1500 chars → Split by ## headers
```

**Metadata Extraction**:
- `#tags` - Preserved as comma-separated metadata
- `[[wiki links]]` - Extracted as related_notes references
- YAML frontmatter - Stripped from content

**Hybrid Retrieval**:
```
Query → _keyword_search() → 1/3 results (title/tag match priority)
      → ChromaDB semantic → 2/3 results (vector similarity)
      → Deduplicate by title → Combined results
```

**Configuration** (`config/app_config.py`):
```python
OBSIDIAN_ENABLED = True
OBSIDIAN_VAULT_PATH = "~/Documents/Luke Notes"
OBSIDIAN_CHUNK_THRESHOLD = 1500
OBSIDIAN_MAX_NOTES_PROMPT = 5
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
- ContextGatherer retrieves via `get_personal_notes()` method
- Builder adds `[USER'S PERSONAL NOTES]` section after dreams, before time context
- TokenManager includes `personal_notes` at priority 6 (high)
- Notes filtered through 3-stage gate system (Cosine → Blended → CrossEncoder)

---

### 2.12.4 knowledge/reference_docs_manager.py (Daemon Documentation / Self-Knowledge) **[NEW 2026-01]**
**Purpose**: Upload and retrieve reference documents that provide Daemon with self-knowledge (architecture docs, PROJECT_SKELETON, etc.)

**Data Classes**:
- `UploadResult`: Statistics from document upload (success, title, total_chunks, file_type, errors, duration)

**Key Methods**:
- `upload_document(file_path, title)` → `UploadResult`: Index file to ChromaDB
- `upload_text(content, title)` → `UploadResult`: Upload text directly (GUI paste)
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

### 2.12.5 utils/daily_notes_generator.py (Auto-Generated Daily Summaries) **[ENHANCED 2026-01-22]**
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
- Uses `gpt-4o-mini` with 800 max tokens

**Configuration** (`config/app_config.py`):
```python
DAILY_NOTES_ENABLED = True
DAILY_NOTES_FOLDER = "Vault/Daily Notes and To Do's"
DAILY_NOTES_MODEL = "gpt-4o-mini"
DAILY_NOTES_MAX_TOKENS = 800
TAG_GENERATION_ENABLED = True  **[NEW 2026-01-22]**
TAG_GENERATION_MODEL = "gpt-4o-mini"  **[NEW 2026-01-22]**
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

**Week Folder Structure**:
```
Obsidian vault/
└── Daily Notes and To Do's/
    └── Week 3 Jan 2026/
        ├── 1 13 26 Daily Note.md
        ├── 1 14 26 Daily Note.md
        └── Week 3 Jan 2026 Summary.md
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

**Configuration** (`config/app_config.py`):
```python
WEEKLY_NOTES_ENABLED = True
WEEKLY_NOTES_MODEL = "gpt-4o-mini"
WEEKLY_NOTES_MAX_TOKENS = 1200
```

**Integration**:
- Writes to Obsidian vault at `OBSIDIAN_VAULT_PATH / DAILY_NOTES_FOLDER / Week N Month Year/`
- Week summaries can be read by narrative context generator
- Atomic file writes (temp → replace pattern)
- **Tag generation** [NEW 2026-01-22]: Analyzes aggregated weekly content using TagGenerator to extract 5-10 contextual tags capturing recurring themes, patterns, and activities

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
**Purpose**: LLM-based conversation summarization and narrative context synthesis

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
- `generate_narrative_context(weeklies, monthlies)` → async narrative synthesis **[NEW 2026-01-17]**
- `_read_obsidian_weekly_summaries(limit)` → List[Dict] from Obsidian vault
- `_read_obsidian_daily_notes(limit)` → List[Dict] from Week * folders only

**Narrative Context Synthesis** [NEW 2026-01-17]:
Generates a "Current Life State" narrative from daily/weekly notes for temporal grounding.
- Primary source: Obsidian daily notes + weekly summaries
- Fallback: Corpus summaries
- Output: ~300 word synthesis covering life chapter, active threads, emotional trajectory, recurring themes
- Cached to `./data/narrative_context.txt` (0ms retrieval latency)
- CLI: `python main.py refresh-narrative`

---

### 2.14.1 Narrative Context System (Temporal Grounding) **[NEW 2026-01-17]**

**Purpose**: Provide trajectory-aware context by synthesizing daily/weekly notes into a cached "Life State" narrative.

**Architecture Decision**: Background cached updates (not per-query synthesis)
```
Per-Query Synthesis: +3-5s latency, ~5k tokens/query  ❌
Background Cached:   0ms latency, ~500 tokens/query   ✅
```

**Data Sources (Hybrid)**:
| Source | Priority | Count | Location |
|--------|----------|-------|----------|
| Obsidian Weekly Summaries | Primary | 2 | `Week */Week * Summary.md` |
| Obsidian Daily Notes | Primary | 7 | `Week */*Daily Note.md` |
| Corpus Summaries | Fallback | 3-5 | `corpus_v4.json` summaries |

**Note**: Only reads daily notes from `Week *` folders (new system created 2026-01-15). Older notes in root folder excluded - they're searchable via `[PERSONAL NOTES]`.

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
NARRATIVE_SYNTHESIS_MODEL = "gpt-4o-mini"
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
│   ├── orchestrator.py        # Main controller (tone, STM, coordinates subsystems)
│   ├── response_parser.py     # Response parsing utilities (thinking blocks, XML stripping) [NEW 2026-01-23]
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
│   ├── file_processor.py      # PDF/DOCX ingestion
│   ├── health_check.py        # Docker/K8s health endpoint
│   ├── conversation_logger.py # Conversation persistence
│   ├── web_search_trigger.py  # Web search detection (LLM-first + heuristics) [ENHANCED 2026-01]
│   ├── daily_notes_generator.py # Auto-generated daily summaries [ENHANCED 2026-01-22]
│   ├── weekly_notes_generator.py # Auto-generated weekly summaries [ENHANCED 2026-01-22]
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
│   └── reference_docs_manager.py  # Daemon self-knowledge docs [NEW 2026-01]
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
| daily_notes_generator.py | Generate: daily summaries with Life Events + tags [ENHANCED 2026-01-22] |
| weekly_notes_generator.py | Organize: weekly folders + summaries + tags [ENHANCED 2026-01-22] |
| tag_generator.py | Tags: LLM-based tag extraction for Obsidian notes (100+ vocabulary, 6 categories) [NEW 2026-01-22] |

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

**Last Updated**: 2025-12-12

**Recent Changes** (2025-12-12):
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
