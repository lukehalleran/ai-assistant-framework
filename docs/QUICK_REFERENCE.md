# Daemon RAG - Quick Reference API

**Ultra-compressed reference for LLM context windows. Function signatures and core logic only.**

---

## Core Entry Point

```python
# core/orchestrator.py
class DaemonOrchestrator:
    async def process_user_query(user_input: str, ...) -> Tuple[str, dict]:
        """Main entry: context pipeline → prompt build → LLM → store → response"""
        context = await self.context_pipeline.build(user_input)
        system_prompt = self._compose_system_prompt(context)
        prompt_ctx = await self.prompt_builder.build_prompt_from_context(context)
        response = await self._generate_response(prompt_ctx, system_prompt)
        await self.memory_system.store_interaction(user_input, response)
        return response, debug_info
```

---

## Context Pipeline **[NEW 2026-01-23]**

```python
# core/context_pipeline.py
class ContextPipeline:
    async def build(user_input: str, files: List = None, use_raw_mode: bool = False,
                    personality: Optional[str] = None, conversation_history: Optional[List[Dict]] = None) -> ContextResult:
        """
        Pipeline: topics‖tone → files → intent → heavy_check → rewrite → stm → identity → thread
        Returns ContextResult with all processed components.

        Stages (latency-optimized 2026-05-18):
        1+2. Topic extraction + Tone detection (PARALLEL via asyncio.gather)
             Tone fast-path: <8 words + no crisis keywords → skip semantic embedding
        3. File processing (FileProcessor)
        4a. Intent classification (regex-first, <1ms) — MOVED BEFORE heavy topic [2026-05-18]
        4b. Heavy topic check — SKIPPED for CASUAL_SOCIAL, META_CONVERSATIONAL,
            and short (<12 word) non-crisis queries [2026-05-18]
        5. Query rewriting (LLM) — SKIPPED for <10 words and casual intents [2026-05-18]
        6. STM analysis (STMAnalyzer)
           - 24h time-windowed recent_memories slice (was: last 8 fixed)
           - Internally injects last 2 daily notes for cross-day recall disambiguation
           - Output now includes reference_type + temporal_facts fields
           6b. STM intent refinement (low-confidence intents upgraded) [NEW 2026-02-15]
        7. Identity injection (UserProfile)
        8. Thread context (memory_system)
        """

@dataclass
class ContextResult:
    processed_query: str          # Rewritten query (or original)
    original_query: str           # Always preserved
    tone_level: ToneLevel         # CRISIS/ELEVATED/CONCERN/CONVERSATIONAL
    tone_instructions: str        # Mode-specific guidelines
    topics: List[str]             # Extracted topics
    primary_topic: Optional[str]  # Main topic
    file_context: Optional[str]   # Processed files
    thread_context: Optional[Dict]# Thread metadata
    stm_summary: Optional[Dict]   # STM analysis
    identity_block: str           # User identity
    is_heavy_topic: bool          # Heavy topic flag
    extracted_facts: List[Dict]   # Inline facts
    intent: Optional[IntentResult]# Intent classification [NEW 2026-02-15]
    is_small_talk: bool           # True when CASUAL_SOCIAL >= 0.70 [NEW 2026-04]

# Usage in orchestrator
context = await self.build_context(user_input, files, use_raw_mode)
# context.processed_query → for retrieval
# context.tone_instructions → for system prompt
# context.topics → for filtering

# Usage with prompt builder
prompt_ctx = await prompt_builder.build_prompt_from_context(context)
```

---

## Memory Operations

```python
# memory/memory_coordinator.py — Thin orchestrator (~551 lines, plus new component wiring)
# All methods delegate to modular components. No inline logic.
# Init: parallel disk reads via ThreadPoolExecutor(3) for graph, profile, claims [CHANGED 2026-05-19]
class MemoryCoordinator:
    def __init__(self, ...):
        self._retriever = MemoryRetriever(...)    # Retrieval operations
        self._storage = MemoryStorage(...)        # Persistence operations
        self._shutdown = ShutdownProcessor(...)   # Session-end processing
        self.scorer = MemoryScorer(...)           # Scoring and ranking
        self.thread_manager = ThreadManager(...)  # Thread tracking
        self.claim_index = ClaimIndex(...)     # Staleness tracking [NEW 2026-03-25]
        self.fact_verifier = FactVerifier(...)  # Pre-storage conflict gate [NEW 2026-03-24]
        self.context_surfacer = ContextSurfacer(...)  # Proactive insights [NEW 2026-03-24]
        self.user_profile = UserProfile()      # Shared with orchestrator [CHANGED 2026-05-19]

    # Retrieval — delegates to MemoryRetriever
    async def get_memories(query, limit=30, topics=None) -> List[Dict]:
        return await self._retriever.get_memories(query, limit, topics)

    async def get_semantic_top_memories(query, limit=10) -> List[Dict]:
        return await self._retriever.get_semantic_top_memories(query, limit)

    # Storage — delegates to MemoryStorage
    async def store_interaction(query, response, tags=None):
        await self._storage.store_interaction(query, response, tags)

    # Shutdown — delegates to ShutdownProcessor
    async def process_shutdown_memory(session_conversations=None):
        await self._shutdown.process_shutdown_memory(session_conversations)

    async def run_shutdown_reflection(session_conversations=None, session_summaries=None) -> bool:
        return await self._shutdown.run_shutdown_reflection(session_conversations, session_summaries)


# memory/memory_retriever.py — Retrieval pipeline (~966 lines)
class MemoryRetriever:
    async def get_memories(query, limit, topic_filter) -> List[Dict]:
        """
        1. Get recent from corpus (last N conversations)
        2. Query ChromaDB collections in parallel
        3. Combine recent + semantic results
        4. Apply multi-stage gating
        5. Rank with MemoryScorer
        6. Deduplicate by memory key
        7. Return top K
        """

    async def get_semantic_top_memories(query, limit=10) -> List[Dict]:
        """Top memories with meta-conversational routing and gating."""

    async def get_facts(query, limit=8) -> List[Dict]:
        """Semantic-primary fact ranking: 0.60*semantic + 0.20*confidence + 0.20*recency.
        Semantic floor: relevance < 0.30 caps recency weight at 0.05.
        Pulls limit*2 candidates for better selection."""

    async def get_skills(query, limit=5) -> List[Dict]:
        """Hybrid retrieval (1/3 recent + 2/3 semantic) from procedural_skills collection."""


# memory/memory_scorer.py — Scoring and ranking
class MemoryScorer:
    _intent_weight_overrides: Optional[Dict[str, float]] = None  # Set by PromptBuilder [NEW 2026-02-15]
    _graph_memory = None          # Set by PromptBuilder for graph-boosted scoring [NEW 2026-03]
    _entity_resolver = None       # Set by PromptBuilder for entity extraction [NEW 2026-03]

    def rank_memories(memories, current_query, current_topic=None, is_meta_conversational=False, weight_overrides=None) -> List[Dict]:
        """
        Score = weights['relevance']*rel + weights['recency']*recency + weights['truth']*truth
              + weights['importance']*importance + weights['continuity']*continuity
              + topic_match + structure + bonuses - penalties
        weight_overrides: per-intent SCORE_WEIGHTS override (from IntentClassifier)
        Falls back to _intent_weight_overrides if no explicit param.
        + staleness_penalty  # staleness_ratio * STALENESS_WEIGHT, 2x at 80%+, capped 0.4 [NEW 2026-03-25]

        Token overlap: _salient_tokens() and memory-side tokens are stemmed via _stem()
        (minimal suffix strip: -ment, -tion, -ing, etc.) to catch anxious/anxiety-type mismatches.
        Tag-keyword bonus: +0.15 max to continuity when query tokens match memory tags.
        Uses time_manager.current() instead of datetime.now() for testable time reference.

        Two-regime temporal decay [UPDATED 2026-05]:
        Small anchors (<=48h, "today"/"yesterday"): flat plateau inside window (~1.0),
        steep dropoff outside. Large anchors (>48h, "last week"/"last month"): peak near
        anchor, penalizes too-recent. Temporal anchor takes priority over time_manager
        and standard fallback.
        """

    def calculate_truth_score(query, response) -> float:
        """Base 0.5 + length/question/confirmation bonuses, capped at 1.0"""

    def calculate_importance_score(content) -> float:
        """Base 0.5 + length/question/keyword bonuses, capped at 1.0"""


# memory/shutdown_processor.py — Session-end processing [ENHANCED 2026-03]
class ShutdownProcessor:
    def __init__(..., memory_coordinator=None, claim_index=None):
        """memory_coordinator enables pipeline-enriched proposal generation.
        claim_index enables staleness tracking on summary creation. [NEW 2026-03-25]"""

    async def process_shutdown_memory(session_conversations=None):
        """Parallelized pipeline [CHANGED 2026-05-19]:
        1) block summaries (source IDs via get_ids_by_timestamp_range),
        Phase A (parallel): session facts, LLM facts, behavioral patterns, procedural skills,
        Phase B (parallel): proposals, impl tracking, threads, synthesis dreaming, wiki enrichment,
        Phase C (sequential): graph save, category cache, cross-dedup (dry_run only).
        After summary storage: extract claims → register in ClaimIndex → add staleness metadata"""

    async def _run_cross_collection_dedup():
        """User mode (CROSS_DEDUP_AUTO_EXECUTE=True): executes deletions automatically.
        Dev mode: dry-run only — logs findings but NEVER deletes. Guarded to run once per process."""

    async def run_shutdown_reflection(session_conversations=None, session_summaries=None) -> bool:
        """LLM reflection generation + ChromaDB storage"""


# memory/cross_deduplicator.py — Cross-collection dedup [NEW 2026-02-13]
class CrossCollectionDeduplicator:
    def __init__(chroma_store, claim_index=None, entity_resolver=None):
        """Thresholds from CROSS_DEDUP_* config. Skips protected collections.
        After contradiction detection, cascades staleness via ClaimIndex. [NEW 2026-03-25]"""

    def run(dry_run=True) -> DedupPlan:
        """
        1. Load docs from target collections (facts, summaries, skills, proposals, reflections)
        2. Compute embeddings via chroma_store.embedding_fn
        3. Pairwise cosine → find duplicates >= 0.92 threshold
        4. Group facts by (subject, predicate) → find contradictions (different objects)
        5. Skip ephemeral predicates (current_feeling, is, has, thinks, etc.)
        6. If not dry_run: execute deletions
        Returns DedupPlan with full audit trail.
        """

    # Keep/delete priority: summaries(5) > reflections(4) > skills(3) > proposals(2) > facts(1)
    # Protected (never scanned): conversations, obsidian_notes, reference_docs, wiki_knowledge
    # GUI: Status tab → "Preview Dedup" (dry_run) / "Run Dedup" (live)

# Config:
CROSS_DEDUP_ENABLED = True
CROSS_DEDUP_DUPLICATE_THRESHOLD = 0.92
CROSS_DEDUP_CONTRADICTION_THRESHOLD = 0.85
CROSS_DEDUP_ON_SHUTDOWN = True  # dry_run=True only, never auto-deletes
CROSS_DEDUP_MAX_DOCS_PER_COLLECTION = 1000  # Per-collection cap

# Protected (never scanned): conversations, obsidian_notes, reference_docs, wiki_knowledge
```

---

## Escalation Tracker (Crisis Cooldown) **[NEW 2026-02-13]**

```python
# core/escalation_tracker.py
class ResponseStrategy(Enum):
    VALIDATE_AND_SUGGEST   # Default: validate + offer suggestions
    GROUNDING_PRESENCE     # Sustained escalation: drop advice, 2-3 sentences max
    QUIET_COMPANIONSHIP    # Sustained + ignored suggestions: 1-2 sentences, just be present
    GENTLE_REENGAGEMENT    # After de-escalation: carefully re-engage

class EscalationTracker:
    def __init__(escalation_threshold=3, deescalation_window=2, max_history=10):
        """Tracks consecutive elevated/calm counts, suggestion engagement, tone velocity."""

    def update(tone_level, user_message, need_type=None) -> ResponseStrategy:
        """Update with new message, return strategy. Tracks engagement with previous suggestions."""

    def record_response(response) -> None:
        """Extract suggestions from response for engagement detection on next turn."""

    def get_strategy_instructions() -> str:
        """Supplemental tone instructions appended to system prompt (or empty for default)."""

    def get_token_budget_override() -> Optional[int]:
        """QUIET=300, GROUNDING=500, GENTLE=800, default=None."""

    def get_escalation_velocity() -> float:
        """0.0 (stable) to 1.0 (rapid escalation) from last 5 tone values."""

    def get_debug_info() -> Dict:
        """Strategy, counts, velocity for debug logging."""

# Strategy transitions:
# VALIDATE → (threshold+ elevated) → GROUNDING → (2+ ignored) → QUIET
#   → (tone drops) → GENTLE → (sustained calm) → VALIDATE
# De-escalation nuance: need_type="PERSPECTIVE" = intensity shift → VALIDATE (not GENTLE)

# Integration in orchestrator:
# tracker.update(tone_level, user_message) → after build_context()
# tracker.get_strategy_instructions() → append to system prompt
# tracker.get_token_budget_override() → override tone-based budget
# tracker.record_response(response) → after generation

# Config (app_config.py):
ESCALATION_ENABLED = True
ESCALATION_THRESHOLD = 3          # Consecutive elevated before strategy shift
ESCALATION_DEESCALATION_WINDOW = 2  # Consecutive calm before gentle ends
```

---

## Intent Classifier **[NEW 2026-02-15]**

```python
# core/intent_classifier.py — Regex-first query intent classification (no LLM calls)
class IntentType(str, Enum):
    FACTUAL_RECALL       # "What's my dog's name?", "do you see X", "show me X"
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

class IntentClassifier:
    def classify(query: str, tone_level: str = None) -> IntentResult:
        """Regex patterns checked in order, highest confidence wins.
        Tone bias: HIGH/MEDIUM → EMOTIONAL_SUPPORT for ambiguous queries.
        TEMPORAL_RECALL: extracts temporal anchor via extract_temporal_window(),
        adds _temporal_anchor_hours to weight_overrides. [UPDATED 2026-02-17]"""

    def refine_with_stm(result: IntentResult, stm_intent: str = None) -> IntentResult:
        """Upgrade low-confidence results (< 0.50) using STM free-text intent.
        Keyword-matches STM text into IntentType. No extra LLM call."""

# Per-intent profiles (_PROFILES dict) define:
#   weights: {relevance, recency, truth, ...} overrides for SCORE_WEIGHTS
#   retrieval: {max_mems, max_recent, max_summaries, ...} overrides for PROMPT_MAX_*
#   gate: threshold override for gate system
# Phase 8 added retrieval keys: max_reference_docs, max_narrative,
#   max_user_uploads, max_proactive, max_personal_notes
# Setting max_X=0 skips retrieval task entirely (no async call launched)

# Integration flow:
# ContextPipeline Stage 4.5 → classify() → IntentResult on ContextResult
# ContextPipeline Stage 6b  → refine_with_stm() after STM analysis
# PromptBuilder             → extracts retrieval_overrides + weight_overrides
# MemoryScorer              → uses weight_overrides in rank_memories()

# Config (app_config.py):
INTENT_ENABLED = True
INTENT_STM_REFINEMENT_THRESHOLD = 0.50
PROMPT_SECTION_GATING_ENABLED = True  # Phase 8 eval-driven gating
```

---

## Gating System

```python
# processing/gate_system.py
class MultiStageGateSystem:
    async def filter_memories(query: str, memories: List[Dict]) -> List[Dict]:
        """
        Stage 1: FAISS semantic search → top 50
        Stage 2: Cosine threshold (0.45) → ~20-30
        Stage 3: Cross-encoder rerank → top k
        """
        # Stage 1: FAISS
        query_emb = self.embed(query)
        faiss_candidates = self.faiss_index.search(query_emb, 50)

        # Stage 2: Cosine filter
        filtered = []
        for mem in faiss_candidates:
            mem_emb = self.embed(mem['text'])
            similarity = cosine_similarity(query_emb, mem_emb)
            if similarity >= self.threshold:  # 0.45
                mem['relevance_score'] = similarity
                filtered.append(mem)

        # Stage 3: Cross-encoder rerank
        pairs = [(query, m['text']) for m in filtered]
        scores = self.cross_encoder.predict(pairs)
        for i, mem in enumerate(filtered):
            mem['relevance_score'] = scores[i]

        return sorted(filtered, key=lambda m: m['relevance_score'], reverse=True)[:k]

# Post-gate filter for personal notes (builder.py) [NEW 2026-03-20]
# After filter_memories(), notes below PERSONAL_NOTES_GATE_THRESHOLD (0.30)
# are dropped. General gate threshold is 0.18; personal notes need stricter filtering.
gated_notes = [n for n in gated_notes if n.get("relevance_score", 0) >= PERSONAL_NOTES_GATE_THRESHOLD]
```

---

## Prompt Building

```python
# core/prompt/builder.py — Thin orchestrator (delegates assembly to formatter.py, hygiene to hygiene.py,
# retrieval to gatherer_memory.py/gatherer_knowledge.py/gatherer_web.py via context_gatherer.py compositor)
class UnifiedPromptBuilder:
    def build_prompt(query: str, memories: List[Dict], topics: List[str] = None, budget: int = 2048,
                     retrieval_overrides: Dict[str, int] = None, weight_overrides: Dict[str, float] = None) -> str:
        """
        Token allocation:
        1. System prompt (fixed ~500-800)
        2. Query (variable)
        3. Recent context (30% of remaining, ~500 tokens)
        4. Episodic memories (35%, ~800 tokens)
        5. Semantic facts (15%, ~400 tokens)
        6. Summaries (10%, ~300 tokens)
        7. Wiki context (10%, ~200 tokens)
        """
        sections = []

        # System prompt
        sections.append(self._load_system_prompt())

        # Recent conversation context
        recent = self._format_recent_context(budget * 0.3)
        sections.append(recent)

        # Memories by type
        episodic = [m for m in memories if m.get('memory_type') == 'episodic']
        semantic = [m for m in memories if m.get('memory_type') == 'semantic']
        summaries = [m for m in memories if m.get('memory_type') == 'summary']

        sections.append(self._format_memories(episodic, budget * 0.35))
        sections.append(self._format_memories(semantic, budget * 0.15))
        sections.append(self._format_memories(summaries, budget * 0.10))

        # Wiki context if relevant (ChromaDB wiki_knowledge first, live API fallback)
        if self._should_use_wiki(query, topics):
            wiki_ctx = self.wiki_manager.search(query, k=3)
            sections.append(self._format_wiki(wiki_ctx, budget * 0.10))

        # Current query
        sections.append(f"User: {query}")

        return "\n\n".join(sections)
```

---

## Response Parsing

```python
# core/response_parser.py
class ResponseParser:
    """Pure static methods for parsing/cleaning LLM responses"""

    @staticmethod
    def parse_thinking_block(response: str) -> Tuple[str, str]:
        """Extract thinking content and final answer. Three layers:
        1. Tag-based: <thinking>/<think>/<output> extraction
        2. Heuristic fallback: _detect_untagged_thinking() for models that dump
           reasoning without tags (meta-reasoning, instruction echoes, ≥2 pattern hits)
        3. Tag cleanup: strip_thinking_tag_leaks() for partial/malformed tags"""
        # Returns (thinking_content, final_answer)
        # If no thinking block: ("", full_response)

    @staticmethod
    def _detect_untagged_thinking(response: str) -> Tuple[str, str]:  # [NEW 2026-04-05]
        """Heuristic fallback: pattern-based detection of untagged chain-of-thought.
        Requires ≥2 distinct patterns, ≥20 char remaining answer."""

    @staticmethod
    def has_incomplete_thinking_block(response: str) -> bool:  # [NEW 2026-03-26]
        """True if opening <thinking>/<think> tag present but closing tag not yet.
        Used during streaming to suppress raw thinking text."""

    @staticmethod
    def extract_incomplete_thinking(response: str) -> str:  # [NEW 2026-03-26]
        """Extracts content after opening think tag when close not yet arrived."""

    @staticmethod
    def strip_reflection_blocks(response: str) -> str:
        """Remove <reflect>...</reflect> and [SYSTEM QUALITY REFLECTION]..."""

    @staticmethod
    def strip_xml_wrappers(text: str) -> str:
        """Remove <result>, <answer>, <final>, <output>, <response> wrappers"""  # [ENHANCED 2026-03-26]

    @staticmethod
    def strip_agentic_tool_tags(text: str) -> str:  # [NEW 2026-05-20]
        """Strip entire <tool>...</tool> blocks (search, memory, github, etc.) leaked into non-agentic responses."""

    @staticmethod
    def strip_prompt_artifacts(text: str) -> str:
        """Remove echoed prompt headers: [TIME CONTEXT], [FACTS], [RECENT CONVERSATION], etc."""
```

---

## LLM Generation

```python
# core/response_generator.py
class ResponseGenerator:
    async def generate_response_stream(prompt: str, model: str = "gpt-4o-mini") -> AsyncIterator[str]:
        """Stream LLM response chunks.
        Detects reasoning/thinking content in streaming chunks and emits synthetic
        <thinking>/<\/thinking> wrapper tags for downstream handling."""  # [ENHANCED 2026-03-26]
        async for chunk in self.model_manager.generate_stream(prompt, model):
            yield chunk

    async def generate_best_of(prompt: str, n: int = 3) -> str:
        """Generate N responses with temp variation, score and pick best"""

    async def generate_duel_and_judge(prompt, model_a, model_b, judge_model) -> dict:
        """Two models compete, judge picks winner. Returns {'answer', 'winner', ...}"""

    async def generate_best_of_ensemble(prompt, generators, selectors) -> str:
        """Multiple generators + multiple judges with weighted scoring"""
```

---

## Best-of Handler [NEW]

```python
# core/best_of_handler.py
@dataclass
class BestOfResult:
    response: str        # Final answer
    used_best_of: bool   # True if best-of was used
    mode: str            # "duel", "ensemble", "single", "fallback"
    metadata: Dict       # Raw duel data, scores, etc.

class BestOfHandler:
    """Orchestrates best-of-N response generation modes"""

    def should_use_best_of(user_input: str, use_raw_mode: bool = False) -> bool:
        """Check if query qualifies (is question, meets token threshold)"""
        cfg = self._get_runtime_config()
        if not cfg["enable"] or use_raw_mode:
            return False
        qinfo = analyze_query(user_input)
        return qinfo.is_question and qinfo.token_count >= cfg["min_tokens"]

    async def generate(prompt, user_input, system_prompt, model_name) -> BestOfResult:
        """
        Mode selection:
        1. Duel: 2 generators + 1 judge (duel_mode=True, 2 generators)
        2. Ensemble: N generators + M judges (generators configured)
        3. Single: 1 model, N temps (fallback)

        Falls back to streaming on timeout (latency_budget_s).
        """
        cfg = self._get_runtime_config()
        if cfg["duel_mode"] and len(cfg["generator_models"]) == 2:
            return await self._generate_duel(...)
        elif cfg["generator_models"]:
            return await self._generate_ensemble(...)
        else:
            return await self._generate_single(...)

# Config (app_config.py):
ENABLE_BEST_OF = True
BEST_OF_DUEL_MODE = False
BEST_OF_GENERATOR_MODELS = ["sonnet-4.5", "gpt-5"]  # Duel needs exactly 2
BEST_OF_SELECTOR_MODELS = ["gpt-4o-mini"]           # Judges
BEST_OF_LATENCY_BUDGET_S = 2.0                      # Timeout before fallback
```

---

## Storage Layers

```python
# memory/corpus_manager.py
class CorpusManager:
    def add_memory(memory: Dict, memory_type: MemoryType):
        """Append to JSON, save atomically"""
        self.data['conversations'].append(memory)
        self.save()

    def get_recent_memories(n: int, memory_type: MemoryType = None) -> List[Dict]:
        """Return last N memories, optionally filtered by type"""
        memories = self.data['conversations']
        if memory_type:
            memories = [m for m in memories if m.get('memory_type') == memory_type.value]
        return memories[-n:]

    def save():
        """Atomic write: temp file → rename"""
        temp_path = self.corpus_path + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
        os.rename(temp_path, self.corpus_path)


# memory/storage/multi_collection_chroma_store.py
class MultiCollectionChromaStore:
    # Collections (14 total): conversations, summaries, wiki_knowledge, facts, reflections, obsidian_notes, reference_docs, procedural, procedural_skills, proposals, threads, synthesis_results, visual_memories, daemon_self_notes

    async def add_memory(text: str, metadata: Dict, collection: str):
        """Embed text and store in ChromaDB collection"""
        embedding = await self._embed(text)
        self.client.get_collection(collection).add(
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[text],
            ids=[metadata['id']]
        )

    async def query(text: str, collection: str, n_results: int = 10) -> List[Dict]:
        """Semantic search in collection"""
        embedding = await self._embed(text)
        results = self.client.get_collection(collection).query(
            query_embeddings=[embedding],
            n_results=n_results
        )
        return self._parse_results(results)

    def get_by_id(collection_name: str, doc_id: str) -> Optional[Dict]:  # [NEW 2026-03-26]
        """Direct document lookup by UUID → {id, content, metadata} or None"""

    # Per-request embedding cache [NEW 2026-05-18]
    def embed_query(text: str) -> list:
        """Pre-embed query text. Returns 384-dim vector for BGE-small."""
    def clear_embedding_cache() -> None:
        """Reset per-request cache. Call at start of each prompt build."""
    def _cached_embed(text: str) -> list:
        """Embed with cache — same text across parallel tasks shares one embedding."""
    # query_collection() auto-checks cache before re-embedding.
    # query_multiple_collections() pre-embeds once for all 4+ collection queries.
```

---

## Code Proposals System **[NEW 2026-02-05]**

```python
# memory/code_proposal.py — Pydantic data models + supervision fields
class CodeProposal(BaseModel):
    id: str                        # UUID
    title: str                     # Short title
    proposal_type: ProposalType    # feature|refactor|bugfix|test|docs|infra
    status: ProposalStatus         # PENDING|APPROVED|REJECTED|COMPLETED|FAILED
    source: ProposalSource         # GOAL_DIRECTED|SESSION_INSIGHT|USER_REQUEST|SHUTDOWN_ANALYSIS|AGENT_BRANCH
    priority: int                  # 1-10 (LLM-assigned)
    reasoning: str                 # Why this change is needed
    description: str               # Detailed description
    implementation_steps: List[ImplementationStep]  # Ordered steps
    affected_files: List[str]      # File paths
    tags: List[str]                # Keywords
    estimated_complexity: str      # low|medium|high
    # Supervision fields (branch gating)
    risk_level: RiskLevel          # low|medium|high|critical
    touches_core_system: bool      # Modifies orchestrator/pipeline/coordinator/safety?
    depends_on: List[str]          # Proposal IDs that must complete first
    test_files: List[str]          # Actual test file paths
    outcome: Optional[ProposalOutcome]  # Review record (accepted, notes, merged_at, reviewed_by)

    def to_metadata() -> dict:     # Flat dict for ChromaDB
    def from_metadata(meta) -> CodeProposal:  # Reconstruct from ChromaDB (backward-compat)
    def record_outcome(accepted, notes, merge_branch, reviewed_by):  # Human review outcome

# config/feature_registry.py — Retrospective shipped-feature catalog
load_registry() -> List[FeatureEntry]           # Cached YAML loader
get_feature(proposal_id) -> FeatureEntry        # Single lookup
get_dependencies(proposal_id) -> List[str]      # Transitive BFS
get_core_features() -> List[FeatureEntry]       # touches_core_system=True
check_conflicts(files) -> List[FeatureEntry]    # File overlap detection


# knowledge/proposal_generator.py — LLM-based proposal generation
class GoalDirectedGenerator:
    def __init__(model_manager, repo_path=".", model_alias="claude-opus-4.6", max_proposals=5):
        pass

    async def generate_proposals(extra_context="", max_proposals=None) -> List[CodeProposal]:
        """
        1. Read CLAUDE.md, QUICK_REFERENCE.md, PROJECT_SKELETON.md (filtered), GOALS.md, git log
        2. Build creative prompt with full project context
        3. Call LLM for structured JSON
        4. Parse + validate into CodeProposal objects
        """

    async def generate_proposals_with_context(pipeline_context, extra_context="", max_proposals=None) -> List[CodeProposal]:
        """Same as above but with pre-gathered rich context from Daemon retrieval pipeline."""

    async def generate_code_for_proposal(proposal, output_dir=None) -> Dict:
        """
        Generate actual implementation code for every step.
        1. For each step: read source (modify), build prompt, call LLM
        2. Write files to staging: data/proposal_code/<proposal_id>/
        3. Create _manifest.json
        Returns: {proposal_id, output_dir, files: {path: content}, errors: [...]}
        """


# memory/proposal_store.py — ChromaDB-backed storage
class ProposalStore:
    def __init__(chroma_store: MultiCollectionChromaStore):
        pass

    def store_proposal(proposal: CodeProposal) -> str:
        """Embed title+reasoning, store in 'proposals' collection. Returns doc_id."""

    def query_proposals(query_text, n_results=20, status=None, proposal_type=None) -> List[CodeProposal]:
        """Semantic search with optional status/type filters."""

    def check_similarity(proposal, threshold=0.70) -> bool:
        """True if similar proposal exists (cosine OR title word overlap >= 0.60 Jaccard)."""

    def get_for_dedup(limit=25) -> str:
        """Get recent proposal titles for dedup context in generator prompt."""


# core/prompt/proposal_filter.py — Prompt-injection retrieval/ranking pipeline [NEW 2026-02-09]
class ProposalFilter:
    def __init__(chroma_store=None, gate_system=None, model_manager=None):
        pass

    async def get_proposals(query, limit=3) -> List[Dict]:
        """
        Full pipeline: project-relevance check → ProposalStore query (utility query)
        → keyword dedup → semantic dedup → gate scoring → composite scoring
        → novelty penalty → diversity selection → optional LLM pairwise ranking.
        Returns [{content, metadata, relevance_score}, ...] ready for prompt.
        """

    # Composite score = w_priority(0.30) * priority + w_breadth(0.20) * breadth
    #                 + w_recency(0.10) * recency + w_goal(0.40) * goal_alignment
    # Novelty penalty: multiplicative factor from git commit overlap (0.5-1.0)
    # Diversity: overlap-coefficient anti-clustering (threshold 0.34)


# memory/shutdown_processor.py — Pipeline-enriched proposal generation [ENHANCED 2026-02-09]
class ShutdownProcessor:
    def __init__(..., memory_coordinator=None):
        """memory_coordinator enables pipeline-enriched proposal generation."""

    async def _generate_proposals(session_conversations):
        """Routes through pipeline-enriched path (rich context from MemoryCoordinator)
        or falls back to _generate_proposals_cold() (truncated excerpts + file reads only)."""

    async def _generate_proposals_cold(sess_items, generator, dedup_context):
        """Original fallback: conversation excerpts + file reads only."""

    async def _gather_proposal_context(sess_items) -> str:
        """Parallel retrieval: memories, skills, facts, reflections, summaries,
        git commits, user profile. Builds structured context sections."""


# Config (app_config.py):
CODE_PROPOSALS_ENABLED = True
CODE_PROPOSALS_MAX_PER_SESSION = 5
CODE_PROPOSALS_DEDUP_THRESHOLD = 0.70
# Proposal generation context: CLAUDE.md, QUICK_REFERENCE.md (full),
# PROJECT_SKELETON.md (filtered, excludes Core Components section),
# GOALS.md, git log (last 10 commits). All read live from disk.
# Prompt integration:
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

---

## File Upload System **[ENHANCED 2026-02-10]**

```python
# utils/file_processor.py — Processes text documents and images
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}

@dataclass
class ProcessedFile:
    filename: str; extension: str; content_text: str; file_path: str
    is_image: bool; media_type: str; base64_data: str; file_size: int; error: str

@dataclass
class ProcessedFilesResult:
    text_content: str               # Backward-compatible merged text
    processed_files: List[ProcessedFile]  # All files
    images: List[ProcessedFile]     # Image files only (base64 encoded)
    documents: List[ProcessedFile]  # Text documents only

class FileProcessor:
    async def process_files(user_text, files) -> str:
        """Legacy API — delegates to process_files_structured(), returns .text_content"""

    async def process_files_structured(user_text, files) -> ProcessedFilesResult:
        """New API — returns structured result with separate image/document handling.
        Images: base64 encoded, saved to data/uploads/<timestamp>_<filename>
        Documents: text extracted, security validated (path traversal, size limits, CSV sanitization)
        PDF: pdfplumber with table extraction — tables rendered as markdown pipe tables
        DOCX: python-docx with table extraction — tables rendered as markdown pipe tables
        XLSX: openpyxl — each sheet rendered as a markdown table with heading
        JSON/YAML/HTML/XML: read as plain text, wrapped in code fences
        MD/TXT/LOG: read as plain text
        """

# gui/handlers.py — Upload persistence (fire-and-forget background task)
async def _persist_uploads(orchestrator, files_result):
    """Store uploads to ChromaDB reference_docs with type='user_upload' metadata.
    Images also store is_image, media_type, image_path metadata.
    Upload images injected into note_images for multimodal model use."""

# core/prompt/gatherer_knowledge.py — Retrieval (KnowledgeRetrievalMixin)
async def get_user_uploads(query, limit=5) -> List[Dict]:
    """Fetch from reference_docs, filter to type='user_upload' only."""
async def get_reference_docs(query, limit) -> List[Dict]:
    """Fetch from reference_docs, filter OUT type='user_upload'."""

# Config (app_config.py):
FILE_UPLOAD_ALLOWED_EXTENSIONS = ['.txt', '.md', '.json', '.yaml', '.yml', '.log', '.html', '.xml', '.docx', '.xlsx', '.csv', '.py', '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.webp']  # expanded [2026-05-18]
FILE_UPLOAD_IMAGE_DIR = "data/uploads"
PROMPT_MAX_USER_UPLOADS = 5
```

---

## Agent Session Safety

```python
# utils/fs_snapshot.py — Filesystem manifest for agent session snapshots
sha256_file(path: str | Path) -> str
should_exclude(rel: str) -> bool
iter_snapshot_files(root: str | Path) -> list[Path]
create_manifest(root: str | Path) -> dict[str, dict]   # {rel_posix: {size, mtime, sha256}}
save_manifest(manifest, path) -> None
load_manifest(path) -> dict
diff_manifests(before, after) -> dict                   # {added: [], deleted: [], modified: []}
# CLI: python -m utils.fs_snapshot create ROOT OUTPUT
# CLI: python -m utils.fs_snapshot diff BEFORE AFTER

# utils/destructive_op_guard.py — Git command classifier
classify_git_args(args: list[str]) -> dict              # {subcmd, destructive: bool, reason}
is_destructive_git_args(args: list[str]) -> bool
unlock_allowed(env=None, root=None) -> bool             # Checks ALLOW_DESTRUCTIVE_OPS=1 or .agent_allow_destructive_once

# utils/shell_cmd_guard.py — Shell command classifier (rm, mv, rmdir, chmod, truncate, find)
classify_shell_cmd(args: list[str], repo_root=None) -> dict  # {command, destructive, severity, reason, targets}
is_destructive_shell_cmd(args: list[str], repo_root=None) -> bool
# PROTECTED_DIRS: 16 dirs (.git, data/, config/, memory/, core/, etc.)
# PROTECTED_FILES: 6 files (main.py, daemon.spec, pytest.ini, pyproject.toml, etc.)
# ALWAYS_BLOCKED_TARGETS: ., .., /, ~, * — cannot be unlocked
# Classifiers for: rm, mv, rmdir, chmod, truncate, find

# utils/python_fs_guard.py — Python filesystem monkey-patch guard (10 patched functions)
activate(repo_root=None)     # Monkey-patches os.remove, os.unlink, os.rmdir, os.rename,
                             #   os.replace, shutil.rmtree, shutil.move,
                             #   shutil.copyfile, shutil.copy, shutil.copy2
deactivate()                 # Restore original stdlib functions
agent_mode()                 # Context manager (ContextVar tokens), sets agent mode for checks
is_active() -> bool
set_agent_mode(active: bool)
# Only blocks in agent mode (Daemon's own runtime is unguarded)
# Checks both source AND destination for rename/replace/move
# Copy functions check DESTINATION only (reading source is safe)
# Always-blocked targets cannot be unlocked
# scripts/bin/usercustomize.py auto-activates guard in child Python interpreters
#   (when scripts/bin/ on PYTHONPATH; skipped during pytest/coverage; disable with DISABLE_FS_GUARD=1)

# Shell scripts:
# scripts/agent_session_start.sh  — Pre-agent snapshot (git state + manifest + untracked tarball)
# scripts/agent_session_audit.sh  — Post-agent audit (manifest diff, git status comparison)
# scripts/safe_git.sh             — Safe git wrapper (blocks restore/reset --hard/clean/push)
# scripts/safe_cmd.sh             — Shell wrapper for destructive commands (mirrors safe_git.sh unlock pattern)
# scripts/activate_guards.sh      — Source to prepend scripts/bin/ to PATH; wraps rm, mv, rmdir,
#                                    chmod, truncate, find through safe_cmd.sh transparently
# scripts/bin/                    — PATH wrapper scripts (rm, mv, rmdir, chmod, truncate, find)
#                                    that route through safe_cmd.sh
```

---

## Helper Utilities

```python
# utils/topic_manager.py
class TopicManager:
    def extract_topics(query: str, use_llm_fallback: bool = False) -> List[str]:
        """
        1. Try spaCy NER (fast)
        2. If empty and fallback enabled, use LLM
        """
        doc = self.nlp(query)
        topics = [ent.text for ent in doc.ents]
        if not topics and use_llm_fallback:
            topics = await self._llm_extract_topics(query)
        return topics


# utils/time_manager.py
class TimeManager:
    def calculate_decay(age_hours: float, decay_rate: float = 0.05) -> float:
        """Temporal decay: 1.0 / (1.0 + decay_rate * age_hours)"""
        return 1.0 / (1.0 + decay_rate * age_hours)


# models/tokenizer_manager.py
class TokenizerManager:
    def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
        """Count tokens using tiktoken"""
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))

    def truncate_to_budget(text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget"""
        tokens = self.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.decode(tokens[:max_tokens])
```

---

## Consolidation & Facts

```python
# memory/memory_consolidator.py
class MemoryConsolidator:
    async def consolidate_memories(conversations: List[Dict], max_tokens: int = 300) -> str:
        """
        1. Format conversations into prompt
        2. Call LLM to summarize
        3. Return summary text
        """
        prompt = self._build_consolidation_prompt(conversations)
        summary = await self.model_manager.generate(prompt, model="gpt-4o-mini", max_tokens=max_tokens)
        return summary


# memory/fact_extractor.py [ENHANCED 2026-03: entity facts, 2026-05: ephemeral filtering]
class FactExtractor:
    def extract_facts(query, response, conversation_context) -> List[MemoryNode]:
        """
        Multi-stage extraction with dual-budget system:
        1. Correction detection (0.9+ confidence): "I'm X, not Y", "actually X"
        2. spaCy dependency parsing for SRO triples
        3. REBEL neural extraction (if enabled)
        4. Regex fallback patterns

        Dual budget [NEW 2026-03]:
        - User facts (subject="user"): capped at USER_FACTS_PER_TURN_CAP (6)
        - Entity facts (subject=entity name): capped at ENTITY_FACTS_PER_TURN_CAP (4)
        - Entity facts require ENTITY_FACT_MIN_CONFIDENCE (0.55)
        - Entity metadata: fact_scope, entity_type (spaCy NER), user_connection

        Ephemeral filtering [NEW 2026-05]:
        - Blocks ephemeral predicates (from PROFILE_EPHEMERAL_RELATIONS config)
        - Blocks boolean-only values ("true"/"false"/"yes"/"no")
        """

# memory/llm_fact_extractor.py [ENHANCED 2026-05: ephemeral filtering]
# _is_ephemeral_relation(rel) — checks against PROFILE_EPHEMERAL_RELATIONS config
# _is_boolean_noise(obj) — rejects facts where object is just true/false/yes/no
# _normalize_triple() blocks both ephemeral predicates and boolean noise before storage
# LLM prompt updated to discourage transient state extraction

# memory/memory_retriever.py [ENHANCED 2026-05: config-driven ephemeral filtering]
# _get_ephemeral_predicates() — replaces hardcoded _EPHEMERAL_PREDICATES with config-driven set
# TTL filter: drops stale ephemeral facts older than PROFILE_EPHEMERAL_TTL_HOURS (default 24h)

# Helper functions [NEW 2026-03]:
def _detect_entity_type(subject, nlp) -> str:
    """spaCy NER → PERSON, ORG, GPE, PRODUCT, WORK_OF_ART, or UNKNOWN"""

def _detect_user_connection(subject, source_text) -> Optional[str]:
    """Regex: 'my X', 'X is my', 'X, my' → "user's boss" etc."""
```

---

## Knowledge Graph [NEW 2026-03]

```python
# memory/graph_models.py — Pydantic data models
class GraphNode(BaseModel):
    entity_id: str; display_name: str; entity_type: str = ""; metadata: Dict = {}
class GraphEdge(BaseModel):
    source_id: str; target_id: str; relation: str; weight: float = 1.0
    def to_natural_language() -> str:  # "Luke has brother Dillion"

# memory/graph_memory.py — NetworkX DiGraph wrapper
class GraphMemory:
    add_entity(node: GraphNode)          # Upsert (merges metadata)
    add_edge(edge: GraphEdge)            # Add or strengthen weight
    get_entity(entity_id) -> GraphNode
    neighbors(entity_id, depth=1) -> Dict[str, List[GraphEdge]]  # BFS
    save() / load()                       # JSON at data/knowledge_graph.json

# memory/entity_resolver.py — Alias resolution
class EntityResolver:
    resolve(text) -> Optional[str]        # Alias → canonical entity_id
    resolve_or_create(text, display_name) -> str
    save() / load()                       # JSON at data/entity_aliases.json

# memory/graph_utils.py — Shared helpers for scoring + expansion
def extract_graph_entities(text, resolver) -> Set[str]:
    """Trigram→bigram→single word alias resolution, strips punctuation, skips stopwords."""
def get_related_display_names(entity_ids, graph_memory, depth=1, skip_ids=None) -> Set[str]:
    """BFS neighbor display names."""
def get_related_entity_ids(entity_ids, graph_memory, depth=1) -> Set[str]:
    """BFS neighbor entity IDs."""
def _is_expansion_junk(name) -> bool:  # [NEW 2026-03-15]
    """Rejects empty/≤2 chars, 4+ words, digit-starting, temporal, measurement, verb phrases."""
def rank_expansion_candidates(entity_ids, graph_memory, depth=2, skip_ids=None, max_terms=8) -> List[str]:  # [NEW 2026-03-15]
    """Ranks by non-hub edge count (lateral connectivity), filters junk, returns ordered names."""

# Ingestion (memory_storage.py:_ingest_fact_to_graph):
# Called after each add_fact(). _is_graph_worthy_object() rejects:
#   4+ word phrases, generics, temporal ("2 years"), measurements ("20lbs"),
#   verb phrases ("stopped being religious"). Rejected → stored as subject-node metadata.
# Duplicate edges strengthen weight (DiGraph, not MultiDiGraph).

# Retrieval (gatherer_knowledge.py / gatherer_memory.py):
# get_graph_context() → BFS from query entities, returns natural language sentences (gatherer_knowledge.py)
# _expand_query_with_graph() → calls rank_expansion_candidates() (connectivity-ranked, junk-filtered) (gatherer_memory.py)

# Scoring (memory_scorer.py):
# graph_bonus = min(0.05 * graph_entity_matches, GRAPH_SCORING_BOOST_CAP)
# Graph refs (_graph_memory, _entity_resolver) set/cleared by builder.py around gather

# Scripts:
# scripts/migrate_facts_to_graph.py --dry-run  # Populate graph from existing facts
# scripts/cleanup_graph_junk.py --execute       # Remove junk nodes, migrate to metadata
```

---

## Thread Surfacing System [NEW 2026-03-23]

```python
# memory/thread_models.py — Pydantic data models
class ThreadType(str, Enum):
    COMMITMENT = "commitment"   # "I'll study for the exam"
    DEADLINE = "deadline"       # "exam next Tuesday"
    UNFINISHED = "unfinished"   # Topic started but not completed
    QUESTION = "question"       # Question not fully answered

class ThreadStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    STALE = "stale"

class OpenThread(BaseModel):
    thread_id: str; topic: str; summary: str; status: ThreadStatus
    thread_type: ThreadType; urgency: float  # 0.0-1.0
    mentioned_at: float; last_referenced: float  # epoch
    resolution_hint: str; source_summary: str; deadline_date: Optional[str]

    def priority_score() -> float:
        """TYPE_PRIORITY[type] * urgency * recency_decay(14 days)"""
    def is_stale(stale_days=14, deadline_grace_hours=48) -> bool:
        """True if no reference in stale_days, or deadline_date passed + grace period."""
    def mark_resolved(resolution="") -> None
    def mark_stale() -> None
    def to_metadata() / from_metadata()   # ChromaDB flat primitives
    def to_dict() / from_dict()           # Full JSON serialization
    def to_embedding_text() -> str        # topic + summary for semantic matching

# TYPE_PRIORITY: deadline(1.0) > commitment(0.8) > question(0.6) > unfinished(0.4)


# memory/thread_store.py — ChromaDB-backed storage
class ThreadStore:
    def __init__(chroma_store):
        """Uses 'threads' collection. Delete-and-re-add for updates."""

    def store_thread(thread: OpenThread) -> Optional[str]:
        """Embed topic+summary, store in 'threads' collection. Returns doc_id."""

    def list_open_threads() -> List[OpenThread]:
        """Get all threads with OPEN status."""

    def get_top_threads(max_results=3, stale_days=14, deadline_grace_hours=48) -> List[OpenThread]:
        """Priority-ranked open threads for session surfacing.
        Lazy staleness: marks stale threads during retrieval."""

    def query_threads(query, n_results=5) -> List[OpenThread]:
        """Semantic search for threads matching a query."""

    def resolve_thread(thread_id, resolution="") -> bool:
        """Mark thread as resolved (delete-and-re-add pattern)."""

    def enforce_cap(max_open=50) -> int:
        """Prune lowest-priority threads when over cap. Returns count pruned."""


# memory/thread_extractor.py — LLM-based extraction [ENHANCED 2026-05-20]
class ThreadExtractor:
    def __init__(model_manager):
        """Two-phase LLM approach: extraction + resolution detection."""

    async def extract_new_threads(session_conversations) -> List[OpenThread]:
        """Extract open loops from session. Few-shot prompt, temp=0.0, cap 5 per session.
        Thread types: commitment, deadline, unfinished, question.
        Today's date injected into prompt for relative date resolution."""

    async def detect_resolutions(session_conversations, open_threads) -> List[Tuple[str, str]]:
        """Detect which existing threads were resolved. Returns (thread_id, resolution) tuples.
        Only marks resolved with clear evidence, NOT just because thread wasn't mentioned.
        Today's date injected for judging whether deadlines have passed."""


# Integration:
# - Shutdown: step 6.5 in process_shutdown_memory() — extract new threads + resolve existing
# - Prompt: [UNRESOLVED THREADS] section (after [KNOWLEDGE GRAPH])
# - Collection: 'threads' in ChromaDB (14 total collections)
# - Builder: top threads retrieved via ThreadStore.get_top_threads() in build_prompt()

# Config (app_config.py):
THREAD_SURFACING_ENABLED = True
THREAD_MAX_OPEN = 50
THREAD_STALE_DAYS = 14
THREAD_DEADLINE_GRACE_HOURS = 48
THREAD_MAX_SURFACED = 3
THREAD_MODEL_ALIAS = ""     # empty = default model
```

---

## Fact Verification Gate **[NEW 2026-03-24]**

```python
# memory/fact_verification.py
class FactVerdict(str, Enum):
    STORE = "store"              # No conflict
    STORE_AND_FLAG = "store_and_flag"  # Store new, mark old superseded
    REJECT = "reject"            # Likely extraction error
    SKIP = "skip"                # Ephemeral relation

class FactVerifier:
    def __init__(chroma_store, model_manager=None):
        """Pre-storage conflict gate. Checks new facts against existing."""

    async def verify_fact(fact_text, collection="facts") -> VerificationResult:
        """Flow: ephemeral check → candidate query → confirmation → user-trust override
        → entity-scope rejection → LLM adjudication → fallback trust-newer."""

# Integration:
# memory_storage.py: intercepts before add_fact()
# shutdown_processor.py: gates _extract_session_facts() and _extract_llm_facts()
# Old facts get superseded_by metadata — never deleted

# Config:
FACT_VERIFICATION_ENABLED = True
FACT_VERIFICATION_CANDIDATE_LIMIT = 5
```

---

## Proactive Context Surfacing **[NEW 2026-03-24]**

```python
# memory/context_surfacer.py + surfacing_models.py + surfacing_history.py
class ContextSurfacer:
    def __init__(graph_memory, entity_resolver, model_manager):
        """Cross-domain insight generation from knowledge graph."""

    async def generate_insights(query, max_insights=3) -> List[ProactiveInsight]:
        """Star topology: classify entities by domain → find cross-domain bridges
        → novelty filter (72h cooldown) → single LLM call → session cache.
        Skips if graph < 20 nodes or < 15 edges."""

# Models (surfacing_models.py):
# DomainEntity, DomainCluster, CrossDomainCandidate, ProactiveInsight

# Integration:
# gatherer_knowledge.py → get_proactive_insights() → context_surfacer.generate_insights()
# Prompt: [PROACTIVE INSIGHTS] after [UNRESOLVED THREADS]
# Parallel task in build_prompt()

# Config:
PROACTIVE_SURFACING_ENABLED = True
PROACTIVE_SURFACING_COOLDOWN_HOURS = 72
PROACTIVE_SURFACING_MAX_INSIGHTS = 3
```

---

## Uncertainty Detector **[NEW 2026-04]**

```python
# core/uncertainty_detector.py
class UncertaintyDetector:
    def detect(response: str) -> UncertaintyResult:
        """Dual-layer detection of 'I don't know' responses:
        Layer 1: Keyword regex (~18 compiled patterns, confidence 0.75-0.90)
        Layer 2: Semantic embedding (8 pre-embedded anchor sentences, cosine threshold 0.70)
        Length guard: strips hedge prefixes, skips if substantive content > 400 chars.
        Returns: UncertaintyResult(is_uncertain, confidence, method, matched_pattern)"""

# Integration:
# gui/handlers.py: after standard streaming, if flagged, retries via agentic search
# Single boolean prevents infinite retry; only fires on non-agentic responses
# Provenance: response_mode="uncertainty-fallback"

# Config:
UNCERTAINTY_FALLBACK_ENABLED = True
UNCERTAINTY_SEMANTIC_THRESHOLD = 0.70
UNCERTAINTY_MAX_LENGTH = 400
```

---

## Response Planner **[NEW 2026-04]**

```python
# core/response_planner.py
class ResponsePlan(BaseModel):
    key_points: List[str]; tone: str; avoid: List[str]; strategy: str; raw_llm_output: str

class ReviewResult(BaseModel):
    passes: bool; confidence: float; issues: List[str]; suggestion: str

class ResponsePlanner:
    def should_plan(context) -> bool:
        """Skips for: small-talk, crisis/elevated tone, CASUAL_SOCIAL intent, <8 word queries, disabled config."""

    async def create_plan(query, context_signals) -> Optional[ResponsePlan]:
        """Lightweight LLM call (~200 tokens, 5s timeout) → ResponsePlan."""

    async def review_answer(query, answer, plan) -> ReviewResult:
        """Post-answer review (~300 tokens). If passes=False and confidence >= 0.90, triggers silent agentic retry."""

# Integration:
# orchestrator.py: plan runs in parallel with build_prompt_from_context() via asyncio.wait()
# Plan injected into system prompt before _assemble_prompt()
# gui/handlers.py: review gate after uncertainty fallback, similarity-guarded retry

# Config:
RESPONSE_PLANNING_ENABLED = True
RESPONSE_PLANNING_MODEL = None         # None → active model (config.yaml overrides to gpt-4o-mini)
RESPONSE_PLANNING_MAX_TOKENS = 200
RESPONSE_PLANNING_TIMEOUT = 5.0
RESPONSE_REVIEW_ENABLED = True
RESPONSE_REVIEW_CONFIDENCE_THRESHOLD = 0.90
```

---

## Memory Staleness System **[NEW 2026-03-25]**

```python
# memory/claim_tracker.py
class ClaimKey(BaseModel):
    subject: str; relation: str; claim_hash: str  # MD5[:12]

class ClaimIndex:
    """Reverse index: claim_hash → [doc_ids], doc_id → [claim_hashes]"""

    def add_claims(doc_id, collection, claims): ...
    def cascade_staleness(claim_key, chroma_store=None) -> List[str]:
        """Find affected docs, recompute staleness_ratio, update ChromaDB metadata."""
    def save() / load(): ...  # JSON at data/claim_index.json

def canonicalize_claim(subject, relation, entity_resolver=None) -> ClaimKey
def extract_claims_from_text(text, entity_resolver=None) -> List[ClaimKey]

# Scoring (memory_scorer.py step 12):
# penalty = ratio * 0.15; if ratio >= 0.8: *= 2.0; reflections *= 0.6; cap 0.4

# Prompt prefix (builder.py):
# staleness_ratio >= 0.6 → "[HISTORICAL — PARTIALLY OUTDATED] ..."

# Cascade triggers:
# 1. Orchestrator: after correction detection
# 2. Cross-deduplicator: after contradiction clusters
# 3. Shutdown: claims extracted at summary creation time

# Config:
STALENESS_ENABLED = True
STALENESS_WEIGHT = 0.15
STALENESS_MAX_PENALTY = 0.4

# Migration: python scripts/migrate_claims.py [--dry-run]
```

---

## Implementation Tracking **[NEW 2026-03-24]**

```python
# knowledge/implementation_detector.py
class DetectionResult(BaseModel):
    proposal_id: str; confidence: float; status: str; evidence: List[str]; stage_reached: int

class ImplementationDetector:
    async def detect(proposal, lightweight=False) -> DetectionResult:
        """4-stage: file existence → code grep → git history → LLM judgment (borderline only).
        Lightweight = stage 1 only (~50ms). Cooldown prevents re-checking (86400s)."""

    async def detect_batch(proposals, lightweight=False) -> List[DetectionResult]:
        """Batch with cooldown respect."""

# Confidence: confirmed (≥0.85), likely (≥0.60), uncertain (≥0.30), not_implemented (<0.30)

# Integration:
# Shutdown: Phase 6.5 (lightweight file-existence check)
# CLI: python main.py check-proposals [--id UUID] [--verbose]
# GUI: "Check Implementation" button (batch), "Check This" (single), badges

# Config:
IMPL_TRACKING_ENABLED = True
IMPL_TRACKING_COOLDOWN = 86400
```

---

## Session-Start Codebase Diff + Feature Inventory **[NEW 2026-03-24]**

```python
# gatherer_knowledge.py:get_codebase_changes() — git log/diff/status since last session
# formatter.py:_build_feature_inventory() — compact 4-line enabled features summary

# Prompt sections:
# [CODEBASE CHANGES SINCE LAST SESSION] — first message only
# [ACTIVE FEATURES] — always present
# Placed after [USER PROFILE] before [TIME CONTEXT]

# Orchestrator: ## CODEBASE CHANGE AWARENESS system prompt (first message only)
# Codebase changes gathered BEFORE small-talk fork (even "Yo" gets change awareness)

# Config:
SESSION_DIFF_ENABLED = True
SESSION_DIFF_MAX_COMMITS = 20
SESSION_DIFF_MAX_FILES = 30
```

---

## Configuration Constants

```python
# config/app_config.py

# Paths
CORPUS_FILE = os.getenv("CORPUS_FILE", "./data/corpus.json")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

# Token budgets (model-aware — see config.yaml token_budget: section)
# Auto-computed: min(context_window * 0.25, ceiling) clamped to [floor, ceiling]
# Override: PROMPT_TOKEN_BUDGET=15000 env var forces a specific value
PROMPT_TOKEN_BUDGET_DEFAULT = 15000   # API models fallback
PROMPT_TOKEN_BUDGET_LOCAL = 12000     # Local model cap
PROMPT_TOKEN_BUDGET_FLOOR = 8000      # Minimum budget
PROMPT_TOKEN_BUDGET_CEILING = 16000   # Maximum budget
# PROMPT_MIN_RECENT_FLOOR = 5  — defined in core/prompt/builder.py (not app_config.py)
PROMPT_MAX_MEMS = int(os.getenv("PROMPT_MAX_MEMS", "15"))

# Decay & scoring
RECENCY_DECAY_RATE = float(os.getenv("RECENCY_DECAY_RATE", "0.05"))
TRUTH_SCORE_UPDATE_RATE = float(os.getenv("TRUTH_SCORE_UPDATE_RATE", "0.02"))
TRUTH_SCORE_MAX = 1.0

# Gating threshold
GATE_COSINE_THRESHOLD = float(os.getenv("GATE_COSINE_THRESHOLD", "0.45"))
# NOTE: context_gatherer.py defaults to 0.45, but gate_system.py defaults to 0.50.
# The env var GATE_COSINE_THRESHOLD overrides both when set.

# Collection boosts
COLLECTION_BOOSTS = {
    "episodic": 0.1,
    "semantic": 0.05,
    "procedural": 0.08,
    "summary": 0.03,
    "meta": 0.02
}

# Score weights (sum to 1.0) — can be overridden per-intent via IntentClassifier
SCORE_WEIGHTS = {
    "relevance": 0.35,
    "recency": 0.25,
    "truth": 0.20,
    "importance": 0.05,
    "continuity": 0.10
}

# Intent Classifier [NEW 2026-02-15]
INTENT_ENABLED = True
INTENT_STM_REFINEMENT_THRESHOLD = 0.50

# Entity Facts [NEW 2026-03]
ENTITY_FACTS_ENABLED = True              # Toggle entity fact extraction
ENTITY_FACTS_PER_TURN_CAP = 4           # Max entity facts per turn
USER_FACTS_PER_TURN_CAP = 6             # Max user facts per turn
ENTITY_FACT_MIN_CONFIDENCE = 0.55        # Min confidence for entity facts

# Ephemeral Fact Filtering [NEW 2026-05]
PROFILE_EPHEMERAL_RELATIONS = [...]      # Config-driven list (mood, feelings, location, etc.)
PROFILE_EPHEMERAL_TTL_HOURS = 24         # Stale ephemeral facts excluded from retrieval
PROFILE_EPHEMERAL_MAX_HISTORY = 20       # Max historical entries per ephemeral relation

# Schedule Extraction [NEW 2026-05]
SCHEDULE_EXTRACTION_ENABLED = True       # Toggle schedule pattern detection in fact extraction
SCHEDULE_PROMPT_MAX_EVENTS = 5           # Max events in [UPCOMING SCHEDULE] section
SCHEDULE_PROMPT_LOOKAHEAD_DAYS = 7       # Days ahead to show in upcoming schedule
# 5 patterns: work_schedule, class_schedule, exam_date, shift_pattern, day_off
# Stored as facts with fact_type="schedule" metadata
# Time normalization: utils/temporal_resolver.py — normalize_time_range(), expand_day_abbreviations(), resolve_date_expression()
# Supersession: same kind + same day → old fact marked superseded via FactVerifier._check_schedule_supersession()
# Tests: 64 in tests/unit/test_schedule_extraction.py

# Knowledge Graph [NEW 2026-03]
KNOWLEDGE_GRAPH_ENABLED = True           # Toggle knowledge graph system
KNOWLEDGE_GRAPH_MAX_DEPTH = 2            # BFS traversal depth
KNOWLEDGE_GRAPH_MAX_EDGES = 50           # Max edges in prompt
GRAPH_SCORING_BOOST_ENABLED = True       # Graph-boosted memory scoring
GRAPH_SCORING_BOOST_CAP = 0.15          # Max graph bonus (0.05 per entity)
GRAPH_QUERY_EXPANSION_ENABLED = True     # Append neighbors to search query
GRAPH_QUERY_EXPANSION_MAX_TERMS = 8      # Max neighbor names appended

# Personal Notes Gate [NEW 2026-03-20]
PERSONAL_NOTES_GATE_THRESHOLD = 0.30     # Post-gate threshold for Obsidian notes (general gate is 0.18)

# Thread Surfacing [NEW 2026-03-23]
THREAD_SURFACING_ENABLED = True          # Toggle thread extraction/surfacing
THREAD_MAX_OPEN = 50                     # Max open threads before pruning
THREAD_STALE_DAYS = 14                   # Days without reference before stale
THREAD_DEADLINE_GRACE_HOURS = 48         # Hours past deadline before stale
THREAD_MAX_SURFACED = 3                  # Max threads in prompt
THREAD_MODEL_ALIAS = ""                  # LLM alias for extraction (empty = default)

# Fact Verification [NEW 2026-03-24]
FACT_VERIFICATION_ENABLED = True
FACT_VERIFICATION_LLM_MODEL = ""        # empty = default model
FACT_VERIFICATION_CANDIDATE_LIMIT = 5

# Proactive Context Surfacing [NEW 2026-03-24] [OPT 2026-05-22]
PROACTIVE_SURFACING_ENABLED = True
PROACTIVE_SURFACING_COOLDOWN_HOURS = 72
PROACTIVE_SURFACING_MAX_INSIGHTS = 3
# Non-blocking: first message fires background warmup, insights appear from message 2+

# Memory Staleness [NEW 2026-03-25]
STALENESS_ENABLED = True
STALENESS_WEIGHT = 0.15
STALENESS_MAX_PENALTY = 0.4
STALENESS_STEEP_THRESHOLD = 0.8
STALENESS_HISTORICAL_THRESHOLD = 0.6
STALENESS_INDEX_PATH = "data/claim_index.json"

# Implementation Tracking [NEW 2026-03-24]
IMPL_TRACKING_ENABLED = True
IMPL_TRACKING_COOLDOWN = 86400          # seconds between re-checks

# Memory Expansion [NEW 2026-03-26]
EXPAND_MEMORY_ENABLED = True            # Toggle expand_memory agentic tool
EXPAND_MAX_PER_SESSION = 3             # Max expansions per agentic session
EXPAND_MAX_WINDOW = 5                  # Max temporal window (±N turns)
EXPAND_DEFAULT_WINDOW = 3              # Default temporal window
EXPAND_ANCHOR_CHAR_LIMIT = 600        # Char limit for anchor document
EXPAND_CONTEXT_CHAR_LIMIT = 300       # Char limit for context documents
EXPAND_ANCHOR_CHAR_LIMIT_LONG = 3000  # Long-form collections (obsidian_notes, reference_docs)
EXPAND_CONTEXT_CHAR_LIMIT_LONG = 2000 # Long-form context limit

# Visual Memory [NEW 2026-05]
VISUAL_MEMORY_ENABLED = False            # Toggle visual memory pipeline
VISUAL_MEMORY_CLIP_MODEL = "ViT-B-32"   # OpenCLIP model for image/text embedding (512-dim)
VISUAL_MEMORY_MAX_IMAGES = 3            # Max images returned per query
VISUAL_MEMORY_CAPTION_MODEL = None      # Vision LLM for captions (None = default model)
VISUAL_MEMORY_FAISS_PATH = "data/visual_faiss.index"  # CLIP vector index path
VISUAL_MEMORY_DEDUP_ENABLED = True      # SHA-256 content dedup on ingestion
# Retrieval: entity-gated (only fires when query mentions an entity with stored images)
# Backfill: scripts/backfill_visual_entities.py (profile-aware re-captioning + entity tagging)

# Skill Activation [NEW 2026-05]
SKILL_ACTIVATION_ENABLED = True          # Toggle post-retrieval skill filtering
SKILL_ACTIVATION_MAX_SKILLS = 3          # Max skills surfaced per turn
SKILL_ACTIVATION_MIN_SCORE = 0.25        # Minimum relevance score threshold
SKILL_ACTIVATION_COOLDOWN_HOURS = 48.0   # Hours before re-surfacing same skill
SKILL_ACTIVATION_FETCH_MULTIPLIER = 3    # Over-fetch factor (fetch N*max, filter to max)
SKILL_ACTIVATION_STM_BONUS = 0.10        # Relevance bonus for STM topic match
SKILL_ACTIVATION_USE_STM = True          # Enable STM topic bonus
# Suppressed intents: EMOTIONAL_SUPPORT, CASUAL_SOCIAL → no skills surfaced
# Cooldown store: data/skill_cooldown.json (JSON-backed TTL tracking)

# Config Schema Validation [NEW 2026-05-10]
# config/schema.py — Pydantic v2 validation for config.yaml (44 section models)
# Validates at startup after YAML load, before constant extraction
# validate_config(config) → returns dict unchanged or sys.exit(1) with errors

# LLM Compression [NEW 2026-03-26] [OPT 2026-05-22]
LLM_COMPRESSION_ENABLED = True         # Use LLM to compress heavily oversized items
LLM_COMPRESSION_MODEL = "gpt-4o-mini"  # Model for compression calls
LLM_COMPRESSION_TIMEOUT = 3.0          # Per-item timeout (seconds)
LLM_COMPRESSION_RATIO_THRESHOLD = 3.0  # Only compress items >= 3x over token limit
LLM_COMPRESSION_MAX_BATCH = 8          # Max items to compress per request (cost guard)
# Items > 6x target tokens skip LLM compression entirely (middle-out handles them instantly)

# Citation & Provenance [NEW 2026-03-26]
PROVENANCE_ENABLED = True              # Toggle provenance metadata on interactions
PROVENANCE_THINKING_MAX_CHARS = 4000   # Max chars for thinking block in metadata

# Uncertainty Fallback [NEW 2026-04]
UNCERTAINTY_FALLBACK_ENABLED = True    # Retry uncertain responses via agentic search
UNCERTAINTY_SEMANTIC_THRESHOLD = 0.70  # Cosine sim threshold for semantic anchor match
UNCERTAINTY_MAX_LENGTH = 400           # Skip detection for responses longer than this

# Response Planning [NEW 2026-04]
RESPONSE_PLANNING_ENABLED = True       # Pre-answer plan from query + context signals
RESPONSE_PLANNING_MODEL = None         # None → active model
RESPONSE_PLANNING_MAX_TOKENS = 200     # Plan LLM call budget
RESPONSE_PLANNING_TIMEOUT = 5.0        # Seconds before plan skipped
RESPONSE_REVIEW_ENABLED = True         # Post-answer review against plan
RESPONSE_REVIEW_MODEL = None           # None → active model
RESPONSE_REVIEW_MAX_TOKENS = 200       # Review LLM call budget
RESPONSE_REVIEW_CONFIDENCE_THRESHOLD = 0.80  # Min confidence to trigger agentic retry
RESPONSE_REVIEW_TIMEOUT = 5.0         # Seconds before review skipped

# Personality / Operating Principles (file-based) [NEW 2026-03-26]
PERSONALITY_DEFAULT_PATH = "config/prompts/default_personality.txt"
PERSONALITY_CUSTOM_PATH = "config/prompts/custom_personality.txt"
OPERATING_PRINCIPLES_PATH = "config/prompts/operating_principles.txt"
PERSONALITY_MAX_CHARS = 15000          # Hard cap on personality text

# Summarization
SUMMARY_EVERY_N = int(os.getenv("SUMMARY_EVERY_N", "20"))
SUMMARIZE_AT_SHUTDOWN_ONLY = True

# Models
MODEL_DEFAULT = os.getenv("LLM_ALIAS", "gpt-4o-mini")
MODEL_SUMMARY = os.getenv("LLM_SUMMARY_ALIAS", "gpt-4o-mini")
# Available: gpt-4o-mini, gpt-4o, gpt-5, sonnet-4.5, claude-opus-4.6, deepseek-v3.1,
#            deepseek-r1, glm-4.6, glm-4.7, glm-5, glm-5-turbo [glm-5* NEW 2026-03-23]
```

---

## Common Patterns (Copy-Paste Templates)

### Adding a new memory type:
```python
# 1. corpus_manager.py - add enum
class MemoryType(Enum):
    NEW_TYPE = "new_type"

# 2. multi_collection_chroma_store.py - initialize collection
COLLECTIONS = ["episodic", "semantic", "procedural", "procedural_skills", "summary", "meta", "new_type"]

# 3. memory/memory_retriever.py - handle in get_memories() pipeline
new_type_results = await self.chroma_store.query(query, "new_type", limit)

# 4. memory/memory_coordinator.py - add delegation wrapper (if new public method needed)
async def get_new_type(self, query, limit=10):
    return await self._retriever.get_new_type(query, limit)

# 5. prompt/formatter.py - add to prompt assembly
new_type_mems = [m for m in memories if m.get('memory_type') == 'new_type']
sections.append(self._format_memories(new_type_mems, budget * 0.05))
```

### Testing a method:
```python
# tests/unit/test_*.py
def test_method_name(coordinator):
    # Setup
    input_data = {...}
    expected = {...}

    # Execute
    result = coordinator.method(input_data)

    # Assert
    assert result == expected
```

### Adding logging:
```python
from utils.logging_utils import get_logger
logger = get_logger(__name__)

logger.debug(f"Details: {variable}")
logger.info("Important milestone")
logger.warning("Unusual condition")
logger.error(f"Error occurred: {e}")
```

---

## Critical Formulas (Quick Lookup)

```python
# Recency decay (standard)
recency = 1.0 / (1.0 + 0.05 * age_hours)

# Two-regime temporal decay [UPDATED 2026-05]
# Small anchor (<=48h, "today"/"yesterday") — flat plateau:
recency = 1.0 - (age_hours / temporal_anchor) * 0.15  # within window
recency = 0.85 / (1.0 + 0.05 * (age_hours - temporal_anchor))  # outside
# Large anchor (>48h, "last week"/"last month") — peak at anchor:
floor = max(0.45, 1.0 - temporal_anchor / 300.0)
recency = floor + (1.0 - floor) * (age_hours / temporal_anchor)  # within window
recency = 1.0 / (1.0 + 0.05 * (age_hours - temporal_anchor))    # outside

# Truth (evidence-based via TruthScorer) [UPDATED 2026-03]
# Initial: user_stated=0.85, corrected=0.90, llm_extracted=0.70, inferred=0.65
# Confirmation: +0.08, Correction: -0.25, Contradiction: -0.15
# Time decay: truth -= (weeks_since_confirmed * 0.02), floor 0.30
truth = TruthScorer.compute_effective_truth(metadata)

# Continuity score (tokens stemmed via _stem() for overlap matching)
continuity = (0.1 if age < 10min else 0) + (0.3 * token_overlap_ratio)
# + tag-keyword bonus: 0.15 * min(tag_hits, 3) / 3.0 when query stems match memory tags

# Structural alignment
alignment = 1.0 - min(1.0, abs(query_density - memory_density) * 3.0)
structure_score = 0.15 * alignment

# Graph-boosted scoring [NEW 2026-03]
graph_bonus = min(0.05 * graph_entity_matches, 0.15)  # capped at GRAPH_SCORING_BOOST_CAP

# Staleness penalty (summaries/reflections only) [NEW 2026-03-25]
staleness_penalty = staleness_ratio * 0.15          # base
if staleness_ratio >= 0.8: staleness_penalty *= 2.0  # steep curve
if collection == 'reflections': staleness_penalty *= 0.6
staleness_penalty = min(staleness_penalty, 0.4)     # capped

# Final memory score
score = (
    0.35 * relevance +
    0.25 * recency +
    0.20 * truth +
    0.05 * importance +
    0.10 * continuity +
    structure_score +
    anchor_bonus +
    graph_bonus +
    penalties
)
```

---

## Knowledge Sources (Prompt Sections)

```python
# Prompt section ordering (formatter.py _assemble_prompt, attention-optimized):
[RECENT CONVERSATION]          # 1. Session continuity (1-15 items)
[RELEVANT MEMORIES]            # 2. Semantic hits (1-15 items)
[RECENT SUMMARIES]             # 3. Compressed recent history (1-5)
[SEMANTIC SUMMARIES]           # 4. Query-relevant compressed history (1-5)
[RECENT REFLECTIONS]           # 5. Meta insights, recent (1-5)
[SEMANTIC REFLECTIONS]         # 6. Meta insights, query-relevant (1-5)
[BACKGROUND KNOWLEDGE]         # 7. Wiki snippets (1-3)
[WEB SEARCH RESULTS]           # 8. Real-time web with [WEB_N] source IDs + citation instruction
[RELEVANT INFORMATION]         # 9. Semantic chunks (1-8)
[DREAMS]                       # 10. Synthesis insights (if enabled; all generators currently disabled)
[USER'S PERSONAL NOTES]        # 11. Obsidian vault notes; post-filtered by PERSONAL_NOTES_GATE_THRESHOLD (0.30)
[USER UPLOADED ITEMS]          # 12. Persisted user file/image uploads from reference_docs collection
[VISUAL MEMORIES]              # 12b. CLIP-retrieved image memories with captions (entity-gated via extract_graph_entities + intent-proximity disambiguation)
[DAEMON DOCUMENTATION]         # 13. Self-knowledge: architecture docs, PROJECT_SKELETON
[PROJECT COMMIT HISTORY]       # 14. Git commit history (procedural memory)
[ADAPTIVE WORKFLOWS]           # 15. Reusable problem-solving patterns (WHEN/THEN)
[PROPOSED FEATURES]            # 16. Code proposals surfaced for project-related queries
[KNOWLEDGE GRAPH]              # 17. Graph traversal: related entities as natural language
[UNRESOLVED THREADS]           # 18. Open commitments, deadlines, unfinished topics
[UPCOMING SCHEDULE]            # 19. Extracted schedule facts (work, class, exams) [NEW 2026-05]
[GOOGLE CALENDAR]              # 20. Real-time Google Calendar events via OAuth2 [NEW 2026-05]
[PROACTIVE INSIGHTS]           # 21. Cross-domain insights from knowledge graph
[USER PROFILE]                 # 22. Categorized facts + anti-confabulation instruction + source excerpts (high-attention zone)
[ACTIVE FEATURES]              # 23. Feature inventory (always present)
[CODEBASE CHANGES SINCE LAST SESSION]  # 24. Git changes (first message only)
[TIME CONTEXT]                 # 25. Current datetime (high-attention zone)
[TEMPORAL GROUNDING]           # 26. Synthesized life context (monthly/weekly/daily notes)
[SHORT-TERM CONTEXT SUMMARY]   # 27. STM analysis + reference_type + temporal_facts + open_threads + constraints
[CURRENT USER QUERY]           # 28. Always last, protected from compression
```

---

## CLI Commands (main.py)

```bash
# Daily Notes - auto-generated conversation summaries
python main.py daily-note                    # Generate for today
python main.py daily-note yesterday          # Generate for yesterday
python main.py daily-note 2026-01-15         # Specific date (YYYY-MM-DD)
python main.py daily-note --force            # Overwrite existing
python main.py daily-note-catchup            # Startup hook (yesterday if missing)

# Weekly Notes - auto-generated weekly summaries [NEW 2026-01-19]
python main.py weekly-note                   # Generate for current week
python main.py weekly-note 2026-01-12        # Generate for week containing date
python main.py weekly-note --force           # Overwrite existing
python main.py weekly-note-catchup           # Generate last week if missing

# Monthly Notes - auto-generated monthly summaries [NEW 2026-03-10]
python main.py monthly-note                  # Generate for last month
python main.py monthly-note 2026-02          # Generate for specific month (YYYY-MM)
python main.py monthly-note last-month       # Explicit last month
python main.py monthly-note --force          # Overwrite existing
python main.py monthly-note-catchup          # Migrate weekly folders + generate last month
python main.py migrate-monthly               # Just run weekly-to-monthly folder migration

# Narrative Context (Temporal Grounding) [ENHANCED 2026-03-10 - now 3-tier: monthly/weekly/daily]
python main.py refresh-narrative             # Regenerate life state from monthly/weekly/daily notes

# Daemon Documentation (self-knowledge)
# Auto-seeded from docs/ on GUI startup (mtime-based idempotency, skips unchanged files)
# Config: REFERENCE_DOCS_AUTO_SEED=True, REFERENCE_DOCS_SEED_PATHS=["docs"]
python main.py upload-doc <file> [title]     # Upload doc to [DAEMON DOCUMENTATION]
python main.py list-docs                     # List uploaded docs
python main.py delete-doc <title>            # Delete specific doc
python main.py clear-docs                    # Clear all docs

# Obsidian Vault (personal notes)
python main.py embed-vault                   # Index vault to ChromaDB
python main.py embed-vault --force           # Force full re-index
python main.py vault-stats                   # Show indexed chunk count
python main.py clear-vault                   # Clear obsidian_notes collection

# Obsidian Relevance Filtering [NEW 2026-03-20]
# Post-gate threshold: PERSONAL_NOTES_GATE_THRESHOLD=0.30 (YAML: obsidian.gate_threshold)
# Notes below this relevance_score are dropped after multi-stage gating (general gate is 0.18)
# Each note header in prompt includes [relevance: X.XX] tag for LLM visibility

# Obsidian Image Support (multimodal models) [NEW 2026-01-30]
# Images (![[image.png]]) in notes are automatically loaded for multimodal models
# Config: OBSIDIAN_INCLUDE_IMAGES=True, OBSIDIAN_MAX_IMAGES_PER_NOTE=3
# Supported: PNG, JPG, GIF, WebP (max 10MB per image, 10MB total per note)
# Image resolution: same folder → parent → attachments → vault root → global search

# Git Memory (procedural knowledge from commit history) [NEW 2026-01-27]
python main.py git-backfill [LIMIT]          # Initial load (default: 200 commits)
python main.py git-update                    # Incremental sync since last backfill
python main.py git-status                    # Show PROCEDURAL collection stats
python main.py git-clear                     # Wipe collection and reset sync state

# Visual Memory [NEW 2026-05]
python scripts/backfill_visual_memory.py     # Re-index existing images into visual_memories + FAISS
python scripts/backfill_visual_memory.py --dry-run  # Preview without writing

# User Profile
python main.py export-profile                # Export to data/user_profile_export.md
python main.py show-profile                  # Print profile to console
```

---

## Daily Notes Generator [ENHANCED 2026-01-22]

```python
# utils/daily_notes_generator.py
class DailyNotesGenerator:
    async def generate_for_date(date: date, force: bool = False) -> GenerationResult:
        """
        1. Get conversations for date from corpus
        2. Skip if no conversations or note exists (unless force)
           2b. Auto-update: if note exists but conversation count grew by
               >= DAILY_NOTES_UPDATE_MIN_NEW (default 3), regenerate the note
        3. Format conversations for LLM
        4. Calculate active duration (estimated actual usage time)
        5. Calculate intensity (1-10 based on count/active_hours/complexity)
        6. Call LLM with structured prompt (includes Life Events section)
        7. Generate contextual tags using TagGenerator [NEW 2026-01-22]
        8. Build YAML frontmatter + markdown content (with tags)
        9. Atomic write to Obsidian vault
        """

    async def generate_yesterday_if_missing() -> Optional[GenerationResult]:
        """Startup catch-up hook (called by GUI on launch)"""

    # Auto-update helpers:
    def _should_auto_update(target_date: date, current_count: int) -> bool
    def _get_existing_conversation_count(target_date: date) -> int
    def _find_existing_note_path(target_date: date) -> Path | None
    # Config: DAILY_NOTES_UPDATE_MIN_NEW (default 3)

    def _calculate_active_duration(convos: List[Dict]) -> float:
        """
        Estimate actual usage time (not wall-clock span):
        - Reading time: ~200 words/min for responses
        - Typing time: ~40 words/min for queries
        - Gap time: Capped at 30 seconds (idle time excluded)
        Example: 8-hour span might = 1.5 hours active
        """

    def _format_filename(date: date) -> str:
        """Format: 'M D YY Daily Note.md' (e.g., '1 16 26 Daily Note.md')"""

# Note structure:
# - Summary (2-3 sentences from Daemon's perspective)
# - Main Quest: [Primary Focus] (RPG-style framing)
# - Side Quests (other topics)
# - Life Events [NEW]: Work, Study, Sleep, Exercise, Other
#   - "Not discussed today" ≠ "didn't happen"
# - Emotional State (mood tracking)
# - Key Decisions, Knowledge Gained, Open Threads
# - Intensity: X/10

# Frontmatter fields (updated):
# - usage_intensity (was: intensity)
# - span_hours (wall-clock), active_hours (estimated usage)
# - tags: [daily, daemon-generated, coding, productive, ...] [NEW 2026-01-22]

# Scheduling:
# - Cron: 0 2 * * * cd /path/to/daemon && python main.py daily-note yesterday
# - GUI startup: _run_daily_notes_catchup() in launch.py (background thread)
```

---

## Tag Generator [NEW 2026-01-22]

```python
# utils/tag_generator.py
class TagGenerator:
    async def generate_tags(content: str, note_type: str, metadata: dict) -> TagGenerationResult:
        """
        LLM-based tag extraction for Obsidian notes.

        1. Truncate content (max 2000 chars: first 1500 + last 500)
        2. Build prompt with: content, metadata, tag vocabulary
        3. Call LLM (gpt-4o-mini, temp=0.3) for tag extraction
        4. Parse response (handles comma/newline/numbered formats)
        5. Normalize tags (lowercase, hyphenate, remove special chars)
        6. Validate tags (min 2 chars, filter stopwords)
        7. Categorize as known vs custom
        8. Enforce min/max limits (3-10 tags)
        9. Fallback to heuristics if LLM fails
        """

# Tag Vocabulary (100+ tags across 7 categories):
# 1. Life domains (17): work, study, health, exercise, sleep, social, family, ...
# 2. Activities (18): coding, programming, learning, reading, writing, debugging, ...
# 3. Emotions (23): stress, anxiety, happy, motivated, tired, focused, overwhelmed, ...
# 4. Productivity (15): productive, deep-work, flow-state, breakthrough, blocked, ...
# 5. Topics (23): ai, programming, python, math, philosophy, psychology, ...
# 6. Meta (16): crisis, decision, reflection, planning, insight, follow-up, ...
# 7. Events (16): appointment, meeting, interview, presentation, deadline, celebration, ...

# Tag format (Obsidian-compatible):
# - Lowercase with hyphens (e.g., deep-work, mental-health)
# - Quoted in YAML: tags: ["daily", "daemon-generated", "coding", "productive"]

# Integration:
# - DailyNotesGenerator: Analyzes daily note content + metadata
# - WeeklyNotesGenerator: Analyzes weekly summary content
# - Future: .md-based memories with tags as 4th filtering stage
```

---

## Narrative Context (Temporal Grounding) [ENHANCED 2026-03-10]

```python
# memory/memory_consolidator.py
class MemoryConsolidator:
    async def generate_narrative_context(weeklies=None, monthlies=None) -> str:
        """
        Synthesize monthly/weekly/daily notes into 'Current Life State' narrative. [3-tier hierarchy]

        Sources (3-tier hierarchy):
        1. Obsidian monthly summaries (<Month YYYY> Summary.md) - 1 max
        2. Obsidian weekly summaries (both monthly-parent and root-level) - 3 max
        3. Obsidian daily notes (both monthly-parent and root-level) - 6 max
        4. Corpus summaries (fallback)

        Output sections:
        - Current Life State (life phase description)
        - Active Threads (ongoing projects/concerns)
        - Emotional Trajectory (mood trend)
        - Recurring Themes (patterns)

        Synthesis prompt includes TEMPORAL ACCURACY RULES:
        - Date attribution derived from daily note frontmatter
        - Multi-day events expressed as date ranges
        - Day-of-week names anchored to actual dates

        Cached to: ./data/narrative_context.txt (0ms retrieval)
        Token budget: Priority 8, 500 token cap
        """

# Update Triggers:
# 1. PRIMARY: Daily note creation → DailyNotesGenerator._trigger_narrative_refresh()
# 2. SECONDARY: After summary consolidation (memory_storage.py)
# 3. FALLBACK: Startup check warns if >24 hours stale
# 4. MANUAL: python main.py refresh-narrative

# Prompt placement:
# [TIME CONTEXT] → [TEMPORAL GROUNDING] → [STM SUMMARY] → [CURRENT USER QUERY]
```

---

## Agentic Search & Tools [NEW 2026-01-22, ENHANCED 2026-03-31]

**Agentic Gate** (gui/handlers.py) — 5-tier trigger before ReAct loop:
1. **Keyword heuristic** (0ms): computation keywords OR memory keywords ("do you remember", "my notes", etc.) OR web search keywords ("web search", "search for", "search online", "look up") OR tool name keywords ("github", "git stats", "wolfram", "sandbox", "pull request", "open issues", etc. → `needs_tools`) [ENHANCED 2026-05-20]. Explicit search keywords bypass intent veto.
1b. **URL detection** (0ms) [NEW 2026-05]: `http://`/`https://` in query triggers agentic mode with `initial_urls` for auto-fetch
1c. **Knowledge keywords** (0ms) [NEW 2026-03-31]: encyclopedic/wiki intent (`explain in depth`, `how does`, `consult wikipedia`, etc.) — fires for 4+ word queries when no computation/memory trigger matched
2. **Entity match** (0ms): `extract_graph_entities()` checks query against knowledge graph aliases; known entities (e.g. "Flapjack") trigger memory search
3. **LLM fallback**: piggybacks on `analyze_for_web_search_llm()`; `needs_memory_search` or `needs_knowledge_search` fields in `WebSearchDecision` catch structural recall/encyclopedic queries. Memory search takes priority: if LLM returns both `should_search` and `needs_memory_search`, the query routes to memory (not web).

Casual skip filter (< 5 words, "thanks", etc.) only applies when no keyword/entity/knowledge/tool-name trigger fired.
`skip_initial_search=True` for computation, memory, knowledge, and tool-name queries. When Tier 1 fires without LLM-generated search terms, initial search is also skipped.

**Post-generation trigger** [NEW 2026-04]: If standard response is uncertain ("I don't know"), `UncertaintyDetector` (`core/uncertainty_detector.py`) retries via agentic search. Keyword regex (~18 patterns) + semantic embedding layer. Config: `UNCERTAINTY_FALLBACK_ENABLED`, `UNCERTAINTY_SEMANTIC_THRESHOLD`, `UNCERTAINTY_MAX_LENGTH`.

```python
# core/agentic/controller.py — orchestration, prompt building, model interaction, quality heuristics,
#   nudge retry (re-prompts model when it narrates tools instead of calling them),
#   no-reasoning decision phase (_generate_decision_no_reasoning bypasses chain-of-thought for XML tool emission),
#   tool hints (_detect_tool_hints injects usage hints when query mentions tool names)
# core/agentic/tools.py — ToolExecutor: dispatch routing + 19 execute methods + get_tool_health()
# core/agentic/gate.py — 4-tier agentic gate: keyword → entity → tool-name → LLM fallback [NEW 2026-05]
# core/agentic/formatters.py — AgenticFormatter: 18 pure formatting methods
# core/actions/ — Internet action executors: telegram.py, discord.py, email.py, google_auth.py, google_calendar.py, google_calendar_create.py, types.py, audit.py, executors.py [NEW 2026-05]
class AgenticSearchController:
    """ReAct loop: Reason → Act (search/compute) → Observe → repeat until done.
    Delegates tool execution to ToolExecutor, formatting to AgenticFormatter."""

    async def run_agentic_search(query: str, system_prompt: str, model_name: str,
                                   initial_search_terms: List[str], initial_context=None,
                                   crisis_level=None, skip_initial_search=False,
                                   initial_urls: List[str] = None) -> AsyncGenerator[Union[ProgressEvent, str], None]:
        """
        Main loop:
        1. Build prompt with gathered context + tool instructions
        1b. Compute context inventory (summarize RAG-gathered sections) [ENHANCED 2026-03-23]
        1c. Inject [TOOL STATUS] via ToolExecutor.get_tool_health() [NEW 2026-05]
        2. Call LLM for decision (search/wolfram/memory/git_stats/done)
        3. Execute tool if requested (via ToolExecutor), add result to context
        3b. Track memory_search_counts per collection (diversity enforcement) [ENHANCED 2026-03-23]
        4. Repeat until done or max_turns reached
        5. Yield AgenticEvent for each stage (thinking/searching/computed/done)
        """

    # ToolExecutor.get_tool_health() → "[TOOL STATUS]" prompt section [UPDATED 2026-05]
    # Reports per-tool availability: web_search (+ site param), wiki_knowledge (FAISS),
    # memory_search (ChromaDB), wolfram, file_access, git_stats, github, expand_memory, visual_memory,
    # fetch_url, search_stackexchange, search_arxiv, search_pubmed, search_hackernews,
    # propose_action (lists calendar_create_event when GOOGLE_CALENDAR_ENABLED).
    # Dedicated API tools are always AVAILABLE (free, no auth).
    # Injected into agentic system prompt + iteration prompts + final prompt.

    # Budget-enforced context accumulation [NEW 2026-03-28]
    def _append_accumulated(session, new_context) -> None:
        """Append to accumulated_context; trims oldest rounds if > context_budget_tokens."""
    # Budget-aware final prompt assembly [NEW 2026-03-28]
    def _build_final_prompt(query, session, initial_context) -> str:
        """Assembles final prompt; trims low-value sections (dreams, reflections, docs,
        summaries) if total exceeds ceiling, preserving recent_conversations + agentic results."""


# core/agentic/types.py
@dataclass
class SearchDecision:
    """LLM's decision at each turn"""
    wants_search: bool = False
    search_query: Optional[str] = None
    search_reason: Optional[str] = None
    wants_wolfram: bool = False         # Computation request
    wolfram_query: Optional[str] = None
    wolfram_reason: Optional[str] = None
    wants_sandbox: bool = False         # Code execution [NEW 2026-01-22]
    sandbox_code: Optional[str] = None
    sandbox_purpose: Optional[str] = None
    wants_memory_search: bool = False   # Memory search [NEW 2026-03-15]
    memory_query: Optional[str] = None
    memory_collection: Optional[str] = None
    wants_memory_expand: bool = False   # Memory expansion [NEW 2026-03-26]
    expand_memory_id: Optional[str] = None
    expand_window: int = 3
    expand_collection: Optional[str] = None
    expand_reason: Optional[str] = None
    wants_git_stats: bool = False       # Git repository stats [NEW 2026-03-29]
    git_stats_query: Optional[str] = None
    git_stats_reason: Optional[str] = None
    wants_recall_image: bool = False   # Visual memory recall [NEW 2026-05]
    recall_image_query: Optional[str] = None
    recall_image_reason: Optional[str] = None
    wants_fetch_url: bool = False      # Fetch URL content [NEW 2026-05]
    fetch_url: Optional[str] = None
    fetch_url_reason: Optional[str] = None
    # Daemon self-note (save working context for future sessions) [NEW 2026-05]
    wants_create_daemon_note: bool = False
    daemon_note_title: Optional[str] = None
    daemon_note_category: Optional[str] = None  # implementation | architecture | research | decisions
    daemon_note_summary: Optional[str] = None
    daemon_note_reason: Optional[str] = None
    # Internet action (propose write action requiring user confirmation) [NEW 2026-05]
    wants_action: bool = False
    action_type: Optional[str] = None  # ActionType value
    action_params: Optional[Dict[str, Any]] = None
    action_summary: Optional[str] = None
    action_reason: Optional[str] = None
    is_done: bool = False
    done_reason: Optional[str] = None
    wants_answer: bool = False
    partial_response: Optional[str] = None


# core/agentic/types.py — Session state [ENHANCED 2026-03-23]
@dataclass
class AgenticSearchSession:
    memory_search_counts: Dict[str, int] = field(default_factory=dict)  # Per-collection diversity tracking
    context_inventory: str = ""  # Summary of what RAG pipeline already gathered
    expand_count: int = 0  # Memory expansion count for session gating [NEW 2026-03-26]
    # ... (also: round_count, search_results, memory_results, etc.)

# MEMORY_SEARCH_TOOL_DEFINITION — Enhanced with per-collection descriptions [ENHANCED 2026-03-23]
# Tool description includes what each collection contains:
#   summaries → profile overviews, biographical questions
#   conversations → temporal recall, specific exchanges
#   facts → individual extracted triples (name=X, age=Y)
#   reflections → session reflections/insights
#   reference_docs → architecture/documentation
#   obsidian_notes → user's personal notes
#   procedural → git commits, how-to knowledge
#   procedural_skills → learned problem-solving patterns
# Diversity enforcement: "avoid searching the same collection repeatedly"

# GIT_STATS_TOOL_DEFINITION — Git repository activity stats [NEW 2026-03-29]
# Function name: "git_stats", single param: query (str, required)
# Natural-language intent parsing → safe git subcommands (log, shortlog, diff, status, etc.)

# FETCH_URL_TOOL_DEFINITION — Fetch web page content by URL [NEW 2026-05]
# Function name: "fetch_url", params: url (str, required), reason (str, optional)
# Calls WebSearchManager._tavily_extract([url]); result registered in web_source_map for [WEB_N] citations
# Gated on web_search_manager.is_available(); auto-triggered for URLs in user messages

# PROPOSE_ACTION_TOOL_DEFINITION — Internet write action proposal [NEW 2026-05, ENHANCED 2026-05]
# Function name: "propose_action"
# Params: action_type (enum: send_telegram, send_discord, send_email,
#   github_create_issue, github_comment_pr, calendar_create_event),
#   recipient (str), subject (str), message (str), reason (str),
#   summary (str, calendar only), description (str, calendar only),
#   start_time (str, ISO 8601, calendar only), end_time (str, ISO 8601, calendar only),
#   time_zone (str, IANA, calendar only), calendar_id (str, calendar only),
#   location (str, calendar only)
# Required: action_type, reason (relaxed from action_type, message, reason — calendar uses summary instead of message)
# User sees confirmation prompt and can approve/reject; at most ONE action per turn
# Execution via core/actions/ executors (Telegram, Discord, email, Gmail API, GitHub, Google Calendar)

# core/git_stats_manager.py [NEW 2026-03-29]
class GitStatsManager:
    """Read-only git repository stats for the agentic loop"""

    def is_available() -> bool:
        """True if cwd is inside a git repo"""

    async def execute_query(query: str) -> Dict[str, Any]:
        """Parse natural-language query → safe git subcommand → structured result dict"""

    def format_for_prompt(result: Dict) -> str:
        """Format result dict for LLM context injection"""

# Safety: only ALLOWED_SUBCOMMANDS may execute (log, shortlog, diff, status, branch, etc.)
# Keyword-based intent parsing maps queries to git commands
# Temporal phrase extraction: "this week", "today", "last 30 days" → --since dates

# core/github_manager.py [NEW 2026-05-19]
class GitHubManager:
    """Read-only GitHub API access via gh CLI for the agentic loop"""

    def is_available() -> bool:
        """True if gh CLI is installed and authenticated"""

    async def execute_query(query: str) -> Dict[str, Any]:
        """Parse natural-language query → safe gh subcommand → structured result dict"""

    def format_for_prompt(result: Dict) -> str:
        """Format result dict for LLM context injection"""

# Safety: only read-only gh subcommands (issue list, pr list, run list, release list, etc.)
# Keyword-based intent parsing: issues, PRs, actions, releases, workflows, labels, milestones,
#   contributors, code search
# Config: GITHUB_API_ENABLED, GITHUB_API_TIMEOUT, GITHUB_API_MAX_OUTPUT_LINES, GITHUB_API_REPO
#   YAML section: github_api


# knowledge/wolfram_manager.py [NEW]
class WolframManager:
    """Wolfram Alpha LLM API for computations"""

    async def query(input_text: str) -> WolframResult:
        """
        1. Check cache (MD5 hash key, 1hr TTL)
        2. Check rate limit (token bucket)
        3. Call Wolfram Alpha LLM API
        4. Parse response, cache success
        5. Return WolframResult
        """

    def format_for_prompt(result: WolframResult) -> str:
        """Format result for LLM context injection"""

    def is_available() -> bool:
        """True if WOLFRAM_APP_ID is configured"""


@dataclass
class WolframResult:
    query: str
    success: bool
    result: str = ""
    input_interpreted: str = ""
    assumptions: List[str] = field(default_factory=list)
    error: Optional[str] = None
    execution_time: float = 0.0
    cached: bool = False


# knowledge/sandbox_manager.py [NEW 2026-01-22, UPDATED 2026-05-18]
class SandboxManager:
    """E2B Code Sandbox for secure Python execution"""

    async def execute_code(code: str) -> SandboxResult:
        """Execute code in ephemeral sandbox"""

    async def create_session() -> PersistentSession:
        """Create persistent session (variables survive across calls)"""

# Persistent sandbox session [NEW 2026-05-18]:
# Session held on AgenticSearchController._sandbox_session (lazy, 10min timeout)
# Variables/dataframes/imports survive across messages in the same conversation.
# System prompt instructs LLM to reuse existing variables.
# Cleanup: controller.close_sandbox() called in main.py _do_shutdown()

    def format_for_prompt(result: SandboxResult, purpose: str) -> str:
        """Format result for LLM context injection"""

    def is_available() -> bool:
        """True if E2B_API_KEY is configured"""


@dataclass
class SandboxResult:
    code: str
    success: bool
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    results: List[Dict] = field(default_factory=list)  # Rich outputs (images, etc.)
    execution_time: float = 0.0
    cached: bool = False


# memory/memory_expander.py [NEW 2026-03-26]
class MemoryExpander:
    """Temporal context expansion + summary drill-down for expand_memory tool"""

    def expand(memory_id: str, window: int = 3, collection: str = None) -> dict:
        """
        Two strategies:
        - timestamp_window: fetch ±window chronological neighbors (conversations, reflections, facts, obsidian_notes)
        - source_docs: for summaries, retrieve original conversations via source_doc_ids metadata
          or temporal_anchor_start/end range. Falls back to timestamp_window if no linkage.
        Returns: {anchor_id, collection, expansion_method, turns, total_in_collection, error}
        """

    def clear_cache():
        """Reset cache between ReAct sessions"""


# Protocol handlers (core/agentic/protocols.py) [ENHANCED 2026-05]
# XMLMarkerHandler - for local models: <search>, <fetch_url>, <wolfram>, <python>, <memory>, <expand_memory>, <file_read>, <file_grep>, <file_list>, <git_stats>, <github>, <recall_image>, <done>
#   Alias patterns: <web_search>/<web_search query="..."> → <search>, <search_memory query="..."> → <memory>
#   Nested XML: <search_memory><query>X</query></search_memory> (DeepSeek-style) via MEMORY_NESTED_PATTERN
#   Helpers: _strip_xml_tags() removes inner tags from content, _extract_nested_tag() extracts child elements
#   Empty args defaulting: git_stats/github/search_memory with empty args fall back to original query
#   Calendar fix: forwards calendar-specific params (summary, start_time, end_time, etc.) to action_params
#   Text parser: only prepends send_ for messaging types (telegram, discord, email), not calendar/github
# NativeToolsHandler - for API models: OpenAI/Anthropic function calling (tool defs gated by github_available param)
```

### Agentic Config Constants

```python
# config/app_config.py

# Web Search (Tavily)
WEB_SEARCH_ENABLED = True
WEB_SEARCH_API_KEY = os.getenv("TAVILY_API_KEY", "")
WEB_SEARCH_DAILY_CREDIT_LIMIT = 100
WEB_SEARCH_CACHE_TTL_HOURS = 72
# Fixes [2026-03-23]: Tavily 400-char query truncation (web_search_manager.py),
#   long paste prefilter >500 chars skips LLM (web_search_trigger.py:quick_prefilter_should_skip)

# Wolfram Alpha [NEW]
WOLFRAM_ENABLED = True
WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID", "")
WOLFRAM_API_URL = "https://www.wolframalpha.com/api/v1/llm-api"
WOLFRAM_TIMEOUT = 30.0
WOLFRAM_CACHE_TTL_SECONDS = 3600
WOLFRAM_RATE_LIMIT_PER_MINUTE = 60
WOLFRAM_MAX_OUTPUT_CHARS = 10000

# E2B Code Sandbox [NEW 2026-01-22]
SANDBOX_ENABLED = True
SANDBOX_API_KEY = os.getenv("E2B_API_KEY", "")
SANDBOX_TIMEOUT_SECONDS = 60
SANDBOX_SESSION_TIMEOUT_MINUTES = 30
SANDBOX_MAX_OUTPUT_CHARS = 4000
SANDBOX_CACHE_TTL_SECONDS = 3600
SANDBOX_RATE_LIMIT_PER_MINUTE = 30

# Git Stats [NEW 2026-03-29]
GIT_STATS_ENABLED = True
GIT_STATS_TIMEOUT = 10                 # subprocess timeout (seconds)
GIT_STATS_MAX_OUTPUT_LINES = 50        # cap git command output

# Internet Actions [NEW 2026-05, ENHANCED 2026-05]
INTERNET_ACTIONS_ENABLED = False       # disabled by default
INTERNET_ACTIONS_TELEGRAM_BOT_TOKEN    # env: TELEGRAM_BOT_TOKEN
INTERNET_ACTIONS_TELEGRAM_CHAT_ID      # env: TELEGRAM_CHAT_ID
INTERNET_ACTIONS_DISCORD_WEBHOOK_URL   # env: DISCORD_WEBHOOK_URL
INTERNET_ACTIONS_SMTP_HOST / _PORT / _USER / _PASSWORD / _FROM
INTERNET_ACTIONS_GITHUB_WRITE_ENABLED = False
INTERNET_ACTIONS_TTL = 300             # action expiry (seconds)
INTERNET_ACTIONS_MAX_PENDING = 5       # max queued actions
INTERNET_ACTIONS_AUDIT_LOG = "logs/actions_audit.jsonl"
# Google OAuth2 + Calendar [NEW 2026-05]
INTERNET_ACTIONS_GOOGLE_CLIENT_ID     # env: GOOGLE_CLIENT_ID
INTERNET_ACTIONS_GOOGLE_CLIENT_SECRET  # env: GOOGLE_CLIENT_SECRET
INTERNET_ACTIONS_GOOGLE_TOKEN_PATH = "data/google_token.json"
GOOGLE_CALENDAR_ENABLED = False        # env: GOOGLE_CALENDAR_ENABLED
GOOGLE_CALENDAR_MAX_EVENTS = 10        # max events in prompt
GOOGLE_CALENDAR_LOOKAHEAD_DAYS = 7     # days ahead to fetch
#   YAML section: internet_actions
```

### Tool Invocation (Prompt Injection)

```python
# Added to system prompt for agentic mode (types.py AGENTIC_SYSTEM_PROMPT_INJECTION):
"""
Available Tools:
1. <search>query</search> - Web search for current events, facts, data
2. <wolfram>query</wolfram> - Computation, math, science, conversions
3. <python purpose="...">code</python> - Execute Python code (NumPy, Pandas, SciPy, SymPy available)
4. <memory collection="...">query</memory> - Search internal memory/knowledge base [NEW 2026-03-15]
5. <expand_memory id="..." collection="..." window="3">reason</expand_memory> - Zoom in on a memory [NEW 2026-03-26]
6. <file_read path="...">reason</file_read> - Read a file from filesystem [NEW 2026-03-26]
7. <file_grep pattern="..." path="...">reason</file_grep> - Search file contents by regex [NEW 2026-03-26]
8. <file_list path="...">reason</file_list> - List directory contents [NEW 2026-03-26]
9. <git_stats>query</git_stats> - Git repository activity stats [NEW 2026-03-29]
10. <recall_image>query</recall_image> - Recall images from visual memory [NEW 2026-05]
11. <fetch_url url="...">reason</fetch_url> - Fetch web page content by URL [NEW 2026-05]
12. <done>reason</done> - Signal task complete

Use Python for: multi-step computation, data analysis, visualization, custom algorithms
Use Wolfram for: single-expression calculations, unit conversions, scientific data, equations
Use search for: current events, recent news, real-time data, general facts
Use fetch_url for: reading a specific URL the user provided or that appeared in search results
Use memory for: user facts, past conversations, personal notes, project history
Use expand_memory for: see surrounding conversation turns or drill into summaries
Use file_read/file_grep/file_list for: reading project files, searching code, listing directories
Use git_stats for: commit counts, recent commits, contributors, files changed, branch activity
Use recall_image for: recalling photos, screenshots, diagrams, or other images from past interactions
"""
```

---

## Personality System (File-Based) **[NEW 2026-03-26]**

```python
# config/app_config.py — Replaces deleted PersonalityManager (JSON configs)
# System prompt = personality text + operating principles, composed via load_personality_text() + load_operating_principles()

def load_personality_text() -> str:
    """Load custom_personality.txt if exists, else default_personality.txt.
    Truncates to PERSONALITY_MAX_CHARS (15000). Returns empty string if no file found."""

def load_default_personality() -> str:
    """Load shipped default personality (for GUI Restore Default button)."""

def load_operating_principles() -> str:
    """Load immutable operating principles (AI limitations, behavioral rules, claim grounding, etc.)."""

# Files:
# config/prompts/default_personality.txt — 94 lines: tone, interaction style, response approach, examples
# config/prompts/custom_personality.txt — created by GUI "Personality" tab Set button
# config/prompts/operating_principles.txt — 255 lines: AI limitations, facts handling, claim grounding,
#   response modes, knowledge source instructions, agentic tool docs, guardrails
# Placeholder substitution: {USER_NAME}, {USER_PRONOUNS}, {PRONOUN_SUBJ}, {PRONOUN_OBJ}, {PRONOUN_POSS}

# gui/launch.py — "Personality" tab:
#   Textbox (25 lines) pre-loaded with current personality text
#   "Set" button: saves to custom_personality.txt, enforces PERSONALITY_MAX_CHARS
#   "Restore Default" button: deletes custom file, shows default
#   Operating principles appended automatically, not editable
```

---

## Citation & Provenance System **[NEW 2026-03-26]**

```python
# Every stored interaction includes provenance metadata for audit trail.

# memory/memory_coordinator.py
class MemoryCoordinator:
    @property
    def session_id(self) -> str:
        """Unique session UUID, generated once per MemoryCoordinator instance."""

# memory/memory_storage.py
class MemoryStorage:
    async def store_interaction(query, response, tags=None, session_id=None, provenance=None):
        """
        Accepts optional provenance dict merged into ChromaDB metadata:
          session_id, response_mode, model_name, thinking_block,
          cited_memory_ids, prompt_hash, agentic_summary
        """

# core/agentic/types.py
@dataclass
class AgenticSearchSession:
    final_prompt_hash: str = ""           # SHA-256 of final assembled prompt
    def get_provenance_summary() -> str:
        """One-line summary: tools used, rounds, searches, memory expansions."""

# core/agentic/controller.py
class AgenticSearchController:
    _last_session: AgenticSearchSession   # Accessible after run_agentic_search()
    # Hashes final prompt (SHA-256), adds [MEM_RECENT_N] / [MEM_SEMANTIC_N] citation markers

# core/response_parser.py
class ResponseParser:
    @staticmethod
    def parse_thinking_block(response) -> Tuple[str, str]:
        """Extract <thinking>...</thinking> and visible answer. Truncated to PROVENANCE_THINKING_MAX_CHARS."""

# gui/handlers.py — Builds provenance dict for all 5 response modes:
#   agentic:     response_mode="agentic", agentic_summary, prompt_hash, cited_memory_ids
#   enhanced:    response_mode="enhanced"
#   best-of:     response_mode="best_of", model_name from best-of metadata
#   best-of-duel: response_mode="best_of_duel", model_name = winner model
#   fallback:    response_mode="fallback"
# Thinking block captured for extended-thinking models (DeepSeek-R1, etc.)

# gui/launch.py — "Provenance" tab (renamed from "Citations")
# Displays per-turn provenance: session_id, response_mode, model, thinking block,
# cited memory IDs, prompt hash, and agentic tool summary.

# Config (app_config.py):
PROVENANCE_ENABLED = True                # Toggle provenance metadata collection
PROVENANCE_THINKING_MAX_CHARS = 4000     # Truncation limit for thinking block storage

# YAML (config.yaml):
# provenance:
#   enabled: true
#   thinking_max_chars: 4000

# gatherer_memory.py fix: recent conversations now build content from Q/A
# when the content field is empty (robustness for older corpus entries).

# Tests: 21 unit tests in tests/unit/test_provenance.py
```

---

## Knowledge Synthesis Filter Pipeline **[NEW 2026-03-28]**

```python
# knowledge/synthesis_models.py — Data models
class CoherenceLevel(Enum):          # INVALID(0.0), WEAK(0.33), MODERATE(0.66), STRONG(1.0)
class CandidateStatus(Enum):         # PENDING, REJECTED, ACCEPTED, CONVERGING

@dataclass
class SynthesisCandidate:
    concept_a: str                   # First endpoint concept
    concept_b: str                   # Second endpoint concept
    connection_claim: str            # Articulated bridge statement
    walk_path: List[str]             # Full random walk node sequence
    source_domains: Set[str]         # Domain tags crossed
    endpoint_distance: float         # Cosine distance between endpoints
    path_hash: str                   # SHA-256[:16] of walk_path

@dataclass
class SynthesisResult:
    candidate: SynthesisCandidate
    stage_results: List[StageResult] # Per-stage pass/fail/score/timing
    coherence_level: Optional[CoherenceLevel]
    novelty_score_external: float    # 1 - nearest_wiki_similarity
    novelty_score_internal: float    # 1 - nearest_synthesis_similarity
    cooccurrence_similarity: float   # how often A and B already co-occur in wiki
    template_similarity: float       # how close claim is to generic bridge templates
    composite_score: float
    status: CandidateStatus
    unique_paths: Set[str]           # Convergence: independent walk paths
    unique_sources: Set[str]         # Convergence: distinct concept pairs
    convergence_strength: float      # |paths| * |sources|
    # to_metadata() -> dict, from_metadata(dict, doc) -> SynthesisResult

# memory/synthesis_memory.py — Persistence + convergence tracking
class SynthesisMemory:
    COLLECTION_NAME = "synthesis_results"  # 12th ChromaDB collection
    def __init__(self, chroma_store, similarity_threshold=0.85): ...
    def find_similar(claim, threshold=None, limit=5) -> List[Tuple[SynthesisResult, float]]: ...
    def store_result(result) -> str:       # Deduplicates; updates convergence if similar exists
    def get_recurring(min_paths=3, min_sources=2) -> List[SynthesisResult]: ...
    def get_stats() -> dict:               # {total_insights, converging_insights, collection}
    # Audit queue methods [NEW 2026-04-01]:
    def grade_result(doc_id, grade, notes, changes_thinking, mechanism_real, heard_before) -> bool:  # Two-layer grading (3 binary screening + 1-5 slider); see `docs/grading_plan.md`
    def get_ungraded(limit=10) -> List[SynthesisResult]: ...
    def get_graded(grade=None, limit=50) -> List[SynthesisResult]: ...
    def get_audit_stats() -> dict:         # {total_graded, valid_count, invalid_count, fp_rate, avg_grade, auto_halt, ungraded_accepted, ungraded_rejected, fp_halt_threshold, min_graded_for_halt}
    def store_rejected_for_audit(result) -> str:  # Stores composite-rejected candidate for FN review

# knowledge/synthesis_filter.py — 7-stage async pipeline (storage is post-pipeline)
# Module-level helpers:
#   _extract_faiss_similarity(results) -> float — extracts top cosine similarity from FAISS search results
#   _compute_template_similarity(claim) -> float — regex-based generic bridge detection
#   _GENERIC_TEMPLATES — compiled regex patterns for vacuous bridge claims
#   _GENERIC_TOKENS — frozenset of generic buzzwords
class SynthesisFilter:
    def __init__(self, chroma_store, model_manager, synthesis_memory=None, graph_memory=None, entity_resolver=None): ...
    async def process_candidate(candidate) -> SynthesisResult:  # Full pipeline, auto-stores accepted
    async def process_batch(candidates) -> dict:
        # Returns: {total, accepted, rejected, rejection_breakdown, accepted_results, avg_stage_times_ms}
        # Also stores composite-rejected candidates for FN audit review via synthesis_memory.store_rejected_for_audit()
    async def _factual_skeptic_pass(result, claim, a, b) -> bool:  # Pass 2 of Stage 5
    @staticmethod
    def _parse_coherence_level(response_text) -> CoherenceLevel:  # Extracted from inline parsing

# Pipeline stages (cheap → expensive):
# 0: text_sanity      — min tokens, verb check, repetition ratio (~0ms)
# 1: domain_crossing   — min 2 distinct domains (~1ms)
# 2: semantic_distance  — endpoint distance in [0.20, 0.90] (~5ms)
# 3: novelty_external   — 3 sub-checks (~15ms, FAISS wiki vector search, 40M vectors):
#      a) claim similarity: full claim vs wiki via FAISS (hard gate at 0.80)
#      b) co-occurrence: bare "concept_a concept_b" vs wiki via FAISS (hard gate at 0.75)
#      c) template specificity: regex generic bridge pattern detection
# 4: novelty_internal   — synthesis memory; new paths pass as convergence (~10ms)
# 5: coherence_judge    — two-pass LLM: Pass 1 structural coherence (4-tier rating),
#                          Pass 2 factual skeptic (MODERATE only, binary PASS/FAIL) (~500ms-2s)
# 6: composite_scoring  — weighted 4-signal novelty composite ≥ 0.65 threshold
# Storage happens post-pipeline in process_candidate(), not as a formal stage

# Composite: 0.30*coherence + 0.40*novelty + 0.15*distance + 0.15*structural
# Novelty (4-signal):
#   0.25*claim_novelty(1-claim_sim) + 0.30*cooccurrence_novelty(1-cooccurrence_sim)
#   + 0.25*specificity(1-template_sim) + 0.20*internal_novelty(synthesis_memory)

# Config (app_config.py):
SYNTHESIS_ENABLED = True
SYNTHESIS_MIN_TOKEN_LENGTH = 10
SYNTHESIS_MAX_REPETITION_RATIO = 0.5
SYNTHESIS_MIN_DOMAINS = 2
SYNTHESIS_DISTANCE_MIN = 0.20
SYNTHESIS_DISTANCE_MAX = 0.90
SYNTHESIS_NOVELTY_KNOWN_THRESHOLD = 0.88   # Claim sim gate — only near-verbatim rehashes
SYNTHESIS_NOVELTY_ADJACENT_THRESHOLD = 0.70  # Label threshold
SYNTHESIS_COOCCURRENCE_KNOWN_THRESHOLD = 0.85  # Co-occurrence gate — 40M-scale recalibrated
SYNTHESIS_MEMORY_SIMILARITY_THRESHOLD = 0.85
SYNTHESIS_COHERENCE_MODEL = "sonnet-4.5"  # code default; config.yaml overrides to "claude-opus-4.6"
SYNTHESIS_COHERENCE_MIN_LEVEL = "MODERATE"
SYNTHESIS_WEIGHT_COHERENCE = 0.30
SYNTHESIS_WEIGHT_NOVELTY = 0.40
SYNTHESIS_WEIGHT_DISTANCE = 0.15
SYNTHESIS_WEIGHT_STRUCTURAL = 0.15
SYNTHESIS_NOVELTY_W_CLAIM = 0.25             # 4-signal novelty weights
SYNTHESIS_NOVELTY_W_COOCCURRENCE = 0.30
SYNTHESIS_NOVELTY_W_SPECIFICITY = 0.25
SYNTHESIS_NOVELTY_W_INTERNAL = 0.20
SYNTHESIS_COMPOSITE_MIN_SCORE = 0.65   # raised from 0.40
SYNTHESIS_CONVERGENCE_STRONG_PATHS = 3
SYNTHESIS_CONVERGENCE_STRONG_SOURCES = 2
SYNTHESIS_DEFAULT_BATCH_SIZE = 100
SYNTHESIS_LOG_ALL_REJECTIONS = True

# Audit queue [NEW 2026-04-01]:
SYNTHESIS_AUDIT_ENABLED = True
SYNTHESIS_AUDIT_FP_HALT_THRESHOLD = 0.50   # Auto-halt synthesis if FP rate exceeds this
SYNTHESIS_AUDIT_MIN_GRADED = 10            # Minimum graded results before auto-halt activates

# YAML (config.yaml):
# synthesis:
#   enabled: true
#   batch_size: 100
#   distance_min: 0.20
#   distance_max: 0.90
#   cooccurrence_known_threshold: 0.85   # 40M-scale recalibrated
#   coherence_model: claude-opus-4.6
#   coherence_min_level: MODERATE
#   weights: {coherence: 0.30, novelty: 0.40, distance: 0.15, structural: 0.15}
#   novelty_weights: {claim: 0.25, cooccurrence: 0.30, specificity: 0.25, internal: 0.20}
#   composite_min_score: 0.65   # raised from 0.40

# synthesis_audit:
#   enabled: true
#   fp_halt_threshold: 0.50
#   min_graded: 10
```

---

## Synthesis Generator (Cross-Store Candidate Generation) **[NEW 2026-03-28]**

```python
# knowledge/synthesis_generator.py — Cross-store sampling + LLM bridge articulation
class SynthesisGenerator:
    def __init__(self, chroma_store, model_manager, graph_memory=None, entity_resolver=None): ...
    async def generate_candidates(count=5) -> List[SynthesisCandidate]:
        # 1. Sample personal entities from facts collection (broad query seeds)
        # 2. Sample wiki articles via FAISS semantic_search_with_neighbors (40M vectors)
        # 3. Form cross-domain pairs (deduplicated, domain-classified)
        # 4. Parallel LLM bridge articulation (semaphore-limited concurrency)
        # 5. Package as SynthesisCandidate objects for filter pipeline
    def _sample_personal_entities(n) -> List[Dict]:  # 12 query seeds, shuffled
    def _sample_wiki_articles(n) -> List[Dict]:       # FAISS search, 12 query seeds, shuffled
    def _classify_domain(item) -> str:                # categorize_relation() 5-layer cascade (direct→prefix→token→embedding→LLM)
    async def _articulate_bridge(concept_a, concept_b, domain_a, domain_b, ctx_a, ctx_b) -> Optional[str]:
        # LLM call; returns None on NO_CONNECTION or <5 words
    def _compute_endpoint_distance(concept_a, concept_b) -> float:
        # Graph shortest path (normalized to [0.15, 1.0]) or default 0.55
    def get_sampling_stats() -> dict:                 # {facts_count, wiki_count, graph_nodes, graph_edges}

# Integration:
# - Shutdown: step 6.8 — three generators run in parallel (retrieval Tier 0,
#   graph walk Tier 1, cross-store Tier 2) with independent quotas
# - Pipeline: generate_candidates() → SynthesisFilter.process_batch() → SynthesisMemory
# - On acceptance: provisional bridge edge created (weight=0.0, status="provisional")
# - Graph sparsity guard: skips if graph < SYNTHESIS_GENERATOR_MIN_GRAPH_NODES nodes

# Config (app_config.py):
SYNTHESIS_GENERATOR_ENABLED = False               # All three generators currently DISABLED in config.yaml pending grading validation
SYNTHESIS_GENERATOR_CANDIDATES_PER_SESSION = 5
SYNTHESIS_GENERATOR_LLM_CONCURRENCY = 5
SYNTHESIS_GENERATOR_MIN_GRAPH_NODES = 20

# YAML (config.yaml):
# synthesis_generator:
#   enabled: false                                  # DISABLED pending grading validation
#   candidates_per_session: 5
#   llm_concurrency: 5
#   min_graph_nodes: 20

# Tests: 18 unit tests in tests/unit/test_synthesis_generator.py
# Calibration: 6 tests in tests/test_synthesis_calibration.py
# Fixtures: tests/fixtures/calibration_candidates.json (72 labeled candidates, 7 tiers)
```

---

## Retrieval-Based Synthesis Generator **[NEW 2026-04-01]**

```python
# knowledge/synthesis_retriever.py — Structural query extraction + FAISS search + adversarial eval
class RetrievalSynthesisGenerator:
    def __init__(self, chroma_store, model_manager, graph_memory=None, entity_resolver=None): ...
    async def generate_candidates(count=5) -> List[SynthesisCandidate]:
        # 1. Sample personal facts from ChromaDB
        # 2. LLM structural query extraction (few-shot): fact → structural pattern query
        # 3. FAISS semantic search (40M vectors) using structural query
        # 4. Adversarial evaluation: LLM judges structural vs surface connection
        # 5. Package as SynthesisCandidate objects
    # Same interface as SynthesisGenerator — drop-in replacement

# Config (app_config.py):
SYNTHESIS_RETRIEVAL_ENABLED = False                  # DISABLED pending grading validation
SYNTHESIS_STRUCTURAL_QUERY_MAX_TOKENS = 100          # Max tokens for structural query LLM call
SYNTHESIS_RETRIEVAL_K = 5                            # FAISS results per structural query
SYNTHESIS_RETRIEVAL_MIN_SIMILARITY = 0.25            # Min cosine similarity for FAISS results
SYNTHESIS_BRIDGE_ON_ACCEPT = True                    # Create provisional graph edge on acceptance
SYNTHESIS_BRIDGE_RELATION = "structural_parallel"    # Relation type for provisional bridges

# Graph walk generator config (app_config.py):
GRAPH_WALK_HUB_DEGREE_THRESHOLD = 15   # Hub dampening: log-scale penalty above this degree
GRAPH_WALK_MIN_DOMAINS = 2             # Min distinct domain categories per walk

# YAML (config.yaml):
# synthesis_retrieval:
#   enabled: false                                  # DISABLED pending grading validation
#   structural_query_max_tokens: 100
#   retrieval_k: 5
#   min_similarity: 0.25
#   bridge_on_accept: true
#   bridge_relation: structural_parallel
```

---

## Visual Memory **[NEW 2026-05]**

```python
# knowledge/clip_manager.py — OpenCLIP ViT-B/32 singleton, lazy-loaded
class CLIPManager:
    """Singleton CLIP model for image/text embedding (512-dim vectors).
    Lazy-loaded on first use — no GPU/model overhead until needed."""

    def encode_image(image_path: str) -> np.ndarray:
        """Load image, preprocess, return 512-dim L2-normalized CLIP embedding."""

    def encode_text(text: str) -> np.ndarray:
        """Tokenize text, return 512-dim L2-normalized CLIP embedding."""


# knowledge/visual_memory_store.py — Dual storage: ChromaDB + FAISS FlatIP
class VisualMemoryStore:
    """Dual-index storage: ChromaDB 'visual_memories' collection (text metadata, captions)
    + FAISS FlatIP index (CLIP 512-dim vectors). SHA-256 content dedup on ingestion."""

    def add(image_path, clip_embedding, caption, metadata) -> str:
        """Store image: FAISS vector + ChromaDB doc. Returns doc_id. Dedup by SHA-256."""

    def search_by_clip(query_embedding, k=3) -> List[Dict]:
        """FAISS inner-product search on CLIP vectors → top K image results."""

    def search_by_text(query, k=3) -> List[Dict]:
        """ChromaDB semantic search on captions/metadata."""


# knowledge/visual_memory_pipeline.py — Ingestion pipeline
class VisualMemoryPipeline:
    """Ingestion: CLIP embed → vision LLM caption → entity tag → store."""

    async def ingest_image(image_path, user_context=None) -> str:
        """Full pipeline: CLIP encode → vision LLM caption → entity extraction → dual store."""


# knowledge/visual_retrieval.py — Hybrid retrieval for prompt injection
class VisualRetriever:
    """Hybrid retrieval: CLIP text→image similarity + ChromaDB text search.
    Entity hard-filtering via target_entities parameter.
    Returns results in note_images format for prompt builder."""

    async def retrieve(query, max_images=3, target_entities=None) -> List[Dict]:
        """Hybrid search: CLIP text embedding vs stored image embeddings + ChromaDB caption search.
        Merges and deduplicates results. When target_entities provided, hard-filters
        to images tagged with at least one target entity (falls back to all on empty).
        Returns note_images-compatible format."""


# Entity-gated retrieval (gatherer_knowledge.py:get_visual_memories):
#   A. Filter junk caption artifacts from stored entity IDs (_VISUAL_JUNK_IDS)
#   B. Entity resolution via extract_graph_entities() (alias-aware)
#   C. Substring fallback for visual-only entities not in graph
#   D. Multi-entity disambiguation via intent-proximity (_VISUAL_INTENT_WORDS)
#   E. Pass target_entities to VisualRetriever for hard-filtering
# Short messages (<=5 words) without intent override suppress retrieval.
# Prompt instruction: do not mention images unless user explicitly asks.

# Integration:
# - Collection: 'visual_memories' in ChromaDB (14 total collections)
# - Prompt section: [VISUAL MEMORIES] (after [USER UPLOADED ITEMS])
# - Agentic tool: recall_image — CLIP-based image recall in ReAct loop
# - Backfill: scripts/backfill_visual_memory.py — re-index existing images

# Config (app_config.py):
VISUAL_MEMORY_ENABLED = False            # Toggle (disabled by default — requires OpenCLIP)
VISUAL_MEMORY_CLIP_MODEL = "ViT-B-32"   # OpenCLIP model variant
VISUAL_MEMORY_MAX_IMAGES = 3            # Max images per query
VISUAL_MEMORY_CAPTION_MODEL = None      # Vision LLM (None = default model)
VISUAL_MEMORY_FAISS_PATH = "data/visual_faiss.index"
VISUAL_MEMORY_DEDUP_ENABLED = True      # SHA-256 content dedup

# Tests: 60 tests across 3 files
# tests/unit/test_clip_manager.py — CLIPManager singleton, encode_image, encode_text
# tests/unit/test_visual_memory_store.py — dual storage, dedup, FAISS/ChromaDB search
# tests/unit/test_visual_retrieval.py — hybrid retrieval, prompt format, agentic tool
```

---

## Google OAuth2 + Calendar Integration **[NEW 2026-05]**

```python
# core/actions/google_auth.py — OAuth2 credential management
class GoogleAuthManager:
    """Manages Google OAuth2 credentials for Gmail and Calendar APIs.
    Handles installed-app OAuth2 flow, token persistence, automatic refresh,
    and scope-upgrade detection."""

    def __init__(client_id, client_secret, token_path="data/google_token.json", scopes=None): ...
    def is_authenticated -> bool:
        """True if valid token exists on disk (auto-refreshes expired tokens)."""
    def get_credentials() -> Optional[google.oauth2.credentials.Credentials]:
        """Return valid credentials or None (refreshes if expired)."""
    def has_scope(scope: str) -> bool:
        """Check if current token has a specific OAuth scope."""
    def authorize() -> Credentials:
        """Run browser-based OAuth consent flow. Persists token to disk."""

def get_google_auth() -> GoogleAuthManager | None:
    """Lazy singleton from config. Returns None if client_id/secret unconfigured."""

# Scopes: gmail.send, calendar.readonly, calendar.events
# One-time auth: python -m core.actions.google_auth


# core/actions/google_calendar.py — Read-only calendar fetch
async def fetch_upcoming_events(max_events=10, lookahead_days=7) -> List[Dict]:
    """Fetch upcoming events from Google Calendar API. 5-min cache.
    Returns: [{summary, start, end, all_day, location}, ...].
    Silent [] on disabled/unconfigured/error."""

def clear_cache() -> None:
    """Reset module-level event cache."""


# core/actions/google_calendar_create.py — Calendar event creation (write action)
async def create_calendar_event(proposal: ActionProposal) -> ActionResult:
    """Create a Google Calendar event via Calendar API POST.
    Requires calendar.events scope. Expects proposal.params:
    summary (str), start_time (str, ISO 8601), end_time (str, ISO 8601),
    description (optional), calendar_id (optional, default "primary"),
    time_zone (optional, default "America/Chicago"), location (optional)."""


# core/actions/email.py — Gmail API primary, SMTP fallback [REWRITTEN 2026-05]
async def send_email(proposal: ActionProposal) -> ActionResult:
    """Gmail API first → SMTP fallback only if Gmail unconfigured.
    No SMTP fallback after Gmail API attempt (prevents duplicate sends)."""
async def _try_gmail_send(proposal, recipient, message) -> ActionResult | None:
    """Returns ActionResult on attempt, None if Gmail not configured."""
async def _smtp_send(proposal, recipient, message) -> ActionResult:
    """SMTP fallback path."""


# core/actions/executors.py — Executor wiring
# _execute_calendar_create_event() now calls google_calendar_create.create_calendar_event()
# (was stub, now real implementation)


# Prompt sections (formatter.py):
# [UPCOMING SCHEDULE] — extracted schedule facts (work, class, exams) from fact_extractor
# [GOOGLE CALENDAR]   — real-time events from Google Calendar API via OAuth2

# Gatherer (gatherer_knowledge.py):
# get_google_calendar_events(max_events=10) — async wrapper around fetch_upcoming_events()

# Eval sections (section_registry.py): 31 entries total (was 29)
# upcoming_schedule (order 29), google_calendar (order 30)

# Config (app_config.py):
INTERNET_ACTIONS_GOOGLE_CLIENT_ID      # env: GOOGLE_CLIENT_ID
INTERNET_ACTIONS_GOOGLE_CLIENT_SECRET   # env: GOOGLE_CLIENT_SECRET
INTERNET_ACTIONS_GOOGLE_TOKEN_PATH = "data/google_token.json"
GOOGLE_CALENDAR_ENABLED = False         # env: GOOGLE_CALENDAR_ENABLED
GOOGLE_CALENDAR_MAX_EVENTS = 10
GOOGLE_CALENDAR_LOOKAHEAD_DAYS = 7

# Tests: tests/unit/test_google_auth.py, test_google_calendar.py, test_calendar_create.py,
#        test_calendar_prompt.py, test_gmail_send.py
```

---

## Backup B2 Upload **[NEW 2026-05]**

```bash
# scripts/backup_data.sh — Enhanced with Backblaze B2 cloud upload
# After local encrypted backup, uploads to B2 via rclone:
#   rclone copy "$BACKUP_FILE" "b2-daemon-crypt:backups/" --progress
# Prunes remote backups older than KEEP_DAYS:
#   rclone delete "b2-daemon-crypt:backups/" --min-age "${KEEP_DAYS}d"
# Requires: rclone configured with b2-daemon-crypt remote
```

---

## Retrieval Quality Benchmarks **[NEW 2026-02-17]**

```bash
# Run benchmarks (real embeddings, ~5s)
pytest tests/benchmarks/ -m benchmark -v

# Exclude benchmarks from regular test runs
pytest -m "not benchmark"
```

```python
# tests/benchmarks/ — Two suites: synth (93 seeds + 96 cases) + real (64 cases from production ChromaDB)
# conftest.py: session-scoped ChromaDB + CorpusManager + MockTimeManager fixtures;
#   real_benchmark_config, real_seeded_stores, real_retrieval_env for real-data suite
# retrieval_benchmark.py: BenchmarkResult + RetrievalBenchmark harness (recall@K, MRR)
#   Content-hash identity matching (_content_hash, _normalize_content)
#   Per-collection retrieval routing via retrieval_method field
# test_retrieval_quality.py: parametrized pytest cases + structural validation
#   TestRealDataRetrievalQuality + TestRealDataStructure test classes [NEW 2026-05-16]
# report_generator.py: markdown report grouped by intent type
# tests/fixtures/retrieval_benchmarks.yaml: synth seed memories + test case definitions
# tests/fixtures/retrieval_benchmarks_real.yaml: 64 real-data cases [NEW 2026-05-16]

# scripts/sample_real_benchmark.py — Sample real data from production ChromaDB [NEW 2026-05-16]
# scripts/benchmark_retrieval.py — Full benchmark runner [NEW 2026-05]
# Metrics: Recall@1/3/10, MRR (aggregate + per-intent), Precision@10,
#   intent classification accuracy, per-component retrieval latency (p50/p90/p95),
#   cold vs warm embedding latency, full prompt build latency
# History: data/benchmark_history.json (append-only, tracks runs over time)
# Docs: docs/BENCHMARK_METRICS.md (metric definitions + interpretation guide)
# Usage: python scripts/benchmark_retrieval.py

# Current scores (2026-05-16):
#   Synth: MRR=0.9149, R@1=0.8056, R@3=0.9236, R@topK=1.0000
#   Real:  MRR=0.8766, R@1=0.8413, R@3=0.8730, R@topK=1.0000
#   Combined: MRR=0.8970, R@1=0.8222, R@3=0.9000, R@topK=1.0000
```

---

## Quick Debug Commands

```bash
# Run full unit suite (either works — root conftest.py adds project to sys.path)
python -m pytest tests/unit/ -q
venv/bin/pytest tests/unit/ -q

# Check coverage
python -m pytest --cov=. --cov-report=term --ignore=tests/memory_test.py --ignore=tests/test_gated_prompt.py

# Run single test file
python -m pytest tests/unit/test_memory_coordinator_methods.py -v

# Check corpus size
wc -l data/corpus.json

# View recent conversations
tail -n 100 data/corpus.json | jq '.conversations[-5:]'

# Check ChromaDB collections
python -c "import chromadb; client = chromadb.PersistentClient('./chroma_db'); print([c.name for c in client.list_collections()])"

# Test prompt builder
python -c "from core.prompt.builder import UnifiedPromptBuilder; pb = UnifiedPromptBuilder(...); print(pb.build_prompt('test query', []))"
```

---

**End of Quick Reference**

**Last verified**: 2026-05-27

This document provides instant lookup for critical functions and patterns.
