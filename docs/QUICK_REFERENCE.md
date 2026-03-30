# Daemon RAG - Quick Reference API

**Ultra-compressed reference for LLM context windows. Function signatures and core logic only.**

---

## Core Entry Point

```python
# core/orchestrator.py
class DaemonOrchestrator:
    async def handle_request(query: str, thread_id: str = None) -> str:
        """Main entry: query → memories → prompt → LLM → response"""
        topics = self.topic_manager.extract_topics(query)
        memories = await self.memory_coordinator.get_memories(query, topics=topics, limit=30)
        prompt = self.prompt_builder.build_prompt(query, memories, topics)
        response = await self.response_generator.generate_response_stream(prompt)
        await self.memory_coordinator.store_interaction(query, response, tags=topics)
        return response
```

---

## Context Pipeline **[NEW 2026-01-23]**

```python
# core/context_pipeline.py
class ContextPipeline:
    async def build(user_input: str, files: List = None, use_raw_mode: bool = False) -> ContextResult:
        """
        Pipeline: topics → tone → files → heavy_check → intent → rewrite → stm → identity → thread
        Returns ContextResult with all processed components.

        Stages:
        1. Topic extraction (TopicManager)
        2. Tone detection (analyze_emotional_context)
        3. File processing (FileProcessor)
        4. Heavy topic check (QueryChecker)
        4.5. Intent classification (IntentClassifier, regex-first, no LLM) [NEW 2026-02-15]
        5. Query rewriting (LLM)
        6. STM analysis (STMAnalyzer)
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
# memory/memory_coordinator.py — Thin orchestrator (~498 lines, plus new component wiring)
# All methods delegate to modular components. No inline logic.
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

    async def get_skills(query, limit=5) -> List[Dict]:
        """Hybrid retrieval (1/3 recent + 2/3 semantic) from procedural_skills collection."""


# memory/memory_scorer.py — Scoring and ranking
class MemoryScorer:
    _intent_weight_overrides: Optional[Dict[str, float]] = None  # Set by PromptBuilder [NEW 2026-02-15]
    _graph_memory = None          # Set by PromptBuilder for graph-boosted scoring [NEW 2026-03]
    _entity_resolver = None       # Set by PromptBuilder for entity extraction [NEW 2026-03]

    def rank_memories(memories, query, current_topic=None, weight_overrides=None) -> List[Dict]:
        """
        Score = weights['relevance']*rel + weights['recency']*recency + weights['truth']*truth
              + weights['importance']*importance + weights['continuity']*continuity
              + topic_match + structure + bonuses - penalties
        weight_overrides: per-intent SCORE_WEIGHTS override (from IntentClassifier)
        Falls back to _intent_weight_overrides if no explicit param.
        + staleness_penalty  # staleness_ratio * STALENESS_WEIGHT, 2x at 80%+, capped 0.4 [NEW 2026-03-25]

        Temporal-aware decay [NEW 2026-02-17]:
        If '_temporal_anchor_hours' in weight_overrides (set by IntentClassifier for TEMPORAL_RECALL),
        recency uses gentle curve (1.0→0.7) within window, standard decay outside.
        Temporal anchor takes priority over time_manager and standard fallback.
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
        """Steps: 1) block summaries, 2) session facts, 3) LLM facts,
        4) procedural skills, 5) code proposals, 6) cross-dedup (dry_run=True only),
        6.5a) open thread extraction + resolution detection [NEW 2026-03-23],
        6.5b) implementation tracking (lightweight file-existence check) [NEW 2026-03-24],
        6.8) synthesis dreaming — cross-store candidate generation + filter pipeline [NEW 2026-03-28],
        7) user profile updates (user-only facts; entity facts stay in ChromaDB only).
        After summary storage: extract claims → register in ClaimIndex → add staleness metadata [NEW 2026-03-25]"""

    async def _run_cross_collection_dedup():
        """Dry-run only — logs findings but NEVER deletes. Guarded to run once per process."""

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

# Integration flow:
# ContextPipeline Stage 4.5 → classify() → IntentResult on ContextResult
# ContextPipeline Stage 6b  → refine_with_stm() after STM analysis
# PromptBuilder             → extracts retrieval_overrides + weight_overrides
# MemoryScorer              → uses weight_overrides in rank_memories()

# Config (app_config.py):
INTENT_ENABLED = True
INTENT_STM_REFINEMENT_THRESHOLD = 0.50
```

---

## Gating System

```python
# processing/gate_system.py
class MultiStageGateSystem:
    async def filter_memories(query: str, memories: List[Dict], k: int = 20) -> List[Dict]:
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
# core/prompt/builder.py
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
        """Extract <thinking>...</thinking> and final answer.
        Also handles <output>...</output> wrappers (pre-output = thinking, inner = answer)."""
        # Returns (thinking_content, final_answer)
        # If no thinking block: ("", full_response)

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
    # Collections (11 total): conversations, summaries, wiki_knowledge, facts, reflections, obsidian_notes, reference_docs, procedural, procedural_skills, proposals, threads

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
```

---

## Code Proposals System **[NEW 2026-02-05]**

```python
# memory/code_proposal.py — Pydantic data models
class CodeProposal(BaseModel):
    id: str                        # UUID
    title: str                     # Short title
    proposal_type: ProposalType    # feature|refactor|bugfix|test|docs|infra
    status: ProposalStatus         # PENDING|APPROVED|REJECTED|COMPLETED|FAILED
    source: ProposalSource         # GOAL_DIRECTED|CONVERSATION|MANUAL
    priority: int                  # 1-10 (LLM-assigned)
    reasoning: str                 # Why this change is needed
    description: str               # Detailed description
    implementation_steps: List[ImplementationStep]  # Ordered steps
    affected_files: List[str]      # File paths
    tags: List[str]                # Keywords
    estimated_complexity: str      # low|medium|high

    def to_metadata() -> dict:     # Flat dict for ChromaDB
    def from_metadata(meta) -> CodeProposal:  # Reconstruct from ChromaDB


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
        """

# gui/handlers.py — Upload persistence (fire-and-forget background task)
async def _persist_uploads(orchestrator, files_result):
    """Store uploads to ChromaDB reference_docs with type='user_upload' metadata.
    Images also store is_image, media_type, image_path metadata.
    Upload images injected into note_images for multimodal model use."""

# core/prompt/context_gatherer.py — Retrieval
async def get_user_uploads(query, limit=5) -> List[Dict]:
    """Fetch from reference_docs, filter to type='user_upload' only."""
async def get_reference_docs(query, limit) -> List[Dict]:
    """Fetch from reference_docs, filter OUT type='user_upload'."""

# Config (app_config.py):
FILE_UPLOAD_ALLOWED_EXTENSIONS = ['.txt', '.docx', '.csv', '.py', '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.webp']  # .pdf added [2026-03-10]
FILE_UPLOAD_IMAGE_DIR = "data/uploads"
PROMPT_MAX_USER_UPLOADS = 5
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


# memory/fact_extractor.py [ENHANCED 2026-03: entity facts]
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
        """

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

# Retrieval (context_gatherer.py):
# get_graph_context() → BFS from query entities, returns natural language sentences
# _expand_query_with_graph() → calls rank_expansion_candidates() (connectivity-ranked, junk-filtered)

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
    def is_stale(stale_days=14) -> bool:
        """True if no reference in stale_days."""
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


# memory/thread_extractor.py — LLM-based extraction
class ThreadExtractor:
    def __init__(model_manager):
        """Two-phase LLM approach: extraction + resolution detection."""

    async def extract_new_threads(session_conversations) -> List[OpenThread]:
        """Extract open loops from session. Few-shot prompt, temp=0.0, cap 5 per session.
        Thread types: commitment, deadline, unfinished, question."""

    async def detect_resolutions(session_conversations, open_threads) -> List[Tuple[str, str]]:
        """Detect which existing threads were resolved. Returns (thread_id, resolution) tuples.
        Only marks resolved with clear evidence, NOT just because thread wasn't mentioned."""


# Integration:
# - Shutdown: step 6.5 in process_shutdown_memory() — extract new threads + resolve existing
# - Prompt: [UNRESOLVED THREADS] section (after [KNOWLEDGE GRAPH])
# - Collection: 'threads' in ChromaDB (11 total collections)
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
# context_gatherer.py → get_proactive_insights() → context_surfacer.generate_insights()
# Prompt: [PROACTIVE INSIGHTS] after [UNRESOLVED THREADS]
# Parallel task in build_prompt()

# Config:
PROACTIVE_SURFACING_ENABLED = True
PROACTIVE_SURFACING_COOLDOWN_HOURS = 72
PROACTIVE_SURFACING_MAX_INSIGHTS = 3
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
# context_gatherer.py:get_codebase_changes() — git log/diff/status since last session
# builder.py:_build_feature_inventory() — compact 4-line enabled features summary

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
PROMPT_TOKEN_BUDGET_DEFAULT = 40000   # API models fallback
PROMPT_TOKEN_BUDGET_LOCAL = 12000     # Local model cap
PROMPT_TOKEN_BUDGET_FLOOR = 8000      # Minimum budget
PROMPT_TOKEN_BUDGET_CEILING = 60000   # Maximum budget
PROMPT_MIN_RECENT_FLOOR = 5           # Min recent conversations guaranteed post-budget [NEW 2026-03-28]
PROMPT_MAX_MEMS = int(os.getenv("PROMPT_MAX_MEMS", "30"))

# Decay & scoring
RECENCY_DECAY_RATE = float(os.getenv("RECENCY_DECAY_RATE", "0.05"))
TRUTH_SCORE_UPDATE_RATE = float(os.getenv("TRUTH_SCORE_UPDATE_RATE", "0.02"))
TRUTH_SCORE_MAX = 1.0

# Gating threshold
GATE_COSINE_THRESHOLD = float(os.getenv("GATE_COSINE_THRESHOLD", "0.45"))

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

# Proactive Context Surfacing [NEW 2026-03-24]
PROACTIVE_SURFACING_ENABLED = True
PROACTIVE_SURFACING_COOLDOWN_HOURS = 72
PROACTIVE_SURFACING_MAX_INSIGHTS = 3

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

# LLM Compression [NEW 2026-03-26]
LLM_COMPRESSION_ENABLED = True         # Use LLM to compress heavily oversized items
LLM_COMPRESSION_MODEL = "gpt-4o-mini"  # Model for compression calls
LLM_COMPRESSION_TIMEOUT = 3.0          # Per-item timeout (seconds)
LLM_COMPRESSION_RATIO_THRESHOLD = 3.0  # Only compress items >= 3x over token limit
LLM_COMPRESSION_MAX_BATCH = 8          # Max items to compress per request (cost guard)

# Citation & Provenance [NEW 2026-03-26]
PROVENANCE_ENABLED = True              # Toggle provenance metadata on interactions
PROVENANCE_THINKING_MAX_CHARS = 4000   # Max chars for thinking block in metadata

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

# 5. prompt_builder.py - add to prompt assembly
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

# Temporal-aware recency decay (TEMPORAL_RECALL only) [NEW 2026-02-17]
# Within window (e.g., 168h for "last week"):
recency = 1.0 - (age_hours / temporal_anchor) * 0.3  # gentle 1.0→0.7
# Outside window:
recency = 0.7 / (1.0 + 0.05 * (age_hours - temporal_anchor))

# Truth boost from access
truth = min(1.0, base_truth + 0.02 * access_count)

# Continuity score
continuity = (0.1 if age < 10min else 0) + (0.3 * token_overlap_ratio)

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
# Prompt section hierarchy (in builder.py _assemble_prompt):
[RECENT CONVERSATION]          # Historical context
[RELEVANT MEMORIES]            # Scored episodic memories
[USER PROFILE]                 # Categorized user facts
[SUMMARIES]                    # Consolidated conversation blocks
[REFLECTIONS]                  # Session reflections
[DREAMS]                       # Dream memories (if enabled)
[USER'S PERSONAL NOTES]        # Obsidian vault notes; post-filtered by PERSONAL_NOTES_GATE_THRESHOLD (0.30); headers show [relevance: X.XX] [ENHANCED 2026-03-20]
[USER UPLOADED ITEMS]          # Persisted user file/image uploads from reference_docs collection [NEW 2026-02-10]
[DAEMON DOCUMENTATION]         # Self-knowledge: architecture docs, PROJECT_SKELETON
[PROJECT COMMIT HISTORY]       # Git commit history (procedural memory)
[ADAPTIVE WORKFLOWS]           # Reusable problem-solving patterns (WHEN/THEN)
[PROPOSED FEATURES]            # Code proposals surfaced for project-related queries [NEW 2026-02-09]
[KNOWLEDGE GRAPH]              # Graph traversal: related entities as natural language [NEW 2026-03]
[UNRESOLVED THREADS]           # Open commitments, deadlines, unfinished topics [NEW 2026-03-23]
[PROACTIVE INSIGHTS]           # Cross-domain insights from knowledge graph [NEW 2026-03-24]
[ACTIVE FEATURES]              # Feature inventory (always present) [NEW 2026-03-24]
[CODEBASE CHANGES SINCE LAST SESSION]  # Git changes (first message only) [NEW 2026-03-24]
[WEB SEARCH RESULTS]           # Real-time Tavily results
[RELEVANT INFORMATION]         # Wikipedia chunks
[TIME CONTEXT]                 # Current datetime
[TEMPORAL GROUNDING]           # Synthesized life context (monthly/weekly/daily notes) [ENHANCED 2026-03-10]
[STM SUMMARY]                  # Short-term memory analysis
[CURRENT USER QUERY]           # The actual query to respond to
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

# Tag Vocabulary (100+ tags across 6 categories):
# 1. Life domains (17): work, study, health, exercise, sleep, social, family, ...
# 2. Activities (18): coding, programming, learning, reading, writing, debugging, ...
# 3. Emotions (23): stress, anxiety, happy, motivated, tired, focused, overwhelmed, ...
# 4. Productivity (15): productive, deep-work, flow-state, breakthrough, blocked, ...
# 5. Topics (23): ai, programming, python, math, philosophy, psychology, ...
# 6. Meta (16): crisis, decision, reflection, planning, insight, follow-up, ...

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

## Agentic Search & Tools [NEW 2026-01-22, ENHANCED 2026-03-23]

**Agentic Gate** (gui/handlers.py) — 3-tier trigger before ReAct loop:
1. **Keyword heuristic** (0ms): computation keywords OR memory keywords ("do you remember", "my notes", etc.)
2. **Entity match** (0ms): `extract_graph_entities()` checks query against knowledge graph aliases; known entities (e.g. "Flapjack") trigger memory search
3. **LLM fallback**: piggybacks on `analyze_for_web_search_llm()`; `needs_memory_search` field in `WebSearchDecision` catches structural recall queries

Casual skip filter (< 5 words, "thanks", etc.) only applies when no keyword/entity trigger fired.
`skip_initial_search=True` for computation and memory queries.

```python
# core/agentic/controller.py
class AgenticSearchController:
    """ReAct loop: Reason → Act (search/compute) → Observe → repeat until done"""

    async def execute_search(query: str, max_turns: int = 8) -> AsyncIterator[AgenticEvent]:
        """
        Main loop:
        1. Build prompt with gathered context + tool instructions
        1b. Compute context inventory (summarize RAG-gathered sections) [ENHANCED 2026-03-23]
        2. Call LLM for decision (search/wolfram/memory/git_stats/done)
        3. Execute tool if requested, add result to context
        3b. Track memory_search_counts per collection (diversity enforcement) [ENHANCED 2026-03-23]
        4. Repeat until done or max_turns reached
        5. Yield AgenticEvent for each stage (thinking/searching/computed/done)
        """
        context = []
        for turn in range(max_turns):
            # Get LLM decision via protocol handler
            decision = await self.protocol_handler.parse_response(llm_response)

            if decision.wants_search:
                results = await self.web_search_manager.search(decision.search_query)
                context.append(self._format_search_context(results))
                yield AgenticEvent(type="searched", data=results)

            elif decision.wants_wolfram:
                result = await self._execute_wolfram(decision.wolfram_query)
                context.append(self._format_wolfram_context(result))
                yield AgenticEvent(type="computed", data=result)

            elif decision.wants_memory_search:  # [ENHANCED 2026-03-23]
                results = await self._execute_memory_search(decision.memory_query, decision.memory_collection)
                context.append(self._format_memory_context(results))
                yield AgenticEvent(type="memory_searched", data=results)

            elif decision.wants_memory_expand:  # [NEW 2026-03-26]
                result = await self._execute_memory_expand(decision.expand_memory_id, decision.expand_window, decision.expand_collection)
                context.append(self._format_expand_context(result))
                yield AgenticEvent(type="memory_expanded", data=result)

            elif decision.is_done:
                yield AgenticEvent(type="done", response=final_response)
                break

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

# core/git_stats_manager.py [NEW 2026-03-29]
class GitStatsManager:
    """Read-only git repository stats for the agentic loop"""

    def is_available() -> bool:
        """True if cwd is inside a git repo"""

    def execute_query(query: str) -> Dict[str, Any]:
        """Parse natural-language query → safe git subcommand → structured result dict"""

    def format_for_prompt(result: Dict) -> str:
        """Format result dict for LLM context injection"""

# Safety: only ALLOWED_SUBCOMMANDS may execute (log, shortlog, diff, status, branch, etc.)
# Keyword-based intent parsing maps queries to git commands
# Temporal phrase extraction: "this week", "today", "last 30 days" → --since dates


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


# knowledge/sandbox_manager.py [NEW 2026-01-22]
class SandboxManager:
    """E2B Code Sandbox for secure Python execution"""

    async def execute_code(code: str) -> SandboxResult:
        """Execute code in ephemeral sandbox"""

    async def create_session() -> PersistentSession:
        """Create persistent session (variables survive across calls)"""

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


# Protocol handlers (core/agentic/protocols.py)
# XMLMarkerHandler - for local models: <search>, <wolfram>, <python>, <memory>, <expand_memory>, <file_read>, <file_grep>, <file_list>, <git_stats>, <done>
# NativeToolsHandler - for API models: OpenAI/Anthropic function calling (10 tool definitions)
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
10. <done>reason</done> - Signal task complete

Use Python for: multi-step computation, data analysis, visualization, custom algorithms
Use Wolfram for: single-expression calculations, unit conversions, scientific data, equations
Use search for: current events, recent news, real-time data, general facts
Use memory for: user facts, past conversations, personal notes, project history
Use expand_memory for: see surrounding conversation turns or drill into summaries
Use file_read/file_grep/file_list for: reading project files, searching code, listing directories
Use git_stats for: commit counts, recent commits, contributors, files changed, branch activity
"""
```

---

## Personality System (File-Based) **[NEW 2026-03-26]**

```python
# config/app_config.py — Replaces deleted PersonalityManager (JSON configs)
# System prompt = personality text + operating principles, composed in orchestrator._build_system_prompt()

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
    _last_session: AgenticSearchSession   # Accessible after execute_search()
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

# context_gatherer.py fix: recent conversations now build content from Q/A
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

# knowledge/synthesis_filter.py — 8-stage async pipeline
# Module-level helpers:
#   _extract_similarity(results) -> float   — converts query_collection result to 0-1 similarity
#   _compute_template_similarity(claim) -> float — regex-based generic bridge detection
#   _GENERIC_TEMPLATES — compiled regex patterns for vacuous bridge claims
#   _GENERIC_TOKENS — frozenset of generic buzzwords
class SynthesisFilter:
    def __init__(self, chroma_store, model_manager, synthesis_memory=None,
                 wiki_collection="wiki_knowledge"): ...
    async def process_candidate(candidate) -> SynthesisResult:  # Full pipeline, auto-stores accepted
    async def process_batch(candidates) -> dict:
        # Returns: {total, accepted, rejected, rejection_breakdown, accepted_results, avg_stage_times_ms}
    async def _factual_skeptic_pass(result, claim, a, b) -> bool:  # Pass 2 of Stage 5
    @staticmethod
    def _parse_coherence_level(response_text) -> CoherenceLevel:  # Extracted from inline parsing

# Pipeline stages (cheap → expensive):
# 0: text_sanity      — min tokens, verb check, repetition ratio (~0ms)
# 1: domain_crossing   — min 2 distinct domains (~1ms)
# 2: semantic_distance  — endpoint distance in [0.20, 0.90] (~5ms)
# 3: novelty_external   — 3 sub-checks (~10ms):
#      a) claim similarity: full claim vs wiki (hard gate at 0.80)
#      b) co-occurrence: bare "concept_a concept_b" vs wiki (hard gate at 0.75)
#      c) template specificity: regex generic bridge pattern detection
# 4: novelty_internal   — synthesis memory; new paths pass as convergence (~10ms)
# 5: coherence_judge    — two-pass LLM: Pass 1 structural coherence (4-tier rating),
#                          Pass 2 factual skeptic (MODERATE only, binary PASS/FAIL) (~500ms-2s)
# 6: composite_scoring  — weighted 4-signal novelty composite ≥ 0.40 threshold
# 7: storage            — accepted → synthesis_results collection

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
SYNTHESIS_NOVELTY_KNOWN_THRESHOLD = 0.80
SYNTHESIS_NOVELTY_ADJACENT_THRESHOLD = 0.50
SYNTHESIS_COOCCURRENCE_KNOWN_THRESHOLD = 0.75  # Stage 3b: co-occurrence hard gate
SYNTHESIS_MEMORY_SIMILARITY_THRESHOLD = 0.85
SYNTHESIS_COHERENCE_MODEL = "sonnet-4.5"
SYNTHESIS_COHERENCE_MIN_LEVEL = "MODERATE"
SYNTHESIS_WEIGHT_COHERENCE = 0.30
SYNTHESIS_WEIGHT_NOVELTY = 0.40
SYNTHESIS_WEIGHT_DISTANCE = 0.15
SYNTHESIS_WEIGHT_STRUCTURAL = 0.15
SYNTHESIS_NOVELTY_W_CLAIM = 0.25             # 4-signal novelty weights
SYNTHESIS_NOVELTY_W_COOCCURRENCE = 0.30
SYNTHESIS_NOVELTY_W_SPECIFICITY = 0.25
SYNTHESIS_NOVELTY_W_INTERNAL = 0.20
SYNTHESIS_COMPOSITE_MIN_SCORE = 0.40
SYNTHESIS_CONVERGENCE_STRONG_PATHS = 3
SYNTHESIS_CONVERGENCE_STRONG_SOURCES = 2
SYNTHESIS_DEFAULT_BATCH_SIZE = 100
SYNTHESIS_LOG_ALL_REJECTIONS = True

# YAML (config.yaml):
# synthesis:
#   enabled: true
#   batch_size: 100
#   distance_min: 0.20
#   distance_max: 0.90
#   cooccurrence_known_threshold: 0.75
#   coherence_model: gpt-4o-mini
#   coherence_min_level: MODERATE
#   weights: {coherence: 0.30, novelty: 0.40, distance: 0.15, structural: 0.15}
#   novelty_weights: {claim: 0.25, cooccurrence: 0.30, specificity: 0.25, internal: 0.20}
#   composite_min_score: 0.40
```

---

## Synthesis Generator (Cross-Store Candidate Generation) **[NEW 2026-03-28]**

```python
# knowledge/synthesis_generator.py — Cross-store sampling + LLM bridge articulation
class SynthesisGenerator:
    def __init__(self, chroma_store, model_manager, graph_memory=None, entity_resolver=None): ...
    async def generate_candidates(count=5) -> List[SynthesisCandidate]:
        # 1. Sample personal entities from facts collection (broad query seeds)
        # 2. Sample wiki articles from wiki_knowledge collection
        # 3. Form cross-domain pairs (deduplicated, domain-classified)
        # 4. Parallel LLM bridge articulation (semaphore-limited concurrency)
        # 5. Package as SynthesisCandidate objects for filter pipeline
    def _sample_personal_entities(n) -> List[Dict]:  # 12 query seeds, shuffled
    def _sample_wiki_articles(n) -> List[Dict]:       # 12 query seeds, shuffled
    def _classify_domain(item) -> str:                # categorize_relation() or keyword heuristics
    async def _articulate_bridge(concept_a, concept_b, domain_a, domain_b, ctx_a, ctx_b) -> Optional[str]:
        # LLM call; returns None on NO_CONNECTION or <5 words
    def _compute_endpoint_distance(concept_a, concept_b) -> float:
        # Graph shortest path (normalized to [0.15, 1.0]) or default 0.55
    def get_sampling_stats() -> dict:                 # {facts_count, wiki_count, graph_nodes, graph_edges}

# Integration:
# - Shutdown: step 6.8 in process_shutdown_memory() — after threads, before graph save
# - Pipeline: generate_candidates() → SynthesisFilter.process_batch() → SynthesisMemory
# - Graph sparsity guard: skips if graph < SYNTHESIS_GENERATOR_MIN_GRAPH_NODES nodes

# Config (app_config.py):
SYNTHESIS_GENERATOR_ENABLED = True
SYNTHESIS_GENERATOR_CANDIDATES_PER_SESSION = 5
SYNTHESIS_GENERATOR_LLM_CONCURRENCY = 5
SYNTHESIS_GENERATOR_MIN_GRAPH_NODES = 20

# YAML (config.yaml):
# synthesis_generator:
#   enabled: true
#   candidates_per_session: 5
#   llm_concurrency: 5
#   min_graph_nodes: 20

# Tests: 18 unit tests in tests/unit/test_synthesis_generator.py
# Calibration: 6 tests in tests/test_synthesis_calibration.py
# Fixtures: tests/fixtures/calibration_candidates.json (54 labeled candidates, 7 tiers)
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
# tests/benchmarks/ — 30 seed memories + 19 test cases across all 9 intent types
# conftest.py: session-scoped ChromaDB + CorpusManager + MockTimeManager fixtures
# retrieval_benchmark.py: BenchmarkResult + RetrievalBenchmark harness (recall@K, MRR)
# test_retrieval_quality.py: parametrized pytest cases + structural validation
# report_generator.py: markdown report grouped by intent type
# tests/fixtures/retrieval_benchmarks.yaml: seed memories + test case definitions
```

---

## Quick Debug Commands

```bash
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

This document is ~2,070 lines → ~14K tokens, providing instant lookup for critical functions and patterns.
