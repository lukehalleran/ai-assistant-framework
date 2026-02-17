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
# memory/memory_coordinator.py — Thin orchestrator (~498 lines)
# All methods delegate to modular components. No inline logic.
class MemoryCoordinator:
    def __init__(self, ...):
        self._retriever = MemoryRetriever(...)    # Retrieval operations
        self._storage = MemoryStorage(...)        # Persistence operations
        self._shutdown = ShutdownProcessor(...)   # Session-end processing
        self.scorer = MemoryScorer(...)           # Scoring and ranking
        self.thread_manager = ThreadManager(...)  # Thread tracking

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

    def rank_memories(memories, query, current_topic=None, weight_overrides=None) -> List[Dict]:
        """
        Score = weights['relevance']*rel + weights['recency']*recency + weights['truth']*truth
              + weights['importance']*importance + weights['continuity']*continuity
              + topic_match + structure + bonuses - penalties
        weight_overrides: per-intent SCORE_WEIGHTS override (from IntentClassifier)
        Falls back to _intent_weight_overrides if no explicit param.

        Temporal-aware decay [NEW 2026-02-17]:
        If '_temporal_anchor_hours' in weight_overrides (set by IntentClassifier for TEMPORAL_RECALL),
        recency uses gentle curve (1.0→0.7) within window, standard decay outside.
        Temporal anchor takes priority over time_manager and standard fallback.
        """

    def calculate_truth_score(query, response) -> float:
        """Base 0.5 + length/question/confirmation bonuses, capped at 1.0"""

    def calculate_importance_score(content) -> float:
        """Base 0.5 + length/question/keyword bonuses, capped at 1.0"""


# memory/shutdown_processor.py — Session-end processing [ENHANCED 2026-02-13]
class ShutdownProcessor:
    def __init__(..., memory_coordinator=None):
        """memory_coordinator enables pipeline-enriched proposal generation."""

    async def process_shutdown_memory(session_conversations=None):
        """Steps: 1) block summaries, 2) session facts, 3) LLM facts,
        4) procedural skills, 5) code proposals, 6) cross-dedup (dry_run=True only),
        7) user profile updates"""

    async def _run_cross_collection_dedup():
        """Dry-run only — logs findings but NEVER deletes. Guarded to run once per process."""

    async def run_shutdown_reflection(session_conversations=None, session_summaries=None) -> bool:
        """LLM reflection generation + ChromaDB storage"""


# memory/cross_deduplicator.py — Cross-collection dedup [NEW 2026-02-13]
class CrossCollectionDeduplicator:
    def __init__(chroma_store):
        """Thresholds from CROSS_DEDUP_* config. Skips protected collections."""

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

        # Wiki context if relevant
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
        """Extract <thinking>...</thinking> and final answer"""
        # Returns (thinking_content, final_answer)
        # If no thinking block: ("", full_response)

    @staticmethod
    def strip_reflection_blocks(response: str) -> str:
        """Remove <reflect>...</reflect> and [SYSTEM QUALITY REFLECTION]..."""

    @staticmethod
    def strip_xml_wrappers(text: str) -> str:
        """Remove <result>...</result>, <answer>...</answer>, <final>...</final>"""

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
        """Stream LLM response chunks"""
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
    # Collections (10 total): conversations, summaries, wiki_knowledge, facts, reflections, obsidian_notes, reference_docs, procedural, procedural_skills, proposals

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
FILE_UPLOAD_ALLOWED_EXTENSIONS = ['.txt', '.docx', '.csv', '.py', '.png', '.jpg', '.jpeg', '.gif', '.webp']
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


# memory/fact_extractor.py
class FactExtractor:
    def extract_facts(conversation: Dict) -> List[Dict]:
        """
        1. Try regex patterns (fast):
           - "X is Y"
           - "X does Y"
           - Entity definitions
        2. Fallback to LLM if complex
        """
        text = f"{conversation['query']} {conversation['response']}"
        facts = []

        # Regex patterns
        is_pattern = re.findall(r'(\w+(?:\s+\w+)?)\s+is\s+(.+?)(?:\.|$)', text)
        for entity, definition in is_pattern:
            facts.append({'entity': entity, 'fact': f"is {definition}", 'confidence': 0.8})

        # LLM fallback if needed
        if not facts and self.use_llm:
            facts = await self.llm_fact_extractor.extract(text)

        return facts
```

---

## Configuration Constants

```python
# config/app_config.py

# Paths
CORPUS_FILE = os.getenv("CORPUS_FILE", "./data/corpus.json")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

# Token budgets
PROMPT_TOKEN_BUDGET = int(os.getenv("PROMPT_TOKEN_BUDGET", "2048"))
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

# Summarization
SUMMARY_EVERY_N = int(os.getenv("SUMMARY_EVERY_N", "20"))
SUMMARIZE_AT_SHUTDOWN_ONLY = True

# Models
MODEL_DEFAULT = os.getenv("LLM_ALIAS", "gpt-4o-mini")
MODEL_SUMMARY = os.getenv("LLM_SUMMARY_ALIAS", "gpt-4o-mini")
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

# Final memory score
score = (
    0.35 * relevance +
    0.25 * recency +
    0.20 * truth +
    0.05 * importance +
    0.10 * continuity +
    structure_score +
    anchor_bonus +
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
[USER'S PERSONAL NOTES]        # Obsidian vault notes (hybrid retrieval) + images for multimodal [ENHANCED 2026-01-30]
[USER UPLOADED ITEMS]          # Persisted user file/image uploads from reference_docs collection [NEW 2026-02-10]
[DAEMON DOCUMENTATION]         # Self-knowledge: architecture docs, PROJECT_SKELETON
[PROJECT COMMIT HISTORY]       # Git commit history (procedural memory)
[ADAPTIVE WORKFLOWS]           # Reusable problem-solving patterns (WHEN/THEN)
[PROPOSED FEATURES]            # Code proposals surfaced for project-related queries [NEW 2026-02-09]
[WEB SEARCH RESULTS]           # Real-time Tavily results
[RELEVANT INFORMATION]         # Wikipedia chunks
[TIME CONTEXT]                 # Current datetime
[TEMPORAL GROUNDING]           # Synthesized life context (daily/weekly notes) [NEW 2026-01-17]
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

# Narrative Context (Temporal Grounding) [NEW 2026-01-17]
python main.py refresh-narrative             # Regenerate life state from daily/weekly notes

# Daemon Documentation (self-knowledge)
python main.py upload-doc <file> [title]     # Upload doc to [DAEMON DOCUMENTATION]
python main.py list-docs                     # List uploaded docs
python main.py delete-doc <title>            # Delete specific doc
python main.py clear-docs                    # Clear all docs

# Obsidian Vault (personal notes)
python main.py embed-vault                   # Index vault to ChromaDB
python main.py embed-vault --force           # Force full re-index
python main.py vault-stats                   # Show indexed chunk count
python main.py clear-vault                   # Clear obsidian_notes collection

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

## Narrative Context (Temporal Grounding) [NEW 2026-01-17]

```python
# memory/memory_consolidator.py
class MemoryConsolidator:
    async def generate_narrative_context(weeklies=None, monthlies=None) -> str:
        """
        Synthesize daily/weekly notes into 'Current Life State' narrative.

        Sources (hybrid):
        1. Obsidian weekly summaries (Week */Week * Summary.md) - 2 max
        2. Obsidian daily notes (Week */*Daily Note.md) - 7 max
        3. Corpus summaries (fallback)

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

## Agentic Search & Tools [NEW 2026-01-22]

```python
# core/agentic/controller.py
class AgenticSearchController:
    """ReAct loop: Reason → Act (search/compute) → Observe → repeat until done"""

    async def execute_search(query: str, max_turns: int = 8) -> AsyncIterator[AgenticEvent]:
        """
        Main loop:
        1. Build prompt with gathered context + tool instructions
        2. Call LLM for decision (search/wolfram/done)
        3. Execute tool if requested, add result to context
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

            elif decision.is_done:
                yield AgenticEvent(type="done", response=final_response)
                break


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
    is_done: bool = False
    done_reason: Optional[str] = None
    wants_answer: bool = False
    partial_response: Optional[str] = None


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


# Protocol handlers (core/agentic/protocols.py)
# XMLMarkerHandler - for local models: <search>query</search>, <wolfram>query</wolfram>, <python>code</python>
# NativeToolsHandler - for API models: OpenAI/Anthropic function calling
```

### Agentic Config Constants

```python
# config/app_config.py

# Web Search (Tavily)
WEB_SEARCH_ENABLED = True
WEB_SEARCH_API_KEY = os.getenv("TAVILY_API_KEY", "")
WEB_SEARCH_DAILY_CREDIT_LIMIT = 100
WEB_SEARCH_CACHE_TTL_HOURS = 72

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
```

### Tool Invocation (Prompt Injection)

```python
# Added to system prompt for agentic mode (types.py AGENTIC_SYSTEM_PROMPT_INJECTION):
"""
Available Tools:
1. <search>query</search> - Web search for current events, facts, data
2. <wolfram>query</wolfram> - Computation, math, science, conversions
3. <python purpose="...">code</python> - Execute Python code (NumPy, Pandas, SciPy, SymPy available) [NEW]
4. <done>reason</done> - Signal task complete

Use Python for: multi-step computation, data analysis, visualization, custom algorithms
Use Wolfram for: single-expression calculations, unit conversions, scientific data, equations
Use search for: current events, recent news, real-time data, general facts
"""
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

This document is ~800 lines → ~5K tokens, providing instant lookup for critical functions and patterns.
