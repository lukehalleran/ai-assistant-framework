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

## Memory Operations

```python
# memory/memory_coordinator.py
class MemoryCoordinator:
    # Retrieval
    async def get_memories(query: str, topics: List[str] = None, limit: int = 30) -> List[Dict]:
        """
        1. Get recent from corpus (last 10)
        2. Query 5 ChromaDB collections in parallel
        3. Gate with MultiStageGateSystem
        4. Rank by _rank_memories()
        5. Return top K
        """
        recent = self.corpus_manager.get_recent_memories(10)
        chroma_results = await asyncio.gather(*[
            self.chroma_store.query(query, col, limit) for col in COLLECTIONS
        ])
        combined = recent + flatten(chroma_results)
        gated = await self.gate_system.filter_memories(query, combined, k=50)
        ranked = self._rank_memories(gated, query)
        return ranked[:limit]

    # Storage
    async def store_interaction(query: str, response: str, tags: List[str] = None):
        """Add to corpus JSON + ChromaDB with embeddings"""
        memory = {
            'id': str(uuid.uuid4()),
            'query': query,
            'response': response,
            'timestamp': datetime.now(),
            'truth_score': self._calculate_truth_score(query, response),
            'importance_score': self._calculate_importance_score(f"{query} {response}"),
            'tags': tags or [],
            'memory_type': 'episodic'
        }
        self.corpus_manager.add_memory(memory, MemoryType.EPISODIC)
        await self.chroma_store.add_memory(f"{query} {response}", memory, 'episodic')

    # Ranking algorithm
    def _rank_memories(memories: List[Dict], query: str) -> List[Dict]:
        """
        Score = 0.35*rel + 0.25*recency + 0.20*truth + 0.05*importance + 0.10*continuity + 0.15*structure + bonuses - penalties

        Components:
        - Relevance: base + collection boost
        - Recency: 1.0 / (1.0 + 0.05 * age_hours)
        - Truth: base + (0.02 * access_count), capped at 1.0
        - Importance: from _calculate_importance_score()
        - Continuity: 0.1 if <10min old, + 0.3*token_overlap
        - Structure: 0.15 * (1 - abs(query_density - memory_density) * 3)
        - Anchor bonus: 0.2*overlap if deictic, else 0.1*overlap
        - Analogy penalty: -0.1 for analogies in math queries
        - Tone penalty: -0.2 truth for negative words
        """
        # See PROJECT_SKELETON.md Section 5.1 for full implementation
        ...
        return sorted(memories, key=lambda m: m['final_score'], reverse=True)

    # Scoring helpers
    def _calculate_truth_score(query: str, response: str) -> float:
        """
        Base: 0.5
        +0.1 if response >200 chars
        +0.1 if '?' in response (clarifying)
        +0.2 if confirmation words (yes/correct/exactly)
        +0.15 if query tokens appear in last conversation response
        Capped at 1.0
        """
        score = 0.5
        if len(response) > 200: score += 0.1
        if '?' in response: score += 0.1
        if any(w in response.lower() for w in ['yes', 'correct', 'exactly', 'right']): score += 0.2
        # Continuity check with conversation_context omitted for brevity
        return min(1.0, score)

    def _calculate_importance_score(content: str) -> float:
        """
        Base: 0.5
        +0.1 if >200 chars
        +0.1 if contains '?'
        +0.2 if keywords: important, remember, note, key, critical, essential, todo, directive
        Capped at 1.0
        """
        score = 0.5
        if len(content) > 200: score += 0.1
        if '?' in content: score += 0.1
        keywords = ['important', 'remember', 'note', 'key', 'critical', 'essential', 'todo', 'directive']
        if any(kw in content.lower() for kw in keywords): score += 0.2
        return min(1.0, score)
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
# core/prompt_builder.py
class PromptBuilder:
    def build_prompt(query: str, memories: List[Dict], topics: List[str] = None, budget: int = 2048) -> str:
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

## LLM Generation

```python
# core/response_generator.py
class ResponseGenerator:
    async def generate_response_stream(prompt: str, model: str = "gpt-4o-mini") -> AsyncIterator[str]:
        """Stream LLM response chunks"""
        async for chunk in self.model_manager.generate_stream(prompt, model):
            yield chunk

    async def generate_best_of_n(prompt: str, n: int = 3) -> str:
        """Generate N responses, use judge to pick best"""
        responses = await asyncio.gather(*[
            self.model_manager.generate(prompt, model) for _ in range(n)
        ])
        best = await self.competitive_scorer.judge_responses(prompt, responses)
        return best
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

# Score weights (sum to 1.0)
SCORE_WEIGHTS = {
    "relevance": 0.35,
    "recency": 0.25,
    "truth": 0.20,
    "importance": 0.05,
    "continuity": 0.10
}

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
COLLECTIONS = ["episodic", "semantic", "procedural", "summary", "meta", "new_type"]

# 3. memory_coordinator.py - handle in get_memories()
new_type_results = await self.chroma_store.query(query, "new_type", limit)

# 4. prompt_builder.py - add to prompt assembly
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
# Recency decay
recency = 1.0 / (1.0 + 0.05 * age_hours)

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
python -c "from core.prompt_builder import PromptBuilder; pb = PromptBuilder(...); print(pb.build_prompt('test query', []))"
```

---

**End of Quick Reference**

This document is ~500 lines → ~3K tokens, providing instant lookup for critical functions and patterns.
