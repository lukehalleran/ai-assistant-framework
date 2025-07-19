import os

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from utils.logging_utils import get_logger, log_and_time
from processing.gate_system import MultiStageGateSystem
logger = get_logger("memory_interface")
logger.debug("Memory_Interface.py is alive")
class MemoryInterface(ABC):

    @abstractmethod
    def store(self, content: Dict) -> str:
        """Store a memory object. Returns memory ID."""
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant memory chunks."""
        pass

    @abstractmethod
    def summarize(self) -> str:
        """Return a high-level summary."""
        pass
class MemoryType(Enum):
    EPISODIC = "episodic"      # Individual interactions
    SEMANTIC = "semantic"       # Extracted facts and knowledge
    PROCEDURAL = "procedural"   # How-to knowledge, patterns
    SUMMARY = "summary"         # Compressed episodic memories
    META = "meta"              # Memories about memory patterns

@dataclass
class MemoryNode:
    """Individual memory unit in the hierarchy"""
    id: str
    content: str
    type: MemoryType
    timestamp: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    importance_score: float = 0.5
    decay_rate: float = 0.1
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    embeddings: Optional[List[float]] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self):
        safe_metadata = {
            k: (
                json.dumps(v) if isinstance(v, (list, dict)) else v
            )
            for k, v in self.metadata.items()
        }

        return {
            "id": self.id,
            "content": self.content,
            "metadata": {
                "type": self.type.value,
                "timestamp": self.timestamp.isoformat(),
                "tags": json.dumps(self.tags),
                **safe_metadata
            }
        }

class HierarchicalMemorySystem:
    def __init__(self, model_manager, chroma_store: Optional[MultiCollectionChromaStore] = None, storage_dir="hierarchical_memory", embed_model=None, cosine_threshold=0.45):
        self.model_manager = model_manager
        self.storage_dir = storage_dir
        self.memories: Dict[str, MemoryNode] = {}
        self.chroma_store = chroma_store
        self.hierarchy: Dict[str, List[str]] = defaultdict(list)
        self.type_index: Dict[MemoryType, List[str]] = defaultdict(list)
        self.tag_index: Dict[str, List[str]] = defaultdict(list)
        self.embed_model = embed_model or SentenceTransformer("all-MiniLM-L6-v2")
        self.episodic_memory = []
        self.cosine_threshold = cosine_threshold

        # Memory configuration
        self.consolidation_threshold = 10
        self.importance_threshold = 0.7
        self.decay_interval = timedelta(days=1)

        # Initialize ChromaDB
        try:
            from chromadb import PersistentClient
            self.client = PersistentClient(path=os.environ.get("CHROMA_PATH", "chroma_db"))
            logger.debug("ChromaDB client initialized")

            try:
                self.chroma_collection = self.client.get_collection("assistant-memory")
                logger.debug("Loaded existing Chroma collection")
            except Exception as e:
                logger.debug(f"Collection not found, attempting to create: {e}")
                try:
                    import onnxruntime as ort
                    providers = ort.get_available_providers()
                    from chromadb.utils import embedding_functions
                    embed_fn = embedding_functions.ONNXMiniLM_L6_V2(providers=providers)
                    self.chroma_collection = self.client.create_collection(
                        "assistant-memory",
                        embedding_function=embed_fn
                    )
                    logger.debug("Created new Chroma collection with ONNX providers")
                except Exception as e:
                    logger.debug(f"ONNX embedding creation failed: {e}")
                    try:
                        self.chroma_collection = self.client.create_collection("assistant-memory")
                        logger.debug("Created new Chroma collection with default embedding")
                    except Exception as e:
                        logger.debug(f"Default collection creation failed: {e}")
                        self.chroma_collection = None

            if self.chroma_collection:
                doc_count = len(self.chroma_collection.get()['documents'])
                logger.debug(f"Chroma collection loaded with {doc_count} documents.")

        except Exception as e:
            logger.debug(f"[ChromaDB Init Error] {e}")
            self.chroma_collection = None



        # Cosine similarity gating system
        self.semantic_gate = MultiStageGateSystem(self.model_manager, cosine_threshold)

        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
        self.load_memories()

        # Safe debug logging
        if self.chroma_collection:
            try:
                logger.debug(f"Chroma collection count: {self.chroma_collection.count()}")
            except:
                logger.debug(f"Chroma collection exists but count unavailable")
        else:
            logger.debug(f"No Chroma collection available")

        logger.debug(f"Number of self.memories: {len(self.memories)}")

    @log_and_time("Chunk Memory Content")
    def chunk_memory_content(self, text: str, max_tokens: int = 256) -> List[str]:
        """Chunk text into smaller pieces using cached tokenizer"""
        tokenizer = get_cached_tokenizer("gpt2")
        tokens = tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk = tokenizer.decode(tokens[i:i + max_tokens])
            chunks.append(chunk.strip())
        return chunks

    def _expand_hierarchical_context(self, memories: List[Dict]) -> List[Dict]:
        """Expand memories with hierarchical relationships"""
        expanded = memories.copy()
        seen_ids = {m['id'] for m in memories}

        for mem_dict in memories[:MEM_NO]:
            memory = mem_dict['memory']

            # Add parent context
            if memory.parent_id and memory.parent_id not in seen_ids:
                if memory.parent_id in self.memories:
                    parent = self.memories[memory.parent_id]
                    if parent.importance_score > MEM_IMPORTANCE_SCORE:
                        expanded.append({
                            'id': parent.id,
                            'memory': parent,
                            'relevance_score': mem_dict['relevance_score'] * 0.7,
                            'content': parent.content,
                            'relationship': 'parent'
                        })
                        seen_ids.add(parent.id)

            # Add highly relevant children
            for child_id in memory.child_ids[:CHILD_MEM_LIMIT]:
                if child_id not in seen_ids and child_id in self.memories:
                    child = self.memories[child_id]
                    if child.importance_score > 0.6:
                        expanded.append({
                            'id': child.id,
                            'memory': child,
                            'relevance_score': mem_dict['relevance_score'] * 0.8,
                            'content': child.content,
                            'relationship': 'child'
                        })
                        seen_ids.add(child.id)

        return expanded

    @log_and_time("store interaction")
    async def store_interaction(self, query: str, response: str, tags: List[str] = None) -> str:
        """Store a new interaction and trigger hierarchical processing"""
        import uuid

        # Create episodic memory
        memory_id = str(uuid.uuid4())
        content = f"User: {query}\nAssistant: {response}"
        metadata = {"query": query, "response": response}

        # Calculate importance score using simple heuristics
        importance = await self._calculate_importance_heuristic(content)

        memory = MemoryNode(
            id=memory_id,
            content=content,
            type=MemoryType.EPISODIC,
            timestamp=datetime.now(),
            importance_score=importance,
            tags=tags or [],
            metadata=metadata
        )

        self.memories[memory_id] = memory
        self.type_index[MemoryType.EPISODIC].append(memory_id)

        # Update tag index
        for tag in memory.tags:
            self.tag_index[tag].append(memory_id)

        # Extract semantic knowledge using simple heuristics
        raw_semantic_memories = await self._extract_semantic_knowledge_heuristic(query, response)
        semantic_memories = []
        for mem in raw_semantic_memories:
            chunks = self.chunk_memory_content(mem.content)
            for i, chunk in enumerate(chunks):
                chunked_mem = MemoryNode(
                    id=f"{mem.id}_chunk{i}",
                    content=chunk,
                    type=mem.type,
                    timestamp=mem.timestamp,
                    importance_score=mem.importance_score,
                    tags=mem.tags,
                    parent_id=mem.parent_id
                )
                semantic_memories.append(chunked_mem)

        for sem_mem in semantic_memories:
            await self._add_child_memory(memory_id, sem_mem)
            await self._index_in_chroma(sem_mem)

        # Check for consolidation
        if len(self.type_index[MemoryType.EPISODIC]) % self.consolidation_threshold == 0:
            await self._consolidate_memories_heuristic()

        # Save to disk
        self.save_memories()
        await self._index_in_chroma(memory)

        return memory_id

    @log_and_time("Retrieve Memory")
    async def retrieve_relevant_memories(self, query: str, max_memories: int = 10) -> List[Dict]:
        """Retrieve memories relevant to the query using cosine similarity"""

        logger.debug(f"\n[DEBUG] Total memories in system: {len(self.memories)}")
        logger.debug(f" Memory types: {[(t.value, len(ids)) for t, ids in self.type_index.items() if ids]}")

        # Stage 1: Quick semantic search for candidates
        candidates = await self._semantic_search_candidates(query, top_k=50)

        # Stage 2: Cosine similarity gating
        relevant_memories = await self._gate_memories_cosine(query, candidates)

        # Stage 3: Expand with hierarchical context
        expanded_memories = self._expand_hierarchical_context(relevant_memories)

        # Stage 4: Apply temporal decay and importance weighting
        scored_memories = self._apply_temporal_decay(expanded_memories)

        # Sort by relevance score and return top N
        sorted_memories = sorted(scored_memories, key=lambda x: x['final_score'], reverse=True)

        # Update access counts
        for mem in sorted_memories[:max_memories]:
            self._update_access(mem['id'])

        logger.debug(f"\n[DEBUG] Memory Retrieval Pipeline:")
        logger.debug(f"  - Candidates from Chroma: {len(candidates)}")
        logger.debug(f"  - After cosine gating: {len(relevant_memories)}")
        logger.debug(f"  - After hierarchical expansion: {len(expanded_memories)}")
        logger.debug(f"  - Final scored memories: {len(sorted_memories[:max_memories])}")

        return sorted_memories[:max_memories]

    @log_and_time("Calculate mem importance heuristic")
    async def _calculate_importance_heuristic(self, content: str) -> float:
        """Calculate importance score using simple heuristics"""
        # Simple heuristics for importance
        score = 0.5

        # Boost for longer content
        if len(content) > 200:
            score += 0.1

        # Boost for questions
        if '?' in content:
            score += 0.1

        # Boost for certain keywords
        important_keywords = ['important', 'remember', 'note', 'key', 'critical', 'essential']
        if any(kw in content.lower() for kw in important_keywords):
            score += 0.2

        return min(score, 1.0)

    @log_and_time("Extract semantic knowledge heuristic")
    async def _extract_semantic_knowledge_heuristic(self, query: str, response: str) -> List[MemoryNode]:
        """Extract semantic facts using simple pattern matching"""
        semantic_memories = []

        # Look for facts pattern: "X is Y", "X are Y", etc.
        fact_patterns = [
            r'(\w+)\s+(?:is|are|was|were)\s+(.+?)(?:\.|,|;|$)',
            r'(?:I|you|we)\s+(?:like|love|hate|prefer)\s+(.+?)(?:\.|,|;|$)',
            r'(?:my|your|our)\s+(\w+)\s+(?:is|are)\s+(.+?)(?:\.|,|;|$)'
        ]

        combined_text = f"{query} {response}"
        facts = []

        for pattern in fact_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches[:3]:  # Limit facts per pattern
                if isinstance(match, tuple):
                    fact = ' '.join(match)
                else:
                    fact = match
                if len(fact) > 10 and len(fact) < 200:  # Reasonable length
                    facts.append(fact.strip())

        # Create memory nodes for unique facts
        seen_facts = set()
        for fact in facts[:5]:  # Limit total facts
            if fact.lower() not in seen_facts:
                seen_facts.add(fact.lower())
                import uuid
                sem_id = str(uuid.uuid4())
                semantic_memories.append(MemoryNode(
                    id=sem_id,
                    content=fact,
                    type=MemoryType.SEMANTIC,
                    timestamp=datetime.now(),
                    importance_score=0.6,
                    tags=["extracted_fact"]
                ))

        return semantic_memories

    @log_and_time("Consolidate Memories Heuristic")
    async def _consolidate_memories_heuristic(self):
        """Consolidate recent episodic memories into summaries using simple compression"""
        recent_episodic = self.type_index[MemoryType.EPISODIC][-self.consolidation_threshold:]

        if len(recent_episodic) < self.consolidation_threshold:
            return

        # Gather content for summarization
        contents = []
        for mem_id in recent_episodic:
            if mem_id in self.memories:
                contents.append(self.memories[mem_id].content)

        # Simple summarization: extract key phrases
        summary_lines = []
        for content in contents[:5]:
            # Extract Q&A pairs
            user_match = re.search(r'User:\s*(.+?)(?=\nAssistant:|$)', content)
            assistant_match = re.search(r'Assistant:\s*(.+?)(?=\n|$)', content)

            if user_match and assistant_match:
                q = user_match.group(1).strip()[:50]
                a = assistant_match.group(1).strip()[:50]
                summary_lines.append(f"Q: {q}... A: {a}...")

        if summary_lines:
            summary = "Summary of recent interactions:\n" + "\n".join(summary_lines)

            import uuid
            summary_id = str(uuid.uuid4())
            summary_node = MemoryNode(
                id=summary_id,
                content=summary,
                type=MemoryType.SUMMARY,
                timestamp=datetime.now(),
                importance_score=0.8,
                tags=["consolidated_summary"]
            )

            # ✅ Register the summary first
            self.memories[summary_id] = summary_node
            self.type_index[MemoryType.SUMMARY].append(summary_id)

            # 🔁 Then link child memories
            for mem_id in recent_episodic:
                if mem_id in self.memories:
                    await self._add_child_memory(summary_id, self.memories[mem_id])



    @log_and_time("Index in Chroma")
    async def _index_in_chroma(self, memory: MemoryNode):
        """Index memory in ChromaDB"""
        try:
            metadata = {
                "type": memory.type.value,
                "timestamp": memory.timestamp.isoformat(),
                "tags": ", ".join(memory.tags),
                **{
                    k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                    for k, v in memory.metadata.items()
                }
            }
            print(f"DEBUG: chroma_store is {'set' if self.chroma_store else 'MISSING'}")
            if self.chroma_store:
                mem_type = memory.type

                if mem_type == MemoryType.EPISODIC:
                    parts = memory.content.split("\nAssistant: ", 1)
                    if len(parts) == 2:
                        query = parts[0].replace("User: ", "")
                        response = parts[1]
                    else:
                        query = memory.content[:100]
                        response = memory.content[100:]

                    self.chroma_store.add_conversation_memory(query, response, metadata)

                elif mem_type == MemoryType.SEMANTIC:
                    self.chroma_store.add_semantic_chunk({
                        "content": memory.content,
                        "source": memory.metadata.get("source", "semantic_extraction"),
                        "type": memory.metadata.get("chunk_type", "unknown"),
                        "importance": memory.importance_score
                    })


                elif mem_type == MemoryType.SUMMARY:
                    self.chroma_store.add_summary(
                        summary=memory.content,
                        period="auto",
                        metadata=metadata
                    )

                else:
                    logger.warning(f"⚠️ Unhandled memory type for Chroma store: {mem_type.value}")

            elif self.chroma_collection:
                self.chroma_collection.add(
                    documents=[memory.content],
                    metadatas=[metadata],
                    ids=[memory.id]
                )

            logger.debug(f"✅ Indexed {memory.type.value} memory: {memory.id}")

        except Exception as e:
            logger.debug(f"[Chroma Index Error] Failed to index memory {memory.id}: {e}")



    @log_and_time("Semantic Search Candidates")
    async def _semantic_search_candidates(self, query: str, top_k: int = 50) -> List[MemoryNode]:
        """Get candidate memories from ChromaDB semantic search"""
        if not self.chroma_collection:
            return []

        candidates = []

        results = self.chroma_collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["distances", "documents", "metadatas"]
        )

        result_ids = results.get("ids", [[]])[0]
        result_distances = results.get("distances", [[]])[0]

        logger.debug(f"Chroma returned {len(result_ids)} candidates")

        if not result_ids or not result_distances:
            logger.debug("[WARNING] No semantic candidates returned by Chroma.")
            return []

        # Build ID to distance mapping
        id_to_distance = dict(zip(result_ids, result_distances))

        # Match with self.memories
        for chroma_id, distance in id_to_distance.items():
            # Handle chunked IDs
            normalized_id = chroma_id.replace("_chunk0", "") if "_chunk0" in chroma_id else chroma_id

            # Try both the normalized and original ID
            if normalized_id in self.memories:
                candidates.append(self.memories[normalized_id])
                logger.debug(f" Matched memory {normalized_id} with distance {distance}")
            elif chroma_id in self.memories:
                candidates.append(self.memories[chroma_id])
                logger.debug(f" Matched memory {chroma_id} with distance {distance}")

        logger.debug(f" Found {len(candidates)} candidates from {len(result_ids)} Chroma results")
        return candidates

    @log_and_time("Gate Memories Cosine")
    async def _gate_memories_cosine(self, query: str, candidates: List[MemoryNode]) -> List[Dict]:
        """Use cosine similarity to filter memories for relevance"""
        if not candidates:
            return []

        try:
            # Encode query once
            query_emb = self.embed_model.encode(query, convert_to_numpy=True)

            # Prepare memory content and encode in batch
            contents = [mem.content[:500] for mem in candidates]
            memory_embs = self.embed_model.encode(contents, convert_to_numpy=True, batch_size=32)

            # Calculate cosine similarities
            similarities = cosine_similarity([query_emb], memory_embs)[0]

            relevant_memories = []
            for mem, similarity in zip(candidates, similarities):
                if similarity >= self.cosine_threshold:
                    relevant_memories.append({
                        'id': mem.id,
                        'memory': mem,
                        'relevance_score': float(similarity),
                        'content': mem.content
                    })
                    logger.debug(f" Memory {mem.id[:8]}... passed gate with score {similarity:.3f}")
                else:
                    logger.debug(f" Memory {mem.id[:8]}... filtered out with score {similarity:.3f}")

            # Sort by relevance
            relevant_memories = sorted(relevant_memories, key=lambda x: x['relevance_score'], reverse=True)

            logger.debug(f" _gate_memories_cosine returning {len(relevant_memories)} relevant memories.")
            logger.debug(f"[GATE COSINE] Distance: {distance:.3f} → Similarity: {similarity:.3f} | Threshold: {GATE_REL_THRESHOLD}")
            return relevant_memories[:30]  # Limit to top 30

        except Exception as e:
            logger.error(f"[Cosine Gate Error] {e}")
            # Fallback: return top candidates with default scores
            return [
                {
                    'id': mem.id,
                    'memory': mem,
                    'relevance_score': 0.5 - (i * 0.01),
                    'content': mem.content
                }
                for i, mem in enumerate(candidates[:10])
            ]

    def _apply_temporal_decay(self, memories: List[Dict]) -> List[Dict]:
        """Apply temporal decay to memory scores"""
        now = datetime.now()

        for mem_dict in memories:
            memory = mem_dict['memory']

            # Calculate age in days
            age_days = (now - memory.timestamp).days

            # Apply decay function
            decay_factor = 1.0 / (1.0 + memory.decay_rate * age_days)

            # Boost recently accessed memories
            access_recency = (now - memory.last_accessed).days
            access_boost = 1.0 if access_recency < 1 else 1.0 / (1.0 + 0.1 * access_recency)

            # Calculate final score
            mem_dict['final_score'] = (
                mem_dict['relevance_score'] *
                memory.importance_score *
                decay_factor *
                access_boost
            )

        return memories

    async def _add_child_memory(self, parent_id: str, child: MemoryNode):
        """Add a child memory to parent"""
        child.parent_id = parent_id
        self.memories[child.id] = child
        self.memories[parent_id].child_ids.append(child.id)
        self.hierarchy[parent_id].append(child.id)
        self.type_index[child.type].append(child.id)
        await self._index_in_chroma(child)

    def _update_access(self, memory_id: str):
        """Update access count and timestamp"""
        if memory_id in self.memories:
            self.memories[memory_id].access_count += 1
            self.memories[memory_id].last_accessed = datetime.now()

    def save_memories(self):
        """Save memories to disk"""
        data = {
            'memories': {
                id: {
                    'id': m.id,
                    'content': m.content,
                    'type': m.type.value,
                    'timestamp': m.timestamp.isoformat(),
                    'access_count': m.access_count,
                    'last_accessed': m.last_accessed.isoformat(),
                    'importance_score': m.importance_score,
                    'decay_rate': m.decay_rate,
                    'parent_id': m.parent_id,
                    'child_ids': m.child_ids,
                    'tags': m.tags,
                    'metadata': m.metadata
                }
                for id, m in self.memories.items()
            },
            'hierarchy': dict(self.hierarchy),
            'type_index': {k.value: v for k, v in self.type_index.items()},
            'tag_index': dict(self.tag_index)
        }

        save_path = os.path.join(self.storage_dir, 'hierarchical_memories.json')
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

    @log_and_time("Load semantic memories")
    def load_semantic_memories_from_chroma(self):
        """Rebuild semantic MemoryNode objects from ChromaDB on startup"""
        logger.debug(f" Attempting to reconstruct semantic memories from Chroma...")
        try:
            results = self.chroma_collection.get(include=["documents", "metadatas"])

            for doc, metadata, id in zip(results["documents"], results["metadatas"], results["ids"]):
                try:
                    timestamp_str = metadata.get("timestamp")
                    timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()

                    memory = MemoryNode(
                        id=id,
                        content=doc,
                        type=MemoryType(metadata.get("type", "semantic")),
                        timestamp=timestamp,
                        tags=json.loads(metadata.get("tags", "[]")),
                        metadata=metadata
                    )
                    self.memories[memory.id] = memory

                except Exception as e:
                    logger.debug(f"[CHROMA LOAD ERROR] Failed to load memory {id}: {e}")
        except Exception as e:
            logger.debug(f"[CHROMA COLLECTION ERROR] Could not query ChromaDB: {e}")

    @log_and_time("Load Memories")
    def load_memories(self):
        """Load memories from disk and sync with Chroma"""
        save_path = os.path.join(self.storage_dir, 'hierarchical_memories.json')

        # Load hierarchical JSON disk memories
        if os.path.exists(save_path):
            try:
                with open(save_path, 'r') as f:
                    data = json.load(f)
                for id, mem_data in data.get('memories', {}).items():
                    try:
                        memory = MemoryNode(
                            id=mem_data['id'],
                            content=mem_data['content'],
                            type=MemoryType(mem_data['type']),
                            timestamp=datetime.fromisoformat(mem_data['timestamp']),
                            access_count=mem_data['access_count'],
                            last_accessed=datetime.fromisoformat(mem_data['last_accessed']),
                            importance_score=mem_data['importance_score'],
                            decay_rate=mem_data['decay_rate'],
                            parent_id=mem_data.get('parent_id'),
                            child_ids=mem_data.get('child_ids', []),
                            tags=mem_data.get('tags', []),
                            metadata=mem_data.get('metadata', {})
                        )
                        self.memories[id] = memory
                    except Exception as e:
                        logger.debug(f"[LOAD ERROR] Could not load memory ID {id}: {e}")

                # Reconstruct indices
                self.hierarchy = defaultdict(list, data.get('hierarchy', {}))
                for type_str, ids in data.get('type_index', {}).items():
                    self.type_index[MemoryType(type_str)] = ids
                self.tag_index = defaultdict(list, data.get('tag_index', {}))
            except Exception as e:
                logger.debug(f"[ERROR] Failed to load hierarchical memory file: {e}")

        # Sync semantic memories from Chroma
        try:
            logger.debug(f" Syncing self.memories with Chroma...")
            results = self.chroma_collection.get()
            docs = results['documents']
            ids = results['ids']
            metadatas = results['metadatas']

            for doc, metadata, mem_id in zip(docs, metadatas, ids):
                # Normalize _chunk0 suffix if needed
                normalized_id = mem_id.replace("_chunk0", "") if mem_id.endswith("_chunk0") else mem_id

                if normalized_id not in self.memories:
                    try:
                        # Validate or default MemoryType
                        raw_type = metadata.get("type", "").lower()
                        if raw_type not in MemoryType._value2member_map_:
                            logger.debug(f"[CHROMA LOAD WARNING] Unknown type '{raw_type}' for memory ID {mem_id}, defaulting to 'semantic'")
                            memory_type = MemoryType.SEMANTIC
                        else:
                            memory_type = MemoryType(raw_type)

                        # Parse timestamp safely
                        raw_timestamp = metadata.get("timestamp")
                        try:
                            timestamp = datetime.fromisoformat(raw_timestamp) if raw_timestamp else datetime.now()
                        except Exception as e:
                            logger.debug(f"[CHROMA LOAD WARNING] Invalid timestamp for ID {mem_id}: {e}, using now()")
                            timestamp = datetime.now()

                        # Parse tags safely
                        raw_tags = metadata.get("tags", "[]")
                        try:
                            tags = json.loads(raw_tags) if isinstance(raw_tags, str) else raw_tags
                        except Exception as e:
                            logger.debug(f"[CHROMA LOAD WARNING] Invalid tags format for ID {mem_id}: {e}")
                            tags = []

                        # Handle doc as either JSON object or raw string
                        try:
                            content_data = json.loads(doc)
                            content = content_data.get("content", "")
                        except Exception:
                            logger.debug(f"[CHROMA LOAD WARNING] Document is not JSON for ID {mem_id}; using raw content.")
                            content = doc

                        memory = MemoryNode(
                            id=normalized_id,
                            content=content,
                            type=memory_type,
                            timestamp=timestamp,
                            tags=tags,
                            metadata=metadata
                        )
                        self.memories[normalized_id] = memory

                    except Exception as e:
                        logger.debug(f"[CHROMA LOAD ERROR] Could not load memory from Chroma ID {mem_id}: {e}")

        except Exception as e:
            logger.debug(f"[CHROMA COLLECTION ERROR] Could not query ChromaDB: {e}")

    def get_summaries(self, query: str = None, limit: int = 5):
        return []

    def get_dreams(self, query: str = None, limit: int = 5):
        return []

