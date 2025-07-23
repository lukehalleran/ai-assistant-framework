import os
import json
import re
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from utils.logging_utils import get_logger, log_and_time
from processing.gate_system import MultiStageGateSystem
from config.config import MEM_NO, MEM_IMPORTANCE_SCORE, CHILD_MEM_LIMIT, GATE_REL_THRESHOLD

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

    @classmethod
    def from_dict(cls, data: Dict):
        """Reconstruct MemoryNode from dictionary"""
        metadata = data.get('metadata', {})

        # Parse tags
        tags = metadata.get('tags', '[]')
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except:
                tags = []

        # Parse timestamp
        timestamp_str = metadata.get('timestamp', '')
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
        except:
            timestamp = datetime.now()

        # Get memory type
        mem_type = MemoryType(metadata.get('type', 'semantic'))

        return cls(
            id=data.get('id', str(uuid.uuid4())),
            content=data.get('content', ''),
            type=mem_type,
            timestamp=timestamp,
            tags=tags,
            metadata=metadata
        )


class HierarchicalMemorySystem:
    def __init__(
        self,
        model_manager,
        chroma_store: MultiCollectionChromaStore,
        storage_dir="hierarchical_memory",
        embed_model=None,
        cosine_threshold=0.45
    ):
        self.model_manager = model_manager
        self.storage_dir = storage_dir
        self.chroma_store = chroma_store
        self.memories: Dict[str, MemoryNode] = {}
        self.hierarchy: Dict[str, List[str]] = defaultdict(list)
        self.type_index: Dict[MemoryType, List[str]] = defaultdict(list)
        self.tag_index: Dict[str, List[str]] = defaultdict(list)
        self.embed_model = embed_model or SentenceTransformer("all-MiniLM-L6-v2")
        self.episodic_memory = []
        self.cosine_threshold = cosine_threshold

        # Config
        self.consolidation_threshold = 10
        self.importance_threshold = 0.7
        self.decay_interval = timedelta(days=1)

        # Initialize from ChromaDB collections
        self._load_from_chroma()

        # Semantic gating
        self.semantic_gate = MultiStageGateSystem(self.model_manager, cosine_threshold)

        # Ensure disk persistence directory
        os.makedirs(self.storage_dir, exist_ok=True)
        self.load_memories()

        logger.debug(f"Initialized HierarchicalMemorySystem with {len(self.memories)} memories")

    def _load_from_chroma(self):
        """Load memories from all ChromaDB collections using a full retrieval."""
        if not self.chroma_store:
            logger.warning("No ChromaDB store provided")
            return

        try:
            # Define collection mappings
            collection_to_memory_type = {
                'conversations': MemoryType.EPISODIC,
                'semantic': MemoryType.SEMANTIC,
                'summaries': MemoryType.SUMMARY,
                'facts': MemoryType.SEMANTIC,
                'wiki': MemoryType.SEMANTIC
            }

            total_loaded = 0

            # Load from each collection
            for collection_key, memory_type in collection_to_memory_type.items():
                try:
                    # Use the new get_all_from_collection method instead of search
                    collection_results = self.chroma_store.get_all_from_collection(collection_key)

                    logger.debug(f"Loading {len(collection_results)} items from {collection_key}")

                    for item in collection_results:
                        try:
                            # Create MemoryNode from result
                            node = MemoryNode(
                                id=item['id'],
                                content=item['content'],
                                type=memory_type,
                                timestamp=datetime.fromisoformat(
                                    item['metadata'].get('timestamp', datetime.now().isoformat())
                                ),
                                tags=self._parse_tags(item['metadata'].get('tags', '')),
                                metadata=item['metadata']
                            )

                            # Store in memory system
                            self.memories[node.id] = node
                            self.type_index[memory_type].append(node.id)

                            for tag in node.tags:
                                self.tag_index[tag].append(node.id)

                            if memory_type == MemoryType.EPISODIC:
                                self.episodic_memory.append(node.content)

                            total_loaded += 1

                        except Exception as e:
                            logger.error(f"Error loading memory {item.get('id', 'unknown')}: {e}")

                except Exception as e:
                    logger.error(f"Error loading collection {collection_key}: {e}")

            logger.info(f"Loaded {total_loaded} memories from ChromaDB")

        except Exception as e:
            logger.error(f"Error loading from ChromaDB: {e}")

    def _parse_tags(self, tags_str: str) -> List[str]:
        """Parse tags from string format"""
        if not tags_str:
            return []
        if isinstance(tags_str, list):
            return tags_str
        try:
            return json.loads(tags_str)
        except:
            # Try comma-separated
            return [t.strip() for t in tags_str.split(',') if t.strip()]

    @log_and_time("store interaction")
    async def store_interaction(self, query: str, response: str, tags: List[str] = None) -> str:
        """Store a new interaction and trigger hierarchical processing"""
        # Create episodic memory
        memory_id = str(uuid.uuid4())
        content = f"User: {query}\nAssistant: {response}"
        metadata = {"query": query, "response": response}

        # Calculate importance score
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

        # Store in local memory
        self.memories[memory_id] = memory
        self.type_index[MemoryType.EPISODIC].append(memory_id)
        self.episodic_memory.append(content)

        # Update tag index
        for tag in memory.tags:
            self.tag_index[tag].append(memory_id)

        # Store in ChromaDB
        self.chroma_store.add_conversation_memory(query, response, metadata)

        # Extract semantic knowledge
        semantic_memories = await self._extract_semantic_knowledge_heuristic(query, response)

        for sem_mem in semantic_memories:
            # Store in local memory
            self.memories[sem_mem.id] = sem_mem
            self.type_index[MemoryType.SEMANTIC].append(sem_mem.id)
            sem_mem.parent_id = memory_id
            memory.child_ids.append(sem_mem.id)

            # Store in ChromaDB
            self.chroma_store.add_semantic_chunk({
                'content': sem_mem.content,
                'source': 'conversation_extraction',
                'type': 'fact',
                'importance': sem_mem.importance_score
            })

        # Check for consolidation
        if len(self.type_index[MemoryType.EPISODIC]) % self.consolidation_threshold == 0:
            await self._consolidate_memories_heuristic()

        # Save to disk
        self.save_memories()

        return memory_id

    @log_and_time("Retrieve Memory")
    async def retrieve_relevant_memories(self, query: str, max_memories: int = 10) -> List[Dict]:
        """Retrieve memories relevant to the query"""
        logger.debug(f"\n[DEBUG] Total memories in system: {len(self.memories)}")
        logger.debug(f"Memory types: {[(t.value, len(ids)) for t, ids in self.type_index.items() if ids]}")

        all_relevant = []

        # 1. Search ChromaDB collections
        if self.chroma_store:
            try:
                # Search all collections
                chroma_results = self.chroma_store.search_all(query, n_results_per_type=10)

                for collection_key, results in chroma_results.items():
                    for result in results:
                        # Check if we have this memory locally
                        mem_id = result['id']
                        if mem_id in self.memories:
                            memory = self.memories[mem_id]
                            all_relevant.append({
                                'id': mem_id,
                                'memory': memory,
                                'relevance_score': result.get('relevance_score', 0.5),
                                'content': memory.content,
                                'source': f'chroma_{collection_key}'
                            })
                        else:
                            # Create temporary memory node for ChromaDB-only results
                            memory = MemoryNode(
                                id=mem_id,
                                content=result['content'],
                                type=self._get_memory_type_for_collection(collection_key),
                                timestamp=datetime.now(),
                                metadata=result.get('metadata', {})
                            )
                            all_relevant.append({
                                'id': mem_id,
                                'memory': memory,
                                'relevance_score': result.get('relevance_score', 0.5),
                                'content': result['content'],
                                'source': f'chroma_{collection_key}'
                            })

            except Exception as e:
                logger.error(f"Error searching ChromaDB: {e}")

        # 2. Search local memories using embeddings
        if self.memories:
            local_results = await self._search_local_memories(query, top_k=20)
            all_relevant.extend(local_results)

        # 3. Remove duplicates (keep highest score)
        seen = {}
        for mem in all_relevant:
            mem_id = mem['id']
            if mem_id not in seen or mem['relevance_score'] > seen[mem_id]['relevance_score']:
                seen[mem_id] = mem

        relevant_memories = list(seen.values())

        # 4. Apply temporal decay and importance weighting
        scored_memories = self._apply_temporal_decay(relevant_memories)

        # 5. Sort by final score
        sorted_memories = sorted(scored_memories, key=lambda x: x['final_score'], reverse=True)

        # 6. Update access counts
        for mem in sorted_memories[:max_memories]:
            self._update_access(mem['id'])

        logger.debug(f"Retrieved {len(sorted_memories[:max_memories])} memories")
        return sorted_memories[:max_memories]

    async def _search_local_memories(self, query: str, top_k: int = 20) -> List[Dict]:
        """Search local memories using embeddings"""
        if not self.memories:
            return []

        try:
            # Encode query
            query_emb = self.embed_model.encode(query, convert_to_numpy=True)

            # Get all memory contents
            memory_ids = list(self.memories.keys())
            contents = [self.memories[mid].content[:500] for mid in memory_ids]

            # Encode all memories
            memory_embs = self.embed_model.encode(contents, convert_to_numpy=True, batch_size=32)

            # Calculate similarities
            similarities = cosine_similarity([query_emb], memory_embs)[0]

            # Get top results
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            results = []
            for idx in top_indices:
                if similarities[idx] >= self.cosine_threshold:
                    mem_id = memory_ids[idx]
                    memory = self.memories[mem_id]
                    results.append({
                        'id': mem_id,
                        'memory': memory,
                        'relevance_score': float(similarities[idx]),
                        'content': memory.content,
                        'source': 'local_search'
                    })

            return results

        except Exception as e:
            logger.error(f"Error in local memory search: {e}")
            return []

    def _get_memory_type_for_collection(self, collection_key: str) -> MemoryType:
        """Map collection key to memory type"""
        mapping = {
            'conversations': MemoryType.EPISODIC,
            'semantic': MemoryType.SEMANTIC,
            'summaries': MemoryType.SUMMARY,
            'facts': MemoryType.SEMANTIC,
            'wiki': MemoryType.SEMANTIC
        }
        return mapping.get(collection_key, MemoryType.SEMANTIC)

    @log_and_time("Calculate mem importance heuristic")
    async def _calculate_importance_heuristic(self, content: str) -> float:
        """Calculate importance score using simple heuristics"""
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

        # Look for facts pattern
        fact_patterns = [
            r'(\w+)\s+(?:is|are|was|were)\s+(.+?)(?:\.|,|;|$)',
            r'(?:I|you|we)\s+(?:like|love|hate|prefer)\s+(.+?)(?:\.|,|;|$)',
            r'(?:my|your|our)\s+(\w+)\s+(?:is|are)\s+(.+?)(?:\.|,|;|$)'
        ]

        combined_text = f"{query} {response}"
        facts = []

        for pattern in fact_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches[:3]:
                if isinstance(match, tuple):
                    fact = ' '.join(match)
                else:
                    fact = match
                if len(fact) > 10 and len(fact) < 200:
                    facts.append(fact.strip())

        # Create memory nodes for unique facts
        seen_facts = set()
        for fact in facts[:5]:
            if fact.lower() not in seen_facts:
                seen_facts.add(fact.lower())
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
        """Consolidate recent episodic memories into summaries"""
        recent_episodic = self.type_index[MemoryType.EPISODIC][-self.consolidation_threshold:]

        if len(recent_episodic) < self.consolidation_threshold:
            return

        # Gather content for summarization
        contents = []
        for mem_id in recent_episodic:
            if mem_id in self.memories:
                contents.append(self.memories[mem_id].content)

        # Simple summarization
        summary_lines = []
        for content in contents[:5]:
            user_match = re.search(r'User:\s*(.+?)(?=\nAssistant:|$)', content)
            assistant_match = re.search(r'Assistant:\s*(.+?)(?=\n|$)', content)

            if user_match and assistant_match:
                q = user_match.group(1).strip()[:50]
                a = assistant_match.group(1).strip()[:50]
                summary_lines.append(f"Q: {q}... A: {a}...")

        if summary_lines:
            summary = "Summary of recent interactions:\n" + "\n".join(summary_lines)

            summary_id = str(uuid.uuid4())
            summary_node = MemoryNode(
                id=summary_id,
                content=summary,
                type=MemoryType.SUMMARY,
                timestamp=datetime.now(),
                importance_score=0.8,
                tags=["consolidated_summary"]
            )

            # Store locally
            self.memories[summary_id] = summary_node
            self.type_index[MemoryType.SUMMARY].append(summary_id)

            # Store in ChromaDB
            self.chroma_store.add_summary(
                summary=summary,
                period=f"consolidation_{datetime.now().strftime('%Y%m%d_%H%M')}",
                metadata={'source': 'auto_consolidation'}
            )

            # Link child memories
            for mem_id in recent_episodic:
                if mem_id in self.memories:
                    self.memories[mem_id].parent_id = summary_id
                    summary_node.child_ids.append(mem_id)

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

    @log_and_time("Load Memories")
    def load_memories(self):
        """Load memories from disk"""
        save_path = os.path.join(self.storage_dir, 'hierarchical_memories.json')

        if os.path.exists(save_path):
            try:
                with open(save_path, 'r') as f:
                    data = json.load(f)

                for id, mem_data in data.get('memories', {}).items():
                    if id not in self.memories:  # Don't overwrite ChromaDB loaded memories
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
                logger.error(f"[ERROR] Failed to load hierarchical memory file: {e}")

    def get_summaries(self, query: str = None, limit: int = 5):
        """Get summaries from ChromaDB"""
        if self.chroma_store:
            return self.chroma_store.search_conversations(query or "summary", n_results=limit)
        return []

    def get_dreams(self, query: str = None, limit: int = 5):
        """Get dreams/meta memories"""
        # This would be implemented based on your dream generation logic
        return []
