# hierarchical_memory.py - Hierarchical memory system with LLM gating
import re
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio
from enum import Enum
from config import GATE_REL_THERESHOLD, MAX_FINAL_MEMORIES, MEM_NO,MEM_IMPORTANCE_SCORE, CHILD_MEM_LIMIT
from transformers import GPT2TokenizerFast
from llm_gates import MultiStageGateSystem

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

class HierarchicalMemorySystem:
    def __init__(self, model_manager, storage_dir="hierarchical_memory"):
        self.model_manager = model_manager
        self.storage_dir = storage_dir
        self.memories: Dict[str, MemoryNode] = {}
        self.hierarchy: Dict[str, List[str]] = defaultdict(list)
        self.type_index: Dict[MemoryType, List[str]] = defaultdict(list)
        self.tag_index: Dict[str, List[str]] = defaultdict(list)

        # Memory configuration
        self.consolidation_threshold = 10
        self.importance_threshold = 0.7
        self.decay_interval = timedelta(days=1)

        # Initialize ChromaDB
        try:
            from chromadb import PersistentClient
            self.client = PersistentClient(path="chroma_db")
            self.chroma_collection = self.client.get_or_create_collection("assistant-memory")
            print(f"[DEBUG] Loaded Chroma collection with {len(self.chroma_collection.get()['documents'])} documents.")
        except Exception as e:
            print(f"[ChromaDB Init Error] {e}")
            self.chroma_collection = None

        # LLM gating system for memory filtering
        self.semantic_gate =MultiStageGateSystem(self.model_manager)

        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
        self.load_memories()

    def chunk_memory_content(self, text: str, max_tokens: int = 256) -> List[str]:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokens = tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk = tokenizer.decode(tokens[i:i + max_tokens])
            chunks.append(chunk.strip())
        return chunks

        tokens = tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk = tokenizer.decode(tokens[i:i + max_tokens])
            chunks.append(chunk.strip())

        return chunks

    async def store_interaction(self, query: str, response: str, tags: List[str] = None) -> str:
        """Store a new interaction and trigger hierarchical processing"""
        import uuid

        # Create episodic memory
        memory_id = str(uuid.uuid4())
        content = f"User: {query}\nAssistant: {response}"

        # Calculate importance score
        importance = await self._calculate_importance(content)

        memory = MemoryNode(
            id=memory_id,
            content=content,
            type=MemoryType.EPISODIC,
            timestamp=datetime.now(),
            importance_score=importance,
            tags=tags or []
        )

        self.memories[memory_id] = memory
        self.type_index[MemoryType.EPISODIC].append(memory_id)

        # Update tag index
        for tag in memory.tags:
            self.tag_index[tag].append(memory_id)

        # Extract semantic knowledge
        raw_semantic_memories = await self._extract_semantic_knowledge(query, response)
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
            await self._consolidate_memories()

        # Save to disk
        self.save_memories()
        await self._index_in_chroma(memory)

        return memory_id

    async def retrieve_relevant_memories(self, query: str, max_memories: int = 10) -> List[Dict]:
        """Retrieve memories relevant to the query using LLM gating"""

        print(f"\n[DEBUG] Total memories in system: {len(self.memories)}")
        print(f"[DEBUG] Memory types: {[(t.value, len(ids)) for t, ids in self.type_index.items() if ids]}")

        # Stage 1: Quick semantic search for candidates
        candidates = await self._semantic_search_candidates(query, top_k=15)

        # Stage 2: LLM-based relevance gating
        relevant_memories = await self._gate_memories(query, candidates)

        # Stage 3: Expand with hierarchical context
        expanded_memories = self._expand_hierarchical_context(relevant_memories)

        # Stage 4: Apply temporal decay and importance weighting
        scored_memories = self._apply_temporal_decay(expanded_memories)

        # Sort by relevance score and return top N
        sorted_memories = sorted(scored_memories, key=lambda x: x['final_score'], reverse=True)

        # Update access counts
        for mem in sorted_memories[:max_memories]:
            self._update_access(mem['id'])

        print(f"\n[DEBUG] Memory Retrieval Pipeline:")
        print(f"  - Candidates from Chroma: {len(candidates)}")
        print(f"  - After LLM gating: {len(relevant_memories)}")
        print(f"  - After hierarchical expansion: {len(expanded_memories)}")
        print(f"  - Final scored memories: {len(sorted_memories[:max_memories])}")

        # Debug output with timestamps
        print(f"\n[DEBUG] Retrieved Memories (showing top {min(5, len(sorted_memories))} of {len(sorted_memories)}):")
        for i, mem in enumerate(sorted_memories[:5]):
            memory = mem['memory']
            print(f"\n  Memory {i+1}:")
            print(f"    Timestamp: {memory.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"    Type: {memory.type.value}")
            print(f"    Score: {mem['final_score']:.3f}")

            # Show content preview for all types
            content_preview = memory.content.replace('\n', ' ')[:150]
            print(f"    Content: {content_preview}...")

            # Parse and display Q&A only for episodic/summary types
            if memory.type in [MemoryType.EPISODIC, MemoryType.SUMMARY]:
                query_match = re.search(r'^User:\s*(.*?)(?=\nAssistant:|\Z)', memory.content, re.IGNORECASE | re.DOTALL)
                response_match = re.search(r'^Assistant:\s*(.*)', memory.content, re.IGNORECASE | re.MULTILINE | re.DOTALL)

                if query_match:
                    print(f"    Q: {query_match.group(1).strip()[:80]}...")
                if response_match:
                    print(f"    A: {response_match.group(1).strip()[:80]}...")

        return sorted_memories[:max_memories]

    async def _calculate_importance(self, content: str) -> float:
        """Calculate importance score using LLM"""
        prompt = f"""Rate the importance of this interaction on a scale of 0.0 to 1.0.
Consider: information density, uniqueness, potential future relevance.

Content: {content[:500]}

Respond with just a number between 0.0 and 1.0."""

        try:
            response = await asyncio.to_thread(
                self.model_manager.generate,
                prompt,
                max_tokens=10,
                temperature=0.1
            )
            return float(response.strip())
        except:
            return 0.5

    async def _extract_semantic_knowledge(self, query: str, response: str) -> List[MemoryNode]:
        """Extract semantic facts from interaction"""
        prompt = f"""Extract key facts and knowledge from this interaction.
List each fact as a separate line. Focus on reusable information.

User: {query}
Assistant: {response[:1000]}

Facts:"""

        try:
            facts_text = await asyncio.to_thread(
                self.model_manager.generate,
                prompt,
                max_tokens=200,
                temperature=0.3
            )

            facts = [f.strip() for f in facts_text.split('\n') if f.strip()]

            semantic_memories = []
            for fact in facts[:5]:  # Limit to 5 facts per interaction
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
        except:
            return []


    async def _consolidate_memories(self):
        """Consolidate recent episodic memories into summaries"""
        recent_episodic = self.type_index[MemoryType.EPISODIC][-self.consolidation_threshold:]

        if len(recent_episodic) < self.consolidation_threshold:
            return

        # Gather content for summarization
        contents = []
        for mem_id in recent_episodic:
            if mem_id in self.memories:
                contents.append(self.memories[mem_id].content)

        # Generate summary
        prompt = f"""Create a concise summary of these interactions, preserving key information:

{chr(10).join(contents[:5])}  # Limit to prevent token overflow

Summary:"""

        try:
            summary = await asyncio.to_thread(
                self.model_manager.generate,
                prompt,
                max_tokens=300,
                temperature=0.5
            )

            # Create summary node
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

            # Link summary to original memories
            for mem_id in recent_episodic:
                await self._add_child_memory(summary_id, self.memories[mem_id])

            self.memories[summary_id] = summary_node
            self.type_index[MemoryType.SUMMARY].append(summary_id)

        except Exception as e:
            print(f"Consolidation error: {e}")

    async def _index_in_chroma(self, memory: MemoryNode):
        """Index memory in ChromaDB"""
        if not self.chroma_collection:
            return
        try:
            self.chroma_collection.add(
                documents=[memory.content],
                metadatas=[{"type": memory.type.value, "tags": json.dumps(memory.tags)}],
                ids=[memory.id]
            )
        except Exception as e:
            print(f"[Chroma Index Error] Failed to index memory {memory.id}: {e}")

    async def _semantic_search_candidates(self, query: str, top_k: int = 15) -> List[MemoryNode]:
        """Get candidate memories from ChromaDB semantic search"""
        if not self.chroma_collection:
            return []

        try:
            # Remove the where clause - ChromaDB doesn't support timestamp filtering
            results = self.chroma_collection.query(
                query_texts=[query],
                n_results=10,  # CHANGED: Reduced from top_k * 2 for performance
                include=["metadatas", "distances"]
            )

            if not results["ids"] or not results["ids"][0]:
                return []  # FIXED: This line was corrupted

            # Manual filtering by recency and distance
            week_ago = datetime.now() - timedelta(days=7)
            candidates = []

            for i, doc_id in enumerate(results["ids"][0]):
                if doc_id in self.memories:
                    memory = self.memories[doc_id]
                    # Manual recency check
                    if memory.timestamp >= week_ago:
                        # Distance threshold (lower is better for ChromaDB)
                        if results["distances"][0][i] < 0.5:
                            candidates.append(memory)

            # Return top_k candidates sorted by distance
            return candidates[:top_k]

        except Exception as e:
            print(f"[Semantic Search Error] {e}")
            # Fallback: return recent memories
            recent_memories = sorted(
                self.memories.values(),
                key=lambda m: m.timestamp,
                reverse=True
            )[:top_k]
            return recent_memories

    async def _gate_memories(self, query: str, candidates: List[MemoryNode]) -> List[Dict]:
        """Parallel batch processing for memory gating"""
        if not candidates:
            return []

        prompts = [
            f"""Rate relevance (0-1) of this memory to query: "{query}"
    Memory: {mem.content[:200]}...
    Just respond with a number 0-1:"""
            for mem in candidates
        ]

        tasks = [
            self.model_manager.generate_async(
                prompt,
                model_name=self.model_manager.get_active_model_name(),
                max_tokens=10,
                temperature=0
            )
            for prompt in prompts
        ]

        # Fully parallel await
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for mem, response in zip(candidates, responses):
            try:
                if isinstance(response, Exception):
                    score = 0.5  # Default score on exception
                    print(f"[LLM Gating Error] {response}")
                else:
                    score = float(response.strip()) if response else 0.5
            except ValueError:
                score = 0.5  # Default score if can't parse float

            if score > GATE_REL_THRESHOLD:
                results.append({
                    'id': mem.id,
                    'memory': mem,
                    'relevance_score': score,
                    'content': mem.content
                })

        return results

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
                    if parent.importance>MEM_IMPORTANCE_SCORE:
                        expanded.append({
                            'id': parent.id,
                            'memory': parent,
                            'relevance_score': mem_dict['relevance_score'] * 0.7,  # Reduced relevance
                            'content': parent.content,
                            'relationship': 'parent'
                        })
                        seen_ids.add(parent.id)

            # Add highly relevant children
            for child_id in memory.child_ids[:CHILD_MEM_LIMIT]:  # Limit children
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

    def load_memories(self):
        """Load memories from disk"""
        save_path = os.path.join(self.storage_dir, 'hierarchical_memories.json')
        if not os.path.exists(save_path):
            return

        try:
            with open(save_path, 'r') as f:
                data = json.load(f)

            # Reconstruct memories
            for id, mem_data in data.get('memories', {}).items():
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

            # Reconstruct indices
            self.hierarchy = defaultdict(list, data.get('hierarchy', {}))
            for type_str, ids in data.get('type_index', {}).items():
                self.type_index[MemoryType(type_str)] = ids
            self.tag_index = defaultdict(list, data.get('tag_index', {}))

        except Exception as e:
            print(f"Error loading memories: {e}")


# Integration with existing LLM gates
class HierarchicalGatedPromptBuilder:
    def __init__(self, prompt_builder, model_manager):
        self.prompt_builder = prompt_builder
        self.model_manager = model_manager
        self.memory_system = HierarchicalMemorySystem(model_manager)

        from llm_gates import MultiStageGateSystem
        self.gate_system = MultiStageGateSystem(model_manager)
        self.semantic_gate = self.gate_system  # One unified gate system

    async def build_hierarchical_prompt(self, user_input: str, context_sources: Dict, system_prompt: str = "", directives_file: str = "structured_directives.txt") -> str:

        filtered_context = {}

        try:
            filtered_context["memories"] = await self.semantic_gate.filter_memories(
                user_input, context_sources.get("memories", [])
            )
        except Exception as e:
            print(f"[PromptBuilder - Memory Filter Error] {e}")
            filtered_context["memories"] = context_sources.get("memories", [])[:5]

        try:
            include_wiki, filtered_wiki = await self.semantic_gate.filter_wiki_content(
                user_input, context_sources.get("wiki_snippet", "")
            )
            if include_wiki:
                filtered_context["wiki_snippet"] = filtered_wiki
        except Exception as e:
            print(f"[PromptBuilder - Wiki Filter Error] {e}")
            filtered_context["wiki_snippet"] = ""

        try:
            filtered_context["semantic_chunks"] = await self.semantic_gate.filter_semantic_chunks(
                user_input, context_sources.get("semantic_chunks", [])
            )
        except Exception as e:
            print(f"[PromptBuilder - Semantic Chunk Error] {e}")
            filtered_context["semantic_chunks"] = []

        return self.prompt_builder.build_prompt(
             model_name=self.model_manager.get_active_model_name(),
            user_input=user_input,
            memories=context_sources.get("memories", []),
            summaries=context_sources.get("summaries", []),
            dreams=[],
            wiki_snippet=context_sources.get("wiki_snippet", ""),
            semantic_snippet=context_sources.get("semantic_chunks", []),
            time_context=context_sources.get("time_context", {}),
            is_api=self.model_manager.is_api_model(self.model_manager.get_active_model_name()),
            include_dreams=False,
            include_code_snapshot=False,
            include_changelog=False,
            system_prompt=system_prompt,
            directives_file=directives_file
                )
