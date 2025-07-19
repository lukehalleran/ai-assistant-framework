# daemon_7_11_25_refactor/memory/memory_coordinator.py
import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from utils.logging_utils import get_logger, log_and_time
from memory.storage.chroma_store import MultiCollectionChromaStore
from memory.memory_interface import HierarchicalMemorySystem
from processing.gate_system import MultiStageGateSystem
logger = get_logger("memory_coordinator")

class MemoryCoordinator:
    def __init__(self,
                 corpus_manager,
                 chroma_collection,
                 chroma_store: Optional[MultiCollectionChromaStore] = None,
                 hierarchical_memory: Optional[HierarchicalMemorySystem] = None,
                 gate_system: Optional[MultiStageGateSystem] = None):
        self.corpus_manager = corpus_manager
        self.chroma_collection = chroma_collection
        self.chroma_store = chroma_store
        self.hierarchical_memory = hierarchical_memory
        self.gate_system = gate_system


    @log_and_time("Retrieve Relevant Memories")
    async def retrieve_relevant_memories(self,
                                       query: str,
                                       config: Dict) -> Dict:
        """
        Retrieve and combine memories from all sources
        """
        logger.debug(f"Retrieving memories for query: {query[:50]}...")

        # Step 1: Get very recent memories (always included)
        very_recent = self.corpus_manager.get_recent_memories(
            config.get('recent_count', 3)
        )
        logger.debug(f"Got {len(very_recent)} very recent memories")

        # Step 2: Get semantic memories from ChromaDB
        semantic_memories = await self._get_semantic_memories(
            query,
            n_results=config.get('semantic_count', 30)
        )
        logger.debug(f"Got {len(semantic_memories)} semantic memories")

        # Step 3: Get hierarchical memories if available
        hierarchical_memories = []
        if self.hierarchical_memory:
            try:
                hierarchical_memories = await self.hierarchical_memory.retrieve_relevant_memories(
                    query,
                    max_memories=config.get('hierarchical_count', 15)
                )
                logger.debug(f"Got {len(hierarchical_memories)} hierarchical memories")
            except Exception as e:
                logger.error(f"Error getting hierarchical memories: {e}")

        # Step 4: Combine and deduplicate
        combined = await self._combine_memories(
            very_recent,
            semantic_memories,
            hierarchical_memories,
            query,
            config
        )

        return {
            'memories': combined,
            'counts': {
                'very_recent': len(very_recent),
                'semantic': len(semantic_memories),
                'hierarchical': len(hierarchical_memories),
                'final': len(combined)
            }
        }

    async def _get_semantic_memories(self, query: str, n_results: int = 30) -> List[Dict]:
        """Get memories from ChromaDB"""
        try:
            results = self.chroma_collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas"]
            )

            memories = []
            if results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    # Parse the document format
                    memory = {
                        'query': self._extract_query_from_doc(doc),
                        'response': self._extract_response_from_doc(doc),
                        'timestamp': results['metadatas'][0][i].get('timestamp', datetime.now()),
                        'source': 'semantic'
                    }
                    memories.append(memory)

            return memories

        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return []

    async def _combine_memories(self,
                              very_recent: List[Dict],
                              semantic: List[Dict],
                              hierarchical: List[Dict],
                              query: str,
                              config: Dict) -> List[Dict]:
        """Combine memories from all sources with deduplication"""

        # Format very recent memories
        combined = []
        seen = set()

        # Always include very recent (these bypass gating)
        for mem in very_recent:
            key = self._get_memory_key(mem)
            if key not in seen:
                mem['source'] = 'very_recent'
                mem['gated'] = False  # Mark as ungated
                combined.append(mem)
                seen.add(key)

        # Combine semantic and hierarchical for gating
        candidates_for_gating = []

        # Add semantic memories
        for mem in semantic:
            key = self._get_memory_key(mem)
            if key not in seen:
                candidates_for_gating.append(mem)

        # Add hierarchical memories (convert format)
        for h_mem in hierarchical:
            if isinstance(h_mem, dict) and 'memory' in h_mem:
                memory = h_mem['memory']
                formatted = self._format_hierarchical_memory(memory)
                key = self._get_memory_key(formatted)
                if key not in seen:
                    formatted['relevance_score'] = h_mem.get('final_score', 0.5)
                    candidates_for_gating.append(formatted)

        # Gate the candidates if gate system available
        if self.gate_system and candidates_for_gating:
            gated_memories = await self._gate_memories(query, candidates_for_gating)
            for mem in gated_memories:
                mem['gated'] = True
                combined.append(mem)
                if self.debug:
                    logger.debug(f"[TOKENS] Memory {i}: {mem_tokens} tokens — running total {current_token_total}/{token_budget}")

        else:
            # No gating, just take top N
            for mem in candidates_for_gating[:config.get('max_memories', 10)]:
                mem['gated'] = False
                combined.append(mem)

        logger.debug(f"Combined {len(combined)} total memories")
        return combined

    async def _gate_memories(self, query: str, memories: List[Dict]) -> List[Dict]:
        """Apply gating to filter memories"""
        try:
            # Convert to format expected by gate system
            chunks = []
            for mem in memories:
                chunk = {
                    "content": f"User: {mem['query']}\nAssistant: {mem['response']}",
                    "metadata": {"timestamp": mem.get("timestamp", datetime.now())}
                }
                chunks.append(chunk)

            # Apply gating
            gated_chunks = await self.gate_system.filter_memories(query, chunks)

            # Convert back to memory format
            gated_memories = []
            for chunk in gated_chunks:
                parts = chunk["content"].split("\nAssistant: ", 1)
                if len(parts) == 2:
                    memory = {
                        'query': parts[0].replace("User: ", ""),
                        'response': parts[1],
                        'timestamp': chunk["metadata"]["timestamp"]
                    }
                    gated_memories.append(memory)

            return gated_memories

        except Exception as e:
            logger.error(f"Error gating memories: {e}")
            return memories[:5]  # Fallback
    def get_summaries(self, limit: int = 3) -> List[str]:
        """Get top-level memory summaries (used for global context in prompt)"""
        return self.corpus_manager.get_summaries(limit)

    def _extract_query_from_doc(self, doc: str) -> str:
        """Extract query from ChromaDB document"""
        if "User:" in doc and "Assistant:" in doc:
            return doc.split("Assistant:")[0].replace("User:", "").strip()
        return doc[:100]

    def _extract_response_from_doc(self, doc: str) -> str:
        """Extract response from ChromaDB document"""
        if "Assistant:" in doc:
            return doc.split("Assistant:")[1].strip()
        return doc[100:]

    def _format_hierarchical_memory(self, memory) -> Dict:
        """Format hierarchical memory to standard format"""
        content = memory.content
        parts = content.split('\nAssistant: ')

        if len(parts) == 2:
            query = parts[0].replace("User: ", "").strip()
            response = parts[1].strip()
        else:
            query = content[:100]
            response = "[Could not parse response]"

        return {
            'query': query,
            'response': response,
            'timestamp': memory.timestamp,
            'source': 'hierarchical'
        }
    def get_dreams(self, limit: int = 2) -> List[str]:
        """Get recent dreams (optional symbolic memory)"""
        if hasattr(self.corpus_manager, 'get_dreams'):
            return self.corpus_manager.get_dreams(limit)
        return []

    def _get_memory_key(self, memory: Dict) -> str:
        """Get unique key for memory deduplication"""
        q = memory.get('query', '')[:50]
        r = memory.get('response', '')[:50]
        return f"{q}_{r}"

    # inside class MemoryCoordinator

    async def store_interaction(self, query: str, response: str, tags: List[str] = None):
        """Store interaction in all memory systems"""

        # 1. Store in short-term corpus
        self.corpus_manager.add_entry(query, response, tags)

        # 2. Store in Chroma (via store or collection)
        try:
            content = f"User: {query}\nAssistant: {response}"
            doc_id = str(uuid.uuid4())
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "tags": ",".join(tags or [])
            }

            if self.chroma_store:
                self.chroma_store.add_conversation_memory(query, response, metadata)

            elif self.chroma_collection:
                self.chroma_collection.add(
                    documents=[content],
                    metadatas=[metadata],
                    ids=[doc_id]
                )

            logger.debug(f"Stored in ChromaDB with ID: {doc_id}")

        except Exception as e:
            logger.error(f"Error storing in ChromaDB: {e}")

        # 3. Store in hierarchical memory
        if self.hierarchical_memory:
            try:
                await self.hierarchical_memory.store_interaction(query, response, tags)
            except Exception as e:
                logger.error(f"Error storing in hierarchical memory: {e}")

    async def get_memories(self, query: str, limit: int = 20) -> List[Dict]:
        """Get top memories for prompt builder use"""
        config = {
            'recent_count': 3,
            'semantic_count': 30,
            'hierarchical_count': 15,
            'max_memories': limit
        }

        results = await self.retrieve_relevant_memories(query, config)
        return results.get('memories', [])
