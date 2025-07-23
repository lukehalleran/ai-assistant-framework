# memory/memory_coordinator.py
import asyncio
import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from utils.logging_utils import get_logger, log_and_time
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from memory.memory_interface import HierarchicalMemorySystem
from processing.gate_system import MultiStageGateSystem

logger = get_logger("memory_coordinator")

class MemoryCoordinator:
    def __init__(self,
                 corpus_manager,
                 chroma_store: MultiCollectionChromaStore,
                 hierarchical_memory: Optional[HierarchicalMemorySystem] = None,
                 gate_system: Optional[MultiStageGateSystem] = None):
        self.corpus_manager = corpus_manager
        self.chroma_store = chroma_store
        self.hierarchical_memory = hierarchical_memory
        self.gate_system = gate_system

    @log_and_time("Retrieve Relevant Memories")
    async def retrieve_relevant_memories(self, query: str, config: Dict) -> Dict:
        logger.debug(f"Retrieving memories for query: {query[:50]}...")

        very_recent = self.corpus_manager.get_recent_memories(config.get('recent_count', 3))
        semantic_memories = await self._get_semantic_memories(query, config.get('semantic_count', 30))
        hierarchical_memories = []

        if self.hierarchical_memory:
            try:
                hierarchical_memories = await self.hierarchical_memory.retrieve_relevant_memories(
                    query,
                    max_memories=config.get('hierarchical_count', 15)
                )
            except Exception as e:
                logger.error(f"Error getting hierarchical memories: {e}")

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
        """Retrieve semantic memory chunks using the new multi-collection store"""
        try:
            results = self.chroma_store.search_all(query, n_results_per_type=n_results)
            semantic_chunks = results.get("semantic", []) + results.get("facts", [])

            memories = []
            for item in semantic_chunks:
                meta = item.get('metadata', {})
                memories.append({
                    'query': meta.get('query', item['content'][:100]),
                    'response': meta.get('response', ""),
                    'timestamp': meta.get('timestamp', datetime.now()),
                    'source': 'semantic',
                    'relevance_score': item.get('relevance_score', 0.5)
                })

            return memories

        except Exception as e:
            logger.error(f"Error in semantic memory retrieval: {e}")
            return []

    async def _combine_memories(self, very_recent, semantic, hierarchical, query, config) -> List[Dict]:
        combined = []
        seen = set()

        for mem in very_recent:
            key = self._get_memory_key(mem)
            if key not in seen:
                mem['source'] = 'very_recent'
                mem['gated'] = False
                combined.append(mem)
                seen.add(key)

        candidates = []

        for mem in semantic:
            key = self._get_memory_key(mem)
            if key not in seen:
                candidates.append(mem)

        for h in hierarchical:
            if isinstance(h, dict) and 'memory' in h:
                mem = self._format_hierarchical_memory(h['memory'])
                key = self._get_memory_key(mem)
                if key not in seen:
                    mem['relevance_score'] = h.get('final_score', 0.5)
                    candidates.append(mem)

        if self.gate_system and candidates:
            gated = await self._gate_memories(query, candidates)
            for mem in gated:
                mem['gated'] = True
                combined.append(mem)
        else:
            for mem in candidates[:config.get('max_memories', 10)]:
                mem['gated'] = False
                combined.append(mem)

        return combined

    async def _gate_memories(self, query: str, memories: List[Dict]) -> List[Dict]:
        try:
            chunks = [{
                "content": f"User: {m['query']}\nAssistant: {m['response']}",
                "metadata": {"timestamp": m.get("timestamp", datetime.now())}
            } for m in memories]

            filtered = await self.gate_system.filter_memories(query, chunks)

            gated = []
            for chunk in filtered:
                parts = chunk["content"].split("\nAssistant: ", 1)
                if len(parts) == 2:
                    gated.append({
                        'query': parts[0].replace("User: ", "").strip(),
                        'response': parts[1].strip(),
                        'timestamp': chunk["metadata"].get("timestamp", datetime.now())
                    })
            return gated

        except Exception as e:
            logger.error(f"Error during memory gating: {e}")
            return memories[:5]

    def _get_memory_key(self, memory: Dict) -> str:
        q = memory.get('query', '')[:50]
        r = memory.get('response', '')[:50]
        return f"{q}_{r}"

    def _format_hierarchical_memory(self, memory) -> Dict:
        parts = memory.content.split('\nAssistant: ')
        if len(parts) == 2:
            return {
                'query': parts[0].replace("User: ", "").strip(),
                'response': parts[1].strip(),
                'timestamp': memory.timestamp,
                'source': 'hierarchical'
            }
        return {
            'query': memory.content[:100],
            'response': "[Could not parse response]",
            'timestamp': memory.timestamp,
            'source': 'hierarchical'
        }

    async def store_interaction(self, query: str, response: str, tags: List[str] = None):
        self.corpus_manager.add_entry(query, response, tags)
        logger.debug("[MEMORY] store_interaction was called.")
        try:
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response,
                "tags": ",".join(tags or [])
            }
            logger.debug(f"[MemoryCoordinator] Attempting to store memory:\nQuery: {query[:100]}\nResponse: {response[:100]}")
            self.chroma_store.add_conversation_memory(query, response, metadata)
            logger.debug("[MemoryCoordinator] ✅ Memory successfully stored")
        except Exception as e:
            logger.error(f"Error storing in ChromaDB: {e}")

        if self.hierarchical_memory:
            try:
                logger.debug(f"[MemoryCoordinator] Attempting to store memory:\nQuery: {query[:100]}\nResponse: {response[:100]}")

                await self.hierarchical_memory.store_interaction(query, response, tags)
                logger.debug("[MemoryCoordinator] ✅ Memory successfully stored")
            except Exception as e:
                logger.error(f"Error storing in hierarchical memory: {e}")

    def get_summaries(self, limit: int = 3) -> List[Dict]:
        summaries = self.corpus_manager.get_summaries(limit)
        return [{
            'content': s.get('response', ''),
            'timestamp': s.get('timestamp', datetime.now()),
            'type': 'summary',
            'tags': s.get('tags', [])
        } for s in summaries]

    def get_dreams(self, limit: int = 2) -> List[str]:
        if hasattr(self.corpus_manager, 'get_dreams'):
            dreams = self.corpus_manager.get_dreams(limit)
            return [{
                'content': d,
                'timestamp': datetime.now(),
                'source': 'dream'
            } for d in dreams]
        return []


    async def get_memories(self, query: str, limit: int = 20) -> List[Dict]:
        """Unified call for prompt builder to fetch top memories"""
        config = {
            'recent_count': 3,
            'semantic_count': 30,
            'hierarchical_count': 15,
            'max_memories': limit
        }
        results = await self.retrieve_relevant_memories(query, config)
        return results.get('memories', [])
