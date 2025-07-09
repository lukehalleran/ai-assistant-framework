# enhanced_reranking.py
# Complete reranking solution for memories, wiki, and semantic chunks

import asyncio
from typing import List, Dict, Tuple, Optional
import logging
from rerank import LLMReranker, MemoryPruner
from config import COSINE_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

class UnifiedGatingSystem:
    """Unified gating for all context types: memories, wiki, semantic chunks"""

    def __init__(self, model_manager, reranker: LLMReranker):
        self.model_manager = model_manager
        self.reranker = reranker
        self.pruner = MemoryPruner()

    async def gate_all_context(self,
                              query: str,
                              memories: List[Dict],
                              wiki_content: str,
                              semantic_chunks: List[Dict]) -> Dict:
        """
        Gate and rerank all context types
        Returns: {
            'memories': List[Dict],
            'wiki_snippet': str,
            'semantic_chunks': List[Dict]
        }
        """

        results = {}

        # 1. Rerank memories
        if memories:
            logger.debug(f"[Unified Gate] Processing {len(memories)} memories")
            results['memories'] = await self._rerank_memories(query, memories)
        else:
            results['memories'] = []

        # 2. Evaluate wiki relevance
        if wiki_content:
            logger.debug(f"[Unified Gate] Evaluating wiki content ({len(wiki_content)} chars)")
            results['wiki_snippet'] = await self._evaluate_wiki(query, wiki_content)
        else:
            results['wiki_snippet'] = ""

        # 3. Rerank semantic chunks
        if semantic_chunks:
            logger.debug(f"[Unified Gate] Processing {len(semantic_chunks)} semantic chunks")
            results['semantic_chunks'] = await self._rerank_chunks(query, semantic_chunks)
        else:
            results['semantic_chunks'] = []

        return results

    async def _rerank_memories(self, query: str, memories: List[Dict]) -> List[Dict]:
        """Enhanced memory reranking"""

        # Use the full reranking pipeline
        reranked = await self.reranker.rerank_memories(
            query,
            memories,
            max_final_memories=8,  # Get a few extra
            use_llm_stage=True
        )

        # Apply diversity pruning
        reranked = self.pruner.prune_by_diversity(reranked, similarity_threshold=0.7)

        # Final selection
        return reranked[:5]

    async def _evaluate_wiki(self, query: str, wiki_content: str) -> str:
        """Use LLM to evaluate wiki relevance and extract key parts"""

        if len(wiki_content) < 100:  # Too short, probably error
            return ""

        # For very casual queries, skip wiki
        casual_patterns = ['hey', 'hi', 'hello', 'how are you', 'whats up', 'sup']
        if any(pattern in query.lower() for pattern in casual_patterns):
            return ""

        eval_prompt = f"""Evaluate if this Wikipedia content is relevant to the user query.
If relevant, extract the most important 1-2 sentences.

User Query: "{query}"

Wikipedia Content: {wiki_content[:800]}

Respond with either:
- "NOT_RELEVANT" if the content doesn't help answer the query
- The 1-2 most relevant sentences if it is relevant

Response:"""

        try:
            response = await self.model_manager.generate_async(
                eval_prompt,
                model_name="gpt-3.5-turbo",
                temperature=0.1,
                max_tokens=150
            )

            # Parse response
            result = ""
            async for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        result += delta.content

            result = result.strip()

            if "NOT_RELEVANT" in result:
                logger.debug("[Wiki Gate] Content deemed not relevant")
                return ""
            else:
                logger.debug(f"[Wiki Gate] Extracted: {result[:100]}...")
                return result[:500]  # Limit length

        except Exception as e:
            logger.error(f"[Wiki Gate] Error: {e}")
            # Fallback: include first 300 chars
            return wiki_content[:300] + "..."

    async def _rerank_chunks(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Rerank semantic chunks using cross-encoder"""

        if len(chunks) <= 3:
            return chunks

        # Convert to format expected by reranker
        chunk_memories = []
        for chunk in chunks[:10]:  # Limit input
            chunk_memories.append({
                'content': f"{chunk.get('title', '')} {chunk.get('text', '')}",
                'relevance_score': chunk.get('score', 0.5),
                'original_chunk': chunk
            })

        # Use cross-encoder only (no LLM for chunks)
        reranked = await self.reranker.cross_encoder_rerank(query, chunk_memories)

        # Extract top chunks
        final_chunks = []
        for mem in reranked[:3]:
            final_chunks.append(mem['original_chunk'])

        return final_chunks


async def integrate_unified_gating():
    """Integration function to update your Daemon with unified gating"""

    print("🔧 Setting up unified gating system...")

    # Import required modules
    from gui import gate_system, model_manager
    from llm_gate_module import MultiStageGateSystem

    # Create reranker and unified system
    reranker = LLMReranker(model_manager)
    unified_gate = UnifiedGatingSystem(model_manager, reranker)

    # Store original methods
    original_filter_memories = gate_system.filter_memories
    original_filter_wiki = gate_system.filter_wiki_content
    original_filter_semantic = gate_system.filter_semantic_chunks

    # Create enhanced filter that uses unified gating
    async def enhanced_filter_all(query: str,
                                 memories: List[Dict] = None,
                                 wiki_content: str = None,
                                 semantic_chunks: List[Dict] = None) -> Dict:
        """Unified filtering for all context types"""

        # First apply original cosine filtering (higher threshold)
        old_threshold = gate_system.gate_system.cosine_threshold
        gate_system.gate_system.cosine_threshold = 0.35

        # Get initial filtering
        filtered_mems = []
        if memories:
            filtered_mems = await original_filter_memories(query, memories)

        # Restore threshold
        gate_system.gate_system.cosine_threshold = old_threshold

        # Apply unified gating
        results = await unified_gate.gate_all_context(
            query,
            filtered_mems,
            wiki_content or "",
            semantic_chunks or []
        )

        return results

    # Patch individual methods to use unified system
    async def new_filter_memories(query: str, memories: List[Dict]) -> List[Dict]:
        results = await enhanced_filter_all(query, memories=memories)
        return results['memories']

    async def new_filter_wiki(query: str, wiki_content: str) -> Tuple[bool, str]:
        results = await enhanced_filter_all(query, wiki_content=wiki_content)
        snippet = results['wiki_snippet']
        return (bool(snippet), snippet)

    async def new_filter_semantic(query: str, chunks: List[Dict]) -> List[Dict]:
        results = await enhanced_filter_all(query, semantic_chunks=chunks)
        return results['semantic_chunks']

    # Replace methods
    gate_system.filter_memories = new_filter_memories
    gate_system.filter_wiki_content = new_filter_wiki
    gate_system.filter_semantic_chunks = new_filter_semantic

    print("✅ Unified gating integration complete!")
    print("\n📊 What's changed:")
    print("- Memories: Cosine → Cross-encoder → LLM rerank → Top 5")
    print("- Wiki: LLM evaluation for relevance + extraction")
    print("- Semantic chunks: Cross-encoder reranking → Top 3")

    return unified_gate


# Quick test for all context types
async def test_unified_gating():
    """Test the unified gating with all context types"""

    from gui import hierarchical_memory, wiki_manager, get_relevant_context
    from models import model_manager

    test_query = "What programming languages do I know?"

    print(f"\n🧪 Testing unified gating with: '{test_query}'")

    # Get all context types
    memories = await hierarchical_memory.retrieve_relevant_memories(test_query, max_memories=50)

    # Use the actual wiki_manager from gui.py
    from gui import wiki_manager
    wiki_content = wiki_manager.search_summary("programming languages", sentences=10)

    # Get semantic chunks using the actual function from gui.py
    from gui import get_relevant_context
    semantic_chunks, _, _ = get_relevant_context(test_query)

    print(f"\n📥 Raw inputs:")
    print(f"- Memories: {len(memories)}")
    print(f"- Wiki: {len(wiki_content)} chars")
    print(f"- Semantic chunks: {len(semantic_chunks)}")

    # Create unified gate
    reranker = LLMReranker(model_manager)
    unified = UnifiedGatingSystem(model_manager, reranker)

    # Test gating
    results = await unified.gate_all_context(
        test_query,
        memories,
        wiki_content,
        semantic_chunks
    )

    print(f"\n📤 Gated outputs:")
    print(f"- Memories: {len(results['memories'])} (target: 5)")
    print(f"- Wiki: {len(results['wiki_snippet'])} chars")
    print(f"- Semantic chunks: {len(results['semantic_chunks'])} (target: 3)")

    if results['memories']:
        print(f"\n🏆 Top memory: {results['memories'][0].get('content', '')[:100]}...")


if __name__ == "__main__":
    # Choose which integration to run
    print("Choose integration type:")
    print("1. Basic memory-only reranking")
    print("2. Unified gating (memories + wiki + semantic)")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "2":
        asyncio.run(integrate_unified_gating())
        # asyncio.run(test_unified_gating())
    else:
        # Run the original integration
        from integrate_reranking import update_daemon_with_reranking
        asyncio.run(update_daemon_with_reranking())
