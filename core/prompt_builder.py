import logging

# Use the root logger or create a child logger that will inherit handlers
logger = logging.getLogger(__name__)
logger.debug("Unified Prompt Builer.py is alive")

import re
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time
from chromadb.config import Settings
import chromadb

from processing.gate_system import MultiStageGateSystem
from memory.memory_interface import HierarchicalMemorySystem, MemoryNode, MemoryType
from memory.memory_coordinator import MemoryCoordinator
from memory.corpus_manager import CorpusManager
from memory.storage.chroma_store import MultiCollectionChromaStore
from models.model_manager import ModelManager
from knowledge.semantic_search import semantic_search
from utils.logging_utils import log_and_time
from models.tokenizer_manager import TokenizerManager
from core.prompt import PromptBuilder
from utils.time_manager import TimeManager

class UnifiedHierarchicalPromptBuilder:
    """
    Unified builder that combines:
    - HierarchicalMemorySystem for structured memory access
    - MultiStageGateSystem for efficient local gating
    - PromptBuilder for final prompt construction
    """

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        model_manager,
        *,
        corpus_manager: CorpusManager = None,
        chroma_collection = None,
        chroma_store: Optional[MultiCollectionChromaStore] = None,
        hierarchical_memory: HierarchicalMemorySystem = None,
        gate_system: MultiStageGateSystem = None,
        tokenizer_manager=None,
        time_manager=None
    ):
        self.prompt_builder = prompt_builder
        self.model_manager  = model_manager
        self.tokenizer_manager = tokenizer_manager or TokenizerManager()
        self.time_manager = time_manager or TimeManager()
        # 1) gating
        self.gate_system = gate_system or MultiStageGateSystem(model_manager)

        # 2) short-term corpus
        self.corpus_manager = corpus_manager or CorpusManager()
        self.chroma_store = chroma_store or MultiCollectionChromaStore()
        # 3) semantic memory (ChromaDB)
        settings = Settings(
            persist_directory="./chroma_convos",
            anonymized_telemetry=False
        )
        client = chromadb.PersistentClient(settings=settings)

        # give it a clear name for just your conversation embeddings
        self.chroma_collection = (
            chroma_collection
            or client.get_or_create_collection(name="daemon_convos")
        )

        # 4) hierarchical memory
        self.hierarchical_memory = (
            hierarchical_memory
            or HierarchicalMemorySystem(
                model_manager=model_manager,
                chroma_store=self.chroma_store
            )
        )


        # 5) now wire them all into the coordinator
        self.memory_system = MemoryCoordinator(
            corpus_manager       = self.corpus_manager,
            chroma_store         = self.chroma_store,
            chroma_collection    = self.chroma_collection,
            hierarchical_memory  = self.hierarchical_memory,
            gate_system          = self.gate_system,
        )

        # keep the rest of your existing init…
        self.model_name = self.model_manager.get_active_model_name()
        self.debug      = True

        self.debug = True  # Enable debugging
    def get_token_count(self, text: str, model_name: str = "gpt2") -> int:
        tokenizer = self.tokenizer_manager.get_tokenizer(model_name)
        if tokenizer is None:
            return 0  # fallback or warning log?
        return len(tokenizer.encode(text, truncation=False))

    @log_and_time("Build async hierichical prompt")
    async def build_hierarchical_prompt(
        self,
        user_input: str,
        include_dreams: bool = True,
        include_wiki: bool = True,
        include_semantic: bool = True,
        semantic_chunks: List[Dict] = None,
        wiki_content: str = "",
        system_prompt: str = "",
        directives_file: str = "structured_directives.txt",
        semantic_memory_results: Optional[Dict] = None,
        time_context: Optional[Dict] = None,
        override_memories: Optional[List[Dict]] = None,
        recent_conversations: Optional[List[Dict]] = None
    ) -> str:
        logger.debug(f"[PROMPT] Building prompt for input: {user_input}")

        start_time = time.time()
        recent_conversations = recent_conversations or []
        context_sources = {}
        MAX_TOKENS = 4096
        RESERVED_FOR_OUTPUT = 512
        token_budget = MAX_TOKENS - RESERVED_FOR_OUTPUT
        current_token_total = 0
        prompt_blocks = []

        # === Step 1: Retrieve memories ===
        if override_memories is not None:
            # Use provided memories but still process wiki and semantic
            formatted_memories = override_memories
            summaries, dreams = [], []
            gated_memories = []  # Empty since we're using override_memories
            logger.debug(f"Using {len(formatted_memories)} override memories")
        else:
            memories = await self._retrieve_memories(user_input)
            summaries = await self._retrieve_summaries()
            dreams = await self._retrieve_dreams() if include_dreams else []

            logger.debug(f"📊 Retrieved: {len(memories)} memories, {len(summaries)} summaries, {len(dreams)} dreams")

            # Gate the memories
            try:
                gated_memories = await self.gate_system.filter_memories(user_input, memories)
            except Exception as e:
                logger.error("❌ Failed to gate memories")
                logger.error(traceback.format_exc())
                gated_memories = []


            # Format memories
            formatted_memories = []
        current_token_total = 0
        model_name = self.model_manager.get_active_model_name()

        for i, mem in enumerate(gated_memories):
            try:
                memory_obj = mem.get("memory")
                content = (
                    mem.get("filtered_content")
                    or mem.get("content")
                    or getattr(memory_obj, "content", "")
                )
                mem_tokens = self.get_token_count(content, model_name)

                if current_token_total + mem_tokens >= token_budget:
                    break  # 🔒 Stop if over budget

                current_token_total += mem_tokens
                timestamp = getattr(memory_obj, "timestamp", None)

                # Extract user/assistant format if available
                parts = content.split("Assistant:")
                if len(parts) == 2:
                    query = parts[0].replace("User:", "").strip()
                    response = parts[1].strip()
                else:
                    query = content[:100]
                    response = content[100:]

                formatted_memories.append({
                    "timestamp": timestamp,
                    "query": query,
                    "response": response
                })

            except Exception as e:
                logger.debug(f"❌ Error formatting memory {i}: {e}")


        # === Step 2: Gate wiki and semantic content (always do this) ===
        gate_start = time.time()

        gated_wiki = ""
        if include_wiki and wiki_content:
            wiki_relevant, wiki_snippet = await self.gate_system.filter_wiki_content(user_input, wiki_content)
            if wiki_relevant:
                gated_wiki = wiki_snippet

        # Ensure semantic chunks are present
        context_sources = {}
        semantic_chunks = context_sources.get("semantic_chunks", [])
        if not semantic_chunks:
            semantic_chunks = semantic_search(user_input, top_k=15)
            context_sources["semantic_chunks"] = semantic_chunks

        gated_chunks = []
        if include_semantic and semantic_chunks:
            gated_chunks = await self.gate_system.filter_semantic_chunks(user_input, semantic_chunks)
            # Wrap in dicts so downstream code expects .get("content")
            gated_chunks = [{"content": chunk} if isinstance(chunk, str) else chunk for chunk in gated_chunks]

        gate_time = time.time() - gate_start
        logger.debug(f"[PROMPT] After gating: wiki={'yes' if gated_wiki else 'no'}, {len(gated_chunks)} chunks")



        # === Step 3: Final prompt build ===
        prompt_start = time.time()

        if not semantic_memory_results and semantic_chunks:
            semantic_memory_results = {
                "documents": [
                    {
                        "title": chunk.get("title", ""),
                        "text": chunk.get("text", ""),
                        "filtered_content": chunk.get("text", "")[:300]
                    }
                    for chunk in semantic_chunks
                ]
            }

       # ✅ Normalize semantic_memory_results format
        if isinstance(semantic_memory_results, list):
            semantic_memory_results = {"documents": semantic_memory_results}
        elif not isinstance(semantic_memory_results, dict):
            semantic_memory_results = {"documents": []}
        elif "documents" not in semantic_memory_results:
            # Handle case where it's a dict but missing 'documents' key
            semantic_memory_results = {"documents": []}

        # Add safety check before debug logging
        if semantic_memory_results.get('documents'):
            logger.debug(f"[DEBUG] semantic_memory_results sample: {semantic_memory_results['documents'][0]}")
        else:
            logger.debug("[DEBUG] semantic_memory_results is empty")

        prompt = await self.prompt_builder.build_gated_prompt(
            model_name=self.model_manager.get_active_model_name(),
            user_input=user_input,
            memories=formatted_memories,
            summaries=summaries,
            dreams=dreams if include_dreams else [],
            wiki_snippet=gated_wiki,
            semantic_snippet=gated_chunks,
            semantic_memory_results=semantic_memory_results,
            time_context=time_context,
            include_dreams=include_dreams,
            include_code_snapshot=False,
            include_changelog=False,
            system_prompt=system_prompt,
            directives_file=directives_file,
            recent_conversations=recent_conversations,
        )

        prompt_time = time.time() - prompt_start
        total_time = time.time() - start_time

        logger.debug(f"⚡ Gating completed in {gate_time:.2f}s")
        logger.debug(f"📝 Prompt built in {prompt_time:.2f}s")
        logger.debug(f"⏱️ Total build time: {total_time:.2f}s")
        logger.debug(f"[MEMORY] Final token usage: {current_token_total} / {token_budget}")


        #logger.debug(f"📏 Final prompt length: {len(prompt)} chars")


        return prompt
    @log_and_time("Async retrieive memories")
    async def _retrieve_memories(self, user_input: str) -> List[Dict]:
        try:
            episodic_results = self.memory_system.chroma_collection.query(
                query_texts=[user_input], n_results=10, where={"type": "episodic"}
            )
            semantic_results = self.memory_system.chroma_collection.query(
                query_texts=[user_input], n_results=5, where={"type": "semantic"}
            )

            episodic_docs = episodic_results["documents"][0] if episodic_results["documents"] else []
            episodic_meta = episodic_results["metadatas"][0] if episodic_results["metadatas"] else []

            semantic_docs = []
            semantic_meta = []

            if semantic_results.get("documents") and semantic_results["documents"]:
                semantic_docs = semantic_results["documents"][0]

            if semantic_results.get("metadatas") and semantic_results["metadatas"]:
                semantic_meta = semantic_results["metadatas"][0]

            memories = []

            for doc, meta in zip(episodic_docs, episodic_meta):
                if isinstance(doc, list): doc = doc[0]
                memories.append({
                    'content': doc,
                    'timestamp': meta.get('timestamp'),
                    'importance': meta.get('importance', 0.5),
                    'type': 'episodic'
                })

            for doc, meta in zip(semantic_docs, semantic_meta):
                if isinstance(doc, list): doc = doc[0]
                memories.append({
                    'content': doc,
                    'timestamp': meta.get('timestamp'),
                    'importance': meta.get('importance', 0.7),
                    'type': 'semantic'
                })
            logger.debug(f"[GATED MEMS] Type: {type(memories)} | Content: {memories}")
            return memories

        except Exception as e:
            import traceback
            logger.debug(f"⚠️ Error retrieving memories: {e}")
            logger.debug("⛏️ Traceback:\n" + traceback.format_exc())
            return []

    async def _retrieve_summaries(self) -> List[str]:
        try:
            summaries = self.memory_system.get_summaries(limit=3)
            return [s.get('content', '') for s in summaries if s.get('content')]
        except Exception as e:
            if self.debug:
                logger.debug(f"⚠️ Error retrieving summaries: {e}")
            return []

    async def _retrieve_dreams(self) -> List[str]:
        try:
            dreams = self.memory_system.get_dreams(limit=2)
            return [d.get('content', '') for d in dreams if d.get('content')]
        except Exception as e:
            if self.debug:
                logger.debug(f"⚠️ Error retrieving dreams: {e}")
            return []

