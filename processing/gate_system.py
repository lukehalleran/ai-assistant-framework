import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
import re
import json
import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from config.config import GATE_REL_THRESHOLD
from utils.logging_utils import log_and_time, get_logger

logger = get_logger(__name__)
logger.debug("llm gates.py is alive - now using cosine similarity")

@dataclass
class GateResult:
    relevant: bool
    confidence: float
    reasoning: str
    filtered_content: str = None

class CosineSimilarityGateSystem:
    """Fast cosine similarity-based gating system"""
    def __init__(self, model_manager, cosine_threshold=GATE_REL_THRESHOLD):
        self.model_manager = model_manager
        self.cosine_threshold = cosine_threshold
        self.cache = {}

        # Load embedding model for query encoding
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Optional: Load cross-encoder for reranking (CPU-friendly)
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.use_reranking = True
            logger.debug("[Cosine Gate] Cross-encoder loaded for reranking")
        except:
            self.cross_encoder = None
            self.use_reranking = False
            logger.debug("[Cosine Gate] No cross-encoder, using cosine only")

    def get_gating_model_name(self):
        return "cosine_similarity"

    def is_meta_query(self, query: str) -> bool:
        meta_keywords = ["memory", "memories", "what do you know", "how much do you remember",
                        "what have you stored", "what do you recall"]
        return any(kw in query.lower() for kw in meta_keywords)

    @log_and_time("Cosine Gate Content")
    async def gate_content_async(self, query: str, content: str, content_type: str) -> GateResult:
        """Gate content using cosine similarity"""
        cache_key = f"{query[:50]}:{content[:100]}:{content_type}"
        if cache_key in self.cache:
            logger.debug(f"[Cosine Gate] Cache hit for {content_type}")
            return self.cache[cache_key]

        try:
            # Encode query and content
            logger.debug(f"[Cosine Gate] Encoding query and {content_type} content")
            query_emb = self.embed_model.encode(query, convert_to_numpy=True)
            content_emb = self.embed_model.encode(content[:500], convert_to_numpy=True)

            # Calculate cosine similarity
            similarity = cosine_similarity([query_emb], [content_emb])[0][0]

            logger.debug(f"[Cosine Gate] {content_type} similarity: {similarity:.4f} (threshold: {self.cosine_threshold})")

            # Binary gating decision
            relevant = similarity >= self.cosine_threshold

            # Add keyword boosting for meta queries
            if self.is_meta_query(query) and any(word in content.lower() for word in ["memory", "stored", "recall"]):
                similarity += 0.1  # Boost score
                relevant = True
                logger.debug(f"[Cosine Gate] Meta query boost applied to {content_type}")

            result = GateResult(
                relevant=relevant,
                confidence=float(similarity),
                reasoning=f"Cosine similarity: {similarity:.3f}",
                filtered_content=content[:300] if relevant else None
            )

            self.cache[cache_key] = result
            logger.info(f"[Cosine Gate] {content_type}: sim={similarity:.3f}, relevant={relevant}, passed={'✓' if relevant else '✗'}")
            return result

        except Exception as e:
            logger.error(f"[Cosine Gate Error] {e}")
            return GateResult(relevant=True, confidence=0.5, reasoning="Gate failed, including by default")

class MultiStageGateSystem:
    def __init__(self, model_manager, cosine_threshold=GATE_REL_THRESHOLD):
        self.gate_system = CosineSimilarityGateSystem(model_manager, cosine_threshold)
        self.model_manager = model_manager
        self.embed_model = self.gate_system.embed_model

    def get_gating_model_name(self):
        return self.gate_system.get_gating_model_name()

    @log_and_time("Batch Cosine Gate Memories")
    async def batch_gate_memories(self, query: str, memories: List[Dict]) -> List[Dict]:
        """Batch gate memories using cosine similarity"""
        if not memories:
            return []

        try:
            # Encode query once
            query_emb = self.embed_model.encode(query, convert_to_numpy=True)

            # Extract content and encode all memories at once
            contents = [mem.get('content', '')[:500] for mem in memories[:50]]  # Limit to 50 for speed
            memory_embs = self.embed_model.encode(contents, convert_to_numpy=True, batch_size=32)

            # Calculate all cosine similarities at once
            similarities = cosine_similarity([query_emb], memory_embs)[0]

            # Filter based on threshold
            filtered_memories = []
            for mem, sim in zip(memories[:50], similarities):
                if sim >= self.gate_system.cosine_threshold:
                    mem['relevance_score'] = float(sim)
                    mem['filtered_content'] = mem.get('content', '')[:300]
                    filtered_memories.append(mem)
                else:
                    logger.debug(f"[Batch Gate] Memory filtered out: score={sim:.3f}")

            # Optional: Rerank using cross-encoder if available
            if self.gate_system.use_reranking and len(filtered_memories) > 5:
                pairs = [[query, mem.get('content', '')[:300]] for mem in filtered_memories]
                rerank_scores = self.gate_system.cross_encoder.predict(pairs)

                for mem, score in zip(filtered_memories, rerank_scores):
                    mem['rerank_score'] = float(score)

                # Sort by rerank score
                filtered_memories = sorted(filtered_memories, key=lambda x: x.get('rerank_score', 0), reverse=True)
                logger.debug(f"[Batch Gate] Reranked {len(filtered_memories)} memories")
            else:
                # Sort by cosine similarity
                filtered_memories = sorted(filtered_memories, key=lambda x: x['relevance_score'], reverse=True)

            logger.debug(f"[Batch Gate] Kept {len(filtered_memories)} of {len(memories[:50])} memories")
            return filtered_memories[:20]  # Return top 20

        except Exception as e:
            logger.error(f"[Batch Gate Error] {e}")
            # Fallback: return top memories with default score
            for i, mem in enumerate(memories[:10]):
                mem['relevance_score'] = 0.5 - (i * 0.05)  # Decreasing scores
                mem['filtered_content'] = mem.get('content', '')[:300]
            return memories[:10]

    @log_and_time("Filter Memories")
    async def filter_memories(self, query: str, memories: List[Dict]) -> List[Dict]:
        """Main entry point - always use batch gating for efficiency"""
        if not memories:
            logger.debug("[Memory Filter] No memories to filter")
            return []

        logger.info(f"[Memory Filter] Starting batch cosine gating for {len(memories)} memories")
        filtered = await self.batch_gate_memories(query, memories)
        logger.info(f"[Memory Filter] Gating complete: {len(filtered)}/{len(memories)} memories passed")
        return filtered

    @log_and_time("Filter Wiki")
    async def filter_wiki_content(self, query: str, wiki_content: str) -> Tuple[bool, str]:
        if not wiki_content:
            return False, ""

        try:
            result = await self.gate_system.gate_content_async(query, wiki_content, "wikipedia")

            # ✅ Standard pass-through if relevant and above threshold
            if result.relevant and result.confidence > self.gate_system.cosine_threshold:
                return True, result.filtered_content or wiki_content[:500]

            # ✅ Secondary check: keyword fallback if cosine fails
            if not result.relevant and result.confidence < 0.25:
                query_keywords = set(re.findall(r'\b\w{4,}\b', query.lower()))
                wiki_keywords = set(re.findall(r'\b\w{4,}\b', wiki_content.lower()[:500]))
                overlap = query_keywords & wiki_keywords

                if len(overlap) >= 2:
                    logger.debug(f"[Wiki Filter Fallback] Cosine low ({result.confidence:.3f}), but keyword match passes: {overlap}")
                    return True, wiki_content[:500]

            # ✅ Backup fallback: slightly more lenient cosine pass
            if result.relevant and result.confidence > 0.4:
                return True, result.filtered_content or wiki_content[:500]

            return False, ""

        except Exception as e:
            logger.debug(f"[Wiki Filter Error] {e}")
            return False, ""


    @log_and_time("Filter Semantic")
    async def filter_semantic_chunks(self, query: str, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            logger.debug("[Semantic Filter] No chunks to filter")
            return []

        try:
            logger.info(f"[Semantic Filter] Processing {len(chunks)} semantic chunks")
            query_emb = self.embed_model.encode(query, convert_to_numpy=True)

            scored_chunks = []
            for i, chunk in enumerate(chunks[:30]):
                text = chunk.get('text', '')
                title = chunk.get('title', '')
                content = f"{title} {text[:300]}"
                chunk_emb = self.embed_model.encode(content, convert_to_numpy=True)

                cosine_score = cosine_similarity([query_emb], [chunk_emb])[0][0]
                chunk['relevance_score'] = float(cosine_score)
                chunk['filtered_content'] = text
                scored_chunks.append(chunk)

                if i < 5:  # Log first 5 for debugging
                    logger.debug(f"[Semantic Filter] Chunk {i} ({title}): score={cosine_score:.3f}")

            # Prefilter
            prefiltered = [c for c in scored_chunks if c['relevance_score'] >= 0.25]
            logger.info(f"[Semantic Filter] Pre-filter: {len(prefiltered)}/{len(scored_chunks)} chunks above 0.25 threshold")

            if self.gate_system.use_reranking and len(prefiltered) > 5:
                logger.info(f"[Semantic Filter] Running cross-encoder reranking on {len(prefiltered)} chunks")
                pairs = [[query, f"{c['title']} {c['text'][:300]}"] for c in prefiltered]
                rerank_scores = self.gate_system.cross_encoder.predict(pairs)

                for chunk, score in zip(prefiltered, rerank_scores):
                    chunk['rerank_score'] = float(score)

                sorted_chunks = sorted(prefiltered, key=lambda x: x['rerank_score'], reverse=True)
            else:
                sorted_chunks = sorted(prefiltered, key=lambda x: x['relevance_score'], reverse=True)

            top_chunks = sorted_chunks[:5]
            logger.info(f"[Semantic Filter] Final result: {len(top_chunks)} chunks selected")

            # Log the selected chunks
            for i, chunk in enumerate(top_chunks):
                logger.debug(f"[Semantic Filter] Selected chunk {i}: {chunk.get('title', 'untitled')} (score: {chunk.get('rerank_score', chunk.get('relevance_score')):.3f})")

            return top_chunks

        except Exception as e:
            logger.error(f"[Semantic Filter Error] {e}")
            import traceback
            logger.error(traceback.format_exc())
            return chunks[:3]



class GatedPromptBuilder:
    def __init__(self, prompt_builder, model_manager):
        self.prompt_builder = prompt_builder
        self.gate_system = MultiStageGateSystem(model_manager)

    @log_and_time("Cosine Gated Prompt Build")
    async def build_gated_prompt(self, user_input, memories, summaries, dreams,
                             wiki_snippet="", semantic_snippet=None,
                             semantic_memory_results=None, time_context=None,
                             recent_conversations=None, model_name=None,
                             include_dreams=True, include_code_snapshot=False,
                             include_changelog=False, system_prompt="",
                             directives_file="structured_directives.txt"):

        """Build prompt with cosine-similarity gated context."""
        logger.debug(f"[Gated Prompt] Building prompt with cosine similarity gating")

        filtered_context = {}

        # Gating each piece of context
        try:
            filtered_context["memories"] = await self.gate_system.filter_memories(user_input, memories)
        except Exception as e:
            logger.debug(f"[Gated Prompt - Memory Error] {e}")
            filtered_context["memories"] = memories[:5]

        try:
            include_wiki, filtered_wiki = await self.gate_system.filter_wiki_content(user_input, wiki_snippet)
            filtered_context["wiki_snippet"] = filtered_wiki if include_wiki else ""
        except Exception as e:
            logger.debug(f"[Gated Prompt - Wiki Error] {e}")
            filtered_context["wiki_snippet"] = ""

        try:
            filtered_context["semantic_chunks"] = await self.gate_system.filter_semantic_chunks(user_input, semantic_snippet or [])
        except Exception as e:
            logger.debug(f"[Gated Prompt - Semantic Error] {e}")
            filtered_context["semantic_chunks"] = []

        # Build prompt
        return self.prompt_builder.build_prompt(
            user_input=user_input,
            memories=filtered_context.get("memories", []),
            summaries=summaries,
            dreams=dreams if include_dreams else [],
            wiki_snippet=filtered_context.get("wiki_snippet", ""),
            semantic_snippet=filtered_context.get("semantic_chunks", []),
            semantic_memory_results=semantic_memory_results,
            time_context=time_context,
            model_name=model_name or self.prompt_builder.tokenizer_manager.active_model_name or
                        self.prompt_builder.model_manager.get_active_model_name(),
            is_api=True,
            include_dreams=include_dreams,
            include_code_snapshot=include_code_snapshot,
            include_changelog=include_changelog,
            system_prompt=system_prompt,
            directives_file=directives_file
        )
