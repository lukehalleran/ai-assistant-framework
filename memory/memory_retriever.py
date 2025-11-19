# memory/memory_retriever.py
"""
Memory retrieval module.

Implements the MemoryRetrieverProtocol contract for retrieving memories.
"""

import os
import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from utils.logging_utils import get_logger
from config.app_config import (
    DEICTIC_THRESHOLD,
    NORMAL_THRESHOLD,
)

logger = get_logger("memory_retriever")


class MemoryRetriever:
    """
    Memory retrieval operations.

    Implements MemoryRetrieverProtocol contract.
    """

    def __init__(
        self,
        corpus_manager,
        chroma_store,
        gate_system=None,
        scorer=None,
        hybrid_retriever=None,
        time_manager=None,
    ):
        """
        Initialize MemoryRetriever.

        Args:
            corpus_manager: CorpusManager for JSON persistence
            chroma_store: MultiCollectionChromaStore for vector storage
            gate_system: Optional MultiStageGateSystem for filtering
            scorer: Optional MemoryScorer for ranking
            hybrid_retriever: Optional HybridRetriever for enhanced search
            time_manager: Optional TimeManager for timestamps
        """
        self.corpus_manager = corpus_manager
        self.chroma_store = chroma_store
        self.gate_system = gate_system
        self.scorer = scorer
        self.hybrid_retriever = hybrid_retriever
        self.time_manager = time_manager

        # State
        self.current_topic: str = "general"
        self.conversation_context: list = []

    def _get_memory_key(self, memory: Dict) -> str:
        """Generate unique key for deduplication."""
        mem_id = memory.get('id')
        if mem_id:
            return f"id:{mem_id}"

        ts = memory.get('timestamp')
        if ts:
            ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
            return f"ts:{ts_str}__{(memory.get('query', '') or '')[:30]}__{(memory.get('response', '') or '')[:30]}"

        # Fallback to content hash
        content = f"{memory.get('query', '')}__{memory.get('response', '')}"
        return f"hash:{hash(content)}__{content[:30]}__{content[-30:]}"

    def _parse_result(self, item: Dict, source: str, default_truth: float = 0.6) -> Dict:
        """Parse a result from ChromaDB into a standardized memory format."""
        if not isinstance(item, dict):
            logger.warning(f"[_parse_result] Expected dict, got {type(item)}")
            return {}

        meta = item.get('metadata', {}) or {}
        ts = meta.get('timestamp', datetime.now())
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except Exception:
                ts = datetime.now()

        tags = meta.get('tags', [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',') if t.strip()]

        return {
            'id': item.get('id', f"{source}::{uuid.uuid4().hex[:8]}"),
            'query': meta.get('query', item.get('content', '')[:100]),
            'response': meta.get('response', ''),
            'content': item.get('content', ''),
            'timestamp': ts,
            'source': source,
            'collection': source,
            'relevance_score': float(item.get('relevance_score', 0.5)),
            'metadata': meta,
            'tags': tags,
            'truth_score': float(meta.get('truth_score', default_truth)),
            'importance_score': float(meta.get('importance_score', 0.5)),
        }

    def _get_recent_conversations(self, k: int = 5) -> List[Dict]:
        """Get recent conversations from corpus (JSON)."""
        entries = self.corpus_manager.get_recent_memories(k) or []
        out: List[Dict] = []
        for e in entries:
            ts = e.get('timestamp', datetime.now())
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except Exception:
                    ts = datetime.now()

            out.append({
                'id': f"recent::{uuid.uuid4().hex[:8]}",
                'query': e.get('query', ''),
                'response': e.get('response', ''),
                'content': f"User: {e.get('query', '')}\nAssistant: {e.get('response', '')}",
                'timestamp': ts,
                'source': 'corpus',
                'collection': 'recent',
                'relevance_score': 0.9,  # fresh bias
                'metadata': {
                    'timestamp': ts.isoformat() if isinstance(ts, datetime) else str(ts),
                    'truth_score': e.get('truth_score', 0.6),
                    'importance_score': e.get('importance_score', 0.5),
                    'tags': e.get('tags', []),
                    'access_count': 0,
                },
                'tags': e.get('tags', []),
                'truth_score': e.get('truth_score', 0.6),
                'importance_score': e.get('importance_score', 0.5),
            })
        return out

    async def get_recent_facts(self, limit: int = 5) -> List[Dict]:
        """Fetch the most recent facts by timestamp."""
        try:
            recent = self.chroma_store.get_recent("facts", limit=limit)
            return recent or []
        except Exception as e:
            logger.debug(f"[MemoryRetriever][RecentFacts] retrieval failed: {e}")
            return []

    async def get_facts(self, query: str, limit: int = 8) -> List[Dict]:
        """Retrieve semantic facts relevant to query."""
        results: List[Dict] = []
        try:
            coll = self.chroma_store.collections.get("facts")

            # Semantic search only if there are rows
            if coll and coll.count() > 0:
                raw = self.chroma_store.query_collection(
                    "facts",
                    query_text=query or "",
                    n_results=min(max(1, limit), coll.count()),
                ) or []

                if not isinstance(raw, list):
                    raw = [raw]

                for item in raw:
                    if not isinstance(item, dict):
                        item = {"content": str(item)}
                    meta = item.get("metadata", {}) or {}
                    content = item.get("content") or meta.get("content") or ""
                    if not content:
                        continue
                    results.append({
                        "id": item.get("id"),
                        "content": content,
                        "confidence": float(meta.get("confidence", 0.6)),
                        "source": meta.get("source", "facts"),
                        "timestamp": meta.get("timestamp"),
                        "tags": meta.get("tags", []),
                        "metadata": meta,
                    })

            # Fallback to most recent if nothing semantic
            if not results and hasattr(self.chroma_store, "get_recent"):
                for item in self.chroma_store.get_recent("facts", limit) or []:
                    content = item.get("content") or ""
                    if not content:
                        continue
                    meta = item.get("metadata", {}) or {}
                    results.append({
                        "id": item.get("id"),
                        "content": content,
                        "confidence": float(meta.get("confidence", 0.6)),
                        "source": meta.get("source", "facts"),
                        "timestamp": meta.get("timestamp"),
                        "metadata": meta,
                    })

        except Exception as e:
            logger.debug(f"[Facts] retrieval error: {e}", exc_info=True)

        # Rank by confidence + light recency
        def _score(x: Dict) -> float:
            ts = x.get("timestamp")
            rec = 1.0
            try:
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                if ts:
                    age_h = (datetime.now() - ts).total_seconds() / 3600.0
                    rec = 1.0 / (1.0 + 0.05 * max(0.0, age_h))
            except Exception:
                pass
            return 0.7 * float(x.get("confidence", 0.6)) + 0.3 * rec

        results.sort(key=_score, reverse=True)
        return results[:limit]

    async def get_reflections(self, limit: int = 2) -> List[Dict]:
        """Fetch recent reflections."""
        out = []

        # Corpus-first
        try:
            cm = self.corpus_manager
            get_by_type = getattr(cm, "get_items_by_type", None)
            if callable(get_by_type):
                items = get_by_type("reflection", limit=limit * 2) or []
            else:
                get_summaries_of_type = getattr(cm, "get_summaries_of_type", None)
                if callable(get_summaries_of_type):
                    items = get_summaries_of_type(types=("reflection",), limit=limit * 2) or []
                else:
                    get_all = getattr(cm, "get_all", None) or getattr(cm, "get_recent_memories", None)
                    items = (get_all(limit * 10) if callable(get_all) else []) or []

            for n in items or []:
                if isinstance(n, dict):
                    t = (n.get("type") or "").strip().lower()
                    tags = n.get("tags") or []
                    if t == "reflection" or "type:reflection" in tags:
                        out.append({
                            "content": n.get("content", "").strip(),
                            "type": "reflection",
                            "tags": tags,
                            "source": n.get("source", "corpus"),
                            "timestamp": n.get("timestamp"),
                        })
                        if len(out) >= limit:
                            return out
        except Exception:
            pass

        # Semantic fallback
        try:
            coll = self.chroma_store.collections.get("reflections") if self.chroma_store else None
            if coll and coll.count() > 0:
                items = self.chroma_store.get_recent("reflections", limit=limit)
                for r in items or []:
                    txt = (r.get("content") if isinstance(r, dict) else str(r)).strip()
                    ts = (r.get("metadata") or {}).get("timestamp") if isinstance(r, dict) else None
                    if txt:
                        out.append({
                            "content": txt,
                            "type": "reflection",
                            "tags": ["source:semantic"],
                            "source": "semantic",
                            "timestamp": ts,
                        })
                        if len(out) >= limit:
                            break
        except Exception:
            pass

        return out[:limit]

    async def get_reflections_hybrid(self, query: str, limit: int = 3) -> List[Dict]:
        """Hybrid retrieval for reflections: n/3 recent + 2n/3 semantic."""
        if limit < 1:
            return []

        recent_budget = max(1, limit // 3)
        semantic_budget = limit - recent_budget

        # Get recent
        recent = await self.get_reflections(limit=recent_budget * 2)

        # Get semantic
        semantic = []
        if query and query.strip():
            try:
                results = self.chroma_store.query_collection(
                    'reflections', query, n_results=semantic_budget * 2
                )
                semantic = [
                    {
                        'content': r.get('content', ''),
                        'timestamp': r.get('metadata', {}).get('timestamp', datetime.now()),
                        'type': 'reflection',
                        'source': 'semantic'
                    }
                    for r in results
                ]
            except Exception:
                pass

        # Deduplicate
        def get_item_id(item):
            if not isinstance(item, dict):
                return str(item)[:50]
            ts = item.get("timestamp", "")
            ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
            content_prefix = (item.get("content", "") or "")[:50].strip()
            return f"ts:{ts_str}::{content_prefix}"

        result = []
        seen_ids = set()

        for item in recent[:recent_budget]:
            item_id = get_item_id(item)
            if item_id not in seen_ids:
                if isinstance(item, dict):
                    item['source'] = 'recent'
                result.append(item)
                seen_ids.add(item_id)

        remaining = limit - len(result)
        for item in semantic:
            if remaining <= 0:
                break
            item_id = get_item_id(item)
            if item_id not in seen_ids:
                result.append(item)
                seen_ids.add(item_id)
                remaining -= 1

        return result[:limit]

    def get_summaries(self, limit: int = 3) -> List[Dict]:
        """Retrieve recent summaries."""
        # Try ChromaDB first
        try:
            results = self.chroma_store.query_collection('summaries', '', n_results=limit)
            chroma_summaries = [{
                'content': r.get('content', ''),
                'timestamp': r.get('metadata', {}).get('timestamp', datetime.now()),
                'type': 'summary'
            } for r in results]
            if chroma_summaries:
                return chroma_summaries
        except:
            pass

        # Fallback to corpus manager
        try:
            if hasattr(self.corpus_manager, 'get_summaries'):
                corpus_summaries = self.corpus_manager.get_summaries(limit)
                if corpus_summaries:
                    return corpus_summaries
        except:
            pass

        return []

    def get_summaries_hybrid(self, query: str, limit: int = 4) -> List[Dict]:
        """Hybrid retrieval: n/4 recent + 3n/4 semantic, with deduplication."""
        if limit < 1:
            return []

        recent_budget = max(1, limit // 4)
        semantic_budget = limit - recent_budget

        # Fetch recent
        recent = []
        try:
            if hasattr(self.corpus_manager, 'get_summaries'):
                recent = self.corpus_manager.get_summaries(recent_budget * 2)
        except Exception:
            pass

        # Fetch semantic
        semantic = []
        if query and query.strip():
            try:
                results = self.chroma_store.query_collection(
                    'summaries', query, n_results=semantic_budget * 2
                )
                semantic = [
                    {
                        'content': r.get('content', ''),
                        'timestamp': r.get('metadata', {}).get('timestamp', datetime.now()),
                        'type': 'summary',
                        'source': 'semantic'
                    }
                    for r in results
                ]
            except Exception:
                pass

        # Deduplicate
        def get_item_id(item):
            if not isinstance(item, dict):
                return str(item)[:50]
            ts = item.get("timestamp", "")
            ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
            content_prefix = (item.get("content", "") or "")[:50].strip()
            return f"ts:{ts_str}::{content_prefix}"

        result = []
        seen_ids = set()

        for item in recent[:recent_budget]:
            item_id = get_item_id(item)
            if item_id not in seen_ids:
                if isinstance(item, dict):
                    item['source'] = 'recent'
                result.append(item)
                seen_ids.add(item_id)

        remaining = limit - len(result)
        for item in semantic:
            if remaining <= 0:
                break
            item_id = get_item_id(item)
            if item_id not in seen_ids:
                result.append(item)
                seen_ids.add(item_id)
                remaining -= 1

        # Fallback if semantic was empty
        if len(result) < limit and not semantic:
            for item in recent[recent_budget:]:
                if len(result) >= limit:
                    break
                item_id = get_item_id(item)
                if item_id not in seen_ids:
                    if isinstance(item, dict):
                        item['source'] = 'recent_fallback'
                    result.append(item)
                    seen_ids.add(item_id)

        return result[:limit]

    def get_dreams(self, limit: int = 2) -> List[Dict]:
        """Retrieve dream memories."""
        if hasattr(self.corpus_manager, 'get_dreams'):
            return [{
                'content': d,
                'timestamp': datetime.now(),
                'source': 'dream'
            } for d in (self.corpus_manager.get_dreams(limit) or [])]
        return []

    async def search_by_type(self, type_name: str, query: str = "", limit: int = 5) -> List[Dict]:
        """Search memories by type."""
        results = []
        try:
            if type_name in self.chroma_store.collections:
                if query:
                    raw = self.chroma_store.query_collection(type_name, query, n_results=limit)
                else:
                    raw = self.chroma_store.get_recent(type_name, limit=limit)

                for item in raw or []:
                    results.append(self._parse_result(item, type_name))
        except Exception as e:
            logger.error(f"[search_by_type] Error: {e}")

        return results

    async def _get_semantic_memories(self, query: str, n_results: int = 30) -> List[Dict]:
        """Get semantic memories using hybrid retrieval."""
        logger.info(f"[Semantic] Using hybrid retrieval for query: '{query[:50]}...'")

        try:
            if self.hybrid_retriever:
                hybrid_results = await self.hybrid_retriever.retrieve(query, limit=n_results)

                memories = []
                for result in hybrid_results:
                    hybrid_score = result.get("hybrid_score", 0.0)
                    keyword_score = result.get("keyword_score", 0.0)

                    boosted_score = hybrid_score + 0.3
                    if keyword_score > 0.5:
                        boosted_score = max(boosted_score, 0.6)

                    memory = {
                        "id": result.get("id", str(hash(str(result)))),
                        "content": result.get("content", ""),
                        "query": result.get("query", ""),
                        "response": result.get("response", ""),
                        "metadata": result.get("metadata", {}),
                        "collection": result.get("collection", "unknown"),
                        "final_score": boosted_score,
                        "semantic_score": result.get("semantic_score", 0.0),
                        "keyword_score": keyword_score,
                        "hybrid_score": hybrid_score,
                        "relevance": boosted_score
                    }
                    memories.append(memory)

                logger.info(f"[Semantic] Hybrid retrieval returned {len(memories)} memories")
                return memories

        except Exception as e:
            logger.error(f"[Semantic] Hybrid retrieval failed: {e}")

        # Fallback to basic search
        return await self._fallback_semantic_search(query, n_results)

    async def _fallback_semantic_search(self, query: str, n_results: int = 30) -> List[Dict]:
        """Fallback semantic search using ChromaDB."""
        memories: List[Dict] = []
        collections_to_query = ['conversations', 'summaries', 'reflections']

        try:
            batch_results = await self.chroma_store.query_multiple_collections(
                collections_to_query,
                query_text=query,
                n_results=n_results
            )

            for collection_name, results in batch_results.items():
                if not results:
                    continue

                for item in results:
                    if item is not None:
                        if not isinstance(item, dict):
                            item = {"content": str(item), "id": str(uuid.uuid4())}
                        memories.append(self._parse_result(item, collection_name))

        except Exception as e:
            logger.error(f"[Semantic] Fallback search failed: {e}")

        return memories

    async def _combine_memories(
        self,
        very_recent: List[Dict],
        semantic: List[Dict],
        hierarchical: List[Dict],
        query: str,
        config: Dict,
        bypass_gate: bool = False
    ) -> List[Dict]:
        """Combine memory pools with optional gating."""
        combined: List[Dict] = []
        candidates: List[Dict] = []
        seen = set()

        # Allow top-N recent memories straight through
        bypass_n = 2
        for mem in very_recent[:bypass_n]:
            key = self._get_memory_key(mem)
            if key not in seen:
                mem['source'] = mem.get('source', 'very_recent')
                mem['gated'] = False
                combined.append(mem)
                seen.add(key)

        # Rest of recent go to candidates
        for mem in very_recent[bypass_n:]:
            key = self._get_memory_key(mem)
            if key not in seen:
                candidates.append(mem)
                seen.add(key)

        # Semantic to candidates
        for mem in semantic:
            key = self._get_memory_key(mem)
            if key not in seen:
                candidates.append(mem)
                seen.add(key)

        # Hierarchical to candidates
        for h in hierarchical:
            if isinstance(h, dict) and 'memory' in h:
                mem = h['memory']
                key = self._get_memory_key(mem)
                if key not in seen:
                    mem['relevance_score'] = h.get('final_score', mem.get('relevance_score', 0.5))
                    candidates.append(mem)
                    seen.add(key)

        # Optional gating
        use_gate_system = self.gate_system and candidates and not bypass_gate

        if use_gate_system:
            gated = await self._gate_memories(query, candidates)
            for mem in gated:
                mem['gated'] = True
                combined.append(mem)
        else:
            cap = config.get('max_memories', 20)
            for mem in candidates[:cap]:
                mem['gated'] = False
                combined.append(mem)

        return combined

    async def _gate_memories(self, query: str, memories: List[Dict]) -> List[Dict]:
        """Apply gate while preserving original metadata."""
        try:
            def _gate_text(m: Dict) -> str:
                txt = (m.get('content') or '').strip()
                if txt:
                    return txt
                q = (m.get('query') or '').strip()
                a = (m.get('response') or '').strip()
                return f"User: {q}\nAssistant: {a}"

            chunks = [{
                "content": _gate_text(m)[:500],
                "metadata": {
                    "timestamp": m.get("timestamp", datetime.now()),
                    "truth_score": m.get('truth_score', 0.5),
                    "original_memory": m
                }
            } for m in memories]

            filtered = await self.gate_system.filter_memories(query, chunks)

            gated: List[Dict] = []
            for ch in filtered:
                md = ch.get("metadata", {}) or {}
                orig = md.get("original_memory")
                if isinstance(orig, dict):
                    orig['gated'] = True
                    gated.append(orig)
            return gated

        except Exception as e:
            logger.error(f"Gating error: {e}")
            return memories[:min(10, len(memories))]

    async def get_memories(
        self,
        query: str,
        limit: int = 20,
        topic_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Main retrieval pipeline: gather -> combine -> gate -> rank -> threshold -> update -> slice
        """
        # Import here to avoid circular imports
        from utils.query_checker import is_meta_conversational, _is_heavy_topic_heuristic
        from memory.memory_scorer import _is_deictic_followup

        # Check for meta-conversational query
        if is_meta_conversational(query):
            logger.debug(f"[MemoryRetriever] Detected meta-conversational query: {query[:50]}...")
            return await self._get_meta_conversational_memories(query, limit, topic_filter)

        # Dynamic configuration
        query_lower = query.lower()
        is_gym_health_query = any(word in query_lower for word in [
            'gym', 'workout', 'work out', 'exercise', 'fitness', 'bench', 'squat',
            'amantadine', 'medication', 'health', 'body', 'tired'
        ])

        if is_gym_health_query:
            cfg = {
                'recent_count': 0,
                'semantic_count': max(50, limit * 5),
                'max_memories': limit,
            }
        else:
            cfg = {
                'recent_count': 1,
                'semantic_count': max(30, limit * 2),
                'max_memories': limit,
            }

        topic_filter = topic_filter or self.current_topic
        if is_gym_health_query:
            topic_filter = None

        # Gather from both sources
        tasks = [
            asyncio.to_thread(self._get_recent_conversations, k=cfg['recent_count']),
            self._get_semantic_memories(query, n_results=cfg['semantic_count'])
        ]
        very_recent, semantic = await asyncio.gather(*tasks)

        hierarchical: List[Dict] = []

        # Topic pre-filter
        if topic_filter and topic_filter != "general":
            def _has_topic_tag(m: Dict) -> bool:
                tags = m.get('tags', [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",") if t.strip()]
                return f"topic:{topic_filter}" in tags

            filtered_recent = [m for m in very_recent if _has_topic_tag(m)]
            filtered_semantic = [m for m in semantic if _has_topic_tag(m)]

            total_before = len(very_recent) + len(semantic)
            total_after = len(filtered_recent) + len(filtered_semantic)

            if total_after > 0:
                very_recent = filtered_recent
                semantic = filtered_semantic

        # Combine with gating
        combined = await self._combine_memories(
            very_recent=very_recent,
            semantic=semantic,
            hierarchical=hierarchical,
            query=query,
            config={'max_memories': max(limit * 2, 30)},
            bypass_gate=is_gym_health_query
        )

        # Rank memories
        if self.scorer:
            ranked = self.scorer.rank_memories(combined, query)
        else:
            ranked = sorted(combined, key=lambda x: x.get('relevance_score', 0.5), reverse=True)

        # Threshold filtering
        is_deictic = _is_deictic_followup(query)
        cutoff = DEICTIC_THRESHOLD if is_deictic else NORMAL_THRESHOLD
        accepted = [m for m in ranked if m.get('final_score', 0.0) >= cutoff]

        if not accepted:
            accepted = ranked

        # Update truth scores
        top_memories = accepted[:limit]
        if self.scorer:
            self.scorer.update_truth_scores_on_access(top_memories)

        return top_memories

    async def _get_meta_conversational_memories(
        self,
        query: str,
        limit: int = 20,
        topic_filter: Optional[str] = None
    ) -> List[Dict]:
        """Special retrieval for meta-conversational queries about conversation history."""
        logger.debug("[MemoryRetriever] Using meta-conversational retrieval strategy")

        # Detect temporal window
        from utils.query_checker import extract_temporal_window
        temporal_days = extract_temporal_window(query)

        if temporal_days == 0:
            recent_limit = min(limit * 5, 50)
        elif temporal_days <= 2:
            recent_limit = min(limit * 3, 30)
        elif temporal_days <= 7:
            recent_limit = min(limit * 8, 80)
        else:
            recent_limit = min(limit * 15, 150)

        # Get recent episodic memories
        very_recent = self._get_recent_conversations(k=recent_limit)

        # Sort chronologically
        def _ts(m):
            ts = m.get('timestamp')
            if isinstance(ts, str):
                try:
                    return datetime.fromisoformat(ts)
                except:
                    return datetime.min
            return ts if isinstance(ts, datetime) else datetime.min

        very_recent.sort(key=_ts, reverse=True)

        # Apply gentle recency weighting
        now = datetime.now()
        for m in very_recent:
            ts = _ts(m)
            if ts:
                age_hours = max(0.0, (now - ts).total_seconds() / 3600.0)
                recency_score = 1.0 / (1.0 + 0.01 * age_hours)
                m['final_score'] = recency_score
                m['relevance_score'] = recency_score

        return very_recent[:limit]
