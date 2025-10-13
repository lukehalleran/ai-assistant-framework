"""
# memory/memory_coordinator.py

Module Contract
- Purpose: Central coordinator for conversational memory. Persists each turn, retrieves relevant memories (recent + semantic), runs shutdown reflections, and synthesizes/stores block summaries at shutdown.
- Inputs:
  - store_interaction(query, response, tags?): adds a turn to corpus + Chroma
  - get_memories(query, limit, topic_filter?): unified retrieval/gating/ranking
  - process_shutdown_memory(): summarize blocks (size N) and extract end‑of‑session facts
  - run_shutdown_reflection(...): generate a short reflection at session end
- Outputs:
  - Lists of normalized memory dicts with scores/metadata; new summary/reflection/fact nodes in storage.
- Key collaborators:
  - CorpusManager (JSON short‑term store)
  - MultiCollectionChromaStore (semantic collections)
  - MemoryConsolidator (LLM summarization API, may be bypassed for micro‑summaries)
  - MultiStageGateSystem (cosine/rerank gating)
- Side effects:
  - Writes to corpus JSON and Chroma collections; updates access/score metadata.
- Async behavior:
  - Most public methods are async to coordinate with model calls and gating.
"""
from __future__ import annotations

import os
import uuid
import logging
import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta, timezone
from collections import deque

from utils.logging_utils import get_logger, log_and_time
from utils.topic_manager import TopicManager
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from memory.fact_extractor import FactExtractor
from memory.memory_consolidator import MemoryConsolidator
from memory.llm_fact_extractor import LLMFactExtractor
from memory.memory_scorer import MemoryScorer
from processing.gate_system import MultiStageGateSystem
from config.app_config import (
    RECENCY_DECAY_RATE,
    TRUTH_SCORE_UPDATE_RATE,
    TRUTH_SCORE_MAX,
    COLLECTION_BOOSTS,
    DEICTIC_THRESHOLD,
    NORMAL_THRESHOLD,
    DEICTIC_ANCHOR_PENALTY,
    DEICTIC_CONTINUITY_MIN,
    SCORE_WEIGHTS,
)

logger = get_logger("memory_coordinator")

REFLECTIONS_ENABLED      = os.getenv("REFLECTIONS_ENABLED", "1").strip() not in ("0","false","no","off")
REFLECTION_MAX_TOKENS    = int(os.getenv("REFLECTION_MAX_TOKENS", "300"))
REFLECTION_MODEL_ALIAS   = os.getenv("LLM_REFLECTION_ALIAS", os.getenv("LLM_SUMMARY_ALIAS", "gpt-4o-mini"))
REFLECTION_MIN_EXCHANGES = int(os.getenv("REFLECTION_MIN_EXCHANGES", "4"))
SUMMARIZE_AT_SHUTDOWN_ONLY = os.getenv("SUMMARIZE_AT_SHUTDOWN_ONLY", "1").strip().lower() not in ("0", "false", "no", "off")
SUMMARY_MAX_BLOCKS_PER_SHUTDOWN = int(os.getenv("SUMMARY_MAX_BLOCKS_PER_SHUTDOWN", "1"))
# Reflection content caps (0 = no cap)
REFLECTION_MAX_EXCERPTS   = int(os.getenv("REFLECTION_MAX_EXCERPTS", "0") or 0)
REFLECTION_MAX_SUMMARIES  = int(os.getenv("REFLECTION_MAX_SUMMARIES", "0") or 0)
# Fallback number of recent memories if session buffer absent (0 = all non-summary, non-reflection)
REFLECTION_FALLBACK_RECENT = int(os.getenv("REFLECTION_FALLBACK_RECENT", "0") or 0)
# Facts extraction policy: default is shutdown-only (no per-turn extraction)
FACTS_EXTRACT_EACH_TURN = os.getenv("FACTS_EXTRACT_EACH_TURN", "0").strip().lower() not in ("0", "false", "no", "off")

# ---------------------------
# Heuristics & token helpers
# ---------------------------

DEICTIC_HINTS = ("explain", "that", "it", "this", "again", "another way", "different way")

def _is_deictic_followup(q: str) -> bool:
    ql = (q or "").lower()
    return any(h in ql for h in DEICTIC_HINTS)

STOPWORDS = set("""
the a an to of in on for with and or if is are was were be been being by at from as this that it its
""".split())

def _salient_tokens(text: str, k: int = 12) -> set:
    toks = re.findall(r"[a-zA-Z0-9\+\-\*/=\^()]+", (text or "").lower())
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 1]
    freq: Dict[str, int] = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    return {t for t, _ in sorted(freq.items(), key=lambda x: (-x[1], -len(x[0])))[:k]}

def _num_op_density(text: str) -> float:
    if not text:
        return 0.0
    nums = len(re.findall(r"\b\d+(?:\.\d+)?\b", text))
    ops = len(re.findall(r"[\+\-\*/=\^]", text))
    toks = max(1, len(re.findall(r"\w+", text)))
    return (nums + ops) / toks

def _analogy_markers(text: str) -> int:
    if not text:
        return 0
    t = text.lower()
    markers = ["it's like", "its like", "imagine", "picture this", "as if", "like when", "metaphor", "analogy"]
    return sum(1 for m in markers if m in t)

def _build_anchor_tokens(conv: list, maxlen: int = 20) -> set:
    """Pull anchor tokens from the last exchange with better math handling."""
    anchors = set()
    if conv:
        last = conv[-1]
        blob = f"{last.get('query','')} {last.get('response','')}"
        math_patterns = [
            r"[a-zA-Z]\([a-zA-Z]\)",       # f(x)
            r"\d+[a-zA-Z]\^\d+",           # 7x^4
            r"\d+[a-zA-Z]\d*",             # 7x4, 9x
            r"[a-zA-Z]′\([a-zA-Z]\)",      # f'(x)
            r"\b\d+(?:\.\d+)?\b",          # numbers
            r"derivative|integral|function|equation|cdf|pdf|variance|expectation",
        ]
        for pattern in math_patterns:
            matches = re.findall(pattern, blob.lower())
            anchors.update(matches[:5])
        anchors |= _salient_tokens(blob, k=8)
    if len(anchors) > maxlen:
        anchors = set(list(anchors)[:maxlen])
    return anchors

# ---------------------------
# Memory Coordinator
# ---------------------------

class MemoryCoordinator:
    """
    Single funnel for short/long-term memory fetch/store:
    - Stores interactions to corpus + Chroma with truth/importance/topic metadata
    - Retrieves from corpus (recent) and Chroma collections (semantic)
    - Combines → gates → ranks → returns top-K
    - Handles shutdown consolidation and fact extraction
    """

    def __init__(self,
                 corpus_manager,
                 chroma_store: MultiCollectionChromaStore,
                 gate_system: Optional[MultiStageGateSystem] = None,
                 topic_manager=None,
                 model_manager=None,
                 time_manager=None):
        self.corpus_manager = corpus_manager
        self.chroma_store = chroma_store
        self.gate_system = gate_system
        self.current_topic = "general"
        self.fact_extractor = FactExtractor()
        # Determine summary cadence from config or env (fallback to 10)
        try:
            from config.app_config import config as _app_cfg
            cfg_n = int(((_app_cfg.get('memory') or {}).get('summary_interval') or 10))
        except Exception:
            cfg_n = 10
        try:
            import os as _os
            env_n = int(_os.getenv('SUMMARY_EVERY_N', str(cfg_n)))
        except Exception:
            env_n = cfg_n
        self.consolidator = MemoryConsolidator(
            consolidation_threshold=max(1, int(env_n)),
            model_manager=model_manager
        )
        self.model_manager = model_manager
        self.scorer = MemoryScorer()
        self.topic_manager = topic_manager or TopicManager()
        self.conversation_context = deque(maxlen=5)
        self.access_history: Dict[str, int] = {}  # Track memory access for truth updates
        self.interactions_since_consolidation = 0
        self.time_manager = time_manager
        self.last_consolidation_time = datetime.now()
        # Mark the start of this application session; used to scope shutdown reflections/facts
        try:
            self.session_start = self._now()
        except Exception:
            from datetime import datetime as _dt
            self.session_start = _dt.now()

    # --------- time helpers (prefer TimeManager) ---------
    def _now(self):
        try:
            if self.time_manager is not None and hasattr(self.time_manager, "current"):
                return self.time_manager.current()
        except Exception:
            pass
        return datetime.now()

    def _now_iso(self):
        try:
            if self.time_manager is not None and hasattr(self.time_manager, "current_iso"):
                return self.time_manager.current_iso()
        except Exception:
            pass
        return self._now().isoformat()



    # ---------------------------
    # Scoring helpers
    # ---------------------------

    @log_and_time("Calc Truth Score")
    def _calculate_truth_score(self, query: str, response: str) -> float:
        """Calculate truth score based on response/continuity characteristics."""
        score = 0.5
        if len(response or "") > 200:
            score += 0.1
        if '?' in (query or ""):
            score += 0.1
        confirms = ('yes', 'correct', 'exactly', 'right', 'understood', 'makes sense', 'good point')
        if any(c in (response or "").lower() for c in confirms):
            score += 0.2
        # Continuity with previous response
        if self.conversation_context:
            last = self.conversation_context[-1]
            last_tokens = set((last.get('response', '') or '').lower().split()[:10])
            if any(t in last_tokens for t in (query or '').lower().split()):
                score += 0.15
        return min(score, 1.0)

    def _calculate_importance_score(self, content: str) -> float:
        """Estimate importance for retention prioritization."""
        score = 0.5
        text = content or ""
        if len(text) > 200:
            score += 0.1
        if '?' in text:
            score += 0.1
        important_keywords = ['important', 'remember', 'note', 'key', 'critical', 'essential', 'todo', 'directive']
        if any(kw in text.lower() for kw in important_keywords):
            score += 0.2
        return min(score, 1.0)

    def _update_truth_scores_on_access(self, memories: List[Dict]):
        """Reinforce truth scores for accessed memories and stamp metadata."""
        for mem in memories:
            mem_id = mem.get('id')
            if mem_id:
                self.access_history[mem_id] = self.access_history.get(mem_id, 0) + 1

            current_truth = float(mem.get('truth_score', 0.5))
            new_truth = min(TRUTH_SCORE_MAX, current_truth + TRUTH_SCORE_UPDATE_RATE)
            mem['truth_score'] = new_truth

            md = mem.setdefault('metadata', {})
            md['truth_score'] = new_truth
            md['access_count'] = md.get('access_count', 0) + 1
            md['last_accessed'] = datetime.now().isoformat()
    # In memory/memory_coordinator.py (add to the class)


    async def get_facts(self, query: str, limit: int = 8) -> List[Dict]:
        results: List[Dict] = []
        try:
            coll = self.chroma_store.collections.get("facts")

            # 1) Semantic search only if there are rows
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

            # 2) Fallback to most recent if nothing semantic
            if not results and hasattr(self.chroma_store, "get_recent"):
                fallback = []
                for item in self.chroma_store.get_recent("facts", limit) or []:
                    content = item.get("content") or ""
                    if not content:
                        continue
                    meta = item.get("metadata", {}) or {}
                    fallback.append({
                        "id": item.get("id"),
                        "content": content,
                        "confidence": float(meta.get("confidence", 0.6)),
                        "source": meta.get("source", "facts"),
                        "timestamp": meta.get("timestamp"),
                        "metadata": meta,
                    })
                results = fallback

        except Exception as e:
            logger.debug(f"[Facts] retrieval error: {e}", exc_info=True)

        # 3) Rank (confidence + light recency)
        def _score(x: Dict) -> float:
            ts = x.get("timestamp")
            rec = 1.0
            try:
                from datetime import datetime
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                if ts:
                    age_h = (datetime.now() - ts).total_seconds() / 3600.0
                    rec = 1.0 / (1.0 + 0.05 * max(0.0, age_h))
            except Exception:
                pass
            return 0.7 * float(x.get("confidence", 0.6)) + 0.3 * rec

        results.sort(key=_score, reverse=True)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Facts] Returning {len(results)} facts for query={query!r}")
            for i, it in enumerate(results[:5], 1):
                logger.debug(f"  #{i}: {it.get('content')!r} (conf={it.get('confidence')})")

        return results[:limit]





    # ---------------------------
    # Thread detection
    # ---------------------------

    def get_thread_context(self) -> Optional[Dict]:
        """
        Retrieve thread context from the most recent conversation.
        Returns thread metadata if a thread is active, None otherwise.
        """
        recent = self.corpus_manager.get_recent_memories(count=1)
        if not recent:
            return None

        last_conv = recent[0]
        thread_id = last_conv.get("thread_id")

        if not thread_id:
            return None

        return {
            "thread_id": thread_id,
            "thread_depth": last_conv.get("thread_depth", 1),
            "thread_started": last_conv.get("thread_started"),
            "thread_topic": last_conv.get("thread_topic"),
            "is_heavy_topic": last_conv.get("is_heavy_topic", False),
        }

    def _detect_or_create_thread(self, query: str, is_heavy: bool) -> Dict:
        """
        Detect if current query continues immediate previous conversation.
        Returns thread metadata dict with thread_id, depth, started, topic.
        Only checks the most recent conversation for strict consecutive threading.
        """
        from utils.query_checker import belongs_to_thread

        # Get only the most recent conversation (limit=1 for strict consecutive check)
        recent = self.corpus_manager.get_recent_memories(count=1)

        if not recent:
            # First conversation - create new thread
            thread_id = f"thread_{self._now().strftime('%Y%m%d_%H%M%S')}"
            logger.debug(f"[Thread] Creating new thread (first conversation): {thread_id}")
            return {
                "thread_id": thread_id,
                "depth": 1,
                "started": self._now_iso(),
                "topic": self.current_topic,
            }

        last_conv = recent[0]

        # Check if current query continues last conversation
        if belongs_to_thread(query, last_conv, current_topic=self.current_topic):
            # Continue existing thread
            thread_id = last_conv.get("thread_id")
            thread_depth = last_conv.get("thread_depth", 0) + 1
            thread_started = last_conv.get("thread_started")
            thread_topic = last_conv.get("thread_topic", self.current_topic)

            logger.debug(
                f"[Thread] Continuing thread {thread_id} at depth {thread_depth} "
                f"(topic: {thread_topic})"
            )

            return {
                "thread_id": thread_id,
                "depth": thread_depth,
                "started": thread_started,
                "topic": thread_topic,
            }
        else:
            # Topic switch or time gap - new thread
            thread_id = f"thread_{self._now().strftime('%Y%m%d_%H%M%S')}"
            logger.debug(
                f"[Thread] Creating new thread (break detected): {thread_id} "
                f"(previous: {last_conv.get('thread_id')})"
            )
            return {
                "thread_id": thread_id,
                "depth": 1,
                "started": self._now_iso(),
                "topic": self.current_topic,
            }

    # ---------------------------
    # Public API: store
    # ---------------------------

    # In memory_coordinator.py, update the store_interaction method
    # In memory_coordinator.py, update the store_interaction method with more debugging
    async def store_interaction(self, query: str, response: str, tags: Optional[List[str]] = None):
        """Persist a turn in both corpus & Chroma with computed metadata"""
        try:
            # Detect heavy topic before anything else (used for both thread detection and facts)
            from utils.query_checker import _is_heavy_topic_heuristic
            is_heavy = _is_heavy_topic_heuristic(query)

            # Thread detection: check if this continues previous conversation
            thread_info = self._detect_or_create_thread(query, is_heavy)

            # Add to corpus (JSON) with stable timestamp and thread metadata
            self.corpus_manager.add_entry(
                query, response, tags or [], timestamp=self._now(),
                thread_id=thread_info.get("thread_id"),
                thread_depth=thread_info.get("depth"),
                thread_started=thread_info.get("started"),
                thread_topic=thread_info.get("topic"),
                is_heavy_topic=is_heavy,
                topic=self.current_topic
            )

            # Update conversation context
            self.conversation_context.append({
                "query": query,
                "response": response,
                "timestamp": self._now()
            })

            # Topic detection
            if hasattr(self.topic_manager, "update_from_user_input"):
                topics = self.topic_manager.update_from_user_input(query)
                primary_topic = topics[0] if topics else "general"
            elif hasattr(self.topic_manager, "detect_topic"):
                primary_topic = self.topic_manager.detect_topic(f"{query} {response}")
            else:
                primary_topic = "general"

            # Ensure tags list exists and includes the topic
            tags = tags or []
            if f"topic:{primary_topic}" not in tags:
                tags.append(f"topic:{primary_topic}")

            # Calculate scores
            truth_score = self._calculate_truth_score(query, response)
            importance_score = self._calculate_importance_score(f"{query} {response}")

            # Create metadata with proper type checking
            # Note: Store full query/response in metadata for retrieval context
            raw_metadata = {
                "timestamp": self._now_iso(),
                "query": query if query else "",
                "response": response if response else "",
                "tags": ",".join(tags) if tags else "",
                "topic": primary_topic or "general",
                "truth_score": float(truth_score),
                "importance_score": float(importance_score),
                "access_count": 0,
                "last_accessed": self._now_iso(),
            }

            # Debug: log the raw metadata
            logger.debug(f"[MemoryCoordinator] Raw metadata: {raw_metadata}")

            # Filter out None values and ensure all values are of the correct type
            clean_metadata = {}
            for k, v in raw_metadata.items():
                logger.debug(f"[Metadata Check] Key: {k}, Value: {v}, Type: {type(v)}")
                if v is not None:
                    if isinstance(v, (str, int, float, bool)):
                        clean_metadata[k] = v
                    else:
                        # Convert to string if not a basic type
                        clean_metadata[k] = str(v)
                        logger.debug(f"[Metadata Conversion] Converted {k} from {type(v)} to string: {clean_metadata[k]}")
                else:
                    logger.warning(f"[Metadata Warning] Key {k} has None value, skipping")

            # Final validation
            for k, v in clean_metadata.items():
                if not isinstance(v, (str, int, float, bool)):
                    logger.error(f"[Metadata Error] Key {k} still has invalid type: {type(v)} after cleaning")

            # Debug: log the clean metadata
            logger.debug(f"[MemoryCoordinator] Clean metadata: {clean_metadata}")

            # Store in Chroma
            memory_id = self.chroma_store.add_conversation_memory(query, response, clean_metadata)
            # --- Facts: by default, only run at shutdown. Per-turn is opt-in via FACTS_EXTRACT_EACH_TURN=1 ---
            if FACTS_EXTRACT_EACH_TURN:
                try:
                    # avoid wasted work on very short/empty outputs
                    if (response or "").strip() and len(response) >= 8:
                        before = datetime.now()
                        facts_before = getattr(self.chroma_store.collections["facts"], "count", lambda: 0)()
                        await self._extract_and_store_facts(query, response, truth_score)
                        facts_after = getattr(self.chroma_store.collections["facts"], "count", lambda: 0)()
                        logger.debug(
                            f"[MemoryCoordinator][Facts] stored {max(0, facts_after - facts_before)} "
                            f"facts in {(datetime.now()-before).total_seconds():.3f}s"
                        )
                    else:
                        logger.debug("[MemoryCoordinator][Facts] skipped (response too short/empty)")
                except Exception as fe:
                    logger.warning(f"[MemoryCoordinator][Facts] extraction/store failed: {fe}", exc_info=True)
            else:
                logger.debug("[MemoryCoordinator][Facts] per-turn extraction disabled (shutdown-only mode)")

            # Increment counter and (optionally) consolidate mid-session unless disabled
            if not SUMMARIZE_AT_SHUTDOWN_ONLY:
                self.interactions_since_consolidation += 1
                if self.interactions_since_consolidation >= self.consolidator.consolidation_threshold:
                    await self._consolidate_and_store_summary()
                    self.interactions_since_consolidation = 0

            logger.debug(f"[MemoryCoordinator] Stored memory {memory_id} (topic={primary_topic}, truth={truth_score:.2f})")
        except Exception as e:
            logger.error(f"Error storing interaction: {e}")
            import traceback
            traceback.print_exc()

    async def _consolidate_and_store_summary(self):
        """Consolidate recent memories and store the summary"""
        if not self.consolidator:
            logger.debug("[MemoryCoordinator] No consolidator available, skipping consolidation")
            return

        try:
            # Get recent memories for consolidation
            recent_memories = self.corpus_manager.get_recent_memories(
                count=self.consolidator.consolidation_threshold
            )

            if not recent_memories:
                logger.debug("[MemoryCoordinator] No recent memories to consolidate")
                return

            # Generate summary
            summary_node = await self.consolidator.consolidate_memories(recent_memories)

            if summary_node:
                # Store summary in corpus
                if hasattr(self.corpus_manager, 'add_summary'):
                    self.corpus_manager.add_summary(
                        content=summary_node.content,
                        tags=summary_node.tags,
                        timestamp=summary_node.timestamp
                    )

                # Store summary in Chroma
                if hasattr(self.chroma_store, 'add_to_collection'):
                    summary_metadata = {
                        "timestamp": summary_node.timestamp.isoformat(),
                        "type": "summary",
                        "importance_score": summary_node.importance_score,
                        "tags": ",".join(summary_node.tags) if summary_node.tags else "",
                        "memory_count": len(recent_memories)
                    }

                    self.chroma_store.add_to_collection(
                        "summaries",
                        summary_node.content,
                        summary_metadata
                    )

                self.last_consolidation_time = datetime.now()
                logger.info(f"[MemoryCoordinator] Consolidated {len(recent_memories)} memories into summary")
            else:
                logger.debug("[MemoryCoordinator] Consolidation returned no summary")
        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")


    # --- inside class MemoryCoordinator ---
    async def add_reflection(self, text: str, *, tags=None, source="reflection", timestamp=None) -> bool:
        if not text:
            return False
        ts = timestamp or self._now()
        tags = list(tags or [])
        if "type:reflection" not in tags:
            tags.append("type:reflection")
        if source and f"source:{source}" not in tags:
            tags.append(f"source:{source}")

        # 1) Corpus (reuse your add_summary schema for compatibility)
        try:
            if hasattr(self.corpus_manager, "add_summary"):
                self.corpus_manager.add_summary({
                    "content": text,
                    "timestamp": ts,
                    "type": "reflection",
                    "tags": tags
                })
        except Exception:
            pass

        # 2) Chroma (semantic)
        try:
            if hasattr(self.chroma_store, "add_to_collection"):
                md = {
                    "timestamp": ts.isoformat(),
                    "type": "reflection",
                    "tags": ",".join(tags),
                    "source": source,
                    "importance_score": 0.7,
                }
                if getattr(self.chroma_store, "collections", None) is not None \
                and "reflections" not in self.chroma_store.collections \
                and hasattr(self.chroma_store, "create_collection"):
                    try:
                        self.chroma_store.create_collection("reflections")
                    except Exception:
                        pass
                self.chroma_store.add_to_collection("reflections", text, md)
        except Exception:
            pass

        return True

    async def get_reflections(self, limit: int = 2):
        """
        Fetch recent reflections (prefer corpus; then semantic). Never silently rely on get_summaries()
        returning non-summary types.
        """
        out = []

        # 1) Corpus-first with multiple compatibility shims
        try:
            cm = self.corpus_manager

            # Preferred: a typed getter if your corpus has one
            get_by_type = getattr(cm, "get_items_by_type", None)
            if callable(get_by_type):
                items = get_by_type("reflection", limit=limit * 2) or []
            else:
                # Alternate: summaries-of-type if available
                get_summaries_of_type = getattr(cm, "get_summaries_of_type", None)
                if callable(get_summaries_of_type):
                    items = get_summaries_of_type(types=("reflection",), limit=limit * 2) or []
                else:
                    # Fallback: a generic getter (may be named differently in your codebase)
                    get_all = getattr(cm, "get_all", None) or getattr(cm, "get_recent_memories", None)
                    if callable(get_all):
                        # Pull a wider window then filter to reflections
                        items = get_all(limit * 10) or []
                    else:
                        items = []

            # Normalize/filter to reflections
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

        # 2) Semantic fallback (Chroma “reflections” collection)
        try:
            coll = self.chroma_store.collections.get("reflections") if getattr(self, "chroma_store", None) else None
            if coll and coll.count() > 0:
                items = self.chroma_store.get_recent("reflections", limit=limit)
                for r in items or []:
                    txt = (r.get("content") if isinstance(r, dict) else str(r)).strip()
                    ts  = (r.get("metadata") or {}).get("timestamp") if isinstance(r, dict) else None
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

    # Optional: generic search by collection/type name (used by some callers)
    async def search_by_type(self, type_name: str, query: str = "", limit: int = 5):
        coll_name = {
            "memories": "conversations",
            "conversations": "conversations",
            "facts": "facts",
            "summaries": "summaries",
            "reflections": "reflections",
        }.get((type_name or "").lower(), (type_name or "").lower())

        try:
            coll = self.chroma_store.collections.get(coll_name)
            if not coll or coll.count() == 0:
                return []
            results = self.chroma_store.query_collection(
                coll_name, query_text=query or "", n_results=min(limit, coll.count())
            ) or []
            if not isinstance(results, list):
                results = [results]
            # Ensure timestamp key surfaces when present in metadata
            norm = []
            for r in results[:limit]:
                if isinstance(r, dict):
                    if "timestamp" not in r and isinstance(r.get("metadata"), dict):
                        r["timestamp"] = r["metadata"].get("timestamp")
                norm.append(r)
            return norm
        except Exception:
            return []


    async def run_shutdown_reflection(self, session_conversations: list[dict] | None = None,
                                    session_summaries: list[dict | str] | None = None) -> bool:
        if not REFLECTIONS_ENABLED or not hasattr(self, "model_manager") or not hasattr(self.model_manager, "generate_once"):
            return False

        conv = list(session_conversations or [])
        sums = list(session_summaries or [])

        if not conv:
            try:
                corpus = list(self.corpus_manager.corpus)
                def _is_summary(e: Dict) -> bool:
                    typ = (e.get('type') or '').lower()
                    tags = e.get('tags') or []
                    return ("summary" in typ) or ("@summary" in tags) or ("type:summary" in tags)
                def _is_reflection(e: Dict) -> bool:
                    typ = (e.get('type') or '').lower()
                    tags = [str(t).lower() for t in (e.get('tags') or [])]
                    return (typ == 'reflection') or ('type:reflection' in tags)
                from datetime import datetime as _dt
                def _ts(e):
                    ts = e.get('timestamp')
                    if isinstance(ts, str):
                        try:
                            ts = _dt.fromisoformat(ts)
                        except Exception:
                            ts = _dt.min
                    if not hasattr(ts, 'isoformat'):
                        ts = _dt.min
                    return ts
                # Build session-only fallback: restrict to entries from this run
                non = [e for e in corpus if (not _is_summary(e)) and (not _is_reflection(e)) and _ts(e) >= self.session_start]
                non.sort(key=_ts, reverse=True)
                if REFLECTION_FALLBACK_RECENT and REFLECTION_FALLBACK_RECENT > 0:
                    conv = non[:REFLECTION_FALLBACK_RECENT]
                else:
                    conv = non
            except Exception:
                conv = []

        if len([c for c in conv if (c.get("query") or c.get("response"))]) < REFLECTION_MIN_EXCHANGES and not sums:
            return False

        def _slice(e):
            q = (e.get("query") or "").strip()
            a = (e.get("response") or "").strip()
            segs = []
            if q: segs.append(f"User: {q[:240]}")
            if a: segs.append(f"Assistant: {a[:300]}")
            return "\n".join(segs)

        conv_blocks = [_slice(e) for e in conv if (e.get("query") or e.get("response"))]
        sum_texts = []
        for s in sums:
            if isinstance(s, dict):
                sum_texts.append(str(s.get("content") or s.get("text") or "").strip())
            elif isinstance(s, str):
                sum_texts.append(s.strip())

        prompt = (
            "You are a neutral QA reviewer for an AI assistant session.\n"
            "Return three sections (3–6 bullets each):\n"
            "1) What went well\n"
            "2) What to improve\n"
            "3) High-level insights (durable heuristics to carry forward)\n"
            "Avoid praise; be specific and actionable.\n\n"
        )
        # Apply configurable caps (0 = unlimited)
        if sum_texts:
            sum_use = sum_texts if REFLECTION_MAX_SUMMARIES <= 0 else sum_texts[:REFLECTION_MAX_SUMMARIES]
            prompt += "SESSION SUMMARIES:\n" + "\n\n".join(f"- {t}" for t in sum_use) + "\n\n"
        conv_use = conv_blocks if REFLECTION_MAX_EXCERPTS <= 0 else conv_blocks[-REFLECTION_MAX_EXCERPTS:]
        prompt += "CONVERSATION EXCERPTS:\n" + "\n\n".join(conv_use) + "\n\n"
        prompt += "Produce the three sections now."

        prev_model = None
        try:
            # capture current for restoration
            if hasattr(self.model_manager, "get_active_model_name"):
                prev_model = self.model_manager.get_active_model_name()

            # switch if alias is configured & known
            if hasattr(self.model_manager, "switch_model"):
                api_models = getattr(self.model_manager, "api_models", {}) or {}
                if REFLECTION_MODEL_ALIAS in api_models:
                    self.model_manager.switch_model(REFLECTION_MODEL_ALIAS)

            # call with supported signature
            text = await self.model_manager.generate_once(
                prompt,
                max_tokens=REFLECTION_MAX_TOKENS
            )
        finally:
            # best-effort restore previous model
            try:
                if prev_model and hasattr(self.model_manager, "switch_model"):
                    self.model_manager.switch_model(prev_model)
                    try:
                        cur = self.model_manager.get_active_model_name() if hasattr(self.model_manager, "get_active_model_name") else prev_model
                        logger.info(f"[Reflections] restored active model after shutdown reflection: {cur}")
                    except Exception:
                        pass
            except Exception:
                pass

        text = (text or "").strip()
        if not text:
            return False

        return await self.add_reflection(text, tags=["session:end"], source="shutdown")

    async def _extract_and_store_facts(self, query: str, response: str, truth_score: float):
        """Extract and store facts from a turn; robust to mixed return types."""
        try:
            logger.debug(f"[MemoryCoordinator] Extracting facts from query: {query[:100]}...")
            # Extract strictly from user input; ignore assistant response to avoid long-model overflows
            facts = await self.fact_extractor.extract_facts(query, "") or []
            total = len(facts)
            logger.debug(f"[MemoryCoordinator] Extracted {total} facts (raw)")

            def _to_dict(item):
                # Normalize MemoryNode / dict / str into a dict with content + metadata
                if isinstance(item, dict):
                    return {"content": item.get("content", ""), "metadata": item.get("metadata", {})}
                # MemoryNode-like
                content = getattr(item, "content", None)
                metadata = getattr(item, "metadata", None)
                if content is not None or metadata is not None:
                    return {"content": content or "", "metadata": metadata or {}}
                # string fallback
                return {"content": str(item or ""), "metadata": {}}

            stored = 0
            for idx, raw in enumerate(facts, 1):
                safe = _to_dict(raw)
                md = safe.get("metadata", {}) or {}

                # Prefer canonical triple if present
                subj = (md.get("subject") or md.get("subj") or "").strip()
                rel  = (md.get("relation") or md.get("rel") or "").strip()
                obj  = (md.get("object") or md.get("obj") or "").strip()

                if subj and rel and obj:
                    fact_text = f"{subj} | {rel} | {obj}"
                else:
                    # fallback to content or a short synthesis
                    fact_text = (safe.get("content") or "").strip()
                    if not fact_text:
                        # last ditch: try composing from query/response snippets
                        fact_text = (query + " -> " + response[:140]).strip()

                if not fact_text:
                    logger.debug(f"[MemoryCoordinator][Facts] Skipping empty fact at index {idx}")
                    continue

                # confidence/source defaults
                conf = float(md.get("confidence", truth_score))
                src  = md.get("source", "conversation")

                try:
                    self.chroma_store.add_fact(
                        fact=fact_text,
                        source=src,
                        confidence=conf,
                    )
                    stored += 1
                    logger.info(
                        f"[MemoryCoordinator][Facts] Stored {idx}/{total}: {fact_text} (conf={conf:.2f})"
                    )
                except Exception as inner:
                    logger.warning(
                        f"[MemoryCoordinator][Facts] add_fact failed at #{idx}: {inner}. "
                        f"fact_text={fact_text!r} meta={md}",
                        exc_info=True
                    )
                    continue

            logger.debug(f"[MemoryCoordinator][Facts] total stored this turn: {stored}")

        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}", exc_info=True)

    # ---------------------------
    # Public API: retrieve
    # ---------------------------

    async def get_memories(self, query: str, limit: int = 20, topic_filter: Optional[str] = None) -> List[Dict]:
        """
        Unified retrieval pipeline: gather → combine → gate → rank → (threshold) → update → (persist) → slice
        """
        cfg = {
            'recent_count': 5,
            'semantic_count': max(30, limit * 2),
            'max_memories': limit,
        }

        topic_filter = topic_filter or self.current_topic

        # 1) Gather from both sources
        very_recent = self._get_recent_conversations(k=cfg['recent_count'])
        semantic = await self._get_semantic_memories(query, n_results=cfg['semantic_count'])

        # Placeholder for hierarchical memories if/when your subsystem is available
        hierarchical: List[Dict] = []

        # 2) Topic pre-filter if specified
        if topic_filter and topic_filter != "general":
            def _has_topic_tag(m: Dict) -> bool:
                tags = m.get('tags', [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",") if t.strip()]
                return f"topic:{topic_filter}" in tags

            very_recent = [m for m in very_recent if _has_topic_tag(m)]
            semantic = [m for m in semantic if _has_topic_tag(m)]
            hierarchical = [m for m in hierarchical if _has_topic_tag(m)]

        # 3) Combine with gating (recent bypass N)
        combined = await self._combine_memories(
            very_recent=very_recent,
            semantic=semantic,
            hierarchical=hierarchical,
            query=query,
            config={'max_memories': max(limit * 2, 30)}
        )

        # 4) Rank memories
        ranked = self._rank_memories(combined, query)

        # 5) Optional acceptance threshold (configurable)
        is_deictic = _is_deictic_followup(query)
        cutoff = DEICTIC_THRESHOLD if is_deictic else NORMAL_THRESHOLD
        accepted = [m for m in ranked if m.get('final_score', 0.0) >= cutoff]
        # If threshold is too strict and empties list, fall back to ranked
        if not accepted:
            accepted = ranked

        # 6) Update truth scores for accessed memories
        top_memories = accepted[:limit]
        self._update_truth_scores_on_access(top_memories)

        # 7) (Optional) persist metadata updates back to store
        try:
            if hasattr(self.chroma_store, "bulk_update_metadata"):
                self.chroma_store.bulk_update_metadata(top_memories)  # no-op if unimplemented
        except Exception as e:
            logger.debug(f"[MemoryCoordinator] bulk_update_metadata skipped/failed: {e}")

        return top_memories

    # ---------------------------
    # Internals: gather
    # ---------------------------

    def _get_recent_conversations(self, k: int = 5) -> List[Dict]:
        """Get recent conversations from corpus (JSON)"""
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
    # In memory_coordinator.py, fix the _parse_result method
    def _parse_result(self, item: Dict, source: str, default_truth: float = 0.6) -> Dict:
        """Parse a result from ChromaDB into a standardized memory format"""
        # Make sure item is a dictionary
        if not isinstance(item, dict):
            logger.warning(f"[_parse_result] Expected dict, got {type(item)}: {item}")
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

    # In memory_coordinator.py, fix the _get_semantic_memories method
    async def _get_semantic_memories(self, query: str, n_results: int = 30) -> List[Dict]:
        """Get semantic memories from Chroma collections"""
        memories: List[Dict] = []
        # Ensure we're querying target collections (semantic across 3 DBs)
        collections_to_query = ['conversations', 'summaries', 'reflections']

        for collection_name in collections_to_query:
            try:
                # Check if collection exists
                if collection_name not in self.chroma_store.collections:
                    logger.debug(f"[Semantic] Collection {collection_name} not found")
                    continue

                # Get the collection count
                collection = self.chroma_store.collections[collection_name]
                count = collection.count()

                if count == 0:
                    logger.debug(f"[Semantic] Collection {collection_name} is empty")
                    continue

                logger.debug(f"[Semantic] Querying {collection_name} (has {count} docs)")

                # Query with proper parameters
                results = self.chroma_store.query_collection(
                    collection_name,
                    query_text=query,
                    n_results=min(n_results, count)  # Don't query for more than exists
                )

                # Ensure results is a list of dictionaries
                if results is None:
                    results = []
                elif not isinstance(results, list):
                    results = [results]

                for item in results:
                    if item is not None:
                        # Ensure item is a dictionary
                        if not isinstance(item, dict):
                            item = {"content": str(item), "id": str(uuid.uuid4())}
                        memories.append(self._parse_result(item, collection_name))

            except Exception as e:
                logger.error(f"[Semantic] Failed to query {collection_name}: {e}")
                import traceback
                traceback.print_exc()

        logger.info(f"[Semantic] Retrieved {len(memories)} total memories from all collections")
        return memories

    async def get_semantic_top_memories(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Return top-k semantic memories across conversations, summaries, reflections
        using the gate system's cosine score. No recent-corpus bypass.
        """
        try:
            raw = await self._get_semantic_memories(query, n_results=max(30, limit * 3))
        except Exception:
            raw = []

        if not raw:
            return []

        # Build chunks for gating with back-reference to original memory
        def _gate_text(m: Dict) -> str:
            txt = (m.get('content') or '').strip()
            if txt:
                return txt
            q = (m.get('query') or '').strip(); a = (m.get('response') or '').strip()
            return f"User: {q}\nAssistant: {a}"

        chunks = [{
            "content": _gate_text(m)[:500],
            "metadata": {"original_memory": m},
        } for m in raw]

        # If no gate_system, return a simple cap by initial relevance score
        if not self.gate_system:
            out = sorted(raw, key=lambda x: float(x.get('relevance_score', 0.5)), reverse=True)[:limit]
            for m in out:
                m['pre_gated'] = True
            return out

        # Run gate and pick top-k by gate score
        try:
            filtered = await self.gate_system.filter_memories(query, chunks)
        except Exception:
            filtered = chunks[:limit]

        # Propagate gate score + mark as pre_gated
        out: List[Dict] = []
        for ch in filtered[:limit]:
            md = ch.get('metadata', {}) or {}
            orig = md.get('original_memory')
            if not isinstance(orig, dict):
                continue
            # Prefer gate scores where available
            score = float(ch.get('relevance_score', ch.get('__score__', orig.get('relevance_score', 0.5))))
            orig = dict(orig)
            orig['relevance_score'] = score
            orig['pre_gated'] = True
            out.append(orig)

        # If fewer than desired passed gating, top up from highest-scoring raw results
        # This ensures the prompt's [RELEVANT MEMORIES] section has up to `limit` items.
        if len(out) < limit and raw:
            # Build a set of keys for quick de-duplication (prefer stable id; fall back to content)
            def _k(m: Dict) -> str:
                mid = str(m.get('id') or '').strip()
                if mid:
                    return f"id::{mid}"
                return f"content::{(m.get('content') or '').strip()[:160].lower()}"

            selected = {_k(m) for m in out}
            # Sort raw by any available score (relevance_score or rank), descending
            def _score(m: Dict) -> float:
                try:
                    return float(m.get('relevance_score', 0.0))
                except Exception:
                    return 0.0

            for cand in sorted(raw, key=_score, reverse=True):
                if len(out) >= limit:
                    break
                key = _k(cand)
                if not key or key in selected:
                    continue
                # Clone and mark as pre-gated so prompt builder doesn't re-gate and drop it
                c = dict(cand)
                c['pre_gated'] = True
                # Carry forward any initial score if present
                if 'relevance_score' not in c:
                    c['relevance_score'] = 0.0
                out.append(c)
                selected.add(key)

        return out[:limit]



    # ---------------------------
    # Internals: combine + gate
    # ---------------------------

    async def _combine_memories(self, very_recent: List[Dict], semantic: List[Dict],
                                hierarchical: List[Dict], query: str, config: Dict) -> List[Dict]:
        """Combine memory pools with (optional) gating; allow some recent to bypass."""
        combined: List[Dict] = []
        candidates: List[Dict] = []
        seen = set()

        # Allow top-N very recent memories straight through (ungated)
        bypass_n = 2
        for mem in very_recent[:bypass_n]:
            key = self._get_memory_key(mem)
            if key not in seen:
                mem['source'] = mem.get('source', 'very_recent')
                mem['gated'] = False
                combined.append(mem)
                seen.add(key)

        # The rest of the very recent go to candidates
        for mem in very_recent[bypass_n:]:
            key = self._get_memory_key(mem)
            if key not in seen:
                candidates.append(mem)
                seen.add(key)

        # semantic → candidates
        for mem in semantic:
            key = self._get_memory_key(mem)
            if key not in seen:
                candidates.append(mem)
                seen.add(key)

        # hierarchical (if present) → normalize & candidates
        for h in hierarchical:
            if isinstance(h, dict) and 'memory' in h:
                mem = self._format_hierarchical_memory(h['memory'])
                key = self._get_memory_key(mem)
                if key not in seen:
                    mem['relevance_score'] = h.get('final_score', mem.get('relevance_score', 0.5))
                    candidates.append(mem)
                    seen.add(key)

        # Optional gating for candidates
        if self.gate_system and candidates:
            gated = await self._gate_memories(query, candidates)
            for mem in gated:
                mem['gated'] = True
                combined.append(mem)
        else:
            # If no gate, just cap the number of candidates added
            cap = config.get('max_memories', 20)
            for mem in candidates[:cap]:
                mem['gated'] = False
                combined.append(mem)

        return combined

    async def _gate_memories(self, query: str, memories: List[Dict]) -> List[Dict]:
        """Apply gate while preserving original metadata."""
        try:
            # Keep original object by reference for robust rehydration
            def _gate_text(m: Dict) -> str:
                # Prefer explicit content when present (summaries/reflections/facts)
                txt = (m.get('content') or '').strip()
                if txt:
                    return txt
                # Fallback to Q/A rendering for conversation-style memories
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
            return memories[: min(10, len(memories))]

    # ---------------------------
    # Internals: ranker
    # ---------------------------

    def _rank_memories(self, memories: List[Dict], current_query: str) -> List[Dict]:
        """
        Score each memory using:
          - base relevance (+ collection/source boost)
          - recency (configurable decay)
          - truth / importance
          - continuity (token overlap + last-10m)
          - structure alignment (numeric/op density)
          - analogy penalty for mathy queries
          - anchor bonus (esp. for deictic follow-ups)
          - optional acceptance threshold (applied by caller after scoring)
        """
        if not memories:
            return []

        now = datetime.now()
        last_10m = now - timedelta(minutes=10)

        is_deictic = _is_deictic_followup(current_query)
        anchors = _build_anchor_tokens(list(self.conversation_context))
        cq_salient = _salient_tokens(current_query, k=12)
        cq_density = _num_op_density(current_query)

        for m in memories:
            # 1) base relevance with collection/source boost
            rel = float(m.get('relevance_score', 0.5))
            collection_key = m.get('collection', m.get('source', ''))
            if collection_key in COLLECTION_BOOSTS:
                rel += COLLECTION_BOOSTS[collection_key]

            # 2) recency with decay
            ts = m.get('timestamp')
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except Exception:
                    ts = now
            elif not isinstance(ts, datetime):
                ts = now
            age_hours = max(0.0, (now - ts).total_seconds() / 3600.0)
            recency = 1.0 / (1.0 + RECENCY_DECAY_RATE * age_hours)

            # 3) truth (access-aware)
            md = m.get('metadata', {}) or {}
            truth = float(m.get('truth_score', md.get('truth_score', 0.6)))
            access_count = int(md.get('access_count', 0))
            if access_count > 0:
                truth = min(TRUTH_SCORE_MAX, truth + (TRUTH_SCORE_UPDATE_RATE * access_count))

            # 4) importance
            importance = float(m.get('importance_score', md.get('importance_score', 0.5)))

            # 5) continuity (overlap + recency)
            # Include content for non Q/A memories (summaries/reflections)
            blob = (m.get('query', '') + ' ' + m.get('response', '') + ' ' + m.get('content', '')).lower()
            m_toks = set(re.findall(r"[a-zA-Z0-9\+\-\*/=\^()]+", blob))
            continuity = 0.0
            if ts >= last_10m:
                continuity += 0.1
            if cq_salient:
                overlap = len(cq_salient & m_toks) / max(1, len(cq_salient))
                continuity += 0.3 * overlap

            # 6) structural alignment
            m_density = _num_op_density(blob)
            density_alignment = 1.0 - min(1.0, abs(cq_density - m_density) * 3.0)
            structure = 0.15 * density_alignment

            # 7) penalties/bonuses
            penalty = 0.0
            if cq_density > 0.08 and _analogy_markers(blob) > 0 and "analogy" not in current_query.lower():
                penalty -= 0.1

            anchor_bonus = 0.0
            if anchors:
                anchor_overlap = len(anchors & m_toks) / max(1, len(anchors))
                if is_deictic:
                    if anchor_overlap < 0.05:
                        penalty -= DEICTIC_ANCHOR_PENALTY
                    else:
                        anchor_bonus += 0.2 * anchor_overlap
                else:
                    anchor_bonus += 0.1 * anchor_overlap

            # 8) tone adjustment
            if any(t in blob for t in ("idiot", "stupid", "dumb", "toddler")):
                truth = max(0.0, truth - 0.2)

            m['final_score'] = (
                SCORE_WEIGHTS.get('relevance', 0.35) * rel +
                SCORE_WEIGHTS.get('recency', 0.25) * recency +
                SCORE_WEIGHTS.get('truth', 0.20) * truth +
                SCORE_WEIGHTS.get('importance', 0.05) * importance +
                SCORE_WEIGHTS.get('continuity', 0.10) * continuity +
                structure +
                anchor_bonus +
                penalty
            )

            if logger.isEnabledFor(logging.DEBUG):
                m['debug'] = {
                    'rel': rel, 'recency': recency, 'truth': truth,
                    'importance': importance, 'continuity': continuity,
                    'structure': structure, 'anchor_bonus': anchor_bonus,
                    'penalty': penalty,
                }

            # extra guardrail for deictic drift
            if is_deictic and continuity < DEICTIC_CONTINUITY_MIN and anchor_bonus < 0.04:
                m['final_score'] *= 0.85

        memories.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)

        if logger.isEnabledFor(logging.DEBUG) and memories:
            logger.debug("\n[Ranker] Top 5 memories:")
            for i, mm in enumerate(memories[:5], 1):
                dbg = mm.get('debug', {})
                logger.debug(
                    f"  #{i}: score={mm.get('final_score', 0):.3f} "
                    f"(rel={dbg.get('rel', 0):.2f}, rec={dbg.get('recency', 0):.2f}, "
                    f"truth={dbg.get('truth', 0):.2f}, imp={dbg.get('importance', 0):.2f}, "
                    f"cont={dbg.get('continuity', 0):.2f}, struct={dbg.get('structure', 0):.2f}, "
                    f"anchor={dbg.get('anchor_bonus', 0):.2f}, pen={dbg.get('penalty', 0):.2f}) "
                    f"Q: {mm.get('query', '')[:48]!r}"
                )

        return memories

    # ---------------------------
    # Shutdown processing
    # ---------------------------

    async def process_shutdown_memory(self, session_conversations: list[dict] | None = None):
        """Create any due summaries in fixed-size blocks and run fact extraction.

        Behavior:
          - Let N = self.consolidator.consolidation_threshold (e.g., 3, 10, 20)
          - Let T = total non-summary conversation entries in the corpus
          - Let S = number of consolidator-produced summaries already stored
          - Target summaries = floor(T / N)
          - Generate (Target - S) new summaries, each summarizing a disjoint block
            of size N, from oldest to newest (so we never duplicate prior work).
        """
        try:
            # 1) Prep corpus slices and counts
            corpus = list(self.corpus_manager.corpus)

            def _is_summary(e: Dict) -> bool:
                typ = (e.get('type') or '').lower()
                tags = e.get('tags') or []
                return ("summary" in typ) or ("@summary" in tags) or ("type:summary" in tags)

            # Exclude summaries and reflections from conversation slices
            def _is_reflection(e: Dict) -> bool:
                typ = (e.get('type') or '').lower()
                tags = [str(t).lower() for t in (e.get('tags') or [])]
                return (typ == 'reflection') or ('type:reflection' in tags)

            non_summ = [e for e in corpus if (not _is_summary(e)) and (not _is_reflection(e))]
            # Sort non-summary items by ascending timestamp to get stable, non-overlapping blocks
            def _ts(e):
                ts = e.get('timestamp')
                from datetime import datetime
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts)
                    except Exception:
                        ts = datetime.min
                if not isinstance(ts, datetime):
                    ts = datetime.min
                return ts
            non_summ.sort(key=_ts)

            # count consolidator-produced summaries (tags are stable across runs)
            def _is_consolidator_summary(e: Dict) -> bool:
                tags = [str(t).lower() for t in (e.get('tags') or [])]
                typ  = (e.get('type') or '').lower()
                if not (("summary" in typ) or ("@summary" in tags) or ("type:summary" in tags)):
                    return False
                return ("source:consolidator" in tags) or ("summary:consolidated" in tags)

            consolidator_summaries = [e for e in corpus if _is_consolidator_summary(e)]
            # Track any known block indices for the CURRENT N via 'block_n:<N>:<idx>' tags
            def _parse_block_n(tags: list, n: int) -> int | None:
                try:
                    key = f"block_n:{n}:"
                    for t in (tags or []):
                        t = str(t).strip().lower()
                        if t.startswith(key):
                            return int(t.split(':', 2)[2])
                except Exception:
                    return None
                return None
            existing_block_indices = []
            # Determine N early to parse indices for the current cadence
            N = max(1, int(getattr(self.consolidator, 'consolidation_threshold', 10)))
            for s in consolidator_summaries:
                bi = _parse_block_n(s.get('tags') or [], N)
                if isinstance(bi, int):
                    existing_block_indices.append(bi)
            max_block_done = max(existing_block_indices) if existing_block_indices else -1

            T = len(non_summ)
            total_blocks = T // N

            # Compute missing block indices among [0, total_blocks)
            existing_set = set(int(x) for x in existing_block_indices if isinstance(x, int))
            all_indices = set(range(total_blocks))
            missing = sorted(all_indices - existing_set)

            # Choose newest missing blocks (from the tail) up to the per-shutdown cap
            cap = max(1, SUMMARY_MAX_BLOCKS_PER_SHUTDOWN)
            selected_blocks = missing[-cap:] if missing else []

            if selected_blocks:
                # 2) Generate summaries for selected blocks (newest missing first)
                for b in selected_blocks:
                    start = b * N
                    end = start + N
                    block = non_summ[start:end]
                    if not block:
                        break
                    content_list = []
                    for c in block:
                        q = (c.get('query') or '').strip()
                        a = (c.get('response') or '').strip()
                        if not (q or a):
                            continue
                        content_list.append({'content': f"User: {q}\nAssistant: {a}", 'q': q, 'a': a})
                    if not content_list:
                        continue
                    # Try consolidator API if available; otherwise do a local LLM summarize
                    summary_text: Optional[str] = None
                    if hasattr(self.consolidator, 'consolidate_memories'):
                        try:
                            summary_node = await self.consolidator.consolidate_memories(content_list)
                            if summary_node and getattr(summary_node, 'content', None):
                                summary_text = summary_node.content
                        except Exception:
                            summary_text = None
                    if summary_text is None:
                        try:
                            # Deterministic micro-summary for tiny blocks (avoids hallucination)
                            if len(content_list) <= 2:
                                def _clip(s, n=160):
                                    s = s or ''
                                    return s if len(s) <= n else (s[:n] + '…')
                                lines = []
                                for it in content_list:
                                    if it.get('q'):
                                        lines.append(f"- User: {_clip(it['q'])}")
                                    if it.get('a'):
                                        lines.append(f"- Assistant: {_clip(it['a'])}")
                                summary_text = "\n".join(lines).strip()
                            else:
                                # Strictly extractive prompt (no invention)
                                excerpts = "\n\n".join(x['content'] for x in content_list if x.get('content'))
                                prompt = (
                                    "You are an extractive note-taker. Using ONLY the EXCERPTS below, "
                                    "write 3–5 factual bullets. Do NOT infer or invent anything not present. "
                                    "If information is minimal, output 1–2 bullets that quote/paraphrase the text.\n\n"
                                    f"EXCERPTS:\n{excerpts}\n\nBullets (no headers):"
                                )
                                if hasattr(self, 'model_manager') and hasattr(self.model_manager, 'generate_once'):
                                    summary_text = await self.model_manager.generate_once(prompt, max_tokens=220)
                        except Exception:
                            summary_text = None
                    summary_text = (summary_text or '').strip()
                    if not summary_text:
                        continue

                    # Store into corpus (kept consistent with add_summary schema)
                    try:
                        # Deduplicate by content against existing summary nodes
                        existing_texts = set()
                        try:
                            for s in self.corpus_manager.get_summaries(500):
                                txt = (s.get('content') or '').strip()
                                if txt:
                                    existing_texts.add(txt)
                        except Exception:
                            pass
                        if summary_text in existing_texts:
                            continue
                        self.corpus_manager.add_summary({
                            'content': summary_text,
                            'timestamp': datetime.now(),
                            'type': 'summary',
                            'tags': [
                                'summary:consolidated',
                                'source:consolidator',
                                f'block_n:{N}:{b}',
                                f'block_span_n:{N}:{start}-{end-1}',
                            ]
                        })
                    except Exception:
                        pass

                    # Store into Chroma as well
                    try:
                        md = {
                            'timestamp': datetime.now().isoformat(),
                            'type': 'summary',
                            'importance_score': getattr(summary_node, 'importance_score', 0.7),
                            'tags': f'summary:consolidated,source:consolidator,block_n:{N}:{b},block_span_n:{N}:{start}-{end-1}',
                            'memory_count': len(block),
                        }
                        if hasattr(self.chroma_store, 'add_to_collection'):
                            self.chroma_store.add_to_collection('summaries', summary_text, md)
                    except Exception:
                        pass

                # For logging, prefer tagged count when present; else total consolidator summaries
                prev_blocks = len(existing_block_indices) if existing_block_indices else len(consolidator_summaries)
                # After generating, recompute handy diagnostics
                new_prev = prev_blocks + len(selected_blocks)
                pending_blocks = max(0, (T // N) - new_prev)
                since_last = 0 if pending_blocks > 0 else (T % N)
                remaining = 0 if pending_blocks > 0 else (N if since_last == 0 else (N - since_last))
                logger.info(
                    f"[Shutdown] Created {len(selected_blocks)} summary block(s) (N={N}, T={T}, prev={prev_blocks}); "
                    f"since_last={since_last}, remaining={remaining}, t={remaining}, backlog={pending_blocks}"
                )
            else:
                prev_blocks = len(existing_block_indices) if existing_block_indices else len(consolidator_summaries)
                # No complete block was due. Report turns since last summary and remaining to next.
                pending_blocks = max(0, (T // N) - prev_blocks)
                since_last = 0 if pending_blocks > 0 else (T % N)
                remaining = 0 if pending_blocks > 0 else (N if since_last == 0 else (N - since_last))
                logger.info(
                    "[Shutdown] No new summaries due (N=%s, T=%s, prev=%s); since_last=%s, remaining=%s, t=%s, backlog=%s",
                    N, T, prev_blocks, since_last, remaining, remaining, pending_blocks
                )

            # 3) Extract facts from this session's user turns only
            session_recent: list[Dict]
            if isinstance(session_conversations, list) and session_conversations:
                # Use provided in-memory buffer
                session_recent = list(session_conversations)
            else:
                try:
                    corpus = list(self.corpus_manager.corpus)
                except Exception:
                    corpus = []
                # Session-only, non-summary/non-reflection, newest first
                def _is_summary(e: Dict) -> bool:
                    typ = (e.get('type') or '').lower()
                    tags = e.get('tags') or []
                    return ("summary" in typ) or ("@summary" in tags) or ("type:summary" in tags)
                def _is_reflection(e: Dict) -> bool:
                    typ = (e.get('type') or '').lower()
                    tags = [str(t).lower() for t in (e.get('tags') or [])]
                    return (typ == 'reflection') or ('type:reflection' in tags)
                session_recent = [e for e in corpus if (not _is_summary(e)) and (not _is_reflection(e))]
                # reuse earlier _ts helper in this function
                try:
                    session_recent = [e for e in session_recent if _ts(e) >= self.session_start]
                except Exception:
                    pass
                session_recent.sort(key=_ts, reverse=True)

            for conv in session_recent[:10]:
                try:
                    # Extract strictly from the user's side only to keep inputs tiny
                    q = (conv.get('query') or '').strip()
                    if not q:
                        continue
                    facts = await self.fact_extractor.extract_facts(q, "")
                except Exception:
                    facts = []
                for fact in (facts or []):
                    try:
                        self.chroma_store.add_fact(
                            fact=getattr(fact, 'content', str(fact)),
                            source='shutdown_extraction',
                            confidence=0.7
                        )
                    except Exception:
                        continue
            
            # 4) Optional LLM-assisted facts over recent user messages (additive)
            try:
                _enabled = int(os.getenv("LLM_FACTS_ENABLED", "1")) == 1
            except Exception:
                _enabled = False

            if _enabled and hasattr(self, 'model_manager') and self.model_manager is not None:
                try:
                    last_turns = int(os.getenv("LLM_FACTS_LAST_TURNS", "12"))
                except Exception:
                    last_turns = 12
                try:
                    max_triples = int(os.getenv("LLM_FACTS_MAX_TRIPLES", "10"))
                except Exception:
                    max_triples = 10
                try:
                    max_chars = int(os.getenv("LLM_FACTS_MAX_INPUT_CHARS", "4000"))
                except Exception:
                    max_chars = 4000
                model_alias = os.getenv("LLM_FACTS_MODEL", "gpt-4o-mini")

                # Collect last user-only queries from THIS SESSION only (prefer in-memory buffer if provided)
                if isinstance(session_conversations, list) and session_conversations:
                    sess_items = session_conversations
                else:
                    try:
                        corpus = list(self.corpus_manager.corpus)
                    except Exception:
                        corpus = []
                    def _is_summary(e: Dict) -> bool:
                        typ = (e.get('type') or '').lower()
                        tags = e.get('tags') or []
                        return ("summary" in typ) or ("@summary" in tags) or ("type:summary" in tags)
                    def _is_reflection(e: Dict) -> bool:
                        typ = (e.get('type') or '').lower()
                        tags = [str(t).lower() for t in (e.get('tags') or [])]
                        return (typ == 'reflection') or ('type:reflection' in tags)
                    try:
                        sess_items = [e for e in corpus if (not _is_summary(e)) and (not _is_reflection(e)) and _ts(e) >= self.session_start]
                    except Exception:
                        sess_items = [e for e in corpus if (not _is_summary(e)) and (not _is_reflection(e))]
                users = [ (e.get('query') or '').strip() for e in sess_items if (e.get('query') or '').strip() ]
                user_tail = users[-max(1,last_turns):]

                if user_tail:
                    llm_ex = LLMFactExtractor(
                        self.model_manager,
                        model_alias=model_alias,
                        max_input_chars=max_chars,
                        max_triples=max_triples,
                    )
                    triples = await llm_ex.extract_triples(user_tail)
                    kept = 0
                    for t in triples:
                        subj = t.get('subject'); rel = t.get('relation'); obj = t.get('object')
                        if not subj or not rel or not obj:
                            continue
                        fact_text = f"{subj} | {rel} | {obj}"
                        try:
                            self.chroma_store.add_fact(
                                fact=fact_text,
                                source='llm_shutdown',
                                confidence=0.75,
                            )
                            kept += 1
                        except Exception:
                            continue
                    logger.info(f"[LLM Facts] kept={kept} (model={model_alias})")

            logger.info("[Shutdown] Memory processing complete")

        except Exception as e:
            logger.error(f"Shutdown processing error: {e}")

    # ---------------------------
    # Helper methods
    # ---------------------------

    def _get_memory_key(self, memory: Dict) -> str:
        """Generate unique key for deduplication"""
        q = (memory.get('query', '') or '')[:80]
        r = (memory.get('response', '') or '')[:80]
        return f"{q}__{r}"

    def _safe_detect_topic(self, text: str) -> str:
        try:
            if hasattr(self.topic_manager, 'detect_topic'):
                return self.topic_manager.detect_topic(text) or 'general'
        except Exception:
            pass
        return 'general'

    def _format_hierarchical_memory(self, memory) -> Dict:
        """Adapt hierarchical memory node to flat dict for ranker."""
        content = getattr(memory, 'content', '')
        parts = content.split('\nAssistant: ')
        q = content[:100]
        r = "[Could not parse response]"
        if len(parts) == 2:
            q = parts[0].replace("User: ", "").strip()
            r = parts[1].strip()

        truth = getattr(memory, 'truth_score', None)
        if truth is None:
            truth = (getattr(memory, 'metadata', {}) or {}).get('truth_score', 0.5)

        ts = getattr(memory, 'timestamp', datetime.now())
        return {
            'id': getattr(memory, 'id', f"hier::{uuid.uuid4().hex[:8]}"),
            'query': q,
            'response': r,
            'content': content or f"User: {q}\nAssistant: {r}",
            'timestamp': ts if isinstance(ts, datetime) else datetime.now(),
            'source': 'hierarchical',
            'collection': 'hierarchical',
            'metadata': getattr(memory, 'metadata', {}) or {},
            'tags': getattr(memory, 'tags', []),
            'truth_score': float(truth or 0.5),
            'relevance_score': float(getattr(memory, 'score', 0.5)),
            'importance_score': float((getattr(memory, 'metadata', {}) or {}).get('importance_score', 0.5)),
        }
    async def debug_memory_state(self):
        """Debug method to check what's in memory and get statistics"""
        stats = {
            'corpus_entries': len(self.corpus_manager.corpus),
            'non_summary_entries': len([e for e in self.corpus_manager.corpus if "@summary" not in e.get("tags", [])]),
            'summaries': len(self.corpus_manager.get_summaries()),
            'chroma_collections': {}
        }

        for name, collection in self.chroma_store.collections.items():
            stats['chroma_collections'][name] = collection.count()

        logger.info(f"[DEBUG] Memory state: {stats}")
        return stats
    # ---------------------------
    # Optional: summaries / dreams passthroughs
    # ---------------------------

    def get_summaries(self, limit: int = 3) -> List[Dict]:
        try:
            results = self.chroma_store.query_collection('summaries', '', n_results=limit)
            return [{'content': r.get('content', ''),
                    'timestamp': r.get('metadata', {}).get('timestamp', datetime.now()),
                    'type': 'summary'} for r in results]
        except:
            return []

    def get_dreams(self, limit: int = 2) -> List[Dict]:
        if hasattr(self.corpus_manager, 'get_dreams'):
            return [{
                'content': d,
                'timestamp': datetime.now(),
                'source': 'dream'
            } for d in (self.corpus_manager.get_dreams(limit) or [])]
        return []
