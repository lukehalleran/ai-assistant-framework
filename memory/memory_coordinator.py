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
import asyncio
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
from memory.hybrid_retriever import HybridRetriever
from memory.memory_storage import MemoryStorage
from memory.memory_retriever import MemoryRetriever
from memory.thread_manager import ThreadManager
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
        self.topic_manager = topic_manager or TopicManager()
        self.conversation_context = deque(maxlen=50)
        # Initialize hybrid retriever for improved semantic search
        self.hybrid_retriever = HybridRetriever(chroma_store=chroma_store)
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

        # Initialize modular components for delegation
        self.scorer = MemoryScorer(
            time_manager=time_manager,
            conversation_context=self.conversation_context
        )

        self.thread_manager = ThreadManager(
            corpus_manager=corpus_manager,
            topic_manager=self.topic_manager,
            time_manager=time_manager
        )

        self._storage = MemoryStorage(
            corpus_manager=corpus_manager,
            chroma_store=chroma_store,
            fact_extractor=self.fact_extractor,
            consolidator=self.consolidator,
            topic_manager=self.topic_manager,
            scorer=self.scorer,
            time_manager=time_manager
        )
        # Connect thread detection to storage
        self._storage._thread_detect_fn = self.thread_manager.detect_or_create_thread
        self._storage.conversation_context = self.conversation_context

        self._retriever = MemoryRetriever(
            corpus_manager=corpus_manager,
            chroma_store=chroma_store,
            gate_system=gate_system,
            scorer=self.scorer,
            hybrid_retriever=self.hybrid_retriever,
            time_manager=time_manager
        )
        self._retriever.conversation_context = list(self.conversation_context)

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


    async def get_recent_facts(self, limit: int = 5) -> List[Dict]:
        """Fetch the most recent facts by timestamp.

        Delegates to MemoryRetriever component.
        """
        return await self._retriever.get_recent_facts(limit)

    async def get_facts(self, query: str, limit: int = 8) -> List[Dict]:
        """Get facts for query.

        Delegates to MemoryRetriever component.
        """
        return await self._retriever.get_facts(query, limit)





    # ---------------------------
    # Thread detection
    # ---------------------------

    def get_thread_context(self) -> Optional[Dict]:
        """
        Retrieve thread context from the most recent conversation.
        Delegates to ThreadManager.
        """
        return self.thread_manager.get_thread_context()

    def _detect_topic_for_query(self, query: str) -> str:
        """
        Detect topic for a specific query string.
        Delegates to ThreadManager.
        """
        return self.thread_manager.detect_topic_for_query(query)

    def _detect_or_create_thread(self, query: str, is_heavy: bool) -> Dict:
        """
        Detect if current query continues immediate previous conversation.
        Delegates to ThreadManager.
        """
        return self.thread_manager.detect_or_create_thread(query, is_heavy)

    def _detect_or_create_thread_legacy(self, query: str, is_heavy: bool) -> Dict:
        """
        Detect if current query continues immediate previous conversation.
        Returns thread metadata dict with thread_id, depth, started, topic.
        Only checks the most recent conversation for strict consecutive threading.
        """
        from utils.query_checker import belongs_to_thread

        # Get only the most recent conversation (limit=1 for strict consecutive check)
        recent = self.corpus_manager.get_recent_memories(count=1)

        # Detect topic for current query to avoid using stale topic
        current_query_topic = self._detect_topic_for_query(query)

        if not recent:
            # First conversation - create new thread
            thread_id = f"thread_{self._now().strftime('%Y%m%d_%H%M%S')}"
            logger.debug(f"[Thread] Creating new thread (first conversation): {thread_id}")
            return {
                "thread_id": thread_id,
                "depth": 1,
                "started": self._now_iso(),
                "topic": current_query_topic,
            }

        last_conv = recent[0]

        # Check if current query continues last conversation
        if belongs_to_thread(query, last_conv, current_topic=current_query_topic):
            # Continue existing thread
            thread_id = last_conv.get("thread_id")
            thread_depth = last_conv.get("thread_depth", 0) + 1
            thread_started = last_conv.get("thread_started")
            thread_topic = last_conv.get("thread_topic", current_query_topic)

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
                "topic": current_query_topic,
            }

    # ---------------------------
    # Public API: store
    # ---------------------------

    # In memory_coordinator.py, update the store_interaction method
    # In memory_coordinator.py, update the store_interaction method with more debugging
    async def store_interaction(self, query: str, response: str, tags: Optional[List[str]] = None):
        """Persist a turn in both corpus & Chroma with computed metadata.

        Delegates to MemoryStorage component.
        """
        # Sync state before delegation
        self._storage.current_topic = self.current_topic
        self._storage.conversation_context = self.conversation_context

        await self._storage.store_interaction(query, response, tags)

        # Sync state back from storage
        self.conversation_context = self._storage.conversation_context
        self.interactions_since_consolidation = self._storage.interactions_since_consolidation

    async def _consolidate_and_store_summary(self):
        """Consolidate recent memories and store the summary.

        Delegates to MemoryStorage component.
        """
        await self._storage.consolidate_and_store_summary()
        self.last_consolidation_time = self._storage.last_consolidation_time or datetime.now()


    # --- inside class MemoryCoordinator ---
    async def add_reflection(self, text: str, *, tags=None, source="reflection", timestamp=None) -> bool:
        """Store a reflection memory.

        Delegates to MemoryStorage component.
        """
        return await self._storage.add_reflection(text, tags=tags, source=source, timestamp=timestamp)

    async def get_reflections(self, limit: int = 2):
        """Fetch recent reflections.

        Delegates to MemoryRetriever component.
        """
        return await self._retriever.get_reflections(limit)

    async def get_reflections_hybrid(self, query: str, limit: int = 3) -> List[Dict]:
        """Hybrid retrieval for reflections.

        Delegates to MemoryRetriever component.
        """
        return await self._retriever.get_reflections_hybrid(query, limit)

    # Optional: generic search by collection/type name (used by some callers)
    async def search_by_type(self, type_name: str, query: str = "", limit: int = 5):
        """Search by memory type.

        Delegates to MemoryRetriever component.
        """
        return await self._retriever.search_by_type(type_name, query, limit)


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
        """Extract and store facts from a turn.

        Delegates to MemoryStorage component.
        """
        await self._storage.extract_and_store_facts(query, response, truth_score)

    # ---------------------------
    # Public API: retrieve
    # ---------------------------

    async def get_memories(self, query: str, limit: int = 20, topic_filter: Optional[str] = None) -> List[Dict]:
        """Unified retrieval pipeline.

        Delegates to MemoryRetriever component.
        """
        # Sync state before delegation
        self._retriever.current_topic = self.current_topic
        self._retriever.conversation_context = list(self.conversation_context)
        return await self._retriever.get_memories(query, limit, topic_filter)

    # ---------------------------
    # Internals: gather
    # ---------------------------

    async def _get_meta_conversational_memories(
        self, query: str, limit: int = 20, topic_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Special retrieval for meta-conversational queries asking about conversation history.

        Strategy:
        1. Detect temporal window from query (e.g., "yesterday", "last week")
        2. Dynamically adjust retrieval limit based on temporal scope
        3. Retrieve recent episodic memories
        4. Sort chronologically (newest first)
        5. Apply gentle recency weighting that preserves older memories
        6. Skip semantic search to avoid cross-contamination from other conversations

        Args:
            query: The meta-conversational query (e.g., "do you recall...")
            limit: Maximum memories to return
            topic_filter: Optional topic to filter by

        Returns:
            List of recent episodic memories in reverse chronological order
        """
        logger.debug("[MemoryCoordinator] Using meta-conversational retrieval strategy")

        # DYNAMIC TEMPORAL WINDOW DETECTION
        # Detect how far back the user is asking about based on temporal markers
        from utils.query_checker import extract_temporal_window
        temporal_days = extract_temporal_window(query)

        # Adjust retrieval limits based on temporal scope
        if temporal_days == 0:
            # No specific temporal marker - use default (3-5 days)
            recent_limit = min(limit * 5, 50)
            return_multiplier = 3
            logger.debug("[MemoryCoordinator] No temporal marker detected, using default window (50 memories)")
        elif temporal_days <= 2:
            # Yesterday / recent (1-2 days)
            recent_limit = min(limit * 3, 30)
            return_multiplier = 2
            logger.debug(f"[MemoryCoordinator] Short temporal window detected ({temporal_days}d), retrieving {recent_limit} memories")
        elif temporal_days <= 7:
            # Last week (3-7 days)
            recent_limit = min(limit * 8, 80)
            return_multiplier = 4
            logger.debug(f"[MemoryCoordinator] Medium temporal window detected ({temporal_days}d), retrieving {recent_limit} memories")
        else:
            # Last month or longer (8+ days)
            recent_limit = min(limit * 15, 150)
            return_multiplier = 6
            logger.debug(f"[MemoryCoordinator] Long temporal window detected ({temporal_days}d), retrieving {recent_limit} memories")

        topic_filter = topic_filter or self.current_topic

        # Gather recent conversations (episodic memories)
        recent = self._get_recent_conversations(k=recent_limit)

        # IMPORTANT: Do NOT filter by topic for meta-conversational queries!
        # When user asks "do you recall...", they're explicitly asking about past conversations
        # which may have been on ANY topic. Topic filtering would exclude relevant memories.
        # Example: "Do you recall when I said I woke up at noon?" should find that memory
        # even if it was tagged with a different topic than the current query.
        logger.debug(f"[MemoryCoordinator] Meta-conversational: skipping topic filter (preserving all {len(recent)} recent memories)")

        # For meta-conversational queries, we want to preserve ALL recent memories
        # without aggressive filtering, since the user is explicitly asking about
        # past conversations which may be from days/weeks ago.
        #
        # Apply LIGHT recency weighting (mostly for sorting order) but don't filter out
        # older memories - the user might be asking about something from "last week"
        #
        # DYNAMIC DECAY: Adjust decay rate based on temporal window
        # Larger temporal window = gentler decay (preserve older memories)
        if temporal_days == 0:
            decay_divisor = 48.0  # Default: 2 days (48 hours)
        elif temporal_days <= 2:
            decay_divisor = 24.0  # Short window: 1 day
        elif temporal_days <= 7:
            decay_divisor = 96.0  # Medium window: 4 days
        else:
            decay_divisor = 168.0  # Long window: 7 days

        for mem in recent:
            ts = mem.get('timestamp', datetime.now())
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except Exception:
                    ts = datetime.now()

            # Calculate recency score using active days if available, otherwise hours
            if (self.time_manager is not None and
                hasattr(self.time_manager, 'calculate_active_day_decay')):
                # Convert decay_divisor from hours to active days for similar behavior
                # decay_divisor ranges from 24-168 hours, so active_days_divisor = 1-7 days
                active_days_divisor = decay_divisor / 24.0
                recency_score = self.time_manager.calculate_active_day_decay(ts, 1.0/active_days_divisor)
            else:
                # Fallback to original hourly calculation
                age_hours = (datetime.now() - ts).total_seconds() / 3600.0
                recency_score = 1.0 / (1.0 + (age_hours / decay_divisor))

            # Use recency primarily for sorting, not aggressive filtering
            current_relevance = mem.get('relevance_score', 0.5)
            mem['final_score'] = (recency_score * 0.4) + (current_relevance * 0.6)
            mem['recency_score'] = recency_score

            # Mark as meta-conversational for debugging
            mem.setdefault('metadata', {})['meta_conversational'] = True

        # Sort by timestamp (chronological - newest first) since all are relevant
        recent.sort(key=lambda m: m.get('timestamp', datetime.min), reverse=True)

        # Return MORE memories than requested to ensure we capture older conversations
        # The multiplier is dynamically adjusted based on detected temporal window
        result = recent[:min(len(recent), limit * return_multiplier)]

        logger.debug(
            f"[MemoryCoordinator] Meta-conversational retrieval returned {len(result)} memories "
            f"(requested: {limit}, multiplier: {return_multiplier}x, temporal_days: {temporal_days}, available: {len(recent)})"
        )

        # Update access tracking
        self._update_truth_scores_on_access(result)

        return result

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

    # Optimized _get_semantic_memories method using batch queries
    async def _get_semantic_memories(self, query: str, n_results: int = 30) -> List[Dict]:
        """
        Get semantic memories using hybrid retrieval (query rewriting + semantic + keyword).
        This replaces the original ChromaDB-only approach with enhanced relevance matching.
        """
        logger.info(f"[Semantic] Using hybrid retrieval for query: '{query[:50]}...'")

        try:
            # Use hybrid retriever for better semantic search quality
            hybrid_results = await self.hybrid_retriever.retrieve(query, limit=n_results)

            # Convert hybrid results to expected format for memory coordinator
            memories = []
            for result in hybrid_results:
                # Create standardized memory format
                # Calculate boosted final_score to pass gate system threshold (0.35)
                hybrid_score = result.get("hybrid_score", 0.0)
                keyword_score = result.get("keyword_score", 0.0)

                # Boost scores to ensure relevant memories pass the 0.35 threshold
                # Hybrid scores are 0.1-0.3, gate threshold is 0.35, so we boost them
                boosted_score = hybrid_score + 0.3  # Add 0.3 to pass threshold
                if keyword_score > 0.5:  # Strong keyword matches get extra boost
                    boosted_score = max(boosted_score, 0.6)

                memory = {
                    "id": result.get("id", str(hash(str(result)))),
                    "content": result.get("content", ""),
                    "query": result.get("query", ""),
                    "response": result.get("response", ""),
                    "metadata": result.get("metadata", {}),
                    "collection": result.get("collection", "unknown"),
                    "final_score": boosted_score,  # Boosted to pass gate threshold
                    "semantic_score": result.get("semantic_score", 0.0),
                    "keyword_score": keyword_score,
                    "hybrid_score": hybrid_score,
                    "scoring": result.get("scoring", {}),
                    "relevance": boosted_score  # For gate system compatibility
                }
                memories.append(memory)

            logger.info(f"[Semantic] Hybrid retrieval returned {len(memories)} memories")
            print(f"[DEBUG] Hybrid retriever returning {len(memories)} memories")
            for i, mem in enumerate(memories[:3]):
                print(f"[DEBUG]   {i+1}. {mem.get('metadata', {}).get('timestamp', 'unknown')} - {mem.get('query', '')[:40]}...")
            return memories

        except Exception as e:
            logger.error(f"[Semantic] Hybrid retrieval failed: {e}")
            import traceback
            traceback.print_exc()

            # Fallback to basic semantic search if hybrid fails
            logger.warning("[Semantic] Falling back to basic semantic search")
            return await self._fallback_semantic_search(query, n_results)

    async def _fallback_semantic_search(self, query: str, n_results: int = 30) -> List[Dict]:
        """
        Fallback semantic search using original ChromaDB approach if hybrid retrieval fails.
        """
        memories: List[Dict] = []
        collections_to_query = ['conversations', 'summaries', 'reflections']

        try:
            # Basic batch query
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
            logger.error(f"[Semantic] Fallback search also failed: {e}")

        return memories

    async def get_semantic_top_memories(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Return top-k semantic memories across conversations, summaries, reflections
        using the gate system's cosine score. No recent-corpus bypass.

        Special handling: If this is a meta-conversational query (e.g., "do you recall"),
        route to specialized retrieval that prioritizes recent episodic memories.
        """
        # Check if this is a meta-conversational query asking about conversation history
        from utils.query_checker import is_meta_conversational
        if is_meta_conversational(query):
            logger.debug(f"[MemoryCoordinator][Semantic] Detected meta-conversational query, routing to specialized retrieval: {query[:50]}...")
            return await self._get_meta_conversational_memories(query, limit, topic_filter=None)

        try:
            raw = await self._get_semantic_memories(query, n_results=max(30, limit * 3))
        except Exception:
            raw = []

        if not raw:
            return []

        # Build chunks for gating with back-reference to original memory
        import re as _re
        _hdr_re = _re.compile(r"^\s*\[[^\]]+\]", _re.IGNORECASE)
        def _strip_headers(s: str) -> str:
            if not s:
                return s
            out = []
            for ln in (s.splitlines() or []):
                if _hdr_re.search(ln):
                    continue
                out.append(ln)
            return "\n".join(out).strip()

        def _gate_text(m: Dict) -> str:
            txt = _strip_headers((m.get('content') or '').strip())
            if txt:
                return txt
            q = _strip_headers((m.get('query') or '').strip()); a = _strip_headers((m.get('response') or '').strip())
            return f"User: {q}\nAssistant: {a}".strip()

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

        # Optional strict top-up: disabled by default to avoid noisy generic memories
        try:
            import os as _os
            enable_topup = str(_os.getenv("MEM_TOPUP_ENABLE", "0")).strip().lower() in {"1","true","yes","on"}
            min_score = float(_os.getenv("MEM_TOPUP_MIN_SCORE", "0.35"))
        except Exception:
            enable_topup = False; min_score = 0.35

        if enable_topup and len(out) < limit and raw:
            # Build a set of keys for quick de-duplication (prefer stable id; fall back to content)
            def _k(m: Dict) -> str:
                mid = str(m.get('id') or '').strip()
                if mid:
                    return f"id::{mid}"
                return f"content::{(m.get('content') or '').strip()[:160].lower()}"

            selected = {_k(m) for m in out}
            def _score(m: Dict) -> float:
                try:
                    return float(m.get('relevance_score', 0.0))
                except Exception:
                    return 0.0
            # Simple lexical overlap guard
            def _overlap(a: str, b: str) -> float:
                import re as __re
                at = set(__re.findall(r"[a-zA-Z0-9]+", (a or "").lower()))
                bt = set(__re.findall(r"[a-zA-Z0-9]+", (b or "").lower()))
                if not at or not bt:
                    return 0.0
                return len(at & bt) / max(1, min(len(at), len(bt)))

            for cand in sorted(raw, key=_score, reverse=True):
                if len(out) >= limit:
                    break
                key = _k(cand)
                if not key or key in selected:
                    continue
                sc = _score(cand)
                txt = _gate_text(cand)
                if sc >= min_score and _overlap(query, txt) >= 0.15:
                    c = dict(cand)
                    c['pre_gated'] = True
                    out.append(c)
                    selected.add(key)

        return out[:limit]



    # ---------------------------
    # Internals: combine + gate
    # ---------------------------

    async def _combine_memories(self, very_recent: List[Dict], semantic: List[Dict],
                                hierarchical: List[Dict], query: str, config: Dict, bypass_gate: bool = False) -> List[Dict]:
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
        print(f"[DEBUG] Processing {len(semantic)} semantic memories into candidates")
        semantic_added = 0
        for mem in semantic:
            key = self._get_memory_key(mem)
            if key not in seen:
                candidates.append(mem)
                seen.add(key)
                semantic_added += 1
        print(f"[DEBUG] Added {semantic_added} semantic memories to candidates (seen: {len(seen)})")

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
        # Skip gate system for gym/health queries to ensure relevant memories get through
        use_gate_system = self.gate_system and candidates and not bypass_gate
        print(f"[DEBUG] _combine_memories: candidates={len(candidates)}, bypass_gate={bypass_gate}, use_gate_system={use_gate_system}")

        if use_gate_system:
            gated = await self._gate_memories(query, candidates)
            print(f"[DEBUG] Gate system filtered {len(candidates)} -> {len(gated)} memories")
            for mem in gated:
                mem['gated'] = True
                combined.append(mem)
        else:
            # If no gate, just cap the number of candidates added
            cap = config.get('max_memories', 20)
            print(f"[DEBUG] Bypassing gate system, taking top {min(cap, len(candidates))} candidates")
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
    # Internals: ranker (delegated to MemoryScorer)
    # ---------------------------

    def _rank_memories(self, memories: List[Dict], current_query: str) -> List[Dict]:
        """
        Score each memory using the MemoryScorer component.
        Delegates to self.scorer.rank_memories() for the actual scoring logic.
        """
        # Sync conversation context to scorer
        self.scorer.conversation_context = list(self.conversation_context)
        return self.scorer.rank_memories(memories, current_query)

    def _rank_memories_legacy(self, memories: List[Dict], current_query: str) -> List[Dict]:
        """
        Legacy implementation - kept for reference.
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

            # 2) recency with decay (using active days)
            ts = m.get('timestamp')
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except Exception:
                    ts = now
            elif not isinstance(ts, datetime):
                ts = now

            # Use active day decay if time_manager supports it, otherwise fall back to hourly decay
            if (self.time_manager is not None and
                hasattr(self.time_manager, 'calculate_active_day_decay')):
                recency = self.time_manager.calculate_active_day_decay(ts, RECENCY_DECAY_RATE)
            else:
                # Fallback to original hourly decay
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
                        result = self.chroma_store.add_fact(
                            fact=getattr(fact, 'content', str(fact)),
                            source='shutdown_extraction',
                            confidence=0.7
                        )
                        if result is None:
                            logger.debug(f"[MemoryCoordinator] Shutdown fact skipped as duplicate: {getattr(fact, 'content', str(fact))}")
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
                            result = self.chroma_store.add_fact(
                                fact=fact_text,
                                source='llm_shutdown',
                                confidence=0.75,
                            )
                            if result is not None:
                                kept += 1
                            else:
                                logger.debug(f"[MemoryCoordinator] LLM fact skipped as duplicate: {fact_text}")
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
        # Include ID or timestamp to avoid false duplicates
        mem_id = memory.get('id', '')
        timestamp = memory.get('timestamp', memory.get('metadata', {}).get('timestamp', ''))
        q = (memory.get('query', '') or '')[:40]  # Shorter to leave room for ID
        r = (memory.get('response', '') or '')[:40]

        # Use multiple identifiers to ensure uniqueness
        if mem_id:
            return f"id:{mem_id}__{q[:20]}__{r[:20]}"
        elif timestamp:
            return f"ts:{timestamp}__{q[:20]}__{r[:20]}"
        else:
            return f"hash:{hash(str(memory))}__{q[:30]}__{r[:30]}"

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
        """Get recent summaries.

        Delegates to MemoryRetriever component.
        """
        return self._retriever.get_summaries(limit)

    def get_summaries_hybrid(self, query: str, limit: int = 4) -> List[Dict]:
        """Hybrid retrieval for summaries.

        Delegates to MemoryRetriever component.
        """
        return self._retriever.get_summaries_hybrid(query, limit)

    def get_dreams(self, limit: int = 2) -> List[Dict]:
        """Get dreams.

        Delegates to MemoryRetriever component.
        """
        return self._retriever.get_dreams(limit)
