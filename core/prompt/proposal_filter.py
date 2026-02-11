# core/prompt/proposal_filter.py
"""
Retrieval, dedup, gating, and ranking of code proposals for prompt injection.

Pipeline: ProposalStore (ChromaDB FAISS top-20)
        -> keyword dedup -> semantic dedup
        -> composite scoring (priority + breadth + recency + goal alignment)
        -> optional LLM pairwise ranking (tournament bracket)
        -> top N

Only surfaces proposals when the query is project-related (fast keyword check).
Composite scoring replaces pure semantic match — a proposal's *actual* value
(priority, breadth of impact, freshness, goal alignment) matters more than
whether its description happens to contain the word "impactful".
"""

import hashlib
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from utils.logging_utils import get_logger

logger = get_logger("proposal_filter")

# Project-related keywords for fast relevance check
_PROJECT_KEYWORDS = frozenset({
    "code", "feature", "implement", "refactor", "bug", "test", "proposal",
    "module", "daemon", "memory", "prompt", "pipeline", "architecture",
    "build", "component", "class", "function", "method", "api", "endpoint",
    "config", "database", "chroma", "collection", "deploy", "docker",
    "system", "design", "improve", "fix", "add", "create", "update",
    "delete", "remove", "migrate", "extract", "optimize", "performance",
    "error", "exception", "handler", "middleware", "route", "schema",
    "model", "integration", "orchestrator", "coordinator", "gate",
    "search", "retrieval", "embedding", "vector", "index", "token",
    "budget", "formatter", "builder", "shutdown", "startup", "skill",
    "workflow", "procedural", "semantic", "corpus", "store", "query",
})

# Recency half-life: proposals older than this (in days) get half credit
_RECENCY_HALF_LIFE_DAYS = 14.0


class ProposalFilter:
    """
    Encapsulates retrieval, dedup, gating, and ranking of code proposals
    for the prompt pipeline.

    Ranking uses a composite score combining:
      - priority (proposal's own 1-10 field, normalized)
      - breadth  (how many system areas it touches: files + tag diversity)
      - recency  (exponential decay from created_at)
      - goal_alignment (semantic similarity to project goals via gate system)

    Optionally, an LLM pairwise ranking pass (tournament bracket) can
    re-rank the top candidates for higher accuracy.
    """

    def __init__(self, chroma_store=None, gate_system=None, model_manager=None):
        self._chroma_store = chroma_store
        self._gate_system = gate_system
        self._model_manager = model_manager
        self._proposal_store = None

        # Cache for utility query (invalidated by file-hash change)
        self._goals_hash: Optional[str] = None
        self._cached_utility_query: Optional[str] = None

    @property
    def proposal_store(self):
        """Lazy-init ProposalStore."""
        if self._proposal_store is None:
            try:
                from memory.proposal_store import ProposalStore
                self._proposal_store = ProposalStore(self._chroma_store)
            except ImportError:
                logger.warning("[ProposalFilter] ProposalStore not available")
        return self._proposal_store

    @property
    def gate_system(self):
        """Lazy-init gate system."""
        if self._gate_system is None:
            try:
                from processing.gate_system import CosineSimilarityGateSystem
                self._gate_system = CosineSimilarityGateSystem()
            except ImportError:
                logger.warning("[ProposalFilter] Gate system not available")
        return self._gate_system

    # ------------------------------------------------------------------
    # Utility query from GOALS.md
    # ------------------------------------------------------------------

    def _build_utility_query(self) -> str:
        """
        Build a query that ranks proposals by project utility.

        Reads docs/GOALS.md, extracts active goals, and constructs a
        composite query. Cached with file-hash invalidation.
        """
        goals_path = Path("docs/GOALS.md")
        if not goals_path.exists():
            # Try relative to project root
            alt = Path(__file__).resolve().parent.parent.parent / "docs" / "GOALS.md"
            if alt.exists():
                goals_path = alt
            else:
                return self._fallback_utility_query()

        try:
            content = goals_path.read_text(encoding="utf-8")
        except Exception:
            return self._fallback_utility_query()

        # Check cache
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if content_hash == self._goals_hash and self._cached_utility_query:
            return self._cached_utility_query

        # Extract active goals (lines starting with ### under Active Goals)
        goals = []
        in_active = False
        for line in content.splitlines():
            if "Active Goals" in line:
                in_active = True
                continue
            if in_active and line.startswith("---"):
                break
            if in_active and line.startswith("### "):
                # Extract goal name: "### 1. Complete Modular Refactor" -> "Complete Modular Refactor"
                goal_text = re.sub(r"^###\s*\d+\.\s*", "", line).strip()
                # Remove backtick annotations
                goal_text = re.sub(r"`[^`]*`", "", goal_text).strip()
                if goal_text:
                    goals.append(goal_text)

        if not goals:
            self._goals_hash = content_hash
            self._cached_utility_query = self._fallback_utility_query()
            return self._cached_utility_query

        numbered = " ".join(f"[{i+1}] {g}" for i, g in enumerate(goals))
        query = f"Given project goals: {numbered} Which features add the most value?"

        self._goals_hash = content_hash
        self._cached_utility_query = query
        return query

    @staticmethod
    def _fallback_utility_query() -> str:
        return (
            "Which code improvements would add the most value to the project? "
            "Consider refactoring, new features, bug fixes, and test coverage."
        )

    # ------------------------------------------------------------------
    # Project-relevance check
    # ------------------------------------------------------------------

    @staticmethod
    def _is_project_related(query: str) -> bool:
        """
        Fast keyword check for project-related queries.

        Returns True if the query contains project-related keywords,
        False for casual/personal queries (zero-cost skip).
        """
        if not query:
            return False
        words = set(re.findall(r"[a-z]+", query.lower()))
        return bool(words & _PROJECT_KEYWORDS)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_dedup(proposals, tag_threshold: float = 0.60) -> list:
        """
        Remove proposals with high keyword overlap.

        Pairwise: if >= tag_threshold tag overlap (Jaccard) AND >= 50%
        title word overlap -> keep higher priority.
        """
        if len(proposals) <= 1:
            return list(proposals)

        keep = list(proposals)
        to_remove = set()

        for i in range(len(keep)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(keep)):
                if j in to_remove:
                    continue

                a, b = keep[i], keep[j]
                tags_a = set(a.tags) if a.tags else set()
                tags_b = set(b.tags) if b.tags else set()

                # Tag Jaccard similarity
                if tags_a or tags_b:
                    tag_union = tags_a | tags_b
                    tag_inter = tags_a & tags_b
                    tag_sim = len(tag_inter) / len(tag_union) if tag_union else 0
                else:
                    tag_sim = 0

                # Title word overlap
                words_a = set(a.title.lower().split())
                words_b = set(b.title.lower().split())
                word_union = words_a | words_b
                word_inter = words_a & words_b
                word_sim = len(word_inter) / len(word_union) if word_union else 0

                if tag_sim >= tag_threshold and word_sim >= 0.50:
                    # Remove the lower-priority one
                    loser = j if a.priority >= b.priority else i
                    to_remove.add(loser)

        return [p for idx, p in enumerate(keep) if idx not in to_remove]

    @staticmethod
    def _semantic_dedup(proposals, threshold: float = 0.85) -> list:
        """
        Remove semantically similar proposals using embedding cosine similarity.

        Embeds to_embedding_text() for each proposal, computes pairwise
        cosine, removes above threshold (keeps higher priority).
        """
        if len(proposals) <= 1:
            return list(proposals)

        try:
            from sentence_transformers import SentenceTransformer, util
            import torch
        except ImportError:
            logger.debug("[ProposalFilter] sentence-transformers not available, skipping semantic dedup")
            return list(proposals)

        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [p.to_embedding_text() for p in proposals]
            embeddings = model.encode(texts, convert_to_tensor=True)
            cosine_scores = util.cos_sim(embeddings, embeddings)

            to_remove = set()
            for i in range(len(proposals)):
                if i in to_remove:
                    continue
                for j in range(i + 1, len(proposals)):
                    if j in to_remove:
                        continue
                    if cosine_scores[i][j].item() >= threshold:
                        loser = j if proposals[i].priority >= proposals[j].priority else i
                        to_remove.add(loser)

            return [p for idx, p in enumerate(proposals) if idx not in to_remove]

        except Exception as e:
            logger.warning(f"[ProposalFilter] Semantic dedup failed: {e}")
            return list(proposals)

    # ------------------------------------------------------------------
    # Composite scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score_priority(proposal) -> float:
        """Normalize priority (1-10) to 0.0-1.0."""
        return max(0.0, min(1.0, (proposal.priority - 1) / 9.0))

    @staticmethod
    def _score_breadth(proposal) -> float:
        """
        Score by how many system areas the proposal touches.

        Combines affected_files count and tag diversity.
        More files + more diverse tags = broader impact.
        """
        n_files = len(proposal.affected_files) if proposal.affected_files else 0
        n_tags = len(proposal.tags) if proposal.tags else 0

        # Count distinct directories in affected files
        dirs = set()
        for f in (proposal.affected_files or []):
            parts = f.replace("\\", "/").split("/")
            if len(parts) > 1:
                dirs.add(parts[0])

        n_dirs = len(dirs)

        # Normalize: 5+ files across 3+ dirs with 4+ tags = 1.0
        file_score = min(1.0, n_files / 5.0)
        dir_score = min(1.0, n_dirs / 3.0)
        tag_score = min(1.0, n_tags / 4.0)

        return (file_score * 0.4) + (dir_score * 0.3) + (tag_score * 0.3)

    @staticmethod
    def _score_recency(proposal) -> float:
        """
        Exponential decay from created_at.

        Half-life of 14 days: a 2-week-old proposal gets 0.5,
        a brand-new one gets ~1.0, a month-old one gets ~0.25.
        """
        created = getattr(proposal, "created_at", 0)
        if not created:
            return 0.5  # unknown age, neutral score

        age_days = (time.time() - created) / 86400.0
        if age_days < 0:
            return 1.0

        return math.exp(-0.693 * age_days / _RECENCY_HALF_LIFE_DAYS)

    @staticmethod
    def _compute_composite_score(
        proposal,
        goal_alignment: float = 0.5,
        w_priority: float = 0.30,
        w_breadth: float = 0.20,
        w_recency: float = 0.20,
        w_goal: float = 0.30,
    ) -> float:
        """
        Compute a composite score for a proposal.

        Args:
            proposal: CodeProposal object
            goal_alignment: Semantic similarity to project goals (0.0-1.0),
                           from gate system. This is one signal, not the
                           sole ranking factor.
            w_priority: Weight for proposal's priority field
            w_breadth: Weight for system breadth (files + dirs + tags)
            w_recency: Weight for how recently the proposal was created
            w_goal: Weight for semantic goal alignment

        Returns:
            Composite score 0.0-1.0
        """
        p_score = ProposalFilter._score_priority(proposal)
        b_score = ProposalFilter._score_breadth(proposal)
        r_score = ProposalFilter._score_recency(proposal)
        g_score = max(0.0, min(1.0, goal_alignment))

        composite = (
            w_priority * p_score
            + w_breadth * b_score
            + w_recency * r_score
            + w_goal * g_score
        )

        return composite

    # ------------------------------------------------------------------
    # Novelty scoring (penalize already-started work)
    # ------------------------------------------------------------------

    def _get_recent_commit_messages(self, limit: int = 30) -> List[str]:
        """
        Fetch recent git commit subjects for novelty comparison.

        Cached for the lifetime of the filter instance.
        """
        if hasattr(self, '_commit_cache'):
            return self._commit_cache

        try:
            result = subprocess.run(
                ["git", "log", f"-n{limit}", "--pretty=format:%s"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                self._commit_cache = [
                    line.strip().lower()
                    for line in result.stdout.splitlines()
                    if line.strip()
                ]
            else:
                self._commit_cache = []
        except Exception:
            self._commit_cache = []

        return self._commit_cache

    def _score_novelty(self, proposal) -> float:
        """
        Score how novel a proposal is relative to recent git history.

        Compares proposal title + reasoning words against recent commit
        messages. High overlap = work already started/done = low score.

        Returns:
            1.0 = fully novel (no overlap with commits)
            0.0 = already implemented (high overlap)
        """
        commits = self._get_recent_commit_messages()
        if not commits:
            return 1.0  # no git history, assume novel

        # Build word set from proposal
        title_words = set(re.findall(r"[a-z]{3,}", proposal.title.lower()))
        reasoning_words = set(re.findall(
            r"[a-z]{3,}", (proposal.reasoning or "").lower()
        ))
        # Title words matter more — use them for matching
        proposal_words = title_words

        if not proposal_words:
            return 1.0

        # Strip common filler words
        _STOP = frozenset({
            "the", "and", "for", "with", "from", "that", "this", "into",
            "should", "would", "could", "using", "based", "when", "more",
            "than", "also", "each", "over", "after", "before", "between",
        })
        proposal_words -= _STOP

        if not proposal_words:
            return 1.0

        # Check overlap against each commit message
        best_overlap = 0.0
        for msg in commits:
            commit_words = set(re.findall(r"[a-z]{3,}", msg)) - _STOP
            if not commit_words:
                continue
            overlap = len(proposal_words & commit_words)
            # Jaccard-like: overlap relative to proposal size
            score = overlap / len(proposal_words) if proposal_words else 0
            best_overlap = max(best_overlap, score)

        # Invert: high overlap = low novelty
        return max(0.0, 1.0 - best_overlap)

    # ------------------------------------------------------------------
    # Topic diversity (anti-clustering)
    # ------------------------------------------------------------------

    @staticmethod
    def _diverse_select(scored_dicts: List[Dict], limit: int = 3,
                        overlap_threshold: float = 0.34) -> List[Dict]:
        """
        Greedy diversity-aware selection from scored proposals.

        Picks the highest-scored proposal, then for each subsequent slot
        skips candidates whose tags overlap too much with already-selected
        proposals. Uses overlap coefficient (|intersection| / |smaller set|)
        which is better than Jaccard for unequal tag sets.

        A threshold of 0.34 means: if more than 1/3 of the smaller
        tag set appears in the larger, they're considered same-topic.
        For two proposals with 3 tags each sharing 1 tag: 1/3 = 0.33
        which just passes. Sharing 2+ tags always triggers.

        Args:
            scored_dicts: Proposals sorted by composite score descending
            limit: How many to select
            overlap_threshold: Max overlap coefficient with any
                already-selected proposal (default 0.34)

        Returns:
            List of diverse top proposals
        """
        if len(scored_dicts) <= limit:
            return list(scored_dicts)

        selected = []
        selected_tag_sets = []

        for candidate in scored_dicts:
            if len(selected) >= limit:
                break

            # Extract tags for this candidate
            meta = candidate.get("metadata", {})
            tags_raw = meta.get("tags_json", "[]")
            try:
                import json as _json
                tags = set(_json.loads(tags_raw)) if isinstance(tags_raw, str) else set(tags_raw or [])
            except Exception:
                tags = set()

            # Check overlap with already-selected proposals
            too_similar = False
            for prev_tags in selected_tag_sets:
                if not tags or not prev_tags:
                    continue
                inter = tags & prev_tags
                # Overlap coefficient: intersection / smaller set
                smaller = min(len(tags), len(prev_tags))
                overlap = len(inter) / smaller if smaller > 0 else 0
                if overlap >= overlap_threshold:
                    too_similar = True
                    break

            if not too_similar:
                selected.append(candidate)
                selected_tag_sets.append(tags)

        # If diversity filtering was too aggressive, backfill with top remaining
        if len(selected) < limit:
            seen_titles = {s.get("metadata", {}).get("title") for s in selected}
            for candidate in scored_dicts:
                if len(selected) >= limit:
                    break
                title = candidate.get("metadata", {}).get("title")
                if title not in seen_titles:
                    selected.append(candidate)
                    seen_titles.add(title)

        return selected

    # ------------------------------------------------------------------
    # LLM pairwise ranking (tournament bracket)
    # ------------------------------------------------------------------

    async def _llm_pairwise_rank(self, proposals: list, limit: int = 3) -> list:
        """
        Tournament-bracket LLM ranking over proposals.

        Takes top candidates, runs pairwise comparisons asking "which of
        these two would improve the system more?", and returns the winners.

        Falls back to input order if LLM is unavailable.
        """
        if not self._model_manager or len(proposals) <= limit:
            return proposals[:limit]

        try:
            from config.app_config import CODE_PROPOSALS_LLM_RANKING_MODEL
        except ImportError:
            CODE_PROPOSALS_LLM_RANKING_MODEL = "gpt-4o-mini"

        # Build summaries for comparison
        def _summarize(p) -> str:
            meta = p.get("metadata", {})
            title = meta.get("title", "Untitled")
            ptype = meta.get("proposal_type", "feature")
            priority = meta.get("priority", 5)
            reasoning = meta.get("reasoning", "")[:150]
            n_files = len(meta.get("affected_files_json", "[]"))
            return f"[{ptype}] P{priority} \"{title}\": {reasoning} (files: {n_files})"

        # Take top 6 candidates for tournament (3 rounds of 2)
        candidates = list(proposals[:min(6, len(proposals))])

        # Run pairwise comparisons
        winners = []
        i = 0
        while i < len(candidates) - 1:
            a = candidates[i]
            b = candidates[i + 1]

            summary_a = _summarize(a)
            summary_b = _summarize(b)

            prompt = (
                f"You are evaluating two proposed code improvements for a software project. "
                f"Which proposal would deliver MORE value to the project overall?\n\n"
                f"Proposal A: {summary_a}\n\n"
                f"Proposal B: {summary_b}\n\n"
                f"Answer with ONLY 'A' or 'B' (the letter of the better proposal)."
            )

            try:
                response = await self._model_manager.generate_once(
                    prompt,
                    model_name=CODE_PROPOSALS_LLM_RANKING_MODEL,
                    system_prompt="You are a senior software architect evaluating proposals. Be concise.",
                    max_tokens=4,
                    temperature=0.1,
                )
                choice = response.strip().upper()
                winner = a if "A" in choice else b
            except Exception as e:
                logger.warning(f"[ProposalFilter] LLM pairwise comparison failed: {e}")
                # Fall back to composite score ordering
                winner = a if a.get("relevance_score", 0) >= b.get("relevance_score", 0) else b

            winners.append(winner)
            i += 2

        # If odd number of candidates, last one gets a bye
        if len(candidates) % 2 == 1:
            winners.append(candidates[-1])

        # If we have more winners than limit, run another round
        if len(winners) > limit:
            return await self._llm_pairwise_rank(
                winners, limit=limit
            )

        logger.debug(f"[ProposalFilter] LLM pairwise ranking: {len(candidates)} -> {len(winners)}")
        return winners[:limit]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def get_proposals(self, query: str, limit: int = 3) -> List[Dict]:
        """
        Retrieve, dedup, and rank proposals for prompt injection.

        Ranking uses composite scoring (priority + breadth + recency +
        goal alignment). Optionally followed by LLM pairwise ranking
        for more accurate "best overall" selection.

        Returns a list of dicts ready for prompt formatting:
        [{content, metadata, relevance_score}, ...]
        """
        try:
            from config.app_config import (
                CODE_PROPOSALS_PROMPT_ENABLED,
                CODE_PROPOSALS_PROMPT_MAX,
                CODE_PROPOSALS_KEYWORD_DEDUP_TAG_THRESHOLD,
                CODE_PROPOSALS_SEMANTIC_DEDUP_THRESHOLD,
                CODE_PROPOSALS_LLM_RANKING,
                CODE_PROPOSALS_WEIGHT_PRIORITY,
                CODE_PROPOSALS_WEIGHT_BREADTH,
                CODE_PROPOSALS_WEIGHT_RECENCY,
                CODE_PROPOSALS_WEIGHT_GOAL_ALIGNMENT,
            )
        except ImportError:
            CODE_PROPOSALS_PROMPT_ENABLED = True
            CODE_PROPOSALS_PROMPT_MAX = 3
            CODE_PROPOSALS_KEYWORD_DEDUP_TAG_THRESHOLD = 0.60
            CODE_PROPOSALS_SEMANTIC_DEDUP_THRESHOLD = 0.75
            CODE_PROPOSALS_LLM_RANKING = False
            CODE_PROPOSALS_WEIGHT_PRIORITY = 0.30
            CODE_PROPOSALS_WEIGHT_BREADTH = 0.20
            CODE_PROPOSALS_WEIGHT_RECENCY = 0.10
            CODE_PROPOSALS_WEIGHT_GOAL_ALIGNMENT = 0.40

        if not CODE_PROPOSALS_PROMPT_ENABLED:
            logger.info("[ProposalFilter] Proposals disabled by config")
            return []

        # Fast check: skip non-project queries
        if not self._is_project_related(query):
            logger.info(f"[ProposalFilter] Query not project-related, skipping: {query[:80]}")
            return []

        store = self.proposal_store
        if not store:
            logger.warning("[ProposalFilter] ProposalStore not available (chroma_store=%s)", type(self._chroma_store))
            return []

        # Step 1: Query proposals with utility query (for goal alignment signal)
        utility_query = self._build_utility_query()
        logger.info(f"[ProposalFilter] Querying with utility_query: {utility_query[:80]}...")
        proposals = store.query_proposals(
            utility_query,
            n_results=20,
            status_filter=["pending", "approved"],
        )

        if not proposals:
            logger.info("[ProposalFilter] No proposals found in store (collection may be empty)")
            return []

        logger.info(f"[ProposalFilter] Retrieved {len(proposals)} proposals from store")

        # Step 2: Keyword dedup
        proposals = self._keyword_dedup(proposals, CODE_PROPOSALS_KEYWORD_DEDUP_TAG_THRESHOLD)
        logger.debug(f"[ProposalFilter] After keyword dedup: {len(proposals)}")

        # Step 3: Semantic dedup
        proposals = self._semantic_dedup(proposals, CODE_PROPOSALS_SEMANTIC_DEDUP_THRESHOLD)
        logger.debug(f"[ProposalFilter] After semantic dedup: {len(proposals)}")

        if not proposals:
            return []

        # Step 4: Gate system for goal_alignment scores (one signal, not sole ranker)
        gate_scores = {}  # proposal index -> gate relevance_score
        gate = self.gate_system
        if gate:
            gate_dicts = [
                {"content": p.to_embedding_text(), "metadata": p.to_metadata(), "relevance_score": 0.5}
                for p in proposals
            ]
            try:
                gated = await gate.filter_memories(utility_query, gate_dicts)
                if gated:
                    # Map back gate scores by matching content
                    content_to_score = {d["content"]: d.get("relevance_score", 0.5) for d in gated}
                    for i, p in enumerate(proposals):
                        gate_scores[i] = content_to_score.get(p.to_embedding_text(), 0.3)
                    logger.debug(f"[ProposalFilter] Gate scored {len(gated)} proposals")
            except Exception as e:
                logger.warning(f"[ProposalFilter] Gate scoring failed: {e}")

        # Step 5: Composite scoring with novelty penalty
        scored_dicts = []
        for i, p in enumerate(proposals):
            goal_alignment = gate_scores.get(i, 0.5)
            composite = self._compute_composite_score(
                p,
                goal_alignment=goal_alignment,
                w_priority=CODE_PROPOSALS_WEIGHT_PRIORITY,
                w_breadth=CODE_PROPOSALS_WEIGHT_BREADTH,
                w_recency=CODE_PROPOSALS_WEIGHT_RECENCY,
                w_goal=CODE_PROPOSALS_WEIGHT_GOAL_ALIGNMENT,
            )

            # Novelty penalty: reduce score for proposals already reflected
            # in recent git commits (work already started/done)
            novelty = self._score_novelty(p)
            if novelty < 1.0:
                # Apply as multiplicative penalty: 50% overlap → score * 0.75
                novelty_factor = 0.5 + 0.5 * novelty  # maps 0.0→0.5, 1.0→1.0
                old_composite = composite
                composite *= novelty_factor
                logger.debug(
                    f"[ProposalFilter] Novelty penalty: '{p.title[:50]}' "
                    f"novelty={novelty:.2f} factor={novelty_factor:.2f} "
                    f"score={old_composite:.3f}->{composite:.3f}"
                )

            scored_dicts.append({
                "content": p.to_embedding_text(),
                "metadata": p.to_metadata(),
                "relevance_score": composite,
            })

        # Sort by composite score descending
        scored_dicts.sort(key=lambda d: d["relevance_score"], reverse=True)
        effective_limit = min(limit, CODE_PROPOSALS_PROMPT_MAX)

        if scored_dicts:
            top = scored_dicts[0]
            logger.info(
                f"[ProposalFilter] Top proposal: '{top['metadata'].get('title', '?')}' "
                f"score={top['relevance_score']:.3f}, returning {min(effective_limit, len(scored_dicts))} proposals"
            )

        # Step 6: Topic diversity selection — prevent top N from clustering
        # on the same topic (e.g. all proposal-system-related)
        diverse = self._diverse_select(scored_dicts, limit=effective_limit)
        logger.debug(
            f"[ProposalFilter] Diversity selection: "
            f"{', '.join(d['metadata'].get('title', '?')[:40] for d in diverse)}"
        )

        # Step 7: Optional LLM pairwise ranking
        if CODE_PROPOSALS_LLM_RANKING and self._model_manager and len(diverse) > 1:
            try:
                result = await self._llm_pairwise_rank(diverse, limit=effective_limit)
                logger.debug(f"[ProposalFilter] LLM ranking: {len(diverse)} -> {len(result)}")
                return result
            except Exception as e:
                logger.warning(f"[ProposalFilter] LLM ranking failed, using composite: {e}")

        logger.debug(f"[ProposalFilter] Returning {len(diverse)} proposals for prompt")
        return diverse
