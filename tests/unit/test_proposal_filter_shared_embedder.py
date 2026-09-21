# tests/unit/test_proposal_filter_shared_embedder.py
"""
Regression test for BC-81: ProposalFilter._semantic_dedup used to build a
brand-new SentenceTransformer("all-MiniLM-L6-v2") on every call (a fresh
~90 MB model load each time, never cached) instead of routing through the
project's shared embedder cache, models.model_manager.ModelManager
._get_cached_embedder() (a static, module-global cache already used by
WikiManager, semantic_search, and need_detector).

This test proves _semantic_dedup never constructs SentenceTransformer
directly and always goes through the cached accessor, while dedup behavior
(collapsing near-duplicate proposals to the higher-priority one, keeping
distinct proposals separate) is unchanged.
"""

import time
import uuid
from unittest.mock import patch

import numpy as np

from memory.code_proposal import CodeProposal, ProposalStatus, ProposalType


def _make_proposal(
    title: str = "Test Proposal",
    tags: list = None,
    priority: int = 5,
    proposal_type: ProposalType = ProposalType.FEATURE,
    status: ProposalStatus = ProposalStatus.PENDING,
    reasoning: str = "Improves the system",
    affected_files: list = None,
    created_at: float = None,
) -> CodeProposal:
    return CodeProposal(
        id=str(uuid.uuid4()),
        title=title,
        proposal_type=proposal_type,
        status=status,
        priority=priority,
        reasoning=reasoning,
        tags=tags or [],
        affected_files=affected_files or [],
        created_at=created_at if created_at is not None else time.time(),
    )


class _FixedVectorEmbedder:
    """Fake embedder standing in for ModelManager._get_cached_embedder()'s
    return value. Identical input text maps to an identical one-hot vector
    (cosine 1.0 -> deduped); distinct input text maps to an orthogonal
    one-hot vector (cosine 0.0 -> kept). Signature matches the real cached
    accessor's stub fallback: `.encode(texts, convert_to_numpy=True,
    normalize_embeddings=True)` — no `convert_to_tensor` kwarg support.
    """

    def __init__(self):
        self.encode_call_count = 0

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, **kwargs):
        self.encode_call_count += 1
        seen = {}
        dim = max(4, len(texts))
        vectors = []
        for t in texts:
            if t not in seen:
                vec = np.zeros(dim, dtype=np.float32)
                vec[len(seen)] = 1.0
                seen[t] = vec
            vectors.append(seen[t])
        return np.array(vectors, dtype=np.float32)


class TestSemanticDedupSharedEmbedder:
    def _get_filter(self):
        from core.prompt.proposal_filter import ProposalFilter
        return ProposalFilter()

    def test_no_sentence_transformer_construction_uses_cached_embedder(self):
        """_semantic_dedup called twice must never construct SentenceTransformer
        directly, and must route through ModelManager._get_cached_embedder both
        times (BC-81: a fresh ~90MB load on every call otherwise)."""
        pf = self._get_filter()
        fake_embedder = _FixedVectorEmbedder()

        # Near-duplicate pair: identical to_embedding_text() (title/reasoning/
        # tags/type all match), differing only by priority.
        p1 = _make_proposal(
            "Add cross-collection deduplication",
            tags=["memory", "dedup"],
            priority=8,
            reasoning="Facts, summaries, and skills often contain overlapping information",
        )
        p2 = _make_proposal(
            "Add cross-collection deduplication",
            tags=["memory", "dedup"],
            priority=5,
            reasoning="Facts, summaries, and skills often contain overlapping information",
        )
        # Distinct proposal: different title/tags/reasoning.
        p3 = _make_proposal(
            "Implement user authentication with OAuth",
            tags=["auth", "security"],
            priority=6,
            reasoning="Enable secure user login via third-party providers",
        )

        with patch("sentence_transformers.SentenceTransformer") as mock_st, \
             patch(
                 "models.model_manager.ModelManager._get_cached_embedder",
                 return_value=fake_embedder,
             ) as mock_get_cached:
            result_dup = pf._semantic_dedup([p1, p2], threshold=0.85)
            result_distinct = pf._semantic_dedup([p1, p3], threshold=0.85)

        # No SentenceTransformer was ever constructed directly.
        assert mock_st.call_count == 0

        # The shared cache accessor was used on every call — never bypassed.
        assert mock_get_cached.call_count == 2
        assert fake_embedder.encode_call_count == 2

        # Dedup behavior is unchanged: near-duplicates collapse to the
        # higher-priority proposal, distinct proposals are both kept.
        assert len(result_dup) == 1
        assert result_dup[0].priority == 8
        assert len(result_distinct) == 2

    def test_empty_and_single_still_skip_embedder(self):
        """Unchanged fast paths: <=1 proposal never touches the embedder."""
        pf = self._get_filter()
        with patch("sentence_transformers.SentenceTransformer") as mock_st, \
             patch("models.model_manager.ModelManager._get_cached_embedder") as mock_get_cached:
            assert pf._semantic_dedup([], threshold=0.85) == []
            p1 = _make_proposal("Solo")
            assert len(pf._semantic_dedup([p1])) == 1

        assert mock_st.call_count == 0
        assert mock_get_cached.call_count == 0
