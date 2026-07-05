"""
# knowledge/synthesis_pooled_generator.py

Module Contract
- Purpose: The discovery ("scientific insight") candidate generator. Pairs PROMINENT
  concepts from the curated CONCEPT_POOL in the NON-OBVIOUS semantic band (cos 0.2-0.45)
  and articulates a candidate structural connection for each, packaged as
  SynthesisCandidate objects for SynthesisFilter.process_batch().
- Class: PooledConceptSynthesisGenerator(model_manager, concept_pool=None,
           min_cos=SYNTHESIS_POOLED_MIN_COS, max_cos=SYNTHESIS_POOLED_MAX_COS,
           concurrency=5, seed=None)
- Key method: async generate_candidates(count) -> List[SynthesisCandidate]
- Substrate: knowledge/synthesis_concept_pool.CONCEPT_POOL (prominent concepts); cosine
  via the sentence embedder alone (semantic_search._load_embedder) — MEMORY-SAFE: embeds
  only the ~48 pool strings, never loads the multi-GB wiki FAISS index or calls
  idx.search (which reads the 33GB text column). Embedding runs via asyncio.to_thread so
  the shutdown dreaming timeout can preempt it.
- Why this design (validated 2026-06-30, scripts/validate_anchored_generator.py): pairing
  two prominent concepts in the non-obvious band, run through the REAL coherence judge,
  gives ~46% MODERATE+STRONG / ~17% accept vs ~0 for the retired personal->wiki generators
  and 31% for FAISS-neighbour anchoring. The lever is concept PROMINENCE, not anchoring.
- Output per candidate: walk_path=["pooled", a, b] (=> midrange-peak distance scoring in
  the filter, NOT the "retrieval" inverted path); source_domains={concept_a, concept_b}
  (two distinct names clear the >=2 domain gate; real non-obviousness is enforced by the
  cosine band + the coherence judge); endpoint_distance = 1 - cos(a,b).
- Side effects: 1 LLM articulation call per attempted pair (parallelized, semaphore;
  staged waves — first `count` attempts, top-up only the NO_CONNECTION shortfall up to
  the oversample budget, so successful articulations are never discarded).
  No ChromaDB writes (the filter/memory own persistence).
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from knowledge.synthesis_concept_pool import CONCEPT_POOL
from knowledge.synthesis_models import SynthesisCandidate
from utils.logging_utils import get_logger

logger = get_logger(__name__)

DISCOVERY_SYSTEM = (
    "You identify non-obvious STRUCTURAL connections between concepts for a discovery "
    "engine. A real connection is a shared mechanism, dynamic, feedback loop, trade-off, or "
    "mathematical form — NOT a shared topic or a vague 'both are systems'. A separate "
    "adversarial judge rejects weak ones, so propose only a genuine shared structure; if "
    "there is none, respond exactly NO_CONNECTION."
)
DISCOVERY_PROMPT = """Two concepts:
  A: {a}
  B: {b}

Name the SINGLE most specific STRUCTURAL connection they share: a mechanism / dynamic /
feedback loop / trade-off / mathematical form such that a non-obvious feature of one
predicts a feature of the other. One or two plain-prose sentences (no headings or lists) —
concrete, mechanistic, falsifiable (someone could argue against it). NOT "both involve
change" / "both are important". If they share no real transferable structure, respond
exactly: NO_CONNECTION."""


class PooledConceptSynthesisGenerator:
    """Discovery generator: prominent curated concepts paired in the non-obvious band."""

    def __init__(
        self,
        model_manager,
        concept_pool: Optional[List[str]] = None,
        min_cos: Optional[float] = None,
        max_cos: Optional[float] = None,
        concurrency: int = 5,
        oversample: float = 2.0,
        seed: Optional[int] = None,
    ):
        from config.app_config import (
            SYNTHESIS_POOLED_MIN_COS,
            SYNTHESIS_POOLED_MAX_COS,
        )
        self.model_manager = model_manager
        self.pool = list(dict.fromkeys(concept_pool or CONCEPT_POOL))  # dedup, order-stable
        self.min_cos = SYNTHESIS_POOLED_MIN_COS if min_cos is None else min_cos
        self.max_cos = SYNTHESIS_POOLED_MAX_COS if max_cos is None else max_cos
        self.concurrency = concurrency
        self.oversample = oversample
        self._rng = random.Random(seed)

    def _in_band_pairs(self) -> List[Tuple[str, str, float]]:
        """All pool pairs whose cosine sits in the non-obvious band [min_cos, max_cos].

        Loads only the sentence embedder (the shared ModelManager singleton when
        warm) — never the multi-GB wiki FAISS index, which is irrelevant for
        embedding ~48 short pool strings and used to stall shutdown cold-loading
        off the external drive. Blocking (model load + encode) — call off-loop.
        """
        try:
            from knowledge.semantic_search import _load_embedder, EMBED_MODEL
            embedder = _load_embedder(EMBED_MODEL)
        except Exception as e:
            logger.warning(f"[PooledGen] embedder unavailable — no candidates: {e}")
            return []
        embs = embedder.encode(self.pool, convert_to_numpy=True, normalize_embeddings=True)
        pairs: List[Tuple[str, str, float]] = []
        n = len(self.pool)
        for i in range(n):
            for j in range(i + 1, n):
                c = float(np.dot(embs[i], embs[j]))
                if self.min_cos <= c <= self.max_cos:
                    pairs.append((self.pool[i], self.pool[j], c))
        self._rng.shuffle(pairs)
        return pairs

    async def _articulate(self, sem, a: str, b: str, cos: float) -> Optional[SynthesisCandidate]:
        from config.app_config import SYNTHESIS_COHERENCE_MODEL
        async with sem:
            try:
                raw = await self.model_manager.generate_once(
                    DISCOVERY_PROMPT.format(a=a, b=b),
                    model_name=SYNTHESIS_COHERENCE_MODEL,
                    system_prompt=DISCOVERY_SYSTEM,
                    # 340 (was 220): the best, most-detailed structural claims were being
                    # truncated mid-sentence at 220, which reads as incoherent to the judge
                    # and to a human grader. 2-sentence mechanistic bridges need the room.
                    max_tokens=340,
                    temperature=0.6,
                )
            except Exception as e:
                logger.debug(f"[PooledGen] articulation failed for {a}<->{b}: {e}")
                return None
        claim = (raw or "").strip()
        if not claim or "NO_CONNECTION" in claim.upper() or len(claim.split()) < 5:
            return None
        return SynthesisCandidate(
            concept_a=a,
            concept_b=b,
            connection_claim=claim,
            walk_path=["pooled", a.lower(), b.lower()],
            source_domains={a, b},
            endpoint_distance=round(1.0 - cos, 4),
            timestamp=datetime.now(),
        )

    async def generate_candidates(self, count: int) -> List[SynthesisCandidate]:
        """Return up to `count` articulated discovery candidates from the curated pool."""
        if count <= 0:
            return []
        # Off-loop: embedding is sync CPU work and the dreaming timeout
        # (asyncio.wait_for) cannot preempt code that blocks the event loop.
        pairs = await asyncio.to_thread(self._in_band_pairs)
        if not pairs:
            return []
        # Staged waves: attempt `count` first, top up only the NO_CONNECTION
        # shortfall — never pay for articulations we would then discard. The
        # oversample factor caps total LLM spend across all waves.
        budget = min(max(count, int(count * self.oversample)), len(pairs))
        sem = asyncio.Semaphore(self.concurrency)
        candidates: List[SynthesisCandidate] = []
        attempted = 0
        while len(candidates) < count and attempted < budget:
            batch = pairs[attempted: min(attempted + (count - len(candidates)), budget)]
            attempted += len(batch)
            results = await asyncio.gather(
                *(self._articulate(sem, a, b, c) for a, b, c in batch)
            )
            candidates.extend(r for r in results if r is not None)
        logger.info(
            "[PooledGen] %d in-band pairs, %d attempted, %d candidates (target %d)",
            len(pairs), attempted, len(candidates), count,
        )
        return candidates[:count]
