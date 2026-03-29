"""
# knowledge/synthesis_models.py

Module Contract
- Purpose: Data models for the knowledge synthesis pipeline. Defines candidate,
  stage result, and fully-evaluated synthesis result types used by the filter
  pipeline and synthesis memory.
- Key classes:
  - CoherenceLevel: 4-tier forced-choice coherence rating (INVALID/WEAK/MODERATE/STRONG)
  - CandidateStatus: Lifecycle status (PENDING/REJECTED/ACCEPTED/CONVERGING)
  - SynthesisCandidate: Raw candidate from a graph walk before filtering
  - StageResult: Result from a single pipeline stage
  - SynthesisResult: Fully evaluated candidate with all stage results + convergence tracking
- Side effects: None (pure data models)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class CoherenceLevel(Enum):
    """4-tier forced-choice coherence rating. No middle ground."""
    INVALID = 0.0
    WEAK = 0.33
    MODERATE = 0.66
    STRONG = 1.0


class CandidateStatus(Enum):
    """Lifecycle status of a synthesis candidate."""
    PENDING = "pending"
    REJECTED = "rejected"
    ACCEPTED = "accepted"
    CONVERGING = "converging"  # seen from multiple independent paths


@dataclass
class SynthesisCandidate:
    """Raw candidate from a graph walk, before filtering."""
    concept_a: str
    concept_b: str
    connection_claim: str       # the articulated bridge statement
    walk_path: List[str]        # full random walk node sequence
    source_domains: Set[str]    # cluster/category tags for crossed domains
    endpoint_distance: float    # cosine distance between concept embeddings
    timestamp: datetime = field(default_factory=datetime.now)
    path_hash: str = ""         # computed from walk_path for dedup

    def __post_init__(self):
        if not self.path_hash:
            path_str = "->".join(self.walk_path)
            self.path_hash = hashlib.sha256(path_str.encode()).hexdigest()[:16]


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage_name: str
    passed: bool
    score: Optional[float] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass
class SynthesisResult:
    """Fully evaluated synthesis candidate with all stage results."""
    candidate: SynthesisCandidate
    stage_results: List[StageResult] = field(default_factory=list)
    coherence_level: Optional[CoherenceLevel] = None
    coherence_justification: str = ""
    novelty_score_external: float = 0.0
    novelty_score_internal: float = 0.0
    nearest_known_external: str = ""    # closest match in wiki corpus
    nearest_known_internal: str = ""    # closest match in synthesis memory
    composite_score: float = 0.0
    status: CandidateStatus = CandidateStatus.PENDING
    rejection_stage: Optional[str] = None
    rejection_reason: str = ""

    # Convergence tracking (populated from synthesis memory)
    unique_paths: Set[str] = field(default_factory=set)
    unique_sources: Set[str] = field(default_factory=set)
    convergence_strength: float = 0.0

    @property
    def passed_all_gates(self) -> bool:
        return all(sr.passed for sr in self.stage_results)

    def reject(self, stage: str, reason: str) -> None:
        self.status = CandidateStatus.REJECTED
        self.rejection_stage = stage
        self.rejection_reason = reason

    def to_metadata(self) -> Dict[str, Any]:
        """Flatten for ChromaDB storage (primitives only)."""
        return {
            "concept_a": self.candidate.concept_a,
            "concept_b": self.candidate.concept_b,
            "connection_claim": self.candidate.connection_claim,
            "path_hash": self.candidate.path_hash,
            "source_domains": ",".join(sorted(self.candidate.source_domains)),
            "endpoint_distance": self.candidate.endpoint_distance,
            "coherence_level": self.coherence_level.name if self.coherence_level else "UNKNOWN",
            "coherence_justification": self.coherence_justification,
            "coherence_score": self.coherence_level.value if self.coherence_level else 0.0,
            "novelty_external": self.novelty_score_external,
            "novelty_internal": self.novelty_score_internal,
            "nearest_known_external": self.nearest_known_external,
            "nearest_known_internal": self.nearest_known_internal,
            "composite_score": self.composite_score,
            "status": self.status.value,
            "unique_paths": ",".join(sorted(self.unique_paths)),
            "unique_sources": ",".join(sorted(self.unique_sources)),
            "convergence_strength": self.convergence_strength,
            "timestamp": self.candidate.timestamp.isoformat(),
        }

    @classmethod
    def from_metadata(cls, metadata: Dict[str, Any], document: str) -> "SynthesisResult":
        """Reconstruct from ChromaDB metadata."""
        candidate = SynthesisCandidate(
            concept_a=metadata.get("concept_a", ""),
            concept_b=metadata.get("concept_b", ""),
            connection_claim=document,
            walk_path=[],  # not stored -- too large
            source_domains=set(metadata.get("source_domains", "").split(",")) if metadata.get("source_domains") else set(),
            endpoint_distance=float(metadata.get("endpoint_distance", 0.0)),
            path_hash=metadata.get("path_hash", ""),
        )
        result = cls(candidate=candidate)
        result.coherence_level = (
            CoherenceLevel[metadata["coherence_level"]]
            if metadata.get("coherence_level") and metadata["coherence_level"] != "UNKNOWN"
            else None
        )
        result.coherence_justification = metadata.get("coherence_justification", "")
        result.novelty_score_external = float(metadata.get("novelty_external", 0.0))
        result.novelty_score_internal = float(metadata.get("novelty_internal", 0.0))
        result.nearest_known_external = metadata.get("nearest_known_external", "")
        result.nearest_known_internal = metadata.get("nearest_known_internal", "")
        result.composite_score = float(metadata.get("composite_score", 0.0))
        result.status = CandidateStatus(metadata.get("status", "pending"))
        result.unique_paths = (
            set(metadata.get("unique_paths", "").split(","))
            if metadata.get("unique_paths") else set()
        )
        result.unique_sources = (
            set(metadata.get("unique_sources", "").split(","))
            if metadata.get("unique_sources") else set()
        )
        result.convergence_strength = float(metadata.get("convergence_strength", 0.0))
        return result
