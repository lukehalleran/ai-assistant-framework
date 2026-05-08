"""
Section utilization analysis.

Answers: for a given corpus of snapshots, which sections actually
contained content? Which are frequently empty? Which fire only for
certain intents?

Inputs:
    - Dict of query_id -> PromptSnapshot (from Phase 1 captures)
    - Dict of query_id -> CorpusQuery (from corpus.py)
Outputs:
    - UtilizationReport with per-section stats, coverage summary,
      always-empty/always-present/high-variance/intent-specific classifications
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

from eval.schema import PromptSnapshot
from eval.corpus import CorpusQuery
from eval.section_registry import SECTION_REGISTRY


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SectionUtilization:
    """Utilization stats for one section across the corpus."""

    section_key: str
    category: str

    # Counts
    total_queries: int = 0
    times_present: int = 0    # Section key existed in layer
    times_nonempty: int = 0   # Content was non-trivial (> 50 chars formatted text)
    times_empty: int = 0      # Section existed but formatted_text was empty/trivial

    # Token stats
    total_tokens: int = 0

    # By intent
    intent_presence: Dict[str, int] = field(default_factory=dict)

    @property
    def presence_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.times_present / self.total_queries

    @property
    def nonempty_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.times_nonempty / self.total_queries

    @property
    def avg_tokens_when_present(self) -> float:
        if self.times_present == 0:
            return 0.0
        return self.total_tokens / self.times_present

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_key": self.section_key,
            "category": self.category,
            "total_queries": self.total_queries,
            "times_present": self.times_present,
            "times_nonempty": self.times_nonempty,
            "times_empty": self.times_empty,
            "total_tokens": self.total_tokens,
            "presence_rate": self.presence_rate,
            "nonempty_rate": self.nonempty_rate,
            "avg_tokens_when_present": self.avg_tokens_when_present,
            "intent_presence": dict(self.intent_presence),
        }


@dataclass
class UtilizationReport:
    """Full utilization analysis across all sections."""

    corpus_size: int
    sections: Dict[str, SectionUtilization]
    intents_covered: Dict[str, int]

    always_empty: List[str] = field(default_factory=list)
    always_present: List[str] = field(default_factory=list)
    high_variance: List[str] = field(default_factory=list)
    intent_specific: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corpus_size": self.corpus_size,
            "intents_covered": self.intents_covered,
            "always_empty": self.always_empty,
            "always_present": self.always_present,
            "high_variance": self.high_variance,
            "intent_specific": self.intent_specific,
            "sections": {
                key: util.to_dict()
                for key, util in self.sections.items()
            },
        }


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class UtilizationAnalyzer:
    """Analyze section utilization across a corpus of snapshots."""

    def analyze(
        self,
        snapshots: Dict[str, PromptSnapshot],
        corpus: Dict[str, CorpusQuery],
        layer: str = "post_hygiene",
    ) -> UtilizationReport:
        """Run full utilization analysis.

        Args:
            snapshots: Map of query_id -> PromptSnapshot.
            corpus: Map of query_id -> CorpusQuery.
            layer: Which snapshot layer to analyze.

        Returns:
            UtilizationReport with per-section stats and classifications.
        """
        # Initialize utilization objects for all registered sections
        utils: Dict[str, SectionUtilization] = {}
        for key, section_def in SECTION_REGISTRY.items():
            utils[key] = SectionUtilization(
                section_key=key,
                category=section_def.category.value,
            )

        intents_covered: Dict[str, int] = defaultdict(int)

        for query_id, snapshot in snapshots.items():
            query = corpus.get(query_id)
            intent = query.intent.value if query else "unknown"
            intents_covered[intent] += 1

            snap_layer = snapshot.layers.get(layer)
            if snap_layer is None:
                # Still count toward total_queries even if layer missing
                for util in utils.values():
                    util.total_queries += 1
                continue

            sections = snap_layer.sections

            for section_key, util in utils.items():
                util.total_queries += 1

                section = sections.get(section_key)
                if section is None:
                    continue

                util.times_present += 1
                util.total_tokens += section.token_count
                util.intent_presence[intent] = (
                    util.intent_presence.get(intent, 0) + 1
                )

                # Check if content is substantive
                text = section.formatted_text or ""
                structured = section.structured_content

                is_nonempty = False
                if len(text.strip()) > 50:
                    is_nonempty = True
                elif isinstance(structured, list) and len(structured) > 0:
                    is_nonempty = True
                elif isinstance(structured, dict) and len(structured) > 0:
                    is_nonempty = True

                if is_nonempty:
                    util.times_nonempty += 1
                else:
                    util.times_empty += 1

        # Classify sections
        always_empty = [
            k for k, u in utils.items()
            if u.presence_rate == 0.0
        ]
        always_present = [
            k for k, u in utils.items()
            if u.total_queries > 0 and u.presence_rate == 1.0
        ]
        high_variance = [
            k for k, u in utils.items()
            if 0.2 <= u.presence_rate <= 0.8
        ]

        # Intent-specific: sections only present for one intent
        intent_specific: Dict[str, List[str]] = defaultdict(list)
        for key, util in utils.items():
            if len(util.intent_presence) == 1 and util.times_present > 0:
                sole_intent = list(util.intent_presence.keys())[0]
                intent_specific[sole_intent].append(key)

        return UtilizationReport(
            corpus_size=len(snapshots),
            sections=utils,
            intents_covered=dict(intents_covered),
            always_empty=always_empty,
            always_present=always_present,
            high_variance=high_variance,
            intent_specific=dict(intent_specific),
        )

    def format_report(self, report: UtilizationReport) -> str:
        """Human-readable utilization summary."""
        lines = [
            "=== Section Utilization Report ===",
            f"Corpus: {report.corpus_size} queries",
            f"Intents covered: {report.intents_covered}",
            "",
            "--- Always Empty (candidates for conditional gating) ---",
        ]

        for key in sorted(report.always_empty):
            section_def = SECTION_REGISTRY.get(key)
            category = section_def.category.value if section_def else "unknown"
            lines.append(f"  [{category}] {key}")

        lines.append("")
        lines.append("--- Always Present ---")
        for key in sorted(report.always_present):
            lines.append(f"  {key}")

        lines.append("")
        lines.append(
            "--- High Variance (20-80% presence -- best optimization targets) ---"
        )
        for key in sorted(report.high_variance):
            util = report.sections[key]
            lines.append(
                f"  {key}: {util.presence_rate:.0%} presence, "
                f"{util.avg_tokens_when_present:.0f} avg tokens"
            )

        lines.append("")
        lines.append("--- Intent-Specific Sections ---")
        for intent, section_keys in sorted(report.intent_specific.items()):
            lines.append(f"  {intent}: {', '.join(sorted(section_keys))}")

        lines.append("")
        lines.append("--- Full Table ---")
        header = (
            f"{'Section':<30} {'Presence':>8} {'Nonempty':>9} "
            f"{'Avg Tok':>8} {'Category':>18}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for key in sorted(
            report.sections.keys(),
            key=lambda k: SECTION_REGISTRY[k].assembly_order
            if k in SECTION_REGISTRY
            else 999,
        ):
            util = report.sections[key]
            section_def = SECTION_REGISTRY.get(key)
            category = section_def.category.value if section_def else "?"
            lines.append(
                f"{key:<30} {util.presence_rate:>7.0%} "
                f"{util.nonempty_rate:>8.0%} "
                f"{util.avg_tokens_when_present:>7.0f} "
                f"{category:>18}"
            )

        return "\n".join(lines)
