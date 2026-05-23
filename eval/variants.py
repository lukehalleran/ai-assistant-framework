"""
Variant generation from frozen snapshots.

Generates ablation variants by systematically dropping, adding, bundling,
or reordering prompt sections. Each variant tests a hypothesis about
section value.

Four strategies:
1. Leave-one-out (LOO): Remove one ablatable section at a time
2. Add-one-in (AOI): Start from structural skeleton, add one section back
3. Bundles: Test pre-defined groups of related sections together
4. Reorder: Move a section to a different assembly_order position

Inputs:
    - PromptSnapshot from Phase 1
    - Section registry (ablatable sections)
Outputs:
    - PromptVariant objects with metadata about what changed
    - Variant manifest mapping variant IDs to ablation descriptions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from eval.schema import PromptSnapshot, SectionSnapshot
from eval.section_registry import (
    SECTION_REGISTRY,
    SectionCategory,
    get_ablatable,
    get_structurally_required,
)


class VariantStrategy(Enum):
    LEAVE_ONE_OUT = "leave_one_out"
    ADD_ONE_IN = "add_one_in"
    BUNDLE = "bundle"
    REORDER = "reorder"


@dataclass
class PromptVariant:
    """One variant of a prompt -- a subset/reordering of sections from a snapshot."""

    variant_id: str  # "{snapshot_id}_{strategy}_{detail}"
    parent_snapshot_id: str
    strategy: VariantStrategy
    layer: str  # "raw_retrieval" or "post_hygiene"

    # What changed
    sections_removed: List[str] = field(default_factory=list)
    sections_added: List[str] = field(default_factory=list)
    reordered_sections: Dict[str, int] = field(default_factory=dict)  # key -> new order
    description: str = ""

    # The section keys for this variant, in the order they should be assembled
    active_sections: List[str] = field(default_factory=list)

    # Token estimates
    token_count_estimate: int = 0
    parent_token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "parent_snapshot_id": self.parent_snapshot_id,
            "strategy": self.strategy.value,
            "layer": self.layer,
            "sections_removed": self.sections_removed,
            "sections_added": self.sections_added,
            "reordered_sections": self.reordered_sections,
            "description": self.description,
            "active_sections": self.active_sections,
            "token_count_estimate": self.token_count_estimate,
            "parent_token_count": self.parent_token_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PromptVariant:
        d = dict(d)  # shallow copy
        d["strategy"] = VariantStrategy(d["strategy"])
        return cls(**d)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section_has_content(section: SectionSnapshot) -> bool:
    """Check whether a section has non-trivial formatted content."""
    text = section.formatted_text
    return bool(text and text.strip())


def _sum_tokens(sections: Dict[str, SectionSnapshot], keys: List[str]) -> int:
    """Sum token counts for the given keys present in sections dict."""
    return sum(
        sections[k].token_count
        for k in keys
        if k in sections
    )


def _sort_keys_by_order(
    keys: List[str],
    sections: Dict[str, SectionSnapshot],
    order_overrides: Optional[Dict[str, int]] = None,
) -> List[str]:
    """Sort section keys by assembly_order, with optional overrides."""

    def _order(k: str) -> int:
        if order_overrides and k in order_overrides:
            return order_overrides[k]
        if k in sections:
            return sections[k].assembly_order
        reg = SECTION_REGISTRY.get(k)
        return reg.assembly_order if reg else 999

    return sorted(keys, key=_order)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class VariantGenerator:
    """Generate ablation variants from a PromptSnapshot."""

    def __init__(self) -> None:
        self.structural = set(get_structurally_required())
        self.ablatable = set(get_ablatable())

    def generate_leave_one_out(
        self,
        snapshot: PromptSnapshot,
        layer: str = "post_hygiene",
        only_sections_with_content: bool = True,
    ) -> List[PromptVariant]:
        """Remove one ablatable section at a time.

        If only_sections_with_content=True (default), skip sections
        that were empty in this snapshot.
        """
        snap_layer = snapshot.layers.get(layer)
        if snap_layer is None:
            return []

        sections = snap_layer.sections
        all_keys = set(sections.keys())
        parent_tokens = _sum_tokens(sections, list(all_keys))
        variants: List[PromptVariant] = []

        for section_key in sorted(self.ablatable):
            if section_key not in all_keys:
                continue

            section = sections[section_key]
            if only_sections_with_content and not _section_has_content(section):
                continue

            active = _sort_keys_by_order(
                [k for k in all_keys if k != section_key],
                sections,
            )
            token_count = _sum_tokens(sections, active)

            variants.append(PromptVariant(
                variant_id=f"{snapshot.snapshot_id}_LOO_{section_key}",
                parent_snapshot_id=snapshot.snapshot_id,
                strategy=VariantStrategy.LEAVE_ONE_OUT,
                layer=layer,
                sections_removed=[section_key],
                active_sections=active,
                description=f"LOO: removed {section_key}",
                token_count_estimate=token_count,
                parent_token_count=parent_tokens,
            ))

        return variants

    def generate_add_one_in(
        self,
        snapshot: PromptSnapshot,
        layer: str = "post_hygiene",
        only_sections_with_content: bool = True,
    ) -> List[PromptVariant]:
        """Start from structural skeleton, add one ablatable section back.

        Identifies which single section provides the most value above
        the bare minimum.
        """
        snap_layer = snapshot.layers.get(layer)
        if snap_layer is None:
            return []

        sections = snap_layer.sections
        all_keys = set(sections.keys())

        skeleton_keys = sorted(
            [k for k in all_keys if k in self.structural],
            key=lambda k: sections[k].assembly_order,
        )
        skeleton_tokens = _sum_tokens(sections, skeleton_keys)

        variants: List[PromptVariant] = []

        for section_key in sorted(self.ablatable):
            if section_key not in all_keys:
                continue

            section = sections[section_key]
            if only_sections_with_content and not _section_has_content(section):
                continue

            active = _sort_keys_by_order(
                skeleton_keys + [section_key],
                sections,
            )
            token_count = skeleton_tokens + section.token_count

            variants.append(PromptVariant(
                variant_id=f"{snapshot.snapshot_id}_AOI_{section_key}",
                parent_snapshot_id=snapshot.snapshot_id,
                strategy=VariantStrategy.ADD_ONE_IN,
                layer=layer,
                sections_added=[section_key],
                active_sections=active,
                description=f"AOI: skeleton + {section_key}",
                token_count_estimate=token_count,
                parent_token_count=skeleton_tokens,
            ))

        return variants

    def generate_bundles(
        self,
        snapshot: PromptSnapshot,
        bundles: Dict[str, List[str]],
        layer: str = "post_hygiene",
    ) -> List[PromptVariant]:
        """Test pre-defined bundles of related sections.

        Each bundle is tested by removing all its sections at once.
        """
        snap_layer = snapshot.layers.get(layer)
        if snap_layer is None:
            return []

        sections = snap_layer.sections
        all_keys = set(sections.keys())
        parent_tokens = _sum_tokens(sections, list(all_keys))
        variants: List[PromptVariant] = []

        for bundle_name, bundle_sections in sorted(bundles.items()):
            removed = [
                s for s in bundle_sections
                if s in all_keys and s in self.ablatable
            ]
            if not removed:
                continue

            active = _sort_keys_by_order(
                [k for k in all_keys if k not in removed],
                sections,
            )
            token_count = _sum_tokens(sections, active)

            variants.append(PromptVariant(
                variant_id=f"{snapshot.snapshot_id}_BUNDLE_{bundle_name}",
                parent_snapshot_id=snapshot.snapshot_id,
                strategy=VariantStrategy.BUNDLE,
                layer=layer,
                sections_removed=removed,
                active_sections=active,
                description=f"Bundle '{bundle_name}': removed {len(removed)} sections",
                token_count_estimate=token_count,
                parent_token_count=parent_tokens,
            ))

        return variants

    def generate_reorder(
        self,
        snapshot: PromptSnapshot,
        section_key: str,
        new_order: int,
        layer: str = "post_hygiene",
    ) -> Optional[PromptVariant]:
        """Move one ablatable section to a different assembly_order position.

        Tests attention-position effects: does moving a section from its
        normal position to a high-attention zone (near the query) or a
        low-attention zone (middle of prompt) change response quality?

        Args:
            section_key: Section to reorder.
            new_order: Target assembly_order position (0-26).

        Returns:
            PromptVariant or None if the section isn't in this snapshot.
        """
        snap_layer = snapshot.layers.get(layer)
        if snap_layer is None:
            return None

        sections = snap_layer.sections
        if section_key not in sections:
            return None

        if section_key not in self.ablatable:
            return None

        section = sections[section_key]
        old_order = section.assembly_order
        if old_order == new_order:
            return None

        all_keys = list(sections.keys())
        overrides = {section_key: new_order}
        active = _sort_keys_by_order(all_keys, sections, order_overrides=overrides)
        parent_tokens = _sum_tokens(sections, all_keys)

        return PromptVariant(
            variant_id=f"{snapshot.snapshot_id}_REORDER_{section_key}_{new_order}",
            parent_snapshot_id=snapshot.snapshot_id,
            strategy=VariantStrategy.REORDER,
            layer=layer,
            reordered_sections={section_key: new_order},
            active_sections=active,
            description=f"Reorder: {section_key} from order {old_order} to {new_order}",
            token_count_estimate=parent_tokens,  # same total, different order
            parent_token_count=parent_tokens,
        )

    def generate_reorder_to_high_attention(
        self,
        snapshot: PromptSnapshot,
        layer: str = "post_hygiene",
        only_sections_with_content: bool = True,
    ) -> List[PromptVariant]:
        """Move each ablatable section to right before the query (high-attention zone).

        Tests whether a section performs better when placed in the high-attention
        area near the end of the prompt.
        """
        snap_layer = snapshot.layers.get(layer)
        if snap_layer is None:
            return []

        sections = snap_layer.sections

        # Place just before current_query (order 26) and after stm_summary (order 25)
        # Use 25 as target -- will sort after existing 24 (narrative_state) but
        # before 26 (current_query). For sections already at 24-25, skip.
        high_attention_order = 25
        variants: List[PromptVariant] = []

        for section_key in sorted(self.ablatable):
            if section_key not in sections:
                continue
            section = sections[section_key]
            if only_sections_with_content and not _section_has_content(section):
                continue
            if section.assembly_order >= 23:
                # Already in the high-attention tail, skip
                continue

            variant = self.generate_reorder(
                snapshot, section_key, high_attention_order, layer
            )
            if variant is not None:
                variants.append(variant)

        return variants

    def generate_all(
        self,
        snapshot: PromptSnapshot,
        bundles: Optional[Dict[str, List[str]]] = None,
        layer: str = "post_hygiene",
        include_reorder: bool = False,
    ) -> List[PromptVariant]:
        """Generate all variant types for a snapshot.

        Args:
            snapshot: Source snapshot.
            bundles: Bundle definitions. Uses DEFAULT_BUNDLES if None.
            layer: Which snapshot layer to use.
            include_reorder: Whether to include reorder-to-high-attention variants.

        Returns:
            LOO + AOI + bundle + (optionally) reorder variants.
        """
        effective_bundles = bundles if bundles is not None else DEFAULT_BUNDLES
        variants: List[PromptVariant] = []
        variants.extend(self.generate_leave_one_out(snapshot, layer))
        variants.extend(self.generate_add_one_in(snapshot, layer))
        variants.extend(self.generate_bundles(snapshot, effective_bundles, layer))
        if include_reorder:
            variants.extend(self.generate_reorder_to_high_attention(snapshot, layer))
        return variants


# ---------------------------------------------------------------------------
# Default bundles
# ---------------------------------------------------------------------------

DEFAULT_BUNDLES: Dict[str, List[str]] = {
    "memory_retrieval": [
        "memories",
        "recent_summaries",
        "semantic_summaries",
        "recent_reflections",
        "semantic_reflections",
    ],
    "external_knowledge": [
        "wiki",
        "web_search_results",
        "semantic_chunks",
    ],
    "proactive_context": [
        "proactive_insights",
        "unresolved_threads",
        "graph_context",
        "daemon_self_notes",
    ],
    "personal_context": [
        "user_profile",
        "personal_notes",
        "dreams",
        "user_uploads",
    ],
    "project_context": [
        "git_commits",
        "codebase_changes",
        "proposed_features",
        "procedural_skills",
    ],
    "narrative_and_meta": [
        "narrative_state",
        "stm_summary",
        "active_features",
    ],
    "system_documentation": [
        "reference_docs",
    ],
    "all_retrieved": [
        key
        for key, defn in SECTION_REGISTRY.items()
        if defn.category == SectionCategory.RETRIEVED
    ],
}
