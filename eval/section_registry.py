"""
Canonical section registry for Daemon's prompt assembly.

Built by inspecting core/prompt/formatter.py:_assemble_prompt() (lines 862-1642).
Each entry maps to exactly one sections.append() call in that method.

The registry is the source of truth for:
- Which sections exist and their assembly order
- Which sections are eligible for ablation
- Which sections are structurally required
- Validation that snapshots match the real builder output
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class SectionCategory(Enum):
    """Category of a prompt section."""
    STRUCTURAL = "structural"       # Required for prompt validity
    SYSTEM = "system"               # System-level instructions (separate from context)
    METADATA = "metadata"           # Computed metadata about the session/config
    RETRIEVED = "retrieved"         # Content retrieved from memory/knowledge stores
    GENERATED_CONTEXT = "generated_context"  # Derived/computed context (graph, insights, STM)


@dataclass(frozen=True)
class SectionDef:
    """Definition of a single prompt section."""
    internal_key: str
    header: str
    source_field: str
    category: SectionCategory
    eligible_for_ablation: bool
    structurally_required: bool
    assembly_order: int
    notes: str = ""

    def __post_init__(self):
        if self.structurally_required and self.eligible_for_ablation:
            raise ValueError(
                f"Section '{self.internal_key}' cannot be both structurally_required "
                f"and eligible_for_ablation"
            )


# ---------------------------------------------------------------------------
# Canonical registry — order matches _assemble_prompt() in formatter.py
# ---------------------------------------------------------------------------

SECTION_REGISTRY: Dict[str, SectionDef] = {}


def _register(s: SectionDef) -> SectionDef:
    SECTION_REGISTRY[s.internal_key] = s
    return s


# --- Structural (never ablatable) ---

_register(SectionDef(
    internal_key="current_query",
    header="[CURRENT USER QUERY]",
    source_field="user_input",
    category=SectionCategory.STRUCTURAL,
    eligible_for_ablation=False,
    structurally_required=True,
    assembly_order=26,
    notes="Always last. Contains [LAST EXCHANGE FOR CONTEXT] + [CURRENT QUERY].",
))

_register(SectionDef(
    internal_key="time_context",
    header="[TIME CONTEXT]",
    source_field="time_context",
    category=SectionCategory.STRUCTURAL,
    eligible_for_ablation=False,
    structurally_required=True,
    assembly_order=23,
    notes="Current date/time + time since last message/session. Computed by formatter.",
))

# --- Retrieved sections (ablatable) ---

_register(SectionDef(
    internal_key="recent_conversation",
    header="[RECENT CONVERSATION]",
    source_field="recent_conversations",
    category=SectionCategory.RETRIEVED,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=1,
))

_register(SectionDef(
    internal_key="memories",
    header="[RELEVANT MEMORIES]",
    source_field="memories",
    category=SectionCategory.RETRIEVED,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=2,
))

_register(SectionDef(
    internal_key="recent_summaries",
    header="[RECENT SUMMARIES]",
    source_field="recent_summaries",
    category=SectionCategory.RETRIEVED,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=3,
))

_register(SectionDef(
    internal_key="semantic_summaries",
    header="[SEMANTIC SUMMARIES]",
    source_field="semantic_summaries",
    category=SectionCategory.RETRIEVED,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=4,
))

_register(SectionDef(
    internal_key="recent_reflections",
    header="[RECENT REFLECTIONS]",
    source_field="recent_reflections",
    category=SectionCategory.RETRIEVED,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=5,
))

_register(SectionDef(
    internal_key="semantic_reflections",
    header="[SEMANTIC REFLECTIONS]",
    source_field="semantic_reflections",
    category=SectionCategory.RETRIEVED,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=6,
))

_register(SectionDef(
    internal_key="wiki",
    header="[BACKGROUND KNOWLEDGE]",
    source_field="wiki",
    category=SectionCategory.RETRIEVED,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=7,
))

_register(SectionDef(
    internal_key="web_search_results",
    header="[WEB SEARCH RESULTS]",
    source_field="web_search_results",
    category=SectionCategory.RETRIEVED,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=8,
    notes="Includes [WEB_N] source IDs and citation instruction.",
))

_register(SectionDef(
    internal_key="semantic_chunks",
    header="[RELEVANT INFORMATION]",
    source_field="semantic_chunks",
    category=SectionCategory.RETRIEVED,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=9,
))

_register(SectionDef(
    internal_key="dreams",
    header="[DREAMS]",
    source_field="dreams",
    category=SectionCategory.RETRIEVED,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=10,
))

_register(SectionDef(
    internal_key="personal_notes",
    header="[USER'S PERSONAL NOTES]",
    source_field="personal_notes",
    category=SectionCategory.RETRIEVED,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=11,
    notes="May include image indicators for multimodal models.",
))

_register(SectionDef(
    internal_key="user_uploads",
    header="[USER UPLOADED ITEMS]",
    source_field="user_uploads",
    category=SectionCategory.RETRIEVED,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=12,
    notes="User-uploaded files/images from sessions.",
))

_register(SectionDef(
    internal_key="reference_docs",
    header="[DAEMON DOCUMENTATION]",
    source_field="reference_docs",
    category=SectionCategory.RETRIEVED,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=13,
))

_register(SectionDef(
    internal_key="git_commits",
    header="[PROJECT COMMIT HISTORY]",
    source_field="git_commits",
    category=SectionCategory.RETRIEVED,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=14,
))

_register(SectionDef(
    internal_key="procedural_skills",
    header="[ADAPTIVE WORKFLOWS]",
    source_field="procedural_skills",
    category=SectionCategory.RETRIEVED,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=15,
))

_register(SectionDef(
    internal_key="proposed_features",
    header="[PROPOSED FEATURES]",
    source_field="proposed_features",
    category=SectionCategory.RETRIEVED,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=16,
))

# --- Generated context (ablatable) ---

_register(SectionDef(
    internal_key="graph_context",
    header="[KNOWLEDGE GRAPH]",
    source_field="graph_context",
    category=SectionCategory.GENERATED_CONTEXT,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=17,
    notes="May include '(derived relationships)' suffix when attribution enabled.",
))

_register(SectionDef(
    internal_key="unresolved_threads",
    header="[UNRESOLVED THREADS]",
    source_field="unresolved_threads",
    category=SectionCategory.GENERATED_CONTEXT,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=18,
))

_register(SectionDef(
    internal_key="proactive_insights",
    header="[PROACTIVE INSIGHTS]",
    source_field="proactive_insights",
    category=SectionCategory.GENERATED_CONTEXT,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=19,
    notes="Includes disclaimer about AI-generated insights.",
))

_register(SectionDef(
    internal_key="visual_memories",
    header="[VISUAL MEMORIES]",
    source_field="visual_memories",
    category=SectionCategory.RETRIEVED,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=27,
    notes="CLIP-matched images from personal collection. Includes captions and entity tags.",
))

_register(SectionDef(
    internal_key="narrative_state",
    header="[TEMPORAL GROUNDING]",
    source_field="narrative_state",
    category=SectionCategory.GENERATED_CONTEXT,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=24,
    notes="Synthesized life-state narrative. Placed after time context.",
))

_register(SectionDef(
    internal_key="stm_summary",
    header="[SHORT-TERM CONTEXT SUMMARY]",
    source_field="stm_summary",
    category=SectionCategory.GENERATED_CONTEXT,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=25,
    notes="Topic, intent, tone, reference_type warnings. Right before query.",
))

# --- Metadata (conditionally ablatable) ---

_register(SectionDef(
    internal_key="user_profile",
    header="[USER PROFILE]",
    source_field="user_profile",
    category=SectionCategory.METADATA,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=20,
    notes="Stored user facts with anti-confabulation instruction.",
))

_register(SectionDef(
    internal_key="active_features",
    header="[ACTIVE FEATURES]",
    source_field="active_features",
    category=SectionCategory.METADATA,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=21,
    notes="Computed from config flags. No source_field in context dict.",
))

_register(SectionDef(
    internal_key="codebase_changes",
    header="[CODEBASE CHANGES SINCE LAST SESSION]",
    source_field="codebase_changes",
    category=SectionCategory.METADATA,
    eligible_for_ablation=True,
    structurally_required=False,
    assembly_order=22,
    notes="First message of session only.",
))

# --- System prompt (separate, never ablatable) ---

_register(SectionDef(
    internal_key="system_prompt",
    header="(system prompt)",
    source_field="system_prompt",
    category=SectionCategory.SYSTEM,
    eligible_for_ablation=False,
    structurally_required=True,
    assembly_order=0,
    notes="Composed in orchestrator. Separate from context prompt. "
          "Includes personality, operating principles, identity substitution, "
          "citation instructions, topic context, thread surfacing, escalation, "
          "thinking instructions, and response plan.",
))


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_all() -> List[SectionDef]:
    """Return all registered sections, sorted by assembly_order."""
    return sorted(SECTION_REGISTRY.values(), key=lambda s: s.assembly_order)


def get_by_key(key: str) -> Optional[SectionDef]:
    """Look up a section by internal key."""
    return SECTION_REGISTRY.get(key)


def get_by_source_field(source_field: str) -> Optional[SectionDef]:
    """Look up a section by its context dict source field."""
    for s in SECTION_REGISTRY.values():
        if s.source_field == source_field:
            return s
    return None


def get_structurally_required() -> List[str]:
    """Return internal keys of structurally required sections."""
    return [s.internal_key for s in SECTION_REGISTRY.values() if s.structurally_required]


def get_ablatable() -> List[str]:
    """Return internal keys of sections eligible for ablation."""
    return [s.internal_key for s in SECTION_REGISTRY.values() if s.eligible_for_ablation]


def get_never_ablate() -> List[str]:
    """Return internal keys of sections that must never be ablated."""
    return [s.internal_key for s in SECTION_REGISTRY.values() if not s.eligible_for_ablation]


def get_context_sections() -> List[SectionDef]:
    """Return all sections that appear in the assembled context prompt (not system)."""
    return sorted(
        [s for s in SECTION_REGISTRY.values() if s.category != SectionCategory.SYSTEM],
        key=lambda s: s.assembly_order,
    )


def validate_registry_against_formatted_sections(section_keys: List[str]) -> List[str]:
    """Check that all emitted section keys are known in the registry.

    Args:
        section_keys: List of section internal_keys found in a real prompt build.

    Returns:
        List of unknown keys not in the registry (ideally empty).
    """
    known = set(SECTION_REGISTRY.keys())
    return [k for k in section_keys if k not in known]


def match_header_to_key(header_line: str) -> Optional[str]:
    """Given a header line like '[RECENT CONVERSATION] n=5', find the registry key.

    Strips the n=... suffix and optional suffixes like '(derived relationships)'.
    """
    # Extract the bracketed header
    import re
    m = re.match(r"(\[[^\]]+\])", header_line)
    if not m:
        return None
    header = m.group(1)
    for s in SECTION_REGISTRY.values():
        if s.header == header:
            return s.internal_key
    return None
