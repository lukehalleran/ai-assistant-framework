"""Tests for eval section registry."""

import pytest
from eval.section_registry import (
    SECTION_REGISTRY,
    SectionCategory,
    SectionDef,
    get_ablatable,
    get_all,
    get_by_key,
    get_by_source_field,
    get_context_sections,
    get_never_ablate,
    get_structurally_required,
    match_header_to_key,
    validate_registry_against_formatted_sections,
)


class TestRegistryIntegrity:
    """Test structural integrity of the section registry."""

    def test_assembly_order_unique(self):
        """All assembly_order values must be unique."""
        orders = [s.assembly_order for s in SECTION_REGISTRY.values()]
        assert len(orders) == len(set(orders)), (
            f"Duplicate assembly_order values: "
            f"{[o for o in orders if orders.count(o) > 1]}"
        )

    def test_structurally_required_not_ablatable(self):
        """Structurally required sections must not be ablatable."""
        for key, s in SECTION_REGISTRY.items():
            if s.structurally_required:
                assert not s.eligible_for_ablation, (
                    f"Section '{key}' is structurally_required but also "
                    f"eligible_for_ablation — this is contradictory"
                )

    def test_current_query_exists_and_required(self):
        """current_query must exist and be structurally required."""
        s = get_by_key("current_query")
        assert s is not None, "current_query not in registry"
        assert s.structurally_required is True
        assert s.eligible_for_ablation is False

    def test_system_prompt_exists_and_required(self):
        """system_prompt must exist and be structurally required."""
        s = get_by_key("system_prompt")
        assert s is not None, "system_prompt not in registry"
        assert s.structurally_required is True
        assert s.eligible_for_ablation is False

    def test_time_context_exists_and_required(self):
        """time_context must exist and be structurally required."""
        s = get_by_key("time_context")
        assert s is not None, "time_context not in registry"
        assert s.structurally_required is True

    def test_stm_summary_is_ablatable(self):
        """STM summary should be ablatable (verified against actual code)."""
        s = get_by_key("stm_summary")
        assert s is not None, "stm_summary not in registry"
        assert s.eligible_for_ablation is True
        assert s.structurally_required is False

    def test_registry_has_26_context_sections(self):
        """Registry should have 27 context sections + system_prompt (matching builder output)."""
        # system_prompt is entry 0 (separate), so 28 total
        assert len(SECTION_REGISTRY) == 28

    def test_all_internal_keys_are_unique(self):
        """Internal keys should all be unique (enforced by dict but be explicit)."""
        keys = list(SECTION_REGISTRY.keys())
        assert len(keys) == len(set(keys))


class TestQueryHelpers:
    """Test registry query functions."""

    def test_get_structurally_required(self):
        required = get_structurally_required()
        assert "current_query" in required
        assert "time_context" in required
        assert "system_prompt" in required
        # Retrieved sections should NOT be required
        assert "memories" not in required
        assert "wiki" not in required

    def test_get_ablatable(self):
        ablatable = get_ablatable()
        assert "memories" in ablatable
        assert "wiki" in ablatable
        assert "stm_summary" in ablatable
        # Required sections should NOT be ablatable
        assert "current_query" not in ablatable
        assert "system_prompt" not in ablatable

    def test_get_never_ablate(self):
        never = get_never_ablate()
        assert "current_query" in never
        assert "system_prompt" in never
        assert "time_context" in never

    def test_ablatable_and_never_are_disjoint(self):
        ablatable = set(get_ablatable())
        never = set(get_never_ablate())
        overlap = ablatable & never
        assert len(overlap) == 0, f"Sections in both ablatable and never_ablate: {overlap}"

    def test_ablatable_plus_never_covers_all(self):
        ablatable = set(get_ablatable())
        never = set(get_never_ablate())
        all_keys = set(SECTION_REGISTRY.keys())
        assert ablatable | never == all_keys

    def test_get_by_source_field(self):
        s = get_by_source_field("memories")
        assert s is not None
        assert s.internal_key == "memories"

    def test_get_by_source_field_returns_none_for_unknown(self):
        assert get_by_source_field("nonexistent_field") is None

    def test_get_context_sections_excludes_system(self):
        ctx = get_context_sections()
        keys = [s.internal_key for s in ctx]
        assert "system_prompt" not in keys
        assert "current_query" in keys

    def test_get_all_sorted_by_assembly_order(self):
        all_sections = get_all()
        orders = [s.assembly_order for s in all_sections]
        assert orders == sorted(orders)


class TestHeaderMatching:
    """Test header-to-key matching."""

    def test_match_simple_header(self):
        assert match_header_to_key("[RECENT CONVERSATION] n=5") == "recent_conversation"

    def test_match_header_with_suffix(self):
        assert match_header_to_key("[KNOWLEDGE GRAPH] n=3 (derived relationships)") == "graph_context"

    def test_match_header_no_count(self):
        assert match_header_to_key("[TIME CONTEXT]") == "time_context"

    def test_match_unknown_header(self):
        assert match_header_to_key("[SOMETHING UNKNOWN] n=1") is None

    def test_match_non_header(self):
        assert match_header_to_key("Just some text") is None


class TestValidation:
    """Test registry validation against real builder output."""

    def test_validate_known_keys_pass(self):
        """Known keys should produce no unknown entries."""
        known = ["recent_conversation", "memories", "current_query"]
        unknown = validate_registry_against_formatted_sections(known)
        assert unknown == []

    def test_validate_unknown_keys_reported(self):
        """Unknown keys should be reported."""
        keys = ["memories", "totally_new_section"]
        unknown = validate_registry_against_formatted_sections(keys)
        assert "totally_new_section" in unknown

    def test_validate_empty_keys(self):
        assert validate_registry_against_formatted_sections([]) == []


class TestSectionDefConstraint:
    """Test SectionDef data constraints."""

    def test_required_and_ablatable_raises(self):
        """SectionDef should reject structurally_required + eligible_for_ablation."""
        with pytest.raises(ValueError, match="cannot be both"):
            SectionDef(
                internal_key="test",
                header="[TEST]",
                source_field="test",
                category=SectionCategory.STRUCTURAL,
                eligible_for_ablation=True,
                structurally_required=True,
                assembly_order=99,
            )

    def test_frozen_dataclass(self):
        """SectionDef instances should be immutable."""
        s = get_by_key("memories")
        with pytest.raises(AttributeError):
            s.eligible_for_ablation = False


class TestRegistryMatchesBuilderSections:
    """Verify that the registry covers the sections the builder actually emits.

    Uses the known section headers from builder.py:_assemble_prompt() grep results.
    """

    KNOWN_BUILDER_HEADERS = [
        "[RECENT CONVERSATION]",
        "[RELEVANT MEMORIES]",
        "[RECENT SUMMARIES]",
        "[SEMANTIC SUMMARIES]",
        "[RECENT REFLECTIONS]",
        "[SEMANTIC REFLECTIONS]",
        "[BACKGROUND KNOWLEDGE]",
        "[WEB SEARCH RESULTS]",
        "[RELEVANT INFORMATION]",
        "[DREAMS]",
        "[USER'S PERSONAL NOTES]",
        "[USER UPLOADED ITEMS]",
        "[DAEMON DOCUMENTATION]",
        "[PROJECT COMMIT HISTORY]",
        "[ADAPTIVE WORKFLOWS]",
        "[PROPOSED FEATURES]",
        "[KNOWLEDGE GRAPH]",
        "[UNRESOLVED THREADS]",
        "[PROACTIVE INSIGHTS]",
        "[USER PROFILE]",
        "[ACTIVE FEATURES]",
        "[CODEBASE CHANGES SINCE LAST SESSION]",
        "[TIME CONTEXT]",
        "[TEMPORAL GROUNDING]",
        "[SHORT-TERM CONTEXT SUMMARY]",
        "[CURRENT USER QUERY]",
    ]

    def test_all_builder_headers_are_in_registry(self):
        """Every header the builder emits should map to a registry entry."""
        registry_headers = {s.header for s in SECTION_REGISTRY.values()}
        missing = []
        for header in self.KNOWN_BUILDER_HEADERS:
            if header not in registry_headers:
                missing.append(header)
        assert missing == [], f"Builder headers not in registry: {missing}"

    def test_all_builder_headers_resolve_via_match(self):
        """match_header_to_key should resolve every known builder header."""
        unresolved = []
        for header in self.KNOWN_BUILDER_HEADERS:
            key = match_header_to_key(header)
            if key is None:
                unresolved.append(header)
        assert unresolved == [], f"Headers that don't resolve: {unresolved}"
