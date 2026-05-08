"""Tests for variant generation from frozen snapshots."""

import pytest

from eval.schema import (
    PromptProvenance,
    PromptSnapshot,
    SectionSnapshot,
    SnapshotLayer,
    compute_prompt_hash,
)
from eval.section_registry import SECTION_REGISTRY, get_ablatable, get_structurally_required
from eval.variants import (
    DEFAULT_BUNDLES,
    PromptVariant,
    VariantGenerator,
    VariantStrategy,
    _section_has_content,
    _sort_keys_by_order,
    _sum_tokens,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_section(key: str, text: str = "", token_count: int = 0) -> SectionSnapshot:
    """Build a SectionSnapshot from the registry, with given text/tokens."""
    reg = SECTION_REGISTRY.get(key)
    if reg is None:
        raise ValueError(f"Unknown section key: {key}")
    return SectionSnapshot(
        key=key,
        header=reg.header,
        structured_content=text if text else None,
        formatted_text=text,
        token_count=token_count if token_count else max(1, len(text) // 4),
        source_field=reg.source_field,
        category=reg.category.value,
        eligible_for_ablation=reg.eligible_for_ablation,
        structurally_required=reg.structurally_required,
        assembly_order=reg.assembly_order,
    )


def _make_snapshot(
    section_keys: list[str] | None = None,
    snapshot_id: str = "test1234",
) -> PromptSnapshot:
    """Build a minimal PromptSnapshot with the given sections populated."""
    if section_keys is None:
        section_keys = [
            "current_query",
            "time_context",
            "memories",
            "user_profile",
            "recent_conversation",
            "wiki",
            "stm_summary",
        ]

    sections = {}
    for key in section_keys:
        sections[key] = _make_section(
            key, text=f"[{key.upper()}] n=1\nContent for {key}", token_count=100
        )

    prompt_text = "\n\n".join(
        s.formatted_text
        for s in sorted(sections.values(), key=lambda s: s.assembly_order)
    )

    layer = SnapshotLayer(
        layer_name="post_hygiene",
        sections=sections,
        layer_content_hash="fakehash",
        prompt_text=prompt_text,
        prompt_hash_exact=compute_prompt_hash(prompt_text),
        prompt_hash_normalized=compute_prompt_hash(prompt_text, normalize=True),
        capture_timestamp="2026-05-05T00:00:00+00:00",
    )

    provenance = PromptProvenance(
        model_name="test-model",
        git_commit_hash="abc1234",
        system_prompt_hash="sys_hash",
    )

    return PromptSnapshot(
        snapshot_id=snapshot_id,
        query_text="test query",
        query_timestamp="2026-05-05T00:00:00+00:00",
        processed_query="test query",
        detected_intent="general",
        detected_tone="CONVERSATIONAL",
        provenance=provenance,
        layers={"post_hygiene": layer},
        retrieval_metadata={},
        assembly_metadata={},
    )


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_section_has_content_nonempty(self):
        s = _make_section("memories", text="Some real content here")
        assert _section_has_content(s) is True

    def test_section_has_content_empty(self):
        s = _make_section("memories", text="")
        assert _section_has_content(s) is False

    def test_section_has_content_whitespace_only(self):
        s = _make_section("memories", text="   \n  ")
        assert _section_has_content(s) is False

    def test_sum_tokens(self):
        sections = {
            "memories": _make_section("memories", token_count=50),
            "wiki": _make_section("wiki", token_count=30),
        }
        assert _sum_tokens(sections, ["memories", "wiki"]) == 80
        assert _sum_tokens(sections, ["memories"]) == 50
        assert _sum_tokens(sections, ["nonexistent"]) == 0

    def test_sort_keys_by_order(self):
        sections = {
            "memories": _make_section("memories"),       # order 2
            "current_query": _make_section("current_query"),  # order 26
            "time_context": _make_section("time_context"),    # order 23
        }
        result = _sort_keys_by_order(list(sections.keys()), sections)
        assert result == ["memories", "time_context", "current_query"]

    def test_sort_keys_with_override(self):
        sections = {
            "memories": _make_section("memories"),       # order 2
            "wiki": _make_section("wiki"),               # order 7
        }
        result = _sort_keys_by_order(
            list(sections.keys()), sections, order_overrides={"memories": 99}
        )
        assert result == ["wiki", "memories"]


# ---------------------------------------------------------------------------
# Leave-one-out tests
# ---------------------------------------------------------------------------

class TestLeaveOneOut:
    def test_loo_removes_one_section_per_variant(self):
        snap = _make_snapshot()
        gen = VariantGenerator()
        variants = gen.generate_leave_one_out(snap)
        for v in variants:
            assert len(v.sections_removed) == 1
            assert v.strategy == VariantStrategy.LEAVE_ONE_OUT

    def test_loo_never_removes_structural(self):
        snap = _make_snapshot()
        gen = VariantGenerator()
        variants = gen.generate_leave_one_out(snap)
        structural = set(get_structurally_required())
        for v in variants:
            for removed in v.sections_removed:
                assert removed not in structural

    def test_loo_skip_empty_sections(self):
        """Empty sections skipped when only_sections_with_content=True."""
        snap = _make_snapshot()
        # Make wiki section empty
        snap.layers["post_hygiene"].sections["wiki"] = _make_section("wiki", text="")
        gen = VariantGenerator()
        variants = gen.generate_leave_one_out(snap, only_sections_with_content=True)
        removed_keys = {v.sections_removed[0] for v in variants}
        assert "wiki" not in removed_keys

    def test_loo_include_empty_when_flag_false(self):
        """Empty sections included when only_sections_with_content=False."""
        snap = _make_snapshot()
        snap.layers["post_hygiene"].sections["wiki"] = _make_section("wiki", text="")
        gen = VariantGenerator()
        variants = gen.generate_leave_one_out(snap, only_sections_with_content=False)
        removed_keys = {v.sections_removed[0] for v in variants}
        assert "wiki" in removed_keys

    def test_loo_token_estimate_lower(self):
        """LOO variant token count < parent token count."""
        snap = _make_snapshot()
        gen = VariantGenerator()
        variants = gen.generate_leave_one_out(snap)
        for v in variants:
            assert v.token_count_estimate < v.parent_token_count

    def test_loo_returns_empty_for_missing_layer(self):
        snap = _make_snapshot()
        gen = VariantGenerator()
        variants = gen.generate_leave_one_out(snap, layer="raw_retrieval")
        assert variants == []

    def test_loo_variant_ids_unique(self):
        snap = _make_snapshot()
        gen = VariantGenerator()
        variants = gen.generate_leave_one_out(snap)
        ids = [v.variant_id for v in variants]
        assert len(ids) == len(set(ids))

    def test_loo_sections_in_assembly_order(self):
        """Active sections are sorted by assembly_order."""
        snap = _make_snapshot()
        gen = VariantGenerator()
        variants = gen.generate_leave_one_out(snap)
        for v in variants:
            sections = snap.layers["post_hygiene"].sections
            orders = [sections[k].assembly_order for k in v.active_sections]
            assert orders == sorted(orders)


# ---------------------------------------------------------------------------
# Add-one-in tests
# ---------------------------------------------------------------------------

class TestAddOneIn:
    def test_aoi_starts_from_skeleton(self):
        """AOI variants always include all structural sections."""
        snap = _make_snapshot()
        gen = VariantGenerator()
        variants = gen.generate_add_one_in(snap)
        structural = set(get_structurally_required())
        snap_structural = structural & set(snap.layers["post_hygiene"].sections.keys())
        for v in variants:
            for s in snap_structural:
                assert s in v.active_sections

    def test_aoi_adds_exactly_one_section(self):
        """Each AOI variant adds exactly one section beyond skeleton."""
        snap = _make_snapshot()
        gen = VariantGenerator()
        variants = gen.generate_add_one_in(snap)
        for v in variants:
            assert len(v.sections_added) == 1
            assert v.strategy == VariantStrategy.ADD_ONE_IN

    def test_aoi_token_count_reasonable(self):
        snap = _make_snapshot()
        gen = VariantGenerator()
        variants = gen.generate_add_one_in(snap)
        for v in variants:
            # parent_token_count is the skeleton, estimate should be >= it
            assert v.token_count_estimate >= v.parent_token_count

    def test_aoi_variant_ids_unique(self):
        snap = _make_snapshot()
        gen = VariantGenerator()
        variants = gen.generate_add_one_in(snap)
        ids = [v.variant_id for v in variants]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Bundle tests
# ---------------------------------------------------------------------------

class TestBundles:
    def test_bundle_removes_all_bundle_members(self):
        """Bundle variant removes all specified sections present in snapshot."""
        snap = _make_snapshot()
        gen = VariantGenerator()
        bundles = {"test_bundle": ["memories", "wiki"]}
        variants = gen.generate_bundles(snap, bundles)
        assert len(variants) == 1
        v = variants[0]
        assert set(v.sections_removed) == {"memories", "wiki"}
        assert "memories" not in v.active_sections
        assert "wiki" not in v.active_sections

    def test_bundle_skips_absent_sections(self):
        """Bundle sections not in snapshot are not listed as removed."""
        snap = _make_snapshot(section_keys=["current_query", "time_context", "memories"])
        gen = VariantGenerator()
        bundles = {"test_bundle": ["memories", "wiki"]}  # wiki not in snapshot
        variants = gen.generate_bundles(snap, bundles)
        assert len(variants) == 1
        assert variants[0].sections_removed == ["memories"]

    def test_bundle_empty_when_no_match(self):
        """No variant if none of the bundle sections are in the snapshot."""
        snap = _make_snapshot(section_keys=["current_query", "time_context"])
        gen = VariantGenerator()
        bundles = {"test_bundle": ["memories", "wiki"]}
        variants = gen.generate_bundles(snap, bundles)
        assert variants == []

    def test_default_bundles_all_retrieved_populated(self):
        """all_retrieved bundle is populated, not an empty list."""
        assert len(DEFAULT_BUNDLES["all_retrieved"]) > 0
        # Should include known retrieved sections
        assert "memories" in DEFAULT_BUNDLES["all_retrieved"]
        assert "wiki" in DEFAULT_BUNDLES["all_retrieved"]
        assert "recent_conversation" in DEFAULT_BUNDLES["all_retrieved"]


# ---------------------------------------------------------------------------
# Reorder tests
# ---------------------------------------------------------------------------

class TestReorder:
    def test_reorder_single_section(self):
        snap = _make_snapshot()
        gen = VariantGenerator()
        v = gen.generate_reorder(snap, "memories", 25)
        assert v is not None
        assert v.strategy == VariantStrategy.REORDER
        assert v.reordered_sections == {"memories": 25}
        assert "Reorder" in v.description

    def test_reorder_changes_position(self):
        """After reorder, memories should appear later in active_sections."""
        snap = _make_snapshot()
        gen = VariantGenerator()
        v = gen.generate_reorder(snap, "memories", 25)
        assert v is not None
        # memories (originally order 2) should now sort near the end
        idx = v.active_sections.index("memories")
        assert idx > 0  # not first anymore

    def test_reorder_noop_same_position(self):
        """Reorder to same position returns None."""
        snap = _make_snapshot()
        gen = VariantGenerator()
        original_order = snap.layers["post_hygiene"].sections["memories"].assembly_order
        v = gen.generate_reorder(snap, "memories", original_order)
        assert v is None

    def test_reorder_structural_rejected(self):
        """Cannot reorder structurally required sections."""
        snap = _make_snapshot()
        gen = VariantGenerator()
        v = gen.generate_reorder(snap, "current_query", 1)
        assert v is None

    def test_reorder_missing_section(self):
        snap = _make_snapshot(section_keys=["current_query", "time_context"])
        gen = VariantGenerator()
        v = gen.generate_reorder(snap, "memories", 25)
        assert v is None

    def test_reorder_to_high_attention(self):
        snap = _make_snapshot()
        gen = VariantGenerator()
        variants = gen.generate_reorder_to_high_attention(snap)
        # Should produce variants for ablatable sections with content
        # that aren't already in positions 23+
        assert len(variants) > 0
        for v in variants:
            assert v.strategy == VariantStrategy.REORDER

    def test_reorder_high_attention_skips_tail_sections(self):
        """Sections already at order >= 23 are not reordered."""
        snap = _make_snapshot()
        gen = VariantGenerator()
        variants = gen.generate_reorder_to_high_attention(snap)
        reordered_keys = set()
        for v in variants:
            reordered_keys.update(v.reordered_sections.keys())
        # stm_summary (25) and narrative_state (24) should not be reordered
        assert "stm_summary" not in reordered_keys

    def test_reorder_token_count_unchanged(self):
        """Reorder doesn't change total tokens, just order."""
        snap = _make_snapshot()
        gen = VariantGenerator()
        v = gen.generate_reorder(snap, "memories", 25)
        assert v is not None
        assert v.token_count_estimate == v.parent_token_count


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_variant_roundtrip(self):
        v = PromptVariant(
            variant_id="snap_LOO_memories",
            parent_snapshot_id="snap1234",
            strategy=VariantStrategy.LEAVE_ONE_OUT,
            layer="post_hygiene",
            sections_removed=["memories"],
            active_sections=["time_context", "current_query"],
            description="LOO: removed memories",
            token_count_estimate=200,
            parent_token_count=300,
        )
        d = v.to_dict()
        v2 = PromptVariant.from_dict(d)
        assert v2.variant_id == v.variant_id
        assert v2.strategy == VariantStrategy.LEAVE_ONE_OUT
        assert v2.sections_removed == ["memories"]
        assert v2.token_count_estimate == 200

    def test_variant_dict_strategy_is_string(self):
        v = PromptVariant(
            variant_id="test",
            parent_snapshot_id="snap",
            strategy=VariantStrategy.REORDER,
            layer="post_hygiene",
        )
        d = v.to_dict()
        assert d["strategy"] == "reorder"


# ---------------------------------------------------------------------------
# generate_all tests
# ---------------------------------------------------------------------------

class TestGenerateAll:
    def test_generate_all_produces_all_types(self):
        snap = _make_snapshot()
        gen = VariantGenerator()
        variants = gen.generate_all(snap, include_reorder=True)
        strategies = {v.strategy for v in variants}
        assert VariantStrategy.LEAVE_ONE_OUT in strategies
        assert VariantStrategy.ADD_ONE_IN in strategies
        assert VariantStrategy.BUNDLE in strategies
        assert VariantStrategy.REORDER in strategies

    def test_generate_all_without_reorder(self):
        snap = _make_snapshot()
        gen = VariantGenerator()
        variants = gen.generate_all(snap, include_reorder=False)
        strategies = {v.strategy for v in variants}
        assert VariantStrategy.REORDER not in strategies

    def test_generate_all_variant_ids_unique(self):
        snap = _make_snapshot()
        gen = VariantGenerator()
        variants = gen.generate_all(snap, include_reorder=True)
        ids = [v.variant_id for v in variants]
        assert len(ids) == len(set(ids))

    def test_generate_all_uses_default_bundles(self):
        """When bundles=None, DEFAULT_BUNDLES are used."""
        snap = _make_snapshot()
        gen = VariantGenerator()
        variants = gen.generate_all(snap)
        bundle_variants = [v for v in variants if v.strategy == VariantStrategy.BUNDLE]
        assert len(bundle_variants) > 0
