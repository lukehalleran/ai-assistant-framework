"""
Tests for config/schema.py Pydantic validation layer.

Covers:
- Happy path with real config.yaml
- Type rejection (wrong types)
- Range rejection (thresholds out of bounds)
- Cross-field validators (weights sum, ordering)
- Unknown key warnings
- Minimal/empty config (defaults fill in)
- String coercion from resolve_vars
"""

import logging
import pytest
from pydantic import ValidationError

from config.schema import (
    DaemonConfig,
    DaemonSection,
    MemorySection,
    ModelsSection,
    GatingSection,
    FeaturesSection,
    WebSearchSection,
    TokenBudgetSection,
    SynthesisSection,
    TagGenerationSection,
    CodeProposalsSection,
    ScoreWeights,
    SynthesisWeights,
    CollectionBoosts,
    validate_config,
)


class TestRealConfig:
    """Validate the actual config.yaml from the project."""

    def test_real_config_validates(self):
        """The real config.yaml should pass validation."""
        from config.app_config import load_yaml_config, ensure_config_defaults
        config = load_yaml_config("config.yaml")
        config = ensure_config_defaults(config)
        result = validate_config(config)
        assert result is config  # Returns same dict

    def test_real_config_parses_to_model(self):
        """The real config.yaml should produce a valid DaemonConfig."""
        from config.app_config import load_yaml_config, ensure_config_defaults
        config = load_yaml_config("config.yaml")
        config = ensure_config_defaults(config)
        model = DaemonConfig(**config)
        assert model.daemon.version == "v4"
        assert model.memory.chroma_path == "./data/chroma_db_v4"


class TestEmptyConfig:
    """Validate that an empty config fills in all defaults."""

    def test_empty_dict_validates(self):
        """An empty config should succeed — all fields have defaults."""
        result = validate_config({})
        assert result == {}

    def test_empty_dict_model(self):
        """An empty config should produce a DaemonConfig with all defaults."""
        model = DaemonConfig()
        assert model.daemon.version == "v4"
        assert model.memory.corpus_file == "./data/corpus_v4.json"
        assert model.gating.cosine_similarity_threshold == 0.25
        assert model.token_budget.default == 15000


class TestTypeRejection:
    """Validate that wrong types are rejected."""

    def test_string_for_int(self):
        """A non-numeric string for an int field should fail."""
        with pytest.raises(ValidationError):
            MemorySection(max_recent="not_a_number")

    def test_string_for_float(self):
        """A non-numeric string for a float field should fail."""
        with pytest.raises(ValidationError):
            GatingSection(cosine_similarity_threshold="high")

    def test_string_for_bool(self):
        """A non-boolean string for a bool field should fail."""
        with pytest.raises(ValidationError):
            DaemonSection(debug_mode="maybe")

    def test_list_for_string(self):
        """A list for a string field should fail."""
        with pytest.raises(ValidationError):
            DaemonSection(version=[1, 2, 3])


class TestStringCoercion:
    """Validate that string-encoded values from resolve_vars are coerced."""

    def test_string_int_coercion(self):
        """String-encoded integers should be coerced to int."""
        m = MemorySection(max_recent="50")
        assert m.max_recent == 50

    def test_string_float_coercion(self):
        """String-encoded floats should be coerced to float."""
        g = GatingSection(cosine_similarity_threshold="0.25")
        assert g.cosine_similarity_threshold == 0.25

    def test_string_bool_coercion(self):
        """String-encoded booleans should be coerced."""
        d = DaemonSection(debug_mode="true")
        assert d.debug_mode is True


class TestRangeRejection:
    """Validate that out-of-range values are rejected."""

    def test_threshold_above_one(self):
        """A 0-1 threshold above 1.0 should fail."""
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            GatingSection(cosine_similarity_threshold=1.5)

    def test_threshold_below_zero(self):
        """A threshold below 0.0 should fail."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            GatingSection(cosine_similarity_threshold=-0.1)

    def test_negative_count(self):
        """A count that must be >= 1 should reject 0."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            MemorySection(max_recent=0)

    def test_zero_timeout(self):
        """A timeout that must be > 0.0 should reject 0."""
        with pytest.raises(ValidationError, match="greater than 0"):
            WebSearchSection(timeout_s=0.0)

    def test_confidence_threshold_allows_above_one(self):
        """confidence_threshold is special — it allows values > 1.0."""
        g = GatingSection(confidence_threshold=2.0)
        assert g.confidence_threshold == 2.0

    def test_temperature_allows_up_to_two(self):
        """Temperature can go up to 2.0."""
        m = ModelsSection(default_temperature=1.5)
        assert m.default_temperature == 1.5

    def test_temperature_rejects_above_two(self):
        """Temperature above 2.0 should fail."""
        with pytest.raises(ValidationError):
            ModelsSection(default_temperature=3.0)


class TestCrossFieldValidators:
    """Validate cross-field constraints."""

    def test_score_weights_sum_warning(self, caplog):
        """Score weights that don't sum to ~1.0 should warn."""
        with caplog.at_level(logging.WARNING, logger="config.schema"):
            ScoreWeights(relevance=0.1, recency=0.1, truth=0.1)
        assert "sum to" in caplog.text

    def test_score_weights_valid_sum(self, caplog):
        """Score weights that sum to ~1.0 should not warn."""
        with caplog.at_level(logging.WARNING, logger="config.schema"):
            ScoreWeights(
                relevance=0.30, recency=0.22, truth=0.18,
                importance=0.05, continuity=0.10, structure=0.05,
                topic_match=0.10
            )
        assert "sum to" not in caplog.text

    def test_synthesis_weights_sum_warning(self, caplog):
        """Synthesis weights that don't sum to ~1.0 should warn."""
        with caplog.at_level(logging.WARNING, logger="config.schema"):
            SynthesisWeights(coherence=0.1, novelty=0.1, distance=0.1, structural=0.1)
        assert "sum to" in caplog.text

    def test_token_budget_floor_above_default(self):
        """floor > default should fail."""
        with pytest.raises(ValidationError, match="floor.*must be <= default"):
            TokenBudgetSection(floor=20000, default=15000, ceiling=25000)

    def test_token_budget_default_above_ceiling(self):
        """default > ceiling should fail."""
        with pytest.raises(ValidationError, match="default.*must be <= ceiling"):
            TokenBudgetSection(floor=5000, default=20000, ceiling=15000)

    def test_token_budget_valid_ordering(self):
        """Valid floor <= default <= ceiling should succeed."""
        t = TokenBudgetSection(floor=8000, default=15000, ceiling=16000)
        assert t.floor <= t.default <= t.ceiling

    def test_synthesis_distance_min_above_max(self):
        """distance_min >= distance_max should fail."""
        with pytest.raises(ValidationError, match="distance_min.*must be < distance_max"):
            SynthesisSection(distance_min=0.9, distance_max=0.2)

    def test_tag_min_above_max(self):
        """min_tags > max_tags should fail."""
        with pytest.raises(ValidationError, match="min_tags.*must be <= max_tags"):
            TagGenerationSection(min_tags=10, max_tags=3)

    def test_code_proposal_weights_sum_warning(self, caplog):
        """Code proposal weights that don't sum to ~1.0 should warn."""
        with caplog.at_level(logging.WARNING, logger="config.schema"):
            CodeProposalsSection(
                weight_priority=0.1, weight_breadth=0.1,
                weight_recency=0.1, weight_goal_alignment=0.1
            )
        assert "sum to" in caplog.text


class TestDaemonModeValidation:
    """Validate daemon.mode enum."""

    def test_valid_dev_mode(self):
        d = DaemonSection(mode="dev")
        assert d.mode == "dev"

    def test_valid_user_mode(self):
        d = DaemonSection(mode="user")
        assert d.mode == "user"

    def test_invalid_mode(self):
        with pytest.raises(ValidationError, match="mode must be"):
            DaemonSection(mode="production")


class TestUnknownKeys:
    """Validate unknown key detection."""

    def test_unknown_section_warning(self, caplog):
        """Unknown top-level sections should produce a warning."""
        config = {"typo_section": {"key": "value"}}
        with caplog.at_level(logging.WARNING, logger="config.schema"):
            validate_config(config)
        assert "typo_section" in caplog.text

    def test_extra_keys_in_section_ignored(self):
        """Unknown keys within a known section should be silently ignored."""
        model = DaemonSection(version="v4", unknown_future_key="value")
        assert model.version == "v4"

    def test_known_sections_no_warning(self, caplog):
        """A config with only known sections should not warn."""
        config = {"daemon": {"version": "v4"}, "memory": {}}
        with caplog.at_level(logging.WARNING, logger="config.schema"):
            validate_config(config)
        assert "Unknown sections" not in caplog.text


class TestSubModels:
    """Validate nested sub-models."""

    def test_collection_boosts_defaults(self):
        cb = CollectionBoosts()
        assert cb.facts == 0.15
        assert cb.wiki == 0.05

    def test_collection_boosts_range(self):
        with pytest.raises(ValidationError):
            CollectionBoosts(facts=1.5)

    def test_score_weights_from_dict(self):
        """Score weights should be constructible from a dict (as in YAML)."""
        sw = ScoreWeights(**{"relevance": 0.3, "recency": 0.2, "truth": 0.2,
                             "importance": 0.1, "continuity": 0.1, "structure": 0.1})
        assert sw.relevance == 0.3

    def test_gating_section_with_score_weights_dict(self):
        """GatingSection should accept score_weights as a raw dict."""
        g = GatingSection(score_weights={"relevance": 0.5, "recency": 0.5})
        assert g.score_weights.relevance == 0.5


class TestPartialSections:
    """Validate partial section configs (some fields present, others defaulted)."""

    def test_partial_memory_section(self):
        m = MemorySection(max_recent=100)
        assert m.max_recent == 100
        assert m.corpus_file == "./data/corpus_v4.json"  # Default

    def test_partial_features_section(self):
        f = FeaturesSection(enable_best_of=False)
        assert f.enable_best_of is False
        assert f.enable_query_rewrite is True  # Default

    def test_partial_root_config(self):
        """Root config with only daemon section should fill in all other defaults."""
        model = DaemonConfig(daemon={"version": "v5"})
        assert model.daemon.version == "v5"
        assert model.memory.max_recent == 50  # Default
        assert model.gating.cosine_similarity_threshold == 0.25  # Default


class TestWebSearchSection:
    """Specific web search validation."""

    def test_valid_config(self):
        from config.schema import WebSearchSection
        ws = WebSearchSection(enabled=True, timeout_s=30.0, daily_credit_limit=100)
        assert ws.timeout_s == 30.0

    def test_negative_credit_limit(self):
        from config.schema import WebSearchSection
        with pytest.raises(ValidationError):
            WebSearchSection(daily_credit_limit=0)


class TestVisualMemorySection:
    """Visual memory validation."""

    def test_defaults(self):
        from config.schema import VisualMemorySection
        vm = VisualMemorySection()
        assert vm.enabled is False
        assert vm.clip_model == "ViT-B-32"
        assert vm.similarity_threshold == 0.20

    def test_threshold_range(self):
        from config.schema import VisualMemorySection
        with pytest.raises(ValidationError):
            VisualMemorySection(similarity_threshold=1.5)
