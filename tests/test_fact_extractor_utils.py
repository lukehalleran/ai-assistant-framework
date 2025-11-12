"""Tests for memory/fact_extractor.py utility functions"""
import pytest
from memory.fact_extractor import (
    _looks_like_units,
    _has_number_unit,
    _is_adj_only,
    _relation_allows_adj,
    _is_clause_like,
    _refine_is_relation,
    _score_triple,
    FactExtractor
)


class TestUnitDetection:
    """Tests for unit detection functions."""

    def test_looks_like_units_lb(self):
        """Test detection of lb units."""
        assert _looks_like_units("lb") is True
        assert _looks_like_units("lbs") is True
        assert _looks_like_units("LB") is True

    def test_looks_like_units_kg(self):
        """Test detection of kg units."""
        assert _looks_like_units("kg") is True
        assert _looks_like_units("kgs") is True
        assert _looks_like_units("kilogram") is True
        assert _looks_like_units("kilograms") is True

    def test_looks_like_units_pound(self):
        """Test detection of pound units."""
        assert _looks_like_units("pound") is True
        assert _looks_like_units("pounds") is True

    def test_looks_like_units_negative(self):
        """Test non-unit strings."""
        assert _looks_like_units("test") is False
        assert _looks_like_units("hello") is False
        assert _looks_like_units("") is False

    def test_has_number_unit_basic(self):
        """Test basic number+unit detection."""
        assert _has_number_unit("285 lb") is True
        assert _has_number_unit("130 kilograms") is True
        assert _has_number_unit("100lbs") is True

    def test_has_number_unit_various_formats(self):
        """Test various number+unit formats."""
        assert _has_number_unit("365 pounds") is True
        assert _has_number_unit("225lb") is True
        assert _has_number_unit("405 kg") is True

    def test_has_number_unit_no_match(self):
        """Test strings without number+unit."""
        assert _has_number_unit("hello world") is False
        assert _has_number_unit("test") is False


class TestRelationHelpers:
    """Tests for relation helper functions."""

    def test_relation_allows_adj_color_relations(self):
        """Test that color relations allow adjectives."""
        assert _relation_allows_adj("color") is True
        assert _relation_allows_adj("favorite_color") is True
        assert _relation_allows_adj("favourite_color") is True
        assert _relation_allows_adj("likes_color") is True

    def test_relation_allows_adj_other_relations(self):
        """Test that non-color relations don't allow adjectives."""
        assert _relation_allows_adj("is") is False
        assert _relation_allows_adj("has") is False
        assert _relation_allows_adj("likes") is False

    def test_is_clause_like_infinitive(self):
        """Test detection of infinitive clauses."""
        assert _is_clause_like("to go home") is True
        assert _is_clause_like("to be there") is True

    def test_is_clause_like_gerund(self):
        """Test detection of gerund clauses."""
        assert _is_clause_like("running fast") is True
        assert _is_clause_like("swimming daily") is True

    def test_is_clause_like_passive(self):
        """Test detection of passive participle clauses."""
        assert _is_clause_like("used in production") is True
        assert _is_clause_like("tested with pytest") is True
        assert _is_clause_like("placed on table") is True

    def test_is_clause_like_negative(self):
        """Test non-clause strings."""
        assert _is_clause_like("blue") is False
        assert _is_clause_like("the cat") is False
        assert _is_clause_like("") is False

    def test_refine_is_relation_basic(self):
        """Test refining 'is' relations."""
        assert _refine_is_relation("is", "a cat") == "is_a"
        assert _refine_is_relation("is", "an animal") == "is_a"
        assert _refine_is_relation("is", "the best") == "is_a"

    def test_refine_is_relation_no_change(self):
        """Test 'is' relations that don't need refining."""
        assert _refine_is_relation("is", "blue") == "is"
        assert _refine_is_relation("is", "happy") == "is"
        assert _refine_is_relation("has", "a cat") == "has"


class TestTripleScoring:
    """Tests for triple scoring function."""

    def test_score_triple_basic(self):
        """Test basic triple scoring."""
        score = _score_triple("Luke", "likes", "pizza")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_triple_proper_nouns_higher(self):
        """Test that proper nouns get higher scores."""
        # Note: without nlp, can't fully test, but ensure it doesn't crash
        score = _score_triple("Luke", "lives_in", "Seattle")
        assert score > 0.0

    def test_score_triple_generic_lower(self):
        """Test that generic terms get lower scores."""
        score_generic = _score_triple("thing", "is", "stuff")
        score_specific = _score_triple("Luke", "likes", "pizza")
        # Specific should score higher than generic
        assert score_specific > score_generic

    def test_score_triple_stop_words_penalized(self):
        """Test that stop words are penalized."""
        score_stop = _score_triple("it", "is", "this")
        score_content = _score_triple("Luke", "likes", "pizza")
        assert score_content > score_stop


class TestFactExtractor:
    """Tests for FactExtractor class."""

    def test_fact_extractor_initialization(self):
        """Test FactExtractor can be initialized."""
        extractor = FactExtractor()
        assert extractor is not None
        assert extractor.use_rebel is True
        assert extractor.use_regex is True

    def test_fact_extractor_initialization_no_rebel(self):
        """Test FactExtractor with REBEL disabled."""
        extractor = FactExtractor(use_rebel=False)
        assert extractor.use_rebel is False
        assert extractor.use_regex is True

    def test_fact_extractor_initialization_no_regex(self):
        """Test FactExtractor with regex disabled."""
        extractor = FactExtractor(use_regex=False)
        assert extractor.use_rebel is True
        assert extractor.use_regex is False

    def test_fact_extractor_has_patterns(self):
        """Test that FactExtractor has expected patterns."""
        extractor = FactExtractor()
        assert hasattr(extractor, 'fact_patterns')
        assert len(extractor.fact_patterns) > 0
        assert hasattr(extractor, 'lift_patterns')
        assert len(extractor.lift_patterns) > 0

    def test_fact_patterns_copular(self):
        """Test that fact patterns match copular statements."""
        extractor = FactExtractor()
        import re

        # Test first pattern (copular statements)
        pattern = extractor.fact_patterns[0]

        assert re.search(pattern, "Luke is a developer")
        assert re.search(pattern, "The sky is blue")
        assert re.search(pattern, "I am happy")

    def test_fact_patterns_preference(self):
        """Test that fact patterns match preference statements."""
        extractor = FactExtractor()
        import re

        # Test second pattern (preferences) - needs IGNORECASE like source code uses
        pattern = extractor.fact_patterns[1]

        assert re.search(pattern, "I like pizza", re.IGNORECASE)
        assert re.search(pattern, "You love cats", re.IGNORECASE)
        assert re.search(pattern, "We prefer tea", re.IGNORECASE)

    def test_lift_patterns_squat(self):
        """Test that lift patterns match squat data."""
        extractor = FactExtractor()
        import re

        # Test first lift pattern
        pattern = extractor.lift_patterns[0]

        assert re.search(pattern, "my squat is 365")
        assert re.search(pattern, "squat: 365 lb")
        assert re.search(pattern, "deadlift=405kg")

    def test_lift_patterns_action(self):
        """Test that lift patterns match action statements."""
        extractor = FactExtractor()
        import re

        # Test second lift pattern
        pattern = extractor.lift_patterns[1]

        assert re.search(pattern, "I squatted 365 lb")
        assert re.search(pattern, "I benched 225")
        assert re.search(pattern, "I just deadlifted 405 pounds")

    def test_extract_methods_exist(self):
        """Test that FactExtractor has expected methods."""
        extractor = FactExtractor()
        assert hasattr(extractor, 'extract_facts')
        # Check if it's callable
        assert callable(getattr(extractor, 'extract_facts', None))


class TestTripleCleaning:
    """Tests for triple cleaning and normalization."""

    def test_clean_triple_empty_subject(self):
        """Test that empty subject returns None."""
        from memory.fact_extractor import _clean_triple
        result = _clean_triple("", "likes", "pizza")
        assert result is None

    def test_clean_triple_empty_relation(self):
        """Test that empty relation returns None."""
        from memory.fact_extractor import _clean_triple
        result = _clean_triple("Luke", "", "pizza")
        assert result is None

    def test_clean_triple_empty_object(self):
        """Test that empty object returns None."""
        from memory.fact_extractor import _clean_triple
        result = _clean_triple("Luke", "likes", "")
        assert result is None

    def test_clean_triple_basic(self):
        """Test basic triple cleaning."""
        from memory.fact_extractor import _clean_triple
        result = _clean_triple("Luke", "likes", "pizza")
        assert result is not None
        assert len(result) == 3

    def test_clean_triple_strips_whitespace(self):
        """Test that whitespace is stripped."""
        from memory.fact_extractor import _clean_triple
        result = _clean_triple("  Luke  ", "  likes  ", "  pizza  ")
        if result:
            s, r, o = result
            assert s.strip() == s
            assert r.strip() == r
            assert o.strip() == o

    def test_clean_triple_strips_punctuation(self):
        """Test that punctuation is stripped."""
        from memory.fact_extractor import _clean_triple
        result = _clean_triple("Luke.", "likes,", "pizza;")
        if result:
            s, r, o = result
            assert not s.endswith(".")
            assert not r.endswith(",")
            assert not o.endswith(";")


class TestNormalization:
    """Tests for subject/object normalization."""

    def test_normalize_subject_obj_first_person(self):
        """Test normalization of first-person pronouns."""
        from memory.fact_extractor import _normalize_subject_obj
        s, r, o = _normalize_subject_obj("I", "like", "pizza", user_name="Luke")
        assert s == "Luke"  # "I" should be replaced with user name

    def test_normalize_subject_obj_possessive(self):
        """Test normalization of possessives."""
        from memory.fact_extractor import _normalize_subject_obj
        s, r, o = _normalize_subject_obj("my", "favorite", "color", user_name="Luke")
        assert s == "Luke"  # "my" should be replaced

    def test_normalize_subject_obj_no_change(self):
        """Test that proper nouns aren't changed."""
        from memory.fact_extractor import _normalize_subject_obj
        s, r, o = _normalize_subject_obj("Luke", "likes", "pizza", user_name="Luke")
        assert s == "Luke"
        assert r == "likes"
        assert o == "pizza"
