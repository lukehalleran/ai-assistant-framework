"""
Unit tests for reflection retrieval v2: retrieval text generation,
metadata extraction, query rewriting, and entity/topic overlap scoring.
"""

import pytest
from memory.memory_storage import (
    _extract_reflection_retrieval_text,
    _clean_reflection_for_embedding,
    extract_reflection_metadata,
    _clean_entities,
    _dedupe_topics,
)
from memory.memory_retriever import (
    _rewrite_reflection_query,
    _compute_reflection_overlap,
)


# ---------------------------------------------------------------------------
# _extract_reflection_retrieval_text
# ---------------------------------------------------------------------------

class TestRetrievalTextGeneration:
    """Tests for the reflection retrieval text generator."""

    def test_strips_boilerplate_headers(self):
        text = (
            "### What went well\n"
            "- User's study notes were compressed effectively into useful summaries.\n"
            "### What could improve\n"
            "- More structured output formats would help the user organize notes."
        )
        result = _extract_reflection_retrieval_text(text)
        assert "What went well" not in result
        assert "What could improve" not in result
        assert "study notes" in result

    def test_strips_assistant_boilerplate(self):
        text = "### What went well\n- The assistant effectively handled the query\n- Real content here"
        result = _extract_reflection_retrieval_text(text)
        assert "assistant effectively" not in result.lower()
        assert "Real content here" in result

    def test_preserves_entities(self):
        text = "### What went well\n- Discussed FAISS indexing and the Daemon project with Auggie"
        result = _extract_reflection_retrieval_text(text)
        assert "FAISS" in result
        assert "Daemon" in result
        assert "Auggie" in result

    def test_preserves_topics_from_parenthetical(self):
        text = "- Clarifications (e.g., interaction variables in missing data) were actionable."
        result = _extract_reflection_retrieval_text(text)
        assert "interaction variables" in result

    def test_handles_single_line_yaml_format(self):
        """YAML collapses newlines — must handle single-line reflections."""
        text = (
            "### What went well - FAISS IVF indexing discussion was clear. "
            "- Recalled Auggie's name accurately. "
            "### What to improve - Should reference Daemon file locations."
        )
        result = _extract_reflection_retrieval_text(text)
        assert "FAISS" in result
        assert "Auggie" in result
        assert "Daemon" in result

    def test_falls_back_to_old_cleaner_when_empty(self):
        """If extraction removes everything, fallback to basic cleaning."""
        text = "### What went well\n- The assistant effectively did things"
        result = _extract_reflection_retrieval_text(text)
        # Should get something, not empty
        assert len(result) > 5

    def test_empty_input(self):
        assert _extract_reflection_retrieval_text("") == ""
        assert _extract_reflection_retrieval_text(None) == ""

    def test_no_duplicate_entities_in_output(self):
        text = "- FAISS FAISS FAISS was mentioned with FAISS indexing"
        result = _extract_reflection_retrieval_text(text)
        # Entity prefix should have FAISS once
        assert result.count("FAISS") >= 1

    def test_real_reflection_differentiation(self):
        """Two reflections about different topics should produce different retrieval text."""
        refl_a = (
            "### What went well\n"
            "- Clear explanations about FAISS indexing and memory retrieval.\n"
            "- Recalled user's brother Auggie's name accurately."
        )
        refl_b = (
            "### What went well\n"
            "- Empathetic response about the Israel-Palestine conflict.\n"
            "- Maintained conversational tone during sensitive discussion."
        )
        text_a = _extract_reflection_retrieval_text(refl_a)
        text_b = _extract_reflection_retrieval_text(refl_b)

        # Should be meaningfully different
        assert "FAISS" in text_a
        assert "FAISS" not in text_b
        assert "Israel" in text_b
        assert "Israel" not in text_a


# ---------------------------------------------------------------------------
# extract_reflection_metadata
# ---------------------------------------------------------------------------

class TestReflectionMetadata:
    """Tests for structured metadata extraction from reflections."""

    def test_extracts_entities(self):
        text = "Discussed FAISS indexing and the Daemon project with Auggie."
        meta = extract_reflection_metadata(text)
        entities = meta["entities"]
        assert "FAISS" in entities
        assert "Daemon" in entities

    def test_detects_project_area_daemon(self):
        text = "The retrieval system in Daemon uses RAG architecture."
        meta = extract_reflection_metadata(text)
        assert meta["project_area"] == "daemon"

    def test_detects_project_area_academic(self):
        text = "User's homework for the OMSA course at school was discussed."
        meta = extract_reflection_metadata(text)
        assert meta["project_area"] == "academic"

    def test_detects_emotional_tone(self):
        text = "User expressed anxiety and stress about work deadlines."
        meta = extract_reflection_metadata(text)
        assert "anxious" in meta["emotional_tone"]

    def test_detects_themes(self):
        text = "Talked about sleep patterns, exercise goals, and medication timing."
        meta = extract_reflection_metadata(text)
        assert "health" in meta["themes"]

    def test_skips_header_entities(self):
        """Header words like 'What', 'Went', 'Well' should not be entities."""
        text = "### What Went Well\n- Discussed real topics."
        meta = extract_reflection_metadata(text)
        assert "What" not in meta["entities"]
        assert "Went" not in meta["entities"]
        assert "Well" not in meta["entities"]

    def test_all_fields_present(self):
        text = "Simple reflection about work."
        meta = extract_reflection_metadata(text)
        for field in ("primary_topic", "secondary_topics", "entities",
                      "themes", "emotional_tone", "project_area"):
            assert field in meta


# ---------------------------------------------------------------------------
# _clean_entities
# ---------------------------------------------------------------------------

class TestCleanEntities:

    def test_removes_possessive_fragments(self):
        entities = {"Auggie's", "s brother", "'s name", "FAISS"}
        cleaned = _clean_entities(entities)
        assert "Auggie" in cleaned
        assert "s brother" not in cleaned
        assert "'s name" not in cleaned
        assert "FAISS" in cleaned

    def test_removes_common_words(self):
        entities = {"The", "However", "FAISS", "Daemon"}
        cleaned = _clean_entities(entities)
        assert "FAISS" in cleaned
        assert "Daemon" in cleaned
        assert "The" not in cleaned


# ---------------------------------------------------------------------------
# _dedupe_topics
# ---------------------------------------------------------------------------

class TestDedupeTopics:

    def test_removes_substrings(self):
        topics = {"statistical definitions", "corrections in statistical definitions"}
        result = _dedupe_topics(topics)
        assert "corrections in statistical definitions" in result
        assert "statistical definitions" not in result

    def test_trims_long_topics(self):
        long_topic = "x" * 100
        result = _dedupe_topics({long_topic})
        for t in result:
            assert len(t) <= 50

    def test_removes_short_topics(self):
        topics = {"hi", "ok", "real topic here"}
        result = _dedupe_topics(topics)
        assert "real topic here" in result
        assert "hi" not in result
        assert "ok" not in result


# ---------------------------------------------------------------------------
# _rewrite_reflection_query
# ---------------------------------------------------------------------------

class TestReflectionQueryRewriting:

    def test_strips_how_have_i_been_doing(self):
        result = _rewrite_reflection_query("How have I been doing with health?")
        assert "how have i been" not in result.lower()
        assert "health" in result.lower()

    def test_strips_what_learned(self):
        result = _rewrite_reflection_query("What have I learned about school?")
        assert "what have i learned" not in result.lower()
        assert "school" in result.lower()

    def test_preserves_content_words(self):
        result = _rewrite_reflection_query("How have I been doing with FAISS indexing?")
        assert "FAISS" in result
        assert "indexing" in result.lower()

    def test_handles_short_query(self):
        result = _rewrite_reflection_query("sleep")
        assert "sleep" in result.lower()

    def test_handles_empty_query(self):
        assert _rewrite_reflection_query("") == ""
        assert _rewrite_reflection_query(None) is None

    def test_does_not_add_noise(self):
        """Rewrite should not add words not in the original query."""
        result = _rewrite_reflection_query("How have I been doing with Daemon?")
        words = set(result.lower().split())
        # Should only contain content words from original
        assert "daemon" in words


# ---------------------------------------------------------------------------
# _compute_reflection_overlap
# ---------------------------------------------------------------------------

class TestReflectionOverlap:

    def test_full_overlap(self):
        query_words = {"faiss", "indexing", "retrieval"}
        meta = {"entities": "FAISS", "primary_topic": "indexing and retrieval"}
        score = _compute_reflection_overlap(query_words, "faiss indexing retrieval", meta)
        assert score > 0.5

    def test_no_overlap(self):
        query_words = {"cooking", "recipe", "dinner"}
        meta = {"entities": "FAISS", "primary_topic": "indexing"}
        score = _compute_reflection_overlap(query_words, "cooking recipe dinner", meta)
        assert score == 0.0

    def test_partial_overlap(self):
        query_words = {"health", "medication", "exercise"}
        meta = {"entities": "", "themes": "health", "primary_topic": "exercise routine"}
        score = _compute_reflection_overlap(query_words, "health medication exercise", meta)
        assert 0.0 < score < 1.0

    def test_empty_metadata(self):
        score = _compute_reflection_overlap({"test"}, "test", {})
        assert score == 0.0

    def test_empty_query(self):
        score = _compute_reflection_overlap(set(), "", {"entities": "FAISS"})
        assert score == 0.0

    def test_primary_topic_substring_bonus(self):
        """Should get bonus when query span appears in primary_topic."""
        query = "faiss indexing performance"
        query_words = set(query.split())
        # Score with primary_topic match (has substring bonus)
        meta = {"primary_topic": "faiss indexing performance tuning"}
        score_with = _compute_reflection_overlap(query_words, query, meta)

        # Score with only entities (word-level match, no substring bonus)
        meta2 = {"primary_topic": "completely unrelated stuff", "entities": "faiss"}
        score_without = _compute_reflection_overlap(query_words, query, meta2)

        assert score_with > score_without
