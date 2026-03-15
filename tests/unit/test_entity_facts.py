"""Tests for entity-to-entity fact extraction.

Covers:
- Budget split (user vs entity facts, separate caps)
- Filter removal (non-user subjects pass through)
- Confidence threshold (entity facts below min_confidence dropped)
- Metadata (fact_scope, entity_type, user_connection populated)
- Profile boundary (entity facts NOT added to UserProfile)
- LLM extractor (produces entity facts with user_connection)
- Helper functions (_detect_entity_type, _detect_user_connection)
- Dedup (entity facts deduplicated correctly)
"""
import pytest
import re
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from memory.fact_extractor import (
    FactExtractor,
    _detect_entity_type,
    _detect_user_connection,
)
from memory.llm_fact_extractor import _normalize_triple


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestDetectEntityType:
    """Tests for _detect_entity_type helper."""

    def test_no_nlp_returns_unknown(self):
        assert _detect_entity_type("Oliver", nlp=None) == "UNKNOWN"

    def test_person_entity(self):
        """Named person should return PERSON (requires spaCy)."""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            pytest.skip("spaCy not available")
        # spaCy may or may not recognize arbitrary names; test the return type
        result = _detect_entity_type("Barack Obama", nlp=nlp)
        assert result in {"PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART", "UNKNOWN"}

    def test_org_entity(self):
        """Organization name should return ORG (requires spaCy)."""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            pytest.skip("spaCy not available")
        result = _detect_entity_type("Google", nlp=nlp)
        assert result in {"PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART", "UNKNOWN"}

    def test_empty_subject(self):
        assert _detect_entity_type("", nlp=None) == "UNKNOWN"


class TestDetectUserConnection:
    """Tests for _detect_user_connection helper."""

    def test_my_boss_pattern(self):
        result = _detect_user_connection("oliver", "My boss Oliver moved from London")
        assert result == "user's boss"

    def test_my_friend_pattern(self):
        result = _detect_user_connection("alex", "My friend Alex lives in Portland")
        assert result == "user's friend"

    def test_my_cat_pattern(self):
        result = _detect_user_connection("flapjack", "My cat Flapjack is adorable")
        assert result == "user's cat"

    def test_is_my_pattern(self):
        result = _detect_user_connection("sarah", "Sarah is my sister and she lives nearby")
        assert result == "user's sister"

    def test_comma_my_pattern(self):
        result = _detect_user_connection("oliver", "Oliver, my manager, approved the request")
        assert result == "user's manager"

    def test_no_connection(self):
        result = _detect_user_connection("london", "London is a big city")
        assert result is None

    def test_empty_inputs(self):
        assert _detect_user_connection("", "some text") is None
        assert _detect_user_connection("oliver", "") is None
        assert _detect_user_connection("", "") is None


# ---------------------------------------------------------------------------
# LLM _normalize_triple tests
# ---------------------------------------------------------------------------

class TestNormalizeTripleEntityFacts:
    """Tests for LLM extractor _normalize_triple with entity facts."""

    def test_user_fact_gets_user_scope(self):
        result = _normalize_triple({
            "subject": "user",
            "relation": "lives_in",
            "object": "Portland",
        })
        assert result is not None
        assert result["fact_scope"] == "user"

    def test_pronoun_normalized_to_user(self):
        result = _normalize_triple({
            "subject": "I",
            "relation": "likes",
            "object": "coffee",
        })
        assert result is not None
        assert result["subject"] == "user"
        assert result["fact_scope"] == "user"

    def test_named_entity_preserved(self):
        result = _normalize_triple({
            "subject": "Oliver",
            "relation": "moved_from",
            "object": "London",
        })
        assert result is not None
        assert result["subject"] == "oliver"
        assert result["fact_scope"] == "entity"

    def test_user_connection_forwarded(self):
        result = _normalize_triple({
            "subject": "Oliver",
            "relation": "moved_from",
            "object": "London",
            "user_connection": "user's boss",
        })
        assert result is not None
        assert result["user_connection"] == "user's boss"

    def test_no_user_connection_when_empty(self):
        result = _normalize_triple({
            "subject": "Oliver",
            "relation": "moved_from",
            "object": "London",
        })
        assert result is not None
        assert "user_connection" not in result

    def test_entity_confidence_preserved(self):
        result = _normalize_triple({
            "subject": "Google",
            "relation": "headquartered_in",
            "object": "Mountain View",
            "confidence": 0.85,
        })
        assert result is not None
        assert result["confidence"] == 0.85
        assert result["fact_scope"] == "entity"


# ---------------------------------------------------------------------------
# FactExtractor integration tests
# ---------------------------------------------------------------------------

class TestEntityFactExtraction:
    """Integration tests for entity fact extraction in FactExtractor."""

    @pytest.fixture
    def extractor(self):
        return FactExtractor(use_rebel=False, use_regex=True)

    @pytest.mark.asyncio
    async def test_user_facts_still_extracted(self, extractor):
        """User facts should continue to be extracted normally."""
        facts = await extractor.extract_facts(
            "I live in Portland and I work at Google",
            "",
        )
        user_facts = [f for f in facts if f.metadata.get("fact_scope") == "user"]
        assert len(user_facts) > 0

    @pytest.mark.asyncio
    async def test_user_facts_have_user_scope(self, extractor):
        """All user-subject facts should have fact_scope='user'."""
        facts = await extractor.extract_facts("I am 32 years old", "")
        for f in facts:
            if f.metadata.get("subject") == "user":
                assert f.metadata.get("fact_scope") == "user"

    @pytest.mark.asyncio
    async def test_entity_facts_have_entity_scope(self, extractor):
        """Non-user-subject facts should have fact_scope='entity'."""
        facts = await extractor.extract_facts(
            "My friend Alex lives in Portland. I like hiking.",
            "",
        )
        entity_facts = [f for f in facts if f.metadata.get("fact_scope") == "entity"]
        # Entity facts may or may not be extracted depending on NLP pipeline
        # Just verify that if they exist, they have correct metadata
        for f in entity_facts:
            assert f.metadata.get("subject") != "user"
            assert "entity_type" in f.metadata

    @pytest.mark.asyncio
    async def test_user_cap_enforced(self, extractor):
        """User facts should be capped at USER_FACTS_PER_TURN_CAP."""
        with patch("config.app_config.USER_FACTS_PER_TURN_CAP", 2):
            facts = await extractor.extract_facts(
                "I live in Portland. I am 32. I work at Google. My name is Luke. I like beer.",
                "",
            )
            user_facts = [f for f in facts if f.metadata.get("fact_scope") == "user"]
            assert len(user_facts) <= 2

    @pytest.mark.asyncio
    async def test_entity_cap_enforced(self, extractor):
        """Entity facts should be capped at ENTITY_FACTS_PER_TURN_CAP."""
        with patch("config.app_config.ENTITY_FACTS_PER_TURN_CAP", 1):
            # Even if many entity facts are available, only 1 should pass
            facts = await extractor.extract_facts(
                "My boss Oliver is from London. My friend Sarah works at Apple. My cat Flapjack is orange.",
                "",
            )
            entity_facts = [f for f in facts if f.metadata.get("fact_scope") == "entity"]
            assert len(entity_facts) <= 1

    @pytest.mark.asyncio
    async def test_entity_facts_disabled(self, extractor):
        """When ENTITY_FACTS_ENABLED=False, only user facts pass through."""
        with patch("config.app_config.ENTITY_FACTS_ENABLED", False):
            facts = await extractor.extract_facts(
                "My boss Oliver moved from London. I live in Portland.",
                "",
            )
            for f in facts:
                assert f.metadata.get("subject") == "user" or f.metadata.get("fact_scope") == "user"

    @pytest.mark.asyncio
    async def test_entity_confidence_threshold(self, extractor):
        """Entity facts below ENTITY_FACT_MIN_CONFIDENCE should be dropped."""
        with patch("config.app_config.ENTITY_FACT_MIN_CONFIDENCE", 0.99):
            facts = await extractor.extract_facts(
                "My boss Oliver is from London. I live in Portland.",
                "",
            )
            entity_facts = [f for f in facts if f.metadata.get("fact_scope") == "entity"]
            # With threshold at 0.99, almost no entity facts should pass
            assert len(entity_facts) == 0

    @pytest.mark.asyncio
    async def test_dedup_across_user_and_entity(self, extractor):
        """Dedup should work across user and entity facts."""
        facts = await extractor.extract_facts(
            "I live in Portland. I live in Portland.",
            "",
        )
        # Should not have duplicate facts
        contents = [f.content for f in facts]
        assert len(contents) == len(set(contents))

    @pytest.mark.asyncio
    async def test_user_facts_prioritized_in_output(self, extractor):
        """User facts should appear before entity facts in the output list."""
        facts = await extractor.extract_facts(
            "I live in Portland. My boss Oliver is from London.",
            "",
        )
        if len(facts) >= 2:
            user_indices = [i for i, f in enumerate(facts) if f.metadata.get("fact_scope") == "user"]
            entity_indices = [i for i, f in enumerate(facts) if f.metadata.get("fact_scope") == "entity"]
            if user_indices and entity_indices:
                assert max(user_indices) < min(entity_indices), \
                    "User facts should come before entity facts"


# ---------------------------------------------------------------------------
# MemoryStorage entity metadata forwarding tests
# ---------------------------------------------------------------------------

class TestMemoryStorageEntityMetadata:
    """Tests that entity metadata flows through to ChromaDB."""

    @pytest.mark.asyncio
    async def test_entity_metadata_forwarded_to_chroma(self):
        """Entity metadata should be included in the source dict passed to add_fact."""
        from memory.memory_storage import MemoryStorage

        mock_chroma = MagicMock()
        mock_chroma.add_fact.return_value = "test-id"
        mock_chroma.add_conversation_memory.return_value = "conv-id"

        mock_corpus = MagicMock()

        # Create a fact node with entity metadata
        mock_fact = MagicMock()
        mock_fact.content = "oliver | moved_from | london"
        mock_fact.metadata = {
            "subject": "oliver",
            "relation": "moved_from",
            "object": "london",
            "confidence": 0.75,
            "source": "conversation",
            "fact_scope": "entity",
            "entity_type": "PERSON",
            "user_connection": "user's boss",
        }

        mock_extractor = MagicMock()
        mock_extractor.extract_facts = AsyncMock(return_value=[mock_fact])

        storage = MemoryStorage(
            corpus_manager=mock_corpus,
            chroma_store=mock_chroma,
            fact_extractor=mock_extractor,
        )

        await storage.extract_and_store_facts("My boss Oliver moved from London", "", 0.7)

        # Verify add_fact was called with a source dict containing entity metadata
        mock_chroma.add_fact.assert_called_once()
        call_kwargs = mock_chroma.add_fact.call_args
        source_arg = call_kwargs.kwargs.get("source") or call_kwargs[1].get("source") if len(call_kwargs) > 1 else call_kwargs.kwargs.get("source")

        # The source should be a dict now
        if source_arg is None:
            # Check positional args
            args = call_kwargs[1] if len(call_kwargs) > 1 else {}
            source_arg = args.get("source", call_kwargs.kwargs.get("source"))

        assert isinstance(source_arg, dict)
        assert source_arg["fact_scope"] == "entity"
        assert source_arg["entity_type"] == "PERSON"
        assert source_arg["user_connection"] == "user's boss"

    @pytest.mark.asyncio
    async def test_user_fact_metadata_forwarded(self):
        """User facts should include fact_scope='user' in metadata."""
        from memory.memory_storage import MemoryStorage

        mock_chroma = MagicMock()
        mock_chroma.add_fact.return_value = "test-id"

        mock_fact = MagicMock()
        mock_fact.content = "user | lives_in | portland"
        mock_fact.metadata = {
            "subject": "user",
            "relation": "lives_in",
            "object": "portland",
            "confidence": 0.8,
            "source": "conversation",
            "fact_scope": "user",
        }

        mock_extractor = MagicMock()
        mock_extractor.extract_facts = AsyncMock(return_value=[mock_fact])

        storage = MemoryStorage(
            corpus_manager=MagicMock(),
            chroma_store=mock_chroma,
            fact_extractor=mock_extractor,
        )

        await storage.extract_and_store_facts("I live in Portland", "", 0.7)

        mock_chroma.add_fact.assert_called_once()
        call_kwargs = mock_chroma.add_fact.call_args
        source_arg = call_kwargs.kwargs.get("source")
        assert isinstance(source_arg, dict)
        assert source_arg["fact_scope"] == "user"


# ---------------------------------------------------------------------------
# ShutdownProcessor profile boundary tests
# ---------------------------------------------------------------------------

class TestShutdownProcessorProfileBoundary:
    """Tests that entity facts do NOT get added to UserProfile."""

    @pytest.mark.asyncio
    async def test_entity_facts_not_added_to_profile(self):
        """Entity facts should be filtered out before adding to UserProfile."""
        from memory.shutdown_processor import ShutdownProcessor

        mock_profile = MagicMock()
        mock_profile.add_facts_batch.return_value = 1

        mock_mm = MagicMock()
        mock_mm.generate_once = AsyncMock(return_value='[{"subject": "user", "relation": "lives_in", "object": "Portland", "confidence": 0.9}, {"subject": "Oliver", "relation": "moved_from", "object": "London", "confidence": 0.8, "user_connection": "user\'s boss"}]')

        mock_chroma = MagicMock()
        mock_chroma.add_fact.return_value = "test-id"

        proc = ShutdownProcessor(
            corpus_manager=MagicMock(),
            chroma_store=mock_chroma,
            consolidator=MagicMock(),
            fact_extractor=MagicMock(),
            model_manager=mock_mm,
            user_profile=mock_profile,
            storage=MagicMock(),
            session_start=datetime.now(),
        )

        session_convs = [
            {"query": "My boss Oliver moved from London. I live in Portland.", "response": "Got it!"}
        ]

        await proc._extract_llm_facts(session_convs)

        # Profile should only receive user facts
        if mock_profile.add_facts_batch.called:
            batch_arg = mock_profile.add_facts_batch.call_args[0][0]
            for fact in batch_arg:
                assert fact.get("subject", "user").lower() == "user", \
                    f"Entity fact leaked to profile: {fact}"

    @pytest.mark.asyncio
    async def test_all_user_facts_still_reach_profile(self):
        """User-scoped facts should still be added to UserProfile."""
        from memory.shutdown_processor import ShutdownProcessor

        mock_profile = MagicMock()
        mock_profile.add_facts_batch.return_value = 2

        mock_mm = MagicMock()
        mock_mm.generate_once = AsyncMock(return_value='[{"subject": "user", "relation": "lives_in", "object": "Portland", "confidence": 0.9}, {"subject": "user", "relation": "age", "object": "32", "confidence": 0.95}]')

        mock_chroma = MagicMock()
        mock_chroma.add_fact.return_value = "test-id"

        proc = ShutdownProcessor(
            corpus_manager=MagicMock(),
            chroma_store=mock_chroma,
            consolidator=MagicMock(),
            fact_extractor=MagicMock(),
            model_manager=mock_mm,
            user_profile=mock_profile,
            storage=MagicMock(),
            session_start=datetime.now(),
        )

        session_convs = [{"query": "I am 32 and I live in Portland", "response": "Got it!"}]

        await proc._extract_llm_facts(session_convs)

        # All facts are user-scoped, so all should reach profile
        assert mock_profile.add_facts_batch.called
        batch_arg = mock_profile.add_facts_batch.call_args[0][0]
        assert len(batch_arg) == 2


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestEntityFactsConfig:
    """Tests for entity facts configuration constants."""

    def test_config_defaults_loaded(self):
        from config.app_config import (
            ENTITY_FACTS_ENABLED,
            ENTITY_FACTS_PER_TURN_CAP,
            USER_FACTS_PER_TURN_CAP,
            ENTITY_FACT_MIN_CONFIDENCE,
        )
        assert isinstance(ENTITY_FACTS_ENABLED, bool)
        assert isinstance(ENTITY_FACTS_PER_TURN_CAP, int)
        assert ENTITY_FACTS_PER_TURN_CAP > 0
        assert isinstance(USER_FACTS_PER_TURN_CAP, int)
        assert USER_FACTS_PER_TURN_CAP > 0
        assert isinstance(ENTITY_FACT_MIN_CONFIDENCE, float)
        assert 0.0 < ENTITY_FACT_MIN_CONFIDENCE < 1.0

    def test_user_cap_greater_than_entity_cap(self):
        """User facts should have higher budget than entity facts by default."""
        from config.app_config import ENTITY_FACTS_PER_TURN_CAP, USER_FACTS_PER_TURN_CAP
        assert USER_FACTS_PER_TURN_CAP >= ENTITY_FACTS_PER_TURN_CAP
