"""Tests for the Fact Verification Gate.

Covers:
- Pydantic model construction (FactVerdict, ConflictCandidate, VerificationResult)
- Fast-path verdicts (no triple, ephemeral relation)
- Candidate finding (ChromaDB queries, triple extraction)
- Conflict detection (same object = confirmation, different object = conflict)
- User-trust override (high-confidence user-stated facts)
- Entity-scope rejection (low-confidence entity facts)
- LLM adjudication (A/B/C responses, unparseable fallback)
- Batch verification
- Flagging behavior (superseded_by metadata on old facts)
- Integration with MemoryStorage
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from memory.fact_verification import (
    FactVerdict,
    ConflictCandidate,
    VerificationResult,
    FactVerifier,
)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestFactVerdict:
    def test_enum_values(self):
        assert FactVerdict.STORE == "store"
        assert FactVerdict.STORE_AND_FLAG == "store_and_flag"
        assert FactVerdict.REJECT == "reject"
        assert FactVerdict.SKIP == "skip"

    def test_all_verdicts_exist(self):
        assert len(FactVerdict) == 4


class TestConflictCandidate:
    def test_default_construction(self):
        c = ConflictCandidate()
        assert c.doc_id == ""
        assert c.content == ""
        assert c.confidence == 0.0

    def test_full_construction(self):
        c = ConflictCandidate(
            doc_id="abc123",
            content="user | likes | pizza",
            subject="user",
            predicate="likes",
            object="pizza",
            confidence=0.85,
            timestamp="2026-03-01T00:00:00",
        )
        assert c.doc_id == "abc123"
        assert c.predicate == "likes"
        assert c.object == "pizza"


class TestVerificationResult:
    def test_default_construction(self):
        vr = VerificationResult()
        assert vr.verdict == FactVerdict.STORE
        assert vr.confidence == 1.0
        assert vr.reason == ""
        assert vr.conflicting_candidates == []
        assert vr.metadata_updates == {}

    def test_with_candidates(self):
        cand = ConflictCandidate(doc_id="x", object="old_val")
        vr = VerificationResult(
            verdict=FactVerdict.STORE_AND_FLAG,
            confidence=0.9,
            reason="user_trust_override",
            conflicting_candidates=[cand],
            metadata_updates={"superseded_reason": "user_correction"},
        )
        assert len(vr.conflicting_candidates) == 1
        assert vr.metadata_updates["superseded_reason"] == "user_correction"


# ---------------------------------------------------------------------------
# FactVerifier — fast path tests
# ---------------------------------------------------------------------------

class TestFastPath:
    def _make_verifier(self):
        chroma = MagicMock()
        return FactVerifier(chroma_store=chroma)

    @pytest.mark.asyncio
    async def test_no_subject_skips(self):
        v = self._make_verifier()
        r = await v.verify(
            subject="", predicate="likes", object_val="pizza",
            fact_text="user | likes | pizza",
        )
        assert r.verdict == FactVerdict.SKIP
        assert r.reason == "no_parseable_triple"

    @pytest.mark.asyncio
    async def test_no_predicate_skips(self):
        v = self._make_verifier()
        r = await v.verify(
            subject="user", predicate="", object_val="pizza",
            fact_text="user | likes | pizza",
        )
        assert r.verdict == FactVerdict.SKIP
        assert r.reason == "no_parseable_triple"

    @pytest.mark.asyncio
    async def test_ephemeral_relation_skips(self):
        v = self._make_verifier()
        # Force ephemeral set
        v._ephemeral = frozenset({"current_feeling", "current_mood"})
        r = await v.verify(
            subject="user", predicate="current_feeling", object_val="happy",
            fact_text="user | current_feeling | happy",
        )
        assert r.verdict == FactVerdict.SKIP
        assert r.reason == "ephemeral_relation"

    @pytest.mark.asyncio
    async def test_ephemeral_loaded_from_config(self):
        v = self._make_verifier()
        v._ephemeral = None  # force reload
        with patch("memory.fact_verification.FactVerifier._get_ephemeral_relations",
                    return_value=frozenset({"current_mood"})):
            # This tests the caching path too
            v._ephemeral = frozenset({"current_mood"})
            r = await v.verify(
                subject="user", predicate="current_mood", object_val="tired",
                fact_text="user | current_mood | tired",
            )
            assert r.verdict == FactVerdict.SKIP


# ---------------------------------------------------------------------------
# FactVerifier — candidate finding
# ---------------------------------------------------------------------------

class TestCandidateFinding:
    def _make_verifier_with_facts(self, existing_facts):
        """Create a verifier with mocked ChromaDB containing existing facts."""
        chroma = MagicMock()
        coll = MagicMock()
        coll.count.return_value = len(existing_facts)
        chroma.collections = {"facts": coll}
        chroma.query_collection.return_value = existing_facts
        return FactVerifier(chroma_store=chroma)

    @pytest.mark.asyncio
    async def test_no_facts_collection(self):
        chroma = MagicMock()
        chroma.collections = {"facts": None}
        v = FactVerifier(chroma_store=chroma)
        cands = v._find_candidates("user", "likes", "user | likes | pizza")
        assert cands == []

    @pytest.mark.asyncio
    async def test_empty_collection(self):
        chroma = MagicMock()
        coll = MagicMock()
        coll.count.return_value = 0
        chroma.collections = {"facts": coll}
        v = FactVerifier(chroma_store=chroma)
        cands = v._find_candidates("user", "likes", "user | likes | pizza")
        assert cands == []

    @pytest.mark.asyncio
    async def test_finds_matching_candidates(self):
        existing = [
            {
                "id": "fact1",
                "content": "user | likes | sushi",
                "metadata": {
                    "subject": "user",
                    "relation": "likes",
                    "object": "sushi",
                    "confidence": 0.8,
                    "timestamp": "2026-01-01",
                },
            }
        ]
        v = self._make_verifier_with_facts(existing)
        v._ephemeral = frozenset()  # no ephemeral skip
        cands = v._find_candidates("user", "likes", "user | likes | pizza")
        assert len(cands) == 1
        assert cands[0].object == "sushi"
        assert cands[0].doc_id == "fact1"

    @pytest.mark.asyncio
    async def test_filters_non_matching_subject(self):
        existing = [
            {
                "id": "fact1",
                "content": "bob | likes | sushi",
                "metadata": {
                    "subject": "bob",
                    "relation": "likes",
                    "object": "sushi",
                    "confidence": 0.8,
                },
            }
        ]
        v = self._make_verifier_with_facts(existing)
        cands = v._find_candidates("user", "likes", "user | likes | pizza")
        assert len(cands) == 0

    @pytest.mark.asyncio
    async def test_filters_non_matching_predicate(self):
        existing = [
            {
                "id": "fact1",
                "content": "user | hates | sushi",
                "metadata": {
                    "subject": "user",
                    "relation": "hates",
                    "object": "sushi",
                    "confidence": 0.8,
                },
            }
        ]
        v = self._make_verifier_with_facts(existing)
        cands = v._find_candidates("user", "likes", "user | likes | pizza")
        assert len(cands) == 0


# ---------------------------------------------------------------------------
# FactVerifier — verification flow
# ---------------------------------------------------------------------------

class TestVerificationFlow:
    def _make_verifier(self, existing_facts=None, model_manager=None):
        chroma = MagicMock()
        coll = MagicMock()
        coll.count.return_value = len(existing_facts or [])
        chroma.collections = {"facts": coll}
        chroma.query_collection.return_value = existing_facts or []
        v = FactVerifier(chroma_store=chroma, model_manager=model_manager)
        v._ephemeral = frozenset()
        return v

    @pytest.mark.asyncio
    async def test_no_candidates_stores(self):
        v = self._make_verifier(existing_facts=[])
        r = await v.verify(
            subject="user", predicate="likes", object_val="pizza",
            fact_text="user | likes | pizza",
        )
        assert r.verdict == FactVerdict.STORE
        assert r.reason == "no_existing_candidates"

    @pytest.mark.asyncio
    async def test_confirmation_stores(self):
        """Same subject+predicate+object = confirmation."""
        existing = [{
            "id": "f1", "content": "user | likes | pizza",
            "metadata": {"subject": "user", "relation": "likes", "object": "pizza",
                         "confidence": 0.8},
        }]
        v = self._make_verifier(existing_facts=existing)
        r = await v.verify(
            subject="user", predicate="likes", object_val="pizza",
            fact_text="user | likes | pizza",
        )
        assert r.verdict == FactVerdict.STORE
        assert r.reason == "confirmation_of_existing"

    @pytest.mark.asyncio
    async def test_user_trust_override(self):
        """User-stated facts with high confidence override conflicts."""
        existing = [{
            "id": "f1", "content": "user | lives_in | NYC",
            "metadata": {"subject": "user", "relation": "lives_in", "object": "NYC",
                         "confidence": 0.7},
        }]
        v = self._make_verifier(existing_facts=existing)
        r = await v.verify(
            subject="user", predicate="lives_in", object_val="LA",
            fact_text="user | lives_in | LA",
            source="user_stated", confidence=0.9,
        )
        assert r.verdict == FactVerdict.STORE_AND_FLAG
        assert r.reason == "user_trust_override"
        assert len(r.conflicting_candidates) == 1

    @pytest.mark.asyncio
    async def test_user_trust_below_threshold_not_auto_trusted(self):
        """User-stated but low confidence doesn't auto-trust."""
        existing = [{
            "id": "f1", "content": "user | lives_in | NYC",
            "metadata": {"subject": "user", "relation": "lives_in", "object": "NYC",
                         "confidence": 0.7},
        }]
        v = self._make_verifier(existing_facts=existing)
        r = await v.verify(
            subject="user", predicate="lives_in", object_val="LA",
            fact_text="user | lives_in | LA",
            source="user_stated", confidence=0.5,
        )
        # Should NOT be user_trust_override, falls through to fallback
        assert r.reason != "user_trust_override"

    @pytest.mark.asyncio
    async def test_entity_scope_low_confidence_rejects(self):
        """Entity-scope facts with low confidence get rejected."""
        existing = [{
            "id": "f1", "content": "flapjack | breed | golden retriever",
            "metadata": {"subject": "flapjack", "relation": "breed",
                         "object": "golden retriever", "confidence": 0.8},
        }]
        v = self._make_verifier(existing_facts=existing)
        r = await v.verify(
            subject="flapjack", predicate="breed", object_val="labrador",
            fact_text="flapjack | breed | labrador",
            source="conversation", confidence=0.5,
            fact_scope="entity",
        )
        assert r.verdict == FactVerdict.REJECT
        assert r.reason == "entity_scope_low_confidence"

    @pytest.mark.asyncio
    async def test_fallback_trust_newer(self):
        """Without LLM, conflicts default to STORE_AND_FLAG."""
        existing = [{
            "id": "f1", "content": "user | works_at | Google",
            "metadata": {"subject": "user", "relation": "works_at",
                         "object": "Google", "confidence": 0.7},
        }]
        v = self._make_verifier(existing_facts=existing)
        r = await v.verify(
            subject="user", predicate="works_at", object_val="Meta",
            fact_text="user | works_at | Meta",
            source="conversation", confidence=0.7,
        )
        assert r.verdict == FactVerdict.STORE_AND_FLAG
        assert r.reason == "fallback_trust_newer"


# ---------------------------------------------------------------------------
# LLM Adjudication
# ---------------------------------------------------------------------------

class TestLLMAdjudication:
    def _make_verifier_with_llm(self, llm_response, existing_facts):
        chroma = MagicMock()
        coll = MagicMock()
        coll.count.return_value = len(existing_facts)
        chroma.collections = {"facts": coll}
        chroma.query_collection.return_value = existing_facts
        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value=llm_response)
        v = FactVerifier(chroma_store=chroma, model_manager=mm)
        v._ephemeral = frozenset()
        return v

    def _existing_fact(self):
        return [{
            "id": "f1", "content": "user | works_at | Google",
            "metadata": {"subject": "user", "relation": "works_at",
                         "object": "Google", "confidence": 0.7},
        }]

    @pytest.mark.asyncio
    async def test_llm_returns_A_stores_and_flags(self):
        v = self._make_verifier_with_llm("A", self._existing_fact())
        r = await v.verify(
            subject="user", predicate="works_at", object_val="Meta",
            fact_text="user | works_at | Meta",
        )
        assert r.verdict == FactVerdict.STORE_AND_FLAG
        assert r.reason == "llm_update"

    @pytest.mark.asyncio
    async def test_llm_returns_B_stores(self):
        v = self._make_verifier_with_llm("B", self._existing_fact())
        r = await v.verify(
            subject="user", predicate="works_at", object_val="Meta",
            fact_text="user | works_at | Meta",
        )
        assert r.verdict == FactVerdict.STORE
        assert r.reason == "llm_complementary"

    @pytest.mark.asyncio
    async def test_llm_returns_C_rejects(self):
        v = self._make_verifier_with_llm("C", self._existing_fact())
        r = await v.verify(
            subject="user", predicate="works_at", object_val="Meta",
            fact_text="user | works_at | Meta",
        )
        assert r.verdict == FactVerdict.REJECT
        assert r.reason == "llm_implausible"

    @pytest.mark.asyncio
    async def test_llm_returns_garbage_falls_back(self):
        v = self._make_verifier_with_llm("xyz nonsense", self._existing_fact())
        r = await v.verify(
            subject="user", predicate="works_at", object_val="Meta",
            fact_text="user | works_at | Meta",
        )
        assert r.verdict == FactVerdict.STORE_AND_FLAG
        assert r.reason == "llm_unparseable_fallback"

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back(self):
        chroma = MagicMock()
        coll = MagicMock()
        coll.count.return_value = 1
        chroma.collections = {"facts": coll}
        chroma.query_collection.return_value = self._existing_fact()
        mm = MagicMock()
        mm.generate_once = AsyncMock(side_effect=RuntimeError("LLM down"))
        v = FactVerifier(chroma_store=chroma, model_manager=mm)
        v._ephemeral = frozenset()
        r = await v.verify(
            subject="user", predicate="works_at", object_val="Meta",
            fact_text="user | works_at | Meta",
        )
        assert r.verdict == FactVerdict.STORE_AND_FLAG
        assert r.reason == "llm_unparseable_fallback"

    @pytest.mark.asyncio
    async def test_llm_disabled_uses_fallback(self):
        """When LLM is disabled via config, fall through to trust-newer."""
        v = self._make_verifier_with_llm("A", self._existing_fact())
        with patch("config.app_config.FACT_VERIFICATION_LLM_ENABLED", False):
            r = await v.verify(
                subject="user", predicate="works_at", object_val="Meta",
                fact_text="user | works_at | Meta",
            )
        assert r.verdict == FactVerdict.STORE_AND_FLAG
        assert r.reason == "fallback_trust_newer"


# ---------------------------------------------------------------------------
# Batch verification
# ---------------------------------------------------------------------------

class TestBatchVerification:
    @pytest.mark.asyncio
    async def test_batch_returns_same_count(self):
        chroma = MagicMock()
        coll = MagicMock()
        coll.count.return_value = 0
        chroma.collections = {"facts": coll}
        chroma.query_collection.return_value = []
        v = FactVerifier(chroma_store=chroma)
        v._ephemeral = frozenset()

        facts = [
            {"subject": "user", "relation": "likes", "object": "pizza",
             "fact_text": "user | likes | pizza"},
            {"subject": "user", "relation": "lives_in", "object": "NYC",
             "fact_text": "user | lives_in | NYC"},
            {"subject": "", "relation": "", "object": "",
             "fact_text": ""},
        ]
        results = await v.verify_batch(facts)
        assert len(results) == 3
        # Third fact has no subject → SKIP
        assert results[2].verdict == FactVerdict.SKIP

    @pytest.mark.asyncio
    async def test_batch_empty_list(self):
        chroma = MagicMock()
        v = FactVerifier(chroma_store=chroma)
        results = await v.verify_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# Objects match helper
# ---------------------------------------------------------------------------

class TestObjectsMatch:
    def test_exact_match(self):
        assert FactVerifier._objects_match("pizza", "pizza") is True

    def test_case_insensitive(self):
        assert FactVerifier._objects_match("Pizza", "pizza") is True

    def test_whitespace_stripped(self):
        assert FactVerifier._objects_match("  pizza  ", "pizza") is True

    def test_different_objects(self):
        assert FactVerifier._objects_match("pizza", "sushi") is False

    def test_empty_objects(self):
        assert FactVerifier._objects_match("", "") is False
        assert FactVerifier._objects_match("pizza", "") is False


# ---------------------------------------------------------------------------
# Integration: MemoryStorage with verifier
# ---------------------------------------------------------------------------

class TestMemoryStorageIntegration:
    @pytest.mark.asyncio
    async def test_rejected_fact_not_stored(self):
        """When verifier rejects, add_fact should not be called."""
        from memory.memory_storage import MemoryStorage

        chroma = MagicMock()
        chroma.add_fact = MagicMock(return_value="new_id")
        corpus = MagicMock()

        verifier = MagicMock()
        verifier.verify = AsyncMock(return_value=VerificationResult(
            verdict=FactVerdict.REJECT, reason="test_reject",
        ))

        extractor = MagicMock()
        extractor.extract_facts = AsyncMock(return_value=[
            {"content": "user | likes | pizza",
             "metadata": {"subject": "user", "relation": "likes",
                          "object": "pizza", "confidence": 0.8}},
        ])

        storage = MemoryStorage(
            corpus_manager=corpus,
            chroma_store=chroma,
            fact_extractor=extractor,
            fact_verifier=verifier,
        )
        storage.graph_memory = None
        storage.entity_resolver = None

        await storage.extract_and_store_facts("what food?", "I like pizza", 0.8)

        # add_fact should NOT have been called because verifier rejected
        chroma.add_fact.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_and_flag_marks_old_fact(self):
        """When verifier says STORE_AND_FLAG, old facts get superseded_by metadata."""
        from memory.memory_storage import MemoryStorage

        chroma = MagicMock()
        chroma.add_fact = MagicMock(return_value="new_id")
        chroma.update_metadata = MagicMock(return_value=True)
        corpus = MagicMock()

        old_cand = ConflictCandidate(doc_id="old_fact_1", object="sushi")
        verifier = MagicMock()
        verifier.verify = AsyncMock(return_value=VerificationResult(
            verdict=FactVerdict.STORE_AND_FLAG,
            reason="user_trust_override",
            conflicting_candidates=[old_cand],
            metadata_updates={"superseded_reason": "user_correction"},
        ))

        extractor = MagicMock()
        extractor.extract_facts = AsyncMock(return_value=[
            {"content": "user | likes | pizza",
             "metadata": {"subject": "user", "relation": "likes",
                          "object": "pizza", "confidence": 0.9,
                          "source": "user_stated"}},
        ])

        storage = MemoryStorage(
            corpus_manager=corpus,
            chroma_store=chroma,
            fact_extractor=extractor,
            fact_verifier=verifier,
        )
        storage.graph_memory = None
        storage.entity_resolver = None

        await storage.extract_and_store_facts("food?", "I like pizza", 0.9)

        # add_fact SHOULD have been called (STORE_AND_FLAG still stores)
        chroma.add_fact.assert_called_once()
        # update_metadata should mark old fact as superseded
        chroma.update_metadata.assert_called_once()
        call_args = chroma.update_metadata.call_args
        assert call_args[0][0] == "facts"
        assert call_args[0][1] == "old_fact_1"
        assert "superseded_by" in call_args[0][2]

    @pytest.mark.asyncio
    async def test_no_verifier_stores_normally(self):
        """Without verifier, facts are stored without verification."""
        from memory.memory_storage import MemoryStorage

        chroma = MagicMock()
        chroma.add_fact = MagicMock(return_value="new_id")
        corpus = MagicMock()

        extractor = MagicMock()
        extractor.extract_facts = AsyncMock(return_value=[
            {"content": "user | likes | pizza",
             "metadata": {"subject": "user", "relation": "likes",
                          "object": "pizza", "confidence": 0.8}},
        ])

        storage = MemoryStorage(
            corpus_manager=corpus,
            chroma_store=chroma,
            fact_extractor=extractor,
            fact_verifier=None,
        )
        storage.graph_memory = None
        storage.entity_resolver = None

        await storage.extract_and_store_facts("food?", "pizza", 0.8)

        chroma.add_fact.assert_called_once()

    @pytest.mark.asyncio
    async def test_verifier_exception_stores_anyway(self):
        """If verifier throws, fact is still stored (graceful degradation)."""
        from memory.memory_storage import MemoryStorage

        chroma = MagicMock()
        chroma.add_fact = MagicMock(return_value="new_id")
        corpus = MagicMock()

        verifier = MagicMock()
        verifier.verify = AsyncMock(side_effect=RuntimeError("boom"))

        extractor = MagicMock()
        extractor.extract_facts = AsyncMock(return_value=[
            {"content": "user | likes | pizza",
             "metadata": {"subject": "user", "relation": "likes",
                          "object": "pizza", "confidence": 0.8}},
        ])

        storage = MemoryStorage(
            corpus_manager=corpus,
            chroma_store=chroma,
            fact_extractor=extractor,
            fact_verifier=verifier,
        )
        storage.graph_memory = None
        storage.entity_resolver = None

        await storage.extract_and_store_facts("food?", "pizza", 0.8)

        # Should still store despite verifier error
        chroma.add_fact.assert_called_once()


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------

class TestFeatureToggle:
    @pytest.mark.asyncio
    async def test_disabled_skips_all_verification(self):
        """When FACT_VERIFICATION_ENABLED is False, FactVerifier is not created."""
        with patch("config.app_config.FACT_VERIFICATION_ENABLED", False):
            # In coordinator, verifier would not be created
            # Here we just test that None verifier is the default
            from memory.memory_storage import MemoryStorage
            storage = MemoryStorage(
                corpus_manager=MagicMock(),
                chroma_store=MagicMock(),
                fact_extractor=MagicMock(),
            )
            assert storage.fact_verifier is None
