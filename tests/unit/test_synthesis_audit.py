"""
Tests for the synthesis audit queue: 1-5 structural grading, stats,
auto-halt, and composite-reject storage.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from knowledge.synthesis_models import (
    CandidateStatus,
    CoherenceLevel,
    SynthesisCandidate,
    SynthesisResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(concept_a="my dog", concept_b="Pavlov", claim="conditioning"):
    return SynthesisCandidate(
        concept_a=concept_a,
        concept_b=concept_b,
        connection_claim=claim,
        walk_path=["a", "b"],
        source_domains={"personal", "psychology"},
        endpoint_distance=0.5,
    )


def _make_result(
    concept_a="my dog",
    concept_b="Pavlov",
    claim="conditioning parallel",
    status=CandidateStatus.ACCEPTED,
    coherence=CoherenceLevel.MODERATE,
    composite=0.70,
    human_grade="",
    rejection_stage=None,
    rejection_reason="",
):
    candidate = _make_candidate(concept_a, concept_b, claim)
    r = SynthesisResult(candidate=candidate)
    r.status = status
    r.coherence_level = coherence
    r.composite_score = composite
    r.human_grade = human_grade
    r.rejection_stage = rejection_stage
    r.rejection_reason = rejection_reason
    return r


class FakeChromaStore:
    """Minimal in-memory ChromaDB mock for synthesis_results collection."""

    def __init__(self):
        self._docs = {}  # doc_id -> {text, metadata}
        self._counter = 0

    def add_to_collection(self, name, text, metadata):
        self._counter += 1
        doc_id = f"synth_{self._counter}"
        self._docs[doc_id] = {"text": text, "metadata": dict(metadata)}
        return doc_id

    def query_collection(self, collection_name, query_text, n_results=10):
        items = []
        for doc_id, doc in self._docs.items():
            items.append({
                "id": doc_id,
                "content": doc["text"],
                "metadata": dict(doc["metadata"]),
                "relevance_score": 0.5,  # below similarity threshold
            })
        return items[:n_results]

    def update_metadata(self, collection_name, doc_id, metadata_updates):
        if doc_id in self._docs:
            self._docs[doc_id]["metadata"].update(metadata_updates)

    def get_collection_stats(self):
        return {"synthesis_results": {"count": len(self._docs)}}


# ---------------------------------------------------------------------------
# SynthesisResult audit field tests
# ---------------------------------------------------------------------------

class TestSynthesisResultAuditFields:
    def test_audit_fields_default_empty(self):
        r = _make_result()
        assert r.human_grade == ""
        assert r.graded_at == ""
        assert r.grade_notes == ""

    def test_audit_fields_in_to_metadata(self):
        r = _make_result(human_grade="4")
        r.graded_at = "2026-04-01T12:00:00"
        r.grade_notes = "strong isomorphism"
        meta = r.to_metadata()
        assert meta["human_grade"] == "4"
        assert meta["graded_at"] == "2026-04-01T12:00:00"
        assert meta["grade_notes"] == "strong isomorphism"

    def test_audit_fields_from_metadata(self):
        meta = {
            "concept_a": "a",
            "concept_b": "b",
            "human_grade": "2",
            "graded_at": "2026-04-01T13:00:00",
            "grade_notes": "surface metaphor only",
            "status": "accepted",
        }
        r = SynthesisResult.from_metadata(meta, "claim text")
        assert r.human_grade == "2"
        assert r.graded_at == "2026-04-01T13:00:00"
        assert r.grade_notes == "surface metaphor only"

    def test_audit_fields_roundtrip(self):
        r = _make_result(human_grade="5")
        r.graded_at = "2026-04-01T14:00:00"
        r.grade_notes = "deep convergence"
        meta = r.to_metadata()
        r2 = SynthesisResult.from_metadata(meta, r.candidate.connection_claim)
        assert r2.human_grade == "5"
        assert r2.graded_at == "2026-04-01T14:00:00"
        assert r2.grade_notes == "deep convergence"


# ---------------------------------------------------------------------------
# SynthesisMemory audit method tests
# ---------------------------------------------------------------------------

class TestSynthesisMemoryGrading:
    def _make_memory(self):
        from memory.synthesis_memory import SynthesisMemory
        store = FakeChromaStore()
        return SynthesisMemory(store), store

    def test_grade_result_numeric_4(self):
        sm, store = self._make_memory()
        r = _make_result()
        doc_id = sm.store_result(r)
        assert sm.grade_result(doc_id, "4")
        assert store._docs[doc_id]["metadata"]["human_grade"] == "4"
        assert store._docs[doc_id]["metadata"]["graded_at"] != ""

    def test_grade_result_numeric_1(self):
        sm, store = self._make_memory()
        r = _make_result()
        doc_id = sm.store_result(r)
        assert sm.grade_result(doc_id, "1", notes="hallucination")
        assert store._docs[doc_id]["metadata"]["human_grade"] == "1"
        assert store._docs[doc_id]["metadata"]["grade_notes"] == "hallucination"

    def test_grade_result_numeric_5(self):
        sm, store = self._make_memory()
        r = _make_result()
        doc_id = sm.store_result(r)
        assert sm.grade_result(doc_id, "5")

    def test_grade_result_numeric_3_borderline(self):
        sm, store = self._make_memory()
        r = _make_result()
        doc_id = sm.store_result(r)
        assert sm.grade_result(doc_id, "3", notes="true but trivial")

    def test_grade_result_legacy_valid_still_works(self):
        sm, store = self._make_memory()
        r = _make_result()
        doc_id = sm.store_result(r)
        assert sm.grade_result(doc_id, "valid")

    def test_grade_result_rejects_invalid_grade(self):
        sm, _ = self._make_memory()
        r = _make_result()
        doc_id = sm.store_result(r)
        assert not sm.grade_result(doc_id, "maybe")
        assert not sm.grade_result(doc_id, "6")
        assert not sm.grade_result(doc_id, "0")

    def test_grade_result_with_notes(self):
        sm, store = self._make_memory()
        r = _make_result()
        doc_id = sm.store_result(r)
        sm.grade_result(doc_id, "4", "strong mechanism naming")
        assert store._docs[doc_id]["metadata"]["grade_notes"] == "strong mechanism naming"


class TestSynthesisMemoryStoreRejected:
    def _make_memory(self):
        from memory.synthesis_memory import SynthesisMemory
        store = FakeChromaStore()
        return SynthesisMemory(store), store

    def test_store_rejected_for_audit(self):
        sm, store = self._make_memory()
        r = _make_result(
            status=CandidateStatus.REJECTED,
            rejection_stage="composite_scoring",
            rejection_reason="score 0.62 < 0.65",
        )
        doc_id = sm.store_rejected_for_audit(r)
        assert doc_id
        assert store._docs[doc_id]["metadata"]["status"] == "rejected"

    def test_store_rejected_preserves_rejection_info(self):
        sm, store = self._make_memory()
        r = _make_result(
            status=CandidateStatus.REJECTED,
            rejection_stage="composite_scoring",
            rejection_reason="score too low",
        )
        doc_id = sm.store_rejected_for_audit(r)
        meta = store._docs[doc_id]["metadata"]
        assert meta["status"] == "rejected"


class TestSynthesisMemoryQueries:
    def _make_memory_with_data(self):
        from memory.synthesis_memory import SynthesisMemory
        store = FakeChromaStore()
        sm = SynthesisMemory(store)

        # 2 accepted, 1 graded (4=valid), 1 ungraded
        r1 = _make_result(concept_a="dog", concept_b="Pavlov", claim="conditioning 1")
        r1.human_grade = "4"
        r1.graded_at = "2026-04-01T10:00:00"
        id1 = store.add_to_collection("synthesis_results", r1.candidate.connection_claim, r1.to_metadata())

        r2 = _make_result(concept_a="cat", concept_b="Schrodinger", claim="superposition 2")
        id2 = store.add_to_collection("synthesis_results", r2.candidate.connection_claim, r2.to_metadata())

        # 2 rejected (composite), 1 graded (2=surface metaphor), 1 ungraded
        r3 = _make_result(
            concept_a="fish", concept_b="Nemo", claim="finding 3",
            status=CandidateStatus.REJECTED,
        )
        r3.human_grade = "2"
        r3.graded_at = "2026-04-01T11:00:00"
        id3 = store.add_to_collection("synthesis_results", r3.candidate.connection_claim, r3.to_metadata())

        r4 = _make_result(
            concept_a="bird", concept_b="Darwin", claim="evolution 4",
            status=CandidateStatus.REJECTED,
        )
        id4 = store.add_to_collection("synthesis_results", r4.candidate.connection_claim, r4.to_metadata())

        return sm, store, [id1, id2, id3, id4]

    def test_get_all_results(self):
        sm, _, ids = self._make_memory_with_data()
        results = sm.get_all_results()
        assert len(results) == 4

    def test_get_ungraded_accepted(self):
        sm, _, ids = self._make_memory_with_data()
        ungraded = sm.get_ungraded(status_filter="accepted")
        assert len(ungraded) == 1
        assert ungraded[0][1].candidate.concept_a == "cat"

    def test_get_ungraded_rejected(self):
        sm, _, ids = self._make_memory_with_data()
        ungraded = sm.get_ungraded(status_filter="rejected")
        assert len(ungraded) == 1
        assert ungraded[0][1].candidate.concept_a == "bird"

    def test_get_ungraded_all(self):
        sm, _, ids = self._make_memory_with_data()
        ungraded = sm.get_ungraded()
        assert len(ungraded) == 2

    def test_get_graded(self):
        sm, _, ids = self._make_memory_with_data()
        graded = sm.get_graded()
        assert len(graded) == 2
        assert graded[0][1].graded_at >= graded[1][1].graded_at


# ---------------------------------------------------------------------------
# Grade classification helpers
# ---------------------------------------------------------------------------

class TestGradeClassification:
    def test_grade_4_is_valid(self):
        from memory.synthesis_memory import SynthesisMemory
        assert SynthesisMemory._grade_is_valid("4")
        assert SynthesisMemory._grade_is_valid("5")

    def test_grade_1_2_3_is_invalid(self):
        from memory.synthesis_memory import SynthesisMemory
        assert SynthesisMemory._grade_is_invalid("1")
        assert SynthesisMemory._grade_is_invalid("2")
        assert SynthesisMemory._grade_is_invalid("3")

    def test_grade_3_not_valid(self):
        from memory.synthesis_memory import SynthesisMemory
        assert not SynthesisMemory._grade_is_valid("3")

    def test_grade_4_not_invalid(self):
        from memory.synthesis_memory import SynthesisMemory
        assert not SynthesisMemory._grade_is_invalid("4")

    def test_legacy_valid_still_valid(self):
        from memory.synthesis_memory import SynthesisMemory
        assert SynthesisMemory._grade_is_valid("valid")
        assert SynthesisMemory._grade_is_valid("should_pass")

    def test_legacy_invalid_still_invalid(self):
        from memory.synthesis_memory import SynthesisMemory
        assert SynthesisMemory._grade_is_invalid("invalid")
        assert SynthesisMemory._grade_is_invalid("correct_reject")


# ---------------------------------------------------------------------------
# Audit stats tests (numeric grades)
# ---------------------------------------------------------------------------

class TestAuditStats:
    def _make_memory_with_grades(self, grades_4_5=0, grades_1_3=0, ungraded_acc=0, ungraded_rej=0):
        """Create memory with numeric grades. grades_4_5 get "4", grades_1_3 get "2"."""
        from memory.synthesis_memory import SynthesisMemory
        store = FakeChromaStore()
        sm = SynthesisMemory(store)
        counter = 0

        for _ in range(grades_4_5):
            counter += 1
            r = _make_result(concept_a=f"v{counter}", claim=f"claim {counter}", human_grade="4")
            r.graded_at = "2026-04-01T10:00:00"
            store.add_to_collection("synthesis_results", r.candidate.connection_claim, r.to_metadata())

        for _ in range(grades_1_3):
            counter += 1
            r = _make_result(concept_a=f"i{counter}", claim=f"claim {counter}", human_grade="2")
            r.graded_at = "2026-04-01T10:00:00"
            store.add_to_collection("synthesis_results", r.candidate.connection_claim, r.to_metadata())

        for _ in range(ungraded_acc):
            counter += 1
            r = _make_result(concept_a=f"ua{counter}", claim=f"claim {counter}")
            store.add_to_collection("synthesis_results", r.candidate.connection_claim, r.to_metadata())

        for _ in range(ungraded_rej):
            counter += 1
            r = _make_result(
                concept_a=f"ur{counter}", claim=f"claim {counter}",
                status=CandidateStatus.REJECTED,
            )
            store.add_to_collection("synthesis_results", r.candidate.connection_claim, r.to_metadata())

        return sm

    @patch("config.app_config.SYNTHESIS_AUDIT_FP_HALT_THRESHOLD", 0.50)
    @patch("config.app_config.SYNTHESIS_AUDIT_MIN_GRADED", 10)
    def test_stats_empty(self):
        sm = self._make_memory_with_grades()
        stats = sm.get_audit_stats()
        assert stats["total_graded"] == 0
        assert stats["fp_rate"] == 0.0
        assert stats["auto_halt"] is False

    @patch("config.app_config.SYNTHESIS_AUDIT_FP_HALT_THRESHOLD", 0.50)
    @patch("config.app_config.SYNTHESIS_AUDIT_MIN_GRADED", 10)
    def test_stats_all_valid(self):
        sm = self._make_memory_with_grades(grades_4_5=10)
        stats = sm.get_audit_stats()
        assert stats["valid_count"] == 10
        assert stats["invalid_count"] == 0
        assert stats["fp_rate"] == 0.0
        assert stats["auto_halt"] is False

    @patch("config.app_config.SYNTHESIS_AUDIT_FP_HALT_THRESHOLD", 0.50)
    @patch("config.app_config.SYNTHESIS_AUDIT_MIN_GRADED", 10)
    def test_stats_high_fp_triggers_halt(self):
        sm = self._make_memory_with_grades(grades_4_5=3, grades_1_3=8)
        stats = sm.get_audit_stats()
        assert stats["fp_rate"] == round(8 / 11, 3)
        assert stats["auto_halt"] is True

    @patch("config.app_config.SYNTHESIS_AUDIT_FP_HALT_THRESHOLD", 0.50)
    @patch("config.app_config.SYNTHESIS_AUDIT_MIN_GRADED", 10)
    def test_stats_high_fp_below_min_graded_no_halt(self):
        sm = self._make_memory_with_grades(grades_4_5=1, grades_1_3=5)
        stats = sm.get_audit_stats()
        assert stats["fp_rate"] > 0.50
        assert stats["auto_halt"] is False

    @patch("config.app_config.SYNTHESIS_AUDIT_FP_HALT_THRESHOLD", 0.50)
    @patch("config.app_config.SYNTHESIS_AUDIT_MIN_GRADED", 10)
    def test_stats_ungraded_counts(self):
        sm = self._make_memory_with_grades(ungraded_acc=5, ungraded_rej=3)
        stats = sm.get_audit_stats()
        assert stats["ungraded_accepted"] == 5
        assert stats["ungraded_rejected"] == 3

    @patch("config.app_config.SYNTHESIS_AUDIT_FP_HALT_THRESHOLD", 0.30)
    @patch("config.app_config.SYNTHESIS_AUDIT_MIN_GRADED", 5)
    def test_stats_custom_threshold(self):
        sm = self._make_memory_with_grades(grades_4_5=3, grades_1_3=2)
        stats = sm.get_audit_stats()
        # FP rate = 2/5 = 0.40 > 0.30 threshold, graded=5 >= min=5
        assert stats["auto_halt"] is True

    @patch("config.app_config.SYNTHESIS_AUDIT_FP_HALT_THRESHOLD", 0.50)
    @patch("config.app_config.SYNTHESIS_AUDIT_MIN_GRADED", 10)
    def test_stats_avg_grade(self):
        sm = self._make_memory_with_grades(grades_4_5=5, grades_1_3=5)
        stats = sm.get_audit_stats()
        # 5 results at grade "4" + 5 at grade "2" = avg 3.0
        assert stats["avg_grade"] == 3.0

    @patch("config.app_config.SYNTHESIS_AUDIT_FP_HALT_THRESHOLD", 0.50)
    @patch("config.app_config.SYNTHESIS_AUDIT_MIN_GRADED", 10)
    def test_stats_mixed(self):
        sm = self._make_memory_with_grades(
            grades_4_5=7, grades_1_3=3,
            ungraded_acc=2, ungraded_rej=1,
        )
        stats = sm.get_audit_stats()
        assert stats["valid_count"] == 7
        assert stats["invalid_count"] == 3
        assert stats["fp_rate"] == 0.3  # 3/10
        assert stats["auto_halt"] is False  # 0.3 < 0.5
        assert stats["ungraded_accepted"] == 2
        assert stats["ungraded_rejected"] == 1


class TestAutoHaltIntegration:
    @patch("config.app_config.SYNTHESIS_AUDIT_FP_HALT_THRESHOLD", 0.50)
    @patch("config.app_config.SYNTHESIS_AUDIT_MIN_GRADED", 10)
    def test_auto_halt_flag_in_stats(self):
        from memory.synthesis_memory import SynthesisMemory
        store = FakeChromaStore()
        sm = SynthesisMemory(store)

        # Add 4 grade-4 + 8 grade-1 = 12 graded, FP rate = 8/12 = 0.67 > 0.50
        for i in range(4):
            r = _make_result(concept_a=f"v{i}", claim=f"c{i}", human_grade="4")
            r.graded_at = "2026-04-01"
            store.add_to_collection("synthesis_results", r.candidate.connection_claim, r.to_metadata())
        for i in range(8):
            r = _make_result(concept_a=f"i{i}", claim=f"ci{i}", human_grade="1")
            r.graded_at = "2026-04-01"
            store.add_to_collection("synthesis_results", r.candidate.connection_claim, r.to_metadata())

        stats = sm.get_audit_stats()
        assert stats["auto_halt"] is True
        assert stats["fp_rate"] == round(8 / 12, 3)


class TestFilterStoresCompositeRejects:
    @pytest.mark.asyncio
    async def test_composite_rejects_stored_for_audit(self):
        from memory.synthesis_memory import SynthesisMemory

        store = FakeChromaStore()
        sm = SynthesisMemory(store)

        r = _make_result(
            status=CandidateStatus.REJECTED,
            rejection_stage="composite_scoring",
            rejection_reason="score 0.62 < 0.65",
        )
        doc_id = sm.store_rejected_for_audit(r)
        assert doc_id
        assert store._docs[doc_id]["metadata"]["status"] == "rejected"
