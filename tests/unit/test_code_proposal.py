# tests/unit/test_code_proposal.py
"""Unit tests for memory.code_proposal data models."""

import json
import time

import pytest

from memory.code_proposal import (
    CodeProposal,
    ImplementationStep,
    ProposalSource,
    ProposalStatus,
    ProposalType,
)


# ------------------------------------------------------------------
# Enum tests
# ------------------------------------------------------------------


class TestEnums:
    def test_proposal_type_values(self):
        assert ProposalType.FEATURE.value == "feature"
        assert ProposalType.REFACTOR.value == "refactor"
        assert ProposalType.BUGFIX.value == "bugfix"
        assert ProposalType.TEST.value == "test"
        assert ProposalType.DOCS.value == "docs"
        assert ProposalType.INFRA.value == "infra"

    def test_proposal_status_values(self):
        assert ProposalStatus.PENDING.value == "pending"
        assert ProposalStatus.APPROVED.value == "approved"
        assert ProposalStatus.REJECTED.value == "rejected"
        assert ProposalStatus.COMPLETED.value == "completed"
        assert ProposalStatus.FAILED.value == "failed"

    def test_proposal_source_values(self):
        assert ProposalSource.GOAL_DIRECTED.value == "goal_directed"
        assert ProposalSource.SESSION_INSIGHT.value == "session_insight"
        assert ProposalSource.USER_REQUEST.value == "user_request"
        assert ProposalSource.SHUTDOWN_ANALYSIS.value == "shutdown_analysis"

    def test_enums_are_strings(self):
        """Enum values should be usable as strings."""
        assert isinstance(ProposalType.FEATURE.value, str)
        assert isinstance(ProposalStatus.PENDING.value, str)
        assert isinstance(ProposalSource.GOAL_DIRECTED.value, str)


# ------------------------------------------------------------------
# ImplementationStep tests
# ------------------------------------------------------------------


class TestImplementationStep:
    def test_basic_creation(self):
        step = ImplementationStep(order=1, description="Create file")
        assert step.order == 1
        assert step.description == "Create file"
        assert step.file_path == ""
        assert step.action == "modify"
        assert step.code_snippet == ""

    def test_full_creation(self):
        step = ImplementationStep(
            order=2,
            description="Add method",
            file_path="core/handler.py",
            action="modify",
            code_snippet="def handle(): pass",
        )
        assert step.order == 2
        assert step.file_path == "core/handler.py"
        assert step.code_snippet == "def handle(): pass"

    def test_to_dict_from_dict_roundtrip(self):
        step = ImplementationStep(
            order=3,
            description="Delete legacy",
            file_path="old/module.py",
            action="delete",
            code_snippet="",
        )
        d = step.to_dict()
        step2 = ImplementationStep.from_dict(d)
        assert step2.order == step.order
        assert step2.description == step.description
        assert step2.file_path == step.file_path
        assert step2.action == step.action

    def test_from_dict_missing_fields(self):
        step = ImplementationStep.from_dict({"description": "Do thing"})
        assert step.order == 1
        assert step.description == "Do thing"
        assert step.action == "modify"


# ------------------------------------------------------------------
# CodeProposal tests
# ------------------------------------------------------------------


class TestCodeProposal:
    def test_defaults(self):
        p = CodeProposal(title="Test proposal")
        assert p.id  # UUID generated
        assert len(p.id) == 36  # UUID format
        assert p.status == ProposalStatus.PENDING
        assert p.source == ProposalSource.GOAL_DIRECTED
        assert p.proposal_type == ProposalType.FEATURE
        assert p.priority == 5
        assert p.reasoning == ""
        assert p.implementation_steps == []
        assert p.affected_files == []
        assert p.tags == []
        assert p.estimated_complexity == "medium"
        assert p.requires_tests is True
        assert p.rejection_reason == ""
        assert p.commit_hash == ""
        assert p.rollback_available is False
        assert p.created_at > 0
        assert p.modified_at > 0
        assert p.executed_at is None

    def test_to_dict_from_dict_roundtrip(self):
        p = CodeProposal(
            title="Add caching layer",
            proposal_type=ProposalType.FEATURE,
            priority=8,
            reasoning="Performance improvement needed",
            description="Add Redis caching for API responses",
            tags=["performance", "caching"],
            affected_files=["core/api.py", "config/redis.py"],
        )
        p.implementation_steps.append(
            ImplementationStep(order=1, description="Add Redis client", file_path="core/redis.py", action="create")
        )

        d = p.to_dict()
        p2 = CodeProposal.from_dict(d)

        assert p2.id == p.id
        assert p2.title == p.title
        assert p2.proposal_type == p.proposal_type
        assert p2.priority == p.priority
        assert p2.reasoning == p.reasoning
        assert p2.description == p.description
        assert p2.tags == p.tags
        assert p2.affected_files == p.affected_files
        assert len(p2.implementation_steps) == 1
        assert p2.implementation_steps[0].description == "Add Redis client"
        assert p2.requires_tests is True
        assert p2.estimated_complexity == "medium"

    def test_to_metadata_from_metadata_roundtrip(self):
        p = CodeProposal(
            title="Refactor auth module",
            proposal_type=ProposalType.REFACTOR,
            source=ProposalSource.SESSION_INSIGHT,
            priority=7,
            reasoning="Auth module is too large",
            tags=["auth", "refactor"],
        )
        p.implementation_steps.append(
            ImplementationStep(order=1, description="Extract helpers")
        )

        md = p.to_metadata()
        p2 = CodeProposal.from_metadata(md)

        assert p2.id == p.id
        assert p2.title == p.title
        assert p2.proposal_type == ProposalType.REFACTOR
        assert p2.source == ProposalSource.SESSION_INSIGHT
        assert p2.priority == 7
        assert p2.tags == ["auth", "refactor"]
        assert len(p2.implementation_steps) == 1

    def test_to_embedding_text(self):
        p = CodeProposal(
            title="Add logging",
            proposal_type=ProposalType.INFRA,
            reasoning="Need better observability",
            tags=["logging", "observability"],
        )
        text = p.to_embedding_text()
        assert "Add logging" in text
        assert "observability" in text
        assert "infra" in text

    def test_from_dict_missing_optional_fields(self):
        """from_dict should handle dicts with only required fields."""
        p = CodeProposal.from_dict({"title": "Minimal"})
        assert p.title == "Minimal"
        assert p.status == ProposalStatus.PENDING
        assert p.implementation_steps == []

    def test_from_dict_missing_implementation_steps_key(self):
        data = {"title": "No steps", "priority": 3}
        p = CodeProposal.from_dict(data)
        assert p.implementation_steps == []
        assert p.priority == 3

    def test_from_dict_with_extra_keys(self):
        """Extra keys should not cause errors."""
        data = {"title": "Test", "unknown_key": "value", "another": 42}
        p = CodeProposal.from_dict(data)
        assert p.title == "Test"


# ------------------------------------------------------------------
# Status lifecycle tests
# ------------------------------------------------------------------


class TestStatusLifecycle:
    def test_mark_approved(self):
        p = CodeProposal(title="Test")
        before = p.modified_at
        time.sleep(0.01)
        p.mark_approved()
        assert p.status == ProposalStatus.APPROVED
        assert p.modified_at > before

    def test_mark_rejected_with_reason(self):
        p = CodeProposal(title="Test")
        p.mark_rejected("Too complex")
        assert p.status == ProposalStatus.REJECTED
        assert p.rejection_reason == "Too complex"

    def test_mark_rejected_without_reason(self):
        p = CodeProposal(title="Test")
        p.mark_rejected()
        assert p.status == ProposalStatus.REJECTED
        assert p.rejection_reason == ""

    def test_mark_completed_with_hash(self):
        p = CodeProposal(title="Test")
        p.mark_completed("abc123def")
        assert p.status == ProposalStatus.COMPLETED
        assert p.commit_hash == "abc123def"
        assert p.executed_at is not None
        assert p.executed_at > 0
        assert p.rollback_available is True

    def test_mark_completed_without_hash(self):
        p = CodeProposal(title="Test")
        p.mark_completed()
        assert p.status == ProposalStatus.COMPLETED
        assert p.commit_hash == ""
        assert p.rollback_available is False

    def test_mark_failed_with_error(self):
        p = CodeProposal(title="Test")
        p.mark_failed("ImportError in module")
        assert p.status == ProposalStatus.FAILED
        assert "Execution failed: ImportError in module" in p.rejection_reason

    def test_mark_failed_without_error(self):
        p = CodeProposal(title="Test")
        p.mark_failed()
        assert p.status == ProposalStatus.FAILED
        assert p.rejection_reason == "Execution failed"


# ------------------------------------------------------------------
# Metadata serialization edge cases
# ------------------------------------------------------------------


class TestMetadataSerialization:
    def test_metadata_values_are_primitives(self):
        """All metadata values must be str/int/float/bool for ChromaDB."""
        p = CodeProposal(
            title="Test",
            tags=["a", "b"],
            affected_files=["f1.py"],
            implementation_steps=[ImplementationStep(order=1, description="s")],
        )
        md = p.to_metadata()
        for key, value in md.items():
            assert isinstance(value, (str, int, float, bool)), (
                f"Key '{key}' has non-primitive type {type(value)}"
            )

    def test_long_reasoning_truncated_in_metadata(self):
        long_reasoning = "x" * 2000
        p = CodeProposal(title="Test", reasoning=long_reasoning)
        md = p.to_metadata()
        assert len(md["reasoning"]) == 1000

    def test_long_description_truncated_in_metadata(self):
        long_desc = "y" * 5000
        p = CodeProposal(title="Test", description=long_desc)
        md = p.to_metadata()
        assert len(md["description"]) == 2000

    def test_full_dict_preserves_full_text(self):
        """to_dict() should NOT truncate (unlike to_metadata)."""
        long_reasoning = "x" * 2000
        p = CodeProposal(title="Test", reasoning=long_reasoning)
        d = p.to_dict()
        assert len(d["reasoning"]) == 2000
