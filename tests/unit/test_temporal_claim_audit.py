"""Regression tests for temporal claims crossing daily/narrative synthesis."""

from datetime import datetime

from memory.memory_consolidator import MemoryConsolidator
from utils.daily_notes_generator import (
    SYSTEM_PROMPT_TEMPLATE,
    build_temporal_claim_audit,
)


def test_competing_recovery_durations_are_preserved_with_context_labels():
    convos = [
        {
            "timestamp": datetime(2026, 9, 2, 16, 20),
            "query": (
                "The last 6 days probably mean I am now good to go. "
                "If I lose function in 2 weeks, that would be a problem."
            ),
        },
        {
            "timestamp": datetime(2026, 9, 2, 16, 45),
            "query": (
                "Look ok? Draft email: My health has stabilized. "
                "I've been consistently functional for 2 weeks now."
            ),
        },
    ]

    audit = build_temporal_claim_audit(convos)

    assert "last 6 days" in audit
    assert "function in 2 weeks" in audit
    assert "functional for 2 weeks" in audit
    assert "[direct-user-statement]" in audit
    assert "[conditional]" in audit
    assert "[draft-or-pasted-text]" in audit
    assert "do not silently choose" in audit


def test_temporal_claim_audit_ignores_unrelated_deadline_duration():
    audit = build_temporal_claim_audit([
        {
            "timestamp": "2026-09-02T10:00:00",
            "query": "The assignment is due in two weeks and has seven questions.",
        }
    ])

    assert audit == "(No mechanically detected status-duration claims.)"


def test_temporal_claim_audit_redacts_identifiers_in_claim_window():
    audit = build_temporal_claim_audit([
        {
            "timestamp": "2026-09-02T10:00:00",
            "query": (
                "I have been functional for 2 weeks; call 312-555-0199 or use "
                "student@example.edu, ID 123456789."
            ),
        }
    ])

    assert "312-555-0199" not in audit
    assert "student@example.edu" not in audit
    assert "123456789" not in audit
    assert "[PHONE]" in audit
    assert "[EMAIL]" in audit
    assert "[ID]" in audit


def test_both_synthesis_prompts_require_temporal_reconciliation():
    assert "TEMPORAL STATUS CLAIM AUDIT" in SYSTEM_PROMPT_TEMPLATE
    assert "Never silently replace one with the other" in SYSTEM_PROMPT_TEMPLATE
    assert "Never silently select, average, or" in MemoryConsolidator.NARRATIVE_SYNTHESIS_PROMPT

