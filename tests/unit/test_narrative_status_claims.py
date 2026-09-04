"""
Status-claim conflict guard (2026-09-04 class-closing fix).

The [TEMPORAL GROUNDING] narrative wrote "he'd withdrawn from the fall
semester" while the profile held a CURRENT `enrolled_in` fact and a
`dropped=<one course>` fact — the user's own words were that one course
was dropped, not the whole semester. utils/status_claims.py is a leaf,
purely-mechanical module (no UserProfile import) imported by both the
narrative path (memory/memory_consolidator.py) and the daily-note path
(utils/daily_notes_generator.py); drives THE deployed functions directly.
"""

from utils.status_claims import (
    STATUS_RELATIONS,
    authoritative_facts_block,
    remove_conflicting_claims,
    status_claim_conflicts,
)


class TestEnrollmentConflict:
    def test_live_withdrawal_sentence_conflicts_with_current_enrollment(self):
        text = (
            "The user is managing coursework this term after he'd withdrawn "
            "from the fall semester following a health scare. He has been "
            "catching up steadily since then."
        )
        facts = [{"relation": "enrolled_in", "value": "Course XYZ", "is_current": True}]

        conflicts = status_claim_conflicts(text, facts)
        assert len(conflicts) == 1
        assert conflicts[0].family == "enrollment"
        assert conflicts[0].relation == "enrolled_in"
        assert conflicts[0].value == "Course XYZ"

        revised, removed = remove_conflicting_claims(text, facts)
        assert len(removed) == 1
        body, _, caution = revised.partition("[CAUTION:")
        assert "withdrawn" not in body
        assert "fall semester" not in body
        assert "catching up steadily" in body
        assert caution  # the caution block itself quotes the removed claim
        assert "enrolled_in=Course XYZ" in revised

    def test_specific_course_drop_is_kept_not_a_conflict(self):
        text = (
            "The user dropped CSE 6200 this term to reduce course load and "
            "feels relieved about the change."
        )
        facts = [{"relation": "dropped", "value": "CSE 6200", "is_current": True}]

        conflicts = status_claim_conflicts(text, facts)
        assert conflicts == []

        revised, removed = remove_conflicting_claims(text, facts)
        assert revised == text
        assert removed == []

    def test_no_profile_facts_leaves_text_unchanged(self):
        text = "He withdrew from the fall semester after a health scare."

        assert status_claim_conflicts(text, []) == []
        assert status_claim_conflicts(text, None) == []

        revised, removed = remove_conflicting_claims(text, [])
        assert revised == text
        assert removed == []


class TestEmploymentConflict:
    def test_job_loss_claim_conflicts_with_current_employment(self):
        text = "He quit his job last month and has been job hunting since."
        facts = [{"relation": "works_at", "value": "Example Corp", "is_current": True}]

        conflicts = status_claim_conflicts(text, facts)
        assert len(conflicts) == 1
        assert conflicts[0].family == "employment"

        revised, removed = remove_conflicting_claims(text, facts)
        assert len(removed) == 1
        body, _, caution = revised.partition("[CAUTION:")
        assert "quit his job" not in body
        assert "works_at=Example Corp" in revised


class TestResidenceConflict:
    def test_move_claim_conflicts_with_current_residence(self):
        text = "She moved back to Rivertown after the semester ended."
        facts = [{"relation": "lives_in", "value": "Example City", "is_current": True}]

        conflicts = status_claim_conflicts(text, facts)
        assert len(conflicts) == 1
        assert conflicts[0].family == "residence"

        revised, removed = remove_conflicting_claims(text, facts)
        assert len(removed) == 1
        body, _, caution = revised.partition("[CAUTION:")
        assert "moved back to Rivertown" not in body
        assert "lives_in=Example City" in revised


class TestAuthoritativeFactsBlock:
    def test_renders_only_status_relations(self):
        facts = [
            {"relation": "enrolled_in", "value": "Course XYZ"},
            {"relation": "favorite_color", "value": "blue"},  # not a status relation
        ]
        block = authoritative_facts_block(facts)
        assert "AUTHORITATIVE CURRENT FACTS" in block
        assert "enrolled_in = Course XYZ" in block
        assert "favorite_color" not in block

    def test_empty_when_no_status_facts(self):
        assert authoritative_facts_block([]) == ""
        assert authoritative_facts_block([{"relation": "favorite_color", "value": "blue"}]) == ""

    def test_status_relations_cover_all_three_families(self):
        assert {"enrolled_in", "program", "dropped"} <= STATUS_RELATIONS
        assert {"works_at", "occupation", "employer"} <= STATUS_RELATIONS
        assert {"lives_in"} <= STATUS_RELATIONS


class TestNarrativePathWiring:
    """memory/memory_consolidator.py imports + uses the guard."""

    def test_prompt_template_has_authoritative_facts_placeholder(self):
        from memory.memory_consolidator import MemoryConsolidator

        assert "{authoritative_status_facts}" in MemoryConsolidator.NARRATIVE_SYNTHESIS_PROMPT

    def test_generate_narrative_context_removes_conflict(self, monkeypatch):
        import asyncio
        from memory.memory_consolidator import MemoryConsolidator

        class FakeModelManager:
            async def generate_once(self, prompt, **kwargs):
                assert "AUTHORITATIVE CURRENT FACTS" in prompt
                assert "enrolled_in = Course XYZ" in prompt
                return (
                    "The user is managing coursework this term after having "
                    "withdrawn from the fall semester following a health "
                    "scare. Mood has been steady this week."
                )

        class FakeProfile:
            def get_current_view(self):
                return {
                    "identity": [
                        {"relation": "enrolled_in", "value": "Course XYZ", "is_current": True},
                    ]
                }

        consolidator = MemoryConsolidator(FakeModelManager(), user_profile=FakeProfile())
        monkeypatch.setattr(
            consolidator, "_read_obsidian_monthly_summaries", lambda limit=1: []
        )
        monkeypatch.setattr(
            consolidator, "_read_obsidian_weekly_summaries", lambda limit=2: []
        )
        monkeypatch.setattr(
            consolidator, "_read_obsidian_daily_notes", lambda limit=7: []
        )

        narrative = asyncio.run(
            consolidator.generate_narrative_context(
                recent_weeklies=[{"timestamp": "2026-09-01", "content": "week stuff"}],
            )
        )
        body, _, caution = narrative.partition("[CAUTION:")
        assert "withdrawn" not in body
        assert "fall semester" not in body
        assert "Mood has been steady this week" in body
        assert "enrolled_in=Course XYZ" in narrative


class TestDailyNotePathWiring:
    """utils/daily_notes_generator.py imports + uses the guard."""

    def test_prompt_template_has_authoritative_facts_placeholder(self):
        from utils.daily_notes_generator import SYSTEM_PROMPT_TEMPLATE

        assert "{authoritative_status_facts}" in SYSTEM_PROMPT_TEMPLATE
