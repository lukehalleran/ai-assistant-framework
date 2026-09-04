"""2026-09-03 night — fixes exposed by two live probe turns after the restart.

1. STM ran on a bare greeting after a gap and summarized the PREVIOUS turn as a
   restatement (recall warning on "Hey").
2. `temporal_recall@0.85` fired on narration ("turned 2 last week") and its
   retrieval profile pulled 20 turns + 8 summaries into a pet anecdote.
3. "[OpenAI unavailable]" sentinel replies were not in API_ERROR_PREFIXES.
4. `has_dad="dad is picking up at 5:30"` — an event mined into a family relation.
5. Weekly/monthly rollup summary notes surfaced on a non-retrospective turn.
6. Thread line said "… thread about general".
"""
import inspect

import pytest

from core.context_pipeline import stm_skip_shape
from core.intent_classifier import IntentClassifier, IntentType
from core.prompt import gatherer_knowledge as gk
from memory.fact_extractor import _is_junk_object
from memory.utils import is_junk_conversation_doc
from models.model_manager import API_ERROR_PREFIXES


class TestStmSkipShape:
    @pytest.mark.parametrize("q", ["Hey", "hey!", "Hi there", "Good morning", "ok cool", "lol", "thanks"])
    def test_greetings_and_acks_skip(self, q):
        assert stm_skip_shape(q) is True

    @pytest.mark.parametrize("q", [
        "Biscuit figured out fetch today, sort of.",
        "hey can you check my calendar for tomorrow",
        "Hey so the doctor never responded and my dad is going the fraud route",
    ])
    def test_substantive_messages_run(self, q):
        assert stm_skip_shape(q) is False

    def test_small_talk_flag_skips(self):
        assert stm_skip_shape("anything at all here", is_small_talk=True) is True

    def test_gate_consults_helper(self):
        from core import context_pipeline
        src = inspect.getsource(context_pipeline.ContextPipeline)
        assert "stm_skip_shape(user_input" in src


class TestTemporalRecallNeedsRecallShape:
    @pytest.fixture(scope="class")
    def clf(self):
        return IntentClassifier()

    @pytest.mark.parametrize("q", [
        "Biscuit turned 2 last week, can't believe it.",
        "I went to the gym yesterday and it was fine",
        "Mochi did the crying thing again this afternoon, same as earlier today",
        "Saw Morgan the other day at the store",
    ])
    def test_narration_is_not_recall(self, clf, q):
        assert clf.classify(q).intent_type != IntentType.TEMPORAL_RECALL

    @pytest.mark.parametrize("q", [
        "what did we talk about last week?",
        "remember when I said yesterday that the doctor never responded",
        "How long have I been on this med",
        "did I mention the thing from the other day",
    ])
    def test_real_recall_still_fires(self, clf, q):
        r = clf.classify(q)
        assert r.intent_type == IntentType.TEMPORAL_RECALL and r.confidence >= 0.85


class TestOpenAIUnavailableSentinel:
    def test_prefix_registered(self):
        assert any(p.startswith("[OpenAI unavailable") for p in API_ERROR_PREFIXES)

    def test_retrieval_junk_filter_drops_it(self):
        assert is_junk_conversation_doc(response="[OpenAI unavailable] (RECENT CONVERSATION) n=10")
        assert not is_junk_conversation_doc(response="Mochi stole the phone again, classic.")


class TestFamilyRelationEventObjects:
    @pytest.mark.parametrize("rel,obj", [
        ("has_dad", "dad is picking up at 5:30"),
        ("has_dad", "dad is providing insurance settlement for next 3 months"),
        ("has_mom", "mom texted about the sandwich"),
        ("has_brother", "coming over tomorrow at 7pm"),
    ])
    def test_event_objects_are_junk(self, rel, obj):
        assert _is_junk_object(obj, rel) is True

    @pytest.mark.parametrize("rel,obj", [
        ("has_dad", "Alex"),
        ("has_brother", "Sam"),
        ("has_doctor", "no patient portal"),          # care-team status objects stay (2026-08-05)
        ("has_therapist", "doesn't respond to messages"),
        ("has_partner", "Casey"),
    ])
    def test_names_and_care_team_status_survive(self, rel, obj):
        assert _is_junk_object(obj, rel) is False


class TestRollupNotes:
    @pytest.mark.parametrize("title,expected", [
        ("Week 6 Feb 2026 Summary", True), ("March 2026 Summary", True),
        ("Week 12 Mar 2026 Summary", True), ("August 2026 Summary", True),
        ("4 25 26 Daily Note", False), ("Summary of my thesis", False), ("", False),
    ])
    def test_title_shape(self, title, expected):
        assert gk._is_rollup_note_title(title) is expected

    @pytest.mark.asyncio
    async def test_rollups_dropped_unless_retrospective(self):
        notes = [
            {"title": "March 2026 Summary", "content": "Study dominated the month. Sleep was variable. " * 6, "relevance_score": 0.9},
            {"title": "4 25 26 Daily Note", "content": "Biscuit chased a moth across the porch and then slept in the sun for an hour. " * 3, "relevance_score": 0.8},
        ]

        class _Mgr:
            async def get_notes(self, q, limit=10, include_images=False, max_images_per_note=3):
                return list(notes)

        g = gk.KnowledgeRetrievalMixin.__new__(gk.KnowledgeRetrievalMixin)
        g.obsidian_manager = _Mgr()
        g.memory_id_map = {}
        kept = await gk.KnowledgeRetrievalMixin.get_personal_notes(g, "Biscuit chased a moth", 5, allow_rollups=False)
        assert [n["title"] for n in kept] == ["4 25 26 Daily Note"]
        kept = await gk.KnowledgeRetrievalMixin.get_personal_notes(g, "what did March look like?", 5, allow_rollups=True)
        assert len(kept) == 2

    def test_builder_derives_allow_rollups(self):
        from core.prompt import builder
        src = inspect.getsource(builder)
        assert "allow_rollups=_notes_allow_rollups" in src


class TestThreadWordingGeneral:
    @pytest.mark.asyncio
    async def test_no_about_general(self):
        from tests.unit.test_process_user_query import _make_bfp_orch, _make_context
        orch = _make_bfp_orch()
        ctx = _make_context(thread_context={"thread_id": "t1", "thread_depth": 1, "thread_topic": "general"})
        _, system_prompt, _ = await orch.build_full_prompt(ctx, return_raw_context=True)
        assert "[THREAD CONTEXT]" in system_prompt
        assert "about general" not in system_prompt


# ── 7. planner embellishment guard ────────────────────────────────────────
class TestPlannerEmbellishmentGuard:
    SRC = "Biscuit turned 2 last week, can't believe it. Mochi did the crying thing again. My mom texted me."

    def test_invented_event_and_name_dropped(self):
        from core.response_planner import ResponsePlanner
        kept, dropped = ResponsePlanner.unsupported_key_points([
            "Daisy's progress with fetch and amusing antics",
            "Biscuit's recent birthday celebration",
            "Mochi's behavior with the ball",
            "Plan Morgan's visit",
        ], self.SRC)
        assert kept == ["Daisy's progress with fetch and amusing antics", "Mochi's behavior with the ball"]
        assert dropped == ["Biscuit's recent birthday celebration", "Plan Morgan's visit"]

    def test_licensed_event_kept(self):
        from core.response_planner import ResponsePlanner
        kept, dropped = ResponsePlanner.unsupported_key_points(
            ["Biscuit's birthday celebration"], "we celebrated Biscuit's birthday yesterday")
        assert kept and not dropped

    def test_never_empties_the_plan(self):
        from core.response_planner import ResponsePlanner
        kept, dropped = ResponsePlanner.unsupported_key_points(["Biscuit's birthday celebration"], "nothing")
        assert kept == ["Biscuit's birthday celebration"] and dropped == []

    def test_plan_model_records_dropped_points(self):
        from core.response_planner import ResponsePlan
        assert ResponsePlan(key_points=["a"]).dropped_points == []

    def test_prompt_carries_no_embellishment_rule(self):
        from core import response_planner
        src = inspect.getsource(response_planner)
        assert "detail the user did not state" in src and "is an age, " in src


# ── 8. negative mood-section notes need an emotional cue ──────────────────
class TestMoodSectionNotes:
    NEG = {"title": "8 12 26 Daily Note", "metadata": {"section": "Emotional State"},
           "content": "Luke feels like shit today, depression and severe sleep deprivation, anxiety about the mania fear.",
           "relevance_score": 0.74}
    POS = {"title": "8 13 26 Daily Note", "metadata": {"section": "Emotional State"},
           "content": "Luke felt great today, calm and energized after the gym and a good night of sleep.",
           "relevance_score": 0.7}
    PET = {"title": "4 25 26 Daily Note", "metadata": {"section": "Main Quest"},
           "content": "Biscuit chased a moth across the porch and then slept in the sun for an hour.",
           "relevance_score": 0.9}

    def test_classifier(self):
        assert gk._is_negative_mood_note(self.NEG) is True
        assert gk._is_negative_mood_note(self.POS) is False
        assert gk._is_negative_mood_note({**self.NEG, "metadata": {"section": "Side Quests"}}) is False

    @pytest.mark.asyncio
    async def test_gatherer_drops_negative_mood_only_when_disallowed(self):
        notes = [self.NEG, self.POS, self.PET]

        class _Mgr:
            async def get_notes(self, q, limit=10, include_images=False, max_images_per_note=3):
                return list(notes)

        g = gk.KnowledgeRetrievalMixin.__new__(gk.KnowledgeRetrievalMixin)
        g.obsidian_manager = _Mgr(); g.memory_id_map = {}
        kept = await gk.KnowledgeRetrievalMixin.get_personal_notes(g, "Biscuit chased a moth lol", 5, allow_mood_sections=False)
        assert [n["title"] for n in kept] == ["8 13 26 Daily Note", "4 25 26 Daily Note"]
        kept = await gk.KnowledgeRetrievalMixin.get_personal_notes(g, "I feel awful today", 5, allow_mood_sections=True)
        assert len(kept) == 3

    def test_builder_derives_mood_flag(self):
        from core.prompt import builder
        src = inspect.getsource(builder)
        assert "allow_mood_sections=_notes_allow_mood" in src
        assert "_is_heavy_topic_heuristic(user_input)" in src
