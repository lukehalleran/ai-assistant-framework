"""2026-09-05 evening two-turn audit (16:06 "haven't heard from Rowan" /
16:27 "functional but lower energy") — regression tests for every code fix.

Fixture text is either synthetic or the assigned topic/relation strings from
the live debug record; no personal data beyond a first name already used as a
test fixture elsewhere in the suite.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. Pacing: "Time since last session" anchors on the previous session's
#    last MESSAGE, not the process shutdown/idle stamp.
# ---------------------------------------------------------------------------

class TestTimeSinceLastSessionAnchor:
    def _tm(self, tmp_path):
        from utils.time_manager import TimeManager
        with patch.object(TimeManager, "_load_last_session_time", return_value=None), \
             patch.object(TimeManager, "_load_active_days", return_value=set()):
            tm = TimeManager(time_file=str(tmp_path / "t.json"))
        tm._save_last_session_time = lambda: None  # never touch data/
        tm._register_active_day = lambda *_a, **_k: None
        return tm

    def test_new_session_measures_from_last_message_not_restart(self, tmp_path):
        tm = self._tm(tmp_path)
        now = datetime.now()
        tm.last_query_time = now - timedelta(hours=2, minutes=21)   # 13:45
        tm.last_session_end_time = now - timedelta(minutes=30)      # 15:37 restart
        tm.mark_query_time()                                         # 16:06 first message
        gap = tm.elapsed_since_last_session()
        assert gap.startswith("2 h"), gap
        assert tm.time_since_previous_message().startswith("N/A")

    def test_anchor_holds_for_the_rest_of_the_session(self, tmp_path):
        tm = self._tm(tmp_path)
        now = datetime.now()
        tm.last_query_time = now - timedelta(hours=2, minutes=21)
        tm.last_session_end_time = now - timedelta(minutes=30)
        tm.mark_query_time()
        tm.mark_query_time()  # second message, same session
        assert tm.elapsed_since_last_session().startswith("2 h")
        assert not tm.time_since_previous_message().startswith("N/A")

    def test_init_seeds_anchor_from_persisted_times(self, tmp_path):
        from utils.time_manager import TimeManager
        now = datetime.now()
        with patch.object(TimeManager, "_load_last_session_time", return_value=now - timedelta(minutes=30)), \
             patch.object(TimeManager, "_load_last_query_time", return_value=now - timedelta(hours=3)), \
             patch.object(TimeManager, "_load_active_days", return_value=set()):
            tm = TimeManager(time_file=str(tmp_path / "t.json"))
        assert tm.previous_session_last_query_time == now - timedelta(hours=3)
        assert tm.elapsed_since_last_session().startswith("3 h")

    def test_first_session_is_still_na(self, tmp_path):
        tm = self._tm(tmp_path)
        assert tm.elapsed_since_last_session() == "N/A (first session)"


# ---------------------------------------------------------------------------
# 2. [THREAD CONTEXT]: a session-start turn never inherits the previous
#    thread's heavy flag.
# ---------------------------------------------------------------------------

class TestSessionStartNeverCarriesHeavyFlag:
    @pytest.mark.asyncio
    async def test_new_session_line_without_heavy_line(self):
        from tests.unit.test_process_user_query import _make_bfp_orch, _make_context
        thread_ctx = {
            "thread_id": "t9",
            "thread_depth": 1,
            "thread_topic": "Homework Part 2 Scope",
            "is_heavy_topic": True,
            "last_timestamp": datetime.now().isoformat(),
        }
        ctx = _make_context(
            thread_context=thread_ctx,
            original_query="Haven't heard back yet. Four commits landed and walking to the gym after breakfast",
            primary_topic="Waiting For A Friend",
            stm_summary=None,
        )
        tm = MagicMock()
        tm.time_since_previous_message = MagicMock(return_value="N/A (first message in session)")
        orch = _make_bfp_orch(time_manager=tm)
        _, system_prompt, _ = await orch.build_full_prompt(ctx, return_raw_context=True)
        assert "New session." in system_prompt
        assert "sensitive/heavy topic" not in system_prompt


# ---------------------------------------------------------------------------
# 3. STM novelty: "I'm" is not a named entity.
# ---------------------------------------------------------------------------

class TestFirstPersonContractionsAreNotNames:
    def test_i_contractions_excluded(self):
        from utils.query_checker import extract_rare_proper_nouns
        q = ("Well haven't heard from Rowan yet which is a bummer. Might be worth it to call "
             "but I'm pretty sure if he is not available today he'll get at me tomorrow. "
             "I'll text again; I've got time and I'd rather wait.")
        assert extract_rare_proper_nouns(q) == ["Rowan"]

    def test_curly_apostrophe(self):
        from utils.query_checker import extract_rare_proper_nouns
        assert extract_rare_proper_nouns("yeah I’m fine, I’ll manage, I’ve eaten") == []

    def test_novelty_override_skips_contraction(self):
        from core.stm_analyzer import novel_named_entities
        assert novel_named_entities("Might call but I'm pretty sure he'll reply", "some window text") == []


# ---------------------------------------------------------------------------
# 4. Knowledge graph: temporal deictics are never entities, and edges that
#    touch a temporal node are suppressed at read time.
# ---------------------------------------------------------------------------

class TestTemporalDeicticsNeverEntities:
    @pytest.mark.parametrize("name", [
        "today", "Tomorrow", "yesterday", "tonight", "recently", "soon",
        "on thursday", "on_thursday", "next monday", "this week", "last night",
        "weekend", "September", "a few days ago", "couple of weeks",
    ])
    def test_is_temporal_deictic(self, name):
        from memory.graph_utils import is_junk_entity, is_temporal_deictic
        assert is_temporal_deictic(name)
        assert is_junk_entity(name)

    @pytest.mark.parametrize("name", ["rowan", "biscuit", "planet fitness", "georgia tech", "sun", "march madness"])
    def test_real_entities_untouched(self, name):
        from memory.graph_utils import is_temporal_deictic
        assert not is_temporal_deictic(name)

    def test_extract_graph_entities_drops_temporal_seeds(self):
        from memory.graph_utils import extract_graph_entities
        known = {"today": "today", "tomorrow": "tomorrow", "rowan": "rowan", "gym": "gym"}
        resolver = MagicMock()
        resolver.resolve = lambda phrase: known.get(phrase)
        q = "haven't heard from Rowan yet; if he is not available today he'll get at me tomorrow, walking to the gym"
        seeds = extract_graph_entities(q, resolver)
        assert "today" not in seeds and "tomorrow" not in seeds
        assert "rowan" in seeds

    def test_temporal_seed_yields_no_context(self, tmp_path):
        from memory.graph_memory import GraphMemory
        from memory.graph_models import GraphEdge, GraphNode
        gm = GraphMemory(persist_path=str(tmp_path / "g.json"))
        gm.add_entity(GraphNode(entity_id="user", display_name="User", entity_type="person"))
        gm.add_entity(GraphNode(entity_id="today", display_name="today", entity_type="other"))
        gm.add_entity(GraphNode(entity_id="biscuit", display_name="Biscuit", entity_type="pet"))
        gm.add_relation(GraphEdge(source_id="user", target_id="today", relation="dad", weight=1.0))
        gm.add_relation(GraphEdge(source_id="user", target_id="biscuit", relation="has_cat", weight=1.0))
        assert gm.get_context_sentences("today") == []
        user_ctx = " | ".join(gm.get_context_sentences("user"))
        assert "today" not in user_ctx and "Biscuit" in user_ctx

    def test_edge_to_temporal_node_is_suppressed(self, tmp_path):
        from memory.graph_memory import GraphMemory
        from memory.graph_models import GraphEdge
        gm = GraphMemory(persist_path=str(tmp_path / "g.json"))
        bad = GraphEdge(source_id="user", target_id="today", relation="dad", weight=1.0)
        bad2 = GraphEdge(source_id="rowan", target_id="on_thursday", relation="texted", weight=1.0)
        good = GraphEdge(source_id="user", target_id="rowan", relation="friend_of", weight=1.0)
        assert gm.edge_is_suppressed(bad)
        assert gm.edge_is_suppressed(bad2)
        assert not gm.edge_is_suppressed(good)


# ---------------------------------------------------------------------------
# 5. Fact junk-object rules.
# ---------------------------------------------------------------------------

class TestJunkObjectRules:
    def test_weekday_objects_junk_outside_schedule_relations(self):
        from memory.fact_extractor import _is_junk_object
        assert _is_junk_object("on thursday", "texted")
        assert _is_junk_object("tuesday", "had_off_day_on")
        assert not _is_junk_object("thursday", "day_off")  # schedule relation exempt

    def test_demonstrative_generic_object_junk(self):
        from memory.fact_extractor import _is_junk_object
        assert _is_junk_object("this assistant", "works_on")
        assert _is_junk_object("that thing", "likes")
        assert not _is_junk_object("Daemon", "works_on")

    def test_care_team_has_relation_rejects_clause_objects(self):
        from memory.fact_extractor import _is_junk_object
        assert _is_junk_object("Rowan is cautious about drinking due to past accident", "has_doctor")
        assert not _is_junk_object("no patient portal", "has_doctor")   # status object stays allowed
        assert not _is_junk_object("Dr. Patel", "has_doctor")
        assert not _is_junk_object("not responsive", "doctor_communication")


# ---------------------------------------------------------------------------
# 6. Provenance boundary (fact_source).
# ---------------------------------------------------------------------------

_MEDS_TURN = ("It's on the news. Ok so I guess I took meds at 9, but I recall being up til 5, "
              "despite zero drinking at all yesterday, and normal meds amt at like 1015.")
_FRIEND_TURN = ("And well so I don't have a car right now. Rowan got in a dangerous accident in "
                "like 2014 due to drinking and driving and he is understandably very cautious about "
                "drinking literally anything if driving involved")
_ADVISOR_TURN = ("I got an email from my advisor in my outlook inbox a couple of days ago about the "
                 "project timeline for this term. Can you read the last email I received from them?")


class TestProvenanceBoundary:
    def _msgs(self, *texts):
        return [{"query": t, "timestamp": f"2026-09-05T12:{i:02d}:00"} for i, t in enumerate(texts)]

    def test_has_doctor_needs_a_clinician_cue(self):
        from memory.fact_source import find_supporting_user_span
        triple = {"subject": "user", "relation": "has_doctor",
                  "object": "Rowan is cautious about drinking due to past accident"}
        assert find_supporting_user_span(triple, self._msgs(_MEDS_TURN, _FRIEND_TURN)) is None
        ok = {"subject": "user", "relation": "has_doctor", "object": "Dr. Patel"}
        assert find_supporting_user_span(ok, self._msgs("My doctor Dr. Patel finally called back")) is not None

    def test_doctor_communication_rejects_advisor_email(self):
        from memory.fact_source import find_supporting_user_span
        triple = {"subject": "user", "relation": "doctor_communication",
                  "object": "received email from advisor about project timeline"}
        assert find_supporting_user_span(triple, self._msgs(_ADVISOR_TURN)) is None
        ok = {"subject": "user", "relation": "doctor_communication", "object": "no patient portal"}
        assert find_supporting_user_span(ok, self._msgs("My psychiatrist has no patient portal at all")) is not None

    def test_works_on_needs_a_project_cue(self):
        from memory.fact_source import find_supporting_user_span
        triple = {"subject": "user", "relation": "works_on", "object": "this assistant"}
        hw = "Attached: HW1 Part 2 PDF, its data file, and two lecture transcripts for this week. Do not solve anything."
        assert find_supporting_user_span(triple, self._msgs(hw)) is None
        ok = {"subject": "user", "relation": "works_on", "object": "Daemon"}
        assert find_supporting_user_span(ok, self._msgs("Spent the afternoon working on Daemon fixes")) is not None

    def test_single_generic_token_overlap_is_not_evidence(self):
        from memory.fact_source import find_supporting_user_span
        triple = {"subject": "user", "relation": "hobby", "object": "casual social drinking with friends"}
        # "drinking" alone (one shared token of four) must not anchor the claim.
        assert find_supporting_user_span(triple, self._msgs(_MEDS_TURN)) is None
        # Two shared content tokens do.
        assert find_supporting_user_span(
            triple, self._msgs("I like casual drinking with friends on weekends")) is not None

    def test_user_text_preferred_over_merged_attachment_query(self):
        from memory.fact_source import find_supporting_user_span
        merged = "Attached: my notes. Scope check only.\n\n[notes.txt]\nI live in Denver and moved here in June."
        triple = {"subject": "user", "relation": "lives_in", "object": "Denver"}
        with_raw = [{"query": merged, "user_text": "Attached: my notes. Scope check only.", "timestamp": "2026-09-05T13:45:00"}]
        assert find_supporting_user_span(triple, with_raw) is None
        without_raw = [{"query": merged, "timestamp": "2026-09-05T13:45:00"}]
        assert find_supporting_user_span(triple, without_raw) is not None  # documents the pre-fix hole


# ---------------------------------------------------------------------------
# 7. Corpus stores the raw user text beside a merged attachment query.
# ---------------------------------------------------------------------------

class TestCorpusUserText:
    def test_user_text_stored_only_when_it_differs(self, tmp_path):
        from memory.corpus_manager import CorpusManager
        cm = CorpusManager(corpus_file=str(tmp_path / "corpus.json"))
        cm.add_entry("Scope check only.\n\n<270K chars of transcripts>", "ok", user_text="Scope check only.")
        cm.add_entry("plain message", "ok", user_text="plain message")
        cm.add_entry("no attachment", "ok")
        assert cm.corpus[0]["user_text"] == "Scope check only."
        assert "user_text" not in cm.corpus[1]
        assert "user_text" not in cm.corpus[2]

    def test_llm_extractor_prompt_prefers_user_text(self):
        from memory.llm_fact_extractor import LLMFactExtractor
        ex = LLMFactExtractor(MagicMock())
        prompt = ex._build_prompt([{"query": "raw\n\nTRANSCRIPT SECRET", "user_text": "raw", "turn_id": "2026-09-05T13:45:00"}])
        assert "TRANSCRIPT SECRET" not in prompt
        assert "User: raw" in prompt


# ---------------------------------------------------------------------------
# 8. Streak claims: extraction, projection, ledger, stale-count removal.
# ---------------------------------------------------------------------------

class TestStreakClaims:
    def test_extracts_day_n_in_a_row(self):
        from utils.streak_claims import extract_streak_claims
        c = extract_streak_claims('Today is day 6 in a row that I am "normal" I think? My sleep was good.', date(2026, 9, 2))
        assert [x.count for x in c] == [6]
        assert c[0].stated_on == date(2026, 9, 2)

    @pytest.mark.parametrize("text,expected", [
        ("i think today is day 8 not 6.", [8]),
        ("Day 8 today and feeling fine", [8]),
        ("six days in a row of feeling ok", [6]),
        ("my 4th straight day at the gym", [4]),
        ("8-day streak", [8]),
        ("the assignment is due in 8 days", []),
        ("I was sick for 3 days last week", []),
        ("stable for six days", []),  # plain duration → build_temporal_claim_audit's job
        ("Draft email: I've been functional 10 days in a row now", []),
    ])
    def test_shapes(self, text, expected):
        from utils.streak_claims import extract_streak_claims
        assert [c.count for c in extract_streak_claims(text, date(2026, 9, 5))] == expected

    def test_projection_and_newest_wins(self):
        from utils.streak_claims import current_streak_count, streak_ledger
        stmts = [
            {"query": "Today is day 6 in a row that I am normal", "timestamp": "2026-09-02T16:00:00"},
            {"query": "no counts here", "response": "six solid days", "timestamp": "2026-09-04T19:57:00"},
        ]
        led = streak_ledger(stmts, as_of=date(2026, 9, 5))
        assert [c.count for c in led] == [6]
        count, newest = current_streak_count(led, date(2026, 9, 5))
        assert count == 9 and newest.stated_on == date(2026, 9, 2)
        stmts.append({"query": "today is day 8", "timestamp": "2026-09-05T18:00:00"})
        led = streak_ledger(stmts, as_of=date(2026, 9, 5))
        assert current_streak_count(led, date(2026, 9, 5))[0] == 8

    def test_activity_streaks_are_separate_from_the_state_count(self):
        from utils.streak_claims import current_streak_count, remove_stale_streak_claims, streak_ledger, streak_ledger_block
        stmts = [
            {"query": "Today is day 6 in a row that I am normal", "timestamp": "2026-09-02T16:00:00"},
            {"query": "I worked out 3 days in a row and took yesterday off", "timestamp": "2026-09-03T16:00:00"},
        ]
        led = streak_ledger(stmts, as_of=date(2026, 9, 5))
        assert [c.kind for c in led] == ["state", "activity"]
        assert current_streak_count(led, date(2026, 9, 5))[0] == 9      # the newer activity claim does not win
        block = streak_ledger_block(led, date(2026, 9, 5))
        assert "activity streak (separate count" in block
        text = "He has worked out three days in a row. He is six days into a stable streak."
        revised, removed = remove_stale_streak_claims(text, led, date(2026, 9, 5))
        assert len(removed) == 1
        assert "worked out three days in a row" in revised.split("[CAUTION")[0]

    def test_ledger_ignores_old_and_assistant_text(self):
        from utils.streak_claims import streak_ledger
        stmts = [
            {"query": "day 20 in a row", "timestamp": "2026-08-01T10:00:00"},
            {"query": "nothing", "response": "you are on day 30 in a row!", "timestamp": "2026-09-04T10:00:00"},
        ]
        assert streak_ledger(stmts, as_of=date(2026, 9, 5)) == []

    def test_block_text(self):
        from utils.streak_claims import streak_ledger, streak_ledger_block
        led = streak_ledger([{"query": "Today is day 6 in a row", "timestamp": "2026-09-02T16:00:00"}], as_of=date(2026, 9, 5))
        block = streak_ledger_block(led, date(2026, 9, 5))
        assert "STREAK LEDGER" in block
        assert "2026-09-02: the user counted day 6" in block
        assert "2026-09-05 is day 9" in block
        assert "CURRENT COUNT as of 2026-09-05: day 9" in block
        assert streak_ledger_block([], date(2026, 9, 5)) == ""

    def test_stale_count_sentences_removed_dated_mention_kept(self):
        from utils.streak_claims import remove_stale_streak_claims, streak_ledger
        led = streak_ledger([{"query": "Today is day 6 in a row", "timestamp": "2026-09-02T16:00:00"}], as_of=date(2026, 9, 5))
        narrative = (
            "Luke is now six days into his longest stable functional streak since June.\n\n"
            "## Emotional Trajectory\n"
            "Luke has maintained stable mood for six consecutive days (August 31–September 5). "
            "On September 2 he counted day 6. He was sick for 3 days in July. "
            "Nine days into the streak now, he feels steadier."
        )
        revised, removed = remove_stale_streak_claims(narrative, led, date(2026, 9, 5))
        assert len(removed) == 2
        body = revised.split("[CAUTION")[0]
        assert "six days into" not in body
        assert "six consecutive days" not in body
        assert "On September 2 he counted day 6." in body         # dated → kept
        assert "sick for 3 days in July" in body                  # plain duration → kept
        assert "Nine days into the streak" in body                # matches current → kept
        assert "[CAUTION:" in revised and "day 9" in revised

    def test_no_ledger_never_touches_text(self):
        from utils.streak_claims import remove_stale_streak_claims
        text = "Luke is six days into a streak."
        assert remove_stale_streak_claims(text, [], date(2026, 9, 5)) == (text, [])


# ---------------------------------------------------------------------------
# 9. Narrative generator wiring: ledger in the prompt, stale count removed,
#    truncated tail trimmed, larger token cap.
# ---------------------------------------------------------------------------

class TestNarrativeWiring:
    @pytest.mark.asyncio
    async def test_generate_narrative_uses_ledger_and_trims(self):
        from memory.memory_consolidator import MemoryConsolidator
        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value=(
            "# Current Life State\n\n## Current Chapter\n"
            "The user is now six days into his longest stable functional streak since June.\n\n"
            "## Emotional Trajectory\n"
            "The user has maintained stable mood since the crisis began in June. September"
        ))
        profile = MagicMock()
        profile.get_current_view = MagicMock(return_value={})
        cons = MemoryConsolidator(mm, user_profile=profile)
        cons._read_obsidian_monthly_summaries = lambda limit: []
        cons._read_obsidian_weekly_summaries = lambda limit: []
        cons._read_obsidian_daily_notes = lambda limit: [{"content": "note", "timestamp": "2026-09-05"}]
        three_days_ago = datetime.now() - timedelta(days=3)
        stmts = [{"query": "Today is day 6 in a row that I am normal", "timestamp": three_days_ago.isoformat()}]

        out = await cons.generate_narrative_context(user_statements=stmts)

        prompt = mm.generate_once.call_args.args[0]
        assert "STREAK LEDGER" in prompt and "day 9" in prompt
        assert mm.generate_once.call_args.kwargs["max_tokens"] == 700
        body = out.split("[CAUTION")[0]
        assert "six days into" not in body
        assert "[CAUTION:" in out and "day 9" in out
        assert "September" not in body
        assert "since the crisis began in June." in body

    def test_trim_truncated_tail(self):
        from memory.memory_consolidator import _trim_truncated_tail
        assert _trim_truncated_tail("A full sentence.\n\n## Heading\nAnother one. Septem") == "A full sentence.\n\n## Heading\nAnother one."
        assert _trim_truncated_tail("A full sentence.\n\n## Heading\nSeptember") == "A full sentence."
        assert _trim_truncated_tail("- **Plans**: fine.\n- Done.") == "- **Plans**: fine.\n- Done."


# ---------------------------------------------------------------------------
# 10. Daily-note audit carries the user's own streak count, from user_text.
# ---------------------------------------------------------------------------

class TestDailyNoteStreakAudit:
    def test_streak_line_present(self):
        from utils.daily_notes_generator import build_temporal_claim_audit
        audit = build_temporal_claim_audit([
            {"timestamp": datetime(2026, 9, 2, 16, 0), "query": 'Today is day 6 in a row that I am "normal" I think?'},
        ])
        assert "[streak-count]" in audit and "day 6" in audit
        assert "user's own count ON THIS DATE" in audit

    def test_audit_reads_user_text_not_attachment(self):
        from utils.daily_notes_generator import build_temporal_claim_audit
        audit = build_temporal_claim_audit([
            {"timestamp": datetime(2026, 9, 5, 13, 45),
             "query": "Scope check only.\n\nTRANSCRIPT: I have been stable for 2 weeks now",
             "user_text": "Scope check only."},
        ])
        assert "2 weeks" not in audit


# ---------------------------------------------------------------------------
# 11. Web trigger: one LLM classification per turn (key normalization +
#     in-flight sharing).
# ---------------------------------------------------------------------------

class TestTriggerCacheNormalization:
    def test_policy_and_credit_buckets(self):
        import utils.web_search_trigger as wst
        assert wst._crisis_policy_key(None) == wst._crisis_policy_key("CONVERSATIONAL") == "OPEN"
        assert wst._crisis_policy_key("HIGH") == wst._crisis_policy_key("medium") == "SUPPRESSED"
        assert wst._credits_bucket(100) == wst._credits_bucket(87.3) == "ok"
        assert wst._credits_bucket(3) == "critical" and wst._credits_bucket(0) == "none"

    def _neutral(self):
        from utils.web_search_trigger import WebSearchDecision, WebSearchDepth
        return WebSearchDecision(
            should_search=True, depth=WebSearchDepth.STANDARD, confidence=0.5,
            reason="neutral", matched_keywords=["news"], matched_patterns=[],
        )

    @pytest.mark.asyncio
    async def test_gate_and_gatherer_share_one_call(self):
        import utils.web_search_trigger as wst
        wst._llm_trigger_cache.clear()
        wst._llm_trigger_inflight.clear()
        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value=(
            '{"should_search": false, "confidence": 0.9, "reason": "personal", '
            '"search_terms": [], "search_depth": "quick", "num_searches": 0}'))
        q = "Well haven't heard back yet, might call, 4 commits landed"
        with patch.object(wst, "should_search_heuristic", return_value=self._neutral()), \
             patch.object(wst, "LLM_FIRST_ENABLED", True):
            a = await wst.analyze_for_web_search_llm(query=q, model_manager=mm, conversation_context="ctx")
            b = await wst.analyze_for_web_search_llm(
                query=q, model_manager=mm, conversation_context="ctx",
                crisis_level="CONVERSATIONAL", remaining_credits=87.3, web_search_enabled=True)
        assert mm.generate_once.call_count == 1
        assert a.should_search is False and b.should_search is False

    @pytest.mark.asyncio
    async def test_concurrent_same_key_calls_share_in_flight_task(self):
        import utils.web_search_trigger as wst
        wst._llm_trigger_cache.clear()
        wst._llm_trigger_inflight.clear()

        async def slow(*_a, **_k):
            await asyncio.sleep(0.05)
            return ('{"should_search": false, "confidence": 0.9, "reason": "personal", '
                    '"search_terms": [], "search_depth": "quick", "num_searches": 0}')

        mm = MagicMock()
        mm.generate_once = AsyncMock(side_effect=slow)
        q = "functional but lower energy than I would like"
        with patch.object(wst, "should_search_heuristic", return_value=self._neutral()), \
             patch.object(wst, "LLM_FIRST_ENABLED", True):
            a, b = await asyncio.gather(
                wst.analyze_for_web_search_llm(query=q, model_manager=mm, conversation_context="ctx"),
                wst.analyze_for_web_search_llm(query=q, model_manager=mm, conversation_context="ctx",
                                               crisis_level="CONVERSATIONAL", remaining_credits=42.0),
            )
        assert mm.generate_once.call_count == 1
        assert a.should_search is False and b.should_search is False
        assert wst._llm_trigger_inflight == {}

    @pytest.mark.asyncio
    async def test_suppressed_policy_still_separate(self):
        import utils.web_search_trigger as wst
        wst._llm_trigger_cache.clear()
        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value=(
            '{"should_search": true, "confidence": 0.9, "reason": "x", '
            '"search_terms": ["t"], "search_depth": "quick", "num_searches": 1}'))
        with patch.object(wst, "should_search_heuristic", return_value=self._neutral()), \
             patch.object(wst, "LLM_FIRST_ENABLED", True):
            open_ = await wst.analyze_for_web_search_llm(query="latest news", model_manager=mm, conversation_context="c")
            high = await wst.analyze_for_web_search_llm(query="latest news", model_manager=mm,
                                                        conversation_context="c", crisis_level="HIGH")
        assert open_.should_search is True
        assert high.should_search is False


# ---------------------------------------------------------------------------
# 12. [RECENT CONVERSATION] session headers carry a time range so two
#     same-day sessions stay distinguishable.
# ---------------------------------------------------------------------------

class TestSessionHeaderSpans:
    def test_same_day_sessions_get_time_ranges(self):
        from core.prompt.formatter import _format_session_header, _session_spans
        today = datetime.now().replace(hour=16, minute=6, second=0, microsecond=0)
        recent = [  # newest-first, as the gatherer returns
            {"query": "a", "response": "b", "timestamp": today.isoformat()},
            {"query": "c", "response": "d", "timestamp": today.replace(hour=11, minute=0).isoformat()},
            {"query": "e", "response": "f", "timestamp": today.replace(hour=10, minute=30).isoformat()},
        ]
        spans = _session_spans(recent)
        assert set(spans) == {1, 2}
        h1 = _format_session_header(today, span=spans[1])
        h2 = _format_session_header(today.replace(hour=11), span=spans[2])
        assert h1.endswith(", 16:06 ---")
        assert h2.endswith(", 10:30–11:00 ---")
        assert h1 != h2 and "--- Session: Today" in h1

    def test_header_without_span_unchanged(self):
        from core.prompt.formatter import _format_session_header
        ts = datetime(2026, 5, 17, 10, 0)
        assert _format_session_header(ts) == "--- Session: Sun, May 17 ---"
