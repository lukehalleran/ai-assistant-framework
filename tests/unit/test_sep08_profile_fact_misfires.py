"""2026-09-08 morning dump: two wrong profile facts from the 09-07 shutdown
LLM extractions, plus STM token glue.

  took=melatonin            <- "So ready to take melatonin … but I think it is
                               early for that still" (prospective, not done;
                               the medication class again)
  gym_schedule=Fridays 8-9  <- the office-hours CALENDAR request, joined on the
                               soft-wrapped fragment "▎ 11, weekly through the
                               end of the semester, zoom" (no gym word anywhere)
  STM novelty note          <- "930and, 3or, 330but": number+unit shape glued a
                               clock shorthand to the conjunction after it
"""
from __future__ import annotations

from datetime import datetime

import pytest

from memory import fact_source as fs
from memory.fact_source import (
    COMPLETION_RELATIONS, classify_claim_time, completion_claim_prospective,
    find_supporting_user_span,
)
from utils.query_checker import extract_data_tokens

T_MELATONIN = ("Is okay, well try again in a min. Yeah today is nuts I don't remember being "
               "both this tired and also capable of somewhat functioning. Fell down for another "
               "hour in bed felt unavoidable. So ready to take melatonin and stuff but I think it "
               "is early for that still idk. I want to get up as early as possible and reasonable "
               "tomorrow because if I can get in a workout before therapy appointment then I can "
               "justify it, but otherwise I probably need to focus on the assignment")
T_CALENDAR = ("add the mgt office hours sessions to my google\n  ▎ calander, fridays 8 to 9 pm "
              "central starting sept\n  ▎ 11, weekly through the end of the semester, zoom\n"
              "  ▎ link https://example-univ.zoom.us/j/00000000001")
T_TOOK = "Took melatonin at 9 and was out by 10. Ready to take the day on now."


def _msgs(*texts):
    return [{"id": f"2026-09-07T18:{i:02d}:00", "query": t, "role": "user"}
            for i, t in enumerate(texts)]


def _find(rel, obj, *texts):
    return find_supporting_user_span({"subject": "user", "relation": rel, "object": obj}, _msgs(*texts))


# ── 1. prospective clause cannot support a completion relation ─────────────
class TestProspectiveCompletion:
    def test_live_melatonin_span_rejected(self):
        assert _find("took", "melatonin", T_MELATONIN) is None

    def test_real_past_tense_span_still_supports(self):
        ev = _find("took", "melatonin", T_MELATONIN, T_TOOK)
        assert ev is not None and ev.text.startswith("Took melatonin at 9")

    def test_plan_relation_keeps_the_prospective_span(self):
        ev = _find("plans_to", "take melatonin", T_MELATONIN)
        assert ev is not None and ev.claim_kind == "plan"

    @pytest.mark.parametrize("clause", [
        "So ready to take melatonin and stuff",
        "about to send the email",
        "I want to go to the gym later",
        "gonna eat after therapy",
        "I'll call the pharmacy",
        "should take the 5 mg soon",
    ])
    def test_prospective_shapes(self, clause):
        assert completion_claim_prospective("took", clause)
        assert completion_claim_prospective("sent", clause)

    @pytest.mark.parametrize("clause", [
        "took the 5 mg at 11",
        "Had melatonin and was out by 10",
        "I was about to skip it but took it anyway",   # past-tense cue present
    ])
    def test_completed_shapes_pass(self, clause):
        assert not completion_claim_prospective("took", clause)

    def test_non_completion_relations_never_fire(self):
        assert not completion_claim_prospective("wants_to", "ready to take melatonin")
        assert not completion_claim_prospective("medication", "ready to take melatonin")
        assert "took" in COMPLETION_RELATIONS

    def test_claim_time_classifies_readiness_as_plan(self):
        ct = classify_claim_time("So ready to take melatonin and stuff but I think it is early for that",
                                 observed_at=datetime(2026, 9, 7, 18, 24))
        assert ct.kind == "plan"
        assert classify_claim_time("Took melatonin last night", observed_at=datetime(2026, 9, 8)).kind == "event"


# ── 2. fitness-schedule relations need an exercise cue ─────────────────────
class TestFitnessCue:
    def test_live_calendar_request_cannot_be_a_gym_schedule(self):
        obj = "Fridays 8 to 9 PM Central starting September 11, weekly through the end of the semester"
        assert _find("gym_schedule", obj, T_MELATONIN, T_CALENDAR) is None

    def test_real_gym_schedule_supported(self):
        ev = _find("gym_schedule", "Fridays 8 to 9 PM", "I hit the gym fridays 8 to 9 pm, weekly.")
        assert ev is not None

    def test_workout_habit_with_workout_word(self):
        ev = _find("workout_habit", "wants to get in a workout before therapy appointment", T_MELATONIN)
        assert ev is not None and ev.claim_kind == "plan"

    def test_typo_calendar_request_still_supports_the_intent_relation(self):
        ev = _find("wants_to", "add MGT office hours sessions to Google Calendar", T_CALENDAR)
        assert ev is not None


# ── 3. soft-wrapped continuation bars are one sentence ─────────────────────
class TestContinuationBars:
    def test_bar_lines_join(self):
        spans = list(fs._claim_spans(T_CALENDAR))
        assert len(spans) == 1
        assert "calander, fridays 8 to 9 pm central starting sept 11, weekly" in spans[0]

    def test_blockquote_still_excluded(self):
        spans = list(fs._claim_spans("my own line\n> quoted thing\nanother line"))
        assert spans == ["my own line", "another line"]

    def test_plain_multiline_unchanged(self):
        assert list(fs._claim_spans("first line.\nsecond line.")) == ["first line.", "second line."]


# ── 4. data tokens: number + function word is not a unit ───────────────────
class TestDataTokenGlue:
    def test_live_sleep_report_yields_no_glued_tokens(self):
        toks = extract_data_tokens("Not really I think the alarm got me at 930 and I got out of bed "
                                   "around 1020. I think I was asleep by 3 or 330 but can't be 100%")
        assert not any(t.endswith(("and", "or", "but")) for t in toks), toks

    def test_real_units_survive(self):
        toks = extract_data_tokens("took 5 mg at 10 AM, slept 6 hours, day 9, 2 16 oz beers, 3 miles")
        for want in ("5mg", "10:00", "6hours", "day9", "16oz", "3miles"):
            assert want in toks
