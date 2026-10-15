"""Thread quick-resolution by identity+status; self-set targets are not due dates.

2026-10-14 incident (synthetic reproduction): a vent plus a pasted progress
report resolved three open threads at once because a completion word
("done") anywhere in the message plus stop-ish keyword overlap counted.
Now each CLAUSE must carry a non-negated completion cue AND name the
thread's own object head (shared judge: completed_plan_claims).

class: BC-26, BC-04, BC-51
"""
import pytest
from unittest.mock import MagicMock

from memory.thread_store import OpenThread, check_quick_resolutions, thread_object_head


def _t(tid, topic, summary="", **kw):
    return OpenThread(thread_id=tid, topic=topic, summary=summary, **kw)


T1 = _t("t1", "Finish sim HW 4 today", "User said they have to finish problems 4 and 5 of sim HW 4 today")
T2 = _t("t2", "STAT 5100 incomplete work due end of semester",
        "User has incomplete work in STAT 5100 that is all due at the end of the semester in December")
T3 = _t("t3", "BIO class deferred to mid next week",
        "The BIO class is deferred to mid next week; user has not yet opened it")

INCIDENT = (
    "I didnt even get more problems done just fixing and finishing old ones, so I am done with those: "
    "Problems 1–3 are done and checked. Problem 4 still needs the final plot. "
    "Problems 6-13 are not started. Target finishing HW4 this week, "
    "Hold the December completion deadline for STAT 5100, based on the end of the fall semester, "
    "that is all I can do. "
    "I don't think I need to open that til mid next week."
)


class TestIncident:
    def test_incident_resolves_nothing(self):
        assert check_quick_resolutions(INCIDENT, [T1, T2, T3]) == []


class TestPositive:
    def test_finished_hw4_resolves_t1(self):
        assert check_quick_resolutions("I finished HW 4 last night", [T1, T2, T3]) == ["t1"]

    def test_turned_in_incomplete_work(self):
        assert check_quick_resolutions(
            "turned in the incomplete work today", [T1, T2, T3]) == ["t2"]

    def test_did_the_email(self):
        email = _t("te", "Send email response", "User needs to send an email response today")
        assert check_quick_resolutions("I did the email yesterday", [email]) == ["te"]

    def test_other_homework_number_does_not_resolve(self):
        hw7 = _t("h7", "Submit homework 7")
        assert check_quick_resolutions("just submitted homework 6", [hw7]) == []
        assert check_quick_resolutions("just submitted homework 7", [hw7]) == ["h7"]


class TestNegation:
    def test_havent_finished_yet(self):
        assert check_quick_resolutions("I haven't finished HW 4 yet", [T1]) == []


class TestObjectHeadTable:
    @pytest.mark.parametrize("topic,heads", [
        ("Finish sim HW 4 today", {"hw", "4"}),
        ("STAT 5100 incomplete work due end of semester", {"stat", "work"}),
        ("6644 incomplete work due end of semester", {"work"}),
        ("MGT class deferred to mid next week", {"class"}),
        ("BIO class deferred to mid next week", {"class"}),
        ("Email Professor Hale about the extension", {"extension"}),
        ("Submit homework 7", {"homework"}),
        ("Send email response", {"response", "email"}),
        ("Homework 6 submission", {"homework"}),
    ])
    def test_heads(self, topic, heads):
        assert thread_object_head(topic)[0] in heads

    def test_digits_from_topic(self):
        from utils import completed_plan_claims as cpc
        assert "4" in cpc._digit_tokens(cpc._norm_tokens("Finish sim HW 4 today"))
        assert cpc._digit_tokens(cpc._norm_tokens("Submit homework 7")) == {"7"}


class TestFormatter:
    def _render(self, threads):
        from core.prompt.formatter import PromptFormatter
        tm = MagicMock()
        tm.count_tokens = MagicMock(return_value=10)
        fmt = PromptFormatter(token_manager=tm, time_manager=None)
        fmt._feature_inventory_cache = None
        ctx = {k: [] for k in (
            "recent_conversations", "memories", "summaries", "reflections", "dreams",
            "semantic_chunks", "wiki", "personal_notes", "reference_docs", "user_uploads",
            "git_commits", "procedural_skills", "proposed_features", "graph_context",
            "upcoming_schedule", "google_calendar", "proactive_insights")}
        ctx.update({"user_profile": "", "narrative_state": "",
                    "web_search_results": None, "unresolved_threads": threads})
        return fmt._assemble_prompt(ctx, "hi")

    def test_date_wording_by_type(self):
        out = self._render([
            {"thread_type": "commitment", "topic": "Finish sim HW", "summary": "s",
             "deadline_date": "2026-10-14"},
            {"thread_type": "deadline", "topic": "Rent", "summary": "s",
             "deadline_date": "2026-10-20"},
            {"thread_type": "question", "topic": "Q", "summary": "s",
             "deadline_date": "2026-10-21"},
        ])
        assert "(target date the user set: 2026-10-14)" in out
        assert "(due: 2026-10-20)" in out
        assert "(date: 2026-10-21)" in out
        assert "(deadline:" not in out


# --- referee additions (2026-10-14) ------------------------------------------
def _rt(tid, topic, ttype="commitment"):
    from memory.thread_models import OpenThread
    return OpenThread(thread_id=tid, topic=topic, summary=topic, thread_type=ttype)


def test_fused_code_matches_spaced_topic():
    from memory.thread_store import check_quick_resolutions
    assert check_quick_resolutions("I finished hw4 last night", [_rt("a", "Finish sim HW 4 today")]) == ["a"]
    assert check_quick_resolutions("I finished HW 4 last night", [_rt("a", "Finish sim HW4 today")]) == ["a"]


def test_course_label_alone_does_not_resolve_labelled_object():
    from memory.thread_store import check_quick_resolutions
    thread = _rt("b", "STAT 5100 incomplete work due end of semester", "deadline")
    assert check_quick_resolutions("I finished the stat lecture today", [thread]) == []
    assert check_quick_resolutions("I finished the incomplete work today", [thread]) == ["b"]
