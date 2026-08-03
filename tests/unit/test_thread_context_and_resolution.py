"""
Regression tests for the 2026-07-25 thread findings:

1. [THREAD CONTEXT] honesty — thread metadata is read off the PREVIOUS stored
   turn, so the injection claimed "This is message #1 … about Forearm Pain" on
   a self-criticism turn. When the current topic clearly diverges, the wording
   now says the previous message's thread shifted, instead of asserting a
   false continuity (_topics_related gate in the orchestrator).

2. Quick thread resolution — "I did the email yesterday" resolved nothing:
   no completion signal matched ("did"/"sent" absent from the regex) AND the
   single keyword overlap ("email") was under the 2-keyword floor. The email
   thread kept surfacing a day past its deadline. Now: past-action verbs count
   as signals (question/second-person guarded), and a signal-adjacent object
   hit resolves without loosening the global floor.
"""

from core.orchestrator import _topics_related
from memory.thread_store import OpenThread, check_quick_resolutions


class TestTopicsRelated:
    def test_live_mismatches_detected(self):
        # The observed lagging pairs — each should read as a topic shift.
        assert not _topics_related("Forearm Pain", "Self-criticism")
        assert not _topics_related("Feeling Ignored", "Cats Arrival")

    def test_shared_word_is_related(self):
        assert _topics_related("Pain Management", "Walking Pain")
        assert _topics_related("Forearm Pain", "Pain While Resting")

    def test_containment_is_related(self):
        assert _topics_related("Resume", "Resume Formats")

    def test_no_signal_defaults_related(self):
        assert _topics_related("", "Self-criticism")
        assert _topics_related("Forearm Pain", "general")
        assert _topics_related(None, None)


def _thread(tid, topic, summary=""):
    return OpenThread(thread_id=tid, topic=topic, summary=summary)


EMAIL_THREAD = _thread(
    "t-email",
    "Send email response",
    "User has an email response they need to send today (Fri 2026-07-24); "
    "assistant offered to review the draft if pasted in",
)


class TestQuickResolutions:
    def test_did_the_email_resolves(self):
        # The live miss, verbatim.
        out = check_quick_resolutions("I did the email yesterday", [EMAIL_THREAD])
        assert out == ["t-email"]

    def test_sent_the_email_resolves(self):
        out = check_quick_resolutions("I sent the email this morning", [EMAIL_THREAD])
        assert out == ["t-email"]

    def test_question_does_not_resolve(self):
        out = check_quick_resolutions("What did the email say?", [EMAIL_THREAD])
        assert out == []

    def test_second_person_does_not_resolve(self):
        out = check_quick_resolutions("Have you sent the email?", [EMAIL_THREAD])
        assert out == []

    def test_unrelated_completion_does_not_resolve(self):
        out = check_quick_resolutions("I did the workout today", [EMAIL_THREAD])
        assert out == []

    def test_existing_signal_path_unchanged(self):
        hw = _thread("t-hw", "Homework 6 submission", "ISyE homework 6 due Wednesday")
        out = check_quick_resolutions(
            "just submitted homework 6 finally", [hw, EMAIL_THREAD]
        )
        assert out == ["t-hw"]

    def test_numbered_task_still_distinguished(self):
        hw7 = _thread("t-hw7", "Homework 7 submission", "ISyE hw7 due next week")
        out = check_quick_resolutions("just submitted homework 6", [hw7])
        assert out == []
