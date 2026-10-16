"""The "[unverified action claim]" read-time marker quotes the claim.

Incident 2026-10-15, class: BC-97 (item-level marker that names no claim),
BC-91 (machinery text). The bare marker could not tell the model WHICH
sentence was unverified. The marker is now built ONLY in
utils/read_time_markers.py and quotes up to 2 of the Daemon's own claim
sentences; the legacy bare form is still recognised everywhere.
"""
from core.action_claim_guard import annotate_unverified_action_claim as annotate
from core.agentic.controller import AgenticSearchController as Ctl
from utils import read_time_markers as rtm

CLAIM = "I've added the meeting to your calendar."


def test_marker_quotes_the_unbacked_claim_sentence():
    out = annotate("Sure thing.\n" + CLAIM)
    last = out.rsplit("\n", 1)[1]
    assert last == f'[unverified action claim: "{CLAIM}"]'
    assert rtm.is_unverified_marker_line(last)


def test_two_claims_both_quoted_three_gives_more():
    two = annotate("I have sent the email. I have also saved the note to disk.")
    last = two.rsplit("\n", 1)[1]
    assert "I have sent the email." in last and "saved the note" in last
    assert "more" not in last
    three = annotate("I have sent the email. I have also saved the note to disk. "
                     "I added the document to the folder.")
    assert three.rsplit("\n", 1)[1].endswith("+1 more]")


def test_long_claim_is_capped_with_ellipsis():
    marker = rtm.unverified_claim_marker(["I added " + "very " * 100 + "long event."])
    quoted = marker[len(rtm.UNVERIFIED_CLAIM_OPEN):-2]
    assert len(quoted) <= rtm.UNVERIFIED_CLAIM_MAX_CHARS and quoted.endswith("…")


def test_no_claim_text_falls_back_to_legacy_marker():
    assert rtm.unverified_claim_marker([""]) == rtm.UNVERIFIED_CLAIM_MARKER


def test_idempotent_on_old_and_new_forms():
    new = annotate(CLAIM)
    assert annotate(new) == new
    old = CLAIM + "\n" + rtm.UNVERIFIED_CLAIM_MARKER
    assert annotate(old) == old


def test_user_text_is_never_quoted():
    from core.action_claim_guard import annotate_conversation_content
    out = annotate_conversation_content("User: " + CLAIM + "\nAssistant: I have sent the email.")
    marker = rtm.trailing_unverified_marker(out)
    assert "I have sent the email." in marker and "meeting" not in marker


def test_strip_machinery_removes_both_forms():
    for marker in (rtm.UNVERIFIED_CLAIM_MARKER, rtm.unverified_claim_marker([CLAIM])):
        assert rtm.strip_machinery("Body text.\n" + marker) == "Body text."


def test_digest_keeps_new_marker_through_truncation():
    long_reply = ("Done. " + "filler words here " * 30 + CLAIM)
    marked = annotate(long_reply)
    assert rtm.trailing_unverified_marker(marked)
    clipped = Ctl._clip_on_word_boundary(marked, Ctl._DIGEST_MSG_CHARS)
    assert len(clipped) <= Ctl._DIGEST_MSG_CHARS
    assert rtm.trailing_unverified_marker(clipped).startswith("[unverified action claim")
    legacy = Ctl._clip_on_word_boundary("x " * 300 + "\n" + rtm.UNVERIFIED_CLAIM_MARKER,
                                        Ctl._DIGEST_MSG_CHARS)
    assert legacy.endswith(rtm.UNVERIFIED_CLAIM_MARKER)
