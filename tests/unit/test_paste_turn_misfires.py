"""2026-08-27 evening-turn audit: a pasted-emails status update ("I sent
these. Hi Morgan and Robin, ... ugh") misfired four independent systems:

1. Agentic gate — the UNANCHORED second-person branch of _REQUEST_SHAPED_RE
   matched "...can you point me to the right process?" (a question addressed
   to the email's recipient), and the substring context words 'document'/
   'file' matched "documented medical circumstances"/"documentation already
   on file" in the prior turns → "File retrieval continuation" → 106s
   agentic loop on a message that requested nothing.
2. Visual gate — bare "see" ("the two options I see are...") in a 700-word
   paste counted as visual intent; entity 'luke' matched the email SIGNATURE;
   two cat photos were attached to the final synthesis and narrated.
3. Intent — bare \\bcommit\\b matched "Before I commit to that" →
   project_work@0.80.
4. Upload staleness gate — two year-old photo uploads (image STUBS whose
   stored text is a filename+bytes line) cleared the 0.62 text-relevance bar
   (embedding noise), for the second time since 2026-08-14.
"""

import pytest

from core.agentic.gate import (
    FILE_DOC_CONTEXT_WORD_RE,
    FILE_DOC_CONTEXT_WORDS,
    _is_request_shaped,
    evaluate_agentic_gate,
)
from core.intent_classifier import IntentClassifier
from core.prompt.gatherer_knowledge import (
    _query_wants_visual,
    _upload_is_image_stub,
    _upload_is_live,
)


# Representative of the live paste: opens with a statement, embeds a
# second-person question addressed to the EMAIL's recipients, contains
# incidental "see", well over the continuation length cap.
PASTED_EMAIL = (
    "I sent these. Hi Morgan and Robin, thank you both again for all your "
    "help this summer. I'm enrolled in two courses this fall and the drop "
    "deadline is tomorrow at 3 PM Central. The two options I see are "
    "petitioning to extend the incomplete deadline, or a retroactive medical "
    "withdrawal for that term. Do you have a recommendation between them, "
    "and can you point me to the right process for each? Given the Friday "
    "deadline, could we speak by phone today or tomorrow morning? "
    "Thank you so much. Luke U_handle GTID 000000000 ugh"
)

# The prior turns from the live session — full of medical-admin vocabulary
# that the substring context words used to match.
MEDICAL_ADMIN_TURNS = [
    {
        "query": "Can we do a web search and attempt to confirm the specific drop date",
        "response": "Yeah, I can search — a medical withdrawal filed after the "
                    "deadline is often possible with documentation. Your note "
                    "said Friday 8/29 but 8/29 is a Saturday.",
        "response_mode": "enhanced",
    },
    {
        "query": "I would like to confirm it's this Friday that's the drop date",
        "response": "Here's what I can tell you: given documented medical "
                    "circumstances and your documentation already on file, "
                    "asking the registrar directly is worth it.",
        "response_mode": "enhanced",
    },
]


class FakeCorpus:
    def __init__(self, turns):
        self._turns = turns

    def get_recent_memories(self, n):
        return self._turns[:n]


# ===========================================================================
# 1. Agentic gate — request shape + context words
# ===========================================================================

class TestRequestShapeAnchoring:

    def test_embedded_can_you_not_request_shaped(self):
        assert not _is_request_shaped(PASTED_EMAIL)

    def test_leading_can_you_still_request_shaped(self):
        assert _is_request_shaped("Can you check the veto logic")
        assert _is_request_shaped("hey, can you pull up the logs")
        assert _is_request_shaped("please could you review the diff")

    def test_leading_imperative_still_request_shaped(self):
        assert _is_request_shaped("check the docs for what changed")
        assert _is_request_shaped("Alright, pull up the veto logic")

    def test_discourse_marker_still_excluded(self):
        assert not _is_request_shaped("Look, I'm just tired")

    def test_statement_opener_not_request_shaped(self):
        assert not _is_request_shaped("I sent these emails this morning")


class TestFileContextWordBounding:

    @pytest.mark.parametrize("blob", [
        "given documented medical circumstances a petition is realistic",
        "my documentation already on file with ods",  # 'on file' is legit but...
        "a medical withdrawal filed after the deadline",
        "check your user profile settings",
    ])
    def test_derived_forms_do_not_match_word_re(self, blob):
        # 'documented'/'documentation'/'filed'/'profile' must not satisfy the
        # word-bounded document/file markers.
        for w in ("document", "file"):
            assert w not in FILE_DOC_CONTEXT_WORDS
        hits = FILE_DOC_CONTEXT_WORD_RE.findall(blob)
        assert all(h.lower() in ("file", "files", "document", "documents") for h in hits)

    def test_real_document_mentions_still_match(self):
        assert FILE_DOC_CONTEXT_WORD_RE.search("I saved the document yesterday")
        assert FILE_DOC_CONTEXT_WORD_RE.search("the files are in the repo")

    def test_derived_forms_rejected(self):
        for blob in ("documented circumstances", "the documentation trail",
                     "filed the petition", "update your profile"):
            assert not FILE_DOC_CONTEXT_WORD_RE.search(blob)


class TestGateLiveReproduction:

    @pytest.mark.asyncio
    async def test_pasted_email_does_not_ride_file_continuation(self):
        """The live turn: pasted emails after medical-admin turns must not
        route to the tool loop via the file-retrieval continuation."""
        d = await evaluate_agentic_gate(
            user_text=PASTED_EMAIL,
            corpus_manager=FakeCorpus(MEDICAL_ADMIN_TURNS),
        )
        # Whatever else the tiers decide (memory/knowledge may still trigger),
        # the file-continuation arm must not have fired — it is what set
        # needs_tools (and veto exemption) on the live turn.
        assert "tools" not in (d.modes or [])

    @pytest.mark.asyncio
    async def test_terse_retrieval_after_repo_turn_still_routes(self):
        repo_turns = [{
            "query": "Pushed yesterdays works docs are updated too, check it out",
            "response": "Nice — pushed and docs updated. I can't actually pull "
                        "up the repo from here.",
            "response_mode": "enhanced",
        }]
        d = await evaluate_agentic_gate(
            user_text="Alright, check it out now",
            corpus_manager=FakeCorpus(repo_turns),
        )
        assert d.should_trigger is True


# ===========================================================================
# 2. Visual gate
# ===========================================================================

class TestVisualGate:

    def test_long_paste_with_incidental_see_blocked(self):
        assert not _query_wants_visual(PASTED_EMAIL, None)
        assert not _query_wants_visual(PASTED_EMAIL, "project_work")

    def test_long_paste_blocked_even_on_recall_intent(self):
        # Recall-classified long pastes are reciting content, not asking to
        # see something.
        assert not _query_wants_visual(PASTED_EMAIL, "factual_recall")

    def test_short_weak_verb_passes(self):
        assert _query_wants_visual("show me Mochi", None)
        assert _query_wants_visual("can I see that again", None)

    def test_visual_noun_passes_when_short_or_requested(self):
        # 2026-08-28 revision: nouns are no longer sufficient at ANY length —
        # "Screenshot saved." narration inside a long ingest paste fired the
        # gate the day after this file shipped. Nouns carry the signal in a
        # short message, or in a short REQUEST line inside a long one.
        assert _query_wants_visual("pull up the screenshot", None)
        assert _query_wants_visual("there was a photo of the whiteboard", None)
        long_with_request = ("word " * 40) + ". Show me the photo of the whiteboard"
        assert _query_wants_visual(long_with_request, None)
        long_with_narration = ("word " * 40) + ". There was a photo of the whiteboard"
        assert not _query_wants_visual(long_with_narration, None)

    def test_short_recall_intent_passes(self):
        assert _query_wants_visual("what does my cat look like", "factual_recall")
        assert _query_wants_visual("what did the kitchen look like before",
                                   "temporal_recall")

    def test_plain_message_still_blocked(self):
        assert not _query_wants_visual("how are you today", None)
        assert not _query_wants_visual("the file is at /home/lukeh/main.py",
                                       "technical_help")


# ===========================================================================
# 3. Intent — "commit to" is not git
# ===========================================================================

class TestCommitToIntent:

    @pytest.fixture(scope="class")
    def classifier(self):
        return IntentClassifier()

    def test_commit_to_not_project_work(self, classifier):
        r = classifier.classify("Before I commit to that, I'd value your honest read")
        assert not (r.intent.value == "project_work" and r.confidence >= 0.8)

    def test_pasted_email_not_project_work(self, classifier):
        r = classifier.classify(PASTED_EMAIL)
        assert not (r.intent.value == "project_work" and r.confidence >= 0.8)

    def test_git_commit_still_project_work(self, classifier):
        for q in ("let me commit the changes", "commit and push this branch",
                  "time to merge the PR"):
            r = classifier.classify(q)
            assert r.intent.value == "project_work", q


# ===========================================================================
# 4. Upload image stubs
# ===========================================================================

def _image_stub(relevance=0.7, ts="2026-01-21T02:09:55"):
    return {
        "content": "User uploaded image: PXL_20260121_020955773.jpg (image/jpeg, 2157175 bytes)",
        "relevance_score": relevance,
        "metadata": {"type": "user_upload", "timestamp": ts},
    }


def _text_doc(relevance, ts="2026-01-01T00:00:00"):
    return {
        "content": "Question 7.1 Describe a situation for exponential smoothing...",
        "relevance_score": relevance,
        "metadata": {"type": "user_upload", "timestamp": ts},
    }


class TestUploadImageStubGate:

    def test_stub_detection(self):
        assert _upload_is_image_stub(_image_stub())
        assert not _upload_is_image_stub(_text_doc(0.5))

    def test_stub_detection_via_mime(self):
        doc = {"content": "binary", "metadata": {"content_type": "image/png"}}
        assert _upload_is_image_stub(doc)

    def test_old_stub_blocked_despite_high_relevance(self):
        """The live failure: a year-old photo stub scoring 0.7 'relevance'
        (embedding noise against a filename line) surfaced on an emotional
        turn with no visual intent."""
        assert not _upload_is_live(_image_stub(relevance=0.9), query=PASTED_EMAIL)

    def test_old_stub_passes_with_visual_intent(self):
        assert _upload_is_live(_image_stub(relevance=0.1),
                               query="show me that photo I uploaded")

    def test_fresh_stub_passes(self):
        from datetime import datetime
        assert _upload_is_live(
            _image_stub(relevance=0.0, ts=datetime.now().isoformat()),
            query="anything at all",
        )

    def test_text_doc_relevance_path_unchanged(self):
        assert _upload_is_live(_text_doc(0.7), query=PASTED_EMAIL)
        assert not _upload_is_live(_text_doc(0.3), query=PASTED_EMAIL)
