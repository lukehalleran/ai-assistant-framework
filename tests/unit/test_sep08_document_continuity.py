"""tests/unit/test_sep08_document_continuity.py

B5 (2026-09-08, bounded active-document continuity) integration tests: the
live incident was a homework PDF attached at turn 0 that Daemon could no
longer see by turn 8 ("ok next q please") because [RECENT CONVERSATION] had
already middle-out trimmed it out of history. These tests drive the
DEPLOYED `handle_submit` (via the `tests/unit/test_handle_submit.py`
harness) end-to-end, plus direct probes of `ResponsePlanner.build_context_digest`
and `AgenticSearchController._build_final_prompt` (F3/F8). All synthetic
content; no store/network access.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.active_document import ActiveDocumentRegistry
from core.agentic.controller import AgenticSearchController
from core.agentic.types import AgenticSearchSession
from core.response_planner import ResponsePlanner
from tests.unit.test_handle_submit import _collect, _make_orchestrator
from utils.attachment_audit import audit_attachments
from utils.file_processor import ProcessedFile, ProcessedFilesResult

# A realistic 3-question homework doc. Body prose deliberately never starts a
# line with "Q<digit>" (a body cross-reference shaped like that collides with
# the word-family regex's bare "q" alternative and would fragment the item
# split — a known trade-off of the line-anchored grammar, not exercised here).
THREE_Q_DOC = (
    "Homework 1\n\n"
    "Question 1\n"
    "What is the sample mean of the dataset?\n\n"
    "Question 2\n"
    "What is the sample standard deviation?\n\n"
    "Question 3\n"
    "Interpret the regression coefficient.\n"
)


def _doc_result(filename, text, ext=".pdf"):
    return ProcessedFilesResult(
        text_content=text,
        documents=[ProcessedFile(filename=filename, extension=ext, content_text=text)],
    )


async def _submit(orch, user_text, files=None, doc_result=None):
    """Drive the deployed handle_submit for one turn, returning
    (yields, user_input) where user_input is exactly what THIS turn's
    orchestrator.prepare_prompt call received for its `user_input` kwarg —
    ctx.analysis_text, which is what carries the active-document injection."""
    from gui.handlers import handle_submit

    fp = MagicMock()
    fp.process_files_structured = AsyncMock(
        return_value=doc_result if doc_result is not None
        else ProcessedFilesResult(text_content=user_text)
    )
    patches = [
        patch("gui.handlers.file_processor", fp),
        patch("gui.handlers.get_conversation_logger", return_value=MagicMock()),
    ]
    for p in patches:
        p.start()
    try:
        gen = handle_submit(
            user_text=user_text, files=files, history=[],
            use_raw_gpt=False, orchestrator=orch, fast_mode=False,
        )
        results = await _collect(gen)
    finally:
        patch.stopall()

    call = orch.prepare_prompt.call_args
    user_input = call.kwargs.get("user_input", "") if call is not None else ""
    return results, user_input


def _fresh_orchestrator():
    orch = _make_orchestrator(streaming_chunks=["Sure, here you go."])
    orch.active_documents = ActiveDocumentRegistry()
    return orch


# ---------------------------------------------------------------------------
# End-to-end continuity through handle_submit
# ---------------------------------------------------------------------------

class TestDocumentContinuityAcrossTurns:
    @pytest.mark.asyncio
    async def test_full_session_flow(self):
        orch = _fresh_orchestrator()

        # Turn 1: attach the doc AND ask for the first question in the same
        # message — the doc just registered this turn is already a
        # candidate, so this resolves in-turn (not just on a later re-ask).
        _, ui1 = await _submit(
            orch, "please show me first question",
            files=[SimpleNamespace(name="/tmp/x.pdf", orig_name="Homework1-2.pdf")],
            doc_result=_doc_result("Homework1-2.pdf", THREE_Q_DOC),
        )
        assert "[ACTIVE DOCUMENT — Homework1-2.pdf, Question 1" in ui1

        # Turns 2-3: unrelated code questions, no files attached — no
        # active-document text should appear.
        _, ui2 = await _submit(orch, "why does my python for loop only run once?")
        assert "[ACTIVE DOCUMENT" not in ui2

        _, ui3 = await _submit(orch, "how do I fix an off by one indexing bug?")
        assert "[ACTIVE DOCUMENT" not in ui3

        # Turn 4: no re-attachment, elliptical navigation — must reach the
        # SECOND question of the ORIGINAL document, and must not have taken
        # the light/self-report path (the injected passage is long).
        _, ui4 = await _submit(orch, "ok next q please")
        assert "Question 2 (2 of 3)" in ui4
        assert len(ui4.split()) > 8

    @pytest.mark.asyncio
    async def test_exhausted_navigation_note(self):
        orch = _fresh_orchestrator()
        await _submit(
            orch, "attaching my homework",
            files=[SimpleNamespace(name="/tmp/x.pdf", orig_name="Homework1-2.pdf")],
            doc_result=_doc_result("Homework1-2.pdf", THREE_Q_DOC),
        )
        _, ui = await _submit(orch, "can I see question 9 please")
        assert "does not exist" in ui
        assert "Homework1-2.pdf" in ui

    @pytest.mark.asyncio
    async def test_ambiguous_navigation_note(self):
        orch = _fresh_orchestrator()
        await _submit(
            orch, "here is my first file",
            files=[SimpleNamespace(name="/tmp/a.pdf", orig_name="Homework-A.pdf")],
            doc_result=_doc_result("Homework-A.pdf", THREE_Q_DOC),
        )
        await _submit(
            orch, "here is my second file",
            files=[SimpleNamespace(name="/tmp/b.pdf", orig_name="Homework-B.pdf")],
            doc_result=_doc_result("Homework-B.pdf", THREE_Q_DOC.replace("Homework 1", "Homework 2")),
        )
        _, ui = await _submit(orch, "next q please")
        assert "Several attached documents contain numbered items" in ui
        assert "Homework-A.pdf" in ui
        assert "Homework-B.pdf" in ui


# ---------------------------------------------------------------------------
# audit_attachments — "previously attached this session"
# ---------------------------------------------------------------------------

class TestAuditAttachmentsAvailableDocuments:
    def test_missing_but_previously_attached_is_distinguished(self):
        note = audit_attachments(
            "Can you use Housing.csv from before?",
            files=[], documents=[],
            available_documents=["Housing.csv", "Other.pdf"],
        )
        assert "Previously attached this session (available on request): Housing.csv." in note
        assert "not attached: Housing.csv" not in note

    def test_missing_and_never_attached_stays_in_missing_list(self):
        note = audit_attachments(
            "Can you use NeverSeen.csv?",
            files=[], documents=[],
            available_documents=["Housing.csv"],
        )
        assert "Pasted material references files not attached: NeverSeen.csv." in note
        assert "Previously attached" not in note

    def test_no_available_documents_is_backward_compatible(self):
        assert audit_attachments("nothing referenced here", [], []) == ""


# ---------------------------------------------------------------------------
# F3 — planner digest carries a reserved user_uploads allowance
# ---------------------------------------------------------------------------

class TestPlannerDigestUserUploads:
    def test_digest_contains_upload_marker(self):
        digest, sections = ResponsePlanner.build_context_digest(
            {"user_uploads": [{"content": "TASK_1_MARKER"}]}
        )
        assert "TASK_1_MARKER" in digest
        assert sections == ["user_uploads"]

    def test_roster_only_marker_is_skipped(self):
        digest, sections = ResponsePlanner.build_context_digest(
            {"user_uploads": [{"content": "", "metadata": {"type": "upload_roster", "roster": []}}]}
        )
        assert digest == ""
        assert sections == []

    def test_user_uploads_does_not_starve_other_sections(self):
        digest, sections = ResponsePlanner.build_context_digest(
            {
                "user_uploads": [{"content": "TASK_1_MARKER"}],
                "stm_summary": {"intent": "homework help"},
            },
            max_chars=6000,
        )
        assert "TASK_1_MARKER" in digest
        assert "homework help" in digest
        assert sections == ["user_uploads", "stm_summary"]


# ---------------------------------------------------------------------------
# F8 — controller._build_final_prompt squeezes an oversized query part
# ---------------------------------------------------------------------------

class TestFinalPromptQuerySqueeze:
    def _controller(self):
        model_manager = MagicMock()
        model_manager.api_models = {}
        controller = AgenticSearchController(model_manager=model_manager, web_search_manager=MagicMock())
        controller._tool_executor = SimpleNamespace(
            get_tool_health=lambda: "All tools nominal.",
            _current_web_source_map=None,
            _current_wiki_source_map=None,
        )
        return controller

    def test_oversized_query_is_squeezed_under_ceiling(self):
        controller = self._controller()
        session = AgenticSearchSession(query="huge")
        huge_query = "HEAD_MARKER_TEXT " + ("X" * 200_000) + " TAIL_MARKER_TEXT"

        final_prompt = controller._build_final_prompt(
            query=huge_query, session=session, initial_context=None
        )

        prompt_ceiling = controller.context_budget_tokens * 5
        assert controller._estimate_tokens(final_prompt) <= prompt_ceiling
        assert "HEAD_MARKER_TEXT" in final_prompt
        assert "characters of attached material omitted from this call" in final_prompt
        assert "get_full_document(title=" in final_prompt
        # Every non-trimmable section still present — nothing else was cut.
        assert "[TIME CONTEXT]" in final_prompt
        assert "[TOOL STATUS" in final_prompt
        assert "[CURRENT USER QUERY" in final_prompt

    def test_small_query_is_untouched(self):
        controller = self._controller()
        session = AgenticSearchSession(query="small")
        final_prompt = controller._build_final_prompt(
            query="What is 2+2?", session=session, initial_context=None
        )
        assert "characters of attached material omitted" not in final_prompt
        assert "What is 2+2?" in final_prompt
