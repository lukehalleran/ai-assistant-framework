"""2026-09-07 follow-ups after the upload/reuse bundle:

1. The visual-intent gate's weak-verb arm fired on "…the user uploads, can we
   look there for it?" (14 words, "look") and admitted five old image stubs
   under [USER UPLOADED ITEMS]. A weak verb now needs a message that is not
   about documents; a visual noun still wins.
2. scripts/daily_note_catchup.py called configure_logging() with the default
   path, which rotates the LIVE daemon's daemon_debug.log by mtime at 02:00.
   The catch-up job now logs to logs/daily_note_catchup.log.
"""
import os

import pytest

from core.prompt.gatherer_knowledge import _query_wants_visual


class TestWeakVerbNeedsNonDocumentContext:
    def test_turn6_upload_lookup_is_not_visual(self):
        q = "It actually should be embedded in the user uploads, can we look there for it?"
        assert _query_wants_visual(q, None) is False

    def test_show_me_the_syllabus_is_not_visual(self):
        assert _query_wants_visual("show me the syllabus", None) is False

    def test_look_at_the_attachment_is_not_visual(self):
        assert _query_wants_visual("can you look at the attachment I sent", None) is False

    def test_weak_verb_without_document_cue_still_visual(self):
        assert _query_wants_visual("show me Mochi", None) is True

    def test_visual_noun_with_document_cue_still_visual(self):
        # The noun arm runs first and is untouched.
        assert _query_wants_visual("look at this photo of my homework", None) is True

    def test_negated_weak_verb_still_not_visual(self):
        assert _query_wants_visual("don't show me Mochi", None) is False


class TestCatchupLogsToOwnFile:
    def test_configure_logging_gets_dedicated_file_path(self, monkeypatch):
        import utils.logging_utils as lu
        import scripts.daily_note_catchup as catchup

        class _Stop(Exception):
            pass

        seen = {}

        def fake_configure_logging(*args, **kwargs):
            seen.update(kwargs)
            raise _Stop()

        monkeypatch.setattr(lu, "configure_logging", fake_configure_logging)
        with pytest.raises(_Stop):
            catchup.run_catchup()
        assert "file_path" in seen
        assert seen["file_path"].replace(os.sep, "/").endswith("logs/daily_note_catchup.log")
        assert os.path.basename(seen["file_path"]) != "daemon_debug.log"


# ===========================================================================
# 3. Retest of the bundle (2026-09-07 11:14/11:15) showed two more layers:
#    the agentic loop never SAW the roster (context inventory omitted the
#    uploads section; the final synthesis prompt never rendered it; a
#    reference_docs memory search returns syllabus chunks, not the homework),
#    and the gate never fired on "Can we look in the user uploads for the ABC
#    1234 homework…" (no first-person anchor; "can we" not request-shaped).
# ===========================================================================

from unittest.mock import MagicMock

from core.agentic.controller import AgenticSearchController
from core.agentic.formatters import AgenticFormatter
from core.agentic.gate import _is_request_shaped
from core.agentic.tools import ToolExecutor
from core.agentic.types import AgenticSearchSession
from utils.query_checker import is_personal_doc_search

Q2 = ("Can we look in the user uploads for the ABC 1234\n  homework, read it in full, "
      "and tell me what the\n  first task asks for?")

ROSTER_ITEM = {
    "content": "",
    "metadata": {"type": "upload_roster",
                 "roster": [{"title": "Homework1-2.pdf", "date": "2026-09-05"},
                            {"title": "UsedCars2.csv", "date": "2026-09-05"}]},
    "relevance_score": 0.0, "match_type": "roster",
}
CHUNK_ITEM = {
    "content": "ABC 1234 - Predictive Modeling for Operations. Homework needs to be submitted as Canvas quizzes.",
    "metadata": {"type": "user_upload", "title": "upload:tmpjvq8cdj4.pdf"},
    "relevance_score": 0.65, "match_type": "semantic",
}


def _controller():
    mm = MagicMock(); mm.api_models = {}
    c = AgenticSearchController(model_manager=mm, web_search_manager=MagicMock())
    c._tool_executor = MagicMock()
    c._tool_executor._current_web_source_map = {}
    return c


class TestGateFiresOnUploadLookup:
    def test_turn2_query_is_personal_doc_search(self):
        assert is_personal_doc_search(Q2) is True

    def test_attachments_noun_is_self_anchoring(self):
        assert is_personal_doc_search("can you look at the attachments for the due date") is True

    def test_web_request_still_excluded(self):
        assert is_personal_doc_search("search the web for the uploads page of that site") is False

    def test_can_we_is_request_shaped(self):
        assert _is_request_shaped("Can we look in the user uploads for the homework?") is True

    def test_plain_we_statement_not_request_shaped(self):
        assert _is_request_shaped("we went to the store and looked at uploads") is False


class TestLoopSeesTheRoster:
    def test_inventory_lists_roster_titles(self):
        c = _controller()
        inv = c._compute_context_inventory({"user_uploads": [ROSTER_ITEM, CHUNK_ITEM]})
        assert "[USER UPLOADED ITEMS]: 1 admitted upload chunks" in inv
        assert "get_full_document" in inv
        assert "Homework1-2.pdf (2026-09-05)" in inv

    def test_inventory_without_uploads_unchanged(self):
        c = _controller()
        assert "USER UPLOADED" not in c._compute_context_inventory({"memories": [{"content": "m"}]})

    def test_final_prompt_renders_uploads_and_roster(self):
        c = _controller()
        session = AgenticSearchSession(query="q")
        prompt = c._build_final_prompt("q", session, {"user_uploads": [ROSTER_ITEM, CHUNK_ITEM]})
        assert "[USER UPLOADED ITEMS]" in prompt
        assert "1) **tmpjvq8cdj4.pdf**" in prompt
        assert "Canvas quizzes" in prompt
        assert "Recently uploaded files" in prompt and "Homework1-2.pdf (2026-09-05)" in prompt

    def test_user_uploads_is_a_direct_rendered_key(self):
        assert "user_uploads" in AgenticSearchController._FINAL_PROMPT_DIRECT_RENDERED_KEYS

    def test_format_user_uploads_names_image_stubs_only(self):
        img = {"content": "User uploaded image: x.png (image/png, 10 bytes)",
               "metadata": {"type": "user_upload", "title": "upload:x.png", "is_image": True}}
        text = AgenticSearchController._format_user_uploads([img])
        assert "x.png" in text and "(image upload)" in text and "image/png" not in text


class _FakeColl:
    def __init__(self, metas):
        self.metas = metas
        self.calls = []

    def get(self, **kwargs):
        self.calls.append(kwargs)
        return {"metadatas": self.metas}


def _executor(metas, results):
    ex = ToolExecutor.__new__(ToolExecutor)
    store = MagicMock()
    coll = _FakeColl(metas)
    store._get_collection = lambda name: coll
    store.query_collection = lambda **kw: results
    ex.chroma_store = store
    ex.formatter = AgenticFormatter()
    ex._current_wiki_source_map = {}
    return ex, coll


class TestMemorySearchListsUploadTitles:
    METAS = [
        {"type": "user_upload", "title": "upload:Homework1-2.pdf", "timestamp": "2026-09-05T13:43:00"},
        {"type": "user_upload", "title": "upload:Homework1-2.pdf", "timestamp": "2026-09-05T13:43:01"},
        {"type": "user_upload", "title": "upload:old.txt", "timestamp": "2026-06-01T10:00:00"},
        {"type": "user_upload", "title": "upload:pic.jpg", "timestamp": "2026-09-07T01:31:00", "is_image": True},
    ]

    @pytest.mark.asyncio
    async def test_reference_docs_search_appends_exact_titles(self):
        ex, coll = _executor(self.METAS, [{"id": "a", "content": "syllabus text", "relevance_score": 0.6,
                                           "metadata": {"title": "upload:tmp.pdf", "section": ""}}])
        out = await ex._execute_memory_search("ABC 1234 first assignment", "reference_docs")
        assert "syllabus text" in out
        assert '"upload:Homework1-2.pdf" (2026-09-05)' in out
        assert "get_full_document" in out
        assert out.index("Homework1-2") < out.index("old.txt")  # newest first
        assert "pic.jpg" not in out
        assert coll.calls and coll.calls[0].get("include") == ["metadatas"]

    @pytest.mark.asyncio
    async def test_empty_reference_docs_search_still_lists_titles(self):
        ex, _ = _executor(self.METAS, [])
        out = await ex._execute_memory_search("nothing", "reference_docs")
        assert out.startswith("[No results found in reference_docs")
        assert "upload:Homework1-2.pdf" in out

    @pytest.mark.asyncio
    async def test_other_collections_do_not_list_titles(self):
        ex, coll = _executor(self.METAS, [{"id": "a", "content": "a memory", "relevance_score": 0.6, "metadata": {}}])
        out = await ex._execute_memory_search("q", "conversations")
        assert "get_full_document" not in out and not coll.calls
