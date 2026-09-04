"""
Regression tests for the 2026-09-04 homework-attachment turn audit
(docs/HANDOFF_20260904_hw_attachment_turn.md, sections B + B2).

A single turn (1 message + UsedCars.csv + Homework1-1.pdf + ~10 lecture
transcripts) produced a 265,651-token prompt, misclassified as
temporal_recall, answered from prior knowledge of a similarly-named public
dataset instead of the attached 1,264-row CSV, and missed both a Part-2
attachment trap and an Eastern-vs-Central deadline conversion. All fixes
here are deterministic (no LLM calls).

Item order mirrors the addendum: 1 (duplication), 4 (CSV/XLSX manifest),
8 (missing-attachment audit), 9 (deadline timezone), 2 (analysis-text
scoping), 3 (same-turn upload dedupe), 5 (phone-redaction newline guard),
6 (TEMPORAL REASONING prompt line).
"""

from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest


def _mock_file(name: str, content: bytes):
    """Minimal file-like object matching the FileProcessor contract used
    throughout tests/unit/test_file_processor.py."""
    mock_file = Mock()
    mock_file.name = name
    mock_file.read = Mock(return_value=content)
    mock_file.seek = Mock()
    return mock_file


# ===========================================================================
# Item 1: attachment bundle duplication
# ===========================================================================

class TestFileProcessorDuplicateGuard:
    """A same-batch duplicate (two distinct file_ids resolving to
    byte-identical uploads, e.g. from a client-side double-fire) used to
    double every attached document's text in the rendered prompt."""

    @pytest.mark.asyncio
    async def test_duplicate_content_appended_once(self):
        from utils.file_processor import FileProcessor

        fp = FileProcessor()
        f1 = _mock_file("a.txt", b"UNIQUE MARKER CONTENT")
        f2 = _mock_file("a.txt", b"UNIQUE MARKER CONTENT")  # same name+bytes

        result = await fp.process_files_structured("query text", [f1, f2])

        assert result.text_content.count("UNIQUE MARKER CONTENT") == 1
        # Both are still tracked as processed (persistence/accounting
        # unaffected) — only the rendered text is deduped.
        assert len(result.documents) == 2

    @pytest.mark.asyncio
    async def test_distinct_content_not_deduped(self):
        from utils.file_processor import FileProcessor

        fp = FileProcessor()
        f1 = _mock_file("a.txt", b"first file content")
        f2 = _mock_file("b.txt", b"second file content")

        result = await fp.process_files_structured("query text", [f1, f2])

        assert "first file content" in result.text_content
        assert "second file content" in result.text_content

    @pytest.mark.asyncio
    async def test_two_real_files_each_appear_exactly_once(self):
        """Item 1's exact spec: fake 2 text files through the merge path and
        assert each file's content occurs EXACTLY once in the result."""
        from utils.file_processor import FileProcessor

        fp = FileProcessor()
        f1 = _mock_file("csv_marker.txt", b"CSV_MARKER_ROWS_1264")
        f2 = _mock_file("pdf_marker.txt", b"PDF_MARKER_PART_ONE")

        result = await fp.process_files_structured("do my homework", [f1, f2])

        assert result.text_content.count("CSV_MARKER_ROWS_1264") == 1
        assert result.text_content.count("PDF_MARKER_PART_ONE") == 1


class TestResolveUploadsDedupe:
    """api/state.py AppState.resolve_uploads: a repeated file_id in the
    request must not resolve to the same upload twice."""

    def test_duplicate_file_ids_deduped_preserving_order(self, tmp_path):
        from api.state import AppState

        p1 = tmp_path / "one.txt"
        p1.write_text("one")
        p2 = tmp_path / "two.txt"
        p2.write_text("two")

        state = AppState()
        fid1 = state.register_upload(path=str(p1), name="one.txt", size=3)
        fid2 = state.register_upload(path=str(p2), name="two.txt", size=3)

        files = state.resolve_uploads([fid1, fid2, fid1])

        assert len(files) == 2
        assert [f.orig_name for f in files] == ["one.txt", "two.txt"]


# ===========================================================================
# Item 4: deterministic CSV/XLSX manifest
# ===========================================================================

class TestTabularManifest:
    @pytest.mark.asyncio
    async def test_csv_manifest_prepended_with_true_row_count(self, tmp_path):
        from utils.file_processor import FileProcessor

        path = tmp_path / "UsedCars.csv"
        df = pd.DataFrame({
            "Id": range(1, 6),
            "Price": [100, 200, 300, 400, 500],
            "Model": ["A", "B", "C", "D", "E"],
        })
        df.to_csv(path, index=False)

        fp = FileProcessor()
        mock_file = _mock_file("UsedCars.csv", path.read_bytes())
        content, _size = fp._process_single_file(mock_file)

        assert content.startswith("[UsedCars.csv: 5 rows")
        assert "5 rows" in content.split("\n")[0]
        assert "Id" in content
        assert "Price" in content
        assert "do not rely on prior knowledge" in content

    @pytest.mark.asyncio
    async def test_csv_manifest_matches_true_count_not_a_guess(self, tmp_path):
        """The exact bug: a model answered '~1,400 rows' from prior
        knowledge of a similarly-named public dataset instead of the 1,264
        rows actually provided. The manifest must report the PARSED count."""
        from utils.file_processor import FileProcessor

        path = tmp_path / "cars.csv"
        df = pd.DataFrame({"Id": range(1, 43), "Val": range(1, 43)})
        df.to_csv(path, index=False)

        fp = FileProcessor()
        mock_file = _mock_file("cars.csv", path.read_bytes())
        content, _size = fp._process_single_file(mock_file)

        assert "42 rows" in content
        assert "1,400" not in content
        assert "1400" not in content

    def test_csv_manifest_truncates_many_columns(self):
        from utils.file_processor import FileProcessor

        cols = [f"col{i}" for i in range(20)]
        manifest = FileProcessor._tabular_manifest("wide.csv", 10, cols)
        assert "…" in manifest
        assert "col0" in manifest
        assert "col19" not in manifest

    @pytest.mark.asyncio
    async def test_xlsx_manifest_per_sheet(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        from utils.file_processor import FileProcessor

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.append(["Name", "Age"])
        ws.append(["Alice", 30])
        ws.append(["Bob", 25])
        path = tmp_path / "people.xlsx"
        wb.save(str(path))

        fp = FileProcessor()
        mock_file = _mock_file("people.xlsx", path.read_bytes())
        content, _size = fp._process_single_file(mock_file)

        assert "2 rows" in content
        assert "Name" in content and "Age" in content
        assert "do not rely on prior knowledge" in content


# ===========================================================================
# Item 8: referenced-but-missing attachment audit
# ===========================================================================

class TestAttachmentAudit:
    def test_glued_canvas_fragment_and_missing_file_and_part_title(self):
        from utils.attachment_audit import audit_attachments

        user_text = (
            "Instructions\n"
            "onHousing.csvHomDownload the following homework instruction file "
            "and data file:\n\n"
            "Homework1-1.pdf\n\n"
            "UsedCars.csvDownload UsedCars.csv"
        )
        attached_names = {"UsedCars.csv", "Homework1-1.pdf"}

        class _Doc:
            def __init__(self, filename, content_text):
                self.filename = filename
                self.content_text = content_text

        files = [Mock(name="UsedCars.csv", orig_name="UsedCars.csv"),
                 Mock(name="Homework1-1.pdf", orig_name="Homework1-1.pdf")]
        documents = [
            _Doc("UsedCars.csv", "Id,Model,Price\n1,A,100\n"),
            _Doc("Homework1-1.pdf",
                 "Homework 1 – Part 1\n\nSTAT 6021 — due Sep 13 11:59pm Eastern."),
        ]

        note = audit_attachments(user_text, files, documents)

        assert note != ""
        assert "Housing.csv" in note
        assert "Part 1" in note
        # Attached files are not reported as missing.
        for name in attached_names:
            assert f"references files not attached: {name}" not in note

    def test_everything_attached_no_part_title_no_note(self):
        from utils.attachment_audit import audit_attachments

        class _Doc:
            def __init__(self, filename, content_text):
                self.filename = filename
                self.content_text = content_text

        user_text = "Please help me with UsedCars.csv and Homework1-1.pdf."
        files = [Mock(name="UsedCars.csv", orig_name="UsedCars.csv"),
                 Mock(name="Homework1-1.pdf", orig_name="Homework1-1.pdf")]
        documents = [
            _Doc("UsedCars.csv", "Id,Model,Price\n1,A,100\n"),
            _Doc("Homework1-1.pdf", "Regression homework, no other parts mentioned here."),
        ]

        note = audit_attachments(user_text, files, documents)
        assert note == ""

    def test_no_documents_no_note(self):
        from utils.attachment_audit import audit_attachments
        assert audit_attachments("just a message, no files", [], []) == ""

    def test_ambiguous_glue_not_salvaged(self):
        """'UsedCars.csvDownload' has no lowercase-run-then-TitleCase shape
        before the extension — must not fabricate a bogus filename."""
        from utils.attachment_audit import _extract_filenames
        found = _extract_filenames("UsedCars.csvDownload UsedCars.csv")
        assert found == {"UsedCars.csv"}


# ===========================================================================
# Item 9: deadline timezone conversion
# ===========================================================================

class TestDeadlineTimezoneNote:
    def test_eastern_to_central_conversion(self):
        from utils.attachment_audit import deadline_timezone_note

        note = deadline_timezone_note(
            "Assignment due 11:59pm Eastern Time.", user_tz="America/Chicago"
        )
        assert note != ""
        assert "10:59 PM Central" in note
        assert "11:59 PM Eastern" in note

    def test_zoned_time_without_deadline_cue_no_note(self):
        # A lecture transcript's "office hours at 3 pm ET" is NOT a deadline
        # (Fable review: the live call fed user text + every attached doc).
        from utils.attachment_audit import deadline_timezone_note
        note = deadline_timezone_note(
            "Office hours are at 3 pm ET on Tuesdays.", user_tz="America/Chicago"
        )
        assert note == ""

    def test_first_cued_match_wins_over_earlier_uncued_time(self):
        from utils.attachment_audit import deadline_timezone_note
        text = ("Office hours are at 3 pm ET on Tuesdays.\n"
                "Homework 1 is due Sunday, September 13 by 11:59 PM Eastern Time.")
        note = deadline_timezone_note(text, user_tz="America/Chicago")
        assert "11:59 PM Eastern = 10:59 PM Central" in note
        assert "3:00 PM" not in note

    def test_canvas_bare_by_phrasing_counts_as_cue(self):
        from utils.attachment_audit import deadline_timezone_note
        note = deadline_timezone_note(
            "Sep 13 by 11:59pm ET", user_tz="America/Chicago"
        )
        assert "10:59 PM Central" in note

    def test_same_zone_no_note(self):
        from utils.attachment_audit import deadline_timezone_note

        note = deadline_timezone_note(
            "Assignment due 11:59pm Central Time.", user_tz="America/Chicago"
        )
        assert note == ""

    def test_no_zone_token_no_note(self):
        from utils.attachment_audit import deadline_timezone_note

        note = deadline_timezone_note(
            "Assignment due 11:59pm local time.", user_tz="America/Chicago"
        )
        assert note == ""

    def test_no_time_at_all_no_note(self):
        from utils.attachment_audit import deadline_timezone_note
        note = deadline_timezone_note("Just some ordinary text.", user_tz="America/Chicago")
        assert note == ""


# ===========================================================================
# Item 2: analysis text scoped to the user's own words, not the attachment
# blob — the ContextPipeline mechanism the gui/handlers.py fix relies on.
# ===========================================================================

class TestAnalysisTextScoping:
    @pytest.fixture
    def mock_model_manager(self):
        manager = Mock()
        manager.generate_once = AsyncMock(return_value="rewritten query")
        return manager

    @pytest.fixture
    def mock_topic_manager(self):
        manager = Mock()
        manager.update_from_user_input = Mock()
        manager.get_primary_topic = Mock(return_value="homework")
        manager.get_entities = Mock(return_value=[])
        return manager

    @pytest.fixture
    def mock_file_processor(self):
        processor = Mock()
        # Simulates the giant merged attachment blob (transcripts full of
        # "previous video" mentions) — Stage 3 output only, never fed back
        # into earlier stages.
        processor.process_files = AsyncMock(
            return_value=(
                "How many hours will this homework take?\n\n"
                + ("In the previous video we covered chat history. " * 200)
            )
        )
        return processor

    @pytest.fixture
    def stm_calls(self):
        return []

    @pytest.fixture
    def mock_stm_analyzer(self, stm_calls):
        analyzer = Mock()

        async def _analyze(**kwargs):
            stm_calls.append(kwargs.get("user_query"))
            return {
                "topic": "homework",
                "user_question": kwargs.get("user_query"),
                "intent": "estimate effort",
                "tone": "neutral",
                "open_threads": [],
                "constraints": [],
            }

        analyzer.analyze = AsyncMock(side_effect=_analyze)
        return analyzer

    @pytest.fixture
    def pipeline(self, mock_model_manager, mock_topic_manager, mock_file_processor,
                 mock_stm_analyzer):
        from core.context_pipeline import ContextPipeline
        return ContextPipeline(
            model_manager=mock_model_manager,
            topic_manager=mock_topic_manager,
            file_processor=mock_file_processor,
            stm_analyzer=mock_stm_analyzer,
            config={"USE_STM_PASS": True, "STM_MIN_CONVERSATION_DEPTH": 1},
        )

    @pytest.mark.asyncio
    async def test_stm_sees_short_user_text_not_merged_blob(
        self, pipeline, stm_calls
    ):
        from core.context_pipeline import ToneLevel

        short_query = "How many hours will this homework take?"
        with patch(
            "core.context_pipeline.ContextPipeline._detect_tone",
            new=AsyncMock(return_value=(ToneLevel.CONVERSATIONAL, None)),
        ):
            result = await pipeline.build(
                short_query, files=[Mock(name="transcript.txt")]
            )

        # The rendered content (file_context) DOES carry the full blob —
        # that's what reaches [CURRENT QUERY].
        assert result.has_files
        assert "previous video" in result.file_context

        # But STM — and by the same parameter, topic/tone/intent/rewrite —
        # was called with the short original query, never the blob.
        assert stm_calls, "STM analyzer was not called"
        assert stm_calls[0] == short_query
        assert "previous video" not in stm_calls[0]
        assert result.original_query == short_query

    @pytest.mark.asyncio
    async def test_intent_not_misled_by_attachment_keywords(self, pipeline):
        from core.context_pipeline import ToneLevel
        from core.intent_classifier import IntentType

        short_query = "How many hours will this homework take?"
        with patch(
            "core.context_pipeline.ContextPipeline._detect_tone",
            new=AsyncMock(return_value=(ToneLevel.CONVERSATIONAL, None)),
        ):
            result = await pipeline.build(
                short_query, files=[Mock(name="transcript.txt")]
            )

        assert result.intent is not None
        assert result.intent.intent != IntentType.TEMPORAL_RECALL


class TestAnalysisQueryCap:
    """core/prompt/builder.py build_prompt_from_context: when
    processed_query is pathologically long and the original (pre-merge)
    user text is short, retrieval-adjacent gathering (obsidian keyword
    search, web trigger, agentic gate, memory search) uses the short text
    instead — the full content still reaches [CURRENT QUERY] via a
    separate code path (context.file_context)."""

    def test_long_processed_query_falls_back_to_short_original(self):
        from core.prompt.builder import ANALYSIS_QUERY_MAX_CHARS

        assert ANALYSIS_QUERY_MAX_CHARS == 2000

    @pytest.mark.asyncio
    async def test_build_prompt_receives_short_text_not_the_blob(self):
        from core.prompt.builder import UnifiedPromptBuilder
        from core.context_pipeline import ContextResult, ToneLevel

        short = "How many hours will this homework take?"
        huge = short + "\n\n" + ("previous video previous video " * 500)
        assert len(huge) > 2000

        context = ContextResult(
            processed_query=huge,
            original_query=short,
            tone_level=ToneLevel.CONVERSATIONAL,
            tone_instructions="",
            file_context=huge,
        )

        builder = object.__new__(UnifiedPromptBuilder)
        captured = {}

        async def fake_build_prompt(user_input, **kwargs):
            captured["user_input"] = user_input
            captured["search_query"] = kwargs.get("search_query")
            return {}

        builder.build_prompt = fake_build_prompt

        await UnifiedPromptBuilder.build_prompt_from_context(builder, context)

        assert captured["user_input"] == short
        assert captured["search_query"] is None

    @pytest.mark.asyncio
    async def test_short_processed_query_passes_through_unchanged(self):
        from core.prompt.builder import UnifiedPromptBuilder
        from core.context_pipeline import ContextResult, ToneLevel

        short = "How many hours will this homework take?"
        context = ContextResult(
            processed_query=short,
            original_query=short,
            tone_level=ToneLevel.CONVERSATIONAL,
            tone_instructions="",
        )

        builder = object.__new__(UnifiedPromptBuilder)
        captured = {}

        async def fake_build_prompt(user_input, **kwargs):
            captured["user_input"] = user_input
            return {}

        builder.build_prompt = fake_build_prompt

        await UnifiedPromptBuilder.build_prompt_from_context(builder, context)
        assert captured["user_input"] == short


# ===========================================================================
# Item 3: same-turn upload dedupe in the uploads gatherer
# ===========================================================================

class TestSameTurnUploadDedupe:
    def test_upload_title_filename_strips_prefix(self):
        from core.prompt.gatherer_knowledge import _upload_title_filename
        doc = {"metadata": {"title": "upload:UsedCars.csv"}}
        assert _upload_title_filename(doc) == "usedcars.csv"

    def test_upload_title_filename_unknown_shape(self):
        from core.prompt.gatherer_knowledge import _upload_title_filename
        assert _upload_title_filename({"metadata": {"title": "not an upload"}}) == ""
        assert _upload_title_filename({}) == ""

    @pytest.mark.asyncio
    async def test_same_turn_upload_dropped_from_retrieval(self):
        from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin

        gatherer = KnowledgeRetrievalMixin()
        gatherer.memory_id_map = {}
        gatherer._current_turn_upload_filenames = ["UsedCars.csv"]

        manager = Mock()
        manager.get_documents = AsyncMock(return_value=[
            {
                "content": "Id,Model,Price\n1,A,100\n",
                "relevance_score": 0.95,
                "metadata": {"type": "user_upload", "title": "upload:UsedCars.csv"},
            },
            {
                "content": "some other earlier upload",
                "relevance_score": 0.95,
                "metadata": {"type": "user_upload", "title": "upload:old_notes.txt"},
            },
        ])
        manager.chroma_store = Mock()
        manager.chroma_store._get_collection = Mock(
            return_value=Mock(get=Mock(return_value={"ids": ["x"]}))
        )
        gatherer.reference_docs_manager = manager

        uploads = await gatherer.get_user_uploads("used cars data", limit=5)

        titles = [u["metadata"]["title"] for u in uploads]
        assert "upload:UsedCars.csv" not in titles
        assert "upload:old_notes.txt" in titles

    @pytest.mark.asyncio
    async def test_no_current_turn_filenames_keeps_all(self):
        from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin

        gatherer = KnowledgeRetrievalMixin()
        gatherer.memory_id_map = {}
        gatherer._current_turn_upload_filenames = []

        manager = Mock()
        manager.get_documents = AsyncMock(return_value=[
            {
                "content": "Id,Model,Price\n1,A,100\n",
                "relevance_score": 0.95,
                "metadata": {"type": "user_upload", "title": "upload:UsedCars.csv"},
            },
        ])
        manager.chroma_store = Mock()
        manager.chroma_store._get_collection = Mock(
            return_value=Mock(get=Mock(return_value={"ids": ["x"]}))
        )
        gatherer.reference_docs_manager = manager

        uploads = await gatherer.get_user_uploads("used cars data", limit=5)
        assert len(uploads) == 1


# ===========================================================================
# Item 5: phone-redaction newline guard
# ===========================================================================

class TestPhoneRedactionNewlineGuard:
    def test_csv_table_fragment_not_redacted_as_phone(self):
        from utils.privacy_redaction import redact_text

        text = "1085\n999  1000"
        out = redact_text(text)
        assert "[REDACTED PHONE]" not in out
        assert "1085" in out

    def test_real_phone_still_redacted_parens(self):
        from utils.privacy_redaction import redact_text
        out = redact_text("Email student@example.edu; phone (404) 555-0123.")
        assert "[REDACTED PHONE]" in out

    def test_real_phone_still_redacted_hyphens(self):
        from utils.privacy_redaction import redact_text
        assert redact_text("Call 404-555-0123") == "Call [REDACTED PHONE]"

    def test_multiline_csv_block_no_false_positive(self):
        from utils.privacy_redaction import redact_text
        text = "Id,KM,CC\n1,2000,1300\n2,3000,\n1400\n"
        out = redact_text(text)
        assert "[REDACTED PHONE]" not in out


# ===========================================================================
# Item 6: TEMPORAL REASONING deadline-timezone prompt line
# ===========================================================================

class TestTemporalReasoningDeadlineLine:
    def test_deadline_timezone_guidance_present(self):
        from core.tone_instructions import get_session_headers_instructions
        text = get_session_headers_instructions()
        assert "deadline" in text.lower()
        assert "timezone" in text.lower()
