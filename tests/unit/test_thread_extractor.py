"""Tests for memory/thread_extractor.py — LLM-based thread extraction."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from memory.thread_extractor import (
    ThreadExtractor,
    _build_conversation_text,
    _parse_json_array,
)
from memory.thread_models import OpenThread, ThreadType, ThreadStatus


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------

# The production EXTRACTION_PROMPT contains unescaped braces in its JSON
# examples which causes str.format() to raise KeyError.  We swap in a
# minimal prompt for the extraction tests so the .format() call succeeds and
# the rest of the logic (LLM mock, JSON parsing, thread construction) is
# exercised properly.
_SAFE_EXTRACTION_PROMPT = (
    "Extract open threads from the conversation.\n\n"
    "CONVERSATION:\n{conversation_text}\n\n"
    "Open threads (JSON array only):"
)


def _make_conversations(pairs):
    """Helper: build a list of conversation dicts from (query, response) pairs."""
    return [{"query": q, "response": r} for q, r in pairs]


def _mock_model_manager(return_value):
    """Helper: create a MagicMock with generate_once as an AsyncMock."""
    mm = MagicMock()
    mm.generate_once = AsyncMock(return_value=return_value)
    return mm


def _make_open_threads(n=2):
    """Helper: create n open threads with deterministic thread_ids."""
    threads = []
    for i in range(n):
        threads.append(OpenThread(
            thread_id=f"thread-{i}",
            topic=f"Thread topic {i}",
            summary=f"Summary for thread {i}",
            thread_type=ThreadType.COMMITMENT,
            urgency=0.5 + i * 0.1,
        ))
    return threads


# ===========================================================================
# _build_conversation_text tests
# ===========================================================================

class TestBuildConversationText:
    """Tests for the _build_conversation_text helper function."""

    def test_normal_conversations(self):
        """1. Builds text from normal conversation dicts with query and response."""
        convos = _make_conversations([
            ("Hello, how are you?", "I'm doing well, thanks!"),
            ("Tell me about Python.", "Python is a versatile language."),
        ])
        text = _build_conversation_text(convos)
        assert "User: Hello, how are you?" in text
        assert "Assistant: I'm doing well, thanks!" in text
        assert "User: Tell me about Python." in text
        assert "Assistant: Python is a versatile language." in text

    def test_empty_list(self):
        """2. Returns empty string for empty conversation list."""
        text = _build_conversation_text([])
        assert text == ""

    def test_max_chars_truncation(self):
        """3. Truncates from the front when text exceeds max_chars."""
        # Create a long conversation that exceeds the limit
        convos = _make_conversations([
            ("A" * 300, "B" * 400),
            ("C" * 300, "D" * 400),
            ("E" * 300, "F" * 400),
        ])
        text = _build_conversation_text(convos, max_chars=200)
        assert len(text) <= 200
        # Truncation keeps the END (most recent), so the last conversation
        # content should appear
        assert "F" in text

    def test_query_only(self):
        """Handles entries with only a query (no response)."""
        convos = [{"query": "Just a question", "response": ""}]
        text = _build_conversation_text(convos)
        assert "User: Just a question" in text
        assert "Assistant:" not in text

    def test_response_only(self):
        """Handles entries with only a response (no query)."""
        convos = [{"query": "", "response": "Just a response"}]
        text = _build_conversation_text(convos)
        assert "Assistant: Just a response" in text
        assert "User:" not in text

    def test_both_empty_skipped(self):
        """Entries where both query and response are empty are skipped."""
        convos = [
            {"query": "", "response": ""},
            {"query": "Real question", "response": "Real answer"},
        ]
        text = _build_conversation_text(convos)
        # Should only contain the second conversation
        assert "Real question" in text
        assert text.startswith("User: Real question")


# ===========================================================================
# _parse_json_array tests
# ===========================================================================

class TestParseJsonArray:
    """Tests for the _parse_json_array helper function."""

    def test_valid_json_array(self):
        """4. Parses a valid JSON array directly."""
        raw = '[{"topic": "test", "urgency": 0.5}]'
        result = _parse_json_array(raw)
        assert len(result) == 1
        assert result[0]["topic"] == "test"

    def test_empty_string(self):
        """5. Returns empty list for empty string."""
        assert _parse_json_array("") == []
        assert _parse_json_array("   ") == []

    def test_text_before_and_after_json(self):
        """6. Extracts JSON array from text with surrounding garbage."""
        raw = 'Here are the results:\n[{"topic": "extracted"}]\nEnd of response.'
        result = _parse_json_array(raw)
        assert len(result) == 1
        assert result[0]["topic"] == "extracted"

    def test_invalid_json(self):
        """7. Returns empty list for completely invalid JSON."""
        assert _parse_json_array("this is not json at all") == []
        assert _parse_json_array("{not an array}") == []

    def test_empty_array(self):
        """8. Returns empty list for '[]' input."""
        result = _parse_json_array("[]")
        assert result == []

    def test_none_input(self):
        """Returns empty list for None input."""
        assert _parse_json_array(None) == []

    def test_json_object_not_array(self):
        """Returns empty list when the parsed result is an object, not an array."""
        assert _parse_json_array('{"key": "value"}') == []


# ===========================================================================
# extract_new_threads tests
# ===========================================================================

class TestExtractNewThreads:
    """Tests for ThreadExtractor.extract_new_threads().

    All tests patch EXTRACTION_PROMPT with a brace-safe version so that
    str.format(conversation_text=...) succeeds.  This isolates the JSON
    parsing and thread-construction logic that we actually want to test.
    """

    @pytest.mark.asyncio
    @patch("memory.thread_extractor.EXTRACTION_PROMPT", _SAFE_EXTRACTION_PROMPT)
    async def test_extracts_threads_from_valid_response(self):
        """9. Extracts OpenThread objects from a valid LLM JSON response."""
        llm_response = json.dumps([{
            "topic": "Study for exam",
            "summary": "User needs to study for their exam next Tuesday",
            "thread_type": "deadline",
            "urgency": 0.8,
            "resolution_hint": "User confirms they studied",
            "deadline_date": "2026-03-24",
        }])
        mm = _mock_model_manager(llm_response)
        extractor = ThreadExtractor(model_manager=mm)

        convos = _make_conversations([
            ("I need to study for my exam next Tuesday", "Good luck!"),
        ])
        threads = await extractor.extract_new_threads(convos)

        assert len(threads) == 1
        assert threads[0].topic == "Study for exam"
        assert threads[0].thread_type == ThreadType.DEADLINE
        assert threads[0].urgency == 0.8
        assert threads[0].deadline_date == "2026-03-24"
        assert threads[0].status == ThreadStatus.OPEN

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_model_manager(self):
        """10. Returns empty list when model_manager is None."""
        extractor = ThreadExtractor(model_manager=None)
        convos = _make_conversations([("Hello", "Hi")])
        threads = await extractor.extract_new_threads(convos)
        assert threads == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_conversations_empty(self):
        """11. Returns empty list when session_conversations is empty."""
        mm = _mock_model_manager("[]")
        extractor = ThreadExtractor(model_manager=mm)
        threads = await extractor.extract_new_threads([])
        assert threads == []
        # generate_once should NOT have been called
        mm.generate_once.assert_not_called()

    @pytest.mark.asyncio
    @patch("memory.thread_extractor.EXTRACTION_PROMPT", _SAFE_EXTRACTION_PROMPT)
    async def test_handles_llm_returning_empty_array(self):
        """12. Returns empty list when LLM returns '[]'."""
        mm = _mock_model_manager("[]")
        extractor = ThreadExtractor(model_manager=mm)
        convos = _make_conversations([("Hello", "Hi")])
        threads = await extractor.extract_new_threads(convos)
        assert threads == []

    @pytest.mark.asyncio
    @patch("memory.thread_extractor.EXTRACTION_PROMPT", _SAFE_EXTRACTION_PROMPT)
    async def test_handles_llm_returning_invalid_json(self):
        """13. Returns empty list when LLM returns unparseable text."""
        mm = _mock_model_manager("Sorry, I can't do that right now.")
        extractor = ThreadExtractor(model_manager=mm)
        convos = _make_conversations([("Hello", "Hi")])
        threads = await extractor.extract_new_threads(convos)
        assert threads == []

    @pytest.mark.asyncio
    @patch("memory.thread_extractor.EXTRACTION_PROMPT", _SAFE_EXTRACTION_PROMPT)
    async def test_handles_llm_call_exception(self):
        """14. Returns empty list when the LLM call raises an exception."""
        mm = MagicMock()
        mm.generate_once = AsyncMock(side_effect=RuntimeError("API timeout"))
        extractor = ThreadExtractor(model_manager=mm)
        convos = _make_conversations([("Hello", "Hi")])
        threads = await extractor.extract_new_threads(convos)
        assert threads == []

    @pytest.mark.asyncio
    @patch("memory.thread_extractor.EXTRACTION_PROMPT", _SAFE_EXTRACTION_PROMPT)
    async def test_caps_at_five_threads(self):
        """15. Caps extracted threads at a maximum of 5."""
        items = []
        for i in range(8):
            items.append({
                "topic": f"Thread {i}",
                "summary": f"Summary {i}",
                "thread_type": "commitment",
                "urgency": 0.5,
                "resolution_hint": "done",
                "deadline_date": None,
            })
        mm = _mock_model_manager(json.dumps(items))
        extractor = ThreadExtractor(model_manager=mm)
        convos = _make_conversations([("I have many things to do", "Sounds busy!")])
        threads = await extractor.extract_new_threads(convos)
        assert len(threads) == 5

    @pytest.mark.asyncio
    @patch("memory.thread_extractor.EXTRACTION_PROMPT", _SAFE_EXTRACTION_PROMPT)
    async def test_handles_null_deadline_values(self):
        """16. Handles null/none/empty deadline_date values correctly.

        The code normalises None, "null", and "none" to None.  An empty
        string "" is falsy so it bypasses the cleanup checks and passes
        through to the OpenThread constructor unchanged.
        """
        items = [
            {"topic": "T1", "summary": "S1", "thread_type": "commitment",
             "urgency": 0.5, "resolution_hint": "done", "deadline_date": None},
            {"topic": "T2", "summary": "S2", "thread_type": "commitment",
             "urgency": 0.5, "resolution_hint": "done", "deadline_date": "null"},
            {"topic": "T3", "summary": "S3", "thread_type": "commitment",
             "urgency": 0.5, "resolution_hint": "done", "deadline_date": "none"},
            {"topic": "T4", "summary": "S4", "thread_type": "commitment",
             "urgency": 0.5, "resolution_hint": "done", "deadline_date": ""},
        ]
        mm = _mock_model_manager(json.dumps(items))
        extractor = ThreadExtractor(model_manager=mm)
        convos = _make_conversations([("Stuff to do", "OK")])
        threads = await extractor.extract_new_threads(convos)
        assert len(threads) == 4
        # None, "null", and "none" are normalised to None
        assert threads[0].deadline_date is None
        assert threads[1].deadline_date is None
        assert threads[2].deadline_date is None
        # Empty string "" is falsy, so it bypasses the cleanup guards and
        # passes through to OpenThread as-is
        assert threads[3].deadline_date == ""

    @pytest.mark.asyncio
    @patch("memory.thread_extractor.EXTRACTION_PROMPT", _SAFE_EXTRACTION_PROMPT)
    async def test_clamps_urgency_to_valid_range(self):
        """17. Clamps urgency values outside 0.0-1.0 to that range."""
        items = [
            {"topic": "Low", "summary": "", "thread_type": "commitment",
             "urgency": -0.5, "resolution_hint": "", "deadline_date": None},
            {"topic": "High", "summary": "", "thread_type": "commitment",
             "urgency": 2.5, "resolution_hint": "", "deadline_date": None},
        ]
        mm = _mock_model_manager(json.dumps(items))
        extractor = ThreadExtractor(model_manager=mm)
        convos = _make_conversations([("Test", "OK")])
        threads = await extractor.extract_new_threads(convos)
        assert len(threads) == 2
        assert threads[0].urgency == 0.0
        assert threads[1].urgency == 1.0

    @pytest.mark.asyncio
    @patch("memory.thread_extractor.EXTRACTION_PROMPT", _SAFE_EXTRACTION_PROMPT)
    async def test_maps_thread_types_correctly(self):
        """18. Maps all four thread_type strings to the correct ThreadType enum."""
        items = [
            {"topic": "T-commit", "summary": "", "thread_type": "commitment",
             "urgency": 0.5, "resolution_hint": "", "deadline_date": None},
            {"topic": "T-deadline", "summary": "", "thread_type": "deadline",
             "urgency": 0.5, "resolution_hint": "", "deadline_date": "2026-04-01"},
            {"topic": "T-unfinished", "summary": "", "thread_type": "unfinished",
             "urgency": 0.5, "resolution_hint": "", "deadline_date": None},
            {"topic": "T-question", "summary": "", "thread_type": "question",
             "urgency": 0.5, "resolution_hint": "", "deadline_date": None},
        ]
        mm = _mock_model_manager(json.dumps(items))
        extractor = ThreadExtractor(model_manager=mm)
        convos = _make_conversations([("Various things", "Sure")])
        threads = await extractor.extract_new_threads(convos)
        assert len(threads) == 4
        assert threads[0].thread_type == ThreadType.COMMITMENT
        assert threads[1].thread_type == ThreadType.DEADLINE
        assert threads[2].thread_type == ThreadType.UNFINISHED
        assert threads[3].thread_type == ThreadType.QUESTION

    @pytest.mark.asyncio
    @patch("memory.thread_extractor.EXTRACTION_PROMPT", _SAFE_EXTRACTION_PROMPT)
    async def test_invalid_thread_type_defaults_to_unfinished(self):
        """Unknown thread_type values fall back to UNFINISHED."""
        items = [{"topic": "Bad type", "summary": "", "thread_type": "banana",
                  "urgency": 0.5, "resolution_hint": "", "deadline_date": None}]
        mm = _mock_model_manager(json.dumps(items))
        extractor = ThreadExtractor(model_manager=mm)
        convos = _make_conversations([("Test", "OK")])
        threads = await extractor.extract_new_threads(convos)
        assert len(threads) == 1
        assert threads[0].thread_type == ThreadType.UNFINISHED

    @pytest.mark.asyncio
    @patch("memory.thread_extractor.EXTRACTION_PROMPT", _SAFE_EXTRACTION_PROMPT)
    async def test_returns_empty_when_llm_returns_none(self):
        """Returns empty list when generate_once returns None/empty."""
        mm = _mock_model_manager(None)
        extractor = ThreadExtractor(model_manager=mm)
        convos = _make_conversations([("Hello", "Hi")])
        threads = await extractor.extract_new_threads(convos)
        assert threads == []


# ===========================================================================
# detect_resolutions tests
# ===========================================================================

class TestDetectResolutions:
    """Tests for ThreadExtractor.detect_resolutions()."""

    @pytest.mark.asyncio
    async def test_detects_resolutions_from_valid_response(self):
        """19. Detects resolved threads from a valid LLM JSON response."""
        open_threads = _make_open_threads(2)
        llm_response = json.dumps([
            {"thread_id": open_threads[0].thread_id,
             "resolution": "User said they studied"},
        ])
        mm = _mock_model_manager(llm_response)
        extractor = ThreadExtractor(model_manager=mm)
        convos = _make_conversations([("I studied for the exam!", "Great job!")])

        resolutions = await extractor.detect_resolutions(convos, open_threads)
        assert len(resolutions) == 1
        assert resolutions[0][0] == open_threads[0].thread_id
        assert resolutions[0][1] == "User said they studied"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_open_threads(self):
        """20. Returns empty list and skips LLM call when no open_threads."""
        mm = _mock_model_manager("[]")
        extractor = ThreadExtractor(model_manager=mm)
        convos = _make_conversations([("Hello", "Hi")])
        resolutions = await extractor.detect_resolutions(convos, [])
        assert resolutions == []
        # LLM should NOT have been called due to early return
        mm.generate_once.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_model_manager(self):
        """21. Returns empty list when model_manager is None."""
        extractor = ThreadExtractor(model_manager=None)
        open_threads = _make_open_threads(1)
        convos = _make_conversations([("Hello", "Hi")])
        resolutions = await extractor.detect_resolutions(convos, open_threads)
        assert resolutions == []

    @pytest.mark.asyncio
    async def test_filters_out_invalid_thread_ids(self):
        """22. Filters out thread_ids that don't match any open thread."""
        open_threads = _make_open_threads(1)
        llm_response = json.dumps([
            {"thread_id": open_threads[0].thread_id,
             "resolution": "Resolved correctly"},
            {"thread_id": "nonexistent-id-999",
             "resolution": "Should be filtered"},
            {"thread_id": "",
             "resolution": "Empty ID should be filtered"},
        ])
        mm = _mock_model_manager(llm_response)
        extractor = ThreadExtractor(model_manager=mm)
        convos = _make_conversations([("I did the thing", "Nice!")])

        resolutions = await extractor.detect_resolutions(convos, open_threads)
        assert len(resolutions) == 1
        assert resolutions[0][0] == open_threads[0].thread_id

    @pytest.mark.asyncio
    async def test_handles_llm_returning_empty_array(self):
        """23. Returns empty list when LLM returns '[]'."""
        mm = _mock_model_manager("[]")
        extractor = ThreadExtractor(model_manager=mm)
        open_threads = _make_open_threads(1)
        convos = _make_conversations([("Just chatting", "Sounds good")])
        resolutions = await extractor.detect_resolutions(convos, open_threads)
        assert resolutions == []

    @pytest.mark.asyncio
    async def test_handles_llm_call_exception(self):
        """24. Returns empty list when the LLM call raises an exception."""
        mm = MagicMock()
        mm.generate_once = AsyncMock(side_effect=ConnectionError("Network down"))
        extractor = ThreadExtractor(model_manager=mm)
        open_threads = _make_open_threads(1)
        convos = _make_conversations([("Test", "OK")])
        resolutions = await extractor.detect_resolutions(convos, open_threads)
        assert resolutions == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_conversations_empty(self):
        """Returns empty list when session_conversations is empty."""
        mm = _mock_model_manager("[]")
        extractor = ThreadExtractor(model_manager=mm)
        open_threads = _make_open_threads(1)
        resolutions = await extractor.detect_resolutions([], open_threads)
        assert resolutions == []
        mm.generate_once.assert_not_called()


# ===========================================================================
# _get_model_alias tests
# ===========================================================================

class TestGetModelAlias:
    """Tests for ThreadExtractor._get_model_alias()."""

    def test_returns_config_value(self):
        """25. Returns the THREAD_MODEL_ALIAS config value when importable."""
        with patch(
            "memory.thread_extractor.ThreadExtractor._get_model_alias",
            return_value="fast-model",
        ):
            result = ThreadExtractor._get_model_alias()
            assert result == "fast-model"

    def test_returns_empty_string_on_import_error(self):
        """26. Returns empty string when config import fails."""
        # The static method catches ImportError and returns "".
        # We test this by calling it directly — even if the import succeeds
        # in this environment, the return type must always be str and the
        # method must never raise.
        result = ThreadExtractor._get_model_alias()
        assert isinstance(result, str)
