"""
Test that file error responses are not stored in memory.

This prevents false memories where the AI remembers old errors
even after the underlying issue has been fixed.
"""
import pytest
from unittest.mock import Mock, AsyncMock
from memory.memory_storage import MemoryStorage


@pytest.fixture
def memory_storage():
    """Create MemoryStorage with mocked dependencies"""
    storage = MemoryStorage(
        corpus_manager=Mock(),
        chroma_store=Mock(),
        fact_extractor=None,
        consolidator=None,
        topic_manager=None,
        scorer=None,
        time_manager=None
    )
    # Mock the add_entry to track calls
    storage.corpus_manager.add_entry = Mock()
    storage.chroma_store.add_conversation_memory = Mock(return_value="test-id")
    return storage


class TestFileErrorDetection:
    """Test _is_file_error_response detection logic"""

    def test_detects_security_error(self, memory_storage):
        """Should detect security error messages"""
        response = "[Security error processing /tmp/file.docx: Path traversal detected]"
        assert memory_storage._is_file_error_response(response) is True

    def test_detects_read_error(self, memory_storage):
        """Should detect file reading errors"""
        response = "[Error reading /tmp/file.txt: Permission denied]"
        assert memory_storage._is_file_error_response(response) is True

    def test_detects_unsupported_type(self, memory_storage):
        """Should detect unsupported file type messages"""
        response = "[Unsupported file type: file.exe]"
        assert memory_storage._is_file_error_response(response) is True

    def test_detects_empty_file(self, memory_storage):
        """Should detect empty file notices"""
        response = "[Empty file: data.csv]"
        assert memory_storage._is_file_error_response(response) is True

    def test_detects_no_text_extracted(self, memory_storage):
        """Should detect DOCX extraction failures"""
        response = "[No text content extracted from document.docx]"
        assert memory_storage._is_file_error_response(response) is True

    def test_detects_size_limit_error(self, memory_storage):
        """Should detect size limit violations"""
        response = "[Total file size exceeds limit: 52428800/52428800 bytes]"
        assert memory_storage._is_file_error_response(response) is True

    def test_ignores_normal_response(self, memory_storage):
        """Should not flag normal conversation as error"""
        response = "That's a great question! Let me help you with that."
        assert memory_storage._is_file_error_response(response) is False

    def test_flags_error_mentioned_in_context(self, memory_storage):
        """Will flag responses that mention error patterns (acceptable tradeoff)"""
        response = """
        Great question about error handling! When dealing with file processing,
        you might encounter various errors like [Security error processing] or
        [Error reading] messages, but these are normal part of robust error handling.

        Here's how to implement proper error handling:
        1. Always validate inputs
        2. Use try-except blocks
        3. Log errors appropriately
        4. Return meaningful error messages

        The key is to anticipate edge cases and handle them gracefully.
        """ * 3  # Make it long enough to pass the heuristic

        # DESIGN DECISION: We flag this as an error (acceptable false positive)
        # Better to be slightly over-aggressive than let actual errors through
        # The patterns are specific enough that mentioning them in teaching is rare
        assert memory_storage._is_file_error_response(response) is True

    def test_detects_short_error_response(self, memory_storage):
        """Should detect short responses dominated by error message"""
        response = "I tried to read the file but got: [Error reading file.txt: Not found]"
        assert memory_storage._is_file_error_response(response) is True


class TestMemoryStorageFiltering:
    """Test that store_interaction skips file error responses"""

    @pytest.mark.asyncio
    async def test_skips_storing_file_error(self, memory_storage):
        """Should not store interactions with file errors"""
        query = "Can you read this file?"
        response = "[Security error processing /tmp/file.docx: Path traversal detected]"

        await memory_storage.store_interaction(query, response)

        # Should NOT have called add_entry or add_conversation_memory
        memory_storage.corpus_manager.add_entry.assert_not_called()
        memory_storage.chroma_store.add_conversation_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_stores_successful_file_upload(self, memory_storage):
        """Should store successful file upload interactions"""
        query = "Here's my document"
        response = "Great! I can see your document contains information about..."

        await memory_storage.store_interaction(query, response)

        # Should have called storage methods
        assert memory_storage.corpus_manager.add_entry.called
        assert memory_storage.chroma_store.add_conversation_memory.called

    @pytest.mark.asyncio
    async def test_stores_normal_conversation(self, memory_storage):
        """Should store normal conversations"""
        query = "What's the weather like?"
        response = "I don't have access to real-time weather data, but..."

        await memory_storage.store_interaction(query, response)

        # Should have called storage methods
        assert memory_storage.corpus_manager.add_entry.called
        assert memory_storage.chroma_store.add_conversation_memory.called


class TestErrorPatternCoverage:
    """Ensure all file processor error patterns are covered"""

    @pytest.mark.parametrize("error_message", [
        "[Security error processing /path/file.txt: Path traversal detected: filename contains directory separators]",
        "[Security error processing /path/file.txt: File object missing 'read()' method]",
        "[Error reading /path/file.csv: Permission denied]",
        "[Unsupported file type: virus.exe]",
        "[Empty file: empty.txt]",
        "[No text content extracted from blank.docx]",
        "[Total file size exceeds limit: 100000000/50000000 bytes]",
    ])
    def test_all_error_patterns_detected(self, memory_storage, error_message):
        """All file processor error patterns should be detected"""
        assert memory_storage._is_file_error_response(error_message) is True
