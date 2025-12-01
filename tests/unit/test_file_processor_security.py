"""
Security tests for utils/file_processor.py

Tests for:
- Path traversal protection
- File size limits
- CSV formula injection sanitization
- Invalid file handling
"""
import pytest
import tempfile
import os
from pathlib import Path
from io import BytesIO, StringIO
from utils.file_processor import FileProcessor
from config.app_config import FILE_UPLOAD_MAX_SIZE, FILE_UPLOAD_MAX_TOTAL_SIZE

# Use config constants for tests
MAX_FILE_SIZE = FILE_UPLOAD_MAX_SIZE
MAX_TOTAL_SIZE = FILE_UPLOAD_MAX_TOTAL_SIZE


class MockFile:
    """Mock file object for testing"""
    def __init__(self, name: str, content: str | bytes):
        self.name = name
        self._content = content
        self._pos = 0

    def read(self):
        return self._content

    def seek(self, pos):
        self._pos = pos


@pytest.fixture
def processor():
    """Create FileProcessor instance"""
    return FileProcessor()


class TestPathTraversalProtection:
    """Test protection against path traversal attacks"""

    @pytest.mark.asyncio
    async def test_blocks_parent_directory_traversal(self, processor):
        """Should block ../ path traversal attempts"""
        malicious_file = MockFile("../../etc/passwd", b"secret data")

        result = await processor.process_files("Test", [malicious_file])

        assert "Path traversal detected" in result
        assert "Security error" in result
        assert "secret data" not in result

    @pytest.mark.asyncio
    async def test_blocks_windows_parent_traversal(self, processor):
        """Should block ..\ path traversal on Windows"""
        malicious_file = MockFile("..\\..\\windows\\system32\\config", b"system data")

        result = await processor.process_files("Test", [malicious_file])

        assert "Path traversal detected" in result or "directory separators" in result

    @pytest.mark.asyncio
    async def test_blocks_absolute_paths(self, processor):
        """Should block absolute path attempts"""
        malicious_file = MockFile("/etc/passwd", b"root:x:0:0")

        result = await processor.process_files("Test", [malicious_file])

        assert "Absolute path not allowed" in result or "Security error" in result

    @pytest.mark.asyncio
    async def test_blocks_windows_absolute_paths(self, processor):
        """Should block Windows absolute paths"""
        malicious_file = MockFile("C:\\Windows\\System32\\config", b"system")

        result = await processor.process_files("Test", [malicious_file])

        assert ("Windows absolute path" in result or
                "Absolute path not allowed" in result or
                "Security error" in result)

    @pytest.mark.asyncio
    async def test_allows_safe_filenames(self, processor):
        """Should allow safe filenames without path components"""
        safe_file = MockFile("document.txt", "Safe content")

        result = await processor.process_files("Test", [safe_file])

        assert "Safe content" in result
        assert "Security error" not in result


class TestFileSizeLimits:
    """Test file size limit enforcement"""

    @pytest.mark.asyncio
    async def test_blocks_oversized_file(self, processor):
        """Should block files exceeding MAX_FILE_SIZE"""
        # Create file larger than 10MB
        large_content = "A" * (MAX_FILE_SIZE + 1024)
        large_file = MockFile("large.txt", large_content)

        result = await processor.process_files("Test", [large_file])

        assert "File too large" in result
        assert "Security error" in result
        assert "AAAA" not in result  # Content should not be in output

    @pytest.mark.asyncio
    async def test_allows_file_at_size_limit(self, processor):
        """Should allow files exactly at the size limit"""
        # Create file exactly at limit
        content = "B" * (MAX_FILE_SIZE - 100)  # Slightly under to account for encoding
        limit_file = MockFile("limit.txt", content)

        result = await processor.process_files("Test", [limit_file])

        assert "BBBB" in result  # Content should be present
        assert "File too large" not in result

    @pytest.mark.asyncio
    async def test_blocks_total_size_exceeded(self, processor):
        """Should block when total size across all files exceeds limit"""
        # Create files that individually are under MAX_FILE_SIZE
        # but together exceed MAX_TOTAL_SIZE
        file_size = 9 * 1024 * 1024  # 9MB each (under individual limit)
        file1 = MockFile("file1.txt", "X" * file_size)
        file2 = MockFile("file2.txt", "Y" * file_size)
        file3 = MockFile("file3.txt", "Z" * file_size)
        file4 = MockFile("file4.txt", "A" * file_size)
        file5 = MockFile("file5.txt", "B" * file_size)
        file6 = MockFile("file6.txt", "C" * file_size)  # 6*9MB = 54MB > 50MB limit

        result = await processor.process_files("Test", [file1, file2, file3, file4, file5, file6])

        assert "Total file size exceeds limit" in result

    @pytest.mark.asyncio
    async def test_handles_empty_files(self, processor):
        """Should handle empty files gracefully"""
        empty_file = MockFile("empty.txt", "")

        result = await processor.process_files("Test", [empty_file])

        assert "Empty file" in result


class TestCSVFormulaSanitization:
    """Test CSV formula injection prevention"""

    @pytest.mark.asyncio
    async def test_sanitizes_equals_formula(self, processor):
        """Should sanitize formulas starting with ="""
        csv_content = "name,command\nJohn,=cmd|'/c calc'!A1"
        csv_file = MockFile("malicious.csv", csv_content)

        result = await processor.process_files("Test", [csv_file])

        # Formula should be escaped with single quote
        assert "'=cmd" in result or "Sanitized" in result.lower()
        # Original formula should not be executed
        assert result.count("=cmd") == 0 or "'=cmd" in result

    @pytest.mark.asyncio
    async def test_sanitizes_plus_formula(self, processor):
        """Should sanitize formulas starting with +"""
        csv_content = "value\n+1+1"
        csv_file = MockFile("plus.csv", csv_content)

        result = await processor.process_files("Test", [csv_file])

        assert "'+1+1" in result or "Sanitized" in result.lower()

    @pytest.mark.asyncio
    async def test_sanitizes_minus_formula(self, processor):
        """Should sanitize formulas starting with -"""
        csv_content = "value\n-1-1"
        csv_file = MockFile("minus.csv", csv_content)

        result = await processor.process_files("Test", [csv_file])

        assert "'-1-1" in result or "Sanitized" in result.lower()

    @pytest.mark.asyncio
    async def test_sanitizes_at_formula(self, processor):
        """Should sanitize formulas starting with @"""
        csv_content = "value\n@SUM(A1:A10)"
        csv_file = MockFile("at.csv", csv_content)

        result = await processor.process_files("Test", [csv_file])

        assert "'@SUM" in result or "Sanitized" in result.lower()

    @pytest.mark.asyncio
    async def test_preserves_safe_csv_values(self, processor):
        """Should not modify safe CSV values"""
        csv_content = "name,value\nAlice,100\nBob,200"
        csv_file = MockFile("safe.csv", csv_content)

        result = await processor.process_files("Test", [csv_file])

        assert "Alice" in result
        assert "100" in result
        assert "Security error" not in result


class TestInvalidFileHandling:
    """Test handling of invalid file inputs"""

    @pytest.mark.asyncio
    async def test_handles_missing_name_attribute(self, processor):
        """Should handle file objects without .name attribute"""
        class BadFile:
            def read(self):
                return "content"

        bad_file = BadFile()
        result = await processor.process_files("Test", [bad_file])

        assert "Security error" in result or "missing 'name'" in result

    @pytest.mark.asyncio
    async def test_handles_missing_read_method(self, processor):
        """Should handle file objects without .read() method"""
        class NoReadFile:
            name = "file.txt"

        no_read_file = NoReadFile()
        result = await processor.process_files("Test", [no_read_file])

        assert "Security error" in result or "missing 'read()'" in result

    @pytest.mark.asyncio
    async def test_handles_unsupported_extension(self, processor):
        """Should handle unsupported file extensions"""
        unsupported_file = MockFile("virus.exe", b"\x00\x01\x02binary")

        result = await processor.process_files("Test", [unsupported_file])

        assert "Unsupported file type" in result
        assert "virus.exe" in result


class TestTemporaryDirectoryIsolation:
    """Test that files are processed in temporary directories"""

    @pytest.mark.asyncio
    async def test_creates_temporary_directory(self, processor, tmp_path):
        """Should use temporary directory for processing"""
        test_file = MockFile("test.txt", "Test content")

        # Get count of temp directories before
        temp_dir = tempfile.gettempdir()
        temp_items_before = os.listdir(temp_dir)

        result = await processor.process_files("Test", [test_file])

        # Temp directory should be cleaned up after processing
        temp_items_after = os.listdir(temp_dir)

        assert "Test content" in result
        # Temp directories should not accumulate
        assert len(temp_items_after) <= len(temp_items_before) + 1

    @pytest.mark.asyncio
    async def test_file_not_accessible_after_processing(self, processor):
        """Temporary files should be deleted after processing"""
        test_content = "Sensitive data"
        test_file = MockFile("secret.txt", test_content)

        result = await processor.process_files("Test", [test_file])

        # File should be processed
        assert test_content in result

        # But temp file should not exist anymore
        temp_dir = tempfile.gettempdir()
        for root, dirs, files in os.walk(temp_dir):
            assert "secret.txt" not in files


class TestSupportedExtensions:
    """Test supported file extensions"""

    def test_returns_copy_of_extensions(self, processor):
        """Should return copy to prevent external modification"""
        extensions1 = processor.get_supported_extensions()
        extensions2 = processor.get_supported_extensions()

        # Modifying one should not affect the other
        extensions1.append('.exe')

        assert '.exe' not in extensions2
        assert '.exe' not in processor.supported_extensions


class TestMultipleFiles:
    """Test processing multiple files"""

    @pytest.mark.asyncio
    async def test_processes_multiple_safe_files(self, processor):
        """Should process multiple safe files successfully"""
        file1 = MockFile("file1.txt", "Content 1")
        file2 = MockFile("file2.txt", "Content 2")

        result = await processor.process_files("Test", [file1, file2])

        assert "Content 1" in result
        assert "Content 2" in result
        assert "Security error" not in result

    @pytest.mark.asyncio
    async def test_continues_after_one_malicious_file(self, processor):
        """Should process safe files even if one is malicious"""
        malicious = MockFile("../../etc/passwd", "secret")
        safe = MockFile("safe.txt", "Safe content")

        result = await processor.process_files("Test", [malicious, safe])

        assert "Security error" in result  # Malicious file blocked
        assert "Safe content" in result  # Safe file processed
        assert "secret" not in result  # Malicious content not included
