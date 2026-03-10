"""
Unit tests for utils/file_processor.py

Tests file processing functionality:
- Supported file types (txt, docx, csv, py)
- File content extraction
- Error handling
- Combined text output
"""

import pytest
import os
import pandas as pd
from unittest.mock import Mock
from utils.file_processor import FileProcessor


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def file_processor():
    """Create a FileProcessor instance"""
    return FileProcessor()


@pytest.fixture
def temp_txt_file(tmp_path):
    """Create a temporary text file"""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Hello from text file!")
    return file_path


@pytest.fixture
def temp_py_file(tmp_path):
    """Create a temporary Python file"""
    file_path = tmp_path / "test.py"
    file_path.write_text("def hello():\n    return 'world'")
    return file_path


@pytest.fixture
def temp_csv_file(tmp_path):
    """Create a temporary CSV file"""
    file_path = tmp_path / "test.csv"
    df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
    df.to_csv(file_path, index=False)
    return file_path


def create_mock_file(name: str, path: str = None):
    """Helper to create a mock file object

    Args:
        name: Filename (without path) for security validation
        path: Full path to the actual file for reading

    Note: After security update (2025-11-30), mock.name should only contain
    the filename without directory components to pass security checks.
    """
    mock_file = Mock()
    # Security: name should only be the filename, not the full path
    mock_file.name = name
    # Store the actual path for file reading if provided
    if path:
        # Read the actual file content
        with open(path, 'rb') as f:
            content = f.read()
        mock_file.read = Mock(return_value=content)
        mock_file.seek = Mock()
    return mock_file


# =============================================================================
# Initialization Tests
# =============================================================================

def test_init_supported_extensions(file_processor):
    """FileProcessor initializes with supported extensions"""
    assert file_processor.supported_extensions == ['.txt', '.docx', '.csv', '.py', '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.webp']


def test_get_supported_extensions(file_processor):
    """get_supported_extensions returns the list"""
    result = file_processor.get_supported_extensions()
    assert result == ['.txt', '.docx', '.csv', '.py', '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.webp']
    assert isinstance(result, list)


# =============================================================================
# Process Files Tests (Async)
# =============================================================================

@pytest.mark.asyncio
async def test_process_files_no_files(file_processor):
    """process_files returns user text when no files provided"""
    result = await file_processor.process_files("Hello", [])
    assert result == "Hello"


@pytest.mark.asyncio
async def test_process_files_empty_file_list(file_processor):
    """process_files handles None files list"""
    result = await file_processor.process_files("Hello", None)
    # Should not crash, returns user text
    assert "Hello" in result


@pytest.mark.asyncio
async def test_process_files_single_txt(file_processor, temp_txt_file):
    """process_files handles single text file"""
    mock_file = create_mock_file("test.txt", str(temp_txt_file))

    result = await file_processor.process_files("User input", [mock_file])

    assert "User input" in result
    assert "Hello from text file!" in result


@pytest.mark.asyncio
async def test_process_files_single_py(file_processor, temp_py_file):
    """process_files handles Python file with code block formatting"""
    mock_file = create_mock_file("test.py", str(temp_py_file))

    result = await file_processor.process_files("Code:", [mock_file])

    assert "Code:" in result
    assert "```python" in result
    assert "def hello():" in result
    assert "```" in result


@pytest.mark.asyncio
async def test_process_files_single_csv(file_processor, temp_csv_file):
    """process_files handles CSV file"""
    mock_file = create_mock_file("test.csv", str(temp_csv_file))

    result = await file_processor.process_files("Data:", [mock_file])

    assert "Data:" in result
    assert "Alice" in result
    assert "Bob" in result


@pytest.mark.asyncio
async def test_process_files_multiple(file_processor, temp_txt_file, temp_py_file):
    """process_files handles multiple files"""
    mock_txt = create_mock_file("test.txt", str(temp_txt_file))
    mock_py = create_mock_file("test.py", str(temp_py_file))

    result = await file_processor.process_files("Files:", [mock_txt, mock_py])

    assert "Files:" in result
    assert "Hello from text file!" in result
    assert "def hello():" in result


@pytest.mark.asyncio
async def test_process_files_unsupported_type(file_processor, tmp_path):
    """process_files handles unsupported file type"""
    # Create an unsupported file
    unsupported = tmp_path / "test.xyz"
    unsupported.write_text("content")
    mock_file = create_mock_file("test.xyz", str(unsupported))

    result = await file_processor.process_files("Input", [mock_file])

    assert "Input" in result
    assert "Unsupported file type" in result
    assert "test.xyz" in result


@pytest.mark.asyncio
async def test_process_files_error_handling(file_processor):
    """process_files handles file read errors gracefully"""
    # Create a mock file that raises an error when read
    mock_file = Mock()
    mock_file.name = "error_file.txt"
    mock_file.read = Mock(side_effect=IOError("Simulated read error"))
    mock_file.seek = Mock()

    result = await file_processor.process_files("Input", [mock_file])

    assert "Input" in result
    # The error should be captured (either as Security error or Error reading)
    assert "error" in result.lower() or "Error" in result


# =============================================================================
# Process Single File Tests
# =============================================================================

def test_process_single_file_txt(file_processor, temp_txt_file):
    """_process_single_file handles text files"""
    mock_file = create_mock_file("test.txt", str(temp_txt_file))

    content, size = file_processor._process_single_file(mock_file)

    assert content == "Hello from text file!"
    assert size == 21


def test_process_single_file_py(file_processor, temp_py_file):
    """_process_single_file handles Python files with formatting"""
    mock_file = create_mock_file("test.py", str(temp_py_file))

    content, size = file_processor._process_single_file(mock_file)

    assert content.startswith("```python\n")
    assert "def hello():" in content
    assert content.endswith("\n```")
    assert size > 0


def test_process_single_file_csv(file_processor, temp_csv_file):
    """_process_single_file handles CSV files"""
    mock_file = create_mock_file("test.csv", str(temp_csv_file))

    content, size = file_processor._process_single_file(mock_file)

    assert "Alice" in content
    assert "Bob" in content
    assert "name" in content
    assert "age" in content
    assert size > 0


def test_process_single_file_unsupported(file_processor):
    """_process_single_file returns message for unsupported types"""
    mock_file = create_mock_file("test.xyz")

    content, size = file_processor._process_single_file(mock_file)

    assert "Unsupported file type" in content
    assert "test.xyz" in content
    assert size == 0


def test_process_single_file_txt_utf8(file_processor, tmp_path):
    """_process_single_file handles UTF-8 encoded text"""
    file_path = tmp_path / "unicode.txt"
    file_path.write_text("Hello 世界 🌍", encoding='utf-8')

    mock_file = create_mock_file("unicode.txt", str(file_path))
    content, size = file_processor._process_single_file(mock_file)

    assert "Hello" in content
    assert "世界" in content
    assert "🌍" in content
    assert size > 0


def test_process_single_file_empty_txt(file_processor, tmp_path):
    """_process_single_file handles empty text file"""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("")

    mock_file = create_mock_file("empty.txt", str(file_path))
    content, size = file_processor._process_single_file(mock_file)

    # Empty files now return a diagnostic message
    assert "[Empty file: empty.txt]" in content
    assert size == 0


def test_process_single_file_empty_csv(file_processor, tmp_path):
    """_process_single_file handles CSV with at least headers"""
    file_path = tmp_path / "minimal.csv"
    # Create CSV with headers but no data
    df = pd.DataFrame(columns=["name", "age"])
    df.to_csv(file_path, index=False)

    mock_file = create_mock_file("minimal.csv", str(file_path))
    content, size = file_processor._process_single_file(mock_file)

    # Should return string representation
    assert isinstance(content, str)
    assert "name" in content or "age" in content or "Empty" in content  # Headers or empty message


# =============================================================================
# PDF Processing Tests
# =============================================================================

def _make_pdf(path, pages_text):
    """Helper to create a real PDF file with given page texts using fpdf2."""
    from fpdf import FPDF
    pdf = FPDF()
    for text in pages_text:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, text)
    pdf.output(str(path))


def test_process_single_file_pdf_single_page(file_processor, tmp_path):
    """_process_single_file extracts text from a single-page PDF"""
    pdf_path = tmp_path / "single.pdf"
    _make_pdf(pdf_path, ["Hello from PDF"])
    mock_file = create_mock_file("single.pdf", str(pdf_path))

    content, size = file_processor._process_single_file(mock_file)

    assert "Hello from PDF" in content
    # Single-page PDFs should NOT have page headers
    assert "## Page" not in content
    assert size > 0


def test_process_single_file_pdf_multi_page(file_processor, tmp_path):
    """_process_single_file adds page headers for multi-page PDFs"""
    pdf_path = tmp_path / "multi.pdf"
    _make_pdf(pdf_path, ["First page content", "Second page content"])
    mock_file = create_mock_file("multi.pdf", str(pdf_path))

    content, size = file_processor._process_single_file(mock_file)

    assert "## Page 1" in content
    assert "## Page 2" in content
    assert "First page content" in content
    assert "Second page content" in content
    assert size > 0


def test_process_single_file_pdf_empty(file_processor, tmp_path):
    """_process_single_file handles PDF with no extractable text"""
    pdf_path = tmp_path / "empty.pdf"
    # Create a PDF with a blank page (no text added)
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.output(str(pdf_path))
    mock_file = create_mock_file("empty.pdf", str(pdf_path))

    content, size = file_processor._process_single_file(mock_file)

    assert "[No text content extracted from empty.pdf]" in content


@pytest.mark.asyncio
async def test_process_files_pdf(file_processor, tmp_path):
    """process_files handles PDF upload end-to-end"""
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path, ["PDF document text"])
    mock_file = create_mock_file("doc.pdf", str(pdf_path))

    result = await file_processor.process_files("Query:", [mock_file])

    assert "Query:" in result
    assert "PDF document text" in result


# =============================================================================
# Edge Cases
# =============================================================================

def test_csv_with_special_characters(file_processor, tmp_path):
    """CSV processing handles special characters"""
    file_path = tmp_path / "special.csv"
    df = pd.DataFrame({"text": ["Hello, world!", "Quote: \"test\"", "Newline\nhere"]})
    df.to_csv(file_path, index=False)

    mock_file = create_mock_file("special.csv", str(file_path))
    content, size = file_processor._process_single_file(mock_file)

    assert isinstance(content, str)
    assert "Hello, world!" in content
    assert size > 0


def test_py_file_with_multiline_code(file_processor, tmp_path):
    """Python file processing preserves formatting"""
    file_path = tmp_path / "complex.py"
    code = """class Example:
    def __init__(self):
        self.value = 42

    def method(self):
        return self.value"""
    file_path.write_text(code)

    mock_file = create_mock_file("complex.py", str(file_path))
    content, size = file_processor._process_single_file(mock_file)

    assert "```python\n" in content
    assert "class Example:" in content
    assert "def __init__" in content
    assert "\n```" in content
    assert size > 0


@pytest.mark.asyncio
async def test_process_files_maintains_order(file_processor, tmp_path):
    """process_files maintains file order in output"""
    file1 = tmp_path / "first.txt"
    file1.write_text("FIRST")
    file2 = tmp_path / "second.txt"
    file2.write_text("SECOND")
    file3 = tmp_path / "third.txt"
    file3.write_text("THIRD")

    mock1 = create_mock_file("first.txt", str(file1))
    mock2 = create_mock_file("second.txt", str(file2))
    mock3 = create_mock_file("third.txt", str(file3))

    result = await file_processor.process_files("START", [mock1, mock2, mock3])

    # Check order is preserved
    start_idx = result.find("START")
    first_idx = result.find("FIRST")
    second_idx = result.find("SECOND")
    third_idx = result.find("THIRD")

    assert start_idx < first_idx < second_idx < third_idx
