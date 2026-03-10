"""
# utils/file_processor.py

Module Contract
- Purpose: Normalize supported file uploads (txt/csv/py/docx/pdf/images) into text and structured data for prompt augmentation.
- Inputs:
  - process_files(user_input: str, files: List[IO]) → concatenated text (backward-compat)
  - process_files_structured(user_input: str, files: List[IO]) → ProcessedFilesResult
- Outputs:
  - Merged text including user input and extracted file contents.
  - ProcessedFilesResult with per-file metadata, images list, and documents list.
- Side effects:
  - Creates temporary files for safe processing (cleaned up automatically)
  - Saves uploaded images to data/uploads/ for persistence
  - Errors are captured and embedded as diagnostic text.

Security Features (Added 2025-11-30):
- Path traversal protection (validates filenames)
- File size limits (10MB max per file)
- CSV formula injection sanitization
- Temporary directory isolation
- Type validation
"""
import os
import base64
import tempfile
import time
import docx2txt
import pandas as pd
from pathlib import Path
from typing import List, Any, Optional
from dataclasses import dataclass, field
from utils.logging_utils import get_logger
from config.app_config import (
    FILE_UPLOAD_MAX_SIZE,
    FILE_UPLOAD_MAX_TOTAL_SIZE,
    FILE_UPLOAD_ALLOWED_EXTENSIONS,
    FILE_UPLOAD_CSV_FORMULA_PREFIXES,
    FILE_UPLOAD_IMAGE_DIR,
)

logger = get_logger("file_processor")

# Security constants (imported from config for centralized management)
MAX_FILE_SIZE = FILE_UPLOAD_MAX_SIZE
MAX_TOTAL_SIZE = FILE_UPLOAD_MAX_TOTAL_SIZE
ALLOWED_EXTENSIONS = FILE_UPLOAD_ALLOWED_EXTENSIONS
CSV_FORMULA_PREFIXES = FILE_UPLOAD_CSV_FORMULA_PREFIXES

# Image file extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}

# Media type mapping for images
IMAGE_MEDIA_TYPES = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
}


@dataclass
class ProcessedFile:
    """Represents a single processed uploaded file."""
    filename: str = ""
    extension: str = ""
    content_text: str = ""       # Extracted text content (empty for images)
    file_path: str = ""          # Persisted path on disk (for images)
    is_image: bool = False
    media_type: str = ""         # MIME type (e.g., 'image/png')
    base64_data: str = ""        # Base64-encoded image data
    file_size: int = 0
    error: str = ""              # Error message if processing failed


@dataclass
class ProcessedFilesResult:
    """Result of structured file processing."""
    text_content: str = ""                          # Combined user text + document text
    processed_files: List[ProcessedFile] = field(default_factory=list)
    images: List[ProcessedFile] = field(default_factory=list)      # Convenience: just the image files
    documents: List[ProcessedFile] = field(default_factory=list)    # Convenience: just the text files


class FileProcessor:
    """Handles processing of uploaded files with security hardening"""

    def __init__(self):
        self.supported_extensions = ALLOWED_EXTENSIONS

    async def process_files(self, user_text: str, files: List[Any]) -> str:
        """
        Process uploaded files and combine with user text (backward-compatible wrapper).

        Args:
            user_text: Original user input
            files: List of uploaded file objects

        Returns:
            Combined text including file contents
        """
        result = await self.process_files_structured(user_text, files)
        return result.text_content

    async def process_files_structured(self, user_text: str, files: List[Any]) -> ProcessedFilesResult:
        """
        Process uploaded files into structured result with separate handling for images and documents.

        Args:
            user_text: Original user input
            files: List of uploaded file objects

        Returns:
            ProcessedFilesResult with text_content, processed_files, images, and documents
        """
        result = ProcessedFilesResult(text_content=user_text)

        if not files:
            return result

        logger.debug(f"Processing {len(files)} uploaded files (structured)")

        # Track total size across all files
        total_size = 0

        for file in files:
            try:
                basename = os.path.basename(getattr(file, 'name', 'unknown'))
                ext = Path(basename).suffix.lower()

                if ext in IMAGE_EXTENSIONS:
                    # Process as image
                    pf = self._process_image_file(file, basename, ext)
                else:
                    # Process as text document
                    pf = self._process_text_file(file)

                # Check total size limit
                total_size += pf.file_size
                if total_size > MAX_TOTAL_SIZE:
                    pf.error = f"Total file size exceeds limit: {total_size}/{MAX_TOTAL_SIZE} bytes"
                    logger.warning(pf.error)
                    result.processed_files.append(pf)
                    result.text_content += f"\n\n[{pf.error}]"
                    break

                result.processed_files.append(pf)

                if pf.error:
                    result.text_content += f"\n\n[Error processing {pf.filename}: {pf.error}]"
                elif pf.is_image:
                    result.images.append(pf)
                    logger.debug(f"Processed image: {pf.filename} ({pf.file_size} bytes)")
                else:
                    result.documents.append(pf)
                    if pf.content_text:
                        result.text_content += "\n\n" + pf.content_text
                    logger.debug(f"Processed document: {pf.filename} ({pf.file_size} bytes)")

            except ValueError as e:
                file_name = getattr(file, 'name', 'unknown_file')
                error_msg = f"[Security error processing {os.path.basename(file_name)}: {str(e)}]"
                logger.error(f"File security validation failed: {e}")
                pf = ProcessedFile(filename=os.path.basename(file_name), error=str(e))
                result.processed_files.append(pf)
                result.text_content += "\n\n" + error_msg

            except Exception as e:
                file_name = getattr(file, 'name', 'unknown_file')
                error_msg = f"[Error reading {file_name}: {str(e)}]"
                logger.error(f"File processing error: {e}")
                pf = ProcessedFile(filename=os.path.basename(file_name), error=str(e))
                result.processed_files.append(pf)
                result.text_content += "\n\n" + error_msg

        logger.info(f"Structured processing complete: {len(result.documents)} documents, {len(result.images)} images")
        return result

    def _process_image_file(self, file, basename: str, ext: str) -> ProcessedFile:
        """
        Process an image file: read bytes, base64 encode, save to data/uploads/.

        Args:
            file: File object
            basename: Filename without path
            ext: File extension (lowercase)

        Returns:
            ProcessedFile with base64_data and file_path populated
        """
        pf = ProcessedFile(
            filename=basename,
            extension=ext,
            is_image=True,
            media_type=IMAGE_MEDIA_TYPES.get(ext, 'application/octet-stream'),
        )

        # Read file bytes using the same security-validated approach as _process_single_file
        file_bytes = self._read_file_bytes(file)
        pf.file_size = len(file_bytes)

        # Enforce size limit
        if pf.file_size > MAX_FILE_SIZE:
            pf.error = f"File too large: {pf.file_size} bytes (max {MAX_FILE_SIZE // (1024*1024)}MB)"
            return pf

        if pf.file_size == 0:
            pf.error = f"Empty image file: {basename}"
            return pf

        # Base64 encode
        pf.base64_data = base64.b64encode(file_bytes).decode('utf-8')

        # Save to data/uploads/ for persistence
        try:
            upload_dir = Path(FILE_UPLOAD_IMAGE_DIR)
            upload_dir.mkdir(parents=True, exist_ok=True)
            # Timestamp prefix to avoid collisions
            safe_name = f"{int(time.time())}_{basename}"
            dest_path = upload_dir / safe_name
            dest_path.write_bytes(file_bytes)
            pf.file_path = str(dest_path)
            logger.debug(f"Saved upload image to {pf.file_path}")
        except Exception as e:
            logger.warning(f"Failed to persist image {basename}: {e}")
            # Non-fatal: image still usable via base64_data for immediate multimodal use

        return pf

    def _read_file_bytes(self, file) -> bytes:
        """Read raw bytes from a file object (Gradio or file-like)."""
        if hasattr(file, 'read') and callable(file.read):
            content = file.read()
            if isinstance(content, str):
                return content.encode('utf-8')
            return content
        else:
            with open(file.name, 'rb') as f:
                return f.read()

    def _process_text_file(self, file) -> ProcessedFile:
        """
        Process a text file using the existing _process_single_file logic.

        Returns:
            ProcessedFile with content_text populated
        """
        basename = os.path.basename(getattr(file, 'name', 'unknown'))
        ext = Path(basename).suffix.lower()
        pf = ProcessedFile(filename=basename, extension=ext)

        try:
            content, file_size = self._process_single_file(file)
            pf.content_text = content
            pf.file_size = file_size
        except ValueError:
            raise  # Re-raise security errors for proper handling upstream
        except Exception as e:
            pf.error = str(e)

        return pf

    def _process_single_file(self, file) -> tuple[str, int]:
        """
        Process a single file based on its extension with security hardening

        Args:
            file: File object with .name and .read() attributes

        Returns:
            Tuple of (file_content: str, file_size: int)

        Raises:
            ValueError: If security validation fails

        Security:
            - Validates filename to prevent path traversal
            - Enforces file size limit
            - Uses temporary directory for processing
            - Sanitizes CSV formula injection
        """
        # SECURITY: Validate filename (prevent path traversal)
        if not hasattr(file, 'name') or not file.name:
            raise ValueError("File object missing 'name' attribute")

        # Extract basename - file.name may be a full path (e.g., from Gradio: /tmp/gradio/.../file.txt)
        # or just a filename. We validate the basename to prevent malicious filenames.
        basename = os.path.basename(file.name)

        # SECURITY: Detect malicious path patterns before checking if it's a legitimate temp path
        # Check for explicit path traversal attempts (../ or ..\)
        if file.name.startswith(('../', '..\\', '../', '..\\')):
            raise ValueError(f"Path traversal detected: filename starts with '..'")

        # Check for path traversal in the middle (/../ or \..\)
        if '/..' in file.name or '\\..' in file.name:
            raise ValueError(f"Path traversal detected: filename contains '/..' or '\\..'")

        # Now differentiate between legitimate temp paths and malicious absolute paths
        # Allow absolute paths ONLY if they're from temp directories (/tmp, /var/tmp, or Windows temp)
        is_temp_path = (
            file.name.startswith('/tmp/') or
            file.name.startswith('/var/tmp/') or
            '\\Temp\\' in file.name or
            '\\AppData\\Local\\Temp\\' in file.name
        )

        # If it starts with / or \ and is NOT a temp path, block it
        if not is_temp_path:
            if file.name.startswith('/') and not file.name.startswith('/tmp'):
                raise ValueError(f"Absolute path not allowed: {file.name}")
            if file.name.startswith('\\') or (len(file.name) >= 3 and file.name[1] == ':'):
                raise ValueError(f"Windows absolute path not allowed: {file.name}")

        # Check for path traversal attempts in the BASENAME
        if '..' in basename:
            raise ValueError(f"Path traversal detected: filename contains '..'")

        if basename.startswith(('/', '\\')):
            raise ValueError(f"Invalid filename: starts with directory separator")

        # Check for Windows absolute path characters in basename (C:, D:, etc.)
        if len(basename) >= 2 and basename[1] == ':':
            raise ValueError(f"Invalid filename: contains drive letter")

        # Validate basename is not empty after extraction
        if not basename or basename in ('.', '..'):
            raise ValueError(f"Invalid filename: {basename}")

        # Validate extension
        file_ext = None
        for ext in ALLOWED_EXTENSIONS:
            if basename.lower().endswith(ext):
                file_ext = ext
                break

        if not file_ext:
            return f"[Unsupported file type: {basename}]", 0

        # SECURITY: Read file content with size limit
        # Handle two types of file objects:
        # 1. File-like objects with .read() method (test mocks, in-memory files)
        # 2. Gradio file objects with .name property pointing to file path

        if hasattr(file, 'read') and callable(file.read):
            # File-like object with read() method
            try:
                file_content = file.read()
            except Exception as e:
                raise ValueError(f"Failed to read file: {e}")
        else:
            # Gradio-style file object - read from file.name path
            # At this point, file.name has been validated by security checks above
            try:
                with open(file.name, 'rb') as f:
                    file_content = f.read()
            except FileNotFoundError:
                raise ValueError(f"File not found: {file.name}")
            except PermissionError:
                raise ValueError(f"Permission denied reading file: {file.name}")
            except Exception as e:
                raise ValueError(f"Failed to read file from path {file.name}: {e}")

        # Handle both bytes and string content
        if isinstance(file_content, bytes):
            file_size = len(file_content)
        elif isinstance(file_content, str):
            file_size = len(file_content.encode('utf-8'))
        else:
            raise ValueError(f"Unexpected file content type: {type(file_content)}")

        # SECURITY: Enforce file size limit
        if file_size > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {file_size} bytes (max {MAX_FILE_SIZE} bytes / "
                f"{MAX_FILE_SIZE // (1024*1024)}MB)"
            )

        if file_size == 0:
            return f"[Empty file: {basename}]", 0

        # Reset file pointer if possible (for re-reading)
        if hasattr(file, 'seek'):
            try:
                file.seek(0)
            except:
                pass  # Not all file objects support seek

        # SECURITY: Process file in temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            safe_path = Path(tmpdir) / basename

            # Write content to temporary file
            if isinstance(file_content, bytes):
                safe_path.write_bytes(file_content)
            else:
                safe_path.write_text(file_content, encoding='utf-8')

            # Process based on extension
            try:
                if file_ext == '.txt':
                    result = safe_path.read_text(encoding='utf-8')

                elif file_ext == '.py':
                    content = safe_path.read_text(encoding='utf-8')
                    result = f"```python\n{content}\n```"

                elif file_ext == '.docx':
                    result = docx2txt.process(str(safe_path))
                    if not result or not result.strip():
                        result = f"[No text content extracted from {basename}]"

                elif file_ext == '.csv':
                    # SECURITY: Sanitize CSV to prevent formula injection
                    df = pd.read_csv(safe_path)

                    # Apply sanitization to all string columns
                    for col in df.columns:
                        if df[col].dtype == 'object':  # String columns
                            df[col] = df[col].apply(self._sanitize_csv_cell)

                    result = df.to_string()

                elif file_ext == '.pdf':
                    import pdfplumber
                    pages = []
                    with pdfplumber.open(safe_path) as pdf:
                        for i, page in enumerate(pdf.pages, 1):
                            text = page.extract_text() or ''
                            if text.strip():
                                pages.append((i, text))
                    if not pages:
                        result = f"[No text content extracted from {basename}]"
                    elif len(pages) == 1:
                        result = pages[0][1]
                    else:
                        # Page headers let chunk_by_headers() split on page boundaries
                        result = '\n\n'.join(f'## Page {num}\n\n{text}' for num, text in pages)

                else:
                    result = f"[Unsupported file type: {basename}]"

            except Exception as e:
                logger.error(f"Error processing {basename}: {e}")
                raise ValueError(f"Failed to process {file_ext} file: {e}")

        return result, file_size

    def _sanitize_csv_cell(self, value: Any) -> Any:
        """
        Sanitize CSV cell value to prevent formula injection

        Args:
            value: Cell value from CSV

        Returns:
            Sanitized value

        Security:
            Escapes values starting with formula characters (=, +, -, @, etc.)
            to prevent execution in spreadsheet applications
        """
        if not isinstance(value, str):
            return value

        # Check if value starts with formula characters
        if value and value[0] in CSV_FORMULA_PREFIXES:
            # Escape by prefixing with single quote
            sanitized = "'" + value
            logger.debug(f"Sanitized CSV formula: {value[:20]}... -> {sanitized[:20]}...")
            return sanitized

        return value

    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions"""
        return self.supported_extensions.copy()  # Return copy to prevent modification
