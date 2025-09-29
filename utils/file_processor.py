"""
# utils/file_processor.py

Module Contract
- Purpose: Normalize supported file uploads (txt/csv/py) into text for prompt augmentation.
- Inputs:
  - process_files(user_input: str, files: List[IO]) → concatenated text
- Outputs:
  - Merged text including user input and extracted file contents.
- Side effects:
  - None (reads files only). Errors are captured and embedded as diagnostic text.
"""
import docx2txt
import pandas as pd
from typing import List, Any
from utils.logging_utils import get_logger
logger = get_logger("file_processor")



class FileProcessor:
    """Handles processing of uploaded files"""

    def __init__(self):
        self.supported_extensions = ['.txt', '.docx', '.csv', '.py']

    async def process_files(self, user_text: str, files: List[Any]) -> str:
        """
        Process uploaded files and combine with user text

        Args:
            user_text: Original user input
            files: List of uploaded file objects

        Returns:
            Combined text including file contents
        """
        combined_text = user_text

        if not files:
            return combined_text

        logger.debug(f"Processing {len(files)} uploaded files")

        for file in files:
            try:
                file_content = self._process_single_file(file)
                combined_text += "\n\n" + file_content
                logger.debug(f"Successfully processed: {file.name}")

            except Exception as e:
                error_msg = f"[Error reading {file.name}: {str(e)}]"
                logger.error(f"File processing error: {e}")
                combined_text += "\n\n" + error_msg

        return combined_text

    def _process_single_file(self, file) -> str:
        """Process a single file based on its extension"""

        if file.name.endswith('.txt'):
            with open(file.name, 'r', encoding='utf-8') as f:
                return f.read()

        elif file.name.endswith('.docx'):
            return docx2txt.process(file.name)

        elif file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
            return df.to_string()

        elif file.name.endswith('.py'):
            with open(file.name, 'r', encoding='utf-8') as f:
                return f"```python\n{f.read()}\n```"

        else:
            return f"[Unsupported file type: {file.name}]"

    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions"""
        return self.supported_extensions

