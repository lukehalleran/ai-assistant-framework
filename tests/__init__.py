import shutil
from pathlib import Path

# Ensure Chroma test directory is recreated fresh each run to avoid schema drift
_TEST_CHROMA_PATH = Path("test_chroma_db")
try:
    if _TEST_CHROMA_PATH.exists():
        shutil.rmtree(_TEST_CHROMA_PATH)
except Exception:
    # If cleanup fails we leave the directory in place; tests will surface the issue.
    pass
