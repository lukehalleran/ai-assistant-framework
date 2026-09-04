"""In-process state for the single-user Daemon API.

AppState holds the orchestrator reference, the one ChatSession (this is a
single-user, localhost-bound app — no cookies/multi-session), and the upload
registry mapping file_id → temp path. Session state here is UI state only;
clearing it never touches stored memory (ChromaDB/corpus).
"""

import asyncio
import os
import uuid
from types import SimpleNamespace
from typing import Dict, List, Optional

from utils.logging_utils import get_logger

logger = get_logger("api_state")


class ChatSession:
    """The single chat session: history in gr.Chatbot 'messages' format."""

    def __init__(self):
        self.history: List[dict] = []
        self.debug_records: List[dict] = []
        self.pending_action_id: Optional[str] = None
        self.personality: str = "default"
        # In-flight guard: one streaming turn at a time (409 on overlap)
        self.stream_lock = asyncio.Lock()

    def clear(self):
        """Reset UI state only — stored memory is never touched here."""
        self.history = []
        self.debug_records = []
        self.pending_action_id = None


class AppState:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.session = ChatSession()
        self._uploads: Dict[str, dict] = {}

    # ---- upload registry -------------------------------------------------
    def register_upload(self, path: str, name: str, size: int) -> str:
        file_id = uuid.uuid4().hex
        self._uploads[file_id] = {"path": path, "name": name, "size": size}
        return file_id

    def resolve_uploads(self, file_ids: List[str]) -> List[SimpleNamespace]:
        """Resolve file_ids to shim objects with `.name` (the FileProcessor contract)."""
        files = []
        for fid in file_ids or []:
            entry = self._uploads.get(fid)
            if not entry:
                logger.warning(f"[API] Unknown upload file_id: {fid}")
                continue
            if not os.path.exists(entry["path"]):
                logger.warning(f"[API] Upload temp file missing: {entry['path']}")
                continue
            files.append(SimpleNamespace(name=entry["path"], orig_name=entry["name"]))
        return files

    def unregister_upload(self, file_id: str) -> None:
        """Remove an upload registration after a failed batch request."""
        self._uploads.pop(file_id, None)
