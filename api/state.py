"""In-process state for the single-user Daemon API.

AppState holds the orchestrator reference, the one ChatSession (this is a
single-user, localhost-bound app — no cookies/multi-session), and the upload
registry mapping file_id → temp path. Session state here is UI state only;
clearing it never touches stored memory (ChromaDB/corpus).
"""

import threading
import time
import asyncio
import os
import uuid
from types import SimpleNamespace
from typing import Dict, List, Optional

from core.active_document import ActiveDocumentRegistry
from utils.logging_utils import get_logger

logger = get_logger("api_state")


class NotesSyncState:
    """Retained outcome of the notes-sync background job (BC-80).

    The worker thread outlives the HTTP request that started it, so a client
    whose fetch dropped (backgrounded tab, network blip) reads the real
    outcome from ``GET /api/sync-notes/status`` after a reload instead of
    showing a failure card for work that finished. Process memory only —
    nothing is persisted; a restart starts from ``idle``.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.status = "idle"  # idle | running | succeeded | failed
        self.message: Optional[str] = None
        self.error: Optional[str] = None
        self.task_id: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        # Terminal payload of the newest finished job (kept when a new job starts).
        self.last_result: Optional[dict] = None
        self.worker: Optional[threading.Thread] = None

    def try_start(self, task_id: str) -> bool:
        """Claim the single flight. False when a job is already running."""
        with self.lock:
            if self.status == "running":
                return False
            self.status = "running"
            self.task_id = task_id
            self.message = "Notes sync started"
            self.error = None
            self.started_at = time.time()
            self.finished_at = None
            return True

    def finish(
        self,
        task_id: str,
        status: str,
        message: str,
        *,
        error: Optional[str] = None,
        result: Optional[dict] = None,
    ) -> None:
        """Record a terminal outcome. Ignored unless ``task_id`` is the current job."""
        with self.lock:
            if self.task_id != task_id:
                return
            self.status = status
            self.message = message
            self.error = error
            self.finished_at = time.time()
            self.last_result = {
                "task_id": task_id,
                "status": status,
                "message": message,
                "error": error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "result": result,
            }

    def snapshot(self) -> dict:
        """JSON-ready view. ``server_time`` lets a client judge whether a
        retained outcome predates a request whose response it lost."""
        with self.lock:
            return {
                "status": self.status,
                "message": self.message,
                "error": self.error,
                "task_id": self.task_id,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "server_time": time.time(),
                "last_result": self.last_result,
            }


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
        # Active-document registry (2026-09-08, B5) lives on AppState (the
        # session is single-user) and is attached here for convenience; a
        # session reset must clear this session's attachments too.
        registry = getattr(self, "active_documents", None)
        if registry is not None:
            registry.clear()


class AppState:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.session = ChatSession()
        self._uploads: Dict[str, dict] = {}
        # Retained notes-sync outcome (BC-80); see NotesSyncState.
        self.notes_sync = NotesSyncState()
        # Bounded active-document continuity (2026-09-08, B5): an in-memory
        # registry of this session's attached documents + numbered-item
        # navigation state, NOT persisted across a restart. Referenced from
        # the session for convenience (ChatSession.clear() above) and
        # attached to the orchestrator so gui/handlers.py and the agentic
        # controller can reach it without threading a new parameter through
        # every call site.
        self.active_documents = ActiveDocumentRegistry()
        self.session.active_documents = self.active_documents
        if orchestrator is not None:
            setattr(orchestrator, "active_documents", self.active_documents)

    # ---- upload registry -------------------------------------------------
    def register_upload(self, path: str, name: str, size: int) -> str:
        file_id = uuid.uuid4().hex
        self._uploads[file_id] = {"path": path, "name": name, "size": size}
        return file_id

    def resolve_uploads(self, file_ids: List[str]) -> List[SimpleNamespace]:
        """Resolve file_ids to shim objects with `.name` (the FileProcessor contract).

        Dedupes file_ids first, preserving order (2026-09-04, homework-
        attachment turn audit item 1): a repeated id in the request — e.g. a
        client-side double-fire of the attach handler that resent the same
        batch — would otherwise resolve to the same upload twice, and every
        downstream consumer (FileProcessor) has no way to know it's the same
        file rather than two files with identical content.
        """
        files = []
        for fid in dict.fromkeys(file_ids or []):
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
