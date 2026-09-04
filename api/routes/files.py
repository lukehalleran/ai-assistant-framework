"""File uploads for the chat: POST /api/uploads (multipart, ≤100MB total).

Files are written to a temp dir and registered by file_id; the chat request
references them via ChatRequest.file_ids. Actual parsing/validation happens in
the existing security-hardened FileProcessor inside handle_submit.
"""

import os
import tempfile
from typing import List

from fastapi import APIRouter, HTTPException, Request, UploadFile

from api.schemas import UploadedFileInfo, UploadResult
from utils.logging_utils import get_logger

logger = get_logger("api_routes")

router = APIRouter(prefix="/api", tags=["files"])

MAX_TOTAL_BYTES = 100 * 1024 * 1024  # match the Gradio 100mb cap
_READ_CHUNK_BYTES = 1024 * 1024
_UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "daemon_api_uploads")


@router.post("/uploads", response_model=UploadResult)
async def upload_files(request: Request, files: List[UploadFile]):
    state = request.app.state.daemon
    os.makedirs(_UPLOAD_DIR, exist_ok=True)

    results = []
    created_paths = []
    total = 0
    try:
        for f in files:
            # Keep the original extension so FileProcessor type-routing works;
            # never trust the client filename for the path itself.
            base = os.path.basename(f.filename or "upload")
            suffix = os.path.splitext(base)[1][:16]
            fd, path = tempfile.mkstemp(suffix=suffix, dir=_UPLOAD_DIR)
            created_paths.append(path)
            size = 0
            with os.fdopen(fd, "wb") as out:
                while True:
                    chunk = await f.read(_READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    size += len(chunk)
                    total += len(chunk)
                    if total > MAX_TOTAL_BYTES:
                        raise HTTPException(status_code=413, detail="Upload exceeds 100MB total limit.")
                    out.write(chunk)

            file_id = state.register_upload(path=path, name=base, size=size)
            results.append(UploadedFileInfo(file_id=file_id, name=base, size=size))
            logger.info(f"[API] Upload registered: {base} ({size} bytes) -> {file_id}")
    except BaseException:
        # A rejected multi-file request must not leave unregistered temp files
        # behind (including the partially written file that exceeded the cap).
        # Cancellation is a BaseException on supported Python versions, and a
        # disconnected client needs the same transactional cleanup.
        for path in created_paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        for info in results:
            state.unregister_upload(info.file_id)
        raise

    return UploadResult(files=results)
