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
_UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "daemon_api_uploads")


@router.post("/uploads", response_model=UploadResult)
async def upload_files(request: Request, files: List[UploadFile]):
    state = request.app.state.daemon
    os.makedirs(_UPLOAD_DIR, exist_ok=True)

    results = []
    total = 0
    for f in files:
        data = await f.read()
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise HTTPException(status_code=413, detail="Upload exceeds 100MB total limit.")

        # Keep the original extension so FileProcessor type-routing works;
        # never trust the client filename for the path itself.
        base = os.path.basename(f.filename or "upload")
        suffix = os.path.splitext(base)[1][:16]
        fd, path = tempfile.mkstemp(suffix=suffix, dir=_UPLOAD_DIR)
        with os.fdopen(fd, "wb") as out:
            out.write(data)

        file_id = state.register_upload(path=path, name=base, size=len(data))
        results.append(UploadedFileInfo(file_id=file_id, name=base, size=len(data)))
        logger.info(f"[API] Upload registered: {base} ({len(data)} bytes) -> {file_id}")

    return UploadResult(files=results)
