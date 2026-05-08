"""
Persistence state fingerprinting and diffing for eval safety.

PersistenceGuard captures fingerprints of all mutable persistent state
(ChromaDB collections, JSON files, graph, profile, etc.) before and after
eval generation. If anything changed, assert_same_as() raises.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.schema import PersistenceSnapshot, StoreFingerprint


# Default data paths relative to project root
_DEFAULT_DATA_PATHS = [
    "data/knowledge_graph.json",
    "data/entity_aliases.json",
    "data/claim_index.json",
    "data/surfacing_history.json",
    "data/category_cache.json",
]


class PersistenceGuard:
    """Captures and compares persistent state fingerprints.

    Usage:
        guard = PersistenceGuard(chromadb_client=client)
        before = guard.capture()
        # ... run eval generation ...
        after = guard.capture()
        before.assert_same_as(after)  # Raises if anything changed
    """

    def __init__(
        self,
        chromadb_client: Any = None,
        data_paths: Optional[List[Path]] = None,
        extra_paths: Optional[List[Path]] = None,
    ):
        """
        Args:
            chromadb_client: ChromaDB client for collection count fingerprints.
                Can be None (skips ChromaDB fingerprinting).
            data_paths: JSON/data file paths to fingerprint.
                Defaults to the standard Daemon data files.
            extra_paths: Additional file paths to fingerprint.
        """
        self._chromadb_client = chromadb_client

        if data_paths is not None:
            self._data_paths = [Path(p) for p in data_paths]
        else:
            self._data_paths = [Path(p) for p in _DEFAULT_DATA_PATHS]

        if extra_paths:
            self._data_paths.extend(Path(p) for p in extra_paths)

    def capture(self) -> PersistenceSnapshot:
        """Capture current state of all monitored stores.

        Returns:
            PersistenceSnapshot with fingerprints for all reachable stores.
        """
        fingerprints: Dict[str, StoreFingerprint] = {}

        # ChromaDB collections
        if self._chromadb_client is not None:
            chromadb_fps = self._fingerprint_chromadb()
            fingerprints.update(chromadb_fps)

        # Data files
        for path in self._data_paths:
            fp = self._fingerprint_file(path)
            if fp is not None:
                fingerprints[f"file:{path}"] = fp

        return PersistenceSnapshot(
            snapshot_id=str(uuid.uuid4())[:8],
            capture_timestamp=datetime.now(timezone.utc).isoformat(),
            fingerprints=fingerprints,
        )

    def _fingerprint_chromadb(self) -> Dict[str, StoreFingerprint]:
        """Fingerprint all ChromaDB collections."""
        fps: Dict[str, StoreFingerprint] = {}

        try:
            collections = self._chromadb_client.list_collections()
        except Exception:
            return fps

        for col in collections:
            try:
                name = col.name if hasattr(col, "name") else str(col)
                count = col.count() if hasattr(col, "count") else None
                fps[f"chromadb:{name}"] = StoreFingerprint(
                    name=name,
                    kind="chromadb_collection",
                    count=count,
                )
            except Exception:
                continue

        return fps

    def _fingerprint_file(self, path: Path) -> Optional[StoreFingerprint]:
        """Fingerprint a single file (size, mtime, sha256)."""
        if not path.exists():
            return None

        try:
            stat = path.stat()
            size_bytes = stat.st_size
            mtime_ns = stat.st_mtime_ns

            # Hash files up to 50MB; skip larger ones (use size+mtime only)
            sha256 = None
            if size_bytes <= 50 * 1024 * 1024:
                sha256 = _file_hash(path)

            return StoreFingerprint(
                name=str(path),
                kind="json_file",
                size_bytes=size_bytes,
                mtime_ns=mtime_ns,
                sha256=sha256,
            )
        except OSError:
            return None


def _file_hash(path: Path) -> str:
    """SHA-256 hash of a file, truncated to 16 hex chars."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]
