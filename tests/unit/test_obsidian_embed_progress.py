"""Regression for the notes-sync progress log (2026-09-14/16, BC-80 batch).

`embed_vault` logged progress on `result.embedded_files % 50 == 0`, which is
true for EVERY updated or skipped note while `embedded_files` stays 0 — the
2026-09-14 sync printed "Embedded 0/810 files..." once per updated note.
Progress is now counted in `processed_files`, incremented in a `finally` so
the skip `continue`s and the error path count too, and `processed_files`
must equal `total_files` when the loop ends.

Drives the DEPLOYED `ObsidianManager.embed_vault` over a temp vault with a
MagicMock store (no ChromaDB, embedder or network).
"""
import logging
from unittest.mock import MagicMock

import pytest

from knowledge.obsidian_manager import ObsidianManager


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "keep.md").write_text("# Keep\n\nA real note with content.\n", encoding="utf-8")
    (tmp_path / "blank.md").write_text("   \n\n", encoding="utf-8")  # skipped as empty
    (tmp_path / "broken.md").write_text("# Broken\n\nThis one will fail to chunk.\n", encoding="utf-8")
    return tmp_path


def _manager(vault):
    store = MagicMock()
    # No pre-existing index: the mtime scan raises → warning → empty map.
    store._get_collection.return_value.get.side_effect = RuntimeError("no index yet")
    return ObsidianManager(chroma_store=store, vault_path=str(vault))


def test_processed_files_counts_embedded_skipped_and_errored(vault, monkeypatch, caplog):
    mgr = _manager(vault)
    real_chunk = mgr._chunk_by_headers

    def _chunk(content, title):
        if title == "broken":
            raise ValueError("chunker exploded")
        return real_chunk(content, title)

    monkeypatch.setattr(mgr, "_chunk_by_headers", _chunk)
    with caplog.at_level(logging.INFO, logger="knowledge.obsidian_manager"):
        result = mgr.embed_vault(force_reindex=False)

    assert result.total_files == 3
    assert result.processed_files == 3  # keep + blank (skipped) + broken (errored)
    assert result.embedded_files == 1
    assert result.skipped_files == 1
    assert len(result.errors) == 1 and "broken" in result.errors[0]
    # The final progress line fires exactly once, on the LAST processed file,
    # and reports processed — never the old "Embedded 0/N" per-note spam.
    progress = [r.getMessage() for r in caplog.records if "Processed " in r.getMessage()]
    assert progress == [
        "[Obsidian] Processed 3/3 files (1 new, 0 updated, 1 skipped)..."
    ]
    assert not any("Embedded 0/" in r.getMessage() for r in caplog.records)


def test_updated_only_run_logs_once_not_per_note(vault, monkeypatch, caplog):
    """The 2026-09-14 shape: every file is an UPDATE (embedded_files stays 0)."""
    (vault / "blank.md").unlink()
    (vault / "broken.md").unlink()
    for i in range(3):
        (vault / f"n{i}.md").write_text(f"# N{i}\n\nbody {i}\n", encoding="utf-8")
    mgr = _manager(vault)
    # Every file is already indexed with an OLDER mtime → all updates.
    monkeypatch.setattr(mgr, "_delete_file_chunks", lambda rel: None)
    store = mgr.chroma_store
    store._get_collection.return_value.get.side_effect = None
    store._get_collection.return_value.get.return_value = {
        "metadatas": [{"file_path": p.name, "file_mtime": 0.0} for p in vault.glob("*.md")]
    }
    with caplog.at_level(logging.INFO, logger="knowledge.obsidian_manager"):
        result = mgr.embed_vault(force_reindex=False)
    assert result.embedded_files == 0 and result.updated_files == 4
    assert result.processed_files == result.total_files == 4
    progress = [r.getMessage() for r in caplog.records if "Processed " in r.getMessage()]
    assert len(progress) == 1
    assert progress[0].startswith("[Obsidian] Processed 4/4 files (0 new, 4 updated")
