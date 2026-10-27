"""A test that builds ``MultiCollectionChromaStore()`` with the default path lands in a tmp dir.

Incident date: 2026-10-27 (wave-2 follow-up: five non-unit tests opened the cwd-relative
``data/chroma_multi`` and never closed it).  Class: BC-81 (a consumer building its own live store).

Exercises the root conftest's autouse ``_redirect_default_chroma_store`` through the deployed constructor.
"""
import os

from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore


def test_default_construction_is_redirected_to_tmp(tmp_path):
    store = MultiCollectionChromaStore()
    try:
        assert os.path.abspath(store.persist_directory).startswith(str(tmp_path))
        assert not os.path.abspath(store.persist_directory).endswith(os.path.join("data", "chroma_multi"))
    finally:
        store.close()


def test_explicit_path_is_untouched(tmp_path):
    p = str(tmp_path / "mine")
    store = MultiCollectionChromaStore(p)
    try:
        assert store.persist_directory == p
    finally:
        store.close()
