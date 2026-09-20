"""BC-81: the context gatherer's lazily-built managers must reuse the live store.

Until 2026-09-19 `ContextGatherer` built ObsidianManager(), ReferenceDocsManager()
and WebSearchManager() with no store, so each lazily constructed its OWN
MultiCollectionChromaStore (a second Chroma client + another bge embedder on the
GPU). A live process logged the embedder load four times. These tests drive the
real lazy properties — every existing test sets the private `_…_manager`
attribute to a mock and never reached this path.
"""
from unittest.mock import MagicMock, patch

import pytest

from core.prompt.context_gatherer import ContextGatherer


class _Store:
    """Identity-only stand-in for the live MultiCollectionChromaStore."""


@pytest.fixture
def gatherer_and_store():
    store = _Store()
    coordinator = MagicMock()
    coordinator.chroma_store = store
    gatherer = ContextGatherer(
        memory_coordinator=coordinator,
        model_manager=MagicMock(),
        token_manager=MagicMock(),
        gate_system=MagicMock(),
        time_manager=MagicMock(),
    )
    return gatherer, store


def test_obsidian_manager_gets_the_live_store(gatherer_and_store):
    gatherer, store = gatherer_and_store
    with patch("config.app_config.OBSIDIAN_ENABLED", True):
        manager = gatherer.obsidian_manager
    assert manager is not None
    assert manager._chroma_store is store
    assert manager.chroma_store is store  # the lazy property must not build another one


def test_reference_docs_manager_gets_the_live_store(gatherer_and_store):
    gatherer, store = gatherer_and_store
    with patch("config.app_config.REFERENCE_DOCS_ENABLED", True):
        manager = gatherer.reference_docs_manager
    assert manager is not None
    assert manager.chroma_store is store


def test_web_search_cache_gets_the_live_store(gatherer_and_store):
    gatherer, store = gatherer_and_store
    manager = gatherer.web_search_manager
    assert manager is not None
    assert manager.cache._store is store


def test_no_store_is_constructed_by_any_lazy_manager(gatherer_and_store):
    gatherer, _store = gatherer_and_store
    with patch(
        "memory.storage.multi_collection_chroma_store.MultiCollectionChromaStore",
        side_effect=AssertionError("a second store was constructed (BC-81)"),
    ), patch("config.app_config.OBSIDIAN_ENABLED", True), \
            patch("config.app_config.REFERENCE_DOCS_ENABLED", True):
        assert gatherer.obsidian_manager.chroma_store is _store
        assert gatherer.reference_docs_manager.chroma_store is _store
        gatherer.web_search_manager.cache._ensure_initialized()  # no `client` attr -> no collection, and no new store


def test_without_a_coordinator_store_the_old_fallback_still_works():
    coordinator = object()  # no chroma_store attribute (CLI / partial stacks)
    gatherer = ContextGatherer(
        memory_coordinator=coordinator,
        model_manager=MagicMock(),
        token_manager=MagicMock(),
        gate_system=MagicMock(),
        time_manager=MagicMock(),
    )
    assert gatherer._live_chroma_store() is None
