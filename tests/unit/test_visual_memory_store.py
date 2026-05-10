"""Tests for knowledge/visual_memory_store.py — dual FAISS + ChromaDB storage."""

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from knowledge.visual_memory_store import VisualMemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_embedding(dim=512) -> np.ndarray:
    """Generate a random L2-normalized embedding."""
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_store(tmp_path, chroma=None) -> VisualMemoryStore:
    """Create a VisualMemoryStore with temp paths."""
    return VisualMemoryStore(
        chroma_store=chroma,
        data_dir=str(tmp_path),
        index_path=str(tmp_path / "clip_index.faiss"),
        meta_path=str(tmp_path / "clip_metadata.json"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVisualMemoryStoreBasic:
    """Test basic add/search operations."""

    def test_add_image_returns_doc_id(self, tmp_path):
        store = _make_store(tmp_path)
        emb = _random_embedding()
        doc_id = store.add_image(
            image_path="/photos/cat.jpg",
            clip_embedding=emb,
            caption="A fluffy orange cat",
            source="upload",
            entity_ids=["cat", "fluffy"],
            media_type="image/jpeg",
            image_hash="abc123",
        )
        assert doc_id is not None
        assert len(doc_id) == 36  # UUID format

    def test_add_image_increments_count(self, tmp_path):
        store = _make_store(tmp_path)
        store.add_image("/a.jpg", _random_embedding(), "image a", image_hash="h1")
        store.add_image("/b.jpg", _random_embedding(), "image b", image_hash="h2")
        stats = store.get_stats()
        assert stats["total_images"] == 2
        assert stats["faiss_vectors"] == 2

    def test_search_by_clip_returns_results(self, tmp_path):
        store = _make_store(tmp_path)
        emb = _random_embedding()
        store.add_image("/cat.jpg", emb, "a cat photo", image_hash="h1")

        # Search with same embedding should return it
        results = store.search_by_clip(emb, k=5)
        assert len(results) >= 1
        assert results[0]["image_path"] == "/cat.jpg"
        assert results[0]["caption"] == "a cat photo"
        assert results[0]["score"] > 0.9  # same vector → near-1.0

    def test_search_by_clip_similarity_ordering(self, tmp_path):
        store = _make_store(tmp_path)

        # Add two images with different embeddings
        emb_cat = _random_embedding()
        emb_dog = _random_embedding()
        store.add_image("/cat.jpg", emb_cat, "cat", image_hash="h1")
        store.add_image("/dog.jpg", emb_dog, "dog", image_hash="h2")

        # Search with cat embedding — cat should rank higher
        results = store.search_by_clip(emb_cat, k=5)
        assert len(results) >= 1
        assert results[0]["image_path"] == "/cat.jpg"

    def test_search_by_clip_empty_index(self, tmp_path):
        store = _make_store(tmp_path)
        store.load()
        results = store.search_by_clip(_random_embedding(), k=5)
        assert results == []

    def test_search_by_clip_threshold_filter(self, tmp_path):
        store = _make_store(tmp_path)
        store._sim_threshold = 0.99  # Very high threshold

        emb1 = _random_embedding()
        emb2 = _random_embedding()  # Different random vector, low similarity
        store.add_image("/a.jpg", emb1, "image", image_hash="h1")

        results = store.search_by_clip(emb2, k=5)
        # Should be filtered out by high threshold
        assert len(results) == 0 or results[0]["score"] >= 0.99


class TestVisualMemoryStoreDedup:
    """Test deduplication by image hash."""

    def test_duplicate_hash_returns_none(self, tmp_path):
        store = _make_store(tmp_path)
        emb = _random_embedding()
        doc1 = store.add_image("/a.jpg", emb, "first", image_hash="same_hash")
        doc2 = store.add_image("/b.jpg", emb, "second", image_hash="same_hash")
        assert doc1 is not None
        assert doc2 is None  # Duplicate

    def test_duplicate_skips_faiss_add(self, tmp_path):
        store = _make_store(tmp_path)
        store.add_image("/a.jpg", _random_embedding(), "first", image_hash="h1")
        store.add_image("/b.jpg", _random_embedding(), "dup", image_hash="h1")
        assert store.get_stats()["faiss_vectors"] == 1

    def test_has_hash(self, tmp_path):
        store = _make_store(tmp_path)
        store.add_image("/a.jpg", _random_embedding(), "first", image_hash="abc123")
        assert store.has_hash("abc123")
        assert not store.has_hash("xyz789")

    def test_empty_hash_allows_duplicates(self, tmp_path):
        store = _make_store(tmp_path)
        doc1 = store.add_image("/a.jpg", _random_embedding(), "first", image_hash="")
        doc2 = store.add_image("/b.jpg", _random_embedding(), "second", image_hash="")
        assert doc1 is not None
        assert doc2 is not None


class TestVisualMemoryStorePersistence:
    """Test save/load round-trip."""

    def test_save_and_load_round_trip(self, tmp_path):
        # Create store, add image, save
        store1 = _make_store(tmp_path)
        emb = _random_embedding()
        store1.add_image("/cat.jpg", emb, "a cat", source="upload", image_hash="h123")

        # Create new store from same paths, load
        store2 = _make_store(tmp_path)
        store2.load()

        assert store2.get_stats()["total_images"] == 1
        assert store2.get_stats()["faiss_vectors"] == 1
        assert store2.has_hash("h123")

        # Search should work
        results = store2.search_by_clip(emb, k=5)
        assert len(results) >= 1
        assert results[0]["image_path"] == "/cat.jpg"

    def test_load_creates_empty_index_if_no_files(self, tmp_path):
        store = _make_store(tmp_path)
        store.load()
        assert store._loaded
        assert store._index is not None
        assert store._index.ntotal == 0

    def test_save_creates_files(self, tmp_path):
        store = _make_store(tmp_path)
        store.add_image("/a.jpg", _random_embedding(), "test", image_hash="h1")

        index_path = tmp_path / "clip_index.faiss"
        meta_path = tmp_path / "clip_metadata.json"
        assert index_path.exists()
        assert meta_path.exists()

        with open(meta_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["image_path"] == "/a.jpg"

    def test_load_idempotent(self, tmp_path):
        store = _make_store(tmp_path)
        store.load()
        store.load()  # Should not error
        assert store._loaded


class TestVisualMemoryStoreChromaDB:
    """Test ChromaDB text search integration."""

    def test_add_image_stores_to_chroma(self, tmp_path):
        mock_chroma = MagicMock()
        store = _make_store(tmp_path, chroma=mock_chroma)
        store.add_image("/a.jpg", _random_embedding(), "a cat photo", image_hash="h1")

        mock_chroma.add_to_collection.assert_called_once()
        call_args = mock_chroma.add_to_collection.call_args
        assert call_args[0][0] == "visual_memories"
        assert call_args[0][1] == "a cat photo"
        meta = call_args[0][2]
        assert meta["is_image"] is True
        assert meta["image_path"] == "/a.jpg"

    def test_search_by_text_uses_chroma(self, tmp_path):
        mock_chroma = MagicMock()
        mock_chroma.query_collection.return_value = [
            {
                "id": "doc1",
                "content": "a cat sitting on a mat",
                "metadata": {
                    "image_path": "/cat.jpg",
                    "source": "upload",
                    "entity_ids": "cat,mat",
                    "media_type": "image/jpeg",
                    "timestamp": "2026-01-01T00:00:00",
                },
                "relevance_score": 0.85,
            }
        ]
        store = _make_store(tmp_path, chroma=mock_chroma)

        results = store.search_by_text("cat", k=5)
        assert len(results) == 1
        assert results[0]["image_path"] == "/cat.jpg"
        assert results[0]["entity_ids"] == ["cat", "mat"]
        assert results[0]["score"] == 0.85

    def test_search_by_text_no_chroma(self, tmp_path):
        store = _make_store(tmp_path, chroma=None)
        results = store.search_by_text("cat", k=5)
        assert results == []

    def test_chroma_failure_graceful(self, tmp_path):
        mock_chroma = MagicMock()
        mock_chroma.add_to_collection.side_effect = Exception("ChromaDB error")
        store = _make_store(tmp_path, chroma=mock_chroma)

        # Should still succeed (FAISS add works, ChromaDB is non-fatal)
        doc_id = store.add_image("/a.jpg", _random_embedding(), "test", image_hash="h1")
        assert doc_id is not None


class TestVisualMemoryStoreMetadata:
    """Test metadata handling."""

    def test_entity_ids_stored_correctly(self, tmp_path):
        store = _make_store(tmp_path)
        emb = _random_embedding()
        store.add_image(
            "/family.jpg", emb, "family photo",
            entity_ids=["alice", "bob", "charlie"],
            image_hash="h1",
        )

        results = store.search_by_clip(emb, k=1)
        assert results[0]["entity_ids"] == ["alice", "bob", "charlie"]

    def test_source_field_persisted(self, tmp_path):
        store = _make_store(tmp_path)
        emb = _random_embedding()
        store.add_image("/note.png", emb, "diagram", source="obsidian", image_hash="h1")

        results = store.search_by_clip(emb, k=1)
        assert results[0]["source"] == "obsidian"

    def test_get_stats(self, tmp_path):
        store = _make_store(tmp_path)
        store.add_image("/a.jpg", _random_embedding(), "a", source="upload", image_hash="h1")
        store.add_image("/b.png", _random_embedding(), "b", source="obsidian", image_hash="h2")

        stats = store.get_stats()
        assert stats["total_images"] == 2
        assert stats["faiss_vectors"] == 2
        assert stats["unique_sources"] == 2
        assert stats["has_faiss"] is True

    def test_no_faiss_graceful(self, tmp_path):
        """Store should work (metadata-only) even if faiss is unavailable."""
        store = _make_store(tmp_path)
        store._loaded = True  # Prevent auto-load from recreating index
        store._index = None  # Simulate no FAISS

        doc_id = store.add_image("/a.jpg", _random_embedding(), "test", image_hash="h1")
        assert doc_id is not None
        assert store.get_stats()["total_images"] == 1
        assert store.get_stats()["faiss_vectors"] == 0
