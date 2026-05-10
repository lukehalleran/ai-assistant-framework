"""Tests for knowledge/visual_memory_pipeline.py — image ingestion pipeline."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from knowledge.visual_memory_pipeline import VisualMemoryPipeline, CAPTION_PROMPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_clip():
    """Create a mock CLIPManager."""
    clip = MagicMock()
    emb = np.random.randn(512).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    clip.encode_image_from_path.return_value = emb
    clip.encode_image.return_value = emb
    return clip


def _mock_store():
    """Create a mock VisualMemoryStore."""
    store = MagicMock()
    store.has_hash.return_value = False
    store.add_image.return_value = "test-doc-id-123"
    return store


def _mock_model_manager():
    """Create a mock ModelManager with async generate_once."""
    mm = MagicMock()
    mm.generate_once = AsyncMock(return_value="A fluffy orange cat sitting on a windowsill.")
    return mm


def _mock_entity_resolver():
    """Create a mock EntityResolver."""
    resolver = MagicMock()
    resolver.resolve.return_value = None  # Default: no match
    return resolver


def _create_test_image(tmp_path, name="test.png"):
    """Create a small test image file."""
    from PIL import Image
    path = tmp_path / name
    Image.new("RGB", (100, 100), color="red").save(path)
    return str(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIngestImage:
    """Test single image ingestion."""

    @pytest.mark.asyncio
    async def test_ingest_returns_doc_id(self, tmp_path):
        img_path = _create_test_image(tmp_path)
        pipeline = VisualMemoryPipeline(_mock_clip(), _mock_store())
        doc_id = await pipeline.ingest_image(img_path, source="upload")
        assert doc_id == "test-doc-id-123"

    @pytest.mark.asyncio
    async def test_ingest_calls_clip_encode(self, tmp_path):
        img_path = _create_test_image(tmp_path)
        clip = _mock_clip()
        pipeline = VisualMemoryPipeline(clip, _mock_store())
        await pipeline.ingest_image(img_path)
        clip.encode_image_from_path.assert_called_once_with(img_path)

    @pytest.mark.asyncio
    async def test_ingest_calls_store_add(self, tmp_path):
        img_path = _create_test_image(tmp_path)
        store = _mock_store()
        pipeline = VisualMemoryPipeline(_mock_clip(), store)
        await pipeline.ingest_image(img_path, source="obsidian")

        store.add_image.assert_called_once()
        call_kwargs = store.add_image.call_args[1]
        assert call_kwargs["image_path"] == img_path
        assert call_kwargs["source"] == "obsidian"
        assert call_kwargs["clip_embedding"] is not None
        assert call_kwargs["image_hash"] != ""

    @pytest.mark.asyncio
    async def test_ingest_skips_duplicate(self, tmp_path):
        img_path = _create_test_image(tmp_path)
        store = _mock_store()
        store.has_hash.return_value = True  # Already exists
        pipeline = VisualMemoryPipeline(_mock_clip(), store)

        result = await pipeline.ingest_image(img_path)
        assert result is None
        store.add_image.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_missing_file(self, tmp_path):
        pipeline = VisualMemoryPipeline(_mock_clip(), _mock_store())
        result = await pipeline.ingest_image("/nonexistent/image.png")
        assert result is None

    @pytest.mark.asyncio
    async def test_ingest_clip_failure_returns_none(self, tmp_path):
        img_path = _create_test_image(tmp_path)
        clip = _mock_clip()
        clip.encode_image_from_path.return_value = None
        pipeline = VisualMemoryPipeline(clip, _mock_store())

        result = await pipeline.ingest_image(img_path)
        assert result is None

    @pytest.mark.asyncio
    async def test_ingest_fallback_caption_on_no_model(self, tmp_path):
        img_path = _create_test_image(tmp_path, "photo.jpg")
        store = _mock_store()
        pipeline = VisualMemoryPipeline(_mock_clip(), store, model_manager=None)
        await pipeline.ingest_image(img_path)

        caption = store.add_image.call_args[1]["caption"]
        assert "photo.jpg" in caption  # Fallback uses filename


class TestCaptionGeneration:
    """Test vision LLM captioning."""

    @pytest.mark.asyncio
    async def test_caption_with_model_manager(self, tmp_path):
        img_path = _create_test_image(tmp_path)
        store = _mock_store()
        mm = _mock_model_manager()
        pipeline = VisualMemoryPipeline(_mock_clip(), store, model_manager=mm)

        await pipeline.ingest_image(img_path)
        mm.generate_once.assert_called_once()

        caption = store.add_image.call_args[1]["caption"]
        assert "fluffy orange cat" in caption

    @pytest.mark.asyncio
    async def test_caption_passes_image_data(self, tmp_path):
        img_path = _create_test_image(tmp_path, "cat.png")
        mm = _mock_model_manager()
        pipeline = VisualMemoryPipeline(_mock_clip(), _mock_store(), model_manager=mm)

        await pipeline.ingest_image(img_path, media_type="image/png")

        call_kwargs = mm.generate_once.call_args[1]
        assert "images" in call_kwargs
        assert len(call_kwargs["images"]) == 1
        assert call_kwargs["images"][0]["media_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_caption_failure_uses_fallback(self, tmp_path):
        img_path = _create_test_image(tmp_path, "diagram.png")
        store = _mock_store()
        mm = MagicMock()
        mm.generate_once = AsyncMock(side_effect=Exception("API error"))
        pipeline = VisualMemoryPipeline(_mock_clip(), store, model_manager=mm)

        doc_id = await pipeline.ingest_image(img_path)
        assert doc_id is not None  # Should still succeed

        caption = store.add_image.call_args[1]["caption"]
        assert "diagram.png" in caption  # Fallback caption


class TestEntityExtraction:
    """Test entity tagging from caption + context."""

    @pytest.mark.asyncio
    async def test_entities_extracted_from_caption(self, tmp_path):
        img_path = _create_test_image(tmp_path)
        store = _mock_store()
        resolver = _mock_entity_resolver()
        pipeline = VisualMemoryPipeline(_mock_clip(), store, entity_resolver=resolver)

        with patch("memory.graph_utils.extract_graph_entities", return_value={"alice", "bob"}):
            await pipeline.ingest_image(img_path, context_text="Photo of Alice and Bob")

        entity_ids = store.add_image.call_args[1]["entity_ids"]
        assert set(entity_ids) == {"alice", "bob"}

    @pytest.mark.asyncio
    async def test_no_resolver_returns_empty_entities(self, tmp_path):
        img_path = _create_test_image(tmp_path)
        store = _mock_store()
        pipeline = VisualMemoryPipeline(_mock_clip(), store, entity_resolver=None)
        await pipeline.ingest_image(img_path)

        entity_ids = store.add_image.call_args[1]["entity_ids"]
        assert entity_ids == []


class TestBatchIngest:
    """Test batch ingestion."""

    @pytest.mark.asyncio
    async def test_batch_ingest_multiple(self, tmp_path):
        paths = [_create_test_image(tmp_path, f"img{i}.png") for i in range(3)]
        pipeline = VisualMemoryPipeline(_mock_clip(), _mock_store())
        results = await pipeline.ingest_batch(paths, source="obsidian")

        assert len(results) == 3
        assert all(r == "test-doc-id-123" for r in results)

    @pytest.mark.asyncio
    async def test_batch_with_context_texts(self, tmp_path):
        paths = [_create_test_image(tmp_path, f"img{i}.png") for i in range(2)]
        store = _mock_store()
        pipeline = VisualMemoryPipeline(_mock_clip(), store)
        await pipeline.ingest_batch(paths, context_texts=["cat photo", "dog photo"])
        assert store.add_image.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_handles_failures(self, tmp_path):
        paths = [
            _create_test_image(tmp_path, "good.png"),
            "/nonexistent/bad.png",
            _create_test_image(tmp_path, "also_good.png"),
        ]
        pipeline = VisualMemoryPipeline(_mock_clip(), _mock_store())
        results = await pipeline.ingest_batch(paths)

        assert results[0] == "test-doc-id-123"
        assert results[1] == ""  # Failed
        assert results[2] == "test-doc-id-123"


class TestHashComputation:
    """Test image hashing."""

    def test_hash_deterministic(self, tmp_path):
        path = _create_test_image(tmp_path)
        h1 = VisualMemoryPipeline._compute_hash(path)
        h2 = VisualMemoryPipeline._compute_hash(path)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_images_different_hash(self, tmp_path):
        from PIL import Image
        p1 = tmp_path / "a.png"
        p2 = tmp_path / "b.png"
        Image.new("RGB", (100, 100), color="red").save(p1)
        Image.new("RGB", (100, 100), color="blue").save(p2)
        assert VisualMemoryPipeline._compute_hash(str(p1)) != VisualMemoryPipeline._compute_hash(str(p2))


class TestMediaTypeDetection:
    """Test media type detection."""

    def test_png(self):
        assert VisualMemoryPipeline._detect_media_type("photo.png") == "image/png"

    def test_jpeg(self):
        assert VisualMemoryPipeline._detect_media_type("photo.jpg") == "image/jpeg"
        assert VisualMemoryPipeline._detect_media_type("photo.jpeg") == "image/jpeg"

    def test_gif(self):
        assert VisualMemoryPipeline._detect_media_type("anim.gif") == "image/gif"

    def test_webp(self):
        assert VisualMemoryPipeline._detect_media_type("photo.webp") == "image/webp"

    def test_unknown_defaults_to_png(self):
        assert VisualMemoryPipeline._detect_media_type("file.xyz") == "image/png"
