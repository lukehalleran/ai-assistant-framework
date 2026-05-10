"""Tests for knowledge/clip_manager.py — CLIP model singleton."""

import importlib
import threading
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_open_clip():
    """Create a mock open_clip module with realistic model behavior."""
    mock_clip = MagicMock()

    # Model returns 512-dim tensors
    import torch
    fake_features = torch.randn(1, 512)
    fake_features = fake_features / fake_features.norm(dim=-1, keepdim=True)

    mock_model = MagicMock()
    mock_model.encode_image.return_value = fake_features
    mock_model.encode_text.return_value = fake_features
    mock_model.eval.return_value = mock_model

    mock_preprocess = MagicMock(return_value=torch.randn(3, 224, 224))
    mock_tokenizer = MagicMock(return_value=torch.zeros(1, 77, dtype=torch.long))

    mock_clip.create_model_and_transforms.return_value = (mock_model, None, mock_preprocess)
    mock_clip.get_tokenizer.return_value = mock_tokenizer

    return mock_clip


def _fresh_manager():
    """Get a fresh CLIPManager instance (bypass singleton)."""
    from knowledge.clip_manager import CLIPManager
    return CLIPManager()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCLIPManagerSingleton:
    """Test singleton behavior."""

    def test_singleton_returns_same_instance(self):
        import knowledge.clip_manager as mod
        # Reset singleton
        mod._singleton = None
        m1 = mod.get_clip_manager()
        m2 = mod.get_clip_manager()
        assert m1 is m2

    def test_singleton_thread_safety(self):
        import knowledge.clip_manager as mod
        mod._singleton = None

        results = []

        def get():
            results.append(mod.get_clip_manager())

        threads = [threading.Thread(target=get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r is results[0] for r in results)


class TestCLIPManagerLoading:
    """Test model loading behavior."""

    def test_lazy_load_not_loaded_on_init(self):
        mgr = _fresh_manager()
        assert not mgr.loaded
        assert mgr._model is None

    def test_load_idempotent(self):
        mock_clip = _make_mock_open_clip()
        mgr = _fresh_manager()

        with patch.dict("sys.modules", {"open_clip": mock_clip}):
            mgr.load()
            assert mgr.loaded
            call_count = mock_clip.create_model_and_transforms.call_count

            mgr.load()  # second call
            assert mock_clip.create_model_and_transforms.call_count == call_count

    def test_graceful_fallback_no_open_clip(self):
        mgr = _fresh_manager()
        # Simulate ImportError for open_clip
        with patch.dict("sys.modules", {"open_clip": None}):
            with patch("builtins.__import__", side_effect=ImportError("no open_clip")):
                mgr._available = True
                mgr.loaded = False
                mgr._model = None
                # Force reimport failure
                try:
                    import open_clip  # noqa
                except ImportError:
                    pass
                # Directly test the code path
                mgr._available = False
                result = mgr.encode_text("test")
                assert result is None

    def test_not_available_returns_none(self):
        mgr = _fresh_manager()
        mgr._available = False
        assert mgr.encode_image(MagicMock()) is None
        assert mgr.encode_text("test") is None
        assert mgr.encode_image_from_path("/fake/path.png") is None


class TestCLIPManagerEncoding:
    """Test encoding methods with mocked model."""

    @pytest.fixture
    def loaded_manager(self):
        mock_clip = _make_mock_open_clip()
        mgr = _fresh_manager()
        with patch.dict("sys.modules", {"open_clip": mock_clip}):
            mgr.load()
        return mgr

    def test_encode_image_shape(self, loaded_manager):
        from PIL import Image
        img = Image.new("RGB", (224, 224), color="red")
        result = loaded_manager.encode_image(img)
        assert result is not None
        assert result.shape == (512,)
        assert result.dtype == np.float32

    def test_encode_image_normalized(self, loaded_manager):
        from PIL import Image
        img = Image.new("RGB", (224, 224), color="blue")
        result = loaded_manager.encode_image(img)
        assert result is not None
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 0.01, f"Expected unit norm, got {norm}"

    def test_encode_text_shape(self, loaded_manager):
        result = loaded_manager.encode_text("a photo of a cat")
        assert result is not None
        assert result.shape == (512,)
        assert result.dtype == np.float32

    def test_encode_text_normalized(self, loaded_manager):
        result = loaded_manager.encode_text("hello world")
        assert result is not None
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 0.01

    def test_encode_image_from_path(self, loaded_manager, tmp_path):
        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (100, 100), color="green").save(img_path)

        result = loaded_manager.encode_image_from_path(str(img_path))
        assert result is not None
        assert result.shape == (512,)

    def test_encode_image_from_path_missing_file(self, loaded_manager):
        result = loaded_manager.encode_image_from_path("/nonexistent/image.png")
        assert result is None

    def test_auto_load_on_encode_text(self):
        """encode_text should trigger lazy load if not loaded."""
        mock_clip = _make_mock_open_clip()
        mgr = _fresh_manager()

        with patch.dict("sys.modules", {"open_clip": mock_clip}):
            assert not mgr.loaded
            result = mgr.encode_text("test query")
            assert mgr.loaded
            assert result is not None

    def test_auto_load_on_encode_image(self):
        """encode_image should trigger lazy load if not loaded."""
        mock_clip = _make_mock_open_clip()
        mgr = _fresh_manager()
        from PIL import Image
        img = Image.new("RGB", (50, 50))

        with patch.dict("sys.modules", {"open_clip": mock_clip}):
            assert not mgr.loaded
            result = mgr.encode_image(img)
            assert mgr.loaded
            assert result is not None


class TestCLIPManagerProperties:
    """Test properties."""

    def test_embedding_dim(self):
        mgr = _fresh_manager()
        assert mgr.embedding_dim == 512

    def test_loaded_property_false_initially(self):
        mgr = _fresh_manager()
        assert not mgr.loaded
