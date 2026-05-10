"""
# knowledge/clip_manager.py

Module Contract
- Purpose: Lazy-loaded singleton for OpenCLIP model. Provides cross-modal encoding
  of images and text into a shared 512-dim embedding space for visual memory retrieval.
- Class: CLIPManager
- Key methods:
  - load() -> None: Lazy, idempotent model loading. No-ops if already loaded.
  - encode_image(image: PIL.Image.Image) -> Optional[np.ndarray]: Encode image to 512-dim L2-normalized vector.
  - encode_text(text: str) -> Optional[np.ndarray]: Encode text to 512-dim L2-normalized vector.
  - encode_image_from_path(path: str) -> Optional[np.ndarray]: Load image from disk and encode.
- Module-level accessor:
  - get_clip_manager() -> CLIPManager: Thread-safe singleton accessor.
- Dependencies:
  - open_clip_torch (optional — graceful fallback if not installed)
  - PIL/Pillow (already available)
  - torch (already available)
  - config.app_config (VISUAL_MEMORY_CLIP_MODEL, VISUAL_MEMORY_CLIP_PRETRAINED)
- Side effects:
  - Loads ~400MB model into RAM on first use (CPU).
  - Model stays resident for process lifetime.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from utils.logging_utils import get_logger

logger = get_logger("knowledge.clip_manager")

# Singleton state
_singleton_lock = threading.Lock()
_singleton: Optional["CLIPManager"] = None


class CLIPManager:
    """
    Lazy-loaded OpenCLIP model for cross-modal image/text encoding.

    Thread-safe via module-level lock for singleton creation.
    Model loading is idempotent — safe to call load() multiple times.
    """

    EMBEDDING_DIM = 512

    def __init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = "cpu"
        self.loaded = False
        self._available = True  # False if open_clip not installed

    def load(self) -> None:
        """Load the CLIP model. Idempotent — no-ops if already loaded."""
        if self.loaded:
            return

        try:
            import open_clip
            import torch
        except ImportError:
            logger.warning(
                "[CLIPManager] open_clip_torch not installed — visual memory disabled. "
                "Install with: pip install open_clip_torch"
            )
            self._available = False
            return

        try:
            from config.app_config import (
                VISUAL_MEMORY_CLIP_MODEL,
                VISUAL_MEMORY_CLIP_PRETRAINED,
            )
            model_name = VISUAL_MEMORY_CLIP_MODEL
            pretrained = VISUAL_MEMORY_CLIP_PRETRAINED
        except ImportError:
            model_name = "ViT-B-32"
            pretrained = "openai"

        try:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                f"[CLIPManager] Loading {model_name} (pretrained={pretrained}) on {self._device}..."
            )

            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self._device
            )
            model.eval()

            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(model_name)
            self.loaded = True

            logger.info(f"[CLIPManager] Model loaded successfully (dim={self.EMBEDDING_DIM})")
        except Exception as e:
            logger.error(f"[CLIPManager] Failed to load model: {e}")
            self._available = False

    def encode_image(self, image) -> Optional[np.ndarray]:
        """
        Encode a PIL Image to a 512-dim L2-normalized embedding.

        Args:
            image: PIL.Image.Image instance

        Returns:
            np.ndarray of shape (512,) or None on failure
        """
        if not self._available:
            return None
        if not self.loaded:
            self.load()
        if not self.loaded:
            return None

        try:
            import torch

            preprocessed = self._preprocess(image).unsqueeze(0).to(self._device)
            with torch.no_grad():
                features = self._model.encode_image(preprocessed)
                features = features / features.norm(dim=-1, keepdim=True)

            return features.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            logger.warning(f"[CLIPManager] encode_image failed: {e}")
            return None

    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """
        Encode text to a 512-dim L2-normalized embedding.

        Args:
            text: Natural language query string

        Returns:
            np.ndarray of shape (512,) or None on failure
        """
        if not self._available:
            return None
        if not self.loaded:
            self.load()
        if not self.loaded:
            return None

        try:
            import torch

            tokens = self._tokenizer([text]).to(self._device)
            with torch.no_grad():
                features = self._model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)

            return features.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            logger.warning(f"[CLIPManager] encode_text failed: {e}")
            return None

    def encode_image_from_path(self, path: str) -> Optional[np.ndarray]:
        """Load an image from disk and encode it."""
        try:
            from PIL import Image

            image = Image.open(path).convert("RGB")
            return self.encode_image(image)
        except Exception as e:
            logger.warning(f"[CLIPManager] Failed to load image from {path}: {e}")
            return None

    @property
    def embedding_dim(self) -> int:
        return self.EMBEDDING_DIM


def get_clip_manager() -> CLIPManager:
    """Thread-safe singleton accessor for CLIPManager."""
    global _singleton
    if _singleton is not None:
        return _singleton

    with _singleton_lock:
        if _singleton is None:
            _singleton = CLIPManager()
        return _singleton
