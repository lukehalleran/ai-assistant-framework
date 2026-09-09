"""Real ModelManager cache, with only the model-loading boundary replaced."""
from unittest.mock import patch

import numpy as np

from models import model_manager as module


def test_model_manager_cross_encoder_caching():
    # The cache is process-global, so verify reuse across manager instances too.
    # Fresh test-owned dict prevents both warm-cache false passes and pollution.
    encoder = object()
    with patch.object(module, "_global_cross_encoders", {}), \
         patch("sentence_transformers.CrossEncoder", return_value=encoder) as load:
        first_manager = module.ModelManager.__new__(module.ModelManager)
        second_manager = module.ModelManager.__new__(module.ModelManager)
        first = first_manager.get_cross_encoder("synthetic-model")
        second = first_manager.get_cross_encoder("synthetic-model")
        third = second_manager.get_cross_encoder("synthetic-model")
        assert first is second is third is encoder
        load.assert_called_once_with("synthetic-model")


def test_different_cross_encoder_models_have_separate_cache_entries():
    first, second = object(), object()
    with patch.object(module, "_global_cross_encoders", {}), \
         patch("sentence_transformers.CrossEncoder", side_effect=[first, second]) as load:
        manager = module.ModelManager.__new__(module.ModelManager)
        assert manager.get_cross_encoder("synthetic-a") is first
        assert manager.get_cross_encoder("synthetic-b") is second
        assert manager.get_cross_encoder("synthetic-a") is first
        assert load.call_count == 2


def test_model_load_failure_returns_and_caches_neutral_fallback():
    with patch.object(module, "_global_cross_encoders", {}), \
         patch("sentence_transformers.CrossEncoder", side_effect=RuntimeError("synthetic load failure")) as load:
        manager = module.ModelManager.__new__(module.ModelManager)
        fallback = manager.get_cross_encoder("synthetic-failing")
        assert fallback is manager.get_cross_encoder("synthetic-failing")
        np.testing.assert_array_equal(fallback.predict([("q", "a"), ("q", "b")]), [0.5, 0.5])
        load.assert_called_once()
