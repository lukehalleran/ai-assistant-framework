#core/model_manager.py
"""Compatibility shim exposing ModelManager under the historical core.* namespace."""

from models.model_manager import ModelManager

__all__ = ["ModelManager"]

