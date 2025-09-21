"""Legacy shim for the relocated TopicManager.

This module used to host the spaCy-based topic manager. The canonical
implementation now lives in ``utils.topic_manager``. We keep this thin wrapper
so any lingering imports continue to work while avoiding duplicate code paths.
"""

from __future__ import annotations

from utils.topic_manager import TopicManager, TopicSpan

__all__ = ["TopicManager", "TopicSpan"]
