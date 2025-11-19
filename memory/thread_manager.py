# memory/thread_manager.py
"""
Thread management module for conversation threading.

Implements the ThreadManagerProtocol contract for tracking conversation threads.
"""

import logging
from datetime import datetime
from typing import Dict, Optional

from utils.logging_utils import get_logger

logger = get_logger("thread_manager")


class ThreadManager:
    """
    Conversation thread management.

    Implements ThreadManagerProtocol contract.
    """

    def __init__(
        self,
        corpus_manager,
        topic_manager=None,
        time_manager=None,
    ):
        """
        Initialize ThreadManager.

        Args:
            corpus_manager: CorpusManager for accessing conversation history
            topic_manager: Optional TopicManager for topic detection
            time_manager: Optional TimeManager for timestamps
        """
        self.corpus_manager = corpus_manager
        self.topic_manager = topic_manager
        self.time_manager = time_manager

    def _now(self) -> datetime:
        """Get current time from time_manager or datetime.now()"""
        if self.time_manager and hasattr(self.time_manager, 'current'):
            return self.time_manager.current()
        return datetime.now()

    def _now_iso(self) -> str:
        """Get current time as ISO string"""
        if self.time_manager and hasattr(self.time_manager, 'current_iso'):
            return self.time_manager.current_iso()
        return self._now().isoformat()

    def get_thread_context(self) -> Optional[Dict]:
        """
        Retrieve thread context from the most recent conversation.
        Returns thread metadata if a thread is active, None otherwise.
        """
        recent = self.corpus_manager.get_recent_memories(count=1)
        if not recent:
            return None

        last_conv = recent[0]
        thread_id = last_conv.get("thread_id")

        if not thread_id:
            return None

        return {
            "thread_id": thread_id,
            "thread_depth": last_conv.get("thread_depth", 1),
            "thread_started": last_conv.get("thread_started"),
            "thread_topic": last_conv.get("thread_topic"),
            "is_heavy_topic": last_conv.get("is_heavy_topic", False),
        }

    def detect_topic_for_query(self, query: str) -> str:
        """
        Detect topic for a specific query string.
        Used by thread detection to avoid stale topic issues.
        """
        try:
            if self.topic_manager:
                # Update topic manager with this specific query
                self.topic_manager.update_from_user_input(query)
                # Get primary topic for this query
                primary = self.topic_manager.get_primary_topic()
                return primary or "general"
            else:
                # Fallback: simple keyword-based topic detection
                query_lower = query.lower()
                if any(word in query_lower for word in ["flapjack", "cat", "kitty"]):
                    return "cats"
                elif any(word in query_lower for word in ["glm", "claude", "model", "ai"]):
                    return "ai models"
                else:
                    return "general"
        except Exception as e:
            logger.debug(f"[Thread] Topic detection failed: {e}")
            return "general"

    def detect_or_create_thread(self, query: str, is_heavy: bool) -> Dict:
        """
        Detect if current query continues immediate previous conversation.
        Returns thread metadata dict with thread_id, depth, started, topic.
        Only checks the most recent conversation for strict consecutive threading.
        """
        from utils.query_checker import belongs_to_thread

        # Get only the most recent conversation (limit=1 for strict consecutive check)
        recent = self.corpus_manager.get_recent_memories(count=1)

        # Detect topic for current query to avoid using stale topic
        current_query_topic = self.detect_topic_for_query(query)

        if not recent:
            # First conversation - create new thread
            thread_id = f"thread_{self._now().strftime('%Y%m%d_%H%M%S')}"
            logger.debug(f"[Thread] Creating new thread (first conversation): {thread_id}")
            return {
                "thread_id": thread_id,
                "depth": 1,
                "started": self._now_iso(),
                "topic": current_query_topic,
            }

        last_conv = recent[0]

        # Check if current query continues last conversation
        if belongs_to_thread(query, last_conv, current_topic=current_query_topic):
            # Continue existing thread
            thread_id = last_conv.get("thread_id")
            thread_depth = last_conv.get("thread_depth", 0) + 1
            thread_started = last_conv.get("thread_started")
            thread_topic = last_conv.get("thread_topic", current_query_topic)

            logger.debug(
                f"[Thread] Continuing thread {thread_id} at depth {thread_depth} "
                f"(topic: {thread_topic})"
            )

            return {
                "thread_id": thread_id,
                "depth": thread_depth,
                "started": thread_started,
                "topic": thread_topic,
            }
        else:
            # Topic switch or time gap - new thread
            thread_id = f"thread_{self._now().strftime('%Y%m%d_%H%M%S')}"
            logger.debug(
                f"[Thread] Creating new thread (break detected): {thread_id} "
                f"(previous: {last_conv.get('thread_id')})"
            )
            return {
                "thread_id": thread_id,
                "depth": 1,
                "started": self._now_iso(),
                "topic": current_query_topic,
            }
