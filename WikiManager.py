# WikiManager.py

"""
WikiManager - Retrieves Wikipedia summaries and full articles.

Responsibilities:
- Search and return article summaries
- Optionally return full article content
- Heuristic for fallback when summary is insufficient
"""
import logging

# Use the root logger or create a child logger that will inherit handlers
logger = logging.getLogger(__name__)
logger.debug("Wiki Manager.py is alive")

import re
import json
import os
from datetime import datetime, timedelta
import wikipedia
from logging_utils import log_and_time

class WikiManager:
    def __init__(self, offline_mode=False):
        self.offline_mode = offline_mode
        # Future: load local index if offline_mode is True
    @log_and_time("Search summary")
    def search_summary(self, topic, sentences=3):
        """Fetch a summary for the given topic."""
        try:
            # Capitalize first letter as wikipedia package is case-sensitive
            topic = topic.strip().capitalize()

            summary = wikipedia.summary(topic, sentences=sentences, auto_suggest=True, redirect=True)
            return summary
        except wikipedia.exceptions.DisambiguationError as e:
            logger.warning(f"[WikiManager] Disambiguation error for '{topic}': {e.options}")
            return f"[Disambiguation Error] Topic '{topic}' is ambiguous. Options: {e.options}"
        except wikipedia.exceptions.PageError:
            logger.warning(f"[WikiManager] Page error: '{topic}' not found.")
            return f"[Page Error] Topic '{topic}' not found."
        except Exception as e:
            logger.error(f"[WikiManager] Unexpected error for '{topic}': {str(e)}")
            return f"[Error fetching summary: {str(e)}]"
    @log_and_time("Fetch full article")
    def fetch_full_article(self, topic):
        """Fetch the full article content for the given topic."""
        try:
            page = wikipedia.page(topic, auto_suggest=True, redirect=True)
            return page.content
        except wikipedia.exceptions.DisambiguationError as e:
            return f"[Disambiguation Error] Topic '{topic}' is ambiguous. Options: {e.options}"
        except wikipedia.exceptions.PageError:
            return f"[Page Error] Topic '{topic}' not found."
        except Exception as e:
            return f"[Error fetching article: {str(e)}]"

    def should_fallback(self, summary, query):
        """Decide whether full article exploration might be needed."""
        if not summary or len(summary.split()) < 50:
            return True
        # If asking for numeric or structured data
        keywords = ["table", "statistics", "list", "timeline", "formula", "equation"]
        if any(k in query.lower() for k in keywords):
            return True
        return False
