# WikiManager.py

"""
WikiManager - Retrieves Wikipedia summaries and full articles.

Responsibilities:
- Search and return article summaries
- Optionally return full article content
- Heuristic for fallback when summary is insufficient
"""

import wikipedia

class WikiManager:
    def __init__(self, offline_mode=False):
        self.offline_mode = offline_mode
        # Future: load local index if offline_mode is True

    def search_summary(self, topic, sentences=3):
        """Fetch a summary for the given topic."""
        try:
            summary = wikipedia.summary(topic, sentences=sentences, auto_suggest=True, redirect=True)
            return summary
        except wikipedia.exceptions.DisambiguationError as e:
            return f"[Disambiguation Error] Topic '{topic}' is ambiguous. Options: {e.options}"
        except wikipedia.exceptions.PageError:
            return f"[Page Error] Topic '{topic}' not found."
        except Exception as e:
            return f"[Error fetching summary: {str(e)}]"

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
