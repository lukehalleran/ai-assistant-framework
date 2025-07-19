# daemon_7_11_25_refactor/memory/corpus_manager.py
import json
import os
from datetime import datetime
from typing import List, Dict
from utils.logging_utils import get_logger, log_and_time

logger = get_logger("corpus_manager")

class CorpusManager:
    """Manages the conversation corpus (short-term memory)"""

    def __init__(self, corpus_file: str = None):
        from config.config import CORPUS_FILE
        self.corpus_file = corpus_file or CORPUS_FILE
        self.corpus = self._load_corpus()

    @log_and_time("Load Corpus")
    def _load_corpus(self) -> List[Dict]:
        """Load corpus from disk"""
        if os.path.exists(self.corpus_file):
            try:
                with open(self.corpus_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert timestamp strings back to datetime
                    for entry in data:
                        if isinstance(entry.get("timestamp"), str):
                            try:
                                entry["timestamp"] = datetime.fromisoformat(entry["timestamp"])
                            except:
                                pass
                    logger.info(f"Loaded {len(data)} corpus entries")
                    return data
            except Exception as e:
                logger.error(f"Error loading corpus: {e}")
        return []

    @log_and_time("Save Corpus")
    def save_corpus(self):
        """Save corpus to disk atomically"""
        tmp_file = self.corpus_file + ".tmp"
        try:
            with open(tmp_file, 'w', encoding='utf-8') as f:
                # Convert datetime to string for JSON
                data_to_save = []
                for entry in self.corpus:
                    entry_copy = entry.copy()
                    if isinstance(entry_copy.get("timestamp"), datetime):
                        entry_copy["timestamp"] = entry_copy["timestamp"].isoformat()
                    data_to_save.append(entry_copy)
                json.dump(data_to_save, f, indent=2)
            os.replace(tmp_file, self.corpus_file)
            logger.debug(f"Saved {len(self.corpus)} entries to corpus")
        except Exception as e:
            logger.error(f"Error saving corpus: {e}")

    def add_entry(self, query: str, response: str, tags: List[str] = None):
        """Add a new interaction to corpus"""
        entry = {
            "query": query,
            "response": response,
            "timestamp": datetime.now(),
            "tags": tags or []
        }
        self.corpus.append(entry)

        # Auto-summarize every 20 non-summary entries
        real_entries = [e for e in self.corpus if "@summary" not in e.get("tags", [])]
        if len(real_entries) % 20 == 0 and len(real_entries) > 0:
            self._create_summary(real_entries[-20:])

        # Trim if too large
        if len(self.corpus) > 500:
            self.corpus = self.corpus[-500:]

        self.save_corpus()

    def _create_summary(self, entries: List[Dict]):
        """Create a summary node for the given entries"""
        from config.config import DEFAULT_SUMMARY_PROMPT_HEADER

        summary_lines = []
        for e in entries[:20]:
            if e.get('response', '').strip():
                q = e.get('query', '[no query]')[:50]
                r = e.get('response', '')[:60]
                summary_lines.append(f"Q: {q}... → A: {r}...")

        if summary_lines:
            summary_entry = {
                "query": "[SUMMARY NODE]",
                "response": DEFAULT_SUMMARY_PROMPT_HEADER + "\n".join(summary_lines),
                "timestamp": datetime.now(),
                "tags": ["@summary"]
            }
            self.corpus.append(summary_entry)
            logger.debug("Created summary node")

    def get_recent_memories(self, count: int = 3) -> List[Dict]:
        """Get most recent non-summary memories"""
        non_summary = [e for e in self.corpus if "@summary" not in e.get("tags", [])]
        return sorted(non_summary, key=lambda x: x.get('timestamp', datetime.min), reverse=True)[:count]

    def get_summaries(self, count: int = 5) -> List[Dict]:
        """Get summary nodes"""
        summaries = [e for e in self.corpus if "@summary" in e.get("tags", [])]
        return sorted(summaries, key=lambda x: x.get('timestamp', datetime.min), reverse=True)[:count]
