# /memory/corpus_manager.py
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from utils.logging_utils import get_logger, log_and_time

logger = get_logger("corpus_manager")

class CorpusManager:
    """Manages the conversation corpus (short-term memory)"""

    def __init__(self, corpus_file: str = None):
        if corpus_file is None:
                from config.app_config import CORPUS_FILE
                corpus_file=CORPUS_FILE
        self.corpus_file = corpus_file or CORPUS_FILE
        self.memories = []
        self.summaries = []
        logger.info(f"[CorpusManager] Initializing with file: {self.corpus_file}")
        self.corpus = self._load_corpus()
        logger.info(f"[CorpusManager] After init, corpus has {len(self.corpus)} entries")

    @log_and_time("Load Corpus")
    def _load_corpus(self) -> List[Dict]:
        """Load corpus from disk"""
        if os.path.exists(self.corpus_file):
            try:
                with open(self.corpus_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"[CorpusManager] Loaded {len(data)} entries from {self.corpus_file}")
                    # Convert timestamp strings back to datetime
                    for entry in data:
                        if isinstance(entry.get("timestamp"), str):
                            try:
                                entry["timestamp"] = datetime.fromisoformat(entry["timestamp"])
                            except:
                                pass
                    logger.info(f"Loaded {len(data)} corpus entries")
                    self.corpus = data
                    return data

            except Exception as e:
                logger.error(f"Error loading corpus: {e}")

        else:
            logger.warning(f"[CorpusManager] Corpus file not found: {self.corpus_file}")

        return []

    def clean_for_json(self, entry):
        if isinstance(entry, dict):
            return {k: self.clean_for_json(v) for k, v in entry.items()}
        elif isinstance(entry, list):
            return [self.clean_for_json(v) for v in entry]
        elif isinstance(entry, datetime):
            return entry.isoformat()
        else:
            return entry


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
                json.dump(self.clean_for_json(data_to_save), f, indent=2)
            os.replace(tmp_file, self.corpus_file)
            logger.debug(f"Saved {len(self.corpus)} entries to corpus")
        except Exception as e:
            logger.error(f"Error saving corpus: {e}")

    def add_entry(self, query: str, response: str, tags: List[str] = None, timestamp: datetime = None):
        """Add a new interaction to corpus"""
        # Trim obvious trailing/leading whitespace to avoid blank lines in prompts
        q = (query or "").strip()
        r = (response or "").strip()
        entry = {
            "query": q,
            "response": r,
            "timestamp": timestamp or datetime.now(),
            "tags": tags or []
        }
        self.corpus.append(entry)


        # Trim if too large
        if len(self.corpus) > 500:
            self.corpus = self.corpus[-500:]

        self.save_corpus()


    def get_recent_memories(self, count: int = 3) -> List[Dict]:
        """Get most recent non-summary memories"""
        non_summary = [e for e in self.corpus if "@summary" not in e.get("tags", [])]
        result = sorted(non_summary, key=lambda x: x.get('timestamp', datetime.min), reverse=True)[:count]
        logger.debug(f"[CorpusManager] Returning {len(result)} recent memories from {len(non_summary)} non-summary entries")
        return result


    def add_summary(self, content, tags: List[str] = None, timestamp: datetime = None):
        """Add a summary-like node to the corpus.

        Accepts either:
          - content: str (classic summary text), or
          - content: dict with keys {content, timestamp?, tags?, type?}
        Ensures type defaults to 'summary' for string inputs. For dict inputs,
        preserves provided type and only adds '@summary' tag when the type is a summary.
        """
        if isinstance(content, dict):
            node = dict(content)  # shallow copy
            # Normalize fields
            node.setdefault("content", "")
            # Accept either 'timestamp' or legacy 'created_at'
            if "timestamp" in node:
                pass
            elif "created_at" in node and isinstance(node["created_at"], datetime):
                node["timestamp"] = node["created_at"]
            elif "created_at" in node and isinstance(node["created_at"], str):
                try:
                    node["timestamp"] = datetime.fromisoformat(node["created_at"].replace("Z", "+00:00"))
                except Exception:
                    node["timestamp"] = timestamp or datetime.now()
            else:
                node["timestamp"] = timestamp or datetime.now()
            node.setdefault("type", node.get("type") or "summary")
            node["tags"] = list(node.get("tags") or [])
            # Only mark as '@summary' if this is actually a summary-type node
            node_type = str(node.get("type", "")).lower()
            if ("summary" in node_type) and ("@summary" not in node["tags"]):
                node["tags"].append("@summary")
            summary = node
        else:
            summary = {
                "content": str(content or ""),
                "timestamp": timestamp or datetime.now(),
                "tags": (tags or []) + ["@summary"],
                "type": "summary"
            }
        # persist in the same list/file that get_summaries() actually reads
        self.corpus.append(summary)
        # bound size like other entries
        if len(self.corpus) > 500:
            self.corpus = self.corpus[-500:]
        self.save_corpus()
        logger.debug(f"[CorpusManager] Added summary: {content[:50]}...")

    def get_summaries(self, count: int = 5) -> List[Dict[str, any]]:
        """Get the most recent summaries, normalizing timestamps for robust sorting."""
        from datetime import timezone

        def _norm_ts(ts):
            # Accept datetime or ISO string; return naive UTC datetime
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    return datetime.min
            if isinstance(ts, datetime):
                try:
                    if ts.tzinfo is not None and ts.tzinfo.utcoffset(ts) is not None:
                        return ts.astimezone(timezone.utc).replace(tzinfo=None)
                    return ts
                except Exception:
                    return datetime.min
            return datetime.min

        def _is_summary_entry(e: Dict) -> bool:
            typ = (e.get("type", "") or "").lower()
            tags = e.get("tags") or []
            # Exclude reflections even if they carry '@summary' from legacy paths
            if ("reflection" in typ) or ("type:reflection" in tags):
                return False
            return ("summary" in typ) or ("@summary" in tags) or ("type:summary" in tags)

        summaries = [e for e in self.corpus if _is_summary_entry(e)]
        summaries.sort(key=lambda x: _norm_ts(x.get("timestamp")), reverse=True)
        return summaries[:count]

    def create_summary_now(self, num_recent: int = 20) -> str:
        """Manually create a summary of recent conversations"""
        recent = self.get_recent_memories(num_recent)
        if not recent:
            return None

        summary_lines = []
        for entry in recent[:num_recent]:
            q = entry.get('query', '')[:50]
            r = entry.get('response', '')[:50]
            summary_lines.append(f"Q: {q}... A: {r}...")

        summary_content = f"Summary of {len(recent)} conversations:\n" + "\n".join(summary_lines)

        self.add_summary(
            content=summary_content,
            tags=["@summary", "manual"],
            timestamp=datetime.now()
        )

        logger.info(f"[CorpusManager] Created manual summary of {len(recent)} conversations")
        return summary_content

    # --- Convenience helpers used by reflections path ---
    def get_summaries_of_type(self, types=("summary",), limit: int = 5) -> List[Dict]:
        tset = set(types or ("summary",))
        items = [e for e in self.corpus if (e.get("type") or "").lower() in tset]
        items.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)
        return items[:limit]

    def get_items_by_type(self, type_name: str, limit: int = 5) -> List[Dict]:
        t = (type_name or "").lower()
        items = [e for e in self.corpus if (e.get("type") or "").lower() == t or f"type:{t}" in (e.get("tags") or [])]
        items.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)
        return items[:limit]
