"""Lane QUAR: corpus readers honour the quarantine flag (class: BC-26, BC-58).

A chroma-only quarantine hid half the copies: every conversation, summary and
reflection also lives in the corpus JSON, and the corpus readers never asked
``is_quarantined``.  These tests drive the DEPLOYED CorpusManager / MemoryRetriever /
ShutdownProcessor against a flagged corpus.  Synthetic text only.

Original batch label: oct15 (renamed 2026-10-16; old path in tests/RENAMES.tsv).
"""
import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

import memory.shutdown_processor as sp
from memory.corpus_manager import CorpusManager
from memory.memory_retriever import MemoryRetriever

NOW = datetime.now().replace(microsecond=0)
FLAG = {"curation_quarantined": True}


def _conv(q, minutes_ago, **extra):
    return {"query": q, "response": f"reply to {q}",
            "timestamp": (NOW - timedelta(minutes=minutes_ago)).isoformat(), "tags": [], **extra}


def _build(tmp_path):
    entries = [
        _conv("visible STAT 5100 question", 50),
        _conv("quarantined STAT 5100 due date claim", 20, **FLAG),
        _conv("another visible turn about BIO 2200", 10),
        {"content": "visible summary about BIO 2200", "type": "summary", "tags": ["@summary"],
         "timestamp": (NOW - timedelta(hours=3)).isoformat()},
        {"content": "quarantined summary STAT 5100 wrong date", "type": "summary", "tags": ["@summary"],
         "timestamp": (NOW - timedelta(hours=2)).isoformat(), **FLAG},
        {"content": "visible reflection", "type": "reflection", "tags": ["type:reflection"],
         "timestamp": (NOW - timedelta(hours=5)).isoformat()},
        {"content": "quarantined reflection wrong date", "type": "reflection", "tags": ["type:reflection"],
         "timestamp": (NOW - timedelta(hours=1)).isoformat(), **FLAG},
    ]
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(entries))
    return CorpusManager(str(path)), path


def _texts(items):
    return " | ".join(str(i.get("query") or i.get("content")) for i in items)


def test_every_corpus_content_reader_excludes_flagged_entries(tmp_path):
    cm, _ = _build(tmp_path)
    assert "quarantined" not in _texts(cm.get_recent_memories(10))
    assert "quarantined" not in _texts(cm.get_recent_within_hours(24, 30))
    assert "quarantined" not in _texts(cm.get_summaries(10))
    assert "quarantined" not in _texts(cm.get_summaries_of_type(("summary",), 10))
    assert "quarantined" not in _texts(cm.get_items_by_type("reflection", 10))
    assert "quarantined" not in _texts(cm.get_items_by_type("summary", 10))
    assert cm.search_keyword("STAT 5100") and all(
        "quarantined" not in h["query_preview"] for h in cm.search_keyword("STAT 5100"))
    # the flagged entries are still STORED (maintenance paths keep them)
    assert len(cm.corpus) == 7 and len(cm.visible_entries()) == 4


def test_flag_survives_save_and_reload(tmp_path):
    cm, path = _build(tmp_path)
    cm.save_corpus()
    again = CorpusManager(str(path))
    assert len(again.corpus) == 7
    assert sum(1 for e in again.corpus if e.get("curation_quarantined")) == 3
    assert "quarantined" not in _texts(again.get_recent_memories(10))


def test_flag_flipped_after_cache_is_built_is_excluded(tmp_path):
    cm, _ = _build(tmp_path)
    assert "another visible" in _texts(cm.get_recent_memories(10))  # builds the episodic cache
    target = next(e for e in cm.corpus if e.get("query", "").startswith("another visible"))
    target["curation_quarantined"] = True  # in-place flip, no add_entry invalidation
    assert "another visible" not in _texts(cm.get_recent_memories(10))
    target["curation_quarantined"] = False  # undo needs the cache dropped
    cm.invalidate_read_cache()
    assert "another visible" in _texts(cm.get_recent_memories(10))


def test_retriever_recent_conversations_and_corpus_first_reflections_exclude_flagged(tmp_path):
    cm, _ = _build(tmp_path)
    r = MemoryRetriever(cm, None)
    recent = r._get_recent_conversations(10)
    assert recent and "quarantined" not in json.dumps(recent, default=str)
    refl = asyncio.run(r.get_reflections(limit=5))
    assert [x["content"] for x in refl] == ["visible reflection"]


# --- deployed shutdown summary selection -----------------------------------------------------

class _Chroma:
    def __init__(self):
        self.added = []

    def list_all(self, name):
        return []

    def add_to_collection(self, name, text, md):
        self.added.append(text)
        return "id"


class _Consolidator:
    consolidation_threshold = 20

    def __init__(self):
        self.seen = []

    async def consolidate_memories(self, content_list):
        self.seen.append(json.dumps(content_list, default=str))
        return SimpleNamespace(content="Summary: the user discussed their project work in some detail here.")


def _proc(corpus, chroma, cons):
    p = sp.ShutdownProcessor(corpus, chroma, cons, None, None, None, None, datetime.now())
    for name in ("_extract_session_facts", "_extract_llm_facts",
                 "_extract_procedural_skills", "_generate_proposals", "_check_implementation_tracking",
                 "_process_open_threads", "_run_wiki_enrichment", "_run_wikidata_enrichment",
                 "_run_cross_collection_dedup"):
        async def _noop(*a, **k):
            return None
        setattr(p, name, _noop)
    p._save_knowledge_graph = lambda: None
    p._maybe_spawn_agent_branch = lambda: None
    return p


def test_shutdown_summary_selection_skips_quarantined_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "SUMMARY_MAX_BLOCKS_PER_SHUTDOWN", 5)
    base = datetime(2026, 8, 1, 12, 0)
    entries = [{"query": f"routine turn {i}", "response": f"reply {i}",
                "timestamp": (base + timedelta(hours=i)).isoformat(), "tags": []} for i in range(41)]
    entries[10]["query"] = "QUARANTINED-DUE-DATE-CLAIM"
    entries[10]["curation_quarantined"] = True
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(entries))
    cm = CorpusManager(str(path))
    cons = _Consolidator()
    asyncio.run(_proc(cm, _Chroma(), cons).process_shutdown_memory([]))
    assert cons.seen, "expected at least one summary block"
    assert not any("QUARANTINED-DUE-DATE-CLAIM" in s for s in cons.seen)
