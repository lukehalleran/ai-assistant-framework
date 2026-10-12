"""B4 item 1: a reflection must render once, as its body (class: BC-46)."""
import asyncio

from core.prompt.formatter import _format_summary_section
from memory.memory_retriever import MemoryRetriever

BODY = ("1) **SESSION TOPIC:** Personal scheduling.\n\n2) **KEY ENTITIES:** Chicago, "
        "Northwind, Adam Serwer, The Atlantic.\n\n3) **WHAT HAPPENED:** dentist moved.")
SOUP = "Adam Adam Serwer Atlantic Atlantic** Chicago appointments and work responsibilities 1) **SESSION TOPIC:**"


class _Corpus:
    def get_items_by_type(self, kind, limit=2):
        return [{"content": BODY, "type": "reflection", "tags": ["type:reflection"],
                 "timestamp": "2026-10-11 23:00:59.372161"}]


class _Coll:
    def count(self):
        return 1


class _Chroma:
    collections = {"reflections": _Coll()}

    def get_recent(self, name, limit=2):
        return [{"content": SOUP,
                 "metadata": {"timestamp": "2026-10-11T23:00:59.372161", "original_text": BODY}}]


def test_reflection_renders_once_as_body():
    r = MemoryRetriever(_Corpus(), _Chroma())
    items = asyncio.run(r.get_reflections(limit=2))
    assert len(items) == 1
    rendered = _format_summary_section(items, "RECENT REFLECTIONS")
    assert "Adam Adam" not in rendered
    assert rendered.count("SESSION TOPIC") == 1


def test_semantic_only_reflection_renders_body_not_retrieval_text():
    class _Empty:
        def get_items_by_type(self, kind, limit=2):
            return []
    items = asyncio.run(MemoryRetriever(_Empty(), _Chroma()).get_reflections(limit=2))
    assert len(items) == 1 and items[0]["content"] == BODY
