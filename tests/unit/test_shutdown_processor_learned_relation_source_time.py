"""Shutdown LLM facts must hand the extractor each turn's timestamp.

Incident 2026-10-16; classes BC-10 (dead wiring: LLMFactExtractor reads the
message ``timestamp`` to date a learned relation, but the only production
producer, ``ShutdownProcessor._extract_llm_facts``, sent only ``turn_id``) and
BC-64 (the PR #79 fixtures carried a ``timestamp`` key the real producer never
sends, so the tests passed while every shutdown logged
"Learned-relation skipped (no source time)").

These tests drive THE deployed ``_extract_llm_facts`` end to end with a fake
model at the ``generate_once`` seam.
"""
import asyncio
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from memory.learned_relations import get_learned_relation_store
from memory.llm_fact_extractor import CORE_RELATIONS
from memory.shutdown_processor import ShutdownProcessor

REL = "brews"
QUERY = "Every morning I make cold coffee before work."
OBJ = "cold coffee"


class _FakeModel:
    async def generate_once(self, **kwargs):
        return json.dumps([
            {"subject": "user", "relation": REL, "object": OBJ, "confidence": 0.9}
        ])


def _processor():
    chroma = MagicMock()
    chroma.add_fact.return_value = "fact-1"
    return ShutdownProcessor(
        corpus_manager=MagicMock(),
        chroma_store=chroma,
        consolidator=MagicMock(),
        fact_extractor=MagicMock(),
        model_manager=_FakeModel(),
        user_profile=None,
        storage=SimpleNamespace(),
        session_start=datetime(2026, 10, 14, 8, 0),
    )


def _item(with_ts=True):
    item = {"query": QUERY, "response": "Nice."}
    if with_ts:
        item["timestamp"] = datetime(2026, 10, 14, 9, 0)
    return item


def _row():
    return get_learned_relation_store()._data["relations"].get(REL)


def test_relation_is_not_core():
    assert REL not in CORE_RELATIONS


def test_shutdown_item_timestamp_dates_learned_relation():
    asyncio.run(_processor()._extract_llm_facts([_item()]))
    row = _row()
    assert row is not None
    assert row["days"] == ["2026-10-14"]
    assert row["last_seen"] == "2026-10-14"


def test_item_without_timestamp_records_nothing():
    asyncio.run(_processor()._extract_llm_facts([_item(with_ts=False)]))
    assert _row() is None


def test_replay_of_same_item_is_one_distinct_day():
    proc = _processor()
    asyncio.run(proc._extract_llm_facts([_item()]))
    asyncio.run(proc._extract_llm_facts([_item()]))
    row = _row()
    assert row["days"] == ["2026-10-14"]
    assert row["count"] == 2
