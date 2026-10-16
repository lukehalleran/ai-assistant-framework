"""Shutdown behavioral patterns must join to the user's own recurring words.

Incident 2026-10-15 (class: BC-46, BC-75): ``eating_habit = orders food after
completing homework`` was stored from ONE evening's messages, with no
source_excerpt, because the behavioral-pattern step called ``add_fact``
directly, bypassing the fact_source provenance boundary. These tests drive the
deployed ``ShutdownProcessor._extract_behavioral_patterns``.
"""
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory.shutdown_processor import ShutdownProcessor
from memory.user_profile import UserProfile


def _run(msgs, model_reply, tmp_path):
    profile = UserProfile(profile_path=str(tmp_path / "profile.json"))
    mm = MagicMock()
    mm.generate_once = AsyncMock(return_value=model_reply)
    proc = ShutdownProcessor(
        corpus_manager=MagicMock(), chroma_store=MagicMock(),
        consolidator=MagicMock(), fact_extractor=MagicMock(),
        model_manager=mm, user_profile=profile, storage=MagicMock(),
        session_start=datetime.now(),
    )
    convs = [{"query": m, "response": "ok", "timestamp": datetime.now().isoformat()}
             for m in msgs]
    asyncio.run(proc._extract_behavioral_patterns(convs))
    return profile


def _facts(profile):
    out = []
    for cat in profile.profile["categories"].values():
        out.extend(f for f in cat if isinstance(f, dict))
    return out


def test_single_evening_is_not_a_habit(tmp_path):
    reply = json.dumps([{"relation": "eating_habit",
                         "object": "orders food after completing homework",
                         "category": "preferences", "confidence": 0.75}])
    p = _run(["I ordered burgers and had a beer", "worked on hw 4 for a while",
              "finished hw 4"], reply, tmp_path)
    assert _facts(p) == []


def test_pattern_absent_from_user_words_is_dropped(tmp_path):
    reply = json.dumps([{"relation": "coding_environment",
                         "object": "codes at the gym", "confidence": 0.7}])
    p = _run(["I always study at the library every morning", "hello", "thanks"],
             reply, tmp_path)
    assert _facts(p) == []


def test_user_stated_regularity_is_stored_with_excerpt(tmp_path):
    reply = json.dumps([{"relation": "study_environment",
                         "object": "library every morning", "confidence": 0.7}])
    p = _run(["I always study at the library every morning", "hello", "thanks"],
             reply, tmp_path)
    facts = _facts(p)
    assert len(facts) == 1
    f = facts[0]
    assert "library" in f["source_excerpt"]
    assert f.get("claim_kind") == "habit"


@pytest.mark.parametrize("reply", [
    "Sure! Here are some patterns you might like.",
    '[{"relation": "x_habit", "object": ',
    json.dumps([{"relation": f"r{i}_habit", "object": f"thing {i} at night"}
                for i in range(5)]),
    json.dumps(["not a dict", 3, None]),
])
def test_misbehaving_model_stores_nothing_unsupported(tmp_path, reply):
    p = _run(["hello there", "how are you", "fine thanks"], reply, tmp_path)
    assert _facts(p) == []
