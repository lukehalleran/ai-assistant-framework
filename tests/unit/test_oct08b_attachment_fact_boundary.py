"""Attachment text must never be mined as the owner's facts (2026-10-08).

On an attachment-only turn the corpus stores ``user_text == ""`` while
``query`` is the merged attachment blob.  Every fact path must read the
AUTHORED text through ``fact_source.authored_text`` — an empty ``user_text`` is
authoritative, never a cue to fall back to the blob.
class: BC-09, BC-54, BC-58
"""
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory.fact_source import authored_text, iter_user_messages
from memory.shutdown_processor import ShutdownProcessor

BLOB = "\n\n[FILE: cv.pdf]\nB.S. in Botany, graduated 2015, Globex analyst"
TRIPLE = json.dumps([
    {"subject": "user", "relation": "completed", "object": "B.S. in Botany",
     "confidence": 0.9}
])


def _processor(model_manager, chroma):
    return ShutdownProcessor(
        corpus_manager=MagicMock(),
        chroma_store=chroma,
        consolidator=MagicMock(),
        fact_extractor=MagicMock(),
        model_manager=model_manager,
        user_profile=None,
        storage=MagicMock(),
        session_start=datetime.now(),
    )


def _mm():
    mm = MagicMock()
    mm.generate_once = AsyncMock(return_value=TRIPLE)
    return mm


class TestAuthoredText:
    def test_empty_user_text_is_authoritative(self):
        assert authored_text({"query": BLOB, "user_text": ""}) == ""

    def test_absent_key_falls_back_to_query(self):
        assert authored_text({"query": "I like tea"}) == "I like tea"

    def test_present_key_wins_over_query(self):
        assert authored_text({"query": BLOB, "user_text": "see attached"}) == "see attached"

    def test_non_mapping_is_text(self):
        assert authored_text("User: hello") == "User: hello"

    def test_iter_user_messages_skips_attachment_only_turn(self):
        assert list(iter_user_messages([{"query": BLOB, "user_text": ""}])) == []


@pytest.mark.asyncio
async def test_attachment_only_turn_mints_no_llm_fact():
    mm, chroma = _mm(), MagicMock()
    chroma.add_fact.return_value = "id"
    proc = _processor(mm, chroma)
    await proc._extract_llm_facts([
        {"query": BLOB, "response": "Looks good.", "user_text": ""},
    ])
    for call in mm.generate_once.call_args_list:
        assert "Botany" not in call.kwargs.get("prompt", "")
    chroma.add_fact.assert_not_called()


@pytest.mark.asyncio
async def test_typed_text_alongside_attachment_still_extracts():
    mm, chroma = _mm(), MagicMock()
    chroma.add_fact.return_value = "id"
    proc = _processor(mm, chroma)
    await proc._extract_llm_facts([
        {"query": "I finished a B.S. in Botany" + BLOB, "response": "Nice.",
         "user_text": "I finished a B.S. in Botany"},
    ])
    prompt = mm.generate_once.call_args.kwargs["prompt"]
    assert "I finished a B.S. in Botany" in prompt
    assert "Globex" not in prompt
    chroma.add_fact.assert_called()


@pytest.mark.asyncio
async def test_ordinary_turn_without_user_text_key_still_extracts():
    mm, chroma = _mm(), MagicMock()
    chroma.add_fact.return_value = "id"
    proc = _processor(mm, chroma)
    await proc._extract_llm_facts([
        {"query": "I finished a B.S. in Botany last spring", "response": "Congrats!"},
    ])
    assert "Botany" in mm.generate_once.call_args.kwargs["prompt"]
    chroma.add_fact.assert_called()


@pytest.mark.asyncio
async def test_extract_session_facts_never_passes_attachment_text():
    fe = MagicMock()
    fe.extract_facts = AsyncMock(return_value=[])
    proc = ShutdownProcessor(
        corpus_manager=MagicMock(),
        chroma_store=MagicMock(),
        consolidator=MagicMock(),
        fact_extractor=fe,
        model_manager=MagicMock(),
        user_profile=None,
        storage=MagicMock(),
        session_start=datetime.now(),
    )
    attach = {"query": BLOB, "response": "ok", "user_text": "",
              "timestamp": datetime.now()}
    typed = {"query": "I adopted a cat named Miso", "response": "Aw!",
             "timestamp": datetime.now()}
    await proc._extract_session_facts([attach, typed])
    seen = [c.args[0] for c in fe.extract_facts.call_args_list]
    assert all("Botany" not in q for q in seen)
    assert any("Miso" in q for q in seen)
