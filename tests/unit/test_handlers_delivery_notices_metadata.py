"""Delivery notices stored as turn metadata (incident 2026-10-27, BC-91 G12;
class: BC-91, BC-58).

Notices appended to the reply at delivery ("> ⚠️ Heads up — ...") were stored
as part of the assistant text, so every later reader had to strip machinery
back out. THE storage dispatch (`gui.handlers._dispatch_storage`) now splits
the trailing registered-notice block off the stored response and carries it on
the turn as a `delivery_notices` metadata list. Display is unchanged (the final
chunk is built before storage); `strip_delivery_notices` stays for legacy rows.

Drives the deployed `_dispatch_storage` -> `_background_store_interaction` ->
`MemoryStorage.store_interaction` chain with only the Chroma store faked.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import gui.handlers as handlers
import utils.read_time_markers as rtm
from memory.memory_storage import MemoryStorage

BODY = "Your noon appointment sounds fine, see you then.\n\nLet me know if the time moves."


def _notice():
    return rtm.delivery_notice(rtm.NOTICE_CALENDAR_UNSEEN, ' Say "add it" and I\'ll queue a card.')


def _wrap(text):
    return text.replace("\n\n", "\n \n")


async def _store(response, provenance=None):
    chroma = MagicMock()
    chroma.add_conversation_memory.return_value = "doc-1"
    storage = MemoryStorage(corpus_manager=MagicMock(), chroma_store=chroma,
                            fact_extractor=MagicMock(), model_manager=MagicMock())
    orch = SimpleNamespace(current_topic="general", memory_system=storage)
    task = handlers._dispatch_storage(
        orch, "q", response, "q", response, "p", [], MagicMock(), "s", provenance, "enhanced")
    await task
    (args, _kw) = chroma.add_conversation_memory.call_args
    return args  # (query, response, metadata)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", [lambda t: t, _wrap])
async def test_stored_response_is_notice_free_and_metadata_carries_it(shape):
    shown = shape(BODY) + _notice()
    assert "⚠️" in shown                       # what the user sees is unchanged
    _q, stored, meta = await _store(shown, {"response_mode": "enhanced"})
    assert "⚠️" not in stored and stored == shape(BODY)
    notices = json.loads(meta["delivery_notices"])
    assert len(notices) == 1 and notices[0].startswith("> ⚠️") and "calendar" in notices[0]


@pytest.mark.asyncio
async def test_two_notices_and_missing_provenance():
    shown = BODY + _notice() + rtm.delivery_notice(rtm.NOTICE_UPLOAD_STALE, " I found was uploaded 2026-10-01.")
    _q, stored, meta = await _store(shown, None)
    assert stored == BODY
    assert len(json.loads(meta["delivery_notices"])) == 2


@pytest.mark.asyncio
async def test_no_notice_no_field_and_authored_quote_kept():
    quoted = BODY + "\n\n> ⚠️ my own aside, not a registered notice"
    _q, stored, meta = await _store(quoted, {"response_mode": "enhanced"})
    assert stored == quoted and "delivery_notices" not in meta


def test_split_delivery_notices_pure():
    body, notices = handlers.split_delivery_notices(BODY + _notice())
    assert body == BODY and len(notices) == 1
    assert handlers.split_delivery_notices(BODY) == (BODY, [])
    assert handlers.split_delivery_notices("") == ("", [])


def test_parity_guard_ignores_the_display_only_notice_block():
    assert handlers._answer_bodies_agree(BODY + _notice(), BODY) is True
