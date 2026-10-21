"""The active-document navigation note/passage is prompt-only.

Incident 2026-10-18 (class: BC-91 machinery stored as user content): the
navigation note and the appended slide passage were added to merged_input,
which is what storage writes as the user's query.

Drives THE deployed gui.handlers._dispatch_storage with the exact suffix the
handler builds from the deployed format_* functions.
"""
import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from core.active_document import (
    ActiveDocumentRegistry,
    format_active_passage,
)
import gui.handlers as handlers

DOC = "Question 1\nWhat is the mean?\nQuestion 2\nDefine variance.\n"


def _run_dispatch(merged_input, suffix):
    store = AsyncMock(return_value="mem-1")
    orch = SimpleNamespace(
        memory_system=SimpleNamespace(store_interaction=store, corpus_manager=None),
        current_topic="general",
    )
    async def go():
        task = handlers._dispatch_storage(
            orch, merged_input, "reply", "next question please", "reply",
            None, [], MagicMock(), "sess", None, "enhanced",
            prompt_only_suffix=suffix,
        )
        await task
    asyncio.run(go())
    return store.await_args.kwargs["query"]


def test_navigation_passage_is_not_stored_in_query():
    reg = ActiveDocumentRegistry()
    reg.register("hw.pdf", DOC, "pdf", reg.next_turn())
    nav = reg.resolve_navigation("next question please", reg.next_turn())
    for base in ("next question please", "  next question\n   please"):
        suffix = "\n\n" + format_active_passage(nav)
        stored = _run_dispatch(base + suffix, suffix)
        assert stored == base
        assert "ACTIVE DOCUMENT" not in stored and "What is the mean" not in stored


def test_no_suffix_leaves_query_untouched():
    assert _run_dispatch("hello there", "") == "hello there"


def test_every_dispatch_site_passes_the_suffix():
    src = inspect.getsource(handlers)
    calls = src.count("= _dispatch_storage(")
    assert calls >= 4
    assert src.count('prompt_only_suffix=getattr(ctx, "prompt_only_suffix", "")') == calls
