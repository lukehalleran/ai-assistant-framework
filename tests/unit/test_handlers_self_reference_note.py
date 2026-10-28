"""Behavioural sibling: the handler appends the self-reference note to the CURRENT query (source-text row).

Incident date: 2026-10-12 (a pasted message / attachment that names the assistant was read as
the assistant's own instructions). Class: BC-63 (a grep for ``self_reference_note(`` and
``_attachment_note`` stands in for "the note rides the current query like the attachment notes").

Drives THE deployed ``gui.handlers.handle_submit`` (harness: tests/unit/test_handle_submit.py)
with a paste-sized message that names the assistant and asserts what ``prepare_prompt``
receives; a short message addressing the assistant, and a paste that never names it, get no
note.
"""
import pytest

from tests.unit.test_handle_submit import _make_orchestrator, _run_submit

FILLER = "The retrieval layer scores candidate documents before a reranker orders them. " * 30
NAMING = "My side project Daemon is a RAG system with long-term memory. " + FILLER
WRAPPED_NAMING = "    My side project Daemon is a RAG system\n    with long-term memory.\n    " + FILLER
NOT_NAMING = "My side project is a RAG system with long-term memory. " + FILLER
NOTE_HEAD = "[SELF-REFERENCE NOTE]"


async def _received(text):
    """The prompt-only notes the handler hands the context pipeline for this turn
    (the single carrier that build_full_prompt renders into [CURRENT QUERY])."""
    import core.context_pipeline as cp
    seen = []
    real = cp.set_prompt_only_notes

    def _spy(notes):
        seen.append(notes)
        return real(notes)

    orch = _make_orchestrator()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cp, "set_prompt_only_notes", _spy)
        await _run_submit(text, orch)
    orch.prepare_prompt.assert_awaited_once()
    assert seen, "handler never set the prompt-only notes"
    # the analysis input (prepare_prompt user_input) must NOT carry the note
    assert NOTE_HEAD not in str(orch.prepare_prompt.call_args.kwargs.get("user_input"))
    return " ".join(str(n) for n in seen)


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [NAMING, WRAPPED_NAMING])
async def test_paste_naming_the_assistant_hands_the_note_to_the_prompt_only_carrier(text):
    assert NOTE_HEAD in await _received(text)


@pytest.mark.asyncio
async def test_paste_that_never_names_the_assistant_gets_no_note():
    assert NOTE_HEAD not in await _received(NOT_NAMING)


@pytest.mark.asyncio
async def test_short_message_addressing_the_assistant_gets_no_note():
    # a short message that names the assistant is ADDRESSING it, not describing it
    assert NOTE_HEAD not in await _received("Daemon is my side project, can you remember that?")
