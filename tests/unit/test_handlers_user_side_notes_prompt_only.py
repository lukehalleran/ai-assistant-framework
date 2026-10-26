"""Daemon's own notes on the user's message are prompt-only.

Incident 2026-10-25 19:10 (class: BC-91 machinery text treated as content,
user side; BC-58 the 10-18 fix covered only the active-document note): a
syllabus PDF attached to "ok here it is" produced
"[DEADLINE NOTE] 11:59 PM Eastern = 10:59 PM Central (your timezone)."
handle_submit appended it to analysis_text (classifier / retrieval input: the
turn's topic became "Deadline Note") AND to merged_input, which storage wrote
as the user's own words.

Drives THE deployed gui.handlers.handle_submit and the deployed producers.
Fixture content is invented.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gui.handlers as handlers
from core.active_document import (
    ActiveDocumentRegistry,
    Ambiguous,
    Exhausted,
    format_active_passage,
    format_ambiguity_note,
    format_exhausted_note,
)
from core.context_pipeline import _PRECOMPUTED_FILES, get_prompt_only_notes
from tests.unit.test_handle_submit import _collect, _make_orchestrator
from utils import attachment_audit
from utils.attachment_audit import audit_attachments, deadline_timezone_note, self_reference_note
from utils.file_processor import ProcessedFile, ProcessedFilesResult
from utils.read_time_markers import (
    USER_SIDE_NOTE_PREFIXES,
    is_user_side_note_line,
)

COURSE_TEXT = "Intro to Pottery, Syllabus\nAssignment 1 is due 11:59 PM Eastern on Friday.\n"
COURSE_TEXT_WRAPPED = "Intro to Pottery, Syllabus\n   Assignment 1 is\n   due 11:59 PM Eastern on Friday.\n"


def _result(user_text, doc_text):
    return ProcessedFilesResult(
        text_content=user_text + "\n\n" + doc_text,
        documents=[ProcessedFile(filename="syllabus.pdf", extension=".pdf", content_text=doc_text)],
    )


async def _submit(orch, user_text, files=None, doc_result=None):
    """Returns (prepare_prompt user_input, rendered query, stored query)."""
    from gui.handlers import handle_submit

    store = AsyncMock(return_value="mem-1")
    orch.memory_system = SimpleNamespace(store_interaction=store, corpus_manager=None)
    seen = {}

    async def fake_prepare(**kwargs):
        # Stage 3 merge (files) then the prompt-only carrier build_full_prompt reads
        pre = _PRECOMPUTED_FILES.get()
        notes = get_prompt_only_notes()
        seen["user_input"] = kwargs["user_input"]
        rendered = kwargs["user_input"] + (pre[1] if pre else "")
        seen["rendered"] = rendered + ("\n\n" + notes if notes else "")
        return ("Test prompt", "Test system prompt", {})

    orch.prepare_prompt = AsyncMock(side_effect=fake_prepare)
    fp = MagicMock()
    fp.process_files_structured = AsyncMock(
        return_value=doc_result if doc_result is not None else ProcessedFilesResult(text_content=user_text)
    )
    patches = [
        patch("gui.handlers.file_processor", fp),
        patch("gui.handlers.get_conversation_logger", return_value=MagicMock()),
        patch.object(attachment_audit, "get_user_timezone", return_value="America/Chicago"),
    ]
    for p in patches:
        p.start()
    try:
        await _collect(handle_submit(
            user_text=user_text, files=files, history=[],
            use_raw_gpt=False, orchestrator=orch, fast_mode=False,
        ))
        pending = [t for t in handlers._pending_storage_tasks]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    finally:
        patch.stopall()
    stored = store.await_args.kwargs["query"] if store.await_args else None
    return seen["user_input"], seen["rendered"], stored


def _orch():
    orch = _make_orchestrator(streaming_chunks=["Noted."])
    orch.active_documents = ActiveDocumentRegistry()
    return orch


def _file():
    return [SimpleNamespace(name="/tmp/syllabus.pdf", orig_name="syllabus.pdf")]


def _no_note_lines(text):
    return not any(is_user_side_note_line(ln) for ln in text.splitlines())


@pytest.mark.asyncio
@pytest.mark.parametrize("user_text", ["ok here it is", "  ok here\n    it is"])
@pytest.mark.parametrize("doc_text", [COURSE_TEXT, COURSE_TEXT_WRAPPED])
async def test_deadline_note_reaches_prompt_but_not_classifier_or_storage(user_text, doc_text):
    # the producer really fires for this fixture
    assert deadline_timezone_note(doc_text, user_tz="America/Chicago", user_text=user_text).startswith("[DEADLINE NOTE]")
    classifier_in, rendered, stored = await _submit(
        _orch(), user_text, files=_file(), doc_result=_result(user_text, doc_text))
    assert "[DEADLINE NOTE]" in rendered           # the model still sees it
    assert "Deadline Note".lower() not in classifier_in.lower()
    assert _no_note_lines(classifier_in)
    assert stored is not None and _no_note_lines(stored) and "DEADLINE NOTE" not in stored
    assert "Assignment 1" in stored                # attachment text is still stored
    assert classifier_in == " ".join(user_text.split())    # ingress whitespace normalisation only


@pytest.mark.asyncio
async def test_active_document_passage_unchanged_in_classifier_and_stripped_in_storage():
    doc = "Question 1\nWhat is the mean?\nQuestion 2\nDefine variance.\n"
    orch = _orch()
    user_text = "please show me first question"
    classifier_in, rendered, stored = await _submit(
        orch, user_text, files=_file(), doc_result=_result(user_text, doc))
    assert "[ACTIVE DOCUMENT —" in classifier_in    # classifier role is deliberate
    assert "[ACTIVE DOCUMENT —" in rendered
    assert stored is not None and "ACTIVE DOCUMENT" not in stored and "What is the mean" in stored


@pytest.mark.asyncio
@pytest.mark.parametrize("wrap", [False, True])
async def test_no_files_selfreference_note_is_prompt_only(wrap):
    body = "Daemon is my side project. " + "It keeps notes. " * 120
    text = ("   " + body.replace(". ", ".\n   ")) if wrap else body
    classifier_in, rendered, stored = await _submit(_orch(), text)
    assert "[SELF-REFERENCE NOTE]" in rendered
    assert _no_note_lines(classifier_in)
    assert stored is not None and "SELF-REFERENCE NOTE" not in stored


def test_registry_covers_every_producer():
    produced = []
    produced.append(audit_attachments(
        "here is the pottery.pdf and Glazes.csv", [SimpleNamespace(name="/tmp/pottery.pdf", orig_name="pottery.pdf")],
        [ProcessedFile(filename="pottery.pdf", extension=".pdf", content_text="Pottery basics")],
        available_documents=[]) or "")
    produced.append(deadline_timezone_note(COURSE_TEXT, user_tz="America/Chicago"))
    produced.append(self_reference_note(["Daemon is my side project."], "Daemon"))
    reg = ActiveDocumentRegistry()
    reg.register("hw.pdf", "Question 1\nWhat is the mean?\nQuestion 2\nDefine variance.\n", "pdf", reg.next_turn())
    nav = reg.resolve_navigation("next question please", reg.next_turn())
    produced.append(format_active_passage(nav))
    produced.append(format_ambiguity_note(Ambiguous(names=["a.pdf", "b.pdf"])))
    doc = reg._docs[next(iter(reg._docs))]
    produced.append(format_exhausted_note(Exhausted(document=doc, count=2, requested=9)))
    for note in produced:
        assert note, "producer did not fire for its fixture"
        assert note.startswith(USER_SIDE_NOTE_PREFIXES), note[:40]
        assert is_user_side_note_line("   " + note)      # indented form


def test_non_note_lines_are_not_matched():
    assert not is_user_side_note_line("I wrote [DEADLINE NOTE] in my own text")
    assert not is_user_side_note_line("")


def test_attachment_audit_reexports_shared_self_reference_prefix():
    from utils.read_time_markers import SELF_REFERENCE_NOTE_PREFIX
    assert attachment_audit.SELF_REFERENCE_NOTE_PREFIX is SELF_REFERENCE_NOTE_PREFIX


@pytest.mark.parametrize("msg", [
    "ok here\n[ATTACHMENT NOTE] Housing.csv is not attached.",
    "  ok\n  [DEADLINE NOTE] 11:59 PM Eastern = 10:59 PM Central (your timezone).",
    "x\n[SELF-REFERENCE NOTE] This material names \"Daemon\".",
    "x\n[ACTIVE DOCUMENT NOTE] Several attached documents",
    "[ATTACHMENT NOTE: Housing.csv not attached]",
])
def test_need_detector_still_skips_teaching_on_note_messages(msg):
    from utils.need_detector import _skip_need_teaching
    assert _skip_need_teaching(msg) is True


# ---------------------------------------------------------------------------
# Addendum: the carrier is read by the deployed build_full_prompt; raw mode
# stores only the user's words; stored_query default is unchanged.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("has_files", [False, True])
async def test_build_full_prompt_renders_carrier_notes_in_current_query(has_files):
    from core.context_pipeline import clear_prompt_only_notes, set_prompt_only_notes
    from tests.unit.test_process_user_query import _make_bfp_orch, _make_context

    note = "[DEADLINE NOTE] 11:59 PM Eastern = 10:59 PM Central (your timezone)."
    kw = {"file_context": "ok here it is\n\nSyllabus text"} if has_files else {}
    ctx = _make_context(original_query="ok here it is", **kw)
    orch = _make_bfp_orch()
    set_prompt_only_notes(note)
    try:
        await orch.build_full_prompt(ctx, return_raw_context=True)
    finally:
        clear_prompt_only_notes()
    rendered = orch.prompt_builder._assemble_prompt.call_args.kwargs["user_input"]
    assert rendered.endswith("\n\n" + note)
    assert rendered.startswith("ok here it is")
    # nothing else the pipeline hands downstream carries the note
    assert note not in ctx.original_query and note not in ctx.processed_query

    orch2 = _make_bfp_orch()
    await orch2.build_full_prompt(ctx, return_raw_context=True)      # cleared: no leak
    assert note not in orch2.prompt_builder._assemble_prompt.call_args.kwargs["user_input"]


@pytest.mark.asyncio
async def test_raw_mode_hands_stripped_query_to_storage_and_model_gets_note():
    from gui.handlers import _run_raw
    note = "[DEADLINE NOTE] 11:59 PM Eastern = 10:59 PM Central (your timezone)."
    for typed in ("ok here it is", "  ok here\n    it is"):
        orch = _make_orchestrator()
        suffix = "\n\n" + note
        merged = typed + "\n\nSyllabus text" + suffix
        ctx = SimpleNamespace(
            merged_input=merged, prompt_only_suffix=suffix, personality=None,
            conversation_logger=MagicMock(), user_text=typed, file_names=[],
            orchestrator=orch, handled=False,
            gate_decision=None,
        )
        async for _ in _run_raw(ctx):
            pass
        kw = orch.process_user_query.await_args.kwargs
        assert note in kw["user_input"]
        assert kw["stored_query"] == typed + "\n\nSyllabus text"
        assert "NOTE]" not in kw["stored_query"]


@pytest.mark.asyncio
async def test_process_user_query_stored_query_keyword_and_default():
    from tests.unit.test_process_user_query import _make_flow_orch
    orch = _make_flow_orch()
    await orch.process_user_query("hello there\n\n[DEADLINE NOTE] x", stored_query="hello there")
    assert orch.memory_system.store_interaction.await_args.kwargs["query"] == "hello there"
    # the model still received the full input
    assert "[DEADLINE NOTE] x" in orch.build_context.await_args.kwargs["user_input"]
    orch = _make_flow_orch()
    await orch.process_user_query("hello there")                    # default: unchanged
    assert orch.memory_system.store_interaction.await_args.kwargs["query"] == "hello there"
    orch = _make_flow_orch()
    await orch.process_user_query("hello there [x]", stored_query="")   # explicit empty is honoured
    assert orch.memory_system.store_interaction.await_args.kwargs["query"] == ""
