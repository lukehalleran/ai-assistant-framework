"""F13b -- GUI inline turn paths + background store record a labelled
``storage_failed`` receipt (CGR-010) instead of a silent ``except: pass``.

Covers the three inline gui/handlers.py sites (``_run_doc_generation``,
``_save_daemon_note``, ``_run_action_retry``) and
``_background_store_interaction``'s transcript metadata, plus
utils/conversation_logger.py's text-format printer. Reuses F13a's
``core.orchestrator._storage_failure_label`` (imported locally at each
site, per the existing 912 precedent) rather than re-deriving the label
format.

Owner decision 4 (2026-09-14): no chat text changes on any path -- every
yielded chunk's ``content`` is asserted byte-for-byte identical between a
failure run and a success run for the same inputs.

Fakes only: LOCAL copies of the ctx/store shapes from
tests/unit/test_doc_conversation_source.py,
tests/unit/test_sep07_calendar_offer_continuation.py and
tests/unit/test_handle_submit.py -- no real Chroma/embedder/ModelManager,
no Gradio build, no real conversation-log directory (tmp_path only).
"""
import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.actions.types import ActionProposal, ActionType
from utils.retrieval_outcome import StoreWriteError

MARKER = "F13BMARKER_6a1f0c3d"


def _failing_store(source="store_interaction", reason="RuntimeError"):
    """AsyncMock store_interaction: raises StoreWriteError chained from a
    plain exception carrying MARKER (F13a pattern) -- a fix that logs or
    records {e} / the chained cause instead of the label would leak it."""
    async def _raise(*_a, **_kw):
        try:
            raise RuntimeError(MARKER)
        except RuntimeError as inner:
            raise StoreWriteError(source=source, reason=reason) from inner
    return AsyncMock(side_effect=_raise)


def _only_final(chunks):
    """Exactly one yielded chunk carries a debug record (contract #1)."""
    finals = [c for c in chunks if "debug" in c]
    assert len(finals) == 1
    return finals[0]


# ---------------------------------------------------------------------------
# 1. _run_doc_generation (gui/handlers.py ~1644)
# ---------------------------------------------------------------------------

class _FakeDocGenerator:
    def __init__(self, **_kw):
        pass

    async def generate(self, **_kw):
        return SimpleNamespace(
            title="X report", path="/tmp/x.md", doc_type="report",
            sources=[], sections_count=1, word_count=10,
        )


def _doc_ctx(memory_system, with_telemetry=True):
    ctx = SimpleNamespace(
        orchestrator=SimpleNamespace(
            prompt_builder=None, memory_system=memory_system,
            model_manager=SimpleNamespace(get_active_model_name=lambda: "test-model"),
        ),
        doc_gen_intent={"topic": "x", "doc_type": "report", "focus": None},
        user_text="write a report about x",
        history=[],
        handled=False,
    )
    if with_telemetry:
        ctx.telemetry = {}
    return ctx


def _run_doc_gen(ctx):
    import gui.handlers as handlers
    import knowledge.document_generator as dg_mod
    with patch.object(dg_mod, "DocumentGenerator", _FakeDocGenerator), \
         patch.object(handlers, "_write_turn_telemetry", lambda *a, **k: None), \
         patch.object(handlers, "_get_session_id", lambda *a: "s1"):
        async def _collect():
            return [c async for c in handlers._run_doc_generation(ctx)]
        return asyncio.run(_collect())


class TestDocGenerationStorageReceipt:
    def test_failure_sets_receipt_matches_success_content(self, caplog):
        ok_ctx = _doc_ctx(SimpleNamespace(store_interaction=AsyncMock(return_value="mem-1")))
        ok_final = _only_final(_run_doc_gen(ok_ctx))

        bad_ctx = _doc_ctx(SimpleNamespace(store_interaction=_failing_store()))
        with caplog.at_level(logging.WARNING, logger="gradio_gui"):
            bad_final = _only_final(_run_doc_gen(bad_ctx))

        assert bad_final["content"] == ok_final["content"]
        assert bad_ctx.handled is True
        label = "store_interaction: RuntimeError"
        assert bad_ctx.telemetry["storage_failed"] == label
        assert bad_final["debug"]["storage_failed"] == label
        assert label in caplog.text
        assert MARKER not in bad_final["content"]
        assert MARKER not in str(bad_final["debug"])
        assert MARKER not in caplog.text
        assert "storage_failed" not in ok_ctx.telemetry
        assert "storage_failed" not in ok_final["debug"]

    def test_memory_system_none_control(self):
        ctx = _doc_ctx(None)
        final = _only_final(_run_doc_gen(ctx))
        assert "storage_failed" not in ctx.telemetry
        assert "storage_failed" not in final["debug"]
        assert ctx.handled is True

    def test_store_interaction_returns_none_control(self):
        ctx = _doc_ctx(SimpleNamespace(store_interaction=AsyncMock(return_value=None)))
        final = _only_final(_run_doc_gen(ctx))
        assert "storage_failed" not in ctx.telemetry
        assert "storage_failed" not in final["debug"]


# ---------------------------------------------------------------------------
# 2. _save_daemon_note (gui/handlers.py ~2563)
# ---------------------------------------------------------------------------

def _note_result():
    r = MagicMock()
    r.title = "Note title"
    r.path = "daemon_notes/note.md"
    r.category = "implementation"
    r.id = "note-1"
    return r


def _note_ctx(memory_system, with_telemetry=True):
    ctx = SimpleNamespace(
        orchestrator=SimpleNamespace(
            memory_system=memory_system,
            model_manager=SimpleNamespace(get_active_model_name=lambda: "test-model"),
        ),
        user_text="remember this",
        handled=False,
    )
    if with_telemetry:
        ctx.telemetry = {}
    return ctx


def _run_self_note(ctx):
    import gui.handlers as handlers
    dnm = MagicMock()
    dnm.create_note = AsyncMock(return_value=_note_result())
    with patch.object(handlers, "_write_turn_telemetry", lambda *a, **k: None), \
         patch.object(handlers, "_get_session_id", lambda *a: "s1"), \
         patch("knowledge.daemon_notes_manager.DaemonNotesManager", return_value=dnm):
        async def _collect():
            return [c async for c in handlers._save_daemon_note(
                ctx, title="Note title", summary="A summary.")]
        return asyncio.run(_collect())


class TestSelfNoteStorageReceipt:
    def test_failure_sets_receipt_matches_success_content(self, caplog):
        ok_ctx = _note_ctx(SimpleNamespace(store_interaction=AsyncMock(return_value="mem-1")))
        ok_final = _only_final(_run_self_note(ok_ctx))

        bad_ctx = _note_ctx(SimpleNamespace(store_interaction=_failing_store()))
        with caplog.at_level(logging.WARNING, logger="gradio_gui"):
            bad_final = _only_final(_run_self_note(bad_ctx))

        assert bad_final["content"] == ok_final["content"]
        assert bad_ctx.handled is True
        label = "store_interaction: RuntimeError"
        assert bad_ctx.telemetry["storage_failed"] == label
        assert bad_final["debug"]["storage_failed"] == label
        assert label in caplog.text
        assert MARKER not in bad_final["content"]
        assert MARKER not in str(bad_final["debug"])
        assert MARKER not in caplog.text
        assert "storage_failed" not in ok_ctx.telemetry
        assert "storage_failed" not in ok_final["debug"]

    def test_memory_system_none_control(self):
        ctx = _note_ctx(None)
        final = _only_final(_run_self_note(ctx))
        assert "storage_failed" not in ctx.telemetry
        assert "storage_failed" not in final["debug"]
        assert ctx.handled is True

    def test_store_interaction_returns_none_control(self):
        ctx = _note_ctx(SimpleNamespace(store_interaction=AsyncMock(return_value=None)))
        final = _only_final(_run_self_note(ctx))
        assert "storage_failed" not in ctx.telemetry
        assert "storage_failed" not in final["debug"]

    def test_ctx_without_telemetry_attribute_is_handled_without_exception(self):
        """_run_pending_proposal's lightweight SimpleNamespace ctx (see
        R_common_rules NON-UNIT TESTS section) has no `telemetry` attr; the
        receipt code must be getattr-defensive and never raise."""
        ctx = _note_ctx(SimpleNamespace(store_interaction=_failing_store()),
                         with_telemetry=False)
        assert not hasattr(ctx, "telemetry")
        final = _only_final(_run_self_note(ctx))
        assert ctx.handled is True
        assert final["debug"]["storage_failed"] == "store_interaction: RuntimeError"


# ---------------------------------------------------------------------------
# 3. _run_action_retry (gui/handlers.py ~3470)
# ---------------------------------------------------------------------------

def _failed_proposal():
    return ActionProposal(
        action_type=ActionType.CALENDAR_CREATE_EVENT,
        params={"summary": "Standup", "start_time": "2026-09-15T09:00:00"},
        summary="calendar_create_event: Standup",
        error="Google token refresh failed.",
    )


class _FakeActionStore:
    def propose(self, proposal):
        return True


def _retry_ctx(memory_system, with_telemetry=True):
    ctx = SimpleNamespace(
        orchestrator=SimpleNamespace(
            memory_system=memory_system,
            model_manager=SimpleNamespace(get_active_model_name=lambda: "test-model"),
        ),
        user_text="try again",
        handled=False,
    )
    if with_telemetry:
        ctx.telemetry = {}
    return ctx


def _run_retry(ctx):
    import gui.handlers as handlers
    from core.agentic.tools import ToolExecutor
    with patch.object(ToolExecutor, "_get_pending_actions_store",
                       return_value=_FakeActionStore()), \
         patch.object(handlers, "_write_turn_telemetry", lambda *a, **k: None), \
         patch.object(handlers, "_get_session_id", lambda *a: "s1"):
        async def _collect():
            return [c async for c in handlers._run_action_retry(ctx, _failed_proposal())]
        return asyncio.run(_collect())


class TestActionRetryStorageReceipt:
    def test_failure_sets_receipt_matches_success_content(self, caplog):
        ok_ctx = _retry_ctx(SimpleNamespace(store_interaction=AsyncMock(return_value="mem-1")))
        ok_final = _only_final(_run_retry(ok_ctx))

        bad_ctx = _retry_ctx(SimpleNamespace(store_interaction=_failing_store()))
        with caplog.at_level(logging.WARNING, logger="gradio_gui"):
            bad_final = _only_final(_run_retry(bad_ctx))

        assert bad_final["content"] == ok_final["content"]
        assert bad_ctx.handled is True
        label = "store_interaction: RuntimeError"
        assert bad_ctx.telemetry["storage_failed"] == label
        assert bad_final["debug"]["storage_failed"] == label
        assert label in caplog.text
        assert MARKER not in bad_final["content"]
        assert MARKER not in str(bad_final["debug"])
        assert MARKER not in caplog.text
        assert "storage_failed" not in ok_ctx.telemetry
        assert "storage_failed" not in ok_final["debug"]
        # Action-retry-specific: the re-queued card id rides both chunks.
        assert ok_final["pending_action_id"] and bad_final["pending_action_id"]

    def test_memory_system_none_control(self):
        ctx = _retry_ctx(None)
        final = _only_final(_run_retry(ctx))
        assert "storage_failed" not in ctx.telemetry
        assert "storage_failed" not in final["debug"]
        assert ctx.handled is True
        assert final["pending_action_id"]

    def test_store_interaction_returns_none_control(self):
        ctx = _retry_ctx(SimpleNamespace(store_interaction=AsyncMock(return_value=None)))
        final = _only_final(_run_retry(ctx))
        assert "storage_failed" not in ctx.telemetry
        assert "storage_failed" not in final["debug"]

    def test_plain_runtime_error_gives_generic_label(self, caplog):
        """Also: a plain (non-StoreWriteError) exception still labels as
        "store_interaction: <ClassName>" via the shared helper."""
        ctx = _retry_ctx(SimpleNamespace(
            store_interaction=AsyncMock(side_effect=RuntimeError(f"boom {MARKER}"))))
        with caplog.at_level(logging.WARNING, logger="gradio_gui"):
            final = _only_final(_run_retry(ctx))
        label = "store_interaction: RuntimeError"
        assert ctx.telemetry["storage_failed"] == label
        assert final["debug"]["storage_failed"] == label
        assert MARKER not in caplog.text
        assert MARKER not in str(final["debug"])


# ---------------------------------------------------------------------------
# 4. _background_store_interaction (gui/handlers.py ~143) + _log_text
#    (utils/conversation_logger.py ~104)
# ---------------------------------------------------------------------------

class TestBackgroundStoreReceipt:
    @pytest.mark.asyncio
    async def test_failure_sets_metadata_receipt(self, caplog):
        import gui.handlers as handlers
        memory = SimpleNamespace(store_interaction=_failing_store())
        orchestrator = SimpleNamespace(memory_system=memory, current_topic="general")
        conversation_logger = MagicMock()

        with caplog.at_level(logging.ERROR, logger="gradio_gui"):
            await handlers._background_store_interaction(
                orchestrator=orchestrator, merged_input="hi",
                response_to_store="Clean response", tags=["t"],
                user_text="hi", final_output="hi", personality="default",
                file_names=[], conversation_logger=conversation_logger,
            )

        assert "Background storage failed" in caplog.text
        assert MARKER not in caplog.text
        conversation_logger.log_interaction.assert_called_once()
        meta = conversation_logger.log_interaction.call_args.kwargs["metadata"]
        assert meta["db_id"] is None
        assert meta["storage_failed"] == "store_interaction: RuntimeError"

    @pytest.mark.asyncio
    async def test_success_control_no_receipt_key(self):
        import gui.handlers as handlers
        memory = SimpleNamespace(store_interaction=AsyncMock(return_value="mem-42"))
        orchestrator = SimpleNamespace(memory_system=memory, current_topic="general")
        conversation_logger = MagicMock()

        await handlers._background_store_interaction(
            orchestrator=orchestrator, merged_input="hi",
            response_to_store="Clean response", tags=[],
            user_text="hi", final_output="hi", personality="default",
            file_names=[], conversation_logger=conversation_logger,
        )

        conversation_logger.log_interaction.assert_called_once()
        meta = conversation_logger.log_interaction.call_args.kwargs["metadata"]
        assert "storage_failed" not in meta
        assert meta["db_id"] == "mem-42"

    @pytest.mark.asyncio
    async def test_text_format_shows_label_only_on_the_failed_entry(self, tmp_path):
        import gui.handlers as handlers
        from utils.conversation_logger import ConversationLogger

        real_logger = ConversationLogger(log_dir=str(tmp_path), log_format="text")

        bad_memory = SimpleNamespace(store_interaction=_failing_store())
        bad_orch = SimpleNamespace(memory_system=bad_memory, current_topic="general")
        await handlers._background_store_interaction(
            orchestrator=bad_orch, merged_input="hi", response_to_store="Bad-path response",
            tags=[], user_text="hi", final_output="hi", personality="default",
            file_names=[], conversation_logger=real_logger,
        )

        ok_memory = SimpleNamespace(store_interaction=AsyncMock(return_value="mem-9"))
        ok_orch = SimpleNamespace(memory_system=ok_memory, current_topic="general")
        await handlers._background_store_interaction(
            orchestrator=ok_orch, merged_input="hi", response_to_store="Ok-path response",
            tags=[], user_text="hi", final_output="hi", personality="default",
            file_names=[], conversation_logger=real_logger,
        )

        text = real_logger.get_current_log_path().read_text(encoding="utf-8")
        assert "Storage failed: store_interaction: RuntimeError" in text
        assert MARKER not in text
        # Exactly one of the two logged entries carries the line.
        assert text.count("Storage failed:") == 1
