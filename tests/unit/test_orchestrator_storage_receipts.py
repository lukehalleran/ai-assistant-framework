"""F13a — orchestrator records a labelled ``storage_failed`` receipt instead
of silently swallowing a failed turn write (CGR-010).

Covers the four store sites named in
docs/execution/generalization/failure_outcome_design.md's real-defects row:
``_handle_deictic``, ``_maybe_document_generation``, ``_maybe_agentic_search``
and ``_store_interaction`` (core/orchestrator.py). Each site's bare
``except: pass`` / ``{e}``-interpolated log becomes a labelled
``debug_info["storage_failed"]`` receipt; the turn still returns exactly one,
unchanged answer on a failed write. No user-visible change on any path
(contract 5) — no new turn record is asserted on the three early-return
sites (deictic/doc-gen/agentic), only on ``_store_interaction``'s normal
turn-record row.

Fakes only — ``DaemonOrchestrator.__new__`` construction via
``tests/unit/test_process_user_query.py``'s ``_make_flow_orch`` (imported,
not duplicated; that file is read-only for this batch), an ``AsyncMock``
memory_system, and a tmp ``config.app_config.TURN_TELEMETRY_PATH`` for the
one test that reads a written turn row (same technique as
``tests/unit/test_section_outcome_receipts.py``'s ``_patch_telemetry_path``).
"""
import json
import logging

import pytest
from unittest.mock import AsyncMock, MagicMock

from utils.retrieval_outcome import StoreWriteError
from tests.unit.test_process_user_query import _make_flow_orch, _agentic_stream

MARKER = "F13AMARKER_9d21fa6c"


def _failing_store(source="store_interaction", reason="RuntimeError"):
    """AsyncMock side effect: raises a StoreWriteError chained from a plain
    exception carrying MARKER, so a fix that ever logs/records {e} (or the
    chained cause) instead of the label would leak it."""
    async def _raise(*_a, **_kw):
        try:
            raise RuntimeError(MARKER)
        except RuntimeError as inner:
            raise StoreWriteError(source=source, reason=reason) from inner
    return AsyncMock(side_effect=_raise)


def _dump(debug):
    return json.dumps(debug, default=str)


# ---------------------------------------------------------------------------
# 1. _handle_deictic
# ---------------------------------------------------------------------------
class TestDeicticStorageFailed:
    @pytest.mark.asyncio
    async def test_failed_store_sets_receipt(self, monkeypatch, caplog):
        monkeypatch.setattr("core.orchestrator.is_deictic", lambda _t: True)
        orch = _make_flow_orch()
        orch.memory_system.get_memories = AsyncMock(
            return_value=[{"metadata": {"needs_clarification": True}}]
        )
        orch.memory_system.store_interaction = _failing_store()
        with caplog.at_level(logging.WARNING):
            text, debug = await orch.process_user_query("what about it")
        assert "not sure what you're referring to" in text
        assert debug["storage_failed"] == "store_interaction: RuntimeError"
        assert MARKER not in _dump(debug)
        assert MARKER not in caplog.text

    @pytest.mark.asyncio
    async def test_success_leaves_no_receipt(self, monkeypatch):
        monkeypatch.setattr("core.orchestrator.is_deictic", lambda _t: True)
        orch = _make_flow_orch()
        orch.memory_system.get_memories = AsyncMock(
            return_value=[{"metadata": {"needs_clarification": True}}]
        )
        text, debug = await orch.process_user_query("what about it")
        assert "not sure what you're referring to" in text
        assert "storage_failed" not in debug


# ---------------------------------------------------------------------------
# 2. _maybe_document_generation
# ---------------------------------------------------------------------------
class TestDocGenStorageFailed:
    @pytest.mark.asyncio
    async def test_failed_store_sets_receipt(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "knowledge.document_generator.detect_document_intent",
            lambda _q: {"topic": "black holes", "doc_type": "report", "focus": None},
        )
        orch = _make_flow_orch(agentic_controller=_agentic_stream("Doc ", "body"))
        orch.logger = logging.getLogger("f13a_test_docgen")
        orch.memory_system.store_interaction = _failing_store()
        with caplog.at_level(logging.WARNING):
            text, debug = await orch.process_user_query(
                "write a report on black holes", use_agentic_search=True
            )
        assert text == "Doc body"  # exactly one answer -- receipt code never raises
        assert "doc_gen_error" not in debug
        assert debug["storage_failed"] == "store_interaction: RuntimeError"
        assert MARKER not in _dump(debug)
        assert MARKER not in caplog.text

    @pytest.mark.asyncio
    async def test_success_leaves_no_receipt(self, monkeypatch):
        monkeypatch.setattr(
            "knowledge.document_generator.detect_document_intent",
            lambda _q: {"topic": "black holes", "doc_type": "report", "focus": None},
        )
        orch = _make_flow_orch(agentic_controller=_agentic_stream("Doc ", "body"))
        text, debug = await orch.process_user_query(
            "write a report on black holes", use_agentic_search=True
        )
        assert text == "Doc body"
        assert "storage_failed" not in debug


# ---------------------------------------------------------------------------
# 3. _maybe_agentic_search
# ---------------------------------------------------------------------------
class TestAgenticStorageFailed:
    @pytest.mark.asyncio
    async def test_failed_store_sets_receipt(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "knowledge.document_generator.detect_document_intent", lambda _q: None
        )
        decision = MagicMock()
        decision.should_search = True
        decision.search_terms = ["weather today"]
        monkeypatch.setattr(
            "utils.web_search_trigger.analyze_for_web_search_llm",
            AsyncMock(return_value=decision),
        )
        orch = _make_flow_orch(agentic_controller=_agentic_stream("Agentic ", "answer"))
        orch.logger = logging.getLogger("f13a_test_agentic")
        orch.memory_system.store_interaction = _failing_store()
        with caplog.at_level(logging.WARNING):
            text, debug = await orch.process_user_query(
                "search the web for weather", use_agentic_search=True
            )
        assert text == "Agentic answer"  # exactly one answer -- receipt code never raises
        assert "agentic_error" not in debug
        assert debug["storage_failed"] == "store_interaction: RuntimeError"
        assert MARKER not in _dump(debug)
        assert MARKER not in caplog.text
        # F6b proximity guard (test_section_outcome_receipts.py) is unaffected:
        # this edit sits above the section_outcomes/task_timings pair.
        assert "section_outcomes" in debug and "task_timings" in debug

    @pytest.mark.asyncio
    async def test_success_leaves_no_receipt(self, monkeypatch):
        monkeypatch.setattr(
            "knowledge.document_generator.detect_document_intent", lambda _q: None
        )
        decision = MagicMock()
        decision.should_search = True
        decision.search_terms = ["weather today"]
        monkeypatch.setattr(
            "utils.web_search_trigger.analyze_for_web_search_llm",
            AsyncMock(return_value=decision),
        )
        orch = _make_flow_orch(agentic_controller=_agentic_stream("Agentic ", "answer"))
        text, debug = await orch.process_user_query(
            "search the web for weather", use_agentic_search=True
        )
        assert text == "Agentic answer"
        assert "storage_failed" not in debug


# ---------------------------------------------------------------------------
# 4. _store_interaction (+ _run_post_response_hooks telemetry, + turn record)
# ---------------------------------------------------------------------------
class TestStoreInteractionStorageFailed:
    @pytest.mark.asyncio
    async def test_failed_store_no_attributeerror_receipt_and_turn_row(
        self, monkeypatch, tmp_path, caplog
    ):
        import config.app_config as config
        monkeypatch.setattr(config, "TURN_TELEMETRY_ENABLED", True)
        path = tmp_path / "turns.jsonl"
        monkeypatch.setattr(config, "TURN_TELEMETRY_PATH", str(path))

        import core.orchestrator as orchestrator_mod
        real_hooks = orchestrator_mod.run_post_response_hooks
        captured = {}

        def _spy(ctx):
            captured["telemetry"] = dict(ctx.telemetry)
            return real_hooks(ctx)

        monkeypatch.setattr(orchestrator_mod, "run_post_response_hooks", _spy)

        orch = _make_flow_orch()
        assert orch.logger is None  # the AttributeError hazard this fix must survive
        orch.memory_system.store_interaction = _failing_store()

        with caplog.at_level(logging.WARNING):
            text, debug = await orch.process_user_query("tell me about python")

        assert text == "Hello world"  # exactly one answer, unchanged
        assert debug["storage_failed"] == "store_interaction: RuntimeError"
        assert captured["telemetry"] == {"storage_failed": "store_interaction: RuntimeError"}
        assert MARKER not in _dump(debug)
        assert MARKER not in caplog.text

        row = json.loads(path.read_text().strip().splitlines()[-1])
        assert row["storage_failed"] == "store_interaction: RuntimeError"
        assert MARKER not in json.dumps(row)

    @pytest.mark.asyncio
    async def test_success_no_receipt_row_and_empty_hook_telemetry(self, monkeypatch, tmp_path):
        import config.app_config as config
        monkeypatch.setattr(config, "TURN_TELEMETRY_ENABLED", True)
        path = tmp_path / "turns.jsonl"
        monkeypatch.setattr(config, "TURN_TELEMETRY_PATH", str(path))

        import core.orchestrator as orchestrator_mod
        real_hooks = orchestrator_mod.run_post_response_hooks
        captured = {}

        def _spy(ctx):
            captured["telemetry"] = dict(ctx.telemetry)
            return real_hooks(ctx)

        monkeypatch.setattr(orchestrator_mod, "run_post_response_hooks", _spy)

        orch = _make_flow_orch()
        text, debug = await orch.process_user_query("tell me about python")

        assert text == "Hello world"
        assert "storage_failed" not in debug
        assert captured["telemetry"] == {}
        row = json.loads(path.read_text().strip().splitlines()[-1])
        assert "storage_failed" not in row

    @pytest.mark.asyncio
    async def test_deliberate_skip_return_none_leaves_no_receipt(self):
        orch = _make_flow_orch()
        orch.memory_system.store_interaction = AsyncMock(return_value=None)
        text, debug = await orch.process_user_query("tell me about python")
        assert text == "Hello world"
        assert "storage_failed" not in debug

    @pytest.mark.asyncio
    async def test_plain_exception_gives_fixed_label(self):
        """A non-StoreWriteError exception always labels as
        "store_interaction: <exception class name>" -- the fixed prefix
        names the write call, not the bypass site that happened to raise."""
        orch = _make_flow_orch()
        orch.memory_system.store_interaction = AsyncMock(side_effect=RuntimeError(MARKER))
        text, debug = await orch.process_user_query("tell me about python")
        assert text == "Hello world"
        assert debug["storage_failed"] == "store_interaction: RuntimeError"
        assert MARKER not in _dump(debug)
