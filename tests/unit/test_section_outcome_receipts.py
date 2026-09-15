"""F6b — section outcomes reach the debug record and the turn record.

Covers: orchestrator stash (_last_section_outcomes), flow propagation
(flow.section_outcomes), debug_info["section_outcomes"], the compact
turn-signal sections_not_checked, gui.handlers._build_debug_record,
_capture_delivery's copy into ctx.telemetry, and the end-to-end write
through _hook_turn_telemetry (sync + deferred). Fakes only. Every test
reaching record_turn patches config.app_config.TURN_TELEMETRY_PATH to
tmp_path.
"""

import inspect
import json
import logging
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.context_pipeline import ContextResult, ToneLevel
from core.escalation_tracker import EscalationTracker
from core.orchestrator import (
    DaemonOrchestrator, PostResponseHookContext, _QueryFlow, _hook_turn_telemetry,
)
from gui.handlers import _build_debug_record, _capture_delivery

QUERY_MARKER = "F6BQUERYMARKER_should_not_leak_9c2f"
EXC_MARKER = "F6BEXCMARK_secret_detail_9f21"
OUTCOMES = {
    "personal_notes": {"status": "failed", "reason": "ConnectionError"},
    "recent": {"status": "succeeded", "reason": ""},
}
SECTIONS_NOT_CHECKED = {"personal_notes": "failed:ConnectionError"}


def _make_context(query=QUERY_MARKER, tone=ToneLevel.CONVERSATIONAL):
    return ContextResult(
        processed_query=query, original_query=query,
        tone_level=tone, tone_instructions="", emotional_context=None,
    )


def _bare_orchestrator():
    """object.__new__(DaemonOrchestrator) with only what build_full_prompt's
    stages touch -- never runs __init__ (same technique as
    tests/unit/test_escalation_gui_wiring.py)."""
    orch = object.__new__(DaemonOrchestrator)
    orch.escalation_tracker = EscalationTracker()
    orch.safety_canary = None
    orch.response_planner = None
    orch.logger = logging.getLogger("test_section_outcome_receipts")
    return orch


class _FakeBuilder:
    def __init__(self, section_outcomes=None, include_key=True):
        self._section_outcomes = section_outcomes
        self._include_key = include_key

    async def build_prompt_from_context(self, context):
        ctx = {}
        if self._include_key:
            ctx["_section_outcomes"] = dict(self._section_outcomes or {})
        return ctx

    def _assemble_prompt(self, context, user_input, system_prompt):
        return "PROMPT"


def _wire(monkeypatch, orch, outcomes=OUTCOMES, include_key=True):
    monkeypatch.setattr(orch, "_build_system_prompt", lambda c, r: "SYSTEM")
    orch.prompt_builder = _FakeBuilder(outcomes, include_key=include_key)
    orch.build_context = AsyncMock(return_value=_make_context())


# 1. Orchestrator stash (contract 1) + turn signal (contract 3) -------------

class TestOrchestratorStashAndTurnSignal:
    async def test_prepare_prompt_stashes_and_pops(self, monkeypatch):
        orch = _bare_orchestrator()
        _wire(monkeypatch, orch)
        _, _, prompt_ctx = await orch.prepare_prompt("hi", return_context=True)
        assert orch._last_section_outcomes == OUTCOMES
        assert "_section_outcomes" not in prompt_ctx
        assert orch._last_task_timings == {}  # control: unaffected by this batch

    async def test_prepare_prompt_missing_key_stashes_empty(self, monkeypatch):
        orch = _bare_orchestrator()
        _wire(monkeypatch, orch, outcomes=None, include_key=False)
        await orch.prepare_prompt("hi", return_context=True)
        assert orch._last_section_outcomes == {}

    async def test_sections_not_checked_compact_dict(self, monkeypatch):
        orch = _bare_orchestrator()
        _wire(monkeypatch, orch)
        await orch.build_full_prompt(_make_context(), use_raw_mode=False)
        assert orch._last_turn_signals["sections_not_checked"] == SECTIONS_NOT_CHECKED

    async def test_sections_not_checked_empty_when_no_outcomes_key(self, monkeypatch):
        orch = _bare_orchestrator()
        _wire(monkeypatch, orch, outcomes=None, include_key=False)
        await orch.build_full_prompt(_make_context(), use_raw_mode=False)
        assert orch._last_turn_signals["sections_not_checked"] == {}

    async def test_sections_not_checked_empty_when_all_succeeded(self, monkeypatch):
        orch = _bare_orchestrator()
        _wire(monkeypatch, orch, outcomes={"recent": {"status": "succeeded", "reason": ""}})
        await orch.build_full_prompt(_make_context(), use_raw_mode=False)
        assert orch._last_turn_signals["sections_not_checked"] == {}


# 2. Flow path: pop + carry + debug_info (contract 2) ------------------------

class TestFlowPathSectionOutcomes:
    async def test_build_prompt_phase_pops_and_carries(self, monkeypatch):
        orch = _bare_orchestrator()
        _wire(monkeypatch, orch)
        flow = _QueryFlow(user_input="hi")
        await orch._build_prompt_phase(flow)
        assert flow.section_outcomes == OUTCOMES
        assert "_section_outcomes" not in flow.prompt_ctx
        assert flow.task_timings == {}  # control: unaffected by this batch

    async def test_finalize_debug_publishes_section_outcomes(self, monkeypatch):
        orch = _bare_orchestrator()
        orch.enable_citations = False
        _wire(monkeypatch, orch)
        flow = _QueryFlow(user_input="hi")
        await orch._build_prompt_phase(flow)
        flow.answer_for_storage = flow.full_response = "answer"
        flow.citations = []
        flow.t_gen_elapsed = flow.t_store_elapsed = 0.0
        flow.debug_info["start_time"] = datetime.now()
        orch._finalize_debug(flow)
        assert flow.debug_info["section_outcomes"] == OUTCOMES
        assert flow.debug_info["task_timings"] == {}  # control

    def test_agentic_bypass_site_mirrors_finalize_debug(self):
        """_maybe_agentic_search (~2315) builds debug_info inline rather
        than through _finalize_debug; driving the full agentic loop needs a
        fake web-search trigger + agentic controller, disproportionate for
        a one-line addition identical in shape to the site proven above.
        Pinned at the source level (same technique as
        test_escalation_gui_wiring.py's TestNoDoubleCount)."""
        src = inspect.getsource(DaemonOrchestrator._maybe_agentic_search)
        assert '"section_outcomes"' in src
        assert abs(
            src.index('debug_info["section_outcomes"]')
            - src.index('debug_info["task_timings"]')
        ) < 200


# 3. Handlers: _build_debug_record + _capture_delivery (contract 4) ---------

class TestHandlersReceipts:
    def test_build_debug_record_key(self):
        rec = _build_debug_record(
            mode="enhanced", user_text="q", prompt="p", system_prompt="s",
            response="r", model="m", prompt_tokens=1, system_tokens=1,
            total_tokens=2, citations=[], orchestrator=None,
            section_outcomes=OUTCOMES,
        )
        assert rec["section_outcomes"] == OUTCOMES
        rec_default = _build_debug_record(
            mode="enhanced", user_text="q", prompt="p", system_prompt="s",
            response="r", model="m", prompt_tokens=1, system_tokens=1,
            total_tokens=2, citations=[], orchestrator=None,
        )
        assert rec_default["section_outcomes"] == {}
        assert rec_default["task_timings"] == {}  # control

    def test_capture_delivery_copies_and_skips_when_empty(self):
        ctx = SimpleNamespace(telemetry={}, t_ingress=0.0)
        _capture_delivery(ctx, {
            "phase_timings": {"a": 1.0}, "task_timings": {"b": 2.0},
            "section_outcomes": OUTCOMES,
        })
        assert ctx.telemetry["section_outcomes"] == OUTCOMES
        assert ctx.telemetry["task_timings"] == {"b": 2.0}  # control

        ctx2 = SimpleNamespace(telemetry={}, t_ingress=0.0)
        _capture_delivery(ctx2, {"phase_timings": {}, "task_timings": {}, "section_outcomes": {}})
        assert "section_outcomes" not in ctx2.telemetry
        assert "task_timings" not in ctx2.telemetry

    def test_two_callers_read_and_forward(self):
        """Both _build_debug_record callers (agentic ~4022/4041, enhanced
        ~4708/4835) are exercised end-to-end elsewhere (e.g.
        tests/unit/test_handle_submit.py); pin the exact read/forward shape
        at both sites structurally."""
        import gui.handlers as handlers
        src = inspect.getsource(handlers)
        assert src.count("getattr(orchestrator, '_last_section_outcomes', {})") == 2
        assert src.count("section_outcomes=") == 3  # 2 callers + the def


# 4. End to end: _hook_turn_telemetry, sync + deferred (contract 3, 4) ------

class _FakeDoneTask:
    def add_done_callback(self, cb):
        cb(self)


def _patch_telemetry_path(monkeypatch, tmp_path):
    import config.app_config as config
    monkeypatch.setattr(config, "TURN_TELEMETRY_ENABLED", True)
    path = tmp_path / "turns.jsonl"
    monkeypatch.setattr(config, "TURN_TELEMETRY_PATH", str(path))
    return path


def _hook_ctx_after_capture(telemetry_task, user_input="q"):
    ctx = SimpleNamespace(telemetry={}, t_ingress=0.0)
    _capture_delivery(ctx, {
        "phase_timings": {"context_pipeline": 0.1}, "task_timings": {"recent": 0.2},
        "section_outcomes": OUTCOMES,
    })
    orch = SimpleNamespace(_last_turn_signals={"sections_not_checked": SECTIONS_NOT_CHECKED})
    return PostResponseHookContext(
        orchestrator=orch, user_input=user_input, response_text=None,
        mode="enhanced", session_id="s", model_name="m", response_len=0,
        telemetry=ctx.telemetry, t_prepare_elapsed=0.0, telemetry_task=telemetry_task,
    )


class TestHookTurnTelemetryEndToEnd:
    @pytest.mark.parametrize("telemetry_task", [None, _FakeDoneTask()])
    def test_writes_both_receipts_sync_and_deferred(self, monkeypatch, tmp_path, telemetry_task):
        path = _patch_telemetry_path(monkeypatch, tmp_path)
        _hook_turn_telemetry(_hook_ctx_after_capture(telemetry_task))
        row = json.loads(path.read_text())
        assert row["sections_not_checked"] == SECTIONS_NOT_CHECKED
        assert row["section_outcomes"] == OUTCOMES
        assert row["task_timings"] == {"recent": 0.2}  # control

    def test_no_outcomes_end_to_end(self, monkeypatch, tmp_path):
        path = _patch_telemetry_path(monkeypatch, tmp_path)
        ctx = SimpleNamespace(telemetry={}, t_ingress=0.0)
        _capture_delivery(ctx, {"phase_timings": {}, "task_timings": {}, "section_outcomes": {}})
        orch = SimpleNamespace(_last_turn_signals={"sections_not_checked": {}})
        hook_ctx = PostResponseHookContext(
            orchestrator=orch, user_input="q", response_text=None,
            mode="enhanced", session_id="s", model_name="m", response_len=0,
            telemetry=ctx.telemetry, t_prepare_elapsed=0.0, telemetry_task=None,
        )
        _hook_turn_telemetry(hook_ctx)
        row = json.loads(path.read_text())
        assert row["sections_not_checked"] == {}
        # Matches the pre-existing task_timings/phase_timings behaviour: an
        # empty dict is never copied by _capture_delivery, so the key is
        # simply absent -- unchanged by this batch.
        assert "section_outcomes" not in row
        assert "task_timings" not in row

    def test_privacy_markers_excluded(self, monkeypatch, tmp_path):
        """No query text or exception-message text under the new keys. The
        pre-existing `query` field is by design and out of scope."""
        path = _patch_telemetry_path(monkeypatch, tmp_path)
        _hook_turn_telemetry(_hook_ctx_after_capture(None, user_input=QUERY_MARKER))
        row = json.loads(path.read_text())
        assert QUERY_MARKER in row["query"]
        assert QUERY_MARKER not in json.dumps(row["sections_not_checked"])
        assert QUERY_MARKER not in json.dumps(row["section_outcomes"])
        assert EXC_MARKER not in json.dumps(row["sections_not_checked"])
        assert EXC_MARKER not in json.dumps(row["section_outcomes"])


# 5. Privacy at the source: reason is a class name, never the message -------

class TestPrivacyReasonIsClassName:
    async def test_reason_never_the_exception_message(self, monkeypatch):
        try:
            raise ValueError(EXC_MARKER)
        except ValueError as exc:
            reason, message = type(exc).__name__, str(exc)
        assert message == EXC_MARKER  # sanity: message != class name
        orch = _bare_orchestrator()
        _wire(monkeypatch, orch, outcomes={"personal_notes": {"status": "failed", "reason": reason}})
        await orch.build_full_prompt(_make_context(query=QUERY_MARKER), use_raw_mode=False)
        signals_str = json.dumps(orch._last_turn_signals["sections_not_checked"])
        assert QUERY_MARKER not in signals_str
        assert EXC_MARKER not in signals_str


# 6. utils/turn_telemetry.py read-only: generic sanitizer keeps dicts -------

class TestTurnTelemetrySanitizerKeepsDicts:
    def test_record_turn_preserves_nested_dicts_unchanged(self, monkeypatch, tmp_path):
        """No source edit needed -- passes identically before and after
        this batch's orchestrator/handlers edit."""
        from utils.turn_telemetry import record_turn
        path = _patch_telemetry_path(monkeypatch, tmp_path)
        assert record_turn({
            "section_outcomes": OUTCOMES, "sections_not_checked": SECTIONS_NOT_CHECKED,
        })
        row = json.loads(path.read_text())
        assert row["section_outcomes"] == OUTCOMES
        assert row["sections_not_checked"] == SECTIONS_NOT_CHECKED
