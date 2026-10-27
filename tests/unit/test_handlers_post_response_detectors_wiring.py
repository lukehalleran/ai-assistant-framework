"""Behavioural sibling for ``TestSourceWiring`` in
tests/unit/test_correction_gui_wiring.py (BC-63: it grepped
``_write_turn_telemetry`` / ``_run_post_response_detectors`` source for the
detector names and passed whether or not the pipeline ran).

Incident date: 2026-08-23 (CorrectionDetector, truth events and the staleness
cascade were dead on the GUI path). Bug classes: BC-63, BC-62. THE deployed
``gui.handlers._write_turn_telemetry`` and ``DaemonOrchestrator`` wrapper are
driven; the public detector entry records what it is called with.
"""
from types import SimpleNamespace

import pytest

from core.orchestrator import DaemonOrchestrator
from gui.handlers import _write_turn_telemetry

CORRECTION = "6 weeks off vryalr not 1."
WRAPPED = "    6 weeks off\n    vryalr not 1.\n   "


class _Tracker:
    def __init__(self):
        self.recorded = []

    def record_response(self, text):
        self.recorded.append(text)


def _orch(detector):
    orch = DaemonOrchestrator.__new__(DaemonOrchestrator)
    orch.run_post_response_detectors = detector
    orch.escalation_tracker = _Tracker()
    orch._last_turn_signals = {}
    return orch


def _ctx(orch, text):
    return SimpleNamespace(orchestrator=orch, telemetry={}, user_text=text,
                           t_prepare_elapsed=0.0)


@pytest.mark.parametrize("text", [CORRECTION, WRAPPED])
def test_turn_hook_runs_detectors_on_the_users_message(text):
    calls = []
    orch = _orch(calls.append)
    _write_turn_telemetry(_ctx(orch, text), "enhanced", "sess", "model", 5,
                          response_text="Noted.")
    assert calls == [text]


def test_blank_user_text_runs_no_detectors():
    calls = []
    orch = _orch(calls.append)
    _write_turn_telemetry(_ctx(orch, ""), "enhanced", "sess", "model", 5,
                          response_text="Noted.")
    assert calls == []


def test_detector_failure_does_not_block_other_hooks():
    def boom(_text):
        raise RuntimeError("detector down")

    orch = _orch(boom)
    _write_turn_telemetry(_ctx(orch, CORRECTION), "enhanced", "sess", "model", 5,
                          response_text="Noted.")
    assert orch.escalation_tracker.recorded == ["Noted."]


def test_flow_wrapper_delegates_to_the_public_pipeline():
    calls = []
    orch = _orch(calls.append)
    DaemonOrchestrator._run_post_response_detectors(
        orch, SimpleNamespace(user_input=CORRECTION))
    assert calls == [CORRECTION]
