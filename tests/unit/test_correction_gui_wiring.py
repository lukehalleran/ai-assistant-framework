"""
Post-response truth pipeline must run on the GUI path (2026-08-23).

The 2026-08-23 log review: "6 weeks off vryalr not 1." never updated any
stored fact because CorrectionDetector / truth events / staleness cascade
were only invoked from the unused process_user_query — the same dead-wiring
class as the 08-18 EscalationTracker GUI bug. Fixes pinned here:

  1. orchestrator exposes public run_post_response_detectors(user_input),
     and the flow-based wrapper delegates to it (one pipeline, two entries).
  2. gui/handlers._write_turn_telemetry calls it per turn.
  3. CorrectionDetector gains a terse numeric-swap pattern and a
     message-level detect_correction_signal() API.
"""

import inspect

import pytest

from core.correction_detector import CorrectionDetector


class TestSourceWiring:
    """Source-level asserts, per the escalation-wiring test pattern."""

    def test_handlers_telemetry_calls_detectors(self):
        import gui.handlers as handlers
        src = inspect.getsource(handlers._write_turn_telemetry)
        assert "run_post_response_detectors" in src

    def test_orchestrator_wrapper_delegates_to_public_method(self):
        from core.orchestrator import DaemonOrchestrator
        wrapper_src = inspect.getsource(
            DaemonOrchestrator._run_post_response_detectors)
        assert "self.run_post_response_detectors(" in wrapper_src
        # The public method holds the real pipeline.
        public_src = inspect.getsource(
            DaemonOrchestrator.run_post_response_detectors)
        assert "detect_corrections" in public_src
        assert "detect_confirmations" in public_src
        assert "detect_entity_corrections" in public_src

    def test_process_user_query_path_still_wired(self):
        """The legacy flow path must keep running the SAME pipeline."""
        from core.orchestrator import DaemonOrchestrator
        src = inspect.getsource(DaemonOrchestrator.process_user_query)
        assert "_run_post_response_detectors" in src


class TestPublicMethodBehavior:
    def test_public_method_drives_detector(self):
        """run_post_response_detectors calls THE deployed detector methods."""
        from core.orchestrator import DaemonOrchestrator

        calls = []

        class SpyDetector:
            def detect_corrections(self, msg, facts):
                calls.append(("corrections", msg))
                return []

            def detect_confirmations(self, msg, facts):
                calls.append(("confirmations", msg))
                return []

            def detect_entity_corrections(self, msg):
                calls.append(("entity", msg))
                return []

            def detect_attributions(self, msg):
                calls.append(("attributions", msg))
                return []

        orch = DaemonOrchestrator.__new__(DaemonOrchestrator)
        orch.user_profile = object()  # truthy — gates the corrections block
        orch.correction_detector = SpyDetector()
        orch.memory_system = None
        orch._get_recent_profile_facts = lambda: [
            {"fact_id": "f1", "relation": "weeks_off_medication", "value": "1"}]

        import logging
        orch.logger = logging.getLogger("test_correction_wiring")

        orch.run_post_response_detectors("6 weeks off vryalr not 1.")

        kinds = {k for k, _ in calls}
        assert "corrections" in kinds
        assert "entity" in kinds
        assert all(m == "6 weeks off vryalr not 1." for _, m in calls)

    def test_public_method_never_raises_without_components(self):
        from core.orchestrator import DaemonOrchestrator
        orch = DaemonOrchestrator.__new__(DaemonOrchestrator)
        orch.user_profile = None
        orch.correction_detector = None
        orch.run_post_response_detectors("anything")  # no-op, no crash


class TestTerseNumericPattern:
    """The live 08-23 correction shape must now register as a signal."""

    def setup_method(self):
        self.det = CorrectionDetector()

    def test_live_message_scores_above_threshold(self):
        assert self.det.detect_correction_signal(
            "6 weeks off vryalr not 1.") >= 0.6

    def test_variants(self):
        for msg in ["3 weeks not 2", "10 mg not 20 mg.",
                    "2 days off it not 5"]:
            assert self.det.detect_correction_signal(msg) >= 0.6, msg

    def test_classic_patterns_still_score(self):
        assert self.det.detect_correction_signal(
            "actually it's 6 weeks") >= 0.6
        assert self.det.detect_correction_signal(
            "no, that's wrong") >= 0.6

    def test_non_corrections_score_zero(self):
        for msg in ["", "how are you today",
                    "I took 900 mg of lorvatin yesterday",
                    "there were 3 people and not everyone stayed for dinner",
                    "6 weeks is a long time"]:
            assert self.det.detect_correction_signal(msg) < 0.6, msg

    def test_detect_corrections_uses_signal(self):
        """The fact-level API builds on the message-level one."""
        src = inspect.getsource(CorrectionDetector.detect_corrections)
        assert "detect_correction_signal" in src

    def test_terse_numeric_matches_fact_with_value_overlap(self):
        facts = [{"fact_id": "f1", "relation": "weeks_off_medication",
                  "value": "1"}]
        events = self.det.detect_corrections("6 weeks off vryalr not 1.", facts)
        assert len(events) == 1
        assert events[0].event_type == "correction"
