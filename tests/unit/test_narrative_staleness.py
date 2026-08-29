"""
Narrative correction-staleness flag (2026-08-23).

The 08-23 session: "6 weeks off vryalr not 1." corrected a duration, but the
narrative context regenerates only at shutdown — the stale "day 8" framing
kept re-entering every prompt's [TEMPORAL GROUNDING] section. The flag
(utils/narrative_staleness.py) doesn't rewrite the narrative; it makes the
prompt honest about it:

  correction signal ≥0.6 in run_post_response_detectors → mark_stale()
  get_narrative_context() → CAUTION line when flag is newer than the file
  save_narrative_context() → clear() (fresh narrative saw the correction)

Drives THE deployed functions; flag + narrative paths sandboxed per test.
"""

import inspect
import os
import time

import pytest

from utils import narrative_staleness as ns


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("NARRATIVE_STALE_FLAG_PATH",
                       str(tmp_path / "narrative_stale.json"))
    yield tmp_path


class TestFlagPrimitives:
    def test_mark_then_is_stale_against_older_narrative(self):
        assert ns.mark_stale("test correction")
        assert ns.is_stale(time.time() - 3600)  # narrative generated an hour ago

    def test_not_stale_when_narrative_newer_than_mark(self):
        ns.mark_stale("test correction")
        assert not ns.is_stale(time.time() + 3600)

    def test_no_flag_is_not_stale(self):
        assert not ns.is_stale(0.0)

    def test_clear_removes_flag(self):
        ns.mark_stale("test correction")
        ns.clear()
        assert not ns.is_stale(0.0)
        ns.clear()  # idempotent, no crash

    def test_repeated_marks_keep_earliest(self, _sandbox):
        """A later correction must not hide an earlier one behind a
        narrative regenerated in between."""
        ns.mark_stale("first")
        import json
        flag_file = _sandbox / "narrative_stale.json"
        first = json.loads(flag_file.read_text())["marked_at"]
        time.sleep(0.01)
        ns.mark_stale("second")
        data = json.loads(flag_file.read_text())
        assert data["marked_at"] == first
        assert data["reason"] == "second"

    def test_corrupt_flag_is_lenient(self, _sandbox):
        (_sandbox / "narrative_stale.json").write_text("{not json")
        assert not ns.is_stale(0.0)
        assert ns.mark_stale("recovers")  # overwrites the corrupt file
        assert ns.is_stale(0.0)


class TestCorpusManagerIntegration:
    def _cm(self, tmp_path, monkeypatch):
        import config.app_config as app_config
        from memory.corpus_manager import CorpusManager
        narrative = tmp_path / "narrative_context.txt"
        monkeypatch.setattr(app_config, "NARRATIVE_CONTEXT_PATH", str(narrative))
        cm = CorpusManager(corpus_file=str(tmp_path / "corpus.json"))
        return cm, narrative

    def test_caution_appended_when_correction_after_generation(
            self, _sandbox, monkeypatch):
        cm, narrative = self._cm(_sandbox, monkeypatch)
        assert cm.save_narrative_context("User is on day 8 off the medication.")
        # Backdate the narrative file so the mark lands after it.
        old = time.time() - 3600
        os.utime(str(narrative), (old, old))
        ns.mark_stale("6 weeks off vryalr not 1.")
        out = cm.get_narrative_context()
        assert "day 8" in out
        assert "[CAUTION:" in out
        assert "corrected" in out

    def test_no_caution_without_flag(self, _sandbox, monkeypatch):
        cm, _ = self._cm(_sandbox, monkeypatch)
        cm.save_narrative_context("User is doing well.")
        assert "[CAUTION:" not in cm.get_narrative_context()

    def test_fresh_save_clears_flag(self, _sandbox, monkeypatch):
        cm, narrative = self._cm(_sandbox, monkeypatch)
        cm.save_narrative_context("Old narrative.")
        old = time.time() - 3600
        os.utime(str(narrative), (old, old))
        ns.mark_stale("correction")
        assert "[CAUTION:" in cm.get_narrative_context()
        # Regeneration (which saw the corrected conversation) supersedes.
        cm.save_narrative_context("New narrative reflecting the correction.")
        out = cm.get_narrative_context()
        assert "[CAUTION:" not in out
        assert "New narrative" in out

    def test_refused_sentinel_save_does_not_clear_flag(
            self, _sandbox, monkeypatch):
        cm, narrative = self._cm(_sandbox, monkeypatch)
        cm.save_narrative_context("Good narrative.")
        old = time.time() - 3600
        os.utime(str(narrative), (old, old))
        ns.mark_stale("correction")
        assert not cm.save_narrative_context("[API Error] 402 whatever")
        assert "[CAUTION:" in cm.get_narrative_context()


class TestOrchestratorWiring:
    def test_detector_pipeline_marks_stale(self):
        src_module = inspect.getsource(
            __import__("core.orchestrator", fromlist=["DaemonOrchestrator"])
            .DaemonOrchestrator.run_post_response_detectors)
        assert "mark_stale" in src_module
        assert "detect_correction_signal" in src_module

    def test_live_correction_message_writes_flag(self, _sandbox):
        """End-to-end: the 08-23 message through THE deployed pipeline."""
        import logging
        from core.correction_detector import CorrectionDetector
        from core.orchestrator import DaemonOrchestrator

        orch = DaemonOrchestrator.__new__(DaemonOrchestrator)
        orch.user_profile = None  # skips the fact-level block
        orch.correction_detector = CorrectionDetector()
        orch.memory_system = None
        orch.logger = logging.getLogger("test_narrative_staleness")

        orch.run_post_response_detectors("6 weeks off vryalr not 1.")
        assert ns.is_stale(0.0)

    def test_non_correction_message_does_not_flag(self, _sandbox):
        import logging
        from core.correction_detector import CorrectionDetector
        from core.orchestrator import DaemonOrchestrator

        orch = DaemonOrchestrator.__new__(DaemonOrchestrator)
        orch.user_profile = None
        orch.correction_detector = CorrectionDetector()
        orch.memory_system = None
        orch.logger = logging.getLogger("test_narrative_staleness")

        orch.run_post_response_detectors("how are you doing today")
        assert not ns.is_stale(0.0)
