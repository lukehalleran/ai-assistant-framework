"""The first-turn stack log is bounded by the startup log maintenance pass.

Incident date: 2026-10-27 (new diagnostics log). Bug classes: BC-41, BC-70.
Drives the deployed run_startup_log_maintenance with a stack log over the cap.
"""
import os

import utils.log_rotation as lr


def test_oversized_stack_log_is_rotated(tmp_path, monkeypatch):
    stacks = tmp_path / "logs" / "first_turn_stacks.log"
    stacks.parent.mkdir()
    stacks.write_text("x" * 3_000_000)
    monkeypatch.setattr(lr, "store_path", lambda rel, frozen=None: str(tmp_path / rel))
    monkeypatch.setattr("config.app_config.LOG_MAINTENANCE_DAILY_NOTES_MAX_MB", 1.0)
    monkeypatch.setattr("config.app_config.TURN_TELEMETRY_PATH", str(tmp_path / "logs" / "turn_records.jsonl"))
    summary = lr.run_startup_log_maintenance()
    assert summary["rotated"] >= 1
    assert not stacks.exists() and os.path.exists(str(stacks) + ".1")
