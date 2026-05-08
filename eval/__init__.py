"""
Daemon Prompt Eval System — Phase 1: Instrumentation & Snapshot Replay.

This module provides measurement infrastructure for prompt section ablation testing.
It captures prompt snapshots, replays them deterministically, and generates LLM
responses without any persistence side effects.

Environment Variables:
    DAEMON_EVAL_CAPTURE: Set to "1" to enable snapshot capture in the builder hook.
    DAEMON_EVAL_CAPTURE_STRICT: Set to "1" to raise on capture/replay errors.
    DAEMON_EVAL_SNAPSHOT_DIR: Override snapshot save directory (default: eval/snapshots).
"""
