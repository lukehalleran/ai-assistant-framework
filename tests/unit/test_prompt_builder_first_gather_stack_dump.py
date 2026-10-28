"""One-shot stack dump when the process's first full gather stalls.

Incident date: 2026-10-27 (3-10 s first-turn stalls with no log line; the
loop-stall monitor never reported more than ~3 s). Bug classes: BC-41, BC-70.
Drives the deployed build_prompt gather with a fake task that blocks the
event-loop thread inside a named function; config is read LIVE (BC-82).
"""
import asyncio
import faulthandler
import time

import pytest

import config.app_config as app_config
from core.prompt import builder as builder_mod
from tests.unit.test_independent_prompt_audit import full_builder, retrieval_limits

RECENT = [{"query": "Synthetic question", "response": "Synthetic retained answer."}]


def synthetic_blocking_worker():
    time.sleep(0.6)


@pytest.fixture
def stack_env(monkeypatch, tmp_path):
    path = tmp_path / "first_turn_stacks.log"
    monkeypatch.setattr(builder_mod, "_first_turn_stack_log_path", lambda: str(path))
    monkeypatch.setattr(builder_mod, "_FIRST_GATHER_CLAIMED", False)
    monkeypatch.setattr(builder_mod, "_STACK_DUMP_IN_TEST_MODE", True)
    monkeypatch.setattr(app_config, "LATENCY_DIAGNOSTICS_FIRST_TURN_STACK_DUMP_S", 0.2)
    return path


async def _gather(monkeypatch, slow):
    b = full_builder(monkeypatch, RECENT, budget=10000)

    async def recent(*args):
        if slow:
            synthetic_blocking_worker()
        return RECENT

    b.context_gatherer._get_recent_conversations = recent
    return await b.build_prompt("Synthetic question", retrieval_overrides=retrieval_limits())


@pytest.mark.asyncio
async def test_stalled_first_gather_dumps_stacks(monkeypatch, stack_env):
    await _gather(monkeypatch, slow=True)
    text = stack_env.read_text()
    assert "first gather > 0.2 s" in text
    assert "synthetic_blocking_worker" in text
    assert "Thread 0x" in text or "Current thread" in text


@pytest.mark.asyncio
async def test_second_gather_does_not_arm(monkeypatch, stack_env):
    await _gather(monkeypatch, slow=False)  # fast first gather claims the slot, header truncated
    assert stack_env.read_text() == ""
    armed = []
    real = faulthandler.dump_traceback_later
    monkeypatch.setattr(faulthandler, "dump_traceback_later", lambda *a, **k: armed.append(a) or real(*a, **k))
    await _gather(monkeypatch, slow=True)
    assert armed == []
    assert not stack_env.exists() or stack_env.read_text() == ""


@pytest.mark.asyncio
async def test_zero_disables_and_does_not_claim(monkeypatch, stack_env):
    monkeypatch.setattr(app_config, "LATENCY_DIAGNOSTICS_FIRST_TURN_STACK_DUMP_S", 0.0)
    armed = []
    monkeypatch.setattr(faulthandler, "dump_traceback_later", lambda *a, **k: armed.append(a))
    await _gather(monkeypatch, slow=True)
    assert armed == []
    assert builder_mod._FIRST_GATHER_CLAIMED is False
    assert not stack_env.exists()


@pytest.mark.asyncio
async def test_timer_is_cancelled_after_gather(monkeypatch, stack_env):
    monkeypatch.setattr(app_config, "LATENCY_DIAGNOSTICS_FIRST_TURN_STACK_DUMP_S", 0.3)
    await _gather(monkeypatch, slow=False)
    await asyncio.sleep(0.5)  # a leaked timer would fire into a closed file here
    assert stack_env.read_text() == ""


@pytest.mark.asyncio
async def test_test_mode_arms_nothing_and_creates_no_file(monkeypatch, stack_env):
    monkeypatch.setattr(builder_mod, "_STACK_DUMP_IN_TEST_MODE", False)
    monkeypatch.setenv("DAEMON_TEST_MODE", "1")
    armed = []
    monkeypatch.setattr(faulthandler, "dump_traceback_later", lambda *a, **k: armed.append(a))
    await _gather(monkeypatch, slow=True)
    assert armed == []
    assert not stack_env.exists()
    assert builder_mod._FIRST_GATHER_CLAIMED is False
