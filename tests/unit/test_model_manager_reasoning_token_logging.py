"""[PromptCache] lines carry reasoning tokens and call duration.

Incident date: 2026-10-24 (lane AGT). Bug class: BC-72 (unobservable decision:
a 25 s / 321-token agentic synthesis could not be explained because the usage
line dropped the provider's reasoning-token count and the call duration).
Drives the DEPLOYED ``ModelManager._log_cache_usage`` and
``generate_once_with_tools`` with fake usage objects / transports.
"""
import logging
from types import SimpleNamespace

import pytest

from models.model_manager import ModelManager


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(ModelManager, "_get_cached_embedder", staticmethod(lambda: None))
    mm = ModelManager(api_key="test-key")
    mm.switch_model("kimi-3")
    return mm


def _line(caplog):
    lines = [r.getMessage() for r in caplog.records if "[PromptCache]" in r.getMessage()]
    assert len(lines) == 1, caplog.text
    return lines[0]


def test_reasoning_tokens_from_dict_details_are_logged(manager, caplog):
    usage = SimpleNamespace(
        prompt_tokens=1000, completion_tokens=900,
        prompt_tokens_details={"cached_tokens": 0},
        completion_tokens_details={"reasoning_tokens": 640},
    )
    with caplog.at_level(logging.INFO):
        manager._log_cache_usage(usage, model_name="kimi-3", where="agentic_synthesis")
    line = _line(caplog)
    assert "reasoning_tokens=640" in line
    assert "duration_s" not in line  # no timing supplied


def test_reasoning_tokens_from_object_details_are_logged(manager, caplog):
    usage = SimpleNamespace(
        prompt_tokens=10, completion_tokens=5,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
    )
    with caplog.at_level(logging.INFO):
        manager._log_cache_usage(usage, model_name="m", where="w")
    assert "reasoning_tokens=3" in _line(caplog)


def test_missing_details_logs_none_and_original_prefix_unchanged(manager, caplog):
    usage = SimpleNamespace(prompt_tokens=20, completion_tokens=7)
    with caplog.at_level(logging.INFO):
        manager._log_cache_usage(usage, model_name="m", where="w", duration_s=1.234)
    line = _line(caplog)
    assert line.startswith("[PromptCache] MISS model=m where=w prompt=20 completion=7 cache_read=0 cache_write=0 cache_discount=None")
    assert "reasoning_tokens=None" in line
    assert line.endswith("duration_s=1.23")


@pytest.mark.asyncio
async def test_tool_round_logs_its_wall_duration(manager, caplog):
    usage = SimpleNamespace(
        prompt_tokens=30, completion_tokens=12,
        completion_tokens_details={"reasoning_tokens": 0},
    )
    message = SimpleNamespace(content="ok", tool_calls=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    async def _create(**kwargs):
        return response

    manager.async_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    tools = [{"type": "function", "function": {"name": "noop", "parameters": {}}}]
    with caplog.at_level(logging.INFO):
        out = await manager.generate_once_with_tools(prompt="p", model_name="kimi-3", tools=tools)
    assert out is message
    line = _line(caplog)
    assert "where=agentic_tools" in line and "reasoning_tokens=0" in line
    assert " duration_s=" in line
