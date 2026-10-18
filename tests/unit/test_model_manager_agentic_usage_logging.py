"""Usage ([PromptCache]) lines for every agentic LLM call.

Incident date: 2026-10-17 (lane AGT). Bug class: BC-72 (unobservable decision - dead/missing
observability - a receipt that exists for some call sites but never gets
emitted for others).

Since 2026-09-30, 126 turns ran the agentic loop; 115 called no tool and the
cost of those turns could not be attributed because the native-tools decision
call (``generate_once_with_tools``) and the agentic synthesis stream emitted no
``[PromptCache]`` usage line (``generate_once`` and ResponseGenerator's
streaming path already did). These tests drive the DEPLOYED methods with fake
transports.
"""
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.agentic.controller import AgenticSearchController
from core.agentic.types import AgenticSearchSession
from models.model_manager import ModelManager

KIMI_SLUG = "moonshotai/kimi-k3"


def _usage(prompt=1200, cached=0, completion=50):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details={"cached_tokens": cached},
    )


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(ModelManager, "_get_cached_embedder", staticmethod(lambda: None))
    mm = ModelManager(api_key="test-key")
    mm.switch_model("kimi-3")
    return mm


def _client_returning(response):
    async def _create(**kwargs):
        return response

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))


def _cache_lines(caplog, where):
    return [
        r.getMessage() for r in caplog.records
        if "[PromptCache]" in r.getMessage() and f"where={where}" in r.getMessage()
    ]


_TOOLS = [{"type": "function", "function": {"name": "noop", "parameters": {}}}]


@pytest.mark.asyncio
async def test_tool_round_emits_one_agentic_tools_usage_line(manager, caplog):
    message = SimpleNamespace(content="ok", tool_calls=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=_usage())
    manager.async_client = _client_returning(response)

    with caplog.at_level(logging.INFO):
        out = await manager.generate_once_with_tools(
            prompt="which tool?", model_name="kimi-3", tools=_TOOLS,
        )

    assert out is message  # return value unchanged
    lines = _cache_lines(caplog, "agentic_tools")
    assert len(lines) == 1, caplog.text
    assert "MISS" in lines[0] or "HIT" in lines[0] or "WRITE" in lines[0]


@pytest.mark.asyncio
async def test_tool_round_with_cache_hit_reports_hit(manager, caplog):
    message = SimpleNamespace(content="ok", tool_calls=None)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)], usage=_usage(prompt=2000, cached=1500),
    )
    manager.async_client = _client_returning(response)

    with caplog.at_level(logging.INFO):
        await manager.generate_once_with_tools(prompt="q", model_name="kimi-3", tools=_TOOLS)

    lines = _cache_lines(caplog, "agentic_tools")
    assert len(lines) == 1 and "HIT" in lines[0]


@pytest.mark.asyncio
async def test_usageless_tool_round_degrades_silently(manager, caplog):
    message = SimpleNamespace(content="ok", tool_calls=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])  # no .usage at all
    manager.async_client = _client_returning(response)

    with caplog.at_level(logging.INFO):
        out = await manager.generate_once_with_tools(
            prompt="q", model_name="kimi-3", tools=_TOOLS,
        )

    assert out is message
    assert _cache_lines(caplog, "agentic_tools") == []


@pytest.mark.asyncio
async def test_exploding_usage_logger_never_breaks_the_tool_round(manager, monkeypatch):
    message = SimpleNamespace(content="ok", tool_calls=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=_usage())
    manager.async_client = _client_returning(response)

    def _boom(*a, **k):
        raise RuntimeError("log sink down")

    monkeypatch.setattr(manager, "_log_cache_usage", _boom)
    out = await manager.generate_once_with_tools(prompt="q", model_name="kimi-3", tools=_TOOLS)
    assert out is message


# --------------------------------------------------------------------------
# Agentic synthesis stream (controller._generate_final_response)
# --------------------------------------------------------------------------
class _Delta:
    def __init__(self, content=""):
        self.content = content
        self.reasoning_content = ""


class _Chunk:
    def __init__(self, content=None, usage=None):
        self.choices = [SimpleNamespace(delta=_Delta(content), finish_reason=None)] if content is not None else []
        self.usage = usage


def _controller(manager, stream):
    async def _generate_async(**kwargs):
        return stream

    manager.generate_async = _generate_async
    ctl = AgenticSearchController(model_manager=manager, web_search_manager=MagicMock())
    ctl._build_final_prompt = MagicMock(return_value="FINAL PROMPT")
    return ctl


async def _drain(ctl):
    out = []
    async for c in ctl._generate_final_response(
        query="q", system_prompt="sys", model_name="kimi-3",
        session=AgenticSearchSession(query="q"),
    ):
        out.append(c)
    return "".join(out)


@pytest.mark.asyncio
async def test_synthesis_stream_logs_trailing_usage_chunk(manager, caplog):
    async def _stream():
        yield _Chunk("Here is the answer.")
        yield _Chunk(usage=_usage(prompt=9000, cached=8000))  # empty-choices usage chunk

    ctl = _controller(manager, _stream())
    with caplog.at_level(logging.INFO):
        text = await _drain(ctl)

    assert "Here is the answer." in text
    lines = _cache_lines(caplog, "agentic_synthesis")
    assert len(lines) == 1, caplog.text
    assert "HIT" in lines[0]


@pytest.mark.asyncio
async def test_synthesis_stream_without_usage_logs_nothing(manager, caplog):
    async def _stream():
        yield _Chunk("Here is the answer.")

    ctl = _controller(manager, _stream())
    with caplog.at_level(logging.INFO):
        text = await _drain(ctl)

    assert "Here is the answer." in text
    assert _cache_lines(caplog, "agentic_synthesis") == []
