"""[PromptCache] lines record the upstream provider that served the call.

Incident date: 2026-10-25 (lane PROV). Bug class: BC-72 (unobservable decision:
kimi-3 showed 0 cache reads in 20 calls after 2026-10-24 19:29 against ~5,120
cached tokens per call before, and the serving provider was never recorded, so
a route change could be neither confirmed nor ruled out).
Drives the DEPLOYED ``ModelManager._log_cache_usage`` and every call path that
emits the line (non-streaming, tool round, ResponseGenerator stream, agentic
synthesis stream) with fake transports whose responses/chunks carry the
OpenRouter top-level ``provider`` field.
"""
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.agentic.controller import AgenticSearchController
from core.agentic.types import AgenticSearchSession
from core.response_generator import ResponseGenerator
from models.model_manager import ModelManager
from utils.time_manager import TimeManager

_TOOLS = [{"type": "function", "function": {"name": "noop", "parameters": {}}}]


def _usage(prompt=1200, cached=0):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=50,
        prompt_tokens_details={"cached_tokens": cached},
    )


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(ModelManager, "_get_cached_embedder", staticmethod(lambda: None))
    mm = ModelManager(api_key="test-key")
    mm.switch_model("kimi-3")
    return mm


def _lines(caplog, where=None):
    return [
        r.getMessage() for r in caplog.records
        if "[PromptCache]" in r.getMessage() and (where is None or f"where={where}" in r.getMessage())
    ]


def _client_returning(response):
    async def _create(**kwargs):
        return response
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))


# ---- unit -----------------------------------------------------------------
def test_provider_is_appended_after_existing_fields(manager, caplog):
    with caplog.at_level(logging.INFO):
        manager._log_cache_usage(_usage(), model_name="m", where="w", duration_s=1.5,
                                 provider="Moonshot AI")
    line = _lines(caplog)[0]
    assert line.startswith(
        "[PromptCache] MISS model=m where=w prompt=1200 completion=50 "
        "cache_read=0 cache_write=0 cache_discount=None reasoning_tokens=None"
    )
    assert line.endswith(" duration_s=1.50 provider=Moonshot AI")


def test_no_provider_emits_no_field(manager, caplog):
    with caplog.at_level(logging.INFO):
        manager._log_cache_usage(_usage(), model_name="m", where="w")
    assert "provider" not in _lines(caplog)[0]


@pytest.mark.parametrize("bad", [None, "", "   ", 7, ["x"], MagicMock()])
def test_odd_provider_never_raises_or_logs_a_field(manager, caplog, bad):
    with caplog.at_level(logging.INFO):
        manager._log_cache_usage(_usage(), model_name="m", where="w", provider=bad)
    assert len(_lines(caplog)) == 1
    assert "provider=" not in _lines(caplog)[0]


def test_provider_is_flattened_and_trimmed(manager, caplog):
    with caplog.at_level(logging.INFO):
        manager._log_cache_usage(_usage(), model_name="m", where="w",
                                 provider="Moon\nshot\t AI " + "x" * 80)
    tail = _lines(caplog)[0].split(" provider=", 1)[1]
    assert "\n" not in tail and "\t" not in tail
    assert len(tail) == 40 and tail.startswith("Moon shot AI x")


# ---- non-streaming / tool round -------------------------------------------
@pytest.mark.asyncio
async def test_generate_once_logs_response_provider(manager, caplog):
    msg = SimpleNamespace(content="ok", tool_calls=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=_usage(),
                               provider="Moonshot AI")
    manager.async_client = _client_returning(response)
    with caplog.at_level(logging.INFO):
        await manager.generate_once(prompt="q", model_name="kimi-3", system_prompt="s")
    assert _lines(caplog, "generate_once")[0].endswith("provider=Moonshot AI")


@pytest.mark.asyncio
async def test_tool_round_logs_provider_from_model_extra(manager, caplog):
    msg = SimpleNamespace(content="ok", tool_calls=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=_usage(),
                               model_extra={"provider": "Fireworks"})
    manager.async_client = _client_returning(response)
    with caplog.at_level(logging.INFO):
        out = await manager.generate_once_with_tools(prompt="q", model_name="kimi-3", tools=_TOOLS)
    assert out is msg
    assert "provider=Fireworks" in _lines(caplog, "agentic_tools")[0]


@pytest.mark.asyncio
async def test_tool_round_without_provider_is_unchanged(manager, caplog):
    msg = SimpleNamespace(content="ok", tool_calls=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=_usage())
    manager.async_client = _client_returning(response)
    with caplog.at_level(logging.INFO):
        await manager.generate_once_with_tools(prompt="q", model_name="kimi-3", tools=_TOOLS)
    assert "provider=" not in _lines(caplog, "agentic_tools")[0]


# ---- streams ---------------------------------------------------------------
class _Delta:
    def __init__(self, content=""):
        self.content = content
        self.reasoning_content = ""


class _Chunk:
    def __init__(self, content=None, usage=None, provider=None, finish=None):
        self.choices = [SimpleNamespace(delta=_Delta(content), finish_reason=finish)] if content is not None else []
        self.usage = usage
        if provider is not None:
            self.provider = provider


def _async_value(value):
    async def _fn():
        return value
    return _fn()


@pytest.mark.asyncio
async def test_response_generator_stream_logs_first_chunk_provider(manager, caplog):
    async def _source():
        yield _Chunk("Hello", provider="Moonshot AI")
        yield _Chunk(" there", provider="Other", finish="stop")
        yield _Chunk(usage=_usage(prompt=9000, cached=8000))  # trailing usage chunk, no provider

    gen = ResponseGenerator(model_manager=manager, time_manager=TimeManager())
    manager.generate_async = lambda *a, **k: _async_value(_source())
    with caplog.at_level(logging.INFO):
        out = [c async for c in gen.generate_streaming_response("q", None)]
    assert "Hello" in "".join(out)
    lines = _lines(caplog, "streaming")
    assert len(lines) == 1, caplog.text
    assert "HIT" in lines[0] and lines[0].endswith("provider=Moonshot AI")


@pytest.mark.asyncio
async def test_agentic_synthesis_stream_logs_provider(manager, caplog):
    async def _stream():
        yield _Chunk("Here is the answer.", provider="Moonshot AI")
        yield _Chunk(usage=_usage(prompt=9000, cached=8000))

    async def _generate_async(**kwargs):
        return _stream()

    manager.generate_async = _generate_async
    ctl = AgenticSearchController(model_manager=manager, web_search_manager=MagicMock())
    ctl._build_final_prompt = MagicMock(return_value="FINAL PROMPT")
    with caplog.at_level(logging.INFO):
        async for _ in ctl._generate_final_response(
            query="q", system_prompt="sys", model_name="kimi-3",
            session=AgenticSearchSession(query="q"),
        ):
            pass
    lines = _lines(caplog, "agentic_synthesis")
    assert len(lines) == 1, caplog.text
    assert lines[0].endswith("provider=Moonshot AI")
