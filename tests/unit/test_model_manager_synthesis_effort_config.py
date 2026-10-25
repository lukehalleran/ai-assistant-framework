"""The agentic synthesis call's reasoning effort is configurable (oct25 lane SPEED).

The effort was a hardcoded "medium" in ``reasoning_request_config``. The knob
``agentic_search.synthesis_reasoning_effort`` -> ``app_config.AGENTIC_SYNTHESIS_REASONING_EFFORT``
is read LIVE at call time (BC-82) and its default keeps the effective request
identical to before ({"effort": "medium"}). Class: BC-82 (live config),
BC-72 (latency lever). No incident date: knob added 2026-10-25, no behaviour change.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from models.model_manager import ModelManager, reasoning_request_config


class _FakeAsyncClient:
    def __init__(self, captured):
        async def _create(**kwargs):
            captured.append(kwargs)

            async def _stream():
                return
                yield  # pragma: no cover
            return _stream()

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(ModelManager, "_get_cached_embedder", staticmethod(lambda: None))
    mm = ModelManager(api_key="test-key")
    mm.captured = []
    mm.async_client = _FakeAsyncClient(mm.captured)
    mm.active_model_name = "kimi-3"
    return mm


def _controller(manager):
    from core.agentic.controller import AgenticSearchController

    ctrl = AgenticSearchController(model_manager=manager, web_search_manager=MagicMock())
    ctrl._build_final_prompt = MagicMock(return_value="FINAL PROMPT")
    return ctrl


async def _drive_final(manager):
    from core.agentic.types import AgenticSearchSession, SearchProtocol

    session = AgenticSearchSession(query="q", max_rounds=3, protocol=SearchProtocol.XML_MARKERS)
    ctrl = _controller(manager)
    async for _ in ctrl._generate_final_response(
        query="q", system_prompt="sys", model_name="kimi-3", session=session
    ):
        pass
    return manager.captured[-1]["extra_body"]["reasoning"]


def test_default_effective_value_is_medium():
    import config.app_config as app_config

    assert app_config.AGENTIC_SYNTHESIS_REASONING_EFFORT == "medium"
    assert reasoning_request_config("moonshotai/kimi-k3") == {"effort": "medium"}


def test_schema_rejects_unknown_effort():
    from pydantic import ValidationError

    from config.schema import AgenticSearchSection

    assert AgenticSearchSection().synthesis_reasoning_effort == "medium"
    with pytest.raises(ValidationError):
        AgenticSearchSection(synthesis_reasoning_effort="extreme")


@pytest.mark.asyncio
async def test_synthesis_request_carries_default_medium(manager):
    assert await _drive_final(manager) == {"effort": "medium"}


@pytest.mark.asyncio
async def test_synthesis_request_follows_live_constant(manager, monkeypatch):
    import config.app_config as app_config

    # Same manager + controller wiring, constant flipped between calls (BC-82).
    assert await _drive_final(manager) == {"effort": "medium"}
    monkeypatch.setattr(app_config, "AGENTIC_SYNTHESIS_REASONING_EFFORT", "low")
    assert await _drive_final(manager) == {"effort": "low"}


@pytest.mark.asyncio
async def test_effort_kwarg_never_reaches_create_and_disable_wins(manager):
    await manager.generate_async("p", max_tokens=8, reasoning_effort="high")
    sent = manager.captured[-1]
    assert sent["extra_body"]["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in sent
    await manager.generate_async(
        "p", max_tokens=8, reasoning_effort="high", disable_reasoning=True)
    assert manager.captured[-1]["extra_body"]["reasoning"] == {"enabled": False}
