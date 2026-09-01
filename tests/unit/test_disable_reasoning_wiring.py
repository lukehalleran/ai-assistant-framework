"""
disable_reasoning must send an EXPLICIT off-switch, not merely omit the param.

2026-08-31 live incident: the insight claim assessor called
generate_once(disable_reasoning=True) against kimi-k3, but the code only
omitted the reasoning request param. kimi-k3 reasons BY DEFAULT when no
reasoning key is sent, so the model reasoned through a ~30K-char evidence
prompt and exceeded even the 75s assessor timeout (HTTP 200 at +1s, body still
pending at +75s — generating, not hung). OpenRouter's documented off-switch is
reasoning={"enabled": false}. Per the validation rule these tests drive THE
deployed generate paths with a kwargs-capturing fake client.
"""
from types import SimpleNamespace

import pytest

from models.model_manager import ModelManager


def _fake_response():
    msg = SimpleNamespace(content="ok", tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


class _FakeAsyncClient:
    def __init__(self, captured):
        async def _create(**kwargs):
            captured.append(kwargs)
            if kwargs.get("stream"):
                async def _stream():
                    return
                    yield  # pragma: no cover
                return _stream()
            return _fake_response()

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(
        ModelManager, "_get_cached_embedder", staticmethod(lambda: None)
    )
    mm = ModelManager(api_key="test-key")
    mm.captured = []
    mm.async_client = _FakeAsyncClient(mm.captured)
    return mm


@pytest.mark.asyncio
async def test_generate_once_disable_sends_explicit_enabled_false(manager):
    await manager.generate_once(
        "judge this", model_name="kimi-3", max_tokens=64, disable_reasoning=True,
    )
    reasoning = manager.captured[-1]["extra_body"].get("reasoning")
    assert reasoning == {"enabled": False}


@pytest.mark.asyncio
async def test_generate_once_default_still_requests_reasoning(manager):
    await manager.generate_once("hello", model_name="kimi-3", max_tokens=64)
    reasoning = manager.captured[-1]["extra_body"].get("reasoning")
    assert reasoning == {"effort": "medium"}


@pytest.mark.asyncio
async def test_generate_async_disable_sends_explicit_enabled_false(manager):
    manager.active_model_name = "kimi-3"  # generate_async uses the active model
    await manager.generate_async(
        "recover", max_tokens=64, disable_reasoning=True,
    )
    reasoning = manager.captured[-1]["extra_body"].get("reasoning")
    assert reasoning == {"enabled": False}


@pytest.mark.asyncio
async def test_non_reasoning_model_sends_no_reasoning_key(manager):
    # enabled=false must not be sprayed at models with no reasoning support.
    await manager.generate_once(
        "hello", model_name="gpt-4o", max_tokens=64, disable_reasoning=True,
    )
    assert "reasoning" not in manager.captured[-1]["extra_body"]
