"""Regression coverage for internal prompt summarizer generation contracts."""

from types import SimpleNamespace
import asyncio

import pytest

from core.prompt.summarizer import LLMSummarizer


class _OnceManager:
    def __init__(self, text: str):
        self.text = text
        self.calls = []

    def get_active_model_name(self):
        return "test-model"

    async def generate_once(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.text


class _StreamingManager:
    def get_active_model_name(self):
        return "test-model"

    async def generate_async(self, prompt, **kwargs):
        async def _stream():
            for text in ("streamed ", "summary"):
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
                )

        return _stream()


class _StreamingManagerNoKwargsSink:
    """No generate_once; generate_async has the model layer's REAL narrow
    signature (raw/images/max_tokens/temperature — no **kwargs sink). A
    stray unexpected kwarg (like the removed `model=model_name`) raises
    TypeError here instead of being silently absorbed, so this reproduces
    the 2026-09-04 latent-bad-kwarg bug in core.prompt.summarizer's
    generate_async fallback."""

    def __init__(self):
        self.calls = []

    def get_active_model_name(self):
        return "test-model"

    async def generate_async(self, prompt, raw=False, images=None,
                              max_tokens=None, temperature=None):
        self.calls.append({
            "prompt": prompt, "raw": raw, "images": images,
            "max_tokens": max_tokens, "temperature": temperature,
        })

        async def _stream():
            for text in ("fallback ", "summary"):
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
                )

        return _stream()


@pytest.mark.asyncio
async def test_summary_uses_complete_non_streaming_text():
    manager = _OnceManager("A real summary")
    summarizer = LLMSummarizer(manager, memory_coordinator=None)

    result = await summarizer._llm_summarize_recent([
        {"query": "What happened?", "response": "Something useful."},
    ])

    assert result == "A real summary"
    assert manager.calls[0][1]["model_name"] == "test-model"


@pytest.mark.asyncio
async def test_summary_compatibility_path_consumes_async_stream():
    summarizer = LLMSummarizer(_StreamingManager(), memory_coordinator=None)

    result = await summarizer._llm_summarize_recent([
        {"query": "What happened?", "response": "Something useful."},
    ])

    assert result == "streamed summary"
    assert "async_generator" not in result


@pytest.mark.asyncio
async def test_generate_async_fallback_has_no_stray_model_kwarg():
    """Regression (2026-09-04): the generate_async fallback used to pass
    `model=model_name` — ModelManager.generate_async(self, prompt, raw=False,
    images=None, **kwargs) has no `model` param, so it silently forwarded
    into local generate()'s kwargs where it could raise. Driven against a
    fake carrying the model layer's real narrow generate_async signature
    (no **kwargs sink) and NO generate_once, so this exercises exactly the
    fallback branch and any stray kwarg raises TypeError."""
    manager = _StreamingManagerNoKwargsSink()
    assert not hasattr(manager, "generate_once")
    summarizer = LLMSummarizer(manager, memory_coordinator=None)

    result = await summarizer._llm_summarize_recent([
        {"query": "What happened?", "response": "Something useful."},
    ])

    assert result == "fallback summary"
    assert len(manager.calls) == 1


@pytest.mark.asyncio
async def test_reflections_accept_complete_string_generation():
    manager = _OnceManager("- First useful reflection.\n- Second useful reflection.")
    summarizer = LLMSummarizer(manager, memory_coordinator=None)

    result = await summarizer._reflect_on_demand(
        context={
            "recent_conversations": [
                {"query": "How is it going?", "response": "Better today."},
            ]
        },
        user_input="What pattern do you see?",
        session_reflections=[],
    )

    assert [item["content"] for item in result[:2]] == [
        "First useful reflection.",
        "Second useful reflection.",
    ]


@pytest.mark.asyncio
async def test_generation_timeout_covers_stream_consumption():
    class _SlowStreamingManager(_StreamingManager):
        async def generate_async(self, prompt, **kwargs):
            async def _stream():
                await asyncio.sleep(1)
                yield "too late"

            return _stream()

    summarizer = LLMSummarizer(_SlowStreamingManager(), memory_coordinator=None)

    with pytest.raises(asyncio.TimeoutError):
        await summarizer._generate_text("prompt", timeout=0.01, max_tokens=10)
