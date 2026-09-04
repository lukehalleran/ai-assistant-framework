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
