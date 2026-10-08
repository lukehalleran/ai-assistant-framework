"""Streamed text reaches the user byte-for-byte (2026-10-08).

The 10-08 04:52 turn was streamed by the model as
``' points to "' + 'less worse,"' + ' not "would' ...`` yet stored and shown as
``points to"less worse," not"wouldn't``. ``generate_streaming_response`` split
the stream on " " and discarded the spaces; ``smart_join`` re-inserted them by
guessing and guessed "no space" before any word starting with a quote. The
same split dropped runs of spaces (code indentation). ~25% of stored turns
carried the glued opening quote.

These tests drive THE deployed generator and THE deployed joiners (enhanced
route ``gui.handlers.smart_join``, CLI route in the orchestrator, best-of
fallback) and require the joined output to equal the model's raw deltas.
"""
import pytest

from core.response_generator import (
    ResponseGenerator,
    smart_join,
    split_at_last_whitespace,
)
from models.model_manager import ModelManager
from utils.time_manager import TimeManager


class _Delta:
    def __init__(self, content=""):
        self.content = content
        self.reasoning_content = ""


class _Choice:
    def __init__(self, content="", finish_reason=None):
        self.delta = _Delta(content)
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, content="", finish_reason=None):
        self.choices = [_Choice(content, finish_reason)]


# Verbatim delta shapes from the 2026-10-08 04:51:56 debug log, plus code
# indentation, a double space, a paragraph break and a tab.
LIVE_DELTAS = [
    "Honest answer: nobody can run that counterfactual,",
    " but the evidence",
    " you gave",
    " me",
    ' points to "',
    'less worse,"',
    ' not "would',
    "n't have",
    ' happened."\n\nHere\'s',
    " the reasoning",
    ". She said 'fine'  twice.",
    "\n\n```python\ndef f(x):\n    if x:\n        return 1\n\treturn 0\n```\n",
    "Done.",
]
EXPECTED = "".join(LIVE_DELTAS)


async def _stream(gen, deltas, finish=True):
    async def _source():
        for i, d in enumerate(deltas):
            last = finish and i == len(deltas) - 1
            yield _Chunk(d, finish_reason="stop" if last else None)
    gen.model_manager.generate_async = lambda *a, **k: _async_value(_source())
    out = []
    async for chunk in gen.generate_streaming_response("q", None):
        out.append(chunk)
    return out


def _async_value(value):
    async def _fn():
        return value
    return _fn()


@pytest.fixture
def gen():
    return ResponseGenerator(model_manager=ModelManager(), time_manager=TimeManager())


def _join(chunks, joiner):
    acc = ""
    for c in chunks:
        acc = joiner(acc, c)
    return acc


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", [True, False], ids=["finish_reason", "stream_end"])
async def test_generator_chunks_concatenate_to_model_text(gen, finish):
    chunks = await _stream(gen, LIVE_DELTAS, finish=finish)
    assert "".join(chunks) == EXPECTED


@pytest.mark.asyncio
async def test_enhanced_route_joiner_preserves_quotes_and_indentation(gen):
    import gui.handlers as handlers

    chunks = await _stream(gen, LIVE_DELTAS)
    joined = _join(chunks, handlers.smart_join)
    assert joined == EXPECTED
    assert 'points to "less worse," not "wouldn\'t' in joined
    assert "\n    if x:\n        return 1\n\treturn 0" in joined


@pytest.mark.asyncio
async def test_cli_and_best_of_routes_use_the_shared_joiner(gen):
    import inspect
    import core.best_of_handler as best_of
    import core.orchestrator as orchestrator

    # Both former `chunk + " "` accumulators now join through smart_join.
    for mod in (orchestrator, best_of):
        src = inspect.getsource(mod)
        assert 'chunk + " "' not in src
        assert mod.smart_join is smart_join

    chunks = await _stream(gen, LIVE_DELTAS)
    assert _join(chunks, best_of.smart_join).strip() == EXPECTED.strip()


def test_bare_word_producers_still_get_spaces():
    # Stubs and synthetic markers yield separator-less words.
    assert _join(["Hello", "world"], smart_join) == "Hello world"
    assert _join(["Hello", ",", "world"], smart_join) == "Hello, world"


@pytest.mark.parametrize("buf,emit,keep", [
    ("abc", "", "abc"),
    ("a b", "a ", "b"),
    ("a\n\n    b", "a\n\n    ", "b"),
    ('to "', 'to ', '"'),
    ("x  ", "x  ", ""),
])
def test_split_at_last_whitespace(buf, emit, keep):
    assert split_at_last_whitespace(buf) == (emit, keep)
    assert emit + keep == buf


@pytest.mark.asyncio
async def test_non_streaming_fallback_keeps_newlines(gen):
    class _Msg:
        content = "Line one.\n\n    indented  two"

    class _Resp:
        choices = [type("C", (), {"message": _Msg()})()]

    gen.model_manager.generate_async = lambda *a, **k: _async_value(_Resp())
    out = [c async for c in gen.generate_streaming_response("q", None)]
    assert "".join(out) == _Msg.content
