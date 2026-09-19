"""Regression: three more sites routed through utils.async_results
(2026-09-19, class BCP-0916-T2, executor L1a — the "remainder" batch after
PR #14's first wave).

Same failure mode as tests/unit/test_gather_cancelled_sites.py: a cancelled
child of an ``asyncio.gather(..., return_exceptions=True)`` is a
``CancelledError`` INSTANCE sitting in the results list (never raised at the
gather site), which is a ``BaseException`` — not an ``Exception`` — so the
old ``isinstance(x, Exception)`` filters at each of these three sites let it
through untouched. Each test reproduces the exact pre-fix failure and asserts
the surviving sibling's result is used instead.
"""
import asyncio
from unittest.mock import MagicMock

import pytest

from core.agentic.controller import AgenticSearchController
from core.response_generator import ResponseGenerator


# ---------------------------------------------------------------------------
# T-A1 — core/response_generator.py:942-948 (judge fan-out in
# generate_best_of_ensemble)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensemble_judge_loop_skips_cancelled_judge():
    """Before the fix: ``for m, res in zip(judge_meta, judge_results): ... sc
    = float(res)`` ran ``float(CancelledError())`` for the cancelled judge
    slot (``isinstance(res, Exception)`` is False for a BaseException) ->
    ``TypeError: float() argument must be a string or a real number, not
    'CancelledError'`` — which killed ensemble selection for BOTH candidates,
    not just the one judge call that was cancelled.

    Two candidates (from two generator models), two judges. Judge "j1" is
    cancelled for every candidate; judge "j2" scores candidate B higher than
    candidate A. Heuristic weight is zeroed so the *returned* winner can only
    be explained by the surviving judge's score actually being used — not
    merely by "no exception was raised".
    """
    mm = MagicMock()

    async def fake_generate_once(*, prompt, model_name, system_prompt, max_tokens, temperature):
        if system_prompt.startswith("You are a strict evaluator"):
            # Judge call.
            if model_name == "j1":
                raise asyncio.CancelledError()
            if "Answer B text." in prompt:
                return '{"score": 9}'
            return '{"score": 1}'
        # Candidate-generation call.
        if model_name == "genA":
            return "Answer A text."
        return "Answer B text."

    mm.generate_once = fake_generate_once
    rg = ResponseGenerator(model_manager=mm)

    result = await rg.generate_best_of_ensemble(
        prompt="p",
        generator_models=["genA", "genB"],
        system_prompt="s",
        question_text="q",
        n_total=2,
        temps=(0.5,),
        selector_models=["j1", "j2"],
        weight_heuristic=0.0,
        weight_llm=1.0,
    )

    assert result == "Answer B text."


# ---------------------------------------------------------------------------
# T-A2 — core/prompt/builder.py:911-914 (LLM-compress fan-out apply loop)
# ---------------------------------------------------------------------------

def _make_item(content: str, **extra) -> dict:
    d = {"content": content}
    d.update(extra)
    return d


@pytest.fixture(autouse=True)
def _enable_optional_compression(monkeypatch):
    monkeypatch.setattr("core.prompt.builder.LLM_COMPRESSION_ENABLED", True)


@pytest.fixture
def _builder():
    from unittest.mock import Mock
    from core.prompt import UnifiedPromptBuilder

    manager = Mock()
    manager.get_active_model_name = Mock(return_value="gpt-4o")
    manager.generate_once = None  # set per-test

    tokenizer = Mock()
    tokenizer.count_tokens = Mock(side_effect=lambda text, model_name: max(1, len(text) // 4))

    coordinator = Mock()
    coordinator.corpus_manager = Mock()
    coordinator.corpus_manager.get_recent_memories = Mock(return_value=[])
    coordinator.get_summaries = Mock(return_value=[])

    return UnifiedPromptBuilder(
        memory_coordinator=coordinator,
        model_manager=manager,
        tokenizer_manager=tokenizer,
        token_budget=40000,
    )


@pytest.mark.asyncio
async def test_llm_compress_apply_loop_skips_cancelled_candidate(_builder):
    """Before the fix: ``for result in results: if result is None or
    isinstance(result, Exception): continue; section, idx, compressed_text =
    result`` — a cancelled ``_compress_one`` task leaves a CancelledError
    INSTANCE in ``results`` (isinstance(..., Exception) is False), so the
    tuple-unpack ``section, idx, compressed_text = result`` raised
    ``TypeError: cannot unpack non-iterable CancelledError object`` and
    propagated out of ``_llm_compress_oversized``, discarding EVERY
    compression for the turn (not just the cancelled one).
    """
    item_a = _make_item("a" * 8000)
    item_b = _make_item("b" * 8000)
    ctx = {"memories": [item_a, item_b]}

    call_num = 0

    async def _partial_cancel(*args, **kwargs):
        nonlocal call_num
        call_num += 1
        if call_num == 1:
            raise asyncio.CancelledError()
        return "second item compressed successfully."

    _builder.model_manager.generate_once = _partial_cancel

    result = await _builder._llm_compress_oversized(ctx)

    contents = [item["content"] for item in result["memories"]]
    assert "second item compressed successfully." in contents
    originals = [c for c in contents if c != "second item compressed successfully."]
    assert len(originals) == 1
    assert len(originals[0]) == 8000


# ---------------------------------------------------------------------------
# T-A3 — core/agentic/controller.py:739-746 (round-1 direct URL fetch)
# ---------------------------------------------------------------------------

@pytest.fixture
def _controller():
    manager = MagicMock()
    manager.api_models = {}
    return AgenticSearchController(model_manager=manager, web_search_manager=MagicMock())


@pytest.mark.asyncio
async def test_round1_url_fetch_skips_cancelled_url(_controller, monkeypatch):
    """Before the fix: ``for url, result in zip(initial_urls[:3],
    fetch_results): if isinstance(result, Exception): ... else: content =
    result`` treated a cancelled fetch's CancelledError instance as a
    SUCCESSFUL result (isinstance(..., Exception) is False for a
    BaseException) — ``content = result`` then set the page content to the
    CancelledError object itself, which downstream string formatting/
    concatenation would choke on, instead of recording it as a fetch error.
    """
    u1 = "https://example.com/cancelled"
    u2 = "https://example.com/ok"
    # >= AGENTIC_FETCH_FASTPATH_MIN_CHARS (400) so the fetch fastpath fires
    # and the turn skips straight to final synthesis (no decision-round
    # mocking needed).
    page_text = "PAGE TEXT " + ("x" * 490)

    async def fake_fetch(url):
        if url == u1:
            raise asyncio.CancelledError()
        return page_text

    monkeypatch.setattr(_controller._tool_executor, "_execute_fetch_url", fake_fetch)

    captured = {}

    async def fake_final(query, system_prompt, model_name, session, initial_context=None):
        captured["session"] = session
        yield "Answer."

    monkeypatch.setattr(_controller, "_generate_final_response", fake_final)

    out = []
    async for ev in _controller.run_agentic_search(
        query="q", system_prompt="sys", model_name="glm-5.2",
        initial_search_terms=[], initial_urls=[u1, u2],
        fetch_fastpath=True,
    ):
        out.append(ev)

    text = "".join(c for c in out if isinstance(c, str))
    assert "Answer." in text

    session = captured["session"]
    assert "PAGE TEXT" in session.accumulated_context
    assert f"[Error fetching {u1}: CancelledError]" in session.accumulated_context
