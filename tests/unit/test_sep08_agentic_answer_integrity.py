"""Regression tests for the 2026-09-08 answer-corruption fix (B1, F1).

`gui/handlers._run_agentic_search` used to strip any line under 4 words
(unless it ended in punctuation or started with a whitelisted char) whenever
the agentic loop ran with zero tool rounds. In a live session this deleted
47 lines across 13 replies — including the ```r/``` fences themselves (32 of
the 47), `print(y_hat - resid)`, `coef(model)`, `coef_table <- summary(model)
$coefficients`, and a lone `)`. Storage used the raw stream, so the user/
debug record saw broken code while stored history stayed intact.

The fix replaces the word-count heuristic with `_strip_bare_tool_name_lines`
— a closed-set check (built from the real agentic tool schemas + dispatch
table) that removes a line ONLY when it IS a bare tool name / call, never a
short line of real content — plus a log-only `_answer_bodies_agree` parity
check right before storage.

Uses synthetic variable names throughout (no real people/documents).
"""

from unittest.mock import AsyncMock, patch

import pytest

from tests.unit.test_handle_submit import (
    _debug_record,
    _final_content,
    _gate_decision,
    _make_orchestrator,
    _run_submit,
)


def _force_agentic_tools_patch():
    """Deterministically routes handle_submit into _run_agentic_search
    regardless of the message's own keyword shape."""
    return patch(
        "core.agentic.gate.evaluate_agentic_gate",
        new_callable=AsyncMock,
        return_value=_gate_decision(modes=["tools"]),
    )


# ---------------------------------------------------------------------------
# (a) A zero-round answer's real content must survive unmangled — display,
#     debug record, and the text dispatched to storage.
# ---------------------------------------------------------------------------

_CODE_HEAVY_ANSWER = (
    "Here's the residual check you asked for:\n\n"
    "```r\n"
    "print(y_hat - resid)\n"
    "```\n\n"
    "Pulling the coefficients and closing out the call:\n\n"
    "coef(model)\n"
    ")\n\n"
    "The overall significance result is:\n\n"
    "TRUE\n\n"
    "| Model | R2 |\n"
    "|-------|-----|\n"
    "| A     | 0.8 |\n\n"
    "To list the working directory contents:\n\n"
    "ls -la\n"
)

_SURVIVING_FRAGMENTS = (
    "```r",
    "print(y_hat - resid)",
    "```",
    "coef(model)",
    ")",
    "TRUE",
    "| A     | 0.8 |",
    "ls -la",
)


class TestZeroRoundAnswerSurvives:
    @pytest.mark.asyncio
    async def test_code_and_short_lines_survive_display_debug_and_storage(self):
        orch = _make_orchestrator(
            agentic_enabled=True, agentic_items=[_CODE_HEAVY_ANSWER],
        )
        # Query text kept identical to an existing proven-safe agentic test
        # (test_handle_submit.py::TestAgenticComputation) so it can never be
        # accidentally reclassified as an insight/pattern request by the real
        # (unmocked) detect_insight_request() the dispatcher also runs — the
        # gate itself is force-routed via the patch below regardless.
        with patch("gui.handlers._dispatch_storage") as mock_dispatch, \
                _force_agentic_tools_patch():
            results = await _run_submit("calculate fibonacci 10", orch)

        content = _final_content(results)
        debug = _debug_record(results)
        assert debug is not None
        assert debug["mode"] == "agentic-search"

        # Storage dispatch: the 3rd and 5th positional args are both the
        # sanitized text passed to persistence (see _dispatch_storage call
        # site in _run_agentic_search).
        assert mock_dispatch.call_count == 1
        stored_args = mock_dispatch.call_args[0]
        stored_text = stored_args[2]

        for fragment in _SURVIVING_FRAGMENTS:
            assert fragment in content, f"{fragment!r} missing from display"
            assert fragment in debug["response"], f"{fragment!r} missing from debug response"
            assert fragment in stored_text, f"{fragment!r} missing from stored text"


# ---------------------------------------------------------------------------
# (b) A genuinely leaked bare tool-call line IS removed from display.
# ---------------------------------------------------------------------------

_NARRATION_WITH_LEAKED_TOOL_LINES = (
    "Let me pull that information together for you.\n\n"
    "github\n\n"
    'web_search("x")\n\n'
    "Here is what I found based on the above."
)


class TestLeakedBareToolLinesRemoved:
    @pytest.mark.asyncio
    async def test_bare_tool_name_and_call_lines_stripped(self):
        orch = _make_orchestrator(
            agentic_enabled=True,
            agentic_items=[_NARRATION_WITH_LEAKED_TOOL_LINES],
        )
        # Same proven-safe query as TestAgenticMemory — content is irrelevant
        # since the gate is force-routed, but this avoids the real (unmocked)
        # insight detector reclassifying an unfamiliar phrasing.
        with _force_agentic_tools_patch():
            results = await _run_submit("do you remember my brother's name?", orch)

        content = _final_content(results)
        lines = [line.strip() for line in content.split("\n")]
        assert "github" not in lines
        assert 'web_search("x")' not in content
        # Surrounding narration sentences must survive.
        assert "Let me pull that information together for you." in content
        assert "Here is what I found based on the above." in content


# ---------------------------------------------------------------------------
# (c) Direct unit tests of the two pure helpers.
# ---------------------------------------------------------------------------

class TestStripBareToolNameLines:
    def test_leaves_real_code_and_short_lines_untouched(self):
        from gui.handlers import _strip_bare_tool_name_lines
        text = "coef(model)\n)\nTRUE\nls -la"
        assert _strip_bare_tool_name_lines(text) == text

    def test_removes_bare_tool_name_line(self):
        from gui.handlers import _strip_bare_tool_name_lines
        text = "Some intro.\n\ngithub\n\nMore text."
        assert _strip_bare_tool_name_lines(text) == "Some intro.\n\nMore text."

    def test_removes_tool_call_with_args(self):
        from gui.handlers import _strip_bare_tool_name_lines
        text = 'Explanation.\n\nweb_search("x")\n\nDone.'
        assert _strip_bare_tool_name_lines(text) == "Explanation.\n\nDone."

    def test_removes_tool_call_with_colon_form(self):
        from gui.handlers import _strip_bare_tool_name_lines
        text = "Before.\n\ngit_stats: recent commits\n\nAfter."
        assert _strip_bare_tool_name_lines(text) == "Before.\n\nAfter."

    def test_tool_name_inside_fence_is_preserved(self):
        from gui.handlers import _strip_bare_tool_name_lines
        text = "```\ngithub\n```"
        assert _strip_bare_tool_name_lines(text) == text

    def test_fence_markers_never_stripped(self):
        from gui.handlers import _strip_bare_tool_name_lines
        text = "```r\nprint(y_hat - resid)\n```"
        assert _strip_bare_tool_name_lines(text) == text

    def test_non_tool_short_lines_untouched(self):
        from gui.handlers import _strip_bare_tool_name_lines
        text = "Yes.\nOK\n42\nDone!"
        assert _strip_bare_tool_name_lines(text) == text

    def test_empty_text_is_noop(self):
        from gui.handlers import _strip_bare_tool_name_lines
        assert _strip_bare_tool_name_lines("") == ""


class TestAnswerBodiesAgree:
    def test_agree_when_stored_contains_every_line(self):
        from gui.handlers import _answer_bodies_agree
        display = "Intro line.\n\n```python\nprint(1)\n```\n\nOutro line."
        stored = (
            "Prefix stuff.\nIntro line.\n```python\nprint(1)\n```\n"
            "Outro line.\nSuffix."
        )
        assert _answer_bodies_agree(display, stored) is True

    def test_disagree_when_a_code_line_is_missing(self):
        from gui.handlers import _answer_bodies_agree
        display = "Intro line.\n\n```python\nprint(1)\nprint(2)\n```\n"
        stored = "Intro line.\n```python\nprint(1)\n```\n"
        assert _answer_bodies_agree(display, stored) is False

    def test_web_n_link_markup_ignored(self):
        from gui.handlers import _answer_bodies_agree
        display = (
            "As reported [[WEB_1](https://example.com)] earlier.\n\n"
            "---\n**Sources:**\n[WEB_1] [Example](https://example.com)"
        )
        stored = "As reported [WEB_1] earlier."
        assert _answer_bodies_agree(display, stored) is True

    def test_empty_display_trivially_agrees(self):
        from gui.handlers import _answer_bodies_agree
        assert _answer_bodies_agree("", "anything") is True
        assert _answer_bodies_agree("   ", "anything") is True
