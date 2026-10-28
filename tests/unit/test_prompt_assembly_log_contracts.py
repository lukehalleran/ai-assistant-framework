"""Behavioural siblings: prompt-assembly logging contracts (source-text rows).

Incident dates: 2026-08-21 (a stored conversation QUOTING the [RECENT CONVERSATION] header
mid-line tripped a false-positive duplicate-header ERROR every turn), 2026-09-03 (the STM
summary VALUE was logged — message content in the debug log). Class: BC-63 (a
grep of the module text stands in for what reaches the log), BC-72 (content in telemetry).

Each test drives THE deployed ``PromptFormatter._assemble_prompt`` and asserts on the
captured log records. (The over-budget WARNING row of test_codex_followups is a GUARD:
no input reaches that branch — see the lane handoff.)
"""
import logging

import pytest

from tests.unit.test_calendar_prompt import TestGoogleCalendarFormatter

HEADER_ERROR = "FINAL PROMPT HAS"


def _formatter():
    return TestGoogleCalendarFormatter()._get_formatter()


def _context(**overrides):
    return TestGoogleCalendarFormatter()._make_context(**overrides)


def _all_log_text(caplog):
    return "\n".join(r.getMessage() for r in caplog.records)


@pytest.fixture
def captured(caplog):
    caplog.set_level(logging.DEBUG)
    # the project loggers may not propagate; attach the handler to the root and the named logger
    for name in ("prompt_formatter", "core.prompt.formatter", "prompt_builder", "token_manager"):
        lg = logging.getLogger(name)
        lg.addHandler(caplog.handler)
        lg.setLevel(logging.DEBUG)
    return caplog


# ----------------------------------------------------- header check is line-anchored
QUOTING = "Here is the dump I pasted: see the [RECENT CONVERSATION] header in the middle of a line"
WRAPPED_QUOTING = "   " + QUOTING.replace("pasted:", "pasted:\n   ") + "\n"


@pytest.mark.parametrize("quote", [QUOTING, WRAPPED_QUOTING])
def test_a_mid_line_quote_of_the_header_is_not_a_duplicate(captured, quote):
    """A section that renders stored text verbatim (semantic chunks) may carry the header
    mid-line; the real [RECENT CONVERSATION] header still counts once."""
    ctx = _context(recent_conversations=[{"query": "hi", "response": "noted"}],
                   semantic_chunks=[{"query": quote, "response": "noted", "content": quote}])
    prompt = _formatter()._assemble_prompt(ctx, "hello")
    assert "see the [RECENT CONVERSATION] header" in prompt or "[RECENT CONVERSATION] header" in prompt  # control: quote rendered
    assert prompt.count("[RECENT CONVERSATION]") >= 2                                              # control: two raw occurrences
    assert HEADER_ERROR not in _all_log_text(captured)


# --------------------------------------------------------------- STM value not logged
SENTINEL = "ZXQ-stm-sentinel-7731"


def test_stm_summary_value_reaches_the_prompt_but_never_the_log(captured):
    stm = {"topic": SENTINEL, "user_question": f"asks about {SENTINEL}", "intent": "share",
           "tone": "neutral", "reference_type": "new_event",
           "temporal_facts": [], "open_threads": [], "constraints": []}
    prompt = _formatter()._assemble_prompt(_context(stm_summary=stm), "hello")
    assert SENTINEL in prompt                      # control: the value IS rendered
    assert SENTINEL not in _all_log_text(captured)  # ...and never logged
