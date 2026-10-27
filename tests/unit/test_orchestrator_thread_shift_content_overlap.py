"""[THREAD CONTEXT] must not assert a topic shift on a follow-up that carries the previous
exchange's own content, whatever the two classifier labels say.

Incident date: 2026-10-27 (audit of logs 10-20..10-26: 16 distinct "appears to shift topic"
injections, 10 of them follow-ups -- e.g. "Bus Reference Baseline" -> "No Covariate Model" within
one regression lesson; "Group By Concept" -> "Data Set Keys And Groups"). Class: BC-08, BC-04.

Drives THE deployed injection (`build_full_prompt`) and `_thread_topic_shifted`. Fixtures are
invented (pottery / garden vocabulary), each asserted clean AND wrapped/indented.
"""
import textwrap
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from core.orchestrator import _thread_topic_shifted
from tests.unit.test_process_user_query import _make_bfp_orch, _make_context

PRIOR = {
    "query": "How hot should the kiln get when I fire the glaze on these bowls?",
    "response": "Cone 6 is typical for that glaze; ramp slowly so the glaze matures without running.",
}

# (thread label, current label, follow-up that shares the exchange's content, nothing in the labels)
FOLLOW_UPS = [
    ("Pottery Glaze Basics", "Kiln Temperature Chart",
     "does the glaze change colour if the kiln runs hotter than the chart says"),
    ("Pottery Glaze Basics", "Bowl Shapes",
     "which bowls crack more often when the kiln cools quickly"),
    ("Pottery Glaze Basics", "Firing Schedule",
     "so a slow ramp is better for the glaze than a fast one?"),
]
# Brand-new subject: shares no content stems with the exchange or the label.
REAL_SHIFTS = [
    ("Pottery Glaze Basics", "Figures Provided", "quick check, what numbers have I given you lately?"),
    ("Pottery Glaze Basics", "Tomato Seedlings", "my tomato seedlings look leggy this week, any thoughts"),
]


def _wrapped(text):
    return "\n".join("    " + ln for ln in textwrap.wrap(text, 24))


def _exchange_text():
    return f"{PRIOR['query']} {PRIOR['response']}"


@pytest.mark.parametrize("wrap", [False, True])
@pytest.mark.parametrize("thread,current,query", FOLLOW_UPS)
def test_content_overlapping_follow_up_is_not_a_shift(thread, current, query, wrap):
    q = _wrapped(query) if wrap else query
    assert _thread_topic_shifted(thread, current, q, prior_exchange_text=_exchange_text()) is False


@pytest.mark.parametrize("wrap", [False, True])
@pytest.mark.parametrize("thread,current,query", REAL_SHIFTS)
def test_real_topic_change_still_asserts_a_shift(thread, current, query, wrap):
    q = _wrapped(query) if wrap else query
    assert _thread_topic_shifted(thread, current, q, prior_exchange_text=_exchange_text()) is True


def test_without_prior_exchange_behaviour_is_unchanged():
    thread, current, query = FOLLOW_UPS[0]
    assert _thread_topic_shifted(thread, current, query) is True


def test_long_paste_is_judged_by_its_authored_head():
    # the paste mentions glaze/kiln many times; the user's own sentence does not
    paste = "Here are my notes:\n\n" + "\n".join(
        f"- kiln glaze firing bowls cone {i} schedule" for i in range(40)
    )
    query = "ok so separately, my tomato seedlings look leggy. " + paste
    assert _thread_topic_shifted(
        "Pottery Glaze Basics", "Tomato Seedlings", query, prior_exchange_text=_exchange_text(),
    ) is True


def _tm():
    tm = MagicMock()
    tm.time_since_previous_message = MagicMock(return_value="3 m")
    return tm


@pytest.mark.asyncio
@pytest.mark.parametrize("wrap", [False, True])
async def test_deployed_injection_renders_no_shift_for_follow_up(wrap):
    thread, current, query = FOLLOW_UPS[0]
    ctx = _make_context(
        thread_context={
            "thread_id": "t1", "thread_depth": 2, "thread_topic": thread,
            "is_heavy_topic": False, "last_timestamp": datetime.now().isoformat(),
        },
        original_query=_wrapped(query) if wrap else query,
        primary_topic=current,
        stm_summary=None,
        last_exchange=dict(PRIOR),
    )
    orch = _make_bfp_orch(time_manager=_tm())
    _, system_prompt, _ = await orch.build_full_prompt(ctx, return_raw_context=True)
    assert "[THREAD CONTEXT]" in system_prompt
    assert "appears to shift topic" not in system_prompt
    assert "Follow the current query" not in system_prompt
    # labels diverge, so the stale label is never endorsed as the current topic
    assert "ongoing conversation thread about Pottery" not in system_prompt


@pytest.mark.asyncio
async def test_deployed_injection_still_asserts_real_shift():
    thread, current, query = REAL_SHIFTS[1]
    ctx = _make_context(
        thread_context={
            "thread_id": "t1", "thread_depth": 2, "thread_topic": thread,
            "is_heavy_topic": False, "last_timestamp": datetime.now().isoformat(),
        },
        original_query=query, primary_topic=current, stm_summary=None,
        last_exchange=dict(PRIOR),
    )
    orch = _make_bfp_orch(time_manager=_tm())
    _, system_prompt, _ = await orch.build_full_prompt(ctx, return_raw_context=True)
    assert "appears to shift topic" in system_prompt
