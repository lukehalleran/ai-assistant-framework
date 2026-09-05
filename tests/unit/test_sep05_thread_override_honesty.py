"""2026-09-05: [THREAD CONTEXT] honesty when the STM continuity override
(2026-08-05) hides a genuine topic divergence.

Live defect (2026-09-05 13:10 debug record): turn 1 was a public-news
question labeled "Politicians Charged With Crimes" (thread depth 1, flagged
heavy). Turn 2 was an unrelated 77-word request to read an advisor email,
labeled "Project Timeline Deadlines"; the STM analyzer classified it
reference_type="recall" because it matched a memory from the previous day.
``_thread_topic_shifted`` correctly refuses to assert a shift on an STM
recall/clarification/correction turn (2026-08-05, anaphoric-continuation
doctrine) — but that override assumes classifier-noisy labels on an
otherwise-continuous conversation, not two genuinely unrelated topics. The
old code fell into the plain continuity ``else`` branch and rendered:

    This is message #1 in an ongoing conversation thread about Politicians
    Charged With Crimes
    This is a sensitive/heavy topic. Be empathetic and specific.

on a turn that was neither. ``core.orchestrator._thread_labels_diverge`` +
``_continuity_asserted_by_stm_override`` detect exactly this combination
(shift not asserted ONLY because of the STM override, AND the labels are
actually unrelated) and the injection now renders a neutral note instead,
without carrying the previous thread's heavy-topic flag forward.

All fixtures use synthetic topic labels; the two from the live debug record
carry no personal data (they are the assigned topic strings, not the user's
own words).
"""
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from core.orchestrator import _continuity_asserted_by_stm_override, _thread_labels_diverge
from tests.unit.test_process_user_query import _make_bfp_orch, _make_context


def _tm(gap: str):
    tm = MagicMock()
    tm.time_since_previous_message = MagicMock(return_value=gap)
    return tm


def _fresh_ts() -> str:
    return datetime.now().isoformat()


# ---------------------------------------------------------------------------
# [THREAD CONTEXT] rendering
# ---------------------------------------------------------------------------

class TestThreadDivergenceHonesty:
    @pytest.mark.asyncio
    async def test_live_shape_unrelated_recall_renders_neutral_note(self):
        thread_ctx = {
            "thread_id": "t1",
            "thread_depth": 1,
            "thread_topic": "Politicians Charged With Crimes",
            "is_heavy_topic": True,
            "last_timestamp": _fresh_ts(),
        }
        query = (
            "I got an email from my advisor in my outlook inbox a couple of days ago "
            "about the project timeline for this term, and I have been meaning to go "
            "back to it but have not had time this week with everything else going "
            "on. Can you read the last email I received from them in outlook and "
            "tell me what deadlines it mentioned? I want to make sure I am not "
            "missing anything before the weekend."
        )
        ctx = _make_context(
            thread_context=thread_ctx,
            original_query=query,
            primary_topic="Project Timeline Deadlines",
            stm_summary={"reference_type": "recall"},
        )
        orch = _make_bfp_orch(time_manager=_tm("3 m"))
        _, system_prompt, _ = await orch.build_full_prompt(ctx, return_raw_context=True)

        assert "[THREAD CONTEXT]" in system_prompt
        assert "ongoing conversation thread about Politicians" not in system_prompt
        assert "sensitive/heavy topic" not in system_prompt
        assert "Follow the current query" not in system_prompt
        assert "Politicians Charged With Crimes" in system_prompt
        assert "Project Timeline Deadlines" in system_prompt
        assert "Maintain conversational continuity" not in system_prompt

    @pytest.mark.asyncio
    async def test_related_labels_with_recall_stay_on_continuity_wording(self):
        thread_ctx = {
            "thread_id": "t2",
            "thread_depth": 3,
            "thread_topic": "Bank Communication",
            "is_heavy_topic": False,
            "last_timestamp": _fresh_ts(),
        }
        ctx = _make_context(
            thread_context=thread_ctx,
            original_query="did they ever answer about the fees",
            primary_topic="Bank Fees",
            stm_summary={"reference_type": "recall"},
        )
        orch = _make_bfp_orch(time_manager=_tm("3 m"))
        _, system_prompt, _ = await orch.build_full_prompt(ctx, return_raw_context=True)

        assert (
            "This is message #3 in an ongoing conversation thread about "
            "Bank Communication" in system_prompt
        )

    @pytest.mark.asyncio
    async def test_anaphoric_continuation_with_diverged_labels_keeps_continuity(self):
        # 2026-07-28 doctrine: an anaphoric fragment never asserts a shift or
        # the neutral note, regardless of how far the fresh-classified label
        # has drifted from the thread topic.
        thread_ctx = {
            "thread_id": "t3",
            "thread_depth": 2,
            "thread_topic": "Long Covid",
            "is_heavy_topic": False,
            "last_timestamp": _fresh_ts(),
        }
        ctx = _make_context(
            thread_context=thread_ctx,
            original_query="It was maybe 3 years of twice a week",
            primary_topic="Exercise Routine",
            stm_summary=None,
        )
        orch = _make_bfp_orch(time_manager=_tm("3 m"))
        _, system_prompt, _ = await orch.build_full_prompt(ctx, return_raw_context=True)

        assert "ongoing conversation thread about Long Covid" in system_prompt
        assert "Follow the current query" not in system_prompt

    @pytest.mark.asyncio
    async def test_genuine_shift_suppresses_carried_heavy_line(self):
        thread_ctx = {
            "thread_id": "t4",
            "thread_depth": 2,
            "thread_topic": "Forearm Pain",
            "is_heavy_topic": True,
            "last_timestamp": _fresh_ts(),
        }
        ctx = _make_context(
            thread_context=thread_ctx,
            original_query="Which tax software handles a 1099 and a W2 together?",
            primary_topic="Tax Filing Software",
            stm_summary=None,
        )
        orch = _make_bfp_orch(time_manager=_tm("3 m"))
        _, system_prompt, _ = await orch.build_full_prompt(ctx, return_raw_context=True)

        assert "appears to shift topic" in system_prompt
        assert "sensitive/heavy topic" not in system_prompt

    @pytest.mark.asyncio
    async def test_continuity_branch_still_renders_heavy_line(self):
        thread_ctx = {
            "thread_id": "t5",
            "thread_depth": 2,
            "thread_topic": "Forearm Pain",
            "is_heavy_topic": True,
            "last_timestamp": _fresh_ts(),
        }
        ctx = _make_context(
            thread_context=thread_ctx,
            original_query="still hurts when I type",
            primary_topic="Forearm Pain",
            stm_summary=None,
        )
        orch = _make_bfp_orch(time_manager=_tm("3 m"))
        _, system_prompt, _ = await orch.build_full_prompt(ctx, return_raw_context=True)

        assert "sensitive/heavy topic" in system_prompt


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

class TestThreadDivergenceHelpers:
    def test_thread_labels_diverge_related(self):
        assert _thread_labels_diverge("Bank Communication", "Bank Fees") is False

    def test_thread_labels_diverge_unrelated(self):
        assert (
            _thread_labels_diverge(
                "Politicians Charged With Crimes", "Project Timeline Deadlines"
            )
            is True
        )

    def test_thread_labels_diverge_empty_or_none_underfires(self):
        assert _thread_labels_diverge("", "Bank Fees") is False
        assert _thread_labels_diverge(None, "Bank Fees") is False
        assert _thread_labels_diverge("Bank Communication", "") is False
        assert _thread_labels_diverge("Bank Communication", None) is False
        assert _thread_labels_diverge("", "") is False
        assert _thread_labels_diverge(None, None) is False

    def test_continuity_asserted_by_stm_override_false_on_anaphoric(self):
        assert _continuity_asserted_by_stm_override("It was maybe 3 years", "recall") is False

    def test_continuity_asserted_by_stm_override_true_on_recall(self):
        query = (
            "Can you read the last email I received from them in outlook and "
            "tell me what deadlines it mentioned?"
        )
        assert _continuity_asserted_by_stm_override(query, "recall") is True

    def test_continuity_asserted_by_stm_override_false_without_stm_override(self):
        query = (
            "Can you read the last email I received from them in outlook and "
            "tell me what deadlines it mentioned?"
        )
        assert _continuity_asserted_by_stm_override(query, "new_event") is False
        assert _continuity_asserted_by_stm_override(query, None) is False
