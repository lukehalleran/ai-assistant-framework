"""Per-section retrieval-outcome recording in the builder's gather loop (F5).

Design: docs/execution/generalization/failure_outcome_design.md, "Decisions
per request" -> CGR-007, batch row F5, and the 2026-09-14 amendments ("The
builder's gather loop turns failures into []"; "get_narrative_context runs
outside the gather loop"). Drives the deployed
UnifiedPromptBuilder.build_prompt through the existing full_builder fake
pattern (test_prompt_timeout.py's technique, test_independent_prompt_audit's
fixture) -- no real store, model or network.

With the `retrieval_limits()` overrides (every max_* zeroed except
max_recent=2), the deployed gather loop creates exactly three tasks:
"recent", "user_profile" and "web_search" (google_calendar/relevant_emails/
daemon_self_notes are config-gated off by full_builder; everything else is
zeroed). "narrative" is recorded separately, outside the loop.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.prompt.formatter import PromptFormatter
from core.prompt.token_manager import TokenManager
from tests.unit.test_independent_prompt_audit import full_builder, retrieval_limits
from utils.retrieval_outcome import OutcomeList, RETRIEVAL_STATES


RECENT = [{"query": "Synthetic question", "response": "Synthetic retained answer."}]
MARKER = "F5MARKQ42_sensitive_detail_must_not_leak"


def _make_builder(monkeypatch, recent=RECENT):
    return full_builder(monkeypatch, recent, budget=10000)


class TestDefaultRunRecordsHealthyAndEmptySections:
    @pytest.mark.asyncio
    async def test_default_fakes_produce_expected_statuses(self, monkeypatch):
        builder = _make_builder(monkeypatch)
        result = await builder.build_prompt(
            "Synthetic question", retrieval_overrides=retrieval_limits()
        )
        outcomes = result["_section_outcomes"]
        # Healthy non-empty section -> succeeded.
        assert outcomes["recent"] == {"status": "succeeded", "reason": ""}
        # Empty sections (default fakes) -> no_results.
        assert outcomes["user_profile"] == {"status": "no_results", "reason": ""}
        assert outcomes["web_search"] == {"status": "no_results", "reason": ""}
        assert outcomes["narrative"] == {"status": "no_results", "reason": ""}
        assert set(outcomes) == {"recent", "user_profile", "web_search", "narrative"}
        assert all(v["status"] in RETRIEVAL_STATES for v in outcomes.values())
        # Controls: gathered values and _task_timings are unaffected by the
        # new bookkeeping -- byte-for-byte what today's build already returns.
        assert result["recent_conversations"] == RECENT
        # Pre-existing behaviour, untouched by F5: `_raw or []` turns the
        # gatherer's "" into [] (design doc's noted `or []` quirk).
        assert result["user_profile"] == []
        assert "_task_timings" in result
        assert set(result["_task_timings"]) == {"recent", "user_profile", "web_search"}
        assert all(isinstance(v, float) for v in result["_task_timings"].values())


class TestSectionExceptionRecordsFailed:
    @pytest.mark.asyncio
    async def test_exception_recorded_failed_others_preserved(self, monkeypatch):
        builder = _make_builder(monkeypatch)
        builder.context_gatherer.get_user_profile_context = AsyncMock(
            side_effect=ValueError(f"{MARKER} while scoring profile")
        )
        result = await builder.build_prompt(
            "Synthetic question", retrieval_overrides=retrieval_limits()
        )
        outcomes = result["_section_outcomes"]
        assert outcomes["user_profile"] == {"status": "failed", "reason": "ValueError"}
        assert outcomes["recent"] == {"status": "succeeded", "reason": ""}
        assert result["recent_conversations"] == RECENT
        assert not result["user_profile"]
        # Privacy: the exception message text never reaches a recorded reason.
        assert MARKER not in str(result)


class TestPendingTaskRecordsTimeout:
    @pytest.mark.asyncio
    async def test_pending_task_recorded_unavailable_timeout(self, monkeypatch):
        builder = _make_builder(monkeypatch)
        fast_done, slow_entered = asyncio.Event(), asyncio.Event()

        async def recent(*args):
            fast_done.set()
            return RECENT

        async def profile(*args, **kwargs):
            slow_entered.set()
            await asyncio.Future()

        builder.context_gatherer._get_recent_conversations = recent
        builder.context_gatherer.get_user_profile_context = profile
        real_wait = asyncio.wait

        async def expired_deadline(tasks, *, timeout, return_when):
            assert timeout == 30.0  # exercise the deployed global gather wait
            await asyncio.wait_for(
                asyncio.gather(fast_done.wait(), slow_entered.wait()), 2
            )
            return await real_wait(tasks, timeout=0, return_when=return_when)

        monkeypatch.setattr("core.prompt.builder.asyncio.wait", expired_deadline)
        result = await builder.build_prompt(
            "Synthetic question", retrieval_overrides=retrieval_limits()
        )
        outcomes = result["_section_outcomes"]
        assert outcomes["user_profile"] == {"status": "unavailable", "reason": "timeout"}
        assert outcomes["recent"] == {"status": "succeeded", "reason": ""}


class TestOutcomeListStatusReadBeforeFlattening:
    """THE TRAP: an empty OutcomeList is falsy, so `task.result() or []`
    would silently replace it with a plain [] if status were read AFTER the
    `or`. The status must be captured from the raw result first."""

    @pytest.mark.asyncio
    async def test_empty_failed_outcomelist_status_survives_flattening(self, monkeypatch):
        builder = _make_builder(monkeypatch)
        builder.context_gatherer._get_recent_conversations = AsyncMock(
            return_value=OutcomeList.failed("UPSTREAM_DOWN")
        )
        result = await builder.build_prompt(
            "Synthetic question", retrieval_overrides=retrieval_limits()
        )
        assert result["_section_outcomes"]["recent"] == {
            "status": "failed",
            "reason": "UPSTREAM_DOWN",
        }
        # The gathered value is still flattened to [] exactly as today.
        assert result["recent_conversations"] == []
        assert not isinstance(result["recent_conversations"], OutcomeList)


class TestGatherExceptionMarksAllFailedGatherError:
    @pytest.mark.asyncio
    async def test_asyncio_wait_raising_marks_every_created_task(self, monkeypatch):
        builder = _make_builder(monkeypatch)

        async def boom(*args, **kwargs):
            raise RuntimeError(f"{MARKER} wait blew up")

        monkeypatch.setattr("core.prompt.builder.asyncio.wait", boom)
        result = await builder.build_prompt(
            "Synthetic question", retrieval_overrides=retrieval_limits()
        )
        outcomes = result["_section_outcomes"]
        for name in ("recent", "user_profile", "web_search"):
            assert outcomes[name] == {"status": "failed", "reason": "gather_error"}
        assert "_build_time" in result, "builder must not fall back to its error path"
        assert MARKER not in str(result)


class TestNarrativeOutcome:
    @pytest.mark.asyncio
    async def test_narrative_exception_recorded_failed(self, monkeypatch):
        builder = _make_builder(monkeypatch)

        def raising_narrative():
            raise ValueError(f"{MARKER} narrative blew up")

        builder.context_gatherer.get_narrative_context = raising_narrative
        result = await builder.build_prompt(
            "Synthetic question", retrieval_overrides=retrieval_limits()
        )
        assert result["_section_outcomes"]["narrative"] == {
            "status": "failed",
            "reason": "ValueError",
        }
        assert MARKER not in str(result["_section_outcomes"])

    @pytest.mark.asyncio
    async def test_narrative_empty_string_is_no_results(self, monkeypatch):
        builder = _make_builder(monkeypatch)
        builder.context_gatherer.get_narrative_context = lambda: ""
        result = await builder.build_prompt(
            "Synthetic question", retrieval_overrides=retrieval_limits()
        )
        assert result["_section_outcomes"]["narrative"] == {"status": "no_results", "reason": ""}

    @pytest.mark.asyncio
    async def test_narrative_non_empty_is_succeeded(self, monkeypatch):
        builder = _make_builder(monkeypatch)
        builder.context_gatherer.get_narrative_context = lambda: "It is currently Monday."
        result = await builder.build_prompt(
            "Synthetic question", retrieval_overrides=retrieval_limits()
        )
        assert result["_section_outcomes"]["narrative"] == {"status": "succeeded", "reason": ""}
        assert result["narrative_state"] == "It is currently Monday."

    @pytest.mark.asyncio
    async def test_max_narrative_zero_skips_entry(self, monkeypatch):
        builder = _make_builder(monkeypatch)
        overrides = {**retrieval_limits(), "max_narrative": 0}
        result = await builder.build_prompt("Synthetic question", retrieval_overrides=overrides)
        assert "narrative" not in result["_section_outcomes"]


class TestSectionOutcomesDoNotLeakIntoRenderedPrompt:
    def _make_formatter(self):
        token_mgr = MagicMock()
        token_mgr.count_tokens = MagicMock(return_value=10)
        fmt = PromptFormatter(token_manager=token_mgr, time_manager=None)
        fmt._feature_inventory_cache = None
        return fmt

    def _base_context(self, **overrides):
        ctx = {
            "recent_conversations": [], "memories": [], "user_profile": "",
            "narrative_state": "", "summaries": [], "reflections": [],
            "dreams": [], "semantic_chunks": [], "wiki": [], "personal_notes": [],
            "reference_docs": [], "user_uploads": [], "git_commits": [],
            "procedural_skills": [], "proposed_features": [], "graph_context": [],
            "unresolved_threads": [], "upcoming_schedule": [], "google_calendar": [],
            "proactive_insights": [], "web_search_results": None,
        }
        ctx.update(overrides)
        return ctx

    def test_rendered_prompt_excludes_key_and_reason_labels(self):
        fmt = self._make_formatter()
        outcomes = {
            "recent": {"status": "succeeded", "reason": ""},
            "user_profile": {"status": "failed", "reason": "ValueError"},
            "web_search": {"status": "unavailable", "reason": "timeout"},
            "narrative": {"status": "failed", "reason": "RuntimeError"},
        }
        ctx = self._base_context(
            _section_outcomes=outcomes,
            _task_timings={"recent": 0.01, "user_profile": 0.02, "web_search": 5.0},
        )
        prompt = fmt._assemble_prompt(ctx, "hello")
        assert "_section_outcomes" not in prompt
        assert "_task_timings" not in prompt
        for label in ("ValueError", "RuntimeError", "gather_error", "cancelled", "timeout"):
            assert label not in prompt


def _char_token_manager(budget):
    return TokenManager(
        SimpleNamespace(get_active_model_name=lambda: "audit"),
        SimpleNamespace(count_tokens=lambda text, model_name=None: len(text or "")),
        budget,
    )


class TestTokenMeterUnaffectedBySectionOutcomes:
    def test_unmetered_total_unchanged_with_and_without_key(self):
        base_ctx = {"recent_conversations": [{"content": "hello world"} for _ in range(3)]}

        tm_without = _char_token_manager(10_000)
        tm_without._manage_token_budget(dict(base_ctx))
        usage_without = tm_without._prompt_token_usage

        outcomes = {
            "recent": {"status": "succeeded", "reason": ""},
            "user_profile": {"status": "failed", "reason": "ValueError"},
        }
        tm_with = _char_token_manager(10_000)
        trimmed_with = tm_with._manage_token_budget({**base_ctx, "_section_outcomes": outcomes})
        usage_with = tm_with._prompt_token_usage

        assert usage_with == usage_without
        assert trimmed_with["_section_outcomes"] == outcomes
