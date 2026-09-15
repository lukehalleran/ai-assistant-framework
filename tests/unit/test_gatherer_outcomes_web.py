"""CGR-20260913-007 #92 (F8b): `_get_web_search_results`'s own except, and
its provider-error/budget-refusal branches, stop collapsing into the same
`None` a genuine empty search or a deliberate non-search already returns —
the return value now agrees with the receipt (`last_web_decision`), which
already distinguished all four cases.

Contract (docs/execution/generalization/briefs/F8b.md):
- an exception escaping `multi_search` -> `OutcomeList.failed(<exception
  class name>)` (ANCHOR #92);
- a budget-refused empty result (`result.blocked == "budget"`, checked
  FIRST -- a budget refusal sets BOTH `error` and `blocked`, per
  knowledge/web_search_manager.py's `search()`) -> `OutcomeList.unavailable("budget")`;
- a provider-error empty result (`result.error` set, not budget-blocked)
  -> `OutcomeList.failed("provider_error")` (a constant label, never
  `str(result.error)`);
- a genuine empty result (no error, not blocked) and every deliberate
  non-search (disabled, vetoed, not triggered, no manager, ...) stay
  `None`;
- the success path and every receipt write are byte-for-byte unchanged.

Every gatherer host here is a bare `WebSearchMixin` instance built with
`__new__`/direct attributes (the test_sep10_web_search_gap.py `_gatherer`
precedent) or a `SimpleNamespace` bound to the REAL mixin method via
`types.MethodType` for the builder-level proof -- never a real
WebSearchManager, rate limiter, cache, Tavily client or network. A local
copy of the `full_builder`/`retrieval_limits` fake-builder pattern from
tests/unit/test_independent_prompt_audit.py is used for the builder-level
case (a local copy, not an import, per the F7b/F7c/F8a precedent against
cross-test-module coupling).

class: BC-20, BC-47, CM-05
"""
from __future__ import annotations

import types
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.prompt.formatter import PromptFormatter
from core.prompt.gatherer_web import WebSearchMixin
from utils.retrieval_outcome import outcome_status

MARKER = "F8BMARKQ23_sensitive_detail_must_not_leak"


def _decision(should_search=True, **overrides):
    base = dict(
        should_search=should_search,
        source="test",
        reason="test reason",
        confidence=0.9,
        depth=NS(value="quick"),
        search_terms=["current status update"],
        evidence_needed=False,
        blocked_reason=None,
        budget_remaining=100.0,
    )
    base.update(overrides)
    return NS(**base)


def _manager(multi_search_mock, *, rate_limiter=None):
    manager = MagicMock()
    manager.is_available.return_value = True
    manager.rate_limiter = rate_limiter
    manager.multi_search = multi_search_mock
    return manager


def _host(manager, *, decision=None, trigger_llm=None):
    """A bare WebSearchMixin host -- no ContextGatherer/UserProfile
    construction, matching test_sep12_web_evidence_budget.py's
    `_GathererHost` (a local copy, not an import)."""
    gatherer = WebSearchMixin.__new__(WebSearchMixin)
    gatherer.web_search_manager = manager
    gatherer.web_search_trigger_llm = trigger_llm or AsyncMock(
        return_value=decision if decision is not None else _decision())
    gatherer.web_search_trigger = None
    gatherer.model_manager = object()
    gatherer.memory_id_map = {}
    return gatherer


class TestTypedFailures:
    """The three sibling except/result branches this batch fixes."""

    @pytest.mark.asyncio
    async def test_raising_multi_search_is_failed_with_exception_class(self):
        manager = _manager(AsyncMock(side_effect=RuntimeError(MARKER)))
        host = _host(manager)

        result = await host._get_web_search_results("query")

        assert outcome_status(result) == ("failed", "RuntimeError")
        assert result == []
        # Receipt unchanged: the except already recorded the class name.
        assert host.last_web_decision["error"] == "RuntimeError"
        assert MARKER not in host.last_web_decision["error"]

    @pytest.mark.asyncio
    async def test_provider_error_result_is_failed_provider_error(self):
        provider_result = NS(has_results=False, pages=[], error=f"{MARKER} provider down",
                             blocked=None)
        manager = _manager(AsyncMock(return_value=provider_result))
        host = _host(manager)

        result = await host._get_web_search_results("query")

        assert outcome_status(result) == ("failed", "provider_error")
        assert result == []
        assert MARKER not in result.reason
        # Receipt unchanged: still the raw provider text, exactly as today
        # (contract point 5) -- the constant label is new only on `result`.
        assert host.last_web_decision["error"] == f"{MARKER} provider down"

    @pytest.mark.asyncio
    async def test_budget_refused_empty_result_is_unavailable_budget(self):
        provider_result = NS(has_results=False, pages=[], error=None, blocked="budget")
        manager = _manager(AsyncMock(return_value=provider_result))
        host = _host(manager)

        result = await host._get_web_search_results("query")

        assert outcome_status(result) == ("unavailable", "budget")
        assert result == []
        assert host.last_web_decision["blocked"] == "budget"

    @pytest.mark.asyncio
    async def test_budget_and_error_both_set_checks_budget_first(self):
        """PARENT-VERIFIED FACT: a budget refusal sets BOTH `error` and
        `blocked="budget"` (knowledge/web_search_manager.py `search()`
        1317-1322, 1401-1406) -- the budget check must run first, or this
        would wrongly read as a provider error."""
        provider_result = NS(
            has_results=False, pages=[],
            error="Daily credit limit reached. Remaining: 0",
            blocked="budget",
        )
        manager = _manager(AsyncMock(return_value=provider_result))
        host = _host(manager)

        result = await host._get_web_search_results("query")

        assert outcome_status(result) == ("unavailable", "budget")
        assert host.last_web_decision["error"] == "Daily credit limit reached. Remaining: 0"
        assert host.last_web_decision["blocked"] == "budget"


class TestUnaffectedControls:
    """Genuine empty searches and deliberate non-searches stay `None`."""

    @pytest.mark.asyncio
    async def test_genuine_empty_result_stays_none(self):
        provider_result = NS(has_results=False, pages=[], error=None, blocked=None)
        manager = _manager(AsyncMock(return_value=provider_result))
        host = _host(manager)

        result = await host._get_web_search_results("query")

        assert result is None

    @pytest.mark.asyncio
    async def test_not_triggered_decision_stays_none(self):
        manager = _manager(AsyncMock(side_effect=AssertionError("must not be called")))
        host = _host(manager, decision=_decision(should_search=False, confidence=0.0))

        result = await host._get_web_search_results("query")

        assert result is None
        manager.multi_search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_disabled_deliberate_non_search_stays_none(self, monkeypatch):
        monkeypatch.setattr("config.app_config.WEB_SEARCH_ENABLED", False)
        manager = _manager(AsyncMock(side_effect=AssertionError("must not be called")))
        host = _host(manager)

        result = await host._get_web_search_results("query")

        assert result is None
        manager.multi_search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_success_returns_the_same_result_object(self):
        page = NS(url="https://example.test/a")
        provider_result = NS(
            has_results=True, pages=[page], error=None, blocked=None,
            total_credits_used=1, from_cache=False,
        )
        manager = _manager(AsyncMock(return_value=provider_result))
        host = _host(manager)

        result = await host._get_web_search_results("query")

        assert result is provider_result
        assert host.memory_id_map["WEB_SEARCH"]["sources"] == ["https://example.test/a"]


# ---------------------------------------------------------------------------
# Builder-level proof (local copy of test_independent_prompt_audit.py's
# full_builder/retrieval_limits pattern, per the F7b/F7c/F8a precedent).
# ---------------------------------------------------------------------------

def _compression_builder():
    from core.prompt.builder import UnifiedPromptBuilder
    from core.prompt.token_manager import TokenManager

    class _CharTokenizer:
        def count_tokens(self, text, model_name):
            return len(text)

    builder = UnifiedPromptBuilder.__new__(UnifiedPromptBuilder)
    builder.model_manager = NS(
        get_active_model_name=lambda: "audit",
        generate_once=AsyncMock(return_value="Compressed."),
        active_model_name="audit",
    )
    builder.token_manager = TokenManager(
        NS(get_active_model_name=lambda: "audit"), _CharTokenizer(), 1200)
    builder._llm_compress_cache = {}
    return builder


def _full_builder(monkeypatch, web_search_host):
    """Run the deployed full builder without optional stores or models, with
    a REAL bound `WebSearchMixin._get_web_search_results` wired in place of
    the usual `AsyncMock(return_value=[])` stub."""
    builder = _compression_builder()
    builder.time_manager = None
    builder.memory_coordinator = NS(
        scorer=None, chroma_store=None, corpus_manager=None,
        get_summaries=lambda count: [], get_reflections=AsyncMock(return_value=[]),
    )
    builder._skill_activation_policy = None
    builder._should_use_light_path = lambda *args: False
    builder._is_continuation_answer = lambda *args: False
    builder._hygiene_and_caps = AsyncMock(side_effect=lambda context, **kw: context)
    builder.context_gatherer = web_search_host
    web_search_host.memory_id_map = {}
    web_search_host.clear_memory_id_map = lambda: None
    web_search_host.get_narrative_context = lambda: ""
    web_search_host._get_recent_conversations = AsyncMock(return_value=[])
    web_search_host.get_user_profile_context = AsyncMock(return_value="")
    for name in ("GOOGLE_CALENDAR_ENABLED", "EMAIL_PASSIVE_CONTEXT_ENABLED", "DAEMON_NOTES_ENABLED"):
        monkeypatch.setattr("config.app_config." + name, False)
    monkeypatch.setattr("config.app_config.WEB_SEARCH_ENABLED", True)
    monkeypatch.setattr("core.prompt.builder.LLM_COMPRESSION_ENABLED", False)
    return builder


def _retrieval_limits():
    return dict.fromkeys([
        "max_mems", "max_summaries", "max_reflections", "max_dreams", "max_semantic",
        "max_wiki", "max_skills", "max_proposals", "max_git_commits",
        "max_surfaced_threads", "max_reference_docs", "max_user_uploads",
        "max_proactive", "max_visual_memories", "max_personal_notes", "max_graph_sentences",
    ], 0) | {"max_recent": 2}


class TestThroughBuilder:

    @pytest.mark.asyncio
    async def test_raising_search_is_failed_and_label_reads_error_web_section_absent(
            self, monkeypatch):
        manager = _manager(AsyncMock(side_effect=RuntimeError(MARKER)))
        host = WebSearchMixin.__new__(WebSearchMixin)
        host.web_search_manager = manager
        host.web_search_trigger_llm = AsyncMock(return_value=_decision())
        host.web_search_trigger = None
        host.model_manager = object()
        # Bind the REAL, unedited-signature `_get_web_search_results` as a
        # bound method on this SimpleNamespace-shaped host, so the builder's
        # task genuinely drives this batch's own edited body end to end.
        host._get_web_search_results = types.MethodType(
            WebSearchMixin._get_web_search_results, host)

        builder = _full_builder(monkeypatch, host)
        result = await builder.build_prompt(
            "Synthetic question", retrieval_overrides=_retrieval_limits())

        assert "_build_time" in result, "builder must not silently return its error fallback"
        assert result["_section_outcomes"]["web_search"] == {
            "status": "failed", "reason": "RuntimeError",
        }
        assert result["web_search_results"] == []
        assert MARKER not in str(result["_section_outcomes"])

        formatter = PromptFormatter(token_manager=MagicMock(), time_manager=None)
        inventory = formatter._build_feature_inventory(result)
        assert "web_search=ON(error)" in inventory
