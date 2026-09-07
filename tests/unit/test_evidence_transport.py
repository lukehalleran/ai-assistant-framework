"""Evidence transport + truthful receipts (2026-09-06, Phase A).

No providers/stores/embedders are constructed — pure unit tests over the
deployed functions/classes.

Verified defects these tests guard against (see
docs/HANDOFF_20260906_phaseAB_contracts.md "Phase A"):
1. `token_manager._manage_token_budget`'s non-list branch did
   `item_text = str(val)` for ANY non-list section; once the repr exceeded
   SEMANTIC_ITEM_MAX_TOKENS (800) it wrote that STRING back. A WebSearchResult
   dataclass repr for more than ~1 page of content always exceeds 800 tokens,
   so the base prompt's web section silently vanished (the formatter requires
   `hasattr(web_search, 'has_results')`).
2. `controller.run_agentic_search`: when round 1 was NOT a web search,
   `initial_context["web_search_results"]` never reached
   `session.accumulated_context`, the decision prompt, or the final prompt —
   decision-answer reuse then answered from counts + a short digest alone.
3. `final_prompt_hash="decision-answer-reuse"` was a sentinel, not a hash;
   nothing recorded which call produced the answer or what it saw.
4. `_build_final_prompt` labeled ALL accumulated tool context
   "[WEB SEARCH RESULTS - N rounds]" and injected a "every claim MUST cite
   [WEB_N]" instruction even for memory-only loops.
"""
import re as _re
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.agentic.controller import AgenticSearchController
from core.agentic.tools import ToolExecutor
from core.agentic.types import AgenticSearchSession, SearchDecision
from core.prompt.formatter import PromptFormatter
from core.prompt.token_manager import STRUCTURED_SECTION_ADAPTERS  # noqa: F401 (existence guard)
from gui.handlers import _attach_agentic_provenance
from knowledge.web_search_manager import WebPage, WebSearchResult

from tests.unit.test_agentic_decision_answer_reuse import LONG_ANSWER
from tests.unit.test_independent_prompt_audit import CharacterTokenizer, manager  # noqa: F401


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mock_tool_executor():
    """Lightweight double exposing exactly what run_agentic_search /
    _build_final_prompt touch on self._tool_executor, with the REAL
    ToolExecutor._merge_web_ids bound (types.MethodType) so id numbering
    behaves exactly as production — the session-wide id map threading is the
    whole point of A3's seeding path."""
    class _Exec:
        pass
    ex = _Exec()
    ex._current_web_source_map = {}
    ex._current_wiki_source_map = {}
    ex.get_tool_health = lambda: "All tools nominal."
    ex._merge_web_ids = types.MethodType(ToolExecutor._merge_web_ids, ex)
    return ex


@pytest.fixture
def controller():
    model_manager = MagicMock()
    model_manager.api_models = {}
    c = AgenticSearchController(model_manager=model_manager, web_search_manager=MagicMock())
    c._tool_executor = _mock_tool_executor()
    return c


def _fake_decision_factory(capture=None, answer=LONG_ANSWER, decisions=None):
    async def fake(*a, **k):
        if capture is not None:
            capture["prompt"] = k.get("prompt")
        if decisions is not None:
            return decisions
        return [SearchDecision(wants_answer=True, partial_response=answer)]
    return fake


async def _run(controller, **kwargs):
    events = []
    async for ev in controller.run_agentic_search(**kwargs):
        events.append(ev)
    text = "".join(e for e in events if isinstance(e, str))
    return events, text


# ---------------------------------------------------------------------------
# T1 — base-prompt web section survives budgeting and renders (write FIRST;
# on HEAD this fails with after_type == "str")
# ---------------------------------------------------------------------------

def _four_pages():
    return [
        WebPage(url="https://example.com/big", title="Big Page", content="A" * 12000, score=0.9),
        WebPage(url="https://example.com/p2", title="P2", content="B" * 100, score=0.8),
        WebPage(url="https://example.com/p3", title="P3", content="C" * 100, score=0.7),
        WebPage(url="https://example.com/p4", title="P4", content="D" * 100, score=0.6),
    ]


class TestT1BudgetNeverStringifiesWebResults:

    def test_survives_budget_and_renders(self):
        tm = manager(10000)
        result = WebSearchResult(query="q", pages=_four_pages())

        after = tm._manage_token_budget({"web_search_results": result})
        val = after["web_search_results"]
        after_type = type(val).__name__

        assert after_type != "str", (
            f"web_search_results was coerced to a string (type={after_type}) "
            "instead of staying a WebSearchResult"
        )
        assert isinstance(val, WebSearchResult)
        assert val.has_results
        assert len(val.pages) <= 8

        fmt = PromptFormatter(token_manager=MagicMock())
        ctx = dict(after)
        prompt = fmt._assemble_prompt(context=ctx)

        assert "[WEB SEARCH RESULTS]" in prompt
        assert "[WEB_1]" in prompt
        # The 12,000-char page must render CLIPPED, not verbatim — clipping
        # happens at render time regardless of whether the budget manager
        # itself needed to shrink (this content was well under
        # WEB_SEARCH_SECTION_MAX_TOKENS at the (8,2000) rendered view).
        assert ("A" * 12000) not in prompt
        assert ("A" * 2000 + "...") in prompt

        rendered_ids = {m.strip("[]") for m in _re.findall(r"\[WEB_\d+\]", prompt)}
        assert rendered_ids == set(ctx["_web_source_map"].keys())


# ---------------------------------------------------------------------------
# T2 — shrinking budgets walk the ladder, then drop to None; never a str
# ---------------------------------------------------------------------------

def _ten_big_pages():
    return [
        WebPage(
            url=f"https://example.com/p{i}", title=f"Page {i}",
            content=chr(65 + i) * 2500, score=1.0 - i * 0.05,
        )
        for i in range(10)
    ]


class TestT2BudgetShrinksThenDrops:

    @pytest.mark.parametrize("budget,expect_none", [(900, False), (300, True), (50, True)])
    def test_shrinks_down_ladder_then_drops(self, budget, expect_none):
        tm = manager(budget)
        result = WebSearchResult(query="q", pages=_ten_big_pages())

        after = tm._manage_token_budget({"web_search_results": result})
        val = after.get("web_search_results")

        assert not isinstance(val, str)
        if expect_none:
            assert val is None
        else:
            assert isinstance(val, WebSearchResult)
            assert val.has_results
            assert len(val.pages) < 8  # shrunk down from the original 8-visible set
        assert tm._prompt_token_usage <= budget


# ---------------------------------------------------------------------------
# T3 — empty / error / dict-shaped results pass through untouched
# ---------------------------------------------------------------------------

class TestT3EmptyErrorAndDictShapesPreserved:

    def test_empty_result_passes_through(self):
        tm = manager(500)
        result = WebSearchResult(query="q", pages=[])
        after = tm._manage_token_budget({"web_search_results": result})
        assert after["web_search_results"] is result

        fmt = PromptFormatter(token_manager=MagicMock())
        prompt = fmt._assemble_prompt(context=dict(after))
        assert "[WEB SEARCH RESULTS]" not in prompt

    def test_error_result_passes_through(self):
        tm = manager(500)
        pages = [WebPage(url="https://x.example.com", title="X", content="body", score=0.5)]
        result = WebSearchResult(query="q", pages=pages, error="boom")
        after = tm._manage_token_budget({"web_search_results": result})
        assert after["web_search_results"] is result

        fmt = PromptFormatter(token_manager=MagicMock())
        prompt = fmt._assemble_prompt(context=dict(after))
        assert "[WEB SEARCH RESULTS]" not in prompt

    def test_dict_shaped_left_unchanged(self):
        tm = manager(500)
        val = {"pages": [{"title": "T", "url": "https://x.example.com", "content": "c"}]}
        after = tm._manage_token_budget({"web_search_results": val})
        assert after["web_search_results"] == val
        assert isinstance(after["web_search_results"], dict)


# ---------------------------------------------------------------------------
# T4 — CJK/emoji content with a 1-token-per-char tokenizer
# ---------------------------------------------------------------------------

class TestT4UnicodeContentSurvivesBudget:

    def test_cjk_and_emoji_content(self):
        tm = manager(2000)
        pages = [
            WebPage(
                url="https://example.jp/a", title="日本語のタイトル",
                content="これはテスト内容です。🎉" * 50, score=0.9,
            ),
            WebPage(url="https://example.jp/b", title="Emoji 🚀", content="🔥" * 200, score=0.5),
        ]
        result = WebSearchResult(query="q", pages=pages)
        after = tm._manage_token_budget({"web_search_results": result})
        val = after["web_search_results"]

        assert not isinstance(val, str)
        assert isinstance(val, WebSearchResult)
        assert tm._prompt_token_usage <= 2000


# ---------------------------------------------------------------------------
# T5 — unknown structured type never becomes a string
# ---------------------------------------------------------------------------

class TestT5UnknownStructuredTypeNeverStringified:

    def test_simplenamespace_left_untouched(self):
        tm = manager(500)
        val = SimpleNamespace(foo=1)
        after = tm._manage_token_budget({"web_search_results": val})
        result_val = after["web_search_results"]

        assert not isinstance(result_val, str)
        assert result_val is val

        fmt = PromptFormatter(token_manager=MagicMock())
        prompt = fmt._assemble_prompt(context=dict(after))
        assert "[WEB SEARCH RESULTS]" not in prompt


# ---------------------------------------------------------------------------
# T6 — list-of-dict web_search_results (existing shape) unaffected
# ---------------------------------------------------------------------------

class TestT6ListShapeUnaffected:

    def test_list_of_dict_web_search_results_unaffected(self):
        tm = manager(100)
        result = tm._manage_token_budget({
            "memories": [{"content": "M" * 100}],
            "google_calendar": [{"content": "C" * 50}],
            "web_search_results": [{"content": "W" * 50}],
        })
        assert result["web_search_results"] == [{"content": "W" * 50}]
        assert result["google_calendar"] == [{"content": "C" * 50}]
        assert result["memories"] == []


# ---------------------------------------------------------------------------
# T7 — controller seeds pre-gathered base web evidence into the loop
# ---------------------------------------------------------------------------

class TestT7SeedsPregatheredWebEvidence:

    @pytest.mark.asyncio
    async def test_seeding_injects_pregathered_web(self, controller):
        base_web = WebSearchResult(
            query="q",
            pages=[WebPage(url="https://ex.org/a", title="Alpha Title", content="alpha body", score=0.9)],
        )
        captured = {}
        controller._get_model_decision = _fake_decision_factory(capture=captured)

        await _run(
            controller,
            query="q", system_prompt="sys", model_name="test-model",
            initial_search_terms=[], skip_initial_search=True,
            initial_context={"web_search_results": base_web},
        )

        assert "[WEB_1]" in captured["prompt"]
        assert "Alpha Title" in captured["prompt"]
        assert "WEB_1" in controller._tool_executor._current_web_source_map

        session = controller._last_session
        assert session.seeded_base_web is True

        final_prompt = controller._build_final_prompt(query="q", session=session, initial_context=None)
        assert "[TOOL RESULTS" in final_prompt
        assert "Alpha Title" in final_prompt


# ---------------------------------------------------------------------------
# T8 — seeding does NOT happen when round 1 was itself a web search
# ---------------------------------------------------------------------------

class TestT8NoDoubleSeedOnRealSearch:

    @pytest.mark.asyncio
    async def test_no_seed_when_round1_is_web_search(self, controller):
        round1_result = WebSearchResult(
            query="term",
            pages=[WebPage(url="https://ex.org/r1", title="Round1", content="round1 body", score=0.9)],
        )
        controller._execute_search = AsyncMock(return_value=round1_result)
        controller._compress_results = AsyncMock(return_value="compressed round1 text")
        controller._get_model_decision = _fake_decision_factory()

        base_web = WebSearchResult(
            query="q",
            pages=[WebPage(url="https://ex.org/base", title="Base Title", content="base body", score=0.9)],
        )
        await _run(
            controller,
            query="q", system_prompt="sys", model_name="test-model",
            initial_search_terms=["term"], skip_initial_search=False,
            initial_context={"web_search_results": base_web},
        )

        session = controller._last_session
        assert session.seeded_base_web is False
        assert "Pre-gathered web results" not in session.accumulated_context
        assert "Base Title" not in session.accumulated_context


# ---------------------------------------------------------------------------
# T9 — reuse gate: admitted evidence the decision round never saw blocks reuse
# ---------------------------------------------------------------------------

class TestT9ReuseGate:

    @pytest.mark.asyncio
    async def test_reuse_skipped_when_context_has_unrendered_evidence(self, controller):
        controller._get_model_decision = _fake_decision_factory()
        captured = {"called": False}

        async def fake_final(query, system_prompt, model_name, session, initial_context=None):
            captured["called"] = True
            yield "SYNTHESIZED."

        controller._generate_final_response = fake_final

        events, text = await _run(
            controller,
            query="q", system_prompt="sys", model_name="test-model",
            initial_search_terms=[], skip_initial_search=True,
            initial_context={"memories": [{"content": "m"}]},
        )

        assert captured["called"] is True
        assert "SYNTHESIZED." in text
        session = controller._last_session
        # 2026-09-07 (B1, docs/HANDOFF_20260907_upload_reuse_contracts.md):
        # reuse_skipped_reason now names the specific retrieval-evidence key
        # that blocked reuse ("memories" here) instead of a bare sentence —
        # updated from the old "decision prompt lacked admitted evidence"
        # pin to match; the underlying block-on-unrendered-evidence behavior
        # this test guards is unchanged (still blocks, still full synthesis).
        assert (
            session.reuse_skipped_reason
            == "decision prompt lacked admitted evidence: memories"
        )
        assert session.answer_call == "final_synthesis"

    @pytest.mark.asyncio
    async def test_reuse_fires_with_seeded_web_and_real_hash(self, controller):
        """A3 + A4/B1 (2026-09-06/07): the pre-gathered base web result is the
        ONE retrieval key the decision round genuinely sees — A3 seeding
        renders it verbatim into accumulated_context before round 2 — so it
        must not block decision-answer reuse (B1 blocks on retrieval keys
        the decision prompt only met as a digest; Delegate B's literal
        reading blocked on web_search_results too and undid A3's purpose —
        Fable referee restored the exemption via session.seeded_base_web).
        Hash receipt must be the real decision-prompt hash, never the old
        sentinel.
        """
        base_web = WebSearchResult(
            query="q",
            pages=[WebPage(url="https://ex.org/a", title="Alpha Title", content="alpha body", score=0.9)],
        )
        controller._get_model_decision = _fake_decision_factory()

        async def fake_final(query, system_prompt, model_name, session, initial_context=None):
            yield "SYNTHESIZED."

        controller._generate_final_response = fake_final

        await _run(
            controller,
            query="q", system_prompt="sys", model_name="test-model",
            initial_search_terms=[], skip_initial_search=True,
            initial_context={"web_search_results": base_web},
        )

        session = controller._last_session
        assert session.seeded_base_web is True
        assert session.answer_call == "decision_reuse"
        assert session.reuse_skipped_reason == ""
        assert session.decision_prompt_hash
        assert session.final_prompt_hash == session.decision_prompt_hash
        assert session.final_prompt_hash != "decision-answer-reuse"

    @pytest.mark.asyncio
    async def test_seeded_web_plus_unrendered_memories_still_blocks(self, controller):
        """Seeding exempts ONLY web_search_results; memories that never
        reached the decision prompt still force full synthesis."""
        base_web = WebSearchResult(
            query="q",
            pages=[WebPage(url="https://ex.org/a", title="Alpha Title", content="alpha body", score=0.9)],
        )
        controller._get_model_decision = _fake_decision_factory()

        async def fake_final(query, system_prompt, model_name, session, initial_context=None):
            yield "SYNTHESIZED."

        controller._generate_final_response = fake_final

        await _run(
            controller,
            query="q", system_prompt="sys", model_name="test-model",
            initial_search_terms=[], skip_initial_search=True,
            initial_context={"web_search_results": base_web, "memories": [{"content": "m1"}]},
        )

        session = controller._last_session
        assert session.seeded_base_web is True
        assert session.answer_call == "final_synthesis"
        assert session.reuse_skipped_reason == "decision prompt lacked admitted evidence: memories"


# ---------------------------------------------------------------------------
# T10 — provenance summary + forwarding carries every new receipt field
# ---------------------------------------------------------------------------

class TestT10ProvenanceForwarding:

    def test_provenance_summary_and_forwarding(self):
        session = AgenticSearchSession(query="q")
        session.answer_call = "decision_reuse"
        session.decision_prompt_hash = "abc123def4567890"
        session.visible_sources = {"web_ids": ["WEB_1"], "sections": ["Search Results So Far"]}
        session.omitted_sections = ["graph_context"]
        session.reuse_skipped_reason = ""
        session.seeded_base_web = True

        summary = session.get_provenance_summary()
        for key in (
            "answer_call", "decision_prompt_hash", "visible_sources",
            "omitted_sections", "reuse_skipped_reason", "seeded_base_web",
        ):
            assert key in summary

        orchestrator = SimpleNamespace(agentic_controller=SimpleNamespace(_last_session=session))
        provenance = {}
        _attach_agentic_provenance(provenance, orchestrator)

        assert provenance["answer_call"] == "decision_reuse"
        assert provenance["decision_prompt_hash"] == "abc123def4567890"
        assert provenance["visible_sources"] == {
            "web_ids": ["WEB_1"], "sections": ["Search Results So Far"],
        }
        assert provenance["omitted_sections"] == ["graph_context"]
        assert provenance["reuse_skipped_reason"] == ""
        assert provenance["seeded_base_web"] is True


# ---------------------------------------------------------------------------
# T11 — final prompt's citation instruction/header reflect actual source kind
# ---------------------------------------------------------------------------

_WEB_CITATION_REQUIREMENT = (
    "Every factual claim from web sources MUST include a [WEB_N] citation."
)


class TestT11FinalPromptSourceKindLabel:

    def test_memory_only_loop_lacks_web_citation_requirement(self, controller):
        session = AgenticSearchSession(query="q")
        session.accumulated_context = "MEMORY: some retrieved memory content."

        final_prompt = controller._build_final_prompt(query="q", session=session, initial_context=None)

        assert "[TOOL RESULTS" in final_prompt
        assert _WEB_CITATION_REQUIREMENT not in final_prompt

    def test_populated_web_source_map_gets_citation_requirement(self, controller):
        controller._tool_executor._current_web_source_map = {
            "WEB_1": {"title": "T", "url": "https://x.example.com", "domain": "x.example.com"}
        }
        session = AgenticSearchSession(query="q")
        session.accumulated_context = "[WEB_1] **T** (https://x.example.com)\nbody"

        final_prompt = controller._build_final_prompt(query="q", session=session, initial_context=None)

        assert "[TOOL RESULTS" in final_prompt
        assert _WEB_CITATION_REQUIREMENT in final_prompt
