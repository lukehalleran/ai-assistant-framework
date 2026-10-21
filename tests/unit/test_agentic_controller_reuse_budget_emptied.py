"""A budget-emptied section no longer blocks decision-answer reuse once the
decision prompt carries the evidence block; git commit reads run off the loop.

Incident 2026-10-18 12:03 (live [test] probe after #88, kimi-3, memory recall):
the decision round got a 5,558-token evidence block and answered, but reuse was
refused because the token budget had emptied a 1-item semantic_summaries
section. The final prompt renders the same post-budget initial_context, so the
synthesis call (13,617 prompt tokens) could not see that section either.
Same turn: get_git_commits ran the procedural collection's get_recent
(list_all) on the event loop, 638 ms.

Classes: BC-25 (consumer blind to gathered evidence, the 10-08 rule kept only
where it still protects), BC-41 (sync work on the event loop).
"""
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.agentic.types import SearchProtocol
from core.agentic.controller import AgenticSearchController
from core.agentic.tools import ToolExecutor
from core.agentic.types import AgenticSearchSession
from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin
from tests.unit.test_agentic_controller_decision_evidence import (
    _FrozenDatetime, _NoPending, _answering_model, _mock_tool_executor, _run, rich_context,
)


@pytest.fixture
def controller(monkeypatch):
    manager = MagicMock()
    manager.api_models = {}
    c = AgenticSearchController(model_manager=manager, web_search_manager=MagicMock())
    c._tool_executor = _mock_tool_executor()
    monkeypatch.setattr("core.agentic.controller.datetime", _FrozenDatetime)
    monkeypatch.setattr(ToolExecutor, "_get_pending_actions_store", staticmethod(lambda: _NoPending()))
    # 2026-10-18: reuse needs a reasoned decision round (opt-in, native tools).
    monkeypatch.setattr("core.agentic.controller.app_config.AGENTIC_DECISION_REASONING", True)
    c.detect_protocol = lambda m: SearchProtocol.NATIVE_TOOLS
    return c


def _emptied_semantic_ctx():
    ctx = rich_context()
    ctx["semantic_summaries"] = []
    ctx["_budget_emptied"] = ["semantic_summaries"]
    return ctx


@pytest.mark.asyncio
async def test_emptied_section_does_not_block_reuse_when_block_present(controller):
    prompts, final_called = [], []
    text, c = await _run(controller, _emptied_semantic_ctx(), _answering_model(prompts), final_called)
    assert final_called == []
    assert c._last_session.answer_call == "decision_reuse"


def test_emptied_section_blocks_without_block(controller):
    # every non-empty key counts as rendered, so only the emptied branch decides
    session = AgenticSearchSession(query="q")
    session.decision_rendered_keys = frozenset(rich_context().keys()) - {"semantic_summaries"}
    assert controller._first_unmet_retrieval_key(_emptied_semantic_ctx(), session) == "semantic_summaries"
    session.decision_evidence_block = "[RELEVANT MEMORIES]\nx"
    assert controller._first_unmet_retrieval_key(_emptied_semantic_ctx(), session) is None


@pytest.mark.asyncio
async def test_only_emptied_evidence_still_blocks(controller):
    # nothing rendered -> no block -> the 10-08 rule stands
    ctx = {"memories": [], "_budget_emptied": ["memories"]}
    prompts, final_called = [], []
    await _run(controller, ctx, _answering_model(prompts), final_called)
    assert final_called == [True]


class _G(KnowledgeRetrievalMixin):
    def __init__(self, chroma):
        self.memory_coordinator = SimpleNamespace(chroma_store=chroma)
        self.memory_id_map = {}


@pytest.mark.asyncio
async def test_git_commit_reads_run_off_loop(monkeypatch):
    monkeypatch.setattr("config.app_config.GIT_MEMORY_ENABLED", True)
    seen = {}

    def fake_recent(name, limit=8):
        seen["recent"] = threading.get_ident()
        return [{"id": "c1", "content": "fix: quasar", "metadata": {"commit_hash": "abc1234"}}]

    def fake_query(name, query_text=None, n_results=3, **kw):
        seen["query"] = threading.get_ident()
        return [{"id": "c2", "content": "feat: nebula", "metadata": {"commit_hash": "def5678"}}]

    chroma = SimpleNamespace(collections={"procedural": object()},
                             get_recent=fake_recent, query_collection=fake_query)
    loop_thread = threading.get_ident()
    out = await _G(chroma).get_git_commits("nebula feature", limit=3)
    assert out
    assert seen["recent"] != loop_thread
    assert seen["query"] != loop_thread
