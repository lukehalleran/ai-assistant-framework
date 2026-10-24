"""Self-identity rule: the assistant is the project that shares its name.

Incident 2026-10-24 12:13: asked what figures the owner had given lately, the
reply said "~50k lines added to <Name> over 8 days" — the assistant spoke of
its own codebase as a third-party project. Fixes: a static, name-agnostic rule
in the cached prefix (UNIVERSAL_GROUNDING) and an own-codebase scope line on
[PROJECT COMMIT HISTORY].

class: BC-91-adjacent identity/speaker confusion (no new class drafted here).
"""
from unittest.mock import AsyncMock, Mock

import pytest

from config.app_config import PROMPT_CACHE_BREAKPOINT
from core.context_pipeline import ContextResult, ToneLevel
from core.prompt.formatter import PromptFormatter
from core.response_guidance import UNIVERSAL_GROUNDING

RULE_MARK = "it is you"


def _orch():
    from core.orchestrator import DaemonOrchestrator
    mm = Mock()
    mm.get_active_model_name = Mock(return_value="m")
    mm.supports_reasoning = Mock(return_value=False)
    rg = Mock()
    rg.time_manager = None
    ms = Mock()
    ms.corpus_manager = Mock()
    pb = Mock()
    pb.build_prompt = AsyncMock(return_value="x")
    return DaemonOrchestrator(model_manager=mm, response_generator=rg,
                              file_processor=Mock(), prompt_builder=pb, memory_system=ms)


def _ctx(query):
    return ContextResult(processed_query=query, original_query=query,
                         tone_level=ToneLevel.CONVERSATIONAL, tone_instructions="",
                         primary_topic="general")


def test_rule_text_is_name_agnostic_and_static():
    assert RULE_MARK in " ".join(UNIVERSAL_GROUNDING.split())
    assert "Daemon" not in UNIVERSAL_GROUNDING
    assert "owner" not in UNIVERSAL_GROUNDING.lower()


@pytest.mark.parametrize("wrapped", [False, True])
def test_rule_above_cache_breakpoint_and_prefix_stable(wrapped):
    o = _orch()
    q1 = "what figures have I given you lately?"
    q2 = "tell me about the weather in Lisbon"
    if wrapped:
        q1 = "    " + q1.replace(" ", "\n    ", 3)
    p1 = o._build_system_prompt(_ctx(q1))
    p2 = o._build_system_prompt(_ctx(q2))
    for p in (p1, p2):
        assert PROMPT_CACHE_BREAKPOINT in p
        head = p.split(PROMPT_CACHE_BREAKPOINT)[0]
        assert RULE_MARK in " ".join(head.split())
    assert p1.split(PROMPT_CACHE_BREAKPOINT)[0] == p2.split(PROMPT_CACHE_BREAKPOINT)[0]


def test_rule_present_with_reference_docs_in_context():
    # The prefix is built independently of prompt_ctx, so a reference-docs turn
    # and a plain turn share the rule; the docs-gated tail must not be the only
    # carrier of self-identity.
    from core.prompt.section_instructions import conditional_instruction_tail
    o = _orch()
    p = o._build_system_prompt(_ctx("how does your memory scoring work?"))
    assert RULE_MARK in " ".join(p.split(PROMPT_CACHE_BREAKPOINT)[0].split())
    tail = conditional_instruction_tail({"reference_docs": [{"x": 1}], "user_query": "how do you score?"})
    assert "Self-Knowledge" in tail
    assert RULE_MARK not in " ".join(tail.split())


def _commit_prompt(name, monkeypatch):
    monkeypatch.setattr("config.app_config.ASSISTANT_NAME", name)
    f = PromptFormatter(token_manager=Mock(), time_manager=None)
    commit = {"content": "feat: add ~50k lines", "metadata": {"commit_hash": "abc1234", "author": "Rowan"}}
    return f._assemble_prompt(context={"git_commits": [commit]}, user_input="hi")


@pytest.mark.parametrize("name", ["Daemon", "Tamsin"])
def test_commit_history_scope_names_own_codebase(name, monkeypatch):
    prompt = _commit_prompt(name, monkeypatch)
    assert "[PROJECT COMMIT HISTORY] n=1" in prompt
    section = prompt.split("[PROJECT COMMIT HISTORY]")[1]
    assert f"your own codebase ({name})" in section.split("1) ")[0]
