"""Per-round scripted driver for the DEPLOYED AgenticSearchController loop.

Purpose (oct27c SRC2, BC-63): a test that greps the source of
``AgenticSearchController.run_agentic_search`` for a name proves only that the
name is still spelled there. This helper runs the real loop with a scripted
model: each decision round returns the decisions the test scripted, every
tool dispatch is a recording fake, and the final synthesis is a recording
fake, so a test can assert what each ROUND actually received (prompt, system
prompt, forced tool choice, narrowed tool list, forced action type) and what
the loop dispatched.

Seam: the three controller methods that talk to a model or a tool
(``_get_model_decision``, ``_dispatch_single``, ``_generate_final_response``)
are replaced on the INSTANCE; everything between them (force arming, retry,
deterministic fallbacks, grounding, backfill, accumulation) is the deployed
code.

Usage::

    run = await run_scripted_loop(
        monkeypatch, query="...",
        rounds=[[SearchDecision(is_done=True)], ...],
        protocol="native",           # or "xml"
    )
    run.rounds[0].forced_action_type
    run.dispatched[0].action_type
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from unittest.mock import MagicMock

from core.agentic.controller import AgenticSearchController
from core.agentic.types import SearchDecision, SearchProtocol, _ToolResult

RoundScript = Union[List[SearchDecision], Callable[["RoundRecord"], List[SearchDecision]]]


@dataclass
class RoundRecord:
    """What one decision round received."""
    index: int
    prompt: str
    system_prompt: str
    tool_choice: Any
    tools_override: Optional[List[Dict]]
    forced_action_type: Optional[str]
    model_name: str


@dataclass
class DispatchRecord:
    decision: SearchDecision
    round_number: int
    context_at_dispatch: str
    sandbox_session: Any = None


@dataclass
class LoopRun:
    controller: AgenticSearchController
    events: List[Any] = field(default_factory=list)
    rounds: List[RoundRecord] = field(default_factory=list)
    dispatched: List[DispatchRecord] = field(default_factory=list)
    final_calls: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(e for e in self.events if isinstance(e, str))

    @property
    def session(self):
        return self.controller._last_session

    @property
    def dispatched_decisions(self) -> List[SearchDecision]:
        return [d.decision for d in self.dispatched]


def make_controller(*, max_rounds: int = 4) -> AgenticSearchController:
    manager = MagicMock()
    manager.api_models = {}
    ctrl = AgenticSearchController(
        model_manager=manager, web_search_manager=MagicMock(), max_rounds=max_rounds,
    )
    return ctrl


async def run_scripted_loop(
    monkeypatch,
    *,
    query: str,
    rounds: Sequence[RoundScript],
    protocol: str = "native",
    controller: Optional[AgenticSearchController] = None,
    configured_actions: Sequence[str] = ("calendar_create_event", "calendar_update_event",
                                         "calendar_delete_event"),
    dispatch_context: Union[str, Callable[[SearchDecision], str]] = "",
    final_text: str = "FINAL.",
    max_rounds: int = 4,
    real_dispatch: bool = False,
    **run_kwargs: Any,
) -> LoopRun:
    """Drive ``run_agentic_search`` to completion with a per-round script.

    ``rounds[i]`` is the decision list (or a callable of the RoundRecord) the
    model returns on decision round i; once the script is exhausted the model
    signals done. ``real_dispatch=True`` leaves ``_dispatch_single`` (and so the
    ``_dispatch_single_inner`` chokepoint + DISPATCH_TABLE routing) deployed; the
    caller then patches the individual ``_dispatch_*`` handlers it cares about.
    ``dispatch_context`` is the formatted_context every faked
    tool dispatch returns (str or callable of the decision).
    ``run_kwargs`` go to run_agentic_search (forced_action, action_query_ws,
    tool_continuation, initial_context, ...).
    """
    ctrl = controller or make_controller(max_rounds=max_rounds)
    proto = SearchProtocol.NATIVE_TOOLS if protocol == "native" else SearchProtocol.XML_MARKERS
    ctrl.detect_protocol = lambda m: proto
    # the action offer is read at call time from these two live names
    monkeypatch.setattr("core.agentic.controller.app_config.INTERNET_ACTIONS_ENABLED", True)
    monkeypatch.setattr("core.actions.registry.configured_action_types",
                        lambda: list(configured_actions))

    run = LoopRun(controller=ctrl)
    script = list(rounds)

    async def fake_decision(prompt, system_prompt, model_name, handler, session,
                            tool_choice="auto", tools_override=None,
                            forced_action_type=None):
        rec = RoundRecord(
            index=len(run.rounds), prompt=prompt, system_prompt=system_prompt,
            tool_choice=tool_choice, tools_override=tools_override,
            forced_action_type=forced_action_type, model_name=model_name,
        )
        run.rounds.append(rec)
        if rec.index >= len(script):
            return [SearchDecision(is_done=True)]
        step = script[rec.index]
        return list(step(rec) if callable(step) else step)

    async def fake_dispatch(decision, round_number, session, crisis_level, sandbox_session):
        run.dispatched.append(DispatchRecord(
            decision=decision, round_number=round_number,
            context_at_dispatch=session.accumulated_context or "",
            sandbox_session=sandbox_session))
        ctx = dispatch_context(decision) if callable(dispatch_context) else dispatch_context
        return _ToolResult(decision=decision, round_data=None, formatted_context=ctx,
                           start_events=[], end_events=[])

    async def fake_final(query, system_prompt, model_name, session, initial_context=None):
        run.final_calls.append({"query": query, "session": session})
        yield final_text

    monkeypatch.setattr(ctrl, "_get_model_decision", fake_decision)
    if not real_dispatch:
        monkeypatch.setattr(ctrl, "_dispatch_single", fake_dispatch)
    monkeypatch.setattr(ctrl, "_generate_final_response", fake_final)

    async for ev in ctrl.run_agentic_search(
        query=query, system_prompt="sys", model_name="test-model",
        initial_search_terms=[], skip_initial_search=True, **run_kwargs,
    ):
        run.events.append(ev)
    return run

