"""
Tool/action wiring parity tests.

These turn "silently dropped tool/action" into a loud red test at commit time. Adding the GitHub
write actions took ~10 debugging rounds because a tool can be wired into one place but not another
and fail by quietly dropping the call. The biggest trap: TWO dispatch routers that must be hand-synced
— ToolExecutor.dispatch_single and the controller's _dispatch_single_inner — which had already drifted
(the controller was missing generate_document / create_daemon_note / lookup_contact / action branches).

If you add a tool or action and one of these fails, you forgot a wiring point — the failure message
tells you which.

Strengthened 2026-09-13: consumption is derived by EVALUATING the real
DISPATCH_TABLE predicates against a one-flag probe (not by grepping router
source), every row's handler is resolved and its argument hand-off is bound
against the real signature, both routers are driven through a sentinel table,
executor references are resolved by the real ActionSpec method, and each
helper has a red control (an unconsumed flag, an unresolvable handler, an
arity mismatch, an unresolvable executor, an action without an executor).
"""
import asyncio
import dataclasses
import inspect
from types import SimpleNamespace

import pytest

import core.agentic.tools as tools_mod
from core.actions.registry import ACTION_SPECS, ActionSpec
from core.actions.types import ActionType, CONFIRMATION_REQUIRED
from core.agentic.controller import AgenticSearchController
from core.agentic.tools import DISPATCH_TABLE, ToolExecutor
from core.agentic.types import PROPOSE_ACTION_TOOL_DEFINITION, SearchDecision


class _Probe:
    """A decision with exactly one wants_* flag set and every payload field present."""

    def __init__(self, flag):
        self._flag = flag

    def __getattr__(self, name):
        if name.startswith("wants_"):
            return name == self._flag
        return "probe"


def declared_tool_flags(decision_cls=SearchDecision) -> set:
    """Every wants_* field that should route to a tool (wants_answer finishes the loop instead)."""
    flags = {f.name for f in dataclasses.fields(decision_cls) if f.name.startswith("wants_")}
    flags.discard("wants_answer")
    return flags


def rows_for(table, flag) -> list:
    return [handler for predicate, handler, _builder in table if predicate(_Probe(flag))]


def unconsumed_flags(flags, table) -> list:
    return sorted(flag for flag in flags if not rows_for(table, flag))


def multiply_routed_flags(flags, table) -> dict:
    return {flag: rows_for(table, flag) for flag in sorted(flags) if len(rows_for(table, flag)) > 1}


def rows_without_a_declared_flag(flags, table) -> list:
    return [handler for predicate, handler, _builder in table
            if not any(predicate(_Probe(flag)) for flag in flags)]


def handoff_problems(table, owner) -> list:
    """Unresolvable handlers and argument hand-offs that the real signature rejects."""
    problems = []
    for predicate, handler_name, builder in table:
        handler = getattr(owner, handler_name, None)
        if not callable(handler):
            problems.append(f"{owner.__name__}.{handler_name} does not resolve to a callable")
            continue
        args = builder(_Probe("wants_probe"), 1, None, None)
        try:
            inspect.signature(handler).bind(None, *args)
        except TypeError as exc:
            problems.append(f"{owner.__name__}.{handler_name} rejects the table's arguments: {exc}")
    return problems


def unresolvable_executors(specs) -> list:
    problems = []
    for key, spec in specs.items():
        try:
            executor = ActionSpec.resolve_executor(spec)
        except (ImportError, AttributeError, ValueError) as exc:
            problems.append(f"{key}: {type(exc).__name__}")
            continue
        if not callable(executor):
            problems.append(f"{key}: executor is not callable")
    return problems


def actions_without_specs(required, specs) -> list:
    return sorted({action.name for action in required} - {action.name for action in specs})


# ---------------------------------------------------------------------------
# Router parity — the dual-router drift trap
# ---------------------------------------------------------------------------
class TestRouterParity:
    def test_every_declared_tool_flag_is_consumed_by_a_dispatch_row(self):
        """Every SearchDecision tool flag must have a DISPATCH_TABLE row — otherwise that tool's
        call is silently dropped. (This is the trap that broke propose_action + 3 others.)"""
        missing = unconsumed_flags(declared_tool_flags(), DISPATCH_TABLE)
        assert not missing, (
            f"DISPATCH_TABLE has no row for: {missing} — those tool calls would be "
            "silently dropped. Add a row in core/agentic/tools.py:DISPATCH_TABLE."
        )

    def test_no_flag_is_claimed_by_two_rows(self):
        assert multiply_routed_flags(declared_tool_flags(), DISPATCH_TABLE) == {}, (
            "A flag matched by two DISPATCH_TABLE rows is silently served by the first one."
        )

    def test_every_dispatch_row_serves_a_declared_flag(self):
        assert rows_without_a_declared_flag(declared_tool_flags(), DISPATCH_TABLE) == [], (
            "A DISPATCH_TABLE row whose predicate matches no SearchDecision flag is dead wiring."
        )

    def test_every_row_hands_off_to_a_real_executor_handler(self):
        problems = handoff_problems(DISPATCH_TABLE, ToolExecutor)
        assert not problems, "\n".join(problems)

    def test_every_row_resolves_on_the_controller_or_its_executor(self):
        for _predicate, handler_name, _builder in DISPATCH_TABLE:
            handler = getattr(AgenticSearchController, handler_name, None) or getattr(ToolExecutor, handler_name, None)
            assert callable(handler), f"controller cannot resolve {handler_name}"

    @pytest.mark.parametrize("router", ["executor", "controller", "controller-delegates"])
    def test_both_routers_dispatch_through_the_shared_table(self, monkeypatch, router):
        """Drive each router with a sentinel table: a router that stops reading DISPATCH_TABLE
        cannot call the sentinel."""
        calls = []

        async def sentinel(decision, round_number):
            calls.append((decision, round_number))
            return "routed"

        monkeypatch.setattr(
            tools_mod, "DISPATCH_TABLE",
            [(lambda d: True, "_sentinel_handler", lambda d, rn, cl, ss: (d, rn))],
        )
        decision = SearchDecision()
        executor = ToolExecutor.__new__(ToolExecutor)
        if router == "executor":
            executor._sentinel_handler = sentinel
            result = asyncio.run(executor.dispatch_single(decision, 3, None, None, None))
        else:
            controller = AgenticSearchController.__new__(AgenticSearchController)
            controller._tool_executor = executor
            owner = executor if router == "controller-delegates" else controller
            owner._sentinel_handler = sentinel
            result = asyncio.run(controller._dispatch_single_inner(decision, 3, None, None, None))
        assert result == "routed"
        assert calls == [(decision, 3)]


class TestRouterHelperControls:
    def test_new_flag_without_a_row_is_reported(self):
        flags = declared_tool_flags() | {"wants_new_tool"}
        assert unconsumed_flags(flags, DISPATCH_TABLE) == ["wants_new_tool"]

    def test_removed_row_is_reported(self):
        table = [row for row in DISPATCH_TABLE if row[1] != "_dispatch_web_search"]
        assert "wants_search" in unconsumed_flags(declared_tool_flags(), table)

    def test_unresolvable_handler_is_reported(self):
        table = list(DISPATCH_TABLE) + [(lambda d: d.wants_search, "_dispatch_does_not_exist", tools_mod._args_basic)]
        assert any("_dispatch_does_not_exist" in p for p in handoff_problems(table, ToolExecutor))

    def test_argument_hand_off_mismatch_is_reported(self):
        wrong = [(lambda d: d.wants_wolfram, "_dispatch_wolfram", lambda d, rn, cl, ss: (d, rn, cl, ss, "extra"))]
        assert any("rejects the table's arguments" in p for p in handoff_problems(wrong, ToolExecutor))

    def test_row_for_a_removed_flag_is_reported(self):
        table = list(DISPATCH_TABLE) + [(lambda d: d.wants_retired_tool, "_dispatch_web_search", tools_mod._args_basic)]
        assert rows_without_a_declared_flag(declared_tool_flags(), table) == ["_dispatch_web_search"]


# ---------------------------------------------------------------------------
# Action coverage — every write action must have an executor
# ---------------------------------------------------------------------------
class TestActionCoverage:
    def test_every_confirmation_action_has_executor(self):
        missing = actions_without_specs(CONFIRMATION_REQUIRED, ACTION_SPECS)
        assert not missing, (
            f"These confirmation-required actions have no executor wired: {missing}. "
            "An approved proposal of this type would fail at execution."
        )

    def test_propose_action_enum_are_valid_actiontypes_with_executors(self):
        enum_vals = (
            PROPOSE_ACTION_TOOL_DEFINITION["function"]["parameters"]
            ["properties"]["action_type"]["enum"]
        )
        keys = {at.name for at in ACTION_SPECS}
        for val in enum_vals:
            at = ActionType(val)  # raises if the schema advertises an unknown action
            assert at.name in keys, (
                f"propose_action advertises '{val}' but it has no executor — the model can "
                f"propose it and the user can approve it, but execution will fail."
            )

    def test_registry_matches_confirmation_required(self):
        """ACTION_SPECS (executors/parse/health/detection source of truth) must not drift from
        the confirmation-required set."""
        assert set(ACTION_SPECS) == set(CONFIRMATION_REQUIRED), (
            f"ACTION_SPECS {set(a.name for a in ACTION_SPECS)} vs "
            f"CONFIRMATION_REQUIRED {set(a.name for a in CONFIRMATION_REQUIRED)} have drifted."
        )

    def test_every_spec_resolves_a_callable_executor(self):
        problems = unresolvable_executors(ACTION_SPECS)
        assert not problems, f"executor_ref does not resolve: {problems}"

    def test_control_unresolvable_executor_module_is_reported(self):
        specs = {"fake": SimpleNamespace(executor_ref="core.actions.does_not_exist:run")}
        assert unresolvable_executors(specs) == ["fake: ModuleNotFoundError"]

    def test_control_unresolvable_executor_function_is_reported(self):
        specs = {"fake": SimpleNamespace(executor_ref="core.actions.registry:no_such_executor")}
        assert unresolvable_executors(specs) == ["fake: AttributeError"]

    def test_control_registered_action_without_an_executor_is_reported(self):
        required = list(CONFIRMATION_REQUIRED) + [SimpleNamespace(name="NEW_WRITE_ACTION")]
        assert actions_without_specs(required, ACTION_SPECS) == ["NEW_WRITE_ACTION"]
