"""
Tool/action wiring parity tests.

These turn "silently dropped tool/action" into a loud red test at commit time. Adding the GitHub
write actions took ~10 debugging rounds because a tool can be wired into one place but not another
and fail by quietly dropping the call. The biggest trap: TWO dispatch routers that must be hand-synced
— ToolExecutor.dispatch_single and the controller's _dispatch_single_inner — which had already drifted
(the controller was missing generate_document / create_daemon_note / lookup_contact / action branches).

If you add a tool or action and one of these fails, you forgot a wiring point — the failure message
tells you which.
"""
import inspect
import re

import pytest

from core.agentic.tools import ToolExecutor
from core.agentic.controller import AgenticSearchController
from core.actions.executors import ActionExecutorRegistry
from core.actions.types import ActionType, CONFIRMATION_REQUIRED
from core.agentic.types import PROPOSE_ACTION_TOOL_DEFINITION


def _dispatch_table_flags() -> set:
    """The wants_* flags the shared DISPATCH_TABLE routes on (table lambdas reference d.wants_*)."""
    import core.agentic.tools as tools_mod
    return set(re.findall(r'\bd\.(wants_\w+)', inspect.getsource(tools_mod)))


def _searchdecision_dispatch_flags() -> set:
    """Every wants_* field on SearchDecision that should route to a tool (excludes wants_answer,
    which the loop handles as 'finish', not a dispatch)."""
    import dataclasses
    from core.agentic.types import SearchDecision
    flags = {f.name for f in dataclasses.fields(SearchDecision) if f.name.startswith("wants_")}
    flags.discard("wants_answer")
    return flags


def _executor_action_keys() -> set:
    """ActionType names that have a registered executor (the ACTION_SPECS source of truth)."""
    from core.actions.registry import ACTION_SPECS
    return {at.name for at in ACTION_SPECS}


# ---------------------------------------------------------------------------
# Router parity — the dual-router drift trap
# ---------------------------------------------------------------------------
class TestRouterParity:
    def test_dispatch_table_covers_every_tool_flag(self):
        """Every SearchDecision tool flag must have a DISPATCH_TABLE row — otherwise that tool's
        call is silently dropped. (This is the trap that broke propose_action + 3 others.)"""
        decl = _searchdecision_dispatch_flags()
        table = _dispatch_table_flags()
        missing = decl - table
        assert not missing, (
            f"DISPATCH_TABLE has no row for: {sorted(missing)} — those tool calls would be "
            "silently dropped. Add a row in core/agentic/tools.py:DISPATCH_TABLE."
        )

    def test_both_routers_iterate_the_shared_table(self):
        """Both routers must drive off DISPATCH_TABLE so they can never drift apart again."""
        te_src = inspect.getsource(ToolExecutor.dispatch_single)
        ctrl_src = inspect.getsource(AgenticSearchController._dispatch_single_inner)
        assert "DISPATCH_TABLE" in te_src, "ToolExecutor.dispatch_single no longer uses the shared table"
        assert "DISPATCH_TABLE" in ctrl_src, "controller _dispatch_single_inner no longer uses the shared table"


# ---------------------------------------------------------------------------
# Action coverage — every write action must have an executor
# ---------------------------------------------------------------------------
class TestActionCoverage:
    def test_every_confirmation_action_has_executor(self):
        keys = _executor_action_keys()
        missing = {a.name for a in CONFIRMATION_REQUIRED} - keys
        assert not missing, (
            f"These confirmation-required actions have no executor wired: {sorted(missing)}. "
            "An approved proposal of this type would fail at execution."
        )

    def test_propose_action_enum_are_valid_actiontypes_with_executors(self):
        enum_vals = (
            PROPOSE_ACTION_TOOL_DEFINITION["function"]["parameters"]
            ["properties"]["action_type"]["enum"]
        )
        keys = _executor_action_keys()
        for val in enum_vals:
            at = ActionType(val)  # raises if the schema advertises an unknown action
            assert at.name in keys, (
                f"propose_action advertises '{val}' but it has no executor — the model can "
                f"propose it and the user can approve it, but execution will fail."
            )

    def test_registry_matches_confirmation_required(self):
        """ACTION_SPECS (executors/parse/health/detection source of truth) must not drift from
        the confirmation-required set."""
        from core.actions.registry import ACTION_SPECS
        assert set(ACTION_SPECS) == set(CONFIRMATION_REQUIRED), (
            f"ACTION_SPECS {set(a.name for a in ACTION_SPECS)} vs "
            f"CONFIRMATION_REQUIRED {set(a.name for a in CONFIRMATION_REQUIRED)} have drifted."
        )

    def test_every_spec_resolves_a_callable_executor(self):
        from core.actions.registry import ACTION_SPECS
        for at, spec in ACTION_SPECS.items():
            assert callable(spec.resolve_executor()), f"{at.value} spec executor_ref does not resolve"
