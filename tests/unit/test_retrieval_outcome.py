"""Tests for the leaf module utils/retrieval_outcome.py.

Drives the deployed utils.retrieval_outcome module directly. See
docs/execution/generalization/failure_outcome_design.md, "F1 Ready decision:
one status vocabulary", for the design this pins: F1 defines the shared
RETRIEVAL_STATES vocabulary once, reusing core.insight.coordinator's
CHANNEL_STATES names, without editing that module (pinned by a parity test
instead).
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from utils.retrieval_outcome import (
    OutcomeList,
    RETRIEVAL_STATES,
    RetrievalError,
    outcome_status,
)

MODULE_PATH = pathlib.Path(__file__).resolve().parents[2] / "utils" / "retrieval_outcome.py"

# Leaf-guard allow-list: only what utils/retrieval_outcome.py itself may import.
_STDLIB_ALLOWLIST = frozenset({"__future__"})
_DISALLOWED_PREFIXES = ("core", "memory", "knowledge", "gui", "api", "models", "utils")


def _top_level_import_names(source: str) -> list[str]:
    """Top-level (module-body) Import/ImportFrom names, one per imported module."""
    tree = ast.parse(source)
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module.split(".")[0])
    return names


def _disallowed_imports(source: str) -> list[str]:
    """Names from _top_level_import_names() that break the leaf contract:
    not on the stdlib allow-list, or a project package (core/memory/
    knowledge/gui/api/models/utils)."""
    return [
        name
        for name in _top_level_import_names(source)
        if name not in _STDLIB_ALLOWLIST or name.startswith(_DISALLOWED_PREFIXES)
    ]


# --- RETRIEVAL_STATES -------------------------------------------------------


def test_states_are_the_four_reused_names():
    assert RETRIEVAL_STATES == frozenset({"succeeded", "no_results", "unavailable", "failed"})


def test_parity_with_insight_channel_states():
    from core.insight.coordinator import CHANNEL_STATES

    assert RETRIEVAL_STATES <= CHANNEL_STATES


# --- OutcomeList: derived status ---------------------------------------------


def test_empty_derives_no_results():
    ol = OutcomeList()
    assert ol.status == "no_results"
    assert ol.reason == ""
    assert ol == []


def test_nonempty_derives_succeeded():
    ol = OutcomeList([1, 2, 3])
    assert ol.status == "succeeded"
    assert ol == [1, 2, 3]


# --- OutcomeList: explicit status --------------------------------------------


def test_failed_without_items():
    ol = OutcomeList(status="failed", reason="ConnectionError")
    assert ol.status == "failed"
    assert ol.reason == "ConnectionError"
    assert ol == []


def test_failed_with_items_is_a_partial_read():
    ol = OutcomeList([1, 2], status="failed", reason="timeout")
    assert ol.status == "failed"
    assert ol == [1, 2]


def test_unavailable_without_items():
    ol = OutcomeList(status="unavailable", reason="not_configured")
    assert ol.status == "unavailable"
    assert ol == []


def test_unavailable_with_items_is_a_partial_read():
    ol = OutcomeList([1], status="unavailable", reason="timeout")
    assert ol.status == "unavailable"
    assert ol == [1]


def test_explicit_succeeded_with_items():
    ol = OutcomeList([1], status="succeeded")
    assert ol.status == "succeeded"


# --- OutcomeList: ValueErrors -------------------------------------------------


def test_unknown_status_raises_value_error():
    with pytest.raises(ValueError):
        OutcomeList(status="bogus")


def test_nonempty_no_results_raises_value_error():
    with pytest.raises(ValueError):
        OutcomeList([1], status="no_results")


# --- OutcomeList: classmethods ------------------------------------------------


def test_failed_classmethod():
    ol = OutcomeList.failed("ConnectionError")
    assert ol.status == "failed"
    assert ol.reason == "ConnectionError"
    assert ol == []


def test_failed_classmethod_with_items():
    ol = OutcomeList.failed("timeout", items=[1, 2])
    assert ol.status == "failed"
    assert ol == [1, 2]


def test_unavailable_classmethod():
    ol = OutcomeList.unavailable("not_configured")
    assert ol.status == "unavailable"
    assert ol == []


# --- OutcomeList: equality/truthiness stay plain-list -------------------------


def test_failed_empty_equals_plain_list_and_is_falsy():
    ol = OutcomeList.failed("x")
    assert ol == []
    assert not ol
    assert not bool(ol)


def test_succeeded_nonempty_is_truthy():
    assert OutcomeList([1])


def test_outcome_list_isinstance_list():
    assert isinstance(OutcomeList.failed("x"), list)


# --- OutcomeList: transforms drop status (documented) --------------------------


def test_slice_drops_status():
    ol = OutcomeList([1, 2, 3], status="failed", reason="x")
    sliced = ol[1:]
    assert type(sliced) is list
    assert not hasattr(sliced, "status")
    assert sliced == [2, 3]


def test_list_call_drops_status():
    ol = OutcomeList([1, 2], status="unavailable", reason="x")
    plain = list(ol)
    assert type(plain) is list
    assert not hasattr(plain, "status")


def test_concat_drops_status():
    ol = OutcomeList([1], status="failed", reason="x")
    combined = ol + [2]
    assert type(combined) is list
    assert not hasattr(combined, "status")
    assert combined == [1, 2]


def test_copy_drops_status():
    ol = OutcomeList([1], status="failed", reason="x")
    copied = ol.copy()
    assert type(copied) is list
    assert not hasattr(copied, "status")


# --- RetrievalError -------------------------------------------------------------


def test_retrieval_error_fields():
    err = RetrievalError(source="tavily", reason="ConnectionError")
    assert err.source == "tavily"
    assert err.reason == "ConnectionError"


def test_retrieval_error_str_includes_both():
    text = str(RetrievalError(source="tavily", reason="ConnectionError"))
    assert "tavily" in text
    assert "ConnectionError" in text


def test_retrieval_error_is_runtime_error():
    assert isinstance(RetrievalError(source="x", reason="y"), RuntimeError)


def test_retrieval_error_raise_and_catch_as_exception():
    with pytest.raises(Exception):
        raise RetrievalError(source="x", reason="y")


def test_retrieval_error_source_and_reason_are_keyword_only():
    with pytest.raises(TypeError):
        RetrievalError("x", "y")  # type: ignore[misc]


# --- outcome_status ---------------------------------------------------------------


def test_outcome_status_outcome_list_succeeded():
    assert outcome_status(OutcomeList([1])) == ("succeeded", "")


def test_outcome_status_outcome_list_no_results():
    assert outcome_status(OutcomeList()) == ("no_results", "")


def test_outcome_status_outcome_list_unavailable():
    assert outcome_status(OutcomeList.unavailable("timeout")) == ("unavailable", "timeout")


def test_outcome_status_outcome_list_failed():
    assert outcome_status(OutcomeList.failed("ConnectionError")) == ("failed", "ConnectionError")


def test_outcome_status_plain_empty_list():
    assert outcome_status([]) == ("no_results", "")


def test_outcome_status_plain_nonempty_list():
    assert outcome_status([1, 2]) == ("succeeded", "")


def test_outcome_status_none():
    assert outcome_status(None) == ("no_results", "")


def test_outcome_status_empty_string():
    assert outcome_status("") == ("no_results", "")


# --- Leaf-module import guard -------------------------------------------------------


def test_module_imports_only_stdlib():
    source = MODULE_PATH.read_text()
    assert _disallowed_imports(source) == []


def test_module_has_at_least_one_import_to_exercise_the_guard():
    source = MODULE_PATH.read_text()
    assert _top_level_import_names(source), "expected at least one top-level import"


def test_negative_control_flags_project_import():
    bad_source = "import core.x\n\nVALUE = 1\n"
    assert _disallowed_imports(bad_source) == ["core"]


def test_negative_control_flags_non_stdlib_from_import():
    bad_source = "from knowledge.web_search_manager import FetchedPages\n"
    assert _disallowed_imports(bad_source) == ["knowledge"]


def test_negative_control_flags_utils_import():
    bad_source = "from utils.logging_utils import get_logger\n"
    assert _disallowed_imports(bad_source) == ["utils"]
