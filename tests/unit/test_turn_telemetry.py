"""Tests for utils/turn_telemetry.py — per-turn JSONL telemetry."""

import json
import os

import pytest
from unittest.mock import patch

from utils.turn_telemetry import record_turn, _sanitize_value
from core.intent_classifier import IntentType


@pytest.fixture
def telemetry_path(tmp_path):
    path = str(tmp_path / "turns" / "turn_records.jsonl")
    with patch("config.app_config.TURN_TELEMETRY_ENABLED", True), \
         patch("config.app_config.TURN_TELEMETRY_PATH", path):
        yield path


class TestRecordTurn:

    def test_writes_one_json_line(self, telemetry_path):
        assert record_turn({"query": "hello", "intent": "casual_social"})
        lines = open(telemetry_path).read().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["query"] == "hello"
        assert rec["intent"] == "casual_social"
        assert "ts" in rec

    def test_appends_across_calls(self, telemetry_path):
        record_turn({"n": 1})
        record_turn({"n": 2})
        lines = open(telemetry_path).read().splitlines()
        assert [json.loads(l)["n"] for l in lines] == [1, 2]

    def test_creates_parent_dir(self, telemetry_path):
        assert not os.path.exists(os.path.dirname(telemetry_path))
        assert record_turn({"a": 1})
        assert os.path.exists(telemetry_path)

    def test_truncates_long_strings(self, telemetry_path):
        record_turn({"query": "x" * 5000})
        rec = json.loads(open(telemetry_path).read())
        assert len(rec["query"]) == 300

    def test_enum_values_serialized(self, telemetry_path):
        record_turn({"intent": IntentType.TECHNICAL_HELP,
                     "modes": ["memory", "web_search"]})
        rec = json.loads(open(telemetry_path).read())
        assert rec["intent"] == "technical_help"
        assert rec["modes"] == ["memory", "web_search"]

    def test_disabled_writes_nothing(self, tmp_path):
        path = str(tmp_path / "off.jsonl")
        with patch("config.app_config.TURN_TELEMETRY_ENABLED", False), \
             patch("config.app_config.TURN_TELEMETRY_PATH", path):
            assert record_turn({"a": 1}) is False
        assert not os.path.exists(path)

    def test_never_raises_on_bad_path(self):
        with patch("config.app_config.TURN_TELEMETRY_ENABLED", True), \
             patch("config.app_config.TURN_TELEMETRY_PATH", "/proc/definitely/not/writable.jsonl"):
            assert record_turn({"a": 1}) is False  # returns False, no exception

    def test_none_record_ok(self, telemetry_path):
        assert record_turn(None)
        rec = json.loads(open(telemetry_path).read())
        assert "ts" in rec


class TestSanitizeValue:

    def test_scalars_pass_through(self):
        assert _sanitize_value(3) == 3
        assert _sanitize_value(0.5) == 0.5
        assert _sanitize_value(True) is True
        assert _sanitize_value(None) is None

    def test_nested_dict_and_set(self):
        out = _sanitize_value({"a": {IntentType.GENERAL}, "b": (1, 2)})
        assert out["a"] == ["general"]
        assert out["b"] == [1, 2]

    def test_arbitrary_object_becomes_string(self):
        class Weird:
            def __str__(self):
                return "weird"
        assert _sanitize_value(Weird()) == "weird"

    def test_magicmock_terminates(self):
        # Regression (2026-07-03): the enum fallback used to recurse on
        # .value — a MagicMock's .value is another mock, infinitely, which
        # allocated gigabytes and OOM-killed the test process. Sanitizing a
        # mock must terminate immediately with a string.
        from unittest.mock import MagicMock
        out = _sanitize_value(MagicMock())
        assert isinstance(out, str)
        assert len(out) <= 300

    def test_object_with_nonscalar_value_attr_terminates(self):
        class Node:
            def __init__(self):
                self.value = self  # self-referential .value chain
            def __str__(self):
                return "node"
        assert _sanitize_value(Node()) == "node"

    def test_deep_container_nesting_capped(self):
        deep = ["x"]
        for _ in range(50):
            deep = [deep]
        out = _sanitize_value(deep)  # must not RecursionError
        assert isinstance(out, list)
