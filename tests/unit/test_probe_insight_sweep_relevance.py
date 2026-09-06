"""Unit tests for scripts/probe_insight_sweep_relevance.py's pure helpers.

Exercises only record filtering, row/summary building, and the redaction
flag against synthetic inputs — no stores, no LLM, no orchestrator. Also
asserts (source-level) that the script never references a store-mutating
method, since it is documented as read-only.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "probe_insight_sweep_relevance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("probe_insight_sweep_relevance", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _load_module()


def _item(**kwargs):
    defaults = dict(
        doc_id="d1", text="hello world", date="2026-09-01", collection="conversations",
        speaker="user", stance_label="user-stated", facet="work",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# --- Source-level safety guard ------------------------------------------

def test_script_never_references_store_write_methods():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    forbidden = ["add_conversation_memory", "store_interaction", ".save(", "upsert", "add_entry"]
    for token in forbidden:
        assert token not in source, f"probe script references a store-write method: {token!r}"


# --- is_insight_record ----------------------------------------------------

def test_is_insight_record_matches_mode_prefix(probe):
    assert probe.is_insight_record({"mode": "insight-assembly"}) is True
    assert probe.is_insight_record({"mode": "insight"}) is True


def test_is_insight_record_matches_gate_reason(probe):
    assert probe.is_insight_record({"gate_reason": "insight-mode: pattern_temporal"}) is True


def test_is_insight_record_rejects_unrelated_records(probe):
    assert probe.is_insight_record({"mode": "enhanced", "gate_reason": "general"}) is False
    assert probe.is_insight_record({"mode": "agentic-search"}) is False
    assert probe.is_insight_record("not-a-dict") is False
    assert probe.is_insight_record({}) is False


# --- load_insight_queries --------------------------------------------------

def test_load_insight_queries_filters_and_limits(tmp_path, probe):
    path = tmp_path / "turn_records.jsonl"
    rows = [
        {"mode": "enhanced", "query": "unrelated"},
        {"mode": "insight-assembly", "query": "q1"},
        {"gate_reason": "insight-mode: pattern_temporal", "query": "q2"},
        {"mode": "insight-assembly", "query": "q3"},
        "not json{{{",
        {"mode": "insight-assembly", "query": "q4"},
    ]
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write((json.dumps(r) if isinstance(r, dict) else r) + "\n")

    all_matches = probe.load_insight_queries(path, last_n=-1)
    assert [r["query"] for r in all_matches] == ["q1", "q2", "q3", "q4"]

    limited = probe.load_insight_queries(path, last_n=2)
    assert [r["query"] for r in limited] == ["q3", "q4"]


def test_load_insight_queries_missing_file_returns_empty(tmp_path, probe):
    assert probe.load_insight_queries(tmp_path / "nope.jsonl", 10) == []


# --- build_evidence_row / summarize_items / build_summary_row -------------

def test_build_evidence_row_shape(probe):
    item = _item(text="a" * 300)
    row = probe.build_evidence_row(
        query="how has work been going",
        facet_name="work",
        facet_keywords=["work", "job"],
        item=item,
        rank=3,
        redact=False,
    )
    assert row["query"] == "how has work been going"
    assert row["facet"] == "work"
    assert row["facet_keywords"] == ["work", "job"]
    assert row["collection"] == "conversations"
    assert row["stance_label"] == "user-stated"
    assert row["speaker"] == "user"
    assert row["date"] == "2026-09-01"
    assert row["doc_id"] == "d1"
    assert row["text_head"] == "a" * 200
    assert row["rank"] == 3
    assert row["grade"] == ""


def test_build_evidence_row_redacts_email_when_enabled(probe):
    item = _item(text="Contact me at someone@example.com about this.")
    row = probe.build_evidence_row(
        query="email someone@example.com about work",
        facet_name="work",
        facet_keywords=["work"],
        item=item,
        rank=1,
        redact=True,
    )
    assert "someone@example.com" not in row["text_head"]
    assert "[REDACTED EMAIL]" in row["text_head"]
    assert "someone@example.com" not in row["query"]
    assert "[REDACTED EMAIL]" in row["query"]


def test_build_evidence_row_no_redact_keeps_email(probe):
    item = _item(text="Contact me at someone@example.com about this.")
    row = probe.build_evidence_row(
        query="q", facet_name="f", facet_keywords=[], item=item, rank=1, redact=False,
    )
    assert "someone@example.com" in row["text_head"]


def test_summarize_items_counts_by_collection_and_facet(probe):
    items = [
        _item(collection="conversations", facet="work"),
        _item(collection="conversations", facet="work"),
        _item(collection="facts", facet="counter-evidence"),
    ]
    per_collection, per_facet = probe.summarize_items(items)
    assert per_collection == {"conversations": 2, "facts": 1}
    assert per_facet == {"work": 2, "counter-evidence": 1}


def test_summarize_items_handles_missing_attrs(probe):
    items = [SimpleNamespace()]
    per_collection, per_facet = probe.summarize_items(items)
    assert per_collection == {"?": 1}
    assert per_facet == {"?": 1}


def test_build_summary_row_shape(probe):
    row = probe.build_summary_row(
        query="q", per_collection={"conversations": 2}, per_facet={"work": 2}, total=2,
    )
    assert row == {
        "type": "summary",
        "query": "q",
        "total_items": 2,
        "per_collection": {"conversations": 2},
        "per_facet": {"work": 2},
    }


def test_format_table_includes_query_and_counts(probe):
    items = [_item(text="short snippet")]
    per_collection, per_facet = probe.summarize_items(items)
    table = probe.format_table("my query", items, per_collection, per_facet)
    assert "my query" in table
    assert "1 evidence items" in table
    assert "conversations" in table
