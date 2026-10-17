"""scripts/remove_graph_node_attrs.py — dry-run-first removal of curated graph node attributes.

Incident date: 2026-10-16 (habit cleanup: `user.social_habit` / `user.study_habit`
stored as node attributes had no owner tool). Bug class: BC-75 (laundered facts
need a curation path in every store). Synthetic graphs in tmp_path only.
"""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location("remove_graph_node_attrs", ROOT / "scripts" / "remove_graph_node_attrs.py")
tool = importlib.util.module_from_spec(_spec)
sys.modules["remove_graph_node_attrs"] = tool
_spec.loader.exec_module(tool)

from memory.graph_memory import GraphMemory  # noqa: E402


def _node(name, md):
    return {"display_name": name, "entity_type": "person", "aliases": [], "first_seen": None,
            "last_seen": None, "mention_count": 3, "metadata": md}


@pytest.fixture
def graph_file(tmp_path):
    payload = {"schema_version": 1, "nodes": {
        "user": _node("User", {"social_habit": "texting Rowan", "study_habit": "evening reading",
                               "gaming_habit": "chess"}),
        "rowan": _node("Rowan", {}),
    }, "edges": []}
    p = tmp_path / "kg.json"
    p.write_text(json.dumps(payload))
    return p


def _attrs(path):
    gm = GraphMemory(persist_path=str(path))
    gm.load()
    return {n: dict(gm.graph.nodes[n].get("metadata") or {}) for n in gm.graph.nodes}


@pytest.fixture(autouse=True)
def daemon_down(monkeypatch):
    monkeypatch.setattr(tool, "_daemon_running", lambda: False)


def _run(graph, tmp_path, *extra):
    return tool.main(["--graph", str(graph), "--backup-dir", str(tmp_path / "bk"), *extra])


def test_dry_run_changes_nothing(graph_file, tmp_path, capsys):
    before = graph_file.read_bytes()
    assert _run(graph_file, tmp_path, "--key", "user.social_habit") == 0
    out = capsys.readouterr().out
    assert "texting Rowan" in out and "DRY RUN" in out
    assert graph_file.read_bytes() == before
    assert not (tmp_path / "bk").exists()


def test_apply_removes_exactly_listed_and_writes_preimage(graph_file, tmp_path):
    before = _attrs(graph_file)
    assert _run(graph_file, tmp_path, "--key", "user.social_habit", "--key", "user.study_habit", "--apply") == 0
    after = _attrs(graph_file)
    assert after["user"] == {"gaming_habit": "chess"}
    assert after["rowan"] == before["rowan"]
    pre = list((tmp_path / "bk").glob("remove_graph_node_attrs_preimage_*.jsonl"))
    assert len(pre) == 1
    recs = [json.loads(line) for line in pre[0].read_text().splitlines()]
    assert {(r["node_id"], r["attr"], r["value"]) for r in recs} == {
        ("user", "social_habit", "texting Rowan"), ("user", "study_habit", "evening reading")}


def test_undo_restores_equal_attributes(graph_file, tmp_path):
    before = copy.deepcopy(_attrs(graph_file))
    _run(graph_file, tmp_path, "--key", "user.social_habit", "--key", "user.study_habit", "--apply")
    pre = next((tmp_path / "bk").glob("remove_graph_node_attrs_preimage_*.jsonl"))
    assert _attrs(graph_file) != before
    assert _run(graph_file, tmp_path, "--undo", "--from-preimage", str(pre), "--apply") == 0
    assert _attrs(graph_file) == before


def test_refuses_under_live_daemon(graph_file, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tool, "_daemon_running", lambda: True)
    before = graph_file.read_bytes()
    assert _run(graph_file, tmp_path, "--key", "user.social_habit", "--apply") == 3
    assert "REFUSED" in capsys.readouterr().out
    assert graph_file.read_bytes() == before
    assert not (tmp_path / "bk").exists()


def test_unknown_keys_reported_not_fatal(graph_file, tmp_path, capsys):
    rc = _run(graph_file, tmp_path, "--key", "ghost.social_habit", "--key", "user.no_such",
              "--key", "user.study_habit", "--apply")
    out = capsys.readouterr().out
    assert rc == 0
    assert "node not found: ghost" in out and "attribute not found: user.no_such" in out
    assert _attrs(graph_file)["user"] == {"social_habit": "texting Rowan", "gaming_habit": "chess"}


OWNER_LINE = ("node attribute (no tool): f12 social_habit=texting Rowan -> "
              "present:user.social_habit;user.study_habit")


@pytest.mark.parametrize("text", [OWNER_LINE + "\n", "    " + OWNER_LINE + "   # note\n",
                                  "user.social_habit\n  user.study_habit  # c\n"])
def test_accepts_owner_decision_lines_and_plain_keys_clean_and_indented(graph_file, tmp_path, text):
    f = tmp_path / "keys.txt"
    f.write_text("# header\n" + text)
    assert _run(graph_file, tmp_path, "--from-file", str(f), "--apply") == 0
    assert _attrs(graph_file)["user"] == {"gaming_habit": "chess"}


def test_dotted_node_id_splits_on_last_dot():
    assert tool.parse_keys("dr.smith.clinic_note") == [("dr.smith", "clinic_note")]
