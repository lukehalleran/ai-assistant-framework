"""Proposer tests on SYNTHETIC trees (2026-10-16, plan v2 section 4.1/4.2; class: convention drift).

Drives the deployed ``scripts/propose_test_renames.py`` functions on throwaway repos built under tmp_path;
never reads the real tests/ tree.
"""

from __future__ import annotations

import json

import pytest

from scripts import propose_test_renames as P
from tests.unit.helpers_test_renames import make_repo, snapshot


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path / "repo")


@pytest.fixture
def rows(repo):
    rs, meta = P.build_rows(repo)
    return {r["old_path"].rsplit("/", 1)[-1]: r for r in rs}, meta


def test_dated_inventory_and_exempt_row(repo):
    dated = P.list_dated_files(repo)
    assert "tests/unit/test_sep10_calendar_forced_action_loop.py" in dated
    assert not any("naming_guard" in d or d.endswith("helpers.py") for d in dated)
    rs, _ = P.build_rows(repo)
    exempt = [r for r in rs if r["kind"] == "exempt"]
    assert [r["old_path"] for r in exempt] == ["tests/unit/test_sep10_calendar_forced_action_loop.py"]
    assert exempt[0]["new_path"] == "-" and exempt[0]["reason"] == "class-guard ledger anchor"


def test_single_module_subject_is_mechanical(rows):
    r = rows[0]["test_sep09_engine_widget.py"]
    assert r["subject_module"] == "core/engine.py"
    assert r["confidence"] == "mechanical"
    assert r["new_path"] == "tests/unit/test_engine_widget.py"
    assert "static:core/engine.py" in r["evidence"]
    assert "imported_by:tests/unit/test_oct03_user_of_helper.py" in r["evidence"]
    assert "commit:" in r["evidence"]


def test_dynamic_load_surfaces_the_script_as_subject(rows):
    r = rows[0]["test_sep10_tool_dry_run.py"]
    assert "dynamic:scripts/tool.py" in r["evidence"]
    assert r["subject_module"] == "scripts/tool.py"
    assert r["confidence"] == "review"
    assert r["new_path"] == "tests/unit/test_tool_dry_run.py"
    assert "core/engine.py" in r["evidence"]


def test_dynamic_versus_static_disagreement_is_owner_without_a_forced_name(rows):
    r = rows[0]["test_oct04_static_dominates.py"]
    assert r["confidence"] == "owner"
    assert r["subject_module"] == "?" and r["new_path"] == "?"
    assert "competing:core/engine.py(" in r["evidence"] and "scripts/tool.py(" in r["evidence"]


def test_tie_between_modules_is_owner(rows):
    r = rows[0]["test_oct02_mixed.py"]
    assert r["confidence"] == "owner" and r["new_path"] == "?"
    assert "competing:" in r["evidence"]


def test_ambiguous_basename_is_qualified_and_flagged(rows):
    r, meta = rows[0]["test_oct01_thing.py"], rows[1]
    assert "alpha" in meta["ambiguous_basenames"]
    assert r["subject_module"] == "pkg_b/alpha.py"
    assert "qualified_basename" in r["flags"].split(",")
    assert r["new_path"] == "tests/unit/test_pkg_b_alpha_thing.py"


def test_ambiguity_set_is_computed_from_the_repo(repo):
    prod = P.production_files(repo)
    assert P.ambiguity_set(prod) == {"alpha"}
    assert P.subject_label("pkg_a/alpha.py", prod, {"alpha"}) == ("pkg_a_alpha", True)
    assert P.subject_label("core/engine.py", prod, {"alpha"}) == ("engine", False)


def test_batch_residual_has_no_invented_behaviour(rows):
    for name in ("test_sep11_B4.py", "test_sep27b_B5.py"):
        r = rows[0][name]
        assert "batch_residual" in r["flags"].split(",")
        assert r["new_path"] == "?"
        assert r["subject_module"] == "core/engine.py"


def test_collision_flag_downgrades_mechanical(rows):
    a, b = rows[0]["test_oct05_flush.py"], rows[0]["test_sep12_flush.py"]
    assert a["new_path"] == b["new_path"] == "tests/unit/test_engine_flush.py"
    for r in (a, b):
        assert "collision" in r["flags"].split(",")
        assert r["confidence"] == "review"


def test_safety_name_change_is_flagged_and_owner(rows, repo):
    r = rows[0]["test_oct06_misc.py"]
    assert r["new_path"] == "tests/unit/test_valence_misc.py"
    assert "safety_name" in r["flags"].split(",") and r["confidence"] == "owner"
    regex = P.load_safety_regex(repo)
    assert P.safety_name_changes(regex, "tests/unit/test_a.py", "tests/unit/test_escalation_a.py")
    assert P.safety_name_changes(regex, "tests/unit/test_escalation_a.py", "tests/unit/test_a.py")
    assert not P.safety_name_changes(regex, "tests/unit/test_a.py", "tests/unit/test_b.py")
    assert not P.safety_name_changes(None, "a", "b")


def test_citation_inventory_is_case_insensitive_and_split(repo):
    stems = {p.rsplit("/", 1)[-1][:-3] for p in P.list_dated_files(repo)}
    inv = P.citation_inventory(repo, stems)
    widget = inv["test_sep09_engine_widget"]
    cur_files = {f for f, _ in widget["current"]}
    his_files = {f for f, _ in widget["historical"]}
    assert "docs/BUG_CLASSES.md" in cur_files and "core/engine.py" in cur_files
    assert "tests/unit/test_oct03_user_of_helper.py" in cur_files
    assert his_files == {"docs/PLAN_20261001_history.md"}
    assert ("docs/notes.md", 1, "TEST_SEP09_ENGINE_WIDGET") in widget["case_variants"]
    # a name with a capital letter (v1's lowercase-only regex missed this shape)
    assert any(f == "docs/BUG_CLASSES.md" for f, _ in inv["test_sep27b_B5"]["current"])


def test_location_classes():
    cls = P.classify_location
    assert cls("docs/BUG_CLASSES.md") == "current" and cls("docs/TEST_LANES.md") == "current"
    assert cls("core/engine.py") == "current" and cls("scripts/smoke_frozen_build.py") == "current"
    for hist in (
        "docs/PLAN_20261016_x.md", "docs/HANDOFF_20260907_y.md", "docs/execution/a/b.md",
        "docs/history/z.md", "docs/audits/q.md", "docs/postmortems/r.md", "CLAUDE_CHANGELOG.md",
        "config/bug_class_dispositions.json", "docs/archive/INDEPENDENT_AUDIT_20260905.md",
        "docs/execution/codex_20260913/guard_receipt.json",
    ):
        assert cls(hist) == "historical", hist


def test_importers_come_from_the_ast_not_text_search(repo):
    _, meta = P.build_rows(repo)
    assert meta["importers"]["tests/unit/test_sep09_engine_widget.py"] == ["tests/unit/test_oct03_user_of_helper.py"]


def test_map_round_trips_through_parse(repo):
    rs, meta = P.build_rows(repo)
    text = P.render_map(rs, meta, "draft")
    header, parsed, errs = P.parse_map(text)
    assert errs == [] and header["status"] == "draft" and header["base_sha"] == meta["base_sha"]
    assert len(parsed) == len(rs) and set(header["hashes"]) == {r["old_path"] for r in rs if r["kind"] == "rename"}
    assert P.parse_map(text.replace(P.COLUMN_LINE, "wrong"))[2]


def test_cli_is_read_only_and_writes_outside_the_checkout(repo, tmp_path):
    before = snapshot(repo)
    out = tmp_path / "out" / "draft.tsv"
    out.parent.mkdir()
    inv = tmp_path / "out" / "inv.json"
    assert P.main(["--root", str(repo), "--out", str(out), "--inventory-out", str(inv),
                   "--summary", str(tmp_path / "out" / "s.txt")]) == 0
    assert snapshot(repo) == before
    assert out.read_text().startswith("#status\tdraft")
    assert "inventory" in json.loads(inv.read_text())
    assert "confidence:" in (tmp_path / "out" / "s.txt").read_text()
