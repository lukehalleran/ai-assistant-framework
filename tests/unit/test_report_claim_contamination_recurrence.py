"""CLS-5: the read-only recurrence detector + per-store disposition (2026-10-16).

``scripts/report_claim_contamination.py`` gains ``recurrence_unsupported`` /
``recurrence_no_evidence``, an ``--inventory`` mode, per-store disposition
cells (profile / Chroma ``facts`` / graph edges / graph node attributes) with
``present`` / ``absent`` / ``unresolved`` semantics, and per-tool id files.

Synthetic stores only (tmp paths, never ``data/``).  The detector runs THE
deployed ``recurrence_claim_supported``; the id files are checked against the
real parsers of the three existing tools (a dry-run through each tool's own
``main``).  Names and values are invented.

class: BC-51, BC-58 (sibling callers), BC-75
"""
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"oct16_{name}", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rcc = _load("report_claim_contamination")
OBSERVED = datetime(2026, 10, 16, 12, 0)

OBJ_EAT = "orders food after completing homework"


def _fact(fid, relation, value, excerpt, current=True):
    return {"fact_id": fid, "relation": relation, "value": value,
            "source_excerpt": excerpt, "is_current": current}


def _categories():
    return {"habits": [
        _fact("fid-eat", "eating_habit", OBJ_EAT, ""),                                   # no evidence
        _fact("fid-sleep", "sleep_pattern", "slept heavy til noon",
              "Slept heavy til noon today after the late night."),                       # unsupported
        _fact("fid-bed", "bedtime_habit", "goes to bed around 11",
              "I always go to bed around 11"),                                           # control
        _fact("fid-env", "home_environment", "quiet", "It is quiet at home these days."),  # non-family
        _fact("fid-tea", "likes", "tea", "I like tea"),                                  # control
        _fact("fid-loose", "eating_habit", "orders food after homework", ""),            # ambiguous
        _fact("fid-old", "study_habit", "completed 5 problems", "", current=False),      # not current
    ]}


# ---------------------------------------------------------------------------
# synthetic stores
# ---------------------------------------------------------------------------

@pytest.fixture
def chroma_dir(tmp_path):
    import chromadb
    path = tmp_path / "chroma"
    coll = chromadb.PersistentClient(path=str(path)).create_collection("facts", embedding_function=None)
    rows = [
        ("c1", f"user | eating_habit | {OBJ_EAT}", {}),
        ("c2", "user | eating_habit | orders pizza after class", {}),
        ("c3", "user | home_environment | quiet", {"curation_quarantined": True}),
        ("c4", "user | likes | tea", {}),
    ]
    coll.add(ids=[r[0] for r in rows], documents=[r[1] for r in rows],
             metadatas=[r[2] or {"type": "fact"} for r in rows],
             embeddings=[[0.1, 0.2, 0.3]] * len(rows))
    return str(path)


@pytest.fixture
def graph_file(tmp_path):
    from memory.graph_memory import GraphMemory
    from memory.graph_models import GraphEdge, GraphNode
    path = str(tmp_path / "graph.json")
    gm = GraphMemory(persist_path=path)
    gm.add_entity(GraphNode(entity_id="user", display_name="User", entity_type="person",
                            metadata={"eating_habit": OBJ_EAT}))
    gm.add_entity(GraphNode(entity_id="order_food", display_name=OBJ_EAT))
    gm.add_relation(GraphEdge(source_id="user", relation="eating_habit", target_id="order_food"))
    gm.save(raise_on_error=True)
    return path


def _rows(chroma_dir, graph_file, inventory=("fid-env",)):
    return rcc.build_disposition_rows(
        _categories(), inventory_ids=list(inventory), observed_at=OBSERVED,
        chroma_docs=rcc.load_chroma_fact_docs(chroma_dir), graph=rcc.load_graph(graph_file),
    )


# ---------------------------------------------------------------------------
# detector reasons
# ---------------------------------------------------------------------------

class TestDetectorReasons:
    def test_empty_excerpt_in_the_family_is_no_evidence(self):
        reasons = rcc.evaluate_fact(_categories()["habits"][0], observed_at=OBSERVED)
        assert reasons == ["recurrence_no_evidence"]

    def test_one_off_excerpt_is_unsupported(self):
        reasons = rcc.evaluate_fact(_categories()["habits"][1], observed_at=OBSERVED)
        assert "recurrence_unsupported" in reasons

    @pytest.mark.parametrize("idx", [2, 3, 4])
    def test_controls_are_not_flagged(self, idx):
        reasons = rcc.evaluate_fact(_categories()["habits"][idx], observed_at=OBSERVED)
        assert not set(reasons) & rcc.RECURRENCE_REASONS

    def test_recurrence_candidates_propose_quarantine(self):
        assert rcc.propose_action(["recurrence_no_evidence"]) == "quarantine"
        assert rcc.propose_action(["recurrence_unsupported"]) == "quarantine"

    def test_existing_report_rows_include_the_new_reasons_and_skip_historical(self):
        rows = rcc.build_report_rows(_categories(), observed_at=OBSERVED)
        by_id = {r["fact_id"]: r for r in rows}
        assert "fid-eat" in by_id and "fid-sleep" in by_id
        assert "fid-bed" not in by_id and "fid-tea" not in by_id and "fid-old" not in by_id


# ---------------------------------------------------------------------------
# disposition rows
# ---------------------------------------------------------------------------

class TestDispositionRows:
    def test_rows_cover_candidates_and_inventory_but_not_controls(self, chroma_dir, graph_file):
        by_id = {r["fact_id"]: r for r in _rows(chroma_dir, graph_file)}
        assert set(by_id) == {"fid-eat", "fid-sleep", "fid-loose", "fid-env"}
        assert by_id["fid-env"]["origin"] == "inventory"
        assert by_id["fid-eat"]["origin"] == "detector"

    def test_present_in_every_store(self, chroma_dir, graph_file):
        row = {r["fact_id"]: r for r in _rows(chroma_dir, graph_file)}["fid-eat"]
        assert row["profile"] == "present:fid-eat"
        assert row["chroma"] == "present:c1" and row["chroma_state"] == "c1=live"
        assert row["graph_edges"] == "present:user|eating_habit|order_food"
        assert row["graph_state"] == "user|eating_habit|order_food=visible"
        assert row["graph_node_attrs"] == "present:user.eating_habit"

    def test_absent_means_searched_and_verifiably_none(self, chroma_dir, graph_file):
        row = {r["fact_id"]: r for r in _rows(chroma_dir, graph_file)}["fid-sleep"]
        assert row["chroma"] == "absent"
        assert row["graph_edges"] == "absent"
        assert row["graph_node_attrs"] == "absent"

    def test_inventory_covers_a_non_family_fact_with_its_quarantine_state(self, chroma_dir, graph_file):
        row = {r["fact_id"]: r for r in _rows(chroma_dir, graph_file)}["fid-env"]
        assert row["chroma"] == "present:c3"
        assert row["chroma_state"] == "c3=quarantined"
        assert row["reasons"] == ""            # the detector says nothing; the row exists anyway

    def test_ambiguous_object_is_unresolved_never_absent(self, chroma_dir, graph_file):
        row = {r["fact_id"]: r for r in _rows(chroma_dir, graph_file)}["fid-loose"]
        assert row["chroma"].startswith("unresolved:ambiguous_object")
        assert row["graph_edges"].startswith("unresolved:ambiguous_target")
        assert row["graph_node_attrs"].startswith("unresolved:ambiguous_value")

    def test_unscanned_stores_are_unresolved(self):
        rows = rcc.build_disposition_rows(_categories(), observed_at=OBSERVED)
        assert rows and all(r["chroma"].startswith("unresolved:") for r in rows)
        assert all(r["graph_edges"].startswith("unresolved:") for r in rows)

    def test_unreadable_store_is_unresolved_with_a_reason(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            rcc.load_graph(str(tmp_path / "missing.json"))
        rows = rcc.build_disposition_rows(
            _categories(), observed_at=OBSERVED,
            chroma_docs=None, chroma_error="chroma_unreadable(FileNotFoundError)",
            graph=None, graph_error="graph_unreadable(FileNotFoundError)")
        assert rows[0]["chroma"] == "unresolved:chroma_unreadable(FileNotFoundError)"
        assert rows[0]["graph_edges"] == "unresolved:graph_unreadable(FileNotFoundError)"

    def test_inventory_id_missing_from_the_profile_is_unresolved(self, chroma_dir, graph_file):
        rows = rcc.build_disposition_rows(
            _categories(), inventory_ids=["fid-gone"], observed_at=OBSERVED,
            chroma_docs=rcc.load_chroma_fact_docs(chroma_dir), graph=rcc.load_graph(graph_file))
        gone = {r["fact_id"]: r for r in rows}["fid-gone"]
        assert gone["profile"] == "absent"
        assert gone["chroma"].startswith("unresolved:")

    def test_corrupt_graph_is_unresolved_and_not_quarantined_by_the_report(self, tmp_path):
        bad = tmp_path / "graph.json"
        bad.write_text("{not json")
        with pytest.raises(ValueError):
            rcc.load_graph(str(bad))
        assert sorted(p.name for p in tmp_path.iterdir()) == ["graph.json"]   # no .corrupt-<ts> copy

    def test_report_leaves_the_stores_untouched(self, chroma_dir, graph_file):
        before = Path(graph_file).read_bytes()
        _rows(chroma_dir, graph_file)
        assert Path(graph_file).read_bytes() == before


# ---------------------------------------------------------------------------
# id files, checked against the three tools' own parsers (dry-run through main)
# ---------------------------------------------------------------------------

class TestIdFiles:
    @pytest.fixture
    def curated(self, chroma_dir, graph_file):
        rows = _rows(chroma_dir, graph_file)
        tsv = rcc.rows_to_tsv(rows)
        lines = tsv.splitlines()
        # the owner drops the sleep row by commenting it out
        lines = [("# " + ln) if "fid-sleep" in ln else ln for ln in lines]
        return "\n".join(lines) + "\n"

    def test_files_contain_only_present_ids_of_kept_rows(self, curated, tmp_path):
        out = rcc.write_id_files(curated, str(tmp_path / "ids"))
        assert out["profile_ids"] == ["fid-eat", "fid-loose", "fid-env"]
        assert out["chroma_ids"] == ["c1", "c3"]
        assert out["graph_edges"] == ["user|eating_habit|order_food"]
        text = (tmp_path / "ids" / "owner_decision.txt").read_text()
        assert "node attribute (no tool)" in text and "user.eating_habit" in text
        assert "unresolved chroma" in text      # fid-loose

    def test_chroma_ids_parse_with_quarantine_facts(self, curated, tmp_path):
        rcc.write_id_files(curated, str(tmp_path / "ids"))
        qf = _load("quarantine_facts")
        ids = qf.parse_id_file((tmp_path / "ids" / "chroma_ids.txt").read_text())
        assert ids == ["c1", "c3"]

    def test_graph_edges_parse_with_quarantine_graph_edges(self, curated, tmp_path, graph_file, monkeypatch, capsys):
        rcc.write_id_files(curated, str(tmp_path / "ids"))
        qg = _load("quarantine_graph_edges")
        keys = qg.parse_edge_keys((tmp_path / "ids" / "graph_edges.txt").read_text())
        assert keys == ["user|eating_habit|order_food"]
        monkeypatch.setattr(sys, "argv", ["q", "--from-file", str(tmp_path / "ids" / "graph_edges.txt"),
                                          "--graph", graph_file])
        assert qg.main() == 0                                   # DRY RUN against the synthetic graph
        assert "DRY RUN" in capsys.readouterr().out

    def test_profile_ids_dry_run_with_purge_profile_facts(self, curated, tmp_path, monkeypatch, capsys):
        from memory.user_profile import UserProfile
        profile_path = str(tmp_path / "profile.json")
        up = UserProfile(profile_path=profile_path)
        up.profile["categories"] = _categories()
        up.save()
        rcc.write_id_files(curated, str(tmp_path / "ids"))
        pp = _load("purge_profile_facts")
        monkeypatch.setattr(pp, "_daemon_running", lambda: False)
        import memory.user_profile as up_mod
        monkeypatch.setattr(up_mod, "UserProfile", lambda: UserProfile(profile_path=profile_path))
        monkeypatch.setattr(sys, "argv", ["p", "--from-file", str(tmp_path / "ids" / "profile_ids.txt")])
        assert pp.main() == 0
        out = capsys.readouterr().out
        assert "Candidate fact_ids: 3; matched in profile: 3" in out
        assert "DRY RUN" in out and "NOT FOUND" not in out

    def test_chroma_ids_dry_run_through_quarantine_facts_main(self, curated, tmp_path, chroma_dir, monkeypatch, capsys):
        rcc.write_id_files(curated, str(tmp_path / "ids"))
        qf = _load("quarantine_facts")
        docs = rcc.load_chroma_fact_docs(chroma_dir)

        class _Store:
            def __init__(self, persist_directory=None):
                pass

            def list_all(self, name):
                return docs

        import config.app_config as app_config
        import memory.storage.multi_collection_chroma_store as mcs
        monkeypatch.setattr(mcs, "MultiCollectionChromaStore", _Store)
        monkeypatch.setattr(app_config, "CHROMA_PATH", chroma_dir)
        monkeypatch.setattr(sys, "argv", ["q", "--from-file", str(tmp_path / "ids" / "chroma_ids.txt")])
        assert qf.main() == 0
        out = capsys.readouterr().out
        assert "Requested 2 id(s)" in out and "0 not found" in out and "DRY RUN" in out


# ---------------------------------------------------------------------------
# CLI end to end
# ---------------------------------------------------------------------------

class TestCli:
    def test_main_writes_report_and_tsv(self, tmp_path, chroma_dir, graph_file, monkeypatch, capsys):
        from memory.user_profile import UserProfile
        profile_path = str(tmp_path / "profile.json")
        up = UserProfile(profile_path=profile_path)
        up.profile["categories"] = _categories()
        up.save()
        inv = tmp_path / "inventory.txt"
        inv.write_text("# the environment facts\nfid-env  # quiet\n")
        out, tsv = tmp_path / "report.json", tmp_path / "disp.tsv"
        monkeypatch.setattr(rcc, "_daemon_running", lambda: False)
        monkeypatch.setattr(sys, "argv", [
            "r", "--profile", profile_path, "--out", str(out), "--inventory", str(inv),
            "--disposition-tsv", str(tsv), "--chroma-path", chroma_dir, "--graph", graph_file])
        assert rcc.main() == 0
        report = json.loads(out.read_text())
        assert {r["fact_id"] for r in report["rows"]} >= {"fid-eat", "fid-sleep"}
        lines = tsv.read_text().splitlines()
        assert lines[0].split("\t") == rcc.TSV_COLUMNS
        assert len(lines) == 1 + 4

    def test_emit_id_files_mode_needs_no_profile(self, tmp_path, monkeypatch, capsys):
        curated = tmp_path / "c.tsv"
        curated.write_text(rcc.rows_to_tsv([]))
        monkeypatch.setattr(sys, "argv", ["r", "--emit-id-files", str(curated),
                                          "--scratch-dir", str(tmp_path / "scratch")])
        assert rcc.main() == 0
        assert (tmp_path / "scratch" / "profile_ids.txt").read_text() == ""
