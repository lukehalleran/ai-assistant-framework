"""Backup/restore contracts on a complete, synthetic temporary data root."""
import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from utils import backup_manager
from scripts import restore_backup


@pytest.fixture
def stores(tmp_path, monkeypatch):
    import config.app_config as config
    import memory.learned_relations as relations
    import utils.adaptive_exemplars as exemplars
    from memory.user_profile import UserProfile
    from memory.graph_memory import GraphMemory
    from memory.graph_models import GraphNode

    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    names = {
        "KNOWLEDGE_GRAPH_PERSIST_PATH": "knowledge_graph.json",
        "KNOWLEDGE_GRAPH_ALIASES_PATH": "entity_aliases.json",
        "CORPUS_FILE": "corpus.json",
        "STALENESS_INDEX_PATH": "claim_index.json",
        "PROACTIVE_SURFACING_HISTORY_PATH": "surfacing_history.json",
    }
    for setting, name in names.items():
        monkeypatch.setattr(config, setting, str(data / name))
    monkeypatch.setattr(UserProfile, "DEFAULT_PATH", str(data / "user_profile.json"))
    monkeypatch.setattr(exemplars, "_STORE_PATH", str(data / "adaptive_exemplars.json"))
    monkeypatch.setattr(relations, "_STORE_PATH", str(data / "learned_relations.json"))
    monkeypatch.setenv("NARRATIVE_STALE_FLAG_PATH", str(data / "narrative_stale.json"))
    expected = set(names.values()) | {
        "user_profile.json", "adaptive_exemplars.json", "learned_relations.json",
        "tone_state.json", "pending_actions.json", "curation_queue.json", "narrative_stale.json",
    }
    for name in expected:
        (data / name).write_text(json.dumps({"synthetic_store": name, "value": 17}))
    # Produce the two critical JSON schemas with their deployed writers.
    (data / "user_profile.json").unlink()
    profile = UserProfile(str(data / "user_profile.json"))
    profile.profile["categories"]["career"] = [{"fact_id": "synthetic", "is_current": True}]
    profile.save()
    (data / "knowledge_graph.json").unlink()
    graph = GraphMemory(persist_path=str(data / "knowledge_graph.json"))
    graph.add_entity(GraphNode(entity_id="synthetic", display_name="Synthetic"))
    graph.save()
    (data / "google_token.json").write_text('{"synthetic_secret": "excluded"}')
    chroma = data / "chroma_db_v4"
    chroma.mkdir()
    with sqlite3.connect(chroma / "chroma.sqlite3") as db:
        db.execute("create table synthetic (value text)")
        db.execute("insert into synthetic values ('original')")
    monkeypatch.setattr(config, "CHROMA_PATH", str(chroma))
    for key, value in dict(BACKUP_DIR=str(tmp_path / "backups"), BACKUP_ENABLED=True,
                           BACKUP_RETENTION=3, BACKUP_INCLUDE_CHROMA=True,
                           BACKUP_MIN_INTERVAL_HOURS=0).items():
        monkeypatch.setattr(config, key, value)
    lock = Mock(return_value=object())
    monkeypatch.setattr("utils.single_instance.acquire_single_instance_lock", lock)
    return SimpleNamespace(root=tmp_path, data=data, expected=expected, lock=lock,
                           original={name: (data / name).read_bytes() for name in expected})


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def backup(stores):
    result = backup_manager.run_backup(reason="synthetic", include_chroma=True)
    assert result.ok, result.error
    manifest = json.loads((Path(result.path) / "manifest.json").read_text())
    assert set(manifest["files"]) == stores.expected
    assert "google_token.json" not in manifest["files"]
    return Path(result.path).name


def test_target_enumerator_includes_every_configured_store(stores):
    paths = backup_manager.backup_targets()
    assert {Path(p).resolve() for p in paths} == {stores.data / name for name in stores.expected}


def test_backup_restore_reopens_real_json_and_sqlite_writers(stores):
    from memory.graph_memory import GraphMemory
    from memory.user_profile import UserProfile

    name = backup(stores)
    for filename in stores.expected:
        (stores.data / filename).write_text('{"synthetic_changed": true}')
    before = snapshot(stores.root)
    assert restore_backup.cmd_restore(name, apply=False) == 0
    assert snapshot(stores.root) == before  # dry-run is read-only
    stores.lock.assert_not_called()
    assert restore_backup.cmd_restore(name, apply=True) == 0
    stores.lock.assert_called_once()
    assert {f: (stores.data / f).read_bytes() for f in stores.expected} == stores.original
    graph = GraphMemory(persist_path=str(stores.data / "knowledge_graph.json"))
    assert graph.get_entity("synthetic").display_name == "Synthetic"
    profile = UserProfile(str(stores.data / "user_profile.json"))
    assert profile.profile["categories"]["career"][0]["fact_id"] == "synthetic"
    with sqlite3.connect(stores.data / "chroma_db_v4/chroma.sqlite3") as db:
        assert db.execute("select value from synthetic").fetchall() == [("original",)]
    assert len(list(stores.data.glob("*.pre-restore-*"))) == len(stores.expected) + 1


def test_restore_recreates_missing_post_july_stores(stores):
    name = backup(stores)
    missing = {"adaptive_exemplars.json", "learned_relations.json", "tone_state.json",
               "pending_actions.json", "curation_queue.json", "narrative_stale.json"}
    for filename in missing:
        (stores.data / filename).unlink()
    assert restore_backup.cmd_restore(name, apply=True) == 0
    assert {f for f in missing if (stores.data / f).is_file()} == missing
    assert {f: (stores.data / f).read_bytes() for f in missing} == {
        f: stores.original[f] for f in missing}


def test_restore_refuses_incomplete_backup_before_touching_targets(stores):
    name = backup(stores)
    (stores.root / "backups" / name / "tone_state.json").unlink()
    before = snapshot(stores.data)
    assert restore_backup.cmd_restore(name, apply=True) == 1
    assert snapshot(stores.data) == before
    stores.lock.assert_not_called()


def test_restore_refuses_when_daemon_lock_is_held(stores):
    from utils.single_instance import SingleInstanceError

    name = backup(stores)
    stores.lock.side_effect = SingleInstanceError("synthetic daemon holds the lock")
    before = snapshot(stores.data)
    with pytest.raises(SystemExit) as exc:
        restore_backup.cmd_restore(name, apply=True)
    assert exc.value.code == 1 and snapshot(stores.data) == before


def test_interrupted_restore_preserves_previous_data_aside(stores, monkeypatch):
    name = backup(stores)
    target = stores.data / "tone_state.json"
    target.write_text('{"synthetic_previous": true}')
    copy = restore_backup.shutil.copy2

    def fail_copy(src, dst, *args, **kwargs):
        if Path(dst).resolve() == target:
            raise OSError("synthetic destination failure")
        return copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(restore_backup.shutil, "copy2", fail_copy)
    with pytest.raises(OSError, match="destination failure"):
        restore_backup.cmd_restore(name, apply=True)
    aside, = stores.data.glob("tone_state.json.pre-restore-*")
    assert json.loads(aside.read_text()) == {"synthetic_previous": True}
    assert (stores.root / "backups" / name / "tone_state.json").read_bytes() == stores.original["tone_state.json"]
