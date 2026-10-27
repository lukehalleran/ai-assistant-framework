"""Self-docs follow the reviewed manifest in every mode (class: BC-59, BC-58).

Incident 2026-10-27: the live seed globbed all docs/*.md (handoffs, private drafts)
into reference_docs and the read side admitted anything under docs/. Every test drives
the deployed function (seed helper, get_reference_docs, normalize_chroma_rows, purge).
"""
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import utils.self_docs_manifest as sdm

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def app_root(tmp_path, monkeypatch):
    """Synthetic app root: docs/ with one listed + two unlisted docs, manifest lists one."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "config").mkdir()
    for n in ("LISTED.md", "HANDOFF_canary.md", "private_canary.md"):
        (tmp_path / "docs" / n).write_text("# x")
    (tmp_path / "config" / "reference_docs_manifest.txt").write_text(
        "# reviewed\ndocs/LISTED.md  # trailing\n")

    def fake_resolve(path):
        return path if os.path.isabs(path) else os.path.normpath(str(tmp_path / path))

    monkeypatch.setattr(sdm, "resolve_bundled_path", fake_resolve)
    sdm.clear_cache()
    yield tmp_path
    sdm.clear_cache()


class RecordingManager:
    def __init__(self):
        self.files, self.dirs = [], []

    def sync_file(self, p):
        self.files.append(Path(p).name)
        return "uploaded"

    def sync_directory(self, d):
        self.dirs.append(d)
        return {"uploaded": 2, "skipped": 0, "failed": 0}


def test_parse_manifest_shared_object():
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_bundle_manifest as cbm
    assert cbm.parse_manifest is sdm.parse_manifest
    assert cbm.MANIFEST_RELPATH == sdm.MANIFEST_RELPATH
    assert sdm.parse_manifest("a # c\n\na\nb") == ["a", "b"]


def test_seed_syncs_only_listed_and_other_dirs_normally(app_root, tmp_path):
    from gui.launch import _seed_reference_docs
    other = tmp_path / "other"
    other.mkdir()
    m = RecordingManager()
    up, sk, fa = _seed_reference_docs(m, ["docs", str(other)])
    assert m.files == ["LISTED.md"]
    assert m.dirs == [str(other.resolve())]
    assert (up, sk, fa) == (3, 0, 0)


def test_seed_is_cwd_independent(app_root, tmp_path, monkeypatch):
    from gui.launch import _seed_reference_docs
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    m = RecordingManager()
    _seed_reference_docs(m, ["docs"])
    assert m.files == ["LISTED.md"] and m.dirs == []


def test_seed_missing_listed_file_is_a_manifest_problem(app_root):
    from gui.launch import _seed_reference_docs
    (app_root / "docs" / "LISTED.md").unlink()
    m = RecordingManager()
    assert _seed_reference_docs(m, ["docs"]) == (0, 0, 0)
    assert m.files == []
    assert any("LISTED.md: file does not exist" in p for p in sdm.manifest_problems())


@pytest.mark.parametrize("seed", ["docs/HANDOFF_canary.md", "docs/sub", "docs/sub/DEEP.md"])
def test_explicit_seed_paths_inside_docs_obey_manifest(app_root, seed):
    # Review 2026-10-27 #4: an explicit seed path inside docs/ must not bypass the
    # manifest (it would re-introduce purged material on the next restart).
    from gui.launch import _seed_reference_docs
    (app_root / "docs" / "sub").mkdir()
    (app_root / "docs" / "sub" / "DEEP.md").write_text("# x")
    m = RecordingManager()
    assert _seed_reference_docs(m, [seed]) == (0, 0, 0)
    assert m.files == [] and m.dirs == []


def test_explicit_listed_file_seed_path_still_syncs(app_root):
    from gui.launch import _seed_reference_docs
    m = RecordingManager()
    assert _seed_reference_docs(m, ["docs/LISTED.md"]) == (1, 0, 0)
    assert m.files == ["LISTED.md"]


class MisbehavingManager:
    def __init__(self, docs):
        self._docs = docs

    async def get_documents(self, query, limit=10):
        return self._docs


def _gatherer(docs):
    from core.prompt.context_gatherer import ContextGatherer
    g = ContextGatherer.__new__(ContextGatherer)
    g._reference_docs_manager = MisbehavingManager(docs)
    g.memory_id_map = {}
    return g


def _chunks(root):
    d = str(root / "docs")
    return [
        {"content": "handoff", "metadata": {"title": "HANDOFF_canary", "file_path": f"{d}/HANDOFF_canary.md"}},
        {"content": "private", "metadata": {"title": "private_canary", "file_path": f"{d}/private_canary.md"}},
        {"content": "upload", "metadata": {"title": "up", "type": "user_upload", "file_path": f"{d}/LISTED.md"}},
        {"content": "unlisted title", "metadata": {"title": "NOPE"}},
        {"content": "transcript", "metadata": {"title": "T", "file_path": "/elsewhere/Downloads/T.txt"}},
        {"content": "listed A", "metadata": {"title": "LISTED", "file_path": f"{d}/LISTED.md"}},
        {"content": "listed B", "metadata": {"title": "LISTED"}},
    ]


@pytest.mark.asyncio
async def test_read_gate_returns_only_listed(app_root):
    # "listed B" carries a listed title but no recorded path: no provenance, dropped.
    docs = await _gatherer(_chunks(app_root)).get_reference_docs("how do you work")
    assert [d["content"] for d in docs] == ["listed A"]


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["/other/project/docs/LISTED.md", "nested"])
async def test_same_filename_elsewhere_is_not_self_knowledge(app_root, where):
    # Review 2026-10-27 #1: a matching filename from another project/install, or nested
    # deeper inside this docs dir, does not establish reviewed provenance.
    fp = str(app_root / "docs" / "private" / "docs" / "LISTED.md") if where == "nested" else where
    chunk = {"content": "foreign", "metadata": {"title": "LISTED", "file_path": fp}}
    assert await _gatherer([chunk]).get_reference_docs("q") == []


def test_moved_install_reseeds_so_provenance_follows(app_root, tmp_path):
    # Same content, different stored path (moved install): sync_file must re-upload so the
    # recorded path matches this install again; same path => skipped.
    from knowledge.reference_docs_manager import ReferenceDocsManager
    doc = app_root / "docs" / "LISTED.md"
    mgr = ReferenceDocsManager.__new__(ReferenceDocsManager)
    h = ReferenceDocsManager._compute_file_hash(str(doc))
    stored = {"content_hash": h, "file_path": "/old/install/docs/LISTED.md", "title": "LISTED"}
    mgr._get_document_chunks = lambda title: [{"metadata": stored}]
    uploads = []
    mgr.upload_document = lambda path, title: uploads.append(path) or MagicMock(success=True)
    assert mgr.sync_file(str(doc)) == "uploaded" and uploads == [str(doc.resolve())]
    stored["file_path"] = str(doc.resolve())
    assert mgr.sync_file(str(doc)) == "skipped" and len(uploads) == 1


@pytest.mark.asyncio
async def test_invalid_entry_keeps_valid_ones_for_reads(app_root):
    (app_root / "config" / "reference_docs_manifest.txt").write_text("not-a-doc\ndocs/LISTED.md\n")
    sdm.clear_cache()
    docs = await _gatherer(_chunks(app_root)).get_reference_docs("q")
    assert [d["content"] for d in docs] == ["listed A"]
    assert any("not-a-doc" in p for p in sdm.manifest_problems())


@pytest.mark.asyncio
async def test_fail_closed_unreadable_manifest(app_root, caplog):
    (app_root / "config" / "reference_docs_manifest.txt").unlink()
    sdm.clear_cache()
    with caplog.at_level(logging.WARNING, logger=sdm.logger.name):
        docs = await _gatherer(_chunks(app_root)).get_reference_docs("q")
        from gui.launch import _seed_reference_docs
        m = RecordingManager()
        _seed_reference_docs(m, ["docs"])
    assert docs == []
    assert m.files == [] and m.dirs == []
    assert len([r for r in caplog.records if "manifest unreadable" in r.getMessage()]) == 1


def test_normalize_files_channel_drops_unlisted(app_root):
    from core.insight.coordinator import normalize_chroma_rows
    d = str(app_root / "docs")
    rows = [
        {"id": "1", "document": "handoff", "metadata": {"title": "H", "file_path": f"{d}/HANDOFF_canary.md"}},
        {"id": "2", "document": "upload", "metadata": {"title": "U", "type": "user_upload", "file_path": f"{d}/HANDOFF_canary.md"}},
        {"id": "3", "document": "listed", "metadata": {"title": "LISTED", "file_path": f"{d}/LISTED.md"}},
        {"id": "4", "document": "outside", "metadata": {"title": "O", "file_path": "/x/y.txt"}},
        {"id": "5", "document": "old install", "metadata": {"title": "HANDOFF_old", "file_path": "/old/install/docs/HANDOFF_old.md"}},
    ]
    kept = [r["text"] for r in normalize_chroma_rows(rows, channel="files")]
    assert kept == ["upload", "listed", "outside"]
    assert len(normalize_chroma_rows(rows, channel="notes")) == 5


def _load_purge():
    spec = importlib.util.spec_from_file_location("purge_unlisted_self_docs", ROOT / "scripts" / "purge_unlisted_self_docs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeColl:
    def __init__(self, got, log):
        self._got, self.log, self.deleted = got, log, []

    def get(self, include=None):
        return self._got

    def delete(self, ids):
        self.log.append("delete")
        self.deleted.extend(ids)


def _got(root):
    d = str(root / "docs")
    metas = [
        {"title": "HANDOFF_canary", "file_path": f"{d}/HANDOFF_canary.md"},
        {"title": "LISTED", "file_path": f"{d}/LISTED.md"},
        {"title": "up", "type": "user_upload", "file_path": f"{d}/private_canary.md"},
        {"title": "T", "file_path": "/elsewhere/T.txt"},
        {"title": "private_canary", "file_path": f"{d}/private_canary.md"},
    ]
    return {"ids": ["a", "b", "c", "d", "e"], "documents": list("vwxyz"), "metadatas": metas}


def test_purge_selection_only_unlisted_repo_docs(app_root):
    purge = _load_purge()
    assert [c["id"] for c in purge.select_unlisted(_got(app_root))] == ["a", "e"]


def test_purge_apply_refused_when_daemon_running(app_root, tmp_path):
    purge = _load_purge()
    coll = FakeColl(_got(app_root), [])
    assert purge.run(coll, True, tmp_path / "bk", daemon_running=lambda: True) == 1
    assert coll.deleted == [] and not (tmp_path / "bk").exists()


@pytest.mark.parametrize("apply", [False, True])
def test_purge_refuses_on_invalid_nonempty_manifest(app_root, tmp_path, apply):
    # Review 2026-10-27 #2: "not-a-doc" alone made every doc look unlisted, so the
    # approved doc was selected and deleted. Any manifest problem => refuse.
    (app_root / "config" / "reference_docs_manifest.txt").write_text("not-a-doc\n")
    sdm.clear_cache()
    purge = _load_purge()
    coll = FakeColl(_got(app_root), [])
    assert purge.run(coll, apply, tmp_path / "bk", daemon_running=lambda: False) == 1
    assert coll.deleted == [] and not (tmp_path / "bk").exists()


def test_purge_reports_old_install_docs_but_never_selects_them(app_root, tmp_path, capsys):
    purge = _load_purge()
    got = _got(app_root)
    got["ids"].append("f")
    got["documents"].append("old")
    got["metadatas"].append({"title": "HANDOFF_old", "file_path": "/old/install/docs/HANDOFF_old.md"})
    coll = FakeColl(got, [])
    assert purge.run(coll, True, tmp_path / "bk", daemon_running=lambda: False) == 0
    assert coll.deleted == ["a", "e"]
    assert "/old/install/docs/HANDOFF_old.md" in capsys.readouterr().out


def test_real_manifest_is_fully_valid():
    sdm.clear_cache()
    try:
        assert sdm.manifest_problems() == ()
        assert sdm.manifest_entries()
    finally:
        sdm.clear_cache()


@pytest.mark.parametrize("apply", [False, True])
def test_purge_refuses_when_manifest_unreadable(app_root, tmp_path, apply):
    # Empty manifest => every repo doc reads as "unlisted"; the purge must not select
    # (or delete) the reviewed self-docs along with the handoffs.
    (app_root / "config" / "reference_docs_manifest.txt").unlink()
    sdm.clear_cache()
    purge = _load_purge()
    coll = FakeColl(_got(app_root), [])
    assert purge.run(coll, apply, tmp_path / "bk", daemon_running=lambda: False) == 1
    assert coll.deleted == [] and not (tmp_path / "bk").exists()


def test_purge_dry_run_deletes_nothing(app_root, tmp_path):
    purge = _load_purge()
    coll = FakeColl(_got(app_root), [])
    assert purge.run(coll, False, tmp_path / "bk", daemon_running=lambda: True) == 0
    assert coll.deleted == []


def test_purge_apply_writes_preimage_before_delete(app_root, tmp_path, monkeypatch):
    purge = _load_purge()
    log = []
    coll = FakeColl(_got(app_root), log)
    real_open = open

    def spy_open(path, *a, **k):
        if "purge_unlisted_self_docs_preimage" in str(path):
            log.append("preimage")
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", spy_open)
    assert purge.run(coll, True, tmp_path / "bk", daemon_running=lambda: False) == 0
    monkeypatch.undo()
    assert log == ["preimage", "delete"]
    assert coll.deleted == ["a", "e"]
    lines = [json.loads(x) for x in next((tmp_path / "bk").glob("*.jsonl")).read_text().splitlines()]
    assert [l["id"] for l in lines] == ["a", "e"]
    assert lines[0]["document"] == "v" and lines[0]["metadata"]["title"] == "HANDOFF_canary"
