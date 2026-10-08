"""
scripts/reembed_visual_memories.py — dry-run-first rebuild of the CLIP FAISS index.

class: BC-16 (model name + pretrained tag must agree), BC-60, BC-71.
Uses a deterministic fake CLIP manager (unit vectors from file bytes); never loads real weights.
"""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

faiss = pytest.importorskip("faiss", reason="faiss is required to build the visual index")

REPO = Path(__file__).resolve().parents[2]
EXPECTED = "ViT-B-32-quickgelu"


def _load_script():
    spec = importlib.util.spec_from_file_location("reembed_visual_memories", REPO / "scripts" / "reembed_visual_memories.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


reembed = _load_script()


def _unit(seed_bytes: bytes) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "little")
    v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeClip:
    def __init__(self, model_name=EXPECTED):
        self.model_name = model_name
        self.loaded = False
        self.paths = []

    def load(self):
        self.loaded = True

    def encode_image_from_path(self, path):
        self.paths.append(path)
        return _unit(Path(path).read_bytes())


def _build_store(tmp_path, n=4, missing=(), mismatch=()):
    """n images on disk + an OLD-arch index/metadata (old vectors = unit(b'old'+bytes))."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    meta, vecs = [], []
    for i in range(n):
        data = f"image-{i}".encode()
        p = img_dir / f"img{i}.png"
        p.write_bytes(data)
        meta.append({
            "faiss_idx": i,
            "image_path": f"images/img{i}.png",
            "image_hash": hashlib.sha256(data).hexdigest(),
            "caption": f"caption {i}",
        })
        vecs.append(_unit(b"old" + data))
    for i in missing:
        (img_dir / f"img{i}.png").unlink()
    for i in mismatch:
        (img_dir / f"img{i}.png").write_bytes(b"tampered")
    idx = faiss.IndexFlatIP(512)
    idx.add(np.stack(vecs))
    index_path = tmp_path / "clip_index.faiss"
    meta_path = tmp_path / "clip_metadata.json"
    faiss.write_index(idx, str(index_path))
    meta_path.write_text(json.dumps(meta, indent=2))
    return index_path, meta_path, np.stack(vecs)


def _argv(tmp_path, index_path, meta_path, *extra):
    return ["--index", str(index_path), "--meta", str(meta_path), "--image-root", str(tmp_path),
            "--backup-root", str(tmp_path / "backups"), *extra]


@pytest.fixture(autouse=True)
def _daemon_down(monkeypatch):
    monkeypatch.setattr(reembed, "_daemon_running", lambda: False)


def _stat(p):
    st = p.stat()
    return p.read_bytes(), st.st_mtime_ns


def test_dry_run_writes_nothing(tmp_path, capsys):
    index_path, meta_path, _ = _build_store(tmp_path)
    before = (_stat(index_path), _stat(meta_path))
    rc = reembed.main(_argv(tmp_path, index_path, meta_path), manager_factory=lambda: pytest.fail("no encoder in dry run"))
    assert rc == 0
    assert (_stat(index_path), _stat(meta_path)) == before
    assert not (tmp_path / "backups").exists()
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "4 verified" in out


def test_apply_backs_up_before_index_changes(tmp_path):
    index_path, meta_path, _ = _build_store(tmp_path)
    old_index_bytes, old_meta_bytes = index_path.read_bytes(), meta_path.read_bytes()
    seen = {}

    class SpyClip(FakeClip):
        def encode_image_from_path(self, path):
            # first encode happens before any write: the backup must already hold the old bytes
            dirs = list((tmp_path / "backups").glob("reembed_visual_preimage_*"))
            seen["backup_dirs"] = dirs
            seen["index_unchanged_at_encode"] = index_path.read_bytes() == old_index_bytes
            return super().encode_image_from_path(path)

    rc = reembed.main(_argv(tmp_path, index_path, meta_path, "--apply"), manager_factory=SpyClip,
                      chroma_exporter=lambda dest: "chroma stub")
    assert rc == 0
    (bdir,) = seen["backup_dirs"]
    assert seen["index_unchanged_at_encode"] is True
    assert (bdir / index_path.name).read_bytes() == old_index_bytes
    assert (bdir / meta_path.name).read_bytes() == old_meta_bytes
    assert index_path.read_bytes() != old_index_bytes  # and the live index did change afterwards


def test_missing_and_mismatched_keep_old_vectors_and_flag(tmp_path):
    index_path, meta_path, old = _build_store(tmp_path, n=12, missing=(3,), mismatch=(7,))
    rc = reembed.main(_argv(tmp_path, index_path, meta_path, "--apply", "--force"), manager_factory=FakeClip,
                      chroma_exporter=lambda dest: "stub")
    assert rc == 0
    new_index = faiss.read_index(str(index_path))
    new = new_index.reconstruct_n(0, new_index.ntotal)
    meta = json.loads(meta_path.read_text())
    assert np.array_equal(new[3], old[3]) and meta[3]["clip_reembed_status"] == "original_missing"
    assert np.array_equal(new[7], old[7]) and meta[7]["clip_reembed_status"] == "hash_mismatch"
    for i in (0, 1, 2, 4):
        assert not np.array_equal(new[i], old[i])
        assert "clip_reembed_status" not in meta[i]
        assert meta[i]["clip_model"] == EXPECTED and meta[i]["clip_pretrained"]


def test_too_many_bad_originals_refuses_without_force(tmp_path):
    index_path, meta_path, _ = _build_store(tmp_path, n=4, missing=(0,))
    before = (_stat(index_path), _stat(meta_path))
    rc = reembed.main(_argv(tmp_path, index_path, meta_path, "--apply"), manager_factory=FakeClip)
    assert rc == 1
    assert (_stat(index_path), _stat(meta_path)) == before
    assert not (tmp_path / "backups").exists()


def test_rebuild_stays_row_aligned_and_searchable(tmp_path):
    index_path, meta_path, _ = _build_store(tmp_path, n=6)
    fake = FakeClip()
    rc = reembed.main(_argv(tmp_path, index_path, meta_path, "--apply"), manager_factory=lambda: fake,
                      chroma_exporter=lambda dest: "stub")
    assert rc == 0
    meta = json.loads(meta_path.read_text())
    index = faiss.read_index(str(index_path))
    assert index.ntotal == len(meta) == 6
    assert [m["faiss_idx"] for m in meta] == list(range(6))
    for k in range(6):
        q = fake.encode_image_from_path(str(tmp_path / "images" / f"img{k}.png"))
        _, ids = index.search(q.reshape(1, -1), 1)
        assert int(ids[0][0]) == k
    assert not Path(str(index_path) + ".tmp").exists()


def test_refuses_when_daemon_running(tmp_path, monkeypatch):
    index_path, meta_path, _ = _build_store(tmp_path)
    monkeypatch.setattr(reembed, "_daemon_running", lambda: True)
    before = (_stat(index_path), _stat(meta_path))
    rc = reembed.main(_argv(tmp_path, index_path, meta_path, "--apply"), manager_factory=FakeClip)
    assert rc == 1
    assert (_stat(index_path), _stat(meta_path)) == before
    assert not (tmp_path / "backups").exists()


def test_guard_failure_fails_closed(tmp_path, monkeypatch):
    index_path, meta_path, _ = _build_store(tmp_path)
    monkeypatch.undo()  # drop the autouse stub so the REAL wrapper runs
    import utils.daemon_guard as guard

    def boom():
        raise RuntimeError("guard exploded")

    monkeypatch.setattr(guard, "daemon_running", boom)
    assert reembed._daemon_running() is True
    before = (_stat(index_path), _stat(meta_path))
    rc = reembed.main(_argv(tmp_path, index_path, meta_path, "--apply"), manager_factory=FakeClip)
    assert rc == 1
    assert (_stat(index_path), _stat(meta_path)) == before


def test_guard_import_failure_fails_closed(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "utils.daemon_guard", None)  # import raises ImportError
    assert reembed._daemon_running() is True


def test_wrong_model_aborts_with_nothing_written(tmp_path):
    index_path, meta_path, _ = _build_store(tmp_path)
    before = (_stat(index_path), _stat(meta_path))
    rc = reembed.main(_argv(tmp_path, index_path, meta_path, "--apply"),
                      manager_factory=lambda: FakeClip(model_name="ViT-B-32"))
    assert rc == 1
    assert (_stat(index_path), _stat(meta_path)) == before
    assert not (tmp_path / "backups").exists()


def test_misaligned_metadata_refused(tmp_path):
    index_path, meta_path, _ = _build_store(tmp_path)
    meta = json.loads(meta_path.read_text())
    meta[2]["faiss_idx"] = 5
    meta_path.write_text(json.dumps(meta))
    before = (_stat(index_path), _stat(meta_path))
    rc = reembed.main(_argv(tmp_path, index_path, meta_path, "--apply"), manager_factory=FakeClip)
    assert rc == 1
    assert (_stat(index_path), _stat(meta_path)) == before


def test_clip_manager_records_model_name_after_load():
    import knowledge.clip_manager as mod
    from config import app_config

    assert app_config.VISUAL_MEMORY_CLIP_MODEL == EXPECTED  # config + weights tag agree (BC-16)
    stub = MagicMock()
    stub.create_model_and_transforms.return_value = (MagicMock(), None, MagicMock())
    mgr = mod.CLIPManager()
    assert mgr.model_name is None and mgr.pretrained is None
    with patch.dict("sys.modules", {"open_clip": stub}):
        mgr.load()
    assert mgr.loaded is True
    assert mgr.model_name == app_config.VISUAL_MEMORY_CLIP_MODEL
    assert mgr.pretrained == app_config.VISUAL_MEMORY_CLIP_PRETRAINED
    assert stub.create_model_and_transforms.call_args[0][0] == EXPECTED
