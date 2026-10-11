"""Oct-10 hygiene (class: BC-70, BC-10).

1. The formatter comment about the wiki-index DISABLED state names the module
   that really consumes ``semantic_search.index_available()``.
2. ``scripts/backfill_visual_memory.py`` built ``VisualMemoryStore`` WITHOUT the
   chroma store, so ``add_image`` skipped the Chroma write and
   ``visual_memories`` stayed empty while ``clip_metadata.json`` held 48
   entries. The script now builds the store as the live path does.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "backfill_visual_memory_under_test", REPO_ROOT / "scripts" / "backfill_visual_memory.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeChroma:
    def __init__(self):
        self.adds = []

    def add_to_collection(self, name, text, meta):
        self.adds.append((name, text, meta))


def test_formatter_comment_names_real_consumer():
    src = (REPO_ROOT / "core" / "prompt" / "formatter.py").read_text()
    start = src.index("Wiki semantic index (FAISS)")
    block = src[start:start + 900]
    assert "core/prompt/builder.py" in block
    assert "_get_semantic_chunks` (gatherer_knowledge.py)\n" not in block
    # the claim is true: builder consumes it
    assert "index_available()" in (REPO_ROOT / "core" / "prompt" / "builder.py").read_text()


def test_build_visual_store_passes_chroma_store(tmp_path):
    mod = _load_script()
    chroma = _FakeChroma()
    store = mod._build_visual_store(
        chroma, data_dir=str(tmp_path)
    )
    assert store._chroma is chroma
    store._index_path = str(tmp_path / "idx.faiss")
    store._meta_path = str(tmp_path / "meta.json")
    doc_id = store.add_image(
        "x.png", np.ones(512, dtype=np.float32), "a cat", image_hash="h1"
    )
    assert doc_id
    assert [a[0] for a in chroma.adds] == ["visual_memories"]


def test_build_visual_store_without_chroma_skips_chroma(tmp_path):
    mod = _load_script()
    store = mod._build_visual_store(None, data_dir=str(tmp_path))
    assert store._chroma is None


def test_dry_run_writes_nothing_and_opens_no_chroma(tmp_path, monkeypatch, capsys):
    mod = _load_script()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "uploads").mkdir(parents=True)
    (tmp_path / "data" / "uploads" / "a.png").write_bytes(b"\x89PNG-fake")
    clip_mod = type(sys)("knowledge.clip_manager")
    clip_mod.get_clip_manager = lambda: object()
    monkeypatch.setitem(sys.modules, "knowledge.clip_manager", clip_mod)
    built = {}

    def _store(chroma, data_dir="data"):
        from knowledge.visual_memory_store import VisualMemoryStore
        built["chroma"] = chroma
        return VisualMemoryStore(
            chroma_store=chroma, data_dir=data_dir,
            index_path=str(tmp_path / "i.faiss"), meta_path=str(tmp_path / "m.json"),
        )

    monkeypatch.setattr(mod, "_build_visual_store", _store)
    monkeypatch.setattr(mod, "_build_chroma_store", lambda: (_ for _ in ()).throw(AssertionError("chroma opened")))
    monkeypatch.setattr(sys, "argv", ["backfill_visual_memory.py"])
    monkeypatch.setattr(mod, "_daemon_running", lambda: True)  # dry run must not need the guard

    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert mod.main() == 0
    after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert before == after
    assert built["chroma"] is None
    assert "Dry run" in capsys.readouterr().out


def test_execute_refused_when_daemon_running(tmp_path, monkeypatch, capsys):
    mod = _load_script()
    monkeypatch.setattr(sys, "argv", ["backfill_visual_memory.py", "--execute"])
    monkeypatch.setattr(mod, "_daemon_running", lambda: True)

    async def _boom(args):
        raise AssertionError("run_backfill reached with a live Daemon")

    monkeypatch.setattr(mod, "run_backfill", _boom)
    assert mod.main() == 1
    assert "ABORT" in capsys.readouterr().out


def test_execute_builds_store_with_chroma(monkeypatch, tmp_path):
    """The execute path hands _build_chroma_store()'s result to _build_visual_store."""
    mod = _load_script()
    chroma = _FakeChroma()
    seen = {}
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "uploads").mkdir(parents=True)
    (tmp_path / "data" / "uploads" / "a.png").write_bytes(b"\x89PNG-fake")

    class _Clip:
        loaded = True

        def load(self):
            pass

    clip_mod = type(sys)("knowledge.clip_manager")
    clip_mod.get_clip_manager = lambda: _Clip()
    monkeypatch.setitem(sys.modules, "knowledge.clip_manager", clip_mod)

    class _Store:
        def get_stats(self):
            return {"total_images": 0}

    class _Pipe:
        def __init__(self, clip, store, model_manager=None):
            pass

        async def ingest_image(self, path, source="upload"):
            return None

    pipe_mod = type(sys)("knowledge.visual_memory_pipeline")
    pipe_mod.VisualMemoryPipeline = _Pipe
    monkeypatch.setitem(sys.modules, "knowledge.visual_memory_pipeline", pipe_mod)
    monkeypatch.setattr(mod, "_build_chroma_store", lambda: chroma)
    monkeypatch.setattr(mod, "_build_visual_store", lambda c, data_dir="data": seen.setdefault("c", c) and _Store())
    import argparse
    import asyncio
    asyncio.run(mod.run_backfill(argparse.Namespace(execute=True, caption=False, obsidian=False)))
    assert seen["c"] is chroma
