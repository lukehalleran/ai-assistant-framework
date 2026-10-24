"""Procedural git index survives a rewritten history and CWD changes.

Incident 2026-10-24, classes BC-58 (a failure read as "nothing new") / BC-16
(CWD-relative paths) / BC-81 (own-store construction): ``data/git_memory_last_hash.txt``
named a commit the 2026-09-03 history rewrite removed, so ``git rev-list
<anchor>..HEAD`` failed and every sync logged "No new commits" for seven weeks.
These tests drive the deployed ``GitMemoryLoader.incremental_update`` against a
real temp git repo and a recording fake store.
"""

import logging
import os
import subprocess
import threading

import pytest

from knowledge.git_memory_loader import GitMemoryLoader, last_hash_path

_GIT = ["git", "-c", "user.name=Rowan", "-c", "user.email=rowan@example.invalid",
        "-c", "commit.gpgsign=false"]


def _git(repo, *args):
    out = subprocess.run(_GIT + list(args), cwd=repo, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _commit(repo, subject):
    _git(repo, "commit", "--allow-empty", "-m", subject)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    return r


class _FakeColl:
    def __init__(self, metas):
        self.metas = metas
        self.calls = []

    def get(self, include=None, limit=None, offset=0, **kw):
        self.calls.append(include)
        assert include == ["metadatas"], "documents must never be loaded"
        return {"metadatas": self.metas[offset:offset + limit]}


class FakeStore:
    def __init__(self, existing_hashes=(), fail_add=False):
        self.coll = _FakeColl([{"full_hash": h, "commit_hash": h[:8]} for h in existing_hashes])
        self.collections = {"procedural": self.coll}
        self.added = []
        self.fail_add = fail_add

    def _get_collection(self, name):
        return self.collections[name]

    def add_batch_to_collection(self, name, texts, metas):
        if self.fail_add:
            raise RuntimeError("chroma write failed")
        assert name == "procedural"
        self.added.extend(metas)
        return ["id"] * len(texts)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "dataroot"
    monkeypatch.setenv("DAEMON_DATA_DIR", str(root))
    return root


def _set_anchor(value):
    os.makedirs(os.path.dirname(last_hash_path()), exist_ok=True)
    with open(last_hash_path(), "w") as f:
        f.write(value)


def _anchor():
    with open(last_hash_path()) as f:
        return f.read().strip()


GONE = "d" * 40  # a hash no repository has


def test_unreachable_anchor_recovers_only_missing_commits(repo, data_root, caplog):
    h1 = _commit(repo, "first change for Tamsin")
    h2 = _commit(repo, "second change")
    h3 = _commit(repo, "third change")
    _set_anchor(GONE)
    store = FakeStore(existing_hashes=[h1, "e" * 40])  # h1 present; one pre-rewrite orphan
    loader = GitMemoryLoader(chroma_store=store, repo_path=str(repo))

    with caplog.at_level(logging.INFO, logger="knowledge.git_memory_loader"):
        n = loader.incremental_update(include_diffs=False)

    assert n == 2
    assert [m["full_hash"] for m in store.added] == [h3, h2]  # newest first, h1 not re-added
    assert _anchor() == h3
    line = next(r.getMessage() for r in caplog.records if "[GitSync]" in r.getMessage())
    assert "anchor unreachable (history rewritten?)" in line
    assert f"re-anchored at {h3[:8]}" in line
    assert "indexed 2" in line and "skipped 1 already present" in line
    assert "1 indexed hashes no longer on HEAD" in line

    # second run: reachable anchor, normal path, nothing new
    again = loader.incremental_update(include_diffs=False)
    assert again == 0
    assert len(store.added) == 2


def test_missing_anchor_file_recovers_not_run_backfill(repo, data_root):
    h1 = _commit(repo, "only commit")
    store = FakeStore()
    loader = GitMemoryLoader(chroma_store=store, repo_path=str(repo))
    assert not os.path.exists(last_hash_path())
    assert loader.incremental_update(include_diffs=False) == 1
    assert [m["full_hash"] for m in store.added] == [h1]
    assert _anchor() == h1


def test_reachable_anchor_keeps_normal_behaviour(repo, data_root):
    h1 = _commit(repo, "old")
    _set_anchor(h1)
    h2 = _commit(repo, "new one")
    store = FakeStore(existing_hashes=[])  # the store would accept h1 again; normal path must not
    loader = GitMemoryLoader(chroma_store=store, repo_path=str(repo))
    assert loader.incremental_update(include_diffs=False) == 1
    assert [m["full_hash"] for m in store.added] == [h2]
    assert _anchor() == h2


def test_cap_defers_anchor_and_drains_backlog_newest_first(repo, data_root, caplog):
    hs = [_commit(repo, f"change {i}") for i in range(5)]
    _set_anchor(GONE)
    store = FakeStore()
    loader = GitMemoryLoader(chroma_store=store, repo_path=str(repo))

    with caplog.at_level(logging.INFO, logger="knowledge.git_memory_loader"):
        assert loader.incremental_update(include_diffs=False, max_commits=2) == 2
    assert [m["full_hash"] for m in store.added] == [hs[4], hs[3]]
    assert _anchor() == GONE  # unchanged: work remains
    assert any("re-anchor deferred: 3 commits still missing (cap 2)" in r.getMessage()
               for r in caplog.records)

    # run 2 (store now holds run 1's commits)
    store2 = FakeStore(existing_hashes=[hs[4], hs[3]])
    loader2 = GitMemoryLoader(chroma_store=store2, repo_path=str(repo))
    assert loader2.incremental_update(include_diffs=False, max_commits=3) == 3
    assert [m["full_hash"] for m in store2.added] == [hs[2], hs[1], hs[0]]
    assert _anchor() == hs[4]

    # run 3: normal path
    assert loader2.incremental_update(include_diffs=False, max_commits=3) == 0
    assert len(store2.added) == 3


def test_cancel_after_first_chunk_keeps_landed_chunk_and_anchor(repo, data_root, monkeypatch):
    import knowledge.git_memory_loader as mod
    monkeypatch.setattr(mod, "_ADD_CHUNK", 2)
    hs = [_commit(repo, f"change {i}") for i in range(5)]
    _set_anchor(GONE)
    cancel = threading.Event()

    class CancelAfterFirst(FakeStore):
        def add_batch_to_collection(self, name, texts, metas):
            out = super().add_batch_to_collection(name, texts, metas)
            cancel.set()
            return out

    store = CancelAfterFirst()
    n = GitMemoryLoader(chroma_store=store, repo_path=str(repo)).incremental_update(
        include_diffs=False, cancel=cancel)
    assert n == 2
    assert [m["full_hash"] for m in store.added] == [hs[4], hs[3]]
    assert _anchor() == GONE


def test_cap_defaults_to_live_config(repo, data_root, monkeypatch):
    import config.app_config as app_config
    hs = [_commit(repo, f"c{i}") for i in range(4)]
    _set_anchor(GONE)
    monkeypatch.setattr(app_config, "GIT_MEMORY_MAX_COMMITS_PER_SYNC", 1)
    store = FakeStore()
    assert GitMemoryLoader(chroma_store=store, repo_path=str(repo)).incremental_update(include_diffs=False) == 1
    assert [m["full_hash"] for m in store.added] == [hs[3]]


def test_anchor_not_advanced_when_store_add_raises(repo, data_root):
    _commit(repo, "a")
    _set_anchor(GONE)
    loader = GitMemoryLoader(chroma_store=FakeStore(fail_add=True), repo_path=str(repo))
    with pytest.raises(RuntimeError):
        loader.incremental_update(include_diffs=False)
    assert _anchor() == GONE
    # retry after the store recovers indexes everything (nothing was half-recorded)
    store = FakeStore()
    assert GitMemoryLoader(chroma_store=store, repo_path=str(repo)).incremental_update(include_diffs=False) == 1


def test_cancelled_run_neither_writes_nor_moves_anchor(repo, data_root):
    _commit(repo, "a")
    _set_anchor(GONE)
    store = FakeStore()
    cancel = threading.Event()
    cancel.set()
    n = GitMemoryLoader(chroma_store=store, repo_path=str(repo)).incremental_update(
        include_diffs=False, cancel=cancel)
    assert n == 0 and store.added == []
    assert _anchor() == GONE


def test_unreadable_collection_fails_closed(repo, data_root):
    _commit(repo, "a")
    _set_anchor(GONE)
    store = FakeStore()
    store.coll.get = lambda **kw: (_ for _ in ()).throw(RuntimeError("cannot read"))
    with pytest.raises(RuntimeError):
        GitMemoryLoader(chroma_store=store, repo_path=str(repo)).incremental_update(include_diffs=False)
    assert _anchor() == GONE


def test_paths_do_not_depend_on_cwd(repo, data_root, tmp_path, monkeypatch):
    h1 = _commit(repo, "a")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    store = FakeStore()
    loader = GitMemoryLoader(chroma_store=store, repo_path=str(repo))
    assert loader.incremental_update(include_diffs=False) == 1
    assert (data_root / "git_memory_last_hash.txt").read_text().strip() == h1
    assert not (elsewhere / "data").exists()


def test_default_repo_root_is_this_checkout_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loader = GitMemoryLoader(chroma_store=FakeStore())
    from pathlib import Path
    import knowledge.git_memory_loader as mod
    assert Path(loader.extractor.repo_path) == Path(mod.__file__).resolve().parents[1]


def test_is_reachable_rejects_non_hash_and_unknown(repo):
    from knowledge.git_memory import GitMemoryExtractor
    h = _commit(repo, "a")
    ex = GitMemoryExtractor(str(repo))
    assert ex.is_reachable(h)
    assert not ex.is_reachable(GONE)
    assert not ex.is_reachable("--output=x")
    assert not ex.is_reachable("")
