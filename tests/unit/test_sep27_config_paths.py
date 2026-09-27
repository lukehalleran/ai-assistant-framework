"""2026-09-27: owner-neutral wiki data root + CorpusManager empty-path crash.

- The wiki FAISS root used to default to an owner mount path hardcoded in
  knowledge/semantic_search.py and two build scripts (BC-59). It is now the
  config key wiki.data_root (env WIKI_DATA_ROOT still wins).
- CorpusManager(corpus_file="") raised NameError: CORPUS_FILE was imported only
  inside `if corpus_file is None:` (latent since 09-16, patch held for re-review).
"""

import importlib
from pathlib import Path

import config.app_config as app_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_no_owner_mount_path_in_wiki_code():
    for rel in ("knowledge/semantic_search.py", "scripts/build_faiss_index.py",
                "scripts/build_wiki_subset.py", "config/app_config.py"):
        text = (_repo_root() / rel).read_text()
        assert "/run/media/" not in text, rel


def test_wiki_data_root_env_overrides_config(monkeypatch, tmp_path):
    monkeypatch.setenv("WIKI_DATA_ROOT", str(tmp_path))
    reloaded = importlib.reload(app_config)
    try:
        assert reloaded.WIKI_DATA_ROOT == str(tmp_path)
    finally:
        monkeypatch.delenv("WIKI_DATA_ROOT", raising=False)
        importlib.reload(app_config)


def test_wiki_data_root_default_is_neutral(monkeypatch):
    monkeypatch.delenv("WIKI_DATA_ROOT", raising=False)
    root = importlib.reload(app_config).WIKI_DATA_ROOT
    assert root and "~" not in root  # expanded, and a string path


def test_empty_string_corpus_file_falls_back_to_config(monkeypatch, tmp_path):
    from memory.corpus_manager import CorpusManager

    fallback = tmp_path / "fallback_corpus.json"
    monkeypatch.setattr(app_config, "CORPUS_FILE", str(fallback))
    mgr = CorpusManager(corpus_file="")
    assert mgr.corpus_file == str(fallback)


def test_none_corpus_file_unchanged(monkeypatch, tmp_path):
    from memory.corpus_manager import CorpusManager

    fallback = tmp_path / "fallback_corpus.json"
    monkeypatch.setattr(app_config, "CORPUS_FILE", str(fallback))
    mgr = CorpusManager(corpus_file=None)
    assert mgr.corpus_file == str(fallback)
