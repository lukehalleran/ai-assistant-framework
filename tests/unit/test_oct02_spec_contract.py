"""Packaging contract for daemon.spec + scripts/stage_frozen_models.py (BC-71, BC-82).

The spec is parsed as TEXT (no PyInstaller import); the staging script is
driven against a fake HF cache in tmp_path.
"""
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = (ROOT / "daemon.spec").read_text()


def _load_stage():
    spec = importlib.util.spec_from_file_location(
        "stage_frozen_models", ROOT / "scripts" / "stage_frozen_models.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stage_frozen_models"] = mod
    spec.loader.exec_module(mod)
    return mod


stage_mod = _load_stage()


# ---------------------------------------------------------------- spec text
def test_spec_bundles_spa_with_build_guard():
    assert "('web/dist', 'web/dist')" in SPEC
    guard = SPEC.index("web/dist/index.html")
    raise_at = SPEC.index("raise SystemExit", guard - 400)
    assert raise_at < SPEC.index("datas = [")
    assert "npm run build" in SPEC[raise_at:raise_at + 300]


def test_spec_collects_api_package():
    assert "'api'" in SPEC and "collect_submodules(pkg)" in SPEC


def test_spec_bundles_staged_models_and_tiktoken():
    assert "'build/models', 'models'" in SPEC
    assert "'build/tiktoken', 'tiktoken'" in SPEC


def test_spec_no_stale_hiddenimports():
    for stale in ("pkg_resources.py2_warn", "pkg_resources._vendor",
                  "numpy.core._multiarray_umath", "numpy.core._dtype_ctypes"):
        assert stale not in SPEC
    assert "'integrations'" in SPEC  # still imported by core/orchestrator.py


def test_spec_data_packages_are_in_requirements():
    reqs = (ROOT / "requirements.txt").read_text().lower()
    assert "python-docx" in reqs and "collect_data_files('docx')" in SPEC
    assert "pdfplumber" in reqs and "collect_data_files('pdfminer')" in SPEC
    assert "googleapiclient" not in SPEC  # not a dependency (raw REST)


def test_spec_does_not_write_tracked_hooks():
    writes = re.findall(r"open\(([^)]*),\s*'w'\)", SPEC)
    assert writes, "expected the generated runtime hook write"
    for target in writes:
        assert target.strip() == "runtime_hook_path"
    assert "'build', 'runtime_hook_generated.py'" in SPEC
    assert "'hooks', 'runtime_hook.py'" not in SPEC


# ------------------------------------------------------------ staging script
def _fake_cache(root: Path, models=stage_mod.MODELS, extra_snapshot=True):
    for m in models:
        repo = root / stage_mod.repo_dirname(m)
        blobs = repo / "blobs"
        snap = repo / "snapshots" / "abc123"
        blobs.mkdir(parents=True)
        snap.mkdir(parents=True)
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text("abc123\n")
        (blobs / "b1").write_bytes(b"config")
        (blobs / "b2").write_bytes(b"weights" * 10)
        (blobs / "b3").write_bytes(b"ptbin")
        (snap / "config.json").symlink_to(blobs / "b1")
        (snap / "model.safetensors").symlink_to(blobs / "b2")
        (snap / "pytorch_model.bin").symlink_to(blobs / "b3")
        (snap / "onnx").mkdir()
        (snap / "onnx" / "model.onnx").write_bytes(b"x")
        if extra_snapshot:  # a stale snapshot refs/main does NOT point to
            old = repo / "snapshots" / "old999"
            old.mkdir()
            (old / "config.json").write_text("stale")


def test_stage_apply_layout_and_refs(tmp_path):
    cache = tmp_path / "hub"
    _fake_cache(cache)
    out = tmp_path / "build"
    logs = []
    total = stage_mod.stage(cache, out / "models", out / "tiktoken", True, log=logs.append)
    assert total > 0
    for m in stage_mod.MODELS:
        repo = out / "models" / stage_mod.repo_dirname(m)
        assert (repo / "refs" / "main").read_text() == "abc123"
        snap = repo / "snapshots" / "abc123"
        assert (snap / "config.json").read_bytes() == b"config"
        assert not (snap / "config.json").is_symlink()  # symlinks resolved
        assert (snap / "model.safetensors").is_file()
        assert not (snap / "pytorch_model.bin").exists()  # redundant format
        assert not (snap / "onnx").exists()
        assert not (repo / "snapshots" / "old999").exists()  # only refs/main's


def test_stage_dry_run_writes_nothing(tmp_path):
    cache = tmp_path / "hub"
    _fake_cache(cache)
    out = tmp_path / "build"
    logs = []
    stage_mod.stage(cache, out / "models", out / "tiktoken", False, log=logs.append)
    assert not out.exists()
    assert any("would copy" in line for line in logs)


def test_stage_missing_model_is_actionable_and_copies_nothing(tmp_path):
    cache = tmp_path / "hub"
    _fake_cache(cache, models=stage_mod.MODELS[:2])  # cross-encoder absent
    out = tmp_path / "build"
    with pytest.raises(stage_mod.StagingError) as exc:
        stage_mod.stage(cache, out / "models", out / "tiktoken", True, log=lambda _: None)
    assert "ms-marco-MiniLM-L-6-v2" in str(exc.value)
    assert "snapshot_download" in str(exc.value)
    assert not out.exists()  # resolution happens before any copy


def test_stage_tiktoken_found_by_url_hash(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    _fake_cache(cache)
    tik = tmp_path / "tikcache"
    tik.mkdir()
    import hashlib
    key = hashlib.sha1(stage_mod.TIKTOKEN_URL.encode()).hexdigest()
    (tik / key).write_bytes(b"ranks")
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tik))
    out = tmp_path / "build"
    stage_mod.stage(cache, out / "models", out / "tiktoken", True, log=lambda _: None)
    assert (out / "tiktoken" / key).read_bytes() == b"ranks"


def test_stage_cli_missing_model_exit_code(tmp_path):
    cache = tmp_path / "hub"
    cache.mkdir()
    r = subprocess.run(
        [sys.executable, "-s", str(ROOT / "scripts" / "stage_frozen_models.py"),
         "--cache", str(cache), "--out", str(tmp_path / "b")],
        capture_output=True, text=True)
    assert r.returncode == 1 and "ERROR" in r.stderr
    assert not (tmp_path / "b").exists()


def test_frozen_bootstrap_sets_tiktoken_cache_dir_and_matches_staged_hf_layout(
        tmp_path, monkeypatch):
    import os
    import utils.bootstrap as bs
    app = tmp_path / "Daemon"
    (app / "_internal" / "models").mkdir(parents=True)
    (app / "_internal" / "tiktoken").mkdir()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(bs, "IS_FROZEN", True)
    monkeypatch.setattr(bs, "IS_WINDOWS", False)
    monkeypatch.setattr(bs, "IS_MACOS", False)
    monkeypatch.setattr(sys, "executable", str(app / "Daemon"))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("APPDATA", raising=False)
    # register for restore: setup_environment setdefaults many keys
    for k in ("TIKTOKEN_CACHE_DIR", "HF_HOME", "TRANSFORMERS_CACHE", "HF_HUB_CACHE",
              "HF_HUB_OFFLINE", "VISUAL_MEMORY_ENABLED", "WIKI_ENABLED", "PROMPT_MAX_WIKI",
              "DAEMON_EXTERNAL_DATA"):
        monkeypatch.delenv(k, raising=False)
    # other keys it may set: snapshot + restore the whole environ
    monkeypatch.setattr(os, "environ", os.environ.copy())
    bs.setup_environment()
    assert os.environ["TIKTOKEN_CACHE_DIR"] == str(app / "_internal" / "tiktoken")
    # staged dirs are models--org--name directly under models/, so HF_HUB_CACHE
    # (not just HF_HOME, which appends /hub) must point at models/
    assert os.environ["HF_HUB_CACHE"] == str(app / "_internal" / "models")


def test_frozen_bootstrap_skips_tiktoken_dir_when_not_staged(tmp_path, monkeypatch):
    import os
    import utils.bootstrap as bs
    app = tmp_path / "Daemon"
    (app / "_internal").mkdir(parents=True)
    monkeypatch.setattr(bs, "IS_FROZEN", True)
    monkeypatch.setattr(bs, "IS_WINDOWS", False)
    monkeypatch.setattr(bs, "IS_MACOS", False)
    monkeypatch.setattr(sys, "executable", str(app / "Daemon"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(os, "environ", os.environ.copy())
    os.environ.pop("TIKTOKEN_CACHE_DIR", None)
    bs.setup_environment()
    assert "TIKTOKEN_CACHE_DIR" not in os.environ


def test_apply_refused_while_daemon_running(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    _fake_cache(cache)
    monkeypatch.setattr(stage_mod, "_daemon_running", lambda: True)
    rc = stage_mod.main(["--apply", "--cache", str(cache), "--out", str(tmp_path / "b")])
    assert rc == 1 and not (tmp_path / "b").exists()
    rc = stage_mod.main(["--cache", str(cache), "--out", str(tmp_path / "b")])  # dry-run ok
    assert rc == 0 and not (tmp_path / "b").exists()
