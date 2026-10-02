#!/usr/bin/env python3
"""Stage the embedding models + tiktoken encoding for the PyInstaller build.

Copies the three HuggingFace snapshots Daemon needs at runtime from the local
HF cache into ``build/models`` (HF-cache layout, which is exactly what the
frozen bootstrap points ``HF_HUB_CACHE`` at) and the tiktoken ``cl100k_base``
file into ``build/tiktoken``. ``daemon.spec`` adds both dirs to the bundle.

Dry-run by default (reports what it would copy and sizes); ``--apply`` copies.
Only the snapshot ``refs/main`` points to is copied, symlinks resolved, and
redundant weight formats (onnx/openvino/tf/flax/rust, and pytorch_model.bin
when safetensors exists) are skipped.

Usage:
    python scripts/stage_frozen_models.py            # dry-run
    python scripts/stage_frozen_models.py --apply
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MODELS = (
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

# tiktoken names its cache file sha1(url); cl100k_base is the fallback encoding.
TIKTOKEN_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

_SKIP_DIRS = {"onnx", "openvino", ".git"}
_SKIP_SUFFIXES = (".onnx", ".h5", ".msgpack", ".ot")


class StagingError(RuntimeError):
    """A required model is missing from the local cache."""


def default_hf_cache() -> Path:
    env = os.environ.get("HF_HUB_CACHE")
    if env:
        return Path(env)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def repo_dirname(model: str) -> str:
    return "models--" + model.replace("/", "--")


def resolve_snapshot(cache: Path, model: str) -> tuple[Path, str]:
    """Return (snapshot dir, commit hash) for ``refs/main``; raise if absent."""
    repo = cache / repo_dirname(model)
    ref = repo / "refs" / "main"
    hint = (
        f"Model {model!r} is not in the HF cache {cache}. Fetch it once with "
        f"`python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{model}')\"` (network needed), then re-run."
    )
    if not ref.is_file():
        raise StagingError(hint)
    commit = ref.read_text().strip()
    snap = repo / "snapshots" / commit
    if not commit or not snap.is_dir():
        raise StagingError(hint)
    return snap, commit


def _snapshot_files(snap: Path) -> list[tuple[Path, Path]]:
    """(source, relative) pairs for the files worth shipping."""
    has_safetensors = any(snap.glob("*.safetensors"))
    out: list[tuple[Path, Path]] = []
    for root, dirs, files in os.walk(snap):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in files:
            if name.endswith(_SKIP_SUFFIXES):
                continue
            if name == "pytorch_model.bin" and has_safetensors:
                continue
            src = Path(root) / name
            out.append((src, src.relative_to(snap)))
    return out


def find_tiktoken_file() -> Path | None:
    key = hashlib.sha1(TIKTOKEN_URL.encode()).hexdigest()
    dirs = [os.environ.get("TIKTOKEN_CACHE_DIR"), os.environ.get("DATA_GYM_CACHE_DIR")]
    dirs.append(os.path.join("/tmp", "data-gym-cache"))
    import tempfile
    dirs.append(os.path.join(tempfile.gettempdir(), "data-gym-cache"))
    for d in dirs:
        if d and (Path(d) / key).is_file():
            return Path(d) / key
    return None


def _mb(n: int) -> str:
    return f"{n / 1_048_576:.1f} MB"


def stage(cache: Path, out_models: Path, out_tiktoken: Path, apply: bool, log=print) -> int:
    """Stage everything; returns total bytes. Raises StagingError on a missing model."""
    plans = []
    for model in MODELS:  # resolve ALL first so a missing model fails before any copy
        snap, commit = resolve_snapshot(cache, model)
        plans.append((model, snap, commit, _snapshot_files(snap)))

    total = 0
    verb = "copy" if apply else "would copy"
    for model, snap, commit, files in plans:
        size = sum(os.path.getsize(s) for s, _ in files)  # follows symlinks
        total += size
        log(f"{verb} {model} @ {commit[:8]}: {len(files)} files, {_mb(size)}")
        if not apply:
            continue
        dest_repo = out_models / repo_dirname(model)
        dest_snap = dest_repo / "snapshots" / commit
        for src, rel in files:
            dst = dest_snap / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)  # copyfile follows symlinks -> real bytes
        (dest_repo / "refs").mkdir(parents=True, exist_ok=True)
        (dest_repo / "refs" / "main").write_text(commit)

    tik = find_tiktoken_file()
    if tik is None:
        log("tiktoken cl100k_base: not found in a local cache (skipped; the "
            "frozen app would try a network download on first use).")
    else:
        size = tik.stat().st_size
        total += size
        log(f"{verb} tiktoken cl100k_base ({tik.name}): {_mb(size)}")
        if apply:
            out_tiktoken.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tik, out_tiktoken / tik.name)

    log(f"total: {_mb(total)}" + ("" if apply else "  (dry-run; pass --apply to copy)"))
    return total


def _daemon_running() -> bool:
    try:
        from utils.daemon_guard import daemon_running
        return daemon_running()
    except Exception as exc:  # fail CLOSED: cannot prove the Daemon is down
        print(f"[daemon-guard] utils.daemon_guard unavailable ({exc!r}) - treating the "
              f"Daemon as RUNNING; --apply is refused. Run from the repo root.",
              file=sys.stderr)
        return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="actually copy (default: dry-run)")
    ap.add_argument("--cache", type=Path, default=None, help="HF hub cache dir")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "build", help="build dir")
    args = ap.parse_args(argv)
    if args.apply and _daemon_running():
        print("Refusing --apply: a live Daemon main.py is detected (the copy competes "
              "for RAM/IO with a build). Shut it down first; dry-run is always allowed.",
              file=sys.stderr)
        return 1
    try:
        stage(args.cache or default_hf_cache(), args.out / "models",
              args.out / "tiktoken", args.apply)
    except StagingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
