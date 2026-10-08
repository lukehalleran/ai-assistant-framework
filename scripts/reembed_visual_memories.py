"""
Re-embed the visual-memory FAISS index with the corrected CLIP architecture (DRY-RUN FIRST).

Background (2026-10-08): the config named `ViT-B-32` with `clip_pretrained: openai`. The OpenAI
weights were trained with the QuickGELU activation, so open_clip builds the plain-GELU arch with a
mismatch warning and every vector in `data/clip_index.faiss` was produced by a slightly different
network than the weights expect. The config now names `ViT-B-32-quickgelu`; vectors already stored
were made with the old arch and should be rebuilt so stored image vectors and freshly encoded text
queries live in the same space.

What is rewritten: ONLY `data/clip_index.faiss` (IndexFlatIP(512)) and `data/clip_metadata.json`.
Row i of the index <-> entry i of the metadata (`faiss_idx == i`); the rebuild keeps that order, so
the Chroma `visual_memories` collection (caption embeddings + `faiss_idx` metadata) stays valid and
is never written. Text queries are encoded fresh at search time (no stored text vectors).

Per row:
  - original image found AND SHA-256 matches `image_hash` -> re-encoded with the deployed
    `CLIPManager.encode_image_from_path`;
  - original missing / hash differs (or none recorded) / encode fails -> the OLD vector is kept
    bit-for-bit and the entry is flagged `clip_reembed_status` = original_missing | hash_mismatch |
    encode_failed. A row is never dropped.

Safety:
  - Default is DRY RUN: reports counts, alignment, originals found/missing/mismatched, configured
    model, and the planned backup dir. Nothing is written.
  - --apply refuses while a live Daemon main.py is detected (fail closed: if the guard cannot be
    imported or raises, the Daemon is treated as running), when the index and metadata are not
    row-aligned, or when more than 10% of originals are missing/mismatched (--force overrides ONLY
    that last check).
  - --apply loads the CLIP manager and aborts with NOTHING written unless it reports
    model_name == ViT-B-32-quickgelu.
  - Before the index changes, copies index + metadata (+ a read-only JSONL export of the Chroma
    `visual_memories` collection, best-effort) to data/backups/reembed_visual_preimage_<ts>/.
  - The new index is written to a unique mkstemp sibling, re-read and checked, then os.replace'd; metadata goes
    through utils.safe_json.atomic_write_json. Nothing is ever deleted.
  - A mean old-vs-new cosine below 0.8 prints a loud WARNING in the receipt; the script does NOT
    auto-revert (restore from the backup dir if you disagree).

Usage:
    python scripts/reembed_visual_memories.py            # dry run (report only)
    python scripts/reembed_visual_memories.py --apply    # backup + rebuild
"""

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

EXPECTED_MODEL = "ViT-B-32-quickgelu"
EMBED_DIM = 512
CHUNK = 16
MAX_BAD_FRACTION = 0.10
COSINE_WARN_BELOW = 0.8


def _daemon_running() -> bool:
    try:
        from utils.daemon_guard import daemon_running
        return daemon_running()
    except Exception as exc:  # fail CLOSED: without the real guard we cannot prove the Daemon is down
        print(f"[daemon-guard] utils.daemon_guard unavailable ({exc!r}) - treating the Daemon as RUNNING; "
              f"--apply is refused. Run from the repo root with the project interpreter.", file=sys.stderr)
        return True


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _resolve_image(raw: str, image_root: Path):
    """Absolute paths as-is; relative ones under the repo root, then under data/uploads/."""
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p if p.is_file() else None
    for cand in (image_root / p, image_root / "data" / "uploads" / p):
        if cand.is_file():
            return cand
    return None


def classify_rows(meta, image_root: Path):
    """Return [(status, resolved_path|None)] per row; status in ok|original_missing|hash_mismatch."""
    out = []
    for entry in meta:
        path = _resolve_image(str(entry.get("image_path") or ""), image_root)
        if path is None:
            out.append(("original_missing", None))
            continue
        want = str(entry.get("image_hash") or "")
        try:
            got = _sha256(path)
        except OSError:
            out.append(("original_missing", None))
            continue
        # No recorded hash = the original cannot be verified; treat like a mismatch (keep old vector).
        out.append(("ok" if want and got == want else "hash_mismatch", path))
    return out


def alignment_problem(meta, ntotal: int):
    """Return a refusal string when index and metadata are not row-aligned, else None."""
    if not isinstance(meta, list) or not all(isinstance(m, dict) for m in meta):
        return "metadata is not a list of entries"
    if len(meta) != ntotal:
        return f"metadata has {len(meta)} entries but the index has {ntotal} vectors"
    for i, m in enumerate(meta):
        if m.get("faiss_idx") != i:
            return f"entry {i} has faiss_idx={m.get('faiss_idx')!r} (expected {i})"
    return None


def export_chroma_visual_memories(dest: Path) -> str:
    """Read-only JSONL export of the Chroma visual_memories collection. Returns a status note."""
    try:
        from config.app_config import CHROMA_PATH
        from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

        store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
        coll = store._get_collection("visual_memories")
        got = coll.get(include=["metadatas", "documents"])
        ids = got.get("ids") or []
        metas = got.get("metadatas") or []
        docs = got.get("documents") or []
        with open(dest, "w", encoding="utf-8") as f:
            for i, m, d in zip(ids, metas, docs):
                f.write(json.dumps({"id": i, "metadata": m, "content": d}, default=str) + "\n")
        return f"exported {len(ids)} visual_memories documents to {dest}"
    except Exception as exc:  # degrades: no Chroma export in the backup (index + metadata copies still made)
        return f"SKIPPED Chroma export ({exc!r}); index + metadata backups are unaffected"


def _paths(index_arg, meta_arg):
    from config.app_config import (
        VISUAL_MEMORY_CLIP_MODEL,
        VISUAL_MEMORY_CLIP_PRETRAINED,
        VISUAL_MEMORY_INDEX_PATH,
        VISUAL_MEMORY_META_PATH,
    )
    index_path = Path(index_arg or VISUAL_MEMORY_INDEX_PATH)
    meta_path = Path(meta_arg or VISUAL_MEMORY_META_PATH)
    if not index_path.is_absolute():
        index_path = REPO_ROOT / index_path
    if not meta_path.is_absolute():
        meta_path = REPO_ROOT / meta_path
    return index_path, meta_path, VISUAL_MEMORY_CLIP_MODEL, VISUAL_MEMORY_CLIP_PRETRAINED


def main(argv=None, *, manager_factory=None, chroma_exporter=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--apply", action="store_true",
                    help="Actually rebuild (after a backup). Default: dry run.")
    ap.add_argument("--force", action="store_true",
                    help="Proceed although more than 10%% of originals are missing/mismatched.")
    ap.add_argument("--index", help="FAISS index path (default: VISUAL_MEMORY_INDEX_PATH).")
    ap.add_argument("--meta", help="Metadata JSON path (default: VISUAL_MEMORY_META_PATH).")
    ap.add_argument("--image-root", help="Base dir for relative image_path values (default: repo root).")
    ap.add_argument("--backup-root", help="Where reembed_visual_preimage_<ts>/ is created "
                                          "(default: <repo>/data/backups).")
    args = ap.parse_args(argv)

    # Guard FIRST, before anything reads or consumes --apply (fail closed via _daemon_running).
    if args.apply and _daemon_running():
        print("ABORT: a live Daemon main.py process is running (or could not be ruled out). "
              "Stop it first; it holds the visual store in memory and would re-save over this rebuild.")
        return 1
    return _run(args, manager_factory, chroma_exporter)


def _run(args, manager_factory, chroma_exporter) -> int:
    import numpy as np
    try:
        import faiss
    except ImportError:
        print("ABORT: faiss is not installed in this interpreter.")
        return 1

    index_path, meta_path, cfg_model, cfg_pretrained = _paths(args.index, args.meta)
    image_root = Path(args.image_root) if args.image_root else REPO_ROOT
    backup_root = Path(args.backup_root) if args.backup_root else REPO_ROOT / "data" / "backups"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root / f"reembed_visual_preimage_{ts}"

    if not index_path.is_file() or not meta_path.is_file():
        print(f"ABORT: index ({index_path}) or metadata ({meta_path}) not found.")
        return 1
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        old_index = faiss.read_index(str(index_path))
    except Exception as exc:  # reported + refused; nothing is written
        print(f"ABORT: could not read the visual store ({exc!r}).")
        return 1

    ntotal = int(old_index.ntotal)
    problem = alignment_problem(meta, ntotal)
    dim_ok = int(old_index.d) == EMBED_DIM
    statuses = (classify_rows(meta, image_root)
                if isinstance(meta, list) and all(isinstance(m, dict) for m in meta) else [])
    counts = {"ok": 0, "original_missing": 0, "hash_mismatch": 0}
    for s, _ in statuses:
        counts[s] += 1
    bad = counts["original_missing"] + counts["hash_mismatch"]
    bad_fraction = (bad / len(statuses)) if statuses else 0.0

    print(f"Index:    {index_path}  ({ntotal} vectors, dim {old_index.d})")
    print(f"Metadata: {meta_path}  ({len(meta) if isinstance(meta, list) else '?'} entries)")
    print(f"Alignment: {'OK' if problem is None and dim_ok else 'PROBLEM - ' + (problem or f'index dim {old_index.d} != {EMBED_DIM}')}")
    print(f"Originals: {counts['ok']} verified, {counts['original_missing']} missing, "
          f"{counts['hash_mismatch']} hash-mismatch/unverifiable ({bad_fraction:.1%} not re-embeddable)")
    print(f"Configured model: {cfg_model} (pretrained={cfg_pretrained}); required: {EXPECTED_MODEL}")
    print(f"Planned backup dir: {backup_dir}")
    print(f"Planned targets:    {index_path.name}, {meta_path.name} (Chroma is never written)")

    if not args.apply:
        print("\nDRY RUN - nothing was written. Re-run with --apply (Daemon stopped) to rebuild.")
        return 0

    # ---- refusals (nothing written before this point) ----
    if problem is not None or not dim_ok:
        print(f"ABORT: {problem or f'index dim {old_index.d} != {EMBED_DIM}'}. Nothing was written.")
        return 1
    if cfg_model != EXPECTED_MODEL:
        print(f"ABORT: configured clip_model is {cfg_model!r}, expected {EXPECTED_MODEL!r}. Nothing was written.")
        return 1
    if bad_fraction > MAX_BAD_FRACTION and not args.force:
        print(f"ABORT: {bad_fraction:.1%} of originals are missing/mismatched (> {MAX_BAD_FRACTION:.0%}). "
              f"Investigate, or pass --force. Nothing was written.")
        return 1

    # ---- load the deployed encoder BEFORE writing anything ----
    if manager_factory is None:
        from knowledge.clip_manager import get_clip_manager as manager_factory  # lazy import: heavy model stack
    mgr = manager_factory()
    mgr.load()
    if getattr(mgr, "model_name", None) != EXPECTED_MODEL:
        print(f"ABORT: CLIP manager reports model_name={getattr(mgr, 'model_name', None)!r} "
              f"(loaded={getattr(mgr, 'loaded', None)}), expected {EXPECTED_MODEL!r}. Nothing was written.")
        return 1

    # ---- backup (before the index changes) ----
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(index_path, backup_dir / index_path.name)
    shutil.copy2(meta_path, backup_dir / meta_path.name)
    note = (chroma_exporter or export_chroma_visual_memories)(backup_dir / "visual_memories_chroma.jsonl")
    print(f"Backup written to {backup_dir}\n  {note}")

    # ---- re-embed in order ----
    old_vecs = np.zeros((ntotal, EMBED_DIM), dtype=np.float32)
    if ntotal:
        old_vecs[:] = old_index.reconstruct_n(0, ntotal)
    new_vecs = old_vecs.copy()
    new_meta = []
    reembedded = []
    for start in range(0, ntotal, CHUNK):
        for i in range(start, min(start + CHUNK, ntotal)):
            entry = dict(meta[i])
            entry.pop("clip_reembed_status", None)
            status, path = statuses[i]
            if status == "ok":
                vec = mgr.encode_image_from_path(str(path))
                vec = None if vec is None else np.asarray(vec, dtype=np.float32).reshape(-1)
                if vec is None or vec.shape[0] != EMBED_DIM or not np.isfinite(vec).all():
                    status = "encode_failed"
                else:
                    new_vecs[i] = vec
                    entry["clip_model"] = cfg_model
                    entry["clip_pretrained"] = cfg_pretrained
                    reembedded.append(i)
            if status != "ok":
                entry["clip_reembed_status"] = status
            new_meta.append(entry)
        gc.collect()

    # ---- write new index (tmp -> verify -> replace), then metadata ----
    new_index = faiss.IndexFlatIP(EMBED_DIM)
    if ntotal:
        new_index.add(np.ascontiguousarray(new_vecs))
    # Unique sibling (same filesystem, so os.replace is atomic); never a
    # derived "<path>.tmp" another writer could share (atomic-writer guard).
    fd, tmp_name = tempfile.mkstemp(prefix=f".{index_path.name}.", suffix=".partial",
                                    dir=str(index_path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    faiss.write_index(new_index, str(tmp))
    check = faiss.read_index(str(tmp))
    if check.ntotal != ntotal or check.d != EMBED_DIM:
        tmp.unlink()
        print(f"ABORT: rebuilt index failed verification (ntotal={check.ntotal}, d={check.d}). "
              f"Originals untouched; temp file removed.")
        return 1
    os.replace(tmp, index_path)
    from utils.safe_json import atomic_write_json
    atomic_write_json(str(meta_path), new_meta, ensure_ascii=True)

    # ---- receipt ----
    cos = None
    if reembedded:
        a = old_vecs[reembedded]
        b = new_vecs[reembedded]
        denom = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        cos = (a * b).sum(axis=1) / np.where(denom == 0, 1.0, denom)
    flagged = [(i, m["clip_reembed_status"]) for i, m in enumerate(new_meta) if "clip_reembed_status" in m]
    print("\nRECEIPT")
    print(f"  rows: {ntotal}  re-embedded: {len(reembedded)}  kept old vector: {len(flagged)}")
    print(f"  flagged rows: {flagged if flagged else 'none'}")
    if cos is not None:
        print(f"  old-vs-new cosine over re-embedded rows: mean {float(cos.mean()):.4f}  min {float(cos.min()):.4f}")
        if float(cos.mean()) < COSINE_WARN_BELOW:
            print(f"  WARNING: mean old-vs-new cosine {float(cos.mean()):.4f} < {COSINE_WARN_BELOW} - the new "
                  f"vectors differ more than an activation fix explains. The index HAS been replaced; "
                  f"restore from {backup_dir} if you disagree.")
    print(f"  backup: {backup_dir}")
    print("  Restart the Daemon afterwards so the visual store reloads the new index.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
