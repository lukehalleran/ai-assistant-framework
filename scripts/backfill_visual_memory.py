#!/usr/bin/env python3
"""
Backfill existing uploaded images into the CLIP visual memory store.

Scans data/uploads/ for images that aren't yet in the visual memory index,
CLIP-embeds them, and optionally generates captions via vision LLM.

Usage:
    python scripts/backfill_visual_memory.py                  # dry-run (default)
    python scripts/backfill_visual_memory.py --execute        # actually ingest
    python scripts/backfill_visual_memory.py --execute --caption   # with LLM captions
    python scripts/backfill_visual_memory.py --execute --obsidian  # also scan Obsidian vault
"""

import argparse
import asyncio
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}


def find_upload_images(upload_dir: str) -> list[str]:
    """Find all image files in the uploads directory."""
    if not os.path.isdir(upload_dir):
        print(f"  Upload directory not found: {upload_dir}")
        return []

    images = []
    for fname in sorted(os.listdir(upload_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            images.append(os.path.join(upload_dir, fname))
    return images


def find_obsidian_images(vault_path: str) -> list[str]:
    """Find all image files in the Obsidian vault."""
    if not os.path.isdir(vault_path):
        print(f"  Obsidian vault not found: {vault_path}")
        return []

    images = []
    for root, _dirs, files in os.walk(vault_path):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                images.append(os.path.join(root, fname))
    return images


async def run_backfill(args):
    from knowledge.clip_manager import get_clip_manager
    from knowledge.visual_memory_store import VisualMemoryStore
    from knowledge.visual_memory_pipeline import VisualMemoryPipeline

    # Collect images to process
    upload_dir = os.path.join("data", "uploads")
    all_images = find_upload_images(upload_dir)
    source_map = {img: "upload" for img in all_images}

    if args.obsidian:
        try:
            from config.app_config import OBSIDIAN_VAULT_PATH
            vault_path = os.path.expanduser(OBSIDIAN_VAULT_PATH)
        except ImportError:
            vault_path = os.path.expanduser("~/Documents/Luke Notes")

        obsidian_images = find_obsidian_images(vault_path)
        for img in obsidian_images:
            if img not in source_map:
                source_map[img] = "obsidian"
        all_images.extend(obsidian_images)

    print(f"\nFound {len(all_images)} images ({len([v for v in source_map.values() if v == 'upload'])} uploads, {len([v for v in source_map.values() if v == 'obsidian'])} obsidian)")

    if not all_images:
        print("Nothing to do.")
        return

    # Initialize CLIP + store
    clip = get_clip_manager()

    if not args.execute:
        # Dry run — just check what would be ingested
        store = VisualMemoryStore(data_dir="data")
        store.load()
        existing = store.get_stats()["total_images"]

        new_count = 0
        for img_path in all_images:
            img_hash = VisualMemoryPipeline._compute_hash(img_path)
            if not store.has_hash(img_hash):
                new_count += 1
                print(f"  [NEW] {os.path.basename(img_path)} ({source_map[img_path]})")
            else:
                print(f"  [SKIP] {os.path.basename(img_path)} (already indexed)")

        print(f"\nDry run: {new_count} new images to ingest, {existing} already indexed")
        print("Run with --execute to actually ingest.")
        return

    # Execute mode
    print("\nLoading CLIP model...")
    clip.load()
    if not clip.loaded:
        print("ERROR: CLIP model failed to load. Is open_clip_torch installed?")
        return

    model_manager = None
    if args.caption:
        try:
            from models.model_manager import ModelManager
            model_manager = ModelManager()
            # Ensure a multimodal model is active for captioning
            try:
                from config.app_config import VISUAL_MEMORY_CAPTION_MODEL
                caption_model = VISUAL_MEMORY_CAPTION_MODEL
            except ImportError:
                caption_model = "gpt-4o-mini"
            model_manager.switch_model(caption_model)
            print(f"Vision LLM captioning enabled (model: {caption_model})")
        except Exception as e:
            print(f"Warning: Could not init ModelManager for captioning: {e}")
            print("Proceeding without captions (filename-only)")
            model_manager = None

    store = VisualMemoryStore(data_dir="data")
    pipeline = VisualMemoryPipeline(clip, store, model_manager=model_manager)

    ingested = 0
    skipped = 0
    failed = 0
    t_start = time.time()

    for i, img_path in enumerate(all_images, start=1):
        source = source_map[img_path]
        basename = os.path.basename(img_path)
        try:
            doc_id = await pipeline.ingest_image(img_path, source=source)
            if doc_id:
                ingested += 1
                print(f"  [{i}/{len(all_images)}] Ingested: {basename} ({source})")
            else:
                skipped += 1
                print(f"  [{i}/{len(all_images)}] Skipped: {basename} (duplicate)")
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(all_images)}] Failed: {basename} — {e}")

    elapsed = time.time() - t_start
    print(f"\nBackfill complete in {elapsed:.1f}s: {ingested} ingested, {skipped} skipped, {failed} failed")
    print(f"Total visual memories: {store.get_stats()['total_images']}")


def main():
    parser = argparse.ArgumentParser(description="Backfill existing images into CLIP visual memory")
    parser.add_argument("--execute", action="store_true", help="Actually ingest (default is dry-run)")
    parser.add_argument("--caption", action="store_true", help="Generate captions via vision LLM (slower, costs API credits)")
    parser.add_argument("--obsidian", action="store_true", help="Also scan Obsidian vault for images")
    args = parser.parse_args()

    if not args.execute:
        print("=== DRY RUN (use --execute to actually ingest) ===")

    asyncio.run(run_backfill(args))


if __name__ == "__main__":
    main()
