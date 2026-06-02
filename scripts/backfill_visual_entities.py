"""
Backfill entity tags on existing visual memory images.

Re-captions images with profile-aware context (pet names, people) so the
vision LLM outputs proper entity names instead of generic "black cat."
Then extracts entities from the enriched captions.

Usage:
    python scripts/backfill_visual_entities.py           # dry-run (default)
    python scripts/backfill_visual_entities.py --execute # write changes
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.app_config import VISUAL_MEMORY_META_PATH
from memory.entity_resolver import EntityResolver
from memory.graph_utils import extract_graph_entities
from utils.logging_utils import get_logger

logger = get_logger("backfill_visual_entities")

# Profile context injected into the captioning prompt so the LLM can name
# entities it recognizes (pets, people). Kept OUT of source — no personal data
# ships in the repo. Provide your own at data/visual_profile_context.txt
# (gitignored) or via the VISUAL_PROFILE_CONTEXT_PATH env var. Empty yields
# generic captions with no personal entity names.
def _load_profile_context() -> str:
    path = os.getenv("VISUAL_PROFILE_CONTEXT_PATH") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "visual_profile_context.txt",
    )
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


PROFILE_CONTEXT = _load_profile_context()

_context_block = (PROFILE_CONTEXT + "\n\n") if PROFILE_CONTEXT else ""
RECAPTION_PROMPT = (
    "You are re-captioning a personal photo for a memory system.\n\n"
    f"{_context_block}"
    "Describe this image in 2-3 sentences. Use proper names for any recognized "
    "people or pets named in the context above.\n"
    "Be specific about details (colors, setting, activity, expressions)."
)


async def recaption_image(model_manager, image_path: str) -> str:
    """Generate a profile-aware caption for an image via generate_async (vision)."""
    import base64

    if not os.path.exists(image_path):
        return ""

    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()

    # Detect media type
    ext = os.path.splitext(image_path)[1].lower()
    media_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    media_type = media_map.get(ext, "image/jpeg")

    images = [{"data": img_data, "media_type": media_type, "filename": os.path.basename(image_path)}]

    try:
        full_response = ""
        gen = await asyncio.wait_for(
            model_manager.generate_async(
                RECAPTION_PROMPT,
                system_prompt="You are an image description assistant for a personal photo archive.",
                max_tokens=200,
                images=images,
            ),
            timeout=15.0,
        )
        async for chunk in gen:
            if isinstance(chunk, str):
                full_response += chunk
            elif hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    full_response += delta.content
        return full_response.strip()
    except Exception as e:
        print(f"    Caption failed: {e}")
        return ""


async def main_async(args):
    meta_path = VISUAL_MEMORY_META_PATH
    if not os.path.exists(meta_path):
        print(f"No metadata file found at {meta_path}")
        return

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    print(f"Loaded {len(metadata)} images from {meta_path}")

    # Initialize entity resolver
    from memory.graph_memory import GraphMemory
    from config.app_config import KNOWLEDGE_GRAPH_PERSIST_PATH, KNOWLEDGE_GRAPH_ALIASES_PATH
    graph = GraphMemory(persist_path=KNOWLEDGE_GRAPH_PERSIST_PATH)
    resolver = EntityResolver(graph_memory=graph, aliases_path=KNOWLEDGE_GRAPH_ALIASES_PATH)

    # Build personal entity set (non-wikidata nodes only)
    personal_entities = set()
    for nid in graph.graph.nodes():
        node = graph.get_entity(nid)
        if node and not getattr(node, 'is_wikidata', False):
            personal_entities.add(nid)
            dn = getattr(node, 'display_name', '')
            if dn:
                personal_entities.add(dn.lower())

    print(f"Graph: {graph.node_count()} nodes, {len(personal_entities)} personal entities")

    # Initialize model manager for re-captioning (needs active vision model)
    from models.model_manager import ModelManager
    mm = ModelManager()
    mm.active_model_name = "gpt-4o-mini"  # Vision-capable, cheap

    updated = 0
    for i, entry in enumerate(metadata):
        image_path = entry.get("image_path", "")
        old_caption = entry.get("caption", "")
        existing = entry.get("entity_ids", [])

        if not image_path or not os.path.exists(image_path):
            print(f"  [{i+1}] SKIP (file not found): {image_path}")
            continue

        # Re-caption with profile context
        print(f"  [{i+1}] {os.path.basename(image_path)}")
        print(f"       Old caption: {old_caption[:70]}...")
        new_caption = await recaption_image(mm, image_path)

        if not new_caption:
            print(f"       SKIP (captioning failed)")
            continue

        print(f"       New caption: {new_caption[:70]}...")

        # Extract entities from new caption
        raw_entities = extract_graph_entities(new_caption, resolver, graph_memory=graph)
        entities = [e for e in raw_entities if e in personal_entities]
        print(f"       Entities:    {entities}")
        print()

        if args.execute:
            entry["caption"] = new_caption
            entry["entity_ids"] = entities
        updated += 1

    print(f"\n{'=' * 50}")
    print(f"Total: {len(metadata)} images, {updated} re-captioned")

    if args.execute and updated > 0:
        # Write metadata
        tmp_path = meta_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(metadata, f, indent=2)
        os.replace(tmp_path, meta_path)
        print(f"Written to {meta_path}")

        # Update ChromaDB
        try:
            from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
            chroma = MultiCollectionChromaStore()
            chroma_updated = 0
            for entry in metadata:
                doc_id = entry.get("doc_id", "")
                entities = entry.get("entity_ids", [])
                caption = entry.get("caption", "")
                if doc_id:
                    updates = {}
                    if entities:
                        updates["entity_ids"] = ",".join(entities)
                    if caption:
                        updates["caption"] = caption
                    if updates:
                        chroma.update_metadata("visual_memories", doc_id, updates)
                        chroma_updated += 1
            print(f"Updated {chroma_updated} ChromaDB entries")
        except Exception as e:
            print(f"ChromaDB update failed (non-fatal): {e}")
    elif not args.execute:
        print("(dry-run — use --execute to write)")


def main():
    parser = argparse.ArgumentParser(description="Backfill entity tags on visual memories")
    parser.add_argument("--execute", action="store_true", help="Write changes (default: dry-run)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
