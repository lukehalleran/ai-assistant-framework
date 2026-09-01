#!/usr/bin/env python3
"""Force-refresh Obsidian embeddings and print a compact result summary."""

import sys
from pathlib import Path

# Running ``python scripts/...`` otherwise puts only ``scripts/`` on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge.obsidian_manager import ObsidianManager
from utils.daemon_guard import daemon_running


def main() -> int:
    # Store-writing script: a live daemon holds the chroma collections open
    # and would race this rebuild (the 08-05 live-clobber class).
    if daemon_running():
        print("Refusing to reindex: a live Daemon instance is running. "
              "Shut it down first.")
        return 2
    result = ObsidianManager().embed_vault(force_reindex=True)
    print(
        "files=%s embedded=%s updated=%s skipped=%s chunks=%s errors=%s"
        % (
            result.total_files,
            result.embedded_files,
            result.updated_files,
            result.skipped_files,
            result.total_chunks,
            result.errors,
        )
    )
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
