#!/usr/bin/env python3
"""
Export all user memory data into a single portable archive.

Usage:
    python scripts/export_user_data.py                      # → daemon_export_<ts>.tar.gz
    python scripts/export_user_data.py --output /path/x.tar.gz
    python scripts/export_user_data.py --no-chroma          # JSON stores only (a few MB)

The archive contains the JSON memory stores (knowledge graph, entity
aliases, user profile, conversation corpus, claim index, surfacing
history), optionally the full ChromaDB directory, and a README with
import instructions. The Google OAuth token is never included.

Import on a new machine: extract into the repo/data directory (or use
scripts/restore_backup.py --restore on a backup created there).
"""

import argparse
import os
import sys
import tarfile
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.backup_manager import backup_targets, chroma_path  # noqa: E402

README = """Daemon memory export — created {ts}

Contents (paths relative to the Daemon data directory):
{listing}

To import on a new machine:
  1. Install Daemon and run it once so the data directory exists, then stop it.
  2. Extract this archive into the data directory, replacing the fresh files:
       tar -xzf {name} -C <daemon>/data
  3. Start Daemon. The corrupt-store guard will refuse to load a damaged
     file, so a bad extraction fails loudly rather than silently.

Not included: the Google OAuth token (re-run the wizard or
scripts/reauth_google.py to re-authenticate) and API keys (.env).
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", help="Archive path (default daemon_export_<ts>.tar.gz)")
    parser.add_argument("--no-chroma", action="store_true",
                        help="Exclude the ChromaDB directory (~600MB)")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or f"daemon_export_{ts}.tar.gz"

    targets = backup_targets()
    if not targets:
        print("No memory stores found — nothing to export.")
        return 1

    listing = "\n".join(f"  - {os.path.basename(p)}" for p in targets)
    include_chroma = not args.no_chroma and os.path.isdir(chroma_path())
    if include_chroma:
        listing += f"\n  - {os.path.basename(chroma_path().rstrip('/'))}/ (ChromaDB)"

    with tarfile.open(out, "w:gz") as tar:
        for p in targets:
            tar.add(p, arcname=os.path.basename(p))
            print(f"  added {os.path.basename(p)}")
        if include_chroma:
            base = os.path.basename(chroma_path().rstrip("/"))
            print(f"  adding {base}/ (this can take a minute)…")
            tar.add(chroma_path(), arcname=base)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(README.format(ts=ts, listing=listing, name=os.path.basename(out)))
            readme_path = f.name
        try:
            tar.add(readme_path, arcname="README_IMPORT.txt")
        finally:
            os.unlink(readme_path)

    size_mb = os.path.getsize(out) / 1e6
    print(f"\nExport written: {out} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
