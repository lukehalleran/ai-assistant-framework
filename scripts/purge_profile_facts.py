#!/usr/bin/env python3
"""Delete curated junk facts from data/user_profile.json by fact_id (DRY-RUN FIRST).

The profile store has no deletion API — junk facts observed live on
2026-08-05 (age_range='14 to 15', nickname='hot bartender' from a joke turn,
greeting stored off an API-error turn, location/lives_in='empty house',
dad_show_up fragment objects) can only be superseded, not removed, and some
have no correct replacement value to supersede with.

Safety model (per CLAUDE.md "NEVER auto-delete user data"):
  * Default is DRY RUN — prints exactly which facts the candidate file
    matches and exits. Nothing is modified.
  * --apply writes a pre-image backup of the FULL profile JSON to
    data/backups/user_profile_preimage_<ts>.json first, then removes only
    the fact_ids listed UNCOMMENTED in the candidate file, then saves via
    the deployed UserProfile.save() (atomic safe_json write).
  * Refuses to run while a live Daemon main.py is detected.

Candidate file format (one fact_id per line, '#' comments allowed):
    data/profile_junk_candidates_<date>.txt

Usage:
    python scripts/purge_profile_facts.py --from-file data/profile_junk_candidates_20260805.txt
    python scripts/purge_profile_facts.py --from-file ... --apply
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _daemon_running() -> bool:
    try:
        from utils.daemon_guard import daemon_running
        return daemon_running()
    except Exception:
        pass
    # Fallback (guard module unavailable): old cmdline heuristic. Known hole:
    # a relative-path launch has no repo name in its cmdline (2026-08-21).
    try:
        out = subprocess.run(["pgrep", "-af", "main.py"], capture_output=True, text=True).stdout
        return any("Daemon_v1" in line for line in out.splitlines())
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-file", required=True, help="curated candidate file (fact_ids, # comments)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if _daemon_running():
        print("Refusing to run: a live Daemon main.py is detected. Shut it down first.")
        return 1

    wanted = set()
    for line in Path(args.from_file).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            wanted.add(line.split()[0])
    if not wanted:
        print("Candidate file has no uncommented fact_ids — nothing to do.")
        return 0

    from memory.user_profile import UserProfile
    profile = UserProfile()

    matches = []  # (category, index, fact)
    for cat, facts in profile.profile.get("categories", {}).items():
        for i, f in enumerate(facts):
            if isinstance(f, dict) and f.get("fact_id") in wanted:
                matches.append((cat, i, f))

    print(f"Candidate fact_ids: {len(wanted)}; matched in profile: {len(matches)}\n")
    for cat, _, f in matches:
        cur = "CURRENT" if f.get("is_current", True) else "historical"
        print(f"  [{cur}] {cat}/{f.get('relation')} = {f.get('value')!r}  (fact_id {f.get('fact_id')})")
    missing = wanted - {f.get("fact_id") for _, _, f in matches}
    for m in sorted(missing):
        print(f"  NOT FOUND: {m}")

    if not args.apply:
        print("\nDRY RUN — re-run with --apply to back up + delete the matched facts.")
        return 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("data/backups")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"user_profile_preimage_{ts}.json"
    shutil.copy2(profile.profile_path, backup_path)
    print(f"\nPre-image backup: {backup_path}")

    removed = 0
    with profile._lock:
        for cat, facts in profile.profile.get("categories", {}).items():
            before = len(facts)
            profile.profile["categories"][cat] = [
                f for f in facts
                if not (isinstance(f, dict) and f.get("fact_id") in wanted)
            ]
            removed += before - len(profile.profile["categories"][cat])
    profile.save()
    print(f"Removed {removed} facts and saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
