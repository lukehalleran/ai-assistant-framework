#!/usr/bin/env python3
"""Add an owner-curated fact to the user profile (dry-run by default).

Why this exists: some durable, high-value facts never get stated in a form
the extractors catch — the 2026-08-05 incident was Daemon advising "message
your prescriber through the portal" when the prescriber has no portal and
doesn't respond well; that context lived only in loose July conversations.
This script lets the owner assert such a fact directly through THE deployed
UserProfile.add_fact path (canonicalization, truth metadata, supersession
all apply — no hand-edited JSON).

Usage:
  python scripts/add_profile_fact.py --relation doctor_communication \
      --value "prescriber has no patient portal and rarely responds" \
      [--category health] [--confidence 0.95] \
      [--excerpt "owner-stated 2026-08-05"] [--apply]

Without --apply it prints what would happen, including any existing facts
for the same (canonical) relation that would be superseded.
"""
import argparse
import subprocess
import sys
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
    ap.add_argument("--relation", required=True)
    ap.add_argument("--value", required=True)
    ap.add_argument("--category", default=None, help="ProfileCategory value (e.g. health); auto-categorized if omitted")
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--excerpt", default="", help="source_excerpt shown at profile injection points")
    ap.add_argument("--apply", action="store_true", help="actually write; default is dry-run")
    args = ap.parse_args()

    # A live Daemon holds the profile IN MEMORY and re-saves it (shutdown,
    # reflection cycles) — any write made here while it runs gets clobbered.
    # 2026-08-05: four curated facts written at 14:03 were silently wiped by
    # the running instance's 18:44 save. Same guard as purge_profile_facts.
    if args.apply and _daemon_running():
        print("Refusing to run --apply: a live Daemon main.py is detected — it holds "
              "the profile in memory and will overwrite this write on its next save. "
              "Shut Daemon down first.")
        return 1

    from memory.user_profile import UserProfile
    from memory.user_profile_schema import ProfileCategory, canonicalize_profile_relation, categorize_relation

    canonical = canonicalize_profile_relation(args.relation.strip().lower(), args.value)
    category = ProfileCategory(args.category) if args.category else categorize_relation(canonical)

    profile = UserProfile()
    existing = [
        f for f in profile.profile["categories"].get(category.value, [])
        if isinstance(f, dict)
        and canonicalize_profile_relation(f.get("relation", ""), f.get("value")) == canonical
    ]

    print(f"relation : {args.relation}  (canonical: {canonical})")
    print(f"category : {category.value}")
    print(f"value    : {args.value}")
    print(f"confidence: {args.confidence}  excerpt: {args.excerpt!r}")
    if existing:
        print(f"\nExisting facts for this relation ({len(existing)}):")
        for f in existing:
            cur = "CURRENT" if f.get("is_current", True) else "historical"
            print(f"  - [{cur}] {f.get('value')!r} (conf {f.get('confidence')}, {f.get('timestamp', '?')})")
        current_other = [
            f for f in existing
            if f.get("is_current", True) and f.get("value", "").lower() != args.value.lower()
        ]
        if current_other:
            print("  → a differing current value exists; --apply will SUPERSEDE it (is_current=False), not delete it.")
    else:
        print("\nNo existing facts for this relation — would append as new.")

    if not args.apply:
        print("\nDRY RUN — re-run with --apply to write.")
        return 0

    ok = profile.add_fact(
        relation=args.relation,
        value=args.value,
        confidence=args.confidence,
        source_excerpt=args.excerpt,
        category=category,
    )
    if not ok:
        print("\nadd_fact rejected the fact (empty relation/value?)")
        return 1
    profile.save()
    print("\nAPPLIED and saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
