#!/usr/bin/env python3
"""
Cleanup profile facts: canonicalize relations, merge duplicates, expire stale ephemeral facts.

Algorithm:
1. Backup user_profile.json
2. Canonicalize all relation names via canonicalize_profile_relation()
3. Recategorize facts via categorize_relation() (fixes the preferences dump)
4. Merge exact-value duplicates within same canonical relation (keep best-scoring)
5. Expire known EPHEMERAL_RELATIONS older than TTL
6. Expire obsolete work schedule facts (user quit job)
7. Expire past upcoming_* facts with resolvable dates

Safety: dry-run by default, backup before execute, never deletes — only marks is_current=False.

Usage:
    python scripts/cleanup_profile_facts.py              # dry-run (preview)
    python scripts/cleanup_profile_facts.py --execute     # backup + apply
    python scripts/cleanup_profile_facts.py --verbose     # show every change
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROFILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "user_profile.json",
)

EPHEMERAL_TTL_DAYS = 7
JOB_QUIT_DATE = datetime(2026, 4, 14)  # User quit job on this date

# Work schedule relations that are obsolete post-quit
OBSOLETE_WORK_RELATIONS = {
    "work_schedule", "work_hours", "work_shift", "work_shift_length",
    "work_shift_duration", "work_start_time", "work_in", "time_until_work",
    "work_hours_left", "work_duration", "work_end_time", "work_time",
    "work_days", "work_environment", "next_work_day", "work_full_time",
    "work_type", "work_status", "last_shift", "work_colleague",
    "coworker_sick", "days_worked", "hours_worked",
}


def compute_fact_score(fact, now):
    """Score a fact for best-of selection. Higher = better."""
    truth = float(fact.get("truth_score", 0.5))
    conf = float(fact.get("confidence", 0.5))
    try:
        ts = datetime.fromisoformat(fact["timestamp"])
        age_days = max((now - ts).days, 1)
        recency = max(0, 1.0 - (age_days / 365.0))
    except (ValueError, KeyError):
        recency = 0.0

    source = fact.get("truth_source", "")
    source_bonus = 0.0
    if source == "corrected":
        source_bonus = 0.75
    elif source == "user_stated":
        source_bonus = 0.5

    return truth * 3.0 + conf * 2.0 + recency * 1.5 + source_bonus


def run_cleanup(execute=False, verbose=False):
    from memory.user_profile_schema import (
        canonicalize_profile_relation,
        categorize_relation,
        EPHEMERAL_RELATIONS,
        SNAPSHOT_RELATIONS,
    )

    if not os.path.exists(PROFILE_PATH):
        print(f"ERROR: {PROFILE_PATH} not found")
        return

    with open(PROFILE_PATH) as f:
        profile = json.load(f)

    now = datetime.now()
    stats = {
        "canonicalized": 0,
        "recategorized": 0,
        "merged_dupes": 0,
        "expired_ephemeral": 0,
        "expired_work": 0,
        "expired_upcoming": 0,
        "total_before": 0,
        "current_before": 0,
    }

    # Count before
    for cat, facts in profile.get("categories", {}).items():
        for f in facts:
            stats["total_before"] += 1
            if f.get("is_current", True):
                stats["current_before"] += 1

    changes = []  # List of (action, description) for reporting

    # ----------------------------------------------------------------
    # Phase 1: Canonicalize relations + recategorize
    # ----------------------------------------------------------------
    for cat in list(profile.get("categories", {}).keys()):
        for f in profile["categories"][cat]:
            if not f.get("is_current", True):
                continue

            old_rel = f.get("relation", "")
            old_cat = cat
            new_rel = canonicalize_profile_relation(old_rel, f.get("value"))
            new_cat = categorize_relation(new_rel).value

            if new_rel != old_rel:
                stats["canonicalized"] += 1
                changes.append(("CANONICALIZE", f"{old_rel} → {new_rel} (val={f['value'][:40]})"))
                if execute:
                    # Preserve original relation in metadata
                    if "original_relation" not in f:
                        f["original_relation"] = old_rel
                    f["relation"] = new_rel

            if new_cat != old_cat:
                stats["recategorized"] += 1
                changes.append(("RECATEGORIZE", f"{old_rel}={f['value'][:30]} : {old_cat} → {new_cat}"))
                if execute:
                    f["_move_to"] = new_cat

    # Execute category moves
    if execute:
        for cat in list(profile["categories"].keys()):
            remaining = []
            for f in profile["categories"][cat]:
                dest = f.pop("_move_to", None)
                if dest and dest != cat:
                    if dest not in profile["categories"]:
                        profile["categories"][dest] = []
                    profile["categories"][dest].append(f)
                else:
                    remaining.append(f)
            profile["categories"][cat] = remaining

    # ----------------------------------------------------------------
    # Phase 2: Merge exact-value duplicates within same canonical relation
    # ----------------------------------------------------------------
    for cat, facts in profile.get("categories", {}).items():
        # Group current facts by canonical relation
        by_canonical = defaultdict(list)
        for i, f in enumerate(facts):
            if not f.get("is_current", True):
                continue
            rel = f.get("relation", "")
            canonical = canonicalize_profile_relation(rel, f.get("value"))
            by_canonical[canonical].append((i, f))

        for canonical, group in by_canonical.items():
            if len(group) < 2:
                continue
            # Group by normalized value
            by_value = defaultdict(list)
            for idx, f in group:
                val = f.get("value", "").strip().lower()
                by_value[val].append((idx, f))

            for val, dupes in by_value.items():
                if len(dupes) < 2:
                    continue
                # Keep best-scoring, mark rest as is_current=False
                scored = [(compute_fact_score(f, now), idx, f) for idx, f in dupes]
                scored.sort(key=lambda x: x[0], reverse=True)
                best_score, best_idx, best_fact = scored[0]
                for score, idx, f in scored[1:]:
                    stats["merged_dupes"] += 1
                    changes.append((
                        "MERGE_DUPE",
                        f"{f['relation']}={f['value'][:30]} (score={score:.2f}) → superseded by {best_fact['relation']} (score={best_score:.2f})"
                    ))
                    if execute:
                        facts[idx]["is_current"] = False
                        facts[idx]["superseded_by_cleanup"] = best_fact.get("fact_id", "")

    # ----------------------------------------------------------------
    # Phase 3: Expire stale ephemeral facts
    # ----------------------------------------------------------------
    for cat, facts in profile.get("categories", {}).items():
        for f in facts:
            if not f.get("is_current", True):
                continue
            rel = f.get("relation", "")
            canonical = canonicalize_profile_relation(rel, f.get("value"))

            # Skip snapshot relations — they survive TTL
            if canonical in SNAPSHOT_RELATIONS:
                continue

            if canonical in EPHEMERAL_RELATIONS:
                try:
                    ts = datetime.fromisoformat(f["timestamp"])
                    age_days = (now - ts).days
                    if age_days > EPHEMERAL_TTL_DAYS:
                        stats["expired_ephemeral"] += 1
                        changes.append((
                            "EXPIRE_EPHEMERAL",
                            f"{rel}={f['value'][:40]} ({age_days}d old)"
                        ))
                        if execute:
                            f["is_current"] = False
                            f["expired_reason"] = "ephemeral_ttl"
                except (ValueError, KeyError):
                    pass

    # ----------------------------------------------------------------
    # Phase 4: Expire obsolete work schedule facts (user quit job)
    # ----------------------------------------------------------------
    for cat, facts in profile.get("categories", {}).items():
        for f in facts:
            if not f.get("is_current", True):
                continue
            rel = f.get("relation", "")
            canonical = canonicalize_profile_relation(rel, f.get("value"))

            if canonical in OBSOLETE_WORK_RELATIONS:
                try:
                    ts = datetime.fromisoformat(f["timestamp"])
                    # Only expire facts from before the quit date
                    if ts < JOB_QUIT_DATE:
                        stats["expired_work"] += 1
                        changes.append((
                            "EXPIRE_WORK",
                            f"{rel}={f['value'][:40]} (pre-quit, {f['timestamp'][:10]})"
                        ))
                        if execute:
                            f["is_current"] = False
                            f["expired_reason"] = "job_quit"
                except (ValueError, KeyError):
                    pass

    # ----------------------------------------------------------------
    # Phase 5: Expire past upcoming_* facts
    # ----------------------------------------------------------------
    for cat, facts in profile.get("categories", {}).items():
        for f in facts:
            if not f.get("is_current", True):
                continue
            rel = f.get("relation", "")
            if not rel.startswith("upcoming_"):
                continue
            try:
                ts = datetime.fromisoformat(f["timestamp"])
                # If the fact is more than 30 days old, the event has passed
                if (now - ts).days > 30:
                    stats["expired_upcoming"] += 1
                    changes.append((
                        "EXPIRE_UPCOMING",
                        f"{rel}={f['value'][:40]} ({(now - ts).days}d old)"
                    ))
                    if execute:
                        f["is_current"] = False
                        f["expired_reason"] = "past_upcoming"
            except (ValueError, KeyError):
                pass

    # ----------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------
    print("=" * 70)
    print(f"PROFILE FACTS CLEANUP {'(EXECUTE)' if execute else '(DRY RUN)'}")
    print("=" * 70)
    print(f"\n  Total facts:          {stats['total_before']}")
    print(f"  Current (before):     {stats['current_before']}")
    print()
    print(f"  Relations canonicalized: {stats['canonicalized']}")
    print(f"  Facts recategorized:     {stats['recategorized']}")
    print(f"  Duplicates merged:       {stats['merged_dupes']}")
    print(f"  Ephemeral expired:       {stats['expired_ephemeral']}")
    print(f"  Work facts expired:      {stats['expired_work']}")
    print(f"  Upcoming expired:        {stats['expired_upcoming']}")
    total_changes = (stats['merged_dupes'] + stats['expired_ephemeral']
                     + stats['expired_work'] + stats['expired_upcoming'])
    print(f"  ---")
    print(f"  Current (after):      {stats['current_before'] - total_changes}")
    print()

    if verbose or not execute:
        # Group changes by action
        by_action = defaultdict(list)
        for action, desc in changes:
            by_action[action].append(desc)

        for action in ["CANONICALIZE", "RECATEGORIZE", "MERGE_DUPE",
                        "EXPIRE_EPHEMERAL", "EXPIRE_WORK", "EXPIRE_UPCOMING"]:
            items = by_action.get(action, [])
            if items:
                print(f"\n  [{action}] ({len(items)})")
                for desc in items[:50]:  # cap display
                    print(f"    {desc}")
                if len(items) > 50:
                    print(f"    ... +{len(items) - 50} more")

    if execute:
        # Backup
        ts_str = now.strftime("%Y%m%d_%H%M%S")
        backup_path = PROFILE_PATH.replace(".json", f"_backup_{ts_str}.json")
        shutil.copy2(PROFILE_PATH, backup_path)
        print(f"\n  Backup saved: {backup_path}")

        # Save
        with open(PROFILE_PATH, "w") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        print(f"  Changes applied to: {PROFILE_PATH}")
    else:
        print(f"\n  (Dry run — no changes made. Use --execute to apply.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup profile facts")
    parser.add_argument("--execute", action="store_true", help="Apply changes (backup first)")
    parser.add_argument("--verbose", action="store_true", help="Show every change")
    args = parser.parse_args()
    run_cleanup(execute=args.execute, verbose=args.verbose)
