#!/usr/bin/env python3
"""
One-time remediation: retire stale illness/recovery facts.

The user's illness/recovery chapter is over, but several "currently recovering"
facts were stored under relation names that the read-side TTL never expired
(``post_illness_recovery``, ``mood_energy``, …), so the agent kept treating
them as present-tense. The system fix (memory/relation_classifier.py) prevents
this going forward; this script retires the facts already on disk.

Non-destructive: sets ``is_current=False`` (+ ``superseded_reason`` /
``superseded_at``) on matching facts in BOTH the user profile JSON and the live
ChromaDB ``facts`` collection. Nothing is deleted. An undo record mapping each
changed id to its prior metadata is written so the change is fully reversible.

Usage:
    python scripts/cleanup_stale_illness.py            # dry-run (default): print plan only
    python scripts/cleanup_stale_illness.py --apply    # back up + apply

Selection: facts whose subject is the user and whose relation names a transient
health state (illness/recovery/sick/symptom, via the shared classifier) OR whose
value explicitly mentions an illness cue — excluding non-self / non-state
relations (coworker_sick, work_while_sick, long_covid_symptoms=none, …).
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from memory.relation_classifier import _is_health_transient  # noqa: E402

PROFILE_PATH = os.path.join(ROOT, "data", "user_profile.json")
CHROMA_PATH_DEFAULT = os.path.join(ROOT, "data", "chroma_db_v4")

# Whole-word value cues for durable relations (e.g. mood_energy) that reference
# illness. Matched with word boundaries so short cues don't match inside other
# words ("flu" must not match "influce", "recover" excluded so it can't match
# "recovering drives" — recovery *relations* are caught by the classifier).
VALUE_CUES = ("sick", "illness", "flu", "symptom", "symptoms", "viral",
              "unwell", "nausea", "nauseous", "covid")
_VALUE_CUE_RE = re.compile(r"\b(" + "|".join(VALUE_CUES) + r")\b", re.IGNORECASE)

# Relations that contain an illness token but are NOT a current self-illness
# state (about someone else, a durable trait, or a negation).
SKIP_RELATIONS = {
    "coworker_sick", "work_while_sick", "project_work_while_sick",
    "rest_duration_when_sick", "long_covid_symptoms", "work_period",
}

# Values that are negations / non-informative — never a "currently sick" signal.
SKIP_VALUES = {"none", "no", "false", ""}

# Don't retire very recent facts — they may describe a genuinely current state.
# Anything younger than this is left to the read-side TTL, which ages it out.
MIN_AGE_HOURS_DEFAULT = 48

REASON = "illness_episode_ended"


def _age_hours(ts) -> float | None:
    """Hours since timestamp, or None if unparseable."""
    try:
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        if ts is None:
            return None
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        return (datetime.now() - ts).total_seconds() / 3600.0
    except (ValueError, TypeError, AttributeError):
        return None


def _is_target(relation: str, value: str) -> bool:
    rel = (relation or "").lower().strip()
    val = (value or "").lower().strip()
    if rel in SKIP_RELATIONS:
        return False
    if val in SKIP_VALUES:
        return False
    if _is_health_transient(rel):
        return True
    # Durable relation but value names an illness/recovery state (e.g. mood_energy)
    return bool(_VALUE_CUE_RE.search(val))


# --------------------------------------------------------------------------
# Profile JSON
# --------------------------------------------------------------------------

def scan_profile(min_age_hours):
    """Return (profile_dict, [(category, index, relation, value, ts)]) of targets."""
    with open(PROFILE_PATH, "r", encoding="utf-8") as fh:
        profile = json.load(fh)
    targets = []
    for cat, facts in profile.get("categories", {}).items():
        for i, f in enumerate(facts):
            if not isinstance(f, dict):
                continue
            if f.get("is_current") is False:
                continue  # already retired
            rel = f.get("relation", "")
            val = f.get("value") or f.get("object", "")
            ts = f.get("timestamp", "")
            age = _age_hours(ts)
            if age is not None and age < min_age_hours:
                continue  # too recent — may be genuinely current
            if _is_target(rel, val):
                targets.append((cat, i, rel, val, ts))
    return profile, targets


# --------------------------------------------------------------------------
# ChromaDB facts collection
# --------------------------------------------------------------------------

def scan_chroma(chroma_path, min_age_hours):
    """Return (collection, [(id, relation, value, ts, old_meta)]) of targets."""
    import chromadb
    client = chromadb.PersistentClient(path=chroma_path)
    col = client.get_collection("facts")
    got = col.get(include=["documents", "metadatas"])
    targets = []
    for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"] or []):
        meta = meta or {}
        if meta.get("is_current") is False:
            continue
        parts = (doc or "").split("|")
        if len(parts) < 3:
            continue
        subj = parts[0].strip().lower()
        rel = parts[1].strip()
        val = "|".join(parts[2:]).strip()
        if subj != "user":
            continue
        ts = meta.get("timestamp", "")
        age = _age_hours(ts)
        if age is not None and age < min_age_hours:
            continue  # too recent — may be genuinely current
        if _is_target(rel, val):
            targets.append((cid, rel, val, ts, dict(meta)))
    return col, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="apply changes (default: dry-run)")
    ap.add_argument("--chroma-path", default=CHROMA_PATH_DEFAULT)
    ap.add_argument("--min-age-hours", type=float, default=MIN_AGE_HOURS_DEFAULT,
                    help="skip facts younger than this (may be genuinely current)")
    args = ap.parse_args()

    now_iso = datetime.now().isoformat()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    profile, p_targets = scan_profile(args.min_age_hours)
    col, c_targets = scan_chroma(args.chroma_path, args.min_age_hours)
    print(f"(min age filter: {args.min_age_hours}h — more recent facts left to the TTL)")

    print(f"\n=== PROFILE targets ({len(p_targets)}) — {PROFILE_PATH} ===")
    for cat, i, rel, val, ts in p_targets:
        print(f"  [{cat:13s}] {rel:26s} = {str(val)[:50]:50s} ({str(ts)[:10]})")

    print(f"\n=== FACTS COLLECTION targets ({len(c_targets)}) — {args.chroma_path} ===")
    for cid, rel, val, ts, _ in c_targets:
        print(f"  [{cid[:12]}] {rel:26s} = {str(val)[:50]:50s} ({str(ts)[:10]})")

    if not args.apply:
        print("\n(dry-run — no changes written. Re-run with --apply to commit.)")
        return

    # ---- Backups + undo record ----
    prof_backup = os.path.join(ROOT, "data", f"user_profile_backup_{stamp}_illness_cleanup.json")
    with open(prof_backup, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2, ensure_ascii=False)
    print(f"\n[backup] profile -> {prof_backup}")

    undo = {
        "created_at": now_iso,
        "reason": REASON,
        "profile_backup": prof_backup,
        "chroma_path": args.chroma_path,
        "chroma_changes": [
            {"id": cid, "old_metadata": old} for cid, _, _, _, old in c_targets
        ],
        "profile_changes": [
            {"category": cat, "index": i, "relation": rel} for cat, i, rel, _, _ in p_targets
        ],
    }
    undo_path = os.path.join(ROOT, "data", f"illness_cleanup_undo_{stamp}.json")
    with open(undo_path, "w", encoding="utf-8") as fh:
        json.dump(undo, fh, indent=2, ensure_ascii=False)
    print(f"[undo]   record  -> {undo_path}")

    # ---- Apply profile ----
    for cat, i, rel, val, ts in p_targets:
        f = profile["categories"][cat][i]
        f["is_current"] = False
        f["superseded_reason"] = REASON
        f["superseded_at"] = now_iso
    profile["updated_at"] = now_iso
    with open(PROFILE_PATH, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2, ensure_ascii=False)
    print(f"[apply]  profile: retired {len(p_targets)} facts")

    # ---- Apply chroma (merge metadata, never replace) ----
    if c_targets:
        ids = [cid for cid, _, _, _, _ in c_targets]
        metas = []
        for cid, rel, val, ts, old in c_targets:
            m = dict(old)
            m["is_current"] = False
            m["superseded_reason"] = REASON
            m["superseded_at"] = now_iso
            metas.append(m)
        col.update(ids=ids, metadatas=metas)
        print(f"[apply]  facts collection: retired {len(ids)} facts")

    print("\nDone. Reversible via the undo record above.")


if __name__ == "__main__":
    main()
