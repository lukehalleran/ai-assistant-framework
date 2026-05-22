#!/usr/bin/env python3
"""
One-time backfill: add supervision metadata to existing ChromaDB proposals.

Scans all documents in the 'proposals' collection and adds missing
supervision fields (risk_level, touches_core_system, depends_on_json,
test_files_json, outcome_json) using heuristic inference from existing
metadata (affected_files, title, proposal_type, implementation_status).

Usage:
    python scripts/migrate_proposals_supervision.py --dry-run    # preview only
    python scripts/migrate_proposals_supervision.py              # write metadata
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.app_config import CHROMA_PATH
from config.feature_registry import load_registry, get_implemented_files
from memory.code_proposal import RiskLevel
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from utils.logging_utils import get_logger

logger = get_logger("migrate_proposals_supervision")

COLLECTION = "proposals"

# Files that indicate core-system changes
CORE_PATHS = {
    "core/orchestrator.py",
    "core/context_pipeline.py",
    "core/prompt/builder.py",
    "memory/memory_coordinator.py",
    "memory/memory_storage.py",
    "memory/memory_scorer.py",
    "memory/memory_retriever.py",
    "processing/gate_system.py",
    "utils/destructive_op_guard.py",
    "utils/python_fs_guard.py",
    "utils/shell_cmd_guard.py",
    "scripts/safe_git.sh",
    "scripts/safe_cmd.sh",
}

# Keywords that suggest high risk
HIGH_RISK_KEYWORDS = re.compile(
    r"safety|guard|security|auth|crisis|escalation|shutdown|delete|purge|migration",
    re.IGNORECASE,
)
CRITICAL_RISK_KEYWORDS = re.compile(
    r"data.loss|destructive|wipe|drop.table|rm.-rf",
    re.IGNORECASE,
)


def infer_risk_level(metadata: dict) -> str:
    """Infer risk_level from title, affected_files, and proposal_type."""
    title = metadata.get("title", "")
    ptype = metadata.get("proposal_type", "feature")
    affected_raw = metadata.get("affected_files_json", "[]")

    try:
        affected = json.loads(affected_raw)
    except (json.JSONDecodeError, TypeError):
        affected = []

    # Critical: destructive keywords
    if CRITICAL_RISK_KEYWORDS.search(title):
        return RiskLevel.CRITICAL.value

    # High: safety keywords or touches core paths
    if HIGH_RISK_KEYWORDS.search(title):
        return RiskLevel.HIGH.value
    if any(f in CORE_PATHS for f in affected):
        return RiskLevel.HIGH.value

    # Low: docs, tests only
    if ptype in ("docs", "test"):
        return RiskLevel.LOW.value

    return RiskLevel.MEDIUM.value


def infer_touches_core(metadata: dict) -> bool:
    """Check if affected_files overlap with core paths."""
    affected_raw = metadata.get("affected_files_json", "[]")
    try:
        affected = json.loads(affected_raw)
    except (json.JSONDecodeError, TypeError):
        return False
    return any(f in CORE_PATHS for f in affected)


def infer_test_files(metadata: dict) -> list:
    """Extract test file paths from affected_files."""
    affected_raw = metadata.get("affected_files_json", "[]")
    try:
        affected = json.loads(affected_raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return [f for f in affected if "test" in f.lower()]


def infer_depends_on(metadata: dict, registry_files: set) -> list:
    """Check if affected_files overlap with registry features → dependency."""
    affected_raw = metadata.get("affected_files_json", "[]")
    try:
        affected = json.loads(affected_raw)
    except (json.JSONDecodeError, TypeError):
        return []

    deps = []
    for entry in load_registry():
        overlap = set(affected) & set(entry.implemented_files)
        if overlap and entry.proposal_id not in deps:
            deps.append(entry.proposal_id)
    return deps


def infer_outcome(metadata: dict) -> dict | None:
    """Infer outcome from status field."""
    status = metadata.get("status", "pending")
    if status == "completed":
        return {
            "accepted": True,
            "notes": "Backfilled from status=completed",
            "merged_at": metadata.get("executed_at") or metadata.get("modified_at"),
            "merge_branch": "",
            "reviewed_by": "",
        }
    elif status == "rejected":
        return {
            "accepted": False,
            "notes": metadata.get("rejection_reason", "Backfilled from status=rejected"),
            "merged_at": None,
            "merge_branch": "",
            "reviewed_by": "",
        }
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Backfill supervision metadata on existing proposals"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only, do not write")
    args = parser.parse_args()

    print(f"{'DRY RUN — ' if args.dry_run else ''}Backfilling proposal supervision fields")
    print(f"ChromaDB path: {CHROMA_PATH}")

    store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    store.create_collection(COLLECTION)

    coll = store.collections.get(COLLECTION)
    if not coll:
        print("No proposals collection found.")
        return

    count = coll.count()
    print(f"Found {count} proposals")

    if count == 0:
        return

    all_items = store.list_all(COLLECTION)
    registry_files = get_implemented_files()

    updated = 0
    skipped = 0

    for item in all_items:
        doc_id = item.get("id")
        md = item.get("metadata") or {}

        # Skip if already backfilled
        if md.get("risk_level") and md.get("risk_level") != "MISSING":
            skipped += 1
            continue

        risk = infer_risk_level(md)
        core = infer_touches_core(md)
        tests = infer_test_files(md)
        deps = infer_depends_on(md, registry_files)
        outcome = infer_outcome(md)

        updates = {
            "risk_level": risk,
            "touches_core_system": core,
            "depends_on_json": json.dumps(deps),
            "test_files_json": json.dumps(tests),
            "outcome_json": json.dumps(outcome),
        }

        title = md.get("title", "Untitled")[:60]
        status = md.get("status", "?")
        dep_str = f" deps={deps}" if deps else ""
        test_str = f" tests={len(tests)}" if tests else ""
        outcome_str = f" outcome={'accepted' if outcome and outcome['accepted'] else 'rejected' if outcome else 'none'}"

        print(
            f"  [{status}] {title:60s}  "
            f"risk={risk:8s} core={str(core):5s}{dep_str}{test_str}{outcome_str}"
        )

        if not args.dry_run:
            store.update_metadata(COLLECTION, doc_id, updates)

        updated += 1

    print(f"\n{'Would update' if args.dry_run else 'Updated'}: {updated}")
    print(f"Already backfilled (skipped): {skipped}")

    if args.dry_run:
        print("\nRe-run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
