#!/usr/bin/env python3
"""Re-classify supervision fields on existing ChromaDB proposals.

Re-runs the CURRENT classifier (``memory.proposal_risk.classify_proposal`` —
directory-PREFIX matching + IMPORT-based detection; safety/supervision touches →
CRITICAL) over EVERY stored proposal and refreshes ``risk_level`` /
``touches_core_system`` / ``depends_on`` accordingly.

Why this exists (vs ``scripts/migrate_proposals_supervision.py``):
- The old migration only backfilled proposals whose ``risk_level`` was MISSING,
  and used exact-string core-path matching. Proposals generated before the
  classifier was wired into ``_parse_proposal`` carry the DEFAULTS (MEDIUM /
  False); proposals from the weaker exact-string era are under-classified. This
  re-runs the live classifier over ALL of them so those get upgraded.

Extraction mirrors ``GoalDirectedGenerator._parse_proposal`` /
``_annotate_conflicts``: every file the proposal would touch (affected_files +
step targets) plus any code/text it introduces (step snippets + description +
reasoning, for import-based detection), and ``depends_on`` is unioned from the
LIVE feature registry.

Safety:
- NON-DESTRUCTIVE: uses ``update_metadata`` (a merge); never deletes a proposal.
- DRY-RUN by default — prints a before→after table and writes NOTHING. Re-run
  with ``--apply`` to persist.

Usage:
    python scripts/reclassify_proposals.py            # preview (dry-run)
    python scripts/reclassify_proposals.py --apply    # write metadata
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.app_config import CHROMA_PATH
from memory.code_proposal import ProposalType
from memory.proposal_risk import classify_proposal
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from utils.logging_utils import get_logger

logger = get_logger("reclassify_proposals")

COLLECTION = "proposals"

# Risk ordering for upgrade/downgrade reporting.
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _loads(raw, default):
    try:
        val = json.loads(raw)
        return val if val is not None else default
    except (json.JSONDecodeError, TypeError):
        return default


def _classify_from_metadata(md: dict):
    """Return (touched_paths, touches_core, risk_level) for one proposal's
    metadata, mirroring the generator's live extraction."""
    affected = _loads(md.get("affected_files_json", "[]"), [])
    steps = _loads(md.get("steps_json", "[]"), [])
    step_paths = [s.get("file_path") for s in steps
                  if isinstance(s, dict) and s.get("file_path")]
    code_texts = [s.get("code_snippet", "") for s in steps
                  if isinstance(s, dict) and s.get("code_snippet")]
    code_texts += [md.get("description", ""), md.get("reasoning", "")]

    touched = list(affected) + step_paths

    try:
        proposal_type = ProposalType(md.get("proposal_type", "feature"))
    except ValueError:
        proposal_type = ProposalType.FEATURE

    touches_core, risk = classify_proposal(
        touched,
        code_texts=code_texts,
        title=md.get("title", ""),
        description=md.get("description", ""),
        proposal_type=proposal_type,
    )
    return touched, touches_core, risk


def _registry_depends_on(touched_paths: list) -> list:
    """Live-registry overlaps (+ transitive deps) for the touched paths — the same
    advisory annotation the generator applies. Best-effort: a missing registry
    degrades to no annotation."""
    try:
        from config.feature_registry import check_conflicts, get_dependencies
    except Exception:  # noqa: BLE001
        return []
    deps: list = []
    try:
        for feat in check_conflicts(touched_paths):
            if feat.proposal_id not in deps:
                deps.append(feat.proposal_id)
            for d in get_dependencies(feat.proposal_id):
                if d not in deps:
                    deps.append(d)
    except Exception as e:  # noqa: BLE001
        logger.debug("registry conflict lookup failed: %s", e)
    return deps


def main():
    parser = argparse.ArgumentParser(
        description="Re-classify supervision fields on stored proposals"
    )
    parser.add_argument("--apply", action="store_true",
                        help="Write metadata (default: dry-run preview only)")
    args = parser.parse_args()
    dry_run = not args.apply

    print(f"{'DRY RUN — ' if dry_run else ''}Re-classifying proposal supervision fields")
    print(f"ChromaDB path: {CHROMA_PATH}\n")

    store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    store.create_collection(COLLECTION)
    coll = store.collections.get(COLLECTION)
    if not coll:
        print("No proposals collection found.")
        return

    count = coll.count()
    print(f"Found {count} proposals\n")
    if count == 0:
        return

    all_items = store.list_all(COLLECTION)

    upgraded = downgraded = core_changed = deps_changed = unchanged = 0

    for item in all_items:
        md = item.get("metadata") or {}
        if not md.get("proposal_id"):
            continue
        doc_id = item.get("id")

        touched, new_core, new_risk = _classify_from_metadata(md)
        new_deps = _registry_depends_on(touched)

        old_risk = (md.get("risk_level") or "medium").lower()
        old_core = bool(md.get("touches_core_system", False))
        old_deps = _loads(md.get("depends_on_json", "[]"), [])
        merged_deps = list(old_deps)
        for d in new_deps:
            if d not in merged_deps:
                merged_deps.append(d)

        updates = {}
        if new_risk.value != old_risk:
            updates["risk_level"] = new_risk.value
        if new_core != old_core:
            updates["touches_core_system"] = new_core
        if merged_deps != old_deps:
            updates["depends_on_json"] = json.dumps(merged_deps)

        if not updates:
            unchanged += 1
            continue

        # classify the kind of change for the summary
        if "risk_level" in updates:
            if _RISK_ORDER.get(new_risk.value, 1) > _RISK_ORDER.get(old_risk, 1):
                upgraded += 1
            else:
                downgraded += 1
        if "touches_core_system" in updates:
            core_changed += 1
        if "depends_on_json" in updates:
            deps_changed += 1

        title = (md.get("title", "Untitled") or "Untitled")[:50]
        risk_str = (f"{old_risk}→{new_risk.value}"
                    if "risk_level" in updates else new_risk.value)
        core_str = (f" core:{old_core}→{new_core}"
                    if "touches_core_system" in updates else "")
        dep_str = f" +deps={[d for d in merged_deps if d not in old_deps]}" \
            if "depends_on_json" in updates else ""
        print(f"  {title:50s}  risk={risk_str}{core_str}{dep_str}")

        if not dry_run:
            store.update_metadata(COLLECTION, doc_id, updates)

    print(f"\n{'Would change' if dry_run else 'Changed'}: "
          f"risk_upgraded={upgraded} risk_downgraded={downgraded} "
          f"core_changed={core_changed} deps_changed={deps_changed} "
          f"unchanged={unchanged}")
    if dry_run:
        print("\nRe-run with --apply to persist these changes.")


if __name__ == "__main__":
    main()
