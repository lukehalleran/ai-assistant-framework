# config/feature_registry.py
"""
Loader for the retrospective feature registry (config/feature_registry.yaml).

Module Contract
- Purpose: Provides typed access to the shipped feature catalog for
  dependency checking, conflict detection, and merge gating by agents.
- Inputs: config/feature_registry.yaml
- Outputs: List[FeatureEntry] with lookup by proposal_id
- Key behaviors:
  - Loads once per process, cached at module level
  - Returns empty list if YAML missing or malformed (graceful degradation)
  - get_feature(proposal_id) for single lookup
  - get_dependencies(proposal_id) for transitive dependency resolution
  - get_core_features() for features that touch core systems
- Side effects: File read on first access
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml
from pydantic import BaseModel, Field


class FeatureOutcome(BaseModel):
    """Outcome record from the registry."""
    accepted: bool = False
    notes: str = ""
    merged_at: str = ""


class FeatureEntry(BaseModel):
    """A single feature in the registry."""
    proposal_id: str
    title: str = ""
    status: str = "implemented"
    source: str = "manual"
    risk_level: str = "medium"
    touches_core_system: bool = False
    depends_on: List[str] = Field(default_factory=list)
    implemented_files: List[str] = Field(default_factory=list)
    test_files: List[str] = Field(default_factory=list)
    outcome: Optional[FeatureOutcome] = None


# Module-level cache
_registry: Optional[List[FeatureEntry]] = None
_index: Optional[Dict[str, FeatureEntry]] = None


def _registry_path() -> Path:
    """Resolve path to feature_registry.yaml."""
    return Path(os.path.dirname(__file__)) / "feature_registry.yaml"


def load_registry(force: bool = False) -> List[FeatureEntry]:
    """Load and cache the feature registry. Returns empty list on failure."""
    global _registry, _index

    if _registry is not None and not force:
        return _registry

    path = _registry_path()
    if not path.exists():
        _registry = []
        _index = {}
        return _registry

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        raw_features = data.get("features", []) if isinstance(data, dict) else []
        entries = []
        idx = {}

        for item in raw_features:
            if not isinstance(item, dict) or "proposal_id" not in item:
                continue

            outcome_raw = item.get("outcome")
            outcome = FeatureOutcome(**outcome_raw) if isinstance(outcome_raw, dict) else None

            entry = FeatureEntry(
                proposal_id=item["proposal_id"],
                title=item.get("title", ""),
                status=item.get("status", "implemented"),
                source=item.get("source", "manual"),
                risk_level=item.get("risk_level", "medium"),
                touches_core_system=bool(item.get("touches_core_system", False)),
                depends_on=item.get("depends_on", []),
                implemented_files=item.get("implemented_files", []),
                test_files=item.get("test_files", []),
                outcome=outcome,
            )
            entries.append(entry)
            idx[entry.proposal_id] = entry

        _registry = entries
        _index = idx
        return _registry

    except Exception:
        _registry = []
        _index = {}
        return _registry


def get_feature(proposal_id: str) -> Optional[FeatureEntry]:
    """Look up a single feature by proposal_id."""
    load_registry()
    return (_index or {}).get(proposal_id)


def get_dependencies(proposal_id: str) -> List[str]:
    """Get transitive dependency chain for a feature (breadth-first)."""
    load_registry()
    if not _index:
        return []

    visited: Set[str] = set()
    queue = [proposal_id]
    result = []

    while queue:
        pid = queue.pop(0)
        if pid in visited:
            continue
        visited.add(pid)

        entry = _index.get(pid)
        if entry and entry.depends_on:
            for dep in entry.depends_on:
                if dep not in visited:
                    result.append(dep)
                    queue.append(dep)

    return result


def get_core_features() -> List[FeatureEntry]:
    """Get all features that touch core system code."""
    return [f for f in load_registry() if f.touches_core_system]


def get_implemented_files() -> Set[str]:
    """Get the union of all implemented file paths across all features."""
    files: Set[str] = set()
    for entry in load_registry():
        files.update(entry.implemented_files)
    return files


def check_conflicts(proposed_files: List[str]) -> List[FeatureEntry]:
    """Return features whose implemented_files overlap with proposed_files."""
    proposed = set(proposed_files)
    conflicts = []
    for entry in load_registry():
        if proposed & set(entry.implemented_files):
            conflicts.append(entry)
    return conflicts
