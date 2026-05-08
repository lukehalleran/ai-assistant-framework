"""
Pure data models for the eval system.

This file intentionally does not import any core Daemon modules. All models are
stdlib dataclasses with JSON serialization via to_dict/from_dict.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

try:
    import orjson

    def _json_dumps(obj: Any) -> str:
        return orjson.dumps(obj, option=orjson.OPT_INDENT_2).decode()

    def _json_loads(s: str) -> Any:
        return orjson.loads(s)

except ImportError:
    import json

    def _json_dumps(obj: Any) -> str:
        return json.dumps(obj, indent=2, default=str)

    def _json_loads(s: str) -> Any:
        return json.loads(s)


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def compute_hash(text_or_bytes: str | bytes) -> str:
    """SHA-256 hash, truncated to 16 hex chars."""
    if isinstance(text_or_bytes, str):
        text_or_bytes = text_or_bytes.encode("utf-8")
    return hashlib.sha256(text_or_bytes).hexdigest()[:16]


def normalize_prompt_text(text: str) -> str:
    """Normalize prompt text for hash comparison.

    Normalization rules (intentionally conservative):
    - Convert CRLF to LF
    - Strip trailing spaces per line
    - Strip leading/trailing full-prompt whitespace
    - Collapse runs of 3+ blank lines to 2

    Does NOT remove ordinary spaces inside sentences.
    """
    text = text.replace("\r\n", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def compute_prompt_hash(prompt_text: str, normalize: bool = False) -> str:
    """Compute hash of prompt text, optionally after normalization."""
    if normalize:
        prompt_text = normalize_prompt_text(prompt_text)
    return compute_hash(prompt_text)


# ---------------------------------------------------------------------------
# Section snapshot
# ---------------------------------------------------------------------------

@dataclass
class SectionSnapshot:
    """Snapshot of a single prompt section."""
    key: str
    header: str
    structured_content: Any
    formatted_text: str
    token_count: int
    source_field: str
    category: str
    eligible_for_ablation: bool
    structurally_required: bool
    assembly_order: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SectionSnapshot:
        return cls(**d)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

@dataclass
class PromptProvenance:
    """Tracks what code/config/model produced a prompt."""
    model_name: str
    git_commit_hash: str
    system_prompt_hash: str
    system_prompt_text: Optional[str] = None
    personality_text_hash: str = ""
    operating_principles_hash: str = ""
    config_hash: str = ""
    formatter_hash: str = ""
    builder_hash: str = ""
    token_manager_hash: str = ""
    token_manager_settings: Dict[str, Any] = field(default_factory=dict)
    capture_timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PromptProvenance:
        return cls(**d)


# ---------------------------------------------------------------------------
# Snapshot layer
# ---------------------------------------------------------------------------

@dataclass
class SnapshotLayer:
    """One layer of a prompt snapshot (raw_retrieval or post_hygiene)."""
    layer_name: str
    sections: Dict[str, SectionSnapshot]
    layer_content_hash: str
    prompt_text: Optional[str]
    prompt_hash_exact: Optional[str]
    prompt_hash_normalized: Optional[str]
    capture_timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # sections are already dicts via asdict
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SnapshotLayer:
        raw_sections = d.pop("sections", {})
        sections = {k: SectionSnapshot.from_dict(v) for k, v in raw_sections.items()}
        return cls(sections=sections, **d)


# ---------------------------------------------------------------------------
# Full prompt snapshot
# ---------------------------------------------------------------------------

@dataclass
class PromptSnapshot:
    """Complete snapshot of a prompt build, including both layers and provenance."""
    snapshot_id: str
    query_text: str
    query_timestamp: str
    processed_query: str
    detected_intent: str
    detected_tone: str
    provenance: PromptProvenance
    layers: Dict[str, SnapshotLayer]
    retrieval_metadata: Dict[str, Any]
    assembly_metadata: Dict[str, Any]
    final_system_prompt_hash: str = ""
    final_user_prompt_hash: str = ""
    final_message_payload_hash: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "snapshot_id": self.snapshot_id,
            "query_text": self.query_text,
            "query_timestamp": self.query_timestamp,
            "processed_query": self.processed_query,
            "detected_intent": self.detected_intent,
            "detected_tone": self.detected_tone,
            "provenance": self.provenance.to_dict(),
            "layers": {k: v.to_dict() for k, v in self.layers.items()},
            "retrieval_metadata": self.retrieval_metadata,
            "assembly_metadata": self.assembly_metadata,
            "final_system_prompt_hash": self.final_system_prompt_hash,
            "final_user_prompt_hash": self.final_user_prompt_hash,
            "final_message_payload_hash": self.final_message_payload_hash,
            "notes": self.notes,
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PromptSnapshot:
        provenance = PromptProvenance.from_dict(d.pop("provenance"))
        raw_layers = d.pop("layers", {})
        layers = {k: SnapshotLayer.from_dict(v) for k, v in raw_layers.items()}
        return cls(provenance=provenance, layers=layers, **d)

    def to_json(self) -> str:
        return _json_dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> PromptSnapshot:
        return cls.from_dict(_json_loads(s))


# ---------------------------------------------------------------------------
# Eval generation result
# ---------------------------------------------------------------------------

@dataclass
class EvalGenerationResult:
    """Result of a side-effect-free eval generation."""
    response_text: str
    model: str
    prompt_hash: str
    prompt_token_count: int
    response_token_count: int
    generation_time_ms: int
    temperature: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> EvalGenerationResult:
        return cls(**d)


# ---------------------------------------------------------------------------
# Persistence fingerprinting
# ---------------------------------------------------------------------------

@dataclass
class StoreFingerprint:
    """Fingerprint of a single persistent store (collection, file, etc.)."""
    name: str
    kind: str  # "chromadb_collection", "json_file", "graph", etc.
    count: Optional[int] = None
    size_bytes: Optional[int] = None
    mtime_ns: Optional[int] = None
    sha256: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> StoreFingerprint:
        return cls(**d)


@dataclass
class PersistenceSnapshot:
    """Fingerprint of all mutable persistent state at a point in time."""
    snapshot_id: str
    capture_timestamp: str
    fingerprints: Dict[str, StoreFingerprint]

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "snapshot_id": self.snapshot_id,
            "capture_timestamp": self.capture_timestamp,
            "fingerprints": {k: v.to_dict() for k, v in self.fingerprints.items()},
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PersistenceSnapshot:
        raw_fps = d.pop("fingerprints", {})
        fps = {k: StoreFingerprint.from_dict(v) for k, v in raw_fps.items()}
        return cls(fingerprints=fps, **d)

    def diff(self, other: PersistenceSnapshot) -> Dict[str, Any]:
        """Compare this snapshot to another. Returns dict of differences.

        Convention: self is 'before', other is 'after'.
        """
        changes: Dict[str, Any] = {}
        all_keys = set(self.fingerprints.keys()) | set(other.fingerprints.keys())

        for key in sorted(all_keys):
            before = self.fingerprints.get(key)
            after = other.fingerprints.get(key)

            if before is None:
                changes[key] = {"change": "added", "after": after.to_dict()}
                continue
            if after is None:
                changes[key] = {"change": "removed", "before": before.to_dict()}
                continue

            field_diffs = {}
            for attr in ("count", "size_bytes", "mtime_ns", "sha256"):
                bval = getattr(before, attr)
                aval = getattr(after, attr)
                if bval != aval:
                    field_diffs[attr] = {"before": bval, "after": aval}

            if field_diffs:
                changes[key] = {"change": "modified", "fields": field_diffs}

        return changes

    def assert_same_as(self, other: PersistenceSnapshot) -> None:
        """Raise AssertionError if any fingerprint changed between self and other."""
        changes = self.diff(other)
        if changes:
            details = []
            for key, info in changes.items():
                change_type = info.get("change", "unknown")
                if change_type == "modified":
                    fields = info.get("fields", {})
                    field_strs = [
                        f"  {f}: {v['before']} -> {v['after']}"
                        for f, v in fields.items()
                    ]
                    details.append(f"{key} ({change_type}):\n" + "\n".join(field_strs))
                else:
                    details.append(f"{key} ({change_type})")
            msg = (
                f"Persistence state changed in {len(changes)} store(s):\n"
                + "\n".join(details)
            )
            raise AssertionError(msg)
