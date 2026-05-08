"""
Snapshot capture, save/load, and replay for prompt eval.

SnapshotCapture: builds PromptSnapshot objects from live builder data.
SnapshotReplay: reconstructs prompts from saved snapshots (no retrieval).
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.schema import (
    PromptProvenance,
    PromptSnapshot,
    SectionSnapshot,
    SnapshotLayer,
    compute_hash,
    compute_prompt_hash,
)
from eval.section_registry import (
    SECTION_REGISTRY,
    get_context_sections,
    match_header_to_key,
)

_SNAPSHOT_DIR = os.environ.get("DAEMON_EVAL_SNAPSHOT_DIR", "eval/snapshots")


# ---------------------------------------------------------------------------
# Snapshot Capture
# ---------------------------------------------------------------------------

class SnapshotCapture:
    """Builds PromptSnapshot objects from live builder output."""

    def capture_layer(
        self,
        layer_name: str,
        structured_context: Dict[str, Any],
        formatted_sections: Dict[str, str],
        prompt_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SnapshotLayer:
        """Capture a single snapshot layer.

        Args:
            layer_name: "raw_retrieval" or "post_hygiene".
            structured_context: The context dict (lists/dicts per field).
            formatted_sections: Map of internal_key -> formatted text for that section.
            prompt_text: The assembled prompt string (None for raw_retrieval).
            metadata: Extra metadata to attach.

        Returns:
            SnapshotLayer with section snapshots and hashes.
        """
        sections: Dict[str, SectionSnapshot] = {}

        for key, formatted_text in formatted_sections.items():
            reg = SECTION_REGISTRY.get(key)
            token_count = _estimate_tokens(formatted_text)
            structured = structured_context.get(
                reg.source_field if reg else key, None
            )

            sections[key] = SectionSnapshot(
                key=key,
                header=reg.header if reg else f"[{key.upper()}]",
                structured_content=_make_serializable(structured),
                formatted_text=formatted_text,
                token_count=token_count,
                source_field=reg.source_field if reg else key,
                category=reg.category.value if reg else "unknown",
                eligible_for_ablation=reg.eligible_for_ablation if reg else True,
                structurally_required=reg.structurally_required if reg else False,
                assembly_order=reg.assembly_order if reg else 999,
                metadata={},
            )

        # Content hash from all formatted texts concatenated in order
        ordered_texts = [
            sections[k].formatted_text
            for k in sorted(sections, key=lambda k: sections[k].assembly_order)
        ]
        layer_content_hash = compute_hash("\n\n".join(ordered_texts))

        prompt_hash_exact = compute_prompt_hash(prompt_text) if prompt_text else None
        prompt_hash_normalized = (
            compute_prompt_hash(prompt_text, normalize=True) if prompt_text else None
        )

        return SnapshotLayer(
            layer_name=layer_name,
            sections=sections,
            layer_content_hash=layer_content_hash,
            prompt_text=prompt_text,
            prompt_hash_exact=prompt_hash_exact,
            prompt_hash_normalized=prompt_hash_normalized,
            capture_timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )

    def capture_full_snapshot(
        self,
        query_text: str,
        processed_query: str,
        detected_intent: str,
        detected_tone: str,
        raw_context: Dict[str, Any],
        raw_formatted_sections: Dict[str, str],
        hygiene_context: Dict[str, Any],
        hygiene_formatted_sections: Dict[str, str],
        post_hygiene_prompt_text: str,
        provenance: PromptProvenance,
        system_prompt_text: str = "",
        retrieval_metadata: Optional[Dict[str, Any]] = None,
        assembly_metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptSnapshot:
        """Capture a complete two-layer snapshot.

        Args:
            query_text: Original user query.
            processed_query: Query after rewriting/processing.
            detected_intent: Intent classification result.
            detected_tone: Tone detection result.
            raw_context: Structured context before hygiene/caps.
            raw_formatted_sections: Formatted sections from raw context (may be empty).
            hygiene_context: Structured context after hygiene/caps/budget.
            hygiene_formatted_sections: Formatted sections after hygiene.
            post_hygiene_prompt_text: Final assembled prompt string.
            provenance: Build provenance metadata.
            system_prompt_text: The composed system prompt (separate from context).
            retrieval_metadata: Optional retrieval timing/counts.
            assembly_metadata: Optional assembly timing/config.

        Returns:
            Complete PromptSnapshot with both layers.
        """
        import uuid

        snapshot_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()

        raw_layer = self.capture_layer(
            layer_name="raw_retrieval",
            structured_context=raw_context,
            formatted_sections=raw_formatted_sections,
            prompt_text=None,  # Raw layer has no assembled prompt
            metadata={"note": "Pre-hygiene structured content only"},
        )

        hygiene_layer = self.capture_layer(
            layer_name="post_hygiene",
            structured_context=hygiene_context,
            formatted_sections=hygiene_formatted_sections,
            prompt_text=post_hygiene_prompt_text,
        )

        system_hash = compute_hash(system_prompt_text) if system_prompt_text else ""
        user_hash = compute_prompt_hash(post_hygiene_prompt_text) if post_hygiene_prompt_text else ""

        # Message payload = system + user concatenated for hash
        if system_prompt_text and post_hygiene_prompt_text:
            payload = system_prompt_text + "\n---\n" + post_hygiene_prompt_text
            payload_hash = compute_hash(payload)
        else:
            payload_hash = ""

        return PromptSnapshot(
            snapshot_id=snapshot_id,
            query_text=query_text,
            query_timestamp=now,
            processed_query=processed_query,
            detected_intent=detected_intent,
            detected_tone=detected_tone,
            provenance=provenance,
            layers={"raw_retrieval": raw_layer, "post_hygiene": hygiene_layer},
            retrieval_metadata=retrieval_metadata or {},
            assembly_metadata=assembly_metadata or {},
            final_system_prompt_hash=system_hash,
            final_user_prompt_hash=user_hash,
            final_message_payload_hash=payload_hash,
        )


# ---------------------------------------------------------------------------
# Snapshot Replay
# ---------------------------------------------------------------------------

class SnapshotReplay:
    """Reconstructs prompts from saved snapshots. No retrieval, no web search."""

    def replay_from_layer(
        self,
        snapshot: PromptSnapshot,
        layer_name: str = "post_hygiene",
        sections_to_include: Optional[List[str]] = None,
    ) -> str:
        """Replay the prompt from a snapshot layer.

        Uses stored formatted_text, ordered by assembly_order. Does NOT call
        retrieval, web search, formatter, or any Daemon pipeline code.

        Args:
            snapshot: The snapshot to replay.
            layer_name: Which layer to replay from.
            sections_to_include: If provided, only include these section keys.
                If None, include all sections present in the layer.

        Returns:
            Reconstructed prompt string.
        """
        layer = snapshot.layers.get(layer_name)
        if layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in snapshot {snapshot.snapshot_id}")

        sections = layer.sections

        # Filter if requested
        if sections_to_include is not None:
            sections = {k: v for k, v in sections.items() if k in sections_to_include}

        # Order by assembly_order
        ordered = sorted(sections.values(), key=lambda s: s.assembly_order)

        # Join formatted texts
        return "\n\n".join(s.formatted_text for s in ordered if s.formatted_text)

    def verify_replay_exact(
        self,
        snapshot: PromptSnapshot,
        layer_name: str = "post_hygiene",
    ) -> bool:
        """Check if replay produces an exact hash match with the stored prompt."""
        layer = snapshot.layers.get(layer_name)
        if layer is None or layer.prompt_hash_exact is None:
            return False

        replayed = self.replay_from_layer(snapshot, layer_name)
        replayed_hash = compute_prompt_hash(replayed)
        return replayed_hash == layer.prompt_hash_exact

    def verify_replay_normalized(
        self,
        snapshot: PromptSnapshot,
        layer_name: str = "post_hygiene",
    ) -> bool:
        """Check if replay produces a normalized hash match with the stored prompt."""
        layer = snapshot.layers.get(layer_name)
        if layer is None or layer.prompt_hash_normalized is None:
            return False

        replayed = self.replay_from_layer(snapshot, layer_name)
        replayed_hash = compute_prompt_hash(replayed, normalize=True)
        return replayed_hash == layer.prompt_hash_normalized


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_snapshot(snapshot: PromptSnapshot, directory: Optional[str] = None) -> Path:
    """Save a snapshot to disk as JSON.

    Filename: YYYY-MM-DD_<snapshot_id>.json

    Returns:
        Path to the saved file.
    """
    directory = directory or _SNAPSHOT_DIR
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{date_str}_{snapshot.snapshot_id}.json"
    file_path = dir_path / filename

    file_path.write_text(snapshot.to_json(), encoding="utf-8")
    return file_path


def load_snapshot(path: Path) -> PromptSnapshot:
    """Load a snapshot from a JSON file."""
    text = path.read_text(encoding="utf-8")
    return PromptSnapshot.from_json(text)


# ---------------------------------------------------------------------------
# Helpers for extracting formatted sections from an assembled prompt
# ---------------------------------------------------------------------------

def extract_sections_from_prompt(prompt_text: str) -> Dict[str, str]:
    """Parse an assembled prompt back into section key -> formatted text.

    Splits on section headers like [HEADER NAME]. Returns a dict mapping
    the registry internal_key to the full section text (including header).
    Sections not in the registry are keyed as "_unknown_N".
    """
    # Split on lines starting with a bracketed header
    parts = re.split(r"(?=^\[[A-Z][A-Z ']+\])", prompt_text, flags=re.MULTILINE)

    result: Dict[str, str] = {}
    unknown_idx = 0

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Try to match the header to a registry key
        first_line = part.split("\n", 1)[0]
        key = match_header_to_key(first_line)

        if key is None:
            # Could be [CURRENT QUERY] (inner header) or [LAST EXCHANGE FOR CONTEXT]
            # These are sub-parts of current_query — skip standalone matching
            if first_line.startswith("[CURRENT QUERY]") or first_line.startswith("[LAST EXCHANGE"):
                # These are inside [CURRENT USER QUERY], handled by that section
                continue
            unknown_idx += 1
            key = f"_unknown_{unknown_idx}"

        result[key] = part

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _make_serializable(obj: Any) -> Any:
    """Convert an object to a JSON-serializable form."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    # For objects with to_dict
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    # For objects with __dict__
    if hasattr(obj, "__dict__"):
        return {k: _make_serializable(v) for k, v in obj.__dict__.items()
                if not k.startswith("_")}
    # Fallback
    return str(obj)
