"""
Daemon Self-Notes Manager.

Module Contract
- Purpose: Allow Daemon to leave structured notes for its future self —
  implementation decisions, architectural rationale, unresolved risks,
  next steps, gotchas. These are project-continuity artifacts, NOT user-facing
  reports and NOT profile facts. They are never ground truth.
- Inputs:
  - create_note(title, category, summary, confidence, ...): create and persist a note
  - detect_self_note_intent(query): regex trigger detection
  - model_manager: ModelManager for LLM topic refinement
  - chroma_store: MultiCollectionChromaStore for embedding
- Outputs:
  - DaemonNote dataclass with id, title, path, metadata
  - Markdown file written to daemon_notes/{slug}-{date}.md
  - ChromaDB embedding in daemon_self_notes collection
  - Index entry appended to daemon_notes/index.json
- Key behaviors:
  - Notes always have ground_truth=False in ChromaDB metadata
  - Notes always have source_type=daemon_self_note
  - Versioned filenames (never overwrites)
  - Path safety: output always inside daemon_notes/
  - Trigger detection: explicit only (no autonomous creation)
- Side effects:
  - File writes to daemon_notes/ directory
  - ChromaDB inserts to daemon_self_notes collection
  - LLM call for topic refinement (optional, lightweight)
- Dependencies:
  - config.app_config (DAEMON_NOTES_* constants)
  - Optional: ModelManager, MultiCollectionChromaStore
"""

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from config import app_config
from utils.logging_utils import get_logger

logger = get_logger("daemon_notes_manager")

# Valid categories and confidence levels
VALID_CATEGORIES = ("implementation", "architecture", "research", "decisions")
VALID_CONFIDENCE = ("low", "medium", "high")
VALID_STATUSES = ("active", "stale", "superseded", "resolved")


# ============================================================================
# Data model
# ============================================================================


@dataclass
class DaemonNote:
    """A structured self-note for Daemon's future sessions."""

    id: str
    title: str
    category: str  # implementation | architecture | research | decisions
    summary: str  # 2-5 sentences, required
    confidence: str  # low | medium | high
    tags: list[str] = field(default_factory=list)
    related_files: list[str] = field(default_factory=list)
    created: str = ""
    status: str = "active"  # active | stale | superseded | resolved
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_chromadb_metadata(self) -> dict[str, Any]:
        """Metadata for ChromaDB embedding. ALWAYS marks as non-ground-truth."""
        return {
            "source_type": "daemon_self_note",
            "author": "daemon",
            "ground_truth": False,
            "trust_level": "working_context",
            "note_id": self.id,
            "category": self.category,
            "status": self.status,
            "confidence": self.confidence,
            "created": self.created,
            "tags": ",".join(self.tags) if self.tags else "",
            "related_files": ",".join(self.related_files) if self.related_files else "",
            "timestamp": self.created,
        }


# ============================================================================
# Core manager
# ============================================================================


class DaemonNotesManager:
    """Creates, stores, and manages Daemon's self-notes.

    Notes are persisted as markdown files in daemon_notes/ and embedded
    into the daemon_self_notes ChromaDB collection for semantic retrieval.
    """

    def __init__(
        self,
        model_manager=None,
        chroma_store=None,
        output_dir: str | Path | None = None,
        repo_root: str | Path | None = None,
    ):
        self.model_manager = model_manager
        self.chroma_store = chroma_store

        root = Path(repo_root) if repo_root else Path(".")
        self.output_dir = Path(output_dir) if output_dir else root / app_config.DAEMON_NOTES_OUTPUT_DIR
        self.repo_root = root.resolve()

    async def create_note(
        self,
        title: str,
        category: str = "implementation",
        summary: str = "",
        confidence: str = "medium",
        tags: list[str] | None = None,
        related_files: list[str] | None = None,
        body: str = "",
    ) -> DaemonNote:
        """Create a daemon self-note: validate, write file, embed, index.

        Args:
            title: Short descriptive title.
            category: implementation | architecture | research | decisions.
            summary: 2-5 sentence summary (required).
            confidence: low | medium | high.
            tags: Optional keyword tags.
            related_files: Optional related file paths.
            body: Optional freeform markdown body below the summary.

        Returns:
            DaemonNote with path and id set.
        """
        # Validate
        if not title or len(title.strip()) < 3:
            raise ValueError("Title must be at least 3 characters")
        if not summary or len(summary.strip()) < 10:
            raise ValueError("Summary must be at least 10 characters")
        if category not in VALID_CATEGORIES:
            category = "implementation"
        if confidence not in VALID_CONFIDENCE:
            confidence = "medium"

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        slug = self._safe_slug(title)
        note_id = f"{slug}-{datetime.now().strftime('%Y-%m-%d')}"

        note = DaemonNote(
            id=note_id,
            title=title.strip(),
            category=category,
            summary=summary.strip(),
            confidence=confidence,
            tags=list(tags or [])[:10],
            related_files=list(related_files or []),
            created=now,
            status="active",
        )

        # Build markdown content
        frontmatter = self._build_frontmatter(note)
        md_body = f"# {note.title}\n\n## Summary\n{note.summary}\n"
        if body:
            md_body += f"\n{body}\n"
        content = frontmatter + "\n" + md_body

        # Write to disk
        path = self._write_versioned_file(slug, content)
        note.path = str(path)
        note.id = path.stem  # Update id to match actual filename

        # Embed into ChromaDB
        if self.chroma_store:
            try:
                embed_text = f"{note.title}\n{note.summary}"
                if body:
                    embed_text += f"\n{body[:500]}"
                self.chroma_store.add_to_collection(
                    "daemon_self_notes",
                    text=embed_text,
                    metadata=note.to_chromadb_metadata(),
                )
                logger.info(f"[DaemonNotes] Embedded note '{note.title}' into ChromaDB")
            except Exception as e:
                logger.warning(f"[DaemonNotes] ChromaDB embed failed (note still saved to disk): {e}")

        # Update index
        self._update_index({
            "id": note.id,
            "path": str(path.resolve().relative_to(self.repo_root.resolve()))
            if self.repo_root else str(path),
            "title": note.title,
            "category": note.category,
            "confidence": note.confidence,
            "created": now,
            "status": "active",
            "tags": note.tags,
        })

        logger.info(
            f"[DaemonNotes] Created note: '{note.title}' ({note.category}, "
            f"confidence={note.confidence}) -> {path}"
        )
        return note

    # ------------------------------------------------------------------
    # LLM summary generation
    # ------------------------------------------------------------------

    async def _generate_summary(self, topic: str, model_manager=None) -> str:
        """Generate a 2-5 sentence summary about a topic using LLM."""
        mm = model_manager or self.model_manager
        if not mm:
            return f"Working notes on: {topic}"

        try:
            result = await mm.generate_once(
                (
                    f"Write a concise 2-5 sentence working note about: {topic}\n\n"
                    f"This is an internal note for an AI assistant's future sessions. "
                    f"Focus on key implementation details, decisions made, or important "
                    f"context that would be useful to recall later. Be specific and factual."
                ),
                system_prompt="You write concise internal technical notes. Output only the note content, nothing else.",
                max_tokens=300,
                temperature=0.3,
            )
            if result and result.strip():
                return result.strip()
        except Exception as e:
            logger.debug(f"[DaemonNotes] Summary generation failed: {e}")

        return f"Working notes on: {topic}"

    # ------------------------------------------------------------------
    # File writing
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_slug(text: str, max_length: int = 60) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        return slug[:max_length].rstrip("-") or "note"

    def _write_versioned_file(self, slug: str, content: str) -> Path:
        """Write content to a versioned file. Never overwrites."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        path = self.output_dir / f"{slug}-{date_str}.md"
        if path.exists():
            for i in range(2, 100):
                path = self.output_dir / f"{slug}-{date_str}-{i}.md"
                if not path.exists():
                    break
            else:
                raise RuntimeError(f"Too many versions for {slug}-{date_str}")

        # Path safety
        resolved = path.resolve()
        output_resolved = self.output_dir.resolve()
        if not str(resolved).startswith(str(output_resolved)):
            raise PermissionError(
                f"Path traversal blocked: {resolved} is not inside {output_resolved}"
            )

        path.write_text(content, encoding="utf-8")
        logger.info(f"[DaemonNotes] Wrote {len(content)} chars to {path}")
        return path

    def _build_frontmatter(self, note: DaemonNote) -> str:
        lines = [
            "---",
            f"id: {note.id}",
            f'title: "{self._yaml_escape(note.title)}"',
            f"type: daemon_self_note",
            f"category: {note.category}",
            f'created: "{note.created}"',
            f"status: {note.status}",
            f"confidence: {note.confidence}",
        ]
        if note.tags:
            lines.append(f"tags: [{', '.join(note.tags)}]")
        if note.related_files:
            lines.append("related_files:")
            for f in note.related_files:
                lines.append(f'  - "{f}"')
        lines.append("---")
        return "\n".join(lines)

    @staticmethod
    def _yaml_escape(text: str | None) -> str:
        if not text:
            return ""
        return text.replace('"', '\\"').replace("\n", " ")

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _update_index(self, entry: dict[str, Any]) -> None:
        index_path = self.output_dir / "index.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        index: list[dict[str, Any]] = []
        if index_path.exists():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                if not isinstance(index, list):
                    index = []
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"[DaemonNotes] Could not read index.json: {e}")
                index = []

        index.append(entry)

        try:
            index_path.write_text(
                json.dumps(index, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as e:
            logger.error(f"[DaemonNotes] Failed to write index.json: {e}")


# ============================================================================
# Trigger detection
# ============================================================================

_SELF_NOTE_VERBS = r"(?:save|leave|write|create|make|record|log)"
_SELF_NOTE_TARGETS = r"(?:note|memo|reminder|observation)"
_SELF_REFS = r"(?:yourself|for yourself|for future|for later|for next time|for your future self)"

SELF_NOTE_TRIGGER_PATTERN = re.compile(
    # "save a note for yourself about X"
    rf"\b{_SELF_NOTE_VERBS}\b.*\b{_SELF_NOTE_TARGETS}\b.*\b{_SELF_REFS}\b"
    # "write yourself a note about X"
    rf"|\b{_SELF_NOTE_VERBS}\b\s+yourself\b.*\b{_SELF_NOTE_TARGETS}\b"
    # "note that we decided X" / "remember this for next time"
    rf"|\bnote\s+that\s+we\b"
    rf"|\bremember\s+this\s+for\s+(?:next\s+time|later|future)\b"
    # "leave an implementation note about X"
    rf"|\b{_SELF_NOTE_VERBS}\b\s+(?:a|an)\s+(?:implementation|architecture|research|decision)\s+{_SELF_NOTE_TARGETS}\b",
    re.IGNORECASE,
)


def detect_self_note_intent(query: str) -> dict[str, Any] | None:
    """Detect whether a query requests daemon self-note creation.

    Returns a dict with {topic, category} if detected, else None.
    """
    if not SELF_NOTE_TRIGGER_PATTERN.search(query):
        return None

    # Detect category from keywords
    category = "implementation"  # default
    if re.search(r"\barchitect(?:ure|ural)\b", query, re.IGNORECASE):
        category = "architecture"
    elif re.search(r"\bresearch\b", query, re.IGNORECASE):
        category = "research"
    elif re.search(r"\b(?:decision|decided|deciding)\b", query, re.IGNORECASE):
        category = "decisions"

    # Extract topic: strip self-references first, then find "about X"
    cleaned = re.sub(rf"\b{_SELF_REFS}\b", "", query, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)  # collapse double spaces

    topic_match = re.search(
        r"\b(?:about|on|regarding)\s+(.+)",
        cleaned, re.IGNORECASE,
    )
    if topic_match:
        topic = topic_match.group(1).strip()
    else:
        # Fallback: strip trigger phrases
        topic = re.sub(
            rf"\b{_SELF_NOTE_VERBS}\b|\b{_SELF_NOTE_TARGETS}\b",
            "", cleaned, flags=re.IGNORECASE,
        ).strip()

    # Clean trailing noise
    topic = re.sub(
        r"\s+(?:please|now|thanks|this time|again)\s*$",
        "", topic, flags=re.IGNORECASE,
    ).strip()
    topic = re.sub(r"^[\s,.\-:!?]+|[\s,.\-:!?]+$", "", topic)

    if not topic or len(topic) < 3:
        return None

    return {
        "topic": topic,
        "category": category,
    }
