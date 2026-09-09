"""core/active_document.py

Module Contract
- Purpose (2026-09-08 homework-session audit, batch B5 / F2 access + F3 +
  F8): an in-memory, per-session registry of this turn's and prior turns'
  attached documents, plus deterministic navigation between their numbered
  items ("Question 1", "Task 2", "1.", "2.") — so "please show me first
  question" and, several turns later with no re-attachment, "ok next q
  please" can both be answered from the ORIGINAL attachment instead of
  asking the user to paste it again. The live incident: a homework PDF
  attached at turn 0 had already been middle-out trimmed out of
  [RECENT CONVERSATION] by the time turn 2/8 asked for the next question,
  and the upload retrieval leg needs a document cue + a 0.62 relevance bar
  that "next q" cannot supply on its own.
- Deliberately a LEAF module: stdlib + utils.logging_utils + utils.
  trigger_match (itself stdlib-only — see its module docstring) so it can be
  imported from gui/handlers.py, core/agentic/controller.py, and
  core/response_planner.py without any cycle risk.
- NOT persisted across a restart: the registry lives on the orchestrator/
  AppState in memory only (api/state.py wires it up + clears it on
  ChatSession.clear()). After a restart the existing upload-roster/
  retrieval path applies as before; this module only closes the SAME-SESSION
  gap.
- Key public API:
  - split_numbered_items(text) -> list[NumberedItem]
  - ActiveDocumentRegistry: register(), documents(), names(), clear(),
    next_turn(), resolve_navigation()
  - format_active_passage(passage) -> str
  - format_ambiguity_note(ambiguous) -> str
  - format_exhausted_note(exhausted) -> str
- Deterministic and pure apart from the registry's own served-items / LRU
  bookkeeping — no LLM calls, no store access, no network.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

from utils.logging_utils import get_logger
from utils.trigger_match import is_negated

logger = get_logger("active_document")

# Env-overridable, matching the pattern of other module-level constants
# across the codebase (config/app_config.py etc).
ACTIVE_DOCUMENT_MAX_CHARS = int(os.getenv("ACTIVE_DOCUMENT_MAX_CHARS", "300000"))
ACTIVE_PASSAGE_MAX_CHARS = int(os.getenv("ACTIVE_PASSAGE_MAX_CHARS", "4000"))
ACTIVE_DOCUMENT_MAX_DOCS = int(os.getenv("ACTIVE_DOCUMENT_MAX_DOCS", "12"))


# ---------------------------------------------------------------------------
# Numbered-item splitting
# ---------------------------------------------------------------------------

@dataclass
class NumberedItem:
    """One numbered item ("Question 1", "Task 2", "3.") in a document's text.

    `start`/`end` are character offsets into the OWNING ActiveDocument's
    (possibly-truncated) `.text` — `end` is the next surviving item's start,
    or `len(text)` for the last one.
    """
    label: str
    number: int
    start: int
    end: int


# Word-labelled family: "Question 1", "Q2", "Task 3.", "# Problem 4",
# "Part 5" — line-anchored so a mid-sentence "question 1" reference doesn't
# fragment the document. re.MULTILINE so ^ matches after every newline.
_WORD_ITEM_RE = re.compile(
    r"^\s*(?:#+\s*)?(?P<word>question|q|task|problem|exercise|part)\s*\.?\s*(?P<num>\d+)\b",
    re.IGNORECASE | re.MULTILINE,
)

# Bare-number family: "1. text", "2) text" — requires at least one
# non-whitespace character right after the marker so a lone "1." on its own
# line isn't treated as an item boundary.
_BARE_NUMBER_ITEM_RE = re.compile(
    r"^\s*(?P<num>\d{1,2})[.)]\s+(?=\S)",
    re.MULTILINE,
)

_WORD_LABELS = {
    "question": "Question",
    "q": "Q",
    "task": "Task",
    "problem": "Problem",
    "exercise": "Exercise",
    "part": "Part",
}


def _extract_run(text: str, pattern: "re.Pattern", *, label_fn) -> List[NumberedItem]:
    """Every `pattern` match in `text`, keeping only the first STRICTLY
    increasing run of numbers (a restart — a lower or repeated number after
    the run began — ends the run; later matches, even if they would resume
    increasing, are not resumed). This is what keeps a repeated bare "1./2."
    bullet list inside each of several word-labelled sections from
    fragmenting a bare-number-family document once word-labels lose the
    family choice."""
    items: List[NumberedItem] = []
    last_number: Optional[int] = None
    starts: List[int] = []
    for m in pattern.finditer(text):
        num = int(m.group("num"))
        if last_number is not None and num <= last_number:
            break
        last_number = num
        starts.append(m.start())
        items.append(NumberedItem(label=label_fn(m), number=num, start=m.start(), end=0))
    for i, item in enumerate(items):
        item.end = items[i + 1].start if i + 1 < len(items) else len(text)
    return items


def _word_label(m: "re.Match") -> str:
    word = m.group("word").lower()
    display = _WORD_LABELS.get(word, word.title())
    return f"{display} {m.group('num')}"


def _bare_label(m: "re.Match") -> str:
    return f"Item {m.group('num')}"


def split_numbered_items(text: str) -> List[NumberedItem]:
    """Split `text` into numbered items, choosing ONE family: the
    word-labelled family ("Question N"/"Task N"/...) when it yields at
    least 2 items, else the bare-number family ("N."/"N)")."""
    if not text:
        return []
    word_items = _extract_run(text, _WORD_ITEM_RE, label_fn=_word_label)
    if len(word_items) >= 2:
        return word_items
    return _extract_run(text, _BARE_NUMBER_ITEM_RE, label_fn=_bare_label)


# ---------------------------------------------------------------------------
# Document / passage / outcome dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ActiveDocument:
    doc_id: str
    display_name: str
    kind: str
    sha256: str
    char_count: int
    text: str
    truncated: bool
    registered_turn: int
    last_selected_turn: int = 0
    items: List[NumberedItem] = field(default_factory=list)
    served: Set[int] = field(default_factory=set)


@dataclass
class ActivePassage:
    document: ActiveDocument
    item: NumberedItem
    text: str
    complete: bool
    position: Tuple[int, int]  # (k, n) — k of n items


@dataclass
class Ambiguous:
    names: List[str]


@dataclass
class Exhausted:
    document: ActiveDocument
    requested: int
    count: int


NavigationResult = Union[ActivePassage, Ambiguous, Exhausted, None]


# ---------------------------------------------------------------------------
# Navigation grammar
# ---------------------------------------------------------------------------

_EXPLICIT_NUMBER_RE = re.compile(r"\b(?:question|q|task|problem|part)\s*(\d+)\b", re.IGNORECASE)
_FIRST_RE = re.compile(r"\bfirst\b", re.IGNORECASE)
_NEXT_RE = re.compile(r"\b(?:next|following)\b", re.IGNORECASE)
_PREVIOUS_RE = re.compile(r"\b(?:previous|prev)\b", re.IGNORECASE)
_LAST_RE = re.compile(r"\blast\b", re.IGNORECASE)


def _first_non_negated(pattern: "re.Pattern", lower_text: str):
    for m in pattern.finditer(lower_text):
        if not is_negated(lower_text, m.start()):
            return m
    return None


def _resolve_target_number(lower_text: str, document: ActiveDocument) -> Optional[int]:
    """Priority order: an explicit number beats the relative-position
    words, which are otherwise checked first→next→previous→last."""
    m = _first_non_negated(_EXPLICIT_NUMBER_RE, lower_text)
    if m:
        return int(m.group(1))
    if _first_non_negated(_FIRST_RE, lower_text):
        return 1
    if _first_non_negated(_NEXT_RE, lower_text):
        served_max = max(document.served) if document.served else 0
        return served_max + 1
    if _first_non_negated(_PREVIOUS_RE, lower_text):
        served_max = max(document.served) if document.served else 0
        return max(served_max - 1, 1)
    if _first_non_negated(_LAST_RE, lower_text):
        return len(document.items)
    return None


def _name_mentioned(lower_text: str, display_name: str) -> bool:
    """Case-insensitive filename mention, with or without its extension.

    Boundary-checked (not a bare substring test): a short stem like "A" (from
    "A.pdf") must not match because the letter happens to occur inside an
    unrelated word ("plea-s-e"). A mention is either the full name or the
    bare stem, each flanked by a non-alphanumeric character (or start/end of
    string) on both sides.
    """
    name = (display_name or "").lower()
    if not name:
        return False
    stem = os.path.splitext(name)[0]
    candidates = {name}
    if stem:
        candidates.add(stem)
    for candidate in candidates:
        pattern = r"(?<![a-z0-9])" + re.escape(candidate) + r"(?![a-z0-9])"
        if re.search(pattern, lower_text):
            return True
    return False


def _bound_passage(document: ActiveDocument, item: NumberedItem) -> Tuple[str, bool]:
    full = document.text[item.start:item.end]
    if len(full) <= ACTIVE_PASSAGE_MAX_CHARS:
        return full, True
    truncated = full[:ACTIVE_PASSAGE_MAX_CHARS]
    marker = (
        f"\n[passage truncated — full text via "
        f'get_full_document(title="upload:{document.display_name}")]'
    )
    return truncated + marker, False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ActiveDocumentRegistry:
    """In-memory, per-session registry of attached documents + numbered-item
    navigation state. See module docstring — attached to the orchestrator/
    AppState, cleared on session reset, never persisted."""

    def __init__(self) -> None:
        self._docs: "OrderedDict[str, ActiveDocument]" = OrderedDict()
        self._by_key: Dict[Tuple[str, str], str] = {}
        self._turn: int = 0

    def next_turn(self) -> int:
        """Monotonically increasing turn counter for callers that don't
        already track one (e.g. the legacy Gradio path)."""
        self._turn += 1
        return self._turn

    def register(self, display_name: str, text: str, kind: str, turn: int) -> ActiveDocument:
        """Register (or refresh) an attached document.

        Same (display_name.lower(), sha256) as an already-registered
        document refreshes `registered_turn` and returns the EXISTING
        object (never a duplicate). LRU-evicts the least-recently-touched
        document once the registry exceeds ACTIVE_DOCUMENT_MAX_DOCS.
        """
        text = text or ""
        sha = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        key = (display_name.strip().lower(), sha)

        existing_id = self._by_key.get(key)
        if existing_id is not None and existing_id in self._docs:
            existing = self._docs[existing_id]
            existing.registered_turn = turn
            self._touch(existing_id)
            return existing

        char_count = len(text)
        truncated = char_count > ACTIVE_DOCUMENT_MAX_CHARS
        stored_text = text[:ACTIVE_DOCUMENT_MAX_CHARS] if truncated else text
        doc = ActiveDocument(
            doc_id=sha[:12],
            display_name=display_name,
            kind=(kind or "").lower().lstrip("."),
            sha256=sha,
            char_count=char_count,
            text=stored_text,
            truncated=truncated,
            registered_turn=turn,
            last_selected_turn=turn,
            items=split_numbered_items(stored_text),
            served=set(),
        )
        self._docs[doc.doc_id] = doc
        self._by_key[key] = doc.doc_id
        self._touch(doc.doc_id)
        self._evict_if_needed()
        return doc

    def documents(self) -> List[ActiveDocument]:
        return list(self._docs.values())

    def names(self) -> List[str]:
        return [d.display_name for d in self._docs.values()]

    def clear(self) -> None:
        self._docs.clear()
        self._by_key.clear()
        self._turn = 0

    def resolve_navigation(self, user_text: str, turn: int) -> NavigationResult:
        """Resolve a task-navigation message ("next q please", "question 3",
        "the last one") against this session's registered documents.

        Candidates = registered documents with >=2 numbered items. A
        mentioned display name narrows to that document; otherwise exactly
        one candidate is used, several is an Ambiguous(names) note, none is
        None. Never chooses by newest timestamp when ambiguous. Deterministic
        aside from marking the resolved item `served` and touching the LRU.
        """
        text = (user_text or "").strip()
        if not text:
            return None
        lower = text.lower()

        candidates = [d for d in self._docs.values() if len(d.items) >= 2]
        if not candidates:
            return None

        named = [d for d in candidates if _name_mentioned(lower, d.display_name)]
        if named:
            candidates = named
        if len(candidates) > 1:
            return Ambiguous(names=[d.display_name for d in candidates])
        document = candidates[0]

        target = _resolve_target_number(lower, document)
        if target is None:
            return None

        if target < 1 or target > len(document.items):
            return Exhausted(document=document, requested=target, count=len(document.items))

        item = document.items[target - 1]
        passage_text, complete = _bound_passage(document, item)
        document.served.add(target)
        document.last_selected_turn = turn
        self._touch(document.doc_id)
        return ActivePassage(
            document=document,
            item=item,
            text=passage_text,
            complete=complete,
            position=(target, len(document.items)),
        )

    # -- internal LRU bookkeeping -----------------------------------------

    def _touch(self, doc_id: str) -> None:
        if doc_id in self._docs:
            self._docs.move_to_end(doc_id)

    def _evict_if_needed(self) -> None:
        while len(self._docs) > ACTIVE_DOCUMENT_MAX_DOCS:
            oldest_id, oldest_doc = self._docs.popitem(last=False)
            stale_keys = [k for k, v in self._by_key.items() if v == oldest_id]
            for k in stale_keys:
                del self._by_key[k]
            logger.info(
                f"[ActiveDocument] Evicted LRU document '{oldest_doc.display_name}' "
                f"(doc_id={oldest_id}) — registry over ACTIVE_DOCUMENT_MAX_DOCS"
            )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_active_passage(p: ActivePassage) -> str:
    k, n = p.position
    return f"[ACTIVE DOCUMENT — {p.document.display_name}, {p.item.label} ({k} of {n})]\n{p.text}"


def format_ambiguity_note(a: Ambiguous) -> str:
    names = ", ".join(a.names)
    return (
        "[ACTIVE DOCUMENT NOTE] Several attached documents contain numbered "
        f"items: {names} — ask which one before answering."
    )


def format_exhausted_note(e: Exhausted) -> str:
    return (
        f"[ACTIVE DOCUMENT NOTE] {e.document.display_name} has {e.count} "
        f"numbered items; item {e.requested} does not exist."
    )
