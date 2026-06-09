"""
Document Research & Generation Pipeline.

Module Contract
- Purpose: Scan available knowledge sources (web, memory, notes, wiki),
  generate a structured markdown document (report or summary) with inline
  citations, YAML frontmatter, and a Sources section, then save it under
  documents/ with versioned filenames and an index.json entry.
- Inputs:
  - topic: str — what to research
  - doc_type: "report" | "summary"
  - focus: optional narrowing string
  - source_material: optional user-provided text (pasted proposal/content) used
    as the PRIMARY source ([INPUT_1]) when substantial — see content-aware below
  - max_sections: int (default from config)
  - model_manager: ModelManager for LLM calls
  - web_search_manager: optional WebSearchManager for Tavily
  - chroma_store: optional MultiCollectionChromaStore for memory/wiki/notes
- Outputs:
  - GeneratedDocument dataclass with path, title, sources, metadata
  - Markdown file written to documents/{type}s/{slug}-{date}.md
  - Index entry appended to documents/index.json
- Key behaviors:
  - Parallel source gathering with graceful degradation per provider
  - Source deduplication by URL/title, capped at DOCUMENT_MAX_SOURCES
  - Stable citation IDs: [WEB_1], [WIKI_1], [NOTES_1]
  - Citation validation: strips invalid IDs, logs warnings
  - Versioned filenames: slug-date.md, slug-date-2.md on collision
  - Path safety: output always inside documents/, slug sanitized
  - Atomic index: file written first, then index updated
  - Summary: single LLM draft call; Report: outline then structured draft
  - LLM-failure safety: generate_once() returns API-error sentinel strings
    ("[API Error] ...", "[CREDITS EXHAUSTED] ...") rather than raising; these
    are detected (topic refine, outline, draft) and abort generation with a
    RuntimeError so a corrupt frontmatter-only file is never written/indexed
  - Trigger detection: detect_document_intent() requires the doc-noun to be the
    (near) OBJECT of the save-verb via DOCUMENT_TRIGGER_PATTERN's bounded gap
    (_VERB_NOUN_GAP, ~4 words). Incidental co-occurrence of a save-verb and a
    doc-noun across a long multi-sentence message (e.g. a pasted homework prompt:
    "Create a new dataframe ... Print the model summary") does NOT trigger — that
    false-fire previously hijacked the turn with a "Document saved" receipt and
    swallowed the conversational reply (see gui/handlers.py:_run_doc_generation).
    Additionally, in a LONG message the trigger must be the head/tail imperative
    (_doc_trigger_is_incidental): a doc-phrase quoted deep in an analytical
    request ("Evaluate this proposal ... [a worker] may write a final report")
    is incidental and answers in chat instead.
  - Content-aware generation: when source_material is substantial it becomes the
    PRIMARY source [INPUT_1] — ranked first, rendered in full, and dominating the
    draft prompt — and web/encyclopedia search is suppressed (personal notes are
    still gathered for grounding). Without it, behavior is the prior topic-driven
    web+memory research. Fixes "evaluate THIS pasted proposal" requests that
    previously web-searched the bare topic and returned irrelevant sources.
- Side effects:
  - File writes to documents/ directory
  - LLM API calls via model_manager
  - Web search via Tavily (if available)
  - ChromaDB queries (if available)
  - Subprocess to git (not used — no git in this module)
- Dependencies:
  - Standard library + config.app_config constants
  - Optional: ModelManager, WebSearchManager, MultiCollectionChromaStore
"""

import asyncio
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

logger = get_logger("document_generator")

# LLM failure sentinels returned by ModelManager._classify_api_error(). These
# are graceful-degradation strings (e.g. "[API Error] Error code: 402 ...",
# "[CREDITS EXHAUSTED] ...") that generate_once() returns INSTEAD of raising.
# They are not content — they must never be written to disk as a document body,
# nor used as a topic or outline. See _looks_like_llm_error().
_LLM_ERROR_SENTINELS = (
    "[API Error]", "[API unavailable]", "[CREDITS EXHAUSTED]", "[RATE LIMITED]",
    "[AUTH ERROR]", "[MODEL NOT SUPPORTED]", "[MODEL NOT FOUND]", "[SERVER ERROR]",
)


def _looks_like_llm_error(text: str | None) -> bool:
    """True if an LLM call returned an error sentinel instead of real content.

    The sentinel is the leading content of the returned string, so we only scan
    the head to avoid false positives from a legitimate doc that mentions an
    error tag deep in its body.
    """
    if not text:
        return False
    head = text.lstrip()[:120]
    return any(sentinel in head for sentinel in _LLM_ERROR_SENTINELS)


# Content-aware generation. When the user pastes substantial material to be
# evaluated/synthesized (e.g. "Write a report evaluating this proposal: ..."),
# that material — not a web search on the topic string — is the PRIMARY source.
# Otherwise the generator web-searches the bare topic and returns irrelevant
# results (a "Daemon architecture" request once pulled the Anarchism Wikipedia
# article and a 1994 Unix-daemon PDF). The provided source is labelled [INPUT_1],
# ranked first, dominates the draft prompt, and suppresses web/encyclopedia
# search; personal notes are still gathered for grounding.
_PROVIDED_SOURCE_TYPE = "provided"
_PROVIDED_SOURCE_ID = "INPUT_1"
DOCUMENT_PROVIDED_MIN_CHARS = 400    # below this, "material" is too thin to anchor on
DOCUMENT_PROVIDED_MAX_CHARS = 8000   # cap of provided material injected into the prompt


# ============================================================================
# Data models
# ============================================================================


@dataclass
class DocumentSource:
    """A single source used in document generation."""

    id: str  # WEB_1, WIKI_1, NOTES_1
    title: str
    url: Optional[str]
    source_type: str  # web, wikipedia, notes, memory
    snippet: str
    relevance: float = 0.0
    retrieved_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_frontmatter_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {"id": self.id, "title": self.title, "type": self.source_type}
        if self.url:
            entry["url"] = self.url
        return entry


@dataclass
class GeneratedDocument:
    """Result of a document generation run."""

    path: str
    title: str
    doc_type: str
    topic: str
    focus: Optional[str]
    sources: list[DocumentSource]
    created_at: str
    status: str = "draft"
    sections_count: int = 0
    word_count: int = 0
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sources"] = [s.to_dict() for s in self.sources]
        return d


# ============================================================================
# Core generator
# ============================================================================


class DocumentGenerator:
    """Orchestrates research, drafting, and saving of structured documents.

    Composes existing search/retrieval utilities — does not reimplement
    Daemon's search stack. All search providers are optional; the generator
    degrades gracefully when providers are unavailable or return no results.
    """

    def __init__(
        self,
        model_manager=None,
        web_search_manager=None,
        chroma_store=None,
        output_dir: str | Path | None = None,
        repo_root: str | Path | None = None,
    ):
        self.model_manager = model_manager
        self.web_search_manager = web_search_manager
        self.chroma_store = chroma_store

        root = Path(repo_root) if repo_root else Path(".")
        self.output_dir = Path(output_dir) if output_dir else root / app_config.DOCUMENT_OUTPUT_DIR
        self.repo_root = root.resolve()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def generate(
        self,
        topic: str,
        doc_type: Literal["report", "summary"] = "report",
        focus: Optional[str] = None,
        max_sections: Optional[int] = None,
        source_material: Optional[str] = None,
    ) -> GeneratedDocument:
        """Generate a structured document on a topic.

        Args:
            topic: What to research.
            doc_type: "report" (multi-section) or "summary" (concise).
            focus: Optional narrowing, e.g. "focus on economic impact".
            max_sections: Override default section count.
            source_material: Optional user-provided text (the pasted proposal /
                content to be evaluated). When substantial, it becomes the
                PRIMARY source [INPUT_1] and web/encyclopedia search is skipped —
                the document is grounded in this material, not a topic web search.

        Returns:
            GeneratedDocument with path to saved file and metadata.

        Raises:
            ValueError: If no model_manager is available.
            RuntimeError: If the LLM returns empty content, or returns an API
                error sentinel (e.g. "[API Error] ...") — in which case no file
                is written.
        """
        if not self.model_manager:
            raise ValueError("No model_manager available — cannot generate documents")

        if doc_type not in ("report", "summary"):
            raise ValueError(f"Invalid doc_type: {doc_type!r}. Must be 'report' or 'summary'.")

        if max_sections is None:
            max_sections = (
                app_config.DOCUMENT_SUMMARY_MAX_SECTIONS
                if doc_type == "summary"
                else app_config.DOCUMENT_REPORT_MAX_SECTIONS
            )

        # 0. Refine topic via LLM if it looks noisy (conversational filler)
        topic = await self._refine_topic(topic, focus)

        logger.info(f"[DocGen] Starting {doc_type} on '{topic}' (focus={focus}, max_sections={max_sections})")

        # 1. Research
        sources = await self._gather_sources(topic, focus, source_material)
        sources = self._dedupe_and_rank(sources)
        logger.info(f"[DocGen] Gathered {len(sources)} sources after dedup")

        # 2. Outline + Draft
        if doc_type == "summary":
            markdown = await self._draft_summary(topic, focus, sources, max_sections)
        else:
            outline = await self._generate_outline(topic, focus, sources, max_sections)
            if _looks_like_llm_error(outline):
                raise RuntimeError(
                    f"LLM call failed during outline — not saving document. "
                    f"Provider returned: {outline.strip()[:160]}"
                )
            markdown = await self._draft_report(topic, focus, outline, sources)

        if not markdown or not markdown.strip():
            raise RuntimeError("LLM returned empty document content")

        # Critical: generate_once() returns API-error sentinels as plain strings
        # rather than raising (e.g. "[API Error] Error code: 402 ..."). Never
        # write one as the document body — abort so the caller reports the
        # failure instead of saving a corrupt frontmatter-only file.
        if _looks_like_llm_error(markdown):
            raise RuntimeError(
                f"LLM call failed during draft — not saving document. "
                f"Provider returned: {markdown.strip()[:160]}"
            )

        # 3. Validate citations
        markdown, sections_count = self._validate_and_clean(markdown, sources, max_sections)

        # 4. Build title from first heading or topic
        title = self._extract_title(markdown, topic)

        # 5. Build frontmatter
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        frontmatter = self._build_frontmatter(title, doc_type, topic, focus, sources, now)

        # 6. Assemble final document
        full_doc = frontmatter + "\n" + markdown

        # 7. Write to disk
        path = self._write_versioned_file(doc_type, topic, full_doc)

        # 8. Update index
        word_count = len(markdown.split())
        model_name = getattr(self.model_manager, "default_model", "") or ""
        index_entry = {
            "id": path.stem,
            "path": str(path.resolve().relative_to(self.repo_root.resolve())) if self.repo_root else str(path),
            "title": title,
            "type": doc_type,
            "topic": topic,
            "focus": focus,
            "created": now,
            "status": "draft",
            "sources_count": len(sources),
            "source_types": sorted(set(s.source_type for s in sources)),
            "model": model_name,
            "version": 1,
        }
        self._update_index(index_entry)

        result = GeneratedDocument(
            path=str(path),
            title=title,
            doc_type=doc_type,
            topic=topic,
            focus=focus,
            sources=sources,
            created_at=now,
            sections_count=sections_count,
            word_count=word_count,
            model=model_name,
        )

        logger.info(
            f"[DocGen] Generated {doc_type}: '{title}' -> {path} "
            f"({word_count} words, {len(sources)} sources, {sections_count} sections)"
        )
        return result

    # ------------------------------------------------------------------
    # Topic refinement
    # ------------------------------------------------------------------

    async def _refine_topic(self, topic: str, focus: str | None) -> str:
        """Use a lightweight LLM call to clean up a noisy topic string.

        Strips conversational filler, trailing noise, and extracts the
        core research subject. Falls back to the original topic on error.
        """
        # Skip if topic already looks clean (short, no obvious filler)
        if len(topic.split()) <= 4 and not any(w in topic.lower() for w in (
            "please", "again", "try", "this time", "now", "should", "let's",
            "bug", "fix", "work", "test",
        )):
            return topic

        try:
            result = await self.model_manager.generate_once(
                (
                    f"Extract the core research topic from this noisy user request. "
                    f"Return ONLY the topic as a short noun phrase (2-8 words). "
                    f"Strip conversational filler, meta-commentary, and format requests.\n\n"
                    f"User request: \"{topic}\"\n"
                    + (f"Focus: \"{focus}\"\n" if focus else "")
                    + "\nTopic:"
                ),
                system_prompt="You extract research topics. Output only the topic, nothing else.",
                max_tokens=30,
                temperature=0.0,
            )
            if (result and result.strip() and len(result.strip()) >= 3
                    and not _looks_like_llm_error(result)):
                refined = result.strip().strip('"').strip("'").strip()
                logger.info(f"[DocGen] Topic refined: {topic!r} → {refined!r}")
                return refined
        except Exception as e:
            logger.debug(f"[DocGen] Topic refinement failed, using original: {e}")

        return topic

    # ------------------------------------------------------------------
    # Research phase
    # ------------------------------------------------------------------

    async def _gather_sources(
        self, topic: str, focus: Optional[str] = None,
        source_material: Optional[str] = None,
    ) -> list[DocumentSource]:
        """Gather sources from all available providers in parallel.

        When `source_material` is substantial it is the PRIMARY source — web and
        encyclopedia (wiki) search are suppressed (they return topic-keyword
        noise for an "evaluate THIS" request); personal notes are still gathered
        for grounding.
        """
        query = f"{topic} {focus}" if focus else topic

        sources: list[DocumentSource] = []
        material = (source_material or "").strip()
        has_material = len(material) >= DOCUMENT_PROVIDED_MIN_CHARS
        if has_material:
            sources.append(DocumentSource(
                id=_PROVIDED_SOURCE_ID,
                title="User-provided material",
                url=None,
                source_type=_PROVIDED_SOURCE_TYPE,
                snippet=material[:DOCUMENT_PROVIDED_MAX_CHARS],
                relevance=10.0,  # dominate ranking + survive the source cap
                retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ))
            logger.info(
                f"[DocGen] Using {len(material)} chars of provided material as "
                f"primary source [{_PROVIDED_SOURCE_ID}]; web/wiki search suppressed"
            )

        tasks: dict[str, Any] = {}

        # Web search (Tavily) — skipped when the user supplied the material.
        if (self.web_search_manager and self.web_search_manager.is_available()
                and not has_material):
            tasks["web"] = self._search_web(query)

        # ChromaDB collections. Encyclopedia (wiki) only when there is no
        # provided material; personal notes always (grounding).
        if self.chroma_store:
            if not has_material:
                tasks["wiki"] = self._search_collection("wiki_knowledge", query, "wikipedia")
            tasks["notes"] = self._search_collection("obsidian_notes", query, "notes")

        if not tasks:
            if sources:
                return sources  # provided material alone is enough to draft from
            logger.warning("[DocGen] No search providers available")
            return []

        labels = list(tasks.keys())
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for label, result in zip(labels, results):
            if isinstance(result, Exception):
                logger.warning(f"[DocGen] {label} search failed: {result}")
                continue
            if isinstance(result, list):
                sources.extend(result)

        return sources

    async def _search_web(self, query: str) -> list[DocumentSource]:
        """Search via Tavily and normalize results."""
        try:
            result = await self.web_search_manager.search(
                query=query,
                max_results=app_config.DOCUMENT_MAX_SOURCES,
            )
            if not result or not hasattr(result, "pages"):
                return []

            sources = []
            for i, page in enumerate(result.pages, 1):
                sources.append(DocumentSource(
                    id=f"WEB_{i}",
                    title=page.title or f"Web result {i}",
                    url=page.url,
                    source_type="web",
                    snippet=page.snippet or page.content[:500] if page.content else "",
                    relevance=page.score if hasattr(page, "score") else 0.0,
                    retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ))
            return sources

        except Exception as e:
            logger.warning(f"[DocGen] Web search failed: {e}")
            return []

    async def _search_collection(
        self, collection: str, query: str, source_type: str,
    ) -> list[DocumentSource]:
        """Search a ChromaDB collection and normalize results."""
        try:
            results = self.chroma_store.query_collection(
                collection, query, n_results=5,
            )
            if not results:
                return []

            prefix = source_type.upper()[:4]
            sources = []
            for i, item in enumerate(results, 1):
                content = item.get("content", "")
                metadata = item.get("metadata", {})
                title = metadata.get("title", "") or metadata.get("source", "") or f"{source_type} result {i}"
                url = metadata.get("url", None) or metadata.get("source_url", None)
                score = item.get("relevance_score", 0.0)

                sources.append(DocumentSource(
                    id=f"{prefix}_{i}",
                    title=str(title)[:200],
                    url=url,
                    source_type=source_type,
                    snippet=content[:500] if content else "",
                    relevance=float(score) if score else 0.0,
                    retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ))
            return sources

        except Exception as e:
            logger.warning(f"[DocGen] {collection} search failed: {e}")
            return []

    def _dedupe_and_rank(self, sources: list[DocumentSource]) -> list[DocumentSource]:
        """Deduplicate by URL/title and rank by relevance, capped at max."""
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()
        deduped: list[DocumentSource] = []

        for src in sources:
            # Dedupe by URL
            if src.url:
                url_key = src.url.rstrip("/").lower()
                if url_key in seen_urls:
                    continue
                seen_urls.add(url_key)

            # Dedupe by title
            title_key = src.title.strip().lower()
            if title_key and title_key in seen_titles:
                continue
            if title_key:
                seen_titles.add(title_key)

            deduped.append(src)

        # Sort by relevance descending, prefer diversity of source types
        deduped.sort(key=lambda s: s.relevance, reverse=True)

        # Cap at max
        cap = app_config.DOCUMENT_MAX_SOURCES
        deduped = deduped[:cap]

        # Re-assign stable IDs after dedup. The user-provided source keeps its
        # recognizable [INPUT_1] id (and is excluded from the type counters).
        counters: dict[str, int] = {}
        for src in deduped:
            if src.source_type == _PROVIDED_SOURCE_TYPE:
                src.id = _PROVIDED_SOURCE_ID
                continue
            prefix = src.source_type.upper()[:4]
            counters[prefix] = counters.get(prefix, 0) + 1
            src.id = f"{prefix}_{counters[prefix]}"

        return deduped

    # ------------------------------------------------------------------
    # LLM outline & draft
    # ------------------------------------------------------------------

    def _format_sources_for_prompt(self, sources: list[DocumentSource]) -> str:
        """Format sources with IDs for LLM prompt injection.

        The user-provided source ([INPUT_1]) renders first and in full (the rest
        are snippet-capped) so the draft is grounded in the actual material.
        """
        if not sources:
            return "(No sources available — generate from general knowledge)"

        # Provided material first.
        ordered = sorted(
            sources, key=lambda s: 0 if s.source_type == _PROVIDED_SOURCE_TYPE else 1
        )
        lines = []
        for src in ordered:
            if src.source_type == _PROVIDED_SOURCE_TYPE:
                parts = [
                    f"[{src.id}] {src.title} — PRIMARY MATERIAL (base the document on this):",
                    src.snippet,
                ]
            else:
                parts = [f"[{src.id}] {src.title}"]
                if src.url:
                    parts.append(f"  URL: {src.url}")
                if src.snippet:
                    parts.append(f"  Content: {src.snippet[:300]}")
            lines.append("\n".join(parts))
        return "\n\n".join(lines)

    @staticmethod
    def _primary_material_instruction(sources: list[DocumentSource]) -> str:
        """Draft-prompt guidance when the user supplied the primary material."""
        if not any(s.source_type == _PROVIDED_SOURCE_TYPE for s in sources):
            return ""
        return (
            f"\nIMPORTANT: [{_PROVIDED_SOURCE_ID}] is the user's own material and is "
            f"the PRIMARY basis for this document — engage with its specifics "
            f"directly (evaluate/synthesize what it actually says). Any other "
            f"sources are secondary grounding only and must not dominate. Do NOT "
            f"pad with generic background that ignores the provided material.\n"
        )

    async def _generate_outline(
        self, topic: str, focus: Optional[str], sources: list[DocumentSource], max_sections: int,
    ) -> str:
        """Generate a structured outline for a report."""
        source_text = self._format_sources_for_prompt(sources)
        focus_line = f"\nFocus area: {focus}" if focus else ""
        primary_note = self._primary_material_instruction(sources)

        prompt = (
            f"Create an outline for a research report on: {topic}{focus_line}\n\n"
            f"Available sources:\n{source_text}\n{primary_note}\n"
            f"Requirements:\n"
            f"- Exactly {max_sections} sections (not counting introduction or conclusion)\n"
            f"- Each section should have a clear heading and 1-2 sentence description\n"
            f"- Reference source IDs (e.g. [WEB_1]) where relevant\n"
            f"- Output only the outline, no other text\n"
        )

        result = await self.model_manager.generate_once(
            prompt,
            system_prompt="You are a research analyst creating document outlines. Output structured outlines only.",
            max_tokens=1000,
            temperature=0.3,
        )
        return result or ""

    async def _draft_report(
        self, topic: str, focus: Optional[str], outline: str, sources: list[DocumentSource],
    ) -> str:
        """Draft a full report from outline and sources."""
        source_text = self._format_sources_for_prompt(sources)
        source_ids = ", ".join(f"[{s.id}]" for s in sources)
        focus_line = f"\nFocus area: {focus}" if focus else ""
        primary_note = self._primary_material_instruction(sources)

        prompt = (
            f"Write a research report on: {topic}{focus_line}\n\n"
            f"Follow this outline:\n{outline}\n\n"
            f"Available sources (cite inline using their IDs):\n{source_text}\n{primary_note}\n"
            f"Requirements:\n"
            f"- Use markdown formatting with ## headings for each section\n"
            f"- Cite sources inline using IDs like {source_ids}\n"
            f"- Every factual claim should have a citation\n"
            f"- End with a ## Sources section listing all cited sources\n"
            f"- Be thorough but concise\n"
            f"- Do NOT include YAML frontmatter — only the report body\n"
        )

        # Budget is doubled to account for reasoning tokens consumed by models
        # with native reasoning (e.g. Claude with effort=medium).
        max_tokens = app_config.DOCUMENT_REPORT_TOKEN_BUDGET * 2
        result = await self.model_manager.generate_once(
            prompt,
            system_prompt="You are a research writer producing well-cited markdown reports. Output the report only. Write ALL sections from the outline — do not stop early.",
            max_tokens=max_tokens,
            temperature=0.4,
        )
        return result or ""

    async def _draft_summary(
        self, topic: str, focus: Optional[str], sources: list[DocumentSource], max_sections: int,
    ) -> str:
        """Draft a concise summary document in one LLM call."""
        source_text = self._format_sources_for_prompt(sources)
        source_ids = ", ".join(f"[{s.id}]" for s in sources)
        focus_line = f"\nFocus area: {focus}" if focus else ""
        primary_note = self._primary_material_instruction(sources)

        prompt = (
            f"Write a concise summary document on: {topic}{focus_line}\n\n"
            f"Available sources (cite inline using their IDs):\n{source_text}\n{primary_note}\n"
            f"Requirements:\n"
            f"- Use markdown formatting\n"
            f"- Maximum {max_sections} sections with ## headings\n"
            f"- Cite sources inline using IDs like {source_ids}\n"
            f"- End with a ## Sources section listing all cited sources\n"
            f"- Keep it concise — this is a summary, not a full report\n"
            f"- Do NOT include YAML frontmatter — only the summary body\n"
        )

        max_tokens = app_config.DOCUMENT_SUMMARY_TOKEN_BUDGET * 2
        result = await self.model_manager.generate_once(
            prompt,
            system_prompt="You are a research writer producing concise, well-cited markdown summaries. Output the summary only.",
            max_tokens=max_tokens,
            temperature=0.4,
        )
        return result or ""

    # ------------------------------------------------------------------
    # Citation validation
    # ------------------------------------------------------------------

    def _validate_and_clean(
        self, markdown: str, sources: list[DocumentSource], max_sections: int,
    ) -> tuple[str, int]:
        """Validate citations and count sections. Returns (cleaned_md, section_count)."""
        valid_ids = {s.id for s in sources}

        # Find all citation IDs in the body
        cited_ids = set(re.findall(r"\[([A-Z]{3,4}_\d+)\]", markdown))

        # Strip invalid citations
        invalid = cited_ids - valid_ids
        if invalid:
            logger.warning(f"[DocGen] Stripping {len(invalid)} invalid citation IDs: {invalid}")
            for bad_id in invalid:
                markdown = markdown.replace(f"[{bad_id}]", "")

        # Count sections (## headings, excluding Sources)
        headings = re.findall(r"^##\s+(.+)$", markdown, re.MULTILINE)
        content_headings = [h for h in headings if h.strip().lower() != "sources"]
        sections_count = len(content_headings)

        # Ensure Sources section exists
        if not re.search(r"^##\s+Sources\s*$", markdown, re.MULTILINE):
            sources_section = self._build_sources_section(sources, markdown)
            markdown = markdown.rstrip() + "\n\n" + sources_section

        return markdown, sections_count

    def _build_sources_section(
        self, sources: list[DocumentSource], markdown: str,
    ) -> str:
        """Build a Sources section from the source list."""
        # Only include sources that were actually cited
        cited_ids = set(re.findall(r"\[([A-Z]{3,4}_\d+)\]", markdown))
        cited_sources = [s for s in sources if s.id in cited_ids]

        # If nothing was cited, include all sources
        if not cited_sources:
            cited_sources = sources

        lines = ["## Sources", ""]
        for src in cited_sources:
            if src.url:
                lines.append(f"- **[{src.id}]** [{src.title}]({src.url}) ({src.source_type})")
            else:
                lines.append(f"- **[{src.id}]** {src.title} ({src.source_type})")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Frontmatter
    # ------------------------------------------------------------------

    def _build_frontmatter(
        self,
        title: str,
        doc_type: str,
        topic: str,
        focus: Optional[str],
        sources: list[DocumentSource],
        created_at: str,
    ) -> str:
        """Build YAML frontmatter block."""
        # Build sources list for YAML
        source_entries = []
        for src in sources:
            entry = src.to_frontmatter_entry()
            source_entries.append(entry)

        # Manual YAML to avoid PyYAML dependency
        lines = [
            "---",
            f'title: "{self._yaml_escape(title)}"',
            f"type: {doc_type}",
            f'topic: "{self._yaml_escape(topic)}"',
            f'focus: "{self._yaml_escape(focus)}"' if focus else "focus: null",
            f'created: "{created_at}"',
            "status: draft",
            "sources:",
        ]

        for entry in source_entries:
            lines.append(f'  - id: {entry["id"]}')
            lines.append(f'    title: "{self._yaml_escape(entry["title"])}"')
            if entry.get("url"):
                lines.append(f'    url: "{entry["url"]}"')
            lines.append(f'    type: {entry["type"]}')

        lines.append("tags:")
        lines.append("  - auto-generated")
        lines.append("---")

        return "\n".join(lines)

    @staticmethod
    def _yaml_escape(text: str | None) -> str:
        """Escape a string for inline YAML."""
        if not text:
            return ""
        return text.replace('"', '\\"').replace("\n", " ")

    @staticmethod
    def _extract_title(markdown: str, fallback: str) -> str:
        """Extract title from first # heading, or use fallback."""
        match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
        if match:
            return match.group(1).strip()
        # Try ## heading
        match = re.search(r"^##\s+(.+)$", markdown, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return fallback.title()

    # ------------------------------------------------------------------
    # File writing
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_slug(text: str, max_length: int = 60) -> str:
        """Convert text to a filesystem-safe slug."""
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        return slug[:max_length].rstrip("-")

    def _write_versioned_file(self, doc_type: str, topic: str, content: str) -> Path:
        """Write content to a versioned file. Never overwrites existing files."""
        type_to_dir = {"report": "reports", "summary": "summaries"}
        subdir = self.output_dir / type_to_dir.get(doc_type, f"{doc_type}s")
        subdir.mkdir(parents=True, exist_ok=True)

        slug = self._safe_slug(topic)
        if not slug:
            slug = "document"
        date_str = datetime.now().strftime("%Y-%m-%d")

        # Find non-colliding filename
        path = subdir / f"{slug}-{date_str}.md"
        if path.exists():
            for i in range(2, 100):
                path = subdir / f"{slug}-{date_str}-{i}.md"
                if not path.exists():
                    break
            else:
                raise RuntimeError(f"Too many versions for {slug}-{date_str}")

        # Path safety: ensure output is inside documents/
        resolved = path.resolve()
        output_resolved = self.output_dir.resolve()
        if not str(resolved).startswith(str(output_resolved)):
            raise PermissionError(
                f"Path traversal blocked: {resolved} is not inside {output_resolved}"
            )

        path.write_text(content, encoding="utf-8")
        logger.info(f"[DocGen] Wrote {len(content)} chars to {path}")
        return path

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _update_index(self, entry: dict[str, Any]) -> None:
        """Append an entry to documents/index.json. Creates file if missing."""
        index_path = self.output_dir / "index.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        index: list[dict[str, Any]] = []
        if index_path.exists():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                if not isinstance(index, list):
                    logger.warning("[DocGen] index.json is not a list, resetting")
                    index = []
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"[DocGen] Could not read index.json: {e}")
                index = []

        index.append(entry)

        try:
            index_path.write_text(
                json.dumps(index, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            logger.info(f"[DocGen] Updated index.json ({len(index)} entries)")
        except OSError as e:
            logger.error(f"[DocGen] Failed to write index.json: {e}")


# ============================================================================
# Trigger detection (for orchestrator integration)
# ============================================================================

# Patterns that indicate document generation intent.
# Must contain a save/write/create verb AND a document/report/summary noun.
_SAVE_VERBS = r"(?:save|write|prepare|create|generate|make|produce|draft)"
_DOC_NOUNS = r"(?:report|document|summary|research\s+note|research\s+document|markdown)"

# Bounded gap between the save-verb and the doc-noun: up to ~4 intervening words
# (article + a modifier or two), tolerant of punctuation. This requires the
# doc-noun to be the (near) OBJECT of the save-verb — "write a report",
# "generate a markdown summary", "save the research document" — instead of
# matching the incidental co-occurrence of a save-verb and a doc-noun ANYWHERE
# in a long, multi-sentence message.
#
# Regression: a pasted homework prompt ("Create a new dataframe ... Print the
# model summary") matched the old unbounded `.*` and spuriously fired document
# generation, which then hijacked the whole turn with a "Document saved" receipt
# (see gui/handlers.py:_run_doc_generation) and swallowed the real reply.
_VERB_NOUN_GAP = r"(?:\W+\w+){0,4}?\W+"

DOCUMENT_TRIGGER_PATTERN = re.compile(
    rf"\b{_SAVE_VERBS}\b{_VERB_NOUN_GAP}\b{_DOC_NOUNS}\b"
    rf"|\b{_DOC_NOUNS}\b{_VERB_NOUN_GAP}\b{_SAVE_VERBS}\b",
    re.IGNORECASE,
)

# Incidental-trigger guard. In a LONG message the document request must be the
# imperative HEAD (or a trailing "...anyway, save that as a doc" TAIL) — a
# trigger phrase buried in the body is almost always quoted/incidental, not a
# request to research+save a document.
#
# Regression: a 2000-word *analytical* request ("Evaluate this proposal and
# produce a plan ...") fired document generation purely on the phrase "write a
# final report" quoted DEEP inside the pasted proposal (describing what a worker
# branch may do). The turn was hijacked into generic web research that ignored
# the proposal entirely. The actual ask ("Evaluate ... produce a plan") has no
# doc-noun and should answer in chat.
_DOC_INTENT_SHORT_MSG_WORDS = 60   # at/under this word count, trust the trigger anywhere
_DOC_INTENT_EDGE_CHARS = 220       # in longer messages, trigger must touch head/tail window


def _doc_trigger_is_incidental(query: str) -> bool:
    """True if the only doc-trigger phrase(s) are buried mid-body of a long message.

    Short messages are trusted wholesale. For longer ones, a genuine request
    leads ("write a report about X ...") or closes ("...save that as a summary");
    a phrase that appears only deep in the interior is treated as incidental.
    """
    if len(query.split()) <= _DOC_INTENT_SHORT_MSG_WORDS:
        return False
    n = len(query)
    tail_start = max(0, n - _DOC_INTENT_EDGE_CHARS)
    for m in DOCUMENT_TRIGGER_PATTERN.finditer(query):
        if m.start() < _DOC_INTENT_EDGE_CHARS or m.start() >= tail_start:
            return False  # at least one trigger is at the head or tail → genuine
    return True


def detect_document_intent(query: str) -> dict[str, Any] | None:
    """Detect whether a query requests document generation.

    Returns a dict with {topic, doc_type, focus} if detected, else None.
    Does NOT trigger on plain "research X" or "summarize X".
    """
    if not DOCUMENT_TRIGGER_PATTERN.search(query):
        return None

    # Incidental-trigger guard: a doc-phrase buried in the body of a long
    # analytical/evaluative message is quoted/incidental, not a request to
    # research+save a document. Let it fall through to normal generation.
    if _doc_trigger_is_incidental(query):
        return None

    # Determine doc_type
    doc_type = "report"
    if re.search(r"\bsummar(?:y|ize)\b", query, re.IGNORECASE):
        doc_type = "summary"

    # Extract topic by finding the trigger phrase and taking the topic after it.
    # Pattern: [filler] <verb> [a/an/the] <noun> [about/on/of] <TOPIC>
    _topic_extract = re.search(
        rf"\b{_SAVE_VERBS}\b\s+(?:a\s+|an\s+|the\s+|me\s+)?"
        rf"\b{_DOC_NOUNS}\b\s+(?:about|on|of|regarding|for)\s+(.+)",
        query, re.IGNORECASE,
    )
    if _topic_extract:
        topic = _topic_extract.group(1).strip()
    else:
        # Fallback: try "about <TOPIC>" anywhere after the trigger
        _about_match = re.search(
            r"\b(?:about|on|of|regarding|for)\s+(.+)",
            query, re.IGNORECASE,
        )
        if _about_match:
            topic = _about_match.group(1).strip()
        else:
            # Last resort: strip trigger phrase, take what's left
            topic = re.sub(
                rf"\b{_SAVE_VERBS}\b\s*(?:a|an|the|me)?\s*\b{_DOC_NOUNS}\b",
                "", query, flags=re.IGNORECASE,
            ).strip()

    # Remove trailing noise: "please", "now", "in docx", "in markdown", etc.
    topic = re.sub(
        r"\s+(?:please|now|thanks|thank you|this\s+time|again|instead|"
        r"in\s+(?:docx|pdf|markdown|md|html)|for\s+me)\s*$",
        "", topic, flags=re.IGNORECASE,
    ).strip()

    # Clean up leading/trailing punctuation and whitespace
    topic = re.sub(r"^[\s,.\-:!?]+|[\s,.\-:!?]+$", "", topic)

    if not topic or len(topic) < 3:
        return None

    # Check for focus phrases
    focus = None
    focus_match = re.search(
        r"\b(?:focus(?:ing)?\s+on|focused\s+on|especially|particularly)\s+(.+?)(?:\s*$)",
        topic, re.IGNORECASE,
    )
    if focus_match:
        focus = focus_match.group(1).strip()

    return {
        "topic": topic,
        "doc_type": doc_type,
        "focus": focus,
    }
