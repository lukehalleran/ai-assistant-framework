"""
Agentic Search Formatters

Contract:
    - Provides AgenticFormatter with pure stateless formatting methods
    - All methods take data in and return formatted strings
    - No side effects, no state mutation, no external dependencies
    - Extracted from AgenticSearchController to reduce god-object size
"""

from typing import Dict, List


class AgenticFormatter:
    """Pure formatting methods for agentic search results."""

    def format_recent_conversations(self, conversations: List[Dict]) -> str:
        """Format recent conversations for the prompt with citation markers."""
        if not conversations:
            return ""
        lines = []
        for i, conv in enumerate(conversations, 1):
            ts = conv.get('timestamp', '')
            user_msg = conv.get('query', conv.get('user', ''))
            assistant_msg = conv.get('response', conv.get('assistant', ''))
            if user_msg:
                lines.append(f"[MEM_RECENT_{i}] {ts}: User: {user_msg[:500]}")
                if assistant_msg:
                    lines.append(f"   Daemon: {assistant_msg[:500]}")
        return "\n".join(lines)

    def format_memories(self, memories: List[Dict]) -> str:
        """Format memories for the prompt with citation markers."""
        if not memories:
            return ""
        lines = []
        for i, mem in enumerate(memories, 1):
            ts = mem.get('timestamp', '')
            content = mem.get('content', mem.get('query', ''))
            response = mem.get('response', '')
            if content:
                lines.append(f"[MEM_SEMANTIC_{i}] {ts}: {content[:400]}")
                if response:
                    lines.append(f"   Response: {response[:400]}")
        return "\n".join(lines)

    def format_summaries(self, summaries: List[Dict]) -> str:
        """Format summaries for the prompt."""
        if not summaries:
            return ""
        lines = []
        for i, s in enumerate(summaries, 1):
            content = s.get('content', s.get('summary', ''))
            ts = s.get('timestamp', '')
            if content:
                lines.append(f"{i}) [{ts}] {content[:600]}")
        return "\n".join(lines)

    def format_personal_notes(self, notes: List[Dict]) -> str:
        """Format personal notes from Obsidian for the prompt."""
        if not notes:
            return ""
        lines = []
        for i, note in enumerate(notes, 1):
            title = note.get('metadata', {}).get('title', 'Untitled')
            content = note.get('content', '')[:500]
            tags = note.get('metadata', {}).get('tags', '')
            if content:
                tag_str = f" [tags: {tags}]" if tags else ""
                lines.append(f"{i}) {title}{tag_str}: {content}")
        return "\n".join(lines)

    def format_dreams(self, dreams: List[Dict]) -> str:
        """Format dreams for the prompt."""
        if not dreams:
            return ""
        lines = []
        for i, d in enumerate(dreams, 1):
            content = d.get('content', d.get('dream', ''))
            ts = d.get('timestamp', '')
            if content:
                lines.append(f"{i}) [{ts}] {content[:400]}")
        return "\n".join(lines)

    def format_reflections(self, reflections: List[Dict]) -> str:
        """Format reflections for the prompt."""
        if not reflections:
            return ""
        lines = []
        for i, r in enumerate(reflections, 1):
            content = r.get('content', r.get('reflection', ''))
            ts = r.get('timestamp', '')
            if content:
                lines.append(f"{i}) [{ts}] {content[:400]}")
        return "\n".join(lines)

    def format_search_context(
        self, round_number: int, query: str, content: str
    ) -> str:
        """Format a single search round for context."""
        return f"[Search Round {round_number}] Query: {query}\n{content}"

    def format_wolfram_context(
        self, round_number: int, query: str, content: str
    ) -> str:
        """Format a single Wolfram Alpha computation for context."""
        return f"[Computation Round {round_number}] Query: {query}\n{content}"

    def format_sandbox_context(
        self, round_number: int, purpose: str, content: str
    ) -> str:
        """Format a single Python code execution for context."""
        return f"[Code Execution Round {round_number}] Purpose: {purpose}\n{content}"

    def format_memory_context(
        self, round_num: int, collection: str, query: str, results: str
    ) -> str:
        """Format memory search results for accumulated context."""
        return (
            f"[MEMORY SEARCH — Round {round_num} — {collection}]\n"
            f"Query: {query}\n"
            f"Results:\n{results}"
        )

    def format_expand_context(self, round_num: int, memory_id: str, results: str) -> str:
        """Format expanded results for accumulated context."""
        return (
            f"[MEMORY EXPANSION — Round {round_num} — {memory_id[:8]}]\n"
            f"{results}"
        )

    def format_file_context(
        self, round_num: int, operation: str, content: str
    ) -> str:
        """Format file access results for accumulated context."""
        return (
            f"[FILE ACCESS — Round {round_num}]\n"
            f"Operation: {operation}\n"
            f"Result:\n{content}"
        )

    def format_full_document_context(
        self, round_num: int, title: str, content: str
    ) -> str:
        """Format full document retrieval for accumulated context."""
        return (
            f"[FULL DOCUMENT — Round {round_num}]\n"
            f"Title: {title}\n"
            f"Content:\n{content}"
        )

    def format_git_stats_context(
        self, round_num: int, query: str, content: str
    ) -> str:
        """Format git stats results for accumulated context."""
        return (
            f"[GIT STATS — Round {round_num}]\n"
            f"Query: {query}\n"
            f"Result:\n{content}"
        )

    def format_memory_results(self, results: list, collection: str) -> str:
        """Format ChromaDB results into readable text for the LLM."""
        lines = []
        for i, r in enumerate(results, 1):
            content = r.get("content", "").strip()
            score = r.get("relevance_score", 0.0)
            meta = r.get("metadata", {})
            doc_id = r.get("id", "")

            header_parts = [f"[{i}]"]
            if doc_id:
                header_parts.append(f"(id: {doc_id})")
            if collection == "reference_docs":
                title = meta.get("title", "")
                section = meta.get("section", "")
                if title:
                    header_parts.append(title)
                if section:
                    header_parts.append(f"({section})")
            elif collection == "facts":
                subject = meta.get("subject", "")
                relation = meta.get("relation", "")
                if subject and relation:
                    header_parts.append(f"{subject} — {relation}")
            elif collection in ("conversations", "summaries", "reflections"):
                ts = meta.get("timestamp", "")
                if ts:
                    header_parts.append(ts[:19])

            header_parts.append(f"(score: {score:.2f})")
            header_parts.append(f"[{collection}]")
            header = " ".join(header_parts)

            if len(content) > 500:
                content = content[:500] + "..."

            lines.append(f"{header}\n{content}")

        return "\n\n".join(lines)

    def format_wiki_faiss_results(self, results: list[dict]) -> str:
        """Format FAISS wiki search results for the agentic LLM."""
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "Unknown")
            section = r.get("section", "")
            text = r.get("text", "").strip()
            score = r.get("similarity", 0.0)
            header = f"[{i}] Wikipedia: {title}"
            if section:
                header += f" / {section}"
            header += f" (score: {score:.3f})"
            lines.append(f"{header}\n{text}")
        return "\n\n".join(lines) if lines else "[No Wikipedia results found]"

    def format_expanded_results(self, result: dict) -> str:
        """Format expand result dict into readable text for the LLM."""
        error = result.get("error")
        turns = result.get("turns", [])
        collection = result.get("collection", "?")
        method = result.get("expansion_method", "timestamp_window")
        total = result.get("total_in_collection", 0)

        if method == "source_docs":
            # Summary expansion — first turn is the summary anchor, rest are source conversations
            anchor_turns = [t for t in turns if t.get("is_anchor")]
            source_turns = [t for t in turns if not t.get("is_anchor")]
            lines = [f"[Summary expanded to {len(source_turns)} source conversations]"]
            if error:
                lines.append(f"Note: {error}")
            if anchor_turns:
                lines.append(f"--- SUMMARY ---")
                lines.append(anchor_turns[0].get("content", ""))
            if source_turns:
                lines.append(f"\n--- ORIGINAL CONVERSATIONS ({len(source_turns)}) ---")
                for t in source_turns:
                    ts = t.get("timestamp", "")[:19]
                    tid = t.get("id", "")[:8]
                    content = t.get("content", "")
                    lines.append(f"[{tid}] {ts}")
                    lines.append(content)
                    lines.append("")
        else:
            lines = [f"[Expanded from {collection} | method: {method} | {len(turns)} turns shown / {total} total]"]
            if error:
                lines.append(f"Note: {error}")
            for t in turns:
                marker = "  <<<< TARGET" if t.get("is_anchor") else ""
                ts = t.get("timestamp", "")[:19]
                tid = t.get("id", "")[:8]
                content = t.get("content", "")
                lines.append(f"--- [{tid}] {ts}{marker} ---")
                lines.append(content)

        return "\n".join(lines)
