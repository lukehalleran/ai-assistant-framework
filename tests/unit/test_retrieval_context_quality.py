"""
Regression tests for the 2026-07-25 retrieval-quality batch:

1. Self-docs suppression — Daemon's own tone-detection docs (crisis keyword
   lists, response-length rules) were retrieved into every distress prompt
   because distress language semantically matches them. reference_docs is now
   suppressed on distress/emotional-support turns.

2. Near-empty note filter — image-only Obsidian note chunks embed as noise
   and scored 0.64 "relevance" against "The cats are here at least";
   get_personal_notes drops chunks with < PERSONAL_NOTES_MIN_CHARS of prose
   (unless the visual-intent gate loaded their images).

3. Read-time thinking strip — a Feb-2026 stored <thinking> block surfaced
   verbatim in a live prompt; _format_memory now applies the storage-boundary
   sanitize at the retrieval boundary too.
"""

from core.prompt.builder import _should_suppress_reference_docs
from core.prompt.gatherer_knowledge import (
    PERSONAL_NOTES_MIN_CHARS,
    _note_text_substance,
)
from core.prompt.formatter import _strip_stored_thinking


class TestSelfDocsSuppression:
    def test_distress_suppresses(self):
        assert _should_suppress_reference_docs(False, True, None)

    def test_emotional_support_intent_suppresses(self):
        assert _should_suppress_reference_docs(False, False, "emotional_support")
        assert _should_suppress_reference_docs(False, False, "EMOTIONAL_SUPPORT")

    def test_files_suppress_preserved(self):
        assert _should_suppress_reference_docs(True, False, None)

    def test_normal_turns_keep_self_docs(self):
        assert not _should_suppress_reference_docs(False, False, None)
        assert not _should_suppress_reference_docs(False, False, "technical_help")
        assert not _should_suppress_reference_docs(False, False, "meta_conversational")


class TestNoteSubstance:
    def test_image_only_chunk_is_empty(self):
        # The live offenders: a heading-ish line plus a pasted-image embed.
        assert _note_text_substance("![[Pasted image 20241117122358.png]]") == 0
        assert _note_text_substance("![alt](https://example.com/img.png)") == 0

    def test_live_offender_shapes_fail_threshold(self):
        # "IID Summations of Various Distributions" note body = one embed.
        assert _note_text_substance("![[Pasted image 20241117122519.png]]\n") < PERSONAL_NOTES_MIN_CHARS
        # Single-letter note ("A") from the dump.
        assert _note_text_substance("A") < PERSONAL_NOTES_MIN_CHARS

    def test_real_note_passes(self):
        text = (
            "The 100pth percentile is the smallest value such that the CDF is "
            "at least p. For discrete X order the values first."
        )
        assert _note_text_substance(text) >= PERSONAL_NOTES_MIN_CHARS

    def test_mixed_note_counts_only_prose(self):
        text = "![[Pasted image.png]]\nshort caption"
        assert _note_text_substance(text) == len("short caption")


class TestReadTimeThinkingStrip:
    def test_leading_thinking_block_removed_answer_kept(self):
        # Shape of the Feb-07 doc that surfaced live: leading tagged block,
        # real answer after it.
        text = (
            "<thinking>\nLuke is in a really dark place tonight. I need to be "
            "careful here and consider safety.\n</thinking>\n\n"
            "That's a lot in your system at once, Luke."
        )
        out = _strip_stored_thinking(text)
        assert "<thinking>" not in out
        assert "dark place" not in out
        assert out.startswith("That's a lot in your system")

    def test_clean_text_untouched_fast_path(self):
        text = "Regular response with no tags at all."
        assert _strip_stored_thinking(text) is text

    def test_all_reasoning_returns_original(self):
        # Whole text is reasoning: dropping content at render time is worse
        # than showing it — conservative fallback keeps the original.
        text = "<thinking>only reasoning, never closed into an answer"
        assert _strip_stored_thinking(text) == text

    def test_reasoning_tag_family_covered(self):
        text = "<reasoning>internal notes</reasoning>\n\nThe real answer."
        out = _strip_stored_thinking(text)
        assert "internal notes" not in out
        assert "The real answer." in out
