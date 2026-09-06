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


class TestUploadStalenessGate:
    """2026-08-05: months-old homework docs and phone photos were injected
    into every turn — hybrid retrieval always returns SOME top-N. An upload
    now surfaces only while fresh (<= USER_UPLOADS_MAX_AGE_DAYS) or when
    relevance clears USER_UPLOADS_MIN_RELEVANCE; undated legacy docs must
    clear the relevance bar."""

    def _doc(self, relevance=0.0, ts=None, match_type=None):
        from datetime import datetime
        meta = {"type": "user_upload"}
        if ts is not None:
            meta["timestamp"] = ts.isoformat() if isinstance(ts, datetime) else ts
        doc = {"relevance_score": relevance, "metadata": meta}
        if match_type is not None:
            doc["match_type"] = match_type
        return doc

    def test_stale_irrelevant_upload_dropped(self):
        from datetime import datetime, timedelta
        from core.prompt.gatherer_knowledge import _upload_is_live
        old = datetime.now() - timedelta(days=180)
        assert not _upload_is_live(self._doc(relevance=0.2, ts=old))

    def test_fresh_upload_survives_regardless_of_relevance(self):
        from datetime import datetime, timedelta
        from core.prompt.gatherer_knowledge import _upload_is_live
        fresh = datetime.now() - timedelta(days=1)
        assert _upload_is_live(self._doc(relevance=0.0, ts=fresh))

    def test_relevant_old_upload_survives(self):
        # Fix 1.1 (2026-09-06): only a SEMANTIC score is real relevance
        # evidence — this is the case the test always meant to pin.
        from datetime import datetime, timedelta
        from core.prompt.gatherer_knowledge import _upload_is_live, USER_UPLOADS_MIN_RELEVANCE
        old = datetime.now() - timedelta(days=180)
        assert _upload_is_live(
            self._doc(relevance=USER_UPLOADS_MIN_RELEVANCE + 0.05, ts=old, match_type="semantic")
        )

    def test_keyword_typed_old_upload_dropped_despite_high_score(self):
        # Fix 1.1 (2026-09-06): reference_docs_manager._keyword_search's
        # empty-section containment bug scored every single-chunk upload
        # chunk 0.9 on EVERY query regardless of actual relevance — a
        # keyword-typed score must not admit a stale, unrelated upload on
        # its own (the freshness/document-cue leg is unaffected).
        from datetime import datetime, timedelta
        from core.prompt.gatherer_knowledge import _upload_is_live, USER_UPLOADS_MIN_RELEVANCE
        old = datetime.now() - timedelta(days=180)
        assert not _upload_is_live(
            self._doc(relevance=USER_UPLOADS_MIN_RELEVANCE + 0.28, ts=old, match_type="keyword"),
            query="rest after taking my medication",
        )

    def test_undated_doc_needs_relevance(self):
        from core.prompt.gatherer_knowledge import _upload_is_live
        assert not _upload_is_live(self._doc(relevance=0.2))
        assert _upload_is_live(self._doc(relevance=0.9))

    def test_garbage_timestamp_treated_as_undated(self):
        from core.prompt.gatherer_knowledge import _upload_is_live
        assert not _upload_is_live(self._doc(relevance=0.1, ts="not-a-date"))

    def test_relevance_bar_calibrated_for_bge_space(self):
        # 2026-08-14: the original 0.5 bar = cosine 0.5 under the store's
        # rel = 1/(1+2(1-cos)) mapping — any-text-vs-any-text in bge space.
        # A homework docx and two year-old photos cleared it on an emotional
        # check-in turn. The bar must sit above the memory gate's 0.60
        # ordinary-relevance cosine (rel ≈ 0.556) to mean CLEARLY relevant.
        from core.prompt.gatherer_knowledge import USER_UPLOADS_MIN_RELEVANCE
        assert USER_UPLOADS_MIN_RELEVANCE >= 0.60

    def test_old_upload_at_legacy_bar_now_dropped(self):
        # The live-incident shape: months-old doc scoring in the 0.5s.
        from datetime import datetime, timedelta
        from core.prompt.gatherer_knowledge import _upload_is_live
        old = datetime.now() - timedelta(days=200)
        assert not _upload_is_live(self._doc(relevance=0.55, ts=old))


class TestSelfDocsAllowGate:
    """2026-08-05: a conversational-tone personal pain turn pulled 15
    [DAEMON DOCUMENTATION] chunks (tone docs, synthesis notes, a lecture
    transcript) — distress suppression didn't fire because tone classified
    CONVERSATIONAL. Self-docs are now ALLOW-listed: meta/technical/project
    intents or an explicit self-referential query cue."""

    def _gate(self, query, intent=None, files=False, distress=False):
        from core.prompt.builder import _should_include_reference_docs
        return _should_include_reference_docs(query, intent, files, distress)

    def test_personal_pain_turn_excluded(self):
        # The live offender: general intent, conversational tone.
        q = ("Yeah. Took more mit about and hour and a half ago and the pain "
             "is probably worse. Ugh. It's so frustrating cuz my mind is working fine")
        assert not self._gate(q, intent="general")

    def test_meta_technical_project_intents_included(self):
        assert self._gate("how did that go", intent="meta_conversational")
        assert self._gate("getting a dimension mismatch error", intent="technical_help")
        assert self._gate("let's work on the retriever", intent="project_work")

    def test_self_referential_cue_includes_on_general_intent(self):
        assert self._gate("how does your memory work?", intent="general")
        assert self._gate("what's the truth score on that fact", intent="general")
        assert self._gate("why did daemon pull that doc", intent=None)
        assert self._gate("how do you decide when to search", intent="general")

    def test_suppressions_still_win(self):
        # Distress / uploads / emotional_support beat the allow-list.
        assert not self._gate("how does your memory work?", intent="technical_help", distress=True)
        assert not self._gate("how does your memory work?", intent="technical_help", files=True)
        assert not self._gate("how does your memory work?", intent="emotional_support")

    def test_ordinary_conversational_turns_excluded(self):
        assert not self._gate("I slept okay, two 4 hour shifts", intent="general")
        assert not self._gate("what should I make for dinner", intent="general")
        assert not self._gate("my mom got home yesterday", intent="casual_social")

    def test_update_fix_cues_included(self):
        # 2026-08-21: the owner asked Daemon to "check out your updates based
        # on docs" after a fix batch — general intent, no cue matched, so the
        # freshly re-seeded self-docs never surfaced. "your updates/fixes/
        # changes" and "fixes/updates on|to|in you" are now cues.
        assert self._gate("check out your updates based on docs", intent="general")
        assert self._gate("I did a decent amount of fixes on you today", intent="general")
        assert self._gate("any updates to you from the audit?", intent="general")
        # non-self uses stay excluded
        assert not self._gate("I need to buy updates for my phone plan", intent="general")
        assert not self._gate("the fixes on the car are done", intent="casual_social")
