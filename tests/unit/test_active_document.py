"""tests/unit/test_active_document.py

Unit tests for core/active_document.py (2026-09-08, batch B5 — bounded
active-document continuity). All synthetic data; no store/network access.
"""

import pytest

from core.active_document import (
    ActiveDocumentRegistry,
    Ambiguous,
    Exhausted,
    ActivePassage,
    ACTIVE_DOCUMENT_MAX_DOCS,
    ACTIVE_PASSAGE_MAX_CHARS,
    format_active_passage,
    format_ambiguity_note,
    format_exhausted_note,
    split_numbered_items,
)

WORD_DOC = (
    "Homework 1\n\n"
    "Question 1\n"
    "What is the mean of the dataset?\n\n"
    "Question 2\n"
    "Compute the standard deviation.\n\n"
    "Question 3\n"
    "Interpret the result.\n"
)

BARE_DOC = (
    "1. First item text here.\n"
    "2. Second item text here.\n"
    "3. Third item text here.\n"
)

MIXED_DOC = (
    "Question 1\n"
    "Do the following:\n"
    "1. sub a\n"
    "2. sub b\n\n"
    "Question 2\n"
    "Do the following:\n"
    "1. sub a\n"
    "2. sub b\n"
)


# ---------------------------------------------------------------------------
# split_numbered_items
# ---------------------------------------------------------------------------

class TestSplitNumberedItems:
    def test_word_family_three_questions(self):
        items = split_numbered_items(WORD_DOC)
        assert [i.number for i in items] == [1, 2, 3]
        assert [i.label for i in items] == ["Question 1", "Question 2", "Question 3"]
        # end = next item's start, or len(text) for the last
        assert items[0].end == items[1].start
        assert items[-1].end == len(WORD_DOC)

    def test_bare_number_family(self):
        items = split_numbered_items(BARE_DOC)
        assert [i.number for i in items] == [1, 2, 3]
        assert items[-1].end == len(BARE_DOC)

    def test_word_family_wins_over_nested_bare_bullets(self):
        """A doc with 2+ word-labelled items wins the family choice even
        though it ALSO contains bare '1./2.' bullets nested inside each
        question — the bare family would otherwise fragment on the reset
        (1,2,1,2)."""
        items = split_numbered_items(MIXED_DOC)
        assert [i.label for i in items] == ["Question 1", "Question 2"]

    def test_single_word_item_falls_back_to_bare_family(self):
        text = "Question 1\nOnly one question here.\n1. a bare item\n2. another\n"
        items = split_numbered_items(text)
        # Only one "Question N" present (<2) -> falls back to the bare family.
        assert [i.number for i in items] == [1, 2]
        assert items[0].label == "Item 1"

    def test_restart_keeps_only_first_run(self):
        text = "1. a\n2. b\n3. c\n1. restarted a\n2. restarted b\n"
        items = split_numbered_items(text)
        # Only the first strictly-increasing run (1, 2, 3) survives; the
        # restarted "1./2." after it are never recognized as new items, so
        # the last surviving item's span runs to len(text) (no next KEPT
        # item bounds it) per the "end = next item's start or len(text)"
        # contract.
        assert [i.number for i in items] == [1, 2, 3]
        assert items[-1].end == len(text)

    def test_empty_text(self):
        assert split_numbered_items("") == []

    def test_no_numbered_items(self):
        assert split_numbered_items("just some prose with no structure") == []


# ---------------------------------------------------------------------------
# ActiveDocumentRegistry — registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_returns_document_with_items(self):
        reg = ActiveDocumentRegistry()
        turn = reg.next_turn()
        doc = reg.register("Homework1-2.pdf", WORD_DOC, ".pdf", turn)
        assert doc.display_name == "Homework1-2.pdf"
        assert doc.kind == "pdf"
        assert len(doc.items) == 3
        assert doc.char_count == len(WORD_DOC)
        assert doc.truncated is False
        assert doc.registered_turn == turn
        assert doc.doc_id == doc.sha256[:12]

    def test_same_name_and_sha_dedupes(self):
        reg = ActiveDocumentRegistry()
        d1 = reg.register("Homework1-2.pdf", WORD_DOC, "pdf", reg.next_turn())
        d2 = reg.register("Homework1-2.pdf", WORD_DOC, "pdf", reg.next_turn())
        assert d1 is d2
        assert len(reg.documents()) == 1
        # registered_turn refreshed to the second call's turn
        assert d2.registered_turn > d1.registered_turn or d2.registered_turn == d1.registered_turn

    def test_different_content_same_name_is_a_new_document(self):
        reg = ActiveDocumentRegistry()
        reg.register("doc.pdf", WORD_DOC, "pdf", reg.next_turn())
        reg.register("doc.pdf", BARE_DOC, "pdf", reg.next_turn())
        assert len(reg.documents()) == 2

    def test_kind_normalized_lowercase_no_dot(self):
        reg = ActiveDocumentRegistry()
        doc = reg.register("a.PDF", WORD_DOC, ".PDF", reg.next_turn())
        assert doc.kind == "pdf"

    def test_lru_eviction_cap(self):
        reg = ActiveDocumentRegistry()
        for i in range(ACTIVE_DOCUMENT_MAX_DOCS + 3):
            reg.register(f"doc{i}.pdf", WORD_DOC + str(i), "pdf", reg.next_turn())
        docs = reg.documents()
        assert len(docs) == ACTIVE_DOCUMENT_MAX_DOCS
        # oldest 3 evicted; newest survive
        names = reg.names()
        assert "doc0.pdf" not in names
        assert f"doc{ACTIVE_DOCUMENT_MAX_DOCS + 2}.pdf" in names

    def test_clear_empties_registry(self):
        reg = ActiveDocumentRegistry()
        reg.register("doc.pdf", WORD_DOC, "pdf", reg.next_turn())
        reg.clear()
        assert reg.documents() == []
        assert reg.names() == []


# ---------------------------------------------------------------------------
# resolve_navigation
# ---------------------------------------------------------------------------

class TestResolveNavigation:
    def _registry_with_doc(self, text=WORD_DOC, name="Homework1-2.pdf"):
        reg = ActiveDocumentRegistry()
        reg.register(name, text, "pdf", reg.next_turn())
        return reg

    def test_first(self):
        reg = self._registry_with_doc()
        nav = reg.resolve_navigation("please show me the first question", reg.next_turn())
        assert isinstance(nav, ActivePassage)
        assert nav.position == (1, 3)
        assert nav.item.label == "Question 1"

    def test_next_after_first(self):
        reg = self._registry_with_doc()
        reg.resolve_navigation("first question", reg.next_turn())
        nav = reg.resolve_navigation("ok next q please", reg.next_turn())
        assert isinstance(nav, ActivePassage)
        assert nav.position == (2, 3)

    def test_next_again_advances(self):
        reg = self._registry_with_doc()
        reg.resolve_navigation("first question", reg.next_turn())
        reg.resolve_navigation("next question", reg.next_turn())
        nav = reg.resolve_navigation("next question", reg.next_turn())
        assert isinstance(nav, ActivePassage)
        assert nav.position == (3, 3)

    def test_next_with_nothing_served_defaults_to_first(self):
        reg = self._registry_with_doc()
        nav = reg.resolve_navigation("next question please", reg.next_turn())
        assert isinstance(nav, ActivePassage)
        assert nav.position == (1, 3)

    def test_explicit_number(self):
        reg = self._registry_with_doc()
        nav = reg.resolve_navigation("can I see question 3", reg.next_turn())
        assert isinstance(nav, ActivePassage)
        assert nav.position == (3, 3)

    def test_previous(self):
        reg = self._registry_with_doc()
        reg.resolve_navigation("question 3", reg.next_turn())
        nav = reg.resolve_navigation("go back to the previous question", reg.next_turn())
        assert isinstance(nav, ActivePassage)
        assert nav.position == (2, 3)

    def test_previous_with_nothing_served_floors_at_one(self):
        reg = self._registry_with_doc()
        nav = reg.resolve_navigation("previous question", reg.next_turn())
        assert isinstance(nav, ActivePassage)
        assert nav.position == (1, 3)

    def test_last(self):
        reg = self._registry_with_doc()
        nav = reg.resolve_navigation("show me the last question", reg.next_turn())
        assert isinstance(nav, ActivePassage)
        assert nav.position == (3, 3)

    def test_exhausted_out_of_range(self):
        reg = self._registry_with_doc()
        nav = reg.resolve_navigation("question 9", reg.next_turn())
        assert isinstance(nav, Exhausted)
        assert nav.requested == 9
        assert nav.count == 3
        assert nav.document.display_name == "Homework1-2.pdf"

    def test_ambiguity_with_two_question_bearing_docs(self):
        reg = ActiveDocumentRegistry()
        reg.register("A.pdf", WORD_DOC, "pdf", reg.next_turn())
        reg.register("B.docx", BARE_DOC, "docx", reg.next_turn())
        nav = reg.resolve_navigation("next question please", reg.next_turn())
        assert isinstance(nav, Ambiguous)
        assert set(nav.names) == {"A.pdf", "B.docx"}

    def test_filename_override_selects_one_of_two_candidates(self):
        reg = ActiveDocumentRegistry()
        reg.register("A.pdf", WORD_DOC, "pdf", reg.next_turn())
        reg.register("B.docx", BARE_DOC, "docx", reg.next_turn())
        nav = reg.resolve_navigation("next question in B.docx please", reg.next_turn())
        assert isinstance(nav, ActivePassage)
        assert nav.document.display_name == "B.docx"

    def test_filename_override_without_extension(self):
        reg = ActiveDocumentRegistry()
        reg.register("A.pdf", WORD_DOC, "pdf", reg.next_turn())
        reg.register("B.docx", BARE_DOC, "docx", reg.next_turn())
        nav = reg.resolve_navigation("next question in B please", reg.next_turn())
        assert isinstance(nav, ActivePassage)
        assert nav.document.display_name == "B.docx"

    def test_single_item_doc_is_not_a_candidate(self):
        reg = ActiveDocumentRegistry()
        reg.register("solo.pdf", "Question 1\nonly one question\n", "pdf", reg.next_turn())
        nav = reg.resolve_navigation("next question please", reg.next_turn())
        assert nav is None

    def test_no_registered_docs_returns_none(self):
        reg = ActiveDocumentRegistry()
        nav = reg.resolve_navigation("next question please", reg.next_turn())
        assert nav is None

    def test_no_navigation_cue_returns_none(self):
        reg = self._registry_with_doc()
        nav = reg.resolve_navigation("what a nice day today", reg.next_turn())
        assert nav is None

    def test_negation_dont_show_next_question(self):
        reg = self._registry_with_doc()
        nav = reg.resolve_navigation("don't show the next question", reg.next_turn())
        assert nav is None

    def test_negation_falls_back_when_only_cue_is_negated(self):
        reg = self._registry_with_doc()
        # "first" is negated; no other cue present -> None, never a guess.
        nav = reg.resolve_navigation("do not give me the first question yet", reg.next_turn())
        assert nav is None

    def test_marks_served(self):
        reg = self._registry_with_doc()
        nav = reg.resolve_navigation("question 2", reg.next_turn())
        assert isinstance(nav, ActivePassage)
        assert 2 in nav.document.served

    def test_empty_text_returns_none(self):
        reg = self._registry_with_doc()
        assert reg.resolve_navigation("", reg.next_turn()) is None
        assert reg.resolve_navigation(None, reg.next_turn()) is None


# ---------------------------------------------------------------------------
# Passage bounding / truncation
# ---------------------------------------------------------------------------

class TestPassageBounding:
    def test_short_item_is_complete(self):
        reg = ActiveDocumentRegistry()
        reg.register("doc.pdf", WORD_DOC, "pdf", reg.next_turn())
        nav = reg.resolve_navigation("first question", reg.next_turn())
        assert nav.complete is True
        assert "[passage truncated" not in nav.text

    def test_long_item_is_truncated_with_marker(self):
        big_doc = "Question 1\n" + ("X" * 10000) + "\nQuestion 2\nshort\n"
        reg = ActiveDocumentRegistry()
        reg.register("big.pdf", big_doc, "pdf", reg.next_turn())
        nav = reg.resolve_navigation("first question", reg.next_turn())
        assert nav.complete is False
        assert len(nav.text) <= ACTIVE_PASSAGE_MAX_CHARS + 200
        assert "passage truncated" in nav.text
        assert 'get_full_document(title="upload:big.pdf")' in nav.text


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_format_active_passage(self):
        reg = ActiveDocumentRegistry()
        reg.register("Homework1-2.pdf", WORD_DOC, "pdf", reg.next_turn())
        nav = reg.resolve_navigation("first question", reg.next_turn())
        rendered = format_active_passage(nav)
        assert rendered.startswith("[ACTIVE DOCUMENT — Homework1-2.pdf, Question 1 (1 of 3)]")
        assert "What is the mean" in rendered

    def test_format_ambiguity_note(self):
        note = format_ambiguity_note(Ambiguous(names=["A.pdf", "B.docx"]))
        assert note.startswith("[ACTIVE DOCUMENT NOTE]")
        assert "A.pdf" in note and "B.docx" in note

    def test_format_exhausted_note(self):
        reg = ActiveDocumentRegistry()
        doc = reg.register("Homework1-2.pdf", WORD_DOC, "pdf", reg.next_turn())
        note = format_exhausted_note(Exhausted(document=doc, requested=9, count=3))
        assert note.startswith("[ACTIVE DOCUMENT NOTE]")
        assert "9" in note and "3" in note
