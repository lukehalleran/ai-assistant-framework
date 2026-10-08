"""Batch 4 (2026-10-08): the shared filename matcher (_name_mentioned) never
treats an English one-letter word ("a", "i") as a bare-stem mention, so the
article in "write a report" does not select a.docx; "in B please" still
selects B.docx (pinned in test_active_document.py).

class: BC-01, BC-58
"""

from core.active_document import (
    ActiveDocumentRegistry,
    ActivePassage,
    Ambiguous,
)

DOC = (
    "Question 1\nWhat is the mean?\n\n"
    "Question 2\nCompute the deviation.\n\n"
    "Question 3\nInterpret it.\n"
)
OTHER = DOC + "\nQuestion 4\nExtra.\n"


def _reg():
    reg = ActiveDocumentRegistry()
    reg.register("a.docx", DOC, "docx", reg.next_turn())
    reg.register("notes.pdf", OTHER, "pdf", reg.next_turn())
    return reg


def test_article_a_does_not_narrow_to_one_letter_stem():
    reg = _reg()
    nav = reg.resolve_navigation("write a report on the first question", reg.next_turn())
    assert isinstance(nav, Ambiguous)
    assert sorted(nav.names) == ["a.docx", "notes.pdf"]


def test_full_filename_still_selects_it():
    reg = _reg()
    nav = reg.resolve_navigation("open a.docx first question", reg.next_turn())
    assert isinstance(nav, ActivePassage)
    assert nav.document.display_name == "a.docx"


def test_normal_stem_unchanged():
    reg = ActiveDocumentRegistry()
    reg.register("syllabus.pdf", DOC, "pdf", reg.next_turn())
    reg.register("notes.pdf", OTHER, "pdf", reg.next_turn())
    nav = reg.resolve_navigation("the syllabus first question", reg.next_turn())
    assert isinstance(nav, ActivePassage)
    assert nav.document.display_name == "syllabus.pdf"
