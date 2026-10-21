"""Active-document item registration and resolution.

Incident 2026-10-18 (class: BC-51 count inferred from a label; BC-04): the
real homework PDF registered 0 numbered items because its questions are table
cells ("| Question 4 1 pts |"), while the slides' own "Part 1./2./3."
headings were counted as items and "problem 12" resolved against them
("item 12 does not exist").

Drives THE deployed ActiveDocumentRegistry / split_numbered_items.
"""
from core.active_document import ActiveDocumentRegistry, split_numbered_items

TABLE_DOC = (
    "Homework 3\n"
    "| Question 1 2 pts |\n| What is the mean? |\n"
    "| Question 2 1 pts |\n| Define variance. |\n"
    "| Question 3 1 pts |\n| State the law. |\n"
    "| Question 4 1 pts |\n| Compute the sum. |\n"
)
SLIDES = (
    "Module 05 slides\nPart 1.\nIntro text about sampling.\n"
    "Part 2.\nMore text.\nPart 3.\nSummary text.\n"
)


def _indent(text: str) -> str:
    return "\n".join("   " + ln for ln in text.split("\n"))


def test_table_cell_questions_register_four_items():
    for text in (TABLE_DOC, _indent(TABLE_DOC)):
        items = split_numbered_items(text)
        assert [i.number for i in items] == [1, 2, 3, 4]
        assert all(i.kind == "question" for i in items)


def test_part_headings_are_sections_not_questions():
    items = split_numbered_items(SLIDES)
    assert len(items) == 3
    assert all(i.kind == "section" for i in items)


def test_question_number_does_not_resolve_onto_part_headings():
    for slides in (SLIDES, _indent(SLIDES)):
        reg = ActiveDocumentRegistry()
        reg.register("Module 05 Lecture Slides.pdf", slides, "pdf", reg.next_turn())
        assert reg.resolve_navigation("problem 12", reg.next_turn()) is None
        assert reg.resolve_navigation("question 2", reg.next_turn()) is None


def test_part_request_still_resolves_on_sections():
    reg = ActiveDocumentRegistry()
    reg.register("Module 05 Lecture Slides.pdf", SLIDES, "pdf", reg.next_turn())
    nav = reg.resolve_navigation("show me part 2", reg.next_turn())
    assert nav is not None and nav.position == (2, 3)


def test_out_of_range_number_yields_no_note():
    reg = ActiveDocumentRegistry()
    reg.register("hw.pdf", TABLE_DOC, "pdf", reg.next_turn())
    assert reg.resolve_navigation("question 12", reg.next_turn()) is None
