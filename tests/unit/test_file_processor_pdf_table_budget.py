"""PDF table extraction: detect once, stop after a time budget.

Incident 2026-10-17, class: BC-69 (latency). A 7.6 MB slide-deck PDF spent ~8 s
in pdfplumber table detection; pages with tables were also detected TWICE
(extract_tables() then find_tables() for the bboxes).
"""
import logging
from pathlib import Path

import pdfplumber
import pytest

import utils.file_processor as fp
from utils.file_processor import FileProcessor

# Output of the pre-change function on this exact fixture (recorded on base).
BASE_OUTPUT = (
    "Intro paragraph before the table.\n\n"
    "| Name | Score |\n| --- | --- |\n| Rowan | 12 |\n| Tamsin | 15 |\n\n"
    "Plain second page.\n\nPlain third page."
)


@pytest.fixture()
def table_pdf(tmp_path) -> Path:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, "Intro paragraph before the table.", new_x="LMARGIN", new_y="NEXT")
    with pdf.table() as t:
        for r in [("Name", "Score"), ("Rowan", "12"), ("Tamsin", "15")]:
            row = t.row()
            for c in r:
                row.cell(c)
    pdf.add_page()
    pdf.cell(0, 10, "Plain second page.")
    pdf.add_page()
    pdf.cell(0, 10, "Plain third page.")
    out = tmp_path / "t.pdf"
    pdf.output(str(out))
    return out


def test_table_page_output_identical_to_base(table_pdf):
    assert FileProcessor._extract_pdf_with_tables(table_pdf, "t.pdf") == BASE_OUTPUT


def test_table_detection_runs_once_per_page(table_pdf, monkeypatch):
    calls = {"find": 0, "extract_tables": 0}
    orig_find = pdfplumber.page.Page.find_tables
    orig_et = pdfplumber.page.Page.extract_tables

    def counting_find(self, *a, **k):
        calls["find"] += 1
        return orig_find(self, *a, **k)

    def counting_et(self, *a, **k):
        calls["extract_tables"] += 1
        return orig_et(self, *a, **k)

    monkeypatch.setattr(pdfplumber.page.Page, "find_tables", counting_find)
    monkeypatch.setattr(pdfplumber.page.Page, "extract_tables", counting_et)
    FileProcessor._extract_pdf_with_tables(table_pdf, "t.pdf")
    # 3 pages: one detection each. extract_tables() internally calls
    # find_tables, so on base the table page counted 2 + extract_tables once.
    assert calls["find"] == 3
    assert calls["extract_tables"] == 0


def test_budget_spent_switches_to_text_only(table_pdf, monkeypatch, caplog):
    monkeypatch.setattr(fp, "PDF_TABLE_EXTRACTION_BUDGET_S", -1.0)
    seen = {"find": 0}
    orig_find = pdfplumber.page.Page.find_tables

    def counting_find(self, *a, **k):
        seen["find"] += 1
        return orig_find(self, *a, **k)

    monkeypatch.setattr(pdfplumber.page.Page, "find_tables", counting_find)
    with caplog.at_level(logging.INFO, logger="file_processor"):
        out = FileProcessor._extract_pdf_with_tables(table_pdf, "t.pdf")
    # Budget is spent before page 0 -> no detection at all, text only.
    assert seen["find"] == 0
    assert "Plain second page." in out and "Plain third page." in out
    assert "| Rowan |" not in out
    budget_lines = [r for r in caplog.records if "PDF table budget spent" in r.getMessage()]
    assert len(budget_lines) == 1
    assert "after 0/3 pages: t.pdf" in budget_lines[0].getMessage()
    summaries = [r for r in caplog.records if "PDF extracted:" in r.getMessage()]
    assert len(summaries) == 1


def test_budget_trips_midway_after_first_page(table_pdf, monkeypatch, caplog):
    clock = {"t": 0.0}

    def fake_perf():
        clock["t"] += 5.0  # every reading advances 5 s: page 0 detection costs 5 s
        return clock["t"]

    monkeypatch.setattr(fp.time, "perf_counter", fake_perf)
    monkeypatch.setattr(fp, "PDF_TABLE_EXTRACTION_BUDGET_S", 3.0)
    with caplog.at_level(logging.INFO, logger="file_processor"):
        out = FileProcessor._extract_pdf_with_tables(table_pdf, "t.pdf")
    assert "| Rowan | 12 |" in out  # page 0 still got its table
    assert "Plain third page." in out
    budget_lines = [r for r in caplog.records if "PDF table budget spent" in r.getMessage()]
    assert len(budget_lines) == 1
    assert "after 1/3 pages" in budget_lines[0].getMessage()
