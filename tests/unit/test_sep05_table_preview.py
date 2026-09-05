"""
Regression tests for the 2026-09-05 attachment table-preview bound
(live defect, 13:44 debug record + stored corpus entry).

A homework scope-check turn attached a 1,264-row CSV. `FileProcessor`
rendered CSVs as `manifest + "\\n\\n" + df.to_string()`, so every row was
rendered: the table block was 216,318 of the 270,559-character query
(~55K tokens), and the agentic final prompt logged "still over ceiling
after trimming: 103926/40000 tokens". The model can't use 1,264 raw rows
anyway — it needs the true shape (the manifest already gives it), column
types, summary statistics, and a small sample. The XLSX path has the same
unbounded-rows shape.

Invocation pattern mirrors tests/unit/test_sep04_attachment_turn.py
(TestTabularManifest) and
tests/unit/test_sep04_evening_fixes.py::test_csv_manifest_and_filename_use_orig_name:
end-to-end cases drive `FileProcessor._process_single_file` (the deployed
extension-routing method), never `_render_dataframe_preview` directly
except for the one item that explicitly calls for it.
"""

from unittest.mock import Mock

import pandas as pd
import pytest

from utils.file_processor import (
    ATTACHMENT_TABLE_FULL_MAX_ROWS,
    ATTACHMENT_TABLE_HEAD_ROWS,
    ATTACHMENT_TABLE_TAIL_ROWS,
    FileProcessor,
)


def _mock_file(name: str, content: bytes):
    """Minimal file-like object matching the FileProcessor contract used
    throughout tests/unit/test_file_processor.py and the 09-04 audit tests."""
    mock_file = Mock()
    mock_file.name = name
    mock_file.read = Mock(return_value=content)
    mock_file.seek = Mock()
    return mock_file


def _round_price_mean(df: pd.DataFrame) -> float:
    """Same rounding rule the implementation applies to describe() stats
    (DataFrame.round(4)) — used to compute an independently-derived
    expected value for the "assert the mean appears" case."""
    return round(float(df["Price"].mean()), 4)


class TestCsvPreviewLargeTable:
    """Case 1: 1,264-row CSV — the exact live-incident shape."""

    def _build_df(self, n=1264):
        return pd.DataFrame({
            "Id": range(1, n + 1),
            "Price": [100 * i for i in range(1, n + 1)],
            "KM": [50 * i for i in range(1, n + 1)],
            "Model": [f"Model{i % 5}" for i in range(n)],
            "Color": [f"Color{i % 3}" for i in range(n)],
        })

    def test_large_csv_preview(self, tmp_path):
        n = 1264
        df = self._build_df(n)
        path = tmp_path / "UsedCars.csv"
        df.to_csv(path, index=False)

        fp = FileProcessor()
        mock_file = _mock_file("UsedCars.csv", path.read_bytes())
        content, _size = fp._process_single_file(mock_file)

        # Manifest still reports the TRUE row count — unreduced by preview.
        assert "[UsedCars.csv: 1,264 rows × 5 columns" in content

        # First-row Id under "First 20 rows"
        first_marker = "First 20 rows:"
        last_marker = "Last 5 rows:"
        omitted_idx = content.index("rows omitted")
        first_idx = content.index(first_marker)
        last_idx = content.index(last_marker)
        assert first_idx < omitted_idx < last_idx

        first_block = content[first_idx:omitted_idx]
        last_block = content[last_idx:]
        assert "1264" not in first_block  # first row's Id is 1, not 1264
        # The first data row's Id value (1) appears in the first block.
        # Check via the actual head-row rendering to avoid false negatives
        # from generic digit collisions.
        head_str = df.head(ATTACHMENT_TABLE_HEAD_ROWS).to_string()
        assert head_str.splitlines()[1].strip() in first_block

        tail_str = df.tail(ATTACHMENT_TABLE_TAIL_ROWS).to_string()
        assert tail_str.splitlines()[-1].strip() in last_block
        assert "1264" in last_block  # last row's Id

        # Omission count: 1264 - 20 - 5 = 1239
        assert "[1,239 rows omitted" in content

        # Numeric summary block present with the correctly-rounded mean.
        assert "Numeric summary" in content
        expected_mean = _round_price_mean(df)
        assert str(expected_mean) in content

        # Bounded overall size — the whole point of the fix.
        assert len(content) < 12_000


class TestCsvPreviewSmallTable:
    """Case 2: small tables render in full, unchanged."""

    def test_ten_row_csv_renders_fully_no_omission(self, tmp_path):
        df = pd.DataFrame({
            "Id": range(1, 11),
            "Value": [f"v{i}" for i in range(1, 11)],
        })
        path = tmp_path / "small.csv"
        df.to_csv(path, index=False)

        fp = FileProcessor()
        mock_file = _mock_file("small.csv", path.read_bytes())
        content, _size = fp._process_single_file(mock_file)

        assert "[small.csv: 10 rows × 2 columns" in content
        assert "rows omitted" not in content
        # Every row's Value should be present since nothing was truncated.
        for i in range(1, 11):
            assert f"v{i}" in content


class TestCsvPreviewBoundary:
    """Case 3: exactly at the cap renders fully; one over triggers preview."""

    def _df(self, n):
        return pd.DataFrame({
            "Id": range(1, n + 1),
            "Value": [i * 2 for i in range(1, n + 1)],
        })

    def test_at_cap_renders_fully(self, tmp_path):
        n = ATTACHMENT_TABLE_FULL_MAX_ROWS
        assert n == 60
        df = self._df(n)
        path = tmp_path / "boundary60.csv"
        df.to_csv(path, index=False)

        fp = FileProcessor()
        mock_file = _mock_file("boundary60.csv", path.read_bytes())
        content, _size = fp._process_single_file(mock_file)

        assert "rows omitted" not in content
        assert f"[boundary60.csv: {n} rows × 2 columns" in content

    def test_one_over_cap_triggers_preview(self, tmp_path):
        n = ATTACHMENT_TABLE_FULL_MAX_ROWS + 1
        assert n == 61
        df = self._df(n)
        path = tmp_path / "boundary61.csv"
        df.to_csv(path, index=False)

        fp = FileProcessor()
        mock_file = _mock_file("boundary61.csv", path.read_bytes())
        content, _size = fp._process_single_file(mock_file)

        assert "rows omitted" in content
        # 61 - 20 - 5 = 36
        assert "[36 rows omitted" in content


class TestCsvPreviewSanitization:
    """Case 4: formula-injection sanitization survives the preview path."""

    def test_sanitized_form_survives_preview(self, tmp_path):
        n = 100
        raw_formula = '=HYPERLINK("x")'
        models = [raw_formula] + [f"Model{i}" for i in range(1, n)]
        df = pd.DataFrame({
            "Id": range(1, n + 1),
            "Model": models,
        })
        path = tmp_path / "formulas.csv"
        df.to_csv(path, index=False)

        fp = FileProcessor()
        expected_sanitized = fp._sanitize_csv_cell(raw_formula)
        assert expected_sanitized == "'" + raw_formula

        mock_file = _mock_file("formulas.csv", path.read_bytes())
        content, _size = fp._process_single_file(mock_file)

        # Sanitized (quote-escaped) form is present...
        assert expected_sanitized in content
        # ...and the raw formula never appears OUTSIDE that escaped form
        # (every occurrence of the raw text is inside a sanitized wrapper).
        assert content.replace(expected_sanitized, "").count(raw_formula) == 0


class TestCsvPreviewNoNumericColumns:
    """Case 5: string-only tables skip the numeric-summary block."""

    def test_string_only_csv_no_numeric_summary(self, tmp_path):
        n = 100
        df = pd.DataFrame({
            "Name": [f"name{i}" for i in range(n)],
            "Tag": [f"tag{i}" for i in range(n)],
        })
        path = tmp_path / "strings.csv"
        df.to_csv(path, index=False)

        fp = FileProcessor()
        mock_file = _mock_file("strings.csv", path.read_bytes())
        content, _size = fp._process_single_file(mock_file)

        assert "Numeric summary" not in content
        assert "First 20 rows" in content
        assert "rows omitted" in content
        assert "Last 5 rows" in content


class TestXlsxPreview:
    """Case 6: XLSX per-sheet preview mirrors the CSV bound."""

    def test_large_sheet_head_tail_only(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Data"
        ws.append(["Id", "Tag"])
        n = 100
        for i in range(1, n + 1):
            ws.append([i, f"TAG_{i:03d}"])
        path = tmp_path / "big.xlsx"
        wb.save(str(path))

        fp = FileProcessor()
        mock_file = _mock_file("big.xlsx", path.read_bytes())
        content, _size = fp._process_single_file(mock_file)

        assert f"big.xlsx [Data]: {n} rows" in content
        assert "TAG_001" in content   # first data row
        assert "TAG_100" in content   # last data row
        assert "rows omitted" in content
        # 100 - 20 - 5 = 75
        assert "75 rows omitted" in content
        # A middle row must NOT be present.
        assert "TAG_050" not in content

    def test_small_sheet_renders_fully(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Small"
        ws.append(["Id", "Tag"])
        for i in range(1, 11):
            ws.append([i, f"TAG_{i:03d}"])
        path = tmp_path / "small.xlsx"
        wb.save(str(path))

        fp = FileProcessor()
        mock_file = _mock_file("small.xlsx", path.read_bytes())
        content, _size = fp._process_single_file(mock_file)

        assert "rows omitted" not in content
        for i in range(1, 11):
            assert f"TAG_{i:03d}" in content


class TestRenderDataframePreviewDirect:
    """Case 7: direct unit test of the block ordering contract."""

    def test_block_order(self):
        n = 200
        df = pd.DataFrame({
            "A": range(n),
            "B": [i * 1.5 for i in range(n)],
            "C": [f"c{i}" for i in range(n)],
        })

        rendered = FileProcessor._render_dataframe_preview(df)

        columns_idx = rendered.index("Columns:")
        summary_idx = rendered.index("Numeric summary")
        first_idx = rendered.index("First 20 rows")
        omitted_idx = rendered.index("rows omitted")
        last_idx = rendered.index("Last 5 rows")

        assert columns_idx < summary_idx < first_idx < omitted_idx < last_idx

        # Sanity: caps are the module constants used throughout.
        assert ATTACHMENT_TABLE_HEAD_ROWS == 20
        assert ATTACHMENT_TABLE_TAIL_ROWS == 5
