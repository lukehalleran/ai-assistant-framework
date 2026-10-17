"""process_files_structured: .R/.Rmd scripts, .odt documents and .xls attachments.

Incident 2026-10-17: 8 reference_docs uploads (R homework scripts, LibreOffice drafts, CSV text
named *.csv.xls) were stored as "[Unsupported file type]". Bug classes: BC-15 (attachment content
never reached the model), BC-91 (machinery text stored as content).
"""
import asyncio
import builtins
import io
import zipfile
from types import SimpleNamespace

from utils import file_processor as fp
from tests.unit.test_odt_text_extraction import make_odt

R_CODE = "fit <- lm(y ~ x, data = rowan_df)\nsummary(fit)\n"
CSV_TEXT = "name,score\nTamsin,3\n=HYPERLINK(\"http://x.example\"),4\n"


def _upload(name, data):
    return SimpleNamespace(name=name, orig_name=name, read=lambda: data)


def _run(files):
    return asyncio.run(fp.FileProcessor().process_files_structured("hi", files))


def _zip(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for n, d in members:
            zf.writestr(n, d)
    return buf.getvalue()


def test_r_and_rmd_come_out_as_fenced_r_text():
    r = _run([_upload("analysis.R", R_CODE.encode()), _upload("notes.Rmd", b"# T\n```{r}\n1+1\n```\n")])
    assert [d.filename for d in r.documents] == ["analysis.R", "notes.Rmd"]
    assert "```r\nfit <- lm(y ~ x, data = rowan_df)" in r.text_content
    assert "Unsupported" not in r.text_content


def test_odt_comes_out_as_text():
    r = _run([_upload("draft.odt", make_odt())])
    assert [d.filename for d in r.documents] == ["draft.odt"]
    assert "# Rowan Project" in r.text_content


def test_broken_odt_is_error_entry_not_document():
    r = _run([_upload("bad.odt", b"nope")])
    assert r.documents == [] and r.processed_files[0].error
    assert "not a valid" in r.text_content


def test_csv_text_xls_uses_csv_path_with_sanitizing():
    r = _run([_upload("data.csv.xls", CSV_TEXT.encode())])
    assert [d.filename for d in r.documents] == ["data.csv.xls"]
    assert "Tamsin" in r.text_content
    assert "'=HYPERLINK" in r.text_content  # sanitized like .csv
    assert "data.csv.xls: 2 rows" in r.text_content or "2 rows" in r.text_content


def test_ole2_xls_without_xlrd_is_error_entry(monkeypatch):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "xlrd":
            raise ImportError("no xlrd")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    r = _run([_upload("old.xls", bytes.fromhex("D0CF11E0A1B11AE1") + b"\x00" * 64)])
    assert r.documents == []
    assert r.processed_files[0].error and "xlrd" in r.processed_files[0].error
    assert "xlrd" in r.text_content


def test_html_xls_is_error_entry():
    r = _run([_upload("page.xls", b"  <html><table><tr><td>1</td></tr></table></html>")])
    assert r.documents == []
    assert "HTML saved as .xls" in r.text_content


def test_zip_with_r_and_odt_expands_both():
    z = _zip([("a.R", R_CODE), ("b.odt", make_odt())])
    r = _run([_upload("pack.zip", z)])
    assert sorted(d.filename for d in r.documents) == ["pack.zip/a.R", "pack.zip/b.odt"]
    assert "```r" in r.text_content and "# Rowan Project" in r.text_content


def test_macos_zip_metadata_does_not_count_toward_member_cap():
    """2026-10-17 live: a macOS-made zip of 26 lecture .srt files carried 51
    __MACOSX/._* AppleDouble entries (plus a directory entry) and was refused as
    "77 members (max 50)". Metadata is never read, so it never counts."""
    ZIP_MAX_MEMBERS = fp.ZIP_MAX_MEMBERS
    srt = "1\n00:00:01,000 --> 00:00:03,000\nInvented lecture line.\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("__MACOSX/", "")
        for n in range(1, 27):
            zf.writestr(f"Lecture_L{n}.srt", srt)
            zf.writestr(f"__MACOSX/._Lecture_L{n}.srt", "\x00" * 4096)  # zero-padded: >100:1
        for n in range(27, 52):
            zf.writestr(f"__MACOSX/._Lecture_L{n}.srt", "\x00" * 4096)
    assert 26 < ZIP_MAX_MEMBERS < 78
    result = _run([_upload("Module 5.zip", buf.getvalue())])
    names = [d.filename for d in result.documents]
    assert len(names) == 26
    assert names[0] == "Module 5.zip/Lecture_L1.srt" and names[-1] == "Module 5.zip/Lecture_L26.srt"
    assert not [p for p in result.processed_files if p.error]
