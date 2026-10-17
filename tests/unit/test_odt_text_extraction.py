"""odt_to_text: stdlib .odt extraction.

Incident 2026-10-17: .odt homework drafts were stored as "[Unsupported file type]". Bug class: BC-15
(attachment content never reached the model).
"""
import io
import zipfile

import pytest

from utils.odt_text import odt_to_text

NS = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
      'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
      'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"')

BODY = """<text:h text:outline-level="1">Rowan Project</text:h>
<text:p>First<text:s text:c="2"/>line<text:tab/>tabbed<text:line-break/>next.</text:p>
<text:h text:outline-level="2">Tasks</text:h>
<text:list><text:list-item><text:p>alpha</text:p></text:list-item>
<text:list-item><text:p>beta</text:p></text:list-item></text:list>
<table:table><table:table-row><table:table-cell><text:p>a</text:p></table:table-cell><table:table-cell><text:p>b</text:p></table:table-cell></table:table-row>
<table:table-row><table:table-cell><text:p>c</text:p></table:table-cell><table:table-cell><text:p>d</text:p></table:table-cell></table:table-row></table:table>
<text:p>Kept<office:annotation><text:p>SECRET NOTE</text:p></office:annotation> tail</text:p>"""

TRACKED = ('<text:tracked-changes><text:changed-region><text:deletion>'
           '<text:p>DELETED TEXT</text:p></text:deletion></text:changed-region></text:tracked-changes>')


def make_odt(body=BODY, tracked=TRACKED, extra=None):
    xml = (f'<?xml version="1.0"?><office:document-content {NS}><office:body><office:text>'
           f'{tracked}{body}</office:text></office:body></office:document-content>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        zf.writestr("content.xml", extra if extra is not None else xml)
    return buf.getvalue()


def test_structure_is_rendered():
    out = odt_to_text(make_odt())
    lines = out.split("\n")
    assert lines[0] == "# Rowan Project"
    assert "First  line\ttabbed\nnext." in out
    assert "## Tasks" in lines
    assert "- alpha" in lines and "- beta" in lines
    assert "a | b" in lines and "c | d" in lines
    assert "Kept tail" in out


def test_annotation_and_tracked_deletion_skipped():
    out = odt_to_text(make_odt())
    assert "SECRET NOTE" not in out
    assert "DELETED TEXT" not in out


def test_malformed_zip_is_value_error():
    with pytest.raises(ValueError, match="not a valid"):
        odt_to_text(b"definitely not a zip")


def test_missing_content_xml():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", "x")
    with pytest.raises(ValueError, match="content.xml"):
        odt_to_text(buf.getvalue())


def test_entity_declaration_refused():
    evil = ('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]>'
            f'<office:document-content {NS}><office:body><office:text><text:p>&a;</text:p>'
            '</office:text></office:body></office:document-content>')
    with pytest.raises(ValueError, match="entity"):
        odt_to_text(make_odt(extra=evil))


def test_invalid_xml_is_value_error():
    with pytest.raises(ValueError, match="valid XML"):
        odt_to_text(make_odt(extra="<office:document-content"))
