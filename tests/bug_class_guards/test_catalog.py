"""Catalog-parser controls: one live structural parity test, rest synthetic.

The parser's red controls are built with ``fixtures.catalog_doc`` rather than
by mutating the real 94 KB document, so a control asserts a property of the
parser instead of a property of today's catalog debt — pinning live debt makes
*fixing* it break the suite, which is the moving-oracle shape BC-65 names.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bug_class_guards.catalog import (
    CatalogParseError,
    catalog_diagnostics,
    incident_digest,
    parse_catalog,
)

from fixtures import catalog_doc

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "docs" / "BUG_CLASSES.md"


def test_live_catalog_parses_with_independent_nonempty_discovery():
    """Body, index, DM, CM and singletons are discovered separately.

    Counts are asserted as parity plus a floor rather than exact numbers: an
    exact count would have to be edited every time a class is added, and the
    property that matters is that no section silently discovers nothing
    (BC-64) and that the index is not simply read off the body.
    """
    catalog = parse_catalog(CATALOG)
    assert set(catalog.entries) == set(catalog.index)
    assert len(catalog.entries) >= 77
    assert len(catalog.detection_methods) >= 29
    assert len(catalog.closure_methods) >= 14
    assert len(catalog.singleton_bullets) >= 16
    # Index order is NOT entry order: BC-77 was appended to family B, so the
    # parser must not assume numeric order anywhere.
    assert catalog.entries["BC-77"].line < catalog.entries["BC-18"].line
    assert list(catalog.entries) != sorted(catalog.entries)


def test_live_catalog_status_fields_all_parse_to_a_known_state():
    catalog = parse_catalog(CATALOG)
    assert all(entry.status for entry in catalog.entries.values())
    assert {record.status for record in catalog.index.values()} <= {
        "closed",
        "partial",
        "recurs",
        "open",
    }


def test_index_body_status_disagreement_is_a_finding():
    catalog = parse_catalog(catalog_doc([("BC-01", "open")], [("BC-01", "closed")]))
    findings = catalog_diagnostics(catalog)
    assert findings == ["index/body status disagreement: BC-01 index=open body=closed"]


def test_agreeing_catalog_has_no_findings():
    catalog = parse_catalog(catalog_doc([("BC-01", "closed"), ("BC-02", "partial")]))
    assert catalog_diagnostics(catalog) == []


def test_incident_digest_is_stable_and_sensitive():
    catalog = parse_catalog(catalog_doc([("BC-01", "closed")]))
    entry = catalog.entries["BC-01"]
    assert incident_digest(entry) == incident_digest(entry)
    assert incident_digest(entry) != incident_digest(entry.fields["Incidents"] + " x")


def test_id_only_in_index_is_reported():
    text = catalog_doc([("BC-01", "open"), ("BC-78", "open")], [("BC-01", "open")])
    findings = catalog_diagnostics(parse_catalog(text))
    assert any("BC-78 appears only in index" in finding for finding in findings)


def test_id_only_in_body_is_reported():
    text = catalog_doc([("BC-01", "open")], [("BC-01", "open"), ("BC-78", "open")])
    findings = catalog_diagnostics(parse_catalog(text))
    assert any("BC-78 appears only in body" in finding for finding in findings)


def test_unknown_method_reference_is_reported():
    text = catalog_doc([("BC-01", "open")]).replace(
        "- Find: DM-01 over the synthetic tree.",
        "- Find: DM-99 over the synthetic tree.",
        1,
    )
    findings = catalog_diagnostics(parse_catalog(text))
    assert any("unknown method reference: BC-01 -> DM-99" in f for f in findings)


def test_missing_field_is_a_named_parse_failure():
    text = catalog_doc([("BC-01", "open")]).replace(
        "- Status: open — synthetic status text.\n", "", 1
    )
    with pytest.raises(CatalogParseError, match="missing field"):
        parse_catalog(text)


def test_malformed_status_is_a_named_parse_failure():
    text = catalog_doc([("BC-01", "open")]).replace(
        "- Status: open — synthetic status text.", "- Status: — synthetic status text.", 1
    )
    with pytest.raises(CatalogParseError, match="malformed status|missing field"):
        parse_catalog(text)


def test_duplicate_or_merged_heading_is_rejected():
    text = catalog_doc([("BC-01", "open")]).replace(
        "### BC-01 Synthetic mechanism heading", "### BC-01 BC-78 merged heading", 1
    )
    with pytest.raises(CatalogParseError, match="malformed or merged class heading"):
        parse_catalog(text)


def test_empty_index_discovery_is_rejected():
    text = catalog_doc([("BC-01", "open")]).replace(
        "| BC-01 | synthetic mechanism | A | open |\n", "", 1
    )
    with pytest.raises(CatalogParseError, match="index discovery is empty"):
        parse_catalog(text)


def test_malformed_utf8_is_rejected():
    with pytest.raises(CatalogParseError, match="malformed UTF-8"):
        parse_catalog(b"# catalog\n\xff")
