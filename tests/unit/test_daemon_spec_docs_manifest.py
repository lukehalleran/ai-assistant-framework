"""Frozen build ships only the reviewed docs manifest (audit A01).

Incident date 2026-10-23, class: BC-59 (nearest; packaging includes unreviewed
working-tree files). daemon.spec bundled ``('docs', 'docs')``, i.e. every file under
the working-directory docs tree incl. untracked/ignored owner files. These tests
drive the deployed ``docs_data_entries`` helper the spec calls, with the tracked set
passed in (no git state is read).
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import check_bundle_manifest as cbm  # noqa: E402

SPEC = (ROOT / "daemon.spec").read_text()
EXCLUDED = re.compile(
    r"(HANDOFF|(^|/)PLAN_|(^|/)REVIEW_|AUDIT|(^|/)audits/|(^|/)execution/|_private|"
    r"RETROSPECTIVE|postmortems/|(^|/)history/|CHANGELOG|WORKPLAN|(^|/)GOALS|"
    r"internship|career|resume|METRICS|CLAIM_SUPPORT|FOLLOWUPS)", re.I)


def _tree(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    for n in ("GUIDE.md", "IGNORED_canary.md", "UNTRACKED_canary.md"):
        (d / n).write_text("x")
    (d / "sub").mkdir()
    (d / "sub" / "DEEP.md").write_text("x")
    return tmp_path


def test_canaries_absent_listed_doc_present(tmp_path):
    root = _tree(tmp_path)
    manifest = "# reviewed\ndocs/GUIDE.md  # trailing comment\n\n"
    tracked = {"docs/GUIDE.md", "docs/sub/DEEP.md"}  # ignored canary: not tracked
    entries = cbm.docs_data_entries(str(root), manifest, tracked)
    assert entries == [("docs/GUIDE.md", "docs")]
    srcs = [e[0] for e in entries]
    assert not any("canary" in s or "DEEP" in s for s in srcs)
    assert all(src != "docs" for src, _ in entries)  # never a directory entry


def test_nested_file_keeps_its_dest_dir(tmp_path):
    root = _tree(tmp_path)
    got = cbm.docs_data_entries(str(root), "docs/sub/DEEP.md", {"docs/sub/DEEP.md"})
    assert got == [("docs/sub/DEEP.md", "docs/sub")]


def test_listed_but_untracked_path_raises(tmp_path):
    root = _tree(tmp_path)
    with pytest.raises(SystemExit) as ei:
        cbm.docs_data_entries(str(root), "docs/GUIDE.md\ndocs/UNTRACKED_canary.md",
                              {"docs/GUIDE.md"})
    assert "UNTRACKED_canary.md: not git-tracked" in str(ei.value)


def test_listed_missing_or_escaping_path_raises(tmp_path):
    root = _tree(tmp_path)
    for bad in ("docs/NOPE.md", "../secret.md", "/etc/passwd", "config/config.yaml"):
        with pytest.raises(SystemExit):
            cbm.docs_data_entries(str(root), bad, {bad})


def test_spec_uses_manifest_not_directory_entry():
    assert "('docs', 'docs')" not in SPEC
    assert "check_bundle_manifest.load_docs_entries(spec_dir)" in SPEC


def test_real_manifest_parses_and_has_no_excluded_category():
    text = (ROOT / cbm.MANIFEST_RELPATH).read_text()
    paths = cbm.parse_manifest(text)
    assert paths, "manifest must list the shipped docs"
    for p in paths:
        assert p.startswith("docs/") and p.endswith(".md"), p
        assert (ROOT / p).is_file(), p
        assert not EXCLUDED.search(p), f"excluded-category file in manifest: {p}"


def test_excluded_pattern_catches_the_categories():
    for name in ("docs/HANDOFF_x.md", "docs/PLAN_1.md", "docs/execution/a.md",
                 "docs/audits/a.md", "docs/resume_review_private.md",
                 "docs/METRICS_SNAPSHOT.md", "docs/REVIEW_x.md"):
        assert EXCLUDED.search(name), name


def test_frozen_seed_paths_resolve_against_bundle(monkeypatch, tmp_path):
    import importlib
    import config.app_config as ac
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    import utils.bootstrap as bs
    monkeypatch.setattr(bs, "resolve_bundled_path", lambda p: str(tmp_path / p))
    try:
        ac = importlib.reload(ac)
        assert ac.REFERENCE_DOCS_SEED_PATHS == [str(tmp_path / "docs")]
    finally:
        monkeypatch.delattr(sys, "frozen", raising=False)
        importlib.reload(ac)
