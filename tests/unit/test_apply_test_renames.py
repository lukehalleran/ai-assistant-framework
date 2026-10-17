"""Applier tests on SYNTHETIC trees (2026-10-16, plan v2 sections 4.2/4.5; class: convention drift).

Every flow runs ``scripts/apply_test_renames.py`` against a throwaway git repo built under tmp_path
(never the real tests/): edit-set contents, each preflight refusal (nothing touched), a full apply, an
interrupted apply plus ``--resume``, and ``--abort-report``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts import apply_test_renames as A
from tests.unit.helpers_test_renames import git, make_repo, snapshot
from tests.unit.test_test_file_naming_guard import check_naming

P = A.P

# new names for the rows the proposer leaves as '?' (the owner's hand-written behaviours)
OWNER_NAMES = {
    "tests/unit/test_oct04_static_dominates.py": "tests/unit/test_engine_many_calls.py",
    "tests/unit/test_oct02_mixed.py": "tests/unit/test_engine_mixed_alpha.py",
    "tests/unit/test_sep11_B4.py": "tests/unit/test_engine_b4_behaviour.py",
    "tests/unit/test_sep27b_B5.py": "tests/unit/test_engine_b5_behaviour.py",
    "tests/unit/test_oct05_flush.py": "tests/unit/test_engine_flush_window_a.py",
    "tests/unit/test_sep12_flush.py": "tests/unit/test_engine_flush_window_b.py",
    "tests/unit/test_oct06_misc.py": "tests/unit/test_core_misc_checks.py",
}


def _draft_map(repo: Path, tmp_path: Path, overrides: dict[str, str] | None = None) -> Path:
    rows, meta = P.build_rows(repo)
    names = dict(OWNER_NAMES)
    names.update(overrides or {})
    for r in rows:
        if r["kind"] == "rename" and r["old_path"] in names:
            r["new_path"] = names[r["old_path"]]
    out = tmp_path / "draft.tsv"
    out.write_text(P.render_map(rows, meta, "draft"), encoding="utf-8")
    return out


def _sealed(repo: Path, tmp_path: Path, overrides: dict[str, str] | None = None) -> Path:
    draft = _draft_map(repo, tmp_path, overrides)
    sealed = tmp_path / "approved.tsv"
    assert A.do_seal(repo, draft, sealed, accept_changed=False) == 0
    return sealed


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path / "repo")


def _editset(repo, map_path, **kw):
    text = map_path.read_text(encoding="utf-8")
    header, rows, errs = P.parse_map(text)
    assert errs == []
    return header, rows, text, A.build_editset(repo, header, rows, map_text=text, **kw)


# ---------------------------------------------------------------------------
# edit-set contents
# ---------------------------------------------------------------------------


def test_dry_run_on_a_draft_is_labelled_and_touches_nothing(repo, tmp_path, capsys):
    draft = _draft_map(repo, tmp_path)
    before = snapshot(repo)
    out = tmp_path / "editset.diff"
    rc = A.main(["--root", str(repo), "--map", str(draft), "--out", str(out)])
    assert rc == 0
    assert snapshot(repo) == before
    text = out.read_text()
    assert text.startswith("# DRAFT")
    assert "rename from tests/unit/test_sep09_engine_widget.py\nrename to tests/unit/test_engine_widget.py" in text
    assert "DRAFT dry-run" in capsys.readouterr().out


def test_edit_set_rewrites_importers_pointers_and_prints_owner_docs(repo, tmp_path):
    _, _, _, es = _editset(repo, _sealed(repo, tmp_path))
    writes = {w["path"]: w["text"] for w in es["writes"]}
    # the importer's dotted import follows the rename
    new_importer = next(t for p, t in writes.items() if "user_of_helper" in p)
    assert "from tests.unit.test_engine_widget import run" in new_importer
    # BUG_CLASSES policy: pointers replaced, narrative keeps the old name plus a pointer
    bug = writes["docs/BUG_CLASSES.md"]
    assert "- Closure: the guard is pinned by `tests/unit/test_engine_widget.py`." in bug
    assert "test_sep09_engine_widget.py (now `test_engine_widget.py`) in the narrative" in bug
    assert "`test_pkg_b_alpha_thing.py`" in bug and "test_engine_b5_behaviour" in bug
    # historical doc untouched, exempt never mapped, CLAUDE.md only printed
    assert "docs/PLAN_20261001_history.md" not in writes and "CLAUDE.md" not in writes
    assert [p for p, _, _ in es["printed"]] == ["CLAUDE.md"]
    assert "test_engine_mixed_alpha" in es["printed"][0][2]
    # permanent resolver + allowlist removal + TEST_LANES prose
    assert writes["tests/RENAMES.tsv"].startswith("#status\tapproved")
    assert "frozenset()" in writes["tests/unit/test_test_file_naming_guard.py"]
    assert "test_sep09" not in writes["tests/unit/test_test_file_naming_guard.py"]
    assert "were renamed by module and behaviour" in writes["docs/TEST_LANES.md"]
    assert "are NOT renamed" not in writes["docs/TEST_LANES.md"]
    assert es["errors"] == [] and es["unresolved"] == []


def test_docstring_batch_label_line_is_added_without_the_old_file_name(repo, tmp_path):
    _, _, _, es = _editset(repo, _sealed(repo, tmp_path), migration_date="2026-10-16")
    widget = next(w for w in es["writes"] if w["path"] == "tests/unit/test_engine_widget.py")
    doc = ast.get_docstring(ast.parse(widget["text"]))
    assert "Original batch label: sep09 (renamed 2026-10-16" in doc
    assert "test_sep09_engine_widget" not in widget["text"]
    b5 = next(w for w in es["writes"] if w["path"] == "tests/unit/test_engine_b5_behaviour.py")
    assert "Original batch label: sep27b_B5" in ast.get_docstring(ast.parse(b5["text"]))


@pytest.mark.parametrize("src,ok", [
    ('"""One line."""\nx = 1\n', True),
    ('"""Multi\n\nline docstring.\n"""\nx = 1\n', True),
    ('"""Multi\nline."""\nx = 1\n', True),
    ("x = 1\n", True),
    ('from __future__ import annotations\n\nx = 1\n', True),
    ("'single quoted docstring'\nx = 1\n", False),
    ("", False),
])
def test_insert_docstring_line_shapes(src, ok):
    line = "Original batch label: sep09 (renamed 2026-10-16; old path in tests/RENAMES.tsv)."
    if ok:
        new = A.insert_docstring_line(src, line)
        assert line in ast.get_docstring(ast.parse(new))
        if "__future__" in src:
            assert new.index('"""') < new.index("from __future__")
    else:
        with pytest.raises(A.EditError):
            A.insert_docstring_line(src, line)


def test_line_locator_is_relocated_to_the_same_source_line(repo, tmp_path):
    _, _, _, es = _editset(repo, _sealed(repo, tmp_path))
    writes = {w["path"]: w["text"] for w in es["writes"]}
    old_line = next(i for i, ln in enumerate(
        (repo / "tests/unit/test_sep09_engine_widget.py").read_text().splitlines(), 1) if "assert run(1)" in ln)
    new_widget = writes["tests/unit/test_engine_widget.py"].splitlines()
    new_line = next(i for i, ln in enumerate(new_widget, 1) if "assert run(1)" in ln)
    assert new_line != old_line  # the docstring line shifted it
    assert f"tests/unit/test_engine_widget.py:{new_line}" in writes["core/engine.py"]


def test_locator_on_a_line_that_vanishes_is_a_hunk_error(repo, tmp_path):
    sealed = _sealed(repo, tmp_path)
    (repo / "core/engine.py").write_text(
        "def run(x):\n    return x\n\n\n# regression: tests/unit/test_sep09_engine_widget.py:999\n", encoding="utf-8")
    git(repo, "commit", "-qam", "bad locator")
    _, _, _, es = _editset(repo, sealed)
    assert any("locator :999" in e for e in es["errors"])


# ---------------------------------------------------------------------------
# preflight refusals: every one leaves the tree untouched
# ---------------------------------------------------------------------------


def _expect_refusal(repo, sealed, journal, capsys, needle, **kw):
    before = snapshot(repo)
    rc = A.do_apply(repo, sealed, journal, editset_file=None, allow_base_mismatch=kw.get("mismatch", False),
                    migration_date=A.MIGRATION_DATE)
    out = capsys.readouterr().out
    assert rc != 0 and needle in out, out
    assert snapshot(repo) == before
    assert not journal.exists()


def test_unapproved_map_cannot_be_applied(repo, tmp_path, capsys):
    _expect_refusal(repo, _draft_map(repo, tmp_path), tmp_path / "j.jsonl", capsys, "not 'approved'")


def test_tampered_approved_body_is_refused(repo, tmp_path, capsys):
    sealed = _sealed(repo, tmp_path)
    sealed.write_text(sealed.read_text().replace("test_engine_widget.py", "test_engine_widget2.py"))
    _expect_refusal(repo, sealed, tmp_path / "j.jsonl", capsys, "approved_body_sha256")


def test_old_file_missing_is_refused(repo, tmp_path, capsys):
    sealed = _sealed(repo, tmp_path)
    git(repo, "rm", "-q", "tests/unit/test_oct01_thing.py")
    git(repo, "commit", "-qm", "gone")
    _expect_refusal(repo, sealed, tmp_path / "j.jsonl", capsys, "old file missing", mismatch=True)


def test_hash_mismatch_is_refused(repo, tmp_path, capsys):
    sealed = _sealed(repo, tmp_path)
    p = repo / "tests/unit/test_oct01_thing.py"
    p.write_text(p.read_text() + "\n# edited after approval\n")
    git(repo, "commit", "-qam", "edit")
    _expect_refusal(repo, sealed, tmp_path / "j.jsonl", capsys, "content hash differs", mismatch=True)


def test_existing_target_is_refused(repo, tmp_path, capsys):
    sealed = _sealed(repo, tmp_path)
    (repo / "tests/unit/test_engine_widget.py").write_text("x = 1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "target exists")
    _expect_refusal(repo, sealed, tmp_path / "j.jsonl", capsys, "target already exists", mismatch=True)


def test_duplicate_new_path_and_chain_are_refused(repo, tmp_path, capsys):
    rows, meta = P.build_rows(repo)
    for r in rows:
        if r["kind"] == "rename" and r["new_path"] == "?":
            r["new_path"] = OWNER_NAMES.get(r["old_path"], "tests/unit/test_engine_x.py")
        if r["old_path"].endswith("test_oct01_thing.py"):
            r["new_path"] = "tests/unit/test_engine_widget.py"  # duplicates the widget's new path
        if r["old_path"].endswith("test_oct03_user_of_helper.py"):
            r["new_path"] = "tests/unit/test_oct01_thing.py"  # chain: target is another row's old path
    draft = tmp_path / "d.tsv"
    draft.write_text(P.render_map(rows, meta, "draft"))
    sealed = tmp_path / "a.tsv"
    assert A.do_seal(repo, draft, sealed, accept_changed=False) == 0
    before = snapshot(repo)
    rc = A.do_apply(repo, sealed, tmp_path / "j.jsonl", editset_file=None, allow_base_mismatch=False,
                    migration_date=A.MIGRATION_DATE)
    assert rc != 0 and snapshot(repo) == before
    header, prow, _ = P.parse_map(sealed.read_text())
    text = sealed.read_text()
    errs = A.preflight(repo, header, prow, A.build_editset(repo, header, prow, map_text=text),
                       map_text=text, require_approval=True, require_clean=True)
    assert any("duplicate new_path" in e for e in errs) and any("cycle/chain" in e for e in errs)


def test_unresolved_rows_block_seal_and_apply(repo, tmp_path, capsys):
    rows, meta = P.build_rows(repo)  # owner rows still '?'
    draft = tmp_path / "d.tsv"
    draft.write_text(P.render_map(rows, meta, "draft"))
    assert A.do_seal(repo, draft, tmp_path / "a.tsv", accept_changed=False) == 1
    assert "new_path '?'" in capsys.readouterr().out
    assert not (tmp_path / "a.tsv").exists()


def test_seal_refuses_files_changed_since_the_draft(repo, tmp_path, capsys):
    draft = _draft_map(repo, tmp_path)
    p = repo / "tests/unit/test_oct01_thing.py"
    p.write_text(p.read_text() + "\n# drift\n")
    assert A.do_seal(repo, draft, tmp_path / "a.tsv", accept_changed=False) == 1
    assert "changed since the draft" in capsys.readouterr().out
    assert A.do_seal(repo, draft, tmp_path / "a.tsv", accept_changed=True) == 0


def test_safety_rule_change_is_refused(repo, tmp_path, capsys):
    sealed = _sealed(repo, tmp_path, {"tests/unit/test_oct06_misc.py": "tests/unit/test_valence_misc_checks.py"})
    # the proposer's own name for that row newly falls under the safety-skip rule (old name did not)
    _expect_refusal(repo, sealed, tmp_path / "j.jsonl", capsys, "safety-skip rule")


def test_hunk_that_does_not_apply_touches_nothing(repo, tmp_path, capsys):
    sealed = _sealed(repo, tmp_path, {"tests/unit/test_oct06_misc.py": "tests/unit/test_core_misc_checks.py"})
    lanes = repo / "docs/TEST_LANES.md"
    lanes.write_text("# Lanes\n\nnothing about naming here\n")
    git(repo, "commit", "-qam", "drop paragraph")
    _expect_refusal(repo, sealed, tmp_path / "j.jsonl", capsys, "section 6 'NOT renamed' paragraph", mismatch=True)


def test_dirty_tree_is_refused_for_apply(repo, tmp_path, capsys):
    sealed = _sealed(repo, tmp_path, {"tests/unit/test_oct06_misc.py": "tests/unit/test_core_misc_checks.py"})
    (repo / "scratch.txt").write_text("dirty")
    _expect_refusal(repo, sealed, tmp_path / "j.jsonl", capsys, "not clean")


def test_base_sha_mismatch_is_refused_unless_allowed(repo, tmp_path, capsys):
    sealed = _sealed(repo, tmp_path, {"tests/unit/test_oct06_misc.py": "tests/unit/test_core_misc_checks.py"})
    (repo / "docs/extra.md").write_text("x")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "advance HEAD")
    _expect_refusal(repo, sealed, tmp_path / "j.jsonl", capsys, "base_sha")


def test_reviewed_edit_set_must_match_regeneration(repo, tmp_path, capsys):
    sealed = _sealed(repo, tmp_path, {"tests/unit/test_oct06_misc.py": "tests/unit/test_core_misc_checks.py"})
    es_file = tmp_path / "es.diff"
    A.main(["--root", str(repo), "--map", str(sealed), "--out", str(es_file)])
    es_file.write_text(es_file.read_text().replace("# editset_sha256: ", "# editset_sha256: 0"))
    before = snapshot(repo)
    rc = A.do_apply(repo, sealed, tmp_path / "j.jsonl", editset_file=es_file, allow_base_mismatch=False,
                    migration_date=A.MIGRATION_DATE)
    assert rc == 1 and "differs from the reviewed" in capsys.readouterr().out
    assert snapshot(repo) == before


# ---------------------------------------------------------------------------
# full apply
# ---------------------------------------------------------------------------


MISC = {"tests/unit/test_oct06_misc.py": "tests/unit/test_core_misc_checks.py"}


def _apply(repo, tmp_path, name="j.jsonl"):
    sealed = _sealed(repo, tmp_path, MISC)
    journal = tmp_path / name
    rc = A.do_apply(repo, sealed, journal, editset_file=None, allow_base_mismatch=False,
                    migration_date=A.MIGRATION_DATE)
    return sealed, journal, rc


def test_full_apply_moves_with_git_and_satisfies_the_guard(repo, tmp_path):
    sealed, journal, rc = _apply(repo, tmp_path)
    assert rc == 0
    status = git(repo, "status", "--porcelain")
    assert any(ln.startswith("R") and ln.endswith("tests/unit/test_sep09_engine_widget.py -> tests/unit/test_engine_widget.py")
               for ln in status.splitlines())
    assert not [p for p in (repo / "tests/unit").iterdir() if P.DATED_FILE_RE.match(p.name)
                and p.name != "test_sep10_calendar_forced_action_loop.py"]
    assert (repo / "tests/unit/test_sep10_calendar_forced_action_loop.py").is_file()
    for p in (repo / "tests/unit").glob("test_*.py"):
        ast.parse(p.read_text())  # every migrated file still parses
    # the permanent resolver is the approved map and the guard accepts the migrated tree
    assert (repo / "tests/RENAMES.tsv").read_text() == sealed.read_text()
    guard_src = (repo / "tests/unit/test_test_file_naming_guard.py").read_text()
    assert "PRE_MIGRATION_ALLOWLIST: frozenset[str] = frozenset()" in guard_src
    assert check_naming(repo, allowlist=frozenset()) == []  # the deployed guard accepts the migrated tree
    # zero-reference check: no migrated old stem in a CURRENT location except the map itself
    stems = {Path(o).stem for o, n in [r.split("\t")[1:3] for r in sealed.read_text().splitlines()
                                       if r.startswith("rename\t")]}
    leaks = []
    for rel, p in P._text_files(repo):
        if rel == "tests/RENAMES.tsv" or rel in A.PRINTED_ONLY or P.classify_location(rel) == "historical":
            continue
        for ln, line in enumerate(p.read_text().splitlines(), 1):
            leaks += [(rel, ln, s) for s in stems if s in line and "(now `" not in line]
    assert leaks == []
    assert [e["step"] for e in A._load_journal(journal)][-1] == "done"
    assert "test_oct02_mixed" in (repo / "CLAUDE.md").read_text()  # printed-only: never written


def test_apply_refuses_to_reuse_a_journal(repo, tmp_path, capsys):
    sealed, journal, rc = _apply(repo, tmp_path)
    assert rc == 0
    again = A.do_apply(repo, sealed, journal, editset_file=None, allow_base_mismatch=True,
                       migration_date=A.MIGRATION_DATE)
    assert again == 2 and "already exists" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# interrupted apply, resume, abort-report
# ---------------------------------------------------------------------------


class Crash(Exception):
    pass


def _work(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    return d


def _clean_reference(tmp_path):
    ref = make_repo(tmp_path / "ref")
    _, _, rc = _apply(ref, _work(tmp_path, "refwork"))
    assert rc == 0
    return snapshot(ref)


def test_interrupt_during_content_writes_then_resume_matches_a_clean_apply(repo, tmp_path, monkeypatch):
    reference = _clean_reference(tmp_path)
    sealed = _sealed(repo, _work(tmp_path, "w"), MISC)
    journal = tmp_path / "j.jsonl"
    real = A._write_text
    calls = {"n": 0}

    def flaky(path, text, mode_from=None):
        calls["n"] += 1
        if calls["n"] == 3:
            raise Crash
        return real(path, text, mode_from)

    monkeypatch.setattr(A, "_write_text", flaky)
    with pytest.raises(Crash):
        A.do_apply(repo, sealed, journal, editset_file=None, allow_base_mismatch=False, migration_date=A.MIGRATION_DATE)
    monkeypatch.setattr(A, "_write_text", real)
    assert A._load_journal(journal)[-1]["step"] != "done"
    assert A.do_resume(repo, journal) == 0
    assert snapshot(repo) == reference
    assert A.do_resume(repo, journal) == 0  # idempotent once complete


def test_interrupt_between_git_mv_and_journal_resumes(repo, tmp_path, monkeypatch):
    reference = _clean_reference(tmp_path)
    sealed = _sealed(repo, _work(tmp_path, "w"), MISC)
    journal = tmp_path / "j.jsonl"
    real = A._journal_append
    seen = {"renames": 0}

    def flaky(j, rec):
        if rec.get("step") == "rename":
            seen["renames"] += 1
            if seen["renames"] == 2:
                raise Crash  # the mv happened, its journal record did not
        return real(j, rec)

    monkeypatch.setattr(A, "_journal_append", flaky)
    with pytest.raises(Crash):
        A.do_apply(repo, sealed, journal, editset_file=None, allow_base_mismatch=False, migration_date=A.MIGRATION_DATE)
    monkeypatch.setattr(A, "_journal_append", real)
    assert A.do_resume(repo, journal) == 0
    assert snapshot(repo) == reference


def test_resume_refuses_when_the_tree_drifted(repo, tmp_path, monkeypatch, capsys):
    sealed = _sealed(repo, _work(tmp_path, "w"), MISC)
    journal = tmp_path / "j.jsonl"
    real = A._write_text
    calls = {"n": 0}

    def flaky(path, text, mode_from=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise Crash
        return real(path, text, mode_from)

    monkeypatch.setattr(A, "_write_text", flaky)
    with pytest.raises(Crash):
        A.do_apply(repo, sealed, journal, editset_file=None, allow_base_mismatch=False, migration_date=A.MIGRATION_DATE)
    monkeypatch.setattr(A, "_write_text", real)
    done = [e for e in A._load_journal(journal) if e["step"] == "write"][0]["path"]
    (repo / done).write_text("# someone edited this meanwhile\n")
    assert A.do_resume(repo, journal) == 1
    assert "RESUME REFUSED" in capsys.readouterr().out


def test_abort_report_lists_progress_and_deletes_nothing(repo, tmp_path, monkeypatch, capsys):
    sealed = _sealed(repo, _work(tmp_path, "w"), MISC)
    journal = tmp_path / "j.jsonl"
    real = A._write_text
    calls = {"n": 0}

    def flaky(path, text, mode_from=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise Crash
        return real(path, text, mode_from)

    monkeypatch.setattr(A, "_write_text", flaky)
    with pytest.raises(Crash):
        A.do_apply(repo, sealed, journal, editset_file=None, allow_base_mismatch=False, migration_date=A.MIGRATION_DATE)
    monkeypatch.setattr(A, "_write_text", real)
    before = snapshot(repo)
    capsys.readouterr()
    assert A.do_abort_report(repo, journal) == 0
    out = capsys.readouterr().out
    assert snapshot(repo) == before
    assert "moved:" in out and "git mv tests/unit/test_engine_widget.py tests/unit/test_sep09_engine_widget.py" in out
    assert "edited:" in out and "pending:" in out and "complete: no" in out
    assert "reset" not in out and "checkout" not in out and "rm " not in out


def test_abort_report_without_a_journal_says_so(tmp_path, capsys):
    assert A.do_abort_report(tmp_path, tmp_path / "none.jsonl") == 0
    assert "nothing was applied" in capsys.readouterr().out


def test_main_apply_refuses_while_a_live_daemon_runs(repo, tmp_path, monkeypatch, capsys):
    sealed = _sealed(repo, tmp_path, MISC)
    monkeypatch.setattr(A, "daemon_running", lambda root=None: True)
    before = snapshot(repo)
    rc = A.main(["--root", str(repo), "--map", str(sealed), "--apply", "--journal", str(tmp_path / "j.jsonl")])
    assert rc == 1 and "live Daemon" in capsys.readouterr().out
    assert snapshot(repo) == before


def test_main_apply_runs_when_no_daemon_is_up(repo, tmp_path, monkeypatch):
    sealed = _sealed(repo, tmp_path, MISC)
    monkeypatch.setattr(A, "daemon_running", lambda root=None: False)
    assert A.main(["--root", str(repo), "--map", str(sealed), "--apply", "--journal", str(tmp_path / "j.jsonl")]) == 0
    assert (repo / "tests/unit/test_engine_widget.py").is_file()
