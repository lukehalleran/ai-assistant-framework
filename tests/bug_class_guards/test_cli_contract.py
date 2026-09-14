"""Deployed-CLI contract: baseline, dispositions, partial scope and report shape.

Every control drives ``check_bug_classes.main`` — the entry point the hook and
CI call — over a synthetic tree from ``fixtures.build_repo`` and asserts the
exit code TOGETHER with the machine-readable failure code, so a control cannot
pass because an unrelated check happened to exit non-zero.

Baseline occurrences are a multiset of full anchors; the dispositions ledger
holds one reviewed record per active occurrence, bound to the source file's
SHA-256.  A missing baseline is never "no debt"; a changed source file needs a
re-review before its accepted debt stays green.
"""

from __future__ import annotations

import json

import pytest

import check_bug_classes

from fixtures import (
    ANCHOR,
    BASELINE_PATH,
    DISPOSITIONS_PATH,
    DM01_RED,
    DM18_GREEN_NARROW,
    DM18_ONE_OF_TWINS,
    DM18_RED,
    DM18_THIRD_HANDLER,
    DM18_TWIN_HANDLERS,
    admit,
    anchor_of,
    build_repo,
    build_tree,
    changelog,
    failure_codes,
    read_ledgers,
    record,
    run_scan,
    sha256_file,
    write_ledgers,
)


def _entry(**overrides):
    entry = {
        "scanner": "dm18_except_returns_empty",
        "path": "memory/store.py",
        "symbol": "Store.get_rows",
        "kind": "broad_except_returns_empty",
        "digest": "a" * 64,
        "excerpt": "return []",
    }
    entry.update(overrides)
    return entry


def _write(root, rel, text):
    (root / rel).write_text(text, encoding="utf-8")


def _admitted(tmp_path, capsys, extra):
    """A green tree whose gate findings are all reviewed accepted debt."""
    build_repo(tmp_path, extra)
    _, payload, _, _ = run_scan(capsys, tmp_path)
    admit(tmp_path, payload)
    code, payload, _, err = run_scan(capsys, tmp_path)
    assert code == 0, err
    return payload


# ---------------------------------------------------------------------------
# Baseline file
# ---------------------------------------------------------------------------


class TestBaselineFailsClosed:
    def test_missing_baseline_on_a_finding_free_tree_exits_two(self, tmp_path, capsys):
        build_repo(tmp_path)
        (tmp_path / BASELINE_PATH).unlink()
        code, payload, _, err = run_scan(capsys, tmp_path)
        assert code == 2
        assert "baseline_invalid" in failure_codes(payload)
        assert "bug-class scan: OK" not in err

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("{not json", id="malformed-json"),
            pytest.param("[]", id="top-level-list"),
            pytest.param(json.dumps({"schema": True, "anchor": ANCHOR, "entries": []}), id="schema-bool"),
            pytest.param(json.dumps({"schema": "2", "anchor": ANCHOR, "entries": []}), id="schema-string"),
            pytest.param(json.dumps({"schema": 1, "entries": []}), id="legacy-schema-one"),
            pytest.param(json.dumps({"schema": 3, "anchor": ANCHOR, "entries": []}), id="future-schema"),
            pytest.param(json.dumps({"schema": 2, "anchor": "bug-class-anchor/0", "entries": []}), id="anchor-version"),
            pytest.param(json.dumps({"schema": 2, "anchor": ANCHOR, "entries": {}}), id="entries-not-list"),
            pytest.param(json.dumps({"schema": 2, "anchor": ANCHOR}), id="entries-missing"),
            pytest.param(json.dumps({"schema": 2, "anchor": ANCHOR, "entries": [], "note": 1}), id="extra-top-level"),
        ],
    )
    def test_structurally_invalid_baseline_exits_two(self, tmp_path, capsys, text):
        build_repo(tmp_path)
        _write(tmp_path, BASELINE_PATH, text)
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 2
        assert "baseline_invalid" in failure_codes(payload)

    @pytest.mark.parametrize(
        "entry",
        [
            pytest.param("row", id="entry-not-object"),
            pytest.param(_entry(path=7), id="path-not-string"),
            pytest.param(_entry(symbol=""), id="empty-symbol"),
            pytest.param(_entry(path=""), id="empty-path"),
            pytest.param(_entry(path="/etc/passwd"), id="absolute-path"),
            pytest.param(_entry(path="../outside.py"), id="parent-path"),
            pytest.param({k: v for k, v in _entry().items() if k != "kind"}, id="missing-field"),
            pytest.param(_entry(line=12), id="extra-field"),
            pytest.param(_entry(scanner="dm99_unknown"), id="unknown-scanner"),
            pytest.param(
                _entry(scanner="dm16_config_key_reachability", kind="config_leaf_unreached"),
                id="report-only-scanner-row",
            ),
            pytest.param(_entry(kind="something_else"), id="undeclared-kind"),
            pytest.param(_entry(digest="abc"), id="short-digest"),
            pytest.param(_entry(digest="A" * 64), id="uppercase-digest"),
            pytest.param(_entry(excerpt=None), id="excerpt-not-string"),
        ],
    )
    def test_invalid_rows_exit_two(self, tmp_path, capsys, entry):
        build_repo(tmp_path)
        _write(tmp_path, BASELINE_PATH, json.dumps({"schema": 2, "anchor": ANCHOR, "entries": [entry]}))
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 2
        assert "baseline_invalid" in failure_codes(payload)

    def test_unsorted_rows_exit_two(self, tmp_path, capsys):
        build_repo(tmp_path)
        rows = [_entry(digest="b" * 64), _entry(digest="a" * 64)]
        _write(tmp_path, BASELINE_PATH, json.dumps({"schema": 2, "anchor": ANCHOR, "entries": rows}))
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 2
        assert "baseline_invalid" in failure_codes(payload)


# ---------------------------------------------------------------------------
# Dispositions ledger
# ---------------------------------------------------------------------------


def _ledger(**overrides):
    ledger = {"schema": 1, "anchor": ANCHOR, "legacy_baseline": None, "records": []}
    ledger.update(overrides)
    return ledger


class TestDispositionsFailClosed:
    def test_missing_dispositions_exits_two(self, tmp_path, capsys):
        build_repo(tmp_path)
        (tmp_path / DISPOSITIONS_PATH).unlink()
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 2
        assert "dispositions_invalid" in failure_codes(payload)

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("{", id="malformed-json"),
            pytest.param(json.dumps(_ledger(schema=True)), id="schema-bool"),
            pytest.param(json.dumps(_ledger(records={})), id="records-not-list"),
            pytest.param(json.dumps(_ledger(extra=1)), id="extra-top-level"),
            pytest.param(json.dumps(_ledger(legacy_baseline=[])), id="legacy-wrong-type"),
        ],
    )
    def test_structurally_invalid_ledger_exits_two(self, tmp_path, capsys, text):
        build_repo(tmp_path)
        _write(tmp_path, DISPOSITIONS_PATH, text)
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 2
        assert "dispositions_invalid" in failure_codes(payload)

    @pytest.mark.parametrize(
        "change",
        [
            pytest.param({"ordinal": True}, id="ordinal-bool"),
            pytest.param({"ordinal": 0}, id="ordinal-zero"),
            pytest.param({"status": "unreviewed"}, id="unknown-status"),
            pytest.param({"reviewer": ""}, id="empty-reviewer"),
            pytest.param({"rationale": ""}, id="empty-rationale"),
            pytest.param({"review_date": "13/09/2026"}, id="bad-date"),
            pytest.param({"source_sha256": "xyz"}, id="bad-source-hash"),
            pytest.param({"assessment": "fine"}, id="unknown-assessment"),
            pytest.param({"note": "x"}, id="extra-field"),
            pytest.param({"status": "confirmed_fixed", "request": None}, id="fixed-without-request"),
        ],
    )
    def test_invalid_record_exits_two(self, tmp_path, capsys, change):
        build_repo(tmp_path, {"memory/store.py": DM18_RED})
        _, payload, _, _ = run_scan(capsys, tmp_path)
        admit(tmp_path, payload)
        baseline, ledger = read_ledgers(tmp_path)
        ledger["records"][0].update(change)
        write_ledgers(tmp_path, baseline["entries"], ledger["records"])
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 2
        assert "dispositions_invalid" in failure_codes(payload)

    def test_duplicate_anchor_ordinal_exits_two(self, tmp_path, capsys):
        build_repo(tmp_path, {"memory/store.py": DM18_RED})
        _, payload, _, _ = run_scan(capsys, tmp_path)
        admit(tmp_path, payload)
        baseline, ledger = read_ledgers(tmp_path)
        write_ledgers(tmp_path, baseline["entries"], ledger["records"] * 2)
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 2
        assert "dispositions_invalid" in failure_codes(payload)


# ---------------------------------------------------------------------------
# Ratchet and reconciliation
# ---------------------------------------------------------------------------


class TestRatchetAndReconciliation:
    def test_new_finding_fails_with_its_anchor(self, tmp_path, capsys):
        build_repo(tmp_path, {"memory/store.py": DM18_RED})
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 1
        assert "new_findings" in failure_codes(payload)
        (row,) = payload["comparison"]["new"]
        assert (row["scanner"], row["path"], row["symbol"], row["kind"]) == (
            "dm18_except_returns_empty", "memory/store.py", "Store.get_rows", "broad_except_returns_empty",
        )

    def test_reviewed_accepted_debt_is_green(self, tmp_path, capsys):
        payload = _admitted(tmp_path, capsys, {"memory/store.py": DM18_RED})
        assert payload["baseline"]["occurrences"] == 1
        assert payload["dispositions"]["active"] == 1
        assert payload["comparison"]["matched"] == 1

    def test_baseline_row_without_a_disposition_fails(self, tmp_path, capsys):
        _admitted(tmp_path, capsys, {"memory/store.py": DM18_RED})
        baseline, _ = read_ledgers(tmp_path)
        write_ledgers(tmp_path, baseline["entries"], [])
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 1
        assert "disposition_missing" in failure_codes(payload)

    def test_accepted_record_without_a_baseline_row_is_an_orphan(self, tmp_path, capsys):
        _admitted(tmp_path, capsys, {"memory/store.py": DM18_RED})
        baseline, ledger = read_ledgers(tmp_path)
        ghost = dict(ledger["records"][0]["anchor"], digest="b" * 64)
        write_ledgers(tmp_path, baseline["entries"], ledger["records"] + [record(tmp_path, ghost, 1)])
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 1
        assert "disposition_orphan" in failure_codes(payload)

    def test_changed_source_needs_re_review_even_when_the_anchor_holds(self, tmp_path, capsys):
        _admitted(tmp_path, capsys, {"memory/store.py": DM18_RED})
        target = tmp_path / "memory/store.py"
        target.write_text("# unrelated edit above\n" + DM18_RED, encoding="utf-8")
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 1
        assert payload["comparison"]["new"] == [] and payload["comparison"]["stale"] == []
        assert "disposition_source_changed" in failure_codes(payload)

    def test_line_movement_keeps_the_baseline_and_re_review_restores_green(self, tmp_path, capsys):
        _admitted(tmp_path, capsys, {"memory/store.py": DM18_RED})
        before = (tmp_path / BASELINE_PATH).read_bytes()
        target = tmp_path / "memory/store.py"
        target.write_text(DM18_RED.replace("import logging\n", "import logging\nimport os\n\n\n"), encoding="utf-8")
        baseline, ledger = read_ledgers(tmp_path)
        for item in ledger["records"]:
            item["source_sha256"] = sha256_file(target)  # the re-review
        write_ledgers(tmp_path, baseline["entries"], ledger["records"])
        code, payload, _, err = run_scan(capsys, tmp_path)
        assert code == 0, err
        assert (tmp_path / BASELINE_PATH).read_bytes() == before

    def test_fixed_debt_left_in_the_baseline_is_stale(self, tmp_path, capsys):
        _admitted(tmp_path, capsys, {"memory/store.py": DM18_RED})
        (tmp_path / "memory/store.py").write_text(DM18_GREEN_NARROW, encoding="utf-8")
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 1
        assert "stale_baseline" in failure_codes(payload)
        assert len(payload["comparison"]["stale"]) == 1

    def test_confirmed_fixed_record_cannot_launder_a_leftover_row(self, tmp_path, capsys):
        _admitted(tmp_path, capsys, {"memory/store.py": DM18_RED})
        (tmp_path / "memory/store.py").write_text(DM18_GREEN_NARROW, encoding="utf-8")
        baseline, ledger = read_ledgers(tmp_path)
        ledger["records"][0].update(status="confirmed_fixed", request="CGR-20260913-999")
        write_ledgers(tmp_path, baseline["entries"], ledger["records"])
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 1
        assert {"stale_baseline", "disposition_missing"} <= failure_codes(payload)

    def test_resolved_history_without_an_active_row_is_green(self, tmp_path, capsys):
        _admitted(tmp_path, capsys, {"memory/store.py": DM18_RED})
        (tmp_path / "memory/store.py").write_text(DM18_GREEN_NARROW, encoding="utf-8")
        _, ledger = read_ledgers(tmp_path)
        ledger["records"][0].update(status="confirmed_fixed", request="CGR-20260913-999")
        write_ledgers(tmp_path, [], ledger["records"])
        code, payload, _, err = run_scan(capsys, tmp_path)
        assert code == 0, err
        assert payload["dispositions"]["active"] == 0
        assert payload["dispositions"]["confirmed_fixed"] == 1


class TestDuplicateMultiplicity:
    def test_identical_occurrences_are_two_rows_and_two_records(self, tmp_path, capsys):
        payload = _admitted(tmp_path, capsys, {"memory/store.py": DM18_TWIN_HANDLERS})
        baseline, ledger = read_ledgers(tmp_path)
        assert len(baseline["entries"]) == 2 and len({json.dumps(e, sort_keys=True) for e in baseline["entries"]}) == 1
        assert sorted(r["ordinal"] for r in ledger["records"]) == [1, 2]
        assert payload["comparison"]["matched"] == 2

    def test_a_third_identical_occurrence_is_exactly_one_new(self, tmp_path, capsys):
        _admitted(tmp_path, capsys, {"memory/store.py": DM18_TWIN_HANDLERS})
        (tmp_path / "memory/store.py").write_text(DM18_THIRD_HANDLER, encoding="utf-8")
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 1
        assert len(payload["comparison"]["new"]) == 1
        assert payload["comparison"]["stale"] == []

    def test_one_fewer_identical_occurrence_is_exactly_one_stale(self, tmp_path, capsys):
        _admitted(tmp_path, capsys, {"memory/store.py": DM18_TWIN_HANDLERS})
        (tmp_path / "memory/store.py").write_text(DM18_ONE_OF_TWINS, encoding="utf-8")
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 1
        assert payload["comparison"]["new"] == []
        assert len(payload["comparison"]["stale"]) == 1


# ---------------------------------------------------------------------------
# Explicit partial scope
# ---------------------------------------------------------------------------


class TestPartialScan:
    def test_partial_scan_is_labelled_and_compares_only_the_selection(self, tmp_path, capsys):
        _admitted(tmp_path, capsys, {"memory/store.py": DM18_RED})
        (tmp_path / "memory/store.py").write_text(DM18_GREEN_NARROW, encoding="utf-8")
        code, payload, _, err = run_scan(capsys, tmp_path, "--scanner", "dm01_raw_substring")
        assert code == 0, err
        assert payload["scope"] == "partial"
        assert payload["selected_scanners"] == ["dm01_raw_substring"]
        assert payload["comparison"]["compared_scanners"] == ["dm01_raw_substring"]
        assert payload["scanners"]["dm18_except_returns_empty"]["status"] == "not_selected"
        assert all(leg["status"] == "not_selected" for leg in payload["scanners"]["dm18_except_returns_empty"]["legs"])
        assert payload["inventory"]["evaluated"] is False
        assert "scope: partial" in err
        assert "bug-class scan: OK" not in err

        code, payload, _, _ = run_scan(capsys, tmp_path)  # the full scan still sees it
        assert code == 1 and "stale_baseline" in failure_codes(payload)

    def test_unknown_scanner_selection_exits_two(self, tmp_path, capsys):
        build_repo(tmp_path)
        code, payload, _, _ = run_scan(capsys, tmp_path, "--scanner", "dm99_nope")
        assert code == 2
        assert "usage" in failure_codes(payload)

    def test_write_baseline_refuses_a_partial_scope(self, tmp_path, capsys):
        build_repo(tmp_path, {"memory/store.py": DM18_RED})
        before = (tmp_path / BASELINE_PATH).read_bytes()
        code, payload, _, _ = run_scan(capsys, tmp_path, "--scanner", "dm18_except_returns_empty", "--write-baseline")
        assert code == 2
        assert "usage" in failure_codes(payload)
        assert (tmp_path / BASELINE_PATH).read_bytes() == before

    def test_partial_scan_still_validates_the_policy(self, tmp_path, capsys):
        build_repo(tmp_path)
        _write(tmp_path, "config/bug_class_policy.json", "{")
        code, payload, _, _ = run_scan(capsys, tmp_path, "--scanner", "catalog")
        assert code == 2 and "policy_invalid" in failure_codes(payload)

    def test_partial_scan_still_validates_baseline_syntax(self, tmp_path, capsys):
        build_repo(tmp_path)
        _write(tmp_path, BASELINE_PATH, json.dumps({"schema": 2, "anchor": ANCHOR, "entries": [_entry(scanner="dm99")]}))
        code, payload, _, _ = run_scan(capsys, tmp_path, "--scanner", "catalog")
        assert code == 2 and "baseline_invalid" in failure_codes(payload)

    def test_partial_scan_still_checks_selected_inputs(self, tmp_path, capsys):
        build_repo(tmp_path, drop=["docs/BUG_CLASSES.md"])
        code, payload, _, _ = run_scan(capsys, tmp_path, "--scanner", "catalog")
        assert code == 1 and "leg_missing" in failure_codes(payload)


class TestWriteBaselineIsBootstrapOnly:
    def test_written_baseline_still_needs_reviewed_dispositions(self, tmp_path, capsys):
        build_repo(tmp_path, {"memory/store.py": DM18_RED})
        code, payload, _, _ = run_scan(capsys, tmp_path, "--write-baseline")
        assert code == 1
        assert "disposition_missing" in failure_codes(payload)
        baseline, _ = read_ledgers(tmp_path)
        assert baseline["schema"] == 2 and len(baseline["entries"]) == 1


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------


class TestReport:
    def test_full_report_carries_the_scoped_evidence(self, tmp_path, capsys):
        build_repo(tmp_path)
        code, payload, _, err = run_scan(capsys, tmp_path)
        assert code == 0, err
        assert payload["report_schema"] == 1
        assert payload["scope"] == "full" and payload["exit"] == 0 and payload["failures"] == []
        assert payload["policy"]["version"] and len(payload["policy"]["sha256"]) == 64
        assert "not behavioral coverage" in payload["claim"].lower()
        coverage = payload["coverage"]
        assert coverage["catalog_classes"] == 3
        assert coverage["uncovered_classes"] == ["BC-03"] and coverage["uncovered_count"] == 1
        assert "BC-01" in coverage["gate_classes"]
        assert {"BC-10", "BC-76"} <= set(coverage["report_only_classes"])
        assert payload["baseline"]["occurrences"] == 0
        assert payload["dispositions"]["active"] == 0
        assert payload["inventory"]["evaluated"] is True

    def test_json_mode_keeps_stdout_pure_and_the_table_on_stderr(self, tmp_path, capsys):
        build_repo(tmp_path)
        code, payload, out, err = run_scan(capsys, tmp_path)
        assert code == 0
        assert payload is not None and out.lstrip().startswith("{")
        assert "scope: full" in err

    def test_receipt_file_matches_the_json_report(self, tmp_path, capsys):
        repo = tmp_path / "repo"
        build_repo(repo)
        receipt = tmp_path / "scan.json"
        code, payload, _, err = run_scan(capsys, repo, "--receipt", str(receipt))
        assert code == 0, err
        assert json.loads(receipt.read_text(encoding="utf-8")) == payload

    def test_receipt_is_written_even_when_the_scan_fails(self, tmp_path, capsys):
        repo = tmp_path / "repo"
        build_repo(repo, {"core/tone.py": DM01_RED})
        receipt = tmp_path / "scan.json"
        code = check_bug_classes.main(["scan", "--root", str(repo), "--receipt", str(receipt)])
        capsys.readouterr()
        assert code == 1
        assert json.loads(receipt.read_text(encoding="utf-8"))["exit"] == 1

    def test_report_only_findings_never_reach_the_baseline_or_exit(self, tmp_path, capsys):
        widening = "`_INFO_SEEKING_CUES` gained one more phrase."
        text = changelog([("2026-08-15", widening), ("2026-09-07", widening), ("2026-09-10", widening)])
        build_repo(tmp_path, {"CLAUDE_CHANGELOG.md": text})
        code, payload, _, err = run_scan(capsys, tmp_path, "--write-baseline")
        assert code == 0, err
        assert payload["scanners"]["dm29_phrase_append_signature"]["findings"]
        baseline, _ = read_ledgers(tmp_path)
        assert baseline["entries"] == []

    def test_findings_carry_anchor_fields_and_a_bounded_excerpt(self, tmp_path, capsys):
        build_tree(tmp_path, {})
        build_repo(tmp_path, {"core/tone.py": DM01_RED})
        _, payload, _, _ = run_scan(capsys, tmp_path)
        findings = payload["scanners"]["dm01_raw_substring"]["findings"]
        assert len(findings) == 2
        for finding in findings:
            assert set(anchor_of("dm01_raw_substring", finding)) == {"scanner", "path", "symbol", "kind", "digest"}
            assert len(finding["excerpt"]) <= 240 and isinstance(finding["line"], int)
