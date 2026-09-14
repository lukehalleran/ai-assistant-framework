"""Input coverage: every declared leg and root is counted on its own.

A nonzero total proves nothing about a root that was never walked.  The
2026-09-13 review found DM-17's two inputs sharing one file count and the
Python walk silently skipping absent roots.  Each control removes or breaks
exactly one input of the minimal green tree (``fixtures.build_repo``) and
requires the deployed CLI to fail closed and name that input.  Private roots
are never traversed: a syntax error planted there must go unnoticed.
"""

from __future__ import annotations

import json

import pytest

import check_bug_classes

from fixtures import (
    BENIGN,
    COMMON_SOURCE_ROOTS,
    DM01_RED,
    DM16_CONSUMER_ROOTS,
    DM18_ROOTS,
    POLICY_PATH,
    REPO_ROOT,
    build_repo,
    changelog,
    failure_codes,
    leg_receipt,
    root_status,
    run_scan,
)

MISSING_OR_EMPTY = {"leg_missing", "leg_empty"}
BROKEN = "def broken(:\n    return 1\n"


def _excluded_root_names() -> list[str]:
    policy = json.loads((REPO_ROOT / POLICY_PATH).read_text(encoding="utf-8"))
    return [item["name"] for item in policy["inventory"]["excluded_roots"]]


def test_minimal_repo_is_green_with_every_required_leg_available(tmp_path, capsys):
    build_repo(tmp_path)
    code, payload, _, err = run_scan(capsys, tmp_path)
    assert code == 0, err
    for scanner in payload["scanners"].values():
        for leg in scanner["legs"]:
            assert {"id", "required", "available", "files_processed", "unresolved", "status", "roots"} <= set(leg)
            if leg["required"]:
                assert leg["available"] is True, leg
                assert leg["files_processed"] > 0, leg


class TestCommonSourceRoots:
    @pytest.mark.parametrize("root_name", COMMON_SOURCE_ROOTS)
    def test_each_root_is_required_on_its_own(self, tmp_path, capsys, root_name):
        build_repo(tmp_path, drop=[root_name])
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 1
        assert failure_codes(payload) & MISSING_OR_EMPTY
        for scanner in ("dm01_raw_substring", "dm31_live_state_default"):
            leg = leg_receipt(payload, scanner, "python_source")
            assert leg["available"] is False
            assert root_status(leg, root_name) in {"missing", "empty"}
            others = [r for r in leg["roots"] if r["path"] != root_name]
            assert all(r["status"] == "available" for r in others), others

    def test_removing_the_whole_common_source_set_fails_every_reader(self, tmp_path, capsys):
        build_repo(tmp_path, drop=[r for r in COMMON_SOURCE_ROOTS if r != "config"])
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 1
        for scanner, leg_id in (
            ("dm01_raw_substring", "python_source"),
            ("dm31_live_state_default", "python_source"),
            ("dm18_except_returns_empty", "dm18_retrieval"),
            ("dm16_config_key_reachability", "dm16_consumers"),  # gitleaks:allow
            ("dm17_apply_without_guard", "dm17_scripts"),
        ):
            assert leg_receipt(payload, scanner, leg_id)["available"] is False, (scanner, leg_id)


class TestDm18Roots:
    @pytest.mark.parametrize("root_name", DM18_ROOTS)
    def test_each_retrieval_root_is_required_on_its_own(self, tmp_path, capsys, root_name):
        build_repo(tmp_path, drop=[root_name])
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 1
        leg = leg_receipt(payload, "dm18_except_returns_empty", "dm18_retrieval")
        assert leg["available"] is False
        assert root_status(leg, root_name) in {"missing", "empty"}

    def test_core_prompt_is_checked_even_though_core_has_files(self, tmp_path, capsys):
        build_repo(tmp_path, drop=["core/prompt"])
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 1
        assert leg_receipt(payload, "dm01_raw_substring", "python_source")["available"] is True
        leg = leg_receipt(payload, "dm18_except_returns_empty", "dm18_retrieval")
        assert root_status(leg, "core/prompt") == "missing"


class TestDm17Legs:
    def test_empty_scripts_leg_fails_even_though_the_tests_leg_has_files(self, tmp_path, capsys):
        build_repo(tmp_path, {"scripts/nested/tool.py": BENIGN}, drop=["scripts/tool.py"])
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 1
        assert "leg_empty" in failure_codes(payload)
        scripts = leg_receipt(payload, "dm17_apply_without_guard", "dm17_scripts")
        tests = leg_receipt(payload, "dm17_apply_without_guard", "dm17_tests")
        assert scripts["status"] == "empty" and scripts["files_processed"] == 0
        assert tests["available"] is True and tests["files_processed"] > 0
        # The old single total would have been nonzero here.
        assert payload["scanners"]["dm17_apply_without_guard"]["files_processed"] > 0
        assert leg_receipt(payload, "dm01_raw_substring", "python_source")["available"] is True

    def test_missing_tests_leg_fails_even_though_the_scripts_leg_has_files(self, tmp_path, capsys):
        build_repo(tmp_path, drop=["tests"])
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 1
        assert "leg_missing" in failure_codes(payload)
        assert leg_receipt(payload, "dm17_apply_without_guard", "dm17_tests")["status"] == "missing"
        assert leg_receipt(payload, "dm17_apply_without_guard", "dm17_scripts")["available"] is True

    def test_tests_root_without_python_is_an_empty_leg(self, tmp_path, capsys):
        build_repo(tmp_path, {"tests/README.md": "notes\n"}, drop=["tests/__init__.py"])
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 1
        assert leg_receipt(payload, "dm17_apply_without_guard", "dm17_tests")["status"] == "empty"


class TestDm16Inputs:
    @pytest.mark.parametrize(
        "input_path, leg_id",
        [("config/config.yaml", "dm16_config_yaml"), ("config/app_config.py", "dm16_app_config")],
    )
    def test_required_config_inputs_fail_closed(self, tmp_path, capsys, input_path, leg_id):
        build_repo(tmp_path, drop=[input_path])
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 1  # report-only scanner, yet a missing REQUIRED input fails
        assert "leg_missing" in failure_codes(payload)
        assert leg_receipt(payload, "dm16_config_key_reachability", leg_id)["status"] == "missing"

    @pytest.mark.parametrize("root_name", DM16_CONSUMER_ROOTS)
    def test_each_consumer_root_is_required(self, tmp_path, capsys, root_name):
        build_repo(tmp_path, drop=[root_name])
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 1
        leg = leg_receipt(payload, "dm16_config_key_reachability", "dm16_consumers")  # gitleaks:allow
        assert root_status(leg, root_name) in {"missing", "empty"}

    def test_schema_is_optional_report_evidence(self, tmp_path, capsys):
        build_repo(tmp_path)
        code, payload, _, err = run_scan(capsys, tmp_path)
        assert code == 0, err
        leg = leg_receipt(payload, "dm16_config_key_reachability", "dm16_schema")
        assert (leg["required"], leg["available"], leg["status"]) == (False, False, "unavailable")

        build_repo(tmp_path, {"config/schema.py": BENIGN})
        code, payload, _, err = run_scan(capsys, tmp_path)
        assert code == 0, err
        assert leg_receipt(payload, "dm16_config_key_reachability", "dm16_schema")["available"] is True


def test_missing_catalog_fails_closed_and_is_not_a_clean_scan(tmp_path, capsys):
    build_repo(tmp_path, drop=["docs/BUG_CLASSES.md"])
    code, payload, _, err = run_scan(capsys, tmp_path)
    assert code == 1
    assert "leg_missing" in failure_codes(payload)
    assert leg_receipt(payload, "catalog", "catalog_document")["status"] == "missing"
    assert payload["scanners"]["catalog"]["findings"] == []
    assert "bug-class scan: OK" not in err


class TestReportOnlyAvailability:
    def test_absent_changelog_is_unavailable_never_a_clean_zero(self, tmp_path, capsys):
        build_repo(tmp_path)
        code, payload, _, err = run_scan(capsys, tmp_path)
        assert code == 0, err
        dm29 = payload["scanners"]["dm29_phrase_append_signature"]
        assert dm29["status"] == "unavailable"
        (leg,) = dm29["legs"]
        assert (leg["required"], leg["available"], leg["status"]) == (False, False, "unavailable")
        assert "BC-76" in payload["coverage"]["unavailable_report_classes"]

    def test_present_changelog_without_the_signature_is_available_with_zero(self, tmp_path, capsys):
        text = changelog([("2026-09-10", "nothing widened here.")])
        build_repo(tmp_path, {"CLAUDE_CHANGELOG.md": text})
        code, payload, _, err = run_scan(capsys, tmp_path)
        assert code == 0, err
        dm29 = payload["scanners"]["dm29_phrase_append_signature"]
        assert dm29["status"] == "ok" and dm29["findings"] == []
        (leg,) = dm29["legs"]
        assert leg["available"] is True and leg["files_processed"] == 1
        assert "BC-76" not in payload["coverage"]["unavailable_report_classes"]

    def test_human_table_says_unavailable_rather_than_zero(self, tmp_path, capsys):
        build_repo(tmp_path)
        check_bug_classes.main(["scan", "--root", str(tmp_path)])
        out = capsys.readouterr().out
        row = next(line for line in out.splitlines() if line.startswith("dm29_phrase_append_signature"))
        assert "unavailable" in row


class TestInventory:
    def test_new_top_level_python_package_must_be_classified(self, tmp_path, capsys):
        build_repo(tmp_path, {"newpkg/__init__.py": BENIGN})
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 1
        assert "inventory_unclassified" in failure_codes(payload)
        assert payload["inventory"]["unclassified_python"] == ["newpkg"]

    def test_new_top_level_python_file_must_be_classified(self, tmp_path, capsys):
        build_repo(tmp_path, {"loose_script.py": BENIGN})
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 1
        assert payload["inventory"]["unclassified_python"] == ["loose_script.py"]

    def test_python_in_a_known_non_python_root_must_be_classified(self, tmp_path, capsys):
        build_repo(tmp_path, {"docs/helper.py": BENIGN})
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 1
        assert payload["inventory"]["unclassified_python"] == ["docs"]

    @pytest.mark.parametrize("private_root", _excluded_root_names())
    def test_every_excluded_private_root_is_never_traversed(self, tmp_path, capsys, private_root):
        build_repo(
            tmp_path,
            {f"{private_root}/broken.py": BROKEN, f"{private_root}/nested/deeper/tone.py": DM01_RED},
        )
        code, payload, _, err = run_scan(capsys, tmp_path)
        assert code == 0, err
        assert private_root in payload["inventory"]["excluded_present"]
        assert payload["syntax"]["errors"] == []
        assert not any(
            finding["path"].startswith(f"{private_root}/")
            for scanner in payload["scanners"].values()
            for finding in scanner["findings"]
        )

    def test_backup_pattern_root_is_never_traversed(self, tmp_path, capsys):
        build_repo(tmp_path, {".handoff-backup-20260910/broken.py": BROKEN})
        code, payload, _, err = run_scan(capsys, tmp_path)
        assert code == 0, err
        assert ".handoff-backup-20260910" in payload["inventory"]["excluded_present"]

    def test_unknown_hidden_root_is_reported_but_not_inspected(self, tmp_path, capsys):
        build_repo(tmp_path, {".scratch/tool.py": BROKEN})
        code, payload, _, err = run_scan(capsys, tmp_path)
        assert code == 0, err
        assert ".scratch" in payload["inventory"]["not_inspected"]

    def test_classified_unscanned_python_root_is_reported_as_a_scope_gap(self, tmp_path, capsys):
        build_repo(tmp_path, {"eval/harness.py": BENIGN})
        code, payload, _, err = run_scan(capsys, tmp_path)
        assert code == 0, err
        assert "eval" in payload["inventory"]["unscanned_python_present"]

    def test_nested_new_directory_under_a_declared_root_is_scanned(self, tmp_path, capsys):
        build_repo(tmp_path, {"core/newpkg/deep/tone.py": DM01_RED})
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 1
        assert "new_findings" in failure_codes(payload)
        assert {row["path"] for row in payload["comparison"]["new"]} == {"core/newpkg/deep/tone.py"}


class TestSyntaxPreflight:
    def test_syntax_error_that_no_detector_shape_matches_exits_two(self, tmp_path, capsys):
        build_repo(tmp_path, {"utils/broken.py": BROKEN})
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 2
        assert "syntax_error" in failure_codes(payload)
        assert [error["path"] for error in payload["syntax"]["errors"]] == ["utils/broken.py"]
        assert payload["comparison"]["compared_scanners"] == []

    @pytest.mark.parametrize("rel", ["tests/unit/test_broken.py", "core/prompt/broken.py", "scripts/broken.py"])
    def test_syntax_error_in_any_declared_leg_exits_two(self, tmp_path, capsys, rel):
        build_repo(tmp_path, {rel: "x = (\n"})
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 2
        assert rel in [error["path"] for error in payload["syntax"]["errors"]]

    def test_every_syntax_error_is_listed(self, tmp_path, capsys):
        build_repo(tmp_path, {"utils/broken.py": BROKEN, "memory/broken.py": BROKEN})
        code, payload, _, _ = run_scan(capsys, tmp_path)
        assert code == 2
        assert sorted(error["path"] for error in payload["syntax"]["errors"]) == [
            "memory/broken.py", "utils/broken.py",
        ]
