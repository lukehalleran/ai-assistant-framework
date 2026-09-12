"""Red/green controls for every repo-static bug-class scanner.

Each control drives the deployed ``scan()`` — the same entry point the gate
calls — over a synthetic tree in ``tmp_path``.  A control that exercised a
private helper instead would pin the test's idea of the scanner rather than
the scanner (BC-63), and a fixture typed to the consumer's assumption rather
than the producer's output is BC-64 itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bug_class_guards.scanners import SCANNERS
from bug_class_guards.scanners import catalog_scanner, dm01_raw_substring
from bug_class_guards.scanners import dm16_config_reachability, dm17_apply_without_guard
from bug_class_guards.scanners import dm18_except_returns_empty, dm29_phrase_append
from bug_class_guards.scanners.common import ScannerError

REPO_ROOT = Path(__file__).resolve().parents[2]

from fixtures import (
    DM01_GREEN_CHOKEPOINT,
    DM01_GREEN_ORDINARY,
    DM01_RED,
    DM16_APP_CONFIG,
    DM16_CONSUMER,
    DM16_CONSUMER_ALL,
    DM16_YAML,
    DM17_GREEN_SCRIPT,
    DM17_GREEN_TEST,
    DM17_RED_SCRIPT,
    DM17_RED_TEST,
    DM18_GREEN_NARROW,
    DM18_GREEN_NO_STORE,
    DM18_GREEN_RERAISE,
    DM18_RED,
    DM29_WIDENING_LINE,
    build_tree,
    catalog_doc,
    changelog,
)



class TestRegistry:
    def test_every_scanner_declares_classes_and_a_known_mode(self):
        assert SCANNERS
        for scanner_id, scanner in SCANNERS.items():
            assert scanner.id == scanner_id
            assert scanner.class_ids and all(
                cid.startswith("BC-") for cid in scanner.class_ids
            )
            assert scanner.mode in {"gate", "report"}
            assert scanner.description


class TestDm01RawSubstring:
    def test_red_flags_both_incident_shapes(self, tmp_path):
        build_tree(tmp_path, {"core/tone.py": DM01_RED})
        result = dm01_raw_substring.scan(tmp_path)
        assert result.files_processed == 1
        texts = [f.text for f in result.findings]
        assert texts == ['if "ice" in text_lower:',
                         "return any(word in text_lower for word in HEAVY_KEYWORDS)"]
        assert all(f.symbol == "is_heavy" for f in result.findings)
        assert all(f.path == "core/tone.py" for f in result.findings)

    def test_green_chokepoint_importer_is_never_flagged(self, tmp_path):
        build_tree(tmp_path, {"core/tone.py": DM01_GREEN_CHOKEPOINT})
        result = dm01_raw_substring.scan(tmp_path)
        assert result.files_processed == 1
        assert result.findings == []

    def test_green_ordinary_membership_is_never_flagged(self, tmp_path):
        build_tree(tmp_path, {"core/tone.py": DM01_GREEN_ORDINARY})
        assert dm01_raw_substring.scan(tmp_path).findings == []

    def test_chokepoint_mentioned_only_in_a_comment_is_not_adoption(self, tmp_path):
        commented = "# TODO: route this through utils.trigger_match one day\n" + DM01_RED
        build_tree(tmp_path, {"core/tone.py": commented})
        assert len(dm01_raw_substring.scan(tmp_path).findings) == 2

    def test_excluded_directories_are_never_walked(self, tmp_path):
        build_tree(
            tmp_path,
            {
                "core/data/tone.py": DM01_RED,
                "core/__pycache__/tone.py": DM01_RED,
                "core/keep.py": DM01_GREEN_ORDINARY,
            },
        )
        result = dm01_raw_substring.scan(tmp_path)
        assert result.files_processed == 1
        assert result.findings == []


class TestDm16ConfigReachability:
    def test_red_flags_leaves_with_no_reader_outside_config(self, tmp_path):
        build_tree(
            tmp_path,
            {
                "config/config.yaml": DM16_YAML,
                "config/app_config.py": DM16_APP_CONFIG,
                "core/consumer.py": DM16_CONSUMER,
            },
        )
        result = dm16_config_reachability.scan(tmp_path)
        symbols = [f.symbol for f in result.findings]
        assert symbols == ["section.dead_key", "section.nested.deep_dead_key"]
        assert all(f.path == "config/config.yaml" for f in result.findings)

    def test_green_every_constant_has_a_consumer(self, tmp_path):
        build_tree(
            tmp_path,
            {
                "config/config.yaml": DM16_YAML,
                "config/app_config.py": DM16_APP_CONFIG,
                "core/consumer.py": DM16_CONSUMER_ALL,
            },
        )
        assert dm16_config_reachability.scan(tmp_path).findings == []

    def test_multiline_scalar_continuation_is_never_read_as_a_key(self):
        leaves = dm16_config_reachability.read_leaves(DM16_YAML)
        dotted = [leaf[0] for leaf in leaves]
        assert "section.prose" in dotted
        assert not any("Input" in name for name in dotted)
        assert not any("item_one" in name for name in dotted)

    def test_missing_config_reports_zero_files_rather_than_passing(self, tmp_path):
        result = dm16_config_reachability.scan(tmp_path)
        assert result.files_processed == 0 and result.findings == []


class TestDm17ApplyWithoutGuard:
    def test_red_flags_apply_script_and_test_data_path(self, tmp_path):
        build_tree(
            tmp_path,
            {
                "scripts/purge_things.py": DM17_RED_SCRIPT,
                "tests/unit/test_store.py": DM17_RED_TEST,
            },
        )
        result = dm17_apply_without_guard.scan(tmp_path)
        paths = [f.path for f in result.findings]
        assert paths == ["scripts/purge_things.py", "tests/unit/test_store.py"]
        script = result.findings[0]
        assert script.symbol == ""
        assert "--apply" in script.text

    def test_green_guarded_script_and_sandboxed_test(self, tmp_path):
        build_tree(
            tmp_path,
            {
                "scripts/purge_things.py": DM17_GREEN_SCRIPT,
                "tests/unit/test_store.py": DM17_GREEN_TEST,
            },
        )
        assert dm17_apply_without_guard.scan(tmp_path).findings == []

    def test_conftest_owns_the_sandbox_redirects_and_is_exempt(self, tmp_path):
        build_tree(tmp_path, {"tests/conftest.py": DM17_RED_TEST})
        result = dm17_apply_without_guard.scan(tmp_path)
        assert result.files_processed == 1
        assert result.findings == []


class TestDm18ExceptReturnsEmpty:
    def test_red_flags_broad_handler_returning_empty_beside_a_store_call(self, tmp_path):
        build_tree(tmp_path, {"memory/store.py": DM18_RED})
        result = dm18_except_returns_empty.scan(tmp_path)
        assert [(f.path, f.symbol, f.text) for f in result.findings] == [
            ("memory/store.py", "Store.get_rows", "return []")
        ]

    def test_green_reraise_and_log_then_raise_are_never_flagged(self, tmp_path):
        build_tree(tmp_path, {"memory/store.py": DM18_GREEN_RERAISE})
        assert dm18_except_returns_empty.scan(tmp_path).findings == []

    def test_green_narrow_handler_and_tri_state_return(self, tmp_path):
        build_tree(tmp_path, {"memory/store.py": DM18_GREEN_NARROW})
        assert dm18_except_returns_empty.scan(tmp_path).findings == []

    def test_green_function_without_a_store_call(self, tmp_path):
        build_tree(tmp_path, {"memory/parse.py": DM18_GREEN_NO_STORE})
        assert dm18_except_returns_empty.scan(tmp_path).findings == []

    def test_scope_is_limited_to_the_retrieval_layers(self, tmp_path):
        build_tree(tmp_path, {"gui/handlers.py": DM18_RED, "memory/store.py": DM18_RED})
        result = dm18_except_returns_empty.scan(tmp_path)
        assert {f.path for f in result.findings} == {"memory/store.py"}


class TestDm29PhraseAppend:
    def test_red_fires_on_three_distinct_dated_batches(self, tmp_path):
        text = changelog(
            [
                ("2026-08-15", DM29_WIDENING_LINE),
                ("2026-09-07", DM29_WIDENING_LINE),
                ("2026-09-10", DM29_WIDENING_LINE),
            ]
        )
        build_tree(tmp_path, {"CLAUDE_CHANGELOG.md": text})
        result = dm29_phrase_append.scan(tmp_path)
        assert [(f.symbol, f.path) for f in result.findings] == [
            ("_INFO_SEEKING_CUES", "CLAUDE_CHANGELOG.md")
        ]
        assert "3 dated batches" in result.findings[0].text

    def test_green_two_batches_is_below_the_signature(self, tmp_path):
        text = changelog(
            [("2026-08-15", DM29_WIDENING_LINE), ("2026-09-07", DM29_WIDENING_LINE)]
        )
        build_tree(tmp_path, {"CLAUDE_CHANGELOG.md": text})
        assert dm29_phrase_append.scan(tmp_path).findings == []

    def test_green_three_mentions_inside_one_batch_is_not_the_signature(self, tmp_path):
        body = "\n".join([DM29_WIDENING_LINE] * 3)
        build_tree(tmp_path, {"CLAUDE_CHANGELOG.md": changelog([("2026-09-10", body)])})
        assert dm29_phrase_append.scan(tmp_path).findings == []

    def test_missing_changelog_reports_zero_files_rather_than_passing(self, tmp_path):
        result = dm29_phrase_append.scan(tmp_path)
        assert result.files_processed == 0
        assert result.findings == []


class TestCatalogScanner:
    def test_red_surfaces_an_index_body_disagreement(self, tmp_path):
        build_tree(
            tmp_path,
            {
                "docs/BUG_CLASSES.md": catalog_doc(
                    [("BC-01", "open")], [("BC-01", "closed")]
                )
            },
        )
        result = catalog_scanner.scan(tmp_path)
        assert result.files_processed == 1
        assert [(f.symbol, f.text) for f in result.findings] == [
            ("BC-01", "index/body status disagreement: BC-01 index=open body=closed")
        ]

    def test_red_surfaces_an_id_present_in_only_one_section(self, tmp_path):
        build_tree(
            tmp_path,
            {
                "docs/BUG_CLASSES.md": catalog_doc(
                    [("BC-01", "open"), ("BC-78", "open")], [("BC-01", "open")]
                )
            },
        )
        findings = catalog_scanner.scan(tmp_path).findings
        assert any("only in index" in f.text for f in findings)

    def test_green_consistent_catalog_has_no_findings(self, tmp_path):
        build_tree(
            tmp_path,
            {"docs/BUG_CLASSES.md": catalog_doc([("BC-01", "closed"), ("BC-02", "open")])},
        )
        assert catalog_scanner.scan(tmp_path).findings == []

    def test_green_live_catalog_is_consistent(self):
        """Parity, not a red control: the real catalog must stay reconciled.

        It lives here rather than in a fixture so that fixing catalog debt
        (BC-77 went index=open/body=closed -> partial on 2026-09-11) makes
        this test pass, never fail.
        """
        assert catalog_scanner.scan(REPO_ROOT).findings == []

    def test_unparseable_catalog_is_a_scanner_error_not_a_clean_pass(self, tmp_path):
        build_tree(tmp_path, {"docs/BUG_CLASSES.md": "# nothing here\n"})
        with pytest.raises(ScannerError):
            catalog_scanner.scan(tmp_path)

    def test_missing_catalog_is_a_scanner_error(self, tmp_path):
        with pytest.raises(ScannerError):
            catalog_scanner.scan(tmp_path)


class TestDeterminism:
    def test_findings_are_stable_across_runs_and_carry_no_absolute_paths(self, tmp_path):
        build_tree(
            tmp_path,
            {"core/tone.py": DM01_RED, "core/other/tone.py": DM01_RED},
        )
        first = [f.fingerprint() for f in dm01_raw_substring.scan(tmp_path).findings]
        second = [f.fingerprint() for f in dm01_raw_substring.scan(tmp_path).findings]
        assert first == second
        assert first == sorted(first)
        assert not any(str(tmp_path) in part for key in first for part in key)
