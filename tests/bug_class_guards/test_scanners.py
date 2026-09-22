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
from bug_class_guards.scanners import dm31_live_state_default
from bug_class_guards.scanners.common import ScannerError

REPO_ROOT = Path(__file__).resolve().parents[2]

from fixtures import (
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
    DM31_GREEN_CALLER_SIZES,
    DM31_GREEN_PRIVATE_HELPER,
    DM31_GREEN_RESOLVED,
    DM31_RED,
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
        texts = [f.excerpt for f in result.findings]
        assert texts == ['if "ice" in text_lower:',
                         "return any(word in text_lower for word in HEAVY_KEYWORDS)"]
        assert all(f.symbol == "is_heavy" for f in result.findings)
        assert all(f.path == "core/tone.py" for f in result.findings)

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
        unreached = [f.symbol for f in result.findings if f.kind == "config_leaf_unreached"]
        assert unreached == ["section.dead_key", "section.nested.deep_dead_key"]
        assert all(f.path == "config/config.yaml" for f in result.findings)
        # Contract v2: a mapping inside a list item is surfaced, not skipped.
        unresolved = [f for f in result.findings if f.kind == "config_yaml_unresolved"]
        assert [(f.symbol, f.unresolved) for f in unresolved] == [("list_section", True)]

    def test_green_every_constant_has_a_consumer(self, tmp_path):
        build_tree(
            tmp_path,
            {
                "config/config.yaml": DM16_YAML,
                "config/app_config.py": DM16_APP_CONFIG,
                "core/consumer.py": DM16_CONSUMER_ALL,
            },
        )
        findings = dm16_config_reachability.scan(tmp_path).findings
        assert [f for f in findings if f.kind == "config_leaf_unreached"] == []

    def test_multiline_scalar_continuation_is_never_read_as_a_key(self):
        leaves = dm16_config_reachability.read_leaves(DM16_YAML)
        dotted = [leaf[0] for leaf in leaves]
        assert "section.prose" in dotted
        assert not any("Input" in name for name in dotted)
        assert not any("item_one" in name for name in dotted)

    def test_missing_config_reports_missing_legs_rather_than_passing(self, tmp_path):
        result = dm16_config_reachability.scan(tmp_path)
        assert result.files_processed == 0 and result.findings == []
        legs = {leg.id: leg.status for leg in result.legs}
        assert legs["dm16_config_yaml"] == "missing" and legs["dm16_app_config"] == "missing"
        assert legs["dm16_schema"] == "unavailable"


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
        assert (script.symbol, script.kind) == ("<module>", "apply_without_guard")
        assert "--apply" in script.excerpt

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
        assert [(f.path, f.symbol, f.excerpt) for f in result.findings] == [
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


class TestDm31LiveStateDefault:
    """BC-78/BC-11/BC-12: a public function that asserts live external state
    through a literal default. The reference case is the 2026-09-11 web
    trigger — `remaining_credits: float = 100` and `web_search_enabled: bool
    = True`, neither passed by the agentic gate."""

    def test_red_flags_both_live_state_defaults(self, tmp_path):
        build_tree(tmp_path, {"utils/trigger.py": DM31_RED})
        result = dm31_live_state_default.scan(tmp_path)
        assert [(f.path, f.symbol, f.excerpt) for f in result.findings] == [
            ("utils/trigger.py", "analyze_for_web_search_llm",
             "web_search_enabled: bool = True,"),
            ("utils/trigger.py", "analyze_for_web_search_llm",
             "remaining_credits: float = 100,"),
        ]

    def test_green_none_default_is_the_fix_not_a_finding(self, tmp_path):
        build_tree(tmp_path, {"utils/trigger.py": DM31_GREEN_RESOLVED})
        assert dm31_live_state_default.scan(tmp_path).findings == []

    def test_green_caller_chosen_sizes_and_fail_closed_toggle(self, tmp_path):
        build_tree(tmp_path, {"core/gather.py": DM31_GREEN_CALLER_SIZES})
        assert dm31_live_state_default.scan(tmp_path).findings == []

    def test_green_private_helper_is_out_of_scope(self, tmp_path):
        build_tree(tmp_path, {"utils/trigger.py": DM31_GREEN_PRIVATE_HELPER})
        assert dm31_live_state_default.scan(tmp_path).findings == []

    def test_the_live_tree_has_no_unbaselined_finding(self):
        """Consistency, not a moving oracle: fixing real debt can only make
        this pass (see the catalog controls' note on BC-65)."""
        result = dm31_live_state_default.scan(REPO_ROOT)
        assert result.files_processed > 100
        for finding in result.findings:
            assert finding.path != "utils/web_search_trigger.py", (
                "the reference defect is back: "
                f"{finding.symbol} {finding.text}")


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
        assert "3 dated batches" in result.findings[0].excerpt

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

    def test_missing_changelog_is_an_unavailable_leg_not_a_clean_zero(self, tmp_path):
        result = dm29_phrase_append.scan(tmp_path)
        assert result.files_processed == 0
        assert result.findings == []
        (leg,) = result.legs
        assert (leg.required, leg.status, leg.available) == (False, "unavailable", False)


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
        assert [(f.symbol, f.excerpt) for f in result.findings] == [
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
        assert any("only in index" in f.excerpt for f in findings)

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

    def test_missing_catalog_is_a_missing_required_leg(self, tmp_path):
        result = catalog_scanner.scan(tmp_path)
        assert result.findings == []
        (leg,) = result.legs
        assert (leg.required, leg.status, leg.files_processed) == (True, "missing", 0)


class TestDeterminism:
    def test_findings_are_stable_across_runs_and_carry_no_absolute_paths(self, tmp_path):
        build_tree(
            tmp_path,
            {"core/tone.py": DM01_RED, "core/other/tone.py": DM01_RED},
        )
        findings = dm01_raw_substring.scan(tmp_path).findings
        first = [f.fingerprint() for f in findings]
        second = [f.fingerprint() for f in dm01_raw_substring.scan(tmp_path).findings]
        assert first == second
        assert [(f.path, f.line) for f in findings] == sorted((f.path, f.line) for f in findings)
        assert not any(str(tmp_path) in part for key in first for part in key)


# ---------------------------------------------------------------------------
# Contract v2 controls (2026-09-13 class-guard completion plan)
# ---------------------------------------------------------------------------

import re  # noqa: E402

from fixtures import (  # noqa: E402
    BENIGN,
    DM01_GREEN_MATCHER_ONLY,
    DM01_RED_BESIDE_MATCHER,
    DM01_RED_IMPORT_ONLY,
    DM01_RED_OUTSIDE_OLD_PREFILTER,
    DM17_GREEN_GUARDED_CALLEE,
    DM17_GREEN_TRY_IMPORT,
    DM17_GREEN_WRAPPER,
    DM17_RED_GUARD_IMPORT_ONLY,
    DM17_RED_GUARD_IN_COMMENT,
    DM17_RED_GUARD_IN_STRING,
    DM17_RED_GUARD_LATE,
    DM17_RED_GUARD_UNCALLED_HELPER,
    DM17_RED_GUARD_UNREACHABLE,
    DM17_RED_TEST_CONCATENATED,
    DM17_ROOT_CONFTEST_SANDBOX,
    DM17_SCOPE_PROSE_ONLY_APPLY,
    DM18_TWIN_HANDLERS,
)


class TestDm01NoModuleExemption:
    """An import of ``utils.trigger_match`` — or a real matcher call — says
    nothing about the OTHER keyword tests in the same module. Contract v1
    skipped the whole module on the import; v2 judges every expression."""

    def test_matcher_import_alone_does_not_exempt_a_raw_test(self, tmp_path):
        build_tree(tmp_path, {"core/tone.py": DM01_RED_IMPORT_ONLY})
        findings = dm01_raw_substring.scan(tmp_path).findings
        assert [(f.symbol, f.excerpt) for f in findings] == [("is_heavy", 'return "ice" in text_lower')]

    def test_real_matcher_use_does_not_hide_a_raw_test_beside_it(self, tmp_path):
        build_tree(tmp_path, {"core/tone.py": DM01_RED_BESIDE_MATCHER})
        findings = dm01_raw_substring.scan(tmp_path).findings
        assert [f.excerpt for f in findings] == ['if "ice" in text_lower:']

    def test_module_that_only_calls_the_matcher_has_no_finding(self, tmp_path):
        build_tree(tmp_path, {"core/tone.py": DM01_GREEN_MATCHER_ONLY})
        assert dm01_raw_substring.scan(tmp_path).findings == []

    def test_lowered_text_shapes_outside_the_removed_prefilter_are_found(self, tmp_path):
        build_tree(tmp_path, {"core/tone.py": DM01_RED_OUTSIDE_OLD_PREFILTER})
        findings = dm01_raw_substring.scan(tmp_path).findings
        assert [(f.symbol, f.excerpt) for f in findings] == [
            ("Detector.is_heavy", 'if "ice" in normalize(text).lower():'),
            ("Detector.is_heavy", 'return "crisis" in self.text_lower'),
        ]


class TestDm17GuardEvidence:
    """BC-37 contract v2: guard TEXT is not guard EVIDENCE.

    Only a recognized guard ``if`` — a call to ``utils.daemon_guard``'s
    ``daemon_running`` (directly or through a module wrapper that returns it)
    whose body refuses — counts, and only when it structurally runs before
    every statement that consumes the ``--apply`` value in its function.
    Anything else with a guard call is an unresolved candidate for a human.
    """

    def _script_findings(self, tmp_path, source):
        build_tree(tmp_path, {"scripts/store_tool.py": source, "tests/unit/test_ok.py": DM17_GREEN_TEST})
        result = dm17_apply_without_guard.scan(tmp_path)
        return result, [f for f in result.findings if f.path == "scripts/store_tool.py"]

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param(DM17_RED_GUARD_IN_COMMENT, id="comment"),
            pytest.param(DM17_RED_GUARD_IN_STRING, id="string"),
            pytest.param(DM17_RED_GUARD_IMPORT_ONLY, id="import-only"),
            pytest.param(DM17_RED_SCRIPT, id="no-mention"),
        ],
    )
    def test_guard_text_without_a_call_is_an_unguarded_script(self, tmp_path, source):
        _, findings = self._script_findings(tmp_path, source)
        assert [(f.kind, f.symbol) for f in findings] == [("apply_without_guard", "<module>")]
        assert "--apply" in findings[0].excerpt

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param(DM17_RED_GUARD_LATE, id="late"),
            pytest.param(DM17_RED_GUARD_UNREACHABLE, id="constant-false-branch"),
            pytest.param(DM17_RED_GUARD_UNCALLED_HELPER, id="uncalled-helper"),
        ],
    )
    def test_guard_call_that_does_not_run_first_is_unresolved(self, tmp_path, source):
        result, findings = self._script_findings(tmp_path, source)
        assert [f.kind for f in findings] == ["apply_guard_unresolved"]
        legs = {leg.id: leg for leg in result.legs}
        assert legs["dm17_scripts"].unresolved == 1

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param(DM17_GREEN_SCRIPT, id="direct-guard"),
            pytest.param(DM17_GREEN_WRAPPER, id="wrapped-guard-with-force"),
            pytest.param(DM17_GREEN_TRY_IMPORT, id="try-import-after-dry-run"),
            pytest.param(DM17_GREEN_GUARDED_CALLEE, id="self-guarded-callee"),
        ],
    )
    def test_recognized_guard_that_runs_before_apply_is_green(self, tmp_path, source):
        _, findings = self._script_findings(tmp_path, source)
        assert findings == []

    def test_scope_boundary_prose_mentioning_the_flag_is_not_an_apply_script(self, tmp_path):
        _, findings = self._script_findings(tmp_path, DM17_SCOPE_PROSE_ONLY_APPLY)
        assert findings == []

    def test_concatenated_data_literal_outside_the_removed_prefilter_is_found(self, tmp_path):
        build_tree(tmp_path, {"tests/unit/test_store.py": DM17_RED_TEST_CONCATENATED, "scripts/tool.py": BENIGN})
        findings = dm17_apply_without_guard.scan(tmp_path).findings
        assert [(f.path, f.kind) for f in findings] == [("tests/unit/test_store.py", "test_data_path_literal")]

    def test_leg_receipts_count_scripts_and_tests_independently(self, tmp_path):
        build_tree(
            tmp_path,
            {"scripts/a.py": BENIGN, "scripts/b.py": BENIGN, "tests/unit/test_x.py": DM17_GREEN_TEST},
        )
        legs = {leg.id: leg for leg in dm17_apply_without_guard.scan(tmp_path).legs}
        assert (legs["dm17_scripts"].files_processed, legs["dm17_tests"].files_processed) == (2, 1)

    def test_empty_scripts_leg_is_not_hidden_by_a_nonempty_tests_leg(self, tmp_path):
        build_tree(tmp_path, {"scripts/nested/tool.py": BENIGN, "tests/unit/test_x.py": DM17_GREEN_TEST})
        legs = {leg.id: leg for leg in dm17_apply_without_guard.scan(tmp_path).legs}
        assert legs["dm17_scripts"].status == "empty"
        assert legs["dm17_tests"].status == "available"


class TestDm17RootConftestException:
    def test_only_the_exact_root_conftest_path_is_exempt(self, tmp_path):
        build_tree(
            tmp_path,
            {
                "tests/conftest.py": DM17_ROOT_CONFTEST_SANDBOX,
                "tests/unit/conftest.py": DM17_ROOT_CONFTEST_SANDBOX,
                "scripts/tool.py": BENIGN,
            },
        )
        result = dm17_apply_without_guard.scan(tmp_path)
        assert {f.path for f in result.findings} == {"tests/unit/conftest.py"}
        legs = {leg.id: leg for leg in result.legs}
        assert legs["dm17_tests"].files_processed == 2


class TestAnchors:
    """Contract v2 identity: (scanner, path, qualname, kind, SHA-256 of the
    candidate's canonical AST). Line numbers are never identity; the excerpt
    is for humans only."""

    def _only(self, tmp_path, source):
        build_tree(tmp_path, {"memory/store.py": source})
        (finding,) = dm18_except_returns_empty.scan(tmp_path).findings
        return finding

    def test_anchor_is_five_nonempty_fields_with_a_sha256_digest(self, tmp_path):
        finding = self._only(tmp_path, DM18_RED)
        assert finding.fingerprint() == (
            "dm18_except_returns_empty", "memory/store.py", "Store.get_rows",
            "broad_except_returns_empty", finding.digest,
        )
        assert re.fullmatch(r"[0-9a-f]{64}", finding.digest)
        assert finding.excerpt == "return []"

    def test_pure_line_movement_keeps_the_anchor(self, tmp_path):
        before = self._only(tmp_path, DM18_RED)
        after = self._only(tmp_path, DM18_RED.replace("import logging\n", "import logging\nimport os\n\n# note\n\n"))
        assert after.fingerprint() == before.fingerprint()
        assert after.line != before.line

    def test_formatting_inside_the_candidate_keeps_the_digest(self, tmp_path):
        before = self._only(tmp_path, DM18_RED)
        after = self._only(tmp_path, DM18_RED.replace("return []", "return [ ]"))
        assert after.digest == before.digest

    def test_editing_the_candidate_changes_the_digest(self, tmp_path):
        before = self._only(tmp_path, DM18_RED)
        after = self._only(tmp_path, DM18_RED.replace("logger.warning(", "logger.error("))
        assert after.digest != before.digest and after.symbol == before.symbol

    def test_moving_the_candidate_into_another_function_changes_the_symbol(self, tmp_path):
        before = self._only(tmp_path, DM18_RED)
        after = self._only(tmp_path, DM18_RED.replace("def get_rows", "def fetch_rows"))
        assert after.symbol == "Store.fetch_rows" and after.fingerprint() != before.fingerprint()

    def test_identical_candidates_in_one_function_share_one_anchor(self, tmp_path):
        build_tree(tmp_path, {"memory/store.py": DM18_TWIN_HANDLERS})
        findings = dm18_except_returns_empty.scan(tmp_path).findings
        assert len(findings) == 2
        assert findings[0].fingerprint() == findings[1].fingerprint()

    def test_every_finding_kind_is_declared_by_its_scanner(self, tmp_path):
        build_tree(
            tmp_path,
            {
                "core/tone.py": DM01_RED,
                "memory/store.py": DM18_RED,
                "utils/trigger.py": DM31_RED,
                "scripts/purge_things.py": DM17_RED_SCRIPT,
                "tests/unit/test_store.py": DM17_RED_TEST,
            },
        )
        for scanner in SCANNERS.values():
            for finding in scanner.scan(tmp_path).findings:
                assert finding.kind in scanner.kinds, (scanner.id, finding.kind)
                assert finding.scanner_id == scanner.id


# ---------------------------------------------------------------------------
# DM-38 / BC-91 — audited machinery consumers + emitter registry (2026-09-22)
# ---------------------------------------------------------------------------

from bug_class_guards.scanners import dm38_machinery_consumers  # noqa: E402


_DM38_BUILDER = """
def _topup_filler(recents, mems, extra_recent, needed):
    used = {_canonical_turn_key(r) for r in (recents or [])}
    used.update(_canonical_turn_key(m) for m in (mems or []))
    for item in (extra_recent or []):
        if _canonical_turn_key(item) not in used:
            pass

def _recency_floor_filler(recent_convos, stored_recent, needed):
    have_keys = {_canonical_turn_key(r) for r in (recent_convos or [])}
    for r in (stored_recent or []):
        key = _canonical_turn_key(r)
        if key not in have_keys:
            have_keys.add(key)
"""

_DM38_HANDLERS = """
import utils.read_time_markers as read_time_markers


async def _apply_delivery_revisions(ctx, response_text, *, source_material=""):
    body = response_text
    grounded, _ = await _apply_grounding_check_for_delivery(ctx, body, source_material=source_material)
    if grounded:
        body = grounded
    personal = await _apply_personal_claim_check_for_delivery(ctx, body)
    if personal:
        body = personal
    return body


async def _run_agentic_search(ctx):
    display_output = "draft"
    _pre_suffix = display_output
    _delivery_body = _pre_suffix
    _ag_guard_suffix = ""
    if _ag_guard_suffix:
        display_output = display_output.rstrip() + _ag_guard_suffix
    _delivery_body = await _apply_delivery_revisions(ctx, _delivery_body)
    display_output = _delivery_body.rstrip() + (_ag_guard_suffix or "")
    return display_output


async def _run_enhanced(ctx):
    _resp_for_debug = "draft"
    _resp_for_debug += _format_card(ctx)
    _pre_suffix = _resp_for_debug
    _delivery_body = _pre_suffix
    _guard_suffix = ""
    if _guard_suffix:
        _resp_for_debug = _resp_for_debug.rstrip() + _guard_suffix
    _delivery_body = await _apply_delivery_revisions(ctx, _delivery_body)
    return _delivery_body.rstrip() + (_guard_suffix or "")


def _emit_calendar_notice():
    return read_time_markers.delivery_notice(read_time_markers.NOTICE_CALENDAR_UNSEEN, " Say add it.")
"""

_DM38_FORMATTER = """
class PromptFormatter:
    def _format_memory(self, mem):
        response = mem.get("response", "")
        response = read_time_markers.strip_delivery_notices(response)
        return response

    def _assemble_prompt(self):
        def mem_parts(mem):
            r = str(mem.get("response", ""))
            r = read_time_markers.strip_delivery_notices(r)
            return r
        last_a = recent[0].get("response", "")
        last_a = read_time_markers.strip_delivery_notices(last_a)
        return last_a
"""

# The leaf may (must) hold the prefix literal; every other file may not.
_DM38_LEAF = """
DELIVERY_NOTICE_PREFIX = "> ⚠️"
NOTICE_CALENDAR_UNSEEN = "I don't see that on your calendar — nothing was created."
DELIVERY_NOTICE_TEXTS = (NOTICE_CALENDAR_UNSEEN,)


def delivery_notice(opening, detail=""):
    return "\\n\\n" + DELIVERY_NOTICE_PREFIX + " " + opening + detail
"""


def _dm38_tree(tmp_path, *, builder=_DM38_BUILDER, handlers=_DM38_HANDLERS,
               formatter=_DM38_FORMATTER, extra=None):
    files = {
        "core/prompt/builder.py": builder,
        "core/prompt/hygiene.py": "def _canonical_turn_key(item): return str(item)\n",
        "core/prompt/formatter.py": formatter,
        "gui/handlers.py": handlers,
        "utils/read_time_markers.py": _DM38_LEAF,
    }
    files.update(extra or {})
    build_tree(tmp_path, files)


def _dm38_findings(tmp_path):
    result = dm38_machinery_consumers.scan(tmp_path)
    assert all(f.scanner_id == "dm38_machinery_consumers" for f in result.findings)
    return result.findings


class TestDM38Green:
    def test_audited_sites_and_registered_emitter_are_clean(self, tmp_path):
        _dm38_tree(tmp_path)
        assert _dm38_findings(tmp_path) == []


class TestDM38BuilderIdentity:
    @pytest.mark.parametrize("helper,old,new", [
        ("_topup_filler", "if _canonical_turn_key(item) not in used:", "if item.get('query', '') not in used:"),
        ("_recency_floor_filler", "key = _canonical_turn_key(r)", "key = r.get('query', '') + r.get('response', '')"),
    ])
    def test_red_when_a_backfill_uses_a_raw_identity(self, tmp_path, helper, old, new):
        builder = _DM38_BUILDER.replace(old, new)
        assert builder != _DM38_BUILDER
        _dm38_tree(tmp_path, builder=builder)
        findings = _dm38_findings(tmp_path)
        assert findings and {f.symbol for f in findings} == {helper}


class TestDM38DeliveryPipeline:
    def test_red_when_a_site_passes_the_suffixed_display_value(self, tmp_path):
        handlers = _DM38_HANDLERS.replace(
            "_delivery_body = await _apply_delivery_revisions(ctx, _delivery_body)\n    display_output",
            "_delivery_body = await _apply_delivery_revisions(ctx, display_output)\n    display_output",
        )
        assert handlers != _DM38_HANDLERS
        _dm38_tree(tmp_path, handlers=handlers)
        findings = _dm38_findings(tmp_path)
        assert [f.symbol for f in findings] == ["_run_agentic_search"]
        assert "delivery-suffixed or unproven" in findings[0].excerpt

    def test_red_when_a_site_calls_a_checker_directly(self, tmp_path):
        handlers = _DM38_HANDLERS.replace(
            "    _delivery_body = await _apply_delivery_revisions(ctx, _delivery_body)\n    return",
            "    await _apply_grounding_check_for_delivery(ctx, _pre_suffix)\n"
            "    _delivery_body = await _apply_delivery_revisions(ctx, _delivery_body)\n    return",
        )
        assert handlers != _DM38_HANDLERS
        _dm38_tree(tmp_path, handlers=handlers)
        findings = _dm38_findings(tmp_path)
        assert [f.symbol for f in findings] == ["_run_enhanced"]
        assert "bypassing the ordered" in findings[0].excerpt

    def test_red_when_a_site_drops_the_pipeline(self, tmp_path):
        handlers = _DM38_HANDLERS.replace(
            "    _delivery_body = await _apply_delivery_revisions(ctx, _delivery_body)\n    return",
            "    return",
        )
        assert handlers != _DM38_HANDLERS
        _dm38_tree(tmp_path, handlers=handlers)
        findings = _dm38_findings(tmp_path)
        assert [f.symbol for f in findings] == ["_run_enhanced"]
        assert "no longer routes" in findings[0].excerpt

    def test_red_when_the_second_revision_reads_the_original_draft(self, tmp_path):
        handlers = _DM38_HANDLERS.replace(
            "personal = await _apply_personal_claim_check_for_delivery(ctx, body)",
            "personal = await _apply_personal_claim_check_for_delivery(ctx, response_text)",
        )
        assert handlers != _DM38_HANDLERS
        _dm38_tree(tmp_path, handlers=handlers)
        findings = _dm38_findings(tmp_path)
        assert [f.symbol for f in findings] == ["_apply_delivery_revisions"]
        assert "sequential revisions must compose" in findings[0].excerpt

    def test_red_when_the_pipeline_drops_a_checker(self, tmp_path):
        handlers = _DM38_HANDLERS.replace(
            "    personal = await _apply_personal_claim_check_for_delivery(ctx, body)\n"
            "    if personal:\n        body = personal\n",
            "",
        )
        assert handlers != _DM38_HANDLERS
        _dm38_tree(tmp_path, handlers=handlers)
        findings = _dm38_findings(tmp_path)
        assert [f.symbol for f in findings] == ["_apply_delivery_revisions"]
        assert "no longer runs the audited _apply_personal_claim_check_for_delivery" in findings[0].excerpt

    def test_red_when_the_pipeline_receives_a_suffixed_name(self, tmp_path):
        handlers = _DM38_HANDLERS.replace(
            "    grounded, _ = await _apply_grounding_check_for_delivery(ctx, body,",
            "    body = body + _guard_notice\n"
            "    grounded, _ = await _apply_grounding_check_for_delivery(ctx, body,",
        )
        assert handlers != _DM38_HANDLERS
        _dm38_tree(tmp_path, handlers=handlers)
        findings = _dm38_findings(tmp_path)
        assert findings and {f.symbol for f in findings} == {"_apply_delivery_revisions"}


class TestDM38Render:
    @pytest.mark.parametrize("variable,old,new", [
        ("response", "        response = read_time_markers.strip_delivery_notices(response)\n", ""),
        ("r", "            r = read_time_markers.strip_delivery_notices(r)\n", ""),
        ("last_a", "        last_a = read_time_markers.strip_delivery_notices(last_a)\n", ""),
    ])
    def test_red_when_a_render_bypasses_the_notice_stripper(self, tmp_path, variable, old, new):
        formatter = _DM38_FORMATTER.replace(old, new)
        assert formatter != _DM38_FORMATTER
        _dm38_tree(tmp_path, formatter=formatter)
        findings = _dm38_findings(tmp_path)
        assert findings and all(variable in f.excerpt for f in findings)


class TestDM38EmitterRegistry:
    def test_red_when_a_notice_literal_appears_outside_the_leaf(self, tmp_path):
        _dm38_tree(tmp_path, extra={
            "core/some_guard.py": 'NEW_NOTICE = "\\n\\n> ⚠️ Heads up — something new happened."\n',
        })
        findings = _dm38_findings(tmp_path)
        assert [(f.path, f.symbol) for f in findings] == [("core/some_guard.py", "<module>")]
        assert "delivery-notice literal outside utils/read_time_markers.py" in findings[0].excerpt

    def test_red_when_an_emitter_passes_an_unregistered_opening(self, tmp_path):
        handlers = _DM38_HANDLERS.replace(
            "read_time_markers.delivery_notice(read_time_markers.NOTICE_CALENDAR_UNSEEN, ",
            'read_time_markers.delivery_notice("I don\'t see that anywhere", ',
        )
        assert handlers != _DM38_HANDLERS
        _dm38_tree(tmp_path, handlers=handlers)
        findings = _dm38_findings(tmp_path)
        assert [f.symbol for f in findings] == ["_emit_calendar_notice"]
        assert "registered NOTICE_* constant" in findings[0].excerpt
