"""Policy contract: the enforced scanner set is pinned, and drift fails closed.

``config/bug_class_policy.json`` is the reviewed statement of what the gate
enforces; ``bug_class_guards.scanners.SCANNERS`` is the implementation.  A
scan compares the two BEFORE any scanner runs, so deleting, renaming,
downgrading or re-classing a scanner — or adding an unreviewed one — cannot
turn a scan green by making its own obligation disappear.  Before this
contract, removing a scanner or downgrading a gate exited 0 (the 2026-09-13
review's sensitivity table).
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from bug_class_guards import scanners as scanners_pkg

from fixtures import (
    EXPECTED_LEGS,
    EXPECTED_REPO_WIDE_GUARDS,
    EXPECTED_SCANNER_LEGS,
    EXPECTED_SCANNERS,
    POLICY_PATH,
    REPO_ROOT,
    build_repo,
    failure_codes,
    run_scan,
)


def _live_policy() -> dict:
    return json.loads((REPO_ROOT / POLICY_PATH).read_text(encoding="utf-8"))


class TestPinnedContract:
    def test_policy_declares_exactly_the_eight_reviewed_scanners(self):
        entries = _live_policy()["scanners"]
        declared = {e["id"]: (e["mode"], tuple(e["class_ids"])) for e in entries}
        assert len(entries) == len(declared) == 8
        assert declared == EXPECTED_SCANNERS

    def test_registry_implements_exactly_the_pinned_scanners(self):
        observed = {
            scanner_id: (scanner.mode, tuple(scanner.class_ids))
            for scanner_id, scanner in scanners_pkg.SCANNERS.items()
        }
        assert observed == EXPECTED_SCANNERS
        assert all(key == scanner.id for key, scanner in scanners_pkg.SCANNERS.items())

    def test_policy_and_registry_agree_on_version_legs_and_kinds(self):
        for entry in _live_policy()["scanners"]:
            scanner = scanners_pkg.SCANNERS[entry["id"]]
            assert tuple(entry["legs"]) == EXPECTED_SCANNER_LEGS[entry["id"]]
            assert [leg.id for leg in scanner.legs] == entry["legs"]
            assert list(scanner.kinds) == entry["kinds"]
            assert type(entry["contract_version"]) is int
            assert scanner.contract_version == entry["contract_version"]

    def test_policy_pins_every_input_leg_and_root(self):
        legs = {
            leg["id"]: (leg["kind"], tuple(leg["roots"]), leg["required"])
            for leg in _live_policy()["legs"]
        }
        assert legs == EXPECTED_LEGS

    def test_registry_legs_match_the_policy_leg_definitions(self):
        declared = {leg["id"]: leg for leg in _live_policy()["legs"]}
        for scanner in scanners_pkg.SCANNERS.values():
            for leg in scanner.legs:
                entry = declared[leg.id]
                assert (leg.kind, list(leg.roots), leg.required) == (
                    entry["kind"], entry["roots"], entry["required"]
                )
                assert [item.path for item in leg.exempt] == [
                    item["path"] for item in entry["exempt"]
                ]

    def test_report_only_scanners_stay_report_only(self):
        report_only = {sid for sid, (mode, _) in EXPECTED_SCANNERS.items() if mode == "report"}
        assert report_only == {"dm16_config_key_reachability", "dm29_phrase_append_signature"}  # gitleaks:allow

    def test_policy_claims_scoped_structural_coverage_only(self):
        claim = _live_policy()["claim"].lower()
        assert "not behavioral coverage" in claim
        assert "coverage gap" in claim

    def test_policy_names_the_five_repo_wide_guards(self):
        assert _live_policy()["repo_wide_guards"] == EXPECTED_REPO_WIDE_GUARDS

    def test_duplicate_scanner_ids_are_rejected_when_the_registry_is_built(self):
        first = next(iter(scanners_pkg.SCANNERS.values()))
        with pytest.raises(ValueError, match="duplicate scanner id"):
            scanners_pkg.build_registry([first, first])

    def test_green_scan_reports_expected_and_observed_registry(self, tmp_path, capsys):
        build_repo(tmp_path)
        code, payload, _, err = run_scan(capsys, tmp_path)
        assert code == 0, err
        registry = payload["registry"]
        expected = {e["id"]: (e["mode"], tuple(e["class_ids"])) for e in registry["expected"]}
        observed = {e["id"]: (e["mode"], tuple(e["class_ids"])) for e in registry["observed"]}
        assert expected == observed == EXPECTED_SCANNERS
        assert registry["mismatches"] == []


# ---------------------------------------------------------------------------
# Registry drift — every mutation must stop the scan before any comparison.
# ---------------------------------------------------------------------------


def _swap(registry, monkeypatch, scanner_id, **changes):
    monkeypatch.setitem(
        registry, scanner_id, dataclasses.replace(registry[scanner_id], **changes)
    )


def _delete_gate(registry, monkeypatch):
    monkeypatch.delitem(registry, "dm18_except_returns_empty")


def _rename(registry, monkeypatch):
    scanner = registry["dm01_raw_substring"]
    monkeypatch.delitem(registry, "dm01_raw_substring")
    monkeypatch.setitem(
        registry, "dm01_raw_substring_v2", dataclasses.replace(scanner, id="dm01_raw_substring_v2")
    )


def _add_unreviewed(registry, monkeypatch):
    scanner = registry["dm31_live_state_default"]
    monkeypatch.setitem(
        registry, "dm99_unreviewed", dataclasses.replace(scanner, id="dm99_unreviewed")
    )


def _narrow_leg(registry, monkeypatch):
    scanner = registry["dm18_except_returns_empty"]
    narrowed = dataclasses.replace(scanner.legs[0], roots=("memory",))
    _swap(registry, monkeypatch, "dm18_except_returns_empty", legs=(narrowed,))


REGISTRY_DRIFT = [
    pytest.param(_delete_gate, id="gate-deleted"),
    pytest.param(_rename, id="scanner-renamed"),
    pytest.param(
        lambda r, m: _swap(r, m, "dm17_apply_without_guard", mode="report"),
        id="gate-downgraded-to-report",
    ),
    pytest.param(
        lambda r, m: _swap(r, m, "dm16_config_key_reachability", mode="gate"),
        id="report-upgraded-to-gate",
    ),
    pytest.param(
        lambda r, m: _swap(r, m, "dm31_live_state_default", class_ids=("BC-78",)),
        id="class-ids-drift",
    ),
    pytest.param(
        lambda r, m: _swap(r, m, "catalog", contract_version=99),
        id="contract-version-drift",
    ),
    pytest.param(_narrow_leg, id="leg-roots-narrowed"),
    pytest.param(_add_unreviewed, id="unreviewed-extra-scanner"),
]


@pytest.mark.parametrize("mutate", REGISTRY_DRIFT)
def test_registry_drift_exits_two_before_any_comparison(tmp_path, capsys, monkeypatch, mutate):
    build_repo(tmp_path)
    mutate(scanners_pkg.SCANNERS, monkeypatch)
    code, payload, _, _ = run_scan(capsys, tmp_path)
    assert code == 2
    assert "registry_mismatch" in failure_codes(payload)
    assert payload["registry"]["mismatches"]
    assert payload["comparison"]["compared_scanners"] == []


# ---------------------------------------------------------------------------
# Malformed policy — fail closed with exit 2, never a clean pass.
# ---------------------------------------------------------------------------


def _mutate_policy(root, mutate):
    path = root / POLICY_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _classify_twice(policy):
    policy["inventory"]["unscanned_python_roots"].append({"name": "core", "reason": "x"})


POLICY_DEFECTS = [
    pytest.param(lambda p: p.update(schema=True), id="schema-bool"),
    pytest.param(lambda p: p.update(schema="1"), id="schema-string"),
    pytest.param(lambda p: p.update(schema=2), id="schema-unsupported"),
    pytest.param(lambda p: p.update(unreviewed_key=True), id="unknown-top-level-key"),
    pytest.param(lambda p: p.pop("legs"), id="legs-missing"),
    pytest.param(lambda p: p["scanners"][0].update(mode="warn"), id="unknown-mode"),
    pytest.param(lambda p: p["scanners"].append(dict(p["scanners"][0])), id="duplicate-scanner"),
    pytest.param(lambda p: p["scanners"][0].update(class_ids=[]), id="empty-class-ids"),
    pytest.param(lambda p: p["scanners"][0].update(class_ids=["bc-1"]), id="malformed-class-id"),
    pytest.param(lambda p: p["scanners"][0].update(contract_version=True), id="version-bool"),
    pytest.param(lambda p: p["scanners"][0].update(legs=["no_such_leg"]), id="unknown-leg"),
    pytest.param(lambda p: p["legs"][0].update(roots=["/etc"]), id="absolute-root"),
    pytest.param(lambda p: p["legs"][0].update(roots=["../core"]), id="parent-root"),
    pytest.param(lambda p: p["legs"][0].update(required="yes"), id="required-not-bool"),
    pytest.param(lambda p: p["legs"][0].update(kind="glob"), id="unknown-leg-kind"),
    pytest.param(_classify_twice, id="root-classified-twice"),
    pytest.param(lambda p: p["harness"].update(expected_test_cases=0), id="empty-harness"),
]


@pytest.mark.parametrize("mutate", POLICY_DEFECTS)
def test_malformed_policy_exits_two(tmp_path, capsys, mutate):
    build_repo(tmp_path)
    _mutate_policy(tmp_path, mutate)
    code, payload, _, _ = run_scan(capsys, tmp_path)
    assert code == 2
    assert "policy_invalid" in failure_codes(payload)


def test_missing_policy_exits_two(tmp_path, capsys):
    build_repo(tmp_path)
    (tmp_path / POLICY_PATH).unlink()
    code, payload, _, _ = run_scan(capsys, tmp_path)
    assert code == 2
    assert "policy_invalid" in failure_codes(payload)


def test_unparseable_policy_exits_two(tmp_path, capsys):
    build_repo(tmp_path)
    (tmp_path / POLICY_PATH).write_text("{not json", encoding="utf-8")
    code, payload, _, _ = run_scan(capsys, tmp_path)
    assert code == 2
    assert "policy_invalid" in failure_codes(payload)
