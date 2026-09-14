"""Receipt verification: the evidence a green required check must carry.

GitHub lets successful, skipped, or neutral checks satisfy a required status,
so the final gate cannot trust a dependency's colour on its own.
``verify-receipts`` re-reads the checked-out policy, baseline and
dispositions, then rejects any scan receipt or JUnit report that is absent,
partial, for another commit, or inconsistent with them.
"""

from __future__ import annotations

import json
from xml.sax.saxutils import quoteattr

import pytest

import check_bug_classes

from fixtures import POLICY_PATH, build_repo, run_scan

SHA = "a" * 40


def _policy(root):
    return json.loads((root / POLICY_PATH).read_text(encoding="utf-8"))


def junit(cases, *, failed=(), skipped=(), errored=()):
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<testsuites>", f'<testsuite name="pytest" tests="{len(cases)}">']
    for index, (classname, name) in enumerate(cases):
        inner = ""
        if index in failed:
            inner = '<failure message="boom" />'
        elif index in skipped:
            inner = '<skipped message="skip" />'
        elif index in errored:
            inner = '<error message="err" />'
        lines.append(f"<testcase classname={quoteattr(classname)} name={quoteattr(name)} time=\"0.01\">{inner}</testcase>")
    lines += ["</testsuite>", "</testsuites>"]
    return "\n".join(lines) + "\n"


def harness_cases(count):
    return [("tests.bug_class_guards.test_synthetic", f"test_case_{index}") for index in range(count)]


def guard_cases(guards):
    return [(path[: -len(".py")].replace("/", "."), "test_guard") for path in guards]


@pytest.fixture
def receipts(tmp_path, capsys, monkeypatch):
    repo = tmp_path / "repo"
    build_repo(repo)
    monkeypatch.setenv("GITHUB_SHA", SHA)
    out = tmp_path / "receipts"
    out.mkdir()
    code, _, _, err = run_scan(capsys, repo, "--receipt", str(out / "scan.json"))
    assert code == 0, err
    policy = _policy(repo)
    (out / "harness.xml").write_text(junit(harness_cases(policy["harness"]["expected_test_cases"])), encoding="utf-8")
    (out / "guards.xml").write_text(junit(guard_cases(policy["repo_wide_guards"])), encoding="utf-8")
    return repo, out


def verify(capsys, repo, out, *, scan="scan.json", harness="harness.xml", guards="guards.xml", sha=SHA):
    args = ["verify-receipts", "--root", str(repo)]
    if scan:
        args += ["--scan", str(out / scan)]
    if harness:
        args += ["--harness-junit", str(out / harness)]
    if guards:
        args += ["--guards-junit", str(out / guards)]
    if sha:
        args += ["--expect-sha", sha]
    code = check_bug_classes.main(args)
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def _edit_scan(out, mutate):
    path = out / "scan.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_complete_receipts_verify(receipts, capsys):
    repo, out = receipts
    code, text = verify(capsys, repo, out)
    assert code == 0, text


def test_absent_scan_receipt_is_rejected(receipts, capsys):
    repo, out = receipts
    (out / "scan.json").unlink()
    code, text = verify(capsys, repo, out)
    assert code == 1 and "scan receipt" in text


def test_receipt_for_another_commit_is_rejected(receipts, capsys):
    repo, out = receipts
    code, text = verify(capsys, repo, out, sha="b" * 40)
    assert code == 1 and "commit" in text


def test_receipt_without_a_commit_is_rejected_when_one_is_expected(receipts, capsys):
    repo, out = receipts
    _edit_scan(out, lambda p: p.update(source_sha=None))
    code, text = verify(capsys, repo, out)
    assert code == 1 and "commit" in text


@pytest.mark.parametrize(
    "mutate, needle",
    [
        pytest.param(lambda p: p.update(exit=1), "exit", id="failed-scan"),
        pytest.param(lambda p: p.update(scope="partial"), "scope", id="partial-scope"),
        pytest.param(lambda p: p["failures"].append({"code": "leg_missing", "detail": "x"}), "failure", id="recorded-failure"),
        pytest.param(
            lambda p: next(e for e in p["registry"]["observed"] if e["id"] == "dm01_raw_substring").update(mode="report"),
            "dm01_raw_substring",
            id="tampered-mode",
        ),
        pytest.param(
            lambda p: p["registry"]["observed"].pop(),
            "registry",
            id="scanner-missing-from-receipt",
        ),
        pytest.param(
            lambda p: p["scanners"]["dm17_apply_without_guard"]["legs"].pop(),
            "dm17_tests",
            id="leg-receipt-missing",
        ),
        pytest.param(
            lambda p: p["scanners"]["dm18_except_returns_empty"]["legs"][0].update(available=False, status="missing"),
            "dm18_retrieval",
            id="required-leg-unavailable",
        ),
        pytest.param(lambda p: p["policy"].update(sha256="0" * 64), "policy", id="another-policy"),
        pytest.param(lambda p: p["baseline"].update(occurrences=5), "baseline", id="baseline-count"),
        pytest.param(lambda p: p["dispositions"].update(active=5), "dispositions", id="disposition-count"),
    ],
)
def test_inconsistent_scan_receipt_is_rejected(receipts, capsys, mutate, needle):
    repo, out = receipts
    _edit_scan(out, mutate)
    code, text = verify(capsys, repo, out)
    assert code == 1
    assert needle in text


def test_harness_count_below_the_expectation_is_rejected(receipts, capsys):
    repo, out = receipts
    expected = _policy(repo)["harness"]["expected_test_cases"]
    (out / "harness.xml").write_text(junit(harness_cases(expected - 1)), encoding="utf-8")
    code, text = verify(capsys, repo, out)
    assert code == 1 and "expected" in text


def test_harness_count_above_the_expectation_is_rejected(receipts, capsys):
    repo, out = receipts
    expected = _policy(repo)["harness"]["expected_test_cases"]
    (out / "harness.xml").write_text(junit(harness_cases(expected + 1)), encoding="utf-8")
    code, text = verify(capsys, repo, out)
    assert code == 1 and "expected" in text


def test_empty_harness_collection_is_rejected(receipts, capsys):
    repo, out = receipts
    (out / "harness.xml").write_text(junit([]), encoding="utf-8")
    code, _ = verify(capsys, repo, out)
    assert code == 1


@pytest.mark.parametrize("flag", ["failed", "skipped", "errored"])
def test_non_passing_harness_case_is_rejected(receipts, capsys, flag):
    repo, out = receipts
    expected = _policy(repo)["harness"]["expected_test_cases"]
    (out / "harness.xml").write_text(junit(harness_cases(expected), **{flag: (0,)}), encoding="utf-8")
    code, _ = verify(capsys, repo, out)
    assert code == 1


def test_missing_repo_wide_guard_module_is_rejected(receipts, capsys):
    repo, out = receipts
    guards = _policy(repo)["repo_wide_guards"]
    (out / "guards.xml").write_text(junit(guard_cases(guards[:-1])), encoding="utf-8")
    code, text = verify(capsys, repo, out)
    assert code == 1 and "test_model_capability_wiring" in text


def test_unreadable_junit_is_rejected(receipts, capsys):
    repo, out = receipts
    (out / "harness.xml").write_text("<testsuites", encoding="utf-8")
    code, _ = verify(capsys, repo, out)
    assert code == 1


def test_verify_without_any_receipt_is_a_usage_error(receipts, capsys):
    repo, out = receipts
    code, _ = verify(capsys, repo, out, scan=None, harness=None, guards=None, sha=None)
    assert code == 2
