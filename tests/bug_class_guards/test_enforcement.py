"""Enforcement wiring: the hook and the workflow run the whole contract.

A ratchet that CI never runs, runs with ``--scanner``, runs with
``--write-baseline``, runs behind a path filter or under
``continue-on-error``, or whose final status can be skipped into green,
enforces nothing.  ``bug_class_guards.enforcement`` reads the literal
commands in ``hooks/pre-push`` and ``.github/workflows/tests.yml``; each
control mutates a copy of the live file and requires a named problem.  The
validator reads text only — it is deliberately not a shell or YAML engine.
"""

from __future__ import annotations

import pytest

from fixtures import REPO_ROOT

HOOK = REPO_ROOT / "hooks" / "pre-push"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tests.yml"

SCAN_HOOK = 'python scripts/check_bug_classes.py scan --root . --receipt "$receipts/bug-class-scan.json"'
SCAN_CI = "python scripts/check_bug_classes.py scan --root . --receipt receipts/bug-class-scan.json"


def _enforcement():
    from bug_class_guards import enforcement

    return enforcement


def _replace_once(text, old, new):
    assert text.count(old) == 1, f"mutation anchor not unique: {old!r}"
    return text.replace(old, new)


def _drop_line_containing(text, needle):
    lines = text.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if needle in line]
    assert len(hits) == 1, f"mutation anchor not unique: {needle!r}"
    del lines[hits[0]]
    return "".join(lines)


def test_live_hook_satisfies_the_enforcement_contract():
    assert _enforcement().hook_problems(HOOK.read_text(encoding="utf-8")) == []


def test_live_workflow_satisfies_the_enforcement_contract():
    assert _enforcement().workflow_problems(WORKFLOW.read_text(encoding="utf-8")) == []


HOOK_MUTATIONS = [
    pytest.param(lambda t: _replace_once(t, SCAN_HOOK, "true"), id="scan-omitted"),
    pytest.param(lambda t: _replace_once(t, "scan --root .", "scan --root . --scanner catalog"), id="partial-selector"),
    pytest.param(lambda t: _replace_once(t, "scan --root .", "scan --root . --write-baseline"), id="baseline-write"),
    pytest.param(lambda t: _replace_once(t, "scan --root .", "scan --root . --baseline /tmp/b.json"), id="baseline-override"),
    pytest.param(lambda t: _replace_once(t, '|| fail "bug-class scan"', "|| true"), id="scan-failure-swallowed"),
    pytest.param(lambda t: _replace_once(t, "  tests/bug_class_guards || fail", "  tests/bug_class_guards/test_policy.py || fail"), id="harness-narrowed"),
    pytest.param(lambda t: _replace_once(t, '|| fail "bug-class receipts"', "|| true"), id="receipt-check-swallowed"),
    pytest.param(lambda t: _drop_line_containing(t, "  tests/unit/test_tool_wiring_parity.py"), id="guard-dropped"),
    pytest.param(
        lambda t: _replace_once(t, '[ -f "$g" ] || fail', '[ -f "$g" ] ||'),
        id="missing-guard-not-fatal",
    ),
    pytest.param(lambda t: _replace_once(t, "set -euo pipefail", "set -uo pipefail"), id="errexit-dropped"),
]


@pytest.mark.parametrize("mutate", HOOK_MUTATIONS)
def test_hook_mutation_is_rejected(mutate):
    text = HOOK.read_text(encoding="utf-8")
    mutated = mutate(text)
    assert mutated != text
    assert _enforcement().hook_problems(mutated)


WORKFLOW_MUTATIONS = [
    pytest.param(lambda t: _replace_once(t, SCAN_CI, "true"), id="scan-omitted"),
    pytest.param(lambda t: _replace_once(t, "scan --root .", "scan --root . --scanner catalog"), id="partial-selector"),
    pytest.param(lambda t: _replace_once(t, "scan --root .", "scan --root . --write-baseline"), id="baseline-write"),
    pytest.param(lambda t: _replace_once(t, SCAN_CI, SCAN_CI + " || true"), id="scan-failure-swallowed"),
    pytest.param(
        lambda t: _replace_once(t, "      - name: Full bug-class scan", "      - name: Full bug-class scan\n        continue-on-error: true"),
        id="continue-on-error",
    ),
    pytest.param(
        lambda t: _replace_once(t, "    branches: [master, refactor/prompt-modular]\n", "    branches: [master, refactor/prompt-modular]\n    paths:\n      - 'core/**'\n"),
        id="push-paths-filter",
    ),
    pytest.param(
        lambda t: _replace_once(t, "  pull_request:\n    branches: [master]\n", "  pull_request:\n    branches: [master]\n    paths-ignore:\n      - 'docs/**'\n"),
        id="pull-request-paths-ignore",
    ),
    pytest.param(lambda t: _replace_once(t, "needs: [bug-class-scan, test, frontend]", "needs: [bug-class-scan, test]"), id="gate-needs-shortened"),
    pytest.param(lambda t: _drop_line_containing(t, "TEST_RESULT: ${{ needs.test.result }}"), id="gate-dependency-unchecked"),
    pytest.param(lambda t: _drop_line_containing(t, "    if: ${{ always() }}"), id="gate-skippable"),
    pytest.param(
        lambda t: _replace_once(t, "      - name: Isolated bug-class harness", "      - name: Isolated bug-class harness\n        if: ${{ always() }}"),
        id="always-outside-the-gate",
    ),
    pytest.param(lambda t: _replace_once(t, "    name: bug-class-gate", "    name: bug-class-gate-v2"), id="gate-renamed"),
    pytest.param(lambda t: _replace_once(t, ' --expect-sha "$GITHUB_SHA"', ""), id="receipt-commit-unchecked"),
    pytest.param(lambda t: _replace_once(t, " --harness-junit receipts/bug-class-harness.xml", ""), id="harness-count-unchecked"),
    pytest.param(lambda t: _drop_line_containing(t, "            tests/unit/test_ordered_slice_guard.py"), id="guard-step-narrowed"),
]


@pytest.mark.parametrize("mutate", WORKFLOW_MUTATIONS)
def test_workflow_mutation_is_rejected(mutate):
    text = WORKFLOW.read_text(encoding="utf-8")
    mutated = mutate(text)
    assert mutated != text
    assert _enforcement().workflow_problems(mutated)
