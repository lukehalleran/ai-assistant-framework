"""Repo-wide guard (2026-09-07): tests must never read the PROJECT repo's git state.

CI run 34146393709 went red on four tests that proved their "failed-before"
evidence by running ``git show HEAD:<file>`` and asserting the old code was
missing a symbol. They were green only while the change was UNCOMMITTED —
the commit that shipped the change moved HEAD onto the new code and the
tests failed forever after, on the runner and locally alike. The full local
suite had run before the commit, so nothing could catch it.

Rule: a test may drive git against a throwaway repository it creates under
``tmp_path`` (``tests/agent_branch`` does), and it may mock ``subprocess``.
It may NOT read a blob from a ref of the project repository (``HEAD:path``)
or run git with ``cwd`` pointing at the project root. "Failed-before"
evidence belongs in the handoff doc as a recorded result, not as an
assertion that depends on where HEAD happens to be.
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]

# A ref-qualified blob path inside a string literal: "HEAD:memory/x.py",
# 'origin/master:utils/y.py', "abc1234:core/z.py".
_REF_BLOB_RE = re.compile(r"""["'](?:HEAD|[A-Za-z0-9_./-]+/[A-Za-z0-9_.-]+|[0-9a-f]{7,40}):[A-Za-z0-9_./-]+["']""")
# f-string form used by the offending tests: f"HEAD:{path}".
_REF_BLOB_FSTRING_RE = re.compile(r"""f["'](?:HEAD|[0-9a-f]{7,40}):\{""")
# A git invocation (list or shell form).
_GIT_CALL_RE = re.compile(r"""(?:\[\s*["']git["']|["']git\s+(?:show|diff|log|rev-parse|cat-file|status|ls-files|stash|checkout|reset)\b)""")
# cwd derived from the test file's own location or a repo-root constant.
_PROJECT_CWD_RE = re.compile(r"""cwd\s*=\s*(?:str\()?\s*(?:REPO_ROOT|PROJECT_ROOT|ROOT|Path\(__file__\)|os\.path\.dirname\(__file__\))""")

_WINDOW = 6  # lines after a git call in which a cwd= kwarg counts


def _label(path: Path) -> str:
    try:
        return str(path.relative_to(TESTS_ROOT.parent))
    except ValueError:  # a tmp_path fixture file in the self-tests below
        return path.name


def _violations(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str] = []
    for i, line in enumerate(lines, 1):
        if _REF_BLOB_RE.search(line) or _REF_BLOB_FSTRING_RE.search(line):
            out.append(f"{_label(path)}:{i}: reads a blob from a project ref: {line.strip()}")
        if _GIT_CALL_RE.search(line):
            window = "\n".join(lines[i - 1 : i - 1 + _WINDOW])
            if _PROJECT_CWD_RE.search(window):
                out.append(f"{_label(path)}:{i}: runs git against the project repo (cwd=<repo root>)")
    return out


def test_no_test_reads_project_git_state():
    violations: list[str] = []
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        if path.name == Path(__file__).name:
            continue
        violations.extend(_violations(path))
    assert not violations, (
        "Tests must not depend on the project repo's git state (green only on a dirty tree, "
        "red the moment the change is committed):\n  " + "\n  ".join(violations)
    )


class TestGuardRecognizesTheClass:
    """The guard must catch the exact shapes that went red on 2026-09-07."""

    def test_git_show_head_fstring(self, tmp_path):
        p = tmp_path / "test_x.py"
        p.write_text(
            'import subprocess\n'
            'def _head_source(path):\n'
            '    return subprocess.run(["git", "show", f"HEAD:{path}"], cwd=str(REPO_ROOT),\n'
            '        capture_output=True, text=True, check=True).stdout\n'
        )
        found = _violations(p)
        assert any("blob from a project ref" in v for v in found)
        assert any("against the project repo" in v for v in found)

    def test_git_show_head_literal_with_file_cwd(self, tmp_path):
        p = tmp_path / "test_y.py"
        p.write_text(
            'old_src = subprocess.run(\n'
            '    ["git", "show", "HEAD:utils/completed_plan_claims.py"],\n'
            '    cwd=str(Path(__file__).resolve().parents[2]),\n'
            '    capture_output=True, text=True, check=True,\n'
            ').stdout\n'
        )
        assert _violations(p)

    def test_tmp_repo_usage_is_allowed(self, tmp_path):
        p = tmp_path / "test_z.py"
        p.write_text(
            'def test_it(tmp_path):\n'
            '    repo = tmp_path / "r"\n'
            '    subprocess.run(["git", "init", "-q", str(repo)], check=True)\n'
            '    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True)\n'
            '    assert _classify_intent("git rev-parse HEAD^{tree}") == "identity"\n'
        )
        assert _violations(p) == []
