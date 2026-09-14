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
or run git with ``cwd``/``-C`` pointing at the project root. "Failed-before"
evidence belongs in the handoff doc as a recorded result, not as an
assertion that depends on where HEAD happens to be.

Detection (strengthened 2026-09-13). The first version matched single lines,
so a call split across lines, ``cwd=`` written before the argument list,
``-C <root>``, a module alias such as ``BASE = Path(__file__).parents[2]``,
and a git read with no ``cwd`` at all all escaped it. A git read with no
``cwd``/``-C`` runs in pytest's working directory, which is the project root.
Git invocations are now found structurally (AST) in every test file,
recursively; blob refs in strings are still matched line by line. Violations
report path, line and rule only, never the source text. Argument lists built
in a variable are out of scope.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()

# A ref-qualified blob path inside a string literal: "HEAD:memory/x.py",
# 'origin/master:utils/y.py', "abc1234:core/z.py".
_REF_BLOB_RE = re.compile(r"""["'](?:HEAD|[A-Za-z0-9_./-]+/[A-Za-z0-9_.-]+|[0-9a-f]{7,40}):[A-Za-z0-9_./-]+["']""")
# f-string form used by the offending tests: f"HEAD:{path}".
_REF_BLOB_FSTRING_RE = re.compile(r"""f["'](?:HEAD|[0-9a-f]{7,40}):\{""")

_SUBPROCESS_FUNCS = frozenset(
    {"run", "check_output", "check_call", "call", "Popen", "getoutput", "getstatusoutput", "system"}
)
# Subcommands that read the repository they run in.
_READ_SUBCOMMANDS = frozenset(
    {
        "show", "diff", "log", "rev-parse", "cat-file", "status", "ls-files", "stash",
        "checkout", "reset", "blame", "grep", "describe", "branch", "tag", "rev-list",
        "archive", "worktree", "ls-tree", "for-each-ref", "reflog", "merge-base", "shortlog",
    }
)
# git options whose NEXT argument is a value, not the subcommand.
_VALUE_OPTIONS = frozenset({"-C", "-c", "--git-dir", "--work-tree", "--namespace"})
_ROOT_ALIASES = frozenset(
    {"REPO_ROOT", "PROJECT_ROOT", "ROOT", "REPO", "BASE_DIR", "ROOT_DIR", "REPO_DIR", "PROJECT_DIR"}
)


def _label(path: Path) -> str:
    try:
        return str(path.relative_to(TESTS_ROOT.parent))
    except ValueError:  # a tmp_path fixture file in the self-tests below
        return path.name


def _is_project_root(expr: ast.AST, aliases: frozenset[str]) -> bool:
    """True for an expression that evaluates to (a path inside) the project root."""
    if isinstance(expr, ast.Name):
        return expr.id in aliases or expr.id == "__file__"
    if isinstance(expr, ast.Constant):
        return expr.value in {".", "./"}
    if isinstance(expr, ast.Attribute):
        if expr.attr == "curdir":
            return True
        return expr.attr in {"parent", "parents"} and _is_project_root(expr.value, aliases)
    if isinstance(expr, ast.Subscript):
        return _is_project_root(expr.value, aliases)
    if isinstance(expr, ast.BinOp):  # REPO_ROOT / "sub" is still inside the repo
        return _is_project_root(expr.left, aliases)
    if isinstance(expr, ast.Call):
        func = expr.func
        if isinstance(func, ast.Attribute):
            if func.attr in {"getcwd", "cwd"}:
                return True
            if func.attr in {"resolve", "absolute"}:
                return _is_project_root(func.value, aliases)
            if func.attr in {"dirname", "abspath", "realpath", "fspath", "join"}:
                return bool(expr.args) and _is_project_root(expr.args[0], aliases)
        if isinstance(func, ast.Name) and func.id in {"str", "Path", "PurePath", "fspath"}:
            return bool(expr.args) and _is_project_root(expr.args[0], aliases)
    return False


def _root_aliases(tree: ast.Module) -> frozenset[str]:
    aliases = set(_ROOT_ALIASES)
    changed = True
    while changed:  # chained aliases: HERE = Path(__file__).parent; ROOT2 = HERE.parent
        changed = False
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            if not _is_project_root(value, frozenset(aliases)):
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in aliases:
                    aliases.add(target.id)
                    changed = True
    return frozenset(aliases)


def _git_invocation(node: ast.AST):
    """(subcommand or None, explicit target expression or None) for a git subprocess call."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
    if name not in _SUBPROCESS_FUNCS:
        return None
    argv = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "args"), None)
    subcommand = target = None
    if isinstance(argv, (ast.List, ast.Tuple)):
        words = argv.elts
        if not (words and isinstance(words[0], ast.Constant) and words[0].value == "git"):
            return None
        index = 1
        while index < len(words):
            word = words[index]
            value = word.value if isinstance(word, ast.Constant) and isinstance(word.value, str) else None
            if value in _VALUE_OPTIONS:
                if value == "-C" and index + 1 < len(words):
                    target = words[index + 1]
                index += 2
                continue
            if value is not None and not value.startswith("-"):
                subcommand = value
                break
            index += 1
    else:
        head = argv
        if isinstance(argv, ast.JoinedStr) and argv.values and isinstance(argv.values[0], ast.Constant):
            head = argv.values[0]
        if not (isinstance(head, ast.Constant) and isinstance(head.value, str) and head.value.startswith("git ")):
            return None
        parts = head.value.split()
        subcommand = next((part for part in parts[1:] if not part.startswith("-")), None)
    cwd = next((k.value for k in node.keywords if k.arg == "cwd"), None)
    return subcommand, target if target is not None else cwd


def _violations(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    label = _label(path)
    out: list[str] = []
    for number, line in enumerate(text.splitlines(), 1):
        if _REF_BLOB_RE.search(line) or _REF_BLOB_FSTRING_RE.search(line):
            out.append(f"{label}:{number}: reads a blob from a project ref")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        out.append(f"{label}:{exc.lineno or 0}: cannot be parsed, so its git use cannot be checked")
        return out
    aliases = _root_aliases(tree)
    for node in ast.walk(tree):
        invocation = _git_invocation(node)
        if invocation is None:
            continue
        subcommand, target = invocation
        if target is not None and _is_project_root(target, aliases):
            out.append(f"{label}:{node.lineno}: runs git against the project repo (cwd/-C is the project root)")
        elif target is None and subcommand in _READ_SUBCOMMANDS:
            out.append(
                f"{label}:{node.lineno}: runs `git {subcommand}` with no cwd/-C, "
                "which is the project repo under pytest"
            )
    return out


def test_no_test_reads_project_git_state():
    violations: list[str] = []
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        if path.resolve() == THIS_FILE:
            continue
        violations.extend(_violations(path))
    assert not violations, (
        "Tests must not depend on the project repo's git state (green only on a dirty tree, "
        "red the moment the change is committed):\n  " + "\n  ".join(violations)
    )


def _write(tmp_path, source, name="test_x.py"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


class TestGuardRecognizesTheClass:
    """The guard must catch the exact shapes that went red on 2026-09-07."""

    def test_git_show_head_fstring(self, tmp_path):
        p = _write(tmp_path,
            'import subprocess\n'
            'def _head_source(path):\n'
            '    return subprocess.run(["git", "show", f"HEAD:{path}"], cwd=str(REPO_ROOT),\n'
            '        capture_output=True, text=True, check=True).stdout\n'
        )
        found = _violations(p)
        assert any("blob from a project ref" in v for v in found)
        assert any("against the project repo" in v for v in found)

    def test_git_show_head_literal_with_file_cwd(self, tmp_path):
        p = _write(tmp_path,
            'old_src = subprocess.run(\n'
            '    ["git", "show", "HEAD:utils/completed_plan_claims.py"],\n'
            '    cwd=str(Path(__file__).resolve().parents[2]),\n'
            '    capture_output=True, text=True, check=True,\n'
            ').stdout\n'
        )
        assert _violations(p)

    def test_tmp_repo_usage_is_allowed(self, tmp_path):
        p = _write(tmp_path,
            'def test_it(tmp_path):\n'
            '    repo = tmp_path / "r"\n'
            '    subprocess.run(["git", "init", "-q", str(repo)], check=True)\n'
            '    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True)\n'
            '    assert _classify_intent("git rev-parse HEAD^{tree}") == "identity"\n'
        )
        assert _violations(p) == []


class TestGuardSeesRewrites:
    """Spellings the single-line version missed (2026-09-13)."""

    def test_call_split_across_lines(self, tmp_path):
        p = _write(tmp_path,
            'out = subprocess.check_output(\n'
            '    [\n'
            '        "git",\n'
            '        "log",\n'
            '        "-1",\n'
            '    ],\n'
            '    text=True,\n'
            '    cwd=REPO_ROOT,\n'
            ')\n'
        )
        assert [v.split(": ", 1)[1] for v in _violations(p)] == [
            "runs git against the project repo (cwd/-C is the project root)"
        ]

    def test_cwd_keyword_written_before_the_argument_list(self, tmp_path):
        p = _write(tmp_path, 'subprocess.run(cwd=PROJECT_ROOT, args=["git", "status"])\n')
        assert _violations(p)

    def test_dash_c_pointing_at_the_project_root(self, tmp_path):
        p = _write(tmp_path, 'subprocess.run(["git", "-C", str(REPO_ROOT), "status"], check=True)\n')
        assert any("against the project repo" in v for v in _violations(p))

    def test_module_alias_of_the_project_root(self, tmp_path):
        p = _write(tmp_path,
            'from pathlib import Path\n'
            'HERE = Path(__file__).resolve().parent\n'
            'TOP = HERE.parents[1]\n'
            'def test_it():\n'
            '    subprocess.run(["git", "diff"], cwd=TOP / "sub")\n'
        )
        assert _violations(p)

    def test_cwd_from_the_working_directory(self, tmp_path):
        p = _write(tmp_path,
            'subprocess.run(["git", "log"], cwd=os.getcwd())\n'
            'subprocess.run(["git", "log"], cwd=".")\n'
        )
        assert len(_violations(p)) == 2

    def test_git_read_with_no_cwd_runs_in_the_project_root(self, tmp_path):
        p = _write(tmp_path, 'head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True)\n')
        assert any("`git rev-parse` with no cwd/-C" in v for v in _violations(p))

    def test_shell_string_git_read_with_no_cwd(self, tmp_path):
        p = _write(tmp_path, 'subprocess.run("git show --stat", shell=True)\n')
        assert _violations(p)

    def test_git_option_value_is_not_mistaken_for_the_subcommand(self, tmp_path):
        p = _write(tmp_path, 'subprocess.run(["git", "-c", "core.pager=cat", "log"])\n')
        assert any("`git log`" in v for v in _violations(p))

    def test_tmp_repo_forms_stay_allowed(self, tmp_path):
        p = _write(tmp_path,
            'def test_it(tmp_path):\n'
            '    repo = tmp_path / "r"\n'
            '    subprocess.run(["git", "status"], cwd=tmp_path)\n'
            '    subprocess.run(["git", "log"], cwd=str(repo))\n'
            '    subprocess.run(["git", "-C", repo, "diff"])\n'
            '    subprocess.run(["git", "clone", str(repo), str(tmp_path / "c")])\n'
            '    mock_run(["git", "show", "--stat"])\n'
        )
        assert _violations(p) == []

    def test_unparseable_test_file_is_a_violation(self, tmp_path):
        p = _write(tmp_path, "def broken(:\n")
        assert any("cannot be parsed" in v for v in _violations(p))

    def test_messages_report_path_line_and_rule_only(self, tmp_path):
        p = _write(tmp_path, 'SECRET_MARKER = subprocess.run(["git", "show", "HEAD:private/notes.py"], cwd=REPO_ROOT)\n')
        found = _violations(p)
        assert found
        assert all(v.startswith("test_x.py:1: ") for v in found)
        assert not any("SECRET_MARKER" in v or "private/notes.py" in v for v in found)
