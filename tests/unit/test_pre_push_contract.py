"""hooks/pre-push contract tests (2026-09-21, PLAN_H2_prepush).

Every test here drives the hook entirely through ``PREPUSH_CHECK=1``, a
dry-run mode added specifically so this suite can exercise the hook's real
selection/probe logic WITHOUT ever running ruff, the bug-class scan, or any
pytest pass — the one rule that matters for this whole plan is that
``hooks/pre-push`` is never executed end to end anywhere, including here: a
live Daemon runs on this machine, and the hook's non-unit pass hard-reset
the box once already when it ran beside one (2026-09-14,
docs/DEVELOPMENT_WORKFLOW.md §3). ``run_hook()`` below asserts
``PREPUSH_CHECK=1`` is set on every call as a structural guard against that.

Each test builds its own throwaway git repository under ``tmp_path``,
copies THIS checkout's ``hooks/pre-push``, ``hooks/pre_push_support.py``,
``hooks/pre-commit-privacy``, ``utils/daemon_guard.py`` and
``utils/__init__.py`` into it, and invokes the hook as a subprocess.

Background: Codex's read-only review at
``~/daemon_exec/followups_0921_runs/codex_pickup/codex_0921_hook_review.md``
and the frontier plan at
``~/daemon_exec/followups_0921_runs/plans/PLAN_H2_prepush.md``.
class: BC-83 (probe resolves live code through the inherited environment),
BC-63 (the check ran, nothing was checked), BC-13.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK = REPO_ROOT / "hooks" / "pre-push"
SUPPORT = REPO_ROOT / "hooks" / "pre_push_support.py"
PRE_COMMIT_PRIVACY = REPO_ROOT / "hooks" / "pre-commit-privacy"
DAEMON_GUARD = REPO_ROOT / "utils" / "daemon_guard.py"
UTILS_INIT = REPO_ROOT / "utils" / "__init__.py"

BASH = "/usr/bin/bash"
STUB_MAIN = "import time\ntime.sleep(30)\n"
ZERO = "0" * 40
DEFAULT_SHA = "a" * 40

# The hook's `guards=(...)` array is checked for FILE EXISTENCE only
# (`[ -f "$g" ] || fail ...`); check mode never executes pytest, so these
# only need to exist, not do anything.
GUARD_FILES = (
    "test_no_git_state_in_tests.py",
    "test_ordered_slice_guard.py",
    "test_budget_meters_rendered_sections.py",
    "test_tool_wiring_parity.py",
    "test_model_capability_wiring.py",
)


# --------------------------------------------------------------- helpers --


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60
    )
    if check and result.returncode != 0:
        raise AssertionError(f"git {args} failed: {result.stderr}")
    return result


def _git_hooks_dir(repo: Path) -> Path:
    raw = _git(repo, "rev-parse", "--git-path", "hooks").stdout.strip()
    path = Path(raw)
    return path if path.is_absolute() else repo / path


def build_repo(tmp_path: Path, name: str = "repo") -> Path:
    """A throwaway git repo carrying THIS checkout's hook code + a stub main.py.

    Mirrors the plan's fixture: ``git init``, copy ``hooks/`` +
    ``utils/daemon_guard.py`` (+ ``utils/__init__.py``) + a stub ``main.py``,
    commit, install an executable ``.git/hooks/pre-commit``.
    """
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")

    (repo / "hooks").mkdir()
    shutil.copy2(HOOK, repo / "hooks" / "pre-push")
    shutil.copy2(SUPPORT, repo / "hooks" / "pre_push_support.py")
    shutil.copy2(PRE_COMMIT_PRIVACY, repo / "hooks" / "pre-commit-privacy")

    (repo / "utils").mkdir()
    shutil.copy2(DAEMON_GUARD, repo / "utils" / "daemon_guard.py")
    shutil.copy2(UTILS_INIT, repo / "utils" / "__init__.py")

    (repo / "main.py").write_text(STUB_MAIN, encoding="utf-8")

    guard_dir = repo / "tests" / "unit"
    guard_dir.mkdir(parents=True)
    for guard_name in GUARD_FILES:
        (guard_dir / guard_name).write_text(
            '"""Stub guard placeholder (existence only; never executed here)."""\n',
            encoding="utf-8",
        )

    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")

    hooks_dir = _git_hooks_dir(repo)
    hooks_dir.mkdir(parents=True, exist_ok=True)
    dest = hooks_dir / "pre-commit"
    shutil.copy2(PRE_COMMIT_PRIVACY, dest)
    dest.chmod(0o755)
    return repo


def _active_pyenv_version() -> str | None:
    """The pyenv version active for this checkout; None on a host without pyenv."""
    if not shutil.which("pyenv"):
        return None
    out = subprocess.run(["pyenv", "version-name"], cwd=str(REPO_ROOT),
                         capture_output=True, text=True, timeout=30)
    return out.stdout.strip() or None


def _stub_tools_dir() -> Path | None:
    """The hook refuses to start without systemd-run (it will not run uncapped).
    Check mode never EXECUTES it, so on a host that lacks it (a container
    runner) a stub on PATH keeps these tests about the hook, not the host."""
    if shutil.which("systemd-run"):
        return None
    import tempfile
    directory = Path(tempfile.mkdtemp(prefix="prepush-stub-tools-"))
    stub = directory / "systemd-run"
    stub.write_text("#!/bin/bash\nexit 0\n")
    stub.chmod(0o755)
    return directory


_ACTIVE_PYENV_VERSION = _active_pyenv_version()
_STUB_TOOLS_DIR = _stub_tools_dir()


def base_env(env_extra: dict | None = None, *, drop_pythonpath: bool = True) -> dict:
    env = dict(os.environ)
    if drop_pythonpath:
        env.pop("PYTHONPATH", None)
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_CEILING_DIRECTORIES"):
        env.pop(key, None)
    env["PREPUSH_CHECK"] = "1"
    if _STUB_TOOLS_DIR is not None:
        env["PATH"] = str(_STUB_TOOLS_DIR) + os.pathsep + env.get("PATH", "")
    # pytest's tmp_path lives under /tmp, outside ~/.python-version's scope,
    # so the bare `python` the hook resolves via PATH would otherwise fall
    # back to pyenv's "system" interpreter (3.13) instead of 3.11.8 — a test
    # environment artifact (production pushes always run from a checkout
    # under $HOME, where .python-version applies) that has nothing to do
    # with the hook's own logic. PYENV_VERSION pins the shim explicitly.
    # The version is whatever is active for THIS checkout, never a literal.
    if _ACTIVE_PYENV_VERSION:
        env.setdefault("PYENV_VERSION", _ACTIVE_PYENV_VERSION)
    if env_extra:
        env.update(env_extra)
    return env


def run_hook(
    repo: Path,
    *,
    env_extra: dict | None = None,
    stdin_text: str | None = None,
    timeout: int = 60,
) -> subprocess.CompletedProcess:
    """Invoke ``hooks/pre-push`` directly (never via ``git push``).

    Always PREPUSH_CHECK=1 — see the module docstring. The assertion below
    is a structural safeguard, not just documentation: it makes it
    impossible for a future edit to this helper to accidentally drop the
    dry-run flag and run the hook for real.
    """
    env = base_env(env_extra)
    assert env.get("PREPUSH_CHECK") == "1", "refusing to run the hook without PREPUSH_CHECK=1"
    if stdin_text is None:
        stdin_text = f"refs/heads/x {DEFAULT_SHA} refs/heads/x {ZERO}\n"
    return subprocess.run(
        [BASH, str(repo / "hooks" / "pre-push")],
        cwd=str(repo),
        input=stdin_text,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


def run_support(
    repo: Path, *args: str, env_extra: dict | None = None, timeout: int = 60
) -> subprocess.CompletedProcess:
    """Invoke ``hooks/pre_push_support.py`` directly, the same way the hook does."""
    env = base_env(env_extra)
    env.pop("PREPUSH_CHECK", None)
    return subprocess.run(
        ["env", "-u", "PYTHONPATH", "python", "-s", str(repo / "hooks" / "pre_push_support.py"), *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


def parse_report(stdout: str) -> dict[str, list[str]]:
    """The check-mode report as ``{key: [values...]}`` (some keys repeat)."""
    report: dict[str, list[str]] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        report.setdefault(key, []).append(value)
    return report


def spawn_stub(cwd: Path) -> subprocess.Popen:
    """A real, relatively-launched ``python main.py`` process for the daemon probe."""
    proc = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=str(cwd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.3)  # let it start before any probe reads /proc/<pid>
    return proc


def _fakebin_without_systemd_run(tmp_path: Path) -> Path:
    """A minimal PATH dir with everything the hook needs EXCEPT systemd-run."""
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    for tool in ("git", "bash", "env", "pgrep", "basename", "wc", "mktemp"):
        found = shutil.which(tool)
        assert found, f"{tool} not found on the test runner's PATH"
        (fakebin / tool).symlink_to(found)
    interpreter = subprocess.run(
        ["env", "-u", "PYTHONPATH", "python", "-s", "-c", "import sys; print(sys.executable)"],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    assert interpreter, "could not resolve the real python interpreter for the fakebin"
    (fakebin / "python").symlink_to(interpreter)
    return fakebin


# -------------------------------------------------------------------- T1 --


def test_t1_poisoned_pythonpath_resolves_utils_inside_the_throwaway_repo(tmp_path):
    """D1: `env -u PYTHONPATH` must neutralize an ambient poisoned PYTHONPATH."""
    repo = build_repo(tmp_path)
    poison = tmp_path / "poison"
    (poison / "utils").mkdir(parents=True)
    (poison / "utils" / "__init__.py").write_text("POISON = True\n", encoding="utf-8")

    result = run_hook(repo, env_extra={"PYTHONPATH": str(poison)})

    assert result.returncode == 0, result.stderr
    report = parse_report(result.stdout)
    assert report["utils"] == [str(repo / "utils" / "__init__.py")]


# -------------------------------------------------------------------- T2 --


def test_t2_missing_pre_commit_hook_blocks(tmp_path):
    repo = build_repo(tmp_path)
    (_git_hooks_dir(repo) / "pre-commit").unlink()

    result = run_hook(repo)

    assert result.returncode != 0
    assert "pre-commit" in result.stderr


def test_t2_non_executable_pre_commit_hook_blocks(tmp_path):
    repo = build_repo(tmp_path)
    (_git_hooks_dir(repo) / "pre-commit").chmod(0o644)

    result = run_hook(repo)

    assert result.returncode != 0
    assert "pre-commit" in result.stderr


def test_t2_installed_pre_commit_hook_passes_that_gate(tmp_path):
    repo = build_repo(tmp_path)

    result = run_hook(repo)

    assert result.returncode == 0, result.stderr


# -------------------------------------------------------------------- T3 --


def test_t3_lowered_soft_nofile_is_raised_to_the_hard_limit(tmp_path):
    repo = build_repo(tmp_path)
    env = base_env()
    stdin_text = f"refs/heads/x {DEFAULT_SHA} refs/heads/x {ZERO}\n"
    hard = subprocess.run(
        [BASH, "-c", "ulimit -Hn"], capture_output=True, text=True, timeout=30
    ).stdout.strip()
    assert hard, "could not read this shell's hard nofile limit"

    cmd = f'ulimit -Sn 256; exec "{BASH}" "{repo}/hooks/pre-push"'
    result = subprocess.run(
        [BASH, "-c", cmd],
        cwd=str(repo),
        input=stdin_text,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    report = parse_report(result.stdout)
    assert report["nofile"] == [hard]


# -------------------------------------------------------------------- T4 --


def test_t4_docs_execution_scratch_tolerated(tmp_path):
    repo = build_repo(tmp_path)
    scratch = repo / "docs" / "execution" / "x"
    scratch.mkdir(parents=True)
    (scratch / "probe.py").write_text("# scratch\n", encoding="utf-8")

    result = run_hook(repo)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "relpath",
    ["tests/unit/test_new.py", "core/new.py", "docs/other/x.py"],
)
def test_t4_untracked_py_outside_scratch_blocks(tmp_path, relpath):
    repo = build_repo(tmp_path)
    target = repo / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("X = 1\n", encoding="utf-8")

    result = run_hook(repo)

    assert result.returncode != 0
    assert "untracked .py" in result.stderr


def test_t4_untracked_py_path_with_a_space_blocks(tmp_path):
    repo = build_repo(tmp_path)
    (repo / "core").mkdir()
    (repo / "core" / "new file.py").write_text("X = 1\n", encoding="utf-8")

    result = run_hook(repo)

    assert result.returncode != 0
    assert "untracked .py" in result.stderr


def test_t4_untracked_py_path_with_a_newline_blocks(tmp_path):
    repo = build_repo(tmp_path)
    (repo / "core").mkdir()
    (repo / "core" / "evil\nfile.py").write_bytes(b"X = 1\n")

    result = run_hook(repo)

    assert result.returncode != 0
    assert "untracked .py" in result.stderr


def test_t4_docs_execution_bypass_via_newline_still_blocks(tmp_path):
    """A "docs/execution/../../x.py"-style string glob-matches "docs/execution/*"
    as plain text; the newline/".." reject must run BEFORE that exception."""
    repo = build_repo(tmp_path)
    scratch = repo / "docs" / "execution"
    scratch.mkdir(parents=True)
    (scratch / "evil\nfile.py").write_bytes(b"X = 1\n")

    result = run_hook(repo)

    assert result.returncode != 0
    assert "untracked .py" in result.stderr


def test_t4_modified_tracked_file_blocks(tmp_path):
    repo = build_repo(tmp_path)
    (repo / "main.py").write_text("import time\ntime.sleep(60)\n", encoding="utf-8")

    result = run_hook(repo)

    assert result.returncode != 0
    assert "modified tracked files" in result.stderr


# -------------------------------------------------------------------- T5 --


def test_t5_pass1_pass3_selection_split_by_lane(tmp_path):
    repo = build_repo(tmp_path)
    (repo / "tests" / "test_b.py").write_text("def test_b():\n    pass\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "baseline test_b")
    base_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    (repo / "tests" / "unit" / "test_a.py").write_text("def test_a():\n    pass\n", encoding="utf-8")
    (repo / "tests" / "test_b.py").write_text("def test_b():\n    assert True\n", encoding="utf-8")
    (repo / "core").mkdir(exist_ok=True)
    (repo / "core" / "b.py").write_text("X = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "push commit")
    local_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    stdin_text = f"refs/heads/x {local_sha} refs/heads/x {base_sha}\n"
    result = run_hook(repo, stdin_text=stdin_text)

    assert result.returncode == 0, result.stderr
    report = parse_report(result.stdout)
    assert "tests/unit/test_a.py" in report["pass1"]
    assert "tests/test_b.py" not in report["pass1"]
    assert all(p.startswith("tests/unit/") for p in report["pass1"])
    assert report["pass3_changed"] == ["tests/test_b.py"]


# -------------------------------------------------------------------- T6 --


def test_t6a_live_daemon_in_the_resolved_root_reports_up(tmp_path):
    live_root = build_repo(tmp_path, name="live")
    clone_root = build_repo(tmp_path, name="clone")
    _git(clone_root, "remote", "add", "origin", str(live_root))

    proc = spawn_stub(live_root)
    try:
        direct = run_support(clone_root, "daemon-state", "--repo-root", ".")
        assert direct.returncode == 0, direct.stderr
        assert direct.stdout.startswith("up "), direct.stdout

        result = run_hook(clone_root)
        assert result.returncode == 0, result.stderr
        report = parse_report(result.stdout)
        assert report["daemon"] == ["up"]
        assert report["pass3"][0].startswith("skip:")
    finally:
        proc.kill()
        proc.wait(timeout=60)


def test_t6b_daemon_in_an_unrelated_cwd_reports_down(tmp_path):
    live_root = build_repo(tmp_path, name="live")
    clone_root = build_repo(tmp_path, name="clone")
    _git(clone_root, "remote", "add", "origin", str(live_root))
    other = tmp_path / "other"
    other.mkdir()
    (other / "main.py").write_text(STUB_MAIN, encoding="utf-8")

    proc = spawn_stub(other)
    try:
        direct = run_support(clone_root, "daemon-state", "--repo-root", ".")
        assert direct.returncode == 0, direct.stderr
        assert direct.stdout.startswith("down "), direct.stdout
    finally:
        proc.kill()
        proc.wait(timeout=60)


def test_t6c_standalone_checkout_with_no_live_root_env_is_unknown(tmp_path):
    repo = build_repo(tmp_path)

    direct = run_support(repo, "daemon-state", "--repo-root", ".")
    assert direct.returncode == 2
    assert direct.stdout.strip() == ""

    result = run_hook(repo)
    assert result.returncode == 0, result.stderr
    report = parse_report(result.stdout)
    assert report["daemon"] == ["unknown"]
    assert report["pass3"][0].startswith("skip:")


def test_t6d_live_root_env_pointing_at_a_non_checkout_is_unknown(tmp_path):
    repo = build_repo(tmp_path)
    bogus = tmp_path / "bogus"
    bogus.mkdir()

    direct = run_support(
        repo,
        "daemon-state",
        "--repo-root",
        ".",
        env_extra={"DAEMON_LIVE_REPO_ROOT": str(bogus)},
    )

    assert direct.returncode == 2


# -------------------------------------------------------------------- T7 --


def test_t7_missing_systemd_run_blocks_before_any_pass(tmp_path):
    repo = build_repo(tmp_path)
    fakebin = _fakebin_without_systemd_run(tmp_path)
    assert shutil.which("systemd-run", path=str(fakebin)) is None

    result = run_hook(repo, env_extra={"PATH": str(fakebin)})

    assert result.returncode != 0
    assert "refusing to run uncapped" in result.stderr


def test_t7_systemd_run_present_reports_capped_yes(tmp_path):
    repo = build_repo(tmp_path)

    result = run_hook(repo)

    assert result.returncode == 0, result.stderr
    report = parse_report(result.stdout)
    assert report["capped"] == ["yes"]


# -------------------------------------------------------------------- T8 --


def test_t8_no_bare_python_rm_or_mv_call_sites():
    """Static check: every Python invocation drops PYTHONPATH; `rm` only
    through `command -p`; no `mv` at all. Comments are excluded — this is a
    check on code, not prose."""
    text = HOOK.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("#")
    )

    bare_python = []
    for m in re.finditer(r"\bpython\b", code):
        start, end = m.start(), m.end()
        if code[max(0, start - 11):start] == "PYTHONPATH ":
            continue
        if start > 0 and code[start - 1] == '"' and code[end:end + 1] == "=":
            continue  # the check-mode report's literal `python=` key
        bare_python.append(code[max(0, start - 20):end + 10])
    assert not bare_python, f"bare python call site(s): {bare_python}"

    bare_rm = []
    for m in re.finditer(r"\brm\b", code):
        start = m.start()
        if code[max(0, start - 11):start] == "command -p ":
            continue
        bare_rm.append(code[max(0, start - 20):start + 10])
    assert not bare_rm, f"bare rm call site(s): {bare_rm}"

    assert not re.search(r"\bmv\b", code), "bare mv call site found"


def test_t8_no_masked_failures_or_forbidden_scan_flags():
    """Sanity companion to T8: the enforcement contract's own MASKS check
    (scripts/bug_class_guards/enforcement.py) must find nothing either, so a
    worktree-cleanup `|| true`/`|| :` never sneaks back in."""
    text = HOOK.read_text(encoding="utf-8")
    for masked in ("|| true", "|| :", "set +e"):
        assert masked not in text, f"masked failure pattern found: {masked!r}"


# ------------------------------------------------ referee additions (R1-R3) --
# Found by the frontier referee reading the rewritten hook, 2026-09-21.


def _overlay(tmp_path: Path, scripts: dict[str, str], *, real_python: bool = False) -> str:
    """A PATH whose first directory holds fake tools; everything else is real."""
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    for name, body in scripts.items():
        tool = overlay / name
        tool.write_text("#!/bin/bash\n" + body)
        tool.chmod(0o755)
    if real_python:
        interpreter = subprocess.run(
            ["env", "-u", "PYTHONPATH", "python", "-s", "-c", "import sys; print(sys.executable)"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        assert interpreter
        (overlay / "python").symlink_to(interpreter)
    rest = os.environ["PATH"]
    if _STUB_TOOLS_DIR is not None:          # a host without systemd-run (see _stub_tools_dir)
        rest = str(_STUB_TOOLS_DIR) + os.pathsep + rest
    return str(overlay) + os.pathsep + rest


def test_r1_a_failing_git_status_blocks_instead_of_reading_as_clean(tmp_path):
    """Read through a process substitution, a failed `git status` produced no
    records and the dirty-tree check passed."""
    repo = build_repo(tmp_path)
    (repo / "main.py").write_text("# modified, uncommitted\n")   # the tree IS dirty
    real_git = shutil.which("git")
    path = _overlay(tmp_path, {"git": f'if [ "$1" = status ]; then exit 128; fi\nexec {real_git} "$@"\n'})

    result = run_hook(repo, env_extra={"PATH": path})

    assert result.returncode != 0
    assert "cannot read the working-tree status" in result.stderr


def test_r2_the_verified_interpreter_is_pinned_for_directories_outside_the_repo(tmp_path):
    """pyenv resolves `python` from the current directory's .python-version
    chain; the pass-3 worktree is created under mktemp's directory, outside
    it. The hook exports the active version once it has verified it."""
    repo = build_repo(tmp_path)
    path = _overlay(tmp_path, {"pyenv": 'echo "7.7.7-pinned"\n'}, real_python=True)
    env = {"PATH": path}

    result = run_hook(repo, env_extra=env)

    assert result.returncode == 0, result.stderr
    assert parse_report(result.stdout)["pyenv_version"] == ["7.7.7-pinned"]


def test_r2_a_pyenv_that_cannot_answer_blocks(tmp_path):
    repo = build_repo(tmp_path)
    path = _overlay(tmp_path, {"pyenv": "exit 1\n"}, real_python=True)

    result = run_hook(repo, env_extra={"PATH": path})

    assert result.returncode != 0
    assert "cannot read the active pyenv version" in result.stderr


def test_r3_pass_3_is_gated_by_a_fresh_probe_and_a_worktree_interpreter_check():
    """Static, because pass 3 must never run in a test: between the pass-3
    header and `git worktree add` the Daemon is probed AGAIN (the first probe
    is minutes old after the unit passes), and between the worktree and the
    8G run the worktree's own interpreter is verified."""
    text = (REPO_ROOT / "hooks" / "pre-push").read_text(encoding="utf-8")
    header = text.index("# ---- non-unit remainder")
    add = text.index("git worktree add", header)
    run = text.index("MemoryMax=8G", add)
    assert "pre_push_support.py daemon-state" in text[header:add]
    assert 'if [ "$daemon_state" = "down" ]; then' in text[header:add]
    assert "wt_pyver" in text[add:run] and "refusing to run the non-unit pass" in text[add:run]
    assert text.count("pre_push_support.py daemon-state") == 2

