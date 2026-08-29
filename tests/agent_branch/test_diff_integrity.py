# tests/agent_branch/test_diff_integrity.py
"""Red-team: the supervisor's diff computation must not let the worker's clone
execute host code or forge the diff.

The worker has full write access to its clone INCLUDING ``.git/``. Git runs
commands from repo-local config (``core.fsmonitor``, ``clean``/``smudge``
filters, ``diff.external``/textconv) and honours replace-refs that change what a
diff says. So the supervisor must NEVER run git against the worker's ``.git`` —
it reconstructs a trusted checkout and diffs that. These tests are pure (git +
filesystem, no podman) so they run everywhere.
"""

import os
import subprocess
from pathlib import Path

from agent_branch.eval_static import parse_diff
from agent_branch.provisioning import compute_trusted_diff


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True, text=True)


def _make_baseline(root: Path) -> tuple[Path, str]:
    """A trusted baseline repo with one tracked file; returns (repo, base_sha)."""
    repo = root / "baseline"
    repo.mkdir(parents=True)
    (repo / "sandbox").mkdir()
    (repo / "sandbox" / "calc.py").write_text("def add(a, b):\n    return a + b\n",
                                              encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         check=True, capture_output=True, text=True).stdout.strip()
    return repo, sha


def _make_malicious_worker_clone(baseline: Path, dest: Path, canary: Path) -> None:
    """Clone the baseline, make a real worktree edit, then weaponise the clone's
    OWN ``.git``: a ``clean`` filter wired to a worktree ``.gitattributes`` runs
    an arbitrary command the moment ``git add`` touches the file."""
    subprocess.run(["git", "clone", "-q", str(baseline), str(dest)], check=True)
    # a genuine, in-scope worktree change the supervisor should still see
    (dest / "sandbox" / "calc.py").write_text(
        "def add(a, b):\n    return a + b  # edited by worker\n", encoding="utf-8")
    # the payload: a clean filter that fires on `git add` and passes content through
    payload = dest / "payload.sh"
    payload.write_text(f"#!/bin/sh\ntouch {canary}\ncat\n", encoding="utf-8")
    payload.chmod(0o755)
    with (dest / ".git" / "config").open("a", encoding="utf-8") as fh:
        fh.write(f'\n[filter "pwn"]\n\tclean = {payload}\n\tsmudge = cat\n')
    (dest / ".gitattributes").write_text("* filter=pwn\n", encoding="utf-8")


def test_collect_diff_does_not_execute_worker_git_config(tmp_path):
    baseline, base_sha = _make_baseline(tmp_path)
    worker = tmp_path / "worker_clone"
    fixed_canary = tmp_path / "fixed_canary"
    _make_malicious_worker_clone(baseline, worker, fixed_canary)

    diff = compute_trusted_diff(
        worker_repo=worker,
        trusted_baseline_repo=baseline,
        base_sha=base_sha,
        work_dir=tmp_path / "diff_root",
    )

    # 1. the worker's .git-config payload NEVER executed on the host
    assert not fixed_canary.exists(), "worker .git clean-filter executed on host"
    # 2. the real worktree change is still captured correctly
    assert "edited by worker" in diff
    assert "sandbox/calc.py" in diff


def test_control_naive_git_add_would_have_executed_the_payload(tmp_path):
    """Control: prove the attack vector is real — the OLD approach (run ``git add``
    directly inside the worker clone) DOES execute the payload. This is what made
    the fix necessary; if this ever stops firing the regression test above is moot."""
    baseline, _ = _make_baseline(tmp_path)
    worker = tmp_path / "worker_clone"
    control_canary = tmp_path / "control_canary"
    _make_malicious_worker_clone(baseline, worker, control_canary)

    subprocess.run(["git", "-C", str(worker), "add", "-A"],
                   capture_output=True, text=True)

    assert control_canary.exists(), (
        "expected the clean-filter vector to fire under naive `git add`; "
        "the regression test's guarantee depends on this being a real threat"
    )


def test_collect_diff_excludes_worker_git_dir(tmp_path):
    """Even a benign worker: nothing from the worker's own .git (refs, objects,
    config) appears in the supervisor-computed diff — only worktree content."""
    baseline, base_sha = _make_baseline(tmp_path)
    worker = tmp_path / "worker_clone"
    subprocess.run(["git", "clone", "-q", str(baseline), str(worker)], check=True)
    (worker / "sandbox" / "new.py").write_text("X = 1\n", encoding="utf-8")
    # a worker-planted replace-ref / packed-refs must not influence the diff base
    (worker / ".git" / "packed-refs").write_text("# tampered\n", encoding="utf-8")

    diff = compute_trusted_diff(
        worker_repo=worker, trusted_baseline_repo=baseline,
        base_sha=base_sha, work_dir=tmp_path / "diff_root",
    )
    assert "sandbox/new.py" in diff and "X = 1" in diff
    assert ".git/" not in diff and "packed-refs" not in diff


def test_collect_diff_does_not_follow_worker_symlinks(tmp_path):
    """A worker symlink pointing at a host file must be diffed AS a symlink, never
    dereferenced (which would copy host file contents into the diff)."""
    baseline, base_sha = _make_baseline(tmp_path)
    secret = tmp_path / "host_secret.txt"
    secret.write_text("TOP-SECRET-HOST-CONTENT\n", encoding="utf-8")
    worker = tmp_path / "worker_clone"
    subprocess.run(["git", "clone", "-q", str(baseline), str(worker)], check=True)
    os.symlink(str(secret), str(worker / "sandbox" / "leak"))

    diff = compute_trusted_diff(
        worker_repo=worker, trusted_baseline_repo=baseline,
        base_sha=base_sha, work_dir=tmp_path / "diff_root",
    )
    assert "TOP-SECRET-HOST-CONTENT" not in diff


def test_nested_git_dir_cannot_mask_changes_as_submodule(tmp_path):
    """A worker drops a NEW directory holding its real files PLUS a nested ``.git``.
    Under a naive overlay (copy nested ``.git`` too) the trusted ``git add -A``
    records only a submodule gitlink (mode 160000, "+Subproject commit <sha>") for
    that directory — every line inside is invisible to the static gate's
    size/scope/content checks (a confirmed diff-masking evasion). The overlay must
    strip ``.git`` at all depths so the real files are diffed line-by-line."""
    baseline, base_sha = _make_baseline(tmp_path)
    worker = tmp_path / "worker_clone"
    subprocess.run(["git", "clone", "-q", str(baseline), str(worker)], check=True)
    newmod = worker / "sandbox" / "newmod"
    newmod.mkdir(parents=True)
    (newmod / "payload.py").write_text("PAYLOAD = 'x'\n" * 40, encoding="utf-8")
    # nest a real git repo inside the new dir (this is what triggers the gitlink)
    subprocess.run(["git", "init", "-q", str(newmod)], check=True)
    _git(newmod, "config", "user.email", "e@e")
    _git(newmod, "config", "user.name", "e")
    _git(newmod, "add", "-A")
    _git(newmod, "commit", "-q", "-m", "hidden")

    diff = compute_trusted_diff(
        worker_repo=worker, trusted_baseline_repo=baseline,
        base_sha=base_sha, work_dir=tmp_path / "diff_root",
    )
    # the real file content is visible — NOT collapsed to a gitlink
    assert "sandbox/newmod/payload.py" in diff
    assert "PAYLOAD = 'x'" in diff
    assert "Subproject commit" not in diff and "160000" not in diff
    # ...and the static gate actually counts those lines now
    touched, added, removed, _empty, _ws = parse_diff(diff)
    assert "sandbox/newmod/payload.py" in touched
    assert added >= 40


def test_worker_file_deletion_is_captured(tmp_path):
    """A worker deleting a tracked file must show as a deletion — the overlay's
    clear-then-copy must not silently drop the removal (which would let a worker
    delete a guard/test on its branch without the supervisor seeing it)."""
    baseline, base_sha = _make_baseline(tmp_path)
    worker = tmp_path / "worker_clone"
    subprocess.run(["git", "clone", "-q", str(baseline), str(worker)], check=True)
    (worker / "sandbox" / "calc.py").unlink()

    diff = compute_trusted_diff(
        worker_repo=worker, trusted_baseline_repo=baseline,
        base_sha=base_sha, work_dir=tmp_path / "diff_root",
    )
    assert "sandbox/calc.py" in diff
    assert "deleted file" in diff or "/dev/null" in diff


def test_diff_external_git_config_not_executed(tmp_path):
    """Defense-in-depth, distinct from the clean-filter vector: a worker ``.git``
    with ``[diff] external = <payload>`` must never execute when the supervisor
    computes the diff — it diffs a TRUSTED checkout, never the worker's ``.git``."""
    baseline, base_sha = _make_baseline(tmp_path)
    worker = tmp_path / "worker_clone"
    canary = tmp_path / "diffext_canary"
    subprocess.run(["git", "clone", "-q", str(baseline), str(worker)], check=True)
    (worker / "sandbox" / "calc.py").write_text(
        "def add(a, b):\n    return a + b  # edit\n", encoding="utf-8")
    payload = worker / "ext.sh"
    payload.write_text(f"#!/bin/sh\ntouch {canary}\n", encoding="utf-8")
    payload.chmod(0o755)
    with (worker / ".git" / "config").open("a", encoding="utf-8") as fh:
        fh.write(f"\n[diff]\n\texternal = {payload}\n")

    diff = compute_trusted_diff(
        worker_repo=worker, trusted_baseline_repo=baseline,
        base_sha=base_sha, work_dir=tmp_path / "diff_root",
    )
    assert not canary.exists(), "worker .git diff.external executed on host"
    assert "edit" in diff


def test_diff_bogus_base_sha_fails_loud(tmp_path):
    """A base_sha the baseline clone doesn't have must RAISE, never silently
    diff against the clone's HEAD or return "" (empty diff reads as "no
    changes" — a silent PASS in a review path). 2026-08-28: the checkout was
    check=False and both failure shapes were reachable."""
    import pytest
    baseline, base_sha = _make_baseline(tmp_path)
    worker = tmp_path / "worker_clone"
    subprocess.run(["git", "clone", "-q", str(baseline), str(worker)], check=True)
    (worker / "sandbox" / "calc.py").write_text(
        "def add(a, b):\n    return a + b  # edit\n", encoding="utf-8")

    with pytest.raises((subprocess.CalledProcessError, RuntimeError)):
        compute_trusted_diff(
            worker_repo=worker, trusted_baseline_repo=baseline,
            base_sha="0" * 40, work_dir=tmp_path / "diff_root",
        )
