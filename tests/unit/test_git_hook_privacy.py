"""Run the deployed privacy hook against disposable Git indexes."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


HOOK = Path(__file__).resolve().parents[2] / "hooks" / "pre-commit-privacy"


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "config").mkdir()
    (tmp_path / "example.txt").write_text("ordinary fixture\n")
    subprocess.run(["git", "add", "example.txt"], cwd=tmp_path, check=True)
    return tmp_path


def run_hook(repo, **overrides):
    env = dict(os.environ)
    env.pop("DAEMON_ALLOW_MISSING_PRIVACY_TERMS", None)
    env.pop("CI", None)
    env.update(overrides)
    return subprocess.run(["bash", str(HOOK)], cwd=repo, env=env,
                          capture_output=True, text=True)


def terms(repo, value="ExampleSecret\n"):
    (repo / "config" / "privacy_terms.local.txt").write_text(value)


@pytest.mark.parametrize("env", [{}, {"CI": "true"}, {"DAEMON_ALLOW_MISSING_PRIVACY_TERMS": "1"}])
def test_missing_terms_refuse_commit(repo, env):
    result = run_hook(repo, **env)
    assert result.returncode != 0
    assert "missing or unreadable" in result.stderr


def test_explicit_ci_opt_out_is_visible(repo):
    result = run_hook(repo, CI="true", DAEMON_ALLOW_MISSING_PRIVACY_TERMS="1")
    assert result.returncode == 0, result.stderr
    assert "explicitly unavailable in CI" in result.stdout


@pytest.mark.parametrize("content", ["", "# comments only\n  \n"])
def test_empty_effective_terms_refuse_commit(repo, content):
    terms(repo, content)
    result = run_hook(repo)
    assert result.returncode != 0
    assert "contains no terms" in result.stderr


@pytest.mark.parametrize("spelling", ["ExampleSecret", "EXAMPLESECRET", "examplesecret"])
def test_staged_match_is_case_insensitive_and_never_printed(repo, spelling):
    terms(repo)
    (repo / "example.txt").write_text(f"{spelling}\n" + "ordinary text\n" * 20000)
    subprocess.run(["git", "add", "example.txt"], cwd=repo, check=True)
    # A clean worktree must not hide a sensitive staged blob.
    (repo / "example.txt").write_text("clean now\n")
    result = run_hook(repo)
    assert result.returncode != 0
    assert "example.txt" in result.stdout
    assert spelling.lower() not in (result.stdout + result.stderr).lower()


def test_clean_index_passes_even_when_worktree_has_term(repo):
    terms(repo, "# comment\r\nExampleSecret\r\n")
    (repo / "example.txt").write_text("ExampleSecret\n")
    result = run_hook(repo)
    assert result.returncode == 0, result.stderr


def test_term_is_literal_not_a_regular_expression(repo):
    terms(repo, "name.*suffix\n")
    (repo / "example.txt").write_text("name any suffix\n")
    subprocess.run(["git", "add", "example.txt"], cwd=repo, check=True)
    result = run_hook(repo)
    assert result.returncode == 0, result.stderr


def test_paths_with_whitespace_are_scanned(repo):
    terms(repo)
    name = "notes with\nnewline.txt"
    (repo / name).write_text("EXAMPLESECRET\n")
    subprocess.run(["git", "add", "--", name], cwd=repo, check=True)
    result = run_hook(repo)
    assert result.returncode != 0
    assert "Privacy term match" in result.stdout


@pytest.mark.parametrize("tool,body,reason", [
    ("grep", "exit 2\n", "private-term scanner failed"),
    ("git", 'if [ "$1" = show ]; then exit 128; fi\nexec {git} "$@"\n', "cannot read a staged blob"),
    ("git", 'if [ "$1" = diff ]; then exit 128; fi\nexec {git} "$@"\n', "cannot enumerate staged paths"),
])
def test_scanner_and_index_errors_refuse_commit(repo, tool, body, reason):
    terms(repo)
    binaries = repo / "bin"
    binaries.mkdir()
    # Only the failure boundary is fake; the hook and Git index are real.
    real_git = shutil.which("git")
    assert real_git
    wrapper = binaries / tool
    wrapper.write_text("#!/bin/bash\n" + body.replace("{git}", real_git))
    wrapper.chmod(0o755)
    # Avoid the independent gitleaks check consuming this test's injected
    # Git failure before the private-term reader gets to exercise it.
    scanner = binaries / "gitleaks"
    scanner.write_text("#!/bin/bash\nexit 0\n")
    scanner.chmod(0o755)
    result = run_hook(repo, PATH=str(binaries) + os.pathsep + os.environ["PATH"])
    assert result.returncode != 0
    assert reason in result.stderr


# --- allow: lines — a term inside a longer PUBLIC literal (a hosting handle) ---

ALLOWING = "ExampleSecret\nallow:myexamplesecret-handle\n"


def stage(repo, text, name="example.txt"):
    (repo / name).write_text(text)
    subprocess.run(["git", "add", "--", name], cwd=repo, check=True)


@pytest.mark.parametrize("spelling", [
    "https://host.example/myexamplesecret-handle/project.git",
    "clone https://host.example/MyExampleSecret-Handle/project",
])
def test_allowed_literal_does_not_block(repo, spelling):
    terms(repo, ALLOWING)
    stage(repo, f"see {spelling}\nordinary text\n")
    result = run_hook(repo)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("body", [
    "https://host.example/myexamplesecret-handle/x and also examplesecret alone\n",
    "myexamplesecret-handleEXAMPLESECRET\n",       # glued to the allowed literal
    "myexamplesecret-handl\n",                       # a near miss of the allowed literal
])
def test_the_term_still_blocks_outside_the_allowed_literal(repo, body):
    terms(repo, ALLOWING)
    stage(repo, body)
    result = run_hook(repo)
    assert result.returncode != 0
    assert "Privacy term match" in result.stdout
    assert "examplesecret" not in (result.stdout + result.stderr).lower()


@pytest.mark.parametrize("line", ["allow:ExampleSecret", "allow:examplesecret", "allow:"])
def test_an_allow_line_cannot_switch_a_term_off(repo, line):
    terms(repo, f"ExampleSecret\n{line}\n")
    stage(repo, "ordinary text\n")
    result = run_hook(repo)
    assert result.returncode != 0
    assert "allow:" in result.stderr


def test_allow_lines_alone_are_not_a_term_list(repo):
    terms(repo, "allow:myexamplesecret-handle\n")
    result = run_hook(repo)
    assert result.returncode != 0
    assert "contains no terms" in result.stderr


def test_binary_blob_with_allow_lines_is_skipped_not_an_error(repo):
    terms(repo, ALLOWING)
    (repo / "image.bin").write_bytes(b"\x00\x01\x02ExampleSecret\x00" * 50)
    subprocess.run(["git", "add", "--", "image.bin"], cwd=repo, check=True)
    result = run_hook(repo)
    assert result.returncode == 0, result.stdout + result.stderr

