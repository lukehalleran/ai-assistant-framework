# agent_branch/eval_sandboxed.py
"""
Layer 1b — DYNAMIC evaluation, in a disposable sandbox (never on the host).

Module Contract
- Purpose: verify the worker's change against the SUPERVISOR-owned tests without
  ever executing worker code on the trusted host. This is the second half of the
  D2 fix: even after the static gate (1a) clears a diff, the act of *running* its
  tests is itself sandboxed.
- Flow:
  1. clone the TRUSTED repo to a throwaway host dir (files only — not executed).
  2. ``git apply`` the supervisor-computed diff there (git apply does not execute
     repo content, so this is safe on the host). Apply failure → killed.
  3. copy the patched checkout into a DISPOSABLE ``--network=none`` container and
     run the trusted test command there; the host never imports/runs the code.
  4. a non-zero exit → killed.
- Inputs: trusted repo path, the diff text, the test command, image, timeout.
- Output: ``SandboxEvalResult`` (applied?, passed?, killed?, exit, output).
- Side effects: spawns a disposable podman container; writes under a temp dir
  that is removed on completion.

Scope note (M1): the default proof command runs each trusted test as a plain
``python <file>`` stdlib script, so it works in the bare worker image (which has
no pytest). The real Daemon suite needs an image carrying the project deps (M3+);
such a caller passes an explicit pytest ``test_cmd``. A branch that reaches this
layer with NO required_tests is unprovable and fails closed (killed), never a
free pass.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from agent_branch.provisioning import (
    PODMAN,
    WORKER_IMAGE,
    _run,
    ensure_worker_image,
    require_podman,
)
from utils.logging_utils import get_logger

logger = get_logger("agent_branch.eval_sandboxed")


class BranchTestEvidence(BaseModel):
    """Result of running the worker's OWN added tests — evidence, never proof.
    Informational only: it never affects the kill decision (D3)."""
    ran: bool = False
    passed: bool = False
    test_files: List[str] = Field(default_factory=list)
    output: str = ""


class SandboxEvalResult(BaseModel):
    applied: bool = False
    passed: bool = False          # PROOF result (supervisor tests) — drives the kill
    killed: bool = True
    exit_code: int = -1
    output: str = ""
    reason: str = ""
    branch_evidence: Optional[BranchTestEvidence] = None  # worker's own tests (evidence)


def default_test_cmd(required_tests: List[str]) -> List[str]:
    """Run each supervisor proof test as a plain stdlib script (module-level
    asserts) — the M1/M2 reality, runnable in the bare worker image which has NO
    pytest. (An M3 deps-image caller passes an explicit pytest ``test_cmd``.)
    Previously this defaulted to ``pytest``, so any manifest using the default got
    killed for "No module named pytest" instead of on its actual merits."""
    import shlex
    joined = " && ".join(f"python {shlex.quote(t)}" for t in required_tests)
    return ["sh", "-c", joined]


def _branch_test_cmd(test_files: List[str]) -> List[str]:
    # Each worker-added test runs as a plain stdlib script (module-level asserts);
    # all must pass for exit 0. Kept distinct from the proof command.
    import shlex
    joined = " && ".join(f"python {shlex.quote(f)}" for f in test_files)
    return ["sh", "-c", joined]


def evaluate_in_sandbox(
    trusted_repo: str | Path,
    diff_text: str,
    required_tests: List[str],
    *,
    test_cmd: Optional[List[str]] = None,
    branch_test_files: Optional[List[str]] = None,
    image: str = WORKER_IMAGE,
    wallclock_seconds: int = 120,
    container_name: str = "agentbranch-eval",
) -> SandboxEvalResult:
    """Apply ``diff_text`` to a fresh trusted checkout and run the trusted PROOF
    tests in a disposable container. ``branch_test_files`` (worker-added tests)
    are run separately as EVIDENCE — they never change the kill verdict (D3).
    Never raises on a normal test failure."""
    if not required_tests:
        # No supervisor-owned PROOF → the branch is UNPROVABLE. Worker-added tests
        # are evidence, never proof (D3), so they can't certify it either. Fail
        # closed: a branch the supervisor cannot verify must NEVER be marked passed
        # (this path previously trivially passed, letting an in-scope diff with
        # zero proof survive — and even rank on diff economy). Kill before podman.
        logger.warning("no required_tests — branch is unprovable; killing (fail-closed)")
        return SandboxEvalResult(
            applied=False, passed=False, killed=True, exit_code=0,
            reason="no supervisor proof tests — branch unprovable (fail-closed)",
        )

    require_podman()
    ensure_worker_image(image=image)
    work = Path(tempfile.mkdtemp(prefix="agentbranch-eval-"))
    checkout = work / "checkout"
    try:
        _run(["git", "clone", "--no-hardlinks", "--quiet",
              f"file://{Path(trusted_repo).resolve()}", str(checkout)], check=True)

        # apply the supervisor-computed diff (safe on host; no code execution)
        diff_file = work / "change.diff"
        diff_file.write_text(diff_text, encoding="utf-8")
        applied = _run(
            ["git", "-C", str(checkout), "apply", "--whitespace=nowarn", str(diff_file)],
            check=False,
        )
        if applied.returncode != 0:
            return SandboxEvalResult(
                applied=False, passed=False, killed=True, exit_code=applied.returncode,
                output=applied.stderr or "", reason="diff failed to apply to trusted checkout",
            )

        # PROOF: trusted tests INSIDE a disposable, no-network container
        passed, exit_code, output = True, 0, ""
        if required_tests:
            cmd = test_cmd or default_test_cmd(required_tests)
            proof = _run_in_disposable(checkout, cmd, image, wallclock_seconds, container_name)
            passed, exit_code, output = proof.exit_code == 0, proof.exit_code, proof.output

        # EVIDENCE: worker-added tests, separate container, never affects the kill
        evidence: Optional[BranchTestEvidence] = None
        if branch_test_files:
            ev = _run_in_disposable(checkout, _branch_test_cmd(branch_test_files),
                                    image, wallclock_seconds, container_name + "-ev")
            evidence = BranchTestEvidence(
                ran=True, passed=ev.exit_code == 0,
                test_files=list(branch_test_files), output=ev.output[:4000],
            )

        return SandboxEvalResult(
            applied=True, passed=passed, killed=not passed,
            exit_code=exit_code, output=output,
            reason="trusted tests passed" if passed else "trusted tests failed",
            branch_evidence=evidence,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


class _Disposable(BaseModel):
    exit_code: int = -1
    output: str = ""


def _run_in_disposable(checkout: Path, cmd: List[str], image: str,
                       timeout: int, name: str) -> _Disposable:
    _run([PODMAN, "rm", "-f", name], check=False)
    try:
        # /work exists in the worker image; /work/checkout is created by cp (its
        # parent exists), mirroring the validated repo-copy path in provisioning.
        _run([
            PODMAN, "create", "--name", name,
            "--network=none", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--pids-limit=256", "--memory=512m", "--workdir", "/work/checkout",
            image, *cmd,
        ], check=True)
        _run([PODMAN, "cp", str(checkout), f"{name}:/work/checkout"], check=True)
        import subprocess
        try:
            proc = _run([PODMAN, "start", "-a", name], timeout=timeout)
            return _Disposable(exit_code=proc.returncode,
                               output=(proc.stdout or "") + (proc.stderr or ""))
        except subprocess.TimeoutExpired:
            _run([PODMAN, "kill", name], check=False)
            return _Disposable(exit_code=124, output="sandbox eval wallclock timeout")
    finally:
        _run([PODMAN, "rm", "-f", name], check=False)
