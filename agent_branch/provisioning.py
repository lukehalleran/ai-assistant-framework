# agent_branch/provisioning.py
"""
WorkerSandbox — rootless-podman provisioning, the real isolation boundary.

Module Contract
- Purpose: provision a single isolated worker, run it, and let the SUPERVISOR
  read results back without that read becoming an escape hatch.
- Isolation model (validated on this host; see plan):
  - rootless podman, base image ``python:3.11-slim`` (present offline).
  - ``--network=none`` → no IP egress at all (the attacker's zero-egress proof).
    LLM mode keeps ``--network=none`` and adds ONE bind-mounted Unix-domain
    socket to the supervisor's allowlist proxy (a UDS is not IP networking).
  - hardening: ``--cap-drop=ALL --security-opt=no-new-privileges --pids-limit
    --memory``; no host writable mount; no remote; no real secrets/stores.
  - the repo is a CLONE copied IN via ``podman cp`` (writable only inside the
    container, never a live host bind-mount). The manifest + fixtures are
    READ-ONLY bind-mounts (writes → EROFS at the mount layer).
- Read-back channel (closes the handoff's "one open question"):
  - the supervisor COMPUTES the diff itself after teardown — it never applies a
    worker-authored patch. The worker controls ``/work/repo/.git``, and git runs
    commands from repo-local config (clean filters, ``core.fsmonitor``, external
    diff) and honours replace-refs, so git is NEVER run against the worker's
    ``.git``. Instead ``compute_trusted_diff`` overlays the worker's WORKTREE
    (sans every ``.git`` at any depth — so a nested ``.git`` can't make the
    trusted ``git add`` record a content-hiding submodule gitlink — symlinks not
    followed) onto a fresh TRUSTED checkout of ``base_sha`` and diffs that with
    hardened flags + a sanitised git env.
  - everything under ``/work/artifacts`` is untrusted DATA: copied out, read as
    text, never executed and never used as a path/instruction.
- Side effects: spawns podman containers, clones/copies under a run dir, removes
  the container on teardown.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from agent_branch.manifest import BranchManifest
from utils.logging_utils import get_logger

logger = get_logger("agent_branch.provisioning")

# Base image must be present offline; the derived worker image just pre-creates
# /work so the supervisor can ``podman cp`` the clone in. Kept minimal on purpose.
DEFAULT_BASE_IMAGE = "docker.io/library/python:3.11-slim"
WORKER_IMAGE = "localhost/agentbranch-worker:m1"
# Curated "light" eval image — Daemon's non-ML deps, so a PROOF can import
# config / data models / graph / pure-logic modules (built on demand from
# eval_image/Containerfile.light). The worker image stays deps-free.
EVAL_IMAGE_LIGHT = "localhost/agentbranch-eval-light:m3"
_EVAL_IMAGE_DIR = Path(__file__).parent / "eval_image"
DEFAULT_IMAGE = WORKER_IMAGE
PODMAN = "podman"

# NOTE: deliberately do NOT pre-create /work/repo — ``podman cp clone :/work/repo``
# must create it AS the clone. (If it pre-exists, cp nests the clone inside it.)
_CONTAINERFILE = (
    "FROM {base}\n"
    "RUN mkdir -p /work/artifacts /work/fixtures\n"
    "WORKDIR /work\n"
)


class NetworkMode(str, Enum):
    NONE = "none"        # attacker — zero egress
    LLM_UDS = "llm_uds"  # llm worker — --network=none + UDS to supervisor proxy


class SandboxConfig(BaseModel):
    image: str = DEFAULT_IMAGE
    memory: str = "512m"
    pids_limit: int = 256
    network: NetworkMode = NetworkMode.NONE
    proxy_uds_host: Optional[str] = None  # host path to proxy UDS (LLM_UDS mode)
    env: Dict[str, str] = Field(default_factory=dict)  # extra worker env vars


class ProvisionResult(BaseModel):
    branch_id: str
    container_name: str
    run_dir: str
    repo_host: str       # host clone (the worker's input)
    base_sha: str        # baseline commit the supervisor diffs against
    artifacts_host: str  # where collected artifacts land after teardown


class RunOutcome(BaseModel):
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# podman availability
# ---------------------------------------------------------------------------

def _run(cmd: List[str], *, timeout: float = 120, check: bool = False,
         capture: bool = True, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    logger.debug("exec: %s", " ".join(cmd))
    return subprocess.run(
        cmd, timeout=timeout, check=check,
        capture_output=capture, text=True, env=env,
    )


# Git options that neutralise EVERY repo-config code-execution / diff-forgery
# vector, used only when running git on the TRUSTED checkout (defense in depth —
# the worker's .git is never used at all, see compute_trusted_diff):
#   core.fsmonitor=   → no fsmonitor hook command
#   core.hooksPath    → no hooks
#   core.quotepath    → stable, unescaped path text
#   --no-replace-objects → ignore any replace-refs that would change the base
_GIT_HARDENED: List[str] = [
    "-c", "core.fsmonitor=",
    "-c", "core.hooksPath=/dev/null",
    "-c", "core.quotepath=false",
    "--no-replace-objects",
]


def _git_safe_env() -> Dict[str, str]:
    """A git environment that ignores ALL ambient config (system/global) and any
    interactive prompt or non-file protocol — so only the trusted repo's own
    (supervisor-created) .git can influence the command."""
    env = dict(os.environ)
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_ALLOW_PROTOCOL"] = "file"
    return env


def _overlay_worktree(*, src: Path, dst: Path) -> None:
    """Replace ``dst``'s working tree (PRESERVING ``dst/.git``) with ``src``'s
    working tree, EXCLUDING every ``.git`` at ANY depth. Symlinks are recreated
    verbatim, never followed — so a worker symlink pointing at a host file can't
    pull that file's contents into the diff.

    Nested ``.git`` entries are stripped at all depths (not just the top level):
    otherwise a worker could drop a NEW untracked directory holding its real
    (malicious) files PLUS a nested ``.git``, and the trusted ``git add -A`` would
    record only a submodule gitlink (mode 160000, "+Subproject commit <sha>") for
    that directory — hiding every line inside it from the static gate's
    size / scope / content checks (a confirmed diff-masking evasion). ``.git`` as a
    file (gitfile) or symlink is excluded by name too."""
    _ignore_git = shutil.ignore_patterns(".git")
    for child in list(dst.iterdir()):
        if child.name == ".git":
            continue
        if child.is_symlink() or not child.is_dir():
            child.unlink()
        else:
            shutil.rmtree(child)
    for child in sorted(src.iterdir()):
        if child.name == ".git":
            continue
        target = dst / child.name
        if child.is_symlink():
            os.symlink(os.readlink(child), target)
        elif child.is_dir():
            shutil.copytree(child, target, symlinks=True, ignore=_ignore_git)
        else:
            shutil.copy2(child, target)


def compute_trusted_diff(*, worker_repo: str | Path, trusted_baseline_repo: str | Path,
                         base_sha: str, work_dir: str | Path) -> str:
    """Compute the worker's diff against ``base_sha`` WITHOUT ever letting the
    worker's clone drive git.

    SECURITY-CRITICAL. The worker has full write access to ``worker_repo/.git``,
    and git executes commands from repo-local config (``core.fsmonitor``,
    ``clean``/``smudge`` filters, ``diff.external``/textconv) and honours
    replace-refs that rewrite what a diff says. Running git against the worker's
    ``.git`` would be host code execution BEFORE the static gate sees anything.

    So: build a fresh TRUSTED checkout at ``base_sha`` (its ``.git`` is
    supervisor-made), overlay only the worker's WORKTREE (sans ``.git``) onto it,
    then diff with hardened flags and a sanitised env. The worker's ``.git`` is
    never read by git.
    """
    worker_repo = Path(worker_repo)
    trusted_baseline_repo = Path(trusted_baseline_repo)
    work_dir = Path(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    env = _git_safe_env()

    # 1. fresh trusted checkout from the supervisor's baseline clone (NOT the worker's)
    _run(["git", *_GIT_HARDENED, "clone", "--no-hardlinks", "--no-local", "--quiet",
          f"file://{trusted_baseline_repo.resolve()}", str(work_dir)], check=True, env=env)
    _run(["git", "-C", str(work_dir), *_GIT_HARDENED, "checkout", "--quiet", base_sha],
         check=False, env=env)
    # 2. overlay the worker's worktree (its .git is excluded → cannot run code)
    _overlay_worktree(src=worker_repo, dst=work_dir)
    # 3. diff with the TRUSTED .git
    _run(["git", "-C", str(work_dir), *_GIT_HARDENED, "add", "-A"], check=False, env=env)
    proc = _run(["git", "-C", str(work_dir), *_GIT_HARDENED, "diff", "--cached", base_sha],
                check=False, env=env)
    return proc.stdout or ""


def podman_available(base_image: str = DEFAULT_BASE_IMAGE) -> bool:
    """True iff rootless podman works AND the base image is present offline (the
    derived worker image is built on demand from it)."""
    if shutil.which(PODMAN) is None:
        return False
    try:
        if _run([PODMAN, "info", "--format", "{{.Host.Security.Rootless}}"], timeout=30).returncode != 0:
            return False
        return _run([PODMAN, "image", "exists", base_image], timeout=30).returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def require_podman(base_image: str = DEFAULT_BASE_IMAGE) -> None:
    if not podman_available(base_image):
        raise RuntimeError(
            f"rootless podman with base image {base_image!r} is required but "
            "unavailable; cannot provision an isolated worker"
        )


def ensure_worker_image(base: str = DEFAULT_BASE_IMAGE, image: str = WORKER_IMAGE) -> str:
    """Build the minimal derived worker image if absent (offline). Returns the
    image ref. Idempotent."""
    if _run([PODMAN, "image", "exists", image]).returncode == 0:
        return image
    ctx = tempfile.mkdtemp(prefix="agentbranch-build-")
    try:
        cf = Path(ctx) / "Containerfile"
        cf.write_text(_CONTAINERFILE.format(base=base), encoding="utf-8")
        _run([PODMAN, "build", "--network=none", "-t", image, "-f", str(cf), ctx],
             check=True, timeout=300)
        logger.info("built worker image %s from %s", image, base)
    finally:
        shutil.rmtree(ctx, ignore_errors=True)
    return image


def ensure_eval_image(image: str = EVAL_IMAGE_LIGHT, *,
                      containerfile: str = "Containerfile.light",
                      timeout: float = 1800) -> str:
    """Build the curated 'light' eval image (Daemon's non-ML deps) if absent, so a
    PROOF test can import config / data-model / graph / pure-logic modules. Used
    ONLY for the disposable proof container — the worker image stays deps-free.

    Unlike ensure_worker_image, the BUILD needs network (pip install); the eval RUN
    that later uses this image still runs --network=none. Idempotent."""
    if _run([PODMAN, "image", "exists", image]).returncode == 0:
        return image
    cf = _EVAL_IMAGE_DIR / containerfile
    if not cf.exists():
        raise FileNotFoundError(f"eval Containerfile not found: {cf}")
    logger.info("building eval image %s from %s (pip install — needs network)...", image, cf)
    proc = _run([PODMAN, "build", "-t", image, "-f", str(cf), str(_EVAL_IMAGE_DIR)],
                timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"eval image build failed:\n{(proc.stderr or '')[-2000:]}")
    logger.info("built eval image %s", image)
    return image


# ---------------------------------------------------------------------------
# sandbox
# ---------------------------------------------------------------------------

class WorkerSandbox:
    """One worker, one container, one run directory."""

    def __init__(
        self,
        branch_id: str,
        manifest: BranchManifest,
        source_repo: str | Path,
        *,
        runs_root: str | Path,
        worker_script: str | Path,
        config: Optional[SandboxConfig] = None,
        seed_files: Optional[Dict[str, str]] = None,
        fake_env_path: Optional[str | Path] = None,
        synthetic_chroma_path: Optional[str | Path] = None,
    ) -> None:
        self.branch_id = branch_id
        self.manifest = manifest
        self.source_repo = Path(source_repo)
        self.runs_root = Path(runs_root)
        self.worker_script = Path(worker_script)
        self.config = config or SandboxConfig()
        self.seed_files = seed_files or {}
        self.fake_env_path = Path(fake_env_path) if fake_env_path else None
        self.synthetic_chroma_path = Path(synthetic_chroma_path) if synthetic_chroma_path else None

        self.container_name = f"agentbranch-{branch_id}"
        self.run_dir = self.runs_root / branch_id
        self.repo_host = self.run_dir / "clone"
        self.manifest_host = self.run_dir / "manifest.json"
        self.artifacts_host = self.run_dir / "artifacts"
        self.base_sha: str = ""
        self.staged_fake_env: Optional[Path] = None
        self.staged_chroma: Optional[Path] = None
        self._provisioned = False

    # -- provisioning --------------------------------------------------------

    def provision(self) -> ProvisionResult:
        require_podman()
        ensure_worker_image(image=self.config.image)
        self.manifest.assert_intact()
        self._prepare_run_dir()
        self._clone_and_seed()
        self._write_readonly_manifest()
        self._stage_fixtures()
        self._create_container()
        self._copy_into_container()
        self._provisioned = True
        logger.info("provisioned worker %s (base_sha=%s)", self.branch_id, self.base_sha[:12])
        return ProvisionResult(
            branch_id=self.branch_id,
            container_name=self.container_name,
            run_dir=str(self.run_dir),
            repo_host=str(self.repo_host),
            base_sha=self.base_sha,
            artifacts_host=str(self.artifacts_host),
        )

    def _prepare_run_dir(self) -> None:
        if self.run_dir.exists():
            shutil.rmtree(self.run_dir)
        self.run_dir.mkdir(parents=True)
        self.artifacts_host.mkdir()

    def _clone_and_seed(self) -> None:
        _run(
            ["git", "clone", "--no-hardlinks", "--quiet",
             f"file://{self.source_repo.resolve()}", str(self.repo_host)],
            check=True,
        )
        # The clone naturally excludes gitignored files (.env, real data) — only
        # tracked content crosses the boundary.
        for rel, content in self.seed_files.items():
            dest = self.repo_host / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
        if self.seed_files:
            _run(["git", "-C", str(self.repo_host), "add", "-A"], check=True)
            _run(["git", "-C", str(self.repo_host), "-c", "user.email=sup@local",
                  "-c", "user.name=supervisor", "commit", "-q", "-m", "seed"], check=True)
        out = _run(["git", "-C", str(self.repo_host), "rev-parse", "HEAD"], check=True)
        self.base_sha = out.stdout.strip()

    def _write_readonly_manifest(self) -> None:
        # mode 644 so a worker write fails with EROFS (the MOUNT is the boundary),
        # not EACCES (which file perms alone would give).
        self.manifest.to_json_file(self.manifest_host)
        self.manifest_host.chmod(0o644)

    def _stage_fixtures(self) -> None:
        """Copy fixtures INTO the run dir so we never bind-mount (or SELinux-
        relabel) files in the repo tree. The run-dir copies are throwaway."""
        fixtures = self.run_dir / "fixtures"
        fixtures.mkdir(exist_ok=True)
        if self.fake_env_path:
            dest = fixtures / "fake_env"
            shutil.copyfile(self.fake_env_path, dest)
            dest.chmod(0o644)
            self.staged_fake_env = dest
        if self.synthetic_chroma_path and Path(self.synthetic_chroma_path).exists():
            dest = fixtures / "chroma"
            shutil.copytree(self.synthetic_chroma_path, dest)
            self.staged_chroma = dest

    def _create_container(self) -> None:
        _run([PODMAN, "rm", "-f", self.container_name], check=False)
        # ``:ro,Z`` keeps the mount read-only (EROFS on write) AND relabels the
        # throwaway run-dir copy so the SELinux-confined container can read it.
        cmd = [
            PODMAN, "create",
            "--name", self.container_name,
            "--network=none",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--pids-limit={self.config.pids_limit}",
            f"--memory={self.config.memory}",
            "--workdir", "/work",
            "-v", f"{self.manifest_host}:/work/manifest.json:ro,Z",
        ]
        if self.staged_fake_env:
            cmd += ["-v", f"{self.staged_fake_env}:/work/fixtures/.env:ro,Z"]
        if self.staged_chroma:
            cmd += ["-v", f"{self.staged_chroma}:/work/fixtures/chroma:ro,Z"]
        if self.config.network == NetworkMode.LLM_UDS and self.config.proxy_uds_host:
            cmd += ["-v", f"{Path(self.config.proxy_uds_host).resolve()}:/work/llm.sock"]
            # SELinux confines the container away from the host-labeled socket; the
            # LLM worker (which already has this deliberate channel) drops only the
            # SELinux layer. Its IP isolation, cap-drop, no-host-mount and
            # no-secrets guarantees are untouched. The attacker (NONE mode) keeps
            # full SELinux confinement — the zero-egress proof stays maximally hard.
            cmd += ["--security-opt", "label=disable"]
        cmd += [
            "-e", "WORKER_REPO=/work/repo",
            "-e", "WORKER_MANIFEST=/work/manifest.json",
            "-e", "WORKER_ARTIFACTS=/work/artifacts",
            "-e", "WORKER_FAKE_ENV=/work/fixtures/.env",
            "-e", "WORKER_CHROMA=/work/fixtures/chroma",
            "-e", "WORKER_LLM_SOCK=/work/llm.sock",
        ]
        for key, val in self.config.env.items():
            cmd += ["-e", f"{key}={val}"]
        cmd += [self.config.image, "python", "/work/worker.py"]
        _run(cmd, check=True)

    def _copy_into_container(self) -> None:
        # repo + worker script copied IN (writable only inside the container).
        _run([PODMAN, "cp", str(self.repo_host), f"{self.container_name}:/work/repo"], check=True)
        _run([PODMAN, "cp", str(self.worker_script), f"{self.container_name}:/work/worker.py"], check=True)

    # -- running -------------------------------------------------------------

    def run_worker(self, *, wallclock_seconds: Optional[int] = None) -> RunOutcome:
        if not self._provisioned:
            raise RuntimeError("call provision() before run_worker()")
        timeout = wallclock_seconds or self.manifest.wallclock_seconds
        start = time.monotonic()
        timed_out = False
        try:
            proc = _run([PODMAN, "start", "-a", self.container_name], timeout=timeout)
            code, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            timed_out = True
            _run([PODMAN, "kill", self.container_name], check=False)
            code, out, err = 124, (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), "wallclock timeout"
        elapsed = time.monotonic() - start
        logger.info("worker %s exited code=%s timed_out=%s in %.1fs",
                    self.branch_id, code, timed_out, elapsed)
        return RunOutcome(exit_code=code, stdout=out or "", stderr=err or "",
                          timed_out=timed_out, elapsed_s=elapsed)

    # -- read-back (supervisor-computed; never trusts worker output) ---------

    def collect_diff(self) -> str:
        """Supervisor computes the diff itself from the worker clone.

        The worker fully controls ``/work/repo/.git``; running git against it on
        the host would execute worker-supplied config (clean filters, fsmonitor,
        external diff) before the static gate ever runs. So we copy the clone out
        and hand it to ``compute_trusted_diff``, which never lets the worker's
        ``.git`` drive git — it diffs the worker WORKTREE against a fresh trusted
        checkout of ``base_sha``."""
        repo_out = self.run_dir / "repo_out"
        if repo_out.exists():
            shutil.rmtree(repo_out)
        _run([PODMAN, "cp", f"{self.container_name}:/work/repo", str(repo_out)], check=True)
        return compute_trusted_diff(
            worker_repo=repo_out,
            trusted_baseline_repo=self.repo_host,
            base_sha=self.base_sha,
            work_dir=self.run_dir / "diff_root",
        )

    def collect_artifacts(self) -> Dict[str, str]:
        """Copy /work/artifacts out and read as untrusted TEXT (never executed)."""
        out_dir = self.run_dir / "artifacts_out"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        proc = _run([PODMAN, "cp", f"{self.container_name}:/work/artifacts", str(out_dir)], check=False)
        artifacts: Dict[str, str] = {}
        if proc.returncode == 0 and out_dir.exists():
            for f in sorted(out_dir.rglob("*")):
                if f.is_file():
                    try:
                        artifacts[str(f.relative_to(out_dir))] = f.read_text(
                            encoding="utf-8", errors="replace")[:50_000]
                    except OSError:
                        continue
        return artifacts

    def container_inspect(self) -> dict:
        """Read podman's view of the container — used by red-team tests to prove
        structural properties (e.g. NetworkMode == none) independent of the
        worker's self-report."""
        proc = _run([PODMAN, "inspect", self.container_name], check=False)
        try:
            return json.loads(proc.stdout)[0] if proc.stdout else {}
        except (json.JSONDecodeError, IndexError):
            return {}

    # -- teardown ------------------------------------------------------------

    def teardown(self, *, remove_run_dir: bool = False) -> None:
        _run([PODMAN, "rm", "-f", self.container_name], check=False)
        if remove_run_dir and self.run_dir.exists():
            shutil.rmtree(self.run_dir, ignore_errors=True)
        logger.info("torn down worker %s (run_dir_removed=%s)", self.branch_id, remove_run_dir)
