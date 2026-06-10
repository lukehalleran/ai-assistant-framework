# agent_branch/manifest.py
"""
BranchManifest — the immutable, supervisor-authored contract issued to one worker.

Module Contract
- Purpose: A frozen, hash-verified description of a single bounded objective and
  the limits the supervisor enforces around it. The manifest is the SINGLE source
  of truth for what a worker is allowed to touch and how much it may spend.
- Inputs: supervisor-supplied fields via ``BranchManifest.issue(...)``.
- Outputs: an immutable model with a canonical SHA-256 ``manifest_hash``;
  JSON (de)serialization for the supervisor-side copy and the read-only mount.
- Key behaviors:
  - Immutable: ``model_config = frozen`` — attribute assignment raises.
  - ``issue()`` computes the canonical hash over all fields except the hash.
  - ``verify()`` / ``assert_intact()`` recompute and compare → detect tamper of
    the supervisor-side copy (the worker's mounted copy is read-only at the
    mount layer, a separate guarantee).
- Side effects: ``to_json_file`` / ``from_json_file`` touch disk; nothing else.

Scope note (M1): this is the TRIMMED manifest — only the fields the isolation
proof exercises. Portfolio fields (risk_level, allowed_tools, required_benchmarks,
success_metric, artifact_format, expiry, approval_gates) are deferred to M3.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List

from pydantic import BaseModel, ConfigDict, Field

from utils.logging_utils import get_logger

logger = get_logger("agent_branch.manifest")


# Real safety-critical paths in this repo. A worker diff that touches any of
# these is killed by the static gate. (The handoff's example listed a
# non-existent ``safety/`` dir — these are the actual paths.)
DEFAULT_FORBIDDEN_PATHS: List[str] = [
    "data/",
    ".env",
    "eval/",
    "tests/",
    "conftest.py",
    # config + thresholds: a worker must not "fake success" by editing a config
    # default or a benchmark threshold the supervisor's check relies on (M2 D2).
    "config/",
    # all files pytest reads config from (forge a passing proof), plus
    # .gitattributes (hide diff content / wire filters) — see eval_static.py.
    "pytest.ini",
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
    ".gitattributes",
    "feature_registry.yaml",
    "utils/destructive_op_guard.py",
    "utils/shell_cmd_guard.py",
    "utils/python_fs_guard.py",
    "utils/fs_snapshot.py",
    "scripts/safe_git.sh",
    "scripts/safe_cmd.sh",
    "scripts/bin/",
    "agent_branch/",
]


class ManifestTamperError(RuntimeError):
    """Raised when a manifest's stored hash does not match its current fields."""


class BranchManifest(BaseModel):
    """Immutable contract issued to a single worker (see module docstring)."""

    model_config = ConfigDict(frozen=True)

    branch_id: str = Field(..., description="Unique id for this branch/run")
    objective: str = Field(..., description="Single bounded goal for the worker")

    allowed_paths: List[str] = Field(
        default_factory=list,
        description="Globs the worker may write (everything else is forbidden)",
    )
    forbidden_paths: List[str] = Field(
        default_factory=lambda: list(DEFAULT_FORBIDDEN_PATHS),
        description="Paths a diff may never touch — touching one is an auto-kill",
    )
    protected_stores: List[str] = Field(
        default_factory=list,
        description="Chroma collections that must never be mounted real (synthetic only)",
    )

    max_diff_lines: int = Field(default=400, ge=0)
    max_commits: int = Field(default=10, ge=0)
    max_cycles: int = Field(default=20, ge=0)
    token_budget: int = Field(default=200_000, ge=0, description="Hard LLM spend cap (enforced at proxy)")
    wallclock_seconds: int = Field(default=900, ge=0, description="Hard timeout (supervisor-side timer)")

    required_tests: List[str] = Field(
        default_factory=list,
        description="Supervisor-owned tests, run OUTSIDE the worker in a disposable sandbox",
    )

    manifest_hash: str = Field(
        default="",
        description="SHA-256 over all other fields, canonicalized; verified every pass",
    )

    # ---- hashing / issuance ------------------------------------------------

    @staticmethod
    def _canonical_payload(data: dict) -> str:
        """Deterministic JSON of all fields except ``manifest_hash``."""
        payload = {k: v for k, v in data.items() if k != "manifest_hash"}
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def compute_hash(cls, data: dict) -> str:
        return hashlib.sha256(cls._canonical_payload(data).encode("utf-8")).hexdigest()

    @classmethod
    def issue(cls, **fields) -> "BranchManifest":
        """Build a manifest and stamp its canonical hash. The only sanctioned
        way to create a manifest the supervisor will accept."""
        fields.pop("manifest_hash", None)
        # Validate field values first (defaults applied) without a hash...
        draft = cls(**fields, manifest_hash="")
        # ...then compute the hash over the realized field set and re-issue frozen.
        digest = cls.compute_hash(draft.model_dump())
        manifest = cls(**draft.model_dump(exclude={"manifest_hash"}), manifest_hash=digest)
        logger.info("Issued manifest branch_id=%s hash=%s", manifest.branch_id, digest[:12])
        return manifest

    # ---- verification ------------------------------------------------------

    def expected_hash(self) -> str:
        return self.compute_hash(self.model_dump())

    def verify(self) -> bool:
        """True iff the stored hash matches the current fields."""
        return bool(self.manifest_hash) and self.manifest_hash == self.expected_hash()

    def assert_intact(self) -> None:
        """Raise ManifestTamperError if the manifest has been tampered with."""
        if not self.verify():
            raise ManifestTamperError(
                f"manifest {self.branch_id!r} hash mismatch: "
                f"stored={self.manifest_hash[:12]!r} expected={self.expected_hash()[:12]!r}"
            )

    # ---- persistence -------------------------------------------------------

    def to_json_file(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
        return p

    @classmethod
    def from_json_file(cls, path: str | Path, *, verify: bool = True) -> "BranchManifest":
        """Load a manifest. With ``verify=True`` (default) a tampered
        supervisor-side copy raises ManifestTamperError (red-team #6)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest = cls(**data)
        if verify:
            manifest.assert_intact()
        return manifest
