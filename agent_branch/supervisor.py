# agent_branch/supervisor.py
"""
Supervisor — the trusted orchestrator (Milestone 1, N=1).

Module Contract
- Purpose: issue an immutable manifest to ONE worker, run it in isolation,
  read results back SAFELY, gate the diff, and write a supervisor-authored
  report. The supervisor owns evaluation and all external writes; the worker
  never grades itself.
- Pipeline (per branch):
    provision → run worker → COMPUTE diff (supervisor-side) → collect artifacts
    → Layer 1a static gate → [if clean] Layer 1b sandboxed tests
    → Layer 2 intent judge (stub) → write report → teardown
- Trust boundaries enforced here:
  - the diff is computed by the supervisor from the worker clone, never a
    worker-authored patch (D6);
  - the static gate runs BEFORE any code execution and gates whether the
    dynamic layer runs at all (D2);
  - token spend is read from the proxy (tamper-proof), not the worker (D1);
  - a tampered manifest can authorize nothing (assert_intact throughout).
- Side effects: provisions/tears down podman containers, writes a report under
  the run dir.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from agent_branch import eval_layer2
from agent_branch.eval_sandboxed import SandboxEvalResult, evaluate_in_sandbox
from agent_branch.eval_static import GateResult, RunStats, run_static_gate
from agent_branch.llm_proxy import LLMProxy
from agent_branch.manifest import BranchManifest
from agent_branch.provisioning import (
    NetworkMode,
    ProvisionResult,
    RunOutcome,
    SandboxConfig,
    WorkerSandbox,
)
from utils.logging_utils import get_logger

logger = get_logger("agent_branch.supervisor")

_FIXTURES = Path(__file__).parent / "fixtures"
DEFAULT_FAKE_ENV = _FIXTURES / "fake_env"
DEFAULT_SYNTH_CHROMA = _FIXTURES / "synthetic_chroma"


def _is_test_file(path: str) -> bool:
    """A worker-added test = a test-named .py in the worker's allowed scope (a
    tests/ edit is already killed by the static gate before we get here)."""
    base = path.rsplit("/", 1)[-1]
    return base.endswith(".py") and (base.startswith("test_") or base.endswith("_test.py"))


def worker_env_for(manifest: BranchManifest, worker_env: Optional[Dict[str, str]]) -> Dict[str, str]:
    """The environment handed to the worker container. The worker's objective is
    ALWAYS the manifest's objective (a worker can never be pointed at a goal the
    supervisor didn't authorize), surfaced as WORKER_OBJECTIVE. Explicit
    ``worker_env`` still wins (tests / per-worker overrides set before this)."""
    env = dict(worker_env or {})
    env.setdefault("WORKER_OBJECTIVE", manifest.objective)
    return env


class BranchReport(BaseModel):
    branch_id: str
    objective: str
    strategy: str = ""            # surgical | refactor | testfirst | saboteur | ...
    outcome: str  # "passed" | "killed" | "error"
    reasons: List[str] = Field(default_factory=list)
    run: RunOutcome
    run_stats: RunStats
    static_gate: GateResult
    sandbox_eval: Optional[SandboxEvalResult] = None
    intent: eval_layer2.IntentVerdict
    diff_excerpt: str = ""
    artifacts: Dict[str, str] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"


class Supervisor:
    def __init__(
        self,
        source_repo: str | Path,
        *,
        runs_root: str | Path,
        trusted_repo: Optional[str | Path] = None,
        fake_env_path: str | Path = DEFAULT_FAKE_ENV,
        synthetic_chroma_path: str | Path = DEFAULT_SYNTH_CHROMA,
    ) -> None:
        self.source_repo = Path(source_repo)
        self.runs_root = Path(runs_root)
        # the trusted checkout the worker's diff is evaluated against (defaults to
        # the same source — the worker cloned from it).
        self.trusted_repo = Path(trusted_repo) if trusted_repo else self.source_repo
        self.fake_env_path = Path(fake_env_path)
        self.synthetic_chroma_path = Path(synthetic_chroma_path)

    def run_branch(
        self,
        manifest: BranchManifest,
        worker_script: str | Path,
        *,
        seed_files: Optional[Dict[str, str]] = None,
        strategy: str = "",
        network: NetworkMode = NetworkMode.NONE,
        proxy: Optional[LLMProxy] = None,
        worker_env: Optional[Dict[str, str]] = None,
        mount_synthetic_chroma: bool = True,
        sandbox_test_cmd: Optional[List[str]] = None,
        eval_image: Optional[str] = None,
        keep_run_dir: bool = False,
    ) -> BranchReport:
        manifest.assert_intact()
        config = SandboxConfig(network=network, env=worker_env_for(manifest, worker_env))
        if network == NetworkMode.LLM_UDS and proxy is not None:
            config.proxy_uds_host = proxy.uds_path

        sandbox = WorkerSandbox(
            manifest.branch_id, manifest, self.source_repo,
            runs_root=self.runs_root, worker_script=worker_script, config=config,
            seed_files=seed_files,
            fake_env_path=self.fake_env_path,
            synthetic_chroma_path=self.synthetic_chroma_path if mount_synthetic_chroma else None,
        )

        run = RunOutcome()
        try:
            provision = sandbox.provision()
            run = sandbox.run_worker()
            diff = sandbox.collect_diff()
            artifacts = sandbox.collect_artifacts()

            run_stats = RunStats(
                tokens_spent=(proxy.tokens_spent if proxy else 0),
                wallclock_elapsed_s=run.elapsed_s,
                commits=0, cycles=0,
            )

            # --- Layer 1a: static gate (NO code execution) ---
            guard_log = artifacts.get("guard.log", "") + artifacts.get("audit.log", "")
            gate = run_static_gate(diff, manifest, run_stats=run_stats,
                                   guard_log=guard_log, audit_log=guard_log)

            # wallclock timeout is also a kill, even if the (empty) diff looked clean
            reasons = list(gate.reasons)
            killed = gate.killed
            if run.timed_out:
                killed = True
                reasons.append("worker exceeded wallclock (killed by supervisor timer)")

            sandbox_eval: Optional[SandboxEvalResult] = None
            intent = eval_layer2.IntentVerdict()

            if killed:
                outcome = "killed"
            else:
                # --- Layer 1b: dynamic, sandboxed (never on host) ---
                # worker-added tests (in allowed scope) run as EVIDENCE, not proof
                branch_test_files = [p for p in gate.touched_paths if _is_test_file(p)]
                sandbox_eval = evaluate_in_sandbox(
                    self.trusted_repo, diff, manifest.required_tests,
                    test_cmd=sandbox_test_cmd,
                    branch_test_files=branch_test_files or None,
                    # per-branch name so concurrent portfolio evals don't collide
                    container_name=f"agentbranch-eval-{manifest.branch_id}",
                    # deps-carrying eval image (proof imports real modules); the
                    # worker image stays deps-free. None -> default bare image.
                    **({"image": eval_image} if eval_image else {}),
                )
                intent = eval_layer2.judge_intent(diff, manifest)
                if sandbox_eval.killed:
                    outcome = "killed"
                    reasons.append(sandbox_eval.reason)
                elif intent.killed:
                    outcome = "killed"
                    reasons.append(intent.reason)
                else:
                    outcome = "passed"
                    reasons.append("survived static gate + sandboxed tests")

            report = BranchReport(
                branch_id=manifest.branch_id, objective=manifest.objective,
                strategy=strategy, outcome=outcome, reasons=reasons,
                run=run, run_stats=run_stats,
                static_gate=gate, sandbox_eval=sandbox_eval, intent=intent,
                diff_excerpt=diff[:4000], artifacts=artifacts,
            )
            self._write_report(provision, report)
            logger.info("branch %s -> %s", manifest.branch_id, outcome)
            return report
        except Exception as e:  # noqa: BLE001
            logger.exception("supervisor error on branch %s", manifest.branch_id)
            return BranchReport(
                branch_id=manifest.branch_id, objective=manifest.objective,
                strategy=strategy, outcome="error", reasons=[f"{type(e).__name__}: {e}"],
                run=run, run_stats=RunStats(), static_gate=GateResult(),
                intent=eval_layer2.IntentVerdict(),
            )
        finally:
            sandbox.teardown(remove_run_dir=not keep_run_dir)

    def _write_report(self, provision: ProvisionResult, report: BranchReport) -> None:
        try:
            Path(provision.run_dir).mkdir(parents=True, exist_ok=True)
            Path(provision.run_dir, "branch_report.json").write_text(
                report.model_dump_json(indent=2), encoding="utf-8")
        except OSError:
            logger.warning("could not persist branch report for %s", report.branch_id)


# ---------------------------------------------------------------------------
# Minimal demo CLI (manual verification per the plan)
# ---------------------------------------------------------------------------

def _make_demo_repo(root: Path) -> Path:
    repo = root / "src"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "demo@local"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "demo"], check=True)
    (repo / "README.md").write_text("demo\n")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True)
    return repo


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Agent-branch supervisor demo (M1)")
    parser.add_argument("--demo", choices=["attacker", "timeout"], default="attacker",
                        help="run a scripted worker against a throwaway repo")
    args = parser.parse_args(argv)

    workers = Path(__file__).parent / "workers"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        repo = _make_demo_repo(root)
        manifest = BranchManifest.issue(
            branch_id=f"demo-{args.demo}", objective="demo isolation",
            allowed_paths=["sandbox/**"], required_tests=[], wallclock_seconds=5,
        )
        sup = Supervisor(repo, runs_root=root / "runs")
        report = sup.run_branch(
            manifest, workers / "attacker.py",
            seed_files={"sandbox/calc.py": "def add(a, b):\n    return a + b\n"},
            worker_env={"WORKER_ATTACK_MODE": args.demo},
            keep_run_dir=False,
        )
    print(json.dumps({
        "outcome": report.outcome,
        "reasons": report.reasons,
        "flags": [f.value for f in report.static_gate.flags],
        "high_risk": report.static_gate.high_risk,
        "touched": report.static_gate.touched_paths,
    }, indent=2))
    return 0 if report.outcome == "killed" else 1


if __name__ == "__main__":
    sys.exit(main())
