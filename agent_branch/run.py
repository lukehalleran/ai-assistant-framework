#!/usr/bin/env python3
# agent_branch/run.py
"""
Runner — drive a REAL coding-worker portfolio against a real repo (M3).

This is the entry point that turns the harness into something that produces actual,
varied, VERIFIED candidate changes for human review. It is the "real run" sibling
of ``portfolio.py --demo`` (which uses scripted offline workers).

The contract (a clean, safe self-improvement primitive):
    you give it  →  an OBJECTIVE, an allowed SCOPE, a committed stdlib PROOF test,
                    and a set of LENSES (per-agent goals files)
    it gives you →  one isolated coding-worker per lens, each making a bounded
                    change; the static gate + sandboxed proof kill the bad ones;
                    survivors are ranked and (with --ingest) land in the GUI
                    Proposals tab as PENDING, human-gated proposals.

Per-lens divergence (reliability / coverage / capability / ...) yields varied
implementations of the same objective for you to compare.

Hard constraints (honest, first version):
- A PROOF is required and gates correctness. The reaper kills anything whose proof
  fails, so without a proof nothing survives. Write a test that proves the objective
  is met.
- The proof + change run in the BARE python:3.11-slim worker image (no project
  deps). So today this verifies SELF-CONTAINED objectives whose proof is a stdlib
  script (module-level asserts), e.g. a new pure module/function. Objectives that
  need the full Daemon deps await the M3 deps-image.
- The proof test must be COMMITTED (the trusted eval clones committed files only).
- Needs an LLM key in the env (OPENROUTER_API_KEY or OPENAI_API_KEY); the proxy
  injects it — the worker never holds it. The worker stays --network=none with a
  single mediated UDS to the proxy.

Usage:
    OPENROUTER_API_KEY=... python -m agent_branch.run \
        --objective "Add utils/strcase.py: snake_to_camel(s) and camel_to_snake(s)" \
        --target utils/strcase.py \
        --allowed "utils/strcase.py" \
        --proof tests/proof_strcase.py \
        --lenses reliability,coverage,capability \
        --ingest
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from agent_branch.llm_proxy import LLMProxy
from agent_branch.portfolio import WORKERS_DIR, WorkerSpec, run_portfolio
from agent_branch.provisioning import NetworkMode
from utils.logging_utils import get_logger

logger = get_logger("agent_branch.run")

GOALS_DIR = "agent_branch/goals"
DEFAULT_UPSTREAM = os.getenv("AGENT_BRANCH_LLM_UPSTREAM", "https://openrouter.ai/api")
DEFAULT_MODEL = os.getenv("AGENT_BRANCH_LLM_MODEL", "anthropic/claude-haiku-4.5")


# --- pure helpers (testable without podman / a key) -------------------------

def api_key() -> Optional[str]:
    """The LLM key the proxy will inject (worker never sees it)."""
    return os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")


def build_template(*, objective: str, allowed: List[str], proofs: List[str],
                   max_diff_lines: int, wallclock: int, token_budget: int) -> dict:
    """The shared manifest template every lens-agent is issued from."""
    return dict(
        objective=objective,
        allowed_paths=list(allowed),
        required_tests=list(proofs),
        max_diff_lines=max_diff_lines,
        wallclock_seconds=wallclock,
        token_budget=token_budget,
    )


def build_specs(lenses: List[str], *, target: str, model: str) -> List[WorkerSpec]:
    """One coding-worker per lens; each gets its goals file + target + model. The
    objective is injected by the supervisor from the manifest (worker_env_for)."""
    specs: List[WorkerSpec] = []
    for lens in lenses:
        specs.append(WorkerSpec(
            branch_id=lens,
            strategy=lens,
            worker_script=str(WORKERS_DIR / "coding_worker.py"),
            worker_env={
                "WORKER_GOALS": f"{GOALS_DIR}/{lens}.md",
                "WORKER_TARGET": target,
                "WORKER_MODEL": model,
            },
            network=NetworkMode.LLM_UDS,
        ))
    return specs


def build_proxies(specs: List[WorkerSpec], *, upstream: str, key: str,
                  runs_root: Path, token_budget: int) -> Dict[str, LLMProxy]:
    """One proxy per branch — so each branch's token budget is enforced and metered
    independently (a shared proxy would attribute the combined spend to every
    branch). Not started here."""
    host = urlparse(upstream).hostname or ""
    proxies: Dict[str, LLMProxy] = {}
    for spec in specs:
        uds = str(runs_root / f"llm-{spec.branch_id}.sock")
        proxies[spec.branch_id] = LLMProxy(
            uds, upstream, api_key=key, allowed_hosts=[host],
            token_budget=token_budget,
            log_path=str(runs_root / f"llm-{spec.branch_id}.log"),
        )
    return proxies


def _csv(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


# --- CLI --------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run a real coding-worker portfolio (M3)")
    p.add_argument("--objective", required=True, help="what to build")
    p.add_argument("--target", required=True, help="the file workers may write")
    p.add_argument("--allowed", required=True,
                   help="comma-separated allowed_paths globs (must cover --target)")
    p.add_argument("--proof", required=True,
                   help="comma-separated COMMITTED stdlib proof test path(s)")
    p.add_argument("--lenses", default="reliability,coverage,capability",
                   help="comma-separated goals lenses (one agent each)")
    p.add_argument("--source", default=".", help="repo to run against")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    p.add_argument("--token-budget", type=int, default=200_000)
    p.add_argument("--wallclock", type=int, default=300)
    p.add_argument("--max-diff-lines", type=int, default=400)
    p.add_argument("--runs-root", default=None)
    p.add_argument("--ingest", action="store_true",
                   help="ingest survivors into the live proposal store (GUI tab)")
    args = p.parse_args(argv)

    key = api_key()
    if not key:
        print("No OPENROUTER_API_KEY / OPENAI_API_KEY in env — the proxy needs a key "
              "to forward to the LLM.", file=sys.stderr)
        return 2

    lenses = _csv(args.lenses)
    if not lenses:
        print("No --lenses given.", file=sys.stderr)
        return 2

    source = Path(args.source).resolve()
    runs_root = Path(args.runs_root) if args.runs_root else source.parent / "agent_branch_runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    template = build_template(
        objective=args.objective, allowed=_csv(args.allowed), proofs=_csv(args.proof),
        max_diff_lines=args.max_diff_lines, wallclock=args.wallclock,
        token_budget=args.token_budget,
    )
    specs = build_specs(lenses, target=args.target, model=args.model)
    proxies = build_proxies(specs, upstream=args.upstream, key=key,
                            runs_root=runs_root, token_budget=args.token_budget)

    store = None
    if args.ingest:
        from config.app_config import CHROMA_PATH
        from memory.proposal_store import ProposalStore
        from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
        chroma = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
        chroma.create_collection("proposals")
        store = ProposalStore(chroma_store=chroma)

    for px in proxies.values():
        px.start()
    time.sleep(0.1)
    try:
        trial = run_portfolio(
            template, specs,
            source_repo=source, runs_root=runs_root,
            sandbox_test_cmd=None,  # default_test_cmd: run each proof as `python <file>`
            proxies=proxies, proposal_store=store,
            max_concurrency=len(specs), keep_run_dir=True,
        )
    finally:
        for px in proxies.values():
            px.stop()

    print(trial.render_markdown())
    spent = sum(px.tokens_spent for px in proxies.values())
    print(f"\nLLM tokens metered (all branches): {spent}")
    if store:
        print("Survivors ingested into the live proposal store — open the GUI "
              "Proposals tab (they carry the 🤖 AGENT-BRANCH badge).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
