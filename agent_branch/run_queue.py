#!/usr/bin/env python3
# agent_branch/run_queue.py
"""
Drain the agent_branch objective queue — the DETACHED process spawned at shutdown.

For each pending queued objective (capped per run), run its lens portfolio via
agent_branch.run.run_objective, ingest survivors into the live proposal store, and
mark the entry done/failed. Runs OFF the critical path: the shutdown hook spawns
this detached and returns immediately, so minutes-long agent_branch work never
blocks the app's exit. Survivors show up in the GUI Proposals tab next session.

Needs an LLM key (the proxy injects it) + podman. Idempotent on the queue: each
entry is marked running -> done/failed and the queue is saved after every entry,
so a crash mid-batch never re-runs a finished objective.

Usage (normally spawned by the shutdown hook; runnable by hand too):
    python -m agent_branch.run_queue --max 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from agent_branch.queue import DEFAULT_QUEUE_PATH, ObjectiveQueue
from agent_branch.run import (
    DEFAULT_MODEL,
    api_key,
    live_proposal_store,
    run_objective,
)
from utils.logging_utils import get_logger

logger = get_logger("agent_branch.run_queue")


def drain(*, queue_path: str | Path = DEFAULT_QUEUE_PATH, max_items: int = 2,
          source: str = ".", runs_root: Optional[str | Path] = None,
          model: str = DEFAULT_MODEL, temperature: float = 0.4,
          ingest: bool = True) -> int:
    """Process up to ``max_items`` pending objectives. Returns the count attempted."""
    key = api_key()
    if not key:
        logger.warning("run_queue: no OPENROUTER_API_KEY/OPENAI_API_KEY — skipping")
        return 0

    q = ObjectiveQueue.load(queue_path)
    pending = q.pending()[:max(0, max_items)]
    if not pending:
        logger.info("run_queue: nothing pending")
        return 0

    source = Path(source).resolve()
    base_runs = Path(runs_root) if runs_root else source.parent / "agent_branch_runs"
    store = live_proposal_store() if ingest else None

    eval_image = None
    if any(e.deps for e in pending):
        from agent_branch.provisioning import ensure_eval_image
        eval_image = ensure_eval_image()  # build the light image once, up front

    attempted = 0
    for e in pending:
        attempted += 1
        e.status = "running"
        q.save(queue_path)
        try:
            trial, spent = run_objective(
                objective=e.objective, target=e.target, allowed=e.allowed,
                proofs=e.proof, lenses=e.lenses, key=key, source=source,
                runs_root=base_runs / e.id,          # unique per objective
                model=model, temperature=temperature,
                eval_image=(eval_image if e.deps else None), store=store,
            )
            survivors = list(trial.ranking)
            e.status = "done"
            e.completed_at = time.time()
            e.note = f"{len(survivors)} survivor(s) [{', '.join(survivors)}], {spent} tokens"
            logger.info("run_queue: %s done — %d survivor(s)", e.id, len(survivors))
        except Exception as ex:  # noqa: BLE001
            e.status = "failed"
            e.completed_at = time.time()
            e.note = f"{type(ex).__name__}: {ex}"
            logger.warning("run_queue: %s failed: %s", e.id, ex)
        q.save(queue_path)  # persist after every entry (crash-safe)

    return attempted


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Drain the agent_branch objective queue")
    p.add_argument("--max", type=int, default=2, help="max objectives this run")
    p.add_argument("--source", default=".")
    p.add_argument("--runs-root", default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--no-ingest", action="store_true")
    args = p.parse_args(argv)
    n = drain(max_items=args.max, source=args.source, runs_root=args.runs_root,
              model=args.model, temperature=args.temperature, ingest=not args.no_ingest)
    print(f"run_queue: attempted {n} objective(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
