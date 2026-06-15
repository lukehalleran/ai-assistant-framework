# agent_branch/queue.py
"""
Objective queue for shutdown-triggered agent_branch runs.

Module Contract
- Purpose: a small persistent queue of bounded objectives the supervisor should
  attempt with isolated coding-workers. Populated cheaply DURING sessions (no LLM);
  drained at shutdown by a detached run_queue process (the expensive part runs
  off the critical path). This is the cost-bounded alternative to a continuous
  autonomous loop: nothing runs unless something was queued.
- Two kinds of entry, same shape:
  - explicit: an objective + a proof test you wrote.
  - failing-test: "make this committed test pass" — the test IS the proof
    (cheapest source; no extra authoring). The agent may not edit the test (the
    static gate kills test edits anyway).
- Every entry carries its own SCOPE (target + allowed) and a PROOF, because the
  reaper kills any branch without a passing supervisor-owned proof — a queued
  objective is only runnable if it says how it will be proven.
- Storage: JSON at data/agent_branch_queue.json (gitignored local state). Never
  auto-deletes; entries are marked done/failed, not removed.
- Side effects: reads/writes the queue file only.
"""

from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

DEFAULT_QUEUE_PATH = Path("data/agent_branch_queue.json")
_DEFAULT_LENSES = ["reliability", "coverage", "capability"]


class QueuedObjective(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    objective: str
    target: str
    allowed: List[str]
    proof: List[str]
    lenses: List[str] = Field(default_factory=lambda: list(_DEFAULT_LENSES))
    deps: bool = False                      # run the proof in the light eval image
    status: str = "pending"                 # pending | running | done | failed
    created_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    result_proposal_ids: List[str] = Field(default_factory=list)
    note: str = ""


class ObjectiveQueue(BaseModel):
    entries: List[QueuedObjective] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_QUEUE_PATH) -> "ObjectiveQueue":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            return cls.model_validate_json(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a corrupt queue must not crash shutdown
            return cls()

    def save(self, path: str | Path = DEFAULT_QUEUE_PATH) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    def pending(self) -> List[QueuedObjective]:
        return [e for e in self.entries if e.status == "pending"]

    def get(self, entry_id: str) -> Optional[QueuedObjective]:
        return next((e for e in self.entries if e.id == entry_id), None)

    def add(self, entry: QueuedObjective) -> QueuedObjective:
        self.entries.append(entry)
        return entry


# --- enqueue helpers (cheap, no LLM) ----------------------------------------

def enqueue(objective: str, *, target: str, allowed: List[str], proof: List[str],
            lenses: Optional[List[str]] = None, deps: bool = False,
            path: str | Path = DEFAULT_QUEUE_PATH) -> QueuedObjective:
    """Queue an explicit objective + the proof test that certifies it."""
    q = ObjectiveQueue.load(path)
    entry = QueuedObjective(objective=objective, target=target, allowed=list(allowed),
                            proof=list(proof), lenses=lenses or list(_DEFAULT_LENSES),
                            deps=deps)
    q.add(entry)
    q.save(path)
    return entry


def enqueue_failing_test(test_path: str, *, target: str, allowed: List[str],
                         lenses: Optional[List[str]] = None, deps: bool = False,
                         path: str | Path = DEFAULT_QUEUE_PATH) -> QueuedObjective:
    """Queue 'make this committed test pass' — the test is the proof. The agent
    may not edit the test (the static gate enforces it)."""
    objective = (f"Make the test at {test_path} pass. Do NOT modify the test itself; "
                 f"implement the code it exercises.")
    return enqueue(objective, target=target, allowed=allowed, proof=[test_path],
                   lenses=lenses, deps=deps, path=path)


# --- tiny CLI: add / list ---------------------------------------------------

def _main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="agent_branch objective queue")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="queue an explicit objective + proof")
    a.add_argument("--objective", required=True)
    a.add_argument("--target", required=True)
    a.add_argument("--allowed", required=True, help="comma-separated globs")
    a.add_argument("--proof", required=True, help="comma-separated committed proof test(s)")
    a.add_argument("--lenses", default=",".join(_DEFAULT_LENSES))
    a.add_argument("--deps", action="store_true")

    t = sub.add_parser("add-test", help="queue 'make this committed test pass'")
    t.add_argument("--test", required=True, help="committed test path (the proof)")
    t.add_argument("--target", required=True)
    t.add_argument("--allowed", required=True, help="comma-separated globs")
    t.add_argument("--lenses", default=",".join(_DEFAULT_LENSES))
    t.add_argument("--deps", action="store_true")

    sub.add_parser("list", help="show the queue")

    args = p.parse_args(argv)
    _csv = lambda s: [x.strip() for x in (s or "").split(",") if x.strip()]

    if args.cmd == "add":
        e = enqueue(args.objective, target=args.target, allowed=_csv(args.allowed),
                    proof=_csv(args.proof), lenses=_csv(args.lenses), deps=args.deps)
        print(f"queued {e.id}: {e.objective[:70]}")
    elif args.cmd == "add-test":
        e = enqueue_failing_test(args.test, target=args.target, allowed=_csv(args.allowed),
                                 lenses=_csv(args.lenses), deps=args.deps)
        print(f"queued {e.id}: {e.objective[:70]}")
    elif args.cmd == "list":
        q = ObjectiveQueue.load()
        if not q.entries:
            print("queue empty")
        for e in q.entries:
            print(f"  [{e.status}] {e.id}  {e.objective[:60]}  -> {e.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
