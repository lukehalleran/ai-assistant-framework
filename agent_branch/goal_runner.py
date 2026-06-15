#!/usr/bin/env python3
# agent_branch/goal_runner.py
"""
Goal-driven proposal mode — the sub-agents pick their OWN work from their goals.

The runner (run.py) needs a hand-written objective + proof. This mode instead lets
each lens behave like the self-proposer (GoalDirectedGenerator reads GOALS.md and
brainstorms aligned ideas) — but produces real, regression-checked CODE, not just
text. Per lens:

  1. DERIVE a bounded, code-targeted objective from the lens's goals file + the
     project goals/context (reusing GoalDirectedGenerator with the lens as extra
     context). Different lenses -> different objectives -> varied proposals.
  2. IMPLEMENT it with one isolated coding-worker (run.run_objective, single lens).
  3. GATE it: the SAFETY static gate always applies; the correctness bar is
     REGRESSION — an auto-generated proof that the target module imports cleanly in
     the deps-image (catches syntax/broken-import garbage) without a hand-written
     test. (b) in the design discussion.
  4. INGEST survivors into the proposal store; you review them in the GUI.

Correctness here is "doesn't break + is real code", not "provably implements the
objective" — because the human is the final reviewer (the goal is MORE, VARIED
proposals, not auto-merge). The strict-proof path (run.py / the queue) still exists
for when you want a verified objective.

Only code targets under safe areas are accepted (never tests/config/safety/
supervision/docs). Needs an LLM key + podman.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from agent_branch.run import DEFAULT_MODEL, api_key, live_proposal_store, run_objective
from utils.logging_utils import get_logger

logger = get_logger("agent_branch.goal_runner")

GOALS_DIR = "agent_branch/goals"
# A goal-driven agent may only TARGET code in these areas...
_ALLOWED_TARGET_PREFIXES = ("utils/", "knowledge/", "memory/", "core/",
                            "processing/", "models/")
# ...and never these (safety / supervision / tests / config / docs).
_FORBIDDEN_TARGET_PREFIXES = ("tests/", "config/", "agent_branch/", "scripts/", "docs/")


# --- pure helpers -----------------------------------------------------------

def module_path(target: str) -> str:
    """'utils/foo.py' -> 'utils.foo' (empty for non-.py)."""
    return target[:-3].replace("/", ".") if target.endswith(".py") else ""


def acceptable_target(target: str) -> bool:
    """A single .py file in a safe code area — the only thing a derived objective
    may target (the regression bar imports it; safety paths are off-limits)."""
    if not target or not target.endswith(".py"):
        return False
    if any(target.startswith(f) for f in _FORBIDDEN_TARGET_PREFIXES):
        return False
    return any(target.startswith(f) for f in _ALLOWED_TARGET_PREFIXES)


def autoproof_for(target: str) -> tuple[str, str]:
    """The regression proof for a target: (repo-relative path, file content). The
    proof imports the target module — so a diff that produces a syntax error or a
    broken import is killed, with no hand-written test."""
    mod = module_path(target)
    rel = f"agent_branch/proofs/_autoproof_{hashlib.sha1(target.encode()).hexdigest()[:8]}.py"
    content = (
        "import importlib, sys\n"
        "sys.path.insert(0, '.')\n"
        f"importlib.import_module({mod!r})\n"
        f"print('import OK: {mod}')\n"
    )
    return rel, content


# --- derivation -------------------------------------------------------------

async def derive_objective(lens_goals: str, *, repo: str, model_manager,
                           extra: str = "") -> Optional[Dict]:
    """Derive ONE bounded, code-targeted objective for a lens by running the
    self-proposer with the lens as its standing mandate. Returns
    {objective, target, allowed} or None if nothing code-targeted came back."""
    from knowledge.proposal_generator import GoalDirectedGenerator
    gen = GoalDirectedGenerator(model_manager=model_manager, repo_path=repo, max_proposals=5)
    ctx = f"## YOUR LENS (standing mandate — let it shape what you propose)\n{lens_goals}"
    if extra:
        ctx += f"\n\n{extra}"
    proposals = await gen.generate_proposals(extra_context=ctx, max_proposals=5)
    for p in proposals:
        target = next((f for f in (p.affected_files or []) if acceptable_target(f)), None)
        if target:
            objective = f"{p.title}. {p.description}".strip().rstrip(".") + "."
            return {"objective": objective, "target": target, "allowed": [target]}
    return None


# --- the goal-driven run ----------------------------------------------------

async def run_goal_driven(lenses: List[str], *, source: str = ".",
                          runs_root: Optional[str] = None, model: str = DEFAULT_MODEL,
                          temperature: float = 0.5, ingest: bool = True,
                          model_manager=None) -> List[Dict]:
    """For each lens: derive its own objective, implement it, regression-gate, and
    ingest survivors as proposals. Returns a per-lens result digest."""
    key = api_key()
    if not key:
        logger.warning("goal_runner: no LLM key — skipping")
        return []
    if model_manager is None:
        from models.model_manager import ModelManager
        model_manager = ModelManager()

    from agent_branch.provisioning import ensure_eval_image
    eval_image = ensure_eval_image()  # regression imports the target -> deps image

    src = Path(source).resolve()
    base = Path(runs_root) if runs_root else src.parent / "agent_branch_runs"
    store = live_proposal_store() if ingest else None

    results: List[Dict] = []
    for lens in lenses:
        lens_file = src / GOALS_DIR / f"{lens}.md"
        lens_goals = lens_file.read_text(encoding="utf-8") if lens_file.exists() else ""
        obj = await derive_objective(lens_goals, repo=str(src), model_manager=model_manager)
        if not obj:
            logger.info("goal_runner: %s derived no code-targeted objective", lens)
            results.append({"lens": lens, "derived": False})
            continue

        # fresh clone with the regression proof committed (the eval clones committed files)
        work = base / f"goal-{lens}"
        clone = work / "clone"
        clone.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", str(src), str(clone)], check=True)
        rel, content = autoproof_for(obj["target"])
        (clone / rel).parent.mkdir(parents=True, exist_ok=True)
        (clone / rel).write_text(content, encoding="utf-8")
        subprocess.run(["git", "-C", str(clone), "add", "-A"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(clone), "commit", "-qm", "autoproof"],
                       check=True, capture_output=True)

        trial, spent = run_objective(
            objective=obj["objective"], target=obj["target"], allowed=obj["allowed"],
            proofs=[rel], lenses=[lens], key=key, source=str(clone),
            runs_root=work / "runs", model=model, temperature=temperature,
            eval_image=eval_image, store=store,
        )
        survived = bool(trial.ranking)
        results.append({"lens": lens, "derived": True, "objective": obj["objective"][:80],
                        "target": obj["target"], "survived": survived, "tokens": spent})
        logger.info("goal_runner: %s -> %s [%s]", lens, obj["target"],
                    "survived" if survived else "killed")
    return results


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Goal-driven proposal mode (M3)")
    p.add_argument("--lenses", default="reliability,coverage,capability")
    p.add_argument("--source", default=".")
    p.add_argument("--runs-root", default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--no-ingest", action="store_true")
    args = p.parse_args(argv)
    lenses = [x.strip() for x in args.lenses.split(",") if x.strip()]
    results = asyncio.run(run_goal_driven(
        lenses, source=args.source, runs_root=args.runs_root, model=args.model,
        temperature=args.temperature, ingest=not args.no_ingest))
    print("\n=== goal-driven results ===")
    for r in results:
        if not r.get("derived"):
            print(f"  {r['lens']}: no code-targeted objective derived")
        else:
            print(f"  {r['lens']}: {r['target']} -> "
                  f"{'survived' if r['survived'] else 'killed'} ({r['tokens']}t)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
