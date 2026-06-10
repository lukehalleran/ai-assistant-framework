# agent_branch/eval_layer2.py
"""
Layer 2 — LLM-judge intent check. **STUB (M3).**

Module Contract
- Purpose (eventual): after a diff survives the mechanical gates (1a static + 1b
  sandboxed tests), an LLM judge assesses whether the change actually serves the
  manifest objective and isn't subtly adversarial (e.g. a plausible-looking edit
  that smuggles intent the tests don't cover).
- M1 status: the HOOK exists so the supervisor can call it in the pipeline, but
  no judge is built. It returns ``verdict="deferred"`` and never kills.
- Do NOT build the judge in M1 (see plan §"Explicitly deferred").
"""

from __future__ import annotations

from pydantic import BaseModel

from agent_branch.manifest import BranchManifest
from utils.logging_utils import get_logger

logger = get_logger("agent_branch.eval_layer2")


class IntentVerdict(BaseModel):
    verdict: str = "deferred"   # M3: "aligned" | "suspect" | "off_objective"
    killed: bool = False
    reason: str = "Layer 2 (LLM intent judge) is deferred to M3"


def judge_intent(diff_text: str, manifest: BranchManifest) -> IntentVerdict:
    """STUB. Always returns a non-killing 'deferred' verdict in M1."""
    logger.debug("Layer 2 intent judge is a stub (branch=%s)", manifest.branch_id)
    return IntentVerdict()
