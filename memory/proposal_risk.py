# memory/proposal_risk.py
"""
ProposalRiskClassifier — single source of truth for a proposal's supervision
fields (``touches_core_system`` + ``risk_level``).

Module Contract
- Purpose: classify a proposed change from its affected files and (optionally)
  the code/text it would introduce, so supervision gating is computed rather than
  defaulted. Previously these fields defaulted to ``False`` / ``MEDIUM`` and were
  never computed in the live generation path — a proposal touching orchestration,
  memory, the safety layer, or the supervision machinery itself looked identical
  to a docs tweak.
- Inputs: affected file paths; optional code/text blobs (step snippets,
  description) for import-based detection; optional title/type for keyword/type
  risk fallback.
- Outputs: ``(touches_core_system: bool, risk_level: RiskLevel)`` from
  ``classify_proposal``; ``requires_human_ack(risk_level, touches_core_system) ->
  bool`` is the companion supervision POLICY (HIGH/CRITICAL or core touch → an
  explicit human acknowledgement is required before approve/merge), kept here so
  the gate can't drift from the classifier that produced the fields.
- Key behaviors:
  - Path matching is by directory PREFIX (not the old exact-string membership),
    so ``core/prompt/anything.py`` is caught by the ``core/prompt/`` entry.
  - IMPORT-based detection catches the refactor-and-move / re-export / new-file
    gap: a brand-new file that ``import``s or ``from``-imports a core/safety/
    supervision module is flagged even though its own path is "clean".
  - The SUPERVISION and SAFETY layers are unconditionally ``CRITICAL`` (no
    exceptions, including transitive/import touches) — they must never change
    without explicit human review.
- Side effects: none (pure functions).
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

from memory.code_proposal import ProposalType, RiskLevel
from utils.logging_utils import get_logger

logger = get_logger("proposal_risk")


# --- path groups (directory prefixes end with "/", files are exact) ---------

# Orchestration + memory spine + the live gate. Touching these is "core" (HIGH).
CORE_SYSTEM_PATHS: Tuple[str, ...] = (
    "core/orchestrator.py",
    "core/context_pipeline.py",
    "core/best_of_handler.py",
    "core/response_generator.py",
    "core/intent_classifier.py",
    "core/prompt/",                 # entire prompt-building pipeline
    "memory/memory_coordinator.py",
    "memory/memory_storage.py",
    "memory/memory_scorer.py",
    "memory/memory_retriever.py",
    "processing/gate_system.py",
)

# The safety layer: guards, snapshots, crisis/escalation, write-action machinery.
# Touching these is CRITICAL.
SAFETY_PATHS: Tuple[str, ...] = (
    "utils/destructive_op_guard.py",
    "utils/python_fs_guard.py",
    "utils/shell_cmd_guard.py",
    "utils/fs_snapshot.py",
    "scripts/safe_git.sh",
    "scripts/safe_cmd.sh",
    "scripts/bin/",
    "core/escalation_tracker.py",
    "core/action_claim_guard.py",
    "core/actions/",                # human-in-the-loop write actions
    "utils/tone_detector.py",       # crisis detection
)

# The supervision machinery itself — the code that classifies/gates proposals and
# isolates branch workers. A proposal must never silently weaken its own referee.
# Touching these is CRITICAL.
SUPERVISION_PATHS: Tuple[str, ...] = (
    "memory/code_proposal.py",
    "memory/proposal_store.py",
    "memory/proposal_risk.py",      # this module
    "knowledge/proposal_generator.py",
    "config/feature_registry.py",
    "config/feature_registry.yaml",
    "agent_branch/",                # the M1/M2 isolation + eval harness
)

# CRITICAL = safety + supervision (the must-not-silently-change layer).
_CRITICAL_PATHS: Tuple[str, ...] = SAFETY_PATHS + SUPERVISION_PATHS
# Everything above is "core" for touches_core_system purposes.
_ALL_CORE_PATHS: Tuple[str, ...] = CORE_SYSTEM_PATHS + _CRITICAL_PATHS

# Keyword fallbacks (kept consistent with scripts/migrate_proposals_supervision.py).
_HIGH_RISK_KEYWORDS = re.compile(
    r"safety|guard|security|auth|crisis|escalation|shutdown|delete|purge|migration",
    re.IGNORECASE,
)
_CRITICAL_RISK_KEYWORDS = re.compile(
    r"data.loss|destructive|wipe|drop.table|rm.-rf", re.IGNORECASE,
)

_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))", re.MULTILINE)


def _norm(path: str) -> str:
    return (path or "").replace("\\", "/").lstrip("./").lstrip("/").strip()


def _matches(path: str, patterns: Sequence[str]) -> bool:
    p = _norm(path)
    if not p:
        return False
    for pat in patterns:
        if pat.endswith("/"):
            if p == pat.rstrip("/") or p.startswith(pat):
                return True
        elif p == pat:
            return True
    return False


def _module_prefixes(patterns: Sequence[str]) -> List[str]:
    """Turn path patterns into dotted-module prefixes for import detection
    (``core/prompt/`` → ``core.prompt``, ``memory/code_proposal.py`` →
    ``memory.code_proposal``)."""
    out: List[str] = []
    for pat in patterns:
        m = pat.rstrip("/")
        if m.endswith(".py"):
            m = m[:-3]
        if m.endswith((".yaml", ".sh")):
            continue  # not importable
        out.append(m.replace("/", "."))
    return out


def _imports_any(code_texts: Iterable[str], module_prefixes: Sequence[str]) -> bool:
    for text in code_texts:
        if not text:
            continue
        for m in _IMPORT_RE.finditer(text):
            mod = m.group(1) or m.group(2) or ""
            mod = mod.strip()
            for pref in module_prefixes:
                if mod == pref or mod.startswith(pref + "."):
                    return True
    return False


def classify_proposal(
    affected_files: Sequence[str],
    *,
    code_texts: Sequence[str] = (),
    title: str = "",
    description: str = "",
    proposal_type: ProposalType = ProposalType.FEATURE,
) -> Tuple[bool, RiskLevel]:
    """Compute ``(touches_core_system, risk_level)`` for a proposal.

    A path under the safety or supervision layer — or a NEW file that imports
    from one (refactor-and-move / re-export) — forces ``CRITICAL`` with no
    exceptions. A path under the core orchestration/memory spine is at least
    ``HIGH``. Otherwise risk falls back to keyword/type heuristics.
    """
    files = list(affected_files or [])

    touches_critical = (
        any(_matches(f, _CRITICAL_PATHS) for f in files)
        or _imports_any(code_texts, _module_prefixes(_CRITICAL_PATHS))
    )
    touches_core_spine = (
        any(_matches(f, CORE_SYSTEM_PATHS) for f in files)
        or _imports_any(code_texts, _module_prefixes(CORE_SYSTEM_PATHS))
    )
    touches_core_system = touches_critical or touches_core_spine

    if touches_critical or _CRITICAL_RISK_KEYWORDS.search(title or ""):
        risk = RiskLevel.CRITICAL
    elif touches_core_spine or _HIGH_RISK_KEYWORDS.search(title or ""):
        risk = RiskLevel.HIGH
    elif proposal_type in (ProposalType.DOCS, ProposalType.TEST):
        risk = RiskLevel.LOW
    else:
        risk = RiskLevel.MEDIUM

    if touches_critical:
        logger.info("proposal classified CRITICAL (touches safety/supervision layer): %s",
                    [f for f in files if _matches(f, _CRITICAL_PATHS)] or "via-import")
    return touches_core_system, risk


def requires_human_ack(risk_level: RiskLevel, touches_core_system: bool) -> bool:
    """Whether FORWARD-progressing a proposal (approve / mark-built / merge) should
    require an explicit human acknowledgement.

    True for HIGH or CRITICAL risk, or any core-system touch — the supervision
    metadata becomes an action the reviewer must take, not just a label. The GUI
    gates Approve / Mark Built on this; rejection is never gated (rejecting is
    always safe). Single source of truth so the gate can't drift from the
    classifier that produced the fields."""
    return bool(touches_core_system) or risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
