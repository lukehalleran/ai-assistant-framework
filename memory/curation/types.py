"""Autonomous curation data models (docs/AUTONOMOUS_CURATION_DESIGN.md).

Prime directive: the curator must never be wrong — always prefer inaction
over a wrong action, and every action taken must be reversible. That is
encoded structurally here:

- Instrument ladder: read-time neutralization > reversible metadata >
  additive > deletion. DELETE proposals can never auto-apply and are only
  executed by an explicit human click.
- Confidence: a single LLM opinion is NEVER sufficient for an automatic
  write (dual-model agreement or a deterministic deployed predicate).
- Every ItemChange carries its pre-image (`before`) so any applied proposal
  can be undone.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Instrument(str, Enum):
    """Ordered preference — curators must use the highest that works."""
    READ_TIME = "read_time"    # no store write at all (documentation of a code-level neutralization)
    METADATA = "metadata"      # reversible metadata flip / content repair with pre-image
    ADDITIVE = "additive"      # new tags/entries only
    DELETE = "delete"          # NEVER automatic; human click only


class Confidence(str, Enum):
    DETERMINISTIC = "deterministic"  # deployed predicate, regression-tested
    DUAL_LLM = "dual_llm"            # two independent models agreed
    SINGLE_LLM = "single_llm"        # one model — queue-only, never auto


class CuratorMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"   # scan + journal only; nothing queued, nothing shown
    QUEUE = "queue"     # proposals appear in the Curation Center for one-click review
    AUTO = "auto"       # auto-apply eligible (still capped by the global max_mode ceiling)


class ProposalStatus(str, Enum):
    PENDING = "pending"
    APPLIED = "applied"
    DISMISSED = "dismissed"
    FAILED = "failed"
    UNDONE = "undone"


class ItemChange(BaseModel):
    """One reversible change to one item in one store.

    `before` holds the pre-image of every field this change touches —
    apply() must populate anything it overwrites so undo() can restore it.
    """
    store: str                       # "chroma:<collection>" | "profile"
    doc_id: str
    change_type: str                 # "set_metadata" | "replace_content" | "quarantine" | "supersede_profile_fact"
    before: Dict[str, Any] = Field(default_factory=dict)
    after: Dict[str, Any] = Field(default_factory=dict)


class CurationProposal(BaseModel):
    proposal_id: str
    curator: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    instrument: Instrument
    confidence: Confidence
    title: str                       # one-line human summary for the queue card
    evidence: str                    # why — shown in the UI, must stand alone
    items: List[ItemChange] = Field(default_factory=list)
    batch: bool = False              # True → one card for the whole item list
    status: ProposalStatus = ProposalStatus.PENDING
    status_detail: str = ""          # dismissal reason / failure message
    resolved_at: Optional[str] = None


class SentinelResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class ScanReport(BaseModel):
    """What one engine run did — journaled and shown in the Activity view."""
    started_at: str
    finished_at: str = ""
    curators_run: List[str] = Field(default_factory=list)
    proposals_queued: int = 0
    proposals_shadowed: int = 0
    sentinel_failures: List[SentinelResult] = Field(default_factory=list)
    halted_curators: Dict[str, str] = Field(default_factory=dict)  # curator -> reason
    errors: Dict[str, str] = Field(default_factory=dict)           # curator -> error
