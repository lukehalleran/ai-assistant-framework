"""CurationEngine — runs curators, disposes proposals, applies with undo.

Disposition rules (docs/AUTONOMOUS_CURATION_DESIGN.md):
- Curators only PROPOSE (their scan() never writes).
- The engine decides shadow / queue / auto per curator mode, capped by the
  global ``max_mode`` ceiling (config `curation.max_mode`, currently
  "queue" — auto is built but locked off until curators graduate).
- DELETE-instrument proposals can never auto-apply regardless of mode.
- Sentinel failure aborts the ENTIRE batch for that curator.
- Anomaly halt: proposals touching more than ANOMALY_FRACTION of a store in
  one run are never auto-applied — mass action is when a rule is most
  likely wrong.
- Rate cap: at most AUTO_RATE_CAP auto-applies per run; overflow queues.
- Every apply captures pre-images (adapters) and journals an undo record.

The queue is DERIVED state (tone_state doctrine): persisted via atomic
write, loaded leniently — a corrupt queue file cold-starts empty and the
next scan regenerates it. Pre-images for APPLIED proposals also live in the
journal, which is append-only.
"""

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from memory.curation.adapters import apply_change, revert_change
from memory.curation.journal import CurationJournal
from memory.curation.types import (
    Confidence,
    CurationProposal,
    CuratorMode,
    Instrument,
    ProposalStatus,
    ScanReport,
    SentinelResult,
)
from utils.logging_utils import get_logger
from utils.safe_json import atomic_write_json

logger = get_logger("curation_engine")

_PROD_QUEUE_PATH = os.path.join("data", "curation_queue.json")
_TEST_QUEUE_PATH = os.path.join("data", "test_curation_queue.json")
_DEFAULT_QUEUE_PATH = _PROD_QUEUE_PATH


def resolve_queue_path(queue_path: str = "") -> str:
    """Explicit path wins; otherwise the module default — redirected to a
    test-only file when a DAEMON_TEST_MODE process would otherwise write the
    owner's prod queue (same doctrine as journal.resolve_journal_path)."""
    resolved = queue_path or _DEFAULT_QUEUE_PATH
    if not queue_path and os.getenv("DAEMON_TEST_MODE") and resolved == _PROD_QUEUE_PATH:
        return _TEST_QUEUE_PATH
    return resolved
QUEUE_SCHEMA_VERSION = 1

_MODE_ORDER = [CuratorMode.OFF, CuratorMode.SHADOW, CuratorMode.QUEUE, CuratorMode.AUTO]


class Curator(Protocol):
    name: str

    def scan(self, stores: "StoreBundle") -> List[CurationProposal]: ...

    def sentinels(self, stores: "StoreBundle") -> List[SentinelResult]: ...


class StoreBundle:
    """Live store handles passed to curators. Curators must treat these as
    READ-ONLY — only engine.apply() writes, through the adapters."""

    def __init__(self, chroma_store=None, user_profile=None, corpus_manager=None):
        self.chroma_store = chroma_store
        self.user_profile = user_profile
        self.corpus_manager = corpus_manager


def _cap_mode(mode: CuratorMode, ceiling: CuratorMode) -> CuratorMode:
    return mode if _MODE_ORDER.index(mode) <= _MODE_ORDER.index(ceiling) else ceiling


class CurationEngine:
    def __init__(
        self,
        stores: StoreBundle,
        *,
        queue_path: str = "",
        journal: Optional[CurationJournal] = None,
        max_mode: str = "queue",
        curator_modes: Optional[Dict[str, str]] = None,
        auto_rate_cap: int = 25,
        anomaly_fraction: float = 0.05,
        max_queue_items_per_curator: int = 50,
    ):
        self.stores = stores
        # Late-bound default so the test sandbox can repoint the module attr.
        self.queue_path = resolve_queue_path(queue_path)
        self.journal = journal or CurationJournal()
        self.max_mode = CuratorMode(max_mode)
        self.curator_modes = {k: CuratorMode(v) for k, v in (curator_modes or {}).items()}
        self.auto_rate_cap = auto_rate_cap
        self.anomaly_fraction = anomaly_fraction
        self.max_queue_items_per_curator = max_queue_items_per_curator
        self._curators: List[Curator] = []
        self._proposals: Dict[str, CurationProposal] = {}
        self._auto_applied_this_run = 0
        self._load_queue()

    # ------------------------------------------------------------------
    # Registration / persistence
    # ------------------------------------------------------------------

    def register(self, curator: Curator) -> None:
        self._curators.append(curator)

    def _load_queue(self) -> None:
        """Lenient load — the queue is derived state; corrupt = cold start."""
        try:
            import json
            if not os.path.exists(self.queue_path):
                return
            with open(self.queue_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for raw in payload.get("proposals", []):
                try:
                    p = CurationProposal(**raw)
                    self._proposals[p.proposal_id] = p
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"[Curation] queue load failed — cold start: {e}")

    def _save_queue(self) -> None:
        payload = {
            "schema_version": QUEUE_SCHEMA_VERSION,
            "proposals": [p.model_dump(mode="json") for p in self._proposals.values()],
        }
        try:
            atomic_write_json(self.queue_path, payload)
        except Exception as e:
            logger.warning(f"[Curation] queue save failed (non-fatal): {e}")

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    def mode_for(self, curator_name: str) -> CuratorMode:
        return _cap_mode(
            self.curator_modes.get(curator_name, CuratorMode.QUEUE), self.max_mode
        )

    def run_scan(self) -> ScanReport:
        report = ScanReport(started_at=datetime.now().isoformat())
        self._auto_applied_this_run = 0
        self.journal.record("scan_started")

        for curator in self._curators:
            mode = self.mode_for(curator.name)
            if mode == CuratorMode.OFF:
                continue
            report.curators_run.append(curator.name)

            # Sentinels first — any failure aborts this curator's batch.
            try:
                sentinel_results = curator.sentinels(self.stores)
            except Exception as e:
                report.errors[curator.name] = f"sentinel crash: {e}"
                self.journal.record("sentinel_crash", curator=curator.name, error=str(e))
                continue
            failed = [s for s in sentinel_results if not s.passed]
            if failed:
                report.sentinel_failures.extend(failed)
                report.halted_curators[curator.name] = (
                    f"sentinel failed: {failed[0].name}"
                )
                self.journal.record(
                    "sentinel_abort", curator=curator.name,
                    failures=[s.model_dump() for s in failed],
                )
                continue

            try:
                proposals = curator.scan(self.stores) or []
            except Exception as e:
                report.errors[curator.name] = str(e)
                self.journal.record("scan_error", curator=curator.name, error=str(e))
                continue

            # De-dup against existing pending/applied proposals (same curator
            # + same item set) so repeat scans don't stack duplicate cards.
            proposals = [p for p in proposals if not self._is_duplicate(p)]

            # Queue-card cap: item-level curators can't flood the review UI.
            non_batch = [p for p in proposals if not p.batch]
            if len(non_batch) > self.max_queue_items_per_curator:
                kept = non_batch[: self.max_queue_items_per_curator]
                dropped = len(non_batch) - len(kept)
                self.journal.record(
                    "queue_cap", curator=curator.name, dropped=dropped,
                    note="rescan after resolving current cards",
                )
                proposals = [p for p in proposals if p.batch] + kept

            for p in proposals:
                disposition = self._dispose(p, mode)
                if disposition == "shadow":
                    report.proposals_shadowed += 1
                else:
                    report.proposals_queued += 1

        report.finished_at = datetime.now().isoformat()
        self._save_queue()
        self.journal.record(
            "scan_finished",
            queued=report.proposals_queued,
            shadowed=report.proposals_shadowed,
            halted=report.halted_curators,
            errors=report.errors,
        )
        return report

    def _is_duplicate(self, p: CurationProposal) -> bool:
        sig = (p.curator, tuple(sorted(i.doc_id for i in p.items)))
        for existing in self._proposals.values():
            if existing.status not in (ProposalStatus.PENDING, ProposalStatus.APPLIED):
                continue
            if (existing.curator, tuple(sorted(i.doc_id for i in existing.items))) == sig:
                return True
        return False

    def _dispose(self, p: CurationProposal, mode: CuratorMode) -> str:
        if mode == CuratorMode.SHADOW:
            self.journal.record(
                "proposal_shadowed", curator=p.curator, title=p.title,
                items=len(p.items),
            )
            return "shadow"

        self._proposals[p.proposal_id] = p
        self.journal.record(
            "proposal_queued", curator=p.curator, proposal_id=p.proposal_id,
            title=p.title, items=len(p.items), instrument=p.instrument.value,
            confidence=p.confidence.value,
        )
        # AUTO disposition (built, currently locked off by max_mode="queue"):
        # only reversible instruments, only deterministic-or-better evidence,
        # bounded by the rate cap and the anomaly fraction.
        if (
            mode == CuratorMode.AUTO
            and p.instrument in (Instrument.METADATA, Instrument.ADDITIVE)
            and p.confidence in (Confidence.DETERMINISTIC, Confidence.DUAL_LLM)
            and not self._anomalous(p)
            and self._auto_applied_this_run < self.auto_rate_cap
        ):
            try:
                self.apply(p.proposal_id, actor="auto")
                self._auto_applied_this_run += 1
            except Exception as e:
                logger.warning(f"[Curation] auto-apply failed, left queued: {e}")
        return "queue"

    def _anomalous(self, p: CurationProposal) -> bool:
        """A proposal touching too much of a store never auto-applies."""
        counts: Dict[str, int] = {}
        for item in p.items:
            counts[item.store] = counts.get(item.store, 0) + 1
        for store_ref, n in counts.items():
            total = self._store_size(store_ref)
            if total and n / total > self.anomaly_fraction:
                return True
        return False

    def _store_size(self, store_ref: str) -> Optional[int]:
        try:
            if store_ref.startswith("chroma:") and self.stores.chroma_store:
                coll = self.stores.chroma_store._get_collection(
                    store_ref.split(":", 1)[1]
                )
                return coll.count() if coll is not None else None
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    # Queue operations (called from the API / UI)
    # ------------------------------------------------------------------

    def pending(self) -> List[CurationProposal]:
        return sorted(
            (p for p in self._proposals.values() if p.status == ProposalStatus.PENDING),
            key=lambda p: p.created_at,
        )

    def get(self, proposal_id: str) -> Optional[CurationProposal]:
        return self._proposals.get(proposal_id)

    def apply(self, proposal_id: str, actor: str = "human") -> CurationProposal:
        p = self._proposals.get(proposal_id)
        if p is None:
            raise KeyError(proposal_id)
        if p.status != ProposalStatus.PENDING:
            raise ValueError(f"proposal is {p.status.value}, not pending")
        if p.instrument == Instrument.DELETE and actor != "human":
            raise ValueError("DELETE proposals require a human")

        applied = []
        try:
            for item in p.items:
                apply_change(
                    item,
                    chroma_store=self.stores.chroma_store,
                    user_profile=self.stores.user_profile,
                )
                applied.append(item)
        except Exception as e:
            # Roll back what landed — an apply is all-or-nothing.
            for item in reversed(applied):
                try:
                    revert_change(
                        item,
                        chroma_store=self.stores.chroma_store,
                        user_profile=self.stores.user_profile,
                    )
                except Exception as re:
                    logger.error(f"[Curation] rollback failed for {item.doc_id}: {re}")
            p.status = ProposalStatus.FAILED
            p.status_detail = str(e)
            p.resolved_at = datetime.now().isoformat()
            self._save_queue()
            self.journal.record(
                "apply_failed", proposal_id=p.proposal_id, curator=p.curator,
                error=str(e), actor=actor,
            )
            raise

        p.status = ProposalStatus.APPLIED
        p.resolved_at = datetime.now().isoformat()
        self._save_queue()
        # Journal carries the full pre-images — the durable undo record.
        self.journal.record(
            "applied", proposal_id=p.proposal_id, curator=p.curator,
            title=p.title, actor=actor,
            items=[i.model_dump(mode="json") for i in p.items],
        )
        return p

    def dismiss(self, proposal_id: str, reason: str = "") -> CurationProposal:
        p = self._proposals.get(proposal_id)
        if p is None:
            raise KeyError(proposal_id)
        if p.status != ProposalStatus.PENDING:
            raise ValueError(f"proposal is {p.status.value}, not pending")
        p.status = ProposalStatus.DISMISSED
        p.status_detail = reason
        p.resolved_at = datetime.now().isoformat()
        self._save_queue()
        # Dismissals are the trust-ladder signal: a dismissed DETERMINISTIC
        # proposal means the rule is wrong — that curator must not graduate.
        self.journal.record(
            "dismissed", proposal_id=p.proposal_id, curator=p.curator,
            title=p.title, reason=reason, confidence=p.confidence.value,
        )
        return p

    def undo(self, proposal_id: str) -> CurationProposal:
        p = self._proposals.get(proposal_id)
        if p is None:
            raise KeyError(proposal_id)
        if p.status != ProposalStatus.APPLIED:
            raise ValueError(f"proposal is {p.status.value}, not applied")
        for item in reversed(p.items):
            revert_change(
                item,
                chroma_store=self.stores.chroma_store,
                user_profile=self.stores.user_profile,
            )
        p.status = ProposalStatus.UNDONE
        p.resolved_at = datetime.now().isoformat()
        self._save_queue()
        self.journal.record(
            "undone", proposal_id=p.proposal_id, curator=p.curator, title=p.title
        )
        return p


def new_proposal_id() -> str:
    return f"cur_{uuid.uuid4().hex[:12]}"
