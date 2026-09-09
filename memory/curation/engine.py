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
- Every apply syncs all pre-images before mutation and journals its outcome.

The queue is DERIVED state (tone_state doctrine): persisted via atomic
write, loaded leniently, then reconciled with append-only journal snapshots.
An unfinished apply or undo is exposed as interrupted for explicit recovery.
Legacy activity-only journal entries still require a surviving queue.
"""

import os
import threading
import uuid
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional, Protocol

from memory.curation.adapters import apply_change, prepare_change, revert_change
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


class CurationBusyError(ValueError):
    """An existing worker owns the engine, including after an HTTP timeout."""


def _serialized(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        if not self._operation_lock.acquire(blocking=False):
            raise CurationBusyError("Curation is busy; the current operation is still running")
        try:
            return method(self, *args, **kwargs)
        finally:
            self._operation_lock.release()
    return locked


class Curator(Protocol):
    name: str

    def scan(self, stores: "StoreBundle") -> List[CurationProposal]: ...

    def sentinels(self, stores: "StoreBundle") -> List[SentinelResult]: ...


class StoreBundle:
    """Live store handles passed to curators. Curators must treat these as
    READ-ONLY — only engine.apply() writes, through the adapters."""

    def __init__(self, chroma_store=None, user_profile=None, corpus_manager=None,
                 graph_memory=None):
        self.chroma_store = chroma_store
        self.user_profile = user_profile
        self.corpus_manager = corpus_manager
        # Knowledge graph (2026-09-05): read by the temporal-node curator,
        # written only through adapters.apply_change (node quarantine flag).
        self.graph_memory = graph_memory


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
        self._operation_lock = threading.RLock()
        self._load_queue()
        self._recover_from_journal()

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
        atomic_write_json(self.queue_path, payload)

    def _recover_from_journal(self) -> None:
        # Full snapshots are new in B2. Older activity-only records are not
        # sufficient to reconstruct a proposal; a surviving queue still works.
        for record in self.journal.records():
            raw = record.get("proposal")
            if not isinstance(raw, dict):
                continue
            recovered = CurationProposal(**raw)
            existing = self._proposals.get(recovered.proposal_id)
            if existing is None or recovered.revision >= existing.revision:
                self._proposals[recovered.proposal_id] = recovered

    def _store_kwargs(self):
        return dict(chroma_store=self.stores.chroma_store,
                    user_profile=self.stores.user_profile,
                    graph_memory=self.stores.graph_memory)

    @staticmethod
    def _transition(p, status, detail=""):
        p.revision += 1
        p.status = status
        p.status_detail = detail
        p.resolved_at = datetime.now().isoformat()

    def _record_state(self, event, p, **detail):
        self.journal.record(event, proposal_id=p.proposal_id, curator=p.curator,
                            title=p.title, proposal=p.model_dump(mode="json"),
                            items=[i.model_dump(mode="json") for i in p.items], **detail)

    def _record_failure(self, event, p, **detail):
        # Called while already propagating an operation error. Try both
        # independent recovery copies; never turn a persistence failure green.
        for save in (lambda: self._record_state(event, p, **detail), self._save_queue):
            try:
                save()
            except Exception:
                logger.exception("[Curation] Failed to persist recovery status")

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    def mode_for(self, curator_name: str) -> CuratorMode:
        return _cap_mode(
            self.curator_modes.get(curator_name, CuratorMode.QUEUE), self.max_mode
        )

    @_serialized
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
            if existing.status not in (ProposalStatus.PENDING, ProposalStatus.APPLIED,
                                       ProposalStatus.INTERRUPTED):
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
            proposal=p.model_dump(mode="json"),
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
            # A failed durable write must also fail an auto-apply scan, not
            # return a successful report after apply() has raised.
            self.apply(p.proposal_id, actor="auto")
            self._auto_applied_this_run += 1
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
            if store_ref == "graph" and self.stores.graph_memory is not None:
                return int(self.stores.graph_memory.node_count())
            if store_ref == "profile" and self.stores.user_profile is not None:
                cats = (getattr(self.stores.user_profile, "profile", None) or {}).get("categories", {})
                return sum(len(v) for v in cats.values() if isinstance(v, list)) or None
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    # Queue operations (called from the API / UI)
    # ------------------------------------------------------------------

    @_serialized
    def pending(self) -> List[CurationProposal]:
        return sorted(
            (p.model_copy(deep=True) for p in self._proposals.values()
             if p.status in (ProposalStatus.PENDING, ProposalStatus.INTERRUPTED)),
            key=lambda p: p.created_at,
        )

    def get(self, proposal_id: str) -> Optional[CurationProposal]:
        return self._proposals.get(proposal_id)

    @_serialized
    def apply(self, proposal_id: str, actor: str = "human") -> CurationProposal:
        p = self._proposals.get(proposal_id)
        if p is None:
            raise KeyError(proposal_id)
        if p.status != ProposalStatus.PENDING:
            raise ValueError(f"proposal is {p.status.value}, not pending")
        if p.instrument == Instrument.DELETE and actor != "human":
            raise ValueError("DELETE proposals require a human")
        if any(q.status == ProposalStatus.INTERRUPTED for q in self._proposals.values()):
            raise ValueError("Undo interrupted curation before applying another proposal")

        # Capture the whole batch before the first write. Repeated targets
        # would need sequential pre-images, so reject ambiguous batches.
        try:
            targets = [(i.store, i.doc_id) for i in p.items]
            if not targets or len(set(targets)) != len(targets):
                raise ValueError("A curation batch must contain distinct, nonempty targets")
            for item in p.items:
                prepare_change(item, **self._store_kwargs())
        except Exception as e:
            self._transition(p, ProposalStatus.FAILED, str(e))
            self._record_failure("apply_failed", p, error=str(e), rollback=[], actor=actor)
            raise

        self._transition(p, ProposalStatus.INTERRUPTED,
                         "Apply interrupted or unfinished; review and undo to restore the pre-image")
        try:
            self._save_queue()
            self._record_state("apply_started", p, actor=actor)
        except Exception as e:
            self._transition(p, ProposalStatus.FAILED, f"Apply not started: {e}")
            self._record_failure("apply_failed", p, error=str(e), rollback=[], actor=actor)
            raise

        attempted = []
        try:
            for item in p.items:
                attempted.append(item)  # include the item that may write then raise
                apply_change(item, prepared=True, **self._store_kwargs())
            self._transition(p, ProposalStatus.APPLIED)
            self._record_state("applied", p, actor=actor)
        except Exception as e:
            outcomes = []
            for item in reversed(attempted):
                outcome = dict(store=item.store, doc_id=item.doc_id, restored=False)
                try:
                    revert_change(item, **self._store_kwargs())
                    outcome["restored"] = True
                except Exception as rollback_error:
                    outcome["error"] = str(rollback_error)
                outcomes.append(outcome)
            status = (ProposalStatus.FAILED if all(x["restored"] for x in outcomes)
                      else ProposalStatus.INTERRUPTED)
            self._transition(p, status, str(e))
            self._record_failure("apply_failed", p, error=str(e), rollback=outcomes, actor=actor)
            raise

        # The journal's committed snapshot is authoritative if this final
        # queue refresh fails. Propagate the error; restart can still undo.
        self._save_queue()
        return p

    @_serialized
    def dismiss(self, proposal_id: str, reason: str = "") -> CurationProposal:
        p = self._proposals.get(proposal_id)
        if p is None:
            raise KeyError(proposal_id)
        if p.status != ProposalStatus.PENDING:
            raise ValueError(f"proposal is {p.status.value}, not pending")
        self._transition(p, ProposalStatus.DISMISSED, reason)
        self._record_state("dismissed", p, reason=reason, confidence=p.confidence.value)
        self._save_queue()
        # Dismissals are the trust-ladder signal: a dismissed DETERMINISTIC
        # proposal means the rule is wrong — that curator must not graduate.
        return p

    @_serialized
    def undo(self, proposal_id: str) -> CurationProposal:
        p = self._proposals.get(proposal_id)
        if p is None:
            raise KeyError(proposal_id)
        if p.status not in (ProposalStatus.APPLIED, ProposalStatus.INTERRUPTED):
            raise ValueError(f"proposal is {p.status.value}, not applied")
        self._transition(p, ProposalStatus.INTERRUPTED,
                         "Undo interrupted or unfinished; retry undo to finish recovery")
        self._save_queue()
        self._record_state("undo_started", p)
        try:
            for item in reversed(p.items):
                revert_change(item, **self._store_kwargs())
            self._transition(p, ProposalStatus.UNDONE)
            self._record_state("undone", p)
        except Exception as e:
            self._transition(p, ProposalStatus.INTERRUPTED, f"Undo incomplete: {e}")
            self._record_failure("undo_failed", p, error=str(e))
            raise
        self._save_queue()
        return p


def new_proposal_id() -> str:
    return f"cur_{uuid.uuid4().hex[:12]}"
