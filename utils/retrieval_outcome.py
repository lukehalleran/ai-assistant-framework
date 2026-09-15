"""Shared retrieval-outcome vocabulary: unavailable is not zero.

A leaf module (stdlib only; no project imports, no logging, no side effects
on import) so any read site — memory stores, knowledge managers, gatherers —
can adopt it without a dependency risk. It defines the vocabulary once and
reuses the state names already established by
``core.insight.coordinator.CHANNEL_STATES`` (pinned by a parity test in
tests/unit/test_retrieval_outcome.py) instead of inventing a second one.

F1 has no production caller: it only defines the vocabulary. F2 onward adopt
it at read sites; see docs/execution/generalization/failure_outcome_design.md,
"F1 Ready decision: one status vocabulary".

``StoreWriteError`` (F10a) is the write-side counterpart to ``RetrievalError``.
"""
from __future__ import annotations

RETRIEVAL_STATES = frozenset({"succeeded", "no_results", "unavailable", "failed"})
"""The four outcomes a read site can report.

``succeeded``: the read ran and returned items.
``no_results``: the read ran and returned nothing.
``unavailable``: the read did not run or could not finish (not configured,
disabled, timed out) — carries a ``reason`` saying which.
``failed``: the read raised, or the store/provider errored — carries a
``reason``.
"""


class RetrievalError(RuntimeError):
    """The typed failure a read site raises instead of swallowing an error.

    ``source`` names the store/provider/section that failed (e.g. "tavily",
    "chroma", "obsidian"). ``reason`` must be a short constant-style label or
    exception class name (e.g. "timeout", "ConnectionError") — never
    free-form user-facing text, so callers can safely log or aggregate it.
    """

    def __init__(self, *, source: str, reason: str) -> None:
        self.source = source
        self.reason = reason
        super().__init__(f"{source}: {reason}")


class StoreWriteError(RuntimeError):
    """The typed failure a WRITE site raises instead of swallowing an error.

    ``source``/``reason`` follow ``RetrievalError``'s contract (labels only,
    never free-form text). ``None`` stays the documented deliberate skip;
    this is only for a write that was attempted and failed. Unrelated to
    ``RetrievalError`` (a read-side failure).
    """

    def __init__(self, *, source: str, reason: str) -> None:
        self.source = source
        self.reason = reason
        super().__init__(f"{source}: {reason}")


class OutcomeList(list):
    """A ``list`` that also carries why it is empty (or a partial read).

    Equality and truthiness stay plain-list: an empty ``OutcomeList`` (of any
    status) equals ``[]`` and is falsy, so existing ``if not x`` callers are
    unchanged until they are migrated to read ``.status`` instead.

    Slicing, ``list(x)``, ``+`` and ``.copy()`` all return a plain ``list``
    and DROP ``status``/``reason`` — this is standard ``list`` subclass
    behaviour, not something this class adds. Callers must read ``.status``
    immediately, before any such transform.
    """

    def __init__(self, items=(), *, status: str | None = None, reason: str = "") -> None:
        items = list(items)
        if status is None:
            status = "succeeded" if items else "no_results"
        elif status not in RETRIEVAL_STATES:
            raise ValueError(f"unknown retrieval status: {status!r}")
        if status == "no_results" and items:
            raise ValueError("status 'no_results' cannot carry items")
        super().__init__(items)
        self.status = status
        self.reason = reason

    @classmethod
    def failed(cls, reason: str, items=()) -> "OutcomeList":
        return cls(items, status="failed", reason=reason)

    @classmethod
    def unavailable(cls, reason: str, items=()) -> "OutcomeList":
        return cls(items, status="unavailable", reason=reason)


def outcome_status(value) -> tuple[str, str]:
    """``(status, reason)`` for any read-site return value.

    The one reader later batches use at their boundaries: an ``OutcomeList``
    reports its own ``(status, reason)``; a plain list reports
    ``("succeeded", "")`` when non-empty, else ``("no_results", "")``; ``None``
    or ``""`` reports ``("no_results", "")``. Nothing else.
    """
    if isinstance(value, OutcomeList):
        return (value.status, value.reason)
    return ("succeeded", "") if value else ("no_results", "")
