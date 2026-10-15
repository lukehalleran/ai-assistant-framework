"""Summary coverage by CONTENT time (BC-95/BC-96 chokepoint).

A summary's identity is the span of conversation it covers (its
``temporal_anchor_start``/``temporal_anchor_end``), never a positional block
index into the rolling corpus and never the time it was created.  Leaf module:
stdlib + logging only.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Iterable, Optional

from utils.date_coerce import to_naive_local

logger = logging.getLogger(__name__)

Range = tuple[datetime, datetime]


def _parse(value: Any) -> Optional[datetime]:
    """Parse an ISO string/datetime to naive local time; None when unparseable."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return to_naive_local(dt)


def _is_consolidator_md(md: dict) -> bool:
    tags = str(md.get("tags") or "").lower()
    return "source:consolidator" in tags or "summary:consolidated" in tags


def ranges_from_metadatas(metadatas: Iterable[dict]) -> list[Range]:
    """Content ranges of consolidator summaries (quarantined ones included: their
    content is still covered).  Rows without parseable anchors are skipped."""
    out: list[Range] = []
    for md in metadatas:
        md = md or {}
        if not _is_consolidator_md(md):
            continue
        start = _parse(md.get("temporal_anchor_start"))
        end = _parse(md.get("temporal_anchor_end"))
        if start is None or end is None:
            continue
        if end < start:
            start, end = end, start
        out.append((start, end))
    return out


def covered_ranges(chroma_store: Any) -> Optional[list[Range]]:
    """All consolidator-summary content ranges, or None when unreadable.

    None means "unknown": callers must not summarize blind.  [] means none exist.
    """
    if chroma_store is None:
        return None
    try:
        rows = chroma_store.list_all("summaries")
    except Exception as e:  # degrades: unreadable coverage -> caller skips summarizing
        logger.warning("[SummaryCoverage] could not read summaries: %s", type(e).__name__)
        return None
    return ranges_from_metadatas((r or {}).get("metadata") or {} for r in rows or [])


def coverage_watermark(ranges: Optional[list[Range]]) -> Optional[datetime]:
    if not ranges:
        return None
    return max(end for _, end in ranges)


def uncovered_blocks(
    entries_sorted: list,
    watermark: Optional[datetime],
    n: int,
    cap: int,
    ts_fn: Callable[[Any], datetime],
) -> list[list]:
    """Full blocks of ``n`` entries newer than the watermark, oldest first, at most ``cap``.

    A partial trailing block waits for more entries.
    """
    n = max(1, int(n))
    pending = [e for e in entries_sorted if watermark is None or ts_fn(e) > watermark]
    full = len(pending) // n
    return [pending[i * n:(i + 1) * n] for i in range(min(full, max(0, int(cap))))]


def remaining_full_blocks(entries_sorted: list, watermark: Optional[datetime], n: int,
                          ts_fn: Callable[[Any], datetime]) -> int:
    n = max(1, int(n))
    return len([e for e in entries_sorted if watermark is None or ts_fn(e) > watermark]) // n


def summary_content_time(md: dict) -> Optional[datetime]:
    """Content time of a summary row (its anchor end), or None for legacy rows."""
    return _parse((md or {}).get("temporal_anchor_end"))


def covers_label(md: dict) -> Optional[str]:
    """'YYYY-MM-DD' or 'YYYY-MM-DD – YYYY-MM-DD' from the anchors; None when absent."""
    md = md or {}
    end = _parse(md.get("temporal_anchor_end"))
    if end is None:
        return None
    start = _parse(md.get("temporal_anchor_start")) or end
    a, b = start.date().isoformat(), end.date().isoformat()
    return a if a == b else f"{a} – {b}"


def overlap_resummaries(rows: list[dict]) -> list[dict]:
    """Rows (id, created, start, end, text) whose [start,end] lies inside the union of
    ranges of EARLIER-CREATED rows (exact time containment, not same-date)."""
    ordered = sorted(rows, key=lambda r: r["created"])
    out = []
    for i, r in enumerate(ordered):
        prior = sorted((p["start"], p["end"]) for p in ordered[:i] if p["created"] < r["created"])
        merged: list[list[datetime]] = []
        for s, e in prior:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        if any(s <= r["start"] and r["end"] <= e for s, e in merged):
            out.append(r)
    return out


def corpus_content_time_raw(entry: dict):
    """Sort/render time of a CORPUS summary entry: ``covers_end`` when present, else
    its creation ``timestamp`` (legacy entries unchanged).  One helper for every
    corpus-side reader (BC-96)."""
    entry = entry or {}
    return entry.get("covers_end") or entry.get("timestamp")


def corpus_anchor_md(entry: dict) -> dict:
    """View a corpus entry's covers_* as chroma-style anchor metadata."""
    entry = entry or {}
    return {"temporal_anchor_start": entry.get("covers_start"),
            "temporal_anchor_end": entry.get("covers_end")}


def entry_time(entry: dict) -> datetime:
    """Naive-local timestamp of a corpus entry (datetime.min when missing/unparseable)."""
    return _parse((entry or {}).get("timestamp")) or datetime.min
