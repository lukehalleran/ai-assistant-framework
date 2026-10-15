#!/usr/bin/env python3
"""Read-only report: summary coverage watermark and re-summaries (BC-95).

Opens chroma.sqlite3 with sqlite URI ``mode=ro`` and NEVER writes to data/, so
no daemon_guard is needed.  Prints the coverage watermark, how many
``conversations`` rows are newer than it, and the candidate RE-SUMMARIES:
consolidator summaries whose [start, end] content range lies inside the union of
ranges of EARLIER-CREATED summaries (exact time overlap, not same-date).
The owner decides any quarantine later via the curation engine.

    python scripts/report_summary_coverage.py [--db PATH] [--out FILE]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory.summary_coverage import _parse, overlap_resummaries  # noqa: E402


def _collection_rows(con: sqlite3.Connection, name: str) -> dict[int, dict]:
    """{row id: {key: value, embedding_id}} for one collection (metadata flattened)."""
    q = (
        "SELECT e.id, e.embedding_id, m.key, m.string_value, m.int_value, m.float_value "
        "FROM embeddings e JOIN segments s ON s.id = e.segment_id "
        "JOIN collections c ON c.id = s.collection "
        "LEFT JOIN embedding_metadata m ON m.id = e.id "
        "WHERE c.name = ? AND s.scope = 'METADATA'"
    )
    rows: dict[int, dict] = {}
    for rid, eid, key, sv, iv, fv in con.execute(q, (name,)):
        r = rows.setdefault(rid, {"embedding_id": eid})
        if key:
            r[key] = sv if sv is not None else (iv if iv is not None else fv)
    return rows


def build_report(db: Path) -> tuple[list[str], list[dict]]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        summ = _collection_rows(con, "summaries")
        convs = _collection_rows(con, "conversations")
    finally:
        con.close()
    cands = []
    for r in summ.values():
        tags = str(r.get("tags") or "").lower()
        if "source:consolidator" not in tags and "summary:consolidated" not in tags:
            continue
        s, e = _parse(r.get("temporal_anchor_start")), _parse(r.get("temporal_anchor_end"))
        c = _parse(r.get("timestamp"))
        if s and e and c:
            cands.append({"id": r["embedding_id"], "created": c, "start": s, "end": e,
                          "text": str(r.get("chroma:document") or "")})
    watermark = max((c["end"] for c in cands), default=None)
    newer = 0
    if watermark:
        for r in convs.values():
            t = _parse(r.get("timestamp"))
            if t and t > watermark:
                newer += 1
    dups = overlap_resummaries(cands)
    lines = [
        f"consolidator summaries with anchors: {len(cands)}",
        f"coverage watermark: {watermark.isoformat() if watermark else 'none'}",
        f"conversations newer than watermark: {newer}",
        f"re-summary candidates: {len(dups)}",
    ]
    for d in dups:
        lines.append(f"{d['id']}\tcreated={d['created'].isoformat(timespec='minutes')}\t"
                     f"covers={d['start'].date()} - {d['end'].date()}\t{d['text'][:80]!r}")
    return lines, dups


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default="data/chroma_db_v4/chroma.sqlite3")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    db = Path(a.db)
    if not db.is_absolute():
        db = ROOT / db
    lines, _ = build_report(db)
    print("\n".join(lines))
    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
