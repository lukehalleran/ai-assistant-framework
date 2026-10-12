#!/usr/bin/env python3
"""Owner-side reader for a Daemon "Report a problem" bundle.

Usage: python scripts/read_report.py <bundle.json>

Read-only; no network.  Prints app info, the turn summary as aligned
key: value lines, context turns as a compact table, rollup headline numbers,
and the full prompt/response if the sender included them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_TABLE_COLS = ("ts", "mode", "intent", "tone_level", "gate_triggered", "wall_elapsed_s")


def _fmt(value) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return "-" if value is None else str(value)


def render(bundle: dict) -> str:
    out: list[str] = []
    out.append(f"== Daemon report (schema {bundle.get('schema_version')}) created {bundle.get('created_at')}")
    if bundle.get("description"):
        out += ["", "-- description", bundle["description"]]
    app = bundle.get("app") or {}
    out += ["", "-- app"] + [f"  {k}: {_fmt(v)}" for k, v in app.items()]
    turn = bundle.get("turn") or {}
    width = max((len(k) for k in turn), default=0)
    out += ["", "-- turn"] + [f"  {k.ljust(width)} : {_fmt(v)}" for k, v in turn.items()]
    ctx = bundle.get("context_turns") or []
    out += ["", f"-- context turns ({len(ctx)})"]
    if ctx:
        rows = [[_fmt(t.get(c)) for c in _TABLE_COLS] for t in ctx]
        widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(_TABLE_COLS)]
        out.append("  " + "  ".join(c.ljust(w) for c, w in zip(_TABLE_COLS, widths)))
        out += ["  " + "  ".join(v.ljust(w) for v, w in zip(r, widths)) for r in rows]
    roll = bundle.get("rollup")
    out += ["", "-- rollup"]
    if isinstance(roll, dict):
        counts = roll.get("counts") or {}
        wall = roll.get("wall_elapsed_s") or {}
        out.append(f"  selected={counts.get('selected')} examined={counts.get('records_examined')}")
        out.append(f"  wall_elapsed_s: {_fmt(wall)}")
    else:
        out.append(f"  unavailable ({bundle.get('rollup_error', 'none')})")
    full = bundle.get("full")
    if isinstance(full, dict):
        out += ["", "-- full prompt", str(full.get("prompt", "")), "", "-- full response", str(full.get("response", ""))]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        bundle = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"cannot read bundle: {type(exc).__name__}", file=sys.stderr)
        return 1
    if not isinstance(bundle, dict):
        print("not a report bundle", file=sys.stderr)
        return 1
    print(render(bundle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
