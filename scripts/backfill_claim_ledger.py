#!/usr/bin/env python3
"""Backfill the user-claim ledger from the last N weeks of the corpus (dry-run first).

Incident 2026-10-20; class: BC-75, BC-52. The live write (memory/claim_ledger_writer.py)
only sees turns from the moment it is enabled; this script seeds the store with the
user's recent statements so the first live turn already has context. It reuses
``scripts/claim_ledger_baseline.py`` for everything that is not new: corpus
selection (same exclusions), the extraction phase (``claim_ledger.extract_claims``),
spend tracking and the 402 stop (``Spend`` / ``_wrap_call``), the price flags and the
deployed ``generate_once`` call shape. There is NO reply-check phase.

    python scripts/backfill_claim_ledger.py                          # dry run: turns + estimate, no calls
    python scripts/backfill_claim_ledger.py --apply --max-usd 0.50   # Daemon DOWN; spends credits
    python scripts/backfill_claim_ledger.py --apply --max-usd 0.50 --store /tmp/x.json
    ... --replace    # the target store already has data: copy it to <store>.pre-backfill-<ts>.json first

Safety:
* dry run is the default and makes no model call and writes nothing;
* ``--apply`` refuses while a live Daemon runs (it holds stores in memory and re-saves);
* ``--apply`` refuses when the target store exists and is non-empty unless ``--replace``,
  which first copies the existing file aside (nothing is deleted);
* the store is written ONCE, at the end of a complete run: a budget / credit / error stop
  writes nothing (a ledger missing the newest turns would leave a gap the live write
  can never fill) -- re-run with a larger ``--max-usd``;
* corpus is opened read-only; the default window is the last 28 days.
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory import claim_ledger as cl  # noqa: E402
from utils.daemon_guard import daemon_running  # noqa: E402
import scripts.claim_ledger_baseline as base  # noqa: E402

DEFAULT_DAYS = 28
EXIT_OK, EXIT_USAGE, EXIT_REFUSED = base.EXIT_OK, base.EXIT_USAGE, 1


def target_store(arg: Optional[Path]) -> Path:
    if arg is not None:
        return arg.expanduser().resolve()
    from memory import claim_ledger_store  # lazy import: live-config (the configured store path is read at call time)
    return Path(claim_ledger_store.store_path()).resolve()


def store_is_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def render_plan(plan: dict, skipped: dict, args, window: dict, store: Path, model: str) -> str:
    ex = plan["extraction"]
    state = "non-empty (needs --replace)" if store_is_nonempty(store) else "absent/empty"
    return "\n".join([
        "DRY RUN -- no model calls made, nothing written (pass --apply with --max-usd to execute)",
        f"window: {window['since']} .. {window['until'] or 'now'}  turns: {plan['turns']}  excluded: {skipped}",
        f"model: {model}   token estimate: chars/4; prices USD/1M in/out = {args.price_in}/{args.price_out}; "
        f"assumed {args.est_subjects} open subjects per prompt",
        f"extraction: {plan['extraction_turns']} calls, in~{ex['in_tokens']:,} out~{ex['out_tokens']:,} "
        f"tokens  ~${ex['usd']:.4f}",
        f"target store: {store}  [{state}]",
    ])


async def run_extraction(turns: List[dict], raw_call: base.ModelCall, spend: base.Spend):
    """Extraction over ``turns`` in time order -> ``(ledger, stats, stop)``; no reply check."""
    ledger = cl.ClaimLedger()
    stats = {"turns": len(turns), "extracted": 0, "skipped": {}, "drops": {}, "claims": 0}
    state = {"consecutive_errors": 0}
    call = base._wrap_call(raw_call, spend, state)

    async def extract_call(system: str, user: str) -> str:
        return await call(system, user, cl.EXTRACT_MAX_TOKENS)

    stop: Optional[base.StopRun] = None
    try:
        for t in turns:
            claims, drops, skip = await cl.extract_claims(
                t["user"], t["ts"], ledger, extract_call, turn_id=f"corpus:{t['idx']}")
            if skip:
                base._bump(stats["skipped"], skip)
            else:
                stats["extracted"] += 1
            for d in drops:
                base._bump(stats["drops"], str(d["reason"]).split(" ")[0])
            for c in claims:
                ledger.add(c)
                stats["claims"] += 1
    except base.StopRun as exc:
        stop = exc
    return ledger, stats, stop


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=base.DEFAULT_CORPUS)
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS, help="window length when --since is not given")
    ap.add_argument("--since", type=date.fromisoformat)
    ap.add_argument("--until", type=date.fromisoformat)
    ap.add_argument("--limit", type=int, help="process at most N turns (trial runs)")
    ap.add_argument("--apply", action="store_true", help="make model calls and write the store (Daemon DOWN)")
    ap.add_argument("--max-usd", type=float, help="required with --apply")
    ap.add_argument("--store", type=Path, help="write here instead of the configured store")
    ap.add_argument("--replace", action="store_true",
                    help="allow a non-empty target store; it is copied to <store>.pre-backfill-<ts>.json first")
    ap.add_argument("--model", help="default: claim_ledger.model from config")
    ap.add_argument("--price-in", type=float, default=0.049, help="USD per 1M input tokens (deepseek-v4.1-flash)")
    ap.add_argument("--price-out", type=float, default=0.525, help="USD per 1M output tokens")
    ap.add_argument("--est-subjects", type=int, default=15)
    ap.add_argument("--est-out-extract", type=int, default=90)
    return ap


def main(argv: Optional[List[str]] = None, *,
         call_factory: Optional[Callable[[str], base.ModelCall]] = None,
         today: Optional[date] = None) -> int:
    from config import app_config  # lazy import: live-config
    args = build_parser().parse_args(argv)
    if args.apply and daemon_running():
        print("REFUSED: a live Daemon main.py process is running (or could not be ruled out) - it holds "
              "stores in memory and would re-save over this change. Shut it down first.", file=sys.stderr)
        return EXIT_REFUSED
    if not args.corpus.exists():
        print(f"corpus not found: {args.corpus}", file=sys.stderr)
        return EXIT_USAGE
    if args.apply and args.max_usd is None:
        print("--apply requires --max-usd", file=sys.stderr)
        return EXIT_USAGE
    model = args.model or app_config.CLAIM_LEDGER_MODEL
    since = args.since or ((today or date.today()) - timedelta(days=max(0, args.days)))
    window = {"since": str(since), "until": str(args.until) if args.until else None}
    store = target_store(args.store)

    if args.apply and store_is_nonempty(store) and not args.replace:
        print(f"REFUSED: {store} already has data. Pass --replace to copy it aside "
              f"(<store>.pre-backfill-<ts>.json) and write the backfill instead.", file=sys.stderr)
        return EXIT_REFUSED

    turns, skipped = base.select_turns(base.load_corpus(args.corpus), since, args.until)
    if args.limit:
        turns = turns[: args.limit]

    if not args.apply:
        plan = base.estimate_plan(turns, price_in=args.price_in, price_out=args.price_out,
                                  est_subjects=args.est_subjects, est_out_extract=args.est_out_extract,
                                  est_out_reply=0)
        print(render_plan(plan, skipped, args, window, store, model))
        return EXIT_OK

    raw_call = (call_factory or base.default_call_factory)(model)
    spend = base.Spend(args.max_usd, args.price_in, args.price_out)
    ledger, stats, stop = asyncio.run(run_extraction(turns, raw_call, spend))
    print(f"turns {stats['turns']}  extracted {stats['extracted']}  claims {stats['claims']}  "
          f"drops {stats['drops']}  est spend ${spend.spent:.4f}")
    if stop:
        print(f"STOPPED: {stop} -- nothing was written (a partial ledger would leave a gap). "
              f"Re-run with a larger --max-usd or after the credit issue is fixed.", file=sys.stderr)
        return stop.code
    if store_is_nonempty(store):
        aside = store.with_name(f"{store.name}.pre-backfill-{datetime.now().strftime('%Y%m%dT%H%M%S')}.json")
        shutil.copy2(store, aside)
        print(f"copied existing store to {aside}")
    store.parent.mkdir(parents=True, exist_ok=True)
    ledger.save(str(store))
    print(f"wrote {len(ledger.claims)} claims to {store}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
