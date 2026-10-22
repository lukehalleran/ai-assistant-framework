#!/usr/bin/env python3
"""Offline baseline: build a user-claim ledger from the corpus and check replies against it.

Incident 2026-10-20; class: BC-75 (self-authored artifact laundered into
evidence), BC-52 (stale claim resurfaces). Companion of ``memory/claim_ledger.py``
(read its docstring first); the deterministic ``report_claim_drift.py`` measured
0 of 2 motivating incidents, so the baseline is measured with the SAME semantic
extractor the ledger will use, and before/after stay comparable.

Walks the corpus window in time order. For each real turn:
  (a) extraction over the USER text -> deterministic validation -> ``ledger.add``;
  (b) reply check over the ASSISTANT text against the ledger as of that turn.
Writes ``--out-dir``: ``ledger.json``, ``findings.jsonl``, ``report.md`` (+
``progress.json``, the resume checkpoint).

READ-ONLY on the corpus (plain ``json.load``); writes only under ``--out-dir``,
which may never sit under a ``data/`` directory. Never touches a live store.

DRY RUN IS THE DEFAULT: without ``--run`` NO model call is made; the script
prints turn counts and the estimated tokens/cost per phase. Token counts are
chars/4 (the repo ``TokenizerManager`` may fetch a tokenizer over the network,
which a dry run must never do). The reply-check estimate is an UPPER bound (it
assumes every turn has open subjects).

    python scripts/claim_ledger_baseline.py --corpus C --since 2026-09-20            # plan only
    python scripts/claim_ledger_baseline.py --corpus C --since 2026-09-20 \\
        --run --model gpt-4o-mini --max-usd 1.50 --out-dir OUT                       # spends credits
    ... --resume                                                                      # continue

``--run`` goes through THE deployed ``ModelManager.generate_once`` (temperature
0, small ``max_tokens``, reasoning off). ``generate_once`` does not return the
provider's usage object, so spend is ESTIMATED (chars/4 in, chars/4 out) and the
run stops BEFORE a call that could push the estimate past ``--max-usd``; a
credit/402 error stops it immediately. Resume is checkpoint-based: the ledger is
saved every ``--checkpoint-every`` turns and ``findings.jsonl`` is truncated back
to the checkpoint's size on ``--resume``.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import copy
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory import claim_ledger as cl  # noqa: E402
from utils import safe_json  # noqa: E402

DEFAULT_CORPUS = ROOT / "data" / "corpus_v4.json"
DEFAULT_MODEL = "gpt-4o-mini"
EXTRACT_MAX_TOKENS = cl.EXTRACT_MAX_TOKENS
REPLY_MAX_TOKENS = cl.REPLY_MAX_TOKENS
EXCERPT_CHARS = 120
TOP_FINDINGS = 15
DROP_EXAMPLES_PER_REASON = 5
DROP_EXAMPLE_CHARS = 120
SUBJECT_TABLE_MAX_ROWS = 200
MAX_CONSECUTIVE_MODEL_ERRORS = 5
SUBJECT_LINE_CHARS = 110      # dry-run assumption: one open-subject line

EXIT_OK, EXIT_USAGE, EXIT_BUDGET, EXIT_CREDITS, EXIT_ERRORS = 0, 2, 3, 4, 5

ModelCall = Callable[[str, str, int], Awaitable[str]]   # (system, user, max_tokens) -> text


class StopRun(Exception):
    """Raised inside the loop to stop cleanly; ``code`` is the process exit code."""

    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


# --- helpers ----------------------------------------------------------------
def est_tokens(text: str) -> int:
    return (len(text or "") + 3) // 4


def check_out_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if any(part == "data" for part in resolved.parts):
        raise ValueError(f"refusing --out-dir under a data/ directory: {resolved}")
    return resolved


def excerpt(text: str, n: int = EXCERPT_CHARS) -> str:
    one = " ".join(str(text or "").split())
    return one if len(one) <= n else one[: n - 1] + "…"


# --- corpus (read-only) -------------------------------------------------------
def load_corpus(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:  # read-only open; nothing here writes
        data = json.load(fh)
    return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []


def select_turns(entries: Iterable[dict], since: Optional[date] = None,
                 until: Optional[date] = None) -> Tuple[List[dict], Dict[str, int]]:
    """Real conversation turns in time order, with exclusion counts.

    Same exclusions as ``report_claim_drift.select_turns`` (summaries/reflections,
    quarantined, [test] probes, junk/API-error docs, unparseable timestamps)."""
    from memory.corpus_manager import is_consolidated_entry  # lazy import: startup-cost (pure predicate; module also defines the corpus writer, never instantiated here)
    from memory.fact_source import authored_text, is_test_turn  # lazy import: startup-cost (extractor stack)
    from memory.utils import is_junk_conversation_doc, is_quarantined  # lazy import: startup-cost (chroma-adjacent helpers)
    from utils.read_time_markers import strip_machinery  # lazy import: startup-cost (marker vocabulary)
    from utils.temporal_resolver import parse_said_on  # lazy import: startup-cost (resolver module)

    skipped: Dict[str, int] = collections.Counter()
    turns: List[dict] = []
    for idx, e in enumerate(entries):
        if is_consolidated_entry(e):
            skipped["summary_or_reflection"] += 1
        elif is_quarantined(e):
            skipped["quarantined"] += 1
        elif is_test_turn(e):
            skipped["test_probe"] += 1
        else:
            user = authored_text(e)
            resp = strip_machinery(str(e.get("response") or ""))
            ts = parse_said_on(e.get("timestamp"))
            if is_junk_conversation_doc(query=user, response=resp):
                skipped["junk"] += 1
            elif ts is None:
                skipped["no_timestamp"] += 1
            elif (since and ts.date() < since) or (until and ts.date() > until):
                skipped["outside_window"] += 1
            else:
                turns.append({"idx": idx, "ts": ts, "user": user, "assistant": resp})
    turns.sort(key=lambda t: (t["ts"], t["idx"]))
    return turns, dict(skipped)


# --- dry run ------------------------------------------------------------------
def estimate_plan(turns: List[dict], *, price_in: float, price_out: float,
                  est_subjects: int, est_out_extract: int, est_out_reply: int) -> dict:
    subj_chars = est_subjects * SUBJECT_LINE_CHARS
    ex_in = ex_out = rc_in = rc_out = prepared = 0
    for t in turns:
        reason, text = cl.prepare_user_text(t["user"])
        if not reason:
            system, user = cl.build_extraction_prompt(text, t["ts"], [])
            ex_in += est_tokens(system) + est_tokens(user) + subj_chars // 4
            ex_out += est_out_extract
            prepared += 1
        reply = (t["assistant"] or "")[: cl.MAX_REPLY_CHARS]
        if reply.strip():
            system, user = cl.build_reply_check_prompt(reply, t["ts"], [])
            rc_in += est_tokens(system) + est_tokens(user) + subj_chars // 4
            rc_out += est_out_reply
    cost = lambda i, o: i * price_in / 1e6 + o * price_out / 1e6  # noqa: E731
    return {
        "turns": len(turns), "extraction_turns": prepared,
        "extraction": {"in_tokens": ex_in, "out_tokens": ex_out, "usd": cost(ex_in, ex_out)},
        "reply_check": {"in_tokens": rc_in, "out_tokens": rc_out, "usd": cost(rc_in, rc_out),
                        "note": "upper bound: assumes open subjects on every turn"},
        "total_usd": cost(ex_in + rc_in, ex_out + rc_out),
    }


def render_plan(plan: dict, skipped: dict, args) -> str:
    ex, rc = plan["extraction"], plan["reply_check"]
    return "\n".join([
        "DRY RUN -- no model calls made (pass --run to execute)",
        f"turns in window: {plan['turns']}  excluded: {skipped}",
        f"token estimate: chars/4; prices USD/1M in/out = {args.price_in}/{args.price_out}; "
        f"assumed {args.est_subjects} open subjects per prompt",
        f"phase (a) extraction  : {plan['extraction_turns']} calls, in~{ex['in_tokens']:,} "
        f"out~{ex['out_tokens']:,} tokens  ~${ex['usd']:.4f}",
        f"phase (b) reply check : <= {plan['turns']} calls, in~{rc['in_tokens']:,} "
        f"out~{rc['out_tokens']:,} tokens  ~${rc['usd']:.4f}  ({rc['note']})",
        f"TOTAL estimated cost  : ~${plan['total_usd']:.4f}",
    ])


# --- run ------------------------------------------------------------------------
class Spend:
    def __init__(self, max_usd: float, price_in: float, price_out: float, spent: float = 0.0):
        self.max_usd, self.price_in, self.price_out, self.spent = max_usd, price_in, price_out, spent

    def project(self, in_tokens: int, out_tokens: int) -> float:
        return in_tokens * self.price_in / 1e6 + out_tokens * self.price_out / 1e6


def _is_credit_failure(text: str) -> bool:
    return isinstance(text, str) and text.lstrip().startswith("[CREDITS EXHAUSTED]")


def _classify_error(exc: Exception) -> str:
    """THE deployed classifier ``generate_once`` applies to a provider exception (402 / quota -> [CREDITS EXHAUSTED])."""
    from models.model_manager import _classify_api_error  # lazy import: startup-cost (openai/httpx stack)
    return _classify_api_error(exc)


def _model_error(text: str) -> bool:
    from models.model_manager import API_ERROR_PREFIXES  # lazy import: startup-cost (openai/httpx stack)
    return isinstance(text, str) and text.lstrip().startswith(API_ERROR_PREFIXES)


def _wrap_call(raw_call: ModelCall, spend: Spend, state: dict) -> Callable[[str, str, int], Awaitable[str]]:
    """Budget guard + credit/error detection around any ``(system, user, max_tokens)`` call."""
    async def call(system: str, user: str, max_tokens: int) -> str:
        in_t = est_tokens(system) + est_tokens(user)
        if spend.spent + spend.project(in_t, max_tokens) > spend.max_usd:
            raise StopRun(f"stopping BEFORE a call that could exceed --max-usd "
                          f"(spent ~${spend.spent:.4f} of ${spend.max_usd:.2f})", EXIT_BUDGET)
        try:
            out = await raw_call(system, user, max_tokens)
        except Exception as exc:  # classified below: credit failure stops the run, the rest is a counted drop
            msg = str(exc)
            if _is_credit_failure(_classify_error(exc)):
                raise StopRun(f"credit exhausted / HTTP 402 -- stopping: {excerpt(msg, 160)}", EXIT_CREDITS) from exc
            state["consecutive_errors"] += 1
            if state["consecutive_errors"] >= MAX_CONSECUTIVE_MODEL_ERRORS:
                raise StopRun(f"{MAX_CONSECUTIVE_MODEL_ERRORS} consecutive model errors -- stopping "
                              f"(last: {excerpt(msg, 120)})", EXIT_ERRORS) from exc
            spend.spent += spend.project(in_t, 0)
            return ""
        if _is_credit_failure(out):
            raise StopRun("credit exhausted / HTTP 402 -- stopping (provider reported insufficient credits)",
                          EXIT_CREDITS)
        spend.spent += spend.project(in_t, est_tokens(out if isinstance(out, str) else ""))
        if _model_error(out):
            state["consecutive_errors"] += 1
            if state["consecutive_errors"] >= MAX_CONSECUTIVE_MODEL_ERRORS:
                raise StopRun(f"{MAX_CONSECUTIVE_MODEL_ERRORS} consecutive model errors -- stopping "
                              f"(last: {excerpt(out, 120)})", EXIT_ERRORS)
            return ""
        state["consecutive_errors"] = 0
        return out if isinstance(out, str) else ""
    return call


def _new_stats() -> dict:
    return {"turns_done": 0, "turns_extracted": 0, "turns_reply_checked": 0,
            "skipped_extraction": {}, "extraction_drops": {}, "reply_drops": {},
            "drop_examples": {"extraction": {}, "reply": {}},
            "findings_by_class": {}, "model_errors": 0, "spent_usd": 0.0}


def _bump(d: dict, key: str, n: int = 1) -> None:
    d[key] = d.get(key, 0) + n


def _record_drops(stats: dict, phase: str, counts_key: str, drops: List[dict]) -> None:
    """Count drops by reason and keep up to N short examples (dropped quote) per reason."""
    examples = stats.setdefault("drop_examples", {}).setdefault(phase, {})
    for d in drops:
        reason = str(d["reason"]).split(" ")[0]
        _bump(stats[counts_key], reason)
        shown = excerpt(d.get("quote") or d.get("detail") or "", DROP_EXAMPLE_CHARS)
        bucket = examples.setdefault(reason, [])
        if shown and len(bucket) < DROP_EXAMPLES_PER_REASON:
            bucket.append(shown)


def _load_progress(out_dir: Path) -> Optional[dict]:
    p = out_dir / "progress.json"
    return safe_json.load_critical_json(str(p), "claim_ledger_baseline_progress")


def _checkpoint(out_dir: Path, ledger: cl.ClaimLedger, progress: dict, findings_path: Path) -> None:
    ledger.save(str(out_dir / "ledger.json"))
    progress["findings_bytes"] = findings_path.stat().st_size if findings_path.exists() else 0
    safe_json.atomic_write_json(str(out_dir / "progress.json"), progress)


async def process_turns(turns: List[dict], out_dir: Path, raw_call: ModelCall, *,
                        spend: Spend, checkpoint_every: int, resume: bool,
                        window: dict) -> Tuple[dict, cl.ClaimLedger, Optional[StopRun]]:
    """Run phases (a)+(b) over ``turns``; returns ``(stats, ledger, stop)``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    findings_path = out_dir / "findings.jsonl"
    prior = _load_progress(out_dir)
    if prior is not None and not resume:
        raise ValueError(f"{out_dir} already holds a run (progress.json); pass --resume or a fresh --out-dir")
    if prior is None and resume:
        raise ValueError(f"--resume given but {out_dir / 'progress.json'} does not exist")
    if prior is not None:
        if prior.get("window") != window or prior.get("n_turns") != len(turns):
            raise ValueError("--resume refused: the corpus window/turn count differs from the checkpointed run")
        ledger = cl.ClaimLedger.load(str(out_dir / "ledger.json"))
        stats, start = prior["stats"], int(prior["next_turn"])
        spend.spent = float(stats.get("spent_usd", 0.0))
        size = int(prior.get("findings_bytes", 0))
        if findings_path.exists() and findings_path.stat().st_size > size:
            with open(findings_path, "r+b") as fh:
                fh.truncate(size)
    else:
        ledger, stats, start = cl.ClaimLedger(), _new_stats(), 0
        findings_path.write_text("", encoding="utf-8")

    state = {"consecutive_errors": 0}
    call = _wrap_call(raw_call, spend, state)
    stop: Optional[StopRun] = None
    progress = {"window": window, "n_turns": len(turns), "next_turn": start, "stats": stats}

    async def extract_call(system: str, user: str) -> str:
        return await call(system, user, EXTRACT_MAX_TOKENS)

    async def reply_call(system: str, user: str) -> str:
        return await call(system, user, REPLY_MAX_TOKENS)

    try:
        for n in range(start, len(turns)):
            t = turns[n]
            snapshot = copy.deepcopy(stats)   # a stop mid-turn restores it: the turn is redone on resume
            claims, drops, skip = await cl.extract_claims(
                t["user"], t["ts"], ledger, extract_call, turn_id=f"corpus:{t['idx']}")
            if skip:
                _bump(stats["skipped_extraction"], skip)
            else:
                stats["turns_extracted"] += 1
            _record_drops(stats, "extraction", "extraction_drops", drops)
            for c in claims:
                ledger.add(c)
            findings, rdrops, rskip = await cl.check_reply(t["assistant"], t["ts"], ledger, reply_call)
            if not rskip:
                stats["turns_reply_checked"] += 1
            _record_drops(stats, "reply", "reply_drops", rdrops)
            if findings:
                with open(findings_path, "a", encoding="utf-8") as fh:
                    for f in findings:
                        row = f.to_dict()
                        row.update({"turn_idx": t["idx"], "ts": t["ts"].isoformat(timespec="seconds"),
                                    "quote": excerpt(f.quote, 200)})
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        _bump(stats["findings_by_class"], f.classification)
            stats["turns_done"] = n + 1
            stats["spent_usd"] = round(spend.spent, 6)
            stats["model_errors"] = state["consecutive_errors"]
            progress["next_turn"] = n + 1
            if (n + 1) % max(1, checkpoint_every) == 0:
                _checkpoint(out_dir, ledger, progress, findings_path)
    except StopRun as exc:
        stop = exc
        stats.clear()
        stats.update(snapshot)
    stats["spent_usd"] = round(spend.spent, 6)
    # A stop mid-turn leaves the turn un-counted: next_turn still points at it, so resume redoes it.
    _checkpoint(out_dir, ledger, progress, findings_path)
    return stats, ledger, stop


# --- report ---------------------------------------------------------------------
def read_findings(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def render_subjects_table(ledger: cl.ClaimLedger) -> List[str]:
    """One row per subject (subject_key + kind + unit): claim count, current value, superseded values."""
    groups: Dict[tuple, List[cl.LedgerClaim]] = {}
    for c in ledger.claims:
        groups.setdefault((c.subject_key, c.kind, c.unit_norm), []).append(c)
    lines = ["", "## Subjects", "",
             "| subject_key | kind | unit | claims | current (value, modality, date) | superseded values |",
             "|---|---|---|---|---|---|"]
    for n, ((key, kind, unit), items) in enumerate(sorted(groups.items())):
        if n >= SUBJECT_TABLE_MAX_ROWS:
            lines.append(f"| ... {len(groups) - n} more subjects not shown | | | | | |")
            break
        cur = ledger._pick_current(items)
        if cur is not None:
            current = f"{cur.value} ({cur.modality}, {cur.stated_at[:10]})"
        else:
            conds = [c for c in items if c.modality == "conditional"]
            current = ("none (conditional only: " + ", ".join(f"{c.value} {c.stated_at[:10]}" for c in conds) + ")"
                       ) if conds else "none"
        sup = [f"{c.value} ({c.modality}, {c.stated_at[:10]})" for c in items if c.superseded_by]
        lines.append(f"| {key} | {kind} | {unit or '-'} | {len(items)} | {current} | {'; '.join(sup) or '-'} |")
    if not groups:
        lines.append("| (no claims) | | | | | |")
    return lines


def render_drop_examples(stats: dict) -> List[str]:
    lines = ["", "## Drop examples (up to %d per reason; dropped quote, %d chars max)"
             % (DROP_EXAMPLES_PER_REASON, DROP_EXAMPLE_CHARS)]
    for phase in ("extraction", "reply"):
        by_reason = (stats.get("drop_examples") or {}).get(phase) or {}
        lines += ["", f"### {phase}"]
        if not by_reason:
            lines.append("none")
        for reason, quotes in sorted(by_reason.items()):
            lines.append(f"- {reason} ({(stats.get(phase + '_drops') or {}).get(reason, len(quotes))} total)")
            lines += [f"  - {q}" for q in quotes]
    return lines


def render_report(stats: dict, ledger: cl.ClaimLedger, findings: List[dict], *,
                  skipped: dict, window: dict, stop: Optional[StopRun]) -> str:
    def table(d: dict) -> str:
        return ", ".join(f"{k}={v}" for k, v in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))) or "none"

    lines = [
        "# Claim-ledger baseline", "",
        f"window: {window}", f"turns done: {stats['turns_done']}  (extracted {stats['turns_extracted']}, "
        f"reply-checked {stats['turns_reply_checked']})", f"corpus exclusions: {skipped}",
        f"estimated spend: ${stats['spent_usd']:.4f}",
    ]
    if stop:
        lines.append(f"STOPPED EARLY: {stop}")
    lines += ["", "## Claims", f"total admitted: {len(ledger.claims)}",
              f"by kind: {table(dict(collections.Counter(c.kind for c in ledger.claims)))}",
              f"by modality: {table(dict(collections.Counter(c.modality for c in ledger.claims)))}",
              f"supersessions: {sum(1 for c in ledger.claims if c.superseded_by)}",
              f"open subjects now: {len(ledger.open_subjects(10**6))}",
              f"extraction skipped (input): {table(stats['skipped_extraction'])}",
              f"extraction drops by reason: {table(stats['extraction_drops'])}", "",
              "## Reply findings", f"by class: {table(stats['findings_by_class'])}",
              f"reply-check drops by reason: {table(stats['reply_drops'])}"]
    for cls in ("stale", "differs_from_estimate", "conditional_as_fact", "estimate_as_fact"):
        rows = [r for r in findings if r.get("classification") == cls][:TOP_FINDINGS]
        lines += ["", f"### top {cls} ({len(rows)} shown)"]
        for r in rows:
            lines.append(f"- {str(r.get('ts'))[:10]} `{r['subject_key']}` reply={r['value']} "
                         f"({r['reply_modality']}) vs ledger={r['claim_value']} ({r['claim_modality']}, "
                         f"stated {str(r['claim_stated_at'])[:10]}): {excerpt(r['quote'])}")
    lines += render_subjects_table(ledger)
    lines += render_drop_examples(stats)
    return "\n".join(lines) + "\n"


# --- model call (deployed path) ----------------------------------------------------
def default_call_factory(model: str) -> ModelCall:
    """THE deployed ``ModelManager.generate_once`` (same construction as backfill_summaries.py)."""
    from dotenv import load_dotenv  # lazy import: optional-dependency (only a --run needs the key)
    from utils.bootstrap import resolve_env_path  # lazy import: startup-cost (bootstrap resolves the live .env location)
    load_dotenv(resolve_env_path(), override=True)
    from models.model_manager import ModelManager  # lazy import: startup-cost (openai/httpx stack)
    mm = ModelManager()
    if model not in mm.api_models:
        raise SystemExit(f"model {model!r} is not a registered API model")
    if getattr(mm, "async_client", None) is None:
        raise SystemExit("no API client (missing OPENROUTER_API_KEY / OPENAI_API_KEY) -- cannot --run")

    async def call(system: str, user: str, max_tokens: int) -> str:
        return await cl.generate_for_ledger(mm, model, system, user, max_tokens)
    return call


# --- CLI --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--since", type=date.fromisoformat)
    ap.add_argument("--until", type=date.fromisoformat)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--run", action="store_true", help="make model calls (default: dry run, no calls)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-usd", type=float, help="required with --run")
    ap.add_argument("--price-in", type=float, default=0.15, help="USD per 1M input tokens")
    ap.add_argument("--price-out", type=float, default=0.60, help="USD per 1M output tokens")
    ap.add_argument("--est-subjects", type=int, default=15, help="dry-run: open subjects per prompt")
    ap.add_argument("--est-out-extract", type=int, default=90)
    ap.add_argument("--est-out-reply", type=int, default=60)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--limit", type=int, help="process at most N turns (trial runs)")
    return ap


def main(argv: Optional[List[str]] = None, *,
         call_factory: Optional[Callable[[str], ModelCall]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        out_dir = check_out_dir(args.out_dir) if args.out_dir else None
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    if not args.corpus.exists():
        print(f"corpus not found: {args.corpus}", file=sys.stderr)
        return EXIT_USAGE
    if args.run and (out_dir is None or args.max_usd is None):
        print("--run requires --out-dir and --max-usd", file=sys.stderr)
        return EXIT_USAGE
    if args.resume and not args.run:
        print("--resume only applies with --run", file=sys.stderr)
        return EXIT_USAGE

    turns, skipped = select_turns(load_corpus(args.corpus), args.since, args.until)
    if args.limit:
        turns = turns[: args.limit]

    if not args.run:
        plan = estimate_plan(turns, price_in=args.price_in, price_out=args.price_out,
                             est_subjects=args.est_subjects, est_out_extract=args.est_out_extract,
                             est_out_reply=args.est_out_reply)
        print(render_plan(plan, skipped, args))
        return EXIT_OK

    raw_call = (call_factory or default_call_factory)(args.model)
    window = {"since": str(args.since), "until": str(args.until), "limit": args.limit}
    spend = Spend(args.max_usd, args.price_in, args.price_out)
    try:
        stats, ledger, stop = asyncio.run(process_turns(
            turns, out_dir, raw_call, spend=spend, checkpoint_every=args.checkpoint_every,
            resume=args.resume, window=window))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    findings = read_findings(out_dir / "findings.jsonl")
    (out_dir / "report.md").write_text(
        render_report(stats, ledger, findings, skipped=skipped, window=window, stop=stop), encoding="utf-8")
    print(f"turns done {stats['turns_done']}/{len(turns)}  claims {len(ledger.claims)}  "
          f"findings {sum(stats['findings_by_class'].values())}  est spend ${stats['spent_usd']:.4f}")
    print(f"wrote {out_dir}/ledger.json findings.jsonl report.md")
    if stop:
        print(f"STOPPED: {stop}", file=sys.stderr)
        return stop.code
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
