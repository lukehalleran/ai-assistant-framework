"""Local, consented, redacted "Report a problem" diagnostic bundle (R1, 2026-10-12).

Purpose: a beta tester's install is invisible to the owner.  The tester builds
a bundle LOCALLY (this module, behind /api/report/*), sees exactly what is in
it, and chooses to send the file or text.  Nothing here writes to disk, opens
a socket or phones home.

Privacy design: the turn SUMMARY is an ALLOWLIST of non-text fields, never a
blocklist.  A field not named here is dropped; a string in an allowlisted
label slot survives only if it looks like a label (``LABEL_RE``), so prose
that leaks into a label field is dropped rather than shipped.  Message text,
plans, gate reasons, memory text, file names, URLs and tool arguments have no
path into the summary at all.  Message text appears ONLY in the optional
``full`` tier (opt-in), redacted through utils.privacy_redaction.

Pure functions over (debug_record, telemetry rows, app info) so the route in
api/routes/report.py stays thin.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import sys
import tomllib
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from eval.section_registry import SECTION_REGISTRY
from utils.privacy_redaction import build_redacted_prompt_export, redact_text
from utils.timezone_resolver import get_user_timezone
from utils.turn_telemetry import _MAX_STR_LEN as _TELEMETRY_STR_LEN

SCHEMA_VERSION = 1
MAX_DESCRIPTION_CHARS = 2000
MAX_CONTEXT_TURNS = 5
MAX_TURN_LIST = 20
PREVIEW_CHARS = 60
TELEMETRY_MAX_LINES = 2000
TELEMETRY_MAX_LINE_BYTES = 262_144
_TAIL_BYTES = 16 * 1024 * 1024
_JOIN_TS_WINDOW_S = 5.0

LABEL_RE = re.compile(r"^[A-Za-z0-9_.:\-]{1,64}$")

# --- the allowlist ----------------------------------------------------------
_LABEL_FIELDS = (
    "mode", "model", "intent", "intent_source", "tone_level", "tone_trigger",
    "tone_arbiter_backend", "web_trigger_source", "grounding_status",
    "personal_claim_status",
)
_NUMBER_FIELDS = (
    "intent_confidence", "tone_confidence", "web_trigger_confidence",
    "web_results_n", "response_len", "prepare_elapsed_s", "wall_elapsed_s",
    "pre_prepare_elapsed_s",
)
_BOOL_FIELDS = (
    "is_small_talk", "gate_triggered", "web_trigger_should_search",
    "uncertainty_fired", "review_fired", "review_passed", "has_images",
)
# Prefix families: only bool / number values survive. Labels from these
# families are NOT trusted (an open-ended family could carry one word of user
# text); the two status labels are explicit _LABEL_FIELDS above.
_SCALAR_PREFIXES = ("grounding_", "personal_claim_")
_TIMING_FIELDS = ("phase_timings", "task_timings")
_MAX_TIMING_KEYS = 20
_MAX_WEB_ERROR = 80


def _label(value: Any) -> Optional[str]:
    if isinstance(value, str) and LABEL_RE.match(value):
        return value
    return None


def _number(value: Any) -> Optional[float | int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _timings(value: Any) -> Optional[dict]:
    if not isinstance(value, dict):
        return None
    out: dict[str, float | int] = {}
    for key, item in value.items():
        lab, num = _label(key), _number(item)
        if lab is None or num is None:
            continue
        out[lab] = round(num, 3) if isinstance(num, float) else num
        if len(out) >= _MAX_TIMING_KEYS:
            break
    return out


def _iso(value: Any) -> Optional[str]:
    if not isinstance(value, str) or len(value) > 40:
        return None
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return None
    return value


def _tool_names(value: Any) -> list[str]:
    names: list[str] = []
    if isinstance(value, (list, tuple)):
        for item in value[:30]:
            raw = item.get("tool") if isinstance(item, dict) else item
            lab = _label(raw)
            if lab:
                names.append(lab)
    return names


def _web_error_label(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return _label(value[:_MAX_WEB_ERROR])


def _section_sizes(prompt: Any) -> dict[str, int]:
    """Approximate token size (chars//4) per KNOWN prompt section.

    Section names come from the code's own registry (a closed set), found as
    line-start headers, so message text can never become a key.
    """
    if not isinstance(prompt, str) or not prompt:
        return {}
    marks: list[tuple[int, str]] = []
    for key, sec in SECTION_REGISTRY.items():
        header = sec.header
        if not header.startswith("["):
            continue
        for m in re.finditer(r"(?m)^" + re.escape(header), prompt):
            marks.append((m.start(), key))
            break
    marks.sort()
    sizes: dict[str, int] = {}
    for i, (start, key) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(prompt)
        sizes[key] = (end - start) // 4
    return sizes


_GATHER_EXTRA_NAMES = frozenset({"narrative", "semantic"})
_KNOWN_NAMES_CACHE: Optional[frozenset] = None


def _known_section_names() -> frozenset:
    """Closed set of code-defined section / gather-task names (never user text)."""
    global _KNOWN_NAMES_CACHE
    if _KNOWN_NAMES_CACHE is None:
        names = set(_GATHER_EXTRA_NAMES)
        names |= set(SECTION_REGISTRY)
        try:
            names |= set(_load_rollup_module().KNOWN_TASKS)
        except Exception:  # degrades: task names absent from the closed set
            pass
        _KNOWN_NAMES_CACHE = frozenset(names)
    return _KNOWN_NAMES_CACHE


def build_summary(debug_record: Optional[dict], telemetry: Optional[dict]) -> dict:
    """Allowlisted, text-free summary of one turn.

    The debug record wins over the telemetry row where both carry a field
    (it is this process's own, newer view); telemetry fills the rest.
    """
    sources = [s for s in (debug_record, telemetry) if isinstance(s, dict)]
    summary: dict[str, Any] = {}

    def first(key: str) -> Any:
        for src in sources:
            if key in src and src[key] is not None:
                return src[key]
        return None

    ts = _iso(first("ts"))
    if ts:
        summary["ts"] = ts
    for key in _LABEL_FIELDS:
        lab = _label(first(key))
        if lab is not None:
            summary[key] = lab
    for key in _NUMBER_FIELDS:
        num = _number(first(key))
        if num is not None:
            summary[key] = num
    for key in _BOOL_FIELDS:
        val = first(key)
        if isinstance(val, bool):
            summary[key] = val
    seen_prefixed: set[str] = set()
    for src in sources:
        for key, val in src.items():
            if (isinstance(key, str) and key.startswith(_SCALAR_PREFIXES)
                    and LABEL_RE.match(key) and key not in summary
                    and key not in seen_prefixed):
                seen_prefixed.add(key)
                if isinstance(val, bool) or _number(val) is not None:
                    summary[key] = val
    gate_modes = first("gate_modes")
    if isinstance(gate_modes, list):
        summary["gate_modes"] = [m for m in (_label(x) for x in gate_modes[:10]) if m]
    web_err = _web_error_label(first("web_error"))
    if web_err:
        summary["web_error"] = web_err
    for key in _TIMING_FIELDS:
        t = _timings(first(key))
        if t is not None:
            summary[key] = t
    tools = _tool_names(first("tool_calls"))
    if tools:
        summary["tool_calls"] = tools
    drift = first("citation_drift")
    if isinstance(drift, list):
        summary["citation_drift_count"] = len(drift)
    outcomes = first("section_outcomes")
    if isinstance(outcomes, dict):
        known = _known_section_names()
        sections: dict[str, str] = {}
        for name, val in outcomes.items():
            if name not in known:
                continue  # closed set: a key that is not a code-defined section name is dropped
            status = val.get("status") if isinstance(val, dict) else val
            sections[name] = _label(status) or "present"
        if sections:
            summary["context_sections"] = dict(sorted(sections.items()))
    prompt = debug_record.get("prompt") if isinstance(debug_record, dict) else None
    sizes = _section_sizes(prompt)
    if sizes:
        summary["prompt_section_tokens_approx"] = sizes
    return summary


# --- telemetry --------------------------------------------------------------
def read_telemetry_tail(path: Any, max_lines: int = TELEMETRY_MAX_LINES) -> list[dict]:
    """Last <= max_lines parsed rows, oldest first.  Bounded; corrupt lines skipped.

    A missing/unreadable file is an empty list (summary falls back to the
    debug record alone) — this is read-only diagnostics, never a store.
    """
    try:
        p = Path(str(path))
        if not p.is_file():
            return []
        size = p.stat().st_size
        with p.open("rb") as fh:
            if size > _TAIL_BYTES:
                fh.seek(size - _TAIL_BYTES)
                fh.readline()  # drop the partial first line
            raw_lines = fh.read().splitlines()
    except (OSError, ValueError):  # degrades: report built without telemetry
        return []
    rows: list[dict] = []
    for raw in raw_lines[-max_lines:]:
        if not raw or len(raw) > TELEMETRY_MAX_LINE_BYTES:
            continue
        try:
            row = json.loads(raw)
        except (UnicodeDecodeError, ValueError, RecursionError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _query_key(value: Any) -> str:
    """Join key for a query: turn telemetry stores strings sliced to its
    _MAX_STR_LEN, so both sides compare on that same prefix (a long paste --
    the turns most worth reporting -- would otherwise never match)."""
    return value[:_TELEMETRY_STR_LEN] if isinstance(value, str) else ""


def _session_id(record: dict) -> str:
    prov = record.get("provenance")
    if isinstance(prov, dict) and isinstance(prov.get("session_id"), str):
        return prov["session_id"]
    return ""


def match_telemetry(records: list, index: int, rows: list[dict]) -> Optional[dict]:
    """The turn_records row for ``records[index]`` (debug records carry no ts).

    Join: same query text (compared here, never emitted) and, when both sides
    carry one, the same session_id.  The N-th most recent debug record with a
    given query pairs with the N-th most recent row with that query, so a
    repeated question maps to the right occurrence.  If a candidate row has a
    ts, it must also sit after the previous occurrence's row.
    """
    if not records or not rows:
        return None
    idx = index if index >= 0 else len(records) + index
    if not (0 <= idx < len(records)) or not isinstance(records[idx], dict):
        return None
    target = records[idx]
    query = target.get("query")
    if not isinstance(query, str) or not query:
        return None
    sid = _session_id(target)
    key = _query_key(query)
    occurrence = sum(
        1 for r in records[idx + 1:]
        if isinstance(r, dict) and _query_key(r.get("query")) == key
    )
    matches = [
        r for r in rows
        if _query_key(r.get("query")) == key
        and not (sid and isinstance(r.get("session_id"), str) and r["session_id"] and r["session_id"] != sid)
    ]
    if occurrence >= len(matches):
        return None
    return matches[-1 - occurrence]


def list_turns(records: list, rows: Optional[list[dict]] = None) -> list[dict]:
    """Newest-first picker rows.  ``preview`` is local display only."""
    out: list[dict] = []
    n = len(records)
    for idx in range(n - 1, -1, -1):
        rec = records[idx]
        if not isinstance(rec, dict):
            continue
        row = match_telemetry(records, idx, rows or []) if rows else None
        q = rec.get("query")
        out.append({
            "index": idx,
            "ts": _iso((row or {}).get("ts")) or _iso(rec.get("ts")),
            "mode": _label(rec.get("mode")),
            "preview": redact_text(q[:PREVIEW_CHARS]) if isinstance(q, str) else "",
        })
        if len(out) >= MAX_TURN_LIST:
            break
    return out


# --- app info / rollup ------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _app_version() -> Optional[str]:
    try:
        from utils.bootstrap import resolve_bundled_path  # lazy import: patch-point
        with Path(resolve_bundled_path("pyproject.toml")).open("rb") as fh:
            v = tomllib.load(fh).get("project", {}).get("version")
        return v if isinstance(v, str) and len(v) <= 40 else None
    except Exception:  # degrades: version omitted (pyproject missing)
        return None


def _git_sha() -> Optional[str]:
    try:
        git = _REPO_ROOT / ".git"
        if git.is_file():  # worktree / clone pointer
            text = git.read_text().strip()
            if text.startswith("gitdir:"):
                git = (_REPO_ROOT / text.split(":", 1)[1].strip()).resolve()
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            ref_file = git / ref
            if ref_file.is_file():
                head = ref_file.read_text().strip()
            else:
                common = git / "commondir"
                base = (git / common.read_text().strip()).resolve() if common.is_file() else git
                head = (base / ref).read_text().strip()
        return head[:10] if re.fullmatch(r"[0-9a-f]{7,40}", head) else None
    except Exception:  # degrades: git_sha omitted
        return None


def app_info(model: Any = None, mode: Any = None) -> dict:
    return {
        "version": _app_version(),
        "git_sha": _git_sha(),
        "frozen": bool(getattr(sys, "frozen", False)),
        "mode": mode if mode in ("user", "dev") else None,
        "model": _label(model),
        "python": "%d.%d.%d" % sys.version_info[:3],
        "platform": sys.platform,
    }


def _load_rollup_module():
    # Through the bundled-path resolver: in a packaged build the script ships
    # as a data file (daemon.spec) under the bundle root, not beside utils/.
    from utils.bootstrap import resolve_bundled_path  # lazy import: patch-point
    path = Path(resolve_bundled_path("scripts/diagnostic_rollup.py"))
    spec = importlib.util.spec_from_file_location("_report_diagnostic_rollup", path)
    if spec is None or spec.loader is None:
        raise ImportError("diagnostic_rollup unavailable")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass/typing helpers expect registration
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return mod


def build_rollup(telemetry_path: Any, ts: Optional[str]) -> tuple[Optional[dict], Optional[str]]:
    """(rollup, error_class) for the selected turn's local date."""
    try:
        when = datetime.fromisoformat(ts) if ts else datetime.now().astimezone()
        zone = None
        if when.tzinfo is not None:
            zone = get_user_timezone()
        if not zone:
            zone = "UTC"
        mod = _load_rollup_module()
        return mod.rollup(str(telemetry_path), when.date() if isinstance(when, datetime) else date.today(), zone), None
    except Exception as exc:  # degrades: rollup omitted, class name recorded
        return None, type(exc).__name__


# --- the bundle -------------------------------------------------------------
def build_bundle(
    records: list,
    index: int,
    *,
    description: str = "",
    include_full: bool = False,
    telemetry_rows: Optional[list[dict]] = None,
    telemetry_path: Any = None,
    info: Optional[dict] = None,
    include_system: bool = False,
    now: Optional[datetime] = None,
) -> dict:
    """Build the schema_version-1 bundle for ``records[index]``.

    Raises IndexError when the index is out of range (the route maps it to 404).
    """
    if not records:
        raise IndexError("no debug records")
    idx = index if index >= 0 else len(records) + index
    if not (0 <= idx < len(records)) or not isinstance(records[idx], dict):
        raise IndexError(index)
    record = records[idx]
    rows = telemetry_rows or []
    row = match_telemetry(records, idx, rows)

    turn = build_summary(record, row)
    if row is None:
        turn["telemetry"] = "unavailable"

    context: list[dict] = []
    for j in range(max(0, idx - MAX_CONTEXT_TURNS), idx):
        prior = records[j]
        if isinstance(prior, dict):
            context.append(build_summary(prior, match_telemetry(records, j, rows)))

    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": (now or datetime.now().astimezone()).isoformat(timespec="seconds"),
        "description": redact_text((description or "")[:MAX_DESCRIPTION_CHARS]),
        "app": info if info is not None else app_info(),
        "turn": turn,
        "context_turns": context,
    }
    if telemetry_path:
        rollup, err = build_rollup(telemetry_path, turn.get("ts"))
    else:
        rollup, err = None, "NoTelemetryPath"
    bundle["rollup"] = rollup
    if err:
        bundle["rollup_error"] = err
    bundle["full"] = None
    if include_full:
        response = record.get("response")
        bundle["full"] = {
            "prompt": build_redacted_prompt_export(record, include_system=include_system),
            "response": redact_text(response if isinstance(response, str) else ""),
        }
    return bundle


def summary_lines(bundle: dict) -> list[str]:
    """Plain-text rendering of the summary tier (never the ``full`` part)."""
    lines = [f"Daemon problem report (schema {bundle.get('schema_version')}) {bundle.get('created_at', '')}"]
    app = bundle.get("app") or {}
    lines.append("app: " + ", ".join(f"{k}={v}" for k, v in app.items() if v is not None))
    if bundle.get("description"):
        lines.append("description: " + bundle["description"])
    for key, val in (bundle.get("turn") or {}).items():
        lines.append(f"{key}: {json.dumps(val) if isinstance(val, (dict, list)) else val}")
    return lines


__all__: Iterable[str] = (
    "SCHEMA_VERSION", "LABEL_RE", "build_bundle", "build_summary", "list_turns",
    "match_telemetry", "read_telemetry_tail", "app_info", "summary_lines",
)
