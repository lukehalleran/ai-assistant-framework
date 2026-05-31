"""
# core/actions/registry.py

Single declarative source of truth for internet WRITE actions (the propose_action family).

Adding a new write action should be ONE entry here + one executor function — not a sweep across
executors.py, protocols.py (parse), tools.py (tool-health), and controller.py (forced-action
detection / instruction / backfill). Each of those consumers reads from ACTION_SPECS instead of
hardcoding per-action logic, so they can't drift out of sync (and the parity tests in
tests/unit/test_tool_wiring_parity.py fail loudly if a spec is incomplete).

Executors are referenced lazily by "module:function" string and resolved at call time — this keeps
import cost deferred (as the original executors.py did) and lets tests patch the module function.

Module Contract
- Public: ActionSpec, ACTION_SPECS, is_action_enabled(spec), enabled_action_types(),
  detect_action_intent(query), backfill_params(action_type, query).
- Dependencies: core.actions.types only (executor modules imported lazily on use); config.app_config
  read lazily for enable flags. No dependency on core.agentic.* (correct layering).
"""

import importlib
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from core.actions.types import ActionType


# ---------------------------------------------------------------------------
# Deterministic param extraction (backfill) — for when a model calls propose_action
# but leaves content fields blank under a large agentic context.
# ---------------------------------------------------------------------------
def _extract_issue_fields_from_query(query: str) -> Tuple[str, str]:
    """Best-effort (title, body) extraction for a 'create a GitHub issue' request."""
    if not query:
        return "", ""
    title, body = "", ""
    m = re.search(r'titled\s*:?\s*["“‘\']([^"”’\']+)["”’\']', query, re.IGNORECASE)
    if not m:
        m = re.search(r'titled\s*:?\s+(.+?)(?:\s+[—–-]\s+|$)', query, re.IGNORECASE)
    if m:
        title = m.group(1).strip().strip('"“”‘’\'')
    bm = re.search(
        r'body\s+should\s+(?:explain|say|describe|cover|note|state|mention)?\s*(?:that\s+)?(.+)$',
        query, re.IGNORECASE | re.DOTALL,
    )
    if bm:
        body = bm.group(1).strip()
    elif m:
        body = query[m.end():].lstrip(" —–-:’'\"").strip()
    return title, body


def _github_issue_backfill(query: str) -> Dict[str, str]:
    title, body = _extract_issue_fields_from_query(query)
    out: Dict[str, str] = {}
    if title:
        out["subject"] = title
    if body:
        out["message"] = body
    return out


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ActionSpec:
    """Everything the rest of the system needs to know about one write action."""
    action_type: ActionType
    executor_ref: str                         # "module.path:function" — resolved lazily at call time
    required: Tuple[str, ...]                 # params that must be present for parse acceptance
    optional: Tuple[str, ...] = ()            # additional params to forward if present
    intent_patterns: Tuple[str, ...] = ()     # regexes for explicit-action detection (forcing)
    backfill: Optional[Callable[[str], Dict[str, str]]] = None  # query -> partial params
    health: str = ""                          # tool-health (TOOL STATUS) line
    field_hint: str = ""                      # the per-action required-field directive line
    enabled_flag: Optional[str] = None        # extra app_config gate beyond INTERNET_ACTIONS_ENABLED
    summary: Optional[Callable[[dict], str]] = None  # params -> human summary

    @property
    def forward_params(self) -> Tuple[str, ...]:
        return tuple(self.required) + tuple(self.optional)

    def resolve_executor(self) -> Callable:
        """Import + return the executor function (lazy; re-resolved each call so patches apply)."""
        module_path, func_name = self.executor_ref.split(":")
        return getattr(importlib.import_module(module_path), func_name)


# Insertion order = forced-action detection priority (issue before pr-comment, etc.).
ACTION_SPECS: Dict[ActionType, ActionSpec] = {
    ActionType.GITHUB_CREATE_ISSUE: ActionSpec(
        action_type=ActionType.GITHUB_CREATE_ISSUE,
        executor_ref="core.actions.github_write:create_github_issue",
        required=("subject",),
        optional=("message",),
        intent_patterns=(r'\b(open|create|file|raise|log)\b[^.?!]{0,40}\bissue\b',),
        backfill=_github_issue_backfill,
        health="github_create_issue (file an issue — subject=title, message=body)",
        field_hint="github_create_issue: subject = the issue TITLE, message = the issue BODY",
        enabled_flag="INTERNET_ACTIONS_GITHUB_WRITE_ENABLED",
        summary=lambda p: f"github issue: {(p.get('subject') or '')[:60]}",
    ),
    ActionType.GITHUB_COMMENT_PR: ActionSpec(
        action_type=ActionType.GITHUB_COMMENT_PR,
        executor_ref="core.actions.github_write:comment_github_pr",
        required=("pr_number", "message"),
        intent_patterns=(
            r'\b(comment|reply|respond|post)\b[^.?!]{0,40}\b(pr|pull[\s-]?request)\b',
            r'\b(pr|pull[\s-]?request)\b[^.?!]{0,25}\bcomment\b',
        ),
        health="github_comment_pr (comment on a PR — pr_number + message)",
        field_hint="github_comment_pr: pr_number = the PR number, message = the comment text",
        enabled_flag="INTERNET_ACTIONS_GITHUB_WRITE_ENABLED",
        summary=lambda p: f"comment on PR #{p.get('pr_number','?')}",
    ),
    ActionType.SEND_EMAIL: ActionSpec(
        action_type=ActionType.SEND_EMAIL,
        executor_ref="core.actions.email:send_email",
        required=("recipient", "message"),
        optional=("subject",),
        intent_patterns=(
            r'\b(send|email|compose|draft|fire off)\b[^.?!]{0,30}\b(email|e-mail)\b',
            r'^\s*email\s+\w',
        ),
        health="send_email (recipient + message; recipient may be a contact name)",
        field_hint="send_email: recipient and message",
        summary=lambda p: f"send_email to {p.get('recipient','')}: {(p.get('message') or '')[:50]}",
    ),
    ActionType.SEND_TELEGRAM: ActionSpec(
        action_type=ActionType.SEND_TELEGRAM,
        executor_ref="core.actions.telegram:send_telegram_message",
        required=("message",),
        optional=("recipient",),
        intent_patterns=(r'\b(send|post|message|ping)\b[^.?!]{0,30}\btelegram\b',),
        health="send_telegram (message; recipient/chat optional)",
        field_hint="send_telegram: message (recipient optional)",
        summary=lambda p: (
            f"send_telegram to {p['recipient']}: {(p.get('message') or '')[:50]}"
            if p.get("recipient") else f"send_telegram: {(p.get('message') or '')[:50]}"
        ),
    ),
    ActionType.SEND_DISCORD: ActionSpec(
        action_type=ActionType.SEND_DISCORD,
        executor_ref="core.actions.discord:send_discord_message",
        required=("message",),
        optional=("recipient",),
        intent_patterns=(r'\b(send|post|message|ping)\b[^.?!]{0,30}\bdiscord\b',),
        health="send_discord (message; webhook optional)",
        field_hint="send_discord: message (recipient optional)",
        summary=lambda p: (
            f"send_discord to {p['recipient']}: {(p.get('message') or '')[:50]}"
            if p.get("recipient") else f"send_discord: {(p.get('message') or '')[:50]}"
        ),
    ),
    ActionType.CALENDAR_CREATE_EVENT: ActionSpec(
        action_type=ActionType.CALENDAR_CREATE_EVENT,
        executor_ref="core.actions.google_calendar_create:create_calendar_event",
        required=("summary", "start_time", "end_time"),
        optional=("description", "time_zone", "calendar_id", "location"),
        intent_patterns=(
            r'\b(create|add|schedule|make|set up|put)\b[^.?!]{0,40}\b(calendar event|event|meeting|appointment)\b',
        ),
        health="calendar_create_event (summary + start_time + end_time)",
        field_hint="calendar_create_event: summary, start_time, end_time",
        enabled_flag="GOOGLE_CALENDAR_ENABLED",
        summary=lambda p: f"calendar_create_event: {p.get('summary','')}",
    ),
}


# ---------------------------------------------------------------------------
# Helpers consumed by executors.py / protocols.py / tools.py / controller.py
# ---------------------------------------------------------------------------
def is_action_enabled(spec: ActionSpec) -> bool:
    """True if internet actions are on AND this spec's extra gate (if any) is on."""
    import config.app_config as cfg
    if not getattr(cfg, "INTERNET_ACTIONS_ENABLED", False):
        return False
    if spec.enabled_flag:
        return bool(getattr(cfg, spec.enabled_flag, False))
    return True


def enabled_action_types() -> Tuple[ActionType, ...]:
    return tuple(at for at, spec in ACTION_SPECS.items() if is_action_enabled(spec))


def detect_action_intent(query: str) -> Optional[ActionType]:
    """Return the ActionType if the query is an EXPLICIT request to perform a write action."""
    if not query:
        return None
    for at, spec in ACTION_SPECS.items():
        for pattern in spec.intent_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return at
    return None


def backfill_params(action_type: ActionType, query: str) -> Dict[str, str]:
    """Deterministically derive missing params from the query, or {} if the spec has no backfill."""
    spec = ACTION_SPECS.get(action_type)
    if spec and spec.backfill:
        return spec.backfill(query) or {}
    return {}
