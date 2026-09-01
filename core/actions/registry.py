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
  detect_action_intent(query), backfill_params(action_type, query),
  get_runtime_action_health().
- Dependencies: core.actions.types only (executor modules imported lazily on use); config.app_config
  read lazily for enable flags. No dependency on core.agentic.* (correct layering).
"""

import importlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

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
    batch_param: Optional[str] = None          # optional list of required-field dicts
    intent_patterns: Tuple[str, ...] = ()     # regexes for explicit-action detection (forcing)
    backfill: Optional[Callable[[str], Dict[str, str]]] = None  # query -> partial params
    health: str = ""                          # tool-health (TOOL STATUS) line
    field_hint: str = ""                      # the per-action required-field directive line
    enabled_flag: Optional[str] = None        # extra app_config gate beyond INTERNET_ACTIONS_ENABLED
    summary: Optional[Callable[[dict], str]] = None  # params -> human summary

    @property
    def forward_params(self) -> Tuple[str, ...]:
        fields = tuple(self.required) + tuple(self.optional)
        if self.batch_param and self.batch_param not in fields:
            fields += (self.batch_param,)
        return fields

    def accepts_params(self, params: Dict[str, Any]) -> bool:
        """Whether params satisfy this action's single-item or batch shape."""
        if all(params.get(field) not in (None, "") for field in self.required):
            return True
        if not self.batch_param:
            return False
        items = params.get(self.batch_param)
        return bool(items) and isinstance(items, list) and all(
            isinstance(item, dict)
            and all(item.get(field) not in (None, "") for field in self.required)
            for item in items
        )

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
        optional=("description", "time_zone", "calendar_id", "location", "all_day"),
        batch_param="events",
        intent_patterns=(
            r'\b(create|add|schedule|make|set up|put)\b[^.?!]{0,40}\b(calendar events?|events?|meetings?|appointments?)\b',
            # "place each in the appropriate time slot on my Google calendar"
            # (live 2026-08-29): verb "place" + bare object "calendar" missed
            # the pattern above, and the verb→object span ran 44 chars — the
            # explicit calendar request produced an offer instead of a
            # proposal. Bare "calendar" only counts as the object of a
            # placement verb (this pattern), never of "make"/"schedule" alone.
            r'\b(add|put|place|drop|slot)\b[^.?!]{0,60}\b(?:google\s+)?calendar\b',
        ),
        health="calendar_create_event (one event or an events[] batch; requires confirmation)",
        field_hint=(
            "calendar_create_event: summary, start_time, end_time; for several "
            "events use one batch proposal containing events[]. Honor any source "
            "timezone. For all-day events set all_day=true and use YYYY-MM-DD "
            "start/end dates (Google end date is exclusive)."
        ),
        enabled_flag="GOOGLE_CALENDAR_ENABLED",
        summary=lambda p: (
            f"calendar_create_event: {len(p.get('events') or [])} events"
            if p.get("events") else
            f"calendar_create_event: {p.get('summary','')}"
        ),
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


def get_runtime_action_health() -> str:
    """Authoritative runtime status for proposal actions and Calendar OAuth.

    Config flags alone are insufficient for Google Calendar: a model must not
    call it unavailable when a token + write scope are present, or available
    when OAuth has not been completed. This helper is shared by agentic and
    enhanced prompts so their self-knowledge cannot drift.
    """
    try:
        import config.app_config as cfg
        if not getattr(cfg, "INTERNET_ACTIONS_ENABLED", False):
            return "propose_action: DISABLED (internet actions not enabled)"
        names = [at.value for at in enabled_action_types()]
        action_list = ", ".join(names) if names else "(no actions enabled)"
        lines = [
            f"propose_action: AVAILABLE ({action_list} — requires user confirmation)"
        ]

        if not getattr(cfg, "GOOGLE_CALENDAR_ENABLED", False):
            lines.append("calendar_create_event backend: DISABLED by config")
            return "\n".join(lines)

        from core.actions.google_auth import get_google_auth
        auth = get_google_auth()
        if auth is None:
            lines.append(
                "calendar_create_event backend: UNAVAILABLE "
                "(Google OAuth client is not configured)"
            )
        elif not auth.is_authenticated:
            lines.append(
                "calendar_create_event backend: UNAVAILABLE "
                "(Google OAuth token is not authenticated)"
            )
        else:
            from core.actions.google_calendar_create import CALENDAR_EVENTS_SCOPE
            if auth.has_scope(CALENDAR_EVENTS_SCOPE):
                lines.append(
                    "calendar_create_event backend: AVAILABLE "
                    "(OAuth token present; calendar.events write scope granted; "
                    "user confirmation required before execution)"
                )
            else:
                lines.append(
                    "calendar_create_event backend: UNAVAILABLE "
                    "(OAuth token lacks calendar.events write scope)"
                )
        return "\n".join(lines)
    except Exception as exc:
        return f"propose_action: STATUS ERROR ({exc})"


_ACTION_REQUEST_MAX_WORDS = 80
_ACTION_COMMAND_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|alright|all\s+right|cool|yeah|yes|sure|right|so|and|"
    r"now|then|also|well|hey)[,\s]+){0,3}"
    r"(?:please\s+)?(?:"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)|"
    r"(?:i\s+(?:want|need|would\s+like)\s+you\s+to\s+)"
    r")?"
    r"(?:open|create|file|raise|log|comment|reply|respond|post|send|e-?mail|"
    r"compose|draft|forward|shoot|fire\s+off|message|ping|add|schedule|make|"
    r"set\s+up|put|place|drop|slot)\b",
    re.IGNORECASE,
)


def _action_request_is_plausible(query: str) -> bool:
    """Reject an action phrase found only inside a long pasted payload.

    Registry patterns need to find compound requests such as "search my docs,
    then place the dates on my calendar", so they are intentionally not
    head-anchored. Short turns are request-local. In a paste-sized turn, the
    action must instead lead the message or appear as a distinct short final
    paragraph written by the user.
    """
    stripped = (query or "").strip()
    if len(stripped.split()) <= _ACTION_REQUEST_MAX_WORDS:
        return True
    if _ACTION_COMMAND_RE.search(stripped):
        return True
    paragraphs = [
        part.strip()
        for part in re.split(r"\n\s*\n", stripped)
        if part.strip()
    ]
    return bool(
        len(paragraphs) > 1
        and len(paragraphs[-1].split()) <= _ACTION_REQUEST_MAX_WORDS
        and _ACTION_COMMAND_RE.search(paragraphs[-1])
    )


def detect_action_intent(query: str) -> Optional[ActionType]:
    """Return the ActionType for an explicit, plausibly user-authored request."""
    if not query:
        return None
    for at, spec in ACTION_SPECS.items():
        for pattern in spec.intent_patterns:
            if (
                re.search(pattern, query, re.IGNORECASE)
                and _action_request_is_plausible(query)
            ):
                return at
    return None


def backfill_params(action_type: ActionType, query: str) -> Dict[str, str]:
    """Deterministically derive missing params from the query, or {} if the spec has no backfill."""
    spec = ACTION_SPECS.get(action_type)
    if spec and spec.backfill:
        return spec.backfill(query) or {}
    return {}
