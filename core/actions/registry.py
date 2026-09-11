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
- Dependencies: core.actions.types + utils.trigger_match (leaf module, negation
  lookback for detect_action_intent) only; executor modules imported lazily on
  use; config.app_config read lazily for enable flags. No dependency on
  core.agentic.* (correct layering).
"""

import importlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from core.actions.types import ActionType
from utils.trigger_match import is_negated as _is_trigger_negated


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


# Calendar noun spelled tolerant of the two common transpositions
# ("calander", "calender", "calandar"). 2026-09-07 live: "add the mgt office
# hours sessions to my google calander in one batch" was an explicit calendar
# request that no calendar pattern saw; the gate ran the turn as a WEB search,
# the model answered with an OFFER, and the affirmation turns that followed
# had no tool route — the reply then narrated "creating the recurring event
# now" with nothing created. Word boundary + optional plural preserved.
_CALENDAR_WORD = r"cal[ae]nd[ae]rs?"


def _calendar_batch_within_cap(params: Dict[str, Any]) -> bool:
    items = params.get("events")
    if not isinstance(items, list):
        return True
    try:
        from config.app_config import GOOGLE_CALENDAR_MAX_EVENTS  # lazy import: live-config read
        cap = int(GOOGLE_CALENDAR_MAX_EVENTS)
    except Exception:
        cap = 10
    return len(items) <= cap


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
    accepts_check: Optional[Callable[[Dict[str, Any]], bool]] = None  # extra shape rule beyond required fields

    @property
    def forward_params(self) -> Tuple[str, ...]:
        fields = tuple(self.required) + tuple(self.optional)
        if self.batch_param and self.batch_param not in fields:
            fields += (self.batch_param,)
        return fields

    def accepts_params(self, params: Dict[str, Any]) -> bool:
        """Whether params satisfy this action's single-item or batch shape."""
        if all(params.get(field) not in (None, "") for field in self.required):
            return self.accepts_check(params) if self.accepts_check else True
        if not self.batch_param:
            return False
        items = params.get(self.batch_param)
        ok = bool(items) and isinstance(items, list) and all(
            isinstance(item, dict)
            and all(item.get(field) not in (None, "") for field in self.required)
            for item in items
        )
        if ok and self.accepts_check:
            return self.accepts_check(params)
        return ok

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
        optional=("description", "time_zone", "calendar_id", "location", "all_day",
                  "recurrence"),
        batch_param="events",
        intent_patterns=(
            # 2026-09-10: "please make repeating through whole course at same
            # time instead" (an amendment of the just-proposed single event)
            # matched nothing — the object list had no recurrence noun.
            r'\b(create|add|schedule|make|set up|put)\b[^.?!]{0,40}\b(' + _CALENDAR_WORD + r'\s+events?|events?|meetings?|appointments?|sessions?|recurring|repeating|repeat(?:s|ing)?\s+weekly)\b',
            # "place each in the appropriate time slot on my Google calendar"
            # (live 2026-08-29): verb "place" + bare object "calendar" missed
            # the pattern above, and the verb→object span ran 44 chars — the
            # explicit calendar request produced an offer instead of a
            # proposal. Bare "calendar" only counts as the object of a
            # placement verb (this pattern), never of "make"/"schedule" alone.
            # 2026-09-07: the noun is spelled through _CALENDAR_WORD — the
            # live "add the mgt office hours sessions to my google calander"
            # missed on the typo alone and the turn ran as a web search.
            r'\b(add|put|place|drop|slot)\b[^.?!]{0,60}\b(?:google\s+)?' + _CALENDAR_WORD + r'\b',
        ),
        health="calendar_create_event (one event or an events[] batch; requires confirmation)",
        field_hint=(
            "calendar_create_event: summary, start_time, end_time; for several "
            "DIFFERENT events use one batch proposal containing events[]. A "
            "REPEATING event (weekly office hours, a standing meeting) is ONE "
            "event whose start/end are the first occurrence plus recurrence "
            "(an RRULE string, e.g. 'RRULE:FREQ=WEEKLY;UNTIL=20261204') — never "
            "N copies; if the source states no end date, use COUNT or say so "
            "and ask, never invent a semester end. Honor any source timezone. "
            "For all-day events set all_day=true and use YYYY-MM-DD start/end "
            "dates (Google end date is exclusive)."
        ),
        enabled_flag="GOOGLE_CALENDAR_ENABLED",
        summary=lambda p: (
            f"calendar_create_event: {len(p.get('events') or [])} events"
            if p.get("events") else
            f"calendar_create_event: {p.get('summary','')}"
        ),
        # An oversize batch (a semester of weekly sessions as 14 copies) used to
        # pass parse and die at APPROVAL ("maximum is 10"); reject at parse so
        # the forced-action retry re-asks — the field hint says: one event +
        # recurrence.
        accepts_check=_calendar_batch_within_cap,
    ),
    # Update/delete (2026-09-01): both require an EXPLICIT calendar anchor
    # ("event(s)" or "calendar") as the object — bare "reschedule my
    # appointment" is a life task the user does with the OFFICE, not a
    # calendar-edit command (live 12:33 turn: "reschedule appointment with
    # new psychiatrist" must NOT force an action loop). Under-fires by design.
    ActionType.CALENDAR_UPDATE_EVENT: ActionSpec(
        action_type=ActionType.CALENDAR_UPDATE_EVENT,
        executor_ref="core.actions.google_calendar_modify:update_calendar_event",
        required=("summary", "date"),
        optional=("event_id", "new_summary", "new_start_time", "new_end_time",
                  "new_description", "new_location", "time_zone", "all_day",
                  "calendar_id"),
        intent_patterns=(
            r'\b(move|reschedule|shift|change|update|edit)\b[^.?!]{0,60}\b' + _CALENDAR_WORD + r'\s+events?\b',
            r'\b(move|reschedule|shift|change|update|edit)\b[^.?!]{0,60}\bevents?\b',
            r'\b(move|reschedule|shift|change|update|edit)\b[^.?!]{0,60}\b(?:on|in|from)\s+(?:my\s+|the\s+)?(?:google\s+)?' + _CALENDAR_WORD + r'\b',
        ),
        health="calendar_update_event (edit an existing event — summary + date identify it; exactly one match required)",
        field_hint=(
            "calendar_update_event: summary and date (YYYY-MM-DD) identify the "
            "EXISTING event; changes go in new_* fields — new_start_time and "
            "new_end_time together (ISO), new_summary, new_description, "
            "new_location. event_id may replace summary+date."
        ),
        enabled_flag="GOOGLE_CALENDAR_ENABLED",
        summary=lambda p: (
            f"calendar_update_event: {p.get('summary','')} on {p.get('date','')}"
        ),
        # An update with no new_* fields can only fail after approval
        # (2026-09-01 live: a changeless marker rendered a card and died at
        # the executor) — reject at parse so the forced retry re-asks.
        accepts_check=lambda p: any(
            p.get(k) for k in ("new_summary", "new_start_time", "new_end_time",
                               "new_description", "new_location")
        ),
    ),
    ActionType.CALENDAR_DELETE_EVENT: ActionSpec(
        action_type=ActionType.CALENDAR_DELETE_EVENT,
        executor_ref="core.actions.google_calendar_modify:delete_calendar_event",
        required=("summary", "date"),
        optional=("event_id", "calendar_id"),
        intent_patterns=(
            r'\b(delete|remove|cancel|clear|drop)\b[^.?!]{0,60}\b' + _CALENDAR_WORD + r'\s+events?\b',
            r'\b(delete|remove|cancel|clear)\b[^.?!]{0,60}\bevents?\b',
            r'\b(delete|remove|cancel|clear|take)\b[^.?!]{0,60}\b(?:off|from)\s+(?:my\s+|the\s+)?(?:google\s+)?' + _CALENDAR_WORD + r'\b',
        ),
        health="calendar_delete_event (remove an existing event — summary + date identify it; exactly one match required, irreversible)",
        field_hint=(
            "calendar_delete_event: summary and date (YYYY-MM-DD) identify the "
            "EXISTING event to remove; event_id may replace summary+date. "
            "Ambiguous matches refuse — never guess on a delete."
        ),
        enabled_flag="GOOGLE_CALENDAR_ENABLED",
        summary=lambda p: (
            f"calendar_delete_event: {p.get('summary','')} on {p.get('date','')}"
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

        # Contacts (2026-09-08, B6): a bare "HTTP 403" told the owner
        # nothing about whether the People API is simply not enabled for
        # the project vs. the OAuth token lacking a contacts scope. Shown
        # only when a call has actually failed this process — silent
        # otherwise, since "never called" is not evidence of a problem.
        # Checked independently of Calendar's enabled state below (a
        # separate Google surface).
        try:
            from core.actions.google_contacts import get_last_error as _contacts_last_error
            _contacts_err = _contacts_last_error()
            if _contacts_err:
                lines.append(f"lookup_contact backend: DEGRADED ({_contacts_err})")
        except Exception:
            pass

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
        elif getattr(auth, "token_expired_no_refresh", False):
            # Audit F15 (2026-08-31): an expired token with no refresh token
            # reported AVAILABLE — the disk shows it cannot be refreshed.
            lines.append(
                "calendar_create_event backend: UNAVAILABLE "
                "(OAuth token expired with no refresh token — "
                "owner must run scripts/reauth_google.py)"
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
    r"set\s+up|put|place|drop|slot|move|reschedule|shift|change|update|edit|"
    r"delete|remove|cancel|clear)\b",
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


# First-person narration frame (2026-09-10): "I only put professors hours in
# calander but there are TA sessions too" matched the calendar-create pattern
# (verb "put" + calendar noun) and FORCED a propose_action round on a message
# whose only request was "can we check course docs" — the model then invented
# a 5 PM TA session. A pattern verb whose immediate subject is the user
# (I / I've / I'd / I'll / we …, optionally separated by an adverb) inside a
# clause that carries no request cue is the user narrating their own action,
# not asking Daemon to act. Clause = the span between sentence punctuation.
_NARRATION_SUBJECT_RE = re.compile(
    r"(?:^|[\s,(])"
    r"(?:i|i'?ve|i'?d|i'?ll|i'?m\s+gonna|i'?m\s+going\s+to|i\s+have|i\s+had|i\s+will|"
    r"i\s+would|i\s+could|i\s+can|i\s+did|i\s+just|we|we'?ve|we'?d|we'?ll|we\s+have|"
    r"we\s+had|we\s+will|we\s+did)"
    r"(?:\s+(?:only|just|already|also|then|even|actually|finally|recently|simply|"
    r"literally|never|still))*\s*$",
    re.IGNORECASE,
)
_REQUEST_CUE_RE = re.compile(
    r"\?|\b(?:can|could|would|will|should|shall)\s+(?:you|we|u)\b|\bplease\b|"
    r"\blet'?s\b|\b(?:want|need|like|ask|help)\s+(?:you|u)\b|\bhelp\s+me\b|"
    r"\bgo\s+ahead\b|\byou\s+(?:to|can|could|should)\b",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY_RE = re.compile(r"[.?!;\n]")


def _match_is_self_narration(query: str, verb_start: int) -> bool:
    """True when the pattern verb at ``verb_start`` is the user's own narrated
    action ("I only put …", "I'll add it later", "we put them in last week")
    and the clause around it asks Daemon for nothing."""
    starts = [m.end() for m in _CLAUSE_BOUNDARY_RE.finditer(query, 0, verb_start)]
    clause_start = starts[-1] if starts else 0
    end_m = _CLAUSE_BOUNDARY_RE.search(query, verb_start)
    clause_end = end_m.end() if end_m else len(query)  # keep the "?" — it is a request cue
    clause = query[clause_start:clause_end]
    if _REQUEST_CUE_RE.search(clause):
        return False
    before = query[clause_start:verb_start]
    return bool(_NARRATION_SUBJECT_RE.search(before))


def detect_action_intent(query: str) -> Optional[ActionType]:
    """Return the ActionType for an explicit, plausibly user-authored request.

    Negation-aware (2026-09-04): "don't add that to my calendar" must not
    return CALENDAR_CREATE_EVENT just because the pattern's verb+object
    co-occur — a negation/avoidance cue within 5 tokens before the match
    (utils.trigger_match.is_negated) disqualifies it.

    Narration-aware (2026-09-10): a match whose verb is governed by a
    first-person subject in a request-free clause ("I only put the hours in
    my calendar") is skipped; later matches in the same message are still
    considered ("I put A in already, can you add B?" fires on the second).
    """
    if not query:
        return None
    for at, spec in ACTION_SPECS.items():
        for pattern in spec.intent_patterns:
            for m in re.finditer(pattern, query, re.IGNORECASE):
                if not _action_request_is_plausible(query):
                    break
                if _is_trigger_negated(query, m.start()):
                    continue
                verb_start = m.start(1) if m.lastindex else m.start()
                if _match_is_self_narration(query, verb_start):
                    continue
                return at
    return None


# Amendment cue (2026-09-10): "make it repeating instead", "actually change
# it to 11", "rather than a single event" — the user is revising the proposal
# just made, so a pending card of the same type is stale, not a duplicate.
_AMENDMENT_CUE_RE = re.compile(
    r"\b(?:instead|rather|actually|change\s+(?:it|that|the)|make\s+(?:it|that|them)|"
    r"not\s+(?:a\s+)?single|different\s+(?:time|day|date|slot)|move\s+(?:it|that)|"
    r"switch\s+(?:it|that)|redo|re-?do\s+(?:it|that)|scrap\s+(?:that|it)|"
    r"replace\s+(?:it|that))\b",
    re.IGNORECASE,
)


def is_amendment_cue(user_text: str) -> bool:
    """True when a short message revises the proposal just offered/queued."""
    text = (user_text or "").strip()
    if not text or len(text.split()) > ACTION_RETRY_MAX_WORDS:
        return False
    return bool(_AMENDMENT_CUE_RE.search(text))


def backfill_params(action_type: ActionType, query: str) -> Dict[str, str]:
    """Deterministically derive missing params from the query, or {} if the spec has no backfill."""
    spec = ACTION_SPECS.get(action_type)
    if spec and spec.backfill:
        return spec.backfill(query) or {}
    return {}


# ---------------------------------------------------------------------------
# Forced-round type pinning (2026-09-09, F12)
# ---------------------------------------------------------------------------
# Live incident: the gate detected an explicit calendar_delete_event request
# and the controller forced propose_action on round 1, but the generic
# native-tools schema's action_type enum never included calendar_delete_event
# (or calendar_update_event) at all — the model had no valid way to express
# the requested type and substituted the only calendar option it could see
# (calendar_create_event), whose required fields the delete-shaped params did
# not satisfy. The round was rejected, the retry was a blind re-ask, and the
# final reply narrated "Queued the deletion… Confirm and it's off" with no
# card ever created. Two complementary fixes: build_forced_tool_schema below
# gives a forced round a tool definition that can only name the ONE required
# type; resolve_forced_action is the single acceptance/coercion/rejection
# decision both protocol handlers call so a model that still names the wrong
# type is corrected (when its params happen to fit the required spec anyway)
# or rejected with a reason the retry prompt can show, instead of the
# rejection being silently dropped.

# Field shapes that are not a bare string, for building a per-action-type
# native tool-calling JSON schema (build_forced_tool_schema). Anything not
# listed here defaults to {"type": "string"} — true of nearly every action
# param across the registry (dates/times are ISO strings, not native types).
_FORCED_TOOL_FIELD_TYPES: Dict[str, Dict[str, Any]] = {
    "all_day": {"type": "boolean"},
    "pr_number": {"type": "integer"},
    "events": {
        "type": "array",
        "description": "Several DIFFERENT calendar events in one batch proposal.",
        "items": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "description": {"type": "string"},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "time_zone": {"type": "string"},
                "calendar_id": {"type": "string"},
                "location": {"type": "string"},
                "all_day": {"type": "boolean"},
                "recurrence": {"type": "string"},
            },
            "required": ["summary", "start_time", "end_time"],
        },
    },
}


def build_forced_tool_schema(action_type: ActionType) -> Optional[Dict[str, Any]]:
    """A propose_action tool definition scoped to exactly ONE action type.

    Used only for a forced decision round (core.agentic.controller) so the
    model cannot silently substitute a sibling action_type the generic,
    all-types tool schema happens to expose — `action_type` is a one-value
    enum and only this spec's own fields are offered. Returns None for an
    unregistered action type (callers fall back to the generic tool).
    """
    spec = ACTION_SPECS.get(action_type)
    if spec is None:
        return None
    properties: Dict[str, Any] = {
        "action_type": {"type": "string", "enum": [action_type.value]},
        "reason": {
            "type": "string",
            "description": "Why you are proposing this action (shown to the user).",
        },
    }
    for field_name in spec.forward_params:
        properties[field_name] = _FORCED_TOOL_FIELD_TYPES.get(field_name, {"type": "string"})
    return {
        "type": "function",
        "function": {
            "name": "propose_action",
            "description": (
                f"Propose the write action {action_type.value}. {spec.field_hint} "
                "The user will see a confirmation prompt and can approve or reject."
            ),
            "parameters": {
                "type": "object",
                "properties": properties,
                # Only action_type/reason are schema-required (matching the
                # generic tool definition) — required CONTENT fields are
                # enforced by resolve_forced_action's acceptance check, not
                # the provider's own schema validation, so a partially-filled
                # call still reaches parsing (and a useful rejection reason)
                # instead of being refused by the API before it is ever sent.
                "required": ["action_type", "reason"],
            },
        },
    }


def resolve_forced_action(
    action_type: str,
    params: Dict[str, Any],
    forced_action_type: Optional[str] = None,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """Validate — and, only in a forced round, possibly coerce — one proposed
    action. Returns (resolved_type, resolved_params, reject_reason); exactly
    one of (resolved_type, reject_reason) is non-None.

    `forced_action_type` is the ActionType.value the controller is currently
    forcing (None outside a forced round — coercion NEVER happens outside a
    forced round, per the audit acceptance criteria). When the model's own
    action_type is accepted as-is, it is returned unchanged (no coercion
    needed). When it is NOT accepted and differs from `forced_action_type`,
    the same raw params are checked against the forced spec; if THEY satisfy
    it (or its backfill), the type is coerced to the forced one. Otherwise a
    human-readable rejection reason is returned naming the required type and
    its fields, for the retry prompt.
    """
    try:
        spec = ACTION_SPECS.get(ActionType(action_type)) if action_type else None
    except ValueError:
        spec = None

    # Referee tightening (Fable, 2026-09-09): inside a forced round a
    # DIFFERENT action_type is never accepted as-is, even when its own spec
    # is satisfied — a well-formed calendar_create_event in a forced
    # calendar_delete_event round would otherwise CREATE what the user asked
    # to delete. Coerce when the params fit the required spec, else reject
    # with the reason; the model's own type stands only when it matches.
    _type_mismatch = bool(forced_action_type) and action_type != forced_action_type
    if spec is not None and not _type_mismatch and (
        spec.accepts_params(params) or spec.backfill is not None
    ):
        return action_type, params, None

    if forced_action_type and action_type != forced_action_type:
        try:
            forced_spec = ACTION_SPECS.get(ActionType(forced_action_type))
        except ValueError:
            forced_spec = None
        if forced_spec is not None:
            if forced_spec.accepts_params(params) or forced_spec.backfill is not None:
                return forced_action_type, params, None
            return None, None, (
                f"proposed action_type={action_type!r} does not match the required "
                f"{forced_action_type!r}, and the given params do not satisfy "
                f"{forced_action_type!r} either (needs: {', '.join(forced_spec.required)})"
            )

    if spec is None:
        return None, None, f"action_type={action_type!r} is not a recognized action"
    return None, None, (
        f"action_type={action_type!r} is missing required fields "
        f"(needs: {', '.join(spec.required)})"
    )


# ---------------------------------------------------------------------------
# Prior-turn action OFFERS (2026-09-07)
# ---------------------------------------------------------------------------
# When a chat-mode reply OFFERS an external action ("Want me to go ahead and
# create the recurring event?") and the user says yes on the next turn, the
# affirmation has to reach the tool loop with the offered action FORCED — the
# NOTE-only PendingProposalStore never carried external kinds, and the gate's
# casual/short skip dropped "please create" into a tool-less turn (live
# 2026-09-07 15:39–15:42: four consecutive turns, no proposal, a confabulated
# "Confirmed — creating the recurring event now"). One mapping here feeds the
# gate arm, the controller force, and the claim guard's expected-to-act set.

_OFFER_UPDATE_VERB_RE = re.compile(
    r"\b(move|reschedule|shift|change|update|edit)\b", re.IGNORECASE)
_OFFER_DELETE_VERB_RE = re.compile(
    r"\b(delete|remove|cancel|clear|take\s+(?:it|that|them)\s+off)\b", re.IGNORECASE)
_OFFER_ISSUE_RE = re.compile(r"\bissue\b", re.IGNORECASE)
_OFFER_PR_RE = re.compile(r"\b(pr|pull\s+request)\b", re.IGNORECASE)
_OFFER_DISCORD_RE = re.compile(r"\bdiscord\b", re.IGNORECASE)


def action_kind_of(action_type: ActionType):
    """The coarse claim-guard ActionKind for an ActionType (None for self-repairable/unknown)."""
    from core.action_claim_guard import ActionKind  # leaf module, no cycle
    mapping = {
        ActionType.SEND_EMAIL: ActionKind.EMAIL,
        ActionType.CALENDAR_CREATE_EVENT: ActionKind.CALENDAR,
        ActionType.CALENDAR_UPDATE_EVENT: ActionKind.CALENDAR,
        ActionType.CALENDAR_DELETE_EVENT: ActionKind.CALENDAR,
        ActionType.SEND_TELEGRAM: ActionKind.MESSAGE,
        ActionType.SEND_DISCORD: ActionKind.MESSAGE,
        ActionType.GITHUB_CREATE_ISSUE: ActionKind.GITHUB,
        ActionType.GITHUB_COMMENT_PR: ActionKind.GITHUB,
    }
    return mapping.get(action_type)


# ---------------------------------------------------------------------------
# Forced-round time grounding (2026-09-10)
# ---------------------------------------------------------------------------
# Live: a forced calendar_create_event round had NO stated time anywhere (the
# user asked to CHECK the course docs for TA sessions) and the model filled
# start_time=2026-09-11T17:00:00 — its own reasoning even said "exact time
# should be confirmed". A proposal card with a guessed time is worse than no
# card. A clock time is grounded when the same hour (:minute when given)
# appears in the request or the gathered context in ANY common spelling:
# "5 pm", "5:00", "17:00", "1700", "730A", "noon", "midnight". No ±1h zone
# tolerance: the executor's timezone doctrine writes a source-stated zone's
# time verbatim with time_zone set, so a grounded proposal matches exactly.
#
# 2026-09-10 referee follow-up: two sibling shapes were still ungrounded.
# (a) A bare hour after a time preposition ("office hours at 3 on Fridays",
#     "meets at 11", "from 9 to 10", "9-10 on Saturdays") had no meridiem and
#     no colon, so none of the arms above matched it — a correctly-stated
#     time was flagged as an invented guess. The `bare`/`r1`+`r2` arms below
#     add it; noise guards keep them off unit-suffixed numbers ("1264 rows",
#     "30 mg") and 4-digit years/ISO dates ("2026", "2026-09-11").
# (b) An ISO timestamp embedded IN the pool text itself (action digests and
#     the controller's own "[ACTION NOT PROPOSED]" note both render
#     "start_time=2026-09-11T17:00:00") has no word boundary between the "T"
#     and the hour digits, so the `h2` arm's leading \b never matched. The
#     `ih`/`im` arm below matches the ISO "T17:00" shape directly.
_TIME_NOISE_GUARD = r"(?!\s*(?:rows?|mg|days?|minutes?|mins?|hours?|%|k)\b)"
_CLOCK_TOKEN_RE = re.compile(
    r"\b(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>a\.?m\.?|p\.?m\.?|a|p)\b"
    r"|\b(?P<h2>\d{1,2}):(?P<m2>\d{2})\b"
    r"|\b(?P<mil>\d{3,4})\s*(?P<ap2>a|p|am|pm)?\b"
    r"|\b(?P<word>noon|midday|midnight)\b"
    r"|T(?P<ih>\d{2}):(?P<im>\d{2})"
    r"|\b(?:at|from|to|until|till|by|around|before|after|@)\s+(?P<bare>\d{1,2})\b"
    r"(?!\s*:)(?!\s*[ap]\.?m\.?\b)" + _TIME_NOISE_GUARD +
    r"|(?<!\d{4}-)\b(?P<r1>\d{1,2})\s*(?:-|–|to)\s*(?P<r2>\d{1,2})\b" + _TIME_NOISE_GUARD,
    re.IGNORECASE,
)


def _add_hour(hours: set, h: int, mi: Optional[str], ap: Optional[str]) -> None:
    """Record a parsed (hour, minute[, ambiguous-pm-reading]) into ``hours``."""
    if h > 24:
        return
    if ap == "p" and h < 12:
        h += 12
    if ap == "a" and h == 12:
        h = 0
    mi_i = None if mi is None else int(mi)
    hours.add((h % 24, mi_i))
    hours.add((h % 24, None))
    if ap is None and 1 <= h <= 12:
        # A 12-hour token with no meridiem ("9:00-10:00pm", "the 3:30",
        # "meets at 11", "9-10 on Saturdays") is ambiguous — it grounds both
        # readings.
        hours.add(((h + 12) % 24, mi_i))
        hours.add(((h + 12) % 24, None))


def _pool_hours(pool: str) -> set:
    """Every (hour, minute) a text mentions, as 24h tuples; minutes=None when
    the mention has no minutes ("5 pm")."""
    hours: set = set()
    for m in _CLOCK_TOKEN_RE.finditer(pool or ""):
        if m.group("word"):
            w = m.group("word").lower()
            hours.add((0 if w == "midnight" else 12, None))
            continue
        if m.group("h") is not None:
            h, mi, ap = int(m.group("h")), m.group("m"), m.group("ap").lower()[0]
        elif m.group("h2") is not None:
            h, mi, ap = int(m.group("h2")), m.group("m2"), None
        elif m.group("ih") is not None:
            h, mi, ap = int(m.group("ih")), m.group("im"), None
        elif m.group("bare") is not None:
            h, mi, ap = int(m.group("bare")), None, None
        elif m.group("r1") is not None:
            _add_hour(hours, int(m.group("r1")), None, None)
            _add_hour(hours, int(m.group("r2")), None, None)
            continue
        else:
            raw = m.group("mil")
            ap = (m.group("ap2") or "").lower()[:1] or None
            if len(raw) == 3:
                h, mi = int(raw[0]), raw[1:]
            else:
                h, mi = int(raw[:2]), raw[2:]
            if not ap and (h > 23 or int(mi) > 59):
                continue  # "1264 rows" — not a clock
            if not ap and len(raw) == 4 and (h < 1 or raw[:2] in ("19", "20")):
                continue  # "2026-09-11" is a year, not 20:26
        _add_hour(hours, h, mi, ap)
    return hours


def _iso_clock(value: Any) -> Optional[Tuple[int, int]]:
    m = re.search(r"T(\d{2}):(\d{2})", str(value or ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def calendar_times_ungrounded(params: Dict[str, Any], pool_text: str) -> list:
    """Return the proposed calendar start/end clock times (as ISO strings) that
    appear nowhere in ``pool_text`` (request + conversation/action context +
    gathered tool output). Empty list = every timed field is grounded. All-day
    events and events with no ISO clock component are never flagged."""
    items = params.get("events") if isinstance(params.get("events"), list) else [params]
    hours = _pool_hours(pool_text)
    bad: list = []
    for ev in items:
        if not isinstance(ev, dict) or ev.get("all_day") in (True, "true", "True"):
            continue
        for key in ("start_time", "end_time"):
            clk = _iso_clock(ev.get(key))
            if clk is None:
                continue
            h, mi = clk
            if (h, mi) in hours or (h, None) in hours:
                continue
            bad.append(f"{key}={ev.get(key)}")
    return bad


def narrated_unbacked_action_type(response_text: str) -> Optional[ActionType]:
    """The EXTERNAL ActionType a prior reply NARRATED without backing (it
    carried the no-card notice or an unbacked-claim correction, or a
    completion claim of an external kind) — or None.

    2026-09-10 live: "Queued up: … Approve the proposal" shipped with no card,
    the appended NO_CARD_NOTICE told the user to say "try again", and "try
    again" had no route because the retry path only re-queues a FAILED card.
    A narrated-but-unbacked external action IS an offer the user can accept.
    """
    text = response_text or ""
    if not text:
        return None
    try:
        from core.action_claim_guard import (
            NO_CARD_NOTICE, ActionKind, detect_completion_claims, detect_kind,
        )
    except Exception:
        return None
    notice = NO_CARD_NOTICE.strip() in text or "I didn't actually" in text
    claims = [c for c in detect_completion_claims(text)
              if c.kind in (ActionKind.CALENDAR, ActionKind.EMAIL,
                            ActionKind.MESSAGE, ActionKind.GITHUB)]
    if not notice and not claims:
        return None
    kind = claims[-1].kind if claims else detect_kind(text)
    clause = claims[-1].matched_text if claims else text
    return _kind_to_action_type(kind, clause)


def _kind_to_action_type(kind, clause: str) -> Optional[ActionType]:
    try:
        from core.action_claim_guard import ActionKind
    except Exception:
        return None
    if kind == ActionKind.CALENDAR:
        if _OFFER_DELETE_VERB_RE.search(clause):
            return ActionType.CALENDAR_DELETE_EVENT
        if _OFFER_UPDATE_VERB_RE.search(clause):
            return ActionType.CALENDAR_UPDATE_EVENT
        return ActionType.CALENDAR_CREATE_EVENT
    if kind == ActionKind.EMAIL:
        return ActionType.SEND_EMAIL
    if kind == ActionKind.MESSAGE:
        return (ActionType.SEND_DISCORD if _OFFER_DISCORD_RE.search(clause)
                else ActionType.SEND_TELEGRAM)
    if kind == ActionKind.GITHUB:
        if _OFFER_PR_RE.search(clause) and not _OFFER_ISSUE_RE.search(clause):
            return ActionType.GITHUB_COMMENT_PR
        return ActionType.GITHUB_CREATE_ISSUE
    return None


def offer_action_type(response_text: str) -> Optional[ActionType]:
    """The EXTERNAL ActionType a reply offered to perform, or None.

    Uses the deployed claim-guard proposal detector (offer marker / question +
    action verb, quoted and drafted blocks stripped). Only external kinds
    return — note/document offers stay with the PendingProposalStore path.
    The newest offer wins when a reply makes several. The kind→type mapping
    reads the offer clause's own verb (an offer to "move" an event is an
    update, to "cancel" it a delete; a bare calendar offer is a create).
    """
    if not response_text:
        return None
    try:
        from core.action_claim_guard import (
            ActionKind, detect_kind, detect_offer_clauses, detect_proposals,
            has_offer_marker,
        )
    except Exception:
        return None
    # Offer FRAMING is required ("want me to…", "I can…", "confirm and I'll…").
    # detect_proposals also admits a bare question + action verb, which is
    # fine for the claim guard but here would turn "Did you add it to your
    # calendar?" + "yes" into a forced calendar create.
    proposals = [p for p in detect_proposals(response_text)
                 if has_offer_marker(p.matched_text or "")]
    candidates = [(p.kind, p.matched_text or "") for p in reversed(proposals)]
    if not candidates:
        # Anaphoric follow-up offer ("Want me to create just the professor
        # one now?", "Confirm and I'll create it"): the offer clause carries
        # no kind word, the reply as a whole does. Live 2026-09-07 turns 4-5.
        _clauses = [c for c in detect_offer_clauses(response_text) if has_offer_marker(c)]
        _kind = detect_kind(response_text) if _clauses else None
        if _kind is not None:
            candidates = [(_kind, _clauses[-1])]
    for kind, clause in candidates:
        resolved = _kind_to_action_type(kind, clause)
        if resolved is not None:
            return resolved
    return None


# Terse go-ahead directive: head-anchored action verb ("please create",
# "create it", "go ahead and add them", "lets just do the first one now").
# Ack/filler openers are allowed; anything longer than the word cap is a
# substantive message that routes through the normal tiers.
OFFER_DIRECTIVE_MAX_WORDS = 14
_OFFER_DIRECTIVE_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|alright|all\s+right|cool|yeah|yes|yep|sure|right|so|and|now|then|well|hey|great|perfect)[,\s]+){0,3}"
    r"(?:please\s+|just\s+|go\s+ahead\s+(?:and\s+)?|let'?s\s+(?:just\s+)?)*"
    r"(?:create|add|make|schedule|book|put|send|post|open|file|do|fire|queue|"
    r"proceed|confirm|approve|proceed)\b(?!,)",
    re.IGNORECASE,
)


_HEAD_FILLER_RE = re.compile(
    r"^(?:(?:well|so|um|uh|hmm|oh|okay|ok|yeah|yes|yep|yup|sure|alright|all\s+right|"
    r"cool|great|perfect|right)[\s,]+)+",
    re.IGNORECASE,
)


def is_offer_affirmation(user_text: str) -> bool:
    """True when ``user_text`` accepts a prior-turn action offer.

    Judged on the message HEAD only (the first clause), so an affirmation
    that goes on to supply the data the offer asked for ("yeah lets do that,
    here are the two links: https://…") still counts. Two shapes:
      1. the head IS an affirmation phrase — exactly, or after stripping
         leading ack fillers ("yes", "sure", "ok please do", "yeah lets do it");
      2. a terse go-ahead directive ("please create", "add them",
         "lets just do the first one now", "okay create both").
    Deliberately NOT the starts-with leniency of pending_proposal.is_affirmation:
    "yeah the Zoom link works" starts with "yeah" but accepts nothing, and a
    forced write action is the wrong thing to hang on an ack word. A
    decline/negation in the head vetoes ("no don't create it", "hold off");
    a question is never an affirmation.
    """
    if not user_text:
        return False
    text = user_text.strip()
    clauses = [c.strip() for c in re.split(r"[,;:\n]|(?<=[.!?])\s", text) if c.strip()]
    if not clauses:
        return False
    if _clause_affirms(clauses[0]):
        return True
    # The go-ahead often CLOSES a short message ("ok that link is right, go
    # ahead and create it"). Judge the tail clause too, but only for a
    # message short enough to be a reply, never a paste.
    if len(clauses) > 1 and len(text.split()) <= OFFER_AFFIRMATION_MAX_WORDS:
        return _clause_affirms(clauses[-1])
    return False


OFFER_AFFIRMATION_MAX_WORDS = 40


def _clause_affirms(clause: str) -> bool:
    from core.pending_proposal import AFFIRMATION_PHRASES, is_decline  # leaf-ish, no cycle
    head = clause.strip()
    if not head or head.endswith("?"):
        return False
    if is_decline(head):
        return False
    norm = re.sub(r"\s+", " ", head.lower()).strip(" .!")
    if norm in AFFIRMATION_PHRASES:
        return True
    core = _HEAD_FILLER_RE.sub("", norm).strip(" .!")
    if not core or core in AFFIRMATION_PHRASES:
        return True
    if len(head.split()) <= OFFER_DIRECTIVE_MAX_WORDS and _OFFER_DIRECTIVE_RE.search(head):
        return True
    return False


# ---------------------------------------------------------------------------
# Retry of a FAILED action (2026-09-07)
# ---------------------------------------------------------------------------
# "Ah didn't work. Can we try that again?" / "had to reauthorize, good now,
# please try again" after an approved action failed at the executor. The
# request is to run the SAME action again; the failed proposal still holds the
# exact params. Cue detection reuses the deployed retry-continuation phrases
# plus "<verb> it again" / "re-run" shapes; a negation scoping the cue vetoes.
_RETRY_EXTRA_RE = re.compile(
    r"\b(?:fire|run|send|create|queue|do|submit|push|kick)\s+(?:it|that|this|them|the\s+\w+)\s+again\b"
    r"|\bre-?(?:run|queue|fire|send|submit|try)\b(?!\s+(?:later|tomorrow))"
    r"|\btry\s+(?:it|that|this|them)\s+(?:again|now|once more)\b"
    r"|\bgive\s+it\s+another\s+(?:go|shot|try)\b",
    re.IGNORECASE,
)
ACTION_RETRY_MAX_WORDS = 25
# A deferral ("try again tomorrow") or the USER retrying something themself
# ("I'll try again later myself") is not a request to re-run Daemon's action now.
_RETRY_DEFER_RE = re.compile(
    r"\b(?:later|tomorrow|tonight|next\s+\w+|in\s+a\s+(?:bit|while|sec|second|minute|few)|"
    r"some\s+other\s+time|another\s+day|not\s+(?:now|yet))\b", re.IGNORECASE)
_RETRY_SELF_RE = re.compile(
    r"\b(?:i'?ll|i\s+will|let\s+me|i\s+can|i'?m\s+gonna|i'?m\s+going\s+to|i\s+should|i\s+might)\s+"
    r"(?:just\s+)?(?:try|retry|re-?run|do\s+it|give\s+it)\b|\bmyself\b|\bby\s+hand\b",
    re.IGNORECASE)


def is_action_retry_request(user_text: str) -> bool:
    """True when a short message asks to run the previously failed action again."""
    if not user_text:
        return False
    text = user_text.strip()
    if len(text.split()) > ACTION_RETRY_MAX_WORDS:
        return False
    if _RETRY_DEFER_RE.search(text) or _RETRY_SELF_RE.search(text):
        return False
    from utils.query_checker import is_retry_continuation  # leaf, no cycle
    m = _RETRY_EXTRA_RE.search(text)
    if m is not None and not _is_trigger_negated(text, m.start()):
        return True
    if is_retry_continuation(text, max_words=ACTION_RETRY_MAX_WORDS):
        # Locate the cue for the negation lookback ("don't try again" / "no
        # need to retry"); fall back to the head when the phrase is fuzzy.
        cue = re.search(r"\b(?:try (?:that |it )?again|one more time|retry|fixed it|"
                        r"should work now|try it now|restarted you)\b", text, re.IGNORECASE)
        pos = cue.start() if cue else 0
        return not _is_trigger_negated(text, pos)
    return False

