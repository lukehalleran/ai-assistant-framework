"""
User display name resolution for prompt templates and output (2026-09-01).

Mirror of utils/institution_resolver.py for the user's NAME. Motivation:
removing hardcoded "Luke" from all output generators and notes templates.

Resolution order:
  1. `DAEMON_USER_NAME` env override (settable via config.local.yaml —
     owner PII never enters source).
  2. User profile — identity facts with relation="name", is_current only,
     highest confidence wins. Falls back to the quick_profile `name` key.
  3. Fallback to "the user" if all sources fail.

Validation: a plausible name is 1-3 tokens, alphabetic+apostrophe/hyphen,
starts uppercase. Junk values fall through to the fallback.
"""

import json
import os
import re
import threading
from typing import Optional

from utils.logging_utils import get_logger

logger = get_logger("user_identity")

USER_NAME_OVERRIDE = os.getenv("DAEMON_USER_NAME", "").strip()

_DEFAULT_PROFILE_PATH = os.path.join("data", "user_profile.json")

# A plausible user name: 1-3 tokens, alphabetic/apostrophe/hyphen, opens uppercase.
_PLAUSIBLE_NAME_RE = re.compile(
    r"^[A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*){0,2}$"
)


class UserIdentityResolver:
    """Profile-backed display name lookup with mtime caching. Never blocks."""

    def __init__(self, profile_path: Optional[str] = None):
        self.profile_path = profile_path or _DEFAULT_PROFILE_PATH
        self._cached: Optional[str] = None
        self._mtime: Optional[float] = None
        self._lock = threading.Lock()

    def get_display_name(self) -> str:
        if USER_NAME_OVERRIDE:
            if _PLAUSIBLE_NAME_RE.match(USER_NAME_OVERRIDE):
                return USER_NAME_OVERRIDE
            logger.debug(
                f"[UserIdentity] env override {USER_NAME_OVERRIDE!r} does not "
                f"match plausible name shape; falling back"
            )
        name = self._from_profile()
        if name:
            return name
        return "the user"

    # ------------------------------------------------------------------

    def _from_profile(self) -> Optional[str]:
        try:
            mtime = os.path.getmtime(self.profile_path)
        except OSError:
            return None
        with self._lock:
            if self._mtime == mtime:
                return self._cached
            value = None
            try:
                with open(self.profile_path, "r", encoding="utf-8") as f:
                    profile = json.load(f)
                value = self._extract(profile)
            except Exception as e:
                logger.debug(f"[UserIdentity] profile read failed: {e}")
            self._cached = value
            self._mtime = mtime
            return value

    @staticmethod
    def _extract(profile: dict) -> Optional[str]:
        candidates = []  # (confidence, value)
        categories = profile.get("categories", {}) or {}
        for facts in categories.values():
            if not isinstance(facts, list):
                continue
            for fact in facts:
                if not isinstance(fact, dict) or not fact.get("is_current", False):
                    continue
                rel = str(fact.get("relation", "")).strip().lower()
                if rel != "name":
                    continue
                val = str(fact.get("value", "")).strip()
                if not _PLAUSIBLE_NAME_RE.match(val):
                    continue
                conf = float(fact.get("confidence", 0.0) or 0.0)
                candidates.append((-conf, val))
        if candidates:
            candidates.sort()
            return candidates[0][1]
        quick = str((profile.get("quick_profile", {}) or {}).get("name", "")).strip()
        if quick and _PLAUSIBLE_NAME_RE.match(quick):
            return quick
        return None


_resolver: Optional[UserIdentityResolver] = None
_resolver_lock = threading.Lock()


def get_user_display_name() -> str:
    """Best currently-known display name for the user, or 'the user'."""
    global _resolver
    if _resolver is None:
        with _resolver_lock:
            if _resolver is None:
                _resolver = UserIdentityResolver()
    return _resolver.get_display_name()
