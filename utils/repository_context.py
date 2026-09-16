"""Repository status context, independent of a turn's conversational tone."""

import re

from utils.trigger_match import compile_keyword_matcher, normalize_ws


# Categorized domain vocabulary: ambiguous operations (push, merge, commit)
# are not domain anchors by themselves. Commit *records* use noun syntax.
_REPOSITORY_CUES = {
    "systems": ("git", "github", "gitlab", "bitbucket"),
    "artifacts": ("repo", "repository", "codebase", "pull request", "merge request"),
    "history": ("commit history", "commit hash", "commit message", "changelog"),
}
_REPOSITORY_MATCHER = compile_keyword_matcher(
    [cue for cues in _REPOSITORY_CUES.values() for cue in cues]
)
_COMMIT_RECORD = re.compile(
    r"\b(?:\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"several|some|another|the|my|our|these|those|latest|recent|new)"
    r"\s+(?:(?:new|recent|latest|more)\s+)?commits?\b",
    re.IGNORECASE,
)


def is_repository_status_report(query: str) -> bool:
    """A request-free status report with repository evidence to ground it.

    Keep questions on the existing hybrid/history route. A casual tone must
    not suppress records of the very activity being reported.
    """
    from utils.query_checker import is_self_report, is_status_report  # lazy import: cycle

    text = normalize_ws(query)
    return bool(
        (is_self_report(text) or is_status_report(text))
        and (_REPOSITORY_MATCHER(text.lower()) or _COMMIT_RECORD.search(text))
    )
