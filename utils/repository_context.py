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


# ---------------------------------------------------------------------------
# Own-code activity QUESTIONS (2026-10-24, BC-31 nearest)
# ---------------------------------------------------------------------------
# "what have I been building lately, and how much code went into it?" is a
# question, not a status report, so is_repository_status_report never routed it
# to live local history; the stale procedural index answered (or nothing did)
# and the reply read absence of context as absence of activity.
#
# ONE categorized software-activity vocabulary (not the probe's phrasings):
#   artifacts  - nouns that only name software work products
#   verbs      - verbs whose software sense dominates for a first-person
#                question (ship/merge/deploy); "build" is handled structurally
#                below because it is ambiguous (a shed, muscle, a habit)
# A question qualifies when it (a) is interrogative/request-shaped, (b) is about
# the user's OWN work (first person, or an ownerless count of commits) and
# (c) carries a software anchor: an artifact noun, a software verb, a bare
# "building" (no object), or "building <software noun>".
_SOFTWARE_ACTIVITY = {
    "artifacts": (
        "code", "lines of code", "loc", "codebase", "commits", "pull requests",
        "pull request", "prs", "pr", "merge requests", "software",
        "bugfix", "bugfixes",
    ),
    "verbs": ("ship", "merge", "deploy", "refactor", "release", "implement"),
}
_SOFTWARE_ARTIFACT_MATCHER = compile_keyword_matcher(
    list(_SOFTWARE_ACTIVITY["artifacts"]) + list(_REPOSITORY_CUES["artifacts"])
    + list(_REPOSITORY_CUES["systems"])
)
_SOFTWARE_VERB_MATCHER = compile_keyword_matcher(list(_SOFTWARE_ACTIVITY["verbs"]))
_BUILD_VERB = re.compile(r"\bbuil(?:d|ding|t)\b(?:\s+(?P<next>[a-z']+))?", re.IGNORECASE)
# What may follow a bare "build": an adverbial/recency word, not an object.
_BARE_BUILD_FOLLOWERS = frozenset({
    "lately", "recently", "today", "yesterday", "this", "last", "so", "these",
    "up", "out", "since", "all", "anything", "much",
})
_FIRST_PERSON = re.compile(r"\b(?:i|i've|i'm|i'd|me|my|mine|we|we've|we're|our|ours)\b", re.IGNORECASE)
_OTHER_PERSON = re.compile(r"\b(?:he|she|they|his|her|their|him|them)\b", re.IGNORECASE)
_QUESTION_SHAPE = re.compile(
    r"^(?:what|which|how|did|have|has|do|does|where|when|can you (?:tell|show|remind)|"
    r"tell me|remind me|show me|give me|summari[sz]e|list)\b",
    re.IGNORECASE,
)
_RECENCY = re.compile(
    r"\b(?:lately|recently|today|yesterday|tonight|this (?:week|month|morning|weekend)|"
    r"last (?:week|month|night)|past (?:\d+|few|couple of) (?:days?|weeks?)|these days|so far)\b",
    re.IGNORECASE,
)


_AUXILIARIES = frozenset({
    "is", "was", "are", "were", "has", "have", "had", "did", "does", "do",
    "will", "would", "can", "could", "should", "shall", "may", "might", "must",
})
_TITLE_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\u2019-]*")
_SENTENCE_START = re.compile(r"(?:^|[.?!]\s+)([A-Za-z][A-Za-z'\u2019-]*)\s*([A-Za-z'\u2019-]*)")


# Activity verbs: base -> (past/participle forms, progressive). A PAST activity
# verb is a past form, or an auxiliary (did/have/has/had) with a base form, or
# "been" + a progressive ("have been building"). Present/modal uses ("can you
# build", "how do I fix") never count.
_ACTIVITY_VERBS = {
    "build": (("built",), "building"), "write": (("wrote", "written"), "writing"),
    "ship": (("shipped",), "shipping"), "merge": (("merged",), "merging"),
    "push": (("pushed",), "pushing"), "commit": (("committed",), "committing"),
    "add": (("added",), "adding"), "deploy": (("deployed",), "deploying"),
    "refactor": (("refactored",), "refactoring"), "release": (("released",), "releasing"),
    "implement": (("implemented",), "implementing"), "make": (("made",), "making"),
    "create": (("created",), "creating"),
}
_PAST_AUX = frozenset({"did", "have", "has", "had", "i've", "we've", "ive", "weve"})
_QUANTITY = re.compile(r"\bhow\s+(?:much|many)\b", re.IGNORECASE)


def _past_activity_verb(lower: str, bases=None) -> bool:
    tokens = re.findall(r"[a-z']+", lower)
    tset = set(tokens)
    has_aux = bool(tset & _PAST_AUX)
    for base, (pasts, ing) in _ACTIVITY_VERBS.items():
        if bases is not None and base not in bases:
            continue
        if tset & set(pasts) or (has_aux and base in tset) or ("been" in tset and ing in tset):
            return True
    return False


def _software_verb_in_past(lower: str) -> bool:
    return _past_activity_verb(lower, bases=set(_SOFTWARE_ACTIVITY["verbs"]))


def _names_another_person(text: str) -> bool:
    """Whether the question is about someone ELSE's code (structural, no name list).

    Another person is marked by: a third-person pronoun; a role possessive
    ("my brother's code", roles from memory.graph_utils.PERSON_ROLE_NOUNS); or a
    capitalized word in a SUBJECT/OWNER position: right after an auxiliary
    ("is Rowan ..."), possessive ("Rowan's code"), or a sentence-initial subject
    followed by an auxiliary/inflected verb. A capitalized word that is the object
    of a preposition or a tool/language/project name ("for Daemon", "in Python")
    never counts; nor do the live assistant name and the repository system names.
    """
    if _OTHER_PERSON.search(text):
        return True
    from memory.graph_utils import PERSON_ROLE_NOUNS  # lazy import: layering (utils must not import memory at load)
    roles = "|".join(sorted(PERSON_ROLE_NOUNS))
    if re.search(rf"\b(?:my|our)\s+(?:{roles})(?:'s|\u2019s)\b", text, re.IGNORECASE):
        return True
    from config.app_config import ASSISTANT_NAME  # lazy import: live-config (BC-82)
    exempt = {s.lower() for s in _REPOSITORY_CUES["systems"]}
    exempt.update({"i", "pr", "prs", "loc", str(ASSISTANT_NAME or "").lower()})
    tokens = _TITLE_TOKEN.findall(text)
    for idx, tok in enumerate(tokens):
        base = re.sub(r"(?:'s|\u2019s)$", "", tok)
        if not base[:1].isupper() or base.lower() in exempt:
            continue
        if tok != base:                                   # possessive owner
            return True
        if idx > 0 and tokens[idx - 1].lower() in _AUXILIARIES:
            return True
    from utils.query_checker import _function_word_lexicon  # lazy import: cycle
    function_words = _function_word_lexicon()
    for m in _SENTENCE_START.finditer(text):
        word, nxt = m.group(1), m.group(2).lower()
        if (word[:1].isupper() and word.lower() not in exempt
                and word.lower() not in function_words
                and (nxt in _AUXILIARIES or re.search(r"(?:ed|ing|s)$", nxt))):
            return True
    return False


def _software_build_anchor(text: str) -> bool:
    """'build' is software only bare ("what have I been building lately") or with
    a software object ("building a feature"); "building a shed/muscle" is not."""
    for m in _BUILD_VERB.finditer(text):
        nxt = (m.group("next") or "").lower()
        if not nxt or nxt in _BARE_BUILD_FOLLOWERS:
            return True
        tail = text[m.end("next") if m.group("next") else m.end():]
        # "built in Python / with Rust": a capitalized tool or language name
        # as the object of in/with/using is a software anchor (a lowercase
        # place or object, "in the garage", is not).
        if nxt in ("in", "with", "using"):
            words = tail.split()
            if words and words[0][:1].isupper() and words[0].lower() not in ("i", "the"):
                return True
        window = (nxt + " " + " ".join(tail.split()[:3])).lower()
        if _SOFTWARE_ARTIFACT_MATCHER(window):
            return True
    return False


def is_own_code_activity_question(query: str) -> bool:
    """Interrogative about the USER'S OWN recent software work (see module note)."""
    text = normalize_ws(query)
    if not text:
        return False
    lower = text.lower()
    if not (text.rstrip().endswith("?") or _QUESTION_SHAPE.match(text)):
        return False
    first_person = bool(_FIRST_PERSON.search(text))
    if not first_person:
        # Ownerless ("how many commits this week"): only a commit-record count
        # inside a recency window, and only when no one else is named.
        return bool(
            re.search(r"\bcommits?\b", lower) and _RECENCY.search(text)
            and not _names_another_person(text)
        )
    if _names_another_person(text):
        return False
    # ACTIVITY framing is what separates "what did I build/ship" from "why is my
    # code broken": a question about a code artifact's correctness, behaviour or
    # structure has no PAST activity verb, so it never qualifies (no help-phrase list).
    past = _past_activity_verb(lower)
    if not past:
        return False
    if _SOFTWARE_VERB_MATCHER(lower) and _software_verb_in_past(lower):
        return True                                   # "what did we ship to Daemon in October?"
    if _software_build_anchor(text):
        return True                                   # "what have I been building lately?"
    framed = bool(_RECENCY.search(text) or _QUANTITY.search(text))
    return bool(framed and _SOFTWARE_ARTIFACT_MATCHER(lower))


def wants_live_repository_history(query: str) -> bool:
    """The ONE routing predicate for live local git history (status reports and
    own-code activity questions); gatherer, builder and token budget share it."""
    return is_repository_status_report(query) or is_own_code_activity_question(query)
