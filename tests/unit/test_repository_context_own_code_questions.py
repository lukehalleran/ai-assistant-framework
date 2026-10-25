"""Own-code activity QUESTIONS route to live local git history.

Oct-25 GITSTAT lane adds "what did I push yesterday?" and "working on ... codebase"
(BC-05/BC-06: predicate coverage via the git-verb / work-on categories).

Incident 2026-10-24 13:57: "what have I been building lately, and how much code
went into it?" is a question, so ``is_repository_status_report`` (request-free
reports only) never routed it to live history; the stale procedural index fed
the turn and the reply claimed the user's building had been "light". Bug class:
BC-31 (nearest: a routing predicate blind to a sibling shape).

Tests drive the deployed predicates and the deployed ``get_git_commits``; every
surface is asserted clean and wrapped/indented.
"""
import textwrap
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.prompt.context_gatherer import ContextGatherer
from utils.repository_context import (
    is_own_code_activity_question,
    is_repository_status_report,
    wants_live_repository_history,
)

POSITIVE = [
    "what have I been building lately, and how much code went into it?",
    "How many commits did I make this week?",
    "what did I ship this week?",
    "how many commits this week?",
    "how many lines of code have I written recently?",
    "What have I been building lately?",
    "tell me what pull requests I merged yesterday",
    "what have I been building in my codebase?",
    "How much code did I write for Daemon this week?",
    "How many PRs did I merge into Daemon this week?",
    "What have I built in Python lately?",
    "what did I ship this week? Anything big?",
    "what did we ship to Daemon in October?",
    "Did I merge the GitHub PR yesterday?",
    "what did I push yesterday?",
    "what have I been working on in the codebase lately?",
]
NEGATIVE = [
    "why isn't my code working?",
    "Can you look at my code for hw 8?",
    "what's wrong with my code here?",
    "how do I fix this error in my code?",
    "Is my code for the CDF part right?",
    "how should I structure my software project for class?",
    "How much code is in my OMSA homework?",
    "what have I been building in the garage, a shed?",
    "how am I building muscle lately?",
    "what is Rowan building these days?",
    "how many commits did Rowan make this week?",
    "what did she ship this week?",
    "What did I commit last week?",
    "what have I been eating lately?",
    "I built a shed this week",
    "how is my brother's code coming along?",
    "How is my friend's code doing this week?",
    "What is Rowan building?",
    "Rowan merged a lot of code this week, what did I ship?",
    "what have I been lifting at the gym lately?",
    "how is my OMSA homework going this week?",
    "I worked on my essay yesterday",
    "what did Rowan push?",
    "push-ups yesterday?",
    "what did I do for push-ups yesterday?",
    "what have I been working on in the garage lately?",
]


def _surface(text, wrapped):
    return textwrap.fill(text, width=24, subsequent_indent="  ") if wrapped else text


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("query", POSITIVE)
def test_own_code_questions_route_live(query, wrapped):
    q = _surface(query, wrapped)
    assert is_own_code_activity_question(q) is True
    assert wants_live_repository_history(q) is True


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("query", NEGATIVE)
def test_non_software_or_other_person_do_not_route(query, wrapped):
    q = _surface(query, wrapped)
    assert is_own_code_activity_question(q) is False
    assert wants_live_repository_history(q) is is_repository_status_report(q)


def test_status_report_route_unchanged():
    assert wants_live_repository_history("Yeah we chilling. Managed to push 6 commits in Uber lol") is True


def _gatherer():
    g = ContextGatherer.__new__(ContextGatherer)
    g.memory_coordinator = SimpleNamespace(chroma_store=None)
    g.memory_id_map = {}
    return g


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_deployed_gatherer_reads_local_git_for_the_incident_query(monkeypatch, wrapped):
    records = [
        f"{i:08x}{'a' * 32}|||feat: invented change {i}|||Body.|||1 hour ago|||"
        "2026-10-23T19:30:00-05:00|||Test Author\x00"
        for i in range(3, 0, -1)
    ]
    run = MagicMock(return_value=SimpleNamespace(returncode=0, stdout="\n".join(records)))
    monkeypatch.setattr("knowledge.git_memory.subprocess.run", run)
    monkeypatch.setattr("config.app_config.GIT_MEMORY_ENABLED", True)
    q = _surface(POSITIVE[0], wrapped)
    commits = await _gatherer().get_git_commits(q)
    assert len(commits) == 3
    assert all(c["metadata"]["retrieval_source"] == "local_git" for c in commits)
    # one `git log` for the records + ONE `git log --numstat` for the window's
    # line stats (never one `git show` per commit)
    assert run.call_count == 2
    assert "--numstat" in run.call_args_list[1].args[0]
