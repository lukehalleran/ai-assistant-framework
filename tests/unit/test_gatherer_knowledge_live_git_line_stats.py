"""Live local-git route carries line stats; the renderer shows them.

Incident 2026-10-24 20:41: "what have I been building lately, and how much code
went into it?" routed to live git, but the commits carried subjects only, so the
"how much code" half was answered from an old user-stated figure. Bug class:
BC-31 (the route lacked the fact). Tests drive the DEPLOYED extractor against a
real throwaway repo, the deployed ``get_git_commits`` and the deployed formatter.
"""
import os
import subprocess
import textwrap
from types import SimpleNamespace

import pytest

from core.prompt.context_gatherer import ContextGatherer
from core.prompt.formatter import PromptFormatter
from knowledge import git_memory
from knowledge.git_memory import GitMemoryExtractor

QUERY = "what have I been building lately, and how much code went into it?"


def _git(repo, *args):
    env = {**os.environ, "GIT_AUTHOR_NAME": "Tamsin", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "Tamsin", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=env)


@pytest.fixture
def repo(tmp_path):
    r = str(tmp_path)
    _git(r, "init", "-q", "-b", "main")
    (tmp_path / "a.py").write_text("1\n2\n3\n")
    _git(r, "add", "."); _git(r, "commit", "-q", "-m", "feat: first")
    (tmp_path / "a.py").write_text("1\n2\n")                 # -1
    (tmp_path / "b.py").write_text("x\ny\n")                 # +2 (new file)
    _git(r, "add", "."); _git(r, "commit", "-q", "-m", "fix: second")
    _git(r, "checkout", "-q", "-b", "side")
    (tmp_path / "c.py").write_text("q\n")
    _git(r, "add", "."); _git(r, "commit", "-q", "-m", "feat: side")
    _git(r, "checkout", "-q", "main")
    (tmp_path / "d.py").write_text("m\n")
    _git(r, "add", "."); _git(r, "commit", "-q", "-m", "feat: main")
    _git(r, "merge", "-q", "--no-ff", "-m", "Merge side", "side")
    return r


def test_extractor_stats_only_one_numstat_call_and_merge_marked(repo, monkeypatch):
    calls = []
    real = subprocess.run
    monkeypatch.setattr(git_memory.subprocess, "run", lambda cmd, **kw: (calls.append(cmd), real(cmd, **kw))[1])
    commits = GitMemoryExtractor(repo).extract_commits(limit=10, include_stats=True)
    assert sum("--numstat" in c for c in calls) == 1          # batched, not per commit
    assert not any(c[:2] == ["git", "show"] for c in calls)
    by_subject = {c["content"].split("\n")[0]: c["metadata"] for c in commits}
    assert by_subject["Commit: Merge side"]["is_merge"] is True
    assert "lines_added" not in by_subject["Commit: Merge side"]
    m = by_subject["Commit: fix: second"]
    assert (m["lines_added"], m["lines_removed"], m["files_changed_count"]) == (2, 1, 2)
    assert "Changes:" not in "".join(c["content"] for c in commits)   # no diff text


def test_extractor_default_has_no_stats(repo):
    commits = GitMemoryExtractor(repo).extract_commits(limit=3)
    assert all("lines_added" not in c["metadata"] for c in commits)


def _gatherer():
    g = ContextGatherer.__new__(ContextGatherer)
    g.memory_coordinator = SimpleNamespace(chroma_store=None)
    g.memory_id_map = {}
    return g


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_deployed_route_carries_stats_into_rendered_section(repo, monkeypatch, wrapped):
    monkeypatch.setattr("config.app_config.GIT_MEMORY_ENABLED", True)
    real = git_memory.GitMemoryExtractor
    monkeypatch.setattr(git_memory, "GitMemoryExtractor", lambda _p: real(repo))
    q = textwrap.fill(QUERY, width=24, subsequent_indent="  ") if wrapped else QUERY
    commits = await _gatherer().get_git_commits(q)
    assert commits and all(c["metadata"]["retrieval_source"] == "local_git" for c in commits)
    assert any("lines_added" in c["metadata"] for c in commits)
    text = _render_section(commits)
    assert "[+2/−1, 2 files]" in text
    assert "[merge]" in text
    assert "Window total (non-merge commits): 4 commits, +7/−1 lines, 4 files." in text


def test_indexed_route_renders_unchanged():
    commits = [{"content": "Commit: x", "metadata": {"commit_hash": "abc", "tags": "", "age_relative": "1 day ago"}}]
    text = _render_section(commits)
    assert "Window total" not in text and "[+" not in text and "[merge]" not in text


def _render_section(commits):
    fmt = PromptFormatter(token_manager=SimpleNamespace(), time_manager=None)
    prompt = fmt._assemble_prompt({"git_commits": commits}, user_input="q")
    return prompt[prompt.index("[PROJECT COMMIT HISTORY]"):]
