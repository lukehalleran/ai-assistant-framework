# tests/agent_branch/test_scoring_divergence.py
"""Content-aware divergence in scoring: file-overlap jaccard is 1.0 whenever two
survivors edit the same file, so it can't tell convergence from genuine divergence
on single-file objectives. content_similarity (diff-content) can — this pins it."""

from types import SimpleNamespace

from agent_branch.scoring import score_branches


def _rep(bid, diff, touched=("f.py",)):
    return SimpleNamespace(
        branch_id=bid, strategy=bid, outcome="passed", reasons=[],
        diff_excerpt=diff,
        static_gate=SimpleNamespace(touched_paths=list(touched),
                                    added_lines=diff.count("\n+"), removed_lines=0, flags=[]),
        run_stats=SimpleNamespace(tokens_spent=1, wallclock_elapsed_s=1.0, commits=0, cycles=0),
        sandbox_eval=SimpleNamespace(passed=True, branch_evidence=None),
    )


def test_identical_diffs_are_content_similar_1():
    diff = "diff --git a/f.py b/f.py\n+++ b/f.py\n@@ +1 @@\n+def f():\n+    return 1\n"
    p = score_branches([_rep("a", diff), _rep("b", diff)])
    o = p.overlaps[0]
    assert o.jaccard == 1.0                 # same file
    assert o.content_similarity == 1.0      # ...and identical content (converged)


def test_divergent_diffs_same_file_have_low_content_similarity():
    p = score_branches([
        _rep("a", "+++ b/f.py\n@@ @@\n+def f():\n+    return 1\n"),
        _rep("b", "+++ b/f.py\n@@ @@\n+def f(x, y, z):\n+    total = x + y + z\n"
                  "+    for _ in range(3):\n+        total += 1\n+    return total\n"),
    ])
    o = p.overlaps[0]
    assert o.jaccard == 1.0                  # file-overlap says "converged"...
    assert o.content_similarity < 0.6        # ...but the content clearly diverged


def test_no_overlap_section_for_single_survivor():
    p = score_branches([_rep("only", "+++ b/f.py\n@@ @@\n+x = 1\n")])
    assert p.overlaps == []
