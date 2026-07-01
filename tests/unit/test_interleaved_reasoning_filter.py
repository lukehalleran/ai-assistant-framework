"""Unit tests for InterleavedReasoningFilter.

Regression coverage for the leak where glm-5.2 interleaved reasoning and content
in the agentic final-response stream and a discarded pre-answer draft
("synthesis system.") fused onto the real answer ("Let me check…"). See
core/reasoning_stream_filter.py and conversation 0f6d70c7 (daemon_debug
2026-06-28).
"""
from core.reasoning_stream_filter import (
    InterleavedReasoningFilter,
    MARKER,
    CONTENT,
    THINKING_OPEN,
    THINKING_CLOSE,
)


def _run(deltas, draft_max_chars=50):
    """Feed (reasoning, content) deltas; return (full_stream, content_only)."""
    f = InterleavedReasoningFilter(draft_max_chars=draft_max_chars)
    items = []
    for dr, dc in deltas:
        items.extend(f.feed(dr, dc))
    items.extend(f.finish())
    full = "".join(t for _, t in items)
    content = "".join(t for k, t in items if k == CONTENT)
    return full, content, f


# ── the actual bug ───────────────────────────────────────────────────────────
def test_leading_draft_fragment_is_dropped():
    """reason → short draft → reason → real answer  ⇒  draft never reaches output."""
    deltas = [
        ("Let me figure out what they mean", ""),   # reasoning
        ("", "synthesis system."),                   # leaked draft fragment (17 chars)
        ("Actually I should just look it up", ""),    # reasoning resumes → draft is junk
        ("", "Let me check what you've been up to — pulling recent commits."),  # real answer
    ]
    full, content, f = _run(deltas)
    assert "synthesis system." not in content
    assert content == "Let me check what you've been up to — pulling recent commits."
    assert f.content_emitted is True
    # The fused string that actually leaked must not appear anywhere.
    assert "synthesis system.Let me check" not in full


def test_no_separatorless_fusion_even_in_full_stream():
    deltas = [
        ("think", ""),
        ("", "synthesis system."),
        ("rethink", ""),
        ("", "Here is the real answer about your synthesis work this week, in detail."),
    ]
    full, content, _ = _run(deltas)
    assert content.startswith("Here is the real answer")
    assert "synthesis system." not in full


# ── must NOT lose genuine content ─────────────────────────────────────────────
def test_substantial_first_run_is_kept():
    """A long leading run is genuine answer text and must stream, not be dropped."""
    long_answer = "Yesterday you replaced the inverted known-oracle with a direct cosine gate."
    deltas = [
        ("thinking about it", ""),
        ("", long_answer),
        ("a trailing afterthought", ""),  # reasoning resumes AFTER real content
    ]
    full, content, f = _run(deltas)
    assert content == long_answer
    assert f.content_emitted is True


def test_short_final_answer_is_restored_when_nothing_replaces_it():
    """A short answer followed by trailing reasoning (and no more content) survives."""
    deltas = [
        ("let me think", ""),
        ("", "Yes."),                 # short final answer
        ("trailing reasoning", ""),    # reasoning resumes, then stream ends
    ]
    full, content, f = _run(deltas)
    assert content == "Yes."
    assert f.content_emitted is True


def test_short_answer_no_trailing_reasoning_streams():
    deltas = [
        ("think", ""),
        ("", "Yes."),
    ]
    _, content, f = _run(deltas)
    assert content == "Yes."
    assert f.content_emitted is True


# ── non-reasoning models unaffected ───────────────────────────────────────────
def test_non_reasoning_content_passes_through_live():
    """No reasoning ever → every content delta passes straight through, in order."""
    deltas = [("", "Hello "), ("", "world."), ("", " More.")]
    items = []
    f = InterleavedReasoningFilter()
    for dr, dc in deltas:
        items.extend(f.feed(dr, dc))
    # Content is emitted live, one item per delta (no buffering/holding).
    content_items = [t for k, t in items if k == CONTENT]
    assert content_items == ["Hello ", "world.", " More."]
    assert f.reasoning_seen is False
    assert f.finish() == []


def test_thinking_markers_wrap_reasoning():
    deltas = [
        ("reasoning one", ""),
        ("reasoning two", ""),     # still reasoning → no second <thinking>
        ("", "The answer is long enough to be confirmed as genuine content here."),
    ]
    items = []
    f = InterleavedReasoningFilter()
    for dr, dc in deltas:
        items.extend(f.feed(dr, dc))
    items.extend(f.finish())
    markers = [t for k, t in items if k == MARKER]
    assert markers == [THINKING_OPEN, THINKING_CLOSE]  # opened once, closed once


# ── reasoning-only swallow (recovery is the caller's job) ─────────────────────
def test_reasoning_only_emits_open_marker_and_no_content():
    f = InterleavedReasoningFilter()
    items = []
    items.extend(f.feed("the whole answer went into reasoning", ""))
    items.extend(f.finish())
    assert items == [(MARKER, THINKING_OPEN)]   # dangling open; caller closes + recovers
    assert f.reasoning_seen is True
    assert f.content_emitted is False
    assert f.in_reasoning is True


def test_multiple_leading_drafts_then_answer():
    """reason → draft1 → reason → draft2 → reason → answer  ⇒  only the answer."""
    deltas = [
        ("r", ""),
        ("", "draft one."),
        ("r", ""),
        ("", "draft two."),
        ("r", ""),
        ("", "The genuine final answer that exceeds the draft threshold comfortably."),
    ]
    _, content, _ = _run(deltas)
    assert content == "The genuine final answer that exceeds the draft threshold comfortably."
    assert "draft one." not in content
    assert "draft two." not in content


def test_content_after_confirmation_streams_even_with_later_reasoning():
    """Once the leading run is confirmed, later short content is NOT dropped."""
    first = "This is a confirmed long answer segment well past the threshold."
    deltas = [
        ("r", ""),
        ("", first),
        ("r", ""),         # reasoning resumes after confirmed content
        ("", " tail."),     # short, but must survive (we only protect the lead)
    ]
    _, content, _ = _run(deltas)
    assert content == first + " tail."
