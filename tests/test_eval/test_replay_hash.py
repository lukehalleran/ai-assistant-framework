"""Tests for snapshot replay and hash verification."""

import pytest

from eval.schema import (
    PromptProvenance,
    compute_prompt_hash,
    normalize_prompt_text,
)
from eval.snapshots import SnapshotCapture, SnapshotReplay


def _make_provenance() -> PromptProvenance:
    return PromptProvenance(
        model_name="test-model",
        git_commit_hash="abc1234",
        system_prompt_hash="sys_hash",
        capture_timestamp="2026-05-04T00:00:00+00:00",
    )


def _make_snapshot_with_sections(formatted_sections: dict, prompt_text: str = None):
    """Build a snapshot from formatted sections."""
    capture = SnapshotCapture()
    if prompt_text is None:
        # Assemble in order
        from eval.section_registry import SECTION_REGISTRY
        ordered = sorted(
            formatted_sections.items(),
            key=lambda kv: SECTION_REGISTRY[kv[0]].assembly_order if kv[0] in SECTION_REGISTRY else 999,
        )
        prompt_text = "\n\n".join(text for _, text in ordered)

    return capture.capture_full_snapshot(
        query_text="test query",
        processed_query="test query",
        detected_intent="GENERAL",
        detected_tone="CONVERSATIONAL",
        raw_context={},
        raw_formatted_sections={},
        hygiene_context={},
        hygiene_formatted_sections=formatted_sections,
        post_hygiene_prompt_text=prompt_text,
        provenance=_make_provenance(),
    )


SECTIONS_FIXTURE = {
    "recent_conversation": "[RECENT CONVERSATION] n=1\n1) 2026-05-04: User: hello\nDaemon: hi there",
    "memories": "[RELEVANT MEMORIES] n=1\n1) 2026-05-01: memory content here",
    "time_context": "[TIME CONTEXT]\nCurrent time: Sunday, 2026-05-04 12:00:00",
    "current_query": "[CURRENT USER QUERY]\n[CURRENT QUERY]\nWhat is happening?",
}


class TestReplayDeterminism:
    """Test that replay produces deterministic output."""

    def test_same_snapshot_replayed_twice_same_hash(self):
        snapshot = _make_snapshot_with_sections(SECTIONS_FIXTURE)
        replay = SnapshotReplay()

        result1 = replay.replay_from_layer(snapshot, "post_hygiene")
        result2 = replay.replay_from_layer(snapshot, "post_hygiene")

        assert compute_prompt_hash(result1) == compute_prompt_hash(result2)

    def test_replay_follows_assembly_order(self):
        snapshot = _make_snapshot_with_sections(SECTIONS_FIXTURE)
        replay = SnapshotReplay()

        result = replay.replay_from_layer(snapshot, "post_hygiene")

        # recent_conversation (order 1) should come before memories (order 2)
        pos_recent = result.index("[RECENT CONVERSATION]")
        pos_memories = result.index("[RELEVANT MEMORIES]")
        pos_time = result.index("[TIME CONTEXT]")
        pos_query = result.index("[CURRENT USER QUERY]")

        assert pos_recent < pos_memories
        assert pos_memories < pos_time
        assert pos_time < pos_query


class TestExactHashMatch:
    """Test exact hash matching for stable formatting."""

    def test_exact_hash_matches_when_stable(self):
        snapshot = _make_snapshot_with_sections(SECTIONS_FIXTURE)
        replay = SnapshotReplay()
        assert replay.verify_replay_exact(snapshot, "post_hygiene")

    def test_exact_hash_fails_on_content_change(self):
        snapshot = _make_snapshot_with_sections(SECTIONS_FIXTURE)

        # Tamper with a section's formatted text
        layer = snapshot.layers["post_hygiene"]
        layer.sections["memories"].formatted_text += "\nEXTRA LINE"

        replay = SnapshotReplay()
        assert not replay.verify_replay_exact(snapshot, "post_hygiene")


class TestNormalizedHashMatch:
    """Test normalized hash matching for whitespace tolerance."""

    def test_normalized_hash_matches_with_trailing_spaces(self):
        # Build sections with trailing spaces
        sections_with_spaces = {
            k: v + "   " for k, v in SECTIONS_FIXTURE.items()
        }
        # Compute the prompt from clean sections for the stored hash
        from eval.section_registry import SECTION_REGISTRY
        ordered = sorted(
            SECTIONS_FIXTURE.items(),
            key=lambda kv: SECTION_REGISTRY[kv[0]].assembly_order if kv[0] in SECTION_REGISTRY else 999,
        )
        clean_prompt = "\n\n".join(text for _, text in ordered)

        snapshot = _make_snapshot_with_sections(SECTIONS_FIXTURE, prompt_text=clean_prompt)

        # Now replace formatted texts with trailing-space versions
        layer = snapshot.layers["post_hygiene"]
        for key in sections_with_spaces:
            if key in layer.sections:
                layer.sections[key].formatted_text = sections_with_spaces[key]

        replay = SnapshotReplay()
        # Exact should fail (trailing spaces change hash)
        assert not replay.verify_replay_exact(snapshot, "post_hygiene")
        # Normalized should pass (trailing spaces stripped)
        assert replay.verify_replay_normalized(snapshot, "post_hygiene")

    def test_normalized_hash_fails_on_content_change(self):
        snapshot = _make_snapshot_with_sections(SECTIONS_FIXTURE)
        layer = snapshot.layers["post_hygiene"]
        layer.sections["memories"].formatted_text = "[RELEVANT MEMORIES] n=1\n1) TOTALLY DIFFERENT CONTENT"

        replay = SnapshotReplay()
        assert not replay.verify_replay_normalized(snapshot, "post_hygiene")

    def test_normalization_preserves_meaningful_differences(self):
        """Normalization must not erase content changes."""
        text_a = "[SECTION]\nfavorite_color=blue"
        text_b = "[SECTION]\nfavorite_color=red"
        assert normalize_prompt_text(text_a) != normalize_prompt_text(text_b)

    def test_normalization_handles_crlf(self):
        text = "line1\r\nline2\r\nline3"
        normalized = normalize_prompt_text(text)
        assert "\r" not in normalized

    def test_normalization_collapses_blank_lines(self):
        text = "section1\n\n\n\n\nsection2"
        normalized = normalize_prompt_text(text)
        assert "\n\n\n" not in normalized


class TestReplayWithSubset:
    """Test replay with section filtering."""

    def test_replay_subset_of_sections(self):
        snapshot = _make_snapshot_with_sections(SECTIONS_FIXTURE)
        replay = SnapshotReplay()

        result = replay.replay_from_layer(
            snapshot, "post_hygiene",
            sections_to_include=["memories", "current_query"],
        )

        assert "[RELEVANT MEMORIES]" in result
        assert "[CURRENT USER QUERY]" in result
        assert "[RECENT CONVERSATION]" not in result

    def test_replay_empty_subset(self):
        snapshot = _make_snapshot_with_sections(SECTIONS_FIXTURE)
        replay = SnapshotReplay()

        result = replay.replay_from_layer(
            snapshot, "post_hygiene",
            sections_to_include=[],
        )
        assert result == ""


class TestReplayErrors:
    """Test replay error handling."""

    def test_replay_missing_layer_raises(self):
        snapshot = _make_snapshot_with_sections(SECTIONS_FIXTURE)
        replay = SnapshotReplay()

        with pytest.raises(ValueError, match="not found"):
            replay.replay_from_layer(snapshot, "nonexistent_layer")

    def test_verify_exact_returns_false_for_missing_layer(self):
        snapshot = _make_snapshot_with_sections(SECTIONS_FIXTURE)
        replay = SnapshotReplay()
        assert not replay.verify_replay_exact(snapshot, "raw_retrieval")

    def test_verify_normalized_returns_false_for_no_hash(self):
        snapshot = _make_snapshot_with_sections(SECTIONS_FIXTURE)
        # raw_retrieval has no prompt hash
        replay = SnapshotReplay()
        assert not replay.verify_replay_normalized(snapshot, "raw_retrieval")
