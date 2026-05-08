"""Tests for snapshot capture, save/load, and section extraction."""

import json
import tempfile
from pathlib import Path

import pytest

from eval.schema import (
    PromptProvenance,
    PromptSnapshot,
    SectionSnapshot,
    SnapshotLayer,
)
from eval.snapshots import (
    SnapshotCapture,
    extract_sections_from_prompt,
    load_snapshot,
    save_snapshot,
)


def _make_provenance(**overrides) -> PromptProvenance:
    defaults = dict(
        model_name="test-model",
        git_commit_hash="abc1234",
        system_prompt_hash="sys_hash_abc",
        capture_timestamp="2026-05-04T00:00:00+00:00",
    )
    defaults.update(overrides)
    return PromptProvenance(**defaults)


def _make_formatted_sections() -> dict:
    return {
        "recent_conversation": "[RECENT CONVERSATION] n=2\n1) 2026-05-04: User: hi\nDaemon: hello\n\n2) 2026-05-03: User: bye\nDaemon: see ya",
        "memories": "[RELEVANT MEMORIES] n=1\n1) 2026-05-01: Some memory content",
        "user_profile": "[USER PROFILE] n=3\nStored facts...\nfavorite_color=blue",
        "time_context": "[TIME CONTEXT]\nCurrent time: Sunday, 2026-05-04 12:00:00",
        "current_query": "[CURRENT USER QUERY]\n[CURRENT QUERY]\nWhat is my favorite color?",
    }


def _make_context() -> dict:
    return {
        "recent_conversations": [{"query": "hi", "response": "hello"}],
        "memories": [{"content": "Some memory content"}],
        "user_profile": "favorite_color=blue",
        "stm_summary": None,
    }


class TestSnapshotCapture:
    """Test SnapshotCapture layer and full snapshot building."""

    def test_capture_layer_creates_sections(self):
        capture = SnapshotCapture()
        layer = capture.capture_layer(
            layer_name="post_hygiene",
            structured_context=_make_context(),
            formatted_sections=_make_formatted_sections(),
            prompt_text="full prompt text here",
        )
        assert layer.layer_name == "post_hygiene"
        assert "recent_conversation" in layer.sections
        assert "memories" in layer.sections
        assert layer.prompt_hash_exact is not None
        assert layer.prompt_hash_normalized is not None

    def test_capture_layer_raw_no_prompt(self):
        capture = SnapshotCapture()
        layer = capture.capture_layer(
            layer_name="raw_retrieval",
            structured_context=_make_context(),
            formatted_sections=_make_formatted_sections(),
            prompt_text=None,
        )
        assert layer.prompt_text is None
        assert layer.prompt_hash_exact is None
        assert layer.prompt_hash_normalized is None

    def test_capture_layer_stores_formatted_text(self):
        capture = SnapshotCapture()
        formatted = _make_formatted_sections()
        layer = capture.capture_layer(
            layer_name="post_hygiene",
            structured_context=_make_context(),
            formatted_sections=formatted,
            prompt_text="...",
        )
        for key, text in formatted.items():
            assert layer.sections[key].formatted_text == text

    def test_capture_layer_stores_structured_content(self):
        capture = SnapshotCapture()
        ctx = _make_context()
        layer = capture.capture_layer(
            layer_name="post_hygiene",
            structured_context=ctx,
            formatted_sections={"memories": "[RELEVANT MEMORIES] n=1\n1) content"},
            prompt_text="...",
        )
        assert layer.sections["memories"].structured_content == ctx["memories"]

    def test_capture_full_snapshot(self):
        capture = SnapshotCapture()
        formatted = _make_formatted_sections()
        prompt_text = "\n\n".join(formatted.values())

        snapshot = capture.capture_full_snapshot(
            query_text="What is my favorite color?",
            processed_query="What is my favorite color?",
            detected_intent="FACTUAL_RECALL",
            detected_tone="CONVERSATIONAL",
            raw_context=_make_context(),
            raw_formatted_sections={},
            hygiene_context=_make_context(),
            hygiene_formatted_sections=formatted,
            post_hygiene_prompt_text=prompt_text,
            provenance=_make_provenance(),
            system_prompt_text="You are Daemon.",
        )
        assert snapshot.snapshot_id
        assert "raw_retrieval" in snapshot.layers
        assert "post_hygiene" in snapshot.layers
        assert snapshot.final_system_prompt_hash != ""
        assert snapshot.final_user_prompt_hash != ""
        assert snapshot.final_message_payload_hash != ""

    def test_section_token_count_estimated(self):
        capture = SnapshotCapture()
        layer = capture.capture_layer(
            layer_name="post_hygiene",
            structured_context={},
            formatted_sections={"memories": "x" * 400},
            prompt_text="...",
        )
        # ~400 chars / 4 = ~100 tokens
        assert layer.sections["memories"].token_count == 100


class TestSnapshotSaveLoad:
    """Test snapshot save/load roundtrip."""

    def test_save_and_load_roundtrip(self):
        capture = SnapshotCapture()
        formatted = _make_formatted_sections()
        prompt_text = "\n\n".join(formatted.values())

        snapshot = capture.capture_full_snapshot(
            query_text="test query",
            processed_query="test query",
            detected_intent="GENERAL",
            detected_tone="CONVERSATIONAL",
            raw_context={},
            raw_formatted_sections={},
            hygiene_context=_make_context(),
            hygiene_formatted_sections=formatted,
            post_hygiene_prompt_text=prompt_text,
            provenance=_make_provenance(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_snapshot(snapshot, directory=tmpdir)
            assert path.exists()
            assert path.suffix == ".json"

            loaded = load_snapshot(path)
            assert loaded.snapshot_id == snapshot.snapshot_id
            assert loaded.query_text == snapshot.query_text
            assert loaded.detected_intent == snapshot.detected_intent

    def test_roundtrip_preserves_sections(self):
        capture = SnapshotCapture()
        formatted = _make_formatted_sections()

        snapshot = capture.capture_full_snapshot(
            query_text="test",
            processed_query="test",
            detected_intent="",
            detected_tone="",
            raw_context={},
            raw_formatted_sections={},
            hygiene_context=_make_context(),
            hygiene_formatted_sections=formatted,
            post_hygiene_prompt_text="...",
            provenance=_make_provenance(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_snapshot(snapshot, directory=tmpdir)
            loaded = load_snapshot(path)

            layer = loaded.layers["post_hygiene"]
            for key in formatted:
                assert key in layer.sections
                assert layer.sections[key].formatted_text == formatted[key]

    def test_roundtrip_preserves_provenance(self):
        capture = SnapshotCapture()
        prov = _make_provenance(model_name="sonnet-4.5", config_hash="cfg_abc")

        snapshot = capture.capture_full_snapshot(
            query_text="test",
            processed_query="test",
            detected_intent="",
            detected_tone="",
            raw_context={},
            raw_formatted_sections={},
            hygiene_context={},
            hygiene_formatted_sections={},
            post_hygiene_prompt_text="",
            provenance=prov,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_snapshot(snapshot, directory=tmpdir)
            loaded = load_snapshot(path)

            assert loaded.provenance.model_name == "sonnet-4.5"
            assert loaded.provenance.config_hash == "cfg_abc"
            assert loaded.provenance.git_commit_hash == "abc1234"

    def test_raw_retrieval_layer_no_prompt_hash(self):
        capture = SnapshotCapture()
        snapshot = capture.capture_full_snapshot(
            query_text="test",
            processed_query="test",
            detected_intent="",
            detected_tone="",
            raw_context=_make_context(),
            raw_formatted_sections={"memories": "[RELEVANT MEMORIES] n=1\nstuff"},
            hygiene_context={},
            hygiene_formatted_sections={},
            post_hygiene_prompt_text="",
            provenance=_make_provenance(),
        )

        raw = snapshot.layers["raw_retrieval"]
        assert raw.prompt_text is None
        assert raw.prompt_hash_exact is None


class TestExtractSections:
    """Test section extraction from assembled prompt text."""

    def test_extract_known_sections(self):
        prompt = (
            "[RECENT CONVERSATION] n=1\n1) hi\n\n"
            "[RELEVANT MEMORIES] n=1\n1) memory\n\n"
            "[TIME CONTEXT]\nCurrent time: now"
        )
        sections = extract_sections_from_prompt(prompt)
        assert "recent_conversation" in sections
        assert "memories" in sections
        assert "time_context" in sections

    def test_unknown_sections_get_indexed_keys(self):
        prompt = "[SOMETHING NEW] n=1\ncontent here"
        sections = extract_sections_from_prompt(prompt)
        assert any(k.startswith("_unknown_") for k in sections)

    def test_empty_prompt(self):
        sections = extract_sections_from_prompt("")
        assert sections == {}
