"""Regression: Obsidian note images only attach on visual-intent queries.

The diet-problem incident (2026-07-14): a course note surfacing on an
unrelated turn ("...gui was literally basically just two prompts") shipped its
embedded lecture screenshot to the multimodal call, and the model narrated the
attached image as a topic pivot despite the prompt-level "do not comment"
guard. Note-image attachment now goes through the same visual-intent gate as
visual memories. Tests call THE deployed _should_include_note_images.
"""

from unittest.mock import patch

from core.prompt.builder import _should_include_note_images


def _gate(model="multi", query="", intent=None, multimodal=True, cfg=True):
    with patch("core.prompt.builder._is_multimodal_model", return_value=multimodal), \
         patch("core.prompt.builder.OBSIDIAN_INCLUDE_IMAGES", cfg):
        return _should_include_note_images(model, query, intent)


class TestNoteImageGate:
    def test_incident_query_does_not_attach(self):
        """The exact failure: casual message, no visual intent → no image."""
        assert _gate(query="Yeah pretty crazy this gui was literally basically just two prompts") is False

    def test_visual_word_attaches(self):
        assert _gate(query="can you look at the diagram in my optimization notes?") is True
        assert _gate(query="show me that slide again") is True

    def test_recall_intents_attach(self):
        assert _gate(query="what was on that homework page?", intent="factual_recall") is True
        assert _gate(query="what did I write yesterday?", intent="temporal_recall") is True

    def test_casual_intent_without_visual_word_blocked(self):
        assert _gate(query="nice work today", intent="casual_social") is False

    def test_non_multimodal_model_blocked(self):
        assert _gate(query="show me the picture", multimodal=False) is False

    def test_config_off_blocked(self):
        assert _gate(query="show me the picture", cfg=False) is False
