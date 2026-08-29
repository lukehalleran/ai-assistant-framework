"""Tests for core/insight/provenance.py — stance labeling + evidence rendering."""

from core.insight.provenance import label_evidence, render_evidence_block
from core.insight.types import EvidenceItem


def _item(**kw):
    defaults = dict(doc_id="d1", text="plain text", date="2026-08-18T13:52:00",
                    collection="conversations", speaker="")
    defaults.update(kw)
    return EvidenceItem(**defaults)


class TestLabeling:
    def test_corpus_user_side(self):
        item = _item(collection="corpus", speaker="user", text="she wasn't abusive")
        label_evidence([item])
        assert item.stance_label == "user-stated"

    def test_corpus_assistant_side_is_inferred(self):
        item = _item(collection="corpus", speaker="assistant",
                     text="she was training you to doubt your perceptions")
        label_evidence([item])
        assert item.stance_label == "assistant-inferred"

    def test_obsidian_is_own_note(self):
        item = _item(collection="obsidian_notes")
        label_evidence([item])
        assert item.stance_label == "users-own-note"

    def test_daemon_authored_collections_are_inferred(self):
        for coll in ("summaries", "reflections", "threads"):
            item = _item(collection=coll)
            label_evidence([item])
            assert item.stance_label == "assistant-inferred", coll

    def test_fact_triple_appraisal(self):
        item = _item(collection="facts", text="casey | is | evil")
        label_evidence([item])
        assert item.stance_label == "extracted-fact"
        assert item.is_appraisal is True

    def test_fact_triple_objective(self):
        item = _item(collection="facts", text="user | lives_in | chicago")
        label_evidence([item])
        assert item.is_appraisal is False

    def test_graph_edge_label(self):
        item = _item(collection="graph", text="Casey is evil", is_appraisal=True)
        label_evidence([item])
        assert item.stance_label == "graph-edge"
        assert item.is_appraisal is True

    def test_conversation_doc_assistant_only_is_inferred(self):
        item = _item(text="Assistant: She chose, deliberately, to dismantle you.")
        label_evidence([item])
        assert item.stance_label == "assistant-inferred"

    def test_conversation_doc_with_user_side(self):
        item = _item(text="User: Casey was evil.\nAssistant: That sounds heavy.")
        label_evidence([item])
        assert item.stance_label == "user-stated"

    def test_evaluative_user_text_flagged_appraisal(self):
        item = _item(collection="corpus", speaker="user", text="Casey was evil.")
        label_evidence([item])
        assert item.is_appraisal is True


class TestRendering:
    def test_every_line_dated_and_attributed(self):
        items = label_evidence([
            _item(collection="corpus", speaker="user", text="she wasn't abusive"),
            _item(doc_id="d2", collection="facts", text="casey | is | evil"),
        ])
        block = render_evidence_block(items)
        lines = block.splitlines()
        assert lines[0].startswith("[E1] 2026-08-18 · corpus · you said:")
        assert "[E2]" in lines[1] and "extracted fact" in lines[1]

    def test_inferred_marker_present(self):
        items = label_evidence([
            _item(collection="summaries", text="User struggled with sleep."),
        ])
        block = render_evidence_block(items)
        assert "[assistant's interpretation, not your words]" in block

    def test_appraisal_marker_present(self):
        items = label_evidence([_item(collection="facts", text="casey | is | evil")])
        block = render_evidence_block(items)
        assert "appraisal, not an objective fact" in block

    def test_max_chars_truncation(self):
        items = label_evidence([
            _item(doc_id=f"d{i}", text="long evidence text " * 10) for i in range(50)
        ])
        block = render_evidence_block(items, max_chars=1000)
        assert len(block) < 1400
        assert "omitted for space" in block

    def test_undated_renders_undated(self):
        items = label_evidence([_item(date=None)])
        assert "undated" in render_evidence_block(items)
