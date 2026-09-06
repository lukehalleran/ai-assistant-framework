"""Exercise deployed retrieval and labeling on mixed-authorship records."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from core.insight.provenance import label_evidence, render_evidence_block
from core.insight.sweep import run_sweep
from core.insight.types import EvidenceItem, FacetPlan, FacetQuery
from memory.corpus_manager import CorpusManager
from memory.fact_source import find_supporting_user_span, quoted_correspondence_lines


def test_mixed_conversation_never_labels_assistant_as_user():
    items = label_evidence([EvidenceItem(
        doc_id="turn1", collection="conversations", date="2026-08-10",
        text="User: I was tired.\nAssistant: That proves a dopamine reset.\nUser: I did not say that.",
    )])
    assert [i.speaker for i in items] == ["user", "assistant", "user"]
    assert len({i.doc_id for i in items}) == 3
    assert items[1].stance_label == "assistant-inferred"
    assert "dopamine" not in " ".join(i.text for i in items if i.stance_label == "user-stated")
    assert label_evidence(items) == items  # no duplicate split or identity drift


@pytest.mark.asyncio
async def test_sweep_splits_before_clipping_and_keeps_note_authorship(tmp_path):
    corpus = CorpusManager(corpus_file=str(tmp_path / "corpus.json"))
    corpus.add_entry("Scope check\nLecture: work with polynomial regression", "Understood",
                     timestamp=datetime(2026, 9, 5), user_text="Scope check")
    store = MagicMock()
    store.query_collection.side_effect = lambda coll, q, n: [{
        "id": "mixed", "content": "User: I rested today.\nAssistant: " + "Speculative mechanism. " * 40,
        "metadata": {"timestamp": "2026-09-05"},
    }] if coll == "conversations" else ([{
        "id": "note1", "content": "The user seems to have reset their energy.",
        "metadata": {"note_date": "2026-08-12", "timestamp": "2026-08-31",
                     "author": "daemon", "source_type": "daemon_daily_summary"},
    }] if coll == "obsidian_notes" else [])
    items = await run_sweep(
        FacetPlan(facets=[FacetQuery(name="rest", query_text="rest and work", keywords=["work"])]),
        chroma_store=store, corpus_manager=corpus,
        caps={"expand_top_k": 0, "evidence_snippet_chars": 100},
    )
    assert not any(i.collection == "corpus" for i in items)
    note = next(i for i in items if i.collection == "obsidian_notes")
    assert note.stance_label == "assistant-inferred"
    assert note.date == "2026-08-12"
    user = next(i for i in items if i.speaker == "user")
    assert user.text == "I rested today."
    reply = next(i for i in items if i.speaker == "assistant")
    assert reply.stance_label == "assistant-inferred"
    assert len(reply.text) <= 101
    assert "your note" not in render_evidence_block(items)


def test_attachment_search_is_available_but_not_user_evidence(tmp_path):
    corpus = CorpusManager(corpus_file=str(tmp_path / "corpus.json"))
    corpus.add_entry("Typed text\nLecture: regression", "ok", user_text="Typed text")
    assert corpus.search_keyword("regression")
    assert not corpus.search_keyword("regression", authored_only=True)
    corpus.add_entry("Attachment only: correlation", "ok", user_text="")
    # Explicit empty authored text remains empty; it must not fall back to query.
    assert corpus.corpus[-1]["user_text"] == ""
    assert not corpus.search_keyword("correlation", authored_only=True)


def test_semantic_evidence_uses_stored_authored_side_and_keeps_reply_separate():
    items = label_evidence([EvidenceItem(
        collection="conversations", doc_id="t1",
        text="User: Lecture transcript on regression\nAssistant: Here is the scope.",
        metadata={"user_text": "Scope check only", "response": "Here is the scope."},
    )])
    assert [i.text for i in items] == ["Scope check only", "Here is the scope."]
    assert [i.stance_label for i in items] == ["user-stated", "assistant-inferred"]
    empty = label_evidence([EvidenceItem(
        collection="conversations", text="Attachment text", metadata={"user_text": "", "response": ""},
    )])
    assert empty == []


def test_empty_authored_side_does_not_support_a_fact():
    assert find_supporting_user_span(
        {"subject": "user", "relation": "works_on", "object": "cache refactor"},
        [{"user_text": "", "query": "I work on the cache refactor"}],
    ) is None


def test_relay_marker_excludes_agent_commitments_but_preserves_user_framing():
    text = ("I am resting today.\n[relay: build-agent]\n"
            "Next I will tackle the cache refactor.\n[/relay]\nI will study tomorrow.")
    assert quoted_correspondence_lines(text) == {1, 2, 3}
    assert find_supporting_user_span(
        {"subject": "user", "relation": "works_on", "object": "cache refactor"}, [text],
    ) is None


def test_relay_without_close_is_quoted_to_end():
    text = "[relay: build-agent]\nI work on the cache.\nI finished the migration."
    assert quoted_correspondence_lines(text) == {0, 1, 2}
    items = label_evidence([EvidenceItem(collection="corpus", speaker="user", text=text)])
    assert all(i.stance_label == "quoted-correspondence" for i in items)


def test_label_evidence_is_idempotent_across_sweep_and_handler_passes():
    """label_evidence runs in sweep._finalize AND again in handlers (engine /
    frozen-contract items join late). A second pass must not re-suffix the
    quoted-correspondence doc_id (":quoted:quoted") or move any label."""
    from core.insight.provenance import label_evidence
    from core.insight.types import EvidenceItem

    items = [
        EvidenceItem(doc_id="c1", collection="conversations", speaker="", date="2026-08-10",
                     text="User: Hi Morgan,\nThanks for the note.\nBest,\nLuke\n"
                          "Assistant: Sure, sounds good."),
        EvidenceItem(doc_id="c2", collection="conversations", speaker="", date="2026-08-11",
                     text="User: I slept badly.\nAssistant: That sounds rough."),
        EvidenceItem(doc_id="n1", collection="obsidian_notes", date="2026-08-10",
                     text="Daily summary text",
                     metadata={"author": "daemon", "source_type": "daemon_daily_summary"}),
    ]
    first = label_evidence(items)
    second = label_evidence([item.model_copy() for item in first])
    key = lambda seq: [(i.doc_id, i.speaker, i.stance_label, i.text) for i in seq]  # noqa: E731
    assert key(first) == key(second)
    assert [i.doc_id for i in first] == ["c1:user:0:quoted", "c1:assistant:1",
                                         "c2:user:0", "c2:assistant:1", "n1"]
