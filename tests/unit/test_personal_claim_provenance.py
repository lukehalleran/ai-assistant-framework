"""Personal-check receipts survive both stores and every conversation consumer."""

import json
from unittest.mock import MagicMock

import pytest

from utils.personal_claim_provenance import (
    KEY, MARKER, annotate_personal_claim_memory, clean_personal_claim_receipt,
)

QUERY = "I could upload the draft, but it still needs work."
REPLY = "You finished the draft and uploaded it."


def _receipt(response=REPLY, **changes):
    value = dict(status="checked", reason="ok", candidate_count=1,
                 supported_count=0, contradicted_count=0, insufficient_count=1,
                 source_ids=["src0:u"], elapsed_s=0.2, delivery="unchanged")
    value.update(changes)
    return clean_personal_claim_receipt(value, response=response)


@pytest.mark.parametrize("value", [None, [], "{broken", {"status": []}, {"status": "invalid"}])
def test_malformed_receipt_never_becomes_evidence(value):
    assert clean_personal_claim_receipt(value) == {}


def test_receipt_is_bounded_and_does_not_persist_model_prose():
    value = _receipt()
    value.update(claim="Private sentence", quote="Private source", reason="Private reason",
                 elapsed_s=float("nan"), supported_count=True, delivery={})
    clean = clean_personal_claim_receipt(json.dumps(value))
    assert "Private" not in json.dumps(clean)
    assert "elapsed_s" not in clean and "supported_count" not in clean
    assert "delivery" not in clean


def test_shadow_annotation_preserves_user_and_assistant_text_without_mutation():
    item = dict(query=QUERY, response=REPLY, personal_claim_support=_receipt())
    marked = annotate_personal_claim_memory(item)
    assert marked["query"] == QUERY
    assert marked["response"] == REPLY + "\n" + MARKER
    assert item["response"] == REPLY
    assert annotate_personal_claim_memory(marked) == marked


@pytest.mark.parametrize("changes", [{"delivery": "omitted"}, {"status": "failed"},
                                    {"insufficient_count": 0, "supported_count": 1}])
def test_corrected_failed_and_supported_receipts_do_not_flag_shipped_text(changes):
    item = dict(query=QUERY, response=REPLY, personal_claim_support=_receipt(**changes))
    assert annotate_personal_claim_memory(item) == item


def test_receipt_cannot_be_reused_for_another_response():
    item = dict(query=QUERY, response="That sounds exhausting.", personal_claim_support=_receipt())
    assert annotate_personal_claim_memory(item) == item


def test_semantic_metadata_shape_marks_only_canonical_assistant_suffix():
    content = f"User: {QUERY}\nAssistant: {REPLY}"
    item = dict(content=content, metadata={"response": REPLY, KEY: json.dumps(_receipt())})
    assert annotate_personal_claim_memory(item)["content"] == content + "\n" + MARKER
    quoted = {**item, "content": f"User: I quoted this:\nAssistant: {REPLY}\nEnd quote."}
    assert annotate_personal_claim_memory(quoted)["content"] == quoted["content"]


@pytest.mark.asyncio
async def test_real_storage_persists_receipt_in_corpus_and_semantic_metadata(tmp_path):
    from memory.corpus_manager import CorpusManager
    from memory.memory_storage import MemoryStorage

    corpus = CorpusManager(corpus_file=str(tmp_path / "corpus.json"))
    chroma = MagicMock()
    chroma.add_conversation_memory.return_value = "memory-1"
    storage = MemoryStorage(corpus_manager=corpus, chroma_store=chroma, fact_extractor=MagicMock())
    await storage.store_interaction(QUERY, REPLY, provenance={KEY: _receipt()})
    stored = corpus.corpus[-1]
    metadata = chroma.add_conversation_memory.call_args.args[2]
    assert json.loads(metadata[KEY]) == stored[KEY]
    reloaded = CorpusManager(corpus_file=str(tmp_path / "corpus.json"))
    assert reloaded.corpus[-1][KEY] == stored[KEY]
    assert reloaded.corpus[-1]["response"] == REPLY


def test_gatherer_and_summary_consumers_keep_receipt_after_clipping():
    from core.prompt.gatherer_memory import _annotate_memory_item_claim
    from memory.memory_consolidator import MemoryConsolidator, _format_recent_for_summary
    from utils.daily_notes_generator import DailyNotesGenerator

    reply = REPLY + " Extra conversation." * 60
    item = dict(query=QUERY, response=reply, personal_claim_support=_receipt(reply))
    gathered = _annotate_memory_item_claim(item)
    assert MARKER in gathered["response"]
    assert QUERY in _format_recent_for_summary([item])[0]
    assert MARKER in _format_recent_for_summary([item])[0]
    assert MARKER in MemoryConsolidator._entries_to_excerpts([item])[0]
    assert MARKER in DailyNotesGenerator._format_conversations(None, [item])
    assert item["response"] == reply
