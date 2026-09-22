"""DM-39: lexical junk filtering preserves synthetic named-entity evidence.

Deployed ingestion, extraction and expansion paths share the lexical filter.
The English stopword lexicon is required in CI; an absent dependency must not
silently skip this guard. class: BC-55, BC-58, BC-76.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import memory.graph_utils as graph_utils
from memory.memory_storage import MemoryStorage


@pytest.mark.parametrize("word", ["again", "also", "very", "still", "then", "because"])
def test_function_words_are_junk(word):
    assert graph_utils._stop_lexicon(), "CI must provision spaCy's English lexicon"
    assert graph_utils.is_junk_entity(word)
    assert graph_utils._is_single_function_word(word)
    assert not MemoryStorage._is_graph_worthy_object(word)


@pytest.mark.parametrize("word", ["quill", "robin", "toby", "pixel", "cedar", "maple grove"])
def test_synthetic_names_are_not_function_words(word):
    assert not graph_utils._is_single_function_word(word)


def test_name_homograph_uses_type_or_original_case():
    assert "may" in graph_utils._stop_lexicon()
    assert graph_utils.is_junk_entity("may")
    assert not graph_utils.is_junk_entity("May")
    assert not graph_utils.is_junk_entity("may", entity_type="person")
    assert MemoryStorage._is_graph_worthy_object("May")


def test_multiword_short_circuit(monkeypatch):
    def fail():
        raise AssertionError("multiword names must not consult the stopword lexicon")
    monkeypatch.setattr(graph_utils, "_stop_lexicon", fail)
    assert not graph_utils._is_single_function_word("maple grove")


def test_missing_lexicon_abstains(monkeypatch):
    monkeypatch.setattr(graph_utils, "_STOP_LEXICON", frozenset())
    assert not graph_utils._is_single_function_word("again")


def test_extraction_preserves_lowercase_typed_name_but_drops_junk():
    nodes = {
        "may": SimpleNamespace(display_name="may", entity_type="person"),
        "again": SimpleNamespace(display_name="again", entity_type="other"),
    }
    resolver = SimpleNamespace(resolve=lambda text: text if text in nodes else None)
    graph = SimpleNamespace(get_entity=nodes.get)
    assert graph_utils.extract_graph_entities("may again", resolver, graph) == {"may"}


def test_expansion_keeps_typed_name_but_drops_junk():
    nodes = {
        name: SimpleNamespace(display_name=name, entity_type=kind, mention_count=3)
        for name, kind in [("seed", "person"), ("may", "person"), ("again", "other")]
    }
    edges = [SimpleNamespace(source_id="seed", target_id=name, metadata={})
             for name in ("may", "again")]
    graph = SimpleNamespace(
        get_entity=nodes.get,
        get_relations=lambda name, direction="both": [e for e in edges if name in (e.source_id, e.target_id)],
    )
    assert graph_utils.rank_expansion_candidates({"seed"}, graph, depth=1) == ["may"]


def test_ingestion_keeps_typed_name_and_rejects_junk_subject(monkeypatch):
    monkeypatch.setattr("memory.memory_storage.app_config.KNOWLEDGE_GRAPH_MIN_CONFIDENCE", 0.0)
    storage = MemoryStorage.__new__(MemoryStorage)
    storage.graph_memory = MagicMock()
    storage.graph_memory.get_entity.return_value = None
    storage.entity_resolver = MagicMock()
    storage.entity_resolver.resolve.return_value = None
    storage.entity_resolver.resolve_or_create.side_effect = lambda text, **kwargs: text
    storage._ingest_fact_to_graph("may", "likes", "painting", entity_type="person", confidence=1.0)
    storage.graph_memory.add_relation.assert_called_once()
    assert storage.graph_memory.add_relation.call_args.args[0].source_id == "may"
    storage.graph_memory.reset_mock()
    storage._ingest_fact_to_graph("again", "likes", "painting", confidence=1.0)
    storage.graph_memory.add_relation.assert_not_called()
