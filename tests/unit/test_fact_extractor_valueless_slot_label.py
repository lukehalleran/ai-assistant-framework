"""Value-less slot-label objects are junk on both extractor paths.

Incident 2026-10-22: the graph/profile held `User | mom name lol`. The 10-15
edge-interjection strip re-judged the remainder "mom name", which names an
identity-attribute SLOT (whose name?) with no value, and it survived.
Class: BC-55 (junk-object admission).
Also pins (item 1 of the lane) that the legacy weekly-notes catch-up builds no
CorpusManager (BC-81 premise check: it never had one).
"""
import pytest

from memory import fact_extractor as fe
from memory import llm_fact_extractor as lfe

# A slot label needs a PERSON-role owner (referee revision 2026-10-23): "home address"/
# "work email" have none, so they are accepted misses (kept), like "domain name".
JUNK = ["mom name", "mom name lol", "dad number", "my sister's address", "boss email"]
KEPT = ["Tamsin", "home address", "work email", "mom's birthday party", "phone number 555-0100",
        "the name of the course", "new phone", "old phone", "my new phone", "the stone age",
        "favorite number", "lucky number", "Rowan's name is Tamsin", "name"]


def _wrap(text):
    return "  " + text.replace(" ", "\n  ", 1) if " " in text else text


@pytest.mark.parametrize("obj", JUNK)
@pytest.mark.parametrize("form", ["clean", "wrapped"])
def test_deployed_chokepoint_drops_slot_label(obj, form):
    o = obj if form == "clean" else _wrap(obj)
    assert fe._is_junk_object(o, "has", "user") is True


@pytest.mark.parametrize("obj", KEPT)
@pytest.mark.parametrize("form", ["clean", "wrapped"])
def test_valued_objects_are_kept(obj, form):
    o = obj if form == "clean" else _wrap(obj)
    assert fe._is_junk_object(o, "has", "user") is False


def test_bare_mom_behaviour_unchanged():
    # bare "mom": slot rule abstains (no modifier); pre-existing verdict kept
    assert fe._is_junk_object("mom", "has", "user") is False
    assert fe.is_valueless_slot_label("mom") is False


@pytest.mark.parametrize("obj", ["mom name lol", "mom name"])
def test_regex_path_clean_triple_drops(obj):
    assert fe._clean_triple("user", "has", obj) is None


def test_llm_path_normalize_triple_drops():
    assert lfe._normalize_triple({"subject": "user", "relation": "has", "object": "mom name lol"}) is None


def test_llm_path_keeps_valued_object():
    out = lfe._normalize_triple({"subject": "user", "relation": "has_mother", "object": "Tamsin"})
    assert out is not None


def test_weekly_catchup_never_builds_a_corpus_manager(monkeypatch):
    import memory.corpus_manager as cm
    import utils.weekly_notes_generator as wg

    def boom(*a, **k):
        raise AssertionError("weekly notes must not construct a CorpusManager")

    monkeypatch.setattr(cm, "CorpusManager", boom)
    gen = wg.WeeklyNotesGenerator()
    assert not hasattr(gen, "corpus_manager") and not hasattr(gen, "_corpus_manager")


def test_no_nlp_never_junk(monkeypatch):
    monkeypatch.setattr(fe, "_get_nlp", lambda: None)
    assert fe.is_valueless_slot_label("mom name") is False


@pytest.mark.parametrize("obj,rel", [("new phone", "needs"), ("new phone", "bought")])
def test_device_objects_survive_deployed_chokepoint(obj, rel):
    assert fe._is_junk_object(obj, rel, "user") is False
