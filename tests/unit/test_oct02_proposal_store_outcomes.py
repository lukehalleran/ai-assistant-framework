"""check_similarity / get_for_dedup: a failure is not "no duplicate" (BC-47).

Typed outcome = RetrievalError, the pattern the file's other readers use.
"""
import pytest

from memory.code_proposal import CodeProposal
from memory.proposal_store import COLLECTION_NAME, ProposalStore
from utils.retrieval_outcome import RetrievalError


class _Coll:
    def __init__(self):
        self.docs = {}

    def count(self):
        return len(self.docs)


class LazyStore:
    """Real-store shape: the collections dict holds a None placeholder until
    ``_get_collection`` opens it; ``create_collection`` only registers."""

    def __init__(self, preload=None, fail=None):
        self.collections = {COLLECTION_NAME: None}
        self._real = _Coll()
        self.fail = fail
        for t in preload or []:
            self._real.docs[t.id] = t

    def create_collection(self, name):
        self.collections.setdefault(name, None)

    def _get_collection(self, name):
        if self.fail == "open":
            raise RuntimeError("cannot open")
        if self.collections[name] is None:
            self.collections[name] = self._real
        return self.collections[name]

    def list_all(self, name):
        if self.fail == "list":
            raise RuntimeError("boom")
        return [
            {"id": k, "content": "", "metadata": {
                "proposal_id": k, "title": v.title, "status": "pending",
                "proposal_type": "feature", "created_at": 1}}
            for k, v in self._get_collection(name).docs.items()
        ]

    def query_collection(self, name, query_text, n_results=5):
        if self.fail == "query":
            raise RuntimeError("query boom")
        coll = self._get_collection(name)
        return [
            {"metadata": {"proposal_id": k, "title": v.title}, "relevance_score": 0.95}
            for k, v in list(coll.docs.items())[:n_results]
        ]


def _p(title):
    return CodeProposal(title=title, reasoning="r")


def test_unopened_collection_still_detects_duplicate():
    existing = _p("Add caching layer")
    store = ProposalStore(LazyStore(preload=[existing]))
    assert store.check_similarity(_p("Add caching layer")) == existing.id


def test_genuine_no_duplicate_still_none():
    store = ProposalStore(LazyStore())
    assert store.check_similarity(_p("Anything")) is None


@pytest.mark.parametrize("fail", ["open", "query"])
def test_check_similarity_failure_is_typed_not_no_duplicate(fail):
    existing = _p("Add caching layer")
    store = ProposalStore(LazyStore(preload=[existing], fail=fail))
    with pytest.raises(RetrievalError) as ei:
        store.check_similarity(_p("Add caching layer"))
    assert ei.value.source == "proposal_store"


def test_get_for_dedup_failure_is_typed_not_empty():
    store = ProposalStore(LazyStore(preload=[_p("Some title")], fail="list"))
    with pytest.raises(RetrievalError):
        store.get_for_dedup()


def test_get_for_dedup_empty_and_populated_unchanged():
    assert ProposalStore(LazyStore()).get_for_dedup() == ""
    out = ProposalStore(LazyStore(preload=[_p("Add caching layer")])).get_for_dedup()
    assert "Add caching layer" in out


def test_not_configured_skip_unchanged():
    s = ProposalStore(None)
    assert s.check_similarity(_p("Some title")) is None
    assert s.get_for_dedup() == ""
