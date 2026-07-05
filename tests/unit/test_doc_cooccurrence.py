"""Unit tests for the doc-co-occurrence oracle (knowledge/doc_cooccurrence.py).

Retrieval is mocked (no FAISS), so these are fast and deterministic. The focus is the
`bidirectional` opt-in: default (either-direction) must keep the validated behaviour;
bidirectional=True must require BOTH directions for the text-mention signal, while leaving
the shared-title signal untouched.
"""
import inspect

import knowledge.doc_cooccurrence as dc

# Distinctive stems: _stems("homeostasis") == {"homeos"}, _stems("thermodynamics") == {"thermo"}
A, B = "homeostasis", "thermodynamics"


def _patch(monkeypatch, store):
    monkeypatch.setattr(dc, "semantic_search_with_neighbors", lambda q, k=8: store.get(q, []))


def test_stems_are_distinctive():
    assert dc._stems(A) == {"homeos"}
    assert dc._stems(B) == {"thermo"}


def test_one_directional_known_by_default_but_not_bidirectional(monkeypatch):
    # A's articles mention B's stem ('thermo'); B's articles do NOT mention A's ('homeos').
    _patch(monkeypatch, {
        A: [{"title": "Homeostasis", "content": "regulation via thermo feedback"}],
        B: [{"title": "Thermodynamics", "content": "energy heat entropy work"}],
    })
    assert dc.doc_cooccurrence(A, B).known is True               # either-direction (default)
    assert dc.doc_cooccurrence(A, B, bidirectional=True).known is False  # one-way stripped


def test_bidirectional_known_in_both_modes(monkeypatch):
    # Each article mentions the other's stem -> real bidirectional co-occurrence.
    _patch(monkeypatch, {
        A: [{"title": "Homeostasis", "content": "regulation via thermo feedback"}],
        B: [{"title": "Thermodynamics", "content": "homeos balance entropy"}],
    })
    assert dc.doc_cooccurrence(A, B).known is True
    r = dc.doc_cooccurrence(A, B, bidirectional=True)
    assert r.known is True and r.mention is True


def test_shared_title_unaffected_by_bidirectional(monkeypatch):
    # Title overlap alone -> known regardless of the flag; mention stays False.
    _patch(monkeypatch, {
        A: [{"title": "Cybernetics", "content": "no stems here"}],
        B: [{"title": "Cybernetics", "content": "none either"}],
    })
    r = dc.doc_cooccurrence(A, B, bidirectional=True)
    assert r.known is True and r.shared == 1 and r.mention is False


def test_no_cooccurrence_not_known_in_either_mode(monkeypatch):
    _patch(monkeypatch, {
        A: [{"title": "Homeostasis", "content": "regulation feedback loop"}],
        B: [{"title": "Thermodynamics", "content": "energy heat entropy work"}],
    })
    assert dc.doc_cooccurrence(A, B).known is False
    assert dc.doc_cooccurrence(A, B, bidirectional=True).known is False


def test_is_known_passes_bidirectional_through(monkeypatch):
    _patch(monkeypatch, {
        A: [{"title": "Homeostasis", "content": "regulation via thermo feedback"}],
        B: [{"title": "Thermodynamics", "content": "energy heat entropy work"}],
    })
    assert dc.is_known(A, B) is True
    assert dc.is_known(A, B, bidirectional=True) is False


def test_default_param_is_either_direction():
    # Backward compatibility: existing callers (no flag) keep the validated behaviour.
    assert inspect.signature(dc.doc_cooccurrence).parameters["bidirectional"].default is False
    assert inspect.signature(dc.is_known).parameters["bidirectional"].default is False
