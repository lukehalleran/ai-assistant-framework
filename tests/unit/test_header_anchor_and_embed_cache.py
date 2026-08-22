"""
2026-08-21 batch (#9 from the 08-18 audit):

1. The formatter's duplicate-[RECENT CONVERSATION]-header debug check counted
   raw substring occurrences — a stored conversation QUOTING the header
   mid-line (e.g. the user pasting a prompt dump) tripped a false-positive
   ERROR log every turn. The check is now anchored to line starts.

2. adaptive_exemplars.encode_texts_cached: the four exemplar adopters (tone,
   need, intent, web trigger) keyed their prototype caches on store.version
   and re-encoded the WHOLE seed+learned set on every accepted record ("one-
   time setup" ran on every learning event). The shared per-text cache means
   a version bump re-encodes only the newly learned texts.
"""

import inspect
import re

import numpy as np
import pytest

from utils.adaptive_exemplars import encode_texts_cached


class CountingEmbedder:
    def __init__(self, dim=4):
        self.dim = dim
        self.calls = []  # list of text-batches encoded

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=False, **kw):
        self.calls.append(list(texts))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, hash(t) % self.dim] = 1.0
        return out


class TestEncodeTextsCached:
    def test_first_call_encodes_all(self):
        emb, cache = CountingEmbedder(), {}
        out = encode_texts_cached(emb, ["a", "b", "c"], cache)
        assert out.shape == (3, 4)
        assert emb.calls == [["a", "b", "c"]]

    def test_second_call_encodes_only_new_texts(self):
        emb, cache = CountingEmbedder(), {}
        encode_texts_cached(emb, ["a", "b", "c"], cache)
        out = encode_texts_cached(emb, ["a", "b", "c", "freshly learned"], cache)
        assert out.shape == (4, 4)
        # only the new text hit the encoder — this is the whole point
        assert emb.calls[1] == ["freshly learned"]

    def test_fully_cached_call_never_encodes(self):
        emb, cache = CountingEmbedder(), {}
        encode_texts_cached(emb, ["a", "b"], cache)
        encode_texts_cached(emb, ["a", "b"], cache)
        assert len(emb.calls) == 1

    def test_output_order_matches_input(self):
        emb, cache = CountingEmbedder(), {}
        first = encode_texts_cached(emb, ["a", "b"], cache)
        again = encode_texts_cached(emb, ["b", "a"], cache)
        assert np.array_equal(again[0], first[1])
        assert np.array_equal(again[1], first[0])

    def test_empty_texts(self):
        out = encode_texts_cached(CountingEmbedder(), [], {})
        assert out.shape[0] == 0

    def test_normalize_flag_forwarded(self):
        seen = {}

        class KwEmbedder(CountingEmbedder):
            def encode(self, texts, **kw):
                seen.update(kw)
                return super().encode(texts, **kw)

        encode_texts_cached(KwEmbedder(), ["a"], {}, normalize=True)
        assert seen.get("normalize_embeddings") is True


class TestAdoptersWired:
    """Each adopter must use the shared helper with its OWN module cache —
    different adopters embed in different spaces; a shared cache would serve
    one model's vectors to another."""

    @pytest.mark.parametrize("mod_name,cache_attr", [
        ("utils.tone_detector", "_exemplar_text_emb_cache"),
        ("utils.need_detector", "_need_text_emb_cache"),
        ("core.intent_classifier", "_intent_text_emb_cache"),
        ("utils.web_search_trigger", "_anchor_text_emb_cache"),
    ])
    def test_adopter_has_cache_and_calls_helper(self, mod_name, cache_attr):
        import importlib
        mod = importlib.import_module(mod_name)
        assert isinstance(getattr(mod, cache_attr), dict)
        src = inspect.getsource(mod)
        assert "encode_texts_cached" in src

    def test_conftest_clears_caches_between_tests(self):
        # the autouse sandbox fixture must clear all four per-text caches —
        # one test's fake-embedder vectors must not leak into the next
        src = open("tests/conftest.py").read()
        for attr in ("_exemplar_text_emb_cache", "_need_text_emb_cache",
                     "_intent_text_emb_cache", "_anchor_text_emb_cache"):
            assert attr in src


class TestHeaderCheckAnchored:
    def test_formatter_uses_line_anchored_regex(self):
        import core.prompt.formatter as fmt
        src = inspect.getsource(fmt)
        assert r"^\[RECENT CONVERSATION\]" in src
        assert 'final_prompt.count("[RECENT CONVERSATION]")' not in src

    def test_anchored_regex_semantics(self):
        # the exact pattern the formatter compiles
        pat = re.compile(r"^\[RECENT CONVERSATION\]", re.MULTILINE)
        real_headers = (
            "[RECENT CONVERSATION]\nUser: hi\n\n"
            "middle section\n\n"
            "[RECENT CONVERSATION]\nUser: again\n"
        )
        assert len(pat.findall(real_headers)) == 2
        quoted_midline = (
            "[RECENT CONVERSATION]\n"
            "User: my prompt dump said [RECENT CONVERSATION] twice\n"
            "Assistant: that's the header label, quoted [RECENT CONVERSATION]\n"
        )
        assert len(pat.findall(quoted_midline)) == 1
