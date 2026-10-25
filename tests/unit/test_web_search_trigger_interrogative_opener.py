"""_INTERROGATIVE_OPENER_RE is word-bounded (2026-10-25, WEB2).

Class: BC-01 (keyword matched inside a longer word). "couldn't focus" opened
as an interrogative "could ..." and kept narration clauses out of narration.
"""
from utils.web_search_trigger import _INTERROGATIVE_OPENER_RE as RE


def test_contraction_is_not_an_interrogative_opener():
    for t in ("couldn't focus all afternoon", "couldn't focus all\n  afternoon".replace("\n  ", " "),
              "isn't working", "didn't sleep well", "wasn't able to focus", "anyone home", "willing to try"):
        assert not RE.match(t), t


def test_real_openers_still_match_clean_and_wrapped():
    for t in ("could you check the news", "could you\n   check the news".replace("\n   ", " "),
              "what's the weather", "how do I fix this", "is it raining", "any news"):
        assert RE.match(t), t


INVERTED = ("Isn't it raining", "Couldn't you check the forecast", "Didn't the election end",
            "Don't they close at 5", "Won't they call", "Isn\u2019t it raining",
            "Haven't we met", "Couldn't there be another way")


def test_inverted_negative_contractions_are_openers_clean_and_wrapped():
    for t in INVERTED:
        assert RE.match(t), t
        flat = " ".join(("\n   ".join(t.split(" ", 1))).split())  # wrapped form flattens
        assert RE.match(flat), flat


def test_uninverted_negative_contractions_are_not_openers():
    for t in ("couldn't focus", "isn't working", "didn't sleep well", "wouldn't help me",
              "Couldn't focus all afternoon"):
        assert not RE.match(t), t
        assert not RE.match(" ".join(t.split())), t
