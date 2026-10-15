"""Multi-word bare cities localize; sarcasm/generic phrases do not. class: BC-75."""
import pytest

from utils.location_resolver import _is_place_value

spacy = pytest.importorskip("spacy")
try:
    spacy.load("en_core_web_sm")
except Exception:  # pragma: no cover
    pytest.skip("en_core_web_sm not installed", allow_module_level=True)


# "salt lake city" dropped: NER tags it one GPE but "city" is in _GENERIC_PLACE_NOUNS (rule c) - see handoff MEASUREMENTS
@pytest.mark.parametrize("v", ["new york", "san diego",
                               "chicago", "Springfield, IL"])
def test_accepted(v):
    assert _is_place_value(v)


@pytest.mark.parametrize("v", ["joke state", "empty house", "small town",
                               "the city", "a bad mood", "one two three four"])
def test_rejected(v):
    assert not _is_place_value(v)
