"""TopicManager._llm_fallback sends a bounded prompt.

Incident 2026-10-17 15:45: a 125K-char zip-merged message went whole to gpt-4o-mini
(28K prompt tokens, 10.5 s) to get a 2-5 word topic. Class: BC-27 (unbounded input to a
small-output LLM call).
"""
from utils.topic_manager import TopicManager, TOPIC_LLM_MAX_INPUT_CHARS


class _MM:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, **kw):
        self.prompts.append(prompt)
        return "Lecture Review"


def _tm():
    mm = _MM()
    return TopicManager(model_manager=mm, llm_model="x", enable_llm_fallback=True), mm


def test_long_paste_bounded_with_authored_head():
    tm, mm = _tm()
    head = "Here are my lecture transcripts for the statistics course, please review them"
    text = head + "\n\n" + ("lorem ipsum dolor sit amet " * 4500)
    assert len(text) > 100_000
    assert tm._llm_fallback(text) == "Lecture Review"
    assert len(mm.prompts[0]) <= TOPIC_LLM_MAX_INPUT_CHARS
    assert mm.prompts[0].startswith("Here are my lecture")


def test_long_message_without_head_cut_at_whitespace():
    tm, mm = _tm()
    text = "word " * 30000
    tm._llm_fallback(text)
    assert 0 < len(mm.prompts[0]) <= TOPIC_LLM_MAX_INPUT_CHARS
    assert mm.prompts[0].endswith("word")


def test_short_message_unchanged():
    tm, mm = _tm()
    tm._llm_fallback("  School starts soon  ")
    assert mm.prompts == ["School starts soon"]
