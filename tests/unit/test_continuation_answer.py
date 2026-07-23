"""
tests/unit/test_continuation_answer.py

Regression for the 2026-07-21 "amplification" misread: a short reply directly
answering the assistant's own prior question ("What was the error?" ->
"amplification") was semantically matched against the whole corpus, which
surfaced a topically-matching but contextually-wrong old memory that hijacked
the response. Such turns now route to the lightweight (recent-turns-only) path.
"""

from utils.query_checker import is_continuation_answer, analyze_query


# --- pure detector -------------------------------------------------------

def test_bare_answer_to_prior_question_is_continuation():
    # The exact incident.
    assert is_continuation_answer("amplification", "Nice — what was the error?")


def test_question_mid_turn_then_trailing_statement():
    # The REAL Daemon shape (the live-test miss): question is not the last
    # sentence — the turn ends on a statement.
    prior = "Nice — what was the error? You mentioned the last exchange was bad but subtle."
    assert is_continuation_answer("amplification", prior)


def test_requires_prior_turn_to_contain_a_question():
    assert not is_continuation_answer("amplification", "Nice, that's a solid fix.")
    assert not is_continuation_answer("amplification", "That's a real weight off after the grind.")
    assert not is_continuation_answer("amplification", "")


def test_short_answer_variants():
    q = "What was the error?"
    assert is_continuation_answer("the tone flatline", q)
    assert is_continuation_answer("a race condition", q)
    assert is_continuation_answer("off by one", q)


def test_counter_question_is_not_an_answer():
    q = "What was the error?"
    assert not is_continuation_answer("why?", q)
    assert not is_continuation_answer("what do you mean", q)   # 'what' is a question lead
    assert not is_continuation_answer("which one?", q)


def test_command_or_meta_short_reply_excluded():
    q = "What should we do next?"
    assert not is_continuation_answer("show me the diff", q)   # command
    assert not is_continuation_answer("what did you say", q)   # meta/question lead


def test_long_reply_is_not_a_bare_answer():
    q = "What was the error?"
    long = "it was a subtle amplification bug in the tone detector fast path that i already fixed"
    assert not is_continuation_answer(long, q)


def test_whitespace_and_case_robust():
    assert is_continuation_answer("  Amplification  ", "So — WHAT WAS THE ERROR?")


# --- builder routing -----------------------------------------------------

def test_builder_routes_continuation_to_light_path(monkeypatch):
    from core.prompt.builder import UnifiedPromptBuilder

    class _CorpusManager:
        def get_recent_memories(self, count=1):
            return [{"query": "...", "response": "Nice — what was the error?"}]

    class _MemCoord:
        corpus_manager = _CorpusManager()

    b = UnifiedPromptBuilder.__new__(UnifiedPromptBuilder)
    b.memory_coordinator = _MemCoord()

    qa = analyze_query("amplification")  # short, non-heavy statement
    assert not qa.is_heavy_topic
    assert b._is_continuation_answer("amplification", qa) is True

    # A heavy/crisis short reply keeps full context even after a question.
    qa_heavy = analyze_query("i want to die")
    assert b._is_continuation_answer("i want to die", qa_heavy) is False


def test_builder_no_prior_question_not_continuation():
    from core.prompt.builder import UnifiedPromptBuilder

    class _CorpusManager:
        def get_recent_memories(self, count=1):
            return [{"response": "That's a solid fix."}]

    class _MemCoord:
        corpus_manager = _CorpusManager()

    b = UnifiedPromptBuilder.__new__(UnifiedPromptBuilder)
    b.memory_coordinator = _MemCoord()
    assert b._is_continuation_answer("amplification", analyze_query("amplification")) is False
