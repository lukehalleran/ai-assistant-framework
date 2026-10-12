"""Oct-12 X1: decision-support trigger needs a personal stake; self-reference note."""
import pytest

from core.response_guidance import include_decision_support, has_personal_stake
from core.prompt.section_instructions import conditional_instruction_tail
from core.response_guidance import DECISION_SUPPORT_GROUNDING
from utils.attachment_audit import self_reference_note


class TestDecisionSupportPersonalStake:
    @pytest.mark.parametrize("q", [
        "tell me about the role of rail in the development of the french state",
        "explain how the printing press changed European politics",
        "what caused the fall of the Roman Republic?",
        "give me an overview of the causes of the First World War",
        "should the government subsidise high speed rail?",
    ])
    def test_historical_or_third_person_excluded(self, q):
        assert include_decision_support(q) is False
        assert DECISION_SUPPORT_GROUNDING.rstrip() not in conditional_instruction_tail({"user_query": q})

    @pytest.mark.parametrize("q", [
        "Should I switch jobs or stay where I am?",
        "Which of these two apartments is better for me?",
        "I'm torn between grad school and working, what do you think I should do?",
        "Weigh both sides of me moving back home.",
        "help me decide whether to drop my second class",
    ])
    def test_first_person_choice_included(self, q):
        assert include_decision_support(q) is True
        assert DECISION_SUPPORT_GROUNDING.rstrip() in conditional_instruction_tail({"user_query": q})

    def test_heavy_topic_still_overrides(self):
        assert include_decision_support("tell me about rail history", is_heavy_topic=True) is True

    def test_stake_helper(self):
        assert not has_personal_stake("tell me about the french state")
        assert has_personal_stake("what is best for me")


class TestSelfReferenceNote:
    def test_titlecase_project_mention_triggers(self):
        note = self_reference_note(["My project Daemon is a RAG system with memory."], "Daemon")
        assert note.startswith("[SELF-REFERENCE NOTE]") and 'names "Daemon"' in note

    @pytest.mark.parametrize("t", [
        "The systemd daemon restarts the service.",
        "run the daemon process in the background",
        "Daemon processes detach from the terminal.",
    ])
    def test_generic_daemon_text_does_not_trigger(self, t):
        assert self_reference_note([t], "Daemon") == ""

    def test_name_comes_from_config(self, monkeypatch):
        from config import app_config
        monkeypatch.setattr(app_config, "ASSISTANT_NAME", "Atlas")
        assert 'names "Atlas"' in self_reference_note(["Atlas is my side project."])
        assert self_reference_note(["Daemon is my side project."]) == ""

    def test_default_config_name(self):
        from config import app_config
        assert app_config.ASSISTANT_NAME == "Daemon"

    def test_sentence_initial_name_with_function_word_triggers(self):
        assert self_reference_note(["Daemon is my side project."], "Daemon")

    def test_rides_the_current_query_like_attachment_notes(self):
        # Same carrier as [ATTACHMENT NOTE]/[DEADLINE NOTE]: appended to
        # merged_input in handle_submit's single note assembly point.
        import inspect
        import gui.handlers as h
        src = inspect.getsource(h._handle_submit_inner)
        assert "self_reference_note(" in src and "_attachment_note" in src
