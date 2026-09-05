"""2026-09-04 evening fixes — three threads closed from the day's live dumps.

1. Recent-conversation cost bug (GPT cost audit): a {query, response} corpus
   entry was metered on its RESPONSE only while the formatter rendered the
   full query+response — a 401,972-char attachment paste rode into the next
   turn's [RECENT CONVERSATION] untrimmed (146K-token prompt for a one-line
   question; the retest hit 279K). token_manager now meters the rendered pair
   and caps the query field on its own.
2. Outlook thread: "…in my outlook inbox recently can you read the last one I
   received from them? …" (95 words) never routed to email_search because the
   email-search arm capped the WHOLE message at 30 words. A clause-level
   read-request arm fires on the ≤30-word sentence that carries the request.
3. Attachment retest defects: server temp basenames leaked into the CSV
   manifest / persisted title / missing-file audit (orig_name threading);
   the audit reported R function names (read.csv, write.csv, adj.r) from
   transcripts as missing files; the deadline scan returned on the user's own
   same-zone "11 pm central" and never converted the document's Eastern time.
"""
import os
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# 1. token_manager — conversation-shaped entries
# ---------------------------------------------------------------------------

class _WordTokenizer:
    def count_tokens(self, text, model_name):
        return len((text or "").split())


class _MM:
    def get_active_model_name(self):
        return "test-model"


def _tm(budget=100_000):
    from core.prompt.token_manager import TokenManager
    return TokenManager(model_manager=_MM(), tokenizer_manager=_WordTokenizer(), token_budget=budget)


class TestConversationEntryMetering:
    def test_extract_text_renders_query_and_response_like_formatter(self):
        tm = _tm()
        assert tm._extract_text({"query": "hello there", "response": "hi"}) == "User: hello there\nDaemon: hi"
        assert tm._extract_text({"query": "only q"}) == "User: only q"

    def test_prerendered_content_key_still_wins(self):
        tm = _tm()
        assert tm._extract_text({"content": "C", "query": "q", "response": "r"}) == "C"

    def test_giant_query_is_capped_and_response_untouched(self):
        from core.prompt import token_manager as tmod
        tm = _tm()
        giant = " ".join(f"w{i}" for i in range(20_000))
        ctx = {"recent_conversations": [
            {"query": giant, "response": "short reply", "timestamp": "2026-09-04T14:14:00"},
        ]}
        out = tm._manage_token_budget(ctx)
        entry = out["recent_conversations"][0]
        capped_q = entry["query"]
        assert "middle-out snipped" in capped_q
        assert len(capped_q.split()) <= tmod.CONVERSATION_QUERY_MAX_TOKENS + 10
        assert capped_q.startswith("w0 w1 ")          # head kept (the user's own words)
        assert capped_q.rstrip().endswith("w19999")   # tail kept
        assert entry["response"] == "short reply"      # never overwritten by a joined blob
        assert entry["timestamp"] == "2026-09-04T14:14:00"

    def test_normal_turn_untouched(self):
        tm = _tm()
        item = {"query": "how long will this take me?", "response": "About two hours."}
        out = tm._manage_token_budget({"recent_conversations": [dict(item)]})
        assert out["recent_conversations"][0] == item

    def test_oversized_response_still_capped_in_place(self):
        from core.prompt import token_manager as tmod
        tm = _tm()
        big_r = " ".join(f"r{i}" for i in range(5_000))
        out = tm._manage_token_budget({"recent_conversations": [{"query": "q", "response": big_r}]})
        e = out["recent_conversations"][0]
        assert e["query"] == "q"
        assert "middle-out snipped" in e["response"]
        assert len(e["response"].split()) <= tmod.SEMANTIC_ITEM_MAX_TOKENS + 10


# ---------------------------------------------------------------------------
# 2. gate — clause-level email-read request
# ---------------------------------------------------------------------------

LIVE_OUTLOOK_QUERY = (
    "ok so got that done. I got an email from career@example.edu in my outlook inbox "
    "recently can you read the last one I received from them? I think it was about the "
    "career fair and a virtual resume review in like a week I can sign up for. Need to "
    "figure out what to do about that, both are useful probably? Also, there is both a "
    "virtual career fair, but I also feel like I remember an in person AI focused one at "
    "the end of this month as well, and that could be a good use of time/money, even "
    "though I would have to fly to atlanta"
)


class TestEmailReadRequestClause:
    def test_helper_fires_on_live_outlook_message(self):
        from core.agentic.gate import _email_read_request_clause
        assert len(LIVE_OUTLOOK_QUERY.split()) > 30
        assert _email_read_request_clause(LIVE_OUTLOOK_QUERY) is True

    def test_helper_silent_on_narration_and_signature(self):
        from core.agentic.gate import _email_read_request_clause
        assert _email_read_request_clause("I emailed the form to them yesterday and then went to the gym.") is False
        long_narration = " ".join(["I sent the email this morning"] + ["word"] * 40)
        assert _email_read_request_clause(long_narration) is False
        sig = "Thanks,\nLuke\nEmail: someone@example.com\nPhone: 555-0100"
        assert _email_read_request_clause(sig) is False

    def test_helper_ignores_request_inside_pasted_correspondence(self):
        from core.agentic.gate import _email_read_request_clause
        pasted = (
            "Here is what she sent me, I did not reply yet.\n\n"
            "Hi Luke,\n\n"
            "Can you check the attached email and confirm the dates?\n\n"
            "Best,\nMorgan\n"
            "Advisor, Example University\n"
        )
        assert _email_read_request_clause(pasted) is False

    def test_helper_respects_negation(self):
        from core.agentic.gate import _email_read_request_clause
        assert _email_read_request_clause("Don't check my email for this, just tell me what you think?") is False

    @pytest.mark.asyncio
    async def test_gate_routes_live_outlook_message_to_tools(self):
        from core.agentic.gate import evaluate_agentic_gate
        decision = await evaluate_agentic_gate(LIVE_OUTLOOK_QUERY)
        assert decision.should_trigger is True
        assert "tools" in decision.modes


# ---------------------------------------------------------------------------
# 3a. orig_name threading
# ---------------------------------------------------------------------------

class TestAttachmentDisplayName:
    def test_prefers_orig_name(self):
        from utils.file_processor import attachment_display_name
        f = SimpleNamespace(name="/tmp/uploads/tmpi6ivi0qj.csv", orig_name="UsedCars.csv")
        assert attachment_display_name(f) == "UsedCars.csv"

    def test_falls_back_to_name_basename(self):
        from utils.file_processor import attachment_display_name
        assert attachment_display_name(SimpleNamespace(name="/tmp/x/Homework1-1.pdf")) == "Homework1-1.pdf"

    def test_pipeline_upload_basename_matches(self):
        from core.context_pipeline import _upload_basename
        f = SimpleNamespace(name="/tmp/uploads/tmpabc.csv", orig_name="UsedCars.csv")
        assert _upload_basename(f) == "UsedCars.csv"
        assert _upload_basename(SimpleNamespace(name="/tmp/uploads/tmpabc.csv")) == "tmpabc.csv"

    @pytest.mark.asyncio
    async def test_csv_manifest_and_filename_use_orig_name(self, tmp_path):
        from utils.file_processor import FileProcessor
        # The processor's security check only accepts temp-directory paths.
        assert str(tmp_path).startswith("/tmp/")
        csv = tmp_path / "tmpi6ivi0qj.csv"
        csv.write_text("Id,Price\n1,100\n2,200\n3,300\n")
        f = SimpleNamespace(name=str(csv), orig_name="UsedCars.csv")
        fp = FileProcessor()
        res = await fp.process_files_structured("how big is this?", [f])
        assert len(res.documents) == 1
        assert res.documents[0].filename == "UsedCars.csv"
        assert "[UsedCars.csv: 3 rows × 2 columns" in res.text_content
        assert "tmpi6ivi0qj" not in res.text_content


# ---------------------------------------------------------------------------
# 3b. attachment audit false positives
# ---------------------------------------------------------------------------

class TestAuditFalsePositives:
    def test_code_identifiers_are_not_filenames(self):
        from utils.attachment_audit import _extract_filenames
        text = ("We call read.csv() to import, write.csv() to export, and retrieve "
                "adj.r.squared from the summary; this read.csv function is simplified.")
        assert _extract_filenames(text) == set()
        assert _extract_filenames(text, strict=True) == set()

    def test_lowercase_r_is_not_a_script(self):
        from utils.attachment_audit import _extract_filenames
        assert _extract_filenames("look at adj.r later") == set()
        assert "analysis.R" in _extract_filenames("open analysis.R in RStudio")

    def test_strict_mode_needs_named_file_shape(self):
        from utils.attachment_audit import _extract_filenames
        assert "temps.txt" in _extract_filenames("use temps.txt for this")
        assert _extract_filenames("use temps.txt for this", strict=True) == set()
        assert "Housing.csv" in _extract_filenames("Download Housing.csv here", strict=True)
        assert "hw2_data.csv" in _extract_filenames("see hw2_data.csv", strict=True)

    def test_temp_basename_of_attached_file_is_not_missing(self):
        from utils.attachment_audit import audit_attachments
        doc = SimpleNamespace(
            filename="UsedCars.csv",
            content_text="[tmpi6ivi0qj.csv: 1,264 rows × 12 columns (Id, Model). Use only this data.]\nId,Model\n",
        )
        files = [SimpleNamespace(name="/tmp/up/tmpi6ivi0qj.csv", orig_name="UsedCars.csv")]
        assert audit_attachments("Here is UsedCars.csv", files, [doc]) == ""

    def test_live_retest_note_names_only_housing_and_part_1(self):
        from utils.attachment_audit import audit_attachments
        user_text = (
            "Instructions\nonHousing.csvHomDownload the following homework instruction "
            "file and data file:\n\nHomework1-1.pdf\n\nUsedCars.csvDownload UsedCars.csv\n"
            "This homework quiz is due on Sep 13 (Sunday) at midnight (11:59pm Eastern Time)."
        )
        transcript = (
            "Hello, everyone. We can just use this read.csv() function whenever we want to "
            "read a CSV file. Similarly write.csv() saves it. We retrieve adj.r.squared from "
            "the summary and df.residual as well. The data file UsedCars.csv is attached."
        )
        files = [
            SimpleNamespace(name="/tmp/up/tmpaaa.csv", orig_name="UsedCars.csv"),
            SimpleNamespace(name="/tmp/up/tmpbbb.pdf", orig_name="Homework1-1.pdf"),
            SimpleNamespace(name="/tmp/up/tmpccc.txt", orig_name="lecture1.txt"),
        ]
        docs = [
            SimpleNamespace(filename="UsedCars.csv", content_text="[UsedCars.csv: 1,264 rows × 12 columns (Id, Model, …)]\nId,Model\n"),
            SimpleNamespace(filename="Homework1-1.pdf", content_text="Data Analytics for Business\nHomework 1 – Part 1\nLinear Models\n"),
            SimpleNamespace(filename="lecture1.txt", content_text=transcript),
        ]
        note = audit_attachments(user_text, files, docs)
        assert note.startswith("[ATTACHMENT NOTE]")
        assert "references files not attached: Housing.csv." in note
        for junk in ("read.csv", "write.csv", "adj.r", "tmpaaa", "df.residual"):
            assert junk not in note
        assert "Part 1" in note


# ---------------------------------------------------------------------------
# 3c. deadline note — same-zone skip + user-stated discrepancy
# ---------------------------------------------------------------------------

LIVE_HW_USER_TEXT = (
    "Now, I have gathered all the module transcripts for the assignment due the 13th at "
    "11 pm central. These are the instructions for the homework: Instructions\n"
    "This homework quiz is due on Sep 13 (Sunday) at midnight (11:59pm Eastern Time).\n"
    "I am also going to attach the dataset and these transcripts."
)


class TestDeadlineSameZoneFix:
    def test_same_zone_first_does_not_block_conversion(self):
        from utils.attachment_audit import deadline_timezone_note
        note = deadline_timezone_note(LIVE_HW_USER_TEXT, user_tz="America/Chicago")
        assert "11:59 PM Eastern = 10:59 PM Central" in note

    def test_user_stated_time_discrepancy_is_named(self):
        from utils.attachment_audit import deadline_timezone_note
        note = deadline_timezone_note(LIVE_HW_USER_TEXT, user_tz="America/Chicago",
                                      user_text=LIVE_HW_USER_TEXT)
        assert "Your message says 11:00 PM Central" in note
        assert "converted deadline is 10:59 PM" in note

    def test_agreeing_user_time_adds_no_discrepancy(self):
        from utils.attachment_audit import deadline_timezone_note
        text = "Due Sunday by 10:59 pm central per the syllabus, which says due 11:59 PM Eastern."
        note = deadline_timezone_note(text, user_tz="America/Chicago", user_text=text)
        assert "10:59 PM Central" in note
        assert "Your message says" not in note

    def test_only_same_zone_times_yield_nothing(self):
        from utils.attachment_audit import deadline_timezone_note
        assert deadline_timezone_note("Assignment due 11:59pm Central Time.", user_tz="America/Chicago") == ""
