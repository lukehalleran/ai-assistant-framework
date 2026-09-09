"""Privacy boundaries for user-shareable debug artifacts."""

from utils.privacy_redaction import (
    build_redacted_prompt_export,
    redact_data,
    redact_text,
)


def test_redact_text_removes_structured_pii_and_credentials():
    source = (
        "Email student@example.edu; phone (404) 555-0123; GTID 900123456; "
        "SSN 123-45-6789; DOB: January 2, 2000\n"
        "Home address: 100 Example Street, Atlanta, GA\n"
        "api_key=sk-example000000000000000000"
    )

    result = redact_text(source)

    for secret in (
        "student@example.edu",
        "404) 555-0123",
        "900123456",
        "123-45-6789",
        "January 2",
        "100 Example Street",
        "sk-example000000000000000000",
    ):
        assert secret not in result
    assert "[REDACTED EMAIL]" in result
    assert "[REDACTED PHONE]" in result
    assert "[REDACTED ID]" in result
    assert "[REDACTED SSN]" in result
    assert "[REDACTED DOB]" in result
    assert "[REDACTED ADDRESS]" in result
    assert "[REDACTED CREDENTIAL]" in result


def test_redact_text_preserves_debug_numbers_and_is_idempotent():
    source = "Turn 42 took 1.25 seconds on 2026-09-02 with 128000 tokens."
    assert redact_text(source) == source
    assert redact_text(redact_text("Call 404-555-0123")) == "Call [REDACTED PHONE]"


def test_redact_data_does_not_mutate_nested_record():
    source = {
        "query": "email me at student@example.edu",
        "provenance": {"snippets": ["GTID: 900123456"]},
        "count": 2,
    }

    result = redact_data(source)

    assert result["query"] == "email me at [REDACTED EMAIL]"
    assert result["provenance"]["snippets"] == ["GTID: [REDACTED ID]"]
    assert result["count"] == 2
    assert source["query"] == "email me at student@example.edu"


def test_shared_prompt_export_redacts_every_included_section():
    record = {
        "mode": "enhanced",
        "model": "test-model",
        "query": "My number is 404-555-0123",
        "prompt": "Recent turn: student@example.edu and ID 900123456",
        "system_prompt": "Private callback: +1 (212) 555-0198",
    }

    exported = build_redacted_prompt_export(record, include_system=True)

    assert "[SYSTEM PROMPT]" in exported
    assert "[REDACTED PHONE]" in exported
    assert "[REDACTED EMAIL]" in exported
    assert "[REDACTED ID]" in exported
    assert "555-" not in exported
    assert "student@example.edu" not in exported
    assert "900123456" not in exported


def test_shared_prompt_export_can_omit_system_prompt():
    exported = build_redacted_prompt_export(
        {"query": "hello", "prompt": "context", "system_prompt": "hidden"},
        include_system=False,
    )
    assert "[SYSTEM PROMPT]" not in exported
    assert "hidden" not in exported


# ---------------------------------------------------------------------------
# F11 (2026-09-08): the 9-digit ID rule corrupted hex hashes, and structured
# `birthday=`/`lives_in=` profile fields were not covered at all.
# ---------------------------------------------------------------------------

def test_nine_digit_id_rule_does_not_corrupt_hex_strings():
    # Nine consecutive digits flanked by hex LETTERS (not just other digits)
    # used to match `(?<!\d)\d{9}(?!\d)` because the old lookaround only
    # excluded adjacent digits, not adjacent letters.
    assert redact_text("abc123456789def") == "abc123456789def"


def test_sha256_hex_digest_survives_byte_identical():
    # A synthetic 64-hex-char "hash" with an embedded 9-digit run flanked by
    # hex letters on both sides.
    fake_sha = "aa" * 5 + "123456789" + "bb" * 5 + "0" * (64 - 10 - 9 - 10)
    assert len(fake_sha) == 64
    assert redact_text(fake_sha) == fake_sha
    assert redact_data({"plan_sha256": fake_sha}) == {"plan_sha256": fake_sha}


def test_labelled_and_standalone_nine_digit_ids_still_redact():
    assert "[REDACTED ID]" in redact_text("GTID 903123456")
    assert "903123456" not in redact_text("GTID 903123456")
    assert "[REDACTED ID]" in redact_text("id: 903123456")
    assert "903123456" not in redact_text("id: 903123456")


def test_structured_dob_and_location_fields_redact():
    assert redact_text("birthday=1990-01-02") == "birthday: [REDACTED DOB]"
    assert redact_text("lives_in: Springfield") == "lives_in: [REDACTED LOCATION]"
    assert "1990-01-02" not in redact_text("birthday=1990-01-02")
    assert "Springfield" not in redact_text("lives_in: Springfield")


def test_structured_field_redaction_is_idempotent():
    once = redact_text("birthday=1990-01-02")
    assert redact_text(once) == once
    once_loc = redact_text("lives_in: Springfield")
    assert redact_text(once_loc) == once_loc


def test_bare_location_word_without_separator_untouched():
    source = "I love this location so much."
    assert redact_text(source) == source
