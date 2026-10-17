"""subtitle_to_text: .srt/.vtt -> running prose with [mm:ss] paragraph anchors.

Incident 2026-10-17 (lecture transcript zip of .srt files attached, never read).
Bug class: BC-15 (attachment content never reached the model).
"""
from utils.subtitle_text import subtitle_to_text

SRT = (
    "1\n00:00:01,000 --> 00:00:04,000\nWelcome to the <i>seminar</i> on orchards.\n\n"
    "2\n00:00:04,500 --> 00:00:08,000\nToday we cover pruning.\n\n"
    "3\n00:01:05,000 --> 00:01:09,000\n{\\an8}Next, grafting.\n"
)
VTT = (
    "WEBVTT\n\nNOTE this is a comment\nstill the note\n\n"
    "00:00:01.000 --> 00:00:04.000 align:start position:0%\nWelcome <c.x>again</c>.\n\n"
    "00:00:04.000 --> 00:00:06.000\nWelcome again.\n\n"
    "01:00:02.000 --> 01:00:05.000\nOne hour in.\n"
)


def _variants(raw):
    return [raw, raw.replace("\n", "\r\n"), "﻿" + raw, "\n".join("    " + l if l else l for l in raw.split("\n"))]


def test_srt_drops_machinery_and_anchors_paragraphs():
    for raw in _variants(SRT):
        out = subtitle_to_text(raw)
        assert "-->" not in out and "{\\an8}" not in out and "<i>" not in out
        paras = out.split("\n\n")
        assert paras[0] == "[00:01] Welcome to the seminar on orchards. Today we cover pruning."
        assert paras[1] == "[01:05] Next, grafting."


def test_vtt_header_note_tags_duplicates_and_hours():
    for raw in _variants(VTT):
        out = subtitle_to_text(raw)
        assert "WEBVTT" not in out and "comment" not in out and "note" not in out.lower().replace("notebook", "")
        paras = out.split("\n\n")
        assert paras[0] == "[00:01] Welcome again."
        assert paras[1] == "[1:00:02] One hour in."


def test_missing_blank_lines_still_parse():
    raw = "1\n00:00:01,000 --> 00:00:02,000\nAlpha\n2\n00:00:02,000 --> 00:00:03,000\nBeta\n"
    assert subtitle_to_text(raw) == "[00:01] Alpha Beta"


def test_malformed_passes_through():
    assert subtitle_to_text("just some\nplain lines") == "just some\nplain lines"
    assert subtitle_to_text("") == ""
