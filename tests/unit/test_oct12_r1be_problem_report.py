"""R1-be: "Report a problem" bundle — the privacy property is the point (2026-10-12).

Drives the DEPLOYED builder (utils.problem_report) and the DEPLOYED route
(api/routes/report.py via create_app, behind LaunchAuthMiddleware).
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from tests.unit.api_launch_auth_client import authed_client as _client
from tests.unit.api_launch_auth_client import make_test_app as create_app
from tests.unit.helpers_orchestrator import _make_orchestrator
from utils import problem_report

CANARIES = {
    "query": "CANARY_QUERY_zx81 where do I work",
    "response": "CANARY_RESPONSE_qk42 you work at the lab",
    "plan": "CANARY_PLAN_mm07 cover the job",
    "gate": "CANARY_GATE_rr93 matched keyword",
    "memory": "CANARY_MEMORY_tt55 my cat is Biscuit",
    "toolarg": "CANARY_TOOLARG_ab12",
    "file": "CANARY_FILE_cd34.pdf",
    "url": "https://canary-host-ef56.example.com/secret",
    "webreason": "CANARY_WEBREASON_gh78",
}
CRED = "api_key=sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD"


def _record(i=0, **over):
    rec = {
        "mode": "enhanced",
        "query": CANARIES["query"] + f" #{i}",
        "prompt": ("[SYSTEM]\nhi\n[RELEVANT MEMORIES]\n" + CANARIES["memory"]
                   + "\n[CURRENT USER QUERY]\n" + CANARIES["query"] + f" #{i}"),
        "system_prompt": "You are Daemon.",
        "response": CANARIES["response"] + " " + CRED,
        "model": "kimi-3",
        "response_plan": {"points": [CANARIES["plan"]]},
        "gate_reason": CANARIES["gate"],
        "provenance": {"session_id": "sess-1", "files": [CANARIES["file"]]},
        "phase_timings": {"total_wall": 2.5, "prompt_build": 1.0},
        "task_timings": {"memories": 0.4},
        "section_outcomes": {"memories": {"status": "ok", "reason": CANARIES["gate"]},
                             CANARIES["file"]: "ok"},
        "tool_calls": [{"tool": "web_search", "status": "ok",
                        "detail": CANARIES["toolarg"] + " " + CANARIES["url"]}],
        "files": [CANARIES["file"]],
        "grounding_status": "skipped",
        "grounding_flagged": False,
        "citation_drift": [{"sentence": CANARIES["response"]}],
        "wall_elapsed_s": 3.2,
    }
    rec.update(over)
    return rec


def _telemetry_row(i=0, ts="2026-10-12T10:00:00-05:00", **over):
    row = {
        "ts": ts, "query": CANARIES["query"] + f" #{i}", "session_id": "sess-1",
        "mode": "enhanced", "model": "kimi-3", "intent": "factual_recall",
        "intent_confidence": 0.8, "tone_level": "conversational",
        "tone_trigger": "keyword", "response_plan": CANARIES["plan"],
        "plan_points": [CANARIES["plan"]], "gate_reason": CANARIES["gate"],
        "web_trigger_reason": CANARIES["webreason"], "web_error": None,
        "tone_dm_reason": CANARIES["gate"], "wall_elapsed_s": 3.1,
        "tool_calls": [{"tool": "web_search", "status": "ok", "detail": CANARIES["url"]}],
        "response_len": 40, "phase_timings": {"total_wall": 3.0},
        "unknown_field": "this is an unknown prose field " + CANARIES["toolarg"],
    }
    row.update(over)
    return row


def _write_telemetry(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@pytest.fixture
def tel(tmp_path):
    p = tmp_path / "logs" / "turn_records.jsonl"
    p.parent.mkdir()
    _write_telemetry(p, [_telemetry_row(i) for i in range(3)])
    return p


def _bundle(records, tel_path, **kw):
    rows = problem_report.read_telemetry_tail(tel_path)
    return problem_report.build_bundle(
        records, kw.pop("index", -1), telemetry_rows=rows, telemetry_path=tel_path,
        info=problem_report.app_info(model="kimi-3", mode="user"), **kw)


class TestCanaries:
    def test_summary_tier_has_no_canary(self, tel):
        b = _bundle([_record(i) for i in range(3)], tel, description="it forgot",
                    include_full=False)
        text = json.dumps(b)
        for name, val in CANARIES.items():
            assert val not in text, f"canary {name} leaked into the summary tier"
        assert "canary-host" not in text
        assert b["full"] is None
        assert b["turn"]["intent"] == "factual_recall"  # telemetry joined
        assert "telemetry" not in b["turn"]
        assert b["turn"]["tool_calls"] == ["web_search"]
        assert b["turn"]["citation_drift_count"] == 1
        assert b["turn"]["context_sections"] == {"memories": "ok"}
        sizes = b["turn"]["prompt_section_tokens_approx"]
        assert sizes and all(isinstance(v, int) for v in sizes.values())

    def test_full_tier_has_prompt_and_response_but_no_credential(self, tel):
        b = _bundle([_record(i) for i in range(3)], tel, include_full=True)
        full = b["full"]
        assert CANARIES["memory"] in full["prompt"]
        assert CANARIES["response"] in full["response"]
        assert "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD" not in json.dumps(b)
        # the summary tier stays clean even when the full tier is on
        summary_text = json.dumps({k: v for k, v in b.items() if k != "full"})
        for val in CANARIES.values():
            assert val not in summary_text

    def test_description_is_redacted_and_capped(self, tel):
        b = _bundle([_record()], tel, description=CRED + " " + "x" * 5000)
        assert "sk-abcdefghijklmnop" not in b["description"]
        assert len(b["description"]) <= 2100


class TestAllowlist:
    def test_unknown_field_prose_never_appears(self):
        s = problem_report.build_summary(
            _record(), _telemetry_row(unknown_field="I am prose and must not appear"))
        assert "I am prose" not in json.dumps(s)
        assert "unknown_field" not in s

    def test_prose_in_label_field_is_dropped(self):
        s = problem_report.build_summary(
            _record(), _telemetry_row(tone_trigger="I feel awful today", intent="a b",
                                      web_error="Connection failed: see https://x.y/z"))
        assert "tone_trigger" not in s and "intent" not in s and "web_error" not in s
        s2 = problem_report.build_summary(_record(), _telemetry_row(tone_trigger="keyword:grief"))
        assert s2["tone_trigger"] == "keyword:grief"

    def test_prefix_family_drops_prose(self):
        s = problem_report.build_summary(
            _record(grounding_note="the user is sad about X", personal_claim_reason="has a dog named Rex"),
            None)
        assert "grounding_note" not in s and "personal_claim_reason" not in s
        assert s["grounding_flagged"] is False and s["grounding_status"] == "skipped"

    def test_missing_telemetry_marks_unavailable(self, tmp_path):
        b = problem_report.build_bundle([_record()], -1, telemetry_rows=[],
                                        telemetry_path=None, info={})
        assert b["turn"]["telemetry"] == "unavailable"
        assert b["rollup"] is None and b["rollup_error"]

    def test_repeated_query_pairs_right_occurrence(self):
        recs = [_record(0), _record(0, wall_elapsed_s=None)]
        rows = [_telemetry_row(0, ts="2026-10-12T10:00:00-05:00", intent="first"),
                _telemetry_row(0, ts="2026-10-12T11:00:00-05:00", intent="second")]
        assert problem_report.build_summary(recs[0], problem_report.match_telemetry(recs, 0, rows))["intent"] == "first"
        assert problem_report.match_telemetry(recs, -1, rows)["intent"] == "second"


class TestRefereeR1:
    """Referee additions (2026-10-12)."""

    def test_prefix_family_drops_single_word_labels(self):
        # A one-word value in an open-ended family could be a word of user
        # text (a name, an employer); only booleans/numbers survive there.
        s = problem_report.build_summary(_record(personal_claim_subject="Northwind",
                                                 grounding_entity="Rex"), None)
        assert "personal_claim_subject" not in s and "grounding_entity" not in s
        assert "Northwind" not in json.dumps(s) and "Rex" not in json.dumps(s)

    def test_long_paste_joins_on_the_telemetry_prefix(self):
        from utils.turn_telemetry import _sanitize_value
        long_q = "paste " + "x" * 5000
        rec = _record(0)
        rec["query"] = long_q
        row = _telemetry_row(0, intent="paste_turn")
        row["query"] = _sanitize_value(long_q)  # what turn telemetry actually stores
        assert len(row["query"]) < len(long_q)
        assert problem_report.match_telemetry([rec], -1, [row])["intent"] == "paste_turn"


class TestTelemetryRead:
    def test_corrupt_and_oversize_lines_skipped_and_bounded(self, tmp_path):
        p = tmp_path / "t.jsonl"
        good = [json.dumps({"ts": "2026-10-12T10:00:00-05:00", "n": i}) for i in range(2100)]
        p.write_text("\n".join(good[:5] + ["{not json", "[1,2]"] + good[5:]) + "\n")
        rows = problem_report.read_telemetry_tail(p)
        assert len(rows) <= 2000 and rows[-1]["n"] == 2099
        assert all(isinstance(r, dict) for r in rows)

    def test_missing_file_is_empty(self, tmp_path):
        assert problem_report.read_telemetry_tail(tmp_path / "nope.jsonl") == []

    def test_rollup_attached_for_the_turn_date(self, tel):
        b = _bundle([_record(i) for i in range(3)], tel)
        assert b.get("rollup_error") is None, b.get("rollup_error")
        assert b["rollup"]["counts"]["selected"] == 3


class TestContext:
    def test_up_to_five_prior_turns(self, tel):
        b = _bundle([_record(i) for i in range(9)], tel)
        assert len(b["context_turns"]) == 5
        b0 = _bundle([_record(0)], tel)
        assert b0["context_turns"] == []

    def test_bad_index_raises(self, tel):
        with pytest.raises(IndexError):
            _bundle([_record()], tel, index=7)


class TestRoutes:
    @pytest.fixture
    def app(self, tel, monkeypatch):
        from config import app_config
        monkeypatch.setattr(app_config, "TURN_TELEMETRY_PATH", str(tel))
        return create_app(_make_orchestrator(), start_background=False)

    @pytest.mark.asyncio
    async def test_404_without_records(self, app):
        async with _client(app) as c:
            assert (await c.post("/api/report/build", json={})).status_code == 404
            r = await c.get("/api/report/turns")
            assert r.status_code == 200 and r.json() == {"turns": []}

    @pytest.mark.asyncio
    async def test_build_and_turn_list(self, app):
        app.state.daemon.session.debug_records.extend(_record(i) for i in range(25))
        async with _client(app) as c:
            turns = (await c.get("/api/report/turns")).json()["turns"]
            assert len(turns) == 20
            assert [t["index"] for t in turns][:3] == [24, 23, 22]
            assert turns[0]["preview"].startswith("CANARY_QUERY_zx81")
            r = await c.post("/api/report/build", json={"index": -1, "description": "d"})
            assert r.status_code == 200
            body = r.json()
            assert body["schema_version"] == 1 and body["full"] is None
            for val in CANARIES.values():
                assert val not in json.dumps(body)
            r2 = await c.post("/api/report/build", json={"include_full": True})
            assert CANARIES["response"] in r2.json()["full"]["response"]
            assert (await c.post("/api/report/build", json={"index": 99})).status_code == 404

    @pytest.mark.asyncio
    async def test_launch_token_required(self, app):
        from tests.unit.api_launch_auth_client import TEST_ORIGIN
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=TEST_ORIGIN) as c:
            assert (await c.get("/api/report/turns")).status_code == 401
            r = await c.post("/api/report/build", json={}, headers={"Origin": TEST_ORIGIN})
            assert r.status_code == 401


def test_read_report_script_prints_bundle(tel, tmp_path):
    b = _bundle([_record(i) for i in range(3)], tel, include_full=True, description="hello")
    f = tmp_path / "b.json"
    f.write_text(json.dumps(b))
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run([sys.executable, "-s", str(root / "scripts" / "read_report.py"), str(f)],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    for needle in ("-- app", "-- turn", "intent", "factual_recall", "-- context turns (2)",
                   "-- rollup", "selected=3", "-- full prompt", "-- full response"):
        assert needle in out.stdout, needle
    bad = subprocess.run([sys.executable, "-s", str(root / "scripts" / "read_report.py"), str(tmp_path / "x")],
                         capture_output=True, text=True, timeout=60)
    assert bad.returncode == 1
